import math
import torch
import random
from torch import nn
from torch import Tensor
from typing import Tuple


# i=9900, ref_loss=0.424, reconstruction_loss=0.436, entropy_loss=0.004, frame_entropy=0.304
# ... for codebook_size=4, num_codebooks=16, frame_entropy_cutoff=0.30000001192092896, entropy_scale=0.01
# i=9900, ref_loss=0.414, reconstruction_loss=0.421, entropy_loss=0.012, frame_entropy=0.457
# ... for codebook_size=16, num_codebooks=8, frame_entropy_cutoff=0.45000001788139343, entropy_scale=0.01
# i=9900, ref_loss=0.407, reconstruction_loss=0.413, entropy_loss=0.161, frame_entropy=0.698
# ... for codebook_size=256, num_codebooks=4, frame_entropy_cutoff=0.675000011920929, entropy_scale=0.01


class MultiKmeansQuantizer(nn.Module):
    def __init__(self, dim: int,
                 codebook_size: int,
                 num_codebooks: int):
        """
        Trainable quantizer that encodes a vector into a sequence of integers (corresponding
        to multiple separate kmeans codebooks), aiming to get the least possible expected squared
        difference.
        """
        super(MultiKmeansQuantizer, self).__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.centers = nn.Parameter((dim ** -0.5) * torch.randn(num_codebooks, codebook_size, dim))

        # these are biases on log-likes, used in training when we take into account
        # the entropy of the class distribution.
        self.biases = nn.Parameter(torch.zeros(num_codebooks, codebook_size))



    def get_product_quantizer(self) -> 'MultiKmeansQuantizer':
        """
        Returns a MultiKmeansQuantizer object with codebook_size = self.codebook_size**2 and
           num_codebooks = self.num_codebooks//2, initialized so that each codebook
           in the result is formed from pairs of codebooks in this object.
        """
        new_codebook_size = self.codebook_size ** 2
        new_num_codebooks = self.num_codebooks // 2

        ans = MultiKmeansQuantizer(self.dim,
                                   new_codebook_size,
                                   new_num_codebooks).to(self.centers.device)

        ans.apply_mask = False

        with torch.no_grad():
            for c_out in range(new_num_codebooks):
                c_in1 = 2 * c_out
                c_in2 = 2 * c_out + 1
                for k_in1 in range(self.codebook_size):
                    for k_in2 in range(self.codebook_size):
                        k_out = k_in1 * self.codebook_size + k_in2
                        ans.centers[c_out,k_out,:] = self.centers[c_in1,k_in1,:] + self.centers[c_in2,k_in2,:]
                        ans.biases[c_out,k_out] = self.biases[c_in1,k_in1] + self.biases[c_in2,k_in2]
        return ans

    def get_initial_centers(self, x: Tensor) -> Tensor:
        """
        Gets an initial set of indexes that encode
        """

    def _reset_parameters(self):
        with torch.no_grad():
            self.to_logits.weight[:] = self.to_logits.weight * self.mask
            self.to_output[:] = self.to_logits.weight


    def _logits(self, x: Tensor) -> Tensor:
        if self.apply_mask:
            return (self.to_logits.bias + torch.matmul(x, (self.to_logits.weight * self.mask).t())) * self.logits_scale
        else:
            return self.to_logits(x) * self.logits_scale

    def _to_output(self) -> Tensor:
        if self.apply_mask:
            return self.to_output * self.mask
        else:
            return self.to_output


    def forward(x: Tensor, as_bytes: bool = False) -> Tensor:
        """
        Compute the quantized output, that can be used to reconstruct x.

        Args:
                x: the Tensor to quantize, of shape (*, dim)
        as_bytes:  if True, the quantized output will be returned as a byte
                 array, combining two codes into a single byte if
                 codebook_size <= 16.

        Returns:  if as_bytes == False, returns a torch.LongTensor of shape (*, num_codebooks);
                  if as_bytes == True, returns a Tensor of dtype=torch.uint8, of
                  (*, num_codebooks/2) if codebook_size <= 16; else, require
                  that codebook_size <= 256, and result will be of shape
                  (*, num_codebooks).
        """
        logits = self._logits(x)

        # reshape logits to (B, self.num_codebooks, self.codebook_size) where B is the
        # product of all dimensions of x except the last one.
        tot_codebook_size = self.num_codebooks * self.codebook_size
        logits = logits.reshape(-1, tot_codebook_size)
        B = logits.shape[0]
        indices = torch.distributions.categorical.Categorical(logits=logits).sample()
        # indices is of shape (B, self.num_codebooks)

        if as_bytes:
            if self.codebook_size <= 16:
                indices = indices.transpose(0, 1)  # easiest to index 1st dim.
                indices = (indices[::2] * 16 + indices[1::2]).to(torch.uint8).transpose(0, 1).contiguous()

        shape = list(x.shape)
        shape[-1] = -1
        return indices.reshape(*shape)

    def compute_ref_loss(self, x: Tensor) -> Tensor:
        """
        Compute the loss function, not for optimization, with deterministic indexes using
        argmax not sampling.

        Args:
                x: the Tensor to quantize, of shape (*, dim)

        Returns:   a scalar torch.Tensor containing the relative sum-squared
                    reconstruction loss.
                    It is the sum-squared of (x - reconstructed_x) / sum-squared of x, which will
                    for already-trained models be between 0 and 1, but could be greater than 1
                    at the start of training.
        """
        logits = self._logits(x)

        # reshape logits to (B, self.num_codebooks, self.codebook_size) where B is the
        # product of all dimensions of x except the last one.
        logits = logits.reshape(-1, self.num_codebooks, self.codebook_size)
        B = logits.shape[0]

        # indices: (B, self.num_codebooks)
        indices = torch.argmax(logits, dim=-1)
        # indexes_expanded: (num_codebooks, B, dim)
        indices_expanded = indices.transpose(0, 1).contiguous().unsqueeze(-1).expand(self.num_codebooks, B, self.dim)
        # to_output_reshaped: (num_codebooks, codebook_size, dim)
        to_output_reshaped = self._to_output().reshape(self.num_codebooks, self.codebook_size, self.dim)
        # chosen_codebooks: (num_codebooks, B, dim).
        chosen_codebooks = torch.gather(to_output_reshaped, dim=1, index=indices_expanded)

        # tot_codebooks: (1, B, dim), this is the sum of the chosen rows of `to_output` corresponding
        # to the chosen codebook entries, this would correspond to the approximated x.
        tot_codebooks = chosen_codebooks.sum(dim=0, keepdim=True)
        # tot_error: (1, B, dim), the error of the approximated vs. real x.
        tot_error = tot_codebooks - x.reshape(1, B, self.dim)
        # tot_error_sumsq: scalar, total squared error.  only needed for diagnostics.
        tot_error_sumsq = (tot_error**2).sum()

        x_tot_sumsq = (x ** 2).sum() + 1.0e-20

        rel_tot_error_sumsq = tot_error_sumsq / x_tot_sumsq

        return rel_tot_error_sumsq

    def encode_training(self, x: Tensor, num_iters: int = 4) -> Tuple[Tensor, Tensor]:
        """
        Version of encode() that is to be used during training, that supports an entropy term
        that can be used to balance class probabilities.

        Args:
              x: a Tensor of shape (*, dim) to be approximated
             num_iters: The number of iterations for optimizing the cluster centers
        Returns (indexes, entropy_loss), where:
              indexes: a LongTensor of shape (*, num_codebooks) containing elements
                   in {0..codebook_size-1}, that (approximately, modulo self.biases)
                   minimize the sum-squared error reconstruction loss.
             entropy_loss: a scalar Tensor of shape (*) that is the difference between
                   the maximum possible average entropy, of log(codebook_size), and the
                   observed class entropy.  Is to be used to encourage classes to have
                   approximately the same probability of being chosen.
        """
        assert x.shape[-1] == self.dim
        x_reshaped = x.reshape(-1, self.dim)
        B = x_reshaped.shape[0]

        indexes = torch.zeros(B, self.num_codebooks, dtype=torch.long, device=x.device)

        for iter in range(num_iters):
            indexes = self.refine_indexes(x, indexes, training=False)
            if False:
                avg_loss = ((self.decode(indexes) - x) ** 2).sum() / ((x ** 2).sum() + 1e-20)
                print(f"iter={iter}, avg_loss={avg_loss.item():.3f}")


        classes_one_hot = torch.zeros(B, self.num_codebooks, self.codebook_size,
                                      device=x.device)
        # indexes: (B, num_codebooks)
        classes_one_hot.scatter_(dim=2, index=indexes.unsqueeze(-1), value=1.0)

        log_probs = (1.0e-20 + classes_one_hot.sum(dim=0) / B).log()  # (num_codebooks, codebook_size)

        log_probs = log_probs + zeros_with_deriv_like(self.biases)  # (num_codebooks, codebook_size)
        log_probs = log_probs.log_softmax(dim=1)                   # (num_codebooks, codebook_size)
        entropy_loss = math.log(self.codebook_size) + (log_probs * log_probs.exp()).sum(dim=1).mean()

        indexes = indexes.reshape(*x.shape[:-1], self.num_codebooks)
        return indexes, entropy_loss


    def encode(self, x: Tensor, num_iters: int = 4) -> Tensor:
        """
        Encode a tensor as integers.
        Args:
              x: a Tensor of shape (*, dim) to be approximated
        Returns (indexes, entropy_loss), where:
              indexes: a LongTensor of shape (*, num_codebooks) containing elements
                   in {0..codebook_size-1}, that can be given to decode(), that should
                   approximately minimize the sum-squared error reconstruction loss.
        """
        pass

    def encode_as_bytes(self, x: Tensor) -> Tensor:
        """
        """
        pass

    def decode(self, code: Tensor) -> Tensor:
        """
        Returns the approximated tensor corresponding to the encoding `code`.
        Args:
            code: a Tensor of integer type, of shape (*, self.num_codebooks),
                  containing elements in {0..self.codebook_size - 1}
        Returns:  a Tensor of float, of shape (*, self.dim).
        """
        code_shape = code.shape
        code = code.reshape(-1, self.num_codebooks)
        B = code.shape[0]

        # indexes_expanded has shape (B, self.num_codebooks, 1, self.dim)
        indexes_expanded = code.unsqueeze(-1).unsqueeze(-1).expand(B, self.num_codebooks, 1, self.dim)

        # centers_expanded has shape (B, self.num_codebooks, self.codebook_size, self.dim)
        centers_expanded = self.centers.unsqueeze(0).expand(B, self.num_codebooks, self.codebook_size, self.dim)

        # centers: (B, self.num_codebooks, self.dim)
        centers = torch.gather(centers_expanded, dim=2, index=indexes_expanded).squeeze(2)

        # x: (B, self.dim)
        x = centers.sum(dim=1)
        return x.reshape(*code_shape[:-1], self.dim)

    def refine_indexes(self,
                       x: Tensor,
                       indexes: Tensor,
                       training: bool) -> Tensor:
        """
        Refine choices of indexes (this is called iteratively starting from
        all-zeros).
        Args:
           x:  A Tensor of shape (B, self.dim) to be approximated.
           indexes: A Tensor of integer type, of shape (B, self.num_codebooks),
                that contains elements in {0..self.codebook_size-1}
           training: If true, will take into account self.biases, which will
                in general make the approximation worse but helps control class
                diversity.
         Returns:  A (hopefully) set of indexes of shape (B, self.num_codebooks) that
                  will hopefully reduce the error w.r.t. x, better or at least no worse
                  than `indexes`.  This algorithm is not exact, but if the codebooks are
                  fairly orthogonal it should work fine.   If they are not fairly orthogonal
                  it may not optimize well, but hopefully the codebooks will then learn
                  to be more orthogona..
        """
        B = indexes.shape[0]
        # indexes_expanded has shape (B, self.num_codebooks, 1, self.dim)
        indexes_expanded = indexes.unsqueeze(-1).unsqueeze(-1).expand(B, self.num_codebooks, 1, self.dim)
        # centers_expanded has shape (B, self.num_codebooks, self.codebook_size, self.dim)
        centers_expanded = self.centers.unsqueeze(0).expand(B, self.num_codebooks, self.codebook_size, self.dim)

        # cur_centers: (B, self.num_codebooks, 1, self.dim)
        cur_centers = torch.gather(centers_expanded, dim=2, index=indexes_expanded)
        # x_err is of shape (B, 1, 1, self.dim), it is the current error of the approximation vs. x.
        x_err = cur_centers.sum(dim=1, keepdim=True) - x.unsqueeze(1).unsqueeze(2)

        all_centers = self.centers.unsqueeze(0) # (1, num_codebooks, codebook_size, dim)

        # TODO: get modified_neg_sumsq_errs by a more efficient expression.

        modified_errs = x_err - cur_centers + all_centers
        modified_neg_sumsq_errs = -((modified_errs ** 2).sum(dim=-1)) # (B, num_codebooks, codebook_size)

        if training:
            # self.biases.unsqueeze(0): (1, num_codebooks, codebook_size)
            modified_neg_sumsq_errs += self.biases.unsqueeze(0)

        indexes = modified_neg_sumsq_errs.argmax(dim=2) # (B, num_codebooks)
        return indexes


class _ZerosWithDerivLike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    @staticmethod
    def backward(ctx, output_deriv: Tensor) -> Tensor:
        return output_deriv

def zeros_with_deriv_like(x: Tensor) -> Tensor:
    """
    Returns torch.zeros_like(x), but for backprop purposes it will be as if
    you had returned x.
    """
    return _ZerosWithDerivLike.apply(x)


def _test_quantization():
    torch.manual_seed(1)
    dim = 256
    device = torch.device('cuda')
    model = nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.LayerNorm(dim),
        nn.Linear(dim, dim),
    ).to(device)


    # out of codebook_size, num_codebooks = (4, 16), (16, 8), (256, 4), all of which
    # give 4 bytes per 512-dimensional vector, the best reconstruction loss
    # SET SIZES:
    codebook_size = 4
    num_codebooks = 16

    quantizer = MultiKmeansQuantizer(dim=dim, codebook_size=codebook_size,
                                     num_codebooks=num_codebooks).to(device)

    _ = quantizer.get_product_quantizer() # testing..

    lr=0.001
    num_iters = 3
    for iter in range(num_iters):


        # training quantizer, not model.
        optim = torch.optim.Adam(
            quantizer.parameters(), lr=lr, betas=(0.9, 0.9), eps=1e-9, weight_decay=0.000001
        )

        # We'll choose in the loop how often to step the scheduler.
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.5)

        for i in range(3000):
            B = 600
            x = torch.randn(B, dim, device=device)
            x = model(x)  + 0.05 * x
            # x is the thing we're trying to quantize: the nnet gives it a non-trivial distribution, which is supposed to
            # emulate a typical output of a neural net.  The "+ 0.1 * x" is a kind of residual term which makes sure
            # the output is not limited to a subspace or anything too-easy like that.


            indexes, entropy_loss = quantizer.encode_training(x)

            rel_err = ((x - quantizer.decode(indexes)) ** 2).sum() / ((x ** 2).sum() + 1.0e-20)

            if i % 100 == 0:
                print(f"i={i}, rel_err={rel_err.item():.3f}, entropy_loss={entropy_loss.item():.3f}")

            # There is no point including a scale on the entropy term, since it
            # only affects the biases, whose derivs are not affected by anything
            # else, and since we are using Adam the optimization is unaffected by the scale
            # of these derivatives.
            tot_loss = rel_err + 0.1 * entropy_loss


            tot_loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

        print(f"... for codebook_size={quantizer.codebook_size}, num_codebooks={quantizer.num_codebooks}")

        if iter + 1 < num_iters:
            quantizer = quantizer.get_product_quantizer()
            lr *= 0.5

if __name__ == "__main__":
    _test_quantization()
