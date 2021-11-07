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


class Quantizer(nn.Module):
    def __init__(self, dim: int,
                 codebook_size: int,
                 num_codebooks: int):
        """
        Trainable quantizer that encodes a vector into a sequence of integers (corresponding
        to multiple separate codebooks), aiming to get the least possible expected squared
        difference.
        """
        super(Quantizer, self).__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.to_logits = nn.Linear(dim, codebook_size * num_codebooks)
        self.logits_scale = 4

        # we will sometimes interpret to_output, which is of shape
        # (num_codebooks * codebook_size, dim), as being of shape
        # (num_codebooks, codebook_size, dim); and similarly with self.to_logits
        self.to_output = nn.Parameter(self.to_logits.weight.detach().clone())


        mask = []
        dims_per_codebook = dim // num_codebooks
        for i in range(codebook_size * num_codebooks):
            this_mask_row = []
            for j in range(dim):
                # append True if j // dims_per_codebook) == i, else False.
                this_mask_row.append((j // dims_per_codebook) == (i // codebook_size))
            mask.append(this_mask_row)

        # self.mask has shape: [ codebook_size * num_codebooks, dim ]
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.bool))
        self.apply_mask = True
        #print("mask = ", self.mask)
        #self._reset_parameters()


    def get_product_quantizer(self) -> 'Quantizer':
        """
        Returns a Quantizer object with codebook_size = self.codebook_size**2 and
           num_codebooks = self.num_codebooks//2, initialized so that each codebook
           in the result is formed from pairs of codebooks in this object.
        """
        new_codebook_size = self.codebook_size ** 2
        new_num_codebooks = self.num_codebooks // 2

        ans = Quantizer(self.dim,
                        new_codebook_size,
                        new_num_codebooks).to(self.to_output.device)

        ans.apply_mask = False

        with torch.no_grad():
            for c_out in range(new_num_codebooks):
                c_in1 = 2 * c_out
                c_in2 = 2 * c_out + 1
                for k_in1 in range(self.codebook_size):
                    row_in1 = self.codebook_size * c_in1 + k_in1
                    for k_in2 in range(self.codebook_size):
                        row_in2 = self.codebook_size * c_in2 + k_in2
                        k_out = k_in1 * self.codebook_size + k_in2
                        row_out = new_codebook_size * c_out + k_out
                        ans.to_logits.weight[row_out,:] = self.to_logits.weight[row_in1] + self.to_logits.weight[row_in2]
                        ans.to_logits.bias[row_out] = self.to_logits.bias[row_in1] + self.to_logits.bias[row_in2]
                        ans.to_output[row_out,:] = self.to_output[row_in1] + self.to_output[row_in2]
        return ans

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

    def refine_indexes(self,
                       x: Tensor,
                       indexes: Tensor) -> Tensor:
        """
        Refine choices of indexes, minimizing sum-squared loss.  Note, this is not guaranteed
        not not increase the sum-squared loss, but works OK in practice.

        Args:
           x:  A Tensor of shape (B, self.dim) to be approximated.
           indexes: A Tensor of integer type, of shape (B, self.num_codebooks),
                that contains elements in {0..self.codebook_size-1}
         Returns:  A tensor of indexes of shape (B, self.num_codebooks) that
                  will hopefully reduce the error w.r.t. x, better or at least no worse
                  than `indexes`.  This algorithm is not exact, but if the codebooks are
                  fairly orthogonal it should work fine.   If they are not fairly orthogonal
                  it may not optimize well, but hopefully the codebooks will then learn
                  to be more orthogona..
        """
        B = indexes.shape[0]
        # indexes_expanded has shape (B, self.num_codebooks, 1, self.dim)
        indexes_expanded = indexes.unsqueeze(-1).unsqueeze(-1).expand(B, self.num_codebooks, 1, self.dim)
        # all_centers: (1, num_codebooks, codebook_size, dim)
        all_centers = self.to_output.reshape(1, self.num_codebooks, self.codebook_size, self.dim)
        # centers_expanded has shape (B, self.num_codebooks, self.codebook_size, self.dim)
        centers_expanded = all_centers.expand(B, self.num_codebooks, self.codebook_size, self.dim)

        # cur_centers: (B, self.num_codebooks, 1, self.dim)
        cur_centers = torch.gather(centers_expanded, dim=2, index=indexes_expanded)
        # x_err is of shape (B, 1, 1, self.dim), it is the current error of the approximation vs. x.
        x_err = cur_centers.sum(dim=1, keepdim=True) - x.unsqueeze(1).unsqueeze(2)

        # TODO: get modified_neg_sumsq_errs by a more efficient expression.

        modified_errs = x_err - cur_centers + all_centers
        modified_neg_sumsq_errs = -((modified_errs ** 2).sum(dim=-1)) # (B, num_codebooks, codebook_size)

        indexes = modified_neg_sumsq_errs.argmax(dim=2) # (B, num_codebooks)
        assert indexes.ndim == 2
        return indexes


    def compute_ref_loss(self, x: Tensor, refine_indexes_iters: int = 0) -> Tensor:
        """
        Compute the loss function, not for optimization, with deterministic indexes using
        argmax not sampling.

        Args:
                x: the Tensor to quantize, of shape (*, dim)
               refine_indexes_iters: number of iterations to refine the indexes
                 from their 1st value.

        Returns:   a scalar torch.Tensor containing the relative sum-squared
                    reconstruction loss.
                    It is the sum-squared of (x - reconstructed_x) / sum-squared of x, which will
                    for already-trained models be between 0 and 1, but could be greater than 1
                    at the start of training.
        """
        x_reshaped = x.reshape(-1, self.dim)
        B = x_reshaped.shape[0]
        logits = self._logits(x_reshaped)
        logits = logits.reshape(B, self.num_codebooks, self.codebook_size)

        # indices: (B, self.num_codebooks)
        indices = torch.argmax(logits, dim=-1)
        for _ in range(refine_indexes_iters):
            indices = self.refine_indexes(x_reshaped, indices)
        assert indices.ndim == 2

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

    def compute_loss(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute three (potential) parts of the loss function.

        Args:
                x: the Tensor to quantize, of shape (*, dim)
        Returns (reconstruction_loss, entropy_loss, frame_entropy), where:
           reconstruction_loss: a scalar torch.Tensor containing the relative sum-squared
                     reconstruction loss, constructed as an expectation over class probs.
                     It is the sum-squared of (x - reconstructed_x) / sum-squared of x, which will
                     for already-trained models be between 0 and 1, but could be greater than 1
                     at the start of training.
          ref_loss:    A deterministic version of reconstruction_loss, picking the best class; this is
                   for reference but not for optimization.
          entropy_loss:  the "relative entropy difference" between log(codebook_size) and the
                    average entropy of each of the codebooks (taken over all frames together,
                    i.e.  (ref_entropy - class_entropy) / ref_entropy, which is a number in [0,1].
          frame_entropy: the average entropy of the codebooks on individual frames, between 0
                    and log(codebook_size).  Training will tend to make this approach 0, but
                    then training gets slow due to small derivatives, so we may want to
                    bound it away from 0 at least in the earlier phases of training.
        """
        logits = self._logits(x)

        # reshape logits to (B, self.num_codebooks, self.codebook_size) where B is the
        # product of all dimensions of x except the last one.
        logits = logits.reshape(-1, self.num_codebooks, self.codebook_size)
        B = logits.shape[0]
        probs = logits.softmax(dim=-1)
        indices = torch.distributions.categorical.Categorical(probs=probs).sample()
        # indices is of shape (B, self.num_codebooks) and contains elements in [0..codebook_size - 1]


        # to_output_reshaped: (num_codebooks, codebook_size, dim)
        to_output_reshaped = self._to_output().reshape(self.num_codebooks, self.codebook_size, self.dim)
        # indexes_expanded: (num_codebooks, B, dim)
        indices_expanded = indices.transpose(0, 1).contiguous().unsqueeze(-1).expand(self.num_codebooks, B, self.dim)

        # chosen_codebooks: (num_codebooks, B, dim).
        chosen_codebooks = torch.gather(to_output_reshaped, dim=1, index=indices_expanded)

        # tot_codebooks: (1, B, dim), this is the sum of the chosen rows of `to_output` corresponding
        # to the chosen codebook entries, this would correspond to the approximated x.
        tot_codebooks = chosen_codebooks.sum(dim=0, keepdim=True)
        # tot_error: (1, B, dim), the error of the approximated vs. real x.
        tot_error = tot_codebooks - x.reshape(1, B, self.dim)
        # tot_error_sumsq: scalar, total squared error.  only needed for diagnostics.
        tot_error_sumsq = (tot_error**2).sum()


        # alt_error: (num_codebooks, 1, B, dim) + (num_codebooks, codebook_size, 1, dim) = (num_codebooks, codebook_size, B, dim)
        # alt_error answers the question: "what if, for this particular codebook, we had chosen this
        # codebook entry; what would the error be then?"
        alt_error = (tot_error - chosen_codebooks).unsqueeze(1) + to_output_reshaped.unsqueeze(2)

        # expected_error_sumsq is like tot_error_sumsq, but replaces the
        # discrete choice with an expectation of the sum-sq error, taken over each codebook
        # while leaving the choices of all the other codebooks fixed.
        expected_error_sumsq = ((alt_error ** 2).sum(dim=-1) * probs.permute(1, 2, 0)).sum() - (tot_error_sumsq * (self.num_codebooks - 1))

        x_tot_sumsq = (x ** 2).sum() + 1.0e-20

        rel_tot_error_sumsq = tot_error_sumsq / x_tot_sumsq
        rel_expected_error_sumsq = expected_error_sumsq / x_tot_sumsq


        # following two should be similar, and the same in expectation, which
        # implies neither is consistently larger or smaller.
        #if random.random() < 0.01:
        #    print(f"expected_error_sumsq={rel_expected_error_sumsq}, tot_error_sumsq={rel_tot_error_sumsq}\n")


        frame_entropy = -((probs * (probs+1.0e-20).log()).sum() / (B * self.num_codebooks))

        # avg_probs: (self.num_codebooks, self.codebook_size)
        avg_probs = probs.sum(0) / B
        tot_entropy = -((avg_probs * (avg_probs+1.0e-20).log()).sum() / self.num_codebooks)
        # entropy_loss > 0, and approaches 0 when tot_entropy approaches log(self.codebook_size),
        # which is its maximum possible value.
        ref_entropy = math.log(self.codebook_size)
        entropy_loss = (ref_entropy - tot_entropy) / ref_entropy

        return rel_expected_error_sumsq, entropy_loss, frame_entropy




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


    # for easier conversion into bytes, we recommend that codebook_size should
    # oalways be of the form 2**(2**n), i.e. 2, 4, 16, 256.
    # num_codebooks should always be a power of 2.
    #
    # out of codebook_size, num_codebooks = (2,(4, 16), (16, 8), (256, 4), all of which
    # give 4 bytes per 512-dimensional vector, the best reconstruction loss
    # SET SIZES:
    codebook_size = 4
    num_codebooks = 16

    quantizer = Quantizer(dim=dim, codebook_size=codebook_size, num_codebooks=num_codebooks).to(device)


    # Train quantizer.
    frame_entropy_cutoff = torch.tensor(0.3, device=device)
    entropy_scale = 0.02

    quantizer.apply_mask = False

    lr=0.005
    num_iters = 3
    for iter in range(num_iters):

        # training quantizer, not model.
        optim = torch.optim.Adam(
            quantizer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.000001
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2500, gamma=0.5)

        for i in range(10000):
            B = 600
            x = torch.randn(B, dim, device=device)
            x = model(x)  + 0.05 * x
            # x is the thing we're trying to quantize: the nnet gives it a non-trivial distribution, which is supposed to
            # emulate a typical output of a neural net.  The "+ 0.05 * x" is a kind of residual term which makes sure
            # the output is not limited to a subspace or anything too-easy like that.  Lots of networks
            # have residuals, so this is quite realistic.


            reconstruction_loss, entropy_loss, frame_entropy = quantizer.compute_loss(x)

            if i % 100 == 0:
                ref_losses = [ float('%.3f' % quantizer.compute_ref_loss(x, i).item()) for i in range(4) ]
                print(f"i={i}, ref_loss{0,1,2,3}={ref_losses}, expected_loss={reconstruction_loss.item():.3f}, "
                      f"entropy_loss={entropy_loss.item():.3f}, frame_entropy={frame_entropy.item():.3f}")


            # reconstruction_loss >= 0, equals 0 when reconstruction is exact.
            tot_loss = reconstruction_loss

            # entropy_loss approaches 0 from above, as the entropy of classes
            # approaches its maximum possible.  (this relates to diversity of
            # chosen codebook entries in classes).
            tot_loss += entropy_loss * entropy_scale

            # We want to maximize frame_entropy if it is less than frame_entropy_cutoff.
            tot_loss -= torch.minimum(frame_entropy_cutoff,
                                      frame_entropy)

            tot_loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

        print(f"... for codebook_size={quantizer.codebook_size}, num_codebooks={quantizer.num_codebooks}, frame_entropy_cutoff={frame_entropy_cutoff.item():.3f}, entropy_scale={entropy_scale}")

        if iter + 1 < num_iters:
            quantizer = quantizer.get_product_quantizer()
            frame_entropy_cutoff = frame_entropy_cutoff * 1.25
            lr *= 0.5

if __name__ == "__main__":
    _test_quantization()
