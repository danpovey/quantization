import h5py
import math
import numpy as np
import torch
import random
import logging
from torch import nn
from torch import Tensor
from typing import Tuple



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
        def is_power_of_two(n: int) -> bool:
            return (n & (n-1) == 0) and n != 0
        assert is_power_of_two(codebook_size)
        assert is_power_of_two(num_codebooks)

        self.to_logits = nn.Linear(dim, codebook_size * num_codebooks)
        self.logits_scale = 4

        # we will sometimes interpret self.centers, which is of shape
        # (num_codebooks * codebook_size, dim), as being of shape
        # (num_codebooks, codebook_size, dim); and similarly with self.to_logits
        self.centers = nn.Parameter(self.to_logits.weight.detach().clone())



    def show_init_invocation(self) -> str:
        return f"quantization.Quantizer(dim={self.dim}, codebook_size={self.codebook_size}, num_codebooks={self.num_codebooks})"


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
                        new_num_codebooks).to(self.centers.device)

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
                        ans.centers[row_out,:] = self.centers[row_in1] + self.centers[row_in2]
        return ans

    def _logits(self, x: Tensor) -> Tensor:
        return self.to_logits(x) * self.logits_scale


    def encode(self,
               x: Tensor, refine_indexes_iters: int = 5,
               as_bytes: bool = True) -> Tensor:
        """
        Compute the quantized output, that can be used to reconstruct x.

        Args:
                x: the Tensor to quantize, of shape (*, dim)
           refine_indexes_iters: a number >= 0: the number of iterations to refine
                the indexes from their initial value.
        as_bytes:  if True, the quantized output will be returned as a byte
                 array, combining as many codes as possible into each bytes
                 codebook_size <= 16.

        Returns:  if as_bytes == False, a torch.LongTensor of shape (*, num_codebooks);
                  if as_bytes == True, a returns a Tensor of dtype=torch.uint8, of shape
                  (*, num_codebooks/n), where n==4 if codebook_size <= 14; or
                     2 if codebook_size <= 16, else 1.
        """
        x_reshaped = x.reshape(-1, self.dim)
        indexes = self._compute_indexes(x_reshaped, refine_indexes_iters)

        if as_bytes:
            codebook_size = self.codebook_size
            while codebook_size ** 2 <= 256:
                indexes = (indexes[:, ::2] + codebook_size * indexes[:, 1::2])
                codebook_size = codebook_size ** 2
            assert codebook_size <= 256
            indexes = indexes.to(torch.uint8)


        return indexes.reshape(*x.shape[:-1], -1)

    def _compute_indexes(self, x: Tensor, refine_indexes_iters: int = 3) -> Tensor:
        """
        Deterministically compute the indexes that encode the tensor x.

        Args:
                x: the Tensor to quantize, of shape (B, dim)
          refine_indexes_iters: a number >= 0: the number of iterations to refine
                the indexes from their initial value.

        Returns:   returns a torch.LongTensor of shape (B, num_codebooks),
              with entries in {0..codebook_size-1}
        """
        assert x.ndim == 2 and x.shape[1] == self.dim
        B = x.shape[0]
        x_reshaped = x.reshape(-1, self.dim)
        B = x_reshaped.shape[0]
        logits = self._logits(x_reshaped)
        logits = logits.reshape(B, self.num_codebooks, self.codebook_size)

        # indexes: (B, self.num_codebooks)
        indexes = torch.argmax(logits, dim=-1)
        for i in range(refine_indexes_iters):
            indexes = self._refine_indexes(x_reshaped, indexes, i)
        assert indexes.ndim == 2
        return indexes.reshape(*x.shape[:-1], self.num_codebooks)


    def _get_combinations(self, k: int, n: int, device: torch.device) -> Tensor:
        """
        Computes two matrix that represent k n-way choices.

          k: the number of separate things we have to choose
          n: the number of items we can choose (per choice)
         device: the devicde that the tensors should be on.
        Returns:
           comb_indexes: a LongTensor of shape (n**k, k), containing elements in {0..n-1}.
                     All rows are distinct.
              comb_mat: a Tensor of shape (n**k, k*(n-1)), containing elements in {0,1}.
                    Each row of comb_mat represents one row of comb_indexes, and represents
                    the same set of choices.  It is obtained by creating a matrix of zeros of
                    shape (n**k, k, n), and using scatter_ to put a 1 in the position
                    corresponding to each choice in the corresponding row of `comb_indexes`,
                    and then retaining only the last n-1 elements in dimension 2.
        """
        p = n ** k
        arange = torch.arange(p, device=device)
        powers = n ** torch.arange(k, device=device)
        comb_indexes = (arange.unsqueeze(1) / powers.unsqueeze(0)).to(torch.long) % n

        comb_one_hot = torch.zeros(p, k, n, device=device)
        src = torch.ones(1, 1, 1, device=device).expand(p, k, n)
        comb_one_hot.scatter_(src=src, dim=2, index=comb_indexes.unsqueeze(2))

        comb_mat =  comb_one_hot[:,:,1:n].contiguous().reshape(p, k*(n-1))

        return comb_indexes, comb_mat

    def _compute_diff_sumsq(self,
                            a: Tensor,
                            b: Tensor) -> Tensor:
        """
        This is utility function that computes a particular expression in an optimized
        way.

        Args:
           a: a Tensor of shape (i, 1, k, l)
           b: a Tensor of shape (i, j, 1, l)
        Returns:
           a Tensor of shape (i, j, k), that is equal to ((a + b)**2).sum(dim=-1)
        """
        assert a.ndim == 4 and a.shape[1] == 1
        assert b.ndim == 4 and b.shape[2] == 1

        a2 = (a ** 2).sum(dim=-1)   # (i, 1, k)
        b2 = (b ** 2).sum(dim=-1)   # (i, j, 1)
        b_permuted = b.permute(0, 2, 3, 1) # (i, 1, l, j)
        ab = torch.matmul(a, b_permuted)  # (i, 1, k, j)
        ab = ab.squeeze(1).transpose(1, 2) # (i, j, j)
        return a2 + b2 + 2 * ab

    def _compute_diff_sumsq2(self,
                            a: Tensor,
                            b: Tensor) -> Tensor:
        """
        This is utility function that computes a particular expression in an optimized
        way.

        Args:
           a: a Tensor of shape (1, j  k, l)
           b: a Tensor of shape (i, j, 1, l)
        Returns:
           a Tensor of shape (i, j, k), that is equal to ((a + b)**2).sum(dim=-1)
        """
        assert a.ndim == 4 and a.shape[0] == 1
        assert b.ndim == 4 and b.shape[2] == 1

        a2 = (a ** 2).sum(dim=-1)   # (1, j, k)
        b2 = (b ** 2).sum(dim=-1)   # (i, j, 1)
        b_permuted = b.permute(2, 1, 3, 0) # (1, j, l, i)
        ab = torch.matmul(a, b_permuted)  # (1, j, k, i)
        ab = ab.squeeze(0).permute(2, 0, 1) # (i, j, k)
        return a2 + b2 + 2 * ab


    def _refine_indexes(self,
                        x: Tensor,
                        indexes: Tensor,
                        i: int) -> Tensor:
        """
        Refine choices of indexes, minimizing sum-squared loss.  Note, this is not guaranteed
        not not increase the sum-squared loss, but works OK in practice.

        Args:
           x:  A Tensor of shape (B, self.dim) to be approximated.
           indexes: A Tensor of integer type, of shape (B, self.num_codebooks),
                that contains elements in {0..self.codebook_size-1}
           i: the iteration of refinement (may affect the groups we choose
               to optimize)
         Returns:  A tensor of indexes of shape (B, self.num_codebooks) that
                  will hopefully reduce the error w.r.t. x, better or at least no worse
                  than `indexes`.  This algorithm is not exact, but if the codebooks are
                  fairly orthogonal it should work fine.   If they are not fairly orthogonal
                  it may not optimize well, but hopefully the codebooks will then learn
                  to be more orthogonal.
        """
        B = indexes.shape[0]
        # indexes_expanded has shape (B, self.num_codebooks, 1, self.dim)
        indexes_expanded = indexes.unsqueeze(-1).unsqueeze(-1).expand(B, self.num_codebooks, 1, self.dim)
        # all_centers: (1, num_codebooks, codebook_size, dim)
        all_centers = self.centers.reshape(1, self.num_codebooks, self.codebook_size, self.dim)
        # centers_expanded has shape (B, self.num_codebooks, self.codebook_size, self.dim)
        centers_expanded = all_centers.expand(B, self.num_codebooks, self.codebook_size, self.dim)

        # cur_centers: (B, self.num_codebooks, 1, self.dim)
        cur_centers = torch.gather(centers_expanded, dim=2, index=indexes_expanded)
        # x_err is of shape (B, 1, 1, self.dim), it is the current error of the approximation vs. x.
        x_err = cur_centers.sum(dim=1, keepdim=True) - x.unsqueeze(1).unsqueeze(2)


        # Below, the 2 args of compute_diff_sumsq2 are:
        #  a: of shape (1, self.num_codebooks, self.codebook_size, self.dim)
        #  b: of shape (B, self.num_codebooks, 1, self.dim)
        # The answer, equivalent to ((a+b)**2).sum(dim=-1), is
        # of shape (B, self.num_codebooks, self.codebook_size).
        modified_sumsq_errs = self._compute_diff_sumsq2(all_centers,
                                                        x_err - cur_centers)

        # The above is an optimization of:
        # modified_errs = x_err - cur_centers + all_centers
        # modified_sumsq_errs = ((modified_errs ** 2).sum(dim=-1))

        # put -inf in modified_sumsq_errs in locations corresponding to the
        # current "index", to make sure the current index stays in the top-n
        # (will ensure prob at least gets no worse on any iter).
        src = torch.full((1, 1, 1), float('-inf'), device=indexes.device).expand(B, self.num_codebooks,
                                                                                 self.codebook_size)
        modified_sumsq_errs.scatter_(dim=2, index=indexes.unsqueeze(-1), src=src)


        if self.codebook_size <= 16 or (i >= 2 and i % 2 == 0):
            N = 2 # for small codebook sizes, it's sufficient to search among the top-2.
            codebooks_per_group = min(8, self.num_codebooks)
        else:
            N = 4
            codebooks_per_group = min(4, self.num_codebooks)

        assert self.num_codebooks % codebooks_per_group == 0


        # `sorted_indexes`, of shape (B, num_codebooks, codebook_size), contains the
        # indexes from best to worst for each codebook.
        _, sorted_indexes = modified_sumsq_errs.sort(dim=2)
        # topn_indexes: (B, num_codebooks, N); contains
        # n-best codebook indexes for each codebook.
        topn_indexes = sorted_indexes[:,:,:N]


        # topn1_indexes_expanded: (B, self.num_codebooks, N-1, self.dim),
        # from this point we exclude the top-1 index because it's the same as the current
        # index and will give us a zero "delta", so we can get a small speedup by
        # excluding it from our matrix multiplication.
        dim = self.dim
        topn1_indexes_expanded = (
            topn_indexes[:,:,1:].unsqueeze(-1).expand(B, self.num_codebooks,
                                                      N - 1, dim))

        # proposed_new_centers: (B, self.num_codebooks, N-1, dim)
        # containing the centers corresponding to 'proposed_indexes'
        proposed_new_centers = torch.gather(centers_expanded, dim=2,
                                            index=topn1_indexes_expanded)

        # proposed_deltas, of shape (B, num_codebooks, N-1, dim), contains the
        # change in the prediction if we were to accept this, among the top-n indexes
        # for this codebook.
        proposed_deltas = proposed_new_centers - cur_centers

        # proposed_deltas: (B, dim, num_codebooks, N-1)
        proposed_deltas = proposed_deltas.permute(0, 3, 1, 2)

        x_err_squeezed = x_err.squeeze(1) # (B, 1, dim)

        # We have to enumerate the number below, so make sure it is no greater
        # than 256.
        num_combinations = N ** codebooks_per_group
        assert num_combinations <= 256

        # comb_indexes: (num_combinations, codebooks_per_group), contains elements in {0..N-1}
        # comb_mat: (num_combinations, codebooks_per_group * (N-1)), contains elements in {0,1}
        comb_indexes, comb_mat = self._get_combinations(codebooks_per_group,
                                                        N, indexes.device)

        for begin_c in range(0, self.num_codebooks, codebooks_per_group):
            end_c = begin_c + codebooks_per_group

            this_proposed_deltas = proposed_deltas[:,:,begin_c:end_c,:].reshape(
                B, dim, codebooks_per_group*(N-1))

            # all_possible_deltas: (B, num_combinations, dim)
            all_possible_deltas = torch.matmul(this_proposed_deltas,
                                               comb_mat.t()).transpose(1,2)

            assert all_possible_deltas.shape == (B, num_combinations, dim)
            all_possible_x_errs = x_err_squeezed + all_possible_deltas  # (B, num_combinations, dim)
            all_possible_x_errs_sumsq = (all_possible_x_errs**2).sum(dim=-1) # (B, num_combinations)
            # chosen_combinations is of shape: (B,); contains elements in {0..num_combinations-1}
            chosen_combinations = (-all_possible_x_errs_sumsq).argmax(dim=1)
            assert chosen_combinations.shape == (B,)

            # selected_indexes will be of shape (B, codebooks_per_group), with
            # elements in 0..N-1.
            selected_indexes = torch.index_select(comb_indexes, dim=0,
                                                  index=chosen_combinations)
            # real_indexes: (B,codebooks_per_group), contains indexes in {0..codebook_size-1}
            real_indexes = torch.gather(topn_indexes[:,begin_c:end_c,:], #(B,codebooks_per_group,N)
                                        dim=2,
                                        index=selected_indexes.unsqueeze(-1)).squeeze(1)

            # Replace selected elements of `indexes` with elements of `real_indexes`
            indexes[:,begin_c:end_c] = real_indexes.squeeze(-1)

            if begin_c + N < self.num_codebooks:
                # not the last iter.. must keep x_err_squeezed updated.

                # chosen_combinations_reshaped: (B, 1, dim), contains elements in {0..num_combinations-1}
                chosen_combinations_reshaped = chosen_combinations.unsqueeze(1).unsqueeze(2).expand(B, 1, dim)
                # selected_deltas contains the changes in x_approx that we selected.
                # Its shape is (B, 1, dim).
                selected_deltas = torch.gather(all_possible_deltas, dim=1,
                                               index=chosen_combinations_reshaped)
                x_err_squeezed += selected_deltas


        assert indexes.ndim == 2
        return indexes


    def _maybe_separate_indexes(self, indexes: Tensor) -> Tensor:
        """
        This reverses the process done in encode() if as_bytes==True, which combines
        multiple codebook entries into a single byte if self.codebook_size is small
        enough.
            Args:
                 indexes: an integer tensor of shape (B, n) where n divides
                       self.num_codebooks
           Returns: a tensor of the same type as `indexes`, of shape (B,
                  self.num_codebooks)
        """
        B = indexes.shape[0]
        if indexes.shape[-1] != self.num_codebooks:
            n = indexes.shape[-1]
            num_repeats = self.num_codebooks // n
            assert num_repeats in [2, 4, 8, 16] and self.num_codebooks == n * num_repeats
            indexes = indexes.unsqueeze(2).expand(B, n, num_repeats)
            size = self.codebook_size
            indexes = (indexes // (size ** torch.arange(num_repeats,
                                                        device=indexes.device))) % size
            indexes = indexes.reshape(B, self.num_codebooks)
        assert indexes.shape == (B, self.num_codebooks)
        return indexes



    def decode(self, indexes: Tensor) -> Tensor:
        """
        Does the (approximate) inverse of _compute_indexes(): constructs from `indexes` the
        corresponding approximated tensor.
        Args:
             indexes:
                    May be an integer tensor of shape (*, self.num_codebooks), with entries
                    in {0..self.num_codebooks-1}
                    May also contain multiple codebook entries combined into one integer, as
                    done by encode() with as_bytes==True; in this case the last dim
                    might be self.num_codebooks/2 or self.num_codebooks/4.
        Returns: a tensor of shape (*, self.dim), consisting of the sum of the specified
                cluster centers.
        """
        orig_shape = indexes.shape
        indexes = indexes.reshape(-1, indexes.shape[-1])
        indexes = self._maybe_separate_indexes(indexes).to(dtype=torch.int64)

        assert indexes.ndim == 2
        B = indexes.shape[0]
        # indexes_expanded: (num_codebooks, B, dim)
        indexes_expanded = indexes.transpose(0, 1).contiguous().unsqueeze(-1).expand(self.num_codebooks, B, self.dim)
        # to_output_reshaped: (num_codebooks, codebook_size, dim)
        to_output_reshaped = self.centers.reshape(self.num_codebooks, self.codebook_size, self.dim)
        # chosen_codebooks: (num_codebooks, B, dim).
        chosen_codebooks = torch.gather(to_output_reshaped, dim=1, index=indexes_expanded)

        # x_approx: (B, dim), this is the sum of the chosen rows of `to_output`
        # corresponding to the chosen codebook entries, this would correspond to
        # the approximated x.
        x_approx = chosen_codebooks.sum(dim=0)
        return x_approx.reshape(*orig_shape[:-1], self.dim)


    def compute_loss(self, x: Tensor, refine_indexes_iters: int = 0) -> Tensor:
        """
        Compute various parts of the loss function.

        Args:
            x: the Tensor to quantize, of shape (*, dim)
           refine_indexes_iters: a number >= 0: the number of iterations to refine
                the indexes from their initial value.

        Returns: (rel_reconstruction_loss, logprob_loss, entropy_loss, index_entropy_loss), where
             rel_reconstruction_loss:  a scalar torch.Tensor containing the relative sum-squared
                    reconstruction loss, based on the indexes chosen after `refine_indexes_iters`
                    iterations of refinement after the argmax of the logits.  This loss is
                    is the sum-squared of (x - reconstructed_x) / sum-squared of x, which
                    for already-trained models will be between 0 and 1, but could be greater than 1
                    at the start of training.
             logprob_loss: the negative average logprob of the selected classes (i.e. those
                   selected after refine_indexes_iters of refinement).  This is added to the
                   loss function, so we can select reasonable classes before refining the indexes.
             logits_entropy_loss: the class entropy loss, from the logits, which approaches
                   zero when all classes of all codebooks are equi-probable (in the logits output).
             index_entropy_loss: the class entropy loss, from the computed indexes,  which approaches
                  zero when all classes of all codebooks are equi-probable (in the indexes output).
                  Not differentiable but useful for diagnostics.
        """
        x = x.reshape(-1, self.dim)
        indexes = self._compute_indexes(x, refine_indexes_iters)
        x_approx = self.decode(indexes)
        # tot_error: (B, dim), the error of the approximated vs. real x.
        tot_error = x_approx - x
        rel_reconstruction_loss = (tot_error**2).sum() / ((x ** 2).sum() + 1.0e-20)

        # Get logprob loss and class-entropy loss
        # wasteful.. already computed logits..
        logits = self._logits(x).reshape(-1, self.num_codebooks, self.codebook_size)
        logits = logits.log_softmax(dim=2)
        # chosen_logits: (B, num_codebooks, 1)
        chosen_logits = torch.gather(logits, dim=2,
                                     index=indexes.unsqueeze(2))
        logprob_loss = -chosen_logits.mean()

        # class_entropy
        B = x.shape[0]
        counts = torch.zeros(B, self.num_codebooks, self.codebook_size, device=x.device)
        ones = torch.ones(1, 1, 1, device=x.device).expand(B, self.num_codebooks, self.codebook_size)
        counts.scatter_(src=ones, dim=2, index=indexes.unsqueeze(2))
        avg_counts = counts.mean(dim=0) + 1.0e-20
        index_entropy = -(avg_counts * avg_counts.log()).sum(dim=1).mean()

        probs = logits.exp().mean(dim=0) + 1.0e-20
        logits_entropy = -(probs * probs.log()).sum(dim=1).mean()
        ref_entropy = math.log(self.codebook_size)

        logits_entropy_loss = (ref_entropy - logits_entropy) / ref_entropy
        index_entropy_loss = (ref_entropy - index_entropy) / ref_entropy

        return rel_reconstruction_loss, logprob_loss, logits_entropy_loss, index_entropy_loss


class _WithDerivOf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return x

    @staticmethod
    def backward(ctx, ans_deriv):
        return None, ans_deriv # deriv goes to y only.

def with_deriv_of(x: Tensor, y: Tensor)  -> Tensor:
    """ Returns x but its deriv gets passed to y in backprop.. """
    return _WithDerivOf.apply(x, y)

class QuantizerTrainer(object):
    def __init__(self,
                 dim: int,
                 bytes_per_frame: int,
                 device: torch.device,
                 phase_one_iters: int = 10000,
                 phase_two_iters: int = 20000,
                 lr: float = 0.005):
        """
        Args:
            dim: The feature dimension we are trying to quantize, e.g. 512
         bytes_per_frame:  The number of bytes to use to quantize each vector of
                `dim` values.
           device: The device to use for training
         phase_one_iters:  The number of iterations to use for the first
               phase of training (with codebook_size=16); after this we
               will convert to have codebook_size=256.  These parameters were
               tuned with a batch size of 600: if your batch size (in frames)
               is smaller than this you may benefit from a larger phase_one_iters and a
               smaller learning rate.
               [Also, note: phase_one_iters should be larger for larger dims;
               for dim=256 and batch_size=600, 10k was enough, but for
               dim=512 and batch_size=600, 20k was better.
         phase_two_iters:  The number of iterations to use for the second
               phase of training (with codebook_size=256)
          lr: The initial learning rate.

        This object trains a Quantizer.  You can use it as follows:

          trainer = QuantizerTrainer(...)
          while not trainer.done():
             # let x be some tensor of shape (*, dim), that you will train on
             # (should not be the same on each minibatch)
             trainer.step(x)
          quantizer = trainer.get_quantizer()
        """
        super(QuantizerTrainer, self).__init__()
        assert bytes_per_frame in [1,2,4,8,16,32]

        # We'll initially train with codebook_size=16 and
        # num_codebooks=bytes_per_frame * 2, then after `phase_one_iters` of
        # training will multiply pairs of codebooks so codebook_size=256 and
        # num_codebooks=bytes_per_frame

        self.phase_one_iters = phase_one_iters
        self.phase_two_iters = phase_two_iters
        self.cur_iter = 0
        self.lr = lr
        self.two_iter_prob = 0.5

        self.quantizer = Quantizer(dim=dim, codebook_size=16,
                                   num_codebooks=bytes_per_frame*2).to(device)
        self._init_optimizer()


    def _init_optimizer(self):
        self.optim = torch.optim.Adam(
            self.quantizer.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1.0e-06
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                         step_size=(self.phase_one_iters
                                                                    if self.cur_iter == 0
                                                                    else self.phase_two_iters)/4,
                                                         gamma=0.5)

    def done(self) -> bool:
        return self.cur_iter > self.phase_one_iters + self.phase_two_iters

    def step(self, x: torch.Tensor) -> None:
        """
        Does one step of training.  You must call this at least 2*phase_one_iters
        iterations.
        Args:
              x: a Tensor of shape (*, dim) containing the frames of data we are
                 trying to accurately encode.
        """
        x = x.reshape(-1, self.quantizer.dim)

        num_iters = 2 if random.random() < self.two_iter_prob else 1
        (reconstruction_loss, logprob_loss,
         logits_entropy_loss, index_entropy_loss) = self.quantizer.compute_loss(x, num_iters)


        if self.cur_iter % 200 == 0:
            det_losses = [ float('%.3f' % self.quantizer.compute_loss(x, j)[0].item())
                           for j in range(6) ]
            phase = 1 if self.cur_iter <= self.phase_one_iters else 2
            i = self.cur_iter - self.phase_one_iters if phase > 1 else self.cur_iter
            # Caution: python's logging level is logging.ERROR by default.  To make the following
            # be printed, you may have to do:
            #  import logging
            #  logging.getLogger().setLevel(logging.INFO)
            # before using this code.
            logging.info(f"phase={phase}/2, iter={i}, "
                         f"dim,nc,csz={self.quantizer.dim},{self.quantizer.num_codebooks},{self.quantizer.codebook_size}, "
                         f"loss_per_iter={det_losses}, "
                         f"logprob_loss={logprob_loss.item():.3f}, "
                         f"logits_entropy_loss={logits_entropy_loss.item():.3f}, "
                         f"index_entropy_loss={index_entropy_loss.item():.3f}")

        entropy_scale = 0.0
        # About the losses:
        # - reconstruction_loss >= 0; it equals 0 when reconstruction is exact.
        #   This is the main loss function, used to train quantizer.centers
        # - logprob_loss trains only quantizer.to_logits, which predicts the
        #   indexes after refinement, so we can initialize them well; it does
        #   not affect the cluster centers.
        # - logits_entropy_loss is currently not used for training, since we
        #   set entropy_scale = 0 above.  It would affect only to_logits, if
        #   used.  The intention was that this might solve problem with
        #   cluster centers having very uneven probabilities of being chosen
        #   (it would do this by biasing the initial choice, relying on
        #   the inexactness of the search).  In our experiments,
        #   logits entropy_loss and index_entropy_loss both end up
        #   less than 0.05, so this does not seem to be a problem in practice,
        #   but it might be a problem if, say, the inputs had a very tiny scale,
        #   so we are keeping the code around.
        # - index_entropy_loss is not differentiable; we have
        #   added it only for diagnostic purposes.  It reflects the entropy of
        #   the distribution over classes, after refining the cluster indexes.
        #   It was computed just in case regularizing logits_entropy_loss was
        #   not enough to affect the final distribution over cluster centers,
        #   so we could diagnose the problem; but we found no problem in practice.
        #

        tot_loss = (reconstruction_loss +
                    logprob_loss +
                    logits_entropy_loss * entropy_scale)
        # We want to maximize frame_entropy if it is less than frame_entropy_cutoff.
        #tot_loss -= torch.minimum(self.frame_entropy_cutoff,
        #                          frame_entropy)

        tot_loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        self.scheduler.step()

        if self.cur_iter == self.phase_one_iters:
            self._begin_second_phase()
        self.cur_iter += 1

    def _begin_second_phase(self):
        """
        This is to be called exactly once, when self.cur_iter reaches self.phase_one_iters
        """
        self.quantizer = self.quantizer.get_product_quantizer()
        self.lr *= 0.5
        self._init_optimizer()

    def get_quantizer(self) -> Quantizer:
        assert self.cur_iter >= 2 * self.phase_one_iters
        return self.quantizer



def read_hdf5_data(filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads the hdf5 archive in the file with name 'filename' into a single
    numpy array of size (tot_frames, dim), shuffles the frames, and returns it
    as a numpy array.  The type will be the same as it was in the archive (e.g. float16).

    Args:
        filename: the name of the filename of your hdf5 archive.  It should
        have been created using code similar to the code in test_write_hdf5.py,
        e.g. something like:

          hf = h5py.File(filename, 'w')
          for i in range(...):
            # get x as some numpy array of type np.float16, and shape (*, dim)
            # the name does not actually matter, except that they should be distinct.
            hf.create_dataset(f'dataset_{i}', data=x)

     Returns (train, valid), where:
          train: a torch.Tensor of shape (tot_train_frames, dim), on CPU, with
                  dtype=torch.float16, with shuffled rows.
          valid: a torch.Tensor of shape (tot_valid_frames, dim), on CPU, with
                  dtype=torch.float16, with shuffled rows (these are distinct
                  frames from those in `train`, but may derive from diffrent
                  rows of the same original tensors.)

    Caution: you should set the logger to INFO level, with:
      logging.getLogger().setLevel(logging.INFO)
    if you want to see the logging output of this function.

    """
    logging.info(f"Opening file {filename}")
    hf = h5py.File(filename, 'r')
    tot_frames = 0
    dim = -1

    def get_num_frames(shape):
        # Returns product of shape[0],shape[1],...,shape[-2]
        num_frames = 1
        for i in shape[:-1]:
            num_frames *= i
        return num_frames

    for key in hf.keys():
        dset = hf[key]
        shape = list(dset.shape)
        if dim == -1:
            dim = shape[-1]
        else:
            assert dim == shape[-1], "Dataset must have consistent dimension (last element of shape"
        tot_frames += get_num_frames(shape)
    logging.info(f"read_data: tot_frames = {tot_frames}")

    ans = np.empty((tot_frames, dim), dtype=np.float16)
    cur_pos = 0
    for key in hf.keys():
        array = hf[key][:] # [:] gets it as NumPy array (I believe).
        array = np.ascontiguousarray(array).reshape(-1, dim)
        num_frames = array.shape[0]
        ans[cur_pos:cur_pos+num_frames,:] = array
        cur_pos += num_frames
    assert cur_pos == tot_frames

    # Shuffle the rows of ans.
    np.random.shuffle(ans)
    ans_torch = torch.from_numpy(ans)

    valid_proportion = 0.05
    valid_frames = valid_proportion * tot_frames
    if valid_frames > 10000:
        valid_frames = 10000
    train_frames = tot_frames - valid_frames
    logging.info(f"read_data: train_frames={train_frames}, valid_frames={valid_frames}")

    # return (train, valid)
    return ans_torch[valid_frames:tot_frames], ans_torch[:valid_frames]
