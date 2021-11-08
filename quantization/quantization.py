import math
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

        self.to_logits = nn.Linear(dim, codebook_size * num_codebooks)
        self.logits_scale = 4

        # we will sometimes interpret self.centers, which is of shape
        # (num_codebooks * codebook_size, dim), as being of shape
        # (num_codebooks, codebook_size, dim); and similarly with self.to_logits
        self.centers = nn.Parameter(self.to_logits.weight.detach().clone())



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
               x: Tensor, refine_indexes_iters: int = 3,
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

    def _compute_indexes(self, x: Tensor, refine_indexes_iters: int = 2) -> Tensor:
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
        for _ in range(refine_indexes_iters):
            indexes = self._refine_indexes(x_reshaped, indexes)
        assert indexes.ndim == 2
        return indexes.reshape(*x.shape[:-1], self.num_codebooks)


    def _get_all_permutations(self, n: int, device: torch.device) -> Tensor:
        """
        For a number n, returns a Tensor of float and shape (2**n, n) whose rows are all
        distinct combinations of 0.0 and 1.0.
        """
        p = 2 ** n
        arange = torch.arange(p, device=device)
        powers = 2 ** torch.arange(n, device=device)
        ans = ((arange.unsqueeze(1) / powers.unsqueeze(0)).to(torch.int32) % 2 == 0).to(torch.float)
        return ans

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
        all_centers = self.centers.reshape(1, self.num_codebooks, self.codebook_size, self.dim)
        # centers_expanded has shape (B, self.num_codebooks, self.codebook_size, self.dim)
        centers_expanded = all_centers.expand(B, self.num_codebooks, self.codebook_size, self.dim)

        # cur_centers: (B, self.num_codebooks, 1, self.dim)
        cur_centers = torch.gather(centers_expanded, dim=2, index=indexes_expanded)
        # x_err is of shape (B, 1, 1, self.dim), it is the current error of the approximation vs. x.
        x_err = cur_centers.sum(dim=1, keepdim=True) - x.unsqueeze(1).unsqueeze(2)

        # TODO: get modified_neg_sumsq_errs by a more efficient expression.

        # Below, the 2 args of compute_diff_sumsq2 are:
        #  a: of shape (1, self.num_codebooks, self.codebook_size, self.dim)
        #  b: of shape (B, self.num_codebooks, 1, self.dim)
        # The answer, equivalent to ((a+b)**2).sum(dim=-1), is
        # of shape (B, self.num_codebooks, self.codebook_size).
        modified_neg_sumsq_errs = -self._compute_diff_sumsq2(all_centers,
                                                             x_err - cur_centers)

        # The above is an optimization of:
        # modified_errs = x_err - cur_centers + all_centers
        # modified_neg_sumsq_errs = -((modified_errs ** 2).sum(dim=-1))

        # put -infinity in modified_neg_sumsq_errs in locations corresponding to the current "index",
        # to disallow them when searching for the best alternative to the current index.
        src = torch.full((B, self.num_codebooks, self.codebook_size), float('-inf'),
                         device=indexes.device)
        modified_neg_sumsq_errs.scatter_(dim=2, index=indexes.unsqueeze(-1), src=src)

        # proposed_indexes contains the best alternative to the index in 'indexes'
        proposed_indexes = modified_neg_sumsq_errs.argmax(dim=2) # (B, num_codebooks)


        # proposed_indexes_expanded: (B, self.num_codebooks, 1, self.dim)
        proposed_indexes_expanded = (
            proposed_indexes.unsqueeze(-1).unsqueeze(-1).expand(B, self.num_codebooks, 1, self.dim))

        # proposed_new_centers: (B, self.num_codebooks, 1, self.dim), the same
        # shape as cur_centers but containing the centers corresponding to 'proposed_indexes'
        proposed_new_centers = torch.gather(centers_expanded, dim=2, index=proposed_indexes_expanded)

        # proposed_deltas, of shape (B, num_codebooks, 1, dim), contains the
        # change in the prediction if we were to accept the best alternative index for
        # each codebook.
        proposed_deltas = proposed_new_centers - cur_centers
        # proposed_deltas: (B, dim, num_codebooks)
        proposed_deltas = proposed_deltas.transpose(1, 3).reshape(B, self.dim,
                                                                  self.num_codebooks)
        x_err_squeezed = x_err.squeeze(1) # (B, 1, dim)

        N = 8  # because 2**8==256 is OK to enumerate; 2**16 is not.

        for begin_c in range(0, self.num_codebooks, N):
            end_c = min(self.num_codebooks, begin_c + N)
            this_num_c = end_c - begin_c

            # perm: (2**this_num_c, this_num_c)
            perm = self._get_all_permutations(this_num_c,
                                              indexes.device)

            this_proposed_deltas = proposed_deltas[:,:,begin_c:end_c]

            # all_possible_deltas: (B, 2**this_num_c, dim)
            all_possible_deltas = torch.matmul(this_proposed_deltas,
                                               perm.t()).transpose(1,2)

            assert all_possible_deltas.shape == (B, 2**this_num_c, self.dim)
            all_possible_x_errs = x_err_squeezed + all_possible_deltas  # (B, 2**this_num_c, dim)
            all_possible_x_errs_sumsq = (all_possible_x_errs**2).sum(dim=-1) # (B, 2**this_num_c)
            perm_rows = (-all_possible_x_errs_sumsq).argmax(dim=1) # (B,)
            assert perm_rows.shape == (B,)

            # selected_perm_rows will be of shape (B, this_num_c), with a 1
            # in each position where we want to select the proposed change.
            selected_perm_rows = torch.index_select(input=perm.to(torch.int32),
                                                    dim=0,
                                                    index=perm_rows)
            assert selected_perm_rows.shape == (B, this_num_c)

            # Replace selected elements of `indexes` with elements of `proposed_indexes`
            indexes[:,begin_c:end_c] = ((proposed_indexes[:,begin_c:end_c] * selected_perm_rows) +
                                        (indexes[:,begin_c:end_c] * (1 - selected_perm_rows)))

            if begin_c + N < self.num_codebooks:
                # not the last iter.. must keep x_err_squeezed updated.
                # perm_rows_reshaped: (B, 1, dim), contains elements in {0..2**this_num_c-1}
                perm_rows_reshaped = perm_rows.unsqueeze(1).unsqueeze(2).expand(B, 1, self.dim)
                # selected_deltas contains the changes in x_approx that we selected.
                # Its shape is (B, 1, dim).
                selected_deltas = torch.gather(all_possible_deltas, dim=1,
                                               index=perm_rows_reshaped)
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


    def compute_loss_deterministic(self, x: Tensor, refine_indexes_iters: int = 0) -> Tensor:
        """
        Compute the loss function with deterministic indexes using argmax not sampling.
        Like compute_loss(), the result is differentiable w.r.t self.centers but
        gives no derivative for self.to_logits

        Args:
                x: the Tensor to quantize, of shape (*, dim)
           refine_indexes_iters: a number >= 0: the number of iterations to refine
                the indexes from their initial value.

        Returns:   a scalar torch.Tensor containing the relative sum-squared
                    reconstruction loss.
                    It is the sum-squared of (x - reconstructed_x) / sum-squared of x, which will
                    for already-trained models be between 0 and 1, but could be greater than 1
                    at the start of training.
        """
        x = x.reshape(-1, self.dim)
        indexes = self._compute_indexes(x, refine_indexes_iters)
        x_approx = self.decode(indexes)
        # tot_error: (B, dim), the error of the approximated vs. real x.
        tot_error = x_approx - x
        rel_reconstruction_loss = (tot_error**2).sum() / ((x ** 2).sum() + 1.0e-20)
        return rel_reconstruction_loss

    def compute_loss(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute three (potential) parts of the loss function.  This version of the loss function
        involves an expectation over class probabilities; see also compute_loss_deterministic(),
        which uses fixed class probabilities, but can't give you a derivative for self.logits.

        Args:
                x: the Tensor to quantize, of shape (*, dim)
        Returns (reconstruction_loss, entropy_loss, frame_entropy), where:
           reconstruction_loss: a scalar torch.Tensor containing the relative sum-squared
                     reconstruction loss, constructed as an expectation over class probs.
                     It is the sum-squared of (x - reconstructed_x) / sum-squared of x, which will
                     for already-trained models be between 0 and 1, but could be greater than 1
                     at the start of training.
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
        indexes = torch.distributions.categorical.Categorical(probs=probs).sample()
        # indexes is of shape (B, self.num_codebooks) and contains elements in [0..codebook_size - 1]


        # to_output_reshaped: (num_codebooks, codebook_size, dim)
        to_output_reshaped = self.centers.reshape(self.num_codebooks, self.codebook_size, self.dim)
        # indexes_expanded: (num_codebooks, B, dim)
        indexes_expanded = indexes.transpose(0, 1).contiguous().unsqueeze(-1).expand(self.num_codebooks, B, self.dim)

        # chosen_codebooks: (num_codebooks, B, dim).
        chosen_codebooks = torch.gather(to_output_reshaped, dim=1, index=indexes_expanded)

        # tot_codebooks: (1, B, dim), this is the sum of the chosen rows of `to_output` corresponding
        # to the chosen codebook entries, this would correspond to the approximated x.
        tot_codebooks = chosen_codebooks.sum(dim=0, keepdim=True)
        # tot_error: (1, B, dim), the error of the approximated vs. real x.
        tot_error = tot_codebooks - x.reshape(1, B, self.dim)
        # tot_error_sumsq: scalar, total squared error.  only needed for diagnostics.
        tot_error_sumsq = (tot_error**2).sum()

        # The two args to _compute_diff_sumsq() below are:
        #    a, of shape: (num_codebooks, 1, B, dim)
        #    b, of shape: (num_codebooks, codebook_size, 1, dim)
        # .. and the answer, which is equivalent to ((a+b)**2).sum(dim-1), is of shape
        #    (num_codebooks, codebook_size, B)
        # alt_error_sumsq answers the question: "what if, for this particular codebook, we had chosen this
        # codebook entry; what would the sum-squared error be then?"
        alt_error_sumsq = self._compute_diff_sumsq((tot_error - chosen_codebooks).unsqueeze(1),
                                                   to_output_reshaped.unsqueeze(2))

        # expected_error_sumsq is like tot_error_sumsq, but replaces the
        # discrete choice with an expectation of the sum-sq error, taken over each codebook
        # while leaving the choices of all the other codebooks fixed.
        expected_error_sumsq = (alt_error_sumsq * probs.permute(1, 2, 0)).sum() - (tot_error_sumsq * (self.num_codebooks - 1))

        x_tot_sumsq = (x ** 2).sum() + 1.0e-20

        rel_tot_error_sumsq = tot_error_sumsq / x_tot_sumsq
        rel_expected_error_sumsq = expected_error_sumsq / x_tot_sumsq

        frame_entropy = -((probs * (probs+1.0e-20).log()).sum() / (B * self.num_codebooks))

        # avg_probs: (self.num_codebooks, self.codebook_size)
        avg_probs = probs.sum(0) / B
        tot_entropy = -((avg_probs * (avg_probs+1.0e-20).log()).sum() / self.num_codebooks)
        # 0 <= entropy_loss <= 1; it approaches 0 when tot_entropy approaches
        # ref_entropy = log(self.codebook_size), which is its maximum possible value.
        ref_entropy = math.log(self.codebook_size)
        entropy_loss = (ref_entropy - tot_entropy) / ref_entropy

        return rel_expected_error_sumsq, entropy_loss, frame_entropy


class QuantizerTrainer(object):
    def __init__(self,
                 dim: int,
                 bytes_per_frame: int,
                 device: torch.device,
                 phase_one_iters: int = 20000,
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
        self.cur_iter = 0
        self.lr = lr
        self.frame_entropy_cutoff = torch.tensor(0.3, device=device)
        self.entropy_scale = 0.02
        self.det_loss_scale = 0.95  # should be in [0..1]


        self.quantizer = Quantizer(dim=dim, codebook_size=16,
                                   num_codebooks=bytes_per_frame*2).to(device)
        self._init_optimizer()


    def _init_optimizer(self):
        self.optim = torch.optim.Adam(
            self.quantizer.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1.0e-06
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                         step_size=self.phase_one_iters/4,
                                                         gamma=0.5)

    def done(self) -> bool:
        return self.cur_iter >= 2 * self.phase_one_iters

    def step(self, x: torch.Tensor) -> None:
        """
        Does one step of training.  You must call this at least 2*phase_one_iters
        iterations.
        Args:
              x: a Tensor of shape (*, dim) containing the frames of data we are
                 trying to accurately encode.
        """
        x = x.reshape(-1, self.quantizer.dim)

        reconstruction_loss, entropy_loss, frame_entropy = self.quantizer.compute_loss(x)

        det_loss = self.quantizer.compute_loss_deterministic(x, 1)

        if self.cur_iter % 100 == 0:
            det_losses = [ float('%.3f' % self.quantizer.compute_loss_deterministic(x, j).item())
                           for j in range(4) ]
            phase = 1 if self.cur_iter <= self.phase_one_iters else 2
            i = self.cur_iter - self.phase_one_iters if phase > 1 else self.cur_iter
            # Caution: python's logging level is logging.ERROR by default.  To make the following
            # be printed, you may have to do:
            #  import logging
            #  logging.getLogger().setLevel(logging.INFO)
            # before using this code.
            logging.info(f"phase={phase}/2, iter={i}, det_loss(0,1,..)={det_losses}, "
                        f"expected_loss={reconstruction_loss.item():.3f}, "
                        f"entropy_loss={entropy_loss.item():.3f}, frame_entropy={frame_entropy.item():.3f}")

        # reconstruction_loss >= 0, equals 0 when reconstruction is exact.
        tot_loss = (reconstruction_loss * (1 - self.det_loss_scale) +
                    det_loss * self.det_loss_scale +
                    entropy_loss * self.entropy_scale)
        # We want to maximize frame_entropy if it is less than frame_entropy_cutoff.
        tot_loss -= torch.minimum(self.frame_entropy_cutoff,
                                  frame_entropy)

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
        self.frame_entropy_cutoff = self.frame_entropy_cutoff * 1.25
        self.lr *= 0.5
        self._init_optimizer()

    def get_quantizer(self) -> Quantizer:
        assert self.cur_iter >= 2 * self.phase_one_iters
        return self.quantizer
