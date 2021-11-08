import logging
import math
from quantization import Quantizer, QuantizerTrainer
import random
import torch
from torch import nn
from torch import Tensor
from typing import Tuple


def _test_quantizer_trainer():
    print("Testing dim=256")
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
    trainer = QuantizerTrainer(dim=256, bytes_per_frame=4,
                               phase_one_iters=10000,
                               device=torch.device('cuda'))

    B = 600
    def generate_x():
        x = torch.randn(B, dim, device=device)
        return model(x)  + 0.05 * x


    while not trainer.done():
        trainer.step(generate_x())

    quantizer = trainer.get_quantizer() # of type Quantizer

    k = 30
    avg_rel_err = 0
    for i in range(k):
        x = generate_x()
        x_approx = quantizer.decode(quantizer.encode(x))
        avg_rel_err += (1/k) * ((x-x_approx)**2).sum() / (x**2).sum()

    print("Done testing dim=256, avg relative approximation error = ", avg_rel_err.item())

def _test_quantizer_trainer_double():
    # doubled means, 2 copies of the same distribution; we do this
    # so we can compare the loss function with the loss in
    # _test_quantizer_trainer()
    # (if we just created a network with larger dim, we would be
    # changing the problem and wouldn't know how to compare)
    print("Testing dim=512, doubled...")
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
    trainer = QuantizerTrainer(dim=512, bytes_per_frame=8,
                               device=torch.device('cuda'),
                               phase_one_iters=20000)
    B = 600
    while not trainer.done():
        x1 = torch.randn(B, dim, device=device)
        x2 = torch.randn(B, dim, device=device)
        x1 = model(x1)  + 0.05 * x1
        x2 = model(x2)  + 0.05 * x2
        x = torch.cat((x1, x2), dim=1)
        trainer.step(x)
    print("Done testing dim=512, doubled...")


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
    # SET SIZES:
    # We start with codebook_size, num_codebooks = (4, 16), but after training
    # the model we expand it to (16, 8), the train more, then expand to
    # (256, 4), then train more.
    codebook_size = 16
    num_codebooks = 8

    quantizer = Quantizer(dim=dim, codebook_size=codebook_size,
                          num_codebooks=num_codebooks).to(device)


    # Train quantizer.
    frame_entropy_cutoff = torch.tensor(0.3, device=device)
    entropy_scale = 0.02
    det_loss_scale = 0.95  # should be in [0..1]

    lr=0.005
    while True:
        # we'll break from this loop when quantizer.codebook_size >= 256.

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

            det_loss = quantizer.compute_loss_deterministic(x, 1)

            if i % 100 == 0:
                if i % 200 == 0:
                    det_losses = [ float('%.3f' % quantizer.compute_loss_deterministic(x, j).item())
                                   for j in range(4) ]
                else:
                    det_losses = []
                    for j in range(4):
                        indexes = quantizer.encode(x, j, as_bytes=False)
                        if random.random()  < 0.1:
                            # Test the as_bytes option.
                            indexes2 = quantizer.encode(x, j, as_bytes=True)
                            assert indexes2.dtype == torch.uint8
                            indexes_ref = quantizer._maybe_separate_indexes(indexes2)
                            assert torch.all(indexes_ref == indexes)

                        x_err = quantizer.decode(indexes) - x
                        rel_err = (x_err**2).sum() / ((x**2).sum() + 1e-20)
                        rel_err = float('%.3f' % rel_err.item())
                        det_losses.append(rel_err)

                print(f"i={i}, det_loss(0,1,..)={det_losses}, expected_loss={reconstruction_loss.item():.3f}, "
                      f"entropy_loss={entropy_loss.item():.3f}, frame_entropy={frame_entropy.item():.3f}")


            # reconstruction_loss >= 0, equals 0 when reconstruction is exact.
            tot_loss = reconstruction_loss * (1 - det_loss_scale)

            tot_loss += det_loss * det_loss_scale

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

        print(f"... for codebook_size={quantizer.codebook_size}, num_codebooks={quantizer.num_codebooks}, "
              f"frame_entropy_cutoff={frame_entropy_cutoff.item():.3f}, entropy_scale={entropy_scale}, "
              f"det_loss_scale={det_loss_scale}")

        if quantizer.codebook_size >= 256:
            break
        quantizer = quantizer.get_product_quantizer()
        #frame_entropy_cutoff = frame_entropy_cutoff * 1.25
        lr *= 0.5

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _test_quantizer_trainer_double()
    _test_quantizer_trainer()
    _test_quantization()
