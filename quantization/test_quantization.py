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



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _test_quantizer_trainer()
    _test_quantizer_trainer_double()
