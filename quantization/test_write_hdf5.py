import h5py
import torch
from torch import nn
import numpy as np


def _write_example_data():
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
    model.eval()

    num_tensors = 2000
    filename = 'training_data.hdf5'
    hf = h5py.File(filename, 'w')

    B = 600 # batch size

    for i in range(num_tensors):
        x = torch.randn(B, dim, device=device)
        x = model(x)  + 0.05 * x
        x = x.to('cpu').detach().numpy().astype(np.float16)
        hf.create_dataset(f'dataset_{i}', data=x)

    hf.close()
    print("Wrote data to ", filename)


if __name__  == '__main__':
    _write_example_data()
