import h5py
import logging
import numpy as np
import torch
from torch import nn



def _test_train_from_file():
    all_data = read_training_data('training_data.hdf5')
    print("shape = ", all_data.shape)
    print("dtype = ", all_data.dtype)

    return

    num_tensors = 1000
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _test_train_from_file()
