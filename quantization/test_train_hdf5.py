import h5py
import logging
import numpy as np
import torch
from torch import nn
from torch import Tensor
from quantization import read_hdf5_data, Quantizer, QuantizerTrainer


def _test_train_from_file():
    train, valid = read_hdf5_data('training_data.hdf5')
    dim = train.shape[1]

    device = torch.device('cuda')

    B = 512  # Minibatch size, this is very arbitrary, it's close to what we used
             # when we tuned this method.
    def minibatch_generator(data: Tensor,
                            repeat: bool):
        assert 3 * B < data.shape[0]
        cur_offset = 0
        while (True if repeat else cur_offset + B <= data.shape[0]):
            start = cur_offset % (data.shape[0] + 1 - B)
            end = start + B
            cur_offset += B
            yield data[start:end,:].to(device).to(dtype=torch.float)

    trainer = QuantizerTrainer(dim=dim, bytes_per_frame=4,
                               phase_one_iters=10000,
                               device=device)

    for x in minibatch_generator(train, repeat=True):
        trainer.step(x)
        if trainer.done():
            break

    quantizer_fn = 'quantizer.pt'

    quantizer = trainer.get_quantizer()
    print(f"You can load the module with: {quantizer.show_init_invocation()}")
    torch.save(quantizer.state_dict(), quantizer_fn)

    quantizer2 = Quantizer(dim=dim, num_codebooks=4, codebook_size=256)
    quantizer2.load_state_dict(torch.load(quantizer_fn))
    quantizer2 = quantizer2.to(device)

    valid_count = 0
    tot_rel_err = 0
    for x in minibatch_generator(valid, repeat=False):
        x_approx = quantizer2.decode(quantizer2.encode(x))
        tot_rel_err += ((x-x_approx)**2).sum() / (x**2).sum()
        valid_count += 1
    print(f"Validation average relative error: {tot_rel_err/valid_count}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    _test_train_from_file()
