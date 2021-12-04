import h5py
import logging
import numpy as np
import torch
from torch import nn
from torch import Tensor
from quantization import read_hdf5_data, Quantizer, QuantizerTrainer
from prediction import JointCodebookPredictor

def _test_train_from_file():
    train, valid = read_hdf5_data('training_data.hdf5')
    dim = train.shape[1]

    device = torch.device('cuda')

    # bytes_per_frame is the key thing you might want to tune: e.g. try 2 or 8
    # or 16.
    bytes_per_frame = 4

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


    trainer = QuantizerTrainer(dim=dim,
                               bytes_per_frame=bytes_per_frame,
                               device=device)

    for x in minibatch_generator(train, repeat=True):
        trainer.step(x)
        if trainer.done():
            break

    # You could also put quantizer.get_id() in the filename if you want.
    quantizer_fn = 'quantizer.pt'

    quantizer = trainer.get_quantizer()
    print(f"You can load the module with: {quantizer.show_init_invocation()}")
    torch.save(quantizer.state_dict(), quantizer_fn)

    quantizer2 = Quantizer(dim=dim, num_codebooks=4, codebook_size=256)
    quantizer2.load_state_dict(torch.load(quantizer_fn))
    quantizer2 = quantizer2.to(device)
    x_mean = quantizer2.get_data_mean()

    assert quantizer2.get_id() == quantizer.get_id()
    print(f"Quantizer id is {quantizer.get_id()}")

    valid_count = 0
    tot_rel_err = 0

    for x in minibatch_generator(valid, repeat=False):
        x_approx = quantizer2.decode(quantizer2.encode(x))
        tot_rel_err += ((x-x_approx)**2).sum() / ((x-x_mean) ** 2).sum()
        valid_count += 1

    print(f"Validation average relative error: {tot_rel_err/valid_count:.5f}")

    # shannon rate-distortion equation-- applicable to Gaussian noise only--
    # says [rate = R, distortion = D]:
    # R  =  1/2 log_2(sigma_x^2 / D)
    # -> solving for D as a function of R,
    # D = sigma_x^2 / (2 ** (2 * R)) = 1 / (2 ** (2 * R)) = 2 ** -(2 * R)
    rate = bytes_per_frame * 8 / dim
    shannon_distortion = 2 ** -(2 * rate)
    print(f"For reference, Shannon distortion rate for is {shannon_distortion:.5f}.\n"
          f"To the extent that the average relative error is lower than this,\n"
          f"it means that the data is easier to compress than if it\n"
          f"were a Gaussian with a spherical covariance matrix.")

def _test_joint_predictor():
    train, valid = read_hdf5_data('training_data.hdf5')
    dim = train.shape[1]

    device = torch.device('cuda')
    quantizer_fn = 'quantizer.pt'
    quantizer = Quantizer(dim=dim, num_codebooks=4, codebook_size=256)
    quantizer.load_state_dict(torch.load(quantizer_fn))
    quantizer = quantizer.to(device)

    # bytes_per_frame is the key thing you might want to tune: e.g. try 2 or 8
    # or 16.
    bytes_per_frame = 4

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

    predictor = JointCodebookPredictor(predictor_dim=dim,
                                       num_codebooks=bytes_per_frame,
                                       self_prediction=True).to(device)

    optim = torch.optim.Adam(
        predictor.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9, weight_decay=1.0e-06
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                step_size=2000,
                                                gamma=0.5)

    count = 0

    x_noise_level = 0.0
    for x in minibatch_generator(train, repeat=True):
        x = x.to(device)
        encoding = quantizer.encode(x + x_noise_level * torch.randn_like(x))
        tot_logprob, tot_count = predictor(x, encoding) # should be easy to predict encoding from x.

        loss = -(tot_logprob / tot_count)
        if count % 200 == 0:
            logging.info(f"Iter={count}, loss = {loss.item():.3f}")
        loss.backward()
        optim.step()
        scheduler.step()
        count += 1
        if count > 10000:
            break


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    #_test_train_from_file()
    _test_joint_predictor()
