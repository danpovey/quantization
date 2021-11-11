# Trainable multi-codebook quantization

This repository implements a utility for use with PyTorch, and ideally GPUs, for training an
efficient quantizer based on multiple single-byte codebooks.   The prototypical scenario
is that you have some distribution over vectors in some space, say, of dimension 512, that
might come from a neural net embedding, and you want a means of encoding a vector into
a short sequence of bytes (say, 4 or 8 bytes) that can be used to reconstruct the
vector with minimal expected loss, measured as squared distance, i.e. squared l2 loss.


This repository provides Quantizer object that lets you do this quantization, and
an associated QuantizerTrainer object that you can use to train the Quantizer.
For example, you might invoke the QuantizerTrainer with 20,000 minibatches
of vectors.



## Usage

#### Installation

```shell script
python3 setup.py install
```


#### Example

```python
import torch
import quantization

trainer = quantization.QuantizerTrainer(dim=256, bytes_per_frame=4,
                                        device=torch.device('cuda'))
while not trainer.done():
   # let x be some tensor of shape (*, dim), that you will train on
   # (should not be the same on each minibatch)
   trainer.step(x)
quantizer = trainer.get_quantizer()

# let x be some tensor of shape (*, dim)..
encoded = quantizer.encode(x)  # (*, 4), dtype=uint8
x_approx = quantizer.decode(quantizer.encode(x))
```

To avoid versioning issues and so on, it may be easier to just include quantization.py
in your repository directly (and add its requirements to your requirements.txt).
