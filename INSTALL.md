To install, you can type (from this directory):
```console
python3 setup.py install
```
You can test as follows:
```console
cd quantization/
python3 test_quantization.py
python3 test_write_hdf5.py
python3 test_train_hdf5.py
```
(This test is configured to use CUDA, and will not work if you don't have a GPU available.
You can modify the code to use CPU if needed, but it will be slow.)
