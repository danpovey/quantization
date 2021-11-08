Type (from this directory):
python3 setup.py install
To test:
cd quantization/
python3 test_quantization.py
(This test is configured to use CUDA, and will not work if you don't have a GPU available.
You can modify the code to use CPU