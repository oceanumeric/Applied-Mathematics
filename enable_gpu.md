# Enable GPU 

It takes me a while (almost 20 hours) to enable GPU computing. Here is some
tips. 

## Check GPU availability

`nvcc --version`

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Wed_Jun__2_19:15:15_PDT_2021
Cuda compilation tools, release 11.4, V11.4.48
Build cuda_11.4.r11.4/compiler.30033411_0
```

`nvidia-smi`

```bash
Sat Dec  3 00:05:01 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.141.03   Driver Version: 470.141.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:09:00.0 Off |                  N/A |
|  0%   49C    P8    30W / 350W |  22239MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1448      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      1683      G   /usr/bin/gnome-shell                8MiB |
|    0   N/A  N/A     74338      C   ...ematics/myvenv/bin/python    22217MiB |
+-----------------------------------------------------------------------------+
```

## Create an virtual environment

`python3 -m venv myvenv`

Then 

`source myvenv/bin/activate`

After activating the virtual environment, you can install `Pytorch` and `JAX` by
following the official guidance. 

Install jupyter kernel

`pip install --user ipykernel`

Then create a kernel within virtual environment

`python -m ipykernel install --user --name=myvenv`

## `code-runner` does not work for `JAX`

I don't know why `jupyter notebook` and `code-runner` in VScode could not
run the script, whereas the script would be run via command line without 
returning any error. 

Then I changed the running path by putting a `settings.json` into `.vscode`. 

## Never name file with the same name of packages

Very bad ideas to call a test file like `pytorch.py`

## GPU memory could be used up easily 

Check GPU usage 

`nvidia-smi`

You might need to kill process (replace YourPID)

```
sudo kill -9 YourPID
```

## Set GPU memory usage

```python
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'
``` 

### GPU memory allocation

JAX will preallocate 90% of the total GPU memory when the first JAX operation is run. Preallocating minimizes allocation overhead and memory fragmentation, but can sometimes cause out-of-memory (OOM) errors. If your JAX process fails with OOM, the following environment variables can be used to override the default behavior:

`XLA_PYTHON_CLIENT_PREALLOCATE=false`

This disables the preallocation behavior. JAX will instead allocate GPU memory as needed, potentially decreasing the overall memory usage. However, this behavior is more prone to GPU memory fragmentation, meaning a JAX program that uses most of the available GPU memory may OOM with preallocation disabled.

`XLA_PYTHON_CLIENT_MEM_FRACTION=.XX`

If preallocation is enabled, this makes JAX preallocate XX% of the total GPU memory, instead of the default 90%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.

`XLA_PYTHON_CLIENT_ALLOCATOR=platform`

This makes JAX allocate exactly what is needed on demand, and deallocate memory that is no longer needed (note that this is the only configuration that will deallocate GPU memory, instead of reusing it). This is very slow, so is not recommended for general use, but may be useful for running with the minimal possible GPU memory footprint or debugging OOM failures.

## Check your hardware

`lspci -v` 