import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import os
import sys


# check executable path
print(sys.executable)
# set memory allocation 
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'


key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

size = 100
x = random.normal(key, (size, size), dtype=jnp.float32)
print(jnp.dot(x, x.T).block_until_ready()) # runs on the GPU


import torch

print(torch.cuda.is_available())

print(torch.cuda.get_device_properties(0).total_memory)

print(torch.cuda.mem_get_info(0))


