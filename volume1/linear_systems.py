import os
import jax
import jax.numpy as jnp

# set the fraction of GPU 
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

x = jnp.arange(10)
print(x)