import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax


print(jax.default_backend())
print(jax.devices())


key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)
