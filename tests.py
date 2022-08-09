import jax
import jax.numpy as jnp
from jochastic import _compute_error_add, _misrounded_add, stochastic_add, stochastic_tree_add

# problem definition
size = 10
dtype = jnp.bfloat16
seed = 1993
key = jax.random.PRNGKey(seed)

# deterministic addition
key, keyx = jax.random.split(key)
x = jax.random.normal(keyx, shape=(size,), dtype=dtype)
key, keyy = jax.random.split(key)
y = jax.random.normal(keyy, shape=(size,), dtype=dtype)
result = x + y

# test error computation
error = jax.jit(_compute_error_add)(x, y, result)
print(f"x + y: {result} ({result.dtype})")
print(f"error: {error} ({error.dtype})")

# checks misrounded addition
alternativeResult = jax.jit(_misrounded_add)(result, error)
print(f"result: {result}")
print(f"alternative: {alternativeResult} ({alternativeResult.dtype})")
print(f"difference: {result - alternativeResult}")

# demonstrate the stochastic additions
key, keysto = jax.random.split(key)
result_sto = jax.jit(stochastic_add, static_argnames=['is_biased'])(keysto, x, y, is_biased=False)
print(f"stochastic addition: {result_sto} ({result_sto.dtype})")
difference = result - result_sto
print(f"difference: {difference}")

key, keysto = jax.random.split(key)
result_bia = jax.jit(stochastic_add, static_argnames=['is_biased'])(keysto, x, y, is_biased=True)
print(f"biased stochastic addition: {result_bia} ({result_bia.dtype})")
difference = result - result_bia
print(f"difference: {difference}")

# tests the tree addition
tree_x = [x, y, x]
tree_y = [y, x, y]
key, keysto = jax.random.split(key)
tree_result = jax.jit(stochastic_tree_add, static_argnames=['is_biased'])(keysto, tree_x, tree_y, is_biased=False)
print(f"tree stochastic addition: {tree_result}")
