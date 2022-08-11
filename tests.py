import jax
import jax.numpy as jnp
import jochastic

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
error = jax.jit(jochastic._compute_error)(x, y, result)
print(f"x + y: {result} ({result.dtype})")
print(f"error: {error} ({error.dtype})")

# checks misrounded addition
alternativeResult = jax.jit(jochastic._misround_result)(result, error)
print(f"result: {result}")
print(f"alternative: {alternativeResult} ({alternativeResult.dtype})")
print(f"difference: {result - alternativeResult}")

# demonstrate the stochastic additions
key, keysto = jax.random.split(key)
result_sto = jax.jit(jochastic.add, static_argnames=['is_biased'])(keysto, x, y, is_biased=False)
print(f"stochastic addition: {result_sto} ({result_sto.dtype})")
difference = result - result_sto
print(f"difference: {difference}")

key, keysto = jax.random.split(key)
result_bia = jax.jit(jochastic.add, static_argnames=['is_biased'])(keysto, x, y, is_biased=True)
print(f"biased stochastic addition: {result_bia} ({result_bia.dtype})")
difference = result - result_bia
print(f"difference: {difference}")

# test mixed precision addition
key, keysto = jax.random.split(key)
y_high = y.astype(jnp.float32)
result_stom = jax.jit(jochastic.add, static_argnames=['is_biased'])(keysto, x, y_high, is_biased=False)
print(f"mixed precision stochastic addition: {result_stom} ({result_stom.dtype})")
difference = result - result_stom
print(f"difference: {difference}")

# tests the tree addition
tree_x = [x, y, x]
tree_y = [y, x, y]
key, keysto = jax.random.split(key)
tree_result = jax.jit(jochastic.tree_add, static_argnames=['is_biased'])(keysto, tree_x, tree_y, is_biased=False)
print(f"tree stochastic addition: {tree_result}")
