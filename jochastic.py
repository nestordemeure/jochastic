"""
This code was written by Nestor Demeure and is licensed under the Apache 2.0 license.
You can find an up-to-date source and full description here: https://github.com/nestordemeure/jochastic
"""
import jax
import jax.numpy as jnp

def _compute_error_add(x, y, result):
    """
    Computes the floating-point error generated when computing x+y=result using the TwoSum error-free transformation.
    In infinite precision (associative maths) this function should return 0.
    WARNING: 
    - the order of the operations *matters*, do not change this operation in a way that would alter the order of operations
      our tests indicate that, at the moment (August 2022), the JAX jit-compiler preserve this operation
    - requires rounding to nearest (the default on modern processors) and assumes that floating points follow the IEEE-754 norm
    """
    y2 = result - x
    x2 = result - y2
    error_y = y - y2
    error_x = x - x2
    return error_x + error_y

def _misrounded_add(result, error):
    """
    Given the result of a floating point operation and the error associated with the operation
    returns a new result which is the result we would have obtained by rounding to the further floating point from the analytical result
    """
    # computes the direction in which the misrounded result lies
    finfo = jnp.finfo(result.dtype)
    direction = jnp.where(error > 0, finfo.max, finfo.min)
    # goes one ULP in that direction
    return jnp.nextafter(result, direction)

def _pseudorandom_bool_biased(prngKey, error, result, alternative_result):
    """
    Returns a boolean generated pseudorandomly (deterministically) from the result of a computation, 
    its numerical error and the result we would have obtained by misrounding
    the random number generator is biased according to the relative error of the addition.
    """
    # gets a random number in [0;1]
    random_unitary_float = jax.random.uniform(key=prngKey, shape=result.shape, dtype=result.dtype)
    # draws a boolean randomly, biasing the draw as a function of the ratio of the error and one ULP
    ulp = jnp.abs(alternative_result - result)
    abs_error = jnp.abs(error)
    return random_unitary_float * ulp > abs_error

def stochastic_add(prngKey, x, y, is_biased=False):
    """
    Adds x and y rounding the result randomly up or down.
    By default the rounding mode is changed half of the time.
    Set is_biased to True to bias the random generator such that a result with a low error is more likely to be rounded correctly.
    """
    # computes both the result and the result that would have been obtained with another rounding
    result = x + y 
    error = _compute_error_add(x, y, result)
    alternative_result = _misrounded_add(result, error)
    # picks the values for which we will use the other rounding
    if is_biased:
        use_result = _pseudorandom_bool_biased(prngKey, error, result, alternative_result)
    else:
        # NOTE: we do not deal with the error==0. case as it is uncommon
        use_result = jax.random.bernoulli(key=prngKey, shape=result.shape)
    # returns either the result or the misrounded result
    return jnp.where(use_result, result, alternative_result)

def _random_split_like_tree(prngKey, tree):
    """
    Takes a random number generator key and a tree, splits the key into a properly structured tree.
    credit: https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    """
    tree_structure = jax.tree_structure(tree)
    key_leaves = jax.random.split(prngKey, tree_structure.num_leaves)
    return jax.tree_unflatten(tree_structure, key_leaves)

def stochastic_tree_add(prngKey, tree_x, tree_y, is_biased=False):
    """
    Adds two pytree tree_x and tree_y rounding the result randomly up or down.
    By default the rounding mode is changed half of the time.
    Set is_biased to True to bias the random generator such that a result with a low error is more likely to be rounded correctly.
    """
    # split the key into a tree
    tree_prngKey = _random_split_like_tree(prngKey, tree_x)
    # applies the addition to all pair of leaves
    def add(x,y,prngKey): return stochastic_add(prngKey, x, y, is_biased)
    return jax.tree_util.tree_map(add, tree_x, tree_y, tree_prngKey)
