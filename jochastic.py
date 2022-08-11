"""
Stochastically rounded operations between JAX tensors.

This code was written by Nestor Demeure and is licensed under the Apache 2.0 license.
You can find an up-to-date source and full description here: https://github.com/nestordemeure/jochastic
"""
import jax
import jax.numpy as jnp

#----------------------------------------------------------------------------------------
# BUILDING BLOCKS

def _compute_error(x, y, result):
    """
    Computes the error introduced during a floating point addition (x+y=result) using the TwoSum error-free transformation.
    In infinite precision (associative maths) this function should return 0.

    WARNING: 
    - the order of the operations *matters*, do not change this operation in a way that would alter the order of operations
    - requires rounding to nearest (the default on modern processors) and assumes that floating points follow the IEEE-754 norm 
      (but, it has been tested with alternative types such as bfloat16)
    """
    # NOTE: computing this quantity via a cast to higher precision would be faster for low precisions
    y2 = result - x
    x2 = result - y2
    error_y = y - y2
    error_x = x - x2
    return error_x + error_y

def _misround_result(result, error):
    """
    Given the result of a floating point operation and the numerical error introduced during that operation
    returns the floating point number on the other side of the interval containing the analytical result of the operation.

    NOTE: the output of this function will be of the type of result, the type of error does not matter.
    """
    # computes the direction in which the misrounded result lies
    finfo = jnp.finfo(result.dtype)
    direction = jnp.where(error > 0, finfo.max, finfo.min)
    # goes one ULP in that direction
    return jnp.nextafter(result, direction)

def _pseudorandom_bool(prngKey, result, alternative_result, error, is_biased=True):
    """
    Takes  the result of a floating point operation, 
    the floating point number on the other side of the interval containing the analytical result of the operation
    and the numerical error introduced during that operation
    returns a randomly generated boolean.

    If is_biased is True, the random number generator is biased according to the relative error of the operation
    else, it will round up 50% of the time and down the other 50%.
    """
    if is_biased:
        # gets a random number in [0;1]
        random_unitary_float = jax.random.uniform(key=prngKey, shape=result.shape, dtype=result.dtype)
        # draws a boolean randomly, biasing the draw as a function of the ratio of the error and one ULP
        ulp = jnp.abs(alternative_result - result)
        abs_error = jnp.abs(error)
        result = random_unitary_float * ulp > abs_error
    else:
        # NOTE: we do not deal with the error==0 case as it is too uncommon to bias the results significantly
        result = jax.random.bernoulli(key=prngKey, shape=result.shape)
    return result

#----------------------------------------------------------------------------------------
# OPERATIONS

def add(prngKey, x, y, is_biased=True):
    """
    Returns the sum of two tensors x and y pseudorandomly rounded up or down to the nearest representable floating-point number.

    If is_biased is True, the random number generator is biased according to the relative error of the addition
    else, it will round up 50% of the time and down the other 50%.
    """
    assert(x.dtype == y.dtype)
    # computes both the result and the result that would have been obtained with another rounding
    result = x + y 
    error = _compute_error(x, y, result)
    alternative_result = _misround_result(result, error)
    # picks the values for which we will use the other rounding
    use_result = _pseudorandom_bool(prngKey, result, alternative_result, error, is_biased)
    return jnp.where(use_result, result, alternative_result)

#----------------------------------------------------------------------------------------
# TREE OPERATIONS

def _random_split_like_tree(prngKey, tree):
    """
    Takes a random number generator key and a tree, splits the key into a properly structured tree.
    credit: https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    """
    tree_structure = jax.tree_structure(tree)
    key_leaves = jax.random.split(prngKey, tree_structure.num_leaves)
    return jax.tree_unflatten(tree_structure, key_leaves)

def tree_add(prngKey, tree_x, tree_y, is_biased=True):
    """
    Returns the sum of two pytree tree_x and tree_y pseudorandomly rounded up or down to the nearest representable floating-point number.

    If is_biased is True, the random number generator is biased according to the relative error of the addition
    else, it will round up 50% of the time and down the other 50%.
    """
    # split the key into a tree
    tree_prngKey = _random_split_like_tree(prngKey, tree_x)
    # applies the addition to all pair of leaves
    def add_leaf(prngKey, x, y): return add(prngKey, x, y, is_biased)
    return jax.tree_util.tree_map(add_leaf, tree_prngKey, tree_x, tree_y)
