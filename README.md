# Jochastic: a Pytorch implementation of stochastic addition

This repository contains a JAX software-based implementation of [stochastic rounding](https://nhigham.com/2020/07/07/what-is-stochastic-rounding/) addition.

When encoding the weights of a neural network in low precision (such as `bfloat16`), one runs into stagnation problems: updates end up being too small relative to the numbers the precision of the encoding.
This leads to weights becoming stuck and the model's accuracy being significantly reduced.

Stochastic arithmetic lets you perform the addition in such a way that the weights have a non-zero probability of being modified anyway.
This avoids the stagnation problem (see [figure 4 of "Revisiting BFloat16 Training"](https://arxiv.org/abs/2010.06192)) without increasing the memory usage (as might happen if one were using a [compensated summation](https://github.com/nestordemeure/pairArithmetic) to solve the problem).

The downside is that software-based stochastic arithmetic is significantly slower than a normal floating-point addition.
It is thus viable for the weight update (when using the output of an [Optax](https://github.com/deepmind/optax) optimizer for example) but would not be appropriate in a hot loop.

Do not hesitate to submit an issue or a pull request if you need added functionalities for your needs!

## Usage

This repository introduces the `stochastic_add` and `stochastic_tree_add` functions which can be used to perform a stochastically rounded addition:

```python
import jax
import jax.numpy as jnp
from jochastic import stochastic_add

# problem definition
size = 10
dtype = jnp.bfloat16
key = jax.random.PRNGKey(1993)

# deterministic addition
key, keyx = jax.random.split(key)
x = jax.random.normal(keyx, shape=(size,), dtype=dtype)
key, keyy = jax.random.split(key)
y = jax.random.normal(keyy, shape=(size,), dtype=dtype)
result = x + y
print(f"deterministic addition: {result}")

# stochastic addition
result_sto = stochastic_add(key, x, y)
print(f"stochastic addition: {result_sto} ({result_sto.dtype})")
difference = result - result_sto
print(f"difference: {difference}")
```

`stochastic_add` takes three inputs:

* `PRNGkey` a key for the JAX random number generator
* `x` a tensor to add
* `y` a tensor to add
* `is_biased` an optional boolean (which defaults to false), setting it to true will make the code slower but more accurate by biasing the random generator such that a result with a low error is more likely to be rounded correctly (otherwise the rounding mode is flipped 50% of the time).

The function will return the sum of `x` and `y`. `stochastic_tree_add` takes the same inputs but expects `x` and `y` to be pytrees.

## Implementation details

We use `TwoSum` to measure the numerical error done by the addition, our tests show that it behaves as needed on `bfloat16` (some edge cases might be invalid, leading to an inexact computation of the numerical error but, it is reliable enough for our purpose).

This and the [`nextafter`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nextafter.html) function let us emulate various rounding modes in software (this is inspired by [Verrou's backend](https://github.com/edf-hpc/verrou)).

Jitting the functions is left to the user's discretion.

## Crediting this work

Please use this reference if you use Stochastorch within a published work:

```bibtex
@misc{Jochastic,
  author = {Nestor, Demeure},
  title = {Jochastic: a Pytorch implementation of stochastic addition},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nestordemeure/jochastic}}
}
```

You will find a Pytorch implementation called StochasTorch [here](https://github.com/nestordemeure/stochastorch).
