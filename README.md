# Jochastic: stochastically rounded operations between JAX tensors.

This repository contains a JAX software-based implementation of some [stochastically rounded operations](https://nhigham.com/2020/07/07/what-is-stochastic-rounding/).

When encoding the weights of a neural network in low precision (such as `bfloat16`), one runs into stagnation problems: updates end up being too small relative to the numbers the precision of the encoding.
This leads to weights becoming stuck and the model's accuracy being significantly reduced.

Stochastic arithmetic lets you perform the operations in such a way that the weights have a non-zero probability of being modified anyway.
This avoids the stagnation problem (see [figure 4 of "Revisiting BFloat16 Training"](https://arxiv.org/abs/2010.06192)) without increasing the memory usage (as might happen if one were using a [compensated summation](https://github.com/nestordemeure/pairArithmetic) to solve the problem).

The downside is that software-based stochastic arithmetic is significantly slower than normal floating-point arithmetic.
It is thus viable for things like the weight update (when using the output of an [Optax](https://github.com/deepmind/optax) optimizer for example) but would not be appropriate in a hot loop.

Do not hesitate to submit an issue or a pull request if you need added functionalities for your needs!

## Usage

This repository introduces the `add` and `tree_add` operations.
They take a [PRNGkey](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.PRNGKey.html) and two tensors (or pytree respectively) to be added but round the result up or down randomly:

```python
import jax
import jax.numpy as jnp
import jochastic

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
result_sto = jochastic.add(key, x, y)
print(f"stochastic addition: {result_sto} ({result_sto.dtype})")
difference = result - result_sto
print(f"difference: {difference}")
```

Both functions take an optional `is_biased` boolean parameter.
If `is_biased` is `True` (the default value), the random number generator is biased according to the relative error of the operation
else, it will round up half of the time on average.

Jitting the functions is left to the user's discretion (you will need to indicate that `is_biased` is static).

**NOTE:**
Very low precision (16 bits floating-point arithmetic or less) is *extremely* brittle.
We recommend using higher precision locally (such as 32 bits) *then* cast down to 16 bits at summing / storage time ([something that Pytorch does transparently when using their `addcdiv` in low precision](https://github.com/pytorch/pytorch/blob/12382f0a38f8199bc74aee701465e847f368e6de/aten/src/ATen/native/cuda/PointwiseOpsKernel.cu?fbclid=IwAR0SdS6mVAGN0TB_TAdKt0WVWWjxiBkmP6Inj9lYH8oB68wjsbQzinlH-xY#L92)).
Both functions will accept mixed-precision inputs (adding a high precision number to a low precision), use that information for the rounding then return the *lowest* precision of their input (contrary to most casting conventions)

## Implementation details

We use `TwoSum` to measure the numerical error done by the addition, our tests show that it behaves as needed on `bfloat16` (some edge cases might be invalid, leading to an inexact computation of the numerical error but, it is reliable enough for our purpose).

This and the [`nextafter`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nextafter.html) function let us emulate various rounding modes in software (this is inspired by [Verrou's backend](https://github.com/edf-hpc/verrou)).

## Crediting this work

You can use this BibTeX reference if you use Jochastic within a published work:

```bibtex
@misc{Jochastic,
  author = {Nestor, Demeure},
  title = {Jochastic: stochastically rounded operations between JAX tensors.},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nestordemeure/jochastic}}
}
```

You will find a Pytorch implementation called StochasTorch [here](https://github.com/nestordemeure/stochastorch).
