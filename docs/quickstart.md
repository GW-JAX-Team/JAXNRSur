# Quick Start

## Basic Usage

To generate a gravitational-wave waveform, instantiate a model and call it with a time array and a parameter vector:

```python
import jax
jax.config.update("jax_enable_x64", True)  # required for accurate surrogate evaluation

import jax.numpy as jnp
from jaxnrsur.NRHybSur3dq8 import NRHybSur3dq8Model

# Time grid in geometric units (units of total mass M)
time = jnp.linspace(-1000, 100, 100000)

# params = [q, chi1z, chi2z]
model = NRHybSur3dq8Model()
hp, hc = model(time, jnp.array([1.5, 0.1, -0.1]))
```

Surrogate data files are downloaded from Zenodo on first use and cached locally — this only happens once.

Switching to the precessing model only requires changing the import and parameter vector:

```python
from jaxnrsur.NRSur7dq4 import NRSur7dq4Model

# params = [q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z]
model = NRSur7dq4Model()
hp, hc = model(time, jnp.array([1.5, 0.0, 0.3, 0.1, 0.0, 0.2, -0.1]))
```

## GPU and Gradient Support

JAXNRSur models are [Equinox](https://github.com/patrick-kidger/equinox) modules, so they work out of the box with `equinox.filter_jit`, `equinox.filter_grad`, and `equinox.filter_vmap`:

```python
import equinox as eqx

# JIT-compile for fast repeated evaluation
fast_model = eqx.filter_jit(model)

# Compute gradient w.r.t. mass ratio and spins
def total_power(params):
    hp, hc = model(time, params)
    return jnp.sum(hp**2 + hc**2)

grad = eqx.filter_jit(eqx.filter_grad(total_power))(jnp.array([1.5, 0.1, -0.1]))
```

GPU execution requires no code changes — JAX will automatically use the GPU if one is available.
See the [Installation](installation.md) page for GPU setup.
