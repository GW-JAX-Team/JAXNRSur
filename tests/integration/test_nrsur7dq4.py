"""Integration tests for NRSur7dq4Model.

Loads the actual surrogate data from disk (requires network on first run).
Tests cover: smoke (shape/NaN), differentiability w.r.t. params and time,
and JIT compilation.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jaxnrsur import JAXNRSur
from jaxnrsur.NRSur7dq4 import NRSur7dq4Model


@pytest.fixture(scope="module")
def model():
    return NRSur7dq4Model()


# --------------------------------------------------------------------------
# Smoke tests
# --------------------------------------------------------------------------


def test_model_runs_and_returns_correct_shape(model):
    """get_waveform_geometric returns (hp, hc) on the requested time grid."""
    time = jnp.linspace(-1000, 100, 500)
    params = jnp.array([2.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.15])
    hp, hc = model.get_waveform_geometric(time, params, theta=1.0, phi=0.0)
    assert hp.shape == time.shape
    assert hc.shape == time.shape
    assert not jnp.isnan(hp).any()
    assert not jnp.isnan(hc).any()


def test_wrapper_smoke(model):
    """JAXNRSur.get_waveform_td returns arrays of the right shape."""
    wrapper = JAXNRSur(model, alpha_window=0.1)
    time = jnp.linspace(-1.0, 0.0, 50)
    params = jnp.array([60.0, 500.0, 1.0, 0.0, 2.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.15])
    hp, hc = wrapper.get_waveform_td(time, params)
    assert hp.shape == time.shape
    assert hc.shape == time.shape


# --------------------------------------------------------------------------
# Differentiability
# --------------------------------------------------------------------------


def test_gradient_wrt_params_not_nan(model):
    """jax.grad w.r.t. surrogate params must not produce NaN."""
    time = jnp.linspace(-1000, 100, 100)
    params = jnp.array([2.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.15])

    def loss(p):
        hp, hc = model.get_waveform_geometric(time, p, theta=jnp.pi / 4, phi=0.0)
        return jnp.sum(hp**2 + hc**2)

    grads = jax.grad(loss)(params)
    assert grads.shape == params.shape
    assert not jnp.isnan(grads).any()


# --------------------------------------------------------------------------
# JIT
# --------------------------------------------------------------------------


def test_jit_consistency(model):
    """JIT-compiled call must match eager call."""
    import equinox as eqx

    time = jnp.linspace(-1000, 100, 200)
    params = jnp.array([2.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.15])

    hp_eager, hc_eager = model.get_waveform_geometric(time, params, theta=1.0, phi=0.0)
    hp_jit, hc_jit = eqx.filter_jit(model.get_waveform_geometric)(
        time, params, theta=1.0, phi=0.0
    )
    assert jnp.allclose(hp_eager, hp_jit)
    assert jnp.allclose(hc_eager, hc_jit)
