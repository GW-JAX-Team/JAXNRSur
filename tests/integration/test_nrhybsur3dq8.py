"""Integration tests for NRHybSur3dq8Model.

Loads the actual surrogate data from disk (requires network on first run).
Tests cover: smoke (shape/NaN/dtype), model attributes, JIT, vmap, and
differentiability w.r.t. params and time.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxnrsur import JAXNRSur
from jaxnrsur.NRHybSur3dq8 import NRHybSur3dq8Model


@pytest.fixture(scope="module")
def model():
    return NRHybSur3dq8Model()


@pytest.fixture(scope="module")
def time():
    return jnp.linspace(-1000, 100, 1000)


@pytest.fixture(scope="module")
def params():
    # [q, chi1z, chi2z]
    return jnp.array([1.5, 0.3, -0.2])


# --------------------------------------------------------------------------
# Smoke / shape / dtype
# --------------------------------------------------------------------------


def _check_hphc(hp, hc, shape):
    assert isinstance(hp, jnp.ndarray) and isinstance(hc, jnp.ndarray)
    assert hp.shape == shape
    assert hc.shape == shape
    assert jnp.issubdtype(hp.dtype, jnp.floating)
    assert jnp.issubdtype(hc.dtype, jnp.floating)
    assert not jnp.isnan(hp).any()
    assert not jnp.isnan(hc).any()


def test_basic_call(model, time, params):
    hp, hc = model(time, params)
    _check_hphc(hp, hc, time.shape)


def test_model_attributes(model):
    assert isinstance(model, NRHybSur3dq8Model)
    assert model.n_modes > 0
    assert hasattr(model, "data")
    assert hasattr(model, "harmonics")


def test_wrapper_smoke(model):
    wrapper = JAXNRSur(model, alpha_window=0.1)
    time = jnp.linspace(-1.0, 0.0, 50)
    params = jnp.array([60.0, 500.0, 1.0, 0.0, 1.5, 0.3, -0.2])
    hp, hc = wrapper.get_waveform_td(time, params)
    assert hp.shape == time.shape
    assert hc.shape == time.shape


# --------------------------------------------------------------------------
# JIT and vmap
# --------------------------------------------------------------------------


def test_jit(model, time, params):
    hp_eager, hc_eager = model(time, params)
    hp_jit, hc_jit = eqx.filter_jit(model)(time, params)
    assert jnp.allclose(hp_eager, hp_jit)
    assert jnp.allclose(hc_eager, hc_jit)


def test_vmap_over_params(model, time, params):
    """vmap over a batch of 5 parameter sets must return (5, n_time) output."""
    params_batch = jnp.repeat(params[None, :], 5, axis=0)
    hp_batch, hc_batch = eqx.filter_jit(
        eqx.filter_vmap(model.get_waveform_geometric, in_axes=(None, 0, None, None))
    )(time, params_batch, 0.0, 0.0)
    _check_hphc(hp_batch, hc_batch, (5, len(time)))


# --------------------------------------------------------------------------
# Differentiability
# --------------------------------------------------------------------------


def test_gradient_wrt_params(model, time, params):
    def loss(p):
        hp, hc = model(time, p)
        return jnp.sum(hp) + jnp.sum(hc)

    grads = jax.grad(loss)(params)
    assert grads.shape == params.shape
    assert not jnp.isnan(grads).any()


def test_gradient_wrt_time(model, time, params):
    def loss(t):
        hp, hc = model(t, params)
        return jnp.sum(hp) + jnp.sum(hc)

    grads = jax.grad(loss)(time)
    assert grads.shape == time.shape
    assert not jnp.isnan(grads).any()
