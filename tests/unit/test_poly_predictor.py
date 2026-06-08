"""Unit tests for PolyPredictor and stable_power."""

import jax
import jax.numpy as jnp
import pytest

from jaxnrsur.PolyPredictor import PolyPredictor, stable_power

jax.config.update("jax_enable_x64", True)


class TestStablePower:
    def test_values(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 2.0])
        result = stable_power(x, y)
        expected = jnp.array([1.0, 1.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_gradient(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 2.0])
        expected_grad = jnp.array([[0.0, 1.0, 4.0], [0.0, 0.0, jnp.log(2.0) * 4.0]])

        def loss(x, y):
            return jnp.sum(stable_power(x, y))

        gx, gy = jax.grad(loss, argnums=(0, 1))(x, y)
        assert jnp.allclose(gx, expected_grad[0])
        assert jnp.allclose(gy, expected_grad[1])


class TestPolyPredictor:
    def test_gradient_not_nan(self):
        coefs = jnp.zeros(1)
        bf_orders = jnp.zeros((1, 7))
        predictor = PolyPredictor(coefs, bf_orders, n_max=1)
        grad = jax.grad(predictor)(jnp.zeros((7,)))
        assert not jnp.isnan(grad).any()
