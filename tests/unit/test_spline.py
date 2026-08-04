import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline as ScipyCubicSpline

from jaxnrsur.Spline import CubicSpline, CubicSplineFactorization


@pytest.fixture(scope="module")
def spline_fixture():
    x = jnp.linspace(0, 1, 5)
    y = x**2
    return CubicSpline(x, y)


def test_cubic_spline_basic(spline_fixture):
    # Check exact interpolation at grid points
    x_grid = jnp.linspace(0, 1, 5)
    y_grid = x_grid**2
    y_interp_grid = spline_fixture(x_grid)
    assert jnp.allclose(y_interp_grid, y_grid, atol=1e-8)

    # Check reasonable accuracy at off-grid points (looser tolerance)
    x_test = jnp.linspace(0, 1, 10)
    y_interp = spline_fixture(x_test)
    y_true = x_test**2
    max_err = jnp.max(jnp.abs(y_interp - y_true))
    print("Max error off-grid:", max_err)
    assert max_err < 0.03  # Allow a small error for natural cubic spline


def test_cubic_spline_grad(spline_fixture):
    def spline_sum(x):
        return jnp.sum(spline_fixture(x))

    grad_fn = jax.grad(spline_sum)
    x_test = jnp.linspace(0, 1, 10)
    grad_val = grad_fn(x_test)
    assert grad_val.shape == x_test.shape


def test_cubic_spline_vmap(spline_fixture):
    x_batch = jnp.stack([jnp.linspace(0, 1, 10), jnp.linspace(0, 1, 10) + 0.1])
    vmap_fn = eqx.filter_vmap(spline_fixture)
    y_batch = vmap_fn(x_batch)
    assert y_batch.shape == x_batch.shape


def test_cubic_spline_jit(spline_fixture):
    x_test = jnp.linspace(0, 1, 10)
    jit_fn = eqx.filter_jit(spline_fixture)
    y_jit = jit_fn(x_test)
    y_ref = spline_fixture(x_test)
    assert jnp.allclose(y_jit, y_ref)


def test_reused_factorization_matches_default():
    x_grid = jnp.linspace(0, 1, 20)
    x_test = jnp.linspace(0, 1, 50)
    factorization = CubicSplineFactorization(x_grid)
    for y_grid in (jnp.sin(x_grid), jnp.exp(x_grid)):
        expected = ScipyCubicSpline(
            np.asarray(x_grid), np.asarray(y_grid), bc_type="natural"
        )(np.asarray(x_test))
        actual = CubicSpline(x_grid, y_grid, factorization)(x_test)
        assert jnp.allclose(actual, expected, atol=1e-8)
        default = CubicSpline(x_grid, y_grid)(x_test)
        assert jnp.allclose(default, expected, atol=1e-8)


def test_reused_factorization_grad_and_vmap():
    x_grid = jnp.linspace(0, 1, 20)
    x_test = jnp.linspace(0, 1, 50)
    factorization = CubicSplineFactorization(x_grid)

    def interpolate(y_grid):
        return CubicSpline(x_grid, y_grid, factorization)(x_test)

    y_batch = jnp.stack([jnp.sin(x_grid), jnp.exp(x_grid)])
    values = jax.jit(jax.vmap(interpolate))(y_batch)
    gradient = jax.grad(lambda y: jnp.sum(interpolate(y)))(y_batch[0])
    assert values.shape == (2, 50)
    assert gradient.shape == x_grid.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_factorization_shape_mismatch_raises():
    factorization = CubicSplineFactorization(jnp.linspace(0, 1, 20))
    x_grid = jnp.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="shape"):
        CubicSpline(x_grid, x_grid**2, factorization)


def test_factorization_value_mismatch_raises():
    factorization = CubicSplineFactorization(jnp.linspace(0, 1, 20))
    x_grid = jnp.linspace(0, 2, 20)  # same shape as factorization, different values
    with pytest.raises(eqx.EquinoxRuntimeError, match="values"):
        CubicSpline(x_grid, x_grid**2, factorization)


def test_factorization_rejects_non_finite_grid():
    x_grid = jnp.array([0.0, 1.0, jnp.nan, 3.0])
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite"):
        CubicSplineFactorization(x_grid)


def test_factorization_rejects_duplicate_grid_points():
    x_grid = jnp.array([0.0, 1.0, 1.0, 2.0])
    with pytest.raises(eqx.EquinoxRuntimeError, match="increasing"):
        CubicSplineFactorization(x_grid)


def test_factorization_rejects_direction_changing_grid():
    x_grid = jnp.array([0.0, 1.0, 0.5, 2.0])
    with pytest.raises(eqx.EquinoxRuntimeError, match="increasing"):
        CubicSplineFactorization(x_grid)


def test_factorization_rejects_duplicate_grid_points_under_jit():
    x_grid = jnp.array([0.0, 1.0, 1.0, 2.0])
    build = eqx.filter_jit(lambda x: CubicSplineFactorization(x))
    with pytest.raises(eqx.EquinoxRuntimeError, match="increasing"):
        build(x_grid)


def test_factorization_rejects_direction_changing_grid_under_jit():
    x_grid = jnp.array([0.0, 1.0, 0.5, 2.0])
    build = eqx.filter_jit(lambda x: CubicSplineFactorization(x))
    with pytest.raises(eqx.EquinoxRuntimeError, match="increasing"):
        build(x_grid)
