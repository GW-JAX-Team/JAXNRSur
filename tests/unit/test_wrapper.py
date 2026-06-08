"""Unit tests for the JAXNRSur wrapper using a dummy surrogate model.

No surrogate data is loaded — the DummyWaveformModel returns simple
sinusoids so we can test the wrapper logic in isolation.
"""

import jax.numpy as jnp
import pytest

from jaxnrsur import C_SI, MPC_SI, RSUN_SI, DataLoader, JAXNRSur, WaveformModel
from jaxtyping import Array, Float


class _DummyDataLoader(DataLoader):
    def __init__(self):
        self.sur_time = jnp.linspace(0, 100, 1000)
        self.modes = [(2, 2), (2, 1), (2, 0)]


class _DummyWaveformModel(WaveformModel):
    def __init__(self):
        self.data = _DummyDataLoader()

    def get_waveform_geometric(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_param"],
        theta: float,
        phi: float,
        omega_lower: float = 0.0,
    ):
        return jnp.sin(time), jnp.cos(time)


@pytest.fixture
def wrapper():
    return JAXNRSur(
        model=_DummyWaveformModel(),
        alpha_window=0.1,
        segment_length=4.0,
        sampling_rate=4096,
    )


def test_get_waveform_td_shape(wrapper):
    time = jnp.linspace(0, 1, 100)
    params = jnp.array([30.0, 100.0, 0.3, 0.2, 1.1, 0.5, 0.2])
    hp, hc = wrapper.get_waveform_td(time, params)
    assert hp.shape == time.shape
    assert hc.shape == time.shape


def test_get_waveform_td_scaling(wrapper):
    """Output should be sinusoid * window * (M * RSUN / d_L / MPC) scaling."""
    time = jnp.linspace(0, 1, 100)
    params = jnp.array([30.0, 100.0, 0.3, 0.2, 1.1, 0.5, 0.2])
    hp, hc = wrapper.get_waveform_td(time, params)

    mtot, dist_mpc = params[0], params[1]
    const = mtot * RSUN_SI / dist_mpc / MPC_SI
    time_m = time * C_SI / RSUN_SI / mtot

    # Window uses the duration of the requested time array (not the surrogate range)
    Tcoorb = time_m[-1] - time_m[0]
    w_start = time_m[0]
    w_end = w_start + wrapper.alpha_window * Tcoorb
    x = (time_m - w_start) / (w_end - w_start)
    window = jnp.select(
        [time_m < w_start, time_m > w_end],
        [0.0, 1.0],
        default=x**3 * (10 + x * (6 * x - 15)),
    )
    assert jnp.allclose(hp, jnp.sin(time_m) * window * const, atol=1e-6)
    assert jnp.allclose(hc, jnp.cos(time_m) * window * const, atol=1e-6)


def test_get_waveform_td_window_zeros_start(wrapper):
    """With alpha_window > 0 the first sample should be zero."""
    wrapper.alpha_window = 0.2
    time = jnp.linspace(0, 1, 100)
    params = jnp.array([30.0, 100.0, 0.3, 0.2, 1.1, 0.5, 0.2])
    hp, hc = wrapper.get_waveform_td(time, params)
    assert jnp.isclose(hp[0], 0.0, atol=1e-8)
    assert jnp.isclose(hc[0], 0.0, atol=1e-8)


def test_get_waveform_fd_shape_and_type(wrapper):
    """FD output should be complex, half-length rfft array."""
    params = jnp.array([30.0, 100.0, 0.3, 0.2, 1.1, 0.5, 0.2])
    hp_fd, hc_fd = wrapper.get_waveform_fd(params)
    expected_len = jnp.fft.rfftfreq(16384, 1.0 / 4096).shape[0]
    assert hp_fd.shape[0] == expected_len
    assert hc_fd.shape[0] == expected_len
    assert jnp.iscomplexobj(hp_fd)
    assert jnp.iscomplexobj(hc_fd)
