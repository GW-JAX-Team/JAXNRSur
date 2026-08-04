"""Time-domain comparison metrics used by cross-validation.

Ported from ripple's ``tests/helpers/metrics.py`` (time-domain half only --
JAXNRSur's cross-validation compares real sampled time series against
gwsurrogate, not frequency-domain waveforms against a PSD-weighted
reference, so the FD inner-product/overlap functions do not apply here).
Kept byte-for-byte identical in formula so both projects report the same
mismatch statistic for TD waveforms.
"""

import numpy as np


def time_domain_overlap_loss(h1, h2) -> float:
    """White, normalized mismatch between two real sampled time series.

    This is ``1 - <h1, h2> / (||h1|| ||h2||)`` with the ordinary sampled
    time-domain inner product. It is appropriate when both series use the same,
    uniformly sampled time grid: the common ``delta_t`` factor cancels. It performs
    no FFT, whitening, or maximization over time or phase.

    The result measures shape and phase agreement, not absolute amplitude agreement:
    multiplying either input by a positive constant leaves it unchanged. Pair it with
    :func:`relative_norm_error` when a validation must also catch a global amplitude
    scale regression.

    Half the squared distance between normalized vectors is algebraically identical
    to the expression above, but remains well-defined for anti-correlated series and
    avoids subtracting nearly equal ``A*B`` and ``C**2`` terms.
    """
    h1 = _as_real_time_series(h1, "h1")
    h2 = _as_real_time_series(h2, "h2")
    _require_matching_shapes(h1, h2)

    norm1 = float(np.linalg.norm(h1))
    norm2 = float(np.linalg.norm(h2))
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("normalized time-domain mismatch is undefined for zero strain")

    difference = h1 / norm1 - h2 / norm2
    loss = 0.5 * float(difference @ difference)
    # Rounding can put an exactly identical/anti-correlated pair a few ulps outside
    # the mathematical [0, 2] range.
    return float(np.clip(loss, 0.0, 2.0))


def relative_norm_error(h1, h2) -> float:
    """Relative difference between two sampled time-series norms.

    This detects a global amplitude-scale regression without conflating it with a
    phase-only error between equal-norm signals. The result is
    ``abs(||h1|| / ||h2|| - 1)``; ``h2`` is normally the reference strain.
    """
    h1 = _as_real_time_series(h1, "h1")
    h2 = _as_real_time_series(h2, "h2")
    _require_matching_shapes(h1, h2)

    norm1 = float(np.linalg.norm(h1))
    norm2 = float(np.linalg.norm(h2))
    if norm2 == 0.0:
        raise ValueError("relative norm error is undefined for zero reference strain")
    return abs(norm1 / norm2 - 1.0)


def _as_real_time_series(values, name: str) -> np.ndarray:
    """Validate one real, finite, non-empty one-dimensional time series."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional time series")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_matching_shapes(h1: np.ndarray, h2: np.ndarray) -> None:
    if h1.shape != h2.shape:
        raise ValueError(
            f"time series must have identical shapes, got {h1.shape} and {h2.shape}"
        )
