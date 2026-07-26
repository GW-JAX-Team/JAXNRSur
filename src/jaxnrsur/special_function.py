import jax

from jaxnrsur.typing import FloatScalar, IntLike


def comb(N: IntLike, k: IntLike) -> FloatScalar:
    """Compute the binomial coefficient (N choose k) using JAX fori_loop.

    Args:
        N (IntLike): The total number of items.
        k (IntLike): The number of items to choose.

    Returns:
        FloatScalar: The binomial coefficient as a scalar JAX array.
    """
    return jax.lax.fori_loop(0, k, lambda i, acc: acc * (N - i) / (i + 1), 1.0)


def effective_spin(q: float, chi1: float, chi2: float) -> tuple[float, float]:
    """Calculate effective spin parameters for a binary system.

    Args:
        q (float): Mass ratio (m1/m2).
        chi1 (float): Spin of the first object.
        chi2 (float): Spin of the second object.

    Returns:
        tuple[float, float]:
            chi_hat (float): Effective spin parameter.
            chi_a (float): Spin difference parameter.
    """
    eta = q / (1.0 + q) ** 2
    chi_wtAvg = (q * chi1 + chi2) / (1 + q)
    chi_hat = (chi_wtAvg - 38.0 * eta / 113.0 * (chi1 + chi2)) / (
        1.0 - 76.0 * eta / 113.0
    )
    chi_a = (chi1 - chi2) / 2.0
    return chi_hat, chi_a
