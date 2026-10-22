"""Convenience type aliases for JAX scalar annotations."""

from jaxtyping import Array, Complex, Float, Int

type FloatScalar = Float[Array, ""]
type IntScalar = Int[Array, ""]
type ComplexScalar = Complex[Array, ""]

type FloatLike = float | FloatScalar
type IntLike = int | IntScalar
