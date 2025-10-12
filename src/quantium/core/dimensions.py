"""
quantium.core.dimensions
=========================

Core representation and algebra for **physical dimensions** used throughout
Quantium.

This module encodes a quantity's physical dimension as a fixed-length
7-tuple of integer exponents over the SI base dimensions:

    Dim = (L, M, T, I, Θ, N, J)
          (length, mass, time, electric current, thermodynamic temperature,
           amount of substance, luminous intensity)

For example:
- meters (m):           LENGTH           = (1, 0, 0, 0, 0, 0, 0)
- seconds (s):          TIME             = (0, 0, 1, 0, 0, 0, 0)
- dimensionless:        DIM_0            = (0, 0, 0, 0, 0, 0, 0)
- speed (m/s):          LENGTH/TIME      = (1, 0, -1, 0, 0, 0, 0)
- acceleration:         LENGTH/TIME^2    = (1, 0, -2, 0, 0, 0, 0)
- force (kg·m/s²):      = (1, 1, -2, 0, 0, 0, 0)
"""

from typing import Tuple

Dim = Tuple[int, int, int, int, int, int, int]  # (L, M, T, I, Θ, N, J)

# ---------------------------------------------------------------------------
# Base dimension vectors (use descriptive names to avoid symbol collisions)
# ---------------------------------------------------------------------------

DIM_0: Dim      = (0, 0, 0, 0, 0, 0, 0)  # dimensionless
LENGTH: Dim     = (1, 0, 0, 0, 0, 0, 0)
MASS: Dim       = (0, 1, 0, 0, 0, 0, 0)
TIME: Dim       = (0, 0, 1, 0, 0, 0, 0)
CURRENT: Dim    = (0, 0, 0, 1, 0, 0, 0)
TEMPERATURE: Dim= (0, 0, 0, 0, 1, 0, 0)
AMOUNT: Dim     = (0, 0, 0, 0, 0, 1, 0)
LUMINOUS: Dim   = (0, 0, 0, 0, 0, 0, 1)

# ---------------------------------------------------------------------------
# Algebraic operations on dimensions
# ---------------------------------------------------------------------------

def dim_mul(a: Dim, b: Dim) -> Dim:
    """Multiply dimensions (add exponents)."""
    return tuple(x + y for x, y in zip(a, b, strict=False))  # type: ignore

def dim_div(a: Dim, b: Dim) -> Dim:
    """Divide dimensions (subtract exponents)."""
    return tuple(x - y for x, y in zip(a, b, strict=False))  # type: ignore

def dim_pow(a: Dim, n: int) -> Dim:
    """Raise a dimension to an integer power."""
    return tuple(x * n for x in a)  # type: ignore
