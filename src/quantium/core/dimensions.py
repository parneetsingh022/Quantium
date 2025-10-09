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
- meters (m):           L           = (1, 0, 0, 0, 0, 0, 0)
- seconds (s):          T           = (0, 0, 1, 0, 0, 0, 0)
- dimensionless:        DIM_0       = (0, 0, 0, 0, 0, 0, 0)
- speed (m/s):          L/T         = (1, 0, -1, 0, 0, 0, 0)
- acceleration:         L/T^2       = (1, 0, -2, 0, 0, 0, 0)
- force                 (kg·m/s^2): = (1, 1, -2, 0, 0, 0, 0)
"""

from typing import Tuple

Dim = Tuple[int, int, int, int, int, int, int]  # (L, M, T, I, Θ, N, J)

# Base dimension vectors
DIM_0: Dim = (0,0,0,0,0,0,0)
L: Dim     = (1,0,0,0,0,0,0)
M: Dim     = (0,1,0,0,0,0,0)
T: Dim     = (0,0,1,0,0,0,0)
I: Dim     = (0,0,0,1,0,0,0)
THETA: Dim = (0,0,0,0,1,0,0)
N: Dim     = (0,0,0,0,0,1,0)
J: Dim     = (0,0,0,0,0,0,1)

def dim_mul(a: Dim, b: Dim) -> Dim: return tuple(x+y for x,y in zip(a,b))  # type: ignore
def dim_div(a: Dim, b: Dim) -> Dim: return tuple(x-y for x,y in zip(a,b))  # type: ignore
def dim_pow(a: Dim, n: int) -> Dim: return tuple(x*n for x in a)           # type: ignore
