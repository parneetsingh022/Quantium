# quantium.core.dimensions

from __future__ import annotations
from typing import Iterable, Union, Tuple, TypeAlias, Any

# --- Public typing -----------------------------------------------------------
# Keep the old exported name "Dim" so external code doesn't change.
# It's now an alias to our object, which is a tuple subclass (still runtime-compatible).
Dim: TypeAlias = "Dimension"
DimTuple = Tuple[int, int, int, int, int, int, int]
DimLike = Union["Dimension", DimTuple, Iterable[int]]

# --- Core object -------------------------------------------------------------

class Dimension(tuple):
    """
    Immutable 7-length vector of integer exponents for SI base dimensions.

    Tuple subclass => backward-compatible (hashable, comparable, usable as dict keys).
    """

    __slots__ = ()

    def __new__(cls, data: DimLike = (0, 0, 0, 0, 0, 0, 0)) -> "Dimension":
        if isinstance(data, Dimension):
            return tuple.__new__(cls, data)

        # allow any iterable of ints
        t = tuple(int(x) for x in data)
        if len(t) != 7:
            raise ValueError("Dimension must have length 7 (L, M, T, I, Θ, N, J).")
        return tuple.__new__(cls, t)

    # --- Algebra (operator overloads) ---
    def __mul__(self, other: DimLike) -> "Dimension": # type: ignore[override]
        o = Dimension(other)
        return Dimension(x + y for x, y in zip(self, o, strict=True))

    def __truediv__(self, other: DimLike) -> "Dimension":
        o = Dimension(other)
        return Dimension(x - y for x, y in zip(self, o, strict=True))

    def __pow__(self, n: int) -> "Dimension":
        if not isinstance(n, int):
            raise TypeError("Exponent must be an int.")
        return Dimension(x * n for x in self)
    
    def __rtruediv__(self, other: DimLike) -> "Dimension":
        """Handles (tuple / Dimension) by calculating (other / self)."""
        # 'other' is the item on the left (the tuple 't')
        # 'self' is the Dimension object on the right (LENGTH)
        o = Dimension(other)  # Convert the left side to a Dimension
        return o / self       # Use the existing __truediv__ (o.__truediv__(self))
    
    def __rmul__(self, other: Any) -> "Dimension":
        """Prevent (int * Dimension) from falling back to tuple repetition."""
        # We explicitly disallow this operation.
        # If we wanted to support it, we'd check if 'other' is a DimLike
        # and call self.__mul__(other), but multiplication is commutative
        # in this context anyway.
        return NotImplemented
    
    def __add__(self, other: Any) -> "Dimension":
        """Block tuple concatenation (e.g., LENGTH + MASS)."""
        return NotImplemented

    def __radd__(self, other: Any) -> "Dimension":
        """Block tuple concatenation (e.g., (1,2) + MASS)."""
        return NotImplemented

    # --- Helpers ---
    @property
    def is_dimensionless(self) -> bool:
        return all(x == 0 for x in self)

    def as_tuple(self) -> DimTuple:
        # explicit narrow type for external APIs
        return tuple(self)

    def __repr__(self) -> str:
        # readable, but still unambiguous
        names = ("L", "M", "T", "I", "Θ", "N", "J")
        parts = "".join(f"[{n}^{v}]" for n, v in zip(names, self, strict=True))

        parts = ""
        for n, v in zip(names, self, strict=True):
            if v != 0:
                parts += f"[{n}^{v}]"

        return parts

# --- Legacy function shims (keep working code alive) ------------------------

def dim_mul(a: DimLike, b: DimLike) -> Dimension:
    return Dimension(a) * b

def dim_div(a: DimLike, b: DimLike) -> Dimension:
    return Dimension(a) / b

def dim_pow(a: DimLike, n: int) -> Dimension:
    return Dimension(a) ** n

# --- Public constants (identical names, now Dimension instances) ------------

DIM_0: Dim       = Dimension((0, 0, 0, 0, 0, 0, 0))
LENGTH: Dim      = Dimension((1, 0, 0, 0, 0, 0, 0))
MASS: Dim        = Dimension((0, 1, 0, 0, 0, 0, 0))
TIME: Dim        = Dimension((0, 0, 1, 0, 0, 0, 0))
CURRENT: Dim     = Dimension((0, 0, 0, 1, 0, 0, 0))
TEMPERATURE: Dim = Dimension((0, 0, 0, 0, 1, 0, 0))
AMOUNT: Dim      = Dimension((0, 0, 0, 0, 0, 1, 0))
LUMINOUS: Dim    = Dimension((0, 0, 0, 0, 0, 0, 1))
