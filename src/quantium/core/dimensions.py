# quantium.core.dimensions

from __future__ import annotations
from fractions import Fraction
from typing import Iterable, Union, Tuple, TypeAlias, Any, Iterator, overload

Number = Union[int, float, Fraction]

# Rationalization settings (tweak with care)
MAX_DENOMINATOR: int = 1024
EXP_TOL: float = 1e-12
EPS_SNAP: float = 1e-12


class IrrationalExponentError(ValueError):
    """Raised when non-rational exponents are supplied for dimensioned units."""


def snap_fraction(value: Fraction) -> Fraction:
    """Collapse tiny residuals produced by arithmetic back to zero."""
    if value == 0:
        return Fraction(0, 1)
    if abs(float(value)) < EPS_SNAP:
        return Fraction(0, 1)
    return value


def rationalize_exponent(value: Number) -> Fraction:
    """Convert a numeric exponent into an exact Fraction within tolerance."""

    if isinstance(value, Fraction):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not value == value or value in (float("inf"), float("-inf")):
            raise IrrationalExponentError("Exponents must be finite real numbers.")
        candidate = Fraction(value).limit_denominator(MAX_DENOMINATOR)
        if abs(value - float(candidate)) <= EXP_TOL:
            return candidate
        raise IrrationalExponentError(
            "Exponent {val:.12g} cannot be represented as a rational with denominator "
            "<= {max_den} within tolerance {tol}. Closest candidate is {num}/{den} (≈ {cand:.12g}).".format(
                val=value,
                max_den=MAX_DENOMINATOR,
                tol=EXP_TOL,
                num=candidate.numerator,
                den=candidate.denominator,
                cand=float(candidate),
            )
        )
    raise TypeError(f"Unsupported exponent type: {type(value)!r}")


def canonical_dim_key(dim: "Dimension") -> Tuple[Tuple[int, int, int], ...]:
    """Tuple form used for hashing/equality: (axis, numerator, denominator)."""
    return tuple((idx, comp.numerator, comp.denominator) for idx, comp in enumerate(dim))

# --- Public typing -----------------------------------------------------------
# Keep the old exported name "Dim" so external code doesn't change.
# It's now an alias to our object, which is a tuple subclass (still runtime-compatible).
Dim: TypeAlias = "Dimension"
DimTuple = Tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]
DimLike = Union["Dimension", DimTuple, Iterable[Number]]

# --- Core object -------------------------------------------------------------

class Dimension(tuple):
    """
    Immutable 7-length vector of integer exponents for SI base dimensions.

    Tuple subclass => backward-compatible (hashable, comparable, usable as dict keys).
    """

    __slots__ = ()

    def __new__(cls, data: DimLike = (0, 0, 0, 0, 0, 0, 0)) -> "Dimension":
        if isinstance(data, Dimension):
            comps = tuple(snap_fraction(Fraction(v.numerator, v.denominator)) for v in data)
        else:
            try:
                iterator: Iterator[Number] = iter(data)
            except TypeError as exc:  # pragma: no cover - defensive branch
                raise TypeError("Dimension requires an iterable of exponents") from exc

            comps = tuple(snap_fraction(rationalize_exponent(x)) for x in iterator)

        if len(comps) != 7:
            raise ValueError("Dimension must have length 7 (L, M, T, I, Θ, N, J).")
        return tuple.__new__(cls, comps)

    # --- Algebra (operator overloads) ---
    def __mul__(self, other: DimLike) -> "Dimension":  # type: ignore[override]
        o = Dimension(other)
        return Dimension(snap_fraction(x + y) for x, y in zip(self, o, strict=True))

    def __truediv__(self, other: DimLike) -> "Dimension":
        o = Dimension(other)
        return Dimension(snap_fraction(x - y) for x, y in zip(self, o, strict=True))

    def __pow__(self, n: Number) -> "Dimension":
        if self.is_dimensionless:
            if not isinstance(n, (int, float, Fraction)):
                raise TypeError("Exponent must be numeric.")
            return Dimension(self)

        exponent = rationalize_exponent(n)
        return Dimension(snap_fraction(x * exponent) for x in self)
    
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

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Dimension):
            return canonical_dim_key(self) == canonical_dim_key(other)
        try:
            other_dim = Dimension(other)
        except Exception:
            return False
        return canonical_dim_key(self) == canonical_dim_key(other_dim)

    def __hash__(self) -> int:
        # Delegate to tuple hashing so values remain hash-compatible with tuples
        # containing the same Fraction entries (preserving legacy behaviour).
        return tuple.__hash__(self)

    # --- Helpers ---
    @property
    def is_dimensionless(self) -> bool:
        return all(x == 0 for x in self)

    def as_tuple(self) -> DimTuple:
        # explicit narrow type for external APIs
        return tuple(self)

    def as_key(self) -> Tuple[Tuple[int, int, int], ...]:
        """Return a stable, canonical key for hashing or dict/set usage."""
        return canonical_dim_key(self)

    def __repr__(self) -> str:
        # readable, but still unambiguous
        names = ("L", "M", "T", "I", "Θ", "N", "J")
        parts = "".join(f"[{n}^{v}]" for n, v in zip(names, self, strict=True))

        parts = ""
        for n, v in zip(names, self, strict=True):
            if v != 0:
                parts += f"[{n}^{v.numerator}/{v.denominator}]" if v.denominator != 1 else f"[{n}^{v.numerator}]"

        return parts


# --- Legacy function shims (keep working code alive) ------------------------


def dim_mul(a: DimLike, b: DimLike) -> Dimension:
    return Dimension(a) * b


def dim_div(a: DimLike, b: DimLike) -> Dimension:
    return Dimension(a) / b


def dim_pow(a: DimLike, n: Number) -> Dimension:
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
