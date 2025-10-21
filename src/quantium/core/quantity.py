"""
quantium.core.quantity
======================

Defines the `Unit` and `Quantity` classes for representing and manipulating
physical quantities with units and dimensions in a consistent, SI-based system.

This module provides:
- A `Unit` class for defining physical units (e.g., meter, second, kilogram)
  with their corresponding scaling factors to SI base units and dimensional
  representation.
- A `Quantity` class for representing values with both magnitude and units,
  enabling dimensional arithmetic and automatic unit consistency checks.

The system supports:
- Dimensional analysis and arithmetic operations between quantities.
- Conversion between compatible units.
- Creation of derived quantities via multiplication, division, and exponentiation.
"""



from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
import re

from quantium.core.dimensions import DIM_0, Dim, dim_div, dim_mul, dim_pow
from quantium.units.parser import extract_unit_expr
from typing import Union

Number = Union[int, float]
_POWER_RE = re.compile(r"^(?P<base>.+?)\^(?P<exp>-?\d+)$")

def _normalize_power_name(name: str) -> str:
    """
    Make names canonical:
    - 'x^1'  -> 'x'
    - 'x^0'  -> '1'   (dimensionless label; adjust if you prefer something else)
    - 'x^-1' stays 'x^-1'
    """
    m = _POWER_RE.match(name)
    if not m:
        return name
    base = m.group("base")
    exp = int(m.group("exp"))
    if exp == 1:
        return base
    if exp == 0:
        return "1"
    return f"{base}^{exp}"


@dataclass(frozen=True, slots=True)
class Unit:
    """
    A physical unit.

    Attributes
    ----------
    name : str
        Symbol or name (e.g., "m", "s", "kg", "cm").
    scale_to_si : float
        Multiplicative factor to convert 1 of this unit to SI for its dimension.
        Examples: m=1.0, cm=0.01, µs=1e-6, ft=0.3048.
    dim : Dim
        Dimension vector (L,M,T,I,Θ,N,J). E.g., meters -> (1,0,0,0,0,0,0).
    system : str
        Optional tag like "si", "imperial", etc.
    """
    name: str
    scale_to_si: float
    dim: Dim

    def __post_init__(self) -> None:
        if len(self.dim) != 7:
            raise ValueError("dim must be a 7-tuple (L,M,T,I,Θ,N,J)")
        if not (self.scale_to_si > 0 and isfinite(self.scale_to_si)):
            raise ValueError("scale_to_si must be a positive, finite number")
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        # dimension must match exactly; scale_to_si can have tiny FP noise
        return (
            self.dim == other.dim
            and isclose(self.scale_to_si, other.scale_to_si, rel_tol=1e-12, abs_tol=0.0)
        )
        
    def __rmul__(self, value: float) -> Quantity:
        return Quantity(float(value), self)
    
    def __mul__(self, other: "Unit") -> "Unit":
        new_dim = dim_mul(self.dim, other.dim)
        new_scale = self.scale_to_si * other.scale_to_si

        # If the two units are equivalent (same dim and scale), collapse to a power.
        # This avoids "K·kelvin" and produces "kelvin^2" (or "K^2" if the LHS was "K").
        if (
            self.dim == other.dim
            and isclose(self.scale_to_si, other.scale_to_si, rel_tol=1e-12, abs_tol=0.0)
        ):
            base_name = self.name if self.name else other.name
            new_unit_name = _normalize_power_name(f"{base_name}^2")
            return Unit(new_unit_name, new_scale, new_dim)

        # Otherwise compose normally.
        new_unit_name = f"{self.name}·{other.name}"
        return Unit(new_unit_name, new_scale, new_dim)


    def __truediv__(self, other: "Unit") -> "Unit":
        new_dim = dim_div(self.dim, other.dim)
        new_scale = self.scale_to_si / other.scale_to_si

        # Parenthesize denominator if it's compound to avoid flattening like "W·s/N·s/m^2"
        def _needs_paren(name: str) -> bool:
            # name contains any operator that would change precedence if ungrouped
            return any(op in name for op in ("·", "*", "/")) and not (name.startswith("(") and name.endswith(")"))

        right = f"({other.name})" if _needs_paren(other.name) else other.name
        new_unit_name = f"{self.name}/{right}"

        return Unit(new_unit_name, new_scale, new_dim)
    
    def __rtruediv__(self, n: int | float) -> Unit:
        if n != 1:
            raise TypeError(
                f"Invalid operation: cannot divide {n} by a Unit ({self.name}). "
                "Only 1/unit (reciprocal) is supported."
            )

        new_dim = dim_div(DIM_0, self.dim)
        name = self.name

        if name.startswith("1/"):
            # 1/(1/x) -> x
            name = name[2:]
        else:
            m = _POWER_RE.match(name)
            if m:
                base = m.group("base")
                k = int(m.group("exp"))
                name = f"{base}^{-k}"    # 1/(s^-3) -> s^3, 1/(s^3) -> s^-3
            else:
                name = f"{name}^-1"      # 1/s -> s^-1   (key change)
        
        normalized_name = _normalize_power_name(name)
        new_scale = 1 / self.scale_to_si
        return Unit(normalized_name, new_scale, new_dim)
        

    def __pow__(self, n: int) -> Unit:
        new_dim = dim_pow(self.dim, n)
        # Canonical naming:
        if n == 0:
            new_unit_name = f"{self.name}^0"  # or maybe a specific "dimensionless" name if you prefer
        elif n == 1:
            new_unit_name = self.name
        else:
            new_unit_name = f"{self.name}^{n}"   # handles negatives like s^-3

        normalized_name = _normalize_power_name(new_unit_name)

        new_scale = self.scale_to_si ** n
        return Unit(normalized_name, new_scale, new_dim)
    
    def as_name(self, name : str) -> Unit:
        return Unit(name, self.scale_to_si, self.dim)


class Quantity:
    """
    Represents a physical quantity with magnitude, dimension, and unit, supporting
    arithmetic operations and unit conversions while maintaining dimensional consistency.

    Attributes
    ----------
    _mag_si : float
        The magnitude of the quantity expressed in SI base units.
    dim : dict or custom dimension object
        The physical dimension of the quantity (e.g., length, time, mass).
    unit : Unit
        The unit in which the quantity is currently represented.
    """
    __slots__ = ["_mag_si", "dim", "unit"]


    def __init__(self, value : float, unit : Unit):
        self._mag_si = float(value) * unit.scale_to_si
        self.dim = unit.dim
        self.unit = unit
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        # Same physical dimension; SI magnitudes equal within tolerance.
        return (
            self.dim == other.dim
            and isclose(self._mag_si, other._mag_si, rel_tol=1e-12, abs_tol=0.0)
        )

    def to(self, new_unit: "Unit|str") -> Quantity:
        if(isinstance(new_unit, str)):
            from quantium.units.registry import DEFAULT_REGISTRY
            new_unit = extract_unit_expr(new_unit, DEFAULT_REGISTRY)
        
        # This proves to mypy that new_unit is a Unit, not a str.
        if not isinstance(new_unit, Unit):
            raise TypeError(
                "Internal error: unit expression did not resolve to a Unit object."
            )

        if new_unit.dim != self.dim:
            raise TypeError("Dimension mismatch in conversion")
        
        # Optimization: Avoid re-allocating if the target unit is
        # *already* our current unit (same name AND dim).
        # We must check name, as 'V/m' == 'W/(A·m)' is True physically,
        # but the user's intent in to() is to get the new name.
        # The dim check has already passed at this point.
        if new_unit.name == self.unit.name:
            return self
        
        return Quantity(self._mag_si / new_unit.scale_to_si, new_unit)
        
    
    def to_si(self) -> Quantity:
        """
        Return an equivalent Quantity expressed in SI with a preferred symbol when possible.
        Strategy:
        1) If the current unit clearly belongs to a specific SI family (atomic symbol with
            scale 1, or a prefixed form of one), keep that family in SI (e.g., kBq → Bq).
        2) Otherwise, use the dimension's preferred symbol (A, N, W, Pa, Hz, …).
        3) If no preferred symbol exists, compose the base-SI name from the dimension.
        """
        # Local imports avoid circular import at module load time.
        from quantium.core.utils import format_dim, preferred_symbol_for_dim
        from quantium.units.registry import DEFAULT_REGISTRY as _ureg

        cur_name = self.unit.name

        # --- (1) Preserve the "family" if we can (Hz vs Bq, Gy vs Sv, …) ---
        # Grab all atomic SI heads (scale==1, same dim) registered in the system.
        si_heads = [name for name, u in _ureg.all().items()
                    if u.scale_to_si == 1.0 and u.dim == self.dim]

        # If our current unit is exactly one of those heads (e.g., "Bq"), or is a prefixed
        # form ending with the head (e.g., "kBq"), keep that head as the SI symbol.
        for head in si_heads:
            if cur_name == head or cur_name.endswith(head):
                si_unit = Unit(head, 1.0, self.dim)
                return Quantity(self._mag_si, si_unit)  # already SI magnitude

        # --- (2) Fall back to the global preferred symbol for this dimension ---
        sym = preferred_symbol_for_dim(self.dim)  # e.g., "A", "N", "W", "Pa", "Hz", …
        if sym:
            si_unit = Unit(sym, 1.0, self.dim)
            return Quantity(self._mag_si, si_unit)

        # --- (3) Compose from base SI if no named symbol exists ---
        si_name = format_dim(self.dim)  # e.g., "kg·m/s²", "1/s", "m"
        si_unit = Unit(si_name, 1.0, self.dim)
        return Quantity(self._mag_si, si_unit)

    def as_name(self, name: str) -> Quantity:
        """Return a copy of this quantity with the same value and dimension but a new unit name."""
        return Quantity(self._mag_si, self.unit.as_name(name))

    @property
    def si(self) -> Quantity:
        return self.to_si()

    # arithmetic
    def __add__(self, other: Quantity) -> Quantity:
        if self.dim != other.dim:
            raise TypeError("Add requires same dimensions")
        # return in left operand's unit
        return Quantity((self._mag_si + other._mag_si)/self.unit.scale_to_si, self.unit)
    
    def __sub__(self, other: Quantity) -> Quantity:
        if self.dim != other.dim:
            raise TypeError("Sub requires same dimensions")
        return Quantity((self._mag_si - other._mag_si)/self.unit.scale_to_si, self.unit)
    
    def __mul__(self, other: "Quantity | Unit | Number") -> "Quantity":
        # scalar × quantity
        if isinstance(other, (int, float)):
            return Quantity((self._mag_si * float(other)) / self.unit.scale_to_si, self.unit)

        # quantity × unit
        if isinstance(other, Unit):
            new_unit = self.unit * other

            return Quantity(self._mag_si, new_unit)
        
        # quantity × quantity
        new_unit = self.unit * other.unit
        # convert SI magnitude back to the composed unit
        return Quantity((self._mag_si * other._mag_si) / new_unit.scale_to_si, new_unit)

    def __rmul__(self, other: float | int) -> "Quantity":
        # allows 3 * (2 m) -> 6 m
        return self.__mul__(other)

    def __truediv__(self, other: "Quantity | Unit | Number") -> "Quantity":
        # quantity / scalar
        if isinstance(other, (int, float)):
            return Quantity((self._mag_si / float(other)) / self.unit.scale_to_si, self.unit)
        
        # quantity / unit
        if isinstance(other, Unit):
            new_unit = self.unit / other

            return Quantity(self._mag_si, new_unit)

        # quantity / quantity
        new_unit = self.unit / other.unit
        if new_unit.dim == DIM_0:
            # dimensionless quantity has no name
            new_unit = Unit('', 1.0, DIM_0)

        return Quantity((self._mag_si / other._mag_si) / new_unit.scale_to_si, new_unit)

    def __rtruediv__(self, other: float | int) -> "Quantity":
        # scalar / quantity  -> returns Quantity with inverse dimension
        if not isinstance(other, (int, float)):
            return NotImplemented
        
        new_dim = dim_div(DIM_0, self.dim)  # or dim_pow(self.dim, -1)
        new_unit_name = f"{1}/{self.unit.name}"
        new_scale = 1.0 / self.unit.scale_to_si
        new_unit = Unit(new_unit_name, new_scale, new_dim)
        return Quantity((float(other) / self._mag_si) / new_unit.scale_to_si, new_unit)

    def __pow__(self, n: int) -> "Quantity":
        new_unit = self.unit ** n
        return Quantity((self._mag_si ** n) / new_unit.scale_to_si, new_unit)
    
    def __repr__(self) -> str:
        # Local imports avoid cyclic imports; modules are cached after the first time.
        from quantium.core.utils import preferred_symbol_for_dim, prettify_unit_name_supers
        from quantium.units.registry import PREFIXES

        # Numeric magnitude in the *current* unit
        mag = self._mag_si / self.unit.scale_to_si

        # Dimensionless: print bare number
        if self.dim == DIM_0:
            return f"{mag:.15g}"

        # Start from the user’s unit name (keeps cm/ms etc.), with superscripts & cancellation
        pretty = prettify_unit_name_supers(self.unit.name, cancel=True)

        # Only consider upgrading when the current unit *looks composed*.
        # We do NOT override atomic SI symbols like Pa, Hz, Bq, Gy, Sv, ...
        name_raw = self.unit.name  # use original to detect composition
        is_composed = any(ch in name_raw for ch in ('/', '·', '^'))

        if is_composed:
            sym = preferred_symbol_for_dim(self.dim)  # e.g., "N", "A", "W", "Pa", "Hz", ...
            if sym:
                scale = self.unit.scale_to_si

                # Exact SI scale -> upgrade to the canonical symbol (e.g., "N")
                if abs(scale - 1.0) <= 1e-12:
                    pretty = sym
                else:
                    # If the composed unit's total scale matches an SI prefix,
                    # print as <prefix><symbol> (e.g., 1e-6 -> "µN")
                    for p in PREFIXES:
                        if abs(scale - p.factor) <= 1e-12:
                            pretty = f"{p.symbol}{sym}"
                            break
                    # else: keep the composed pretty name as-is

        # If the pretty name reduces to "1", show just the number
        return f"{mag:.15g}" if pretty == "1" else f"{mag:.15g} {pretty}"


    
    def __format__(self, spec: str) -> str:
        """
        Custom string formatting for Quantity objects.

        The format specifier controls whether the quantity is shown in its
        current unit or converted to SI units before printing.

        Supported specifiers
        --------------------
        "" (empty), "unit", or "u"
            Display the quantity in its current unit (default).
        "si"
            Display the quantity converted to SI units.

        Examples
        --------
        >>> v = 1000 @ (ureg.get("cm") / ureg.get("s"))
        >>> f"{v}"           # default: show in current unit (cm/s)
        '1000 cm/s'
        >>> f"{v:unit}"      # explicit but same as above
        '1000 cm/s'
        >>> f"{v:si}"        # convert and show in SI (m/s)
        '10 m/s'

        Raises
        ------
        ValueError
            If the format specifier is not one of "", "unit", "u", or "si".
        """
        spec = (spec or "").strip().lower()
        if spec in ("", "unit", "u"):
            return repr(self)          # current unit (default)
        if spec == "si":
            return repr(self.to_si())  # force SI
        raise ValueError("Unknown format spec; use '', 'unit', or 'si'")




        

        

    