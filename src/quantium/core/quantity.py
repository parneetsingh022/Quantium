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
from math import isfinite
import re

from quantium.core.dimensions import DIM_0, Dim, dim_div, dim_mul, dim_pow

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
        return self.scale_to_si == other.scale_to_si and self.dim == other.dim
        
    def __rmatmul__(self, value: float) -> Quantity:
        return Quantity(float(value), self)
    
    def __mul__(self, other : Unit) -> Unit:
        new_dim = dim_mul(self.dim, other.dim)
        # compose unit name and scale
        new_unit_name = f"{self.name}·{other.name}"
        new_scale = self.scale_to_si * other.scale_to_si
        return Unit(new_unit_name, new_scale, new_dim)

    def __truediv__(self, other : Unit) -> Unit:
        new_dim = dim_div(self.dim, other.dim)
        new_unit_name = f"{self.name}/{other.name}"
        new_scale = self.scale_to_si / other.scale_to_si
        return Unit(new_unit_name, new_scale, new_dim)
    
    def __rtruediv__(self, n: int | float):
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
        
        return self._mag_si == other._mag_si and self.unit.dim == other.unit.dim

    def to(self, new_unit: Unit) -> Quantity:
        if new_unit.dim != self.dim:
            raise TypeError("Dimension mismatch in conversion")
        return Quantity(self._mag_si / new_unit.scale_to_si, new_unit)
        
    
    def to_si(self) -> "Quantity":
        """
        Return an equivalent Quantity expressed in SI with a preferred symbol when possible.
        Examples:
        (C/s)  -> A
        (kg·m/s²) -> N
        (J/s)  -> W
        (1/s)  -> Hz
        (cm)   -> m
        """
        # Local imports avoid circular import at module load time.
        from quantium.core.utils import format_dim, preferred_symbol_for_dim

        # 1) Try a preferred named SI unit for this dimension (A, N, W, Pa, …)
        sym = preferred_symbol_for_dim(self.dim)  # returns e.g. "A", "N", "W", or None
        if sym:
            # Use the registry instance if you want (optional)
            try:
                from quantium.units.registry import get_unit
                si_unit = get_unit(sym)            # should have scale_to_si == 1.0
            except Exception:
                # Fallback if you want to avoid importing the registry here
                si_unit = Unit(sym, 1.0, self.dim)
            return Quantity(self._mag_si, si_unit) # _mag_si is already in SI

        # 2) Fall back to composed base-SI name for this dimension
        si_name = format_dim(self.dim)             # e.g., "kg·m/s²", "m", "1"
        si_unit = Unit(si_name, 1.0, self.dim)
        return Quantity(self._mag_si, si_unit)

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
    
    def __mul__(self, other: "Quantity | float | int") -> "Quantity":
        # scalar × quantity
        if isinstance(other, (int, float)):
            return Quantity((self._mag_si * float(other)) / self.unit.scale_to_si, self.unit)

        # quantity × quantity
        new_unit = self.unit * other.unit
        # convert SI magnitude back to the composed unit
        return Quantity((self._mag_si * other._mag_si) / new_unit.scale_to_si, new_unit)

    def __rmul__(self, other: float | int) -> "Quantity":
        # allows 3 * (2 m) -> 6 m
        return self.__mul__(other)

    def __truediv__(self, other: "Quantity | float | int") -> "Quantity":
        # quantity / scalar
        if isinstance(other, (int, float)):
            return Quantity((self._mag_si / float(other)) / self.unit.scale_to_si, self.unit)

        # quantity / quantity
        new_unit = self.unit / other.unit
        if new_unit.dim == DIM_0:
            # dimensionless quantity has no name
            new_unit = new_unit = Unit('', 1.0, DIM_0)

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
        new_unit = self.unit ** 2
        return Quantity((self._mag_si ** n) / new_unit.scale_to_si, new_unit)
    
    def __repr__(self) -> str:
        # Local imports avoid cyclic imports; modules are cached after the first time.
        from quantium.core.utils import preferred_symbol_for_dim, prettify_unit_name_supers

        # Always compute magnitude in the unit currently stored on the object
        mag = self._mag_si / self.unit.scale_to_si

        # Start from the user’s unit name (keeps cm/ms etc.), with superscripts and cancellation
        pretty = prettify_unit_name_supers(self.unit.name, cancel=True)

        # If this quantity’s *current unit* is exactly SI-scaled (factor 1),
        # upgrade to a preferred *named* unit for this dimension (A, N, J, W, Pa, Hz, …)
        # This turns 'C/s' -> 'A', 'kg·m/s²' -> 'N', 'J/s' -> 'W', etc.
        if self.unit.scale_to_si == 1.0:
            sym = preferred_symbol_for_dim(self.dim)
            if sym:
                pretty = sym  # symbol only; mag stays the same because factor is 1.0
        
        if self.dim == DIM_0:
            return f"{mag:g}"
        
        return f"{mag:g}" if pretty == "1" else f"{mag:g} {pretty}"


        

        

    