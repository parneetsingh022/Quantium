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
from quantium.core.dimensions import Dim, dim_div, dim_mul, dim_pow, DIM_0
from quantium.core.utils import format_dim
import re


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
    system: str = "si"

    def __post_init__(self) -> None:
        if len(self.dim) != 7:
            raise ValueError("dim must be a 7-tuple (L,M,T,I,Θ,N,J)")
        if not (self.scale_to_si > 0 and isfinite(self.scale_to_si)):
            raise ValueError("scale_to_si must be a positive, finite number")
        
    def __rmatmul__(self, value: float) -> Quantity:
        return Quantity(float(value), self)


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
        

    def to(self, new_unit: Unit) -> Quantity:
        if new_unit.dim != self.dim:
            raise TypeError("Dimension mismatch in conversion")
        return Quantity(self._mag_si / new_unit.scale_to_si, new_unit)

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
        new_dim = dim_mul(self.dim, other.dim)
        # compose unit name and scale
        new_unit_name = f"{self.unit.name}·{other.unit.name}"
        new_scale = self.unit.scale_to_si * other.unit.scale_to_si
        new_unit = Unit(new_unit_name, new_scale, new_dim)
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
        new_dim = dim_div(self.dim, other.dim)
        new_unit_name = f"{self.unit.name}/{other.unit.name}"
        new_scale = self.unit.scale_to_si / other.unit.scale_to_si
        new_unit = Unit(new_unit_name, new_scale, new_dim)
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
        new_dim = dim_pow(self.dim, n)
        new_unit_name = f"{self.unit.name}^{n}"
        new_scale = self.unit.scale_to_si ** n
        new_unit = Unit(new_unit_name, new_scale, new_dim)
        return Quantity((self._mag_si ** n) / new_unit.scale_to_si, new_unit)
    
    def __repr__(self) -> str:
        # local import to avoid circular deps; module is cached after first import
        from quantium.core.utils import prettify_unit_name_supers

        mag = self._mag_si / self.unit.scale_to_si          # keep value in current unit
        pretty = prettify_unit_name_supers(self.unit.name)  # keep symbols as written
        return f"{mag:g}" if pretty == "1" else f"{mag:g} {pretty}"
        

        

    