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
from functools import lru_cache
from math import isclose, isfinite
import math
import re

from quantium.core.dimensions import DIM_0, Dim, dim_div, dim_mul, dim_pow
from quantium.core.utils import _tokenize_name_merge
from quantium.units.parser import extract_unit_expr
from typing import Dict, List, Tuple, Union

Number = Union[int, float]
SymbolComponents = Dict[str, Tuple[int, Tuple[int, int]]]
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


_MAX_CANON_POWER = 12

_ALLOWED_CANON_PREFIX_SYMBOLS = frozenset({"T", "G", "M", "k", "m", "µ", "p"})


def _dim_key(dim: Dim) -> tuple[int, ...]:
    """Return a hashable key for cached dimension lookups."""
    return tuple(dim)


@lru_cache(maxsize=None)
def _preferred_dim_symbol_map() -> dict[tuple[int, ...], str]:
    """Cache symbols for SI-coherent units (scale 1) to speed canonicalisation."""
    from quantium.core.utils import preferred_symbol_for_dim
    from quantium.units.registry import DEFAULT_REGISTRY

    mapping: dict[tuple[int, ...], str] = {}
    for name, unit in DEFAULT_REGISTRY.all().items():
        sym = preferred_symbol_for_dim(unit.dim)
        if sym:
            mapping.setdefault(_dim_key(unit.dim), sym)
    return mapping


def _match_preferred_power(dim: Dim) -> tuple[str, int] | None:
    """Detect if ``dim`` is a power of a known preferred symbol dimension."""
    base_map = _preferred_dim_symbol_map()
    for base_dim_tuple, symbol in base_map.items():
        base_dim = base_dim_tuple
        if dim_pow(base_dim, -1) == dim:
            return symbol, -1
        for power in range(2, _MAX_CANON_POWER + 1):
            if dim_pow(base_dim, power) == dim:
                return symbol, power
            if dim_pow(base_dim, -power) == dim:
                return symbol, -power
    return None


def _canonical_unit_for_dim(dim: Dim) -> Unit:
    """Return a canonical unit (scale 1) for the provided dimension."""
    if dim == DIM_0:
        return Unit("", 1.0, DIM_0)

    from quantium.core.utils import format_dim, preferred_symbol_for_dim

    sym = preferred_symbol_for_dim(dim)
    if sym:
        return Unit(sym, 1.0, dim)

    match = _match_preferred_power(dim)
    if match:
        base_sym, power = match
        name = _normalize_power_name(f"{base_sym}^{power}")
        return Unit(name, 1.0, dim)

    name = format_dim(dim)
    if name == "1":
        return Unit("", 1.0, DIM_0)
    return Unit(name, 1.0, dim)


def _unit_symbol_map(unit: "Unit", priority: int = 0) -> SymbolComponents:
    if not unit.name:
        return {}

    raw = list(_tokenize_name_merge(unit.name).items())
    if raw:
        mapping: SymbolComponents = {}
        for idx, (symbol, exponent) in enumerate(raw):
            mapping[symbol] = (exponent, (priority, idx))
        return mapping

    # Fallback for simple units that tokeniser didn't decompose (e.g., "cm")
    return {unit.name: (1, (priority, 0))}


def _scale_symbol_map(components: SymbolComponents, factor: int) -> SymbolComponents:
    if factor == 1:
        return dict(components)
    scaled: SymbolComponents = {}
    for symbol, (exponent, order) in components.items():
        new_exp = exponent * factor
        if new_exp != 0:
            scaled[symbol] = (new_exp, order)
    return scaled


def _combine_symbol_maps(*maps: SymbolComponents) -> SymbolComponents:
    combined: SymbolComponents = {}
    for mapping in maps:
        for symbol, (exponent, order) in mapping.items():
            if exponent == 0:
                continue
            if symbol in combined:
                existing_exp, existing_order = combined[symbol]
                new_exp = existing_exp + exponent
                if new_exp == 0:
                    combined.pop(symbol, None)
                    continue
                combined[symbol] = (new_exp, min(existing_order, order))
            else:
                combined[symbol] = (exponent, order)
    return combined


def _format_unit_components(components: SymbolComponents) -> str:
    if not components:
        return ""

    def fmt(sym: str, exp: int) -> str:
        if abs(exp) == 1:
            return sym
        return f"{sym}^{abs(exp)}"

    positives: List[str] = []
    negatives: List[str] = []

    for sym in sorted(components):
        exp, _ = components[sym]
        if exp > 0:
            positives.append(fmt(sym, exp))
        elif exp < 0:
            negatives.append(fmt(sym, exp))

    numerator = "·".join(positives) if positives else "1"
    denominator = "·".join(negatives)

    if denominator:
        if "·" in denominator:
            denominator = f"({denominator})"
        return f"{numerator}/{denominator}"
    return numerator


def _dim_single_axis(dim: Dim) -> tuple[int, int] | None:
    axes = [idx for idx, power in enumerate(dim) if power != 0]
    if len(axes) != 1:
        return None
    axis = axes[0]
    return axis, dim[axis]


def _choose_symbol_for_axis(
    components: SymbolComponents,
    axis_idx: int,
    target_exp: int,
) -> Unit | None:
    from quantium.units.registry import DEFAULT_REGISTRY

    target_sign = 1 if target_exp > 0 else -1
    best: tuple[str, int, Unit] | None = None
    best_score = -1
    fallback: tuple[str, int, Unit] | None = None
    fallback_score = -1

    ordered_items = sorted(components.items(), key=lambda item: item[1][1])

    for symbol, (exponent, _) in ordered_items:
        if exponent == 0:
            continue
        try:
            candidate = DEFAULT_REGISTRY.get(symbol)
        except ValueError:
            continue

        axis_info = _dim_single_axis(candidate.dim)
        if axis_info is None:
            continue

        cand_axis, _ = axis_info
        if cand_axis != axis_idx:
            continue

        score = abs(exponent)
        entry = (symbol, exponent, candidate)

        if (exponent > 0 and target_sign > 0) or (exponent < 0 and target_sign < 0):
            if score > best_score:
                best = entry
                best_score = score
        else:
            if score > fallback_score:
                fallback = entry
                fallback_score = score

    chosen = best or fallback
    if not chosen:
        return None

    _, _, unit = chosen
    return unit


def _unit_from_components(components: SymbolComponents) -> Unit | None:
    """Reconstruct a composite unit from component symbol exponents."""
    if not components:
        return None

    from quantium.units.registry import DEFAULT_REGISTRY

    def _mul_units(units: list[Unit]) -> Unit | None:
        result: Unit | None = None
        for u in units:
            result = u if result is None else result * u
        return result

    numer_parts: list[Unit] = []
    denom_parts: list[Unit] = []

    for symbol, (exponent, _) in components.items():
        if exponent == 0:
            continue
        try:
            base = DEFAULT_REGISTRY.get(symbol)
        except ValueError:
            return None

        abs_exp = abs(exponent)
        unit_part = base if abs_exp == 1 else (base ** abs_exp)
        if exponent > 0:
            numer_parts.append(unit_part)
        else:
            denom_parts.append(unit_part)

    numerator = _mul_units(numer_parts)
    denominator = _mul_units(denom_parts)

    if numerator is None and denominator is None:
        return Unit("", 1.0, DIM_0)
    if numerator is None:
        numerator = Unit("", 1.0, DIM_0)
    if denominator is None:
        return numerator
    return numerator / denominator


def _si_to_value_unit(
    mag_si: float,
    dim: Dim,
    components: SymbolComponents | None = None,
    requested_unit: Unit | None = None,
) -> tuple[float, Unit]:
    """Convert an SI magnitude into a value/unit tuple using canonical units."""
    if dim == DIM_0:
        unit = Unit("", 1.0, DIM_0)
        return mag_si / unit.scale_to_si, unit

    def _should_preserve(name: str) -> bool:
        return not any(ch in name for ch in ("*", "/", "^", "·"))

    if requested_unit is not None and _should_preserve(requested_unit.name):
        return mag_si / requested_unit.scale_to_si, requested_unit

    if components:
        single_axis = _dim_single_axis(dim)
        if single_axis is not None:
            axis_idx, target_exp = single_axis
            chosen = _choose_symbol_for_axis(components, axis_idx, target_exp)
            if chosen is not None:
                final_unit = chosen ** target_exp

                if (
                    requested_unit is None
                    and abs(target_exp) == 1
                ):
                    from quantium.core.utils import preferred_symbol_for_dim
                    from quantium.units.registry import DEFAULT_REGISTRY, PREFIXES

                    pref_sym = preferred_symbol_for_dim(dim)
                    if pref_sym and final_unit.name == pref_sym and dim != DIM_0:
                        allowed_prefixes = [
                            p for p in PREFIXES if p.symbol in _ALLOWED_CANON_PREFIX_SYMBOLS
                        ]

                        candidates: list[tuple[float, Unit]] = []

                        base_value = mag_si / final_unit.scale_to_si
                        candidates.append((base_value, final_unit))

                        for prefix in allowed_prefixes:
                            symbol = f"{prefix.symbol}{pref_sym}"
                            try:
                                candidate_unit = DEFAULT_REGISTRY.get(symbol)
                            except Exception:
                                continue
                            if target_exp == -1:
                                candidate_unit = candidate_unit ** -1
                            candidate_value = mag_si / candidate_unit.scale_to_si
                            candidates.append((candidate_value, candidate_unit))

                        def _score(entry: tuple[float, Unit]) -> tuple[int, float]:
                            value, _ = entry
                            if value == 0.0:
                                return (0, 0.0)
                            abs_val = abs(value)
                            if 1.0 <= abs_val < 1000.0:
                                return (0, abs(math.log10(abs_val)))
                            return (1, abs(math.log10(abs_val)))

                        best_value, best_unit = min(candidates, key=_score)
                        return best_value, best_unit

                return mag_si / final_unit.scale_to_si, final_unit

        from quantium.units.registry import DEFAULT_REGISTRY

        ordered_components = sorted(
            (
                (symbol, exponent, order)
                for symbol, (exponent, order) in components.items()
                if exponent
            ),
            key=lambda item: item[2],
        )

        if ordered_components:
            # If all component symbols resolve to the same base dimension we can
            # attempt to collapse them into a single power of a single unit.
            # Pick the "best" reference unit among the components. Previously
            # we always used the first symbol which caused undesired results
            # like `N * kN^2 -> N^3`. Prefer the component with the largest
            # absolute exponent (tie-break: earlier/left-most component).
            candidates: list[tuple[str, int, Unit, tuple[int, int]]] = []
            total_exp = 0
            all_same_dim = True
            reference_dim: Dim | None = None

            for symbol, exponent, order in ordered_components:
                try:
                    candidate_unit = DEFAULT_REGISTRY.get(symbol)
                except ValueError:
                    all_same_dim = False
                    break

                if reference_dim is None:
                    reference_dim = candidate_unit.dim
                elif candidate_unit.dim != reference_dim:
                    all_same_dim = False
                    break

                candidates.append((symbol, exponent, candidate_unit, order))
                total_exp += exponent

            if all_same_dim and candidates and total_exp != 0:
                # choose by (abs(exponent), -priority, -index) so larger exponents
                # win; ties prefer lower priority (earlier operand) and then lower
                # index.
                def _pick_key(entry: tuple[str, int, Unit, tuple[int, int]]):
                    _, exp, _, ordt = entry
                    return (abs(exp), -ordt[0], -ordt[1])

                best = max(candidates, key=_pick_key)
                _, _, reference_unit, _ = best
                canonical_unit = reference_unit ** total_exp
                if canonical_unit.dim == dim:
                    return mag_si / canonical_unit.scale_to_si, canonical_unit

        composite = _unit_from_components(components)
        if composite is not None and composite.dim == dim:
            from quantium.core.utils import preferred_symbol_for_dim
            from quantium.units.registry import DEFAULT_REGISTRY, PREFIXES

            match = _match_preferred_power(dim)
            if match:
                base_sym, power = match
                try:
                    base_unit = DEFAULT_REGISTRY.get(base_sym)
                except Exception:
                    base_unit = None

                if base_unit is not None:
                    canonical_unit = base_unit ** power
                    return mag_si / canonical_unit.scale_to_si, canonical_unit

            pref_sym = preferred_symbol_for_dim(dim)
            if pref_sym and dim != DIM_0:
                try:
                    pref_unit = DEFAULT_REGISTRY.get(pref_sym)
                except Exception:
                    pref_unit = None

                if pref_unit is not None and pref_unit.scale_to_si > 0:
                    allowed_prefixes = [
                        p for p in PREFIXES if p.symbol in _ALLOWED_CANON_PREFIX_SYMBOLS
                    ]

                    candidates: list[tuple[float, Unit]] = []

                    base_value = mag_si / pref_unit.scale_to_si
                    candidates.append((base_value, pref_unit))

                    for prefix in allowed_prefixes:
                        symbol = f"{prefix.symbol}{pref_sym}"
                        try:
                            candidate_unit = DEFAULT_REGISTRY.get(symbol)
                        except Exception:
                            continue
                        candidate_value = mag_si / candidate_unit.scale_to_si
                        candidates.append((candidate_value, candidate_unit))

                    def _score(entry: tuple[float, Unit]) -> tuple[int, float]:
                        value, _ = entry
                        if value == 0.0:
                            return (0, 0.0)
                        abs_val = abs(value)
                        if 1.0 <= abs_val < 1000.0:
                            return (0, abs(math.log10(abs_val)))
                        return (1, abs(math.log10(abs_val)))

                    best_value, best_unit = min(candidates, key=_score)
                    return best_value, best_unit

            return mag_si / composite.scale_to_si, composite

    unit = _canonical_unit_for_dim(dim)
    return mag_si / unit.scale_to_si, unit


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
        scalar = float(value)
        if scalar == 0.0:
            mag_si = 0.0  # keep exact zero to avoid floating noise
            components = _unit_symbol_map(self)
            val, unit = _si_to_value_unit(mag_si, self.dim, components)
            return Quantity(val, unit)
        return Quantity(scalar, self)
    
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

        components = _combine_symbol_maps(
            _unit_symbol_map(self, 0),
            _unit_symbol_map(other, 1),
        )

        if (
            new_dim != DIM_0
            and isclose(new_scale, 1.0, rel_tol=1e-12, abs_tol=0.0)
            and "1" not in components
        ):
            from quantium.core.utils import preferred_symbol_for_dim

            preferred = preferred_symbol_for_dim(new_dim)
            if preferred:
                return Unit(preferred, 1.0, new_dim)

        new_unit_name = _format_unit_components(components)
        if new_dim == DIM_0:
            new_unit_name = ""
        return Unit(new_unit_name, new_scale, new_dim)


    def __truediv__(self, other: "Unit") -> "Unit":
        new_dim = dim_div(self.dim, other.dim)
        new_scale = self.scale_to_si / other.scale_to_si

        components = _combine_symbol_maps(
            _unit_symbol_map(self, 0),
            _scale_symbol_map(_unit_symbol_map(other, 1), -1),
        )

        if (
            new_dim != DIM_0
            and isclose(new_scale, 1.0, rel_tol=1e-12, abs_tol=0.0)
            and "1" not in components
        ):
            from quantium.core.utils import preferred_symbol_for_dim

            preferred = preferred_symbol_for_dim(new_dim)
            if preferred:
                return Unit(preferred, 1.0, new_dim)

        new_unit_name = _format_unit_components(components)
        if new_dim == DIM_0:
            new_unit_name = ""

        return Unit(new_unit_name, new_scale, new_dim)
    
    def __rtruediv__(self, n: int | float) -> Unit:
        if n != 1:
            raise TypeError(
                f"Invalid operation: cannot divide {n} by a Unit ({self.name}). "
                "Only 1/unit (reciprocal) is supported."
            )

        new_dim = dim_div(DIM_0, self.dim)
        components = _scale_symbol_map(_unit_symbol_map(self, 0), -1)
        new_scale = 1 / self.scale_to_si
        new_name = _format_unit_components(components)
        if new_dim == DIM_0:
            new_name = ""
        return Unit(new_name, new_scale, new_dim)
        

    def __pow__(self, n: int) -> Unit:
        new_dim = dim_pow(self.dim, n)
        # Canonical naming:
        if n == 0:
            normalized_name = ""
        else:
            components = _scale_symbol_map(_unit_symbol_map(self, 0), n)
            normalized_name = _format_unit_components(components)
            if not normalized_name:
                normalized_name = ""

        new_scale = self.scale_to_si ** n
        return Unit(normalized_name, new_scale, new_dim)


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

    def _symbol_component_map(self, priority: int) -> SymbolComponents:
        return _unit_symbol_map(self.unit, priority)
        
    def _check_dim_compatible(self, other: object) -> None:
        """Internal helper to raise TypeError on dimension mismatch."""
        if not isinstance(other, Quantity):
            # Allow comparison with 0 (dimensionless)
            if isinstance(other, (int, float)) and other == 0:
                if self.dim != DIM_0:
                    raise TypeError("Cannot compare a dimensioned quantity to 0")
                return  # It's a 0 dimensionless quantity, OK
            raise TypeError(f"Cannot compare Quantity with type {type(other)}")

        if self.dim != other.dim:
            raise TypeError(
                f"Cannot compare quantities with different dimensions: "
                f"'{self.unit.name}' and '{other.unit.name}'"
            )
        
    def _is_close(self, other_si_mag: float) -> bool:
        """Internal helper for fuzzy equality."""
        return isclose(self._mag_si, other_si_mag, rel_tol=1e-12, abs_tol=0.0)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        # Same physical dimension; SI magnitudes equal within tolerance.
        return (
            self.dim == other.dim
            and isclose(self._mag_si, other._mag_si, rel_tol=1e-12, abs_tol=0.0)
        )
    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.dim != other.dim:
            return True # Not equal if dims don't match
        return not self._is_close(other._mag_si)
    
    def __lt__(self, other: object) -> bool:
        self._check_dim_compatible(other)
        other_si_mag = getattr(other, '_mag_si', 0.0)
        # Strictly less than AND not fuzzy-equal
        return self._mag_si < other_si_mag and not self._is_close(other_si_mag)

    def __le__(self, other: object) -> bool:
        self._check_dim_compatible(other)
        other_si_mag = getattr(other, '_mag_si', 0.0)
        # Less than OR fuzzy-equal
        return self._mag_si < other_si_mag or self._is_close(other_si_mag)

    def __gt__(self, other: object) -> bool:
        self._check_dim_compatible(other)
        other_si_mag = getattr(other, '_mag_si', 0.0)
        # Strictly greater than AND not fuzzy-equal
        return self._mag_si > other_si_mag and not self._is_close(other_si_mag)

    def __ge__(self, other: object) -> bool:
        self._check_dim_compatible(other)
        other_si_mag = getattr(other, '_mag_si', 0.0)
        # Greater than OR fuzzy-equal
        return self._mag_si > other_si_mag or self._is_close(other_si_mag)
    
    # --- Hashing Solution ---

    def as_key(self, precision: int = 12) -> tuple:
        """
        Returns a hashable, discretized key for this quantity.

        This is the recommended way to use Quantities in dictionaries
        or sets, as it forces the user to choose a precision
        level for "fuzzy" hashing.

        The standard `__hash__` is not implemented because `__eq__`
        uses `isclose`, which would violate the Python hash contract.

        Usage:
        >>> my_dict = {}
        >>> q1 = (1.0 + 1e-13) * u.m
        >>> q2 = (1.0 - 1e-13) * u.m
        >>>
        >>> # q1 and q2 are "equal" but not hash-equal
        >>> q1 == q2  # True
        >>>
        >>> # Using as_key forces them to be hash-equal
        >>> my_dict[q1.as_key(precision=9)] = "value"
        >>> print(my_dict[q2.as_key(precision=9)])
        "value"

        Parameters
        ----------
        precision : int, optional
            The number of decimal places to round the *SI magnitude*
            to for hashing, by default 12 (which is typically
            near 64-bit float precision limits).

        Returns
        -------
        tuple
            A hashable tuple of (dimension, rounded_si_magnitude).
        """
        # Round the SI magnitude to the specified precision
        rounded_mag_si = round(self._mag_si, precision)
        
        # We must also handle -0.0 vs 0.0, which round identically
        # but have different hashes.
        if rounded_mag_si == 0.0:
            rounded_mag_si = 0.0
            
        return (self.dim, rounded_mag_si)

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

        components = _unit_symbol_map(new_unit)
        value, unit = _si_to_value_unit(self._mag_si, self.dim, components, new_unit)
        return Quantity(value, unit)
        
    
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

    @property
    def si(self) -> Quantity:
        return self.to_si()
    
    @property
    def value(self) -> float:
        return self._mag_si / self.unit.scale_to_si

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
            result_mag_si = self._mag_si * other.scale_to_si
            result_dim = dim_mul(self.dim, other.dim)
            components = _combine_symbol_maps(
                self._symbol_component_map(0),
                _unit_symbol_map(other, 1),
            )
            value, unit = _si_to_value_unit(result_mag_si, result_dim, components)
            return Quantity(value, unit)
        
        # quantity × quantity
        result_mag_si = self._mag_si * other._mag_si
        result_dim = dim_mul(self.dim, other.dim)
        components = _combine_symbol_maps(
            self._symbol_component_map(0),
            other._symbol_component_map(1),
        )
        value, unit = _si_to_value_unit(result_mag_si, result_dim, components)
        return Quantity(value, unit)

    def __rmul__(self, other: float | int) -> "Quantity":
        # allows 3 * (2 m) -> 6 m
        return self.__mul__(other)

    def __truediv__(self, other: "Quantity | Unit | Number") -> "Quantity":
        # quantity / scalar
        if isinstance(other, (int, float)):
            return Quantity((self._mag_si / float(other)) / self.unit.scale_to_si, self.unit)
        
        # quantity / unit
        if isinstance(other, Unit):
            result_mag_si = self._mag_si / other.scale_to_si
            result_dim = dim_div(self.dim, other.dim)
            components = _combine_symbol_maps(
                self._symbol_component_map(0),
                _scale_symbol_map(_unit_symbol_map(other, 1), -1),
            )
            value, unit = _si_to_value_unit(result_mag_si, result_dim, components)
            return Quantity(value, unit)

        # quantity / quantity
        result_mag_si = self._mag_si / other._mag_si
        result_dim = dim_div(self.dim, other.dim)
        components = _combine_symbol_maps(
            self._symbol_component_map(0),
            _scale_symbol_map(other._symbol_component_map(1), -1),
        )
        value, unit = _si_to_value_unit(result_mag_si, result_dim, components)
        return Quantity(value, unit)

    def __rtruediv__(self, other: float | int) -> "Quantity":
        # scalar / quantity  -> returns Quantity with inverse dimension
        if not isinstance(other, (int, float)):
            return NotImplemented
        result_dim = dim_div(DIM_0, self.dim)
        result_mag_si = float(other) / self._mag_si
        components = _scale_symbol_map(self._symbol_component_map(0), -1)
        value, unit = _si_to_value_unit(result_mag_si, result_dim, components)
        return Quantity(value, unit)

    def __pow__(self, n: int) -> "Quantity":
        new_unit = self.unit ** n
        return Quantity((self._mag_si ** n) / new_unit.scale_to_si, new_unit)
    
    def __repr__(self) -> str:
        # Local imports avoid cyclic imports; modules are cached after the first time.
        from quantium.core.utils import (
            preferred_symbol_for_dim,
            prettify_unit_name_supers,
        )
        from quantium.units.registry import PREFIXES
        from math import log10, floor  # Added import for math functions

        # Numeric magnitude in the *current* unit
        mag = self._mag_si / self.unit.scale_to_si

        # Dimensionless: print bare number
        if self.dim == DIM_0:
            return f"{mag:.15g}"

        # Start from the user’s unit name (keeps cm/ms etc.), with superscripts & cancellation
        # This is CRITICAL: it cancels "kg·mg/kg" to "mg" *before* we check composition.
        pretty = prettify_unit_name_supers(self.unit.name, cancel=True)

        # CRITICAL: Check if the *prettified* name is composed.
        # This check prevents re-formatting of simple units like "cm", "mg", "Pa", "Bq",
        # which fixes regressions.
        is_composed = any(ch in pretty for ch in ("/", "·", "^"))

        if is_composed:
            # Respect the stored composed unit name for representation.
            # We deliberately avoid performing an additional canonicalisation
            # here so that `Quantity.unit.name` matches the printed output.
            return f"{mag:.15g}" if pretty == "1" else f"{mag:.15g} {pretty}"

        # If the pretty name reduces to "1", show just the number
        # This also handles all non-composed units that skipped the `if` block.
        return f"{mag:.15g}" if pretty == "1" else f"{mag:.15g} {pretty}"


    
    def __format__(self, spec: str) -> str:
        """
        Custom string formatting for Quantity objects.

        The format specifier controls whether the quantity is shown in its
        current unit or converted to SI units before printing.

        Supported specifiers
        --------------------
        "" (empty), or "native"
            Display the quantity in its current unit (default).
        "si"
            Display the quantity converted to SI units.

        Examples
        --------
        >>> v = 1000 @ (ureg.get("cm") / ureg.get("s"))
        >>> f"{v}"           # default: show in current unit (cm/s)
        '1000 cm/s'
        >>> f"{v:native}"      # explicit but same as above
        '1000 cm/s'
        >>> f"{v:si}"        # convert and show in SI (m/s)
        '10 m/s'

        Raises
        ------
        ValueError
            If the format specifier is not one of "", "native", or "si".
        """
        spec = (spec or "").strip().lower()
        if spec in ("", "native"):
            return repr(self)          # current native (default)
        if spec == "si":
            return repr(self.to_si())  # force SI
        raise ValueError("Unknown format spec; use '', 'native', or 'si'")




        

        

    