"""Utilities for canonical unit name simplification.

This module centralises the logic used to canonicalise and simplify unit names
within the quantity system.  The functionality is wrapped inside
``UnitNameSimplifier`` so it can be reused (and more easily tested) without
keeping a tight coupling to ``quantium.core.quantity``.
"""

from __future__ import annotations

from functools import lru_cache
import math
import re
from typing import Dict, List, Tuple, TYPE_CHECKING

from quantium.core.dimensions import DIM_0, Dim, dim_pow
from quantium.core.utils import _tokenize_name_merge

if TYPE_CHECKING:  # pragma: no cover - import only used for typing
    from quantium.core.quantity import Unit

SymbolComponents = Dict[str, Tuple[int, Tuple[int, int]]]


_POWER_RE = re.compile(r"^(?P<base>.+?)\^(?P<exp>-?\d+)$")
_MAX_CANON_POWER = 12
_ALLOWED_CANON_PREFIX_SYMBOLS = frozenset({"T", "G", "M", "k", "m", "\u00b5", "p"})


def _dim_key(dim: Dim) -> tuple[int, ...]:
    """Return a hashable key for cached dimension lookups."""
    return tuple(dim)


class UnitNameSimplifier:
    """Encapsulates the heuristics for producing canonical unit names."""

    def __init__(self, unit_cls: type["Unit"]) -> None:
        self._unit_cls = unit_cls

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_power_name(name: str) -> str:
        """Normalise power expressions such as ``x^1`` or ``x^0``."""
        match = _POWER_RE.match(name)
        if not match:
            return name
        base = match.group("base")
        exp = int(match.group("exp"))
        if exp == 1:
            return base
        if exp == 0:
            return "1"
        return f"{base}^{exp}"

    @staticmethod
    @lru_cache(maxsize=None)
    def _preferred_dim_symbol_map() -> dict[tuple[int, ...], str]:
        """Cache symbols for SI-coherent units (scale 1)."""
        from quantium.core.utils import preferred_symbol_for_dim
        from quantium.units.registry import DEFAULT_REGISTRY

        mapping: dict[tuple[int, ...], str] = {}
        for name, unit in DEFAULT_REGISTRY.all().items():
            sym = preferred_symbol_for_dim(unit.dim)
            if sym:
                mapping.setdefault(_dim_key(unit.dim), sym)
        return mapping

    @classmethod
    def _match_preferred_power(cls, dim: Dim) -> tuple[str, int] | None:
        """Detect if ``dim`` is a power of a known preferred symbol dimension."""
        base_map = cls._preferred_dim_symbol_map()
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

    def canonical_unit_for_dim(self, dim: Dim) -> "Unit":
        """Return a canonical unit (scale 1) for the provided dimension."""
        unit_cls = self._unit_cls
        if dim == DIM_0:
            return unit_cls("", 1.0, DIM_0)

        from quantium.core.utils import format_dim, preferred_symbol_for_dim

        sym = preferred_symbol_for_dim(dim)
        if sym:
            return unit_cls(sym, 1.0, dim)

        match = self._match_preferred_power(dim)
        if match:
            base_sym, power = match
            name = self.normalize_power_name(f"{base_sym}^{power}")
            return unit_cls(name, 1.0, dim)

        name = format_dim(dim)
        if name == "1":
            return unit_cls("", 1.0, DIM_0)
        return unit_cls(name, 1.0, dim)

    # ------------------------------------------------------------------
    # Symbol component utilities
    # ------------------------------------------------------------------
    def unit_symbol_map(self, unit: "Unit", priority: int = 0) -> SymbolComponents:
        if not unit.name:
            return {}

        raw = list(_tokenize_name_merge(unit.name).items())
        if raw:
            mapping: SymbolComponents = {}
            for idx, (symbol, exponent) in enumerate(raw):
                mapping[symbol] = (exponent, (priority, idx))
            return mapping

        return {unit.name: (1, (priority, 0))}

    @staticmethod
    def scale_symbol_map(components: SymbolComponents, factor: int) -> SymbolComponents:
        if factor == 1:
            return dict(components)
        scaled: SymbolComponents = {}
        for symbol, (exponent, order) in components.items():
            new_exp = exponent * factor
            if new_exp != 0:
                scaled[symbol] = (new_exp, order)
        return scaled

    @staticmethod
    def combine_symbol_maps(*maps: SymbolComponents) -> SymbolComponents:
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

    @staticmethod
    def format_unit_components(components: SymbolComponents) -> str:
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

    @staticmethod
    def _dim_single_axis(dim: Dim) -> tuple[int, int] | None:
        axes = [idx for idx, power in enumerate(dim) if power != 0]
        if len(axes) != 1:
            return None
        axis = axes[0]
        return axis, dim[axis]

    def _choose_symbol_for_axis(
        self,
        components: SymbolComponents,
        axis_idx: int,
        target_exp: int,
    ) -> "Unit" | None:
        from quantium.units.registry import DEFAULT_REGISTRY

        target_sign = 1 if target_exp > 0 else -1
        best: tuple[str, int, "Unit"] | None = None
        best_score = -1
        fallback: tuple[str, int, "Unit"] | None = None
        fallback_score = -1

        ordered_items = sorted(components.items(), key=lambda item: item[1][1])

        for symbol, (exponent, _) in ordered_items:
            if exponent == 0:
                continue
            try:
                candidate = DEFAULT_REGISTRY.get(symbol)
            except ValueError:
                continue

            axis_info = self._dim_single_axis(candidate.dim)
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

    def _unit_from_components(self, components: SymbolComponents) -> "Unit" | None:
        """Reconstruct a composite unit from component symbol exponents."""
        if not components:
            return None

        from quantium.units.registry import DEFAULT_REGISTRY

        def _mul_units(units: List["Unit"]) -> "Unit" | None:
            result: "Unit" | None = None
            for u in units:
                result = u if result is None else result * u
            return result

        numer_parts: List["Unit"] = []
        denom_parts: List["Unit"] = []

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

        unit_cls = self._unit_cls
        if numerator is None and denominator is None:
            return unit_cls("", 1.0, DIM_0)
        if numerator is None:
            numerator = unit_cls("", 1.0, DIM_0)
        if denominator is None:
            return numerator
        return numerator / denominator

    # ------------------------------------------------------------------
    # Public API used by quantity/unit logic
    # ------------------------------------------------------------------
    def si_to_value_unit(
        self,
        mag_si: float,
        dim: Dim,
        components: SymbolComponents | None = None,
        requested_unit: "Unit" | None = None,
    ) -> tuple[float, "Unit"]:
        """Convert an SI magnitude into a value/unit tuple using canonical units."""
        unit_cls = self._unit_cls

        if dim == DIM_0:
            unit = unit_cls("", 1.0, DIM_0)
            return mag_si / unit.scale_to_si, unit

        def _should_preserve(name: str) -> bool:
            return not any(ch in name for ch in ("*", "/", "^", "·"))

        if requested_unit is not None and _should_preserve(requested_unit.name):
            return mag_si / requested_unit.scale_to_si, requested_unit

        if components:
            single_axis = self._dim_single_axis(dim)
            if single_axis is not None:
                axis_idx, target_exp = single_axis
                chosen = self._choose_symbol_for_axis(components, axis_idx, target_exp)
                if chosen is not None:
                    final_unit = chosen ** target_exp

                    if requested_unit is None and abs(target_exp) == 1:
                        from quantium.core.utils import preferred_symbol_for_dim
                        from quantium.units.registry import DEFAULT_REGISTRY, PREFIXES

                        pref_sym = preferred_symbol_for_dim(dim)
                        if pref_sym and final_unit.name == pref_sym and dim != DIM_0:
                            allowed_prefixes = [
                                p for p in PREFIXES if p.symbol in _ALLOWED_CANON_PREFIX_SYMBOLS
                            ]

                            prefix_candidates: list[tuple[float, "Unit"]] = []

                            base_value = mag_si / final_unit.scale_to_si
                            prefix_candidates.append((base_value, final_unit))

                            for prefix in allowed_prefixes:
                                symbol = f"{prefix.symbol}{pref_sym}"
                                try:
                                    candidate_unit = DEFAULT_REGISTRY.get(symbol)
                                except Exception:
                                    continue
                                if target_exp == -1:
                                    candidate_unit = candidate_unit ** -1
                                candidate_value = mag_si / candidate_unit.scale_to_si
                                prefix_candidates.append((candidate_value, candidate_unit))

                            def _score(entry: tuple[float, "Unit"]) -> tuple[int, float]:
                                value, _ = entry
                                if value == 0.0:
                                    return (0, 0.0)
                                abs_val = abs(value)
                                if 1.0 <= abs_val < 1000.0:
                                    return (0, abs(math.log10(abs_val)))
                                return (1, abs(math.log10(abs_val)))

                            best_value, best_unit = min(prefix_candidates, key=_score)
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
                component_candidates: list[tuple[str, int, "Unit", tuple[int, int]]] = []
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

                    component_candidates.append((symbol, exponent, candidate_unit, order))
                    total_exp += exponent

                if all_same_dim and component_candidates and total_exp != 0:
                    def _pick_key(entry: tuple[str, int, "Unit", tuple[int, int]]) -> tuple[int, int, int]:
                        _, exp, _, ordt = entry
                        return (abs(exp), -ordt[0], -ordt[1])

                    best = max(component_candidates, key=_pick_key)
                    _, _, reference_unit, _ = best
                    canonical_unit = reference_unit ** total_exp
                    if canonical_unit.dim == dim:
                        return mag_si / canonical_unit.scale_to_si, canonical_unit

            composite = self._unit_from_components(components)
            if composite is not None and composite.dim == dim:
                from quantium.core.utils import preferred_symbol_for_dim
                from quantium.units.registry import DEFAULT_REGISTRY, PREFIXES

                match = self._match_preferred_power(dim)
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

                        preferred_candidates: list[tuple[float, "Unit"]] = []

                        base_value = mag_si / pref_unit.scale_to_si
                        preferred_candidates.append((base_value, pref_unit))

                        for prefix in allowed_prefixes:
                            symbol = f"{prefix.symbol}{pref_sym}"
                            try:
                                candidate_unit = DEFAULT_REGISTRY.get(symbol)
                            except Exception:
                                continue
                            candidate_value = mag_si / candidate_unit.scale_to_si
                            preferred_candidates.append((candidate_value, candidate_unit))

                        def _score(entry: tuple[float, "Unit"]) -> tuple[int, float]:
                            value, _ = entry
                            if value == 0.0:
                                return (0, 0.0)
                            abs_val = abs(value)
                            if 1.0 <= abs_val < 1000.0:
                                return (0, abs(math.log10(abs_val)))
                            return (1, abs(math.log10(abs_val)))

                        best_value, best_unit = min(preferred_candidates, key=_score)
                        return best_value, best_unit

                return mag_si / composite.scale_to_si, composite

        unit = self.canonical_unit_for_dim(dim)
        return mag_si / unit.scale_to_si, unit


__all__ = ["SymbolComponents", "UnitNameSimplifier"]
