"""
quantium.units.registry
=======================

A structured, extensible, and testable units registry for Quantium.

Key improvements over the prior design
--------------------------------------
- Encapsulates global state in a `UnitsRegistry` class (thread-safe).
- Data-driven registration of SI base/derived units.
- Normalization that handles ASCII fallbacks and Unicode NFC.
- Lazy, safe synthesis of prefixed units with anti-stacking checks.
- Support for aliases (e.g., "ohm" → "Ω", "Ohm" → "Ω").
- Clear public API: `register`, `register_alias`, `get`, `has`, `all`.
- Easily testable and embeddable (multiple registries for testing).

Assumptions
-----------
- `Unit(name: str, scale_to_si: float, dim)` is available from `quantium.core.quantity`.
- Dimension arithmetic helpers are in `quantium.core.dimensions`.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import re
import unicodedata
from typing import Dict, Iterable, Mapping, Optional, Tuple

from quantium.core.quantity import Unit
from quantium.core.dimensions import (
    L, M, T, I, THETA, N, J, DIM_0,
    dim_mul, dim_div, dim_pow,
)

# ---------------------------------------------------------------------------
# Prefix model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Prefix:
    symbol: str
    factor: float

# SI prefixes including 2022 additions (quetta/ronna/ronto/quecto)
PREFIXES: Tuple[Prefix, ...] = (
    # large
    Prefix("Q", 1e30),  # quetta
    Prefix("R", 1e27),  # ronna
    Prefix("Y", 1e24),  # yotta
    Prefix("Z", 1e21),  # zetta
    Prefix("E", 1e18),  # exa
    Prefix("P", 1e15),  # peta
    Prefix("T", 1e12),  # tera
    Prefix("G", 1e9),   # giga
    Prefix("M", 1e6),   # mega
    Prefix("k", 1e3),   # kilo
    Prefix("h", 1e2),   # hecto
    Prefix("da", 1e1),  # deca (two letters)
    # small
    Prefix("d", 1e-1),  # deci
    Prefix("c", 1e-2),  # centi
    Prefix("m", 1e-3),  # milli
    Prefix("µ", 1e-6),  # micro (Greek mu)
    Prefix("n", 1e-9),  # nano
    Prefix("p", 1e-12), # pico
    Prefix("f", 1e-15), # femto
    Prefix("a", 1e-18), # atto
    Prefix("z", 1e-21), # zepto
    Prefix("y", 1e-24), # yocto
    Prefix("r", 1e-27), # ronto
    Prefix("q", 1e-30), # quecto
)

# Ordered list of prefix symbols by descending length for robust matching
_PREFIX_SYMBOLS_DESC = tuple(sorted((p.symbol for p in PREFIXES), key=len, reverse=True))
_PREFIX_FACTORS: Mapping[str, float] = {p.symbol: p.factor for p in PREFIXES}

# ---------------------------------------------------------------------------
# Normalization & aliases
# ---------------------------------------------------------------------------
_ALIAS_TO_CANONICAL: Dict[str, str] = {
    # ohm aliases (case-insensitive handled separately)
    "ohm": "Ω",
}

_OHM_RE = re.compile(r"(?i)ohm")


def normalize_symbol(s: str) -> str:
    """Normalize user-provided unit symbols.

    Rules:
    - Unicode normalize to NFC (composed forms like "µ").
    - Replace ASCII leading 'u' micro with Greek 'µ' **only** at start.
    - Map textual aliases to canonical symbols (e.g. any 'ohm' → 'Ω').
    - Strip surrounding whitespace.
    - Leave case as-is except for alias mapping handled via regex.
    """
    if not s:
        return s

    s = s.strip()
    s = unicodedata.normalize("NFC", s)

    # Leading 'u' as ASCII micro → 'µ'
    if s.startswith("u"):
        s = "µ" + s[1:]

    # Replace all forms of 'ohm' with Ω
    s = _OHM_RE.sub("Ω", s)
    return s


# ---------------------------------------------------------------------------
# Units registry
# ---------------------------------------------------------------------------
class UnitsRegistry:
    """Thread-safe registry for `Unit` objects with SI prefix synthesis.

    This registry does *not* parse compound expressions (like "m/s^2").
    It focuses on atomic symbols (possibly prefixed). That higher-level
    parsing can be layered above this API.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._units: Dict[str, Unit] = {}
        self._aliases: Dict[str, str] = {}

    # -------------------------- public API ---------------------------------
    def register(self, unit: Unit) -> None:
        """Register (or overwrite) a `Unit` under its canonical name.

        Use `register_alias` to add additional spellings without duplication.
        """
        with self._lock:
            self._units[unit.name] = unit

    def register_alias(self, alias: str, canonical: str) -> None:
        with self._lock:
            self._aliases[normalize_symbol(alias)] = canonical

    def has(self, symbol: str) -> bool:
        try:
            self.get(symbol)
            return True
        except ValueError:
            return False

    def get(self, symbol: str) -> Unit:
        """Lookup a unit by symbol. If missing, try to synthesize via SI prefix.

        Raises `ValueError` if unknown.
        """
        sym = normalize_symbol(symbol)
        with self._lock:
            # alias redirect
            target = self._aliases.get(sym)
            if target is not None:
                sym = target

            u = self._units.get(sym)
            if u is not None:
                return u

            # Attempt to synthesize prefixed unit
            synthesized = self._try_synthesize_prefixed(sym)
            if synthesized is not None:
                return synthesized

        raise ValueError(f"Unknown unit symbol: {symbol}")

    def all(self) -> Mapping[str, Unit]:
        with self._lock:
            return dict(self._units)

    # ------------------------- internals -----------------------------------
    def _split_prefix(self, symbol: str) -> Tuple[Optional[str], str]:
        for p in _PREFIX_SYMBOLS_DESC:
            if symbol.startswith(p):
                return p, symbol[len(p):]
        return None, symbol

    def _looks_prefixed(self, symbol: str) -> bool:
        p, base = self._split_prefix(symbol)
        return p is not None and base in self._units

    def _try_synthesize_prefixed(self, sym: str) -> Optional[Unit]:
        # Already registered due to race? (cheap check)
        if sym in self._units:
            return self._units[sym]

        prefix, base_sym = self._split_prefix(sym)
        if prefix is None or not base_sym:
            return None

        base = self._units.get(base_sym)
        if base is None:
            return None

        # Prevent stacked prefixes: base itself must not be prefixed
        if self._looks_prefixed(base_sym):
            return None

        factor = _PREFIX_FACTORS[prefix]
        new_unit = Unit(sym, base.scale_to_si * factor, base.dim)
        self._units[sym] = new_unit
        return new_unit


# ---------------------------------------------------------------------------
# Bootstrap a default registry with SI units
# ---------------------------------------------------------------------------

def _bootstrap_default_registry() -> UnitsRegistry:
    reg = UnitsRegistry()

    # Base SI units
    base_units = (
        Unit("m", 1.0, L),           # length
        Unit("kg", 1.0, M),          # mass
        Unit("s", 1.0, T),           # time
        Unit("A", 1.0, I),           # electric current
        Unit("K", 1.0, THETA),       # temperature
        Unit("mol", 1.0, N),         # amount of substance
        Unit("cd", 1.0, J),          # luminous intensity
    )

    # Named, dimensionless
    derived_named = (
        Unit("rad", 1.0, DIM_0),
        Unit("sr", 1.0, DIM_0),
    )

    # Derived (data-driven to reduce boilerplate)
    # Format: (symbol, scale_to_si, dim)
    derived_units = (
        ("g", 1e-3, M),
        ("Hz", 1.0, dim_pow(T, -1)),
        ("N", 1.0, dim_mul(M, dim_div(L, dim_pow(T, 2)))),            # kg·m/s²
        ("Pa", 1.0, dim_div(dim_mul(M, dim_div(L, dim_pow(T, 2))), dim_pow(L, 2))),  # N/m²
        ("J", 1.0, dim_mul(dim_mul(M, dim_div(L, dim_pow(T, 2))), L)), # N·m
        ("W", 1.0, dim_div(dim_mul(dim_mul(M, dim_div(L, dim_pow(T, 2))), L), T)),   # J/s
        ("C", 1.0, dim_mul(I, T)),
        ("V", 1.0, dim_div(dim_div(dim_mul(dim_mul(M, dim_div(L, dim_pow(T, 2))), L), T), I)),
        ("F", 1.0, dim_div(dim_mul(I, T), dim_div(dim_div(dim_mul(dim_mul(M, dim_div(L, dim_pow(T, 2))), L), T), I))),
        ("Ω", 1.0, dim_div(dim_div(dim_mul(dim_mul(M, dim_div(L, dim_pow(T, 2))), L), T), I) ),
        ("S", 1.0, dim_div(I, dim_div(dim_div(dim_mul(dim_mul(M, dim_div(L, dim_pow(T, 2))), L), T), I))),
        ("Wb", 1.0, dim_mul(dim_div(dim_div(dim_mul(dim_mul(M, dim_div(L, dim_pow(T, 2))), L), T), I), T)),
        ("T", 1.0, dim_div(dim_mul(dim_div(dim_div(dim_mul(dim_mul(M, dim_div(L, dim_pow(T, 2))), L), T), I), T), dim_pow(L, 2))),
        ("H", 1.0, dim_div(dim_mul(dim_div(dim_div(dim_mul(dim_mul(M, dim_div(L, dim_pow(T, 2))), L), T), I), T), I)),
        ("lm", 1.0, dim_mul(J, DIM_0)),  # cd·sr (sr is DIM_0)
        ("lx", 1.0, dim_div(dim_mul(J, DIM_0), dim_pow(L, 2))),
        ("Bq", 1.0, dim_pow(T, -1)),
        ("Gy", 1.0, dim_div(dim_mul(dim_div(L, dim_pow(T, 2)), M), M)),  # J/kg = (N·m)/kg simplified; same as original Gy dim
        ("Sv", 1.0, dim_div(dim_mul(dim_div(L, dim_pow(T, 2)), M), M)),  # same as Gy
        ("kat", 1.0, dim_div(N, T)),
    )

    # Register all
    for u in base_units:
        reg.register(u)
    for u in derived_named:
        reg.register(u)
    for sym, scale, dim in derived_units:
        reg.register(Unit(sym, scale, dim))

    # Common aliases
    reg.register_alias("ohm", "Ω")
    reg.register_alias("Ohm", "Ω")
    reg.register_alias("OHM", "Ω")

    return reg


# Public, shared default registry
DEFAULT_REGISTRY: UnitsRegistry = _bootstrap_default_registry()


# ---------------------------------------------------------------------------
# Convenience functions mirroring the old API (optional)
# ---------------------------------------------------------------------------

def register_unit(unit: Unit) -> None:
    DEFAULT_REGISTRY.register(unit)


def get_unit(symbol: str) -> Unit:
    return DEFAULT_REGISTRY.get(symbol)


def register_alias(alias: str, canonical: str) -> None:
    DEFAULT_REGISTRY.register_alias(alias, canonical)


def all_units() -> Mapping[str, Unit]:
    return DEFAULT_REGISTRY.all()


__all__ = [
    "Prefix",
    "UnitsRegistry",
    "DEFAULT_REGISTRY",
    "register_unit",
    "register_alias",
    "get_unit",
    "all_units",
]
