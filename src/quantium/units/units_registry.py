"""
quantium.units.units_registry
=============================

A centralized registry of all defined units used by the Quantium framework.

This module maintains a global mapping from unit symbols (e.g., "m", "s", "kg")
to their corresponding `Unit` instances, allowing dynamic lookup and extension
of supported units.
"""

from quantium.core.quantity import Unit
from quantium.core.dimensions import (
    L, M, T, I, THETA, N, J, DIM_0,
    dim_mul, dim_div, dim_pow
)
import re

# ---------------------------------------------------------------------------
# Base SI Units
# ---------------------------------------------------------------------------
meter = Unit("m", 1.0, L)           # Length
kilogram = Unit("kg", 1.0, M)       # Mass
second = Unit("s", 1.0, T)          # Time
ampere = Unit("A", 1.0, I)          # Electric current
kelvin = Unit("K", 1.0, THETA)      # Thermodynamic temperature
mole = Unit("mol", 1.0, N)          # Amount of substance
candela = Unit("cd", 1.0, J)        # Luminous intensity

# ---------------------------------------------------------------------------
# Common Derived SI Units
# ---------------------------------------------------------------------------
# Plane angle & solid angle (dimensionless but named)
radian = Unit("rad", 1.0, DIM_0)
steradian = Unit("sr", 1.0, DIM_0)

# Gram
gram = Unit("g", 1e-3, M)

# Frequency
hertz = Unit("Hz", 1.0, dim_pow(T, -1))

# Force
newton = Unit("N", 1.0, dim_mul(M, dim_div(L, dim_pow(T, 2))))  # kg·m/s²

# Pressure
pascal = Unit("Pa", 1.0, dim_div(newton.dim, dim_pow(L, 2)))    # N/m²

# Energy / Work / Heat
joule = Unit("J", 1.0, dim_mul(newton.dim, L))                  # N·m

# Power
watt = Unit("W", 1.0, dim_div(joule.dim, T))                    # J/s

# Electric charge
coulomb = Unit("C", 1.0, dim_mul(I, T))                         # A·s

# Electric potential
volt = Unit("V", 1.0, dim_div(watt.dim, I))                     # W/A

# Capacitance
farad = Unit("F", 1.0, dim_div(coulomb.dim, volt.dim))          # C/V

# Resistance
ohm = Unit("Ω", 1.0, dim_div(volt.dim, I))                      # V/A

# Conductance
siemens = Unit("S", 1.0, dim_div(I, volt.dim))                  # A/V

# Magnetic flux
weber = Unit("Wb", 1.0, dim_mul(volt.dim, T))                   # V·s

# Magnetic flux density
tesla = Unit("T", 1.0, dim_div(weber.dim, dim_pow(L, 2)))       # Wb/m²

# Inductance
henry = Unit("H", 1.0, dim_div(weber.dim, I))                   # Wb/A

# Luminous flux
lumen = Unit("lm", 1.0, dim_mul(candela.dim, steradian.dim))    # cd·sr

# Illuminance
lux = Unit("lx", 1.0, dim_div(lumen.dim, dim_pow(L, 2)))        # lm/m²

# Radioactivity
becquerel = Unit("Bq", 1.0, dim_pow(T, -1))                     # 1/s

# Absorbed dose, specific energy, kerma
gray = Unit("Gy", 1.0, dim_div(joule.dim, kilogram.dim))         # J/kg

# Dose equivalent
sievert = Unit("Sv", 1.0, gray.dim)                             # same as Gy

# Catalytic activity
katal = Unit("kat", 1.0, dim_div(mole.dim, T))                  # mol/s

# ---------------------------------------------------------------------------
# Global Unit Registry
# ---------------------------------------------------------------------------
UNIT_REGISTRY = {
    # Base units
    "m": meter,
    "kg": kilogram,
    "s": second,
    "A": ampere,
    "K": kelvin,
    "mol": mole,
    "cd": candela,

    # Derived units
    "g": gram,
    "rad": radian,
    "sr": steradian,
    "Hz": hertz,
    "N": newton,
    "Pa": pascal,
    "J": joule,
    "W": watt,
    "C": coulomb,
    "V": volt,
    "F": farad,
    "Ω": ohm,
    "S": siemens,
    "Wb": weber,
    "T": tesla,
    "H": henry,
    "lm": lumen,
    "lx": lux,
    "Bq": becquerel,
    "Gy": gray,
    "Sv": sievert,
    "kat": katal,
}

# ---------------------------------------------------------------------------
# Registry Helpers
# ---------------------------------------------------------------------------
PREFIXES = {
    # Large prefixes
    "Q": 1e30,   # quetta
    "R": 1e27,   # ronna
    "Y": 1e24,   # yotta
    "Z": 1e21,   # zetta
    "E": 1e18,   # exa
    "P": 1e15,   # peta
    "T": 1e12,   # tera
    "G": 1e9,    # giga
    "M": 1e6,    # mega
    "k": 1e3,    # kilo
    "h": 1e2,    # hecto
    "da": 1e1,   # deca (note: 2 letters)

    # Small prefixes
    "d": 1e-1,   # deci
    "c": 1e-2,   # centi
    "m": 1e-3,   # milli
    "µ": 1e-6,   # micro (Greek mu)
    "n": 1e-9,   # nano
    "p": 1e-12,  # pico
    "f": 1e-15,  # femto
    "a": 1e-18,  # atto
    "z": 1e-21,  # zepto
    "y": 1e-24,  # yocto
    "r": 1e-27,  # ronto
    "q": 1e-30,  # quecto
}

# ---------------------------------------------------------------------------
# Prefix Utilities
# ---------------------------------------------------------------------------

def register_unit(unit: Unit) -> None:
    """Add a new `Unit` instance to the registry."""
    UNIT_REGISTRY[unit.name] = unit


def _normalize_symbol(s: str) -> str:
    """
    Normalize common ASCII variants in unit symbols:
      - Leading 'u' → 'µ' (treat as micro prefix)
      - Any 'ohm' (case-insensitive) → 'Ω'
    """
    if s.startswith("u"):           # don't consult PREFIXES; accept ASCII 'u' as micro alias
        s = "µ" + s[1:]

    # Replace all 'ohm' substrings, case-insensitive
    s = re.sub(r"(?i)ohm", "Ω", s)
    return s

_PREFIXES_BY_LEN_DESC = tuple(sorted(PREFIXES.keys(), key=len, reverse=True))

def _split_prefix(symbol: str):
    for p in _PREFIXES_BY_LEN_DESC:
        if symbol.startswith(p):
            return p, symbol[len(p):]
    return None, symbol


def _is_prefixed_symbol(sym: str) -> bool:
    """
    Return True if `sym` looks like a prefixed unit we could support:
    i.e., starts with a valid prefix and the rest exists in UNIT_REGISTRY.
    """
    p, rest = _split_prefix(sym)
    return p is not None and rest in UNIT_REGISTRY


def _prefix_add(symbol: str) -> bool:
    """
    Try to add a prefixed unit named `symbol` to UNIT_REGISTRY.

    Rules:
    - `symbol` must begin with a known SI prefix from PREFIXES.
    - The remainder (base symbol) must already exist in UNIT_REGISTRY.
    - Do NOT allow stacking prefixes (e.g., 'k' + 'um' -> reject).
    - If `symbol` already exists, do nothing and return False.

    Returns:
        True if a new unit was created and added; False otherwise.
    """

    symbol = _normalize_symbol(symbol)
    
    # Already present? Nothing to do.
    if symbol in UNIT_REGISTRY:
        return False

    # Split into prefix and base part
    prefix, base_sym = _split_prefix(symbol)
    if prefix is None or not base_sym:
        return False  # no valid prefix or empty base

    # Base must exist
    base_unit = UNIT_REGISTRY.get(base_sym)
    if base_unit is None:
        return False

    # Prevent stacking: if the base itself looks prefixed, reject
    if _is_prefixed_symbol(base_sym):
        return False

    # Create and register the new prefixed unit via helper
    scale = PREFIXES[prefix]
    new_unit = Unit(symbol, base_unit.scale_to_si * scale, base_unit.dim)
    register_unit(new_unit)
    return True

def get_unit(symbol: str) -> Unit:
    """Return the Unit for `symbol`, generating a prefixed unit on the fly if needed."""
    # Fast path: already known
    symbol = _normalize_symbol(symbol)
    unit = UNIT_REGISTRY.get(symbol)
    if unit is not None:
        return unit

    # Try to synthesize a prefixed unit (no-op if invalid or stacked)
    if _prefix_add(symbol):
        return UNIT_REGISTRY[symbol]

    raise ValueError(f"Unknown unit symbol: {symbol}")

