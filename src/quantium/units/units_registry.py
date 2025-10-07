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
def get_unit(symbol: str) -> Unit:
    """Return the `Unit` instance corresponding to the given symbol."""
    try:
        return UNIT_REGISTRY[symbol]
    except KeyError:
        raise ValueError(f"Unknown unit symbol: {symbol}")

def register_unit(unit: Unit) -> None:
    """Add a new `Unit` instance to the registry."""
    UNIT_REGISTRY[unit.name] = unit