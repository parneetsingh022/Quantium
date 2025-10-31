from dataclasses import dataclass
from typing import Tuple

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