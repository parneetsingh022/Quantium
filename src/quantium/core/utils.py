"""
quantium.core.utils
====================

Utility functions for formatting and displaying physical dimensions and units
within the Quantium framework.

This module provides helper functions for representing dimensional exponents
and unit strings in a readable scientific format (e.g., 'kg·m/s²').
"""

from __future__ import annotations

# --- superscript + name-based prettifier (keeps units as written) ---
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Pattern,
    Tuple,
    cast,
)

# Backcompat for typing features if needed
try:
    from typing import Protocol, TypeAlias, runtime_checkable
except Exception:  # pragma: no cover - for older Python
    from typing_extensions import Protocol, TypeAlias, runtime_checkable

# A dimension is a 7-tuple of integer exponents: (L, M, T, I, Θ, N, J)
Dim: TypeAlias = Tuple[int, int, int, int, int, int, int]

_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sup(n: int) -> str:
    return "" if n == 1 else str(n).translate(_SUPERSCRIPTS)


# match token like "cm", "s^2", "m^(2)", "kg^sup(3)", or with unicode superscripts "m²"
_TOKEN_RE: Pattern[str] = re.compile(
    r"""
    \s*
    # Exclude superscript digits (⁰¹²³⁴⁵⁶⁷⁸⁹) and superscript minus (⁻)
    (?P<sym>[^·/\s^⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)
    (?:
        \^(\(?(?P<exp1>-?\d+)\)?          # ^2 or ^(2)
          |sup\((?P<exp2>-?\d+)\)         # ^sup(2)
        )
        |
        (?P<usup>[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)          # or existing unicode superscripts
    )?
    \s*
""",
    re.X,
)



def _parse_exponent(m: re.Match[str]) -> int:
    e = m.group("exp1") or m.group("exp2")
    if e is not None:
        return int(e)
    us = m.group("usup")
    if us:
        # map unicode superscripts back to int
        tbl = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
        return int(us.translate(tbl) or "1")
    return 1


def _tokenize_name_merge(name: str) -> Dict[str, int]:
    """Merge exponents from a composed unit name; cancel identical symbols.

    Correctly handles sequences like 'cm/ms^3·ms' as (cm / ms^3) · ms.
    """
    if not name or name == "1":
        return {}

    # Normalize ASCII separators
    name = name.replace("*", "·")

    parts = re.split(r"([·/])", name)  # keep separators
    op = "·"  # last operator seen; '·' or '/'
    exps: Dict[str, int] = {}

    for tok in parts:
        if not tok:
            continue
        if tok in ("·", "/"):
            op = tok
            continue
        if tok == "1":
            op = "·"
            continue

        m = _TOKEN_RE.fullmatch(tok)
        if m:
            sym = m.group("sym")
            e = _parse_exponent(m)
        else:
            sym, e = tok, 1

        # Apply operator to THIS token only
        if op == "/":
            e = -e
        exps[sym] = exps.get(sym, 0) + e

        # Reset to multiply for the next token
        op = "·"

    # Drop zeros
    return {k: v for k, v in exps.items() if v != 0}


def prettify_unit_name_supers(name: str, *, cancel: bool = True) -> str:
    """
    Pretty-print using the *existing* unit symbols (no SI conversion).
    Produces middle-dots and unicode superscripts, e.g. 'cm/ms', 'kg·m/s²'.
    """
    if not cancel:
        # just restyle exponents to superscripts and normalize separators
        name = name.replace("*", "·")
        name = re.sub(r"\^sup\((-?\d+)\)", lambda m: _sup(int(m.group(1))), name)
        name = re.sub(r"\^\((-?\d+)\)", lambda m: _sup(int(m.group(1))), name)
        name = re.sub(r"\^(-?\d+)", lambda m: _sup(int(m.group(1))), name)
        return name

    exps = _tokenize_name_merge(name)

    num: List[Tuple[str, int]] = sorted(
        [(s, e) for s, e in exps.items() if e > 0], key=lambda x: x[0]
    )
    den: List[Tuple[str, int]] = sorted(
        [(s, -e) for s, e in exps.items() if e < 0], key=lambda x: x[0]
    )

    def join(parts: List[Tuple[str, int]]) -> str:
        if not parts:
            return "1"
        return "·".join(f"{s}{_sup(p)}" for s, p in parts)

    num_s = join(num)
    den_s = join(den)
    return f"{num_s}/{den_s}" if den else num_s


# ---------- Dimension → pretty unit string ----------
def format_dim(dim: Dim) -> str:
    """
    Turn a dimension tuple (L,M,T,I,Θ,N,J) into 'kg·m/s²' style.
    Conventional order: M, L, T, I, Θ, N, J.
    """
    # indices: L=0 M=1 T=2 I=3 Θ=4 N=5 J=6
    labels: List[str] = ["m", "kg", "s", "A", "K", "mol", "cd"]
    order: List[int] = [1, 0, 2, 3, 4, 5, 6]  # M, L, T, I, Θ, N, J  (fixed order)

    num: List[str] = []
    den: List[str] = []
    for i in order:
        e = dim[i]
        if e > 0:
            num.append(labels[i] + _sup(e))
        elif e < 0:
            den.append(labels[i] + _sup(-e))

    numerator = "·".join(num) if num else "1"
    denominator = "·".join(den)
    return f"{numerator}/{denominator}" if denominator else numerator


def _dim_key(dim: Dim) -> Dim:
    return cast(Dim, tuple(dim))


_PREFERRED_ORDER = [ "A", "C", "N", "Pa", "J", "W", "V", "Ω", "S", "F", "Wb", "T", "H", "Hz", "lm", "lx", "Bq", "Gy", "Sv", "kat", "rad", "sr", "m", "kg", "s", "K", "mol", "cd", ]


@runtime_checkable
class _HasDimScale(Protocol):
    """Protocol for registry unit entries with SI scale and a dimension."""

    dim: Dim
    scale_to_si: float


# Build lazily (no import at module load)
_PREFERRED_BY_DIM: Optional[Dict[Dim, str]] = None


def _build_pref_map() -> Dict[Dim, str]:
    """
    Build a dimension -> preferred symbol map from the default registry.
    Only includes units whose scale to SI is exactly 1.0.
    """
    from quantium.units.registry import DEFAULT_REGISTRY
    pref: Dict[Dim, str] = {}
    reg: Any = DEFAULT_REGISTRY

    for sym in _PREFERRED_ORDER:                 # earlier = higher preference
        u = cast(Optional[_HasDimScale], reg.get(sym))
        if u and getattr(u, "scale_to_si", None) == 1.0:
            k = _dim_key(u.dim)
            pref.setdefault(k, sym)              # do not overwrite if already set
    return pref


def preferred_symbol_for_dim(dim: Dim) -> Optional[str]:
    """Return the preferred symbol (e.g., 'A', 'N', 'W') for a dimension, or None."""
    global _PREFERRED_BY_DIM
    if _PREFERRED_BY_DIM is None:
        _PREFERRED_BY_DIM = _build_pref_map()
    return _PREFERRED_BY_DIM.get(_dim_key(dim))


# Optional helper if you ever want to refresh after registering new units:
def invalidate_preferred_cache() -> None:
    global _PREFERRED_BY_DIM
    _PREFERRED_BY_DIM = None
