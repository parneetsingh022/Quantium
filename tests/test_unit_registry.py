# pytest tests for quantium.units.registry
#
# These tests exercise normalization, aliases, SI-prefix synthesis,
# anti-stacking rules, thread-safety, and the convenience functions that
# delegate to DEFAULT_REGISTRY. They purposefully use an isolated registry
# instance for most tests, and monkeypatch DEFAULT_REGISTRY where needed.

import threading
import time
import types
import importlib
import builtins
import pytest

from quantium.core.quantity import Unit
from quantium.core.dimensions import (
    L, M, T, I, THETA, N, J, DIM_0,
    dim_mul, dim_div, dim_pow,
)

# We import the module under test once, and access internals we intentionally
# rely on in tests (like _bootstrap_default_registry).
import quantium.units.registry as regmod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def reg():
    """Fresh, fully-bootstrapped UnitsRegistry for isolation per test."""
    # Using the module's bootstrap helper keeps the registry consistent with
    # the shipped set of units while avoiding global mutations.
    return regmod._bootstrap_default_registry()


@pytest.fixture()
def patched_default(monkeypatch, reg):
    """Temporarily replace DEFAULT_REGISTRY with an isolated instance."""
    monkeypatch.setattr(regmod, "DEFAULT_REGISTRY", reg)
    yield reg


# ---------------------------------------------------------------------------
# Base/derived presence & correctness
# ---------------------------------------------------------------------------

def test_base_units_present(reg):
    for sym, dim in [("m", L), ("kg", M), ("s", T), ("A", I), ("K", THETA), ("mol", N), ("cd", J)]:
        u = reg.get(sym)
        assert isinstance(u, Unit)
        assert u.name == sym
        assert u.scale_to_si == pytest.approx(1.0)
        assert u.dim == dim


def test_common_derived_units_present(reg):
    # spot-check a few dimensions/scales
    assert reg.get("rad").dim == DIM_0
    assert reg.get("sr").dim == DIM_0

    Hz = reg.get("Hz")
    assert Hz.dim == dim_pow(T, -1)
    assert Hz.scale_to_si == pytest.approx(1.0)

    N_unit = reg.get("N")
    assert N_unit.dim == dim_mul(M, dim_div(L, dim_pow(T, 2)))  # kg·m/s²

    Pa = reg.get("Pa")
    assert Pa.dim == dim_div(N_unit.dim, dim_pow(L, 2))

    J_unit = reg.get("J")
    assert J_unit.dim == dim_mul(N_unit.dim, L)

    W = reg.get("W")
    assert W.dim == dim_div(J_unit.dim, T)

    V = reg.get("V")
    assert V.dim == dim_div(W.dim, I)

    F = reg.get("F")
    assert F.dim == dim_div(reg.get("C").dim, V.dim)

    ohm = reg.get("Ω")
    assert ohm.dim == dim_div(V.dim, I)

    S = reg.get("S")
    assert S.dim == dim_div(I, V.dim)


# ---------------------------------------------------------------------------
# Normalization & aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp, expected", [
    ("um", "µm"),   # ASCII micro-prefix fallback at start
    ("uA", "µA"),
    ("ohm", "Ω"),   # textual alias → canonical
    ("Ohm", "Ω"),
    ("OHM", "Ω"),
])
def test_normalization_maps_to_canonical(inp, expected, reg):
    # `get` should just work with the normalized symbol
    # For prefixes, we expect synthesis if not pre-registered.
    u1 = reg.get(expected)
    u2 = reg.get(inp)
    assert u1.dim == u2.dim
    assert u1.scale_to_si == pytest.approx(u2.scale_to_si)


def test_alias_registration_custom(reg):
    # Create a new alias for ohm and confirm both path & object identity
    reg.register_alias("ohms", "Ω")
    u_alias = reg.get("ohms")
    u_ohm = reg.get("Ω")
    assert u_alias is u_ohm


# ---------------------------------------------------------------------------
# Prefix synthesis & anti-stacking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pref, base, factor", [
    ("k", "m", 1e3),
    ("M", "s", 1e6),
    ("µ", "A", 1e-6),
    ("n", "mol", 1e-9),
])
def test_valid_prefix_synthesis(pref, base, factor, reg):
    base_u = reg.get(base)
    sym = f"{pref}{base}"
    u = reg.get(sym)
    assert u.name == sym
    assert u.dim == base_u.dim
    assert u.scale_to_si == pytest.approx(base_u.scale_to_si * factor)


def test_synthesis_is_lazy_and_idempotent(reg):
    base_len = len(reg.all())
    km1 = reg.get("km")
    after_first = len(reg.all())
    km2 = reg.get("km")
    after_second = len(reg.all())

    assert after_first == base_len + 1  # created once
    assert after_second == after_first  # no duplicates
    assert km1 is km2


def test_anti_stacking_prefixed_base_rejected(reg):
    # 'mm' is milli + m (valid and synthesized)
    mm = reg.get("mm")
    assert mm.name == "mm"

    # 'kmm' would be kilo + (mm) which is stacking → must fail
    with pytest.raises(ValueError):
        reg.get("kmm")

    # Another explicit stacking case: 'kµm'
    with pytest.raises(ValueError):
        reg.get("kµm")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_symbol_raises(reg):
    with pytest.raises(ValueError):
        reg.get("nope")


# ---------------------------------------------------------------------------
# Thread-safety: concurrent synthesis of the same symbol
# ---------------------------------------------------------------------------

def test_thread_safe_prefixed_creation(reg):
    created = []
    errs = []

    def worker():
        try:
            created.append(reg.get("km"))
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errs
    # All returned the exact same Unit object
    first = created[0]
    assert all(u is first for u in created)
    # Registry has exactly one 'km'
    assert "km" in reg.all() and sum(1 for k in reg.all().keys() if k == "km") == 1


# ---------------------------------------------------------------------------
# Convenience functions should use DEFAULT_REGISTRY
# ---------------------------------------------------------------------------

def test_convenience_functions_use_default_registry(patched_default):
    # register a custom unit through the free function and fetch it back
    regmod.register_unit(Unit("ft", 0.3048, L))
    out = regmod.get_unit("ft")
    assert out.name == "ft"
    assert out.scale_to_si == pytest.approx(0.3048)

    # alias convenience function
    regmod.register_alias("foot", "ft")
    assert regmod.get_unit("foot") is out


# ---------------------------------------------------------------------------
# Dimension sanity checks via relationships
# ---------------------------------------------------------------------------

def test_dimension_relationships(reg):
    # N = kg·m/s², J = N·m, W = J/s, V = W/A, Ω = V/A, S = A/V
    kg = reg.get("kg").dim
    m = reg.get("m").dim
    s = reg.get("s").dim
    A = reg.get("A").dim

    N_dim = dim_mul(kg, dim_div(m, dim_pow(s, 2)))
    assert reg.get("N").dim == N_dim

    J_dim = dim_mul(N_dim, m)
    assert reg.get("J").dim == J_dim

    W_dim = dim_div(J_dim, s)
    assert reg.get("W").dim == W_dim

    V_dim = dim_div(W_dim, A)
    assert reg.get("V").dim == V_dim

    ohm_dim = dim_div(V_dim, A)
    assert reg.get("Ω").dim == ohm_dim

    S_dim = dim_div(A, V_dim)
    assert reg.get("S").dim == S_dim
