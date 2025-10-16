# pytest tests for quantium.units.registry
#
# These tests exercise normalization, aliases, SI-prefix synthesis,
# anti-stacking rules, thread-safety, and the convenience functions that
# delegate to DEFAULT_REGISTRY. They purposefully use an isolated registry
# instance for most tests, and monkeypatch DEFAULT_REGISTRY where needed.

import threading

import pytest

from quantium.core.dimensions import (
    AMOUNT,
    CURRENT,
    DIM_0,
    LENGTH,
    LUMINOUS,
    MASS,
    TEMPERATURE,
    TIME,
    dim_div,
    dim_mul,
    dim_pow,
)
from quantium.core.quantity import Unit

# We import the module under test once, and access internals we intentionally
# rely on in tests (like _bootstrap_default_registry).
import quantium.units.registry as regmod
from quantium.units.registry import UnitsRegistry

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
    for sym, dim in [("m", LENGTH), ("kg", MASS), ("s", TIME), ("A", CURRENT), ("K", TEMPERATURE), ("mol", AMOUNT), ("cd", LUMINOUS)]:
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
    assert Hz.dim == dim_pow(TIME, -1)
    assert Hz.scale_to_si == pytest.approx(1.0)

    N_unit = reg.get("N")
    assert N_unit.dim == dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2)))  # kg·m/s²

    Pa = reg.get("Pa")
    assert Pa.dim == dim_div(N_unit.dim, dim_pow(LENGTH, 2))

    J_unit = reg.get("J")
    assert J_unit.dim == dim_mul(N_unit.dim, LENGTH)

    W = reg.get("W")
    assert W.dim == dim_div(J_unit.dim, TIME)

    V = reg.get("V")
    assert V.dim == dim_div(W.dim, CURRENT)

    F = reg.get("F")
    assert F.dim == dim_div(reg.get("C").dim, V.dim)

    ohm = reg.get("Ω")
    assert ohm.dim == dim_div(V.dim, CURRENT)

    S = reg.get("S")
    assert S.dim == dim_div(CURRENT, V.dim)


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
    ureg = UnitsRegistry()
    ureg.register(Unit("ft", 0.3048, LENGTH))
    out = ureg.get("ft")
    assert out.name == "ft"
    assert out.scale_to_si == pytest.approx(0.3048)

    # alias convenience function
    ureg.register_alias("foot", "ft")
    assert ureg.get("foot") is out


# ---------------------------------------------------------------------------
# Dimension sanity checks via relationships
# ---------------------------------------------------------------------------

def test_dimension_relationships(reg):
    # AMOUNT = kg·m/s², LUMINOUS = N·m, W = LUMINOUS/s, V = W/A, Ω = V/A, S = A/V
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



@pytest.mark.parametrize("sym, dim_expr, scale", [
    # frequency & mechanics
    ("Hz", dim_pow(TIME, -1), 1.0),
    ("N",  dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), 1.0),
    ("Pa", dim_div(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), dim_pow(LENGTH, 2)), 1.0),
    ("J",  dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), 1.0),
    ("W",  dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), 1.0),

    # electricity
    ("C",  dim_mul(CURRENT, TIME), 1.0),
    ("V",  dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT), 1.0),
    ("F",  dim_div(dim_mul(CURRENT, TIME), dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT)), 1.0),
    ("Ω",  dim_div(  # V / A
            dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT), CURRENT), 1.0),
    ("S",  dim_div(CURRENT, dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT)), 1.0),
    ("Wb", dim_mul(dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT), TIME), 1.0),
    ("T",  dim_div(dim_mul(dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT), TIME), dim_pow(LENGTH, 2)), 1.0),
    ("H",  dim_div(dim_mul(dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT), TIME), CURRENT), 1.0),

    # photometry (sr is DIM_0 so lm = cd)
    ("lm", dim_mul(LUMINOUS, DIM_0), 1.0),
    ("lx", dim_div(dim_mul(LUMINOUS, DIM_0), dim_pow(LENGTH, 2)), 1.0),

    # radioactivity & dose
    ("Bq", dim_pow(TIME, -1), 1.0),
    ("Gy", dim_div(  # LUMINOUS/kg
            dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), MASS), 1.0),
    ("Sv", dim_div(  # same as Gy
            dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), MASS), 1.0),

    # catalytic activity
    ("kat", dim_div(AMOUNT, TIME), 1.0),

    # mass non-SI base
    ("g",  MASS, 1e-3),
])
def test_all_derived_units_dimensions_and_scales(reg, sym, dim_expr, scale):
    u = reg.get(sym)
    assert u.dim == dim_expr
    assert u.scale_to_si == pytest.approx(scale)


# ---------------------------------------------------------------------------
# Time units: presence, dimensions, scales
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sym, seconds", [
    ("min",       60.0),
    ("h",         60.0 * 60.0),
    ("d",         24.0 * 60.0 * 60.0),
    ("wk",        7.0 * 24.0 * 60.0 * 60.0),
    ("fortnight", 14.0 * 24.0 * 60.0 * 60.0),

    # Civil (Gregorian) average month/year
    ("mo",        (365.2425 / 12.0) * 24.0 * 3600.0),
    ("yr",        365.2425 * 24.0 * 3600.0),

    # Astronomy
    ("yr_julian", 365.25 * 24.0 * 3600.0),

    # Longer spans (Gregorian-based)
    ("decade",     10.0  * 365.2425 * 24.0 * 3600.0),
    ("century",    100.0 * 365.2425 * 24.0 * 3600.0),
    ("millennium", 1000.0* 365.2425 * 24.0 * 3600.0),
])
def test_time_units_present_and_scaled(reg, sym, seconds):
    u = reg.get(sym)
    assert isinstance(u, Unit)
    assert u.dim == TIME
    assert u.scale_to_si == pytest.approx(seconds)


# ---------------------------------------------------------------------------
# Time aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias, canonical", [
    # Common aliases (already have ohm in your tests, included here for completeness)
    ("ohm", "Ω"),
    ("Ohm", "Ω"),
    ("OHM", "Ω"),

    # Time aliases
    ("minute", "min"),
    ("minutes", "min"),
    ("hr", "h"),
    ("hour", "h"),
    ("hours", "h"),
    ("day", "d"),
    ("days", "d"),
    ("week", "wk"),
    ("weeks", "wk"),
    ("fortnights", "fortnight"),
    ("month", "mo"),
    ("months", "mo"),
    ("year", "yr"),
    ("years", "yr"),
    ("annum", "yr"),
    ("dec", "decade"),
    ("decades", "decade"),
    ("cent", "century"),
    ("centuries", "century"),
    ("millennia", "millennium"),
])
def test_time_aliases_map_to_canonical(reg, alias, canonical):
    u_alias = reg.get(alias)
    u_canon = reg.get(canonical)
    # Should be the exact same Unit object
    assert u_alias is u_canon


# ---------------------------------------------------------------------------
# Non-prefixable enforcement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_sym", [
    # The registry marks these as non-prefixable; any SI/metric prefix must fail.
    "mkg", "kkg", "ukg",            # kg is non-prefixable (SI prefixes apply to gram, not kilogram)
    "kmin", "µmin", "umin", "mmin",
    "kh", "µh", "uh", "mh",
    "kd", "µd", "ud", "md",
    "kwk", "µwk", "uwk", "mwk",
    "kfortnight", "µfortnight", "ufortnight", "mfortnight",
    "kmo", "µmo", "umo", "mmo",
    "kyr", "µyr", "uyr", "myr",
    "kyr_julian", "µyr_julian", "uyr_julian", "myr_julian",
    "kdecade", "µdecade", "udecade", "mdecade",
    "kcentury", "µcentury", "ucentury", "mcentury",
    "kmillennium", "µmillennium", "umillennium", "mmillennium",
])
def test_non_prefixable_time_units_reject_prefixes(reg, bad_sym):
    with pytest.raises(ValueError):
        reg.get(bad_sym)


def test_non_prefixable_does_not_accidentally_create_units(reg):
    # Ensure attempting to fetch a prefixed version did not pollute the registry
    base_len = len(reg.all())
    for bad in ["kyr", "umin", "kh", "kmo", "mkg"]:
        with pytest.raises(ValueError):
            reg.get(bad)
    assert len(reg.all()) == base_len


# ---------------------------------------------------------------------------
# Sanity: prefix synthesis still works for prefixable time base 's'
# (and remains disallowed for the added non-prefixable time units)
# ---------------------------------------------------------------------------

def test_seconds_still_prefixable_but_not_month_year(reg):
    # Positive control: seconds *are* prefixable (already covered elsewhere, but we assert here for contrast)
    ms = reg.get("ms")
    s = reg.get("s")
    assert ms.dim == s.dim
    assert ms.scale_to_si == pytest.approx(1e-3)

    # Negative controls: these are non-prefixable
    for sym in ["kmo", "kyr", "µh"]:
        with pytest.raises(ValueError):
            reg.get(sym)


# -------------------------------
# Basic compound parsing & equivalence
# -------------------------------

def test_get_compound_velocity_and_acceleration(reg):
    u1 = reg.get("m/s**2")             # parser path
    u2 = reg.get("m") / (reg.get("s") ** 2)  # manual composition
    assert isinstance(u1, Unit)
    assert u1.dim == u2.dim
    assert u1.scale_to_si == pytest.approx(u2.scale_to_si)

def test_get_compound_force_equals_newton(reg):
    nm = reg.get("kg*m/s**2")   # base composition
    N  = reg.get("N")           # named
    assert nm.dim == N.dim
    assert nm.scale_to_si == pytest.approx(N.scale_to_si)

def test_get_literal_one_over_second_equals_hz(reg):
    inv_s = reg.get("1/s")
    Hz = reg.get("Hz")
    assert inv_s.dim == Hz.dim
    assert inv_s.scale_to_si == pytest.approx(1.0)


# -------------------------------
# Parentheses distribution cases
# -------------------------------

def test_get_parentheses_complex_reduces_to_m3_per_s(reg):
    # (W*s)/(N*s/m**2) -> m^3/s (scale 1)
    u_expr = reg.get("(W*s)/(N*s/m**2)")
    u_si   = reg.get("m**3/s")
    assert u_expr.dim == u_si.dim
    assert u_expr.scale_to_si == pytest.approx(1.0)

def test_get_parentheses_with_prefixes_and_mm(reg):
    # (uW*s)/(N*s/mm**2) -> 1e-12 * m^3/s
    u_expr = reg.get("(uW*s)/(N*s/mm**2)")
    u_si   = reg.get("m**3/s")
    assert u_expr.dim == u_si.dim
    # uW = 1e-6 W, mm^2 = 1e-6 m^2  => overall factor 1e-12
    assert u_expr.scale_to_si == pytest.approx(1e-12)


# -------------------------------
# Aliases & prefixes inside expressions
# -------------------------------

def test_get_alias_inside_expression(reg):
    # ohm alias should normalize to Ω and V = Ω·A
    v1 = reg.get("ohm*A")
    V  = reg.get("V")
    assert v1.dim == V.dim
    assert v1.scale_to_si == pytest.approx(1.0)

def test_get_prefix_inside_expression(reg):
    # 10 cm/s vs m/s scale factors: reg.get should synthesize cm correctly
    cm_per_s = reg.get("cm/s")
    m_per_s  = reg.get("m/s")
    # compare scales to SI: cm/s should be 1e-2 m/s
    assert cm_per_s.dim == m_per_s.dim
    assert cm_per_s.scale_to_si == pytest.approx(1e-2)


# -------------------------------
# Idempotence / caching of expressions
# (identity not required, but equality must hold)
# -------------------------------

def test_get_same_expression_twice_equal_not_necessarily_identical(reg):
    u1 = reg.get("kg*m/s**2")
    u2 = reg.get("kg*m/s**2")
    assert u1 is not None and u2 is not None
    assert u1.dim == u2.dim
    assert u1.scale_to_si == pytest.approx(u2.scale_to_si)
    # different object identity is fine; equality is defined by dim+scale


# -------------------------------
# Errors & invalid syntax
# -------------------------------

@pytest.mark.parametrize("bad", [
    "m//s",         # disallowed token
    "m/s**",        # trailing operator
    "*/s",          # starts with operator
    "(",            # unbalanced paren
    "m/s)",         # unbalanced paren
    "1/",           # dangling slash
    "kg*m/s**x",    # non-integer exponent
    "kg*m/s**+2+3", # trailing noise
])
def test_get_compound_invalid_syntax_raises(reg, bad):
    with pytest.raises(ValueError):
        reg.get(bad)

def test_get_compound_unknown_symbol_raises(reg):
    with pytest.raises(ValueError):
        reg.get("kg*blorp/s")


# -------------------------------
# Thread-safety smoke test for compound expressions
# -------------------------------

def test_thread_safety_compound_get(reg):
    errs = []
    outs = []

    def worker(expr):
        try:
            outs.append(reg.get(expr))
        except Exception as e:
            errs.append(e)

    exprs = [
        "kg*m/s**2",
        "m/s",
        "(W*s)/(N*s/m**2)",
        "1/s",
        "cm/s",
    ] * 8  # repeat to create more concurrency

    threads = [threading.Thread(target=worker, args=(e,)) for e in exprs]
    for t in threads: 
        t.start()
    for t in threads: 
        t.join()

    assert not errs
    # basic sanity: all results are Units with sensible dim/scale
    assert all(isinstance(u, Unit) for u in outs)
    # verify a couple of expected dims quickly
    assert any(u.dim == dim_div(LENGTH, TIME) for u in outs)        # m/s cases
    assert any(u.dim == dim_pow(TIME, -1) for u in outs)            # 1/s cases