from dataclasses import FrozenInstanceError
import math

import pytest

from quantium.core.dimensions import DIM_0, LENGTH, TEMPERATURE, TIME, dim_div, dim_mul, dim_pow
from quantium.core.quantity import Quantity, Unit
from quantium.units.registry import DEFAULT_REGISTRY as ureg

# -------------------------------
# Unit: construction & validation
# -------------------------------

def test_unit_valid():
    m = Unit("m", 1.0, LENGTH)
    assert m.name == "m"
    assert m.scale_to_si == 1.0
    assert m.dim == LENGTH

def test_unit_invalid_dim_length():
    with pytest.raises(ValueError):
        Unit("bad", 1.0, (1, 0, 0))  # not 7-tuple

@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_unit_invalid_scale(scale):
    with pytest.raises(ValueError):
        Unit("x", scale, LENGTH)

def test_unit_is_frozen_and_slotted():
    m = Unit("m", 1.0, LENGTH)

    # frozen => normal assignment raises FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        m.name = "meter"

    # slots => adding a new attribute should fail (AttributeError or TypeError depending on Python)
    with pytest.raises((AttributeError, TypeError)):
        m.some_new_attr = 42


# -------------------------------
# Quantity: basics & conversion
# -------------------------------

def test_quantity_construct_and_to():
    m  = Unit("m", 1.0, LENGTH)
    cm = Unit("cm", 0.01, LENGTH)

    q_cm = Quantity(200, cm)          # 200 cm
    q_m  = q_cm.to(m)                  # -> 2 m

    assert isinstance(q_m, Quantity)
    assert q_m.unit is m
    assert q_m.dim == LENGTH
    # _mag_si is internal, so check using units:
    assert math.isclose(q_m._mag_si, 2.0)  # 2 m in SI
    # magnitude shown in the *current* unit:
    assert math.isclose(q_m._mag_si / q_m.unit.scale_to_si, 2.0)

def test_quantity_to_dimension_mismatch_raises():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TEMPERATURE)
    q = Quantity(3, m)
    with pytest.raises(TypeError):
        q.to(s)


# -------------------------------
# __rmatmul__: value @ Unit
# -------------------------------

def test_rmatmul_operator():
    m = Unit("m", 1.0, LENGTH)
    q = 3 @ m
    assert isinstance(q, Quantity)
    assert q.dim == LENGTH
    assert q.unit is m
    assert math.isclose(q._mag_si, 3.0)


# -------------------------------
# Arithmetic: +, -, *, /, **, scalars
# -------------------------------

def test_add_and_sub_same_dim():
    m = Unit("m", 1.0, LENGTH)
    cm = Unit("cm", 0.01, LENGTH)
    q1 = 1 @ m
    q2 = 50 @ cm  # 0.5 m

    s = q1 + q2   # left unit ("m") retained
    d = q1 - q2

    assert s.unit is m and d.unit is m
    assert math.isclose(s._mag_si / s.unit.scale_to_si, 1.5)
    assert math.isclose(d._mag_si / d.unit.scale_to_si, 0.5)

def test_add_dim_mismatch_raises():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TEMPERATURE)
    with pytest.raises(TypeError):
        _ = (1 @ m) + (1 @ s)

def test_scalar_multiplication_and_division():
    m = Unit("m", 1.0, LENGTH)
    q = 2 @ m

    q2 = q * 3
    q3 = 3 * q
    q4 = q / 2

    assert q2.dim == LENGTH and q3.dim == LENGTH and q4.dim == LENGTH
    assert math.isclose(q2._mag_si / q2.unit.scale_to_si, 6.0)
    assert math.isclose(q3._mag_si / q3.unit.scale_to_si, 6.0)
    assert math.isclose(q4._mag_si / q4.unit.scale_to_si, 1.0)

def test_quantity_times_quantity():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TEMPERATURE)
    q = (2 @ m) * (3 @ s)  # -> 6 m·s

    assert q.dim == dim_mul(LENGTH, TEMPERATURE)
    assert q.unit.name == "m·s"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 6.0)

def test_quantity_div_quantity():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TEMPERATURE)
    q = (10 @ m) / (2 @ s)  # -> 5 m/s

    assert q.dim == dim_div(LENGTH, TEMPERATURE)
    assert q.unit.name == "m/s"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 5.0)

def test_scalar_divided_by_quantity():
    m = Unit("m", 1.0, LENGTH)
    q = 2 / (2 @ m)  # -> 1 (1/m)

    assert q.dim == dim_div(DIM_0, LENGTH)
    assert q.unit.name == "1/m"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 1.0)

def test_power_of_quantity():
    m = Unit("m", 1.0, LENGTH)
    q2 = (2 @ m) ** 2  # -> 4 m^2

    assert q2.dim == dim_pow(LENGTH, 2)
    assert q2.unit.name == "m^2"
    assert math.isclose(q2._mag_si / q2.unit.scale_to_si, 4.0)


# -------------------------------
# to_si(): preferred symbol & fallback
# -------------------------------

def test_to_si_uses_preferred_symbol_when_available(monkeypatch):
    # Arrange: make preferred_symbol_for_dim return a symbol for LENGTH
    import quantium.core.utils as utils
    # Monkeypatch utils functions that to_si imports locally
    def fake_preferred(dim):
        return "m" if dim == LENGTH else None
    def fake_format(dim):
        return "LENGTH?"  # should not be used in this test

    monkeypatch.setattr(utils, "preferred_symbol_for_dim", fake_preferred, raising=True)
    monkeypatch.setattr(utils, "format_dim", fake_format, raising=True)

    cm = Unit("cm", 0.01, LENGTH)
    q_si = (123 @ cm).to_si()

    assert isinstance(q_si, Quantity)
    assert q_si.unit.name == "m"         # preferred symbol chosen
    assert q_si.unit.scale_to_si == 1.0  # SI unit
    # magnitudes in SI should match _mag_si:
    assert math.isclose(q_si._mag_si, 1.23)

def test_to_si_fallbacks_to_formatted_dim_when_no_symbol(monkeypatch):
    import quantium.core.utils as utils
    def fake_preferred(dim):
        return None  # force fallback
    def fake_format(dim):
        # For LENGTH/TEMPERATURE, produce a composed name:
        return "m/s" if dim == dim_div(LENGTH, TEMPERATURE) else "1"

    monkeypatch.setattr(utils, "preferred_symbol_for_dim", fake_preferred, raising=True)
    monkeypatch.setattr(utils, "format_dim", fake_format, raising=True)

    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TEMPERATURE)
    q = ((5 @ m) / (2 @ s)).to_si()

    assert q.unit.name == "m/s"          # composed SI name from format_dim
    assert q.unit.scale_to_si == 1.0
    assert math.isclose(q._mag_si, 2.5)


# -------------------------------
# __repr__: pretty printing
# -------------------------------

def test_repr_keeps_non_si_unit_name(monkeypatch):
    # Ensure repr uses the current unit name ("cm") and does not replace with a symbol
    import quantium.core.utils as utils

    # Make prettifier a no-op passthrough so we can assert on exact string.
    monkeypatch.setattr(utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True)
    # Ensure it would *try* to upgrade only when scale_to_si == 1.0; here it's 0.01, so no upgrade.
    monkeypatch.setattr(utils, "preferred_symbol_for_dim", lambda d: "m", raising=True)

    cm = Unit("cm", 0.01, LENGTH)
    q = 2 @ cm
    assert repr(q) == "2 cm"

def test_repr_upgrades_to_preferred_symbol_when_scale_is_1(monkeypatch):
    from quantium import core as _core
    # prettifier just returns what it's given
    monkeypatch.setattr(
        _core.utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True
    )
    # preferred symbol for LENGTH is "m"
    monkeypatch.setattr(_core.utils, "preferred_symbol_for_dim", lambda d: "m" if d == LENGTH else None, raising=True)

    m = Unit("m", 1.0, LENGTH)
    q = 3 @ m
    # scale_to_si == 1.0 -> allowed to upgrade pretty name to "m"
    assert repr(q) == "3 m"


# -------------------------------
# Reciprocal test for units
# -------------------------------
@pytest.mark.regression(reason="Issue #19: Unit division with 1 error + edge cases")
def test_unit_reciprocal():
    s = ureg.get("s")

    # --- 1) No exceptions on valid reciprocal forms ---
    for expr in [lambda: 1/s, lambda: s**-1, lambda: 1/s**3, lambda: 1/s**-1]:
        try:
            expr()
        except Exception as e:
            pytest.fail(f"Reciprocal operation raised: {e}")

    # --- 2) Equivalence between reciprocal and power forms ---
    # Operator precedence note: ** binds tighter than /, so 1/s**-3 == 1/(s**-3)
    assert (1/s) == (s ** -1)
    assert (1/s**3) == (s ** -3)
    assert (1/s**-3) == (s ** 3)

    # --- 3) Idempotence / normalization of reciprocals ---
    # 1/(1/s) == s
    assert 1 / (1 / s) == s
    # Double-negative power via reciprocal should normalize:
    assert 1 / (s ** -1) == s
    # Reciprocal of a reciprocal stays stable (two flips -> original)
    assert 1 / (1 / (1 / s)) == (1 / s)

    # --- 4) Mixed powers with reciprocals ---
    # 1/(s^k) == s^-k for positive k
    assert 1 / (s ** 4) == (s ** -4)
    # 1/(s^-k) == s^k
    assert 1 / (s ** -5) == (s ** 5)

    # --- 5) Name normalization sanity (if names matter in equality) ---
    # Expect canonical power-style names (after your recent normalization)
    # These asserts fail only if names diverge while dim/scale match.
    assert (1 / s).name == (s ** -1).name
    assert (1 / (s ** 3)).name == (s ** -3).name
    assert (1 / (s ** -3)).name == (s ** 3).name

    # --- 6) Zero and one powers (corner cases) ---
    s0 = s ** 0               # dimensionless
    s1 = s ** 1
    # (1 / s^0) should flip exponent sign: ^0 -> ^-0 (still 0); stays dimensionless
    r0 = 1 / s0
    assert r0.dim == s0.dim
    assert r0.scale_to_si == pytest.approx(1.0)
    # 1/s^1 == s^-1
    assert 1 / s1 == (s ** -1)

    # --- 7) Chained expressions don't drift scale/dim ---
    # (1/s) * s -> dimensionless
    one = (1 / s) * s
    assert one.name  # any name; just ensure object exists
    assert one.dim == DIM_0  # or however your dimensionless dim is represented

@pytest.mark.regression(reason="Issue #19: Invalid numerators must raise TypeError")
@pytest.mark.parametrize("n", [0, 2, 3, -1, 3.14])
def test_unit_reciprocal_invalid_numerators_raise(n):
    s = ureg.get("s")
    with pytest.raises(TypeError):
        _ = n / s

@pytest.mark.regression(reason="Issue #19: Parentheses / precedence and deep nesting")
def test_unit_reciprocal_parentheses_and_nesting():
    s = ureg.get("s")

    # Precedence: ** before /
    assert (1 / s**-1) == (1 / (s**-1)) == (s ** 1)

    # Deep nesting: 1/(1/(1/s)) == 1/s
    u = 1 / (1 / (1 / s))
    assert u == (1 / s)

    # Compound powers: ((1/s)**3) * (s**3) -> dimensionless
    left = (1 / s) ** 3
    right = s ** 3
    prod = left * right
    assert prod.dim == DIM_0


@pytest.mark.regression(reason="Issue #3: Verify all units produce dimensionless result when divided by themselves")
def test_all_units_self_division_is_dimensionless_and_nameless():
    """
    For every registered unit:
      - Create a Quantity with magnitude 1 and that unit
      - Divide it by itself
      - Ensure the resulting dimension == DIM_0 (dimensionless)
      - Ensure the resulting unit name is "" (empty string)
      - Ensure __repr__ returns only the numeric value (no unit symbol)
    """
    for name, unit in ureg.all().items():
        q = Quantity(1.0, unit)
        result = q / q

        # Check dimensionless
        assert result.dim == DIM_0, f"{name}: expected DIM_0, got {result.dim}"

        # Check unit name strictly empty string
        assert result.unit.name == "", f"{name}: expected unit name '', got '{result.unit.name}'"

        # Check __repr__ output: should be only a number (no space or unit)
        rep = repr(result)
        assert rep.strip().replace('.', '', 1).isdigit(), (
            f"{name}: expected numeric-only repr, got '{rep}'"
        )

@pytest.mark.regression(reason="Dimensionless results must be SI-normalized and numerically correct")
@pytest.mark.parametrize("sym_a, val_a, sym_b, val_b", [
    # base + prefixes
    ("m",   3.2,  "cm",  5.0),
    ("s",   7.0,  "ms",  2.0),
    ("g",   1.0,  "kg",  0.5),
    # derived units with prefixes
    ("N",   10.0, "uN",  5.0),
    ("Pa",  2.0,  "kPa", 3.0),
    ("J",   8.0,  "MJ",  2.0),
    ("Hz",  5.0,  "kHz", 1.0),
])
def test_dimensionless_result_is_si_and_correct(sym_a, val_a, sym_b, val_b):
    ua = ureg.get(sym_a)
    ub = ureg.get(sym_b)

    qa = val_a @ ua
    qb = val_b @ ub

    r = qa / qb

    # Expected pure number computed via SI magnitudes
    expected = (val_a * ua.scale_to_si) / (val_b * ub.scale_to_si)

    # 1) Dimensionless
    assert r.dim == DIM_0

    # 2) In SI (canonical dimensionless): empty name and scale 1.0
    assert r.unit.name == ""
    assert r.unit.scale_to_si == 1.0

    # 3) Numeric correctness
    # r is already in SI since r.unit.scale_to_si == 1.0
    assert float(repr(r)) == pytest.approx(expected)


# -------------------------------
# Unit equality (regressions)
# -------------------------------

@pytest.mark.regression(reason="Issue #24: Units with identical dim & scale must compare equal regardless of name")
def test_unit_equality_ignores_name_and_matches_scale_and_dim():
    # Construct Newton from base units and compare with predefined "N"
    kg = ureg.get("kg")
    m  = ureg.get("m")
    s  = ureg.get("s")
    N1 = kg * m / (s ** 2)   # name "kg·m/s^2", scale 1.0, dim (1,1,-2,0,0,0,0)
    N2 = ureg.get("N")       # name "N",         scale 1.0, dim (1,1,-2,0,0,0,0)

    assert N1 == N2
    # Sanity: names may differ but equality should be based on dim+scale only
    assert N1.name != N2.name
    assert N1.scale_to_si == N2.scale_to_si
    assert N1.dim == N2.dim


@pytest.mark.regression(reason="Units with different scales are not equal even if dimensions match")
def test_unit_inequality_different_scale_same_dim():
    m  = Unit("m", 1.0, LENGTH)
    cm = Unit("cm", 0.01, LENGTH)
    assert m != cm


@pytest.mark.regression(reason="Units with different dimensions are not equal even if scales match")
def test_unit_inequality_different_dim_same_scale():
    # scale 1.0 but different dimensions (LENGTH vs TIME)
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TIME)
    assert m != s


@pytest.mark.regression(reason="__eq__ should return NotImplemented for incompatible types")
def test_unit_equality_with_incompatible_type_returns_notimplemented():
    m = Unit("m", 1.0, LENGTH)
    # Direct dunder call to observe NotImplemented (== would coerce to False)
    assert Unit.__eq__(m, 42) is NotImplemented
    assert (m == 42) is False


# -------------------------------
# Quantity equality (regressions)
# -------------------------------

@pytest.mark.regression(reason="Quantities equal when SI magnitude and unit (by dim+scale) match")
def test_quantity_equality_same_si_and_equivalent_units():
    # Build equivalent units with different names but same dim/scale
    kg = ureg.get("kg")
    m = ureg.get("m")
    s = ureg.get("s")
    N = ureg.get("N")
    unit_from_bases = kg * m / (s ** 2)   # equals N by dim+scale

    q1 = 10 @ N
    q2 = 10 @ unit_from_bases

    # _mag_si identical & units compare equal (ignoring name)
    assert q1 == q2


@pytest.mark.regression(reason="Issue: #28 Quantities with same SI magnitude but different unit scales should be equal")
def test_quantity_equality_same_si_magnitude_different_units():
    # 100 cm and 1 m have same SI magnitude (both represent 1.0 m in SI),
    # and __eq__ now compares normalized SI magnitudes and dimensions, not unit identity.
    m  = Unit("m", 1.0, LENGTH)
    cm = Unit("cm", 0.01, LENGTH)

    q_cm = 100 @ cm  # _mag_si = 1.0
    q_m  = 1 @ m     # _mag_si = 1.0

    # They should now compare equal because their physical values are identical.
    assert q_cm == q_m


@pytest.mark.regression(reason="Quantities with different SI magnitudes must not be equal even if units match")
def test_quantity_inequality_different_si_magnitude_same_unit():
    m = Unit("m", 1.0, LENGTH)
    q1 = 2 @ m
    q2 = 3 @ m
    assert q1 != q2


@pytest.mark.regression(reason="Quantity __eq__ returns NotImplemented for incompatible types")
def test_quantity_equality_with_incompatible_type_returns_notimplemented():
    m = Unit("m", 1.0, LENGTH)
    q = 1 @ m
    assert Quantity.__eq__(q, "not-a-quantity") is NotImplemented
    assert (q == "not-a-quantity") is False


# -------------------------------
# Mixed constructions (extra safety)
# -------------------------------

@pytest.mark.regression(reason="Derived unit equality is stable across different build paths")
def test_unit_equality_across_multiple_construction_paths():
    # (kg·m)/s^2 == (kg/s^2)·m == kg·(m/s^2)
    kg = ureg.get("kg")
    m = ureg.get("m")
    s = ureg.get("s")
    u1 = kg * m / (s ** 2)
    u2 = (kg / (s ** 2)) * m
    u3 = kg * (m / (s ** 2))
    assert u1 == u2 == u3


@pytest.mark.regression(reason="Quantity equality consistent when units simplify to same dim/scale")
def test_quantity_equality_when_units_simplify_to_same_dim_and_scale():
    # Build two different-looking but equivalent units for velocity
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TIME)
    ms = m / s

    # Another velocity path: (m*s)/s^2 simplifies to m/s
    alt = (m * s) / (s ** 2)
    assert ms == alt  # unit equality check

    q1 = 12 @ ms
    q2 = 12 @ alt
    assert q1 == q2


@pytest.mark.regression(reason="Name differences never drive equality; only scale & dim")
def test_unit_name_changes_do_not_affect_equality():
    m = Unit("m", 1.0, LENGTH)
    m_alias = m.as_name("meter")
    assert m_alias.name != m.name
    assert m_alias == m


@pytest.mark.regression(reason="Issue: #33 Unit raised to power 0 is not dimensionless")
def test_quantity_pow_zero_and_one_and_negative():
    m = Unit("m", 1.0, LENGTH)
    q = 2 @ m
    q0 = q ** 0
    q1 = q ** 1
    qn = q ** -2
    assert q0.dim == DIM_0 and q0.unit.scale_to_si == 1.0
    assert q1.dim == LENGTH
    assert qn.dim == dim_pow(LENGTH, -2)



def _name(sym: str, n: int) -> str:
    return "1" if n == 0 else (sym if n == 1 else f"{sym}^{n}")

# -------------------------
# Regression: Issue #33 (Unit)
# -------------------------

@pytest.mark.regression(reason="Issue #33: negative exponents should produce correct dim/name/scale")
@pytest.mark.parametrize("n", [-3, -4])
@pytest.mark.parametrize("sym, scale", [("m", 1.0), ("cm", 0.01)])
def test_unit_pow_negative_high_exponents_regression_issue_33(sym: str, scale: float, n: int):
    u = Unit(sym, scale, LENGTH)
    up = u ** n

    # Dimension and name
    assert up.dim == dim_pow(LENGTH, n)
    assert up.name == _name(sym, n)

    # Scale: scale_to_si ** n (e.g., (0.01)**-3 = 1_000_000)
    assert up.scale_to_si == pytest.approx(scale ** n)

    # Reciprocal sanity
    assert (1 / (u ** (-n))) == (u ** n)


# -------------------------
# Regression: Issue #33 (Quantity)
# -------------------------

@pytest.mark.regression(reason="Issue #33: Quantity ** n must match dim, unit, and magnitude for negative exponents")
@pytest.mark.parametrize("n", [-3, -4])
@pytest.mark.parametrize("sym, scale, value", [
    ("m",  1.0,  2.0),
    ("cm", 0.01, 5.0),
])
def test_quantity_pow_negative_high_exponents_regression_issue_33(sym: str, scale: float, value: float, n: int):
    u = Unit(sym, scale, LENGTH)
    q = value @ u
    qp = q ** n

    # Dimension & unit name/scale
    assert qp.dim == dim_pow(LENGTH, n)
    assert qp.unit.name == _name(sym, n)
    assert qp.unit.scale_to_si == pytest.approx(scale ** n)

    # Magnitude in the resulting unit should be value**n
    mag_in_unit = qp._mag_si / qp.unit.scale_to_si
    assert math.isclose(mag_in_unit, value ** n, rel_tol=1e-12, abs_tol=1e-12)



# NOTE:
# These tests rely on the classic binary rounding facts:
#   0.1 * 0.2 != 0.02 exactly
#   3 * 0.1 != 0.3 exactly
# so exact float equality will fail. The library should tolerate tiny drift.


@pytest.mark.regression(reason="Float drift: Unit equality should tolerate tiny scale differences")
def test_unit_equality_tolerates_scale_float_drift():
    # Two mathematically identical scales obtained via different FP paths.
    u1 = Unit("a", 0.1, LENGTH) * Unit("b", 0.2, LENGTH)     # scale ~ 0.020000000000000004
    u2 = Unit("c", 0.02, dim_pow(LENGTH, 2))                  # scale 0.02

    # If Unit.__eq__ uses exact float equality, this will fail.
    # After fixing, __eq__ should consider them equal (same dim, scales within tolerance).
    assert u1 == u2, f"scales differ slightly: {u1.scale_to_si} vs {u2.scale_to_si}"


@pytest.mark.regression(reason="Float drift: Quantity equality should use tolerant SI magnitude comparison")
def test_quantity_equality_tolerates_si_float_drift():
    m = Unit("m", 1.0, LENGTH)
    a = Unit("a", 0.1, LENGTH)

    q1 = 3 @ a      # _mag_si ~ 0.30000000000000004
    q2 = 0.3 @ m    # _mag_si ~ 0.29999999999999999

    # Exact equality on _mag_si will fail. After fix, should pass.
    assert q1 == q2, f"_mag_si differ slightly: {q1._mag_si} vs {q2._mag_si}"


@pytest.mark.regression(reason="Float drift: Dimensionless ratios should be numerically 1 within tolerance")
def test_dimensionless_ratio_avoids_float_drift():
    m = Unit("m", 1.0, LENGTH)
    a = Unit("a", 0.1, LENGTH)

    num = 3 @ a       # SI ~ 0.30000000000000004
    den = 0.3 @ m     # SI ~ 0.29999999999999999
    r = num / den     # should be dimensionless 1

    # 1) Dimensionless dim
    assert r.dim == DIM_0

    # 2) Canonical dimensionless unit (empty name, scale 1.0)
    assert r.unit.name == ""
    assert r.unit.scale_to_si == 1.0

    # 3) Numeric value ~ 1, allow tiny drift (repr shows the magnitude in current unit)
    # If repr prints a bare number for dimensionless quantities, cast to float safely:
    val = float(repr(r))
    assert val == pytest.approx(1.0, rel=1e-12, abs=1e-15)

# -------------------------------
# .si property
# -------------------------------

def test_si_equivalent_to_to_si():
    m = ureg.get("m")
    cm = ureg.get("cm")

    q = 123 @ cm        # 1.23 m in SI
    q_si = q.si
    q_to_si = q.to_si()

    # Same dim and SI scale
    assert q_si.unit.scale_to_si == 1.0
    assert q_si.dim == q_to_si.dim
    assert q_si.unit == q_to_si.unit
    assert math.isclose(q_si._mag_si, q_to_si._mag_si, rel_tol=1e-12, abs_tol=0.0)


def test_si_uses_preferred_symbol_when_available(monkeypatch):
    # Arrange: make preferred_symbol_for_dim return a symbol for LENGTH
    import quantium.core.utils as utils

    def fake_preferred(dim):
        return "m" if dim == LENGTH else None

    def fake_format(dim):
        return "LENGTH?"  # should not be used in this test

    monkeypatch.setattr(utils, "preferred_symbol_for_dim", fake_preferred, raising=True)
    monkeypatch.setattr(utils, "format_dim", fake_format, raising=True)

    cm = ureg.get("cm")
    q_si = (123 @ cm).si  # 1.23 m

    assert isinstance(q_si, Quantity)
    assert q_si.unit.name == "m"         # preferred symbol chosen
    assert q_si.unit.scale_to_si == 1.0  # SI unit
    assert math.isclose(q_si._mag_si, 1.23)


def test_si_fallbacks_to_formatted_dim_when_no_symbol(monkeypatch):
    import quantium.core.utils as utils

    def fake_preferred(dim):
        return None  # force fallback

    def fake_format(dim):
        # composed SI name for velocity
        return "m/s" if dim == dim_div(LENGTH, TIME) else "1"

    monkeypatch.setattr(utils, "preferred_symbol_for_dim", fake_preferred, raising=True)
    monkeypatch.setattr(utils, "format_dim", fake_format, raising=True)

    m = ureg.get("m")
    s = ureg.get("s")
    q = ((5 @ m) / (2 @ s)).si  # 2.5 m/s

    assert q.unit.name == "m/s"
    assert q.unit.scale_to_si == 1.0
    assert math.isclose(q._mag_si, 2.5)


def test_si_does_not_mutate_original_quantity():
    cm = ureg.get("cm")
    ms = ureg.get("ms")

    v = 1000 @ (cm / ms)  # original in cm/ms
    v_si = v.si           # 10000 m/s

    # original unchanged
    assert v.unit == (cm / ms)
    # new object is SI
    assert v_si.unit.scale_to_si == 1.0
    assert math.isclose(v_si._mag_si, 10000.0)


def test_si_repr_respects_preferred_symbol_when_scale_is_1(monkeypatch):
    # Ensure repr of q.si upgrades to the preferred symbol (since SI scale is 1.0)
    import quantium.core.utils as utils

    # prettifier passthrough to assert exact string
    monkeypatch.setattr(utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True)
    # preferred symbol for LENGTH is "m"
    monkeypatch.setattr(utils, "preferred_symbol_for_dim", lambda d: "m" if d == LENGTH else None, raising=True)

    cm = ureg.get("cm")
    q_si = (123 @ cm).si  # 1.23 m in SI
    assert repr(q_si) == "1.23 m"
