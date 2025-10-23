import pytest

from quantium.core.dimensions import DIM_0, TIME, LENGTH, MASS
from quantium.core.quantity import Unit
from quantium.units.registry import DEFAULT_REGISTRY as ureg
from math import isclose
from quantium import u

# Helper function to mock prettify (if you use it in other tests)
def _nop_prettifier(monkeypatch):
    import quantium.core.utils as utils
    monkeypatch.setattr(
        utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True
    )


TIME_INV = TIME ** -1

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


# --- Tests for __repr__ Auto-Prefixing of Composed Units ---

@pytest.mark.regression(reason="Issue #75: Fix for non-standard SI scales in __repr__")
def test_repr_upgrades_non_standard_scale_N_per_cm2():
    """
    Tests the original problem: 4 N/cm² (scale 10^4) should be
    auto-formatted to 40 kPa (using engineering notation).
    """
    N = ureg.get("N")
    cm = ureg.get("cm")

    force = 100 * N
    area = 25 * cm**2
    pressure = force / area  # 4 N/cm²

    # Check that the value is correct (4e4 Pa)
    assert pressure.dim == u.Pa.dim
    assert isclose(pressure._mag_si, 40000.0)
    assert pressure.unit.name == "N/cm^2"
    assert isclose(pressure.unit.scale_to_si, 10000.0)

    # Check that __repr__ now correctly formats 4e4 Pa as 40 kPa
    assert f"{pressure}" == "40 kPa"


@pytest.mark.regression(reason="Issue #75: Fix for non-standard SI scales in __repr__")
def test_repr_upgrades_non_standard_scale_kN_per_cm2():
    """
    Tests a more complex non-standard scale: 4 kN/cm² (scale 10^7)
    should be auto-formatted to 40 MPa.
    """
    kN = ureg.get("kN")
    cm = ureg.get("cm")

    force = 100 * kN
    area = 25 * cm**2
    pressure = force / area  # 4 kN/cm²

    # Check that the value is correct (4e7 Pa)
    assert pressure.dim == u.Pa.dim
    assert isclose(pressure._mag_si, 40_000_000.0)
    assert pressure.unit.name == "kN/cm^2"
    assert isclose(pressure.unit.scale_to_si, 10**7)

    # Check that __repr__ now correctly formats 4e7 Pa as 40 MPa
    assert f"{pressure}" == "40 MPa"


@pytest.mark.regression(reason="Issue #75: Fix for non-standard SI scales in __repr__")
def test_repr_upgrades_non_standard_scale_small_value():
    """
    Tests a non-standard small scale: 4 N/km² (scale 10^-6)
    This scale *does* match a prefix (micro, µ), so the *old* logic
    path should handle it correctly, formatting it as 4 µPa.
    """
    N = ureg.get("N")
    km = ureg.get("km")

    force = 100 * N
    area = 25 * km**2
    pressure = force / area  # 4 N/km²

    # Check value (4e-6 Pa)
    assert isclose(pressure._mag_si, 4.0e-6)
    assert isclose(pressure.unit.scale_to_si, 1.0e-6)

    # Check that __repr__ formats this as 4 µPa
    # This confirms the pre-existing prefix-matching logic still works
    assert f"{pressure}" == "4 µPa"


# --- Tests to Confirm Regressions Stay Fixed ---

@pytest.mark.regression(reason="Issue #75: Confirm fix for __repr__ bug #72")
def test_repr_regression_fix_issue72_kg_mg_per_kg(monkeypatch):
    """
    Confirms that 'kg·mg/kg' is prettified to 'mg' *before*
    the composed-unit check, preventing it from being upgraded to 'mkg'.
    """
    # This test doesn't need the nop_prettifier, it relies on the real one
    import quantium.core.utils as utils
    utils.invalidate_preferred_cache()

    kg = ureg.get("kg")
    mg = ureg.get("mg")

    dose_rate = 15 * (mg / kg)
    patient_mass = 75 * kg
    required_dose = patient_mass * dose_rate  # 1125 kg·mg/kg

    assert required_dose.unit.name == "kg·mg/kg"
    assert required_dose.dim == MASS

    # The fix ensures this prints "1125 mg", not "1.125 mkg"
    assert f"{required_dose}" == "1125 mg"


@pytest.mark.regression(reason="Issue #75: Confirm fix for __repr__ bug")
def test_repr_regression_fix_keeps_non_si_unit(monkeypatch):
    """
    Confirms that a simple non-SI unit ('cm') is not auto-formatted
    by the new logic.
    """
    _nop_prettifier(monkeypatch) # Use nop to check the raw unit name
    import quantium.core.utils as utils
    monkeypatch.setattr(utils, "preferred_symbol_for_dim", lambda d: "m", raising=True)

    cm = Unit("cm", 0.01, LENGTH)
    q = 2 * cm

    # The fix ensures this prints "2 cm", not "20 mm"
    assert repr(q) == "2 cm"


@pytest.mark.regression(reason="Issue #75: Confirm fix for __repr__ bug")
def test_repr_regression_fix_atomic_units_not_flipped(monkeypatch):
    """
    Confirms that atomic SI symbols (Bq, Hz) are not auto-formatted,
    even though they share a dimension.
    """
    _nop_prettifier(monkeypatch)
    import quantium.core.utils as utils
    utils.invalidate_preferred_cache()
    
    Bq = ureg.get("Bq")
    Hz = ureg.get("Hz")
    q_Bq = 100 * Bq
    q_Hz = 100 * Hz
    
    assert q_Bq.dim == TIME_INV
    assert q_Hz.dim == TIME_INV

    # The fix ensures these print as-is, not flipping to the other
    assert f"{q_Bq}" == "100 Bq"
    assert f"{q_Hz}" == "100 Hz"


@pytest.mark.regression(reason="Issue #75: Confirm fix for __repr__ bug")
def test_repr_regression_fix_format_si_works(monkeypatch):
    """
    Confirms that the ':si' formatter still works, as it relies on
    __repr__ not re-formatting the base SI unit ('Pa').
    """
    _nop_prettifier(monkeypatch)

    kPa = ureg.get("kPa")
    uF = ureg.get("uF")
    
    p = 2 * kPa   # 2000 Pa
    q = 3 * uF    # 3e-6 F

    # p.to_si() creates a Quantity(2000, unit=Pa).
    # The fix ensures repr(Quantity(2000, unit=Pa)) is "2000 Pa",
    # not "2 kPa".
    assert f"{p:si}" == "2000 Pa"

    # q.to_si() creates Quantity(3e-6, unit=F).
    # The fix ensures repr(Quantity(3e-6, unit=F)) is "3e-06 F",
    # not "3 µF".
    assert f"{q:si}" == "3e-06 F"