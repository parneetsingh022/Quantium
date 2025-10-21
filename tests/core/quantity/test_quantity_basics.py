import math
import pytest
from quantium.core.dimensions import LENGTH, TEMPERATURE,TIME, DIM_0
from quantium.core.quantity import Quantity, Unit
from quantium.units.registry import DEFAULT_REGISTRY as dreg
from quantium import u
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
# __rmatmul__: value * Unit
# -------------------------------

def test_rmatmul_operator():
    m = Unit("m", 1.0, LENGTH)
    q = 3 * m
    assert isinstance(q, Quantity)
    assert q.dim == LENGTH
    assert q.unit is m
    assert math.isclose(q._mag_si, 3.0)


# -------------------------------
# quantity as_name()
# -------------------------------

def test_quantity_has_as_name_and_returns_quantity():
    K = dreg.get("K")
    q = 100 * K

    out = q.as_name("kelvin")

    assert isinstance(out, Quantity)
    assert out is not q  # returns a new object
    assert out.unit.name == "kelvin"
    # same SI magnitude and same dimension
    assert math.isclose(out._mag_si, q._mag_si)
    assert out.dim == q.dim
    # unit scale and dim unchanged; only the name should differ
    assert math.isclose(out.unit.scale_to_si, q.unit.scale_to_si)
    assert out.unit.dim == q.unit.dim


def test_quantity_as_name_does_not_mutate_original():
    m = Unit("m", 1.0, LENGTH)
    q = 5 * m

    out = q.as_name("meter")

    # original untouched
    assert q.unit.name == "m"
    assert math.isclose(q._mag_si, 5.0)
    # new has the new name
    assert out.unit.name == "meter"
    # values/dims consistent
    assert math.isclose(out._mag_si, 5.0)
    assert out.dim == LENGTH


def test_quantity_as_name_works_with_registry_kelvin_repr():
    K = dreg.get("K")
    q = (100 * K).as_name("kelvin")
    # repr should reflect the new unit name
    assert repr(q) == "100 kelvin"


def test_quantity_as_name_on_composed_quantity_preserves_scale_and_value():
    m = Unit("m", 1.0, LENGTH)
    q = (2 * m) * (3 * m)   # 6 m·m (your Unit.__mul__ may collapse to m^2)
    shown_before = q._mag_si / q.unit.scale_to_si

    out = q.as_name("area")
    shown_after = out._mag_si / out.unit.scale_to_si

    # Only the unit name changes
    assert out.unit.name == "area"
    assert math.isclose(out.unit.scale_to_si, q.unit.scale_to_si)
    assert out.unit.dim == q.unit.dim
    # magnitude shown to users remains identical
    assert math.isclose(shown_after, shown_before)
    # repr matches the new name
    assert repr(out) == f"{shown_before:g} area"


def test_quantity_as_name_chainable():
    K = dreg.get("K")
    q = (100 * K).as_name("kelvin").as_name("K")
    assert q.unit.name == "K"
    assert repr(q) == "100 K"


def test_quantity_as_name_on_dimensionless_is_allowed():
    # Create a dimensionless quantity by dividing equivalent units
    s = Unit("s", 1.0, TIME)
    q = (10 * s) / (2 * s)  # -> 5 (dimensionless)
    assert q.dim == DIM_0

    out = q.as_name("1")  # rename the unit label for display
    assert out.dim == DIM_0
    assert out.unit.name == "1"
    assert math.isclose(out._mag_si, q._mag_si)
    assert repr(out) == "5"


# ----------------------------
# Helpers
# ----------------------------
def shown(q: Quantity) -> float:
    """Return the magnitude shown in q's current unit (not SI)."""
    return q._mag_si / q.unit.scale_to_si


# ----------------------------
# Happy-path conversions using string expressions
# ----------------------------

def test_to_string_simple_prefix_change_velocity():
    # 10 m/s -> 1000 cm/s
    q = 10 * dreg.get("m/s")
    out = q.to("cm/s")
    assert math.isclose(shown(out), 1000.0)
    assert out.dim == q.dim

def test_to_string_acceleration_power_syntax():
    # 9.8 m/s^2 -> 980 cm/s^2, using ** syntax required by your parser
    q = 9.8 * dreg.get("m/s**2")
    out = q.to("cm/s**2")
    assert math.isclose(shown(out), 980.0, rel_tol=1e-12)

def test_to_string_force_from_newton_to_base_composed():
    # 1 N -> 1 kg·m/s^2
    q = 1 * dreg.get("N")
    out = q.to("kg*m/s**2")
    assert math.isclose(shown(out), 1.0)
    assert out.dim == q.dim

def test_to_string_energy_newton_meter_equivalence():
    # 3 J -> 3 N·m
    q = 3 * dreg.get("J")
    out = q.to("N*m")
    assert math.isclose(shown(out), 3.0)
    assert out.dim == q.dim

def test_to_string_power_joule_per_second():
    # 7 W -> 7 J/s
    q = 7 * dreg.get("W")
    out = q.to("J/s")
    assert math.isclose(shown(out), 7.0)
    assert out.dim == q.dim

def test_to_string_frequency_from_kHz_to_one_over_s():
    # 1 kHz -> 1000 1/s
    q = 1 * dreg.get("kHz")
    out = q.to("1/s")
    assert math.isclose(shown(out), 1000.0)
    assert out.dim == q.dim

def test_to_string_pressure_pascal_to_N_per_m2():
    # 101325 Pa -> 101325 N/m^2
    q = 101_325 * dreg.get("Pa")
    out = q.to("N/m**2")
    assert math.isclose(shown(out), 101_325.0)
    assert out.dim == q.dim

def test_to_string_parentheses_and_mixed_ops():
    # 100 m/s -> 10000 (cm)/s using parentheses and explicit *
    q = 100 * dreg.get("m/s")
    out = q.to("(cm) / (s)")
    assert math.isclose(shown(out), 10_000.0)
    assert out.dim == q.dim

def test_to_string_with_micro_alias_in_denominator():
    # 1 / ms -> 1000 1/s (Hz dimension), using 'ms' in target string
    q = 1 * (1 / dreg.get("ms"))  # Quantity with T^-1
    out = q.to("1/s")
    assert math.isclose(shown(out), 1000.0)
    assert out.dim == q.dim

def test_to_string_with_ohm_alias_normalization():
    # 'ohm' alias should resolve to 'Ω' under the hood
    q = 5 * dreg.get("Ω")
    out = q.to("ohm")
    assert math.isclose(shown(out), 5.0)
    # Registry normalizes alias to canonical "Ω"
    assert out.unit.name == "Ω"
    assert out.dim == q.dim


# --------------------------------------------------------
# Tests conversion from one naming system to other
# --------------------------------------------------------

def test_to_physically_equivalent_different_name():
    """
    Tests the bug fix: converting to a unit that is physically
    identical but has a different name should return a NEW object.
    """
    q1 = Quantity(5.0, u.W/(u.A*u.m))  # 5.0 W/(A·m)

    # Test conversion using a Unit object
    q2 = q1.to(u.V/u.m)            # Convert to V/m

    # 1. Check physical equivalence (value is the same)
    assert q1 == q2
    assert f"{q2}" == "5 V/m"

    # 2. Check that it is a NEW object with the new unit name
    assert q1 is not q2
    assert q2.unit.name == 'V/m'
    assert q1.unit.name == 'W/(A·m)' # Original is unchanged

    # Test conversion using a string name
    q3 = q1.to("V/m")            # Convert to V/m

    # 3. Check physical equivalence
    assert q1 == q3
    assert f"{q3}" == "5 V/m"

    # 4. Check that it is a NEW object
    assert q1 is not q3
    assert q3.unit.name == 'V/m'

def test_to_identical_name_optimization():
    """
    Tests the optimization path: converting to the *exact same unit*
    (identical name) should return the SAME object (`self`).
    """
    q1 = Quantity(10.0, u.V/u.m)  # 10.0 V/m

    # Test conversion using the *same* Unit object
    q2 = q1.to(u.V/u.m)

    # Check that it returned the *exact same object*
    assert q1 is q2
    assert f"{q2}" == "10 V/m"

    # Test conversion using the *same* string name
    q3 = q1.to("V/m")

    # Check that this also returned the *exact same object*
    assert q1 is q3
    assert f"{q3}" == "10 V/m"

# ----------------------------
# Identity / fast path
# ----------------------------

def test_to_string_identity_fast_path_returns_same_object():
    # .to("m") on a Quantity already in meters should return self
    q = 2.5 * dreg.get("m")
    r = q.to("m")
    assert r is q


# ----------------------------
# Dimension mismatch and invalid syntax
# ----------------------------

def test_to_string_dimension_mismatch_raises():
    # 3 m -> "kg" should raise TypeError
    q = 3 * dreg.get("m")
    with pytest.raises(TypeError):
        _ = q.to("kg")

def test_to_string_invalid_expression_raises_value_error():
    # Parser should reject invalid tokens like '//' and raise ValueError
    q = 1 * dreg.get("m/s")
    with pytest.raises(ValueError):
        _ = q.to("m//s")


# ----------------------------
# More complex target expressions
# ----------------------------

def test_to_string_complex_nested_equivalence():
    # 1 N -> (kg·m)/(s^2) using parentheses and ** syntax
    q = 1 * dreg.get("N")
    out = q.to("(kg*m)/(s**2)")
    assert math.isclose(shown(out), 1.0)
    assert out.dim == q.dim

def test_to_string_uses_registry_parser_consistently_with_sources():
    # Construct a unit via a complex expression and convert to a different but equivalent one
    q = 4 * dreg.get("(W*s)/(N*s/m**2)")  # dimension L^3/T
    out = q.to("m**3/s")
    assert math.isclose(shown(out), 4.0)
    assert out.dim == q.dim


# ----------------------------
# Dimensionless support
# ----------------------------

def test_to_string_dimensionless_no_change():
    # (10 s) / (2 s) -> 5 (dimensionless); converting to "1" keeps it dimensionless
    s = dreg.get("s")
    q = (10 * s) / (2 * s)
    assert q.dim == DIM_0
    out = q.to("1")
    assert out.dim == DIM_0
    assert math.isclose(shown(out), 5.0)