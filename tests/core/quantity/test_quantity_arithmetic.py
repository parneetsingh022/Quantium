import math

import pytest

from quantium.core.dimensions import DIM_0, TIME, LENGTH, TEMPERATURE, dim_div, dim_mul
from quantium.core.quantity import Unit, Quantity
from quantium import u


# -------------------------------
# Arithmetic: +, -, *, /, **, scalars
# -------------------------------

def test_add_and_sub_same_dim():
    m = Unit("m", 1.0, LENGTH)
    cm = Unit("cm", 0.01, LENGTH)
    q1 = 1 * m
    q2 = 50 * cm  # 0.5 m

    s = q1 + q2   # left unit ("m") retained
    d = q1 - q2

    assert s.unit is m and d.unit is m
    assert math.isclose(s._mag_si / s.unit.scale_to_si, 1.5)
    assert math.isclose(d._mag_si / d.unit.scale_to_si, 0.5)

def test_add_dim_mismatch_raises():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TEMPERATURE)
    with pytest.raises(TypeError):
        _ = (1 * m) + (1 * s)

def test_scalar_multiplication_and_division():
    m = Unit("m", 1.0, LENGTH)
    q = 2 * m

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
    q = (2 * m) * (3 * s)  # -> 6 m·s

    assert q.dim == dim_mul(LENGTH, TEMPERATURE)
    assert q.unit.name == "m·s"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 6.0)

def test_quantity_div_quantity():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TEMPERATURE)
    q = (10 * m) / (2 * s)  # -> 5 m/s

    assert q.dim == dim_div(LENGTH, TEMPERATURE)
    assert q.unit.name == "m/s"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 5.0)

def test_scalar_divided_by_quantity():
    m = Unit("m", 1.0, LENGTH)
    q = 2 / (2 * m)  # -> 1 (1/m)

    assert q.dim == dim_div(DIM_0, LENGTH)
    assert q.unit.name == "1/m"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 1.0)



# -------------------------------
# Quantity * Unit
# -------------------------------

def test_quantity_times_unit_basic():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TIME)

    q = (2 * m) * s  # → 2 m·s

    assert isinstance(q, Quantity)
    assert q.dim == dim_mul(LENGTH, TIME)
    assert q.unit.name in ("m·s", "s·m")  # order may depend on your Unit.__mul__
    # magnitude shown in the current (composed) unit:
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 2.0)


def test_quantity_times_unit_with_prefix_does_not_change_numeric_value():
    cm = Unit("cm", 0.01, LENGTH)
    m = Unit("m", 1.0, LENGTH)

    q_cm = 300 * cm   # SI = 3.0 m
    out = q_cm * m    # new_unit scale = 0.01, constructor multiplies again

    assert out.dim == dim_mul(LENGTH, LENGTH)
    # Displayed magnitude should still be 3.0 in the composed unit
    # because shown_value = out._mag_si / out.unit.scale_to_si = (3.0*0.01)/0.01 = 3.0
    assert math.isclose(out._mag_si / out.unit.scale_to_si, 300.0)
    # Name depends on your Unit.__mul__ formatting:
    assert out.unit.name in ("cm·m", "m·cm")



def test_quantity_times_unit_does_not_mutate_original():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TIME)

    q = 5 * m
    _ = q * s

    # original unchanged
    assert q.unit is m
    assert math.isclose(q._mag_si, 5.0)


# -------------------------------
# Quantity / Unit
# -------------------------------

def test_quantity_div_unit_basic():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TIME)

    q = (10 * m) / s  # → 10 m/s

    assert isinstance(q, Quantity)
    assert q.dim == dim_div(LENGTH, TIME)
    assert q.unit.name in ("m/s", "m·s^-1")  # accept either normalized style
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 10.0)


def test_quantity_div_unit_dimensionless_normalization():
    """With current impl, quantity / unit yields DIM_0 but keeps a composed name (e.g., 'm/m')."""
    m = Unit("m", 1.0, LENGTH)

    q = (7 * m) / m   # DIM_0, but name not normalized in this code path

    assert q.dim == DIM_0
    # Current behavior: not normalized to empty name for quantity ÷ unit
    assert q.unit.name in ("", "m/m")
    # Scale of 'm/m' is 1, so displayed value is 7 either way
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 7.0)



def test_quantity_div_unit_with_prefix_dimensionless_value_is_correct():
    """Even if name is 'cm/m', the displayed value should be the correct pure number."""
    cm = Unit("cm", 0.01, LENGTH)
    m = Unit("m", 1.0, LENGTH)

    q = (200 * cm) / m  # 200 cm / 1 m -> 2 (dimensionless)

    assert q.dim == DIM_0
    # Current behavior keeps 'cm/m' (scale 0.01) rather than empty name
    assert q.unit.name in ("", "cm/m")
    # Regardless of name normalization, shown value must be 2.0
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 2.0)



def test_quantity_div_unit_does_not_mutate_original():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TIME)

    q = 12 * m
    _ = q / s

    # original unchanged
    assert q.unit is m
    assert math.isclose(q._mag_si, 12.0)

# --- Tests for Issue #67 ---

@pytest.mark.regression(reason="Issue #67: Operator precedence bug in scalar * unit / unit")
def test_regression_67_scalar_times_unit_div_unit_precedence():
    """
    Tests the exact failing case from Issue #67.
    1000 * u.cm / u.s was evaluated as (1000 * u.cm) / u.s,
    and the bug in Quantity.__truediv__(Unit) caused an incorrect value.
    """
    cm =  u("cm")
    s =  u("s")

    # This is evaluated as (1000 * cm) / s
    q = 1000 * cm / s

    # q1 = 1000 * cm -> (value=1000, _mag_si=10.0)
    # q_new = q1 / s
    # Fixed code uses q1.value (1000.0)
    # q_new = Quantity(1000.0, cm/s)
    #   q_new._mag_si = 1000.0 * 0.01 = 10.0
    #   q_new.value = 10.0 / 0.01 = 1000.0

    assert isinstance(q, Quantity)
    assert q.unit.name == "cm/s"
    assert math.isclose(q._mag_si, 10.0)
    assert math.isclose(q.value, 1000.0)


@pytest.mark.regression(reason="Issue #67: Fix for Quantity * Unit constructor")
def test_regression_67_quantity_times_unit_uses_value():
    """
    Explicitly tests that (Quantity) * (Unit) uses self.value, not self._mag_si.
    """
    cm =  u("cm") # scale 0.01
    m =  u("m")   # scale 1.0

    q1 = 1000 * cm  # (value=1000, _mag_si=10.0)
    q2 = q1 * m     # Should be 1000 cm·m

    # Fixed code: Quantity(q1.value, cm·m) -> Quantity(1000.0, cm·m)
    #   q2.unit = cm·m (scale 0.01)
    #   q2._mag_si = 1000.0 * 0.01 = 10.0
    #   q2.value = 10.0 / 0.01 = 1000.0
    
    assert q2.unit.name == "cm·m"
    assert math.isclose(q2._mag_si, 10.0)
    assert math.isclose(q2.value, 1000.0)


@pytest.mark.regression(reason="Issue #67: Fix for Quantity / Unit constructor")
def test_regression_67_quantity_div_unit_uses_value():
    """
    Explicitly tests that (Quantity) / (Unit) uses self.value, not self._mag_si.
    """
    cm =  u("cm") # scale 0.01
    s =  u("s")   # scale 1.0

    q1 = 1000 * cm  # (value=1000, _mag_si=10.0)
    q2 = q1 / s     # Should be 1000 cm/s

    # Fixed code: Quantity(q1.value, cm/s) -> Quantity(1000.0, cm/s)
    #   q2.unit = cm/s (scale 0.01)
    #   q2._mag_si = 1000.0 * 0.01 = 10.0
    #   q2.value = 10.0 / 0.01 = 1000.0
    
    assert q2.unit.name == "cm/s"
    assert math.isclose(q2._mag_si, 10.0)
    assert math.isclose(q2.value, 1000.0)


@pytest.mark.regression(reason="Bugfix: Quantity * Unit dimensionless path")
def test_quantity_times_unit_resulting_in_dimensionless():
    """
    Tests the `if new_unit.dim == DIM_0:` branch in Quantity.__mul__.
    Ensures the SI magnitude is calculated correctly and a scale=1 unit is used.
    """
    m = u.m
    cm = u.cm

    # 1. Simple case: (10 m) * (1/m)
    q1 = 10 * m
    inv_m = 1 / m  # scale_to_si = 1.0

    q_final_1 = q1 * inv_m

    assert q_final_1.dim == DIM_0
    assert q_final_1.unit.scale_to_si == 1.0
    assert math.isclose(q_final_1._mag_si, 10.0)  # 10.0 * 1.0
    assert math.isclose(q_final_1.value, 10.0)

    # 2. Prefixed case: (10 m) * (1/cm)
    q2 = 10 * m  # _mag_si = 10.0
    inv_cm = 1 / cm  # scale_to_si = 1 / 0.01 = 100.0

    q_final_2 = q2 * inv_cm  # 10 m * (1 / 0.01 m) = 1000

    assert q_final_2.dim == DIM_0
    assert q_final_2.unit.scale_to_si == 1.0
    assert math.isclose(q_final_2._mag_si, 1000.0)  # 10.0 * 100.0
    assert math.isclose(q_final_2.value, 1000.0)

    # 3. Prefixed case (other way): (10 cm) * (1/m)
    q3 = 10 * cm  # _mag_si = 0.1
    # inv_m is from case 1 (scale_to_si = 1.0)

    q_final_3 = q3 * inv_m  # 10 cm * (1/m) = 0.1 m * (1/m) = 0.1

    assert q_final_3.dim == DIM_0
    assert q_final_3.unit.scale_to_si == 1.0
    assert math.isclose(q_final_3._mag_si, 0.1)  # 0.1 * 1.0
    assert math.isclose(q_final_3.value, 0.1)


@pytest.mark.regression(reason="Bugfix: Cover Quantity * Unit dimensionless path")
def test_quantity_times_inverse_unit_simple():
    """
    Explicitly tests the `if new_unit.dim == DIM_0:` branch in Quantity.__mul__
    with a single simple case to satisfy code coverage.
    """
    m = u("m")
    q = 5 * m       # _mag_si = 5.0
    inv_m = 1 / m   # scale_to_si = 1.0

    # This calls q.__mul__(inv_m)
    result = q * inv_m
    
    # Test the exact lines from the coverage report
    # new_mag_si = self._mag_si * other.scale_to_si (5.0 * 1.0)
    # return Quantity(new_mag_si, unit_dimless)
    assert result.dim == DIM_0
    assert result.unit.scale_to_si == 1.0
    assert math.isclose(result._mag_si, 5.0)
    assert math.isclose(result.value, 5.0)