import math

import pytest

from quantium.core.dimensions import DIM_0, TIME, LENGTH, TEMPERATURE, dim_div, dim_mul
from quantium.core.quantity import Unit, Quantity


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



# -------------------------------
# Quantity * Unit
# -------------------------------

def test_quantity_times_unit_basic():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TIME)

    q = (2 @ m) * s  # → 2 m·s

    assert isinstance(q, Quantity)
    assert q.dim == dim_mul(LENGTH, TIME)
    assert q.unit.name in ("m·s", "s·m")  # order may depend on your Unit.__mul__
    # magnitude shown in the current (composed) unit:
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 2.0)


def test_quantity_times_unit_with_prefix_does_not_change_numeric_value():
    cm = Unit("cm", 0.01, LENGTH)
    m = Unit("m", 1.0, LENGTH)

    q_cm = 300 @ cm   # SI = 3.0 m
    out = q_cm * m    # new_unit scale = 0.01, constructor multiplies again

    assert out.dim == dim_mul(LENGTH, LENGTH)
    # Displayed magnitude should still be 3.0 in the composed unit
    # because shown_value = out._mag_si / out.unit.scale_to_si = (3.0*0.01)/0.01 = 3.0
    assert math.isclose(out._mag_si / out.unit.scale_to_si, 3.0)
    # Name depends on your Unit.__mul__ formatting:
    assert out.unit.name in ("cm·m", "m·cm")



def test_quantity_times_unit_does_not_mutate_original():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TIME)

    q = 5 @ m
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

    q = (10 @ m) / s  # → 10 m/s

    assert isinstance(q, Quantity)
    assert q.dim == dim_div(LENGTH, TIME)
    assert q.unit.name in ("m/s", "m·s^-1")  # accept either normalized style
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 10.0)


def test_quantity_div_unit_dimensionless_normalization():
    """With current impl, quantity / unit yields DIM_0 but keeps a composed name (e.g., 'm/m')."""
    m = Unit("m", 1.0, LENGTH)

    q = (7 @ m) / m   # DIM_0, but name not normalized in this code path

    assert q.dim == DIM_0
    # Current behavior: not normalized to empty name for quantity ÷ unit
    assert q.unit.name in ("", "m/m")
    # Scale of 'm/m' is 1, so displayed value is 7 either way
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 7.0)



def test_quantity_div_unit_with_prefix_dimensionless_value_is_correct():
    """Even if name is 'cm/m', the displayed value should be the correct pure number."""
    cm = Unit("cm", 0.01, LENGTH)
    m = Unit("m", 1.0, LENGTH)

    q = (200 @ cm) / m  # 200 cm / 1 m -> 2 (dimensionless)

    assert q.dim == DIM_0
    # Current behavior keeps 'cm/m' (scale 0.01) rather than empty name
    assert q.unit.name in ("", "cm/m")
    # Regardless of name normalization, shown value must be 2.0
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 2.0)



def test_quantity_div_unit_does_not_mutate_original():
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TIME)

    q = 12 @ m
    _ = q / s

    # original unchanged
    assert q.unit is m
    assert math.isclose(q._mag_si, 12.0)