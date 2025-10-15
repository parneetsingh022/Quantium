import math

import pytest

from quantium.core.dimensions import DIM_0, LENGTH, TEMPERATURE, dim_div, dim_mul, dim_pow
from quantium.core.quantity import Unit

def _name(sym: str, n: int) -> str:
    return "1" if n == 0 else (sym if n == 1 else f"{sym}^{n}")

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

