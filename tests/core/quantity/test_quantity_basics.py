import math
import pytest
from quantium.core.dimensions import LENGTH, TEMPERATURE
from quantium.core.quantity import Quantity, Unit

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