import math
import pytest
from quantium.core.dimensions import LENGTH, TEMPERATURE,TIME, DIM_0
from quantium.core.quantity import Quantity, Unit
from quantium.units.registry import DEFAULT_REGISTRY as dreg
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
# quantity as_name()
# -------------------------------

def test_quantity_has_as_name_and_returns_quantity():
    K = dreg.get("K")
    q = 100 @ K

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
    q = 5 @ m

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
    q = (100 @ K).as_name("kelvin")
    # repr should reflect the new unit name
    assert repr(q) == "100 kelvin"


def test_quantity_as_name_on_composed_quantity_preserves_scale_and_value():
    m = Unit("m", 1.0, LENGTH)
    q = (2 @ m) * (3 @ m)   # 6 m·m (your Unit.__mul__ may collapse to m^2)
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
    q = (100 @ K).as_name("kelvin").as_name("K")
    assert q.unit.name == "K"
    assert repr(q) == "100 K"


def test_quantity_as_name_on_dimensionless_is_allowed():
    # Create a dimensionless quantity by dividing equivalent units
    s = Unit("s", 1.0, TIME)
    q = (10 @ s) / (2 @ s)  # -> 5 (dimensionless)
    assert q.dim == DIM_0

    out = q.as_name("1")  # rename the unit label for display
    assert out.dim == DIM_0
    assert out.unit.name == "1"
    assert math.isclose(out._mag_si, q._mag_si)
    assert repr(out) == "5"