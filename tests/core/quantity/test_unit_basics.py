from dataclasses import FrozenInstanceError
import pytest
from quantium.core.dimensions import LENGTH
from quantium.core.quantity import Unit


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