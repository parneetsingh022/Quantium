import pytest

from quantium.core.dimensions import LENGTH, TIME
from quantium.core.quantity import Quantity, Unit
from quantium.units.registry import DEFAULT_REGISTRY as ureg


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