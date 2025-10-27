import math
from fractions import Fraction

import pytest

from quantium.core.dimensions import (
    DIM_0,
    LENGTH,
    IrrationalExponentError,
    dim_div,
    dim_mul,
    dim_pow,
)
from quantium.core.quantity import Unit

def _name(sym: str, n: int) -> str:
    #return "1" if n == 0 else (sym if n == 1 else f"{sym}^{n}")
    if n == 0:
        return 1
    if n >= 1: return f"{sym}^{n}"

    return f"1/{sym}^{abs(n)}"

def test_power_of_quantity():
    m = Unit("m", 1.0, LENGTH)
    q2 = (2 * m) ** 2  # -> 4 m^2

    assert q2.dim == dim_pow(LENGTH, 2)
    assert q2.unit.name == "m^2"
    assert math.isclose(q2._mag_si / q2.unit.scale_to_si, 4.0)


@pytest.mark.regression(reason="Issue: #33 Unit raised to power 0 is not dimensionless")
def test_quantity_pow_zero_and_one_and_negative():
    m = Unit("m", 1.0, LENGTH)
    q = 2 * m
    q0 = q ** 0
    q1 = q ** 1
    qn = q ** -2
    assert q0.dim == DIM_0 and q0.unit.scale_to_si == 1.0
    assert q1.dim == LENGTH
    assert qn.dim == dim_pow(LENGTH, -2)


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
    q = value * u
    qp = q ** n

    # Dimension & unit name/scale
    assert qp.dim == dim_pow(LENGTH, n)
    assert qp.unit.name == _name(sym, n)
    assert qp.unit.scale_to_si == pytest.approx(scale ** n)

    # Magnitude in the resulting unit should be value**n
    mag_in_unit = qp._mag_si / qp.unit.scale_to_si
    assert math.isclose(mag_in_unit, value ** n, rel_tol=1e-12, abs_tol=1e-12)


def test_quantity_pow_fraction_exponent_dimensioned():
    m = Unit("m", 1.0, LENGTH)
    q = 9 * m
    exponent = Fraction(1, 2)

    qp = q ** exponent

    assert qp.dim == dim_pow(LENGTH, exponent)
    assert qp.unit.name == "m^(1/2)"
    assert math.isclose(qp.unit.scale_to_si, 1.0)
    mag_in_unit = qp._mag_si / qp.unit.scale_to_si
    assert math.isclose(mag_in_unit, math.sqrt(9), rel_tol=1e-12, abs_tol=1e-12)


def test_quantity_pow_float_rationalized_dimensioned():
    m = Unit("m", 1.0, LENGTH)
    q = 16 * m
    exponent = 0.5

    qp = q ** exponent

    assert qp.dim == dim_pow(LENGTH, Fraction(1, 2))
    assert qp.unit.name == "m^(1/2)"
    mag_in_unit = qp._mag_si / qp.unit.scale_to_si
    assert math.isclose(mag_in_unit, math.sqrt(16), rel_tol=1e-12, abs_tol=1e-12)


def test_quantity_pow_dimensionless_real_exponent():
    one = Unit("", 1.0, DIM_0)
    q = 3 * one

    exponent = math.pi
    qp = q ** exponent

    assert qp.dim == DIM_0
    assert qp.unit.name == ""
    assert math.isclose(qp._mag_si / qp.unit.scale_to_si, 3 ** exponent, rel_tol=1e-12, abs_tol=1e-12)


def test_quantity_pow_dimensionless_imaginary_exponent_error():
    one = Unit("", 1.0, DIM_0)
    q = 2 * one

    with pytest.raises(TypeError):
        _ = q ** complex(0, 1)


def test_quantity_pow_dimensioned_irrational_exponent_error():
    m = Unit("m", 1.0, LENGTH)
    q = 2 * m

    with pytest.raises(IrrationalExponentError):
        _ = q ** math.pi


def test_quantity_repr_fractional_power():
    m = Unit("m", 1.0, LENGTH)
    q = 100 * m
    half = q ** Fraction(1, 2)
    assert repr(half) == "10 m^(1/2)"

    exp = Fraction(89, 100)
    frac = q ** exp
    assert repr(frac) == "60.2559586074358 m^(89/100)"



