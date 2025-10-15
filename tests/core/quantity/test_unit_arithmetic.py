import math
import pytest

from quantium.core.dimensions import (
    DIM_0,
    TEMPERATURE,
    LENGTH,
    TIME,
    dim_mul,
    dim_div,
    dim_pow
)
from quantium.core.quantity import Unit


# -------------------------------
# Multiplication: equivalent units collapse to a power
# -------------------------------

def test_unit_mul_equivalent_names_collapses_to_power_prefers_lhs_name():
    """K * kelvin (same dim, same scale) collapses to a squared head, not 'K·kelvin'."""
    K = Unit("K", 1.0, TEMPERATURE)
    kelvin = Unit("kelvin", 1.0, TEMPERATURE)

    out1 = K * kelvin
    out2 = kelvin * K

    # Dimension = Θ^2
    assert out1.dim == dim_mul(TEMPERATURE, TEMPERATURE)
    assert out2.dim == dim_mul(TEMPERATURE, TEMPERATURE)

    # Scale multiplies (1*1=1)
    assert math.isclose(out1.scale_to_si, 1.0)
    assert math.isclose(out2.scale_to_si, 1.0)

    # Name should be a power, not a dotted product. We accept either base symbol,
    # depending on implementation preference.
    valid_names = {"K^2", "kelvin^2"}
    assert out1.name in valid_names
    assert out2.name in valid_names
    # Ensure no lingering '·' when units are equivalent
    assert "·" not in out1.name and "·" not in out2.name


def test_unit_mul_non_equivalent_names_composes_normally():
    """m * s composes with a dot and correct scale & dimension."""
    m = Unit("m", 1.0, LENGTH)
    s = Unit("s", 1.0, TIME)

    out = m * s

    assert out.dim == dim_mul(LENGTH, TIME)
    assert math.isclose(out.scale_to_si, 1.0)
    # exact order can depend on implementation; allow either
    assert out.name in ("m·s", "s·m")


# -------------------------------
# Division: equivalent units → dimensionless
# -------------------------------

def test_unit_div_equivalent_is_dimensionless_even_if_name_not_normalized():
    """K / kelvin is DIM_0 with scale 1; name may or may not be normalized."""
    K = Unit("K", 1.0, TEMPERATURE)
    kelvin = Unit("kelvin", 1.0, TEMPERATURE)

    out1 = K / kelvin
    out2 = kelvin / K

    assert out1.dim == DIM_0
    assert out2.dim == DIM_0
    assert math.isclose(out1.scale_to_si, 1.0)
    assert math.isclose(out2.scale_to_si, 1.0)

    # Current Unit.__truediv__ may leave a composed name (e.g., "K/kelvin").
    # Accept either normalized or raw composed names.
    acceptable = {"", "1", "K/kelvin", "kelvin/K"}
    assert out1.name in acceptable
    assert out2.name in acceptable


def test_unit_div_non_equivalent_builds_fraction_name_and_scale():
    """cm / m should retain correct dimension and scale (0.01) and a composed name."""
    cm = Unit("cm", 0.01, LENGTH)
    m = Unit("m", 1.0, LENGTH)

    out = cm / m

    assert out.dim == DIM_0  # same dimension -> cancels
    assert math.isclose(out.scale_to_si, 0.01)  # 0.01 / 1
    # Name depends on implementation; accept either direction
    assert out.name in ("cm/m", "m^-1·cm") or "/" in out.name or "^-1" in out.name


# -------------------------------
# Powers and reciprocals: name normalization
# -------------------------------

def test_unit_pow_name_normalization_and_scale():
    """Check x^0 -> '1', x^1 -> 'x', x^2 -> 'x^2' with correct scales/dims."""
    s = Unit("s", 1.0, TIME)

    s0 = s ** 0
    s1 = s ** 1
    s2 = s ** 2
    s_1 = s ** -1

    # dims
    assert s0.dim == dim_pow(TIME, 0) == DIM_0
    assert s1.dim == dim_pow(TIME, 1) == TIME
    assert s2.dim == dim_pow(TIME, 2)
    assert s_1.dim == dim_pow(TIME, -1) == dim_div(DIM_0, TIME)

    # scales
    assert math.isclose(s0.scale_to_si, 1.0)
    assert math.isclose(s1.scale_to_si, 1.0)
    assert math.isclose(s2.scale_to_si, 1.0)
    assert math.isclose(s_1.scale_to_si, 1.0)

    # names (using ^, not unicode superscript, per implementation)
    assert s0.name in ("1", "s^0")
    assert s1.name == "s"
    assert s2.name == "s^2"
    assert s_1.name == "s^-1"



def test_unit_reciprocal_with_rtruediv_normalizes_power():
    """1 / (s^-3) -> s^3, and 1 / s -> s^-1."""
    # Build s^-3 directly using dim_pow for clarity
    s_neg3 = Unit("s^-3", 1.0, dim_pow(TIME, -3))

    one_over_sneg3 = 1 / s_neg3
    s = Unit("s", 1.0, TIME)
    one_over_s = 1 / s

    # (T^-(-3)) = T^3
    assert one_over_sneg3.dim == dim_pow(TIME, 3)
    assert one_over_sneg3.name == "s^3"
    assert math.isclose(one_over_sneg3.scale_to_si, 1.0)

    # (1 / s) = s^-1
    assert one_over_s.dim == dim_pow(TIME, -1) == dim_div(DIM_0, TIME)
    assert one_over_s.name == "s^-1"
    assert math.isclose(one_over_s.scale_to_si, 1.0)


def test_kelvin_equivalent_units_mul_div_pretty_output():
    # Use the registry units so we match your canonical K definition.
    from quantium.units.registry import DEFAULT_REGISTRY as dreg
    kelvin = dreg.get("K")

    # Rename the left operand to "kelvin" to exercise name-collapsing preference.
    k_named = kelvin.as_name("kelvin")

    q1 = (100 @ k_named) * (10 @ dreg.get("K"))
    q2 = (10 @ dreg.get("K")) * (100 @ k_named)
    q3 = (100 @ k_named) / (10 @ dreg.get("K"))
    q4 = (10 @ dreg.get("K")) / (100 @ k_named)

    # Exact repr/print expectations (note: uses the superscript ² character)
    assert repr(q1) == "1000 kelvin²"
    assert repr(q2) == "1000 K²"
    assert repr(q3) == "10"
    assert repr(q4) == "0.1"
    

# -------------------------------
# Smoke: rmatmul creates a Quantity with the right unit
# -------------------------------

def test_unit_rmatmul_creates_quantity_with_unit():
    m = Unit("m", 1.0, LENGTH)
    q = 5 @ m
    from quantium.core.quantity import Quantity
    assert isinstance(q, Quantity)
    assert q.unit is m
    assert math.isclose(q._mag_si, 5.0)
