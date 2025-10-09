import math
import pytest
from dataclasses import FrozenInstanceError

from quantium.core.dimensions import (
    DIM_0, L, T, dim_mul, dim_div, dim_pow
)
from quantium.core.quantity import Unit, Quantity


# -------------------------------
# Unit: construction & validation
# -------------------------------

def test_unit_valid():
    m = Unit("m", 1.0, L)
    assert m.name == "m"
    assert m.scale_to_si == 1.0
    assert m.dim == L

def test_unit_invalid_dim_length():
    with pytest.raises(ValueError):
        Unit("bad", 1.0, (1, 0, 0))  # not 7-tuple

@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_unit_invalid_scale(scale):
    with pytest.raises(ValueError):
        Unit("x", scale, L)

def test_unit_is_frozen_and_slotted():
    m = Unit("m", 1.0, L)

    # frozen => normal assignment raises FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        m.name = "meter"

    # slots => adding a new attribute should fail (AttributeError or TypeError depending on Python)
    with pytest.raises((AttributeError, TypeError)):
        m.some_new_attr = 42


# -------------------------------
# Quantity: basics & conversion
# -------------------------------

def test_quantity_construct_and_to():
    m  = Unit("m", 1.0, L)
    cm = Unit("cm", 0.01, L)

    q_cm = Quantity(200, cm)          # 200 cm
    q_m  = q_cm.to(m)                  # -> 2 m

    assert isinstance(q_m, Quantity)
    assert q_m.unit is m
    assert q_m.dim == L
    # _mag_si is internal, so check using units:
    assert math.isclose(q_m._mag_si, 2.0)  # 2 m in SI
    # magnitude shown in the *current* unit:
    assert math.isclose(q_m._mag_si / q_m.unit.scale_to_si, 2.0)

def test_quantity_to_dimension_mismatch_raises():
    m = Unit("m", 1.0, L)
    s = Unit("s", 1.0, T)
    q = Quantity(3, m)
    with pytest.raises(TypeError):
        q.to(s)


# -------------------------------
# __rmatmul__: value @ Unit
# -------------------------------

def test_rmatmul_operator():
    m = Unit("m", 1.0, L)
    q = 3 @ m
    assert isinstance(q, Quantity)
    assert q.dim == L
    assert q.unit is m
    assert math.isclose(q._mag_si, 3.0)


# -------------------------------
# Arithmetic: +, -, *, /, **, scalars
# -------------------------------

def test_add_and_sub_same_dim():
    m = Unit("m", 1.0, L)
    cm = Unit("cm", 0.01, L)
    q1 = 1 @ m
    q2 = 50 @ cm  # 0.5 m

    s = q1 + q2   # left unit ("m") retained
    d = q1 - q2

    assert s.unit is m and d.unit is m
    assert math.isclose(s._mag_si / s.unit.scale_to_si, 1.5)
    assert math.isclose(d._mag_si / d.unit.scale_to_si, 0.5)

def test_add_dim_mismatch_raises():
    m = Unit("m", 1.0, L)
    s = Unit("s", 1.0, T)
    with pytest.raises(TypeError):
        _ = (1 @ m) + (1 @ s)

def test_scalar_multiplication_and_division():
    m = Unit("m", 1.0, L)
    q = 2 @ m

    q2 = q * 3
    q3 = 3 * q
    q4 = q / 2

    assert q2.dim == L and q3.dim == L and q4.dim == L
    assert math.isclose(q2._mag_si / q2.unit.scale_to_si, 6.0)
    assert math.isclose(q3._mag_si / q3.unit.scale_to_si, 6.0)
    assert math.isclose(q4._mag_si / q4.unit.scale_to_si, 1.0)

def test_quantity_times_quantity():
    m = Unit("m", 1.0, L)
    s = Unit("s", 1.0, T)
    q = (2 @ m) * (3 @ s)  # -> 6 m·s

    assert q.dim == dim_mul(L, T)
    assert q.unit.name == "m·s"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 6.0)

def test_quantity_div_quantity():
    m = Unit("m", 1.0, L)
    s = Unit("s", 1.0, T)
    q = (10 @ m) / (2 @ s)  # -> 5 m/s

    assert q.dim == dim_div(L, T)
    assert q.unit.name == "m/s"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 5.0)

def test_scalar_divided_by_quantity():
    m = Unit("m", 1.0, L)
    q = 2 / (2 @ m)  # -> 1 (1/m)

    assert q.dim == dim_div(DIM_0, L)
    assert q.unit.name == "1/m"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 1.0)

def test_power_of_quantity():
    m = Unit("m", 1.0, L)
    q2 = (2 @ m) ** 2  # -> 4 m^2

    assert q2.dim == dim_pow(L, 2)
    assert q2.unit.name == "m^2"
    assert math.isclose(q2._mag_si / q2.unit.scale_to_si, 4.0)


# -------------------------------
# to_si(): preferred symbol & fallback
# -------------------------------

def test_to_si_uses_preferred_symbol_when_available(monkeypatch):
    # Arrange: make preferred_symbol_for_dim return a symbol for L
    from quantium import core as _core
    # Monkeypatch utils functions that to_si imports locally
    def fake_preferred(dim):
        return "m" if dim == L else None
    def fake_format(dim):
        return "L?"  # should not be used in this test

    monkeypatch.setattr(_core.utils, "preferred_symbol_for_dim", fake_preferred, raising=True)
    monkeypatch.setattr(_core.utils, "format_dim", fake_format, raising=True)

    cm = Unit("cm", 0.01, L)
    q_si = (123 @ cm).to_si()

    assert isinstance(q_si, Quantity)
    assert q_si.unit.name == "m"         # preferred symbol chosen
    assert q_si.unit.scale_to_si == 1.0  # SI unit
    # magnitudes in SI should match _mag_si:
    assert math.isclose(q_si._mag_si, 1.23)

def test_to_si_fallbacks_to_formatted_dim_when_no_symbol(monkeypatch):
    from quantium import core as _core
    def fake_preferred(dim):
        return None  # force fallback
    def fake_format(dim):
        # For L/T, produce a composed name:
        return "m/s" if dim == dim_div(L, T) else "1"

    monkeypatch.setattr(_core.utils, "preferred_symbol_for_dim", fake_preferred, raising=True)
    monkeypatch.setattr(_core.utils, "format_dim", fake_format, raising=True)

    m = Unit("m", 1.0, L)
    s = Unit("s", 1.0, T)
    q = ((5 @ m) / (2 @ s)).to_si()

    assert q.unit.name == "m/s"          # composed SI name from format_dim
    assert q.unit.scale_to_si == 1.0
    assert math.isclose(q._mag_si, 2.5)


# -------------------------------
# __repr__: pretty printing
# -------------------------------

def test_repr_keeps_non_si_unit_name(monkeypatch):
    # Ensure repr uses the current unit name ("cm") and does not replace with a symbol
    from quantium import core as _core

    # Make prettifier a no-op passthrough so we can assert on exact string.
    monkeypatch.setattr(_core.utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True)
    # Ensure it would *try* to upgrade only when scale_to_si == 1.0; here it's 0.01, so no upgrade.
    monkeypatch.setattr(_core.utils, "preferred_symbol_for_dim", lambda d: "m", raising=True)

    cm = Unit("cm", 0.01, L)
    q = 2 @ cm
    assert repr(q) == "2 cm"

def test_repr_upgrades_to_preferred_symbol_when_scale_is_1(monkeypatch):
    from quantium import core as _core
    # prettifier just returns what it's given
    monkeypatch.setattr(
        _core.utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True
    )
    # preferred symbol for L is "m"
    monkeypatch.setattr(_core.utils, "preferred_symbol_for_dim", lambda d: "m" if d == L else None, raising=True)

    m = Unit("m", 1.0, L)
    q = 3 @ m
    # scale_to_si == 1.0 -> allowed to upgrade pretty name to "m"
    assert repr(q) == "3 m"
