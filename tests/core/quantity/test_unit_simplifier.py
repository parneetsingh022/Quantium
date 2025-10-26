import pytest

from quantium.core.dimensions import DIM_0, dim_mul, dim_pow
from quantium.core.quantity import Unit
from quantium.core.unit_simplifier import UnitNameSimplifier
from quantium.units.registry import DEFAULT_REGISTRY


@pytest.fixture(scope="module")
def simplifier() -> UnitNameSimplifier:
    return UnitNameSimplifier(Unit)


def test_canonical_unit_for_dim_prefers_registered_symbol(simplifier: UnitNameSimplifier):
    newton = DEFAULT_REGISTRY.get("N")
    canonical = simplifier.canonical_unit_for_dim(newton.dim)
    assert canonical.name == "N"
    assert canonical.scale_to_si == pytest.approx(1.0)
    assert canonical.dim == newton.dim


def test_canonical_unit_for_dim_detects_preferred_powers(simplifier: UnitNameSimplifier):
    joule = DEFAULT_REGISTRY.get("J")
    squared_dim = dim_pow(joule.dim, 2)
    canonical = simplifier.canonical_unit_for_dim(squared_dim)
    assert canonical.name == "J^2"
    assert canonical.scale_to_si == pytest.approx(1.0)
    assert canonical.dim == squared_dim


def test_normalize_power_name():
    normalize = UnitNameSimplifier.normalize_power_name
    assert normalize("m^1") == "m"
    assert normalize("m^0") == "1"
    assert normalize("m^-2") == "m^-2"
    assert normalize("Pa") == "Pa"


def test_unit_symbol_map_records_priority_and_index(simplifier: UnitNameSimplifier):
    metre = DEFAULT_REGISTRY.get("m")
    mapping = simplifier.unit_symbol_map(metre, priority=2)
    assert mapping == {"m": (1, (2, 0))}


def test_scale_symbol_map_scales_and_drops_zero(simplifier: UnitNameSimplifier):
    mapping = {"m": (1, (0, 0)), "s": (0, (1, 0))}
    scaled = simplifier.scale_symbol_map(mapping, 2)
    assert scaled["m"] == (2, (0, 0))
    assert "s" not in scaled


def test_combine_symbol_maps_merges_and_cancels(simplifier: UnitNameSimplifier):
    map_a = {"m": (1, (0, 0))}
    map_b = {"m": (-1, (1, 0)), "s": (2, (1, 1))}
    combined = simplifier.combine_symbol_maps(map_a, map_b)
    assert "m" not in combined
    assert combined["s"] == (2, (1, 1))


def test_format_unit_components_handles_fraction(simplifier: UnitNameSimplifier):
    components = {"m": (1, (0, 0)), "s": (-2, (1, 0))}
    assert simplifier.format_unit_components(components) == "m/s^2"


def test_unit_from_components_rebuilds_registered_units(simplifier: UnitNameSimplifier):
    newton = DEFAULT_REGISTRY.get("N")
    metre = DEFAULT_REGISTRY.get("m")
    components = simplifier.combine_symbol_maps(
        simplifier.unit_symbol_map(newton, 0),
        simplifier.unit_symbol_map(metre, 1),
    )
    composite = simplifier._unit_from_components(components)
    expected = newton * metre
    assert composite == expected


def test_si_to_value_unit_dimensionless_returns_identity(simplifier: UnitNameSimplifier):
    value, unit = simplifier.si_to_value_unit(5.0, DIM_0)
    assert value == pytest.approx(5.0)
    assert unit.name == ""
    assert unit.dim == DIM_0


def test_si_to_value_unit_preserves_requested_simple_unit(simplifier: UnitNameSimplifier):
    centimetre = DEFAULT_REGISTRY.get("cm")
    components = simplifier.unit_symbol_map(centimetre)
    value, unit = simplifier.si_to_value_unit(0.05, centimetre.dim, components, centimetre)
    assert value == pytest.approx(5.0)
    assert unit.name == "cm"


def test_si_to_value_unit_prefers_prefix_for_readable_values(simplifier: UnitNameSimplifier):
    metre = DEFAULT_REGISTRY.get("m")
    components = simplifier.unit_symbol_map(metre)
    value, unit = simplifier.si_to_value_unit(0.001, metre.dim, components)
    assert unit.name == "mm"
    assert value == pytest.approx(1.0)


def test_si_to_value_unit_prefers_highest_exponent_component(simplifier: UnitNameSimplifier):
    newton = DEFAULT_REGISTRY.get("N")
    kilonewton = DEFAULT_REGISTRY.get("kN")
    kilonewton_sq = kilonewton ** 2

    components = simplifier.combine_symbol_maps(
        simplifier.unit_symbol_map(newton, 0),
        simplifier.unit_symbol_map(kilonewton_sq, 1),
    )

    combined_dim = dim_mul(newton.dim, kilonewton_sq.dim)
    combined_scale = newton.scale_to_si * kilonewton_sq.scale_to_si

    value, unit = simplifier.si_to_value_unit(combined_scale, combined_dim, components)

    expected_unit = kilonewton ** 3
    assert unit.name == expected_unit.name
    assert unit.dim == expected_unit.dim
    assert value == pytest.approx(combined_scale / expected_unit.scale_to_si)