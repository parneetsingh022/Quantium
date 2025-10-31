import importlib
import types
import pytest

import quantium.units.registry as regmod
from quantium.units.registry import _bootstrap_default_registry


@pytest.fixture()
def fresh_registry():
    return _bootstrap_default_registry()


def test__get_default_registry_returns_DEFAULT(monkeypatch, fresh_registry):
    # Patch the DEFAULT_REGISTRY and verify the helper returns it
    monkeypatch.setattr(regmod, "DEFAULT_REGISTRY", fresh_registry, raising=True)

    # Import the private helper from the package module
    import quantium.units as units
    get_default = getattr(units, "_get_default_registry")

    assert get_default() is fresh_registry


def test_lazy_u_binds_to_default_registry(monkeypatch, fresh_registry):
    # When DEFAULT_REGISTRY is patched, accessing `quantium.units.u` should
    # lazily resolve to a UnitNamespace bound to that registry.
    monkeypatch.setattr(regmod, "DEFAULT_REGISTRY", fresh_registry, raising=True)

    from quantium.units import u
    # UnitNamespace has an internal _reg reference to the registry
    assert hasattr(u, "_reg")
    assert u._reg is fresh_registry

    # Sanity: attribute access flows through to the registry
    assert u.m is fresh_registry.get("m")


def test_unknown_module_attribute_raises_attributeerror():
    import quantium.units as units
    with pytest.raises(AttributeError):
        _ = getattr(units, "definitely_not_a_public_attr")


def test_dir_includes_u():
    import quantium.units as units
    names = dir(units)
    assert "u" in names
    # Should be sorted for better discoverability
    assert names == sorted(names)
