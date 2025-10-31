# tests/conftest.py
import pytest
from quantium.units.registry import DEFAULT_REGISTRY as _ureg



@pytest.fixture(scope="session")
def ureg():
    return _ureg

@pytest.fixture
def nop_prettifier(monkeypatch):
    import quantium.core.utils as utils
    monkeypatch.setattr(utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True)


