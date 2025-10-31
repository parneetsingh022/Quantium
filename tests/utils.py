# tests/utils.py
from quantium.catalog.registry import DEFAULT_REGISTRY as _ureg

def _nop_prettifier(monkeypatch):
    import quantium.core.utils as utils
    monkeypatch.setattr(utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True)

def _name(sym: str, n: int) -> str:
    return "1" if n == 0 else (sym if n == 1 else f"{sym}^{n}")