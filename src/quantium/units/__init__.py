from quantium.units.registry import UnitsRegistry
from typing import Any

# Lazy access helpers -------------------------------------------------------

def _get_default_registry() -> UnitsRegistry:
    # Import here to avoid import-time side-effects / circular imports.
    from quantium.units.registry import DEFAULT_REGISTRY  # local import
    return DEFAULT_REGISTRY

def __getattr__(name: str) -> Any:
    """
    Lazy attribute access. Accessing 'u' will construct a namespace from the
    package's default registry on first use.
    """
    if name == "u":
        return _get_default_registry().as_namespace()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list[str]:
    # Improve discoverability in REPL / autocomplete.
    return sorted(list(globals().keys()) + ["u"])