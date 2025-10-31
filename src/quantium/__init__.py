"""
Quantium: A modern Python library for dimensional analysis and unit-safe scientific computation.

Quantium provides tools for defining, manipulating, and validating physical quantities with
units, ensuring robust and expressive modeling across scientific and engineering workflows.
This module exposes a minimal, stable public API. Heavy subsystems (e.g. the units registry)
are imported lazily to avoid import-time side effects and circular imports.
"""

from importlib import metadata as _metadata

__author__ = "Parneet Sidhu"
__license__ = "MIT"
__version__ = _metadata.version("quantium")

__all__ = ["__version__", "__author__", "__license__"]
