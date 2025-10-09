"""
Quantium: A modern Python library for dimensional analysis and unit-safe scientific computation.

Quantium provides tools for defining, manipulating, and validating physical quantities with
units, ensuring robust and expressive modeling across scientific and engineering workflows.
"""

from quantium.units.units_registry import get_unit

__version__ = "0.0.1a0"
__author__ = "Parneet Sidhu"
__license__ = "MIT"

__all__ = [
    "__version__", 
    "__author__", 
    "__license__",
    "get_unit"]
