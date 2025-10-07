"""
quantium.units.utils
====================

Utility functions for formatting and displaying physical dimensions and units
within the Quantium framework.

This module provides helper functions for representing dimensional exponents
and unit strings in a readable scientific format (e.g., 'kg·m/s²').

Functions
---------
_sup(n: int) -> str
    Converts an integer exponent into its Unicode superscript representation.
    Used for displaying powers in formatted unit strings (e.g., 'm²' or 's⁻¹').

format_dim(dim: Dim) -> str
    Converts a dimension tuple `(L, M, T, I, Θ, N, J)` into a conventional
    formatted string like `'kg·m/s²'`, following standard SI notation.

"""

from quantium.core.dimensions import Dim

_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
def _sup(n: int) -> str:
    return ("" if n == 1 else str(n).translate(_SUPERSCRIPTS))

def format_dim(dim: Dim) -> str:
    """
    Turn a dimension tuple (L,M,T,I,Θ,N,J) into 'kg·m/s²' style.
    Conventional order: M, L, T, I, Θ, N, J.
    """
    # Map indices in (L,M,T,I,Θ,N,J) to conventional labels
    # index:   0   1   2   3    4   5   6
    labels = ["m", "kg", "s", "A", "K", "mol", "cd"]
    order  = [1,   0,    3,   2,   4,   5,    6]  # M, L, T, I, Θ, N, J

    num, den = [], []
    for i in order:
        e = dim[i]
        if e > 0:
            num.append(labels[i] + _sup(e))
        elif e < 0:
            den.append(labels[i] + _sup(-e))

    numerator = "·".join(num) if num else "1"
    denominator = "·".join(den)
    return f"{numerator}/{denominator}" if denominator else numerator