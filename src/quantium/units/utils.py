# quantium/units/utils.py
from quantium.units.dimensions import Dim

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