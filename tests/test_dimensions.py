import pytest

# If your file lives at src/quantium/units/dimensions.py, this import will work:
from quantium.core.dimensions import (
    Dim,
    LENGTH, MASS, TIME, CURRENT, TEMPERATURE, AMOUNT, LUMINOUS, DIM_0,
    dim_mul, dim_div, dim_pow
)


# --- Basic structure & base vectors -------------------------------------------------

def test_base_vectors_shape_and_types():
    bases = [DIM_0, LENGTH, MASS, TIME, CURRENT, TEMPERATURE, AMOUNT, LUMINOUS]
    for b in bases:
        assert isinstance(b, tuple)
        assert len(b) == 7
        assert all(isinstance(x, int) for x in b)

def test_dimensional_basis():
    # Check each base unit has a single 1 in the right place and 0 elsewhere
    assert LENGTH     == (1,0,0,0,0,0,0)
    assert MASS     == (0,1,0,0,0,0,0)
    assert TIME     == (0,0,1,0,0,0,0)
    assert CURRENT     == (0,0,0,1,0,0,0)
    assert TEMPERATURE == (0,0,0,0,1,0,0)
    assert AMOUNT     == (0,0,0,0,0,1,0)
    assert LUMINOUS     == (0,0,0,0,0,0,1)
    assert DIM_0 == (0,0,0,0,0,0,0)

# --- Algebra: multiplication, division, power --------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    (LENGTH, dim_pow(TIME, -1), (1,0,-1,0,0,0,0)),  # LENGTH * TIME^{-1}  (speed)
    (LENGTH, LENGTH, (2,0,0,0,0,0,0)),
    (MASS, TIME, (0,1,1,0,0,0,0)),
    (DIM_0, LENGTH, LENGTH),                          # identity
    (DIM_0, DIM_0, DIM_0),
])
def test_dim_mul(a, b, expected):
    # vector add of exponents
    assert dim_mul(a, b) == tuple(x+y for x,y in zip(a,b))
    # commutativity
    assert dim_mul(a, b) == dim_mul(b, a)
    # selected expected checks
    if expected is not None:
        assert dim_mul(a, b) == expected

@pytest.mark.parametrize("a", [LENGTH, MASS, TIME, CURRENT, TEMPERATURE, AMOUNT, LUMINOUS, (2, -1, 3, 0, 0, 0, -4)])
def test_dim_div_and_identities(a: Dim):
    # a / a = DIM_0
    assert dim_div(a, a) == DIM_0
    # a / DIM_0 = a (subtracting zeros)
    assert dim_div(a, DIM_0) == a
    # DIM_0 / a = inverse exponents
    assert dim_div(DIM_0, a) == tuple(0 - x for x in a)

@pytest.mark.parametrize("a,n,expected", [
    (LENGTH, 0, DIM_0),
    (LENGTH, 1, LENGTH),
    (LENGTH, 2, (2,0,0,0,0,0,0)),
    (TIME, -2, (0,0,-2,0,0,0,0)),
    ((1, -1, 2, 0, 0, 3, -4), 3, (3, -3, 6, 0, 0, 9, -12)),
])
def test_dim_pow(a: Dim, n: int, expected: Dim):
    assert dim_pow(a, n) == expected

# --- Algebraic laws ----------------------------------------------------------------

@pytest.mark.parametrize("a,b,n", [
    (LENGTH, TIME, 3),
    (MASS, CURRENT, -2),
    ((1,0,-1,0,0,0,0), (0,1,0,0,0,0,0), 4),  # speed * mass, ^4
])
def test_power_distributes_over_mul(a: Dim, b: Dim, n: int):
    left  = dim_pow(dim_mul(a, b), n)
    right = dim_mul(dim_pow(a, n), dim_pow(b, n))
    assert left == right

@pytest.mark.parametrize("a,m,n", [
    (LENGTH, 2, 3),
    (TIME, -1, 5),
    ((1,0,-2,0,0,0,0), 4, -2),
])
def test_same_base_adds_exponents(a: Dim, m: int, n: int):
    left  = dim_mul(dim_pow(a, m), dim_pow(a, n))
    right = dim_pow(a, m + n)
    assert left == right

@pytest.mark.parametrize("a,m,n", [
    (MASS, 2, 3),
    (CURRENT, -2, 4),
    ((0,1,0,1,0,0,-1), 5, -3),
])
def test_power_of_power_multiplies_exponents(a: Dim, m: int, n: int):
    left  = dim_pow(dim_pow(a, m), n)
    right = dim_pow(a, m * n)
    assert left == right

# --- Physical sanity checks (examples from docstring) -------------------------------

def test_dimension_examples_from_docs():
    # meters: LENGTH
    assert LENGTH == (1,0,0,0,0,0,0)
    # seconds: TIME
    assert TIME == (0,0,1,0,0,0,0)
    # dimensionless
    assert DIM_0 == (0,0,0,0,0,0,0)
    # speed: LENGTH/TIME = LENGTH * TIME^{-1}
    speed = dim_mul(LENGTH, dim_pow(TIME, -1))
    assert speed == (1,0,-1,0,0,0,0)
    # acceleration: LENGTH/TIME^2
    accel = dim_mul(LENGTH, dim_pow(TIME, -2))
    assert accel == (1,0,-2,0,0,0,0)
    # force: (kg·m/s^2) -> MASS * LENGTH * TIME^{-2}
    force = dim_mul(MASS, dim_mul(LENGTH, dim_pow(TIME, -2)))
    assert force == (1,1,-2,0,0,0,0)

# --- Immutability / tuple semantics -------------------------------------------------

@pytest.mark.parametrize("op", ["mul", "div", "pow"])
def test_results_are_tuples_and_not_lists(op):
    if op == "mul":
        out = dim_mul(LENGTH, TIME)
    elif op == "div":
        out = dim_div(LENGTH, TIME)
    else:
        out = dim_pow(LENGTH, 3)
    assert isinstance(out, tuple)
    assert not isinstance(out, list)
