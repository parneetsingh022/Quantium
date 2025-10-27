import pytest

# If your file lives at src/quantium/units/dimensions.py, this import will work:
from quantium.core.dimensions import (
    AMOUNT,
    CURRENT,
    DIM_0,
    LENGTH,
    LUMINOUS,
    MASS,
    TEMPERATURE,
    TIME,
    Dim,
    IrrationalExponentError,
    dim_div,
    dim_mul,
    dim_pow,
    Dimension
)

from fractions import Fraction
import math

# --- Basic structure & base vectors -------------------------------------------------

def test_base_vectors_shape_and_types():
    bases = [DIM_0, LENGTH, MASS, TIME, CURRENT, TEMPERATURE, AMOUNT, LUMINOUS]
    for b in bases:
        assert isinstance(b, tuple)
        assert len(b) == 7
        assert all(isinstance(x, Fraction) for x in b)

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
    assert dim_mul(a, b) == tuple(x+y for x,y in zip(a,b, strict=False))
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



def test_public_constants_are_dimension_and_tuple():
    for c in [DIM_0, LENGTH, MASS, TIME, CURRENT, TEMPERATURE, AMOUNT, LUMINOUS]:
        assert isinstance(c, tuple)        # legacy expectation
        assert isinstance(c, Dimension)    # new object type


def test_equality_with_plain_tuple_and_hashing():
    t = (1, 0, -1, 0, 0, 0, 0)
    d = Dimension(t)
    assert d == t
    assert hash(d) == hash(t)

def test_dimension_as_dict_key():
    d1 = Dimension((1,0,0,0,0,0,0))
    d2 = (1,0,0,0,0,0,0)  # plain tuple
    m = {d1: "ok"}
    # Because equality & hash match, inserting the tuple should overwrite
    m[d2] = "overwritten"
    assert m[d1] == "overwritten"


def test_fractional_hash_matches_fraction_tuples():
    fractional = (
        Fraction(1, 2),
        Fraction(0, 1),
        Fraction(-3, 2),
        Fraction(0, 1),
        Fraction(5, 3),
        Fraction(0, 1),
        Fraction(0, 1),
    )
    d = Dimension(fractional)
    assert d == fractional
    assert hash(d) == hash(fractional)


def test_fractional_dimension_as_dict_key():
    fractional = (
        Fraction(2, 5),
        Fraction(0, 1),
        Fraction(1, 3),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(-4, 7),
        Fraction(0, 1),
    )
    d = Dimension(fractional)
    store = {d.as_key(): "dimension"}
    store[Dimension(fractional).as_key()] = "tuple"
    assert store[d.as_key()] == "tuple"


def test_pow_rejects_irrational_exponents():
    d = LENGTH * (TIME ** -1)
    with pytest.raises(IrrationalExponentError):
        _ = d ** (2 ** 0.5)
        _ = d ** math.pi

def test_operator_mul_div_pow_equivalence():
    left = (LENGTH / TIME) * MASS
    right = dim_mul(dim_div(LENGTH, TIME), MASS)
    assert left == right

def test_power_operator_equivalence():
    assert (LENGTH ** 2) == dim_pow(LENGTH, 2)
    assert (TIME ** -2) == dim_pow(TIME, -2)

def test_tuple_times_dimension_raises():
    t = (0, 1, 0, 0, 0, 0, 0)
    with pytest.raises(TypeError):
        _ = t * LENGTH  # tuple tries to "repeat", not dimensional multiply

def test_dimension_constructs_from_iterable_and_coerces_ints():
    d = Dimension([1.0, 0, -1, 0, 0, 0, 0])  # floats coerced to int
    assert d == (1, 0, -1, 0, 0, 0, 0)

def test_dimension_rejects_wrong_length():
    with pytest.raises(ValueError):
        Dimension((1,2,3))  # not 7

def test_allow_non_int_exponent_for_pow():
    d = Dimension((1,0,0,0,0,0,0))
    _ = d ** 2.5

def test_operator_algebraic_laws():
    # (ab)^n = a^n b^n
    a, b, n = LENGTH, TIME, 3
    assert ((a * b) ** n) == ((a ** n) * (b ** n))

    # a^m a^n = a^(m+n)
    m, n = 2, -5
    assert ((a ** m) * (a ** n)) == (a ** (m+n))

    # (a^m)^n = a^(mn)
    m, n = 4, -2
    assert ((a ** m) ** n) == (a ** (m*n))

def test_shim_functions_return_dimension_instances():
    out1 = dim_mul(LENGTH, TIME)
    out2 = dim_div(LENGTH, TIME)
    out3 = dim_pow(LENGTH, 3)
    for out in (out1, out2, out3):
        assert isinstance(out, tuple)
        assert isinstance(out, Dimension)


@pytest.mark.parametrize("dim, expected", [
    (DIM_0, ""),                      # dimensionless -> no parts
    (LENGTH, "[L^1]"),
    (TIME, "[T^1]"),
    (MASS, "[M^1]"),
    (CURRENT, "[I^1]"),
    (TEMPERATURE, "[Θ^1]"),
    (AMOUNT, "[N^1]"),
    (LUMINOUS, "[J^1]"),
])
def test_repr_single_bases(dim, expected):
    assert repr(dim) == expected

def test_repr_negative_and_multi_components():
    # Construct: L^1 * M^1 * T^-2  -> "[L^1][M^1][T^-2]" (order L, M, T, I, Θ, N, J)
    force_like = dim_mul(MASS, dim_mul(LENGTH, dim_pow(TIME, -2)))
    assert isinstance(force_like, Dimension)
    assert repr(force_like) == "[L^1][M^1][T^-2]"

def test_repr_omits_zero_components_and_has_no_spaces():
    # (1, 0, -1, 0, 0, 3, -4) -> "[L^1][T^-1][N^3][J^-4]"
    d = Dimension((1, 0, -1, 0, 0, 3, -4))
    s = repr(d)
    assert s == "[L^1][T^-1][N^3][J^-4]"
    assert " " not in s

def test_repr_deterministic_and_ordered():
    # Same dimensional values, different construction order must give same repr
    a = LENGTH * MASS * (TIME ** -2)
    b = MASS * (TIME ** -2) * LENGTH
    assert repr(a) == "[L^1][M^1][T^-2]"
    assert repr(a) == repr(b)

# --- Helpers: as_tuple, is_dimensionless ------------------------------------

def test_as_tuple_returns_plain_tuple_not_dimension():
    d = LENGTH / TIME
    t = d.as_tuple()
    assert isinstance(t, tuple)
    assert not isinstance(t, Dimension)
    assert t == (1, 0, -1, 0, 0, 0, 0)

def test_as_tuple_round_trip_equality():
    d = Dimension((1, 0, -1, 0, 0, 0, 0))
    d2 = Dimension(d.as_tuple())
    assert d2 == d
    assert hash(d2) == hash(d)  # should behave like tuples

def test_is_dimensionless_true_for_dim0_false_for_others():
    assert DIM_0.is_dimensionless is True
    assert LENGTH.is_dimensionless is False
    assert (LENGTH / LENGTH).is_dimensionless is True
    assert (TIME ** 0).is_dimensionless is True

def test_is_dimensionless_composite_zero_vector():
    d = Dimension((0, 0, 0, 0, 0, 0, 0))
    assert d.is_dimensionless

# --- Sanity around Unicode Theta -------------------------------------------

def test_repr_includes_unicode_theta_for_temperature():
    # Ensure the character Θ is present when temperature exponent is non-zero
    s = repr(TEMPERATURE ** 2)
    assert s == "[Θ^2]"

# --- Extra guardrails -------------------------------------------------------

def test_repr_no_leading_or_trailing_whitespace():
    s = repr(LENGTH * (TIME ** -2))
    assert s == s.strip()

def test_repr_handles_all_zeros_cleanly():
    # Explicitly verify empty string is the chosen representation for DIM_0
    assert repr(DIM_0) == ""


def test_reverse_ops_iterable_on_left_supported_paths():
    t = (0, 1, 0, 0, 0, 0, 0)  # MASS

    # Multiplication must have Dimension on the left to avoid tuple repetition semantics
    assert Dimension(t) * LENGTH == (1, 1, 0, 0, 0, 0, 0)

    # Division can work either way if __rtruediv__ is implemented on Dimension
    assert t / LENGTH == (-1, 1, 0, 0, 0, 0, 0)   # uses Dimension.__rtruediv__
    assert LENGTH / t == (1, -1, 0, 0, 0, 0, 0)   # normal __truediv__

def test_operations_with_integers_fail_correctly():
    # 1. Test (int * Dimension)
    # This should fail because Dimension.__rmul__ returns NotImplemented
    # to prevent falling back to tuple repetition semantics.
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = 2 * LENGTH

    # 2. Test (Dimension * int)
    # This should fail because Dimension.__mul__ expects a DimLike object.
    # It will try to call Dimension(2), which fails because 'int' is not iterable.
    with pytest.raises(TypeError, match="Dimension requires an iterable"):
        _ = LENGTH * 2

    # 3. Test (Dimension / int)
    # This should fail because Dimension.__truediv__ expects DimLike.
    with pytest.raises(TypeError, match="Dimension requires an iterable"):
        _ = LENGTH / 2

    # 4. Test (int / Dimension)
    # This should fail because Dimension.__rtruediv__ expects DimLike.
    with pytest.raises(TypeError, match="Dimension requires an iterable"):
        _ = 2 / LENGTH

    # Sanity check: The *only* valid algebraic operation with an int is __pow__
    assert LENGTH ** 2 == (2, 0, 0, 0, 0, 0, 0)
    assert isinstance(LENGTH ** 2, Dimension)

def test_invalid_operations_fail():
    # 1. Test (int * Dimension)
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = 2 * LENGTH

    # 2. Test (Dimension * int)
    with pytest.raises(TypeError, match="Dimension requires an iterable"):
        _ = LENGTH * 2

    # 3. Test (Dimension / int)
    with pytest.raises(TypeError, match="Dimension requires an iterable"):
        _ = LENGTH / 2

    # 4. Test (int / Dimension)
    with pytest.raises(TypeError, match="Dimension requires an iterable"):
        _ = 2 / LENGTH

    # Sanity check: The *only* valid algebraic operation with an int is __pow__
    assert LENGTH ** 2 == (2, 0, 0, 0, 0, 0, 0)
    assert isinstance(LENGTH ** 2, Dimension)
    
    # 5. Test (Dimension + Dimension)
    # This should fail because we block tuple concatenation.
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = LENGTH + MASS
    
    # 6. Test (Dimension - Dimension)
    # This fails because tuple.__sub__ is not implemented.
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = LENGTH - MASS

    # 7. Test (Dimension - tuple)
    # This also fails.
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = LENGTH - (1, 0, 0, 0, 0, 0, 0)