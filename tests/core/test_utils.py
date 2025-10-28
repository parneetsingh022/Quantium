import pytest
import math
from fractions import Fraction

# Target module
import quantium.core.utils as utils


# Dimension helpers
from quantium.core.dimensions import (
    DIM_0, LENGTH, MASS, TIME, CURRENT, TEMPERATURE, AMOUNT, LUMINOUS,
    dim_mul, dim_div, dim_pow
)

# Registry (used only to ensure preferred map uses real units)
from quantium.units.registry import DEFAULT_REGISTRY as ureg


# -------------------------------
# _sup
# -------------------------------

@pytest.mark.parametrize("n, expected", [
    (1, ""),         # 1 -> empty
    (2, "²"),
    (3, "³"),
    (10, "¹⁰"),
    (-1, "⁻¹"),      # superscript minus + superscript ONE
    (-3, "⁻³"),      # superscript minus + superscript THREE
])
def test__sup(n, expected):
    assert utils._sup(n) == expected



# -------------------------------
# _parse_exponent via _TOKEN_RE
# -------------------------------

def test__parse_exponent():
    # Shorthands for exact superscript chars:
    SUP2 = "\N{SUPERSCRIPT TWO}"          # U+00B2
    SUP3 = "\N{SUPERSCRIPT THREE}"        # U+00B3
    SUP_MINUS = "\N{SUPERSCRIPT MINUS}"   # U+207B

    # Sanity: plain, caret, and ^( ) and ^sup( ) all work
    for token, expected in [
        ("m", 1),
        ("m^3", 3),
        ("m^(4)", 4),
        ("m^sup(5)", 5),
        ("m" + SUP2, 2),                 # "m²" with explicit codepoint
        ("s" + SUP_MINUS + SUP3, -3),    # "s⁻³" with explicit codepoints
    ]:
        m = utils._TOKEN_RE.fullmatch(token)
        assert m, f"Pattern did not match token {token!r}"
        assert utils._parse_exponent(m) == expected



# -------------------------------
# _tokenize_name_merge
# -------------------------------

@pytest.mark.parametrize("name, expected", [
    ("m", {"m": 1}),
    ("s^2", {"s": 2}),
    ("kg*m/s^2", {"kg": 1, "m": 1, "s": -2}),     # ASCII '*' normalized to '·'
    ("cm/ms^3·ms", {"cm": 1, "ms": -2}),          # (cm / ms^3) · ms -> cm / ms^2
    ("m·s/s", {"m": 1}),                          # cancellation
    ("1", {}),                                    # dimensionless token
    ("", {}),                                     # empty
])
def test__tokenize_name_merge(name, expected):
    assert utils._tokenize_name_merge(name) == expected


# -------------------------------
# prettify_unit_name_supers
# -------------------------------

def test_prettify_cancel_true_cancels_and_supscripts():
    # cancellation and superscripts
    # 'kg*m/s^2' -> 'kg·m/s²'
    out = utils.prettify_unit_name_supers("kg*m/s^2", cancel=True)
    assert out == "kg·m/s²"

    # 'cm/ms^3·ms' -> 'cm/ms²'
    out2 = utils.prettify_unit_name_supers("cm/ms^3·ms", cancel=True)
    assert out2 == "cm/ms²"

def test_prettify_cancel_false_restylizes_only():
    # No cancellation, just styling and superscripts
    out = utils.prettify_unit_name_supers("kg*m/s^2", cancel=False)
    # '*' -> '·', '^2' -> '²', but keep full tokens (no cancel)
    assert out == "kg·m/s²"


# -------------------------------
# format_dim
# -------------------------------

def test_format_dim_basic_and_ordering():
    # Force dimension: MASS * LENGTH / TIME^2 => 'kg·m/s²'
    force_dim = dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2)))
    assert utils.format_dim(force_dim) == "kg·m/s²"

    # Just LENGTH -> 'm'
    assert utils.format_dim(LENGTH) == "m"

    # Dimensionless -> '1'
    assert utils.format_dim(DIM_0) == "1"

    # TIME^-1 -> '1/s'
    assert utils.format_dim(dim_pow(TIME, -1)) == "1/s"


# -------------------------------
# preferred_symbol_for_dim + cache build (first-wins)
# -------------------------------

def test_preferred_symbol_first_wins_for_time_inverse():
    # Ensure map is rebuilt fresh
    utils.invalidate_preferred_cache()

    # TIME^-1 has both Hz and Bq; _PREFERRED_ORDER lists 'Hz' before 'Bq'
    time_inv = dim_pow(TIME, -1)
    assert utils.preferred_symbol_for_dim(time_inv) == "Hz"

def test_preferred_symbol_energy_over_mass_and_length():
    utils.invalidate_preferred_cache()

    # Force dimension should prefer 'N'
    force_dim = dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2)))
    assert utils.preferred_symbol_for_dim(force_dim) == "N"

def test_preferred_symbol_for_base_dims():
    utils.invalidate_preferred_cache()

    assert utils.preferred_symbol_for_dim(LENGTH) == "m"
    assert utils.preferred_symbol_for_dim(MASS) == "kg"
    assert utils.preferred_symbol_for_dim(TIME) == "s"

def test_preferred_symbol_none_when_no_symbol_defined():
    utils.invalidate_preferred_cache()

    # Velocity (L/T) has no preferred named symbol in _PREFERRED_ORDER
    vel_dim = dim_div(LENGTH, TIME)
    assert utils.preferred_symbol_for_dim(vel_dim) is None


# -------------------------------
# invalidate_preferred_cache
# -------------------------------

def test_invalidate_preferred_cache_rebuilds_map(monkeypatch):
    # Build once
    utils.invalidate_preferred_cache()
    sym1 = utils.preferred_symbol_for_dim(dim_pow(TIME, -1))
    assert sym1 == "Hz"

    # Monkeypatch _PREFERRED_ORDER to swap preference (put 'Bq' first)
    orig = utils._PREFERRED_ORDER[:]
    try:
        monkeypatch.setattr(utils, "_PREFERRED_ORDER", ["Bq", "Hz"] + orig, raising=False)
        utils.invalidate_preferred_cache()
        sym2 = utils.preferred_symbol_for_dim(dim_pow(TIME, -1))
        assert sym2 == "Bq"
    finally:
        # Restore original list and cache
        monkeypatch.setattr(utils, "_PREFERRED_ORDER", orig, raising=False)
        utils.invalidate_preferred_cache()
        sym3 = utils.preferred_symbol_for_dim(dim_pow(TIME, -1))
        assert sym3 == "Hz"


# -------------------------------
# _expand_parentheses
# -------------------------------

@pytest.mark.parametrize("expr, expected", [
    # --- docstring examples / core behaviors ---
    ("W·s/(N·s/m^2)", "W·s·m^2/N/s"),
    ("(kg·m/s^2)·m/s", "kg·m/s^2·m/s"),
    ("x/(a/b·c)", "x·b/a/c"),

    # --- simple cases (no change) ---
    ("kg·m/s^2", "kg·m/s^2"),            # no parentheses → unchanged
    ("m", "m"),

    # --- ASCII '*' normalization inside and around parens ---
    ("a*(b/c)", "a·b/c"),
    ("a*(b*c)", "a·b·c"),
    ("a/(b*c)", "a/b/c"),

    # --- neutral/single-token parentheses are removed ---
    ("(m^2)", "m^2"),
    ("((m))", "m"),

    # --- multiple groups ---
    ("a/(b·c)·(d/e)", "a/b/c·d/e"),
    ("(a/b)/(c/d)", "a/b/c·d"),          # outer division by (c/d) → multiply by d/c

    # --- nested parentheses (expand innermost-first) ---
    ("x/(a/(b/c))", "x/(a/b·c)"),        # inner expands; outer left as-is for next pass by caller
    # If the caller feeds the result back in again, it would become: "x·c/b/a"
    # but this test checks the function's single-pass contract.

    # --- spaces & mixed separators handled ---
    ("a * ( b / c ) / ( d / e )", "a·b/c/d·e"),
])
def test_expand_parentheses_various(expr, expected):
    assert utils._expand_parentheses(expr) == expected


def test_expand_parentheses_unbalanced_returns_input():
    # The helper bails out (returns input) when it can't find a matching ')'
    expr = "a/(b"
    assert utils._expand_parentheses(expr) == expr


def test_expand_parentheses_idempotent():
    # Once expanded, a second call should be a no-op
    expr = "W·s/(N·s/m^2)"
    once = utils._expand_parentheses(expr)
    twice = utils._expand_parentheses(once)
    assert once == twice

#############################################
# rationalize function test
#############################################


@pytest.mark.parametrize("value,expected", [
    (0, 0),
    (3, 3),
    (-7, -7),
])
def test_integers_return_int(value, expected):
    assert utils.rationalize(value) == expected


@pytest.mark.parametrize("value,expected", [
    (0, Fraction(0, 1)),
    (3, Fraction(3, 1)),
    (-7, Fraction(-7, 1)),
])
def test_integers_as_fraction(value, expected):
    assert utils.rationalize(value, as_fraction=True) == expected


@pytest.mark.parametrize("value,expected_frac", [
    (0.5, Fraction(1, 2)),     # exact as float
    (1.25, Fraction(5, 4)),    # exact as float
    (2.0, Fraction(2, 1)),     # denominator 1 -> will return int if as_fraction=False
    (-0.5, Fraction(-1, 2)),
])
def test_exact_float_rationals(value, expected_frac):
    out = utils.rationalize(value, as_fraction=True)
    assert out == expected_frac

    out_default = utils.rationalize(value, as_fraction=False)
    if expected_frac.denominator == 1:
        assert isinstance(out_default, int)
        assert out_default == expected_frac.numerator
    else:
        assert isinstance(out_default, Fraction)
        assert out_default == expected_frac


def test_returns_int_when_denominator_is_one():
    assert utils.rationalize(2.0) == 2
    assert isinstance(utils.rationalize(2.0), int)


# ----- Error cases -----

@pytest.mark.parametrize("value", [
    float("inf"),
    float("-inf"),
    float("nan"),
])
def test_nonfinite_raises_value_error(value):
    with pytest.raises(ValueError):
        utils.rationalize(value)


@pytest.mark.parametrize("value", [
    math.pi,                # irrational
    math.sqrt(2),           # irrational
    1.3333333333,           # not exactly 4/3 as a float -> should raise
    0.142857,               # close to 1/7 but not exact -> should raise
])
def test_irrational_or_nonexact_decimal_raises(value):
    with pytest.raises(ValueError):
        utils.rationalize(value)


def test_type_error_on_unsupported_types():
    with pytest.raises(TypeError):
        utils.rationalize("3/2")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        utils.rationalize(Fraction(3, 2))  # type: ignore[arg-type]


# ----- max_denominator behavior -----

def test_max_denominator_does_not_force_acceptance():
    # Even with small max_denominator, 0.5 still maps to 1/2 exactly (float equality holds)
    assert utils.rationalize(0.5, max_denominator=2) == Fraction(1, 2)

    # A value like 0.142857 is not exactly equal to 1/7 in binary float;
    # round-trip check should fail regardless of max_denominator.
    with pytest.raises(ValueError):
        utils.rationalize(0.142857, max_denominator=7)

    with pytest.raises(ValueError):
        utils.rationalize(0.142857, max_denominator=10_000)

#############################################
# simplify_fraction function test
#############################################

def test_reduces_to_int_simple():
    assert utils.simplify_fraction(Fraction(4, 2)) == 2
    assert isinstance(utils.simplify_fraction(Fraction(4, 2)), int)

def test_reduces_to_int_zero():
    # Any zero Fraction reduces to 0/1
    assert utils.simplify_fraction(Fraction(0, 5)) == 0
    assert isinstance(utils.simplify_fraction(Fraction(0, 5)), int)

def test_reduces_to_int_negative():
    assert utils.simplify_fraction(Fraction(-6, 3)) == -2
    assert isinstance(utils.simplify_fraction(Fraction(-6, 3)), int)

def test_keeps_fraction_when_not_integer():
    out = utils.simplify_fraction(Fraction(3, 2))
    assert out == Fraction(3, 2)
    assert isinstance(out, Fraction)

def test_keeps_already_reduced_fraction_when_not_integer():
    out = utils.simplify_fraction(Fraction(7, 5))
    assert out == Fraction(7, 5)
    assert isinstance(out, Fraction)

def test_large_values():
    # 1000000/250000 reduces to 4 -> int
    out_int = utils.simplify_fraction(Fraction(1_000_000, 250_000))
    assert out_int == 4
    assert isinstance(out_int, int)

    # Non-integer large fraction stays Fraction
    out_frac = utils.simplify_fraction(Fraction(1_000_001, 250_000))
    assert out_frac == Fraction(1_000_001, 250_000)
    assert isinstance(out_frac, Fraction)

def test_pass_through_int_and_float():
    assert utils.simplify_fraction(5) == 5
    assert isinstance(utils.simplify_fraction(5), int)

    # Function is specified to return floats unchanged
    val = 1.5
    out = utils.simplify_fraction(val)
    assert out == val
    assert isinstance(out, float)

def test_idempotence_on_fraction():
    f = Fraction(9, 6)  # reduces to 3/2 internally
    first = utils.simplify_fraction(f)     # -> Fraction(3, 2)
    second = utils.simplify_fraction(first)
    assert first == Fraction(3, 2)
    assert second == Fraction(3, 2)  # calling again doesn’t change result

def test_sign_handling():
    # Sign should be on numerator; reduction should still yield integer
    assert utils.simplify_fraction(Fraction(-10, -5)) == 2
    assert utils.simplify_fraction(Fraction(10, -5)) == -2

