from math import isclose
import pytest

from quantium.core.dimensions import LENGTH, MASS
from quantium.core.quantity import Unit
from quantium.units.registry import DEFAULT_REGISTRY as ureg
import quantium.core.utils as utils



from tests.utils import _nop_prettifier
from quantium.core import utils

# -------------------------------
# __repr__: pretty printing
# -------------------------------

def test_repr_keeps_non_si_unit_name(monkeypatch):
    # Ensure repr uses the current unit name ("cm") and does not replace with a symbol
    import quantium.core.utils as utils

    # Make prettifier a no-op passthrough so we can assert on exact string.
    monkeypatch.setattr(utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True)
    # Ensure it would *try* to upgrade only when scale_to_si == 1.0; here it's 0.01, so no upgrade.
    monkeypatch.setattr(utils, "preferred_symbol_for_dim", lambda d: "m", raising=True)

    cm = Unit("cm", 0.01, LENGTH)
    q = 2 * cm
    assert repr(q) == "2 cm"

def test_repr_upgrades_to_preferred_symbol_when_scale_is_1(monkeypatch):
    from quantium import core as _core
    # prettifier just returns what it's given
    monkeypatch.setattr(
        _core.utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True
    )
    # preferred symbol for LENGTH is "m"
    monkeypatch.setattr(_core.utils, "preferred_symbol_for_dim", lambda d: "m" if d == LENGTH else None, raising=True)

    m = Unit("m", 1.0, LENGTH)
    q = 3 * m
    # scale_to_si == 1.0 -> allowed to upgrade pretty name to "m"
    assert repr(q) == "3 m"


# -------------------------------
# __format__: formatting specs (broad coverage)
# -------------------------------


def test_format_equivalents_across_units(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    cm = ureg.get("cm")
    s  = ureg.get("s")
    kPa = ureg.get("kPa")
    ohm = ureg.get("ohm")  # alias -> Ω
    min_ = ureg.get("min")

    # 1) Velocity in non-SI (cm/s)
    v = 250 * (cm / s)
    assert f"{v}" == "250 cm/s"
    assert f"{v:unit}" == "250 cm/s"
    assert f"{v:u}" == "250 cm/s"

    # 2) Pressure with prefix (kPa)
    p = 2 * kPa
    assert f"{p}" == "2 kPa"
    assert f"{p:unit}" == "2 kPa"

    # 3) Alias normalization (ohm → Ω)
    r = 5 * ohm
    assert f"{r}" == "5 Ω"
    assert f"{r:unit}" == "5 Ω"

    # 4) Mixed time unit in denominator (m/min)
    speed = 120 * (ureg.get("m") / min_)
    assert f"{speed}" == "120 m/min"
    assert f"{speed:unit}" == "120 m/min"


def test_format_si_converts_various_units(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    cm = ureg.get("cm")
    s  = ureg.get("s")
    kPa = ureg.get("kPa")
    min_ = ureg.get("min")

    # 1) 1000 cm/s -> 10 m/s
    v = 1000 * (cm / s)
    assert f"{v:si}" == "10 m/s"

    # 2) 2 kPa -> 2000 Pa
    p = 2 * kPa
    assert f"{p:si}" == "2000 Pa"

    # 3) 120 m/min -> 2 m/s (since 1 min = 60 s)
    speed = 120 * (ureg.get("m") / min_)
    assert f"{speed:si}" == "2 m/s"

    # 4) Frequency with prefix: 2 kHz -> 2000 Hz
    kHz = ureg.get("kHz")
    f = 2 * kHz
    assert f"{f:si}" == "2000 Hz"


def test_format_with_micro_and_normalization(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    # Leading 'u' maps to Greek micro 'µ' during lookup/registration
    uF = ureg.get("uF")     # normalized to µF internally
    q = 3 * uF
    # Current unit uses the canonical symbol
    assert f"{q}" == "3 µF"
    # SI format converts to base Farads (3e-06 F)
    assert f"{q:si}" == "3e-06 F"


def test_format_whitespace_and_case_insensitivity(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    cm = ureg.get("cm")
    s  = ureg.get("s")
    v = 1000 * (cm / s)  # 10 m/s in SI

    # Varied spacing/casing should still resolve to SI
    assert f"{v: SI }" == "10 m/s"
    assert f"{v:Si}" == "10 m/s"
    assert f"{v:  sI}" == "10 m/s"


def test_format_dimensionless_is_numeric_only(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    # Dimensionless via equal ratio (kPa / Pa)
    kPa = ureg.get("kPa")
    Pa  = ureg.get("Pa")

    q = (3 * kPa) / (3000 * Pa)  # equals 1 (dimensionless)
    assert f"{q}" == "1"
    assert f"{q:unit}" == "1"
    assert f"{q:si}" == "1"


def test_format_invalid_spec_raises(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    m = ureg.get("m")
    q = 3 * m
    with pytest.raises(ValueError):
        _ = f"{q:unknown}"

# -------------------------------
# .si preserves correct SI family (Hz/Bq, Gy/Sv)
# -------------------------------

def test_si_preserves_family_for_time_inverse_and_dose(monkeypatch):
    _nop_prettifier(monkeypatch)
    import quantium.core.utils as utils
    utils.invalidate_preferred_cache()

    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    Hz = ureg.get("Hz")
    Bq = ureg.get("Bq")
    Gy = ureg.get("Gy")
    Sv = ureg.get("Sv")

    kHz = ureg.get("kHz")
    kBq = ureg.get("kBq")
    kGy = ureg.get("kGy")
    kSv = ureg.get("kSv")

    # Sanity: original units print as-is
    assert f"{100 * Hz}" == "100 Hz"
    assert f"{100 * Bq}" == "100 Bq"
    assert f"{100 * Gy}" == "100 Gy"
    assert f"{100 * Sv}" == "100 Sv"

    # Prefixed forms print as-is
    assert f"{100 * kHz}" == "100 kHz"
    assert f"{100 * kBq}" == "100 kBq"
    assert f"{100 * kGy}" == "100 kGy"
    assert f"{100 * kSv}" == "100 kSv"

    # .si should preserve family heads (Hz↔Hz, Bq↔Bq, Gy↔Gy, Sv↔Sv)
    assert f"{(100 * kHz):si}" == "100000 Hz"
    assert f"{(100 * kBq):si}" == "100000 Bq"
    assert f"{(100 * kGy):si}" == "100000 Gy"
    assert f"{(100 * kSv):si}" == "100000 Sv"


# -------------------------------
# __repr__ should not flip atomic symbols but upgrade composed SI names
# -------------------------------

def test_repr_preserves_atomic_symbols_and_prefixed(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    # Atomic SI units stay unchanged
    for sym in ("Hz", "Bq", "Gy", "Sv"):
        u = ureg.get(sym)
        assert f"{100 * u}" == f"100 {sym}"

    # Prefixed atomic SI units also stay unchanged
    for sym in ("kHz", "kBq", "kGy", "kSv"):
        u = ureg.get(sym)
        assert f"{100 * u}" == f"100 {sym}"


def test_repr_upgrades_only_composed_si(monkeypatch):
    _nop_prettifier(monkeypatch)
    import quantium.core.utils as utils
    utils.invalidate_preferred_cache()

    from quantium.units.registry import DEFAULT_REGISTRY as ureg
    C = ureg.get("C")
    s = ureg.get("s")
    kg = ureg.get("kg")
    m = ureg.get("m")

    # C/s -> A (preferred symbol for electric current)
    q_current = (1 * C) / (1 * s)
    assert f"{q_current}" == "1 A"

    # kg·m/s² -> N (preferred symbol for force)
    # wrap s as a Quantity: (1 * s) ** 2, not (s ** 2)
    q_force = (2 * kg) * (3 * m) / ((1 * s) ** 2)
    assert f"{q_force}" == "6 N"



def test_si_fallback_to_composed_when_no_named_symbol(monkeypatch):
    _nop_prettifier(monkeypatch)
    import quantium.core.utils as utils
    utils.invalidate_preferred_cache()

    from quantium.units.registry import DEFAULT_REGISTRY as ureg
    cm = ureg.get("cm")
    s = ureg.get("s")

    v = 1000 * (cm / s)  # velocity: no named SI symbol
    assert f"{v:si}" == "10 m/s"


def test_force_micro_and_kilo_newton(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    mg = ureg.get("mg")    # 1e-6 kg
    um = ureg.get("µm")    # 1e-6 m
    ms = ureg.get("ms")    # 1e-3 s
    # mg·µm/ms² -> (1e-6*1e-6)/(1e-6) = 1e-6 -> µN
    q_micro = 1 * (mg * um / (ms ** 2))
    assert f"{q_micro}" == "1 µN"

    kg = ureg.get("kg")    # 1 kg
    km = ureg.get("km")    # 1e3 m
    s  = ureg.get("s")     # 1 s
    # kg·km/s² -> (1*1e3)/(1) = 1e3 -> kN
    q_kilo = 1 * (kg * km / (s ** 2))
    assert f"{q_kilo}" == "1 kN"


def test_current_symbol_and_prefix(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    C  = ureg.get("C")
    s  = ureg.get("s")
    mC = ureg.get("mC")   # 1e-3 C
    ms = ureg.get("ms")   # 1e-3 s

    # C/s -> A
    q_A = (1 * C) / (1 * s)
    assert f"{q_A}" == "1 A"

    # µC/ms -> (1e-6 / 1e-3) = 1e-3 -> mA
    uC = ureg.get("µC")
    q_mA = (1 * uC) / (1 * ms)
    assert f"{q_mA}" == "1 mA"

    # mC/s -> (1e-3 / 1) = 1e-3 -> mA
    q_mA2 = (1 * mC) / (1 * s)
    assert f"{q_mA2}" == "1 mA"


def test_power_symbol_and_prefix(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    J  = ureg.get("J")
    s  = ureg.get("s")
    mJ = ureg.get("mJ")
    ms = ureg.get("ms")
    uJ = ureg.get("µJ")

    # J/s -> W
    q_W = (1 * J) / (1 * s)
    assert f"{q_W}" == "1 W"

    # mJ/ms -> (1e-3 / 1e-3) = 1 -> W
    q_W2 = (1 * mJ) / (1 * ms)
    assert f"{q_W2}" == "1 W"

    # µJ/ms -> (1e-6 / 1e-3) = 1e-3 -> mW
    q_mW = (1 * uJ) / (1 * ms)
    assert f"{q_mW}" == "1 mW"


def test_pressure_symbol_and_prefix(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    mN = ureg.get("mN")   # 1e-3 N composed from kg·m/s² but available via registry
    mm = ureg.get("mm")   # 1e-3 m

    # mN/mm² -> (1e-3) / (1e-3)^2 = 1e3 -> kPa
    # construct as a Quantity / Quantity so the unit algebra flows through
    q_kPa = (1 * mN) / ((1 * mm) ** 2)
    assert f"{q_kPa}" == "1 kPa"

    # N/mm² -> (1) / (1e-3)^2 = 1e6 -> MPa
    N = ureg.get("N")
    q_MPa = (1 * N) / ((1 * mm) ** 2)
    assert f"{q_MPa}" == "1 MPa"


def test_frequency_symbol_and_prefix(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    ms = ureg.get("ms")
    us = ureg.get("µs")
    s  = ureg.get("s")

    # 1/ms -> 1e3 1/s -> kHz
    q_kHz = 1 * (1 / ms)         # Unit reciprocal → Quantity via @
    assert f"{q_kHz}" == "1 kHz"

    # 1/µs -> 1e6 1/s -> MHz
    q_MHz = 1 * (1 / us)
    assert f"{q_MHz}" == "1 MHz"

    # 1/s -> Hz
    q_Hz = 1 * (1 / s)
    assert f"{q_Hz}" == "1 Hz"


def test_atomic_units_not_flipped(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    # Atomic SI heads should print as-is (no cross-family flip)
    for sym in ("Hz", "Bq", "Gy", "Sv", "Pa", "A", "W", "N"):
        u = ureg.get(sym)
        assert f"{100 * u}" == f"100 {sym}"


def test_dimensionless_prints_number(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    m = ureg.get("m")
    # (1 m) / (1 m) is dimensionless → bare number
    q = (1 * m) / (1 * m)
    # repr should be just the number; tolerate "1" exactly
    assert f"{q}" == "1"


# -------------------------------
# __repr__: Regression tests for cancellation + upgrade logic (Issue #72)
# (These tests use the *real* prettifier)
# -------------------------------

@pytest.mark.regression(reason="Issue #72: Quantity.__repr__ incorrectly upgrades a cancelled unit to a prefixed SI unit (e.g., mg becomes µkg)")
def test_issue72_repr_bug_fix_kg_mg_per_kg(monkeypatch):
    """
    REGRESSION TEST for the "µkg" bug.
    Ensures that a unit like "kg·mg/kg" simplifies to "mg" *first*,
    and then __repr__ does NOT upgrade "mg" to "µkg".
    """
    # Invalidate cache to be safe
    utils.invalidate_preferred_cache()
    
    kg = ureg.get("kg")
    mg = ureg.get("mg")
    
    # This is the exact calculation from the bug report
    dose_rate = 15 * (mg / kg)
    patient_mass = 75 * kg
    
    required_dose = patient_mass * dose_rate
    
    # 1. Check the unit name and scale *after* multiplication
    # The raw name is 'kg·mg/kg'
    assert required_dose.unit.name == "kg·mg/kg" 
    # The scale is 1 (for kg) * 1e-6 (for mg/kg) = 1e-6
    assert isclose(required_dose.unit.scale_to_si, 1e-6)
    assert required_dose.dim == MASS
    
    # 2. Check the __repr__
    # The prettifier should cancel 'kg·mg/kg' to 'mg'.
    # The __repr__ logic should see 'mg', see it's *not* composed,
    # and print it as-is.
    # It must NOT see 'kg·mg/kg', see it *is* composed,
    # and upgrade its 1e-6 scale to 'µkg'.
    assert f"{required_dose}" == "1125 mg"
    
    # 3. Check the reverse order
    required_dose_rev = dose_rate * patient_mass
    # The raw name is 'mg/kg·kg'
    assert required_dose_rev.unit.name == "mg/kg·kg"
    assert isclose(required_dose_rev.unit.scale_to_si, 1e-6)
    assert f"{required_dose_rev}" == "1125 mg"

@pytest.mark.regression(reason="Issue #72: Quantity.__repr__ incorrectly upgrades a cancelled unit to a prefixed SI unit (e.g., mg becomes µkg)")
def test_issue72_repr_cancellation_to_prefixed_si(monkeypatch):
    """
    Tests that cancellation still allows a *correct* upgrade when
    the *simplified* unit is still composed.
    'm·mg/kg' -> 'm·(1e-6 kg) / kg' -> 1e-6 m -> 'µm'
    """
    utils.invalidate_preferred_cache()

    m = ureg.get("m")
    mg = ureg.get("mg")
    kg = ureg.get("kg")
    
    q = (1 * m) * (1 * mg) / (1 * kg) # 1 * (m·mg/kg)
    
    # 1. Check unit properties
    assert isclose(q.unit.scale_to_si, 1e-6) # m * (mg/kg) = 1 * 1e-6
    assert q.dim == LENGTH # Dimension is Length
    
    # 2. Check __repr__
    # prettify simplifies 'm·mg/kg' to 'm·mg/kg' (it's already simplified)
    # __repr__ logic:
    #   pretty = "m·mg/kg"
    #   is_composed = True
    #   dim is Length, preferred = "m"
    #   scale_to_si is 1e-6
    #   Logic finds prefix 'µ'.
    #   Logic *correctly* upgrades to "µm".
    assert f"{q}" == "1 µm"

@pytest.mark.regression(reason="Issue #72: Quantity.__repr__ incorrectly upgrades a cancelled unit to a prefixed SI unit (e.g., mg becomes µkg)")
def test_issue72_repr_cancellation_to_dimensionless(monkeypatch):
    """
    Tests that 'mg/kg' simplifies to a dimensionless number.
    The prettifier returns "1", and __repr__ should only print the number.
    """
    utils.invalidate_preferred_cache()
    
    mg = ureg.get("mg")
    kg = ureg.get("kg")
    
    q = (1 * mg) / (1 * kg) # 1 * (mg/kg)
    
    # Value is 1e-6 kg / 1 kg = 1e-6
    assert isclose(q.value, 1e-6)
    # __repr__ should just be the number
    assert f"{q}" == "1e-06"
    
    # Test with equal values
    q_equal = (10 * mg) / (10 * mg)
    assert isclose(q_equal.value, 1)
    assert f"{q_equal}" == "1"

@pytest.mark.regression(reason="Issue #72: Quantity.__repr__ incorrectly upgrades a cancelled unit to a prefixed SI unit (e.g., mg becomes µkg)")
def test_issue72_repr_cancellation_to_si_symbol(monkeypatch):
    """
    Tests that 'mJ/ms' simplifies to 'W'.
    This confirms the existing behavior still works with the *real* prettifier.
    """
    utils.invalidate_preferred_cache()
    
    mJ = ureg.get("mJ")
    ms = ureg.get("ms")
    
    q_W = (1 * mJ) / (1 * ms) # (1e-3 J) / (1e-3 s) = 1 J/s = 1 W
    
    # Prettify returns 'mJ/ms'
    # __repr__ sees 'mJ/ms', is_composed = True
    # dim is Power, preferred = 'W'
    # scale_to_si is 1.0
    # Logic upgrades to 'W'
    assert f"{q_W}" == "1 W"