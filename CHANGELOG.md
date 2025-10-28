# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - **Unreleased**

### Added
- Allowed raising Quantity objects to float and fractional powers, enabling operations such as x**0.5 or x**Fraction(1,3).

### Fixed

- Added robust rules for multiplication/division of same-dimension units, now the symbol with the higher total power is chosen (e.g. m * cm**2 -> cm³). [#91]
- Enhanced handling of complex mixed-prefix units (e.g. kg·mm/s² -> mN). [#91]
- Improves readability, prefix consistency, and reduces ambiguity in derived units. [#91]



[#91]: https://github.com/parneetsingh022/quantium/issues/91

## [0.1.1] - 2025-10-25

### Added
- Added Python 3.13 and 3.14 classifiers to `pyproject.toml` for up-to-date Python version metadata. [#86]

### Fixed
- Prevented registration or aliasing of unit names that conflict with existing `UnitNamespace` attributes or methods (e.g., `define`, `__init__`, `_reserved_names`), ensuring consistent and unambiguous behavior when defining units. [#69]
- Added `_reserved_names` declaration inside `UnitNamespace` and post-class initialization to correctly include it in reserved name checks. [#69]
- Updated `UnitsRegistry.register()`, `register_alias()`, and `UnitNamespace.define()` to raise clear `ValueError` messages when attempting to register conflicting names. [#69]

[#86]: https://github.com/parneetsingh022/quantium/issues/86
[#69]: https://github.com/parneetsingh022/quantium/issues/69

## [0.1.0] - 2025-10-24

### Breaking Change
- Removed old `get_unit()` function and introduced new UnitRegistry class with `register()`, `register_alias()`, `has()`, `get()`, `all()` functions.
- Replaced the `@` operator (`4 @ unit`) with the standard multiplication operator (`4 * unit`) for creating quantities.

### Added
- Support for unit algebra: units can now be combined using multiplication (`*`), division (`/`), and exponentiation (`**`) to produce new derived units with correct dimensional analysis (e.g., `m/s`, `m^2`, `N·m`, etc.).

- Added common time units (min, h, d, wk, fortnight, mo, yr, yr_julian, decade, century, millennium) with full alias support and SI-based scaling in `Default Registry`.

- Added `.si` property to a quantity to convert any quantity to its respective SI unit.

- Added support for formatted string output of quantities using the `__format__` method.  
  Quantities can now be printed in their current or SI units directly in f-strings:  
  - `f"{q}"` or `f"{q:native}"` → displays the quantity in its current unit.  
  - `f"{q:si}"` → displays the quantity converted to SI units.  
  This provides a cleaner and more Pythonic way to print quantities without calling `.to_si()` manually.

- Units and quantities now support string-based compound expressions in `.get()` and `.to()` (e.g., `"m/s**2"`, `"(W*s)/(N*s/m**2)"`, `"1/s"`), enabling intuitive text-based conversions and registry lookups for mixed or derived units.

- Added `UnitNamespace` to provide a user-friendly interface for accessing units.

- Added full set of comparison operators (==, !=, <, <=, >, >=) to the Quantity class. Equality comparisons (==, !=, <=, >=) automatically account for small floating-point rounding errors.

- Added .as_key(precision=12) method to Quantity to provide a safe, explicit way to create hashable keys for use in dictionaries and sets.

## [0.0.1a0] - 2025-10-09
### Added
- Initial alpha release of **Quantium**.
- Core support for unit-safe mathematical calculations.
- `get_unit()` API for creating and combining physical units.
- Basic arithmetic operations between unit quantities (`+`, `-`, `*`, `/`, `**`).
- String representation of unit results (e.g., `10 m/s`).

### Notes
- NumPy interoperability is **not yet supported** but planned for future versions.
