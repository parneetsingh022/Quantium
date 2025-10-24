# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - **Unreleased**

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
