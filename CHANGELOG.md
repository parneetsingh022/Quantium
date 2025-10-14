# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking Change
- Removed old `get_unit()` function and introduced new UnitRegistry class with `register()`, `register_alias()`, `has()`, `get()`, `all()` functions.

### Added
- Support for unit algebra: units can now be combined using multiplication (`*`), division (`/`), and exponentiation (`**`) to produce new derived units with correct dimensional analysis (e.g., `m/s`, `m^2`, `N·m`, etc.).

- Added common time units (min, h, d, wk, fortnight, mo, yr, yr_julian, decade, century, millennium) with full alias support and SI-based scaling in `Default Registry`.

- Added `.si` property to a quantity to convert any quantity to its respective SI unit.


## [0.0.1a0] - 2025-10-09
### Added
- Initial alpha release of **Quantium**.
- Core support for unit-safe mathematical calculations.
- `get_unit()` API for creating and combining physical units.
- Basic arithmetic operations between unit quantities (`+`, `-`, `*`, `/`, `**`).
- String representation of unit results (e.g., `10 m/s`).

### Notes
- NumPy interoperability is **not yet supported** but planned for future versions.
