<h1 align="center">Quantium</h1>


<p align="center">
  <img src="https://github.com/user-attachments/assets/f2edd31b-5091-4432-a8c9-34c664aa2b2f" 
       alt="Quantium logo" 
       width="300" 
       height="300">
</p>

<p>
  <a href="https://badge.fury.io/py/quantium">
    <img src="https://badge.fury.io/py/quantium.svg" alt="PyPI version">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://img.shields.io/pypi/pyversions/quantium">
    <img src="https://img.shields.io/pypi/pyversions/quantium" alt="Python Version">
  </a>
  <a href="https://quantium.readthedocs.io/en/latest/">
    <img src="https://readthedocs.org/projects/quantium/badge/?version=latest" alt="Documentation Status">
  </a>
  <a href="https://github.com/parneetsingh022/quantium/actions">
    <img src="https://github.com/parneetsingh022/quantium/actions/workflows/ci.yml/badge.svg" alt="Build Status">
  </a>
  <a href="http://mypy-lang.org/">
    <img src="https://www.mypy-lang.org/static/mypy_badge.svg" alt="Checked with mypy">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://codecov.io/gh/parneetsingh022/quantium">
  <img src="https://codecov.io/gh/parneetsingh022/quantium/graph/badge.svg" alt="Test Coverage">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://pepy.tech/project/quantium">
  <img src="https://static.pepy.tech/personalized-badge/quantium?period=total&units=international_system&left_color=black&right_color=blue&left_text=downloads" alt="PyPI Downloads">
  </a>
  <a href="https://github.com/parneetsingh022/quantium/commits/main">
    <img src="https://img.shields.io/github/last-commit/parneetsingh022/quantium" alt="GitHub last commit">
  </a>
</p>

### Readable units. Reliable math.



Quantium is a lightweight Python library for unit-safe scientific and mathematical computation.

It combines a clean, dependency-minimal architecture with a powerful system for dimensional analysis — ensuring that every calculation you perform respects physical consistency.



Beyond correctness, Quantium emphasizes clarity.

Its advanced formatting engine automatically simplifies, normalizes, and beautifully renders units using Unicode superscripts, SI prefixes, and canonical symbols.



## Key Features

- Dimensional Analysis: Guarantees physical consistency in all calculations.



- Unit Simplification: Automatically recognizes and simplifies composite units to their standard named forms (e.g., `kg*m/s**2` is displayed as `N`).



- Beautiful Formatting: Renders all units into a clean, human-readable format using Unicode dots for multiplication and superscripts for exponents (e.g., `kg*m**2` becomes `kg·m²`).



- SI Prefix Support: Easily convert between base units and their prefixed forms (e.g., a Quantity of `1000 m` can be converted to `1 km`).



- Extensible: Easily define your own custom units and dimensions.



## Documentation

View the [**official Quantium documentation**](https://quantium.readthedocs.io/) for installation guides, tutorials, and the complete API reference.



## Installation & Setup

Quantium can be installed from the Python Package Index (PyPI):



```bash

pip install quantium

```



After installation, verify that Quantium is correctly installed by checking its version:



```python

import quantium

print("Quantium version:", quantium.__version__)

```



To make sure Quantium is ready to use, open a Python shell and run:

```python

>>> from quantium import u

>>> (10 * u.kg) * (5 * u.m) / (2 * u.s**2)

25 N

```



## Requirements

Quantium is built to work seamlessly in modern environments and is compatible with current development tools and workflows.



Quantium currently supports **Python 3.10 and above**.


## Contributing

Contributions are welcome! If you'd like to fix a bug, add a feature, or improve the documentation, please feel free to open an issue or submit a pull request.
