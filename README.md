<h1 align="center">Quantium</h1>


<p align="center">
  <img src="https://github.com/user-attachments/assets/f2edd31b-5091-4432-a8c9-34c664aa2b2f" 
       alt="Quantium logo" 
       width="300" 
       height="300">
</p>

### Readable units. Reliable math.

Quantium is a lightweight Python library for unit-safe scientific and mathematical computation.
It combines a clean, dependency-minimal architecture with a powerful system for dimensional analysis — ensuring that every calculation you perform respects physical consistency.

Beyond correctness, Quantium emphasizes clarity.
Its advanced formatting engine automatically simplifies, normalizes, and beautifully renders units using Unicode superscripts, SI prefixes, and canonical symbols.

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
