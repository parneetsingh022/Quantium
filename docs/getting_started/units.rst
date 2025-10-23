Units
=======================

Public API for the Quantium units registry and namespace access.

Classes
-------

.. py:class:: UnitsRegistry()

   Thread-safe registry for :class:`quantium.core.quantity.Unit` objects.
   Handles aliases, normalization, and **lazy** SI-prefix synthesis for atomic symbols.

   .. py:method:: register(unit: Unit, replace: bool = False) -> None

      Register a unit by its canonical symbol (``unit.name``).
      Raises :class:`ValueError` on conflicts unless ``replace=True``.

   .. py:method:: register_alias(alias: str, canonical: str, replace: bool = False) -> None

      Map an alias spelling (case/Unicode tolerant) to a canonical symbol.
      Raises :class:`ValueError` if the alias collides with an existing unit
      (unless ``replace=True``).

   .. py:method:: set_non_prefixable(symbols: Iterable[str]) -> None

      Mark symbols that must **not** accept SI prefixes (e.g., ``"kg"``, ``"min"``).

   .. py:method:: has(symbol: str) -> bool

      Return ``True`` if ``symbol`` resolves (via alias, synthesis, or parsing).

   .. py:method:: get(symbol: str) -> Unit

      Resolve a unit:
      - Atomic symbols (direct/alias/prefixed synthesis)
      - Compound expressions (e.g., ``"m/s**2"``) via :func:`quantium.units.parser.extract_unit_expr`

      Raises :class:`ValueError` if not found.

   .. py:method:: all() -> Mapping[str, Unit]

      Return a **copy** of the internal symbol→Unit mapping.

   .. py:method:: as_namespace() -> UnitNamespace

      Return a convenience façade for attribute and callable access.


.. py:class:: UnitNamespace(reg: UnitsRegistry)

   Namespace view over a registry.

   .. py:method:: __call__(spec: str) -> Unit
      :noindex:

      Resolve atomic or compound ``spec`` using :meth:`UnitsRegistry.get`.

   .. py:method:: __getattr__(name: str) -> Unit
      :noindex:

      Attribute access for symbols (e.g., ``u.m``). Unknown names raise :class:`AttributeError`.

   .. py:method:: define(expr: str, scale: float|int, reference: Unit, replace: bool = False) -> None
      :noindex:

      Define a new unit by scale relative to a reference unit with the same dimension.

   .. py:method:: __contains__(spec: str) -> bool
      :noindex:

      Delegates to :meth:`UnitsRegistry.has`.

   .. py:method:: __dir__() -> list[str]
      :noindex:

      Returns Python attributes + registered symbols + known aliases (useful for autocomplete).


Functions
---------

.. py:function:: normalize_symbol(s: str) -> str

   Normalize a user-provided unit symbol:
   - Unicode NFC
   - Leading ASCII ``'u'`` → micro ``'µ'`` (at start)
   - Case-insensitive ``"ohm"`` → ``"Ω"``
   - Trim surrounding whitespace


Module attributes
-----------------

.. py:data:: DEFAULT_REGISTRY
   :type: UnitsRegistry

   Shared, pre-bootstrapped registry. Typical usage:

   .. code-block:: python

      from quantium.units.registry import DEFAULT_REGISTRY
      u = DEFAULT_REGISTRY.as_namespace()

      3 * u.m
      9.81 * u("m/s**2")
      10 * u("kg*m/s**2")   # N
      6 * u("ohm")          # Ω


Units in the default registry
--------------------------------

The table below lists **canonical symbols** and their **aliases** (if any) bundled
in :data:`DEFAULT_REGISTRY`. Prefixed forms (e.g., ``km``, ``mA``) are synthesized
lazily where valid. Symbols in the **Non-prefixable** list will not accept SI prefixes.

.. tip::
   You can reference units via attributes (``u.m``) **or** strings (``u("m")``,
   ``u("m/s**2")``, ``u("kg*m/s**2")``).

.. list-table::
   :header-rows: 1
   :widths: 16 24 60

   * - Symbol
     - Dimension (informal)
     - Aliases

   * - m
     - length
     - —
   * - kg
     - mass
     - —
   * - s
     - time
     - —
   * - A
     - electric current
     - —
   * - K
     - thermodynamic temperature
     - —
   * - mol
     - amount of substance
     - —
   * - cd
     - luminous intensity
     - —

   * - rad
     - dimensionless (radian)
     - —
   * - sr
     - dimensionless (steradian)
     - —

   * - g
     - mass (1e-3 kg)
     - —
   * - Hz
     - frequency (s⁻¹)
     - —
   * - N
     - force (kg·m/s²)
     - —
   * - Pa
     - pressure (N/m²)
     - —
   * - J
     - energy (N·m)
     - —
   * - W
     - power (J/s)
     - —
   * - C
     - electric charge (A·s)
     - —
   * - V
     - electric potential (W/A)
     - —
   * - F
     - capacitance (C/V)
     - —
   * - Ω
     - resistance (V/A)
     - ohm, Ohm, OHM
   * - S
     - conductance (A/V)
     - —
   * - Wb
     - magnetic flux (V·s)
     - —
   * - T
     - magnetic flux density (Wb/m²)
     - —
   * - H
     - inductance (Wb/A)
     - —
   * - lm
     - luminous flux (cd·sr)
     - —
   * - lx
     - illuminance (lm/m²)
     - —
   * - Bq
     - activity (s⁻¹)
     - —
   * - Gy
     - absorbed dose (J/kg)
     - —
   * - Sv
     - dose equivalent (J/kg)
     - —
   * - kat
     - catalytic activity (mol/s)
     - —

   * - min
     - time (60 s)
     - minute, minutes
   * - h
     - time (3600 s)
     - hr, hour, hours
   * - d
     - time (86400 s)
     - day, days
   * - wk
     - time (7 d)
     - week, weeks
   * - fortnight
     - time (14 d)
     - fortnights
   * - mo
     - time (~30.436875 d; Gregorian mean month)
     - month, months
   * - yr
     - time (365.2425 d; Gregorian mean year)
     - year, years, annum
   * - yr_julian
     - time (365.25 d; Julian year)
     - —
   * - decade
     - time (10 yr)
     - dec, decades
   * - century
     - time (100 yr)
     - cent, centuries
   * - millennium
     - time (1000 yr)
     - millennia


Non-prefixable symbols
----------------------

These canonical symbols are **not** eligible for SI prefixes in the default registry:

``kg``, ``min``, ``h``, ``d``, ``wk``, ``fortnight``, ``mo``, ``yr``, ``yr_julian``,
``decade``, ``century``, ``millennium``.
