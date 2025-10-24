Unit Registry and Namespace
===========================

.. module:: quantium.units.registry
   :noindex:

The ``UnitsRegistry`` class manages all known unit definitions, including aliases and rules
for SI prefixes. The ``UnitNamespace`` provides a convenient, attribute-based way to access
units from a registry.

Accessing Predefined Units
--------------------------

Quantium provides a default registry of predefined units. This is accessed through a
``UnitNamespace``, conventionally imported as ``u``.

.. code-block:: python

   from quantium import u

   # Access units by attribute
   m = u.m      # Access the 'meter' unit
   s = u.s      # Access the 'second' unit

   # 1. Combine units during calculation
   speed = 100 * u.m / u.s

   # 2. Access compound units using string parsing
   speed_unit = u("m/s")
   acceleration = 9.8 * u("m/s**2")
   pressure = 101.3 * u("kPa")

Adding New Units
----------------

The easiest way to define new units is to use the ``define()`` method on the ``u`` namespace.
This method creates a new ``Unit`` and registers it with the default registry.

.. class:: UnitNamespace
   :noindex:

   Provides attribute-style access to a ``UnitsRegistry``. The default instance is ``u``.

   .. method:: define(expr: str, scale: float | int, reference: Unit, replace: bool = False)
      :noindex:

      Defines a new unit based on an existing reference unit.

      :param expr: The name/symbol for the new unit (e.g., "ft", "inch").
      :param scale: The scaling factor *from the reference unit*.
      :param reference: The existing ``Unit`` to base this new unit on.
      :param replace: If ``True``, overwrite any existing unit with this name.

      **Example:**

      .. code-block:: python

         from quantium import u

         # Define imperial length units based on the meter
         u.define("inch", 0.0254, u.m)
         u.define("ft", 12, u.inch)  # Can be based on other defined units
         u.define("mile", 5280, u.ft)

         # Now they are available like any other unit
         dist = 100 * u.ft
         print(dist.to(u.m))  # 30.48 m

   .. method:: __call__(spec: str) -> Unit
      :noindex:

      Parses a string expression to get a unit (e.g., ``u("km/h")``).

   .. method:: __getattr__(name: str) -> Unit
      :noindex:

      Accesses a unit by its name or symbol (e.g., ``u.km``).

SI Prefixes
-----------

The unit registry can automatically synthesize units with SI prefixes (e.g., "k", "m",
"µ", "n"). You can access them directly as attributes.

.. code-block:: python

   from quantium import u

   dist_mm = 10 * u.mm  # millimeter
   cap_pf = 22 * u.pF   # picofarad
   freq_ghz = 5.1 * u.GHz  # gigahertz
   current_ua = 50 * u.uA  # microamp (u or µ)

Note: Not all units can be prefixed. For example, ``kg`` (kilogram) is already a base unit
with a prefix, so ``mkg`` (millikilogram) is not allowed. Non-SI units like ``min`` (minute)
or ``h`` (hour) are also not prefixable, as noted in the tables below.

Advanced Registry Management
----------------------------

For advanced use cases (like creating an isolated set of units), you can instantiate
``UnitsRegistry`` directly.

.. class:: UnitsRegistry
   :noindex:

   A thread-safe registry for ``Unit`` objects with SI prefix synthesis.

   .. method:: register(unit: Unit, replace: bool = False)
      :noindex:

      Registers a new ``Unit`` object directly. Use ``UnitNamespace.define()`` for a simpler
      interface.

   .. method:: register_alias(alias: str, canonical: str, replace: bool = False)
      :noindex:

      Registers an alternative name for an existing unit.

      :param alias: The new name (e.g., "meter").
      :param canonical: The existing, canonical symbol (e.g., "m").

      **Example:**

      .. code-block:: python

         # In the default registry 'u', this is already done:
         # u.register_alias("ohm", "Ω")

         print(u.ohm == u.Ω)  # True

   .. method:: get(symbol: str) -> Unit
      :noindex:

      Looks up a unit by its symbol, parsing expressions or synthesizing prefixes as needed.
      Raises ``ValueError`` if unknown.

   .. method:: has(symbol: str) -> bool
      :noindex:

      Returns ``True`` if the symbol is known or can be synthesized.

   .. method:: set_non_prefixable(symbols: Iterable[str])
      :noindex:

      Marks a set of unit symbols that should not accept SI prefixes (e.g., "kg", "min", "h").
