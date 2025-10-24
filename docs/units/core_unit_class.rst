Core Unit Class
================

.. module:: quantium.core.quantity
   :noindex:

The ``Unit`` class is the foundation for all physical units in Quantium. Each ``Unit`` object
stores its name (symbol), its conversion factor to the equivalent base SI unit, and its
physical dimension.

.. class:: Unit(name: str, scale_to_si: float, dim: Dim)
   :noindex:

   Represents a physical unit. This class is generally not instantiated directly by users.
   Instead, users access predefined units from the default registry (see below) or create new
   units using the registry's helper methods.

   :param name: The symbol or name of the unit (e.g., "m", "kg", "cm").
   :param scale_to_si: The multiplicative factor to convert 1 of this unit to its SI equivalent.
   :param dim: The ``Dim`` object representing the unit's physical dimension (L, M, T, I, Θ, N, J).

   **Attributes**

   .. attribute:: name
      :noindex:
      :type: str

      Symbol or name (e.g., "m", "s", "kg", "cm").

   .. attribute:: scale_to_si
      :noindex:
      :type: float

      Multiplicative factor to convert 1 of this unit to SI for its dimension
      (e.g., m=1.0, cm=0.01, ft=0.3048).

   .. attribute:: dim
      :noindex:
      :type: quantium.core.dimensions.Dim

      Dimension vector (L,M,T,I,Θ,N,J).

   .. rubric:: Unit Arithmetic

   ``Unit`` objects can be combined using arithmetic operators to create new, derived units.

   .. method:: __mul__(other: Unit) -> Unit
      :noindex:

      Multiplies two units. (e.g., ``m * m`` returns ``Unit(name='m^2', ...)``)

   .. method:: __truediv__(other: Unit) -> Unit
      :noindex:

      Divides two units. (e.g., ``m / s`` returns ``Unit(name='m/s', ...)``)

   .. method:: __rtruediv__(n: int | float) -> Unit
      :noindex:

      Calculates the reciprocal of a unit. Only ``1 / unit`` is supported
      (e.g., ``1 / s`` returns ``Unit(name='s^-1', ...)``).

   .. method:: __pow__(n: int) -> Unit
      :noindex:

      Raises a unit to an integer power (e.g., ``m ** 2`` returns ``Unit(name='m^2', ...)``).

   .. method:: __rmul__(value: float) -> Quantity
      :noindex:

      Creates a ``Quantity`` by multiplying a number with a unit.
      This is the magic that enables the ``100 * u.m`` syntax.
