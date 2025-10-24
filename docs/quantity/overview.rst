Overview
========

The ``Quantity`` class is the primary object for performing calculations in Quantium.
It binds a numerical ``value`` to a specific ``Unit``, allowing for
dimensionally-aware arithmetic, comparisons, and unit conversions.

Instantiation and Access
------------------------

A ``Quantity`` object holds your value and its unit. Multiplying a number by a unit
(e.g., ``100 * u.m``) is the standard, convenient way to create one. This operation
calls the ``Quantity`` constructor and returns the resulting object.

.. code-block:: python

   from quantium.core.quantity import Quantity
   from quantium import u

   # Preferred shorthand (returns a Quantity)
   dist_short = 100 * u.m

   # Direct constructor (equivalent)
   dist_long = Quantity(100, u.m)

   assert dist_short == dist_long

Accessing values
----------------

Once you have a ``Quantity`` object, you can access its numeric value in the
current unit via the ``.value`` property, or get an SI-expressed ``Quantity``
using the ``.si`` property.

.. code-block:: python

   time = Quantity(10, u.s)
   print(time.value)       # 10.0

   dist_km = Quantity(5, u.km)
   print(dist_km)          # 5 km
   print(dist_km.value)    # 5.0
   print(dist_km.si)       # 5000 m
   print(dist_km.si.value) # 5000.0
