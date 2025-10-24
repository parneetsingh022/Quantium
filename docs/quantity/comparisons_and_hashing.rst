Comparisons and hashing
=======================

Quantities can be compared as long as they have the same dimension. Comparisons
automatically handle unit conversion.

.. code-block:: python

   from quantium import u

   speed_1 = 100 * u.km / u.h
   speed_2 = 30 * u.m / u.s

   print(speed_1 == speed_2) # False
   print(speed_1 < speed_2)  # True (100 km/h is ~27.8 m/s)
   print(speed_2 > speed_1)  # True

Floating point noise
---------------------

The ``Quantity`` equality uses ``math.isclose`` to provide fuzzy equality. This
means small floating-point noise won't break comparisons:

.. code-block:: python

   q1 = 1.0 * u.m
   q2 = (1.0 + 1e-13) * u.m
   print(q1 == q2) # True

Hashing and ``as_key()``
------------------------

Because equality is fuzzy, ``Quantity`` objects are not hashable by default
(doing so would break the hash/equality contract). Use ``.as_key()`` to create
a stable, hashable representation if you need to use quantities as dictionary
keys or in sets.

.. code-block:: python

   q1 = (1.0 + 1e-13) * u.m
   q2 = 1.0 * u.m

   print(q1 == q2) # True

   # Use as_key to create a stable, hashable key
   safe_dict = {}
   safe_dict[q1.as_key()] = "value"
   print(safe_dict[q2.as_key()])  # "value"
