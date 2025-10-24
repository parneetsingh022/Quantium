Unit conversions
================

You can convert any ``Quantity`` to another compatible unit using the ``.to()`` method.

.. code-block:: python

   from quantium import u

   speed_ms = 10 * u.m / u.s
   print(speed_ms)  # 10 m/s

   # Convert to km/h using the .to() method
   speed_kmh = speed_ms.to(u.km / u.h)
   print(speed_kmh)  # 36 km/h

   # You can also pass a string
   speed_kmh_str = speed_ms.to("km/h")
   print(speed_kmh_str) # 36 km/h

   # Convert back to SI
   print(speed_kmh.si) # 10 m/s

Notes
-----

- ``.to()`` accepts either a ``Unit`` object or a string that will be parsed
  by the unit registry.
- Converting to an incompatible dimension raises ``TypeError``.
