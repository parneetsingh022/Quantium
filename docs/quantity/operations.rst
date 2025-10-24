Arithmetic operations
=====================

All arithmetic operations are dimensionally-aware. The short ``100 * u.m``
syntax is used throughout for readability.

Addition / Subtraction
----------------------

Quantities must have the same physical dimension (e.g., Length). The result is
returned in the units of the left-hand operand.

.. code-block:: python

   from quantium import u

   total_dist = (1 * u.km) + (500 * u.m)
   print(total_dist)  # 1500 m

Multiplication / Division
-------------------------

Combining quantities produces new dimensions and units (e.g., Length / Time = Speed).

.. code-block:: python

   speed = (100 * u.m) / (10 * u.s)
   print(speed)  # 10 m/s

   force = (10 * u.kg) * (9.8 * u.m / u.s**2)
   print(force)  # 98 N

Exponentiation
--------------

Units can be raised to integer powers.

.. code-block:: python

   area = (5 * u.m) ** 2
   print(area)   # 25 m²

   volume = (2 * u.cm) ** 3
   print(volume) # 8 cm³
