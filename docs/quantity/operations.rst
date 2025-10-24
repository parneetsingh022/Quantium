Arithmetic operations
=====================

All arithmetic operations are dimensionally-aware. The short ``100 * u.m``
syntax is used throughout for readability.

Addition / Subtraction
----------------------

Quantities must have the same physical dimension (e.g., both represent length, time, energy, etc.).
When two compatible quantities are added or subtracted, the result keeps the units of the left-hand operand (the first quantity in the expression).

.. code-block:: python

   from quantium import u

   total_dist = (1 * u.km) + (500 * u.m)
   print(total_dist)  # 1.5 m

   delta_t = (10 * u.min) - (30 * u.s)
   print(delta_t) # 9.5 min

.. note::

    This ensures intuitive and stable behavior — the left-hand operand determines the display unit,
    while the underlying computation is always performed in SI base units for numerical accuracy.

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
