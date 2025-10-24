String formatting
=================

Quantities support custom string formatting via the built-in ``__format__``
implementation. By default quantities print in their current unit. Use the
``:si`` specifier to force conversion to SI units first.

.. code-block:: python

   from quantium import u

   v = 1000 * (u.cm / u.s)

   # Default format (current unit)
   print(f"{v}")      # 1000 cm/s
   print(f"{v:native}") # 1000 cm/s

   # Force SI format
   print(f"{v:si}")   # 10 m/s

The format specifiers accepted are: ``''`` (empty), ``'unit'``/``'u'`` (current unit)
and ``'si'`` (convert and display SI). Any other specifier raises ``ValueError``.
