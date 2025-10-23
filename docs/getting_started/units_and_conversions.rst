Units and Conversions
=====================================

Once you have a ``Quantity`` object, Quantium provides a simple API for converting, inspecting, and defining units.

---------------------------------
1. Converting Units
---------------------------------

You can easily convert any ``Quantity`` to a compatible unit using the ``.to()`` method.

.. code-block:: python

    from quantium import u

    # Simple conversions
    distance = 1 * u.km
    print(distance.to(u.m))
    print(distance.to(u.cm))

    # Compound unit conversions
    velocity = (10 * u.m / u.s).to(u.km / u.h)
    print(velocity)

Output:

.. code-block::

    1000 m
    100000 cm
    36.0 km/h

---------------------------------
2. String-Based Definitions
---------------------------------

For complex or dynamically-defined units, you can pass a string expression directly to the ``u()`` object:

.. code-block:: python

    from quantium import u

    print(3 * u("m"))
    print(5 * u("m/s**2"))
    print(10 * u("kg*m/s**2"))

Output:

.. code-block::

    3 m
    5 m/s²
    10 N

--------------------------------------------------
3. Inspecting and Simplifying Units
--------------------------------------------------

Quantium gives you tools to look "under the hood" of an expression. You can independently access the **value**, **units**, and fundamental **dimensions** (in MLT notation).

.. code-block:: python

    from quantium import u

    expr = (3 * u.kg) * (2 * u.m / u.s**2) # This is a force

    # Access the numerical value
    print(f"Value: {expr.value}")

    # Access the unit object
    print(f"Units: {expr.unit}")

    # Access the base dimensions
    print(f"Dimension: {expr.dim}")

    # Printing the unit
    print(expr)


Output:

.. code-block::

    Value: 6.0
    Units: Unit(name='kg·m/s^2', scale_to_si=1.0, dim=[L^1][M^1][T^-2])
    Dimension: [L^1][M^1][T^-2]
    6.0 N

This separation of **value**, **units**, and **dimensions** helps in both debugging and documentation.