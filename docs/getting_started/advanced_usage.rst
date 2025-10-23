Advanced Usage
================

Quantium is also extensible, allowing you to define your own units or integrate with other scientific computing libraries.

--------------------------------------
1. Defining Custom Units
--------------------------------------

Quantium lets you define your own units when needed for domain-specific or even historical or humorous use cases.

.. code-block:: python

    from quantium import u

    # Define a new unit based on its SI equivalent
    u.define("furlong_per_fortnight", 0.0001663 , u.m / u.s)

    speed = 10 * u.furlong_per_fortnight
    print(speed)

    # You can now convert back and forth
    print(speed.to(u.m / u.s))
    print((1 * u.m / u.s).to(u.furlong_per_fortnight))

Output:

.. code-block::

    10.0 furlong_per_fortnight
    0.001663 m/s
    6013.23 furlong_per_fortnight

--------------------------------------
2. Working with NumPy Arrays
--------------------------------------

Quantium ``Quantity`` objects can wrap **NumPy arrays**, allowing you to perform unit-aware, vectorized calculations.

.. code-block:: python

    from quantium import u
    import numpy as np

    # Create an array of distances in meters
    distances_m = np.array([100, 500, 1000]) * u.m

    # Create a time
    time = 10 * u.s

    # Perform a vectorized operation
    speeds = distances_m / time
    print(speeds)

    # You can convert the entire array
    speeds_kmh = speeds.to(u.km / u.h)
    print(speeds_kmh)

Output:

.. code-block::

    [ 10.  50. 100.] m/s
    [ 36. 180. 360.] km/h