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
    6013.22910402886 furlong_per_fortnight
