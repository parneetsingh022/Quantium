Getting Started
=======================

.. toctree::
   :maxdepth: 2
   :hidden:

   Units <units>
   Quantity <quantity>

Welcome to the **Quantium Getting Started Guide** — a practical walkthrough designed to help you explore the power and simplicity of Quantium through real examples.

If you haven’t already installed Quantium, see the :doc:`Installation & Setup <../index>` section on the main page.


--------------------------------------
1. Your First Calculation
--------------------------------------

Let’s start with something simple: computing force using **Newton’s second law**.

.. code-block:: python

   from quantium import u

   mass = 10 * u.kg
   acceleration = 5 * u.m / (2 * u.s**2)

   force = mass * acceleration
   print(force)

Output:

.. code-block::

   25 N

Quantium automatically infers that the resulting unit is **Newtons**, ensuring dimensional consistency.


--------------------------------------------------
2. Working with Units and Conversion
--------------------------------------------------

You can inspect or convert units easily:

.. code-block:: python

   from quantium import u

   distance = 1 * u.km
   print(distance.to(u.m))     # 1000 m
   print(distance.to(u.cm))    # 100000 cm

You can even convert compound units:

.. code-block:: python

   velocity = (10 * u.m / u.s).to(u.km / u.h)
   print(velocity)  # 36.0 km/h

Quantium also allows you to define units using **string expressions** rather than attributes:

.. code-block:: python

   from quantium import u

   print(3 * u("m"))            # 3 m
   print(5 * u("m/s**2"))       # 5 m/s²
   print(10 * u("kg*m/s**2"))   # 10 N


--------------------------------------------------
3. Automatic Dimensional Checking
--------------------------------------------------

Quantium prevents you from performing physically invalid operations.

.. code-block:: python

   from quantium import u

   try:
       invalid = (5 * u.kg) + (10 * u.m)
   except ValueError as e:
       print(e)

Output:

.. code-block::

   Incompatible units: 'kg' and 'm'

This helps you catch mistakes early and ensures all computations remain physically meaningful.


--------------------------------------
4. Derived Quantities
--------------------------------------

Quantium supports derived physical quantities automatically.  
For instance, you can compute **kinetic energy**:

.. code-block:: python

   from quantium import u

   mass = 2 * u.kg
   velocity = 3 * u.m / u.s

   energy = 0.5 * mass * velocity**2
   print(energy)  # 9 J

Quantium recognizes that ``kg * (m/s)^2`` equals **Joules (J)**.


--------------------------------------
5. Chained Operations
--------------------------------------

Units are fully preserved through complex mathematical expressions.

.. code-block:: python

   from quantium import u

   h = 12 * u.m
   g = 9.81 * u.m / u.s**2
   m = 70 * u.kg

   potential_energy = m * g * h
   print(potential_energy)         # 8234.4 J
   print(potential_energy.to(u.kJ))  # Convert to kilojoules

Output:

.. code-block::

   8240.4 J
   8.2404 kJ


--------------------------------------
6. Defining Custom Units
--------------------------------------

Quantium lets you define your own units when needed.

.. code-block:: python

   from quantium import u

   u.define("furlong_per_fortnight", 0.0001663 , u.m / u.s)

   speed = 10 * u.furlong_per_fortnight
   print(speed.to(u.m / u.s))

This makes Quantium adaptable to educational, engineering, or domain-specific use cases.


.. --------------------------------------
.. 7. Inspecting and Simplifying Units
.. --------------------------------------

.. Quantium gives you tools to inspect or simplify expressions:

.. .. code-block:: python

..    from quantium import u

..    expr = (3 * u.kg) * (2 * u.m / u.s**2)
..    print(expr.units)       # N
..    print(expr.dimension)   # [M][L][T^-2]
..    print(expr.value)       # 6.0

.. This separation of **value**, **units**, and **dimensions** helps in both debugging and documentation.


--------------------------------------
7. Advanced Example — Ohm’s Law
--------------------------------------

Let’s compute electrical resistance from voltage and current.

.. code-block:: python

   from quantium import u

   voltage = 12 * u.V
   current = 2 * u.A

   resistance = voltage / current
   print(resistance)  # 6 Ω

Quantium automatically simplifies ``V / A`` into **Ohms (Ω)**.


--------------------------------------
8. Summary
--------------------------------------

By now, you’ve seen how Quantium:

- Handles units automatically and safely.  
- Detects invalid or inconsistent operations.  
- Converts and simplifies units seamlessly.  
- Supports both attribute-style and string-based unit definitions.  

For a deeper dive, explore:

- **API Reference:** Detailed documentation of `Quantity`, `Unit`, and mathematical operations.
- **Examples:** Advanced real-world scenarios and physics simulations.
- **Contributing:** Help expand Quantium by contributing your ideas and improvements.

Quantium makes unit-safe computation **simple, expressive, and reliable** — so you can focus on the science, not the syntax.
