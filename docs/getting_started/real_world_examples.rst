Real-World Examples
=====================

Quantium shines when applied to real-world scientific, engineering, and even everyday problems.

--------------------------------------
1. Physics: Kinetic Energy
--------------------------------------

Quantium supports derived physical quantities automatically. For instance, you can compute kinetic energy.

.. math::
   E_k = \frac{1}{2}mv^2

.. code-block:: python

    from quantium import u

    mass = 2 * u.kg
    velocity = 3 * u.m / u.s

    energy = 0.5 * mass * velocity**2
    print(energy)

Quantium recognizes that ``kg * (m/s)^2`` equals **Joules (J)**.

Output:

.. code-block::

    9.0 J

--------------------------------------
2. Physics: Potential Energy
--------------------------------------

Units are fully preserved through complex mathematical expressions, like calculating **gravitational potential energy**.

.. math::
   E_g = mgh

.. code-block:: python

    from quantium import u

    h = 12 * u.m
    g = 9.81 * (u.m / u.s**2)
    m = 70 * u.kg

    potential_energy = m * g * h
    print(potential_energy)
    print(potential_energy.to(u.kJ)) # Convert to kilojoules

Output:

.. code-block::

    8240.4 J
    8.2404 kJ


--------------------------------------
3. Electrical Engineering: Ohm’s Law
--------------------------------------

Let’s compute electrical resistance from voltage and current using **Ohm's Law**.

.. math::

    V = IR

.. code-block:: python

    from quantium import u

    voltage = 12 * u.V
    current = 2 * u.A

    resistance = voltage / current
    print(resistance)

Quantium automatically simplifies ``V / A`` into **Ohms (Ω)**.

Output:

.. code-block::

    6.0 Ω

--------------------------------------
4. Healthcare: Medical Dosage
--------------------------------------

Unit safety is critical in medicine. Imagine a drug dose specified as **15 mg per kg of body weight**.

.. code-block:: python

    from quantium import u

    patient_mass = 75 * u.kg
    dose_rate = 15 * (u.mg / u.kg)

    required_dose = patient_mass * dose_rate
    print(required_dose)

    # You can also convert to a different mass unit, like grams
    print(required_dose.to(u.g))

Output:

.. code-block::

    1125.0 mg
    1.125 g

Quantium prevents a user from, for example, accidentally multiplying by a mass in *pounds* without conversion, or by *patient height*, which would raise a ``ValueError``.

--------------------------------------
5. Mechanical Engineering: Pressure
--------------------------------------

Calculating pressure involves multiple derived units. Let's find the pressure exerted by a 100 Newton force on a 25 cm² area.

.. math::

    P = \frac{F}{A}

.. code-block:: python

    from quantium import u

    force = 100 * u.N
    area = 25 * u.cm**2

    pressure = force / area

    # Auto detects Pa symbol and scale i.e. kPa
    print(pressure)

    # In standard si unit
    print(pressure.si)

    # Convert to a different scale
    print(pressure.to('uPa'))


Output:

.. code-block::

    40 kPa
    40000 Pa
    40000000000 µPa

.. --------------------------------------
.. 6. Everyday Life: Fuel Efficiency
.. --------------------------------------

.. You can easily convert between different cultural conventions, such as fuel efficiency.

.. .. code-block:: python

..     from quantium import u

..     # Define a US gallon
..     u.define("gallon", 3.78541, u.L)

..     efficiency_mpg = 35 * u.mile / u.gallon

..     # Convert to the international standard (Liters per 100km)
..     # Note: L/100km is an inverse unit, so we take the reciprocal
..     efficiency_L_per_100km = (100 * u.km) / efficiency_mpg
..     print(efficiency_L_per_100km.to(u.L))

.. Output:

.. .. code-block::

..     6.72 L