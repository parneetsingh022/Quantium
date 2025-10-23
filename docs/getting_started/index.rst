.. _getting_started:

Getting Started
=======================

Welcome to the **Quantium Getting Started Guide**! This guide will walk you through the core features of Quantium with practical, hands-on examples.

If you haven’t already installed Quantium, see the :doc:`Installation & Setup <../index>` section on the main page.

This guide is divided into the following sections:

.. toctree::
   :maxdepth: 2

   units_and_conversions
   real_world_examples
   advanced_usage
   next_steps


--------------------------------------
1. Your First Calculation
--------------------------------------

Let’s start with something simple: computing force using Newton’s second law (:math:`F = ma`).

.. code-block:: python

    from quantium import u

    mass = 10 * u.kg
    acceleration = 5 * u.m / (2 * u.s**2)

    force = mass * acceleration
    print(force)

Output:

.. code-block::

    25 N

Notice how Quantium automatically infers that the resulting unit is **Newtons**, ensuring dimensional consistency from the start.

--------------------------------------------------
2. Automatic Dimensional Checking
--------------------------------------------------

Quantium's primary goal is to prevent you from performing physically invalid operations. For example, you cannot add **mass** to **length**.

.. code-block:: python

    from quantium import u

    try:
        invalid = (5 * u.kg) + (10 * u.m)
    except ValueError as e:
        print(e)

Output:

.. code-block::

    Incompatible units: 'kg' and 'm'

This automatic checking helps you catch mistakes early, long before they can corrupt your data or simulations, ensuring all computations remain physically meaningful.