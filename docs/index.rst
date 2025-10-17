.. quantium documentation master file, created by
   sphinx-quickstart on Wed Oct  8 19:59:28 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quantium
=======================

.. raw:: html

   <div class="intro-container">
      <img src="_static/quantium_logo.png" alt="Quantium Logo">
      <div>
        <p><strong>Welcome to Quantium</strong> — a lightweight Python library for mathematical and scientific computations 
        with units. It enables dimensional analysis and unit-safe calculations through a simple, dependency-minimal design. 
        NumPy integration is planned for future releases..</p>
      </div>
   </div>



Getting Started
---------------

To install Quantium, simply run:

.. code-block:: bash

   pip install quantium

Once installed, you can start performing unit-safe calculations:

.. code-block:: python

   from quantium import get_unit

   m = get_unit('m')
   s = get_unit('s')

   d = 10 @ m
   t = 1 @ s
   v = d / t

   print(v)

**Output**

.. code-block:: none

   10 m/s


Contributing
------------

We welcome contributions from the community!  
To get started, See the `CONTRIBUTING guide <https://github.com/parneetsingh022/quantium/blob/main/CONTRIBUTING.md>`_ for details.

----

License
-------

Quantium is distributed under the MIT License.  
See the `CHANGELOG <https://github.com/parneetsingh022/quantium/blob/main/CHANGELOG.md>`_ for version history and recent updates.
