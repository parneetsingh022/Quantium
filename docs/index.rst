.. quantium documentation master file, created by
   sphinx-quickstart on Wed Oct  8 19:59:28 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation & Setup
=======================

.. toctree::
   :maxdepth: 2
   :caption: Overview
   :hidden:

   Installation & Setup <self>
   Getting Started <getting_started/index>
   units/index
   quantity/index

.. raw:: html

   <div class="intro-container" style="background-color: #0669b4ff;">
      <!-- <img src="_static/quantium_logo.png" alt="Quantium Logo"> -->
      <div>
         <div>
         <p><strong>Welcome to Quantium</strong> —  a lightweight Python library for unit-safe scientific and mathematical computation. It combines a clean, dependency-minimal architecture with a powerful system for dimensional analysis — ensuring that every calculation you perform respects physical consistency.</p>
         </div>

         <!-- GitHub Buttons -->
         <div style="margin-top: 15px;">
            <a class="github-button" href="https://github.com/parneetsingh022/quantium" data-icon="octicon-star" data-show-count="true" aria-label="Star parneetsingh022/quantium on GitHub">Star</a>
            <a class="github-button" href="https://github.com/parneetsingh022/quantium/fork" data-icon="octicon-repo-forked" aria-label="Fork parneetsingh022/quantium on GitHub">Fork</a>
            <a class="github-button" href="https://github.com/parneetsingh022/quantium" data-icon="octicon-mark-github" aria-label="View Quantium on GitHub">View on GitHub</a>
         </div>
      </div>

      <!-- GitHub Buttons Script -->
      <script async defer src="https://buttons.github.io/buttons.js"></script>
   </div>




Installation & Setup
--------------------

Quantium can be installed from the Python Package Index (PyPI):

.. code-block:: bash

   pip install quantium

After installation, verify that Quantium is correctly installed by checking its version:

.. code-block:: python

   import quantium
   print("Quantium version:", quantium.__version__)

To make sure Quantium is ready to use, open a Python shell and run:

.. code-block:: python

   >>> from quantium import u
   >>> (10 * u.kg) * (5 * u.m) / (2 * u.s**2)
   25 N

To update Quantium to the latest version:

.. code-block:: bash

   pip install --upgrade quantium

Requirements
--------------------

Quantium is built to work seamlessly in modern environments and is compatible with  
current development tools and workflows.

Quantium currently supports **Python 3.10 and above**.

Contributing
--------------------

We welcome contributions from the community!  
To get started, See the `CONTRIBUTING guide <https://github.com/parneetsingh022/quantium/blob/main/CONTRIBUTING.md>`_ for details.

----

License
--------------------

Quantium is distributed under the MIT License.  
See the `CHANGELOG <https://github.com/parneetsingh022/quantium/blob/main/CHANGELOG.md>`_ for version history and recent updates.
