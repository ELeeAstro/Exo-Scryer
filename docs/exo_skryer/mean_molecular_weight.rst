***********************
Mean Molecular Weight (μ)
***********************

Overview
========

The mean molecular weight module provides functions to compute the average molecular weight
of the atmospheric mixture. This is critical for converting pressure to number density
and for computing altitude profiles.

Available Schemes
=================

Constant μ
----------

Use a single constant value for mean molecular weight throughout the atmosphere.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_mu: constant_mu

   params:
     - { name: mu, dist: uniform, low: 2.0, high: 3.0, transform: logit, init: 2.3 }


Dynamic μ from Chemistry
-------------------------

Compute mean molecular weight from chemical abundances at each layer.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_mu: dynamic


Functions
=========

constant_mu
-----------

Returns a constant μ profile for all atmospheric layers.


compute_mu
----------

Computes μ from volume mixing ratios and molecular masses.


See Also
========

- :doc:`chemistry` for chemical abundance schemes
- :doc:`api` for complete API reference
