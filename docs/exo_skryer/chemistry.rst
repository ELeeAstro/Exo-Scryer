***********************
Chemistry Schemes
***********************

Overview
========

The chemistry module provides functions to compute chemical abundances (volume mixing ratios)
as a function of altitude. Schemes range from simple constant abundances to full chemical
equilibrium calculations.

Available Chemistry Schemes
============================

Constant VMR
------------

Vertically uniform volume mixing ratios for all species.
Most commonly used scheme for retrievals.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_chem: constant_vmr

   params:
     - { name: log_10_f_H2O, dist: uniform, low: -9, high: -1, transform: logit, init: -3 }
     - { name: log_10_f_CO, dist: uniform, low: -9, high: -1, transform: logit, init: -5 }
     - { name: log_10_f_CO2, dist: uniform, low: -9, high: -1, transform: logit, init: -5 }
     - { name: log_10_f_CH4, dist: uniform, low: -9, high: -1, transform: logit, init: -3 }


Chemical Equilibrium (FastChem-JAX)
------------------------------------

Compute chemical equilibrium abundances using FastChem algorithm implemented in JAX.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_chem: CE_fastchem_jax


Chemical Equilibrium (Rate-based)
----------------------------------

Compute chemical equilibrium abundances using Gibbs free energy minimization.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_chem: CE_rate_jax


Quenched Chemistry (Approximation)
-----------------------------------

Apply vertical quenching approximation to chemical equilibrium abundances.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_chem: quench_approx

   params:
     - { name: log_10_p_quench, dist: uniform, low: -3, high: 2, transform: logit, init: 0.0 }


Functions
=========

constant_vmr
------------

Builder function that creates a kernel for constant VMR profiles.


build_constant_vmr_kernel
--------------------------

Low-level function to construct constant VMR chemistry kernels.


CE_fastchem_jax
---------------

Chemical equilibrium using FastChem-JAX algorithm.


CE_rate_jax
-----------

Chemical equilibrium using rate equations and Gibbs energy.


quench_approx
-------------

Vertical quenching approximation for disequilibrium chemistry.


See Also
========

- :doc:`mean_molecular_weight` for computing atmospheric μ
- :doc:`api` for complete API reference
