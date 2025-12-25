***********************
Cloud Opacities
***********************

Overview
========

Cloud opacities provide absorption and scattering from condensate particles. Clouds can
dramatically alter spectra by muting absorption features and adding wavelength-dependent
slopes. Several parametric and physical cloud models are available.

Available Cloud Models
======================

Grey Cloud
----------

Wavelength-independent opacity deck at a specified pressure level.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     opac_cloud: grey_cloud

   params:
     - { name: log_10_p_cloud, dist: uniform, low: -3, high: 2, transform: logit, init: 0.0 }
     - { name: kappa_cloud, dist: uniform, low: 0.0, high: 1e5, transform: logit, init: 1e3 }


Power-Law Cloud
---------------

Simple parametric cloud with power-law wavelength dependence.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     opac_cloud: powerlaw_cloud

   params:
     - { name: log_10_p_cloud, dist: uniform, low: -3, high: 2, transform: logit, init: 0.0 }
     - { name: log_10_kappa_0, dist: uniform, low: -2, high: 5, transform: logit, init: 2.0 }
     - { name: gamma, dist: uniform, low: -5.0, high: 5.0, transform: logit, init: 0.0 }


Physically-Motivated Clouds (F18)
----------------------------------

Physically-motivated cloud model based on Mie scattering and particle size distributions.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     opac_cloud: F18_cloud

   params:
     - { name: log_10_p_cloud, dist: uniform, low: -3, high: 2, transform: logit, init: 0.0 }
     - { name: f_sed, dist: uniform, low: 0.1, high: 10.0, transform: logit, init: 1.0 }
     - { name: log_10_Kzz, dist: uniform, low: 5, high: 10, transform: logit, init: 7.0 }


Direct n-k Cloud
----------------

Cloud model using directly specified refractive indices.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     opac_cloud: direct_nk

   params:
     - { name: log_10_p_cloud, dist: uniform, low: -3, high: 2, transform: logit, init: 0.0 }
     - { name: log_10_Kzz, dist: uniform, low: 5, high: 10, transform: logit, init: 7.0 }


Cloud Parameters
================

Common Parameters
-----------------

- **log_10_p_cloud**: Base pressure of cloud deck (bar)
- **kappa_cloud** or **log_10_kappa_0**: Cloud opacity strength (cm² g⁻¹)
- **f_sed**: Sedimentation efficiency (dimensionless)
- **log_10_Kzz**: Vertical mixing coefficient (cm² s⁻¹)
- **gamma**: Power-law index for wavelength dependence


Wavelength Dependence
---------------------

Different cloud models produce different spectral slopes:

- **Grey**: Flat (no wavelength dependence)
- **Power-law**: κ ∝ λ^γ
- **Mie scattering**: Physical wavelength dependence from particle size


Functions
=========

grey_cloud
----------

Simple grey (wavelength-independent) cloud deck.


powerlaw_cloud
--------------

Parametric cloud with power-law wavelength scaling.


F18_cloud
---------

Physically-motivated cloud model with Mie scattering.


direct_nk
---------

Cloud model using tabulated refractive indices.


kk_n_from_k_wavenumber_cached
------------------------------

Kramers-Kronig relation for computing real refractive index from imaginary part.


zero_cloud_opacity
------------------

Returns zero opacity (used when clouds are disabled).


See Also
========

- :doc:`radiative_transfer` for RT scheme documentation
- :doc:`api` for complete API reference
