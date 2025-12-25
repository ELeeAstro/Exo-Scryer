***********************
Rayleigh Scattering
***********************

Overview
========

Rayleigh scattering provides wavelength-dependent scattering from atmospheric molecules
and particles much smaller than the wavelength of light. This is the dominant opacity
source at short wavelengths (UV to visible).

Configuration
=============

**YAML Configuration:**

.. code-block:: yaml

   physics:
     opac_ray: lbl

   opac:
     ray:
       - {species: H2}
       - {species: He}


Available Species
=================

Common Rayleigh scatterers:

- **H₂** - Molecular hydrogen
- **He** - Helium
- **H₂O** - Water vapor
- **CO₂** - Carbon dioxide
- **N₂** - Molecular nitrogen


Physical Description
=====================

Rayleigh scattering cross-section scales as λ⁻⁴:

.. math::

   σ_{\rm Ray} ∝ λ^{-4}

This creates the characteristic blue slope in transmission spectra at short wavelengths.


Typical Applications
====================

**Transmission Spectroscopy:**

- Creates distinctive slope at optical wavelengths
- Helps constrain atmospheric composition (H₂/He ratio)
- Important for rocky planets with H₂-dominated atmospheres

**Reflection Spectroscopy:**

- Dominates reflected light at short wavelengths
- Creates blue color of clear atmospheres


Functions
=========

compute_ray_opacity
-------------------

Computes Rayleigh scattering cross-sections for atmospheric gases.


zero_ray_opacity
----------------

Returns zero opacity (used when Rayleigh scattering is disabled).


See Also
========

- :doc:`opacity_cia` for collision-induced absorption
- :doc:`radiative_transfer` for RT scheme documentation
- :doc:`api` for complete API reference
