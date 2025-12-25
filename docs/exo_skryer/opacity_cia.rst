***********************
Collision-Induced Absorption (CIA)
***********************

Overview
========

Collision-induced absorption (CIA) arises from transient dipole moments created during
molecular collisions. This is particularly important for H₂-H₂ and H₂-He pairs in
hydrogen-dominated atmospheres, providing significant continuum opacity at near-infrared
wavelengths.

Configuration
=============

**YAML Configuration:**

.. code-block:: yaml

   physics:
     opac_cia: lbl

   opac:
     cia:
       - {species: H2-H2, path: ../../opac_data/cia/H2-H2_2011.npz}
       - {species: H2-He, path: ../../opac_data/cia/H2-He_2011.npz}


Available CIA Pairs
===================

Common collision pairs:

- **H₂-H₂** - Hydrogen self-broadening (dominant in hot Jupiters)
- **H₂-He** - Hydrogen-helium collisions
- **H₂-CH₄** - Hydrogen-methane collisions
- **N₂-N₂** - Nitrogen self-broadening (Earth-like planets)
- **N₂-H₂** - Nitrogen-hydrogen collisions


Physical Description
====================

CIA opacity depends on:

- Wavelength
- Temperature
- Number densities of both collision partners

The opacity coefficient is:

.. math::

   κ_{\rm CIA} = n_1 \, n_2 \, σ_{\rm CIA}(λ, T)

where n₁ and n₂ are the number densities of the collision partners.


Data Format
===========

CIA opacity files contain absorption coefficients.

**Required Arrays:**

- ``wavelength``: Wavelength grid (µm)
- ``temperature``: Temperature grid (K)
- ``alpha``: CIA coefficient (cm⁵ molecule⁻²)


Typical Applications
====================

**Hot Jupiters:**

- H₂-H₂ and H₂-He CIA provides continuum opacity
- Important for thermal structure and emission spectra
- Affects near-IR colors

**Brown Dwarfs:**

- CIA broadens pressure-broadened wings
- Affects energy balance


Functions
=========

compute_cia_opacity
-------------------

Computes collision-induced absorption from tabulated coefficients.


zero_cia_opacity
----------------

Returns zero opacity (used when CIA is disabled).


See Also
========

- :doc:`opacity_rayleigh` for Rayleigh scattering
- :doc:`radiative_transfer` for RT scheme documentation
- :doc:`api` for complete API reference
