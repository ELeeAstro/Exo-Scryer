***********************
Radiative Transfer (RT) Schemes
***********************

Overview
========

The radiative transfer module provides 1D plane-parallel RT solvers for different observing geometries.
Each scheme computes synthetic spectra for comparison with observations.

Available RT Schemes
====================

Transit/Transmission Spectroscopy
----------------------------------

Computes the wavelength-dependent transit depth (or effective radius) for primary transit observations.
This geometry is used for exoplanet atmospheres observed against their host star.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     rt_scheme: transmission_1d


**Typical Use Cases:**

- Hot Jupiters transiting bright stars
- Rocky planets in the habitable zone
- Atmospheric characterization via transmission features


Emission Spectroscopy
----------------------

Computes thermal emission spectra from the day-side or night-side of a planet.
Two emission modes are supported: planet emission and brown dwarf emission.

**YAML Configuration (Dayside Emission):**

.. code-block:: yaml

   physics:
     rt_scheme: emission_1d
     emission_mode: dayside


**YAML Configuration (Brown Dwarf):**

.. code-block:: yaml

   physics:
     rt_scheme: emission_1d
     emission_mode: brown_dwarf

   params:
     - { name: R_s, dist: delta, value: 0.0, transform: identity, init: 0.0 }


**Typical Use Cases:**

- Secondary eclipse observations
- Directly imaged exoplanets
- Brown dwarfs
- Phase curve observations


Reflection/Albedo Spectroscopy
-------------------------------

Computes reflected light spectra for planets observed in scattered starlight.
This geometry is relevant for direct imaging and high-contrast observations.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     rt_scheme: albedo_1d


**Typical Use Cases:**

- Direct imaging of reflected light
- Phase curves (orbital phase variations)
- Future missions like HabEx, LUVOIR


RT Solver Details
=================

Opacity Treatment
-----------------

All RT schemes support multiple opacity types:

- **Line-by-line (LBL)**: High-resolution molecular line lists
- **Correlated-k (c-k)**: Pre-tabulated opacity distributions for faster computation
- **Rayleigh scattering**: Wavelength-dependent scattering
- **CIA (Collision-Induced Absorption)**: H₂-H₂, H₂-He pairs
- **Clouds**: Various parametric and physical cloud models


Numerical Methods
-----------------

The 1D RT schemes use:

- Plane-parallel atmosphere approximation
- Two-stream or discrete ordinates methods
- Support for both line-by-line and correlated-k opacity
- Instrument bandpass convolution


Performance Considerations
--------------------------

**Line-by-line (LBL)**:
- Higher spectral resolution
- Slower computation
- Best for high-resolution spectroscopy

**Correlated-k (c-k)**:
- Faster computation (10-100x speedup)
- Lower effective resolution
- Best for broadband photometry and low-resolution spectroscopy


Functions
=========

compute_transit_depth_1d
------------------------

Core function for transmission spectroscopy calculations.

Returns the transit depth (Rp/Rs)² as a function of wavelength.


compute_emission_spectrum_1d
-----------------------------

Core function for emission spectroscopy calculations.

Returns flux at the observer (dayside) or at 10 pc (brown dwarf).


compute_albedo_spectrum_1d
---------------------------

Core function for reflection spectroscopy calculations.

Returns geometric albedo as a function of wavelength.


See Also
========

- :doc:`tp_profiles` for temperature-pressure profile schemes
- :doc:`chemistry` for chemical abundance schemes
- :doc:`api` for complete API reference
