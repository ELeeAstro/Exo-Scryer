***********************
Correlated-k Opacities
***********************

Overview
========

Correlated-k (c-k) opacities provide a fast approximation to line-by-line opacities by
grouping similar absorption strengths into bins (g-points). This dramatically speeds up
radiative transfer calculations while maintaining reasonable accuracy for low-resolution
spectroscopy.

Configuration
=============

**YAML Configuration:**

.. code-block:: yaml

   physics:
     opac_line: ck

   opac:
     ck: True
     ck_mix: RORR  # or PRAS

     line:
       - {species: H2O, path: ../../opac_data/ck/H2O_R100.npz}
       - {species: CO, path: ../../opac_data/ck/CO_R100.npz}
       - {species: CO2, path: ../../opac_data/ck/CO2_R100.npz}
       - {species: CH4, path: ../../opac_data/ck/CH4_R100.npz}


Mixing Schemes
==============

Random Overlap with Resort and Rebin (RORR)
--------------------------------------------

The RORR method assumes random overlap of absorption lines between different species.

**YAML Configuration:**

.. code-block:: yaml

   opac:
     ck_mix: RORR


Pre-mixed Absorption and Scattering (PRAS)
-------------------------------------------

Alternative mixing scheme for combining c-k opacities.

**YAML Configuration:**

.. code-block:: yaml

   opac:
     ck_mix: PRAS


Data Format
===========

C-k opacity files store pre-binned absorption coefficients.

**Required Arrays:**

- ``wavelength``: Wavelength grid (µm)
- ``temperature``: Temperature grid (K)
- ``pressure``: Pressure grid (bar)
- ``g_points``: Quadrature points in cumulative probability
- ``g_weights``: Weights for each g-point
- ``kcoeff``: Absorption coefficients (cm² g⁻¹)


Performance Considerations
===========================

**Advantages:**

- 10-100x faster than line-by-line
- Much smaller file sizes
- Suitable for broadband photometry and R < 1000 spectroscopy

**Disadvantages:**

- Lower effective spectral resolution
- Cannot resolve individual absorption lines
- Requires pre-computed c-k tables


Functions
=========

compute_ck_opacity
------------------

Computes opacity from correlated-k tables.


zero_ck_opacity
---------------

Returns zero opacity (used when c-k opacities are disabled).


See Also
========

- :doc:`opacity_line` for line-by-line opacities (slower but more accurate)
- :doc:`radiative_transfer` for RT scheme documentation
- :doc:`api` for complete API reference
