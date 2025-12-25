***********************
Line-by-Line Opacities
***********************

Overview
========

Line-by-line (LBL) opacities provide the most accurate spectral resolution by using pre-computed
molecular line lists. This opacity source is essential for high-resolution spectroscopy and
resolving individual absorption lines.

Configuration
=============

**YAML Configuration:**

.. code-block:: yaml

   physics:
     opac_line: lbl

   opac:
     line:
       - {species: H2O, path: ../../opac_data/lbl/H2O_dnu_1.npz}
       - {species: CO, path: ../../opac_data/lbl/CO_dnu_1.npz}
       - {species: CO2, path: ../../opac_data/lbl/CO2_dnu_1.npz}
       - {species: CH4, path: ../../opac_data/lbl/CH4_dnu_1.npz}
       - {species: NH3, path: ../../opac_data/lbl/NH3_dnu_1.npz}


Data Format
===========

LBL opacity files are pre-tabulated opacity cross-sections stored as NumPy `.npz` archives.

**Required Arrays:**

- ``wavelength``: Wavelength grid (µm)
- ``temperature``: Temperature grid (K)
- ``pressure``: Pressure grid (bar)
- ``sigma``: Cross-section array (cm² molecule⁻¹)


Available Species
=================

Common molecular absorbers:

- **H₂O** - Water vapor
- **CO** - Carbon monoxide
- **CO₂** - Carbon dioxide
- **CH₄** - Methane
- **NH₃** - Ammonia
- **HCN** - Hydrogen cyanide
- **C₂H₂** - Acetylene
- **Na** - Sodium
- **K** - Potassium
- **TiO** - Titanium oxide
- **VO** - Vanadium oxide


Performance Considerations
===========================

**Advantages:**

- High spectral resolution
- Accurate line shapes
- Essential for R > 10,000 spectroscopy

**Disadvantages:**

- Large memory footprint
- Slower computation (10-100x slower than c-k)
- Requires large opacity files


Functions
=========

compute_line_opacity
--------------------

Computes molecular absorption cross-sections from pre-tabulated line lists.


zero_line_opacity
-----------------

Returns zero opacity (used when line opacities are disabled).


See Also
========

- :doc:`opacity_ck` for correlated-k opacities (faster alternative)
- :doc:`radiative_transfer` for RT scheme documentation
- :doc:`api` for complete API reference
