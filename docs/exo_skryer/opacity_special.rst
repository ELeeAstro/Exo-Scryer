***********************
Special Opacities
***********************

Overview
========

Special opacity sources that don't fit into the standard molecular line, Rayleigh, or CIA
categories. Currently includes bound-free and free-free absorption from H⁻ (important for
cool stars and brown dwarfs).

H⁻ (H-minus) Opacity
====================

The H⁻ ion provides significant continuum opacity in cool atmospheres (T < 10,000 K).
It arises from:

- **Bound-free absorption**: H⁻ → H + e⁻
- **Free-free absorption**: H⁻ + hν → H⁻ (excited)

Configuration
=============

**YAML Configuration:**

.. code-block:: yaml

   physics:
     opac_special: hminus


Physical Description
====================

H⁻ opacity is important in:

- Cool stars (< 7000 K)
- Brown dwarfs
- Cool gas giant atmospheres

The opacity depends on:

- Temperature
- Electron density (requires chemical equilibrium calculation)
- Wavelength (continuum absorption)


Wavelength Dependence
=====================

H⁻ has characteristic absorption edges:

- **Bound-free edge**: ~1.6 µm (near-infrared)
- **Free-free**: Continuum absorption throughout visible/IR


Functions
=========

compute_hminus_opacity
----------------------

Computes H⁻ bound-free and free-free absorption.


compute_special_opacity
-----------------------

General interface for special opacity sources (currently H⁻).


See Also
========

- :doc:`opacity_line` for molecular line opacities
- :doc:`chemistry` for chemical equilibrium (needed for electron density)
- :doc:`api` for complete API reference
