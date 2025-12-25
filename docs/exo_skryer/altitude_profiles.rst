***********************
Altitude Profiles
***********************

Overview
========

Altitude calculation schemes convert pressure levels to altitude/height coordinates
using the hypsometric equation. Different schemes handle gravitational acceleration
in different ways depending on the application.


Hypsometric (Constant Gravity)
===============================

Simplest altitude scheme assuming constant gravitational acceleration throughout the
atmosphere. Suitable for shallow atmospheres where gravity variation is negligible.

**Physics:**

Uses the barometric formula with constant scale height:

.. math::

   dz = H \\ln(p_{\\rm lower} / p_{\\rm upper})

where :math:`H = k_B T / (\\mu m_u g_{\\rm ref})`.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_alt: hypsometric

   params:
     # Required parameter
     - { name: log_10_g, dist: uniform, low: 2.0, high: 5.0, transform: logit, init: 3.5 }

     # Alternative: compute from M_p and R_p (if using these as retrieval params)
     - { name: M_p, dist: uniform, low: 0.5, high: 3.0, transform: logit, init: 1.0 }
     - { name: R_p, dist: uniform, low: 0.8, high: 1.5, transform: logit, init: 1.0 }

**Required Parameters:**

- ``log_10_g``: Log₁₀ surface gravity (cm s⁻²)

  - Typical range: 2.0 to 5.0 (100 to 100,000 cm s⁻²)
  - Gas giants: ~3.0 to 3.5 (1000-3000 cm s⁻²)
  - Super-Earths: ~3.5 to 4.0 (3000-10,000 cm s⁻²)

**Notes:**

- If ``log_10_g`` is not directly provided as a parameter, the forward model can
  compute it from ``M_p`` (planet mass in M_jup) and ``R_p`` (planet radius in R_jup)
- Constant gravity is a good approximation when atmospheric scale height << planet radius


Hypsometric Variable Gravity
=============================

Altitude scheme accounting for gravitational acceleration varying with altitude.
Uses predictor-corrector integration for accuracy.

**Physics:**

Gravity decreases with altitude in spherical geometry:

.. math::

   g(z) = g_{\\rm ref} \\left(\\frac{R_0}{R_0 + z}\\right)^2

where :math:`R_0 = R_p \\times R_{\\rm Jup}` is the planetary radius.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_alt: hypsometric_variable_g

   params:
     # Required parameters
     - { name: log_10_g, dist: uniform, low: 2.0, high: 5.0, transform: logit, init: 3.5 }
     - { name: R_p, dist: uniform, low: 0.8, high: 1.5, transform: logit, init: 1.0 }

     # Alternative: compute log_10_g from M_p and R_p
     - { name: M_p, dist: uniform, low: 0.5, high: 3.0, transform: logit, init: 1.0 }
     - { name: R_p, dist: uniform, low: 0.8, high: 1.5, transform: logit, init: 1.0 }

**Required Parameters:**

- ``log_10_g``: Log₁₀ gravity at reference radius (cm s⁻²)

  - Same range as constant gravity case

- ``R_p``: Planet radius (Jupiter radii)

  - Typical range: 0.8 to 2.0 R_jup for gas giants
  - Mini-Neptunes: 0.3 to 0.5 R_jup
  - Super-Earths: 0.1 to 0.3 R_jup

**When to Use:**

- Extended atmospheres where scale height is not << R_p
- High-altitude features (e.g., extended hydrogen coronae)
- Accurate transmission spectroscopy modeling
- Planets with low surface gravity (brown dwarfs, inflated hot Jupiters)

**Algorithm:**

Uses predictor-corrector per layer:

1. Predict dz using g at current altitude
2. Correct using g at predicted mid-layer altitude


Hypsometric Variable Gravity with Reference Pressure
=====================================================

Altitude scheme with variable gravity anchored at a specific pressure level. The
altitude grid is set such that z = 0 at the reference pressure, then integrated
both upward (decreasing pressure) and downward (increasing pressure).

**Physics:**

Same gravitational variation as ``hypsometric_variable_g``, but altitude zero-point
is defined at a reference pressure ``p_ref`` instead of at the bottom of the grid.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_alt: hypsometric_variable_g_pref

   params:
     # Required parameters
     - { name: log_10_g, dist: uniform, low: 2.0, high: 5.0, transform: logit, init: 3.5 }
     - { name: R_p, dist: uniform, low: 0.8, high: 1.5, transform: logit, init: 1.0 }
     - { name: log_10_p_ref, dist: uniform, low: -3.0, high: 2.0, transform: logit, init: 0.0 }

     # Alternative: compute log_10_g from M_p and R_p
     - { name: M_p, dist: uniform, low: 0.5, high: 3.0, transform: logit, init: 1.0 }
     - { name: R_p, dist: uniform, low: 0.8, high: 1.5, transform: logit, init: 1.0 }
     - { name: log_10_p_ref, dist: uniform, low: -3.0, high: 2.0, transform: logit, init: 0.0 }

**Required Parameters:**

- ``log_10_g``: Log₁₀ gravity at reference radius (cm s⁻²)

  - Same range as other schemes

- ``R_p``: Planet radius (Jupiter radii)

  - Same range as ``hypsometric_variable_g``

- ``log_10_p_ref``: Log₁₀ reference pressure (bar)

  - Typical range: -3.0 to 2.0 (0.001 to 100 bar)
  - Transmission spectroscopy: Often use ~0.01 to 0.1 bar (optical depth ~ 1)
  - Common choice: 0.0 (p_ref = 1 bar)

**When to Use:**

- Transmission spectroscopy where planet radius is defined at a specific pressure
- When comparing to radius measurements at a standard pressure level
- Retrievals where absolute radius at a pressure is better constrained than surface radius

**Example Use Cases:**

**Transmission Spectroscopy:**

.. code-block:: yaml

   physics:
     vert_alt: hypsometric_variable_g_pref
     rt_scheme: transmission_1d

   params:
     - { name: log_10_g, dist: uniform, low: 2.5, high: 4.0, transform: logit, init: 3.2 }
     - { name: R_p, dist: uniform, low: 0.9, high: 1.2, transform: logit, init: 1.05 }
     - { name: log_10_p_ref, dist: delta, value: -1.0 }  # Fix ref at 0.1 bar

In this case, ``R_p`` represents the radius at 0.1 bar.


Comparison of Schemes
=====================

+-----------------------------------+-------------------+------------------------+--------------------+
| Scheme                            | Gravity Treatment | Altitude Zero Point    | Best For           |
+===================================+===================+========================+====================+
| ``hypsometric``                   | Constant          | Bottom of grid         | Shallow atm.       |
+-----------------------------------+-------------------+------------------------+--------------------+
| ``hypsometric_variable_g``        | Variable with z   | Bottom of grid         | Extended atm.      |
+-----------------------------------+-------------------+------------------------+--------------------+
| ``hypsometric_variable_g_pref``   | Variable with z   | At p_ref               | Transmission spec. |
+-----------------------------------+-------------------+------------------------+--------------------+


Typical Parameter Values
=========================

**Gas Giants (Hot Jupiters):**

.. code-block:: yaml

   params:
     - { name: log_10_g, dist: uniform, low: 2.8, high: 3.8, init: 3.3 }
     - { name: R_p, dist: uniform, low: 0.9, high: 1.5, init: 1.2 }
     - { name: log_10_p_ref, dist: delta, value: -1.0 }  # 0.1 bar

**Sub-Neptunes:**

.. code-block:: yaml

   params:
     - { name: log_10_g, dist: uniform, low: 3.0, high: 4.0, init: 3.5 }
     - { name: R_p, dist: uniform, low: 0.3, high: 0.5, init: 0.4 }
     - { name: log_10_p_ref, dist: delta, value: 0.0 }  # 1 bar

**Brown Dwarfs / Directly Imaged Planets:**

.. code-block:: yaml

   params:
     - { name: log_10_g, dist: uniform, low: 3.5, high: 5.0, init: 4.5 }
     - { name: R_p, dist: uniform, low: 0.8, high: 1.2, init: 1.0 }

(Brown dwarfs typically use emission spectroscopy, so p_ref is less relevant)


Computing log_10_g from M_p and R_p
====================================

If you prefer to retrieve planet mass and radius directly, the forward model can
compute gravity:

.. math::

   g = \\frac{G M_p}{R_p^2}

   \\log_{10} g = \\log_{10}\\left(\\frac{G M_p M_{\\rm Jup}}{(R_p R_{\\rm Jup})^2}\\right)

**Example configuration:**

.. code-block:: yaml

   params:
     # Don't include log_10_g explicitly
     - { name: M_p, dist: uniform, low: 0.5, high: 2.0, transform: logit, init: 1.0 }
     - { name: R_p, dist: uniform, low: 0.9, high: 1.3, transform: logit, init: 1.1 }

The forward model will automatically compute ``log_10_g`` from ``M_p`` and ``R_p``
when evaluating the altitude profile.


See Also
========

- :doc:`tp_profiles` for temperature-pressure structure
- :doc:`mean_molecular_weight` for mean molecular weight schemes
- :doc:`radiative_transfer` for RT geometry
- :doc:`api` for function API reference
