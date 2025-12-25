***********************
Post-Processing Scripts
***********************

Overview
========

After running an atmospheric retrieval, several post-processing scripts are available to
analyze and visualize the results. These scripts are typically located in individual
experiment directories and read the ``posterior.nc`` file produced by the retrieval.


Prerequisites
=============

All post-processing scripts require:

- A completed retrieval with ``posterior.nc`` file in the experiment directory
- The original ``retrieval_config.yaml`` configuration file
- Python environment with exo_skryer and dependencies installed


Best-Fit Spectrum Plots
========================

Emission Spectrum (Dayside/Brown Dwarf)
----------------------------------------

**Script:** ``bestfit_em_plot.py``

Plots the median emission spectrum with credible intervals, comparing model predictions
to observed data. Produces multiple output formats:

- Flux ratio (F_p/F_star) vs wavelength
- Planet flux vs wavelength
- Brightness temperature vs wavelength
- Zoomed plots for specific wavelength ranges

**Usage:**

.. code-block:: bash

   python bestfit_em_plot.py --config retrieval_config.yaml

**Options:**

- ``--config``: Path to YAML configuration file (required)
- ``--outname``: Output filename prefix (default: "model_emission")
- ``--max-samples``: Maximum posterior samples to evaluate (default: 2000)
- ``--seed``: Random seed for reproducibility (default: 123)
- ``--no-show``: Don't display plots (save only)

**Outputs:**

- ``model_emission.png/pdf``: Flux ratio plot
- ``model_emission_zoom.png/pdf``: Zoomed flux ratio (7-12 µm)
- ``model_emission_planet_flux.png/pdf``: Planet flux
- ``model_emission_brightness_temperature.png/pdf``: Brightness temperature
- ``model_emission_quantiles.npz``: Numerical results (percentiles)


Transmission Spectrum
----------------------

**Script:** ``bestfit_plot.py``

Plots the median transmission spectrum with credible intervals for transiting exoplanet
observations.

**Usage:**

.. code-block:: bash

   python bestfit_plot.py --config retrieval_config.yaml

**Options:**

Similar to ``bestfit_em_plot.py``

**Outputs:**

- ``model_transmission.png/pdf``: Transit depth vs wavelength
- ``model_transmission_quantiles.npz``: Numerical results


Temperature-Pressure Profiles
==============================

**Script:** ``T_p_plot.py``

Plots the median temperature-pressure profile with 95% credible intervals from posterior
samples.

**Usage:**

.. code-block:: bash

   python T_p_plot.py --config retrieval_config.yaml

**Options:**

- ``--config``: Path to YAML configuration file (required)
- ``--outname``: Output filename prefix (default: "Tp_band")
- ``--max-samples``: Maximum samples to evaluate (default: 2000)
- ``--seed``: Random seed (default: 123)
- ``--no-show``: Don't display plot

**Outputs:**

- ``Tp_band.png/pdf``: Temperature vs pressure with credible band
- ``Tp_band_quantiles.npz``: T(p) percentiles at each pressure level


Contribution Functions
======================

**Script:** ``contri_func_plot.py``

Computes and visualizes the emission contribution function, showing which atmospheric
layers contribute most to the observed spectrum at each wavelength.

**Usage:**

.. code-block:: bash

   python contri_func_plot.py --config retrieval_config.yaml

**Options:**

- ``--config``: Path to YAML configuration file (required)
- ``--outname``: Output filename prefix (default: "contribution_function")
- ``--no-show``: Don't display plots
- ``--wavelengths``: Specific wavelengths in µm for 1D profiles (e.g., ``--wavelengths 1.5 3.0 8.0``)

**Outputs:**

- ``contribution_function_1d.png``: 1D profiles at selected wavelengths
- ``contribution_function_2d.png``: 2D contour map (wavelength vs pressure)
- ``contribution_function_2d_zoom.png``: Zoomed 2D map (7-12 µm)
- ``contribution_function_data.npz``: Numerical contribution function array

**Configuration:**

The script temporarily enables ``contri_func: true`` in the physics section when running
the forward model. This computes the normalized contribution function:

.. math::

   C(p, λ) = \\frac{dB/dT \\cdot dT/dp \\cdot e^{-τ}}{F_{\\rm total}}


Corner Plots (Parameter Correlations)
======================================

Basic Corner Plot
-----------------

**Script:** ``corner_plot.py``

Creates corner plots showing 1D and 2D marginalized posterior distributions for
selected parameters.

**Usage:**

.. code-block:: python

   # Edit the script to select variables:
   var_names = ["R_p", "T_iso", "log_10_f_H2O"]
   scales = ["linear", "linear", "linear"]

   # Run:
   python corner_plot.py

**Customization:**

Edit the script to modify:

- ``var_names``: List of parameter names to plot
- ``scales``: Scale for each parameter ("linear" or "log")
- ``quants``: Quantiles to show (default: [0.16, 0.5, 0.84])

**Outputs:**

- Interactive matplotlib window with corner plot
- Can be saved manually or modified to auto-save


Anesthetic Corner Plot
----------------------

**Script:** ``corner_anesthetic.py``

Alternative corner plot using the anesthetic package, which provides additional
features like KDE-based contours and better handling of large sample sets.

**Usage:**

.. code-block:: python

   python corner_anesthetic.py

**Features:**

- Kernel density estimation for smooth contours
- Automatic detection of parameter ranges
- Built-in support for nested sampling outputs


Cloud Profile Plots
====================

Refractive Index Profiles
--------------------------

**Script:** ``plot_nk_profile.py``

Plots the vertical profile of cloud particle complex refractive index (n and k)
for direct n-k cloud models.

**Usage:**

.. code-block:: bash

   python plot_nk_profile.py --config retrieval_config.yaml

**Outputs:**

- Vertical profiles of real (n) and imaginary (k) parts of refractive index
- Wavelength-dependent values at each atmospheric layer


Cloud Property Profiles
------------------------

**Script:** ``plot_cloud_profile.py``

Plots vertical profiles of cloud particle properties (size distribution, number density,
optical depth) for physical cloud models (F18, direct_nk).

**Usage:**

.. code-block:: bash

   python plot_cloud_profile.py --config retrieval_config.yaml

**Outputs:**

- Particle radius vs pressure
- Number density vs pressure
- Cloud optical depth vs wavelength


Typical Post-Processing Workflow
=================================

After a retrieval completes, a typical analysis workflow is:

1. **Verify convergence**: Check sampling diagnostics in posterior.nc

   .. code-block:: python

      import arviz as az
      idata = az.from_netcdf("posterior.nc")
      print(az.summary(idata))

2. **Plot best-fit spectrum**:

   .. code-block:: bash

      python bestfit_em_plot.py --config retrieval_config.yaml

3. **Plot T-P profile**:

   .. code-block:: bash

      python T_p_plot.py --config retrieval_config.yaml

4. **Plot contribution function** (for emission):

   .. code-block:: bash

      python contri_func_plot.py --config retrieval_config.yaml

5. **Create corner plot** to examine parameter degeneracies:

   .. code-block:: python

      # Edit corner_plot.py to select parameters
      python corner_plot.py

6. **Plot cloud profiles** (if using clouds):

   .. code-block:: bash

      python plot_cloud_profile.py --config retrieval_config.yaml


Common Output Files
===================

All post-processing scripts save both figures and numerical data:

**Figure Formats:**

- ``.png``: Raster graphics (300 DPI) for presentations
- ``.pdf``: Vector graphics for publications

**Data Formats:**

- ``.npz``: Compressed NumPy arrays with quantiles and raw data

  .. code-block:: python

     import numpy as np
     data = np.load("model_emission_quantiles.npz")
     print(data.files)  # List available arrays
     wavelength = data["lam"]
     median_flux = data["depth_p50"]


Tips and Best Practices
========================

**Sample Size:**

- Default ``--max-samples=2000`` is usually sufficient for smooth credible intervals
- Increase for very detailed plots or if credible bands look noisy
- Decrease for faster prototyping

**Random Seeds:**

- Use consistent ``--seed`` values for reproducible plots
- Different seeds sample different posterior draws but should give similar results

**Wavelength Ranges:**

- Zoom plots (7-12 µm) focus on mid-infrared features common in JWST data
- Edit scripts to customize wavelength ranges for your data

**Parameter Selection for Corner Plots:**

- Start with 3-5 key parameters to avoid cluttered plots
- Common choices: ``R_p``, ``T_iso``/``T_int``, major molecular abundances
- Check for degeneracies between cloud and chemistry parameters


See Also
========

- :doc:`yaml_configuration` for retrieval setup
- :doc:`radiative_transfer` for forward model details
- :doc:`tp_profiles` for T-P profile schemes
