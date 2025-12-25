***********************
Testing Forward Model
***********************

Overview
========

Before running a full atmospheric retrieval, it's essential to test that your forward
model is correctly configured. The ``test_forward_model.py`` script runs a single
forward model evaluation using parameter values from your configuration file.


Purpose
=======

Testing the forward model serves several purposes:

- **Verify Configuration**: Ensure all opacity files, stellar spectra, and data files are correctly specified
- **Check Parameter Ranges**: Confirm that initial parameter values produce reasonable spectra
- **Debug Issues**: Identify problems before waiting hours for a retrieval to fail
- **Validate Setup**: Test that JAX/GPU acceleration is working correctly
- **Explore Parameter Space**: Quickly see how changes to parameters affect the spectrum


Prerequisites
=============

- A complete ``retrieval_config.yaml`` configuration file
- All required opacity data files (specified in config)
- Observational data file (wavelength, half-bandwidth, flux, error, response_mode)
- Stellar spectrum file (for dayside emission and reflection)
- Python environment with exo_skryer and dependencies installed


Usage
=====

Basic Usage
-----------

.. code-block:: bash

   cd experiments/your_setup/
   python test_forward_model.py

The script automatically looks for ``retrieval_config.yaml`` in the current directory.


What the Script Does
====================

The test script performs the following steps:

1. **Load Configuration**: Read ``retrieval_config.yaml``

2. **Load Observational Data**: Read observed spectrum from data file

3. **Build Opacities**: Initialize all opacity sources (line-by-line, c-k, Rayleigh, CIA, clouds)

4. **Load Bandpass Registry**: Set up instrument response functions

5. **Read Stellar Spectrum**: Load host star spectrum (if needed)

6. **Build Forward Model**: Construct the complete forward model function

7. **Extract Parameters**: Get parameter values from config:

   - **Fixed parameters** (``dist: delta``): Use ``value`` field
   - **Free parameters**: Use ``init`` field as test value

8. **Run Forward Model**: Evaluate model at test parameters

9. **Generate Plots**: Create diagnostic plots comparing model to data

10. **Save Results**: Store outputs for further analysis


Parameter Value Selection
==========================

The script uses different parameter values depending on parameter type:

**Fixed Parameters** (``dist: delta``):

.. code-block:: yaml

   params:
     - { name: p_top, dist: delta, value: 1.0e-6 }  # Uses value=1e-6

**Free Parameters** (``dist: uniform``, etc.):

.. code-block:: yaml

   params:
     - { name: T_int, dist: uniform, low: 1000, high: 2000, init: 1400 }  # Uses init=1400


Output Files
============

The script generates three output files:

**1. Diagnostic Plots** (``forward_model_test.png/pdf``)

Two-panel figure:

- **Top panel**: Model spectrum vs observed data

  - High-resolution model (thin line)
  - Binned model (points connected by line)
  - Observed data with error bars

- **Bottom panel**: Residuals (model - data) / error

  - Shows goodness of fit in units of σ
  - Dashed lines at 0, ±1σ

**2. Numerical Output** (``forward_model_test_output.npz``)

Compressed NumPy archive containing:

- ``lam_obs``: Observed wavelengths (µm)
- ``dlam_obs``: Half-bandwidths (µm)
- ``y_obs``: Observed flux/depth
- ``dy_obs``: Observational uncertainties
- ``model_binned``: Model convolved to observed wavelengths
- ``lam_hires``: High-resolution wavelength grid (µm)
- ``model_hires``: High-resolution model spectrum
- ``theta``: Dictionary of parameter values used

**3. Console Output**

Prints diagnostic information:

.. code-block:: text

   ============================================================
   FORWARD MODEL TEST
   ============================================================
   Experiment dir: /path/to/experiment
   Config file: /path/to/retrieval_config.yaml

   [1/5] Loading observed data...
     Found 25 data points
     Wavelength range: 1.100 - 11.800 µm

   [2/5] Building opacities...
     Master grid: 50000 wavelength points

   [3/5] Loading bandpass registry...

   [4/5] Building forward model...
     Forward model built successfully

   [5/5] Extracting parameters from config...

   Parameter values:
   ------------------------------------------------------------
     R_p                  =      1.00000  (delta)
     M_p                  =      1.00000  (delta)
     T_int                =   1400.00000  (uniform)
     log_10_k_ir          =     -1.00000  (uniform)
     ...
   ------------------------------------------------------------

   Running forward model...
     Success! Generated spectrum with 25 binned points
              and 50000 hi-res points

   Generating plots...

   Plots saved:
     forward_model_test.png
     forward_model_test.pdf
     forward_model_test_output.npz

   ============================================================
   FORWARD MODEL TEST COMPLETE
   ============================================================


Interpreting Results
====================

Good Forward Model Test
------------------------

Signs that your forward model is working correctly:

✓ **No Errors**: Script completes without exceptions

✓ **Reasonable Spectrum**: Model flux/depth values are similar order of magnitude to data

✓ **Spectral Features**: Model shows expected molecular absorption/emission features

✓ **Finite Values**: No NaN or infinity values in output

✓ **Smooth Spectrum**: High-resolution spectrum is continuous (no discontinuities)

✓ **Residuals**: Model-data residuals are not wildly large (< 10σ expected at this stage)


Common Issues and Solutions
============================

Issue: "Opacity file not found"
--------------------------------

**Cause**: Path in config doesn't point to opacity data

**Solution**: Check paths in ``opac`` section are correct relative to experiment directory

.. code-block:: yaml

   opac:
     line:
       - {species: H2O, path: ../../opac_data/line/H2O_ExoMol.npz}


Issue: Model spectrum is all zeros or NaN
------------------------------------------

**Cause**: Parameter values out of physical range or opacity data issue

**Solution**:

- Check parameter ``init`` values are reasonable
- Verify opacity files are not corrupted
- Check temperature profile doesn't have negative values
- Ensure pressure grid (``p_top``, ``p_bot``) is physically reasonable


Issue: "CUDA out of memory"
----------------------------

**Cause**: High-resolution model too large for GPU memory

**Solution**:

- Reduce wavelength range to just your observed data
- Use correlated-k opacities instead of line-by-line
- Set ``platform: cpu`` in runtime section for testing


Issue: Model doesn't match data at all
---------------------------------------

**Cause**: Expected for initial test - parameter values are just guesses

**Solution**:

- Check if model is in reasonable range (e.g., transit depth 0-10%, not 1000%)
- Adjust ``init`` values in config to be closer to expected values
- This is normal - retrieval will find best-fit values


Customizing the Test Script
============================

You can modify ``test_forward_model.py`` to:

**Test Specific Parameter Values**:

.. code-block:: python

   # After line ~174, manually override parameters:
   theta["T_int"] = 1500.0  # Test different temperature
   theta["log_10_f_H2O"] = -3.0  # Test different H2O abundance

**Plot Specific Wavelength Ranges**:

.. code-block:: python

   # After line ~203, add xlim:
   ax1.set_xlim(7.0, 12.0)  # Focus on mid-IR

**Compare Multiple Models**:

.. code-block:: python

   # Run forward model multiple times with different parameters
   for T_int in [1200, 1400, 1600]:
       theta["T_int"] = T_int
       result = fm(theta)
       ax1.plot(lam_obs, result["binned"], label=f"T_int={T_int}")


Integration with Retrieval Workflow
====================================

Recommended workflow:

1. **Create initial config**: Set up ``retrieval_config.yaml`` with reasonable parameter ranges

2. **Test forward model**: Run ``test_forward_model.py``

   .. code-block:: bash

      python test_forward_model.py

3. **Inspect results**: Check that model is reasonable and outputs are finite

4. **Adjust config**: Modify parameter ranges or physics settings based on test results

5. **Re-test**: Iterate until forward model produces sensible output

6. **Run retrieval**: Launch full retrieval with confidence

   .. code-block:: bash

      python run_retrieval.py

7. **Post-process**: Use post-processing scripts to analyze retrieval results


Example: Testing Different T-P Profiles
========================================

To test how different temperature-pressure profiles affect your spectrum:

.. code-block:: yaml

   # Test 1: Isothermal
   physics:
     vert_Tp: isothermal

   params:
     - { name: T_iso, dist: uniform, low: 1000, high: 2000, init: 1400 }

Run test, examine output, then try:

.. code-block:: yaml

   # Test 2: Milne (self-luminous)
   physics:
     vert_Tp: Milne

   params:
     - { name: T_int, dist: uniform, low: 1000, high: 2000, init: 1400 }
     - { name: log_10_k_ir, dist: uniform, low: -2, high: 0, init: -1.0 }

Compare the resulting spectra to understand which profile is more appropriate.


Performance Benchmarking
=========================

The test script can benchmark forward model evaluation time:

.. code-block:: python

   import time

   # After building forward model:
   start = time.time()
   for _ in range(100):
       result = fm(theta)
   elapsed = time.time() - start
   print(f"Average forward model time: {elapsed/100:.3f} seconds")

Typical evaluation times:

- **Line-by-line, CPU**: 1-10 seconds
- **Line-by-line, GPU**: 0.1-1 seconds
- **Correlated-k, CPU**: 0.1-1 seconds
- **Correlated-k, GPU**: 0.01-0.1 seconds


See Also
========

- :doc:`yaml_configuration` for configuration file format
- :doc:`postprocessing` for analyzing retrieval results
- :doc:`radiative_transfer` for forward model details
- :doc:`opacity_line` and :doc:`opacity_ck` for opacity options
