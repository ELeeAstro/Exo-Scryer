"""
sampler_pymultinest_NS.py
==========================
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
from pathlib import Path

import numpy as np

from .build_prepared import Prepared

# Try to import PyMultiNest
try:
    import pymultinest
    from pymultinest.analyse import Analyzer
    PYMULTINEST_AVAILABLE = True
except ImportError:
    PYMULTINEST_AVAILABLE = False
    pymultinest = None
    Analyzer = None

__all__ = [
    "build_prior_transform_pymultinest",
    "build_loglikelihood_pymultinest",
    "run_nested_pymultinest"
]


def build_prior_transform_pymultinest(cfg) -> Tuple[callable, List[str]]:
    """
    Build PyMultiNest prior transform from cfg.params.

    Parameters
    ----------
    cfg : object
        Configuration object with cfg.params list

    Returns
    -------
    prior_transform : callable
        Function with signature (cube, ndim, nparams) that modifies cube in-place
    param_names : List[str]
        Ordered list of non-delta parameter names
    """
    from scipy.stats import norm, lognorm

    # Extract non-delta parameters
    params_cfg = [p for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]
    param_names = [p.name for p in params_cfg]

    def prior_transform(cube, ndim, nparams):
        """
        PyMultiNest prior transform: maps unit cube to physical parameters.

        Modifies cube in-place (PyMultiNest requirement).
        """
        # Clip to avoid edge cases in inverse CDF
        eps = 1e-10

        for i, p in enumerate(params_cfg):
            u = np.clip(cube[i], eps, 1.0 - eps)
            dist_name = str(getattr(p, "dist", "")).lower()

            if dist_name == "uniform":
                low = float(getattr(p, "low"))
                high = float(getattr(p, "high"))
                cube[i] = low + u * (high - low)

            elif dist_name in ("gaussian", "normal"):
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                cube[i] = norm.ppf(u, loc=mu, scale=sigma)

            elif dist_name == "lognormal":
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                cube[i] = lognorm.ppf(u, s=sigma, scale=np.exp(mu))

            else:
                raise ValueError(f"Unsupported distribution '{dist_name}' for parameter '{p.name}'")

    return prior_transform, param_names


def build_loglikelihood_pymultinest(cfg, prep: Prepared, param_names: List[str]) -> callable:
    """
    Build PyMultiNest log-likelihood function.

    Implements the same split-normal likelihood as JAXNS and BlackJAX NS,
    but uses NumPy operations (except for the JAX forward model).

    Parameters
    ----------
    cfg : object
        Configuration object
    prep : Prepared
        Prepared model bundle with forward model and observed data
    param_names : List[str]
        Ordered list of parameter names

    Returns
    -------
    loglikelihood : callable
        Function with signature (cube, ndim, nparams) returning log-likelihood
    """
    # Observed data - convert to NumPy for efficient NumPy operations
    y_obs = np.asarray(prep.y)
    dy_obs_p = np.asarray(prep.dy_p)
    dy_obs_m = np.asarray(prep.dy_m)

    def loglikelihood(cube, ndim, nparams):
        """
        PyMultiNest log-likelihood: split-normal likelihood.

        Parameters
        ----------
        cube : array
            Parameter values in constrained space (after prior transform)
        ndim : int
            Number of dimensions
        nparams : int
            Number of parameters

        Returns
        -------
        logL : float
            Log-likelihood value
        """
        try:
            # Build parameter dictionary from cube values
            theta_dict = {param_names[i]: float(cube[i]) for i in range(ndim)}

            # Call forward model (JAX) and convert to NumPy
            mu = np.asarray(prep.fm(theta_dict))  # (N,)
            r = y_obs - mu  # residuals

            # Split-normal likelihood using NumPy operations
            c = theta_dict.get("c", -99.0)  # log10(sigma_jit)
            sig_jit = 10.0**c
            sig_jit2 = sig_jit * sig_jit

            # Inflate BOTH sides in quadrature
            sigp_eff = np.sqrt(dy_obs_p**2 + sig_jit2)
            sigm_eff = np.sqrt(dy_obs_m**2 + sig_jit2)

            # Choose side for exponent
            sig_eff = np.where(r >= 0.0, sigp_eff, sigm_eff)

            # Normalisation must use the SAME effective scales
            norm = np.clip(sigm_eff + sigp_eff, 1e-300, np.inf)
            sig_eff = np.clip(sig_eff, 1e-300, np.inf)

            logC = 0.5 * np.log(2.0 / np.pi) - np.log(norm)
            logL = np.sum(logC - 0.5 * (r / sig_eff) ** 2)

            # Convert to Python float
            result = float(logL)

            # Handle non-finite values
            if not np.isfinite(result):
                return -1e100

            return result

        except Exception as e:
            print(f"[PyMultiNest] Likelihood evaluation error: {e}")
            return -1e100

    return loglikelihood


def run_nested_pymultinest(
    cfg,
    prep: Prepared,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run PyMultiNest nested sampling.

    Parameters
    ----------
    cfg : object
        Configuration with cfg.sampling.pymultinest settings
    prep : Prepared
        Prepared model bundle
    exp_dir : Path
        Output directory

    Returns
    -------
    samples_dict : Dict[str, np.ndarray]
        Posterior samples for each parameter
    evidence_info : Dict[str, Any]
        Evidence and diagnostic information
    """
    if not PYMULTINEST_AVAILABLE:
        raise ImportError(
            "PyMultiNest is not installed. Install with:\n"
            "  pip install pymultinest\n"
            "Note: This also requires the MultiNest library.\n"
            "Recommended installation via conda:\n"
            "  conda install -c conda-forge multinest pymultinest\n"
            "See: https://johannesbuchner.github.io/PyMultiNest/"
        )

    mn_cfg = cfg.sampling.pymultinest

    # Extract configuration with defaults
    n_live = int(getattr(mn_cfg, "n_live_points", 500))
    evidence_tol = float(getattr(mn_cfg, "evidence_tolerance", 0.5))
    sampling_eff = float(getattr(mn_cfg, "sampling_efficiency", 0.8))
    multimodal = bool(getattr(mn_cfg, "multimodal", False))
    const_eff = bool(getattr(mn_cfg, "const_efficiency_mode", False))
    importance_ns = bool(getattr(mn_cfg, "importance_nested_sampling", True))
    resume = bool(getattr(mn_cfg, "resume", False))
    verbose = bool(getattr(mn_cfg, "verbose", True))
    seed = int(getattr(mn_cfg, "seed", -1))
    max_modes = int(getattr(mn_cfg, "max_modes", 100))

    # Setup output directory
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_basename = str(exp_dir / "mn_")

    # Build prior and likelihood from cfg (not prep!)
    prior_fn, param_names = build_prior_transform_pymultinest(cfg)
    loglike_fn = build_loglikelihood_pymultinest(cfg, prep, param_names)

    ndim = len(param_names)

    print(f"[PyMultiNest] Running nested sampling...")
    print(f"[PyMultiNest] Free parameters: {ndim}")
    print(f"[PyMultiNest] Parameter names: {param_names}")
    print(f"[PyMultiNest] Live points: {n_live}")
    print(f"[PyMultiNest] Evidence tolerance: {evidence_tol}")
    print(f"[PyMultiNest] Sampling efficiency: {sampling_eff}")

    # Run PyMultiNest
    pymultinest.run(
        loglike_fn,
        prior_fn,
        ndim,
        outputfiles_basename=output_basename,
        n_live_points=n_live,
        evidence_tolerance=evidence_tol,
        sampling_efficiency=sampling_eff,
        multimodal=multimodal,
        const_efficiency_mode=const_eff,
        importance_nested_sampling=importance_ns,
        resume=resume,
        verbose=verbose,
        seed=seed,
        max_modes=max_modes,
    )

    print(f"[PyMultiNest] Sampling complete. Loading results...")

    # Load results using Analyzer
    analyzer = Analyzer(ndim, outputfiles_basename=output_basename)
    stats = analyzer.get_stats()

    # Extract evidence information
    evidence_info: Dict[str, Any] = {
        "logZ": float(stats["global evidence"]),
        "logZ_err": float(stats["global evidence error"]),
        "sampler": "pymultinest",
        "n_live": n_live,
        "output_basename": output_basename,
    }

    print(f"[PyMultiNest] Evidence: {evidence_info['logZ']:.2f} ± {evidence_info['logZ_err']:.2f}")

    # Extract equal-weighted posterior samples
    samples = analyzer.get_equal_weighted_posterior()[:, :-1]  # Drop logL column
    n_samples = samples.shape[0]

    print(f"[PyMultiNest] Posterior samples: {n_samples}")

    # Build samples_dict (same as JAXNS/BlackJAX)
    samples_dict: Dict[str, np.ndarray] = {}

    # Add free parameters
    for i, name in enumerate(param_names):
        samples_dict[name] = samples[:, i]

    # Add fixed/delta parameters (same as JAXNS/BlackJAX)
    for param in cfg.params:
        name = param.name
        if name not in samples_dict:
            dist_name = str(getattr(param, "dist", "")).lower()
            if dist_name == "delta":
                val = getattr(param, "value", getattr(param, "init", None))
                if val is not None:
                    samples_dict[name] = np.full(
                        (n_samples,),
                        float(val),
                        dtype=np.float64,
                    )

    return samples_dict, evidence_info
