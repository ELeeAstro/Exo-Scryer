"""
sampler_blackjax_NS.py
======================

Minimal BlackJAX nested-sampling scaffold:
- build priors from cfg.params (delta, uniform, normal, lognormal)
- Gaussian log-likelihood via prep.fm and observed data
- run blackjax.nss and return samples/evidence info
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
from anesthetic import NestedSamples

from build_prepared import Prepared


def run_nested_blackjax(cfg, prep: Prepared, exp_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    ns_cfg = getattr(cfg.sampling, "blackjax_ns", None)
    if ns_cfg is None:
        raise ValueError("Missing cfg.sampling.blackjax_ns configuration.")

    num_live = int(getattr(ns_cfg, "num_live_points", 500))
    num_inner_steps = int(getattr(ns_cfg, "num_inner_steps", 20))
    num_delete = int(getattr(ns_cfg, "num_delete", max(1, num_live // 2)))
    seed = int(getattr(ns_cfg, "seed", 0))
    dlogz_stop = float(getattr(ns_cfg, "dlogz_stop", -3.0))

    # Extract prior bounds from cfg.params for uniform priors only
    prior_bounds = {}
    for param in cfg.params:
        if param.dist == "uniform":
            prior_bounds[param.name] = (float(param.low), float(param.high))

    rng_key = jax.random.PRNGKey(seed)
    rng_key, prior_key = jax.random.split(rng_key)
    particles, logprior_fn = blackjax.ns.utils.uniform_prior(prior_key, num_live, prior_bounds)


    print(f"Particle structure: {particles.keys()}")


    lam   = jnp.asarray(prep.lam)
    dlam  = jnp.asarray(prep.dlam)
    y_obs = jnp.asarray(prep.y)
    dy_obs = jnp.asarray(prep.dy)

    def loglikelihood_fn(params):
        mu = prep.fm(params)  # (N_obs,)
        sig = jnp.clip(dy_obs, 1e-300, jnp.inf)
        r = (y_obs - mu) / sig
        return -0.5 * jnp.sum(r * r + jnp.log(2 * jnp.pi) + 2 * jnp.log(sig))

    nested_sampler = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )

    print(f"[blackjax_ns] Initializing nested sampler with {num_live} live points...")
    init_fn = jax.jit(nested_sampler.init)
    step_fn = jax.jit(nested_sampler.step)

    live = init_fn(particles)
    dead = []
    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while not live.logZ_live - live.logZ <  dlogz_stop:
            rng_key, subkey = jax.random.split(rng_key, 2)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            pbar.update(num_delete)

    dead = blackjax.ns.utils.finalise(live, dead)

    rng_key, weight_key, sample_key = jax.random.split(rng_key,3)
    re_samples = blackjax.ns.utils.sample(sample_key, dead, shape = num_live)
    log_w = blackjax.ns.utils.log_weights(weight_key, dead, shape=100)
    ns_ess = blackjax.ns.utils.ess(sample_key, dead)
    logzs = jax.scipy.special.logsumexp(log_w, axis=0)

    print(f"ESS: {int(ns_ess)}")
    print(f"logZ estimate: {logzs.mean():.2f} +- {logzs.std():.2f}")

    # Build labels from prior names with proper LaTeX formatting
    def make_latex_label(name: str) -> str:
        """Convert parameter name to LaTeX-friendly label."""
        # Split on underscores
        parts = name.split('_')
        if len(parts) == 1:
            # No underscores, return as-is
            return rf"${name}$"
        elif len(parts) == 2:
            # Single underscore: treat second part as subscript
            return rf"${parts[0]}_{{{parts[1]}}}$"
        else:
            # Multiple underscores: use nested subscripts or escape
            # e.g., "log_10_f_H2O" -> "log_{10,f_{H2O}}"
            base = parts[0]
            subscript = ','.join(parts[1:])
            return rf"${base}_{{{subscript}}}$"

    labels = {name: make_latex_label(name) for name in prior_bounds.keys()}

    samples = NestedSamples(
        dead.particles,
        logL=dead.loglikelihood,
        logL_birth=dead.loglikelihood_birth,
        labels=labels,
    )

    csv_path = exp_dir / "nested_samples.csv"
    samples.to_csv(csv_path)
    print(f"[blackjax_ns] Saved nested samples to {csv_path}")

    # Extract evidence info
    evidence_info: Dict[str, Any] = {}

    # Build samples_dict from dead particles
    samples_dict: Dict[str, np.ndarray] = {}

    # Extract samples for each varying parameter
    # dead.particles is a dict with parameter names as keys
    param_names = list(prior_bounds.keys())
    for name in param_names:
        samples_dict[name] = np.asarray(re_samples[name])

    # Add fixed/delta parameters
    for param in cfg.params:
        name = param.name
        if name not in samples_dict:
            if param.dist == "delta":
                val = float(param.value)
                n_samples = len(re_samples)
                samples_dict[name] = np.full((n_samples,), val, dtype=np.float64)

    # print(f"[blackjax_ns] logZ = {evidence_info['logZ']:.2f} ± {evidence_info['logZ_err']:.2f}")
    # print(f"[blackjax_ns] ESS = {evidence_info['ESS']}")

    return samples_dict, evidence_info
