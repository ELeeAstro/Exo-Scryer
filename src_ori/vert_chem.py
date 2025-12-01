"""
vert_chem.py
============

Overview:
    Vertical chemistry profiles for atmospheric models.

Notes:
    Species names in config parameters (after stripping f_ or log_10_f_ prefix)
    must match opacity table species names exactly (including capitalization).

Sections to complete:
    - Usage
    - Key Functions
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

solar_h2 = 0.5
solar_he = 0.085114
solar_h2_he = solar_h2 + solar_he


def constant_vmr(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """
    Create constant VMR profiles from parameters.

    Species names are taken directly from parameter keys after stripping prefixes.
    User must ensure these match opacity table species names exactly.
    """
    del p_lay, T_lay  # unused but kept for consistent signature

    vmr: Dict[str, jnp.ndarray] = {}
    for k, v in params.items():
        if k.startswith("log_10_f_"):
            species = k[len("log_10_f_"):]
            vmr[species] = 10.0 ** jnp.asarray(v)
        elif k.startswith("f_"):
            species = k[len("f_"):]
            vmr[species] = jnp.asarray(v)

    trace_values = list(vmr.values())
    if trace_values:
        total_trace_vmr = jnp.sum(jnp.stack(trace_values))
    else:
        total_trace_vmr = jnp.asarray(0.0)
    background_vmr = 1.0 - total_trace_vmr

    vmr["H2"] = background_vmr * solar_h2 / solar_h2_he
    vmr["He"] = background_vmr * solar_he / solar_h2_he

    vmr_lay = {species: jnp.full((nlay,), value) for species, value in vmr.items()}
    return vmr_lay


def chemical_equilibrium(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    del p_lay, T_lay, params, nlay
    raise NotImplementedError("chemical_equilibrium is not implemented yet.")
