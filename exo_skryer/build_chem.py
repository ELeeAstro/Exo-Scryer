"""
build_chem.py
=============
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

__all__ = [
    'infer_trace_species',
    'infer_log10_vmr_keys',
    'validate_log10_vmr_params',
    'prepare_chemistry_kernel',
    'load_gibbs_if_needed'
]


def _extract_species_list(block) -> list[str]:
    if not block:
        return []
    if isinstance(block, bool):
        return []
    names: list[str] = []
    try:
        iterator = iter(block)
    except TypeError:
        iterator = iter((block,))
    for item in iterator:
        name = getattr(item, "species", item)
        names.append(str(name).strip())
    return names


def _append_unique(seq: list[str], name: str) -> None:
    name = str(name).strip()
    if not name:
        return
    if name not in seq:
        seq.append(name)


def infer_trace_species(
    cfg,
    line_opac_scheme_str: str,
    ray_opac_scheme_str: str,
    cia_opac_scheme_str: str,
    special_opac_scheme_str: str,
) -> tuple[str, ...]:
    required: list[str] = []

    def add_many(names: Iterable[str]) -> None:
        for n in names:
            _append_unique(required, n)

    if line_opac_scheme_str.lower() == "lbl":
        add_many(_extract_species_list(getattr(cfg.opac, "line", None)))
    elif line_opac_scheme_str.lower() == "ck":
        ck_mode = getattr(cfg.opac, "ck", None)
        if isinstance(ck_mode, bool):
            ck_block = getattr(cfg.opac, "line", None)
        else:
            ck_block = ck_mode
        add_many(_extract_species_list(ck_block))

    if ray_opac_scheme_str.lower() in ("lbl", "ck"):
        add_many(_extract_species_list(getattr(cfg.opac, "ray", None)))

    if cia_opac_scheme_str.lower() in ("lbl", "ck"):
        for cia_name in _extract_species_list(getattr(cfg.opac, "cia", None)):
            if cia_name == "H-":
                _append_unique(required, "H-")
                continue
            parts = cia_name.split("-")
            if len(parts) == 2:
                _append_unique(required, parts[0])
                _append_unique(required, parts[1])

    if special_opac_scheme_str.lower() not in ("none", "off", "false", "0"):
        for cia_name in _extract_species_list(getattr(cfg.opac, "cia", None)):
            if cia_name == "H-":
                _append_unique(required, "H-")

    trace_species = tuple(s for s in required if s not in ("H2", "He"))
    return trace_species


def infer_log10_vmr_keys(trace_species: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"log_10_f_{s}" for s in trace_species)


def validate_log10_vmr_params(cfg, trace_species: tuple[str, ...]) -> None:
    cfg_param_names = {p.name for p in cfg.params}
    missing = [s for s in trace_species if f"log_10_f_{s}" not in cfg_param_names]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing required VMR parameters for: {joined}. "
            f"Add `log_10_f_<species>` entries to cfg.params."
        )


def prepare_chemistry_kernel(cfg, chemistry_kernel, opacity_schemes: dict):
    """Prepare and validate chemistry kernel with inferred species.

    This function infers the required trace species from the opacity configuration,
    validates that necessary parameters are present, and returns an optimized
    chemistry kernel ready for use in the forward model.

    Parameters
    ----------
    cfg : config object
        Configuration object containing params and opac settings.
    chemistry_kernel : callable
        The base chemistry kernel function from vert_chem (e.g., constant_vmr,
        CE_fastchem_jax, CE_rate_jax).
    opacity_schemes : dict
        Dictionary containing opacity scheme strings with keys:
        'line_opac', 'ray_opac', 'cia_opac', 'special_opac'.

    Returns
    -------
    chemistry_kernel : callable
        The prepared chemistry kernel, potentially optimized for JIT compilation.
    trace_species : tuple of str
        Tuple of trace species names inferred from the opacity configuration.

    Notes
    -----
    For constant VMR mode, this function validates required parameters and builds
    an optimized JIT-friendly kernel using `constant_vmr` from vert_chem.
    """
    from .vert_chem import constant_vmr

    # Infer required species from opacity configuration
    trace_species = infer_trace_species(
        cfg,
        line_opac_scheme_str=opacity_schemes['line_opac'],
        ray_opac_scheme_str=opacity_schemes['ray_opac'],
        cia_opac_scheme_str=opacity_schemes['cia_opac'],
        special_opac_scheme_str=opacity_schemes['special_opac'],
    )

    # For constant VMR: validate and build optimized kernel
    if chemistry_kernel is constant_vmr:
        validate_log10_vmr_params(cfg, trace_species)
        chemistry_kernel = constant_vmr(trace_species)

    return chemistry_kernel, trace_species


def load_gibbs_if_needed(cfg: Any, exp_dir: Path) -> None:
    """Load Gibbs free energy tables if chemical equilibrium chemistry requires it.

    This function checks if the configured chemistry scheme requires Gibbs free
    energy data (RateJAX chemical equilibrium modes) and loads the JANAF tables
    if needed. If data is already loaded or not required, it returns immediately.

    Parameters
    ----------
    cfg : config object
        Parsed YAML configuration object with `cfg.physics.vert_chem` attribute.
    exp_dir : `~pathlib.Path`
        Experiment directory used to resolve relative paths to JANAF data.

    Raises
    ------
    ValueError
        If chemical equilibrium mode is selected but JANAF data path is not
        specified in `cfg.data.janaf`.

    Notes
    -----
    Only relevant when ``physics.vert_chem`` is set to one of:
    - 'rate_ce'
    - 'rate_jax'
    - 'ce_rate_jax'

    The JANAF data path can be specified relative to the experiment directory
    or as an absolute path in the configuration.

    Examples
    --------
    >>> load_gibbs_if_needed(cfg, exp_dir)
    [info] Loading Gibbs free energy tables from /path/to/JANAF_data
    [info] Gibbs cache loaded: 123 species
    """
    phys = getattr(cfg, "physics", None)
    if phys is None:
        return

    vert_chem_raw = getattr(phys, "vert_chem", None)
    if vert_chem_raw is None:
        return

    vert_chem_name = str(vert_chem_raw).lower()
    if vert_chem_name not in ("rate_ce", "rate_jax", "ce_rate_jax"):
        return

    from rate_jax import is_gibbs_cache_loaded, load_gibbs_cache

    if is_gibbs_cache_loaded():
        print("[info] Gibbs cache already loaded")
        return

    data_cfg = getattr(cfg, "data", None)
    janaf_rel_path = getattr(data_cfg, "janaf", None) if data_cfg is not None else None
    if janaf_rel_path is None:
        raise ValueError(
            "JANAF data path not found in config. Please add 'janaf: path/to/JANAF_data' "
            "under 'data:' section in your YAML config."
        )

    janaf_path = (
        str(exp_dir / janaf_rel_path)
        if not Path(janaf_rel_path).is_absolute()
        else janaf_rel_path
    )

    print(f"[info] Loading Gibbs free energy tables from {janaf_path}")
    gibbs = load_gibbs_cache(janaf_path)
    print(f"[info] Gibbs cache loaded: {len(gibbs.data)} species")
