"""The files the computations write and the figures read.

Each results file is one dataclass below, saved as one ``.npz`` whose keys
are the dataclass's field paths (``truth.theta_training``). The writer
(``run/``) and the reader (``plots.py``) share these classes, so they cannot
disagree about what a file holds. Shapes use ``S`` training sites, ``H``
held-out sites, ``G`` map grid points, ``P`` parameters, ``N`` model
outputs per site, ``I`` IPL iterations and ``n`` grid points per axis.
"""

import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "CalibrationOptions",
    "IPLHistory",
    "MethodResults",
    "NorthAmericaPrior",
    "SiteGrids",
    "ToyResults",
    "UncertaintyMaps",
    "TruthResults",
    "load",
    "save",
]


# ── the toy experiment ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class TruthResults:
    """The synthetic truth, split into the prior's terms (``FieldDraw``)."""

    #: (S, P) and (H, P): the true parameters.
    theta_training: np.ndarray
    theta_held_out: np.ndarray
    #: (S, P): the three terms that sum to ``theta_training``.
    trend_training: np.ndarray
    smooth_variation_training: np.ndarray
    site_noise_training: np.ndarray
    #: (G, P): the trend and the smooth variation on the map grid.
    trend_map: np.ndarray
    smooth_variation_map: np.ndarray


@dataclass(frozen=True)
class MethodResults:
    """One method's posterior and its predictions."""

    #: (S, P) and (S, P, P): the posterior at the training sites.
    posterior_mean: np.ndarray
    posterior_site_covariance: np.ndarray
    #: (G, P): the smooth field's mean and SD on the map; NaN for a method
    #: that makes no map (site by site).
    map_mean: np.ndarray
    map_sd: np.ndarray
    #: (H, P): the parameters at the held-out sites.
    held_out_mean: np.ndarray
    held_out_sd: np.ndarray
    #: (P,): scores at the held-out sites (``evaluation.held_out_scores``).
    held_out_root_mean_square_error: np.ndarray
    held_out_interval_coverage: np.ndarray


@dataclass(frozen=True)
class SiteGrids:
    """Site-by-site posteriors on their grids (``SiteGridPosteriors``)."""

    #: (S, P, n) and (S, n, n).
    grid_axes: np.ndarray
    density: np.ndarray


@dataclass(frozen=True)
class IPLHistory:
    """IPL after each iteration, scored against the reference."""

    #: (I,): cumulative model runs per site.
    runs_per_site: np.ndarray
    #: (I, S, P).
    mean: np.ndarray
    sd: np.ndarray
    #: (I, P): the median over tower (sparse) sites of
    #: ``|mean - reference mean| / reference SD`` and of ``SD / reference SD``.
    median_standardized_error_towers: np.ndarray
    median_standardized_error_sparse: np.ndarray
    median_sd_ratio_towers: np.ndarray
    median_sd_ratio_sparse: np.ndarray


@dataclass(frozen=True)
class ToyResults:
    """Everything ``run/toy_experiment.py`` computes."""

    # The region (toy.domain.Region).
    training_locations: np.ndarray  # (S, 2)
    held_out_locations: np.ndarray  # (H, 2)
    map_grid_locations: np.ndarray  # (G, 2)
    map_grid_size: int
    # The observations (toy.observations).
    is_tower: np.ndarray  # (S,)
    observed: np.ndarray  # (S, N)
    noise_sd: np.ndarray  # (S, N)
    data: np.ndarray  # (S, N), 0 where unobserved
    true_model_output: np.ndarray  # (S, N), noise-free
    temperature: np.ndarray  # (S, months), the drivers
    light: np.ndarray  # (S, months)
    truth: TruthResults
    # The methods. "prior" is no calibration at all.
    prior: MethodResults
    site_by_site: MethodResults
    spatial_reference: MethodResults
    ipl: MethodResults
    site_grids: SiteGrids
    ipl_history: IPLHistory
    # The reference's draws and cost (inference.spatial_mcmc).
    reference_samples: np.ndarray  # (draws, S, P)
    reference_model_runs_per_site: int
    reference_max_r_hat: float
    reference_min_effective_sample_size: float


# ── the illustrations ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class NorthAmericaPrior:
    """Prior draws over North America (``illustrations.north_america``)."""

    #: (sites,): the site pool, and the flux-tower sites among it (the tower
    #: table's primary towers).
    site_ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    tower_site_ids: np.ndarray
    #: (nx,) and (ny,): the display raster's cell centers, projected meters
    #: (``SITE_PROJECTION``).
    grid_x: np.ndarray
    grid_y: np.ndarray
    #: (draws, ny, nx): the draws on the raster; NaN where the map is not drawn.
    draws: np.ndarray


@dataclass(frozen=True)
class CalibrationOptions:
    """Parameter values under the non-spatial options (``illustrations.calibration_options``)."""

    #: (sites,): the site pool.
    site_ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    #: (pfts,) the PFT names, and (sites,) each site's position among them.
    pft_names: np.ndarray
    pft_codes: np.ndarray
    #: (sites,): a value drawn independently at each site, and one drawn per
    #: PFT and shared by its sites.
    independent: np.ndarray
    by_pft: np.ndarray


@dataclass(frozen=True)
class UncertaintyMaps:
    """A field's prior and posterior SD over North America (``illustrations.uncertainty``)."""

    #: (sites,): the site pool, and (towers,) the tower sites with strong data.
    site_lon: np.ndarray
    site_lat: np.ndarray
    tower_lon: np.ndarray
    tower_lat: np.ndarray
    #: (nx,) and (ny,): the display raster's cell centers, projected meters.
    grid_x: np.ndarray
    grid_y: np.ndarray
    #: (ny, nx): the SDs on the raster; NaN where the map is not drawn.
    prior_sd: np.ndarray
    posterior_sd: np.ndarray


# ── reading and writing ─────────────────────────────────────────────────────


def save(result, path: Path) -> None:
    """Write a results dataclass to ``path`` (``.npz``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **_flatten(result))


def load(result_type: type, path: Path):
    """Read a results dataclass of type ``result_type`` from ``path``."""
    if not path.exists():
        raise FileNotFoundError(f"no results at {path}; README.md says which script writes it")
    with np.load(path) as arrays:
        return _unflatten(result_type, dict(arrays))


def _flatten(result, prefix: str = "") -> dict[str, np.ndarray]:
    flat = {}
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        key = prefix + field.name
        if dataclasses.is_dataclass(value):
            flat.update(_flatten(value, key + "."))
        else:
            flat[key] = np.asarray(value)
    return flat


def _unflatten(result_type: type, arrays: dict[str, np.ndarray], prefix: str = ""):
    hints = typing.get_type_hints(result_type)
    values = {}
    for field in dataclasses.fields(result_type):
        key, field_type = prefix + field.name, hints[field.name]
        if dataclasses.is_dataclass(field_type):
            values[field.name] = _unflatten(field_type, arrays, key + ".")
        elif field_type in (int, float):
            values[field.name] = field_type(arrays[key])
        else:
            values[field.name] = arrays[key]
    return result_type(**values)
