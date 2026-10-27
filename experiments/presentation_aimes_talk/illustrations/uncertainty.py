"""How data shrink a spatial parameter's uncertainty over North America.

The prior is the Gaussian process part of the North America prior
(``north_america``): ``theta(x) = trend(x) + u(x)`` with
``u ~ GP(0, field_sd^2 R)``, Matern-3/2 correlation ``R`` in the chordal
distance. The trend's coefficient is treated as known, so the prior
uncertainty is the process's alone: ``field_sd``, the same everywhere.

**The data.** Every site of the pool measures the parameter at its location,
``m_s = theta(x_s) + e_s``, independently, with ``e_s ~ N(0, site_noise_sd^2)``
at most sites (weak data, such as remote sensing) and
``N(0, tower_noise_sd^2)`` at a few flux-tower sites (strong data). A
"measurement of theta" stands in for what a site's own data say about its
parameter.

**The posterior uncertainty** is exact Gaussian process regression: with
``K = field_sd^2 R(X, X) + diag(noise^2)`` over the sites ``X``,

    var(theta(g) | m) = field_sd^2 - k_gX K^{-1} k_Xg,

which depends on where the sites are and how noisy their data are, never
on the measured values, so no synthetic data are needed. It is computed at
the field points and shown on the display raster (``continent``).

**Which towers.** A few of the pool's flux-tower sites, spread out so that
each one's effect is visible: farthest-point sampling, starting from the
tower nearest the towers' median location, then repeatedly adding the
tower farthest from those already chosen.
"""

import numpy as np
from scipy.linalg import cho_factor, solve_triangular

from ..results import UncertaintyMaps
from ..toy.spatial_prior import matern32_correlation
from .continent import ContinentGrids, flux_tower_site_ids, points_on_sphere

__all__ = ["compute_uncertainty_maps", "spread_out_towers"]

#: Field points per block of the posterior variance computation, which holds
#: a (sites x block) matrix at a time.
FIELD_POINTS_PER_BLOCK = 1000


def compute_uncertainty_maps(
    grids: ContinentGrids,
    *,
    length_scale: float,
    field_sd: float,
    n_towers: int,
    tower_noise_sd: float,
    site_noise_sd: float,
) -> UncertaintyMaps:
    """The prior and posterior SD of the field, on the display raster (``length_scale`` in meters)."""
    towers = spread_out_towers(grids, flux_tower_site_ids(), n_towers)
    noise_sd = np.full(len(grids.site_ids), site_noise_sd)
    noise_sd[towers] = tower_noise_sd

    posterior_sd = _posterior_sd(
        points_on_sphere(grids.site_lon, grids.site_lat),
        noise_sd,
        grids.field_points_on_sphere(),
        length_scale=length_scale,
        field_sd=field_sd,
    )
    prior_sd = np.full(len(grids.field_points), field_sd)
    return UncertaintyMaps(
        site_lon=grids.site_lon,
        site_lat=grids.site_lat,
        tower_lon=grids.site_lon[towers],
        tower_lat=grids.site_lat[towers],
        grid_x=grids.display_x,
        grid_y=grids.display_y,
        prior_sd=grids.to_display(prior_sd),
        posterior_sd=grids.to_display(posterior_sd),
    )


def spread_out_towers(grids: ContinentGrids, tower_site_ids: np.ndarray, n_towers: int) -> np.ndarray:
    """Positions in ``grids.site_ids`` of ``n_towers`` spread-out tower sites (module docstring)."""
    candidates = np.flatnonzero(np.isin(grids.site_ids, tower_site_ids))
    locations = np.column_stack([grids.site_x[candidates], grids.site_y[candidates]])
    first = np.argmin(np.linalg.norm(locations - np.median(locations, axis=0), axis=1))
    chosen = [first]
    distance_to_chosen = np.linalg.norm(locations - locations[first], axis=1)
    while len(chosen) < n_towers:
        farthest = int(np.argmax(distance_to_chosen))
        chosen.append(farthest)
        distance_to_chosen = np.minimum(distance_to_chosen, np.linalg.norm(locations - locations[farthest], axis=1))
    return candidates[np.sort(chosen)]


# ── helpers ─────────────────────────────────────────────────────────────────


def _posterior_sd(
    site_points: np.ndarray,
    noise_sd: np.ndarray,
    target_points: np.ndarray,
    *,
    length_scale: float,
    field_sd: float,
) -> np.ndarray:
    """The posterior SD of the field at ``target_points`` (module docstring)."""
    data_covariance = field_sd**2 * matern32_correlation(site_points, site_points, length_scale)
    data_covariance[np.diag_indices_from(data_covariance)] += noise_sd**2
    lower, _ = cho_factor(data_covariance, lower=True, overwrite_a=True)

    variance = np.empty(len(target_points))
    for start in range(0, len(target_points), FIELD_POINTS_PER_BLOCK):
        block = slice(start, start + FIELD_POINTS_PER_BLOCK)
        cross_covariance = field_sd**2 * matern32_correlation(site_points, target_points[block], length_scale)
        # k_gX K^{-1} k_Xg = |L^{-1} k_Xg|^2, column by column.
        whitened = solve_triangular(lower, cross_covariance, lower=True)
        variance[block] = field_sd**2 - np.sum(whitened**2, axis=0)
    return np.sqrt(np.maximum(variance, 0.0))
