"""Draws from a spatial prior over North America, as continuous maps.

One generic parameter (on a log scale, centered at 0) at every location
``x`` of the continental grids (``continent``):

    theta(x) = slope * MAT_std(x) + field_sd * u(x),

the prior model the slides state: a trend in a spatial covariate plus a
Gaussian process. ``MAT_std`` is mean annual temperature (the site covariate
table's ``MAT``), standardized over the pool's sites and carried to each
field point from its nearest site. ``u`` is a unit-variance Gaussian
process with Matern-3/2 correlation in the chordal distance. It is drawn
exactly at the field points and shown on the display raster.

It reads the site covariate table (tracked) beyond what the grids read. It
illustrates the prior's form only: the numbers are chosen to look like a
plausible parameter map, not fitted to anything.
"""

import numpy as np
import pandas as pd
from scipy.linalg import cholesky

from sipnet_calibration.conventions import tracked_data_root

from ..results import NorthAmericaPrior
from ..toy.spatial_prior import CHOLESKY_JITTER, matern32_correlation
from .continent import ContinentGrids, flux_tower_site_ids

__all__ = ["draw_north_america_prior"]


def draw_north_america_prior(
    rng: np.random.Generator,
    grids: ContinentGrids,
    *,
    n_draws: int,
    length_scale: float,
    field_sd: float,
    trend_slope: float,
) -> NorthAmericaPrior:
    """Draw the prior ``n_draws`` times at the field points and show it on the display raster.

    ``length_scale`` is in meters.
    """
    covariate = _standardized_mean_annual_temperature(grids.site_ids)[grids.field_nearest_site]
    on_sphere = grids.field_points_on_sphere()
    correlation = matern32_correlation(on_sphere, on_sphere, length_scale)
    # Factorized in place: the matrix is about a gigabyte.
    correlation[np.diag_indices_from(correlation)] += CHOLESKY_JITTER
    lower = cholesky(correlation, lower=True, overwrite_a=True)
    smooth_variation = field_sd * (lower @ rng.standard_normal((len(on_sphere), n_draws))).T
    field = trend_slope * covariate[None, :] + smooth_variation

    return NorthAmericaPrior(
        site_ids=grids.site_ids,
        lon=grids.site_lon,
        lat=grids.site_lat,
        tower_site_ids=flux_tower_site_ids(),
        grid_x=grids.display_x,
        grid_y=grids.display_y,
        draws=grids.to_display(field),
    )


# ── helpers ─────────────────────────────────────────────────────────────────


def _standardized_mean_annual_temperature(site_ids: np.ndarray) -> np.ndarray:
    """Each site's ``MAT`` from the site covariate table, standardized over the sites given."""
    path = tracked_data_root() / "raw" / "covariates" / "site_covariates_pft_assignment.csv"
    # ``index`` is the covariate table's site id (data/raw/covariates/provenance.md).
    covariates = pd.read_csv(path, usecols=["index", "MAT"]).set_index("index")
    mean_annual_temperature = covariates.loc[site_ids, "MAT"].to_numpy()
    return (mean_annual_temperature - mean_annual_temperature.mean()) / mean_annual_temperature.std()
