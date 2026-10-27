"""Draws from a spatial prior over the real North American site pool.

One generic parameter (on a log scale, centered at 0) at each of the pool's
sites ``x``:

    theta(x) = slope * MAT_std(x) + field_sd * u(x) + site_noise_sd * eta(x),

where ``MAT_std`` is the site's mean annual temperature standardized over
the pool (the site covariate table's ``MAT``), ``u`` is a unit-variance
Gaussian process with Matern-3/2 correlation in the chordal distance
between sites (the straight line through the Earth, which is a valid
distance for this kernel on the sphere and close to the great-circle
distance at these length scales), and ``eta`` is independent at each site.

It reads the site table and the covariate table, both tracked inputs, and
the tower table to mark the flux-tower sites. It illustrates the prior's
form only: the numbers are chosen to look like a plausible parameter map,
not fitted to anything.
"""

import numpy as np
import pandas as pd
from scipy.linalg import cholesky

from sipnet_calibration.conventions import tracked_data_root
from sipnet_calibration.net_ecosystem_exchange.towers import read_tower_table
from sipnet_calibration.sites import load_sites

from ..results import NorthAmericaPrior
from ..toy.spatial_prior import CHOLESKY_JITTER, matern32_correlation

__all__ = ["EARTH_RADIUS", "draw_north_america_prior"]

#: The Earth's mean radius, kilometers.
EARTH_RADIUS = 6371.0


def draw_north_america_prior(
    rng: np.random.Generator,
    *,
    n_draws: int,
    length_scale_km: float,
    field_sd: float,
    trend_slope: float,
    site_noise_sd: float,
) -> NorthAmericaPrior:
    """Draw the prior ``n_draws`` times over every site of the pool."""
    site_table = load_sites()
    site_ids = site_table["site_id"].to_numpy()
    lon, lat = site_table["lon"].to_numpy(), site_table["lat"].to_numpy()

    mean_annual_temperature = _mean_annual_temperature(site_ids)
    standardized = (mean_annual_temperature - mean_annual_temperature.mean()) / mean_annual_temperature.std()
    trend = trend_slope * standardized

    points = EARTH_RADIUS * _unit_vectors(lon, lat)
    correlation = matern32_correlation(points, points, length_scale_km)
    # Factorized in place: the 8000 x 8000 matrix is half a gigabyte.
    correlation[np.diag_indices_from(correlation)] += CHOLESKY_JITTER
    lower = cholesky(correlation, lower=True, overwrite_a=True)
    smooth_variation = field_sd * (lower @ rng.standard_normal((len(site_ids), n_draws))).T
    site_noise = site_noise_sd * rng.standard_normal((n_draws, len(site_ids)))

    towers = read_tower_table()
    tower_site_ids = np.unique(towers.loc[towers["primary"], "site_id"].to_numpy())
    return NorthAmericaPrior(
        site_ids=site_ids,
        lon=lon,
        lat=lat,
        trend=trend,
        draws=trend[None, :] + smooth_variation + site_noise,
        tower_site_ids=tower_site_ids,
    )


# ── helpers ─────────────────────────────────────────────────────────────────


def _mean_annual_temperature(site_ids: np.ndarray) -> np.ndarray:
    """Each site's ``MAT`` (deg C) from the site covariate table, in ``site_ids`` order."""
    path = tracked_data_root() / "raw" / "covariates" / "site_covariates_pft_assignment.csv"
    # ``index`` is the covariate table's site id (data/raw/covariates/provenance.md).
    covariates = pd.read_csv(path, usecols=["index", "MAT"]).set_index("index")
    return covariates.loc[site_ids, "MAT"].to_numpy()


def _unit_vectors(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Points on the unit sphere, ``(n, 3)``, from degrees."""
    lon_radians, lat_radians = np.radians(lon), np.radians(lat)
    return np.column_stack(
        [
            np.cos(lat_radians) * np.cos(lon_radians),
            np.cos(lat_radians) * np.sin(lon_radians),
            np.sin(lat_radians),
        ]
    )
