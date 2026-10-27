"""Draws from a spatial prior over North America, as continuous maps.

One generic parameter (on a log scale, centered at 0) at every location
``x`` of a map grid covering the site pool:

    theta(x) = slope * MAT_std(x) + field_sd * u(x),

the prior model the slides state: a trend in a spatial covariate plus a
Gaussian process. ``MAT_std`` is mean annual temperature (the site covariate
table's ``MAT``), standardized over the pool's sites and carried to each
grid point from its nearest site. ``u`` is a unit-variance Gaussian process
with Matern-3/2 correlation in the chordal distance between locations (the
straight line through the Earth, a valid distance for this kernel on the
sphere and close to the great-circle distance at these length scales).

**Two grids**, both regular in the display projection (``SITE_PROJECTION``,
meters) and spanning the sites:

- the *field grid*, ``grid_spacing`` apart, where the Gaussian process is
  drawn exactly; it covers every display cell, plus a margin;
- the *display raster*, ``display_spacing`` apart, finer, where the draws
  are shown, interpolated bilinearly from the field grid. The field is
  smooth at the field grid's scale (the length scale is far longer), so
  this loses nothing and draws smooth edges.

**Where the map is drawn.** The repository has no land polygons, so the
map is drawn where the site pool is: display cells within ``land_radius``
of a site, with any enclosed hole smaller than ``max_hole_area`` filled
(gaps between sparse sites), while larger ones (Hudson Bay, say) stay
blank.

It reads the site table and the covariate table, both tracked inputs, and
the tower table to mark the flux-tower sites. It illustrates the prior's
form only: the numbers are chosen to look like a plausible parameter map,
not fitted to anything.
"""

import numpy as np
import pandas as pd
import pyproj
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cholesky
from scipy.spatial import cKDTree

from sipnet_calibration.conventions import tracked_data_root
from sipnet_calibration.net_ecosystem_exchange.towers import read_tower_table
from sipnet_calibration.projection import SITE_PROJECTION
from sipnet_calibration.sites import load_sites

from ..results import NorthAmericaPrior
from ..toy.spatial_prior import CHOLESKY_JITTER, matern32_correlation

__all__ = ["EARTH_RADIUS", "draw_north_america_prior"]

#: The Earth's mean radius, meters.
EARTH_RADIUS = 6.371e6


def draw_north_america_prior(
    rng: np.random.Generator,
    *,
    n_draws: int,
    length_scale: float,
    field_sd: float,
    trend_slope: float,
    grid_spacing: float,
    display_spacing: float,
    land_radius: float,
    max_hole_area: float,
) -> NorthAmericaPrior:
    """Draw the prior ``n_draws`` times and show it on a raster over the site pool.

    Lengths are in meters and ``max_hole_area`` in square meters.
    """
    site_table = load_sites()
    site_ids = site_table["site_id"].to_numpy()
    lon, lat = site_table["lon"].to_numpy(), site_table["lat"].to_numpy()
    site_x, site_y = SITE_PROJECTION.forward(lon, lat)
    site_tree = cKDTree(np.column_stack([site_x, site_y]))

    # Where the map is drawn, on the display raster.
    display_x, display_y = _axes_spanning(site_x, site_y, display_spacing)
    distance_to_site, _ = site_tree.query(_mesh_points(display_x, display_y))
    drawn = _with_small_holes_filled(
        (distance_to_site <= land_radius).reshape(len(display_y), len(display_x)),
        max_hole_cells=max_hole_area / display_spacing**2,
    )

    # The field, drawn exactly on the field grid. Its points reach a cell
    # diagonal beyond the drawn area, so every drawn display cell lies
    # inside a square of field-grid points to interpolate from.
    grid_x, grid_y = _axes_spanning(site_x, site_y, grid_spacing, margin=land_radius + grid_spacing)
    grid_points = _mesh_points(grid_x, grid_y)
    distance_to_site, nearest_site = site_tree.query(grid_points)
    in_field = distance_to_site <= land_radius + np.sqrt(2) * grid_spacing
    field_on_grid = np.full((n_draws, len(grid_y) * len(grid_x)), np.nan)
    field_on_grid[:, in_field] = _draw_field(
        rng,
        grid_points[in_field],
        _standardized_mean_annual_temperature(site_ids)[nearest_site[in_field]],
        n_draws=n_draws,
        length_scale=length_scale,
        field_sd=field_sd,
        trend_slope=trend_slope,
    )

    # The draws on the display raster, blank where the map is not drawn.
    display_points = _mesh_points(display_x, display_y)[:, ::-1]  # (y, x), the interpolator's order
    draws = np.stack([
        RegularGridInterpolator((grid_y, grid_x), field.reshape(len(grid_y), len(grid_x)))(display_points)
        for field in field_on_grid
    ]).reshape(n_draws, len(display_y), len(display_x))
    draws[:, ~drawn] = np.nan

    towers = read_tower_table()
    tower_site_ids = np.unique(towers.loc[towers["primary"], "site_id"].to_numpy())
    return NorthAmericaPrior(
        site_ids=site_ids,
        lon=lon,
        lat=lat,
        tower_site_ids=tower_site_ids,
        grid_x=display_x,
        grid_y=display_y,
        draws=draws,
    )


def _draw_field(
    rng: np.random.Generator,
    points: np.ndarray,
    standardized_covariate: np.ndarray,
    *,
    n_draws: int,
    length_scale: float,
    field_sd: float,
    trend_slope: float,
) -> np.ndarray:
    """Draws ``(n_draws, n_points)`` of the trend plus the Gaussian process at projected ``points``."""
    point_lon, point_lat = _to_lon_lat(points[:, 0], points[:, 1])
    on_sphere = EARTH_RADIUS * _unit_vectors(point_lon, point_lat)
    correlation = matern32_correlation(on_sphere, on_sphere, length_scale)
    # Factorized in place: the matrix is about a gigabyte.
    correlation[np.diag_indices_from(correlation)] += CHOLESKY_JITTER
    lower = cholesky(correlation, lower=True, overwrite_a=True)
    smooth_variation = field_sd * (lower @ rng.standard_normal((len(points), n_draws))).T
    return trend_slope * standardized_covariate[None, :] + smooth_variation


# ── helpers ─────────────────────────────────────────────────────────────────


def _standardized_mean_annual_temperature(site_ids: np.ndarray) -> np.ndarray:
    """Each site's ``MAT`` from the site covariate table, standardized over the sites given."""
    path = tracked_data_root() / "raw" / "covariates" / "site_covariates_pft_assignment.csv"
    # ``index`` is the covariate table's site id (data/raw/covariates/provenance.md).
    covariates = pd.read_csv(path, usecols=["index", "MAT"]).set_index("index")
    mean_annual_temperature = covariates.loc[site_ids, "MAT"].to_numpy()
    return (mean_annual_temperature - mean_annual_temperature.mean()) / mean_annual_temperature.std()


def _axes_spanning(x: np.ndarray, y: np.ndarray, spacing: float, *, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Regular grid axes covering the points ``x``, ``y`` and ``margin`` beyond them."""
    return (
        np.arange(x.min() - margin, x.max() + margin + spacing, spacing),
        np.arange(y.min() - margin, y.max() + margin + spacing, spacing),
    )


def _mesh_points(axis_x: np.ndarray, axis_y: np.ndarray) -> np.ndarray:
    """Every grid point, ``(ny * nx, 2)`` as (x, y), row by row (x fastest)."""
    mesh_x, mesh_y = np.meshgrid(axis_x, axis_y)
    return np.column_stack([mesh_x.ravel(), mesh_y.ravel()])


def _with_small_holes_filled(mask: np.ndarray, *, max_hole_cells: float) -> np.ndarray:
    """``mask`` with every enclosed hole of fewer than ``max_hole_cells`` cells set."""
    holes, n_holes = ndimage.label(~mask)
    hole_sizes = ndimage.sum_labels(np.ones_like(holes), holes, index=np.arange(1, n_holes + 1))
    # A component touching the edge is open water or outside the pool, not a hole.
    touches_edge = np.zeros(n_holes + 1, dtype=bool)
    touches_edge[np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))] = True
    small = np.zeros(n_holes + 1, dtype=bool)
    small[1:] = (hole_sizes < max_hole_cells) & ~touches_edge[1:]
    return mask | small[holes]


def _to_lon_lat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projected meters back to longitude and latitude, degrees."""
    transformer = pyproj.Transformer.from_crs(SITE_PROJECTION.crs(), "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)


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
