"""The grids a field over North America is computed on and shown on.

The continental maps (prior draws, uncertainty) share two grids, both
regular in the display projection (``SITE_PROJECTION``, meters) and spanning
the site pool:

- the **field grid**, ``grid_spacing`` apart, where a Gaussian process is
  computed exactly; it covers every display cell, plus a margin;
- the **display raster**, ``display_spacing`` apart, finer, where results
  are shown, interpolated bilinearly from the field grid
  (``ContinentGrids.to_display``). A field whose length scale is far longer
  than the field grid's spacing is smooth at that scale, so this loses
  nothing and draws smooth edges.

**Where the map is drawn.** The repository has no land polygons, so a map
is drawn where the site pool is: display cells within ``land_radius`` of a
site, with any enclosed hole smaller than ``max_hole_area`` filled (gaps
between sparse sites), while larger ones (Hudson Bay, say) stay blank.

**Distance.** Covariances are functions of the chordal distance between
locations (the straight line through the Earth, ``points_on_sphere``): a
valid distance for the Matern kernels on the sphere, and close to the
great-circle distance at the length scales used here.
"""

from dataclasses import dataclass

import numpy as np
import pyproj
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from sipnet_calibration.net_ecosystem_exchange.towers import read_tower_table
from sipnet_calibration.projection import SITE_PROJECTION
from sipnet_calibration.sites import load_sites

__all__ = ["EARTH_RADIUS", "ContinentGrids", "build_continent_grids", "flux_tower_site_ids", "points_on_sphere"]

#: The Earth's mean radius, meters.
EARTH_RADIUS = 6.371e6


@dataclass(frozen=True)
class ContinentGrids:
    """The site pool, the field grid and the display raster."""

    #: (sites,): the site pool, and its locations in projected meters.
    site_ids: np.ndarray
    site_lon: np.ndarray
    site_lat: np.ndarray
    site_x: np.ndarray
    site_y: np.ndarray
    #: The field grid's axes, and its points near the pool ``(n_field, 2)``,
    #: their positions in the grid (``in_field``, flattened row by row) and
    #: their nearest sites.
    field_x: np.ndarray
    field_y: np.ndarray
    in_field: np.ndarray
    field_points: np.ndarray
    field_nearest_site: np.ndarray
    #: The display raster's axes, and ``(ny, nx)`` where the map is drawn.
    display_x: np.ndarray
    display_y: np.ndarray
    drawn: np.ndarray

    def field_points_on_sphere(self) -> np.ndarray:
        """The field points as ``points_on_sphere`` gives them, ``(n_field, 3)`` meters."""
        lon, lat = _to_lon_lat(self.field_points[:, 0], self.field_points[:, 1])
        return points_on_sphere(lon, lat)

    def to_display(self, values: np.ndarray) -> np.ndarray:
        """Values at the field points ``(..., n_field)`` on the display raster ``(..., ny, nx)``.

        Interpolated bilinearly, and NaN where the map is not drawn.
        """
        leading_shape = values.shape[:-1]
        flat_values = values.reshape(-1, values.shape[-1])
        on_field_grid = np.full((len(flat_values), len(self.field_y) * len(self.field_x)), np.nan)
        on_field_grid[:, self.in_field] = flat_values
        display_points = _mesh_points(self.display_x, self.display_y)[:, ::-1]  # (y, x), the interpolator's order
        shown = np.stack([
            RegularGridInterpolator(
                (self.field_y, self.field_x), field.reshape(len(self.field_y), len(self.field_x))
            )(display_points)
            for field in on_field_grid
        ]).reshape(len(flat_values), len(self.display_y), len(self.display_x))
        shown[:, ~self.drawn] = np.nan
        return shown.reshape(*leading_shape, len(self.display_y), len(self.display_x))


def build_continent_grids(
    *, grid_spacing: float, display_spacing: float, land_radius: float, max_hole_area: float
) -> ContinentGrids:
    """Lay out both grids over the site pool (lengths in meters, area in square meters)."""
    site_table = load_sites()
    site_lon, site_lat = site_table["lon"].to_numpy(), site_table["lat"].to_numpy()
    site_x, site_y = SITE_PROJECTION.forward(site_lon, site_lat)
    site_tree = cKDTree(np.column_stack([site_x, site_y]))

    display_x, display_y = _axes_spanning(site_x, site_y, display_spacing)
    distance_to_site, _ = site_tree.query(_mesh_points(display_x, display_y))
    drawn = _with_small_holes_filled(
        (distance_to_site <= land_radius).reshape(len(display_y), len(display_x)),
        max_hole_cells=max_hole_area / display_spacing**2,
    )

    # The field grid's points reach a cell diagonal beyond the drawn area, so
    # every drawn display cell lies inside a square of them to interpolate from.
    field_x, field_y = _axes_spanning(site_x, site_y, grid_spacing, margin=land_radius + grid_spacing)
    grid_points = _mesh_points(field_x, field_y)
    distance_to_site, nearest_site = site_tree.query(grid_points)
    in_field = distance_to_site <= land_radius + np.sqrt(2) * grid_spacing
    return ContinentGrids(
        site_ids=site_table["site_id"].to_numpy(),
        site_lon=site_lon,
        site_lat=site_lat,
        site_x=site_x,
        site_y=site_y,
        field_x=field_x,
        field_y=field_y,
        in_field=in_field,
        field_points=grid_points[in_field],
        field_nearest_site=nearest_site[in_field],
        display_x=display_x,
        display_y=display_y,
        drawn=drawn,
    )


def flux_tower_site_ids() -> np.ndarray:
    """The sites with a flux tower: the NEE tower table's primary towers' sites, ascending."""
    towers = read_tower_table()
    return np.unique(towers.loc[towers["primary"], "site_id"].to_numpy())


def points_on_sphere(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Locations as points on the Earth's surface in 3D, ``(n, 3)`` meters, from degrees."""
    lon_radians, lat_radians = np.radians(lon), np.radians(lat)
    return EARTH_RADIUS * np.column_stack(
        [
            np.cos(lat_radians) * np.cos(lon_radians),
            np.cos(lat_radians) * np.sin(lon_radians),
            np.sin(lat_radians),
        ]
    )


# ── helpers ─────────────────────────────────────────────────────────────────


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
