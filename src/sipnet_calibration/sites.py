"""The site pool: its grid, its table, and site selection.

The site table is read from ``data/processed/sites/sites.csv``, produced by
``scripts/ingest_sites.py`` from the point shapefile in ``data/raw/sites/``, the
PFT assignment table, and ``data/site_id_map.csv`` (Ameriflux ``Site_ID`` ->
integer site id, exact matching). See ``data/README.md`` for the column set.

``select_sites`` is the most-reused operation in the project and is deliberately
not a plotting concern: subsetting by PFT, bounding box, data availability, or
random sample happens once and the result is passed to adapters and plotters
alike.

Note the sites are 8000 *irregular points* spanning 7-82 deg N and
178 W-20 W. Only ~3640 fall inside a CONUS bounding box, so CONUS-only
assumptions are wrong.

This module also defines the geographic lattice the sites sit on. That lives
here, beside the site table, because the grid is a property of the site pool and
because the constants and the two conversions that use them must not be able to
disagree with each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Grid", "SITE_GRID"]


@dataclass(frozen=True, slots=True)
class Grid:
    """A regular geographic lattice, addressed by the centers of its cells.

    The step is held as an integer number of cells per degree rather than as a
    floating-point width, so that a step of exactly 1/120 degree stays exact
    instead of accumulating error across 19080 columns.

    Indices are zero-based, with ``lon_idx`` increasing east from ``west`` and
    ``lat_idx`` increasing north from ``south``. Cell centers are at::

        lon = west  + (lon_idx + 0.5) / cells_per_degree
        lat = south + (lat_idx + 0.5) / cells_per_degree

    Parameters
    ----------
    west, south:
        Outer edge of the first cell, in degrees. Not a cell center.
    n_lon, n_lat:
        Number of cells along each axis.
    cells_per_degree:
        Cells per degree, the reciprocal of the step.
    """

    west: float
    south: float
    n_lon: int
    n_lat: int
    cells_per_degree: int

    def __post_init__(self) -> None:
        if self.n_lon <= 0 or self.n_lat <= 0:
            raise ValueError(f"grid must have positive extent, got {self.n_lon}x{self.n_lat}")
        if self.cells_per_degree <= 0:
            raise ValueError(f"cells_per_degree must be positive, got {self.cells_per_degree}")

    # ── derived geometry ──────────────────────────────────────────────────────

    @property
    def step(self) -> float:
        """Cell width in degrees."""
        return 1.0 / self.cells_per_degree

    @property
    def east(self) -> float:
        """Outer edge of the last column, in degrees."""
        return self.west + self.n_lon / self.cells_per_degree

    @property
    def north(self) -> float:
        """Outer edge of the last row, in degrees."""
        return self.south + self.n_lat / self.cells_per_degree

    @property
    def step_arcsec(self) -> float:
        """Cell width in arcseconds."""
        return 3600.0 / self.cells_per_degree

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_lat, n_lon)``, matching the row-major convention of a raster."""
        return (self.n_lat, self.n_lon)

    # ── conversions ──────────────────────────────────────────────────────────

    def index_to_lonlat(self, lon_idx, lat_idx):
        """Cell centers for the given indices.

        Parameters
        ----------
        lon_idx, lat_idx:
            Zero-based indices, scalar or array-like. Broadcast against each
            other.

        Returns
        -------
        tuple
            ``(lon, lat)`` in degrees. Scalars in, scalars out.

        Raises
        ------
        ValueError
            If any index falls outside the grid.
        """
        j = np.asarray(lon_idx)
        k = np.asarray(lat_idx)
        if not (np.issubdtype(j.dtype, np.integer) and np.issubdtype(k.dtype, np.integer)):
            if np.any(j != np.floor(j)) or np.any(k != np.floor(k)):
                raise ValueError("indices must be integers; use lonlat_to_index for coordinates")
            j, k = j.astype(np.int64), k.astype(np.int64)
        if np.any(j < 0) or np.any(j >= self.n_lon):
            raise ValueError(f"lon_idx outside 0..{self.n_lon - 1}")
        if np.any(k < 0) or np.any(k >= self.n_lat):
            raise ValueError(f"lat_idx outside 0..{self.n_lat - 1}")

        lon = self.west + (j + 0.5) / self.cells_per_degree
        lat = self.south + (k + 0.5) / self.cells_per_degree
        if lon.ndim == 0 and lat.ndim == 0:
            return float(lon), float(lat)
        return lon, lat

    def lonlat_to_index(self, lon, lat, *, tol: float = 1e-4):
        """Indices of the cells whose centers the given coordinates sit at.

        The coordinates are expected to *be* cell centers, not arbitrary points:
        this is a lookup, not a binning operation. Coordinates that are further
        than *tol* from any center raise rather than being snapped, since a point
        that is not on the grid usually means the wrong grid or the wrong CRS.

        Parameters
        ----------
        lon, lat:
            Degrees, scalar or array-like. Broadcast against each other.
        tol:
            Largest accepted departure from a cell center, in degrees. The
            default of 1e-4 (about 11 m) is loose enough for coordinates that
            have passed through 32-bit storage and tight enough to reject a point
            that is genuinely off-grid, given a cell width of 1/120 degree.

        Returns
        -------
        tuple
            ``(lon_idx, lat_idx)`` as integers. Scalars in, scalars out.

        Raises
        ------
        ValueError
            If any coordinate lies further than *tol* from a cell center, or the
            resulting index falls outside the grid.
        """
        x = np.asarray(lon, dtype=float)
        y = np.asarray(lat, dtype=float)
        jf = (x - self.west) * self.cells_per_degree - 0.5
        kf = (y - self.south) * self.cells_per_degree - 0.5
        j = np.rint(jf).astype(np.int64)
        k = np.rint(kf).astype(np.int64)

        off = np.maximum(np.abs(jf - j), np.abs(kf - k)) / self.cells_per_degree
        if np.any(off > tol):
            worst = float(np.max(off))
            raise ValueError(
                f"coordinates are not on the grid: worst departure from a cell center is "
                f"{worst:.3g} degrees, tolerance is {tol:g}"
            )
        if np.any(j < 0) or np.any(j >= self.n_lon):
            raise ValueError(f"longitude outside the grid ({self.west} to {self.east})")
        if np.any(k < 0) or np.any(k >= self.n_lat):
            raise ValueError(f"latitude outside the grid ({self.south} to {self.north})")

        if j.ndim == 0 and k.ndim == 0:
            return int(j), int(k)
        return j, k


#: The grid the 8000 sites are defined on: 30 arcsecond (1/120 degree) cells
#: spanning 179 W to 20 W and 7 N to 85 N, as 19080 x 9360 cells. This is the
#: grid of the North American Land Carbon Reanalysis; see ``data/README.md``.
#:
#: Site coordinates are cell centers of this grid, but as stored they depart from
#: exact centers by up to 1.02e-6 degrees (about 0.11 m), consistent with having
#: passed through 32-bit floating point somewhere upstream. The integer indices
#: are therefore the exact representation of a site's position and the stored
#: coordinates are a lossy rendering of it.
SITE_GRID = Grid(west=-179.0, south=7.0, n_lon=19080, n_lat=9360, cells_per_degree=120)
