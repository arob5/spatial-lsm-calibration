"""The data model for the site pool, and the geographic lattice it sits on.

Overview
--------
This module defines how the site pool is represented -- its columns, dtypes and
identifiers -- and provides the functions for reading it, selecting from it, and
converting between coordinates and grid indices. It is the single description of
that layout: the ingest script that writes the table gets its column set and
dtypes from here rather than declaring its own.

It sits downstream of the one script that builds the table, and the dependency
runs one way::

    raw/sites/pts.*  +  data/site_id_map.csv
      -> scripts/ingest_sites.py    processed/sites/sites.csv
      -> this module                load_sites() -> pandas.DataFrame

Every other processed file joins against this table on ``site_id``, so this
is where the meaning of a site identifier is fixed. ``data/README.md`` documents
the source data, the coordinate reference system and the open questions.

Input data
----------
``data/processed/sites/sites.csv``
    The site table, read by :func:`load_sites`, whose layout is the
    `Data model`_ below. :func:`default_sites_path` says where it is expected
    to be, honoring the ``$SIPNET_CALIBRATION_DATA`` override in
    :data:`sipnet_calibration.conventions.DATA_ROOT_ENV_VAR`.

The grid itself is not read from anywhere. :data:`SITE_GRID` is defined in code
beside the conversions that use it, so the constants and the arithmetic cannot
disagree with each other.

Data model
----------
:func:`load_sites` returns a ``pandas.DataFrame`` with one row per site, in
ascending ``site_id`` order, holding the columns of :data:`SITE_COLUMNS` with
the dtypes of :data:`SITE_COLUMN_DTYPES`, which the read imposes. A missing or
unexpected column, a duplicate or non-ascending ``site_id``, or an
out-of-range integer raises.

================================ ============= ==============================
Column                           Dtype         Meaning
================================ ============= ==============================
``site_id``                      ``int32``     handed-down site identifier
``lon``, ``lat``                 ``float64``   coordinates, in degrees
``lon_index``, ``lat_index``     ``int32``     position on :data:`SITE_GRID`
``site_name``                    ``str``       label, empty where absent
``site_order``                   ``int32``     0 sampled, else the named rank
``cluster``, ``landcover``       ``int8``      sampling stratum, cover class
``ameriflux_site_id``            ``str``       identifier, empty where absent
================================ ============= ==============================

``site_id`` is left as a column rather than made the index, so the frame is a
table; callers wanting lookup call :func:`site_lookup`.

**Missing values.** The empty string, not ``NaN``, in both text columns: the
table is read with ``keep_default_na=False``, so a name that happens to read as
a null word survives. No numeric column can be missing.

**Geography.** The sites are irregular points spanning roughly 7-82 degrees
north and 178-20 degrees west, of which fewer than half fall inside a
conterminous-US bounding box. Their coordinates are cell centers of
:data:`SITE_GRID`, a regular geographic lattice; ``lon_index``/``lat_index`` are
the exact representation of a position and the stored floats are a lossy
rendering of it.

There is deliberately **no plant functional type column**. A PFT class is not
an intrinsic property of a site: a calibration may not use PFTs at all, and
several site-labels data sources can be applied to the same pool. Each site
labels data source has its own processed file under
``data/processed/site_labels/``, keyed on ``site_id``, and a caller joins one
on before selecting.

Constants
---------
:data:`N_SITES`
    The size of the site pool the raw inputs define, which raw-data code
    checks its inputs against: the ingest scripts, the raw-data specs
    (``expected_rows`` of :mod:`sipnet_calibration.site_labels`), the survey
    scripts and ``scripts/raw_sources/split_site_pft_16class.py``.

Functions
---------
:func:`load_sites`
    Read the site table and check it against the data model above.

:func:`select_sites`
    A subset of a site table, by identifier, bounding box, arbitrary predicate,
    or a number of sites drawn at random. The filters compose.

:func:`default_sites_path`
    Where the table is expected to be.

:func:`site_lookup`, :func:`site_locations`, :func:`site_coordinates`
    The table keyed on ``site_id`` for repeated lookups; the ``lon``/``lat``
    coordinates of given sites, with their CF attributes, as a field carries
    them; and those with the ``site`` coordinate itself.

The checks
    One invariant each, with the two groups callers use:
    :func:`check_site_table_locates_the_sites`, the check every ``lon``/``lat``
    lookup makes of a site table, and
    :func:`check_site_table_is_keyed_on_site_ids`, for a table that need not
    carry ``lon``/``lat``; and :func:`check_site_table_lists_the_sites` and
    :func:`check_sites_are_the_site_table`, which a raw data source's sites
    are checked against.

:data:`EXTENTS`
    Named longitude/latitude boxes -- ``CONUS``, ``NORTH_AMERICA``, ``ALASKA``
    -- in the form *bbox* takes, shared with the spatial plotting layer so that
    a figure and the sites it plots agree on what a region is.

:class:`Grid` and :data:`SITE_GRID`
    The lattice, and the conversions between coordinates and indices:
    :meth:`Grid.lonlat_to_index` and :meth:`Grid.index_to_lonlat`.

Notes
-----
**The grid is not equal-area.** A cell is about 928 m tall everywhere, but its
width shrinks from roughly 921 m at the south of the pool to a small fraction of
that at the north. Density and per-area calculations have to account for it, and
spatial plots need a real projection rather than plotting degrees directly.

**Coordinates are stored longitude before latitude**, which is the traditional
GDAL and PROJ ordering rather than the axis order EPSG:4326 formally declares.
Coordinate transformations should be configured accordingly.

**:meth:`Grid.lonlat_to_index` is a lookup, not a binning operation.** It expects
coordinates that already *are* cell centers and raises on anything further than
its tolerance from one, because a point that is off-grid usually means the wrong
grid or the wrong CRS rather than a point needing rounding.

Usage
-----
Read the table and select from it::

    from sipnet_calibration.sites import (
        SITE_GRID,
        load_sites,
        select_sites,
    )

    site_table = load_sites()               # or load_sites(path)

    # Named sites, by identifier, in the order given.
    named = select_sites(site_table, ids=[4102, 4113, 5584])

    # A bounding box, as (west, south, east, north), edges included. Every
    # longitude in the pool is negative.
    conus = select_sites(site_table, bbox=(-125, 24, -66, 50))

    # Any predicate over the table's columns.
    flux_towers = select_sites(site_table, where=lambda s: s["ameriflux_site_id"] != "")

    # The filters compose, and n_random always draws from whatever survived.
    subset = select_sites(
        site_table,
        bbox=(-125, 24, -66, 50),
        where=lambda s: s["ameriflux_site_id"] != "",
        n_random=20,
        seed=0,
    )

Select on site labels by joining them on first, since PFT is not a column here::

    import pandas as pd

    # Real site labels are their own data source under processed/site_labels/,
    # keyed on site_id. The join is the same whatever the source is called.
    site_labels = pd.DataFrame({"site_id": [4102, 4113], "pft": ["DBF", "ENF"]})
    deciduous = select_sites(
        site_table.merge(site_labels, on="site_id"), where=lambda s: s["pft"] == "DBF"
    )

Convert between coordinates and grid indices::

    row = site_table.iloc[0]

    # A stored coordinate back to its exact position on the lattice.
    lon_index, lat_index = SITE_GRID.lonlat_to_index(row["lon"], row["lat"])

    # And back to the cell center, which is where the stored value came from.
    lon, lat = SITE_GRID.index_to_lonlat(lon_index, lat_index)

    # Both are vectorized, so a whole column converts at once.
    lon_indices, lat_indices = SITE_GRID.lonlat_to_index(
        site_table["lon"].to_numpy(), site_table["lat"].to_numpy()
    )

Look sites up, and give a field its coordinates::

    from sipnet_calibration.sites import site_coordinates, site_locations, site_lookup

    keyed = site_lookup(site_table)                     # indexed on site_id
    keyed.loc[4102, "lon"]
    coords = site_locations([4113, 4102], site_table)   # {"lon", "lat"} on site, CF attributes
    coords = site_coordinates([4113, 4102], site_table)  # {"site", "lon", "lat"}
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.conventions import (
    LAT,
    LAT_ATTRIBUTES,
    LON,
    LON_ATTRIBUTES,
    SITE,
    SITE_ATTRIBUTES,
    SITE_DTYPE,
    SITE_ID,
    FrozenMapping,
    data_root,
)
from sipnet_calibration.validation import (
    as_bbox,
    as_bounded_integer,
    as_positive_integer,
    as_site_ids,
    truncated,
)

__all__ = [
    "EXTENTS",
    "N_SITES",
    "SITE_COLUMNS",
    "SITE_COLUMN_DTYPES",
    "SITE_GRID",
    "Grid",
    "check_site_table_has_locations",
    "check_site_table_has_site_ids",
    "check_site_table_is_a_dataframe",
    "check_site_table_is_keyed_on_site_ids",
    "check_site_table_lists_each_site_once",
    "check_site_table_lists_the_sites",
    "check_site_table_locates_the_sites",
    "check_site_table_site_ids_are_integers",
    "check_site_table_sites_are_all_listed",
    "check_sites_are_the_site_table",
    "default_sites_path",
    "load_sites",
    "select_sites",
    "site_coordinates",
    "site_locations",
    "site_lookup",
]


# ── the grid ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Grid:
    """A regular geographic lattice, addressed by the centers of its cells.

    The step is held as an integer number of cells per degree rather than as a
    floating-point width, so that a step of exactly 1/120 degree stays exact
    instead of accumulating error across 19080 columns.

    Indices are zero-based, with ``lon_index`` increasing east from ``west`` and
    ``lat_index`` increasing north from ``south``. Cell centers are at::

        lon = west  + (lon_index + 0.5) / cells_per_degree
        lat = south + (lat_index + 0.5) / cells_per_degree

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
        for name in ("n_lon", "n_lat", "cells_per_degree"):
            as_positive_integer(getattr(self, name), message_name=f"the grid's {name}")

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

    def index_to_lonlat(self, lon_index, lat_index):
        """Cell centers for the given indices.

        Parameters
        ----------
        lon_index, lat_index:
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
        j = np.asarray(lon_index)
        k = np.asarray(lat_index)
        if not (np.issubdtype(j.dtype, np.integer) and np.issubdtype(k.dtype, np.integer)):
            if np.any(j != np.floor(j)) or np.any(k != np.floor(k)):
                raise ValueError("indices must be integers; use lonlat_to_index for coordinates")
            j, k = j.astype(np.int64), k.astype(np.int64)
        if np.any(j < 0) or np.any(j >= self.n_lon):
            raise ValueError(f"lon_index outside 0..{self.n_lon - 1}")
        if np.any(k < 0) or np.any(k >= self.n_lat):
            raise ValueError(f"lat_index outside 0..{self.n_lat - 1}")

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
            ``(lon_index, lat_index)`` as integers. Scalars in, scalars out.

        Raises
        ------
        ValueError
            If any coordinate is not finite, lies further than *tol* from a cell
            center, or resolves to an index outside the grid.
        """
        x = np.asarray(lon, dtype=float)
        y = np.asarray(lat, dtype=float)
        # Checked first because NaN defeats both guards below: np.rint(nan) is 0
        # on this platform, and every comparison against NaN is False, so a NaN
        # coordinate would silently resolve to a real cell.
        non_finite = ~(np.isfinite(x) & np.isfinite(y))
        if np.any(non_finite):
            n = int(np.count_nonzero(non_finite))
            raise ValueError(
                f"{n} coordinate(s) are not finite, so they are not on the grid"
            )
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


# ── the site table ────────────────────────────────────────────────────────────

#: The size of the site pool the raw inputs define: the records of
#: ``raw/sites/pts.*`` and the rows of each whole-pool raw data source. Only
#: raw-data code reads it -- the ingest scripts, the raw-data specs
#: (``expected_rows``), the survey scripts and the split script -- to check
#: its inputs have that size; code working on a site table counts the sites
#: of the table in hand instead.
N_SITES = 8000

#: Columns of ``processed/sites/sites.csv``, in order. The ingest script writes
#: exactly these and :func:`load_sites` requires exactly these, so the two
#: cannot drift apart.
#:
#: ``lon``/``lat`` are the stored coordinates at full precision and
#: ``lon_index``/``lat_index`` are their exact representation on :data:`SITE_GRID`;
#: the table carries both because the floats are a lossy rendering of the
#: indices rather than the other way round. ``ameriflux_site_id`` is the empty
#: string for the sites with no Ameriflux counterpart, which is most of them.
SITE_COLUMNS = (
    SITE_ID,
    LON,
    LAT,
    "lon_index",
    "lat_index",
    "site_name",
    "site_order",
    "cluster",
    "landcover",
    "ameriflux_site_id",
)

#: Dtype per column, matching the site-table schema in the processed-format
#: plan. ``lon_index`` genuinely needs ``int32``; the other integer columns would
#: fit ``int16`` and are widened to match it.
#:
#: The two text columns are declared ``str`` so that nothing is inferred from
#: their content. Note that this alone does **not** save the eight sites named
#: ``NA`` -- ``dtype=str`` still yields ``nan`` for them. What saves them is
#: ``keep_default_na=False`` in :func:`load_sites`.
#:
#: Read-only: this is the schema, and a caller that mutated it would change what
#: every later read of the table produces.
SITE_COLUMN_DTYPES = FrozenMapping(
    {
        SITE_ID: SITE_DTYPE,
        LON: np.float64,
        LAT: np.float64,
        "lon_index": np.int32,
        "lat_index": np.int32,
        "site_name": str,
        "site_order": np.int32,
        "cluster": np.int8,
        "landcover": np.int8,
        "ameriflux_site_id": str,
    }
)


def default_sites_path() -> Path:
    """Where the site table is expected to be.

    ``$SIPNET_CALIBRATION_DATA/processed/sites/sites.csv`` when that variable is
    set, and otherwise the ``data/`` directory of this checkout. Experiments name
    their paths in ``config.py`` rather than relying on this; it exists so that
    tests, notebooks and the ingest script agree on one default.
    """
    return data_root() / "processed" / "sites" / "sites.csv"


def load_sites(path: Path | str | None = None) -> pd.DataFrame:
    """Read the site table.

    Parameters
    ----------
    path:
        The CSV to read. Defaults to :func:`default_sites_path`.

    Returns
    -------
    pandas.DataFrame
        One row per site, the columns of :data:`SITE_COLUMNS` in that order with
        the dtypes of :data:`SITE_COLUMN_DTYPES`, in ascending ``site_id`` order.
        ``site_id`` is left as a column rather than made the index, so that the
        frame is a table rather than a lookup; callers wanting lookup call
        :func:`site_lookup`.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with the command that produces it.
    ValueError
        If the columns are not the expected set, the file holds no rows,
        ``site_id`` is not unique, not ascending, or below 1, or any integer
        column holds a value outside the range of its declared dtype.

    Notes
    -----
    Two reader settings are load-bearing rather than stylistic.

    ``keep_default_na`` is off and ``na_values`` is empty, so no site name or
    identifier is reinterpreted as a missing value. Eight of the 8000 sites are
    named literally ``NA``, and an unmapped ``ameriflux_site_id`` reads back as
    the empty string it was written as. The cost is that the float columns must
    never be blank, which the ingest script guarantees.

    ``float_precision="round_trip"`` selects the exact float parser over the
    fast one pandas uses by default. The fast parser is not exact: it reads
    ``-93.287501017252595`` as -93.28750101725261, off by 1.4e-14. This column
    is the exact geometry of the site pool, so that is not acceptable, and the
    setting makes the read correct whatever decimal representation the file
    happens to carry.
    """
    csv_path = Path(path) if path is not None else default_sites_path()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"no site table at {csv_path}; build it with "
            "`python scripts/ingest_sites.py` from the project environment"
        )

    # Read integers wide, then narrow after checking. Reading straight into the
    # declared widths wraps out-of-range values silently: a site_id of
    # 4294967297 becomes 1 in int32 and then satisfies every check below.
    wide = {
        column: (np.int64 if np.issubdtype(np.dtype(dtype), np.integer) else dtype)
        for column, dtype in SITE_COLUMN_DTYPES.items()
    }
    table = pd.read_csv(
        csv_path,
        dtype=wide,
        keep_default_na=False,
        na_values=[],
        float_precision="round_trip",
    )
    _check_site_table(table, source=csv_path)
    return table[list(SITE_COLUMNS)].astype(SITE_COLUMN_DTYPES)


# ── site selection ────────────────────────────────────────────────────────────

#: Named regions, as ``(west, south, east, north)`` in degrees, in the form
#: :func:`select_sites` takes for *bbox* and
#: :meth:`sipnet_calibration.projection.Projection.projected_bounds` takes for
#: axes limits. They live here, beside the selection they parametrize, so that a
#: figure and the site subset it plots cannot disagree about what a region means.
#:
#: - ``CONUS`` is the conterminous-US box ``data/README.md`` uses, holding a
#:   little under half the pool; the count is asserted in the test suite.
#: - ``NORTH_AMERICA`` is the extent of :data:`SITE_GRID` itself, so it contains
#:   every site by construction rather than by a bound anyone chose.
#: - ``ALASKA`` is the EPSG area of use of "United States (USA) - Alaska", as
#:   registered for EPSG:3338, clipped on the west at the grid's own edge: the
#:   registered extent runs from 172.42 E across the antimeridian, whereas the
#:   grid, the site pool and :func:`select_sites` are all in negative longitudes
#:   and none of them wraps. No site is lost, since every site longitude is
#:   negative, but a basemap drawn to this box omits the western Aleutians.
#:
#: The plotting design spec calls the middle one ``NA``. It is spelled out here
#: under the project's convention against abbreviations, and because ``NA`` is
#: an unhappy name in a module that has to read ``NA`` as a literal site name.
#: Read-only, like :data:`SITE_COLUMN_DTYPES`: reassigning an entry would
#: silently change every later figure in the process.
EXTENTS = FrozenMapping(
    {
        "CONUS": (-125.0, 24.0, -66.0, 50.0),
        "NORTH_AMERICA": (SITE_GRID.west, SITE_GRID.south, SITE_GRID.east, SITE_GRID.north),
        "ALASKA": (SITE_GRID.west, 51.3, -129.99, 71.4),
    }
)


def select_sites(
    site_table: pd.DataFrame,
    *,
    ids: Iterable[int] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    where: Callable[[pd.DataFrame], object] | None = None,
    n_random: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """A subset of the site table.

    The filters compose, and are applied in the order below so that
    *n_random* always draws from what survived the rest.

    Parameters
    ----------
    site_table:
        A site table, from :func:`load_sites`, or one with extra columns joined
        on. Never modified.
    ids:
        Site ids to keep, a sequence of them; every one must exist. The result
        is in the order given, unless *n_random* is also passed, which re-sorts
        by ``site_id``.
    bbox:
        ``(west, south, east, north)`` in degrees, edges included. Longitudes are
        negative throughout the pool, so ``(-125, 24, -66, 50)`` is the
        conterminous US and ``(66, 24, 125, 50)`` selects nothing. The named
        regions are in :data:`EXTENTS`, so ``bbox=EXTENTS["CONUS"]`` is the same
        box as the CONUS figure uses.
    where:
        A callable taking the table and returning a boolean mask over its rows —
        anything ``.loc`` accepts. This is the general filter: it covers the
        columns of the table and any joined on beside them.
    n_random:
        Keep this many rows, drawn at random without replacement, in ascending
        ``site_id`` order. Fewer rows available is an error.
    seed:
        Seed for *n_random*. Passing one makes the draw reproducible; leaving it
        out does not.

    Returns
    -------
    pandas.DataFrame
        A copy, with the index reset.

    Raises
    ------
    KeyError
        If *ids* names a site the table does not hold.
    TypeError
        If *ids* is one id, a string, a set or a mapping, or holds a
        boolean, a float or a value that is not a number; if *bbox* is not a
        sequence of four numbers; or if *n_random* is a boolean or not an
        integer.
    ValueError
        If *ids* holds duplicates or values that are not site ids, is a
        two-dimensional array, or the table lists a site twice; if *bbox* is
        malformed; if *where* does not return a usable mask; or if *n_random*
        is negative or exceeds the number of rows available.

    Notes
    -----
    There is no ``pft=`` argument. PFT is not a column of the site table (see the
    module docstring). PFT selection can be done by joining site labels on and
    passing ``where``::

        labeled = site_table.merge(pd.read_csv(site_labels), on="site_id")
        select_sites(labeled, where=lambda t: t["pft"] == "DBF")

    Examples
    --------
    Twenty conterminous-US sites with an Ameriflux counterpart::

        select_sites(
            load_sites(),
            bbox=(-125, 24, -66, 50),
            where=lambda s: s["ameriflux_site_id"] != "",
            n_random=20,
            seed=0,
        )
    """
    selected = site_table

    if ids is not None:
        selected = _select_by_id(selected, ids)
    if bbox is not None:
        selected = selected.loc[_bbox_mask(selected, bbox)]
    if where is not None:
        selected = selected.loc[_predicate_mask(selected, where)]
    if n_random is not None:
        selected = _draw_n_random(selected, n_random, seed)

    return selected.reset_index(drop=True)


# ── site lookup ───────────────────────────────────────────────────────────────


def site_lookup(site_table: pd.DataFrame) -> pd.DataFrame:
    """The site table keyed on ``site_id``, so looking a site up is not a scan.

    Parameters
    ----------
    site_table:
        A site table, as :func:`load_sites` returns it or with columns joined
        on, or one already indexed on ``site_id``.

    Returns
    -------
    pandas.DataFrame
        *site_table* indexed on ``site_id``, which stays a column as well when
        it was one; *site_table* itself when it is already indexed so, so
        passing the result back in costs nothing.

    Raises
    ------
    TypeError
        If *site_table* is not a ``DataFrame``.
    ValueError
        If it has no ``site_id`` column or index.
    """
    check_site_table_is_a_dataframe(site_table)
    check_site_table_has_site_ids(site_table)
    if site_table.index.name == SITE_ID:
        return site_table
    return site_table.set_index(SITE_ID, drop=False)


def site_locations(
    site_ids: Iterable[int], site_table: pd.DataFrame | None = None
) -> dict[str, xr.DataArray]:
    """The ``lon``/``lat`` coordinates of *site_ids*, as a field carries them.

    Parameters
    ----------
    site_ids:
        The sites to locate, a sequence of site ids, each once.
    site_table:
        The site table to read them from, as :func:`load_sites` returns it or
        keyed by :func:`site_lookup`. Read from its default location when
        omitted.

    Returns
    -------
    dict
        ``{"lon": ..., "lat": ...}``, each a ``float64`` ``DataArray`` on the
        ``site`` dimension, in the order of *site_ids*, carrying the CF
        attributes of :data:`sipnet_calibration.conventions.LON_ATTRIBUTES`
        and :data:`~sipnet_calibration.conventions.LAT_ATTRIBUTES`. They have
        no ``site`` coordinate of their own, so they assign by position onto
        a ``site`` dimension in that order.

    Raises
    ------
    TypeError
        For a refusal of :func:`sipnet_calibration.validation.as_site_ids`
        (one id, a string, a set, a float or a boolean), or if *site_table*
        is not a ``DataFrame`` or its ``site_id`` is not integers.
    ValueError
        If *site_ids* is a two-dimensional array, an id is not a site id or is
        named twice, or for any refusal of
        :func:`check_site_table_locates_the_sites`.
    KeyError
        If a site is not in the site table.
    FileNotFoundError
        If *site_table* is omitted and the site table is absent.
    """
    wanted = list(as_site_ids(site_ids, message_name="site_ids"))
    table = site_table if site_table is not None else load_sites()
    check_site_table_locates_the_sites(table, wanted)
    located = site_lookup(table).loc[wanted, [LON, LAT]]
    return {
        LON: xr.DataArray(located[LON].to_numpy(np.float64), dims=SITE, attrs=LON_ATTRIBUTES),
        LAT: xr.DataArray(located[LAT].to_numpy(np.float64), dims=SITE, attrs=LAT_ATTRIBUTES),
    }


def site_coordinates(
    site_ids: Iterable[int], site_table: pd.DataFrame
) -> dict[str, xr.DataArray]:
    """The ``site``, ``lon`` and ``lat`` coordinates of *site_ids*, as a field carries them.

    Parameters
    ----------
    site_ids:
        The sites, a sequence of site ids, each once, in the order the
        ``site`` dimension takes them.
    site_table:
        The site table to locate them in, as :func:`load_sites` returns it or
        keyed by :func:`site_lookup`.

    Returns
    -------
    dict
        ``{"site": ..., "lon": ..., "lat": ...}``, each a ``DataArray`` on the
        ``site`` dimension: ``site`` the ids as ``int32`` with
        :data:`~sipnet_calibration.conventions.SITE_ATTRIBUTES`, and
        ``lon``/``lat`` as :func:`site_locations` gives them.

    Raises
    ------
    TypeError, ValueError, KeyError
        As :func:`site_locations` raises them.
    """
    wanted = as_site_ids(site_ids, message_name="site_ids")
    return {
        SITE: xr.DataArray(np.asarray(wanted, dtype=SITE_DTYPE), dims=SITE, attrs=SITE_ATTRIBUTES),
        **site_locations(wanted, site_table),
    }


# ── helpers ───────────────────────────────────────────────────────────────────
#
# Private: the shape of the table and of a selection, not part of the API.


def _check_site_table(table: pd.DataFrame, *, source: Path) -> None:
    """Raise unless *table* is a usable site table, naming what is wrong."""
    found = set(table.columns)
    expected = set(SITE_COLUMNS)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(
            f"{source} is not a site table: missing columns {missing}, "
            f"unexpected columns {extra}"
        )
    site_ids = table[SITE_ID].to_numpy()
    if site_ids.size == 0:
        raise ValueError(f"{source} holds no rows")
    if np.unique(site_ids).size != site_ids.size:
        raise ValueError(f"{source} holds duplicate site_id values")
    if np.any(np.diff(site_ids) <= 0):
        raise ValueError(f"{source} is not in ascending site_id order")
    if site_ids.min() < 1:
        raise ValueError(f"{source} holds a site_id below 1")

    for column, dtype in SITE_COLUMN_DTYPES.items():
        if not np.issubdtype(np.dtype(dtype), np.integer):
            continue
        info = np.iinfo(dtype)
        values = table[column].to_numpy()
        outside = np.flatnonzero((values < info.min) | (values > info.max))
        if outside.size:
            index = int(outside[0])
            raise ValueError(
                f"{source} row {index} has {column}={values[index]}, outside the "
                f"range of {np.dtype(dtype).name} ({info.min}..{info.max})"
            )


def _select_by_id(site_table: pd.DataFrame, ids: Iterable[int]) -> pd.DataFrame:
    """Rows for *ids*, in the order given. Raises on an unknown identifier.

    Selection is positional rather than ``set_index(...).loc[...]``, which would
    change ``site_id``'s dtype and move it to the first column -- so the frame
    this path returns would differ in shape from the one every other path
    returns, on a table with joined columns.
    """
    wanted = list(as_site_ids(ids, message_name="ids"))
    # A repeated site would make ids= return more rows than it was asked for.
    check_site_table_lists_each_site_once(site_table)
    check_site_table_lists_the_sites(site_table, wanted)
    position = pd.Series(np.arange(len(site_table)), index=site_table[SITE_ID].to_numpy())
    return site_table.iloc[position.loc[wanted].to_numpy()]


def _site_ids_of(site_table: pd.DataFrame) -> pd.Index:
    """The site table's ``site_id``, from its index or its column."""
    if site_table.index.name == SITE_ID:
        return site_table.index
    return pd.Index(site_table[SITE_ID])


def _bbox_mask(site_table: pd.DataFrame, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """A boolean mask of the sites inside *bbox*, edges included."""
    west, south, east, north = as_bbox(bbox, message_name="bbox")
    lon = site_table[LON].to_numpy()
    lat = site_table[LAT].to_numpy()
    return (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)


def _predicate_mask(
    site_table: pd.DataFrame, where: Callable[[pd.DataFrame], object]
) -> np.ndarray:
    """The mask *where* returns, aligned and checked before it is used.

    A ``Series`` is aligned on its index, the way ``.loc`` would, rather than
    being read positionally. Reading it positionally is the dangerous case: a
    reordered mask of the right length then selects the wrong rows and every
    shape and dtype check still passes.
    """
    result = where(site_table)

    if isinstance(result, pd.Series):
        if not result.index.equals(site_table.index):
            if len(result) != len(site_table) or set(result.index) != set(site_table.index):
                raise ValueError(
                    "where returned a Series whose index does not match the "
                    "table's, so it cannot be aligned; return a mask over the "
                    "frame that was passed in"
                )
            result = result.reindex(site_table.index)
        if isinstance(result.dtype, pd.BooleanDtype):
            if result.isna().any():
                raise ValueError(
                    f"where returned a nullable boolean mask with "
                    f"{int(result.isna().sum())} missing value(s); pandas cannot "
                    "index with those. Say what a missing label means, for "
                    'example (t["pft"] == "DBF").fillna(False)'
                )
            result = result.astype(bool)

    mask = np.asarray(result)
    if mask.dtype != bool:
        raise ValueError(
            f"where must return a boolean mask, got dtype {mask.dtype}"
        )
    if mask.shape != (len(site_table),):
        raise ValueError(
            f"where returned a mask of shape {mask.shape}, expected "
            f"{(len(site_table),)}"
        )
    return mask


def _draw_n_random(
    site_table: pd.DataFrame, n_random: int, seed: int | None
) -> pd.DataFrame:
    """*n_random* rows drawn without replacement, in ascending ``site_id`` order."""
    n_random = as_bounded_integer(
        n_random, minimum=0, maximum=len(site_table), message_name="n_random"
    )
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(site_table), size=n_random, replace=False))
    return site_table.iloc[positions].sort_values(SITE_ID)


# ── checks ────────────────────────────────────────────────────────────────────


def check_site_table_locates_the_sites(site_table: Any, site_ids: Iterable[int]) -> None:
    """The site table, as :func:`load_sites` or :func:`site_lookup` gives it, locates
    each of *site_ids* once."""
    check_site_table_is_keyed_on_site_ids(site_table)
    check_site_table_has_locations(site_table)
    check_site_table_lists_the_sites(site_table, site_ids)


def check_site_table_is_keyed_on_site_ids(site_table: Any) -> None:
    """The site table is a ``DataFrame`` whose integer ``site_id`` lists each site once."""
    check_site_table_is_a_dataframe(site_table)
    check_site_table_has_site_ids(site_table)
    check_site_table_site_ids_are_integers(site_table)
    check_site_table_lists_each_site_once(site_table)


def check_site_table_is_a_dataframe(site_table: Any) -> None:
    """The site table is a ``pandas.DataFrame``."""
    if not isinstance(site_table, pd.DataFrame):
        raise TypeError(
            f"the site table must be a DataFrame, got {type(site_table).__name__}; pass "
            "the table load_sites() returns."
        )


def check_site_table_has_site_ids(site_table: pd.DataFrame) -> None:
    """The site table has a ``site_id`` column or index."""
    if site_table.index.name != SITE_ID and SITE_ID not in site_table.columns:
        raise ValueError(
            f"the site table has no {SITE_ID!r} column or index; pass the table "
            "load_sites() returns, or one keyed by site_lookup()."
        )


def check_site_table_site_ids_are_integers(site_table: pd.DataFrame) -> None:
    """The site table's ``site_id`` holds integers."""
    site_ids = _site_ids_of(site_table)
    if not pd.api.types.is_integer_dtype(site_ids.dtype):
        raise TypeError(
            f"the site table's {SITE_ID} must hold integers, got dtype {site_ids.dtype}; "
            "cast it with .astype(int) where every value is whole."
        )


def check_site_table_lists_each_site_once(site_table: pd.DataFrame) -> None:
    """The site table lists no site twice."""
    site_ids = _site_ids_of(site_table)
    if site_ids.has_duplicates:
        repeated = sorted(set(site_ids[site_ids.duplicated()].tolist()))
        raise ValueError(
            f"the site table lists site(s) {truncated(repeated)} more than once; a site "
            "has one row, as load_sites() gives it, so drop the repeated rows."
        )


def check_site_table_has_locations(site_table: pd.DataFrame) -> None:
    """The site table has ``lon`` and ``lat`` columns."""
    absent = [name for name in (LON, LAT) if name not in site_table.columns]
    if absent:
        raise ValueError(
            f"the site table has no {absent} column(s), and needs {LON!r} and {LAT!r} to "
            "locate a site; pass the table load_sites() returns."
        )


def check_site_table_lists_the_sites(
    site_table: pd.DataFrame, site_ids: Iterable[int], *, message_name: str = "site(s)"
) -> None:
    """Every site of *site_ids* is in the site table."""
    wanted = pd.Index(list(site_ids))
    missing = wanted[~wanted.isin(_site_ids_of(site_table))].tolist()
    if missing:
        raise KeyError(
            f"{message_name} {truncated(missing)} are not in the site table, which holds "
            f"{len(site_table)} sites; site ids are never renumbered, so pass a table "
            "holding every site asked for."
        )


def check_site_table_sites_are_all_listed(
    site_table: pd.DataFrame, site_ids: Iterable[int], *, message_name: str
) -> None:
    """Every site of the site table is among *site_ids*."""
    listed = set(pd.Index(list(site_ids)).tolist())
    absent = sorted(set(_site_ids_of(site_table).tolist()) - listed)
    if absent:
        raise ValueError(
            f"{message_name} lack {len(absent)} site(s) of the site table, "
            f"{truncated(absent)}; a whole-pool data source covers every site, so check it "
            "was made for this site table."
        )


def check_sites_are_the_site_table(
    site_table: pd.DataFrame, site_ids: Iterable[int], *, message_name: str
) -> None:
    """*site_ids* are exactly the sites of the site table.

    The group of :func:`check_site_table_lists_the_sites` and
    :func:`check_site_table_sites_are_all_listed`, which a whole-pool raw data
    source's sites are checked with.
    """
    site_ids = list(site_ids)
    check_site_table_lists_the_sites(site_table, site_ids, message_name=message_name)
    check_site_table_sites_are_all_listed(site_table, site_ids, message_name=message_name)
