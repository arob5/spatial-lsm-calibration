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

Every other processed product joins against this table on ``site_id``, so this
is where the meaning of a site identifier is fixed. ``data/README.md`` documents
the source data, the coordinate reference system and the open questions.

Input data
----------
``data/processed/sites/sites.csv``
    The site table, read by :func:`load_sites`, whose layout is the
    `Data model`_ below. :func:`default_sites_path` says where it is expected
    to be, honoring the ``$SIPNET_CALIBRATION_DATA`` override in
    :data:`DATA_ROOT_ENV_VAR`.

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
table; callers wanting lookup call ``.set_index("site_id")``.

**Missing values.** The empty string, not ``NaN``, in both text columns: the
table is read with ``keep_default_na=False``, so a name that happens to read as
a null word survives. No numeric column can be missing.

**Geography.** The sites are irregular points spanning roughly 7-82 degrees
north and 178-20 degrees west, of which fewer than half fall inside a
conterminous-US bounding box. Their coordinates are cell centers of
:data:`SITE_GRID`, a regular geographic lattice; ``lon_index``/``lat_index`` are
the exact representation of a position and the stored floats are a lossy
rendering of it.

There is deliberately **no plant functional type column**. A PFT labeling is not
an intrinsic property of a site: a calibration may not use PFTs at all, and
different labelings can be applied to the same pool. Labelings are their own
product under ``processed/labelings/``, keyed on ``site_id``, and a caller joins
one on before selecting.

Functions
---------
:func:`load_sites`
    Read the site table and check it against the data model above.

:func:`select_sites`
    A subset of a site table, by identifier, bounding box, arbitrary predicate,
    or random sample. The filters compose.

:func:`default_sites_path`
    Where the table is expected to be.

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

    sites = load_sites()                    # or load_sites(path)

    # Named sites, by identifier, in the order given.
    named = select_sites(sites, ids=[4102, 4113, 5584])

    # A bounding box, as (west, south, east, north), edges included. Every
    # longitude in the pool is negative.
    conus = select_sites(sites, bbox=(-125, 24, -66, 50))

    # Any predicate over the table's columns.
    flux_towers = select_sites(sites, where=lambda s: s["ameriflux_site_id"] != "")

    # The filters compose, and sample is always of whatever survived.
    subset = select_sites(
        sites,
        bbox=(-125, 24, -66, 50),
        where=lambda s: s["ameriflux_site_id"] != "",
        sample=20,
        seed=0,
    )

Select on a labeling by joining it on first, since PFT is not a column here::

    import pandas as pd

    # A real labeling is its own product under processed/labelings/, keyed on
    # site_id. The join is the same whatever the labeling is called.
    labeling = pd.DataFrame({"site_id": [4102, 4113], "pft": ["DBF", "ENF"]})
    deciduous = select_sites(
        sites.merge(labeling, on="site_id"), where=lambda s: s["pft"] == "DBF"
    )

Convert between coordinates and grid indices::

    row = sites.iloc[0]

    # A stored coordinate back to its exact position on the lattice.
    lon_index, lat_index = SITE_GRID.lonlat_to_index(row["lon"], row["lat"])

    # And back to the cell center, which is where the stored value came from.
    lon, lat = SITE_GRID.index_to_lonlat(lon_index, lat_index)

    # Both are vectorized, so a whole column converts at once.
    lon_indices, lat_indices = SITE_GRID.lonlat_to_index(
        sites["lon"].to_numpy(), sites["lat"].to_numpy()
    )
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DATA_ROOT_ENV_VAR",
    "Grid",
    "SITE_COLUMNS",
    "SITE_COLUMN_DTYPES",
    "SITE_GRID",
    "default_sites_path",
    "load_sites",
    "select_sites",
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
    "site_id",
    "lon",
    "lat",
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
SITE_COLUMN_DTYPES = {
    "site_id": np.int32,
    "lon": np.float64,
    "lat": np.float64,
    "lon_index": np.int32,
    "lat_index": np.int32,
    "site_name": str,
    "site_order": np.int32,
    "cluster": np.int8,
    "landcover": np.int8,
    "ameriflux_site_id": str,
}

#: Environment variable naming the ``data/`` directory, for a checkout whose
#: data lives elsewhere. Unset, the repository's own ``data/`` is used.
DATA_ROOT_ENV_VAR = "SIPNET_CALIBRATION_DATA"


def default_sites_path() -> Path:
    """Where the site table is expected to be.

    ``$SIPNET_CALIBRATION_DATA/processed/sites/sites.csv`` when that variable is
    set, and otherwise the ``data/`` directory of this checkout. Experiments name
    their paths in ``config.py`` rather than relying on this; it exists so that
    tests, notebooks and the ingest script agree on one default.
    """
    root = os.environ.get(DATA_ROOT_ENV_VAR)
    data_root = Path(root) if root else Path(__file__).resolve().parents[2] / "data"
    return data_root / "processed" / "sites" / "sites.csv"


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
        ``.set_index("site_id")``.

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


def select_sites(
    sites: pd.DataFrame,
    *,
    ids: Iterable[int] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    where: Callable[[pd.DataFrame], object] | None = None,
    sample: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """A subset of the site table.

    The filters compose, and are applied in the order below so that *sample* is
    always a sample of what survived the rest.

    Parameters
    ----------
    sites:
        A site table, from :func:`load_sites`, or one with extra columns joined
        on. Never modified.
    ids:
        Site identifiers to keep; every identifier must exist. The result is in
        the order given, unless *sample* is also passed, which re-sorts by
        ``site_id``.
    bbox:
        ``(west, south, east, north)`` in degrees, edges included. Longitudes are
        negative throughout the pool, so ``(-125, 24, -66, 50)`` is the
        conterminous US and ``(66, 24, 125, 50)`` selects nothing.
    where:
        A callable taking the table and returning a boolean mask over its rows —
        anything ``.loc`` accepts. This is the general filter: it covers the
        columns of the table and any joined on beside them.
    sample:
        Keep this many rows, drawn without replacement, in ascending ``site_id``
        order. Fewer rows available is an error.
    seed:
        Seed for *sample*. Passing one makes the draw reproducible; leaving it
        out does not.

    Returns
    -------
    pandas.DataFrame
        A copy, with the index reset.

    Raises
    ------
    KeyError
        If *ids* names a site the table does not hold.
    ValueError
        If *ids* holds duplicates, *bbox* is malformed, *where* does not return
        a usable mask, or *sample* is negative or exceeds the number of rows
        available.

    Notes
    -----
    There is no ``pft=`` argument. PFT is not a column of the site table (see the
    module docstring). PFT selection can be done by joining a labeling and passing
    ``where``::

        labeled = sites.merge(pd.read_csv(labeling), on="site_id")
        select_sites(labeled, where=lambda t: t["pft"] == "DBF")

    Examples
    --------
    Twenty conterminous-US sites with an Ameriflux counterpart::

        select_sites(
            load_sites(),
            bbox=(-125, 24, -66, 50),
            where=lambda s: s["ameriflux_site_id"] != "",
            sample=20,
            seed=0,
        )
    """
    selected = sites

    if ids is not None:
        selected = _select_by_id(selected, ids)
    if bbox is not None:
        selected = selected.loc[_bbox_mask(selected, bbox)]
    if where is not None:
        selected = selected.loc[_predicate_mask(selected, where)]
    if sample is not None:
        selected = _draw_sample(selected, sample, seed)

    return selected.reset_index(drop=True)


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
    site_ids = table["site_id"].to_numpy()
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


def _select_by_id(sites: pd.DataFrame, ids: Iterable[int]) -> pd.DataFrame:
    """Rows for *ids*, in the order given. Raises on an unknown identifier.

    Selection is positional rather than ``set_index(...).loc[...]``, which would
    change ``site_id``'s dtype and move it to the first column -- so the frame
    this path returns would differ in shape from the one every other path
    returns, on a table with joined columns.
    """
    wanted = []
    for site_id in ids:
        as_int = int(site_id)
        if as_int != site_id:
            raise ValueError(f"ids must be whole numbers, got {site_id!r}")
        wanted.append(as_int)
    if len(set(wanted)) != len(wanted):
        raise ValueError("ids holds duplicate site ids")

    site_ids = sites["site_id"]
    if site_ids.duplicated().any():
        repeated = site_ids[site_ids.duplicated()].unique().tolist()
        raise ValueError(
            f"the table holds {len(repeated)} repeated site id(s) "
            f"{sorted(repeated)[:10]}, so ids= would return more rows than it "
            "was asked for; de-duplicate it first"
        )

    position = pd.Series(np.arange(len(sites)), index=site_ids.to_numpy())
    unknown = [site_id for site_id in wanted if site_id not in position.index]
    if unknown:
        raise KeyError(
            f"{len(unknown)} site id(s) are not in the table: {unknown[:10]}"
        )
    return sites.iloc[position.loc[wanted].to_numpy()]


def _bbox_mask(sites: pd.DataFrame, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """A boolean mask of the sites inside *bbox*, edges included."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (west, south, east, north), got {bbox!r}")
    west, south, east, north = (float(value) for value in bbox)
    if west > east:
        raise ValueError(
            f"bbox west {west} is east of east {east}; the pool spans "
            f"{SITE_GRID.west} to {SITE_GRID.east}, all negative, and this "
            "function does not wrap the antimeridian"
        )
    if south > north:
        raise ValueError(f"bbox south {south} is north of north {north}")
    lon = sites["lon"].to_numpy()
    lat = sites["lat"].to_numpy()
    return (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)


def _predicate_mask(sites: pd.DataFrame, where: Callable[[pd.DataFrame], object]) -> np.ndarray:
    """The mask *where* returns, aligned and checked before it is used.

    A ``Series`` is aligned on its index, the way ``.loc`` would, rather than
    being read positionally. Reading it positionally is the dangerous case: a
    reordered mask of the right length then selects the wrong rows and every
    shape and dtype check still passes.
    """
    result = where(sites)

    if isinstance(result, pd.Series):
        if not result.index.equals(sites.index):
            if len(result) != len(sites) or set(result.index) != set(sites.index):
                raise ValueError(
                    "where returned a Series whose index does not match the "
                    "table's, so it cannot be aligned; return a mask over the "
                    "frame that was passed in"
                )
            result = result.reindex(sites.index)
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
    if mask.shape != (len(sites),):
        raise ValueError(
            f"where returned a mask of shape {mask.shape}, expected "
            f"{(len(sites),)}"
        )
    return mask


def _draw_sample(sites: pd.DataFrame, sample: int, seed: int | None) -> pd.DataFrame:
    """*sample* rows drawn without replacement, in ascending ``site_id`` order."""
    if sample < 0:
        raise ValueError(f"sample must not be negative, got {sample}")
    if sample > len(sites):
        raise ValueError(
            f"asked for a sample of {sample} from {len(sites)} site(s) available"
        )
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(sites), size=sample, replace=False))
    return sites.iloc[positions].sort_values("site_id")
