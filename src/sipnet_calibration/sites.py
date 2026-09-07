"""Helpers for working with sites and their geography.

The site table is read from ``data/processed/sites/sites.csv``, produced by
``scripts/ingest_sites.py`` from the point shapefile in ``data/raw/sites/`` and
``data/site_id_map.csv`` (Ameriflux ``Site_ID`` -> integer site id, exact
matching). :data:`SITE_COLUMNS` is the column set; ``data/README.md`` describes
what each column means.

There is deliberately no plant functional type column. A PFT labeling is not an
intrinsic property of a site: calibrations may or may not use PFTs, and different
PFT labelings can be applied to the same site pool. Labelings are their own product,
``processed/labelings/``, keyed on ``site_id``, and a caller joins one on before
selecting.

``select_sites`` is a site selection helper: subsetting by bounding box, by an
arbitrary predicate or to a random sample.

Note the sites are 8000 *irregular points* spanning 7-82 deg N and
178 W-20 W. Only ~3640 fall inside a CONUS bounding box.

This module also defines the geographic lattice the sites sit on.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
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


# ── the site table ────────────────────────────────────────────────────────────

#: Columns of ``processed/sites/sites.csv``, in order. The ingest script writes
#: exactly these and :func:`load_sites` requires exactly these, so the two
#: cannot drift apart.
#:
#: ``lon``/``lat`` are the stored coordinates at full precision and
#: ``lon_idx``/``lat_idx`` are their exact representation on :data:`SITE_GRID`;
#: the table carries both because the floats are a lossy rendering of the
#: indices rather than the other way round. ``ameriflux_site_id`` is the empty
#: string for the sites with no Ameriflux counterpart, which is most of them.
SITE_COLUMNS = (
    "site_id",
    "lon",
    "lat",
    "lon_idx",
    "lat_idx",
    "site_name",
    "site_order",
    "cluster",
    "landcover",
    "ameriflux_site_id",
)

#: Dtype per column. The integer widths are the narrowest that hold the data,
#: and the two identifier columns are read as text rather than left to
#: inference, so that a site named ``NA`` stays a name.
SITE_COLUMN_DTYPES = {
    "site_id": np.int32,
    "lon": np.float64,
    "lat": np.float64,
    "lon_idx": np.int32,
    "lat_idx": np.int32,
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
        If the columns are not the expected set, or ``site_id`` is not unique.

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

    table = pd.read_csv(
        csv_path,
        dtype=SITE_COLUMN_DTYPES,
        keep_default_na=False,
        na_values=[],
        float_precision="round_trip",
    )
    _check_site_table(table, source=csv_path)
    return table[list(SITE_COLUMNS)]


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
        Site identifiers to keep. The result is in the order given, every identifier
        must exist.
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
        If *bbox* is malformed, *where* does not return a usable mask, or
        *sample* exceeds the number of rows available.

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


def _select_by_id(sites: pd.DataFrame, ids: Iterable[int]) -> pd.DataFrame:
    """Rows for *ids*, in the order given. Raises on an unknown identifier."""
    wanted = [int(site_id) for site_id in ids]
    known = set(sites["site_id"].tolist())
    unknown = [site_id for site_id in wanted if site_id not in known]
    if unknown:
        raise KeyError(
            f"{len(unknown)} site id(s) are not in the table: {unknown[:10]}"
        )
    if len(set(wanted)) != len(wanted):
        raise ValueError("ids holds duplicate site ids")
    return sites.set_index("site_id").loc[wanted].reset_index()


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
    """The mask *where* returns, checked for shape and dtype before it is used."""
    mask = np.asarray(where(sites))
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
