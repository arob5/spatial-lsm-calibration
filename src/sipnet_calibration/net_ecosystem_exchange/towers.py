"""The tower table: which pool site each AmeriFlux tower is, and its clock.

``data/raw/net_ecosystem_exchange/ameriflux_towers.csv`` has one row per
downloaded tower. It records the tower's pool site and how that match was
made, whether the tower is the one whose series the products carry for that
site, the tower's UTC offset and how it was established, and why a tower is
left out. It is the one place those decisions are written down, and it is
tracked, so a change to any of them shows in a diff.

``scripts/raw_sources/build_ameriflux_towers.py`` writes it from the raw files,
AmeriFlux's site listing, the reanalysis's list of towers added to the pool,
and the site table; :func:`read_tower_table` reads it back.

Data model
----------
:func:`read_tower_table` returns a ``pandas.DataFrame``, one row per tower in
ascending ``tower`` order, with the columns of :data:`TOWER_COLUMNS` and the
dtypes of :data:`TOWER_COLUMN_DTYPES`. In the CSV an absent value is the empty
string; in the frame it is ``<NA>`` for a nullable integer, ``NaN`` for a
float and ``""`` for text.

========================== =========== ===================================================
Column                     Dtype       Meaning
========================== =========== ===================================================
``tower``                  ``str``     AmeriFlux site identifier
``site_id``                ``Int32``   the pool site; ``<NA>`` if none
``match_basis``            ``str``     how the site was found (:data:`MATCH_BASES`), or ``""``
``primary``                ``bool``    whether the products carry this tower's series
``primary_reason``         ``str``     why this tower is or is not primary, at a shared site
``excluded_reason``        ``str``     why the tower is left out, or ``""``
``tower_lon``, ``tower_lat`` ``float64`` AmeriFlux's coordinates
``lon_index``, ``lat_index`` ``Int32`` the tower's cell on the pool grid
``distance_m``             ``float64`` tower to its site's center; a diagnostic only
``utc_offset_hours``       ``float64`` the tower's standard-time offset from UTC
``utc_offset_source``      ``str``     where the offset came from
``utc_offset_separation``  ``float64`` how clearly the offset was recovered
``shortwave_lag_steps``    ``Int32``   steps measured shortwave lags ``SW_IN_POT``; <NA> unchecked
``resolution_minutes``     ``int16``   30 or 60
``record_steps``           ``int32``   steps with an ``NEE_VUT_REF`` value in the raw file
``doi``                    ``str``     the site's AmeriFlux FLUXNET DOI
``site_version``           ``str``     the FULLSET version
``igbp``                   ``str``     the site's IGBP class, from AmeriFlux's listing
``source_file``            ``str``     the FULLSET file
``comment``                ``str``     anything else, such as a coordinate disagreement
========================== =========== ===================================================

**The matching rule.** A tower is matched to a pool site on one of three
bases, tried in order, and never by distance:

``named_in_pool``
    The site's name embeds the tower's identifier in parentheses, as
    ``Intermediate hardwood (IHW) (US-Wi1)`` does. Where several sites' names
    embed it, the one whose cell holds the tower is taken, and if none does the
    next basis is tried.
``pool_input_list``
    The site's ``site_name`` is ``ameriflux``, and the tower is the first entry
    of the pool input list (the reanalysis's list of towers added to the pool)
    to fall in its cell.
``same_cell``
    The tower's AmeriFlux coordinates fall in the site's cell, on the pool
    raster's real edges (:meth:`sipnet_calibration.sites.Grid.cell_of`).

**One tower per site.** Where several towers match one site, the products
carry one: the earliest basis in :data:`MATCH_BASES`, then the longer record in
time, then the identifier. A tower whose clock fails its check is never primary.

**The clock.** A tower's offset is recovered from its own ``SW_IN_POT``, which
ONEFlux computes from the site's coordinates and the offset in its metadata, on
the file's own stamps (:func:`recover_utc_offset`). The recovery must be clear,
``utc_offset_separation`` at least :data:`MINIMUM_OFFSET_SEPARATION`, and the
tower's measured shortwave must peak where ``SW_IN_POT`` does, to within
:data:`MAXIMUM_SHORTWAVE_LAG_MINUTES` (:func:`shortwave_lag_steps`); at hourly
resolution that means no lag at all. A tower is also excluded when it has too
little measured shortwave to check, no ``SW_IN_POT`` in 2012-2024, or an offset
that is not a whole number of its steps. The reason is recorded; a lag within
the tolerance but not zero is noted in ``comment``.

Functions
---------
:func:`read_tower_table`
    Read the table and check it.
:func:`build_tower_table`
    Assemble it from per-tower summaries, the site listing, the pool's list
    and the site table. Pure.
:func:`summarize_raw`
    The per-tower summaries of one raw file: offset, clock check, record.
:func:`match_towers`
    The matching rule on its own.
:func:`recover_utc_offset`, :func:`shortwave_lag_steps`
    The two clock computations.
:func:`read_ameriflux_site_list`, :func:`read_pool_input_list`
    The two inputs besides the raw files and the site table.

Notes
-----
**Why the offset is recovered rather than read.** It lives in each site's
BADM (``UTC_OFFSET``), which AmeriFlux distributes only to account holders;
the FULLSET files and AmeriFlux's public site listing do not carry it. The
recovery reads back the value ONEFlux itself used, so it is the producer's
number, and the shortwave check then tests that the data are on that clock.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.net_ecosystem_exchange.names import (
    TOWER,
    ameriflux_site_list_path,
    pool_input_list_path,
    resolve_resolution,
    tower_table_path,
)
from sipnet_calibration.sites import SITE_GRID

__all__ = [
    "MATCH_BASES",
    "MAXIMUM_SHORTWAVE_LAG_MINUTES",
    "MINIMUM_OFFSET_SEPARATION",
    "OffsetFit",
    "TOWER_COLUMNS",
    "TOWER_COLUMN_DTYPES",
    "UTC_OFFSET_RECOVERED",
    "build_tower_table",
    "match_towers",
    "read_ameriflux_site_list",
    "read_pool_input_list",
    "read_tower_table",
    "recover_utc_offset",
    "shortwave_lag_steps",
    "summarize_raw",
]

#: The matching bases, in the order they are tried.
MATCH_BASES = ("named_in_pool", "pool_input_list", "same_cell")

#: ``utc_offset_source`` for an offset recovered from ``SW_IN_POT``.
UTC_OFFSET_RECOVERED = "recovered from SW_IN_POT"

#: The smallest ratio of the runner-up offset's misfit to the best one's that
#: counts as a clear recovery.
MINIMUM_OFFSET_SEPARATION = 2.0

#: The candidate offsets, in hours.
_CANDIDATE_OFFSETS = np.arange(-12.0, 12.01, 0.5)

#: The largest shift of measured shortwave from ``SW_IN_POT`` a tower may show,
#: in minutes: one half-hour step, the resolution of the check at half-hourly
#: towers. At hourly towers only a zero lag passes.
MAXIMUM_SHORTWAVE_LAG_MINUTES = 30

#: The largest shift, in steps either way, the shortwave check looks for.
_MAXIMUM_LAG_STEPS = 4

#: The fewest measured shortwave steps the check needs, in days of steps.
_MINIMUM_MEASURED_DAYS = 30

TOWER_COLUMN_DTYPES: Mapping[str, str] = MappingProxyType(
    {
        "tower": "str",
        "site_id": "Int32",
        "match_basis": "str",
        "primary": "bool",
        "primary_reason": "str",
        "excluded_reason": "str",
        "tower_lon": "float64",
        "tower_lat": "float64",
        "lon_index": "Int32",
        "lat_index": "Int32",
        "distance_m": "float64",
        "utc_offset_hours": "float64",
        "utc_offset_source": "str",
        "utc_offset_separation": "float64",
        "shortwave_lag_steps": "Int32",
        "resolution_minutes": "int16",
        "record_steps": "int32",
        "doi": "str",
        "site_version": "str",
        "igbp": "str",
        "source_file": "str",
        "comment": "str",
    }
)

#: The columns, in file order.
TOWER_COLUMNS: tuple[str, ...] = tuple(TOWER_COLUMN_DTYPES)


@dataclass(frozen=True)
class OffsetFit:
    """The offset that best reproduces a tower's ``SW_IN_POT``, and its runner-up."""

    offset_hours: float
    rmse: float
    runner_up_hours: float
    runner_up_rmse: float

    @property
    def separation(self) -> float:
        """The runner-up's misfit over the best one's; large means clear."""
        if not math.isfinite(self.rmse):
            return math.nan
        return self.runner_up_rmse / self.rmse if self.rmse > 0 else math.inf


def read_tower_table(path: Path | str | None = None) -> pd.DataFrame:
    """Read the tower table and check it against the data model.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with the command that produces it.
    ValueError
        If the columns, dtypes or invariants are not the data model's.
    """
    path = Path(path) if path is not None else tower_table_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file. It is tracked; regenerate it with\n"
            "  python scripts/raw_sources/build_ameriflux_towers.py"
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != TOWER_COLUMNS:
        raise ValueError(f"{path}: columns are {list(frame.columns)}, expected {list(TOWER_COLUMNS)}")
    table = _typed(frame, path)
    _check_tower_table(table, path)
    return table


def read_ameriflux_site_list(path: Path | str | None = None) -> pd.DataFrame:
    """AmeriFlux's site listing, reduced to what the tower table needs.

    Returns
    -------
    pandas.DataFrame
        ``tower``, ``tower_lat``, ``tower_lon``, ``igbp`` and ``doi``, one row
        per site, ascending.
    """
    path = Path(path) if path is not None else ameriflux_site_list_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file. It is not tracked, because it carries contact details; "
            "copy it from beside the FLUXNET download (see "
            "data/raw/net_ecosystem_exchange/provenance.md)."
        )
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    columns = {
        "Site ID": "tower",
        "Latitude (degrees)": "tower_lat",
        "Longitude (degrees)": "tower_lon",
        "Vegetation Abbreviation (IGBP)": "igbp",
        "AmeriFlux FLUXNET DOI": "doi",
    }
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    listing = frame[list(columns)].rename(columns=columns)
    for column in ("tower_lat", "tower_lon"):
        listing[column] = pd.to_numeric(listing[column].replace("", "nan"), errors="coerce")
    if listing["tower"].duplicated().any():
        raise ValueError(f"{path}: a site is listed twice")
    return listing.sort_values("tower", ignore_index=True)


def read_pool_input_list(path: Path | str | None = None) -> pd.DataFrame:
    """The reanalysis's list of AmeriFlux towers added to the site pool.

    Returns
    -------
    pandas.DataFrame
        ``input_order`` (the list's own 1-based row label), ``tower``,
        ``input_lat`` and ``input_lon``, in list order.
    """
    path = Path(path) if path is not None else pool_input_list_path()
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not a file; it is tracked, so restore it from git.")
    # R's write.csv header: an empty name for the row labels, then R's
    # make.names versions of AmeriFlux's column titles.
    expected = ["", "Site.ID", "Latitude..degrees.", "Longitude..degrees.", "Elevation..m."]
    with path.open() as handle:
        header = [name.strip('"') for name in handle.readline().strip().split(",")]
    if header != expected:
        raise ValueError(f"{path}: columns are {header}, expected {expected}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, header=0, names=["row", *expected[1:]])
    listing = pd.DataFrame(
        {
            "input_order": frame["row"].astype(np.int64),
            "tower": frame["Site.ID"].str.strip(),
            "input_lat": frame["Latitude..degrees."].astype(np.float64),
            "input_lon": frame["Longitude..degrees."].astype(np.float64),
        }
    )
    if not (listing["input_order"].diff().dropna() > 0).all():
        raise ValueError(f"{path}: the row labels are not strictly increasing")
    if listing["tower"].duplicated().any():
        raise ValueError(f"{path}: a tower is listed twice: {listing.loc[listing['tower'].duplicated(), 'tower'].tolist()}")
    return listing


def match_towers(
    towers: pd.DataFrame, site_table: pd.DataFrame, pool_input_list: pd.DataFrame
) -> pd.DataFrame:
    """Each tower's pool site under the matching rule.

    Parameters
    ----------
    towers:
        ``tower``, ``tower_lon`` and ``tower_lat``.
    site_table:
        As :func:`sipnet_calibration.sites.load_sites` returns it.
    pool_input_list:
        As :func:`read_pool_input_list` returns it.

    Returns
    -------
    pandas.DataFrame
        ``tower``, ``site_id`` (``<NA>`` if unmatched), ``match_basis``,
        ``lon_index``, ``lat_index``, ``distance_m``, ``comment`` and
        ``excluded_reason``, in the order of *towers*.
    """
    site_of_cell = {
        (int(j), int(k)): int(site)
        for site, j, k in site_table[["site_id", "lon_index", "lat_index"]].itertuples(index=False)
    }
    named = _named_sites(site_table)
    input_owner = _pool_input_owners(pool_input_list, site_table, site_of_cell)
    site_lonlat = site_table.set_index("site_id")[["lon", "lat"]]

    rows = []
    for tower, lon, lat in towers[["tower", "tower_lon", "tower_lat"]].itertuples(index=False):
        cell = _cell_or_none(lon, lat)
        own_site = site_of_cell.get(cell) if cell is not None else None
        site, basis, comment = _match_one(tower, own_site, named, input_owner)
        distance = math.nan
        if site is not None:
            distance = _haversine_m(lon, lat, *site_lonlat.loc[site])
            if own_site != site:
                comment = (
                    f"AmeriFlux's coordinates fall {distance:.0f} m from the site's center, "
                    f"outside its cell; matched by {basis}"
                )
        rows.append(
            {
                "tower": tower,
                "site_id": site,
                "match_basis": basis,
                "lon_index": cell[0] if cell else None,
                "lat_index": cell[1] if cell else None,
                "distance_m": distance,
                "comment": comment,
                "excluded_reason": "" if site is not None else _unmatched_reason(cell, comment),
            }
        )
    matched = pd.DataFrame(rows)
    for column in ("site_id", "lon_index", "lat_index"):
        matched[column] = matched[column].astype("Int32")
    return matched


def recover_utc_offset(
    step_starts: pd.DatetimeIndex,
    step_minutes: int,
    potential_shortwave: np.ndarray,
    lat: float,
    lon: float,
) -> OffsetFit:
    """The offset that best reproduces ONEFlux's ``SW_IN_POT``.

    For each candidate offset from -12 to +12 hours in half hours, top-of-
    atmosphere shortwave is computed at each step's midpoint in UTC (local
    standard time minus the offset), with Spencer's (1971) solar declination,
    equation of time and eccentricity, and compared with ``SW_IN_POT``.

    Parameters
    ----------
    step_starts:
        The steps' starts in local standard time, naive.
    step_minutes:
        The step length.
    potential_shortwave:
        ``SW_IN_POT`` at those steps, ``NaN`` where missing.
    lat, lon:
        The tower's coordinates, in degrees.

    Returns
    -------
    OffsetFit
        The best candidate and the runner-up, with their root mean square
        misfits in W m-2.

    Raises
    ------
    ValueError
        If there is no ``SW_IN_POT`` value to fit.
    """
    keep = np.isfinite(potential_shortwave)
    if not keep.any():
        raise ValueError("no SW_IN_POT value to recover the offset from")
    middle = step_starts[keep] + pd.Timedelta(minutes=step_minutes / 2)
    target = potential_shortwave[keep]
    hours_since_epoch = (middle - pd.Timestamp("1970-01-01")) / pd.Timedelta(hours=1)
    scores = []
    for offset in _CANDIDATE_OFFSETS:
        utc = pd.Timestamp("1970-01-01") + pd.to_timedelta(hours_since_epoch - offset, unit="h")
        predicted = _top_of_atmosphere_shortwave(utc, lat, lon)
        scores.append((float(np.sqrt(np.mean((predicted - target) ** 2))), float(offset)))
    scores.sort()
    (best_rmse, best), (runner_rmse, runner) = scores[0], scores[1]
    return OffsetFit(best, best_rmse, runner, runner_rmse)


def shortwave_lag_steps(
    step_starts: pd.DatetimeIndex,
    step_minutes: int,
    measured_shortwave: np.ndarray,
    measured_flag: np.ndarray,
    potential_shortwave: np.ndarray,
) -> int | None:
    """How many steps measured shortwave's mean day sits from ``SW_IN_POT``'s.

    The mean diurnal cycles of measured ``SW_IN_F`` (``SW_IN_F_QC`` 0) and of
    ``SW_IN_POT`` over the same steps are cross-correlated at shifts of up to
    four steps either way. Zero means the data are on the clock ``SW_IN_POT``
    was computed for; a positive lag means the measured cycle comes later.

    Returns
    -------
    int or None
        The shift of highest correlation, or ``None`` if fewer than 30 days of
        measured steps exist.
    """
    per_day = 1440 // step_minutes
    keep = (measured_flag == 0) & np.isfinite(measured_shortwave) & np.isfinite(potential_shortwave)
    if keep.sum() < _MINIMUM_MEASURED_DAYS * per_day:
        return None
    minute_of_day = step_starts.hour * 60 + step_starts.minute
    slot = np.asarray(minute_of_day // step_minutes)[keep]
    counts = np.bincount(slot, minlength=per_day)
    if (counts == 0).any():
        return None
    measured = np.bincount(slot, weights=measured_shortwave[keep], minlength=per_day) / counts
    potential = np.bincount(slot, weights=potential_shortwave[keep], minlength=per_day) / counts
    lags = range(-_MAXIMUM_LAG_STEPS, _MAXIMUM_LAG_STEPS + 1)
    correlations = [np.corrcoef(np.roll(potential, lag), measured)[0, 1] for lag in lags]
    return int(lags[int(np.argmax(correlations))])


def summarize_raw(raw: xr.Dataset, site_list: pd.DataFrame) -> pd.DataFrame:
    """Per-tower summaries of one raw file: its record, offset and clock check.

    Parameters
    ----------
    raw:
        As :func:`sipnet_calibration.net_ecosystem_exchange.raw.read_raw` returns it.
    site_list:
        As :func:`read_ameriflux_site_list` returns it; the source of each
        tower's coordinates.

    Returns
    -------
    pandas.DataFrame
        ``tower``, ``resolution_minutes``, ``source_file``, ``site_version``,
        ``record_steps``, ``utc_offset_hours``, ``utc_offset_separation`` and
        ``shortwave_lag_steps``.

    Raises
    ------
    ValueError
        If a tower of the raw file is not in the site listing.
    """
    resolution = resolve_resolution(raw.attrs["resolution"])
    starts = resolution.raw_step_starts()
    coordinates = site_list.set_index("tower")
    rows = []
    for tower in raw[TOWER].values.tolist():
        if tower not in coordinates.index:
            raise ValueError(f"tower {tower} is not in AmeriFlux's site listing")
        one = raw.sel({TOWER: tower})
        potential = one["SW_IN_POT"].values
        lat, lon = coordinates.at[tower, "tower_lat"], coordinates.at[tower, "tower_lon"]
        if not (np.isfinite(lat) and np.isfinite(lon)):
            raise ValueError(f"tower {tower} has no coordinates in AmeriFlux's site listing")
        if np.isfinite(potential).any():
            fit = recover_utc_offset(starts, resolution.minutes, potential, lat, lon)
        else:  # no step of the file falls in the raw window
            fit = OffsetFit(math.nan, math.nan, math.nan, math.nan)
        lag = shortwave_lag_steps(
            starts, resolution.minutes, one["SW_IN_F"].values, one["SW_IN_F_QC"].values, potential
        )
        rows.append(
            {
                "tower": tower,
                "resolution_minutes": resolution.minutes,
                "source_file": str(one["source_file"].values),
                "site_version": str(one["site_version"].values),
                "record_steps": int(np.isfinite(one["NEE_VUT_REF"].values).sum()),
                "utc_offset_hours": fit.offset_hours,
                "utc_offset_separation": fit.separation,
                "shortwave_lag_steps": lag,
            }
        )
    summary = pd.DataFrame(rows)
    summary["shortwave_lag_steps"] = summary["shortwave_lag_steps"].astype("Int32")
    return summary


def build_tower_table(
    summaries: pd.DataFrame,
    site_list: pd.DataFrame,
    pool_input_list: pd.DataFrame,
    site_table: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the tower table.

    Parameters
    ----------
    summaries:
        One row per tower, as :func:`summarize_raw` returns them, for every
        raw file.
    site_list, pool_input_list, site_table:
        As :func:`read_ameriflux_site_list`, :func:`read_pool_input_list` and
        :func:`sipnet_calibration.sites.load_sites` return them.

    Returns
    -------
    pandas.DataFrame
        The data model above, one row per tower of *summaries*.

    Raises
    ------
    ValueError
        If a tower appears twice in *summaries* or is not in the site listing.
    """
    if summaries["tower"].duplicated().any():
        raise ValueError("a tower appears in more than one raw file")
    table = summaries.merge(site_list, on="tower", how="left", validate="one_to_one")
    if table["tower_lon"].isna().any():
        raise ValueError(f"towers not in the site listing: {table.loc[table['tower_lon'].isna(), 'tower'].tolist()}")
    matched = match_towers(table[["tower", "tower_lon", "tower_lat"]], site_table, pool_input_list)
    table = table.merge(matched, on="tower", how="left", validate="one_to_one")
    table["utc_offset_source"] = np.where(table["utc_offset_hours"].notna(), UTC_OFFSET_RECOVERED, "")
    table["excluded_reason"] = [
        existing or _clock_reason(offset, separation, lag, minutes)
        for existing, offset, separation, lag, minutes in zip(
            table["excluded_reason"],
            table["utc_offset_hours"],
            table["utc_offset_separation"],
            table["shortwave_lag_steps"],
            table["resolution_minutes"],
        )
    ]
    table["comment"] = [
        _joined(comment, _lag_note(lag, minutes))
        for comment, lag, minutes in zip(table["comment"], table["shortwave_lag_steps"], table["resolution_minutes"])
    ]
    table = _choose_primaries(table)
    table = table.sort_values("tower", ignore_index=True)[list(TOWER_COLUMNS)]
    return _typed(table, None)


# ── supporting helpers ────────────────────────────────────────────────────────


#: An AmeriFlux identifier in parentheses inside a pool site name.
_NAMED_IDENTIFIER = re.compile(r"\(([A-Za-z]{2}-[A-Za-z0-9]{3})\)")


def _named_sites(site_table: pd.DataFrame) -> dict[str, list[int]]:
    """Tower identifier to the pool sites whose names embed it."""
    named: dict[str, list[int]] = {}
    for site, name in site_table[["site_id", "site_name"]].itertuples(index=False):
        for identifier in _NAMED_IDENTIFIER.findall(name):
            tower = identifier[:2].upper() + identifier[2:]
            named.setdefault(tower, []).append(int(site))
    return named


def _pool_input_owners(
    pool_input_list: pd.DataFrame, site_table: pd.DataFrame, site_of_cell: Mapping
) -> dict[str, int]:
    """Tower to the ``ameriflux``-labeled site the pool's list placed it in."""
    labeled = set(site_table.loc[site_table["site_name"] == "ameriflux", "site_id"].astype(int))
    owners: dict[str, int] = {}
    seen_cells: set[tuple[int, int]] = set()
    for tower, lon, lat in pool_input_list[["tower", "input_lon", "input_lat"]].itertuples(index=False):
        cell = _cell_or_none(lon, lat)
        if cell is None or cell in seen_cells:
            continue
        seen_cells.add(cell)
        site = site_of_cell.get(cell)
        if site in labeled:
            owners[tower] = site
    return owners


def _match_one(
    tower: str, own_site: int | None, named: Mapping, input_owner: Mapping
) -> tuple[int | None, str, str]:
    candidates = named.get(tower, [])
    if len(candidates) == 1:
        return candidates[0], "named_in_pool", ""
    note = ""
    if len(candidates) > 1:
        if own_site in candidates:
            return own_site, "named_in_pool", f"named by several pool sites {candidates}; its own cell's taken"
        note = f"named by several pool sites {candidates}, none in its cell"
    if tower in input_owner:
        return input_owner[tower], "pool_input_list", note
    if own_site is not None:
        return own_site, "same_cell", note
    return None, "", note


def _unmatched_reason(cell, comment: str) -> str:
    if cell is None:
        return "no pool site: the tower's coordinates are missing or outside the pool grid"
    if comment:
        return f"no pool site: {comment}, and its own cell holds no pool point"
    return "no pool site: no site names the tower and its cell holds no pool point"


def _clock_reason(offset: float, separation: float, lag, step_minutes: int) -> str:
    if not np.isfinite(offset):
        return "clock: no SW_IN_POT in 2012-2024 to recover the offset from"
    if (offset * 60) % step_minutes != 0:
        return f"clock: a {offset} h offset is not a whole number of {step_minutes}-minute steps"
    if not separation >= MINIMUM_OFFSET_SEPARATION:
        return f"clock: the UTC offset is not clearly recovered (separation {separation:.2f})"
    if lag is pd.NA or lag is None:
        return "clock: too little measured shortwave to check the offset against"
    if abs(int(lag)) * step_minutes > MAXIMUM_SHORTWAVE_LAG_MINUTES:
        return f"clock: measured shortwave sits {int(lag) * step_minutes} minutes from SW_IN_POT"
    return ""


def _lag_note(lag, step_minutes: int) -> str:
    """A note for a lag the check allows but that is not zero; excluded lags have a reason instead."""
    if lag is pd.NA or lag is None or int(lag) == 0:
        return ""
    if abs(int(lag)) * step_minutes > MAXIMUM_SHORTWAVE_LAG_MINUTES:
        return ""
    return (
        f"measured shortwave sits {int(lag) * step_minutes} minutes from SW_IN_POT, within "
        f"the check's tolerance of {MAXIMUM_SHORTWAVE_LAG_MINUTES} minutes"
    )


def _joined(*parts: str) -> str:
    return "; ".join(part for part in parts if part)


def _choose_primaries(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["primary"] = False
    table["primary_reason"] = ""
    eligible = table[table["site_id"].notna() & (table["excluded_reason"] == "")]
    rank = {basis: i for i, basis in enumerate(MATCH_BASES)}
    for site, group in eligible.groupby("site_id"):
        ordered = group.assign(
            _rank=group["match_basis"].map(rank),
            _record_minutes=group["record_steps"].astype(np.int64) * group["resolution_minutes"].astype(np.int64),
        ).sort_values(["_rank", "_record_minutes", "tower"], ascending=[True, False, True])
        chosen = ordered.index[0]
        table.loc[chosen, "primary"] = True
        if len(ordered) > 1:
            others = ", ".join(table.loc[ordered.index[1:], "tower"])
            first, second = ordered.iloc[0], ordered.iloc[1]
            why = (
                f"matched {first['match_basis']}, ahead of {second['match_basis']}"
                if first["_rank"] != second["_rank"]
                else "the longer record" if first["_record_minutes"] != second["_record_minutes"]
                else "the first identifier"
            )
            table.loc[chosen, "primary_reason"] = f"{why}; also at this site: {others}"
            for other in ordered.index[1:]:
                table.loc[other, "primary_reason"] = f"not primary: {table.loc[chosen, 'tower']} is"
    return table


def _cell_or_none(lon: float, lat: float) -> tuple[int, int] | None:
    try:
        return SITE_GRID.cell_of(lon, lat)
    except ValueError:
        return None


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6_371_008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = phi2 - phi1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _top_of_atmosphere_shortwave(utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    hour = np.asarray(utc.hour + utc.minute / 60 + utc.second / 3600, dtype=np.float64)
    day = np.asarray(utc.dayofyear, dtype=np.float64)
    gamma = 2 * np.pi / 365.0 * (day - 1 + (hour - 12) / 24)
    declination = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    equation_of_time = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    eccentricity = (
        1.000110 + 0.034221 * np.cos(gamma) + 0.001280 * np.sin(gamma)
        + 0.000719 * np.cos(2 * gamma) + 0.000077 * np.sin(2 * gamma)
    )
    solar_time = hour + lon / 15 + equation_of_time / 60
    hour_angle = np.radians(15 * (solar_time - 12))
    latitude = math.radians(lat)
    cosine_zenith = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    return np.maximum(0.0, 1361.0 * eccentricity * cosine_zenith)


def _typed(frame: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    """*frame*, all text, converted to :data:`TOWER_COLUMN_DTYPES`."""
    where = f"{path}: " if path is not None else ""
    table = pd.DataFrame(index=frame.index)
    for column, dtype in TOWER_COLUMN_DTYPES.items():
        text = frame[column].astype(object).map(lambda value: "" if pd.isna(value) else str(value))
        try:
            if dtype == "str":
                table[column] = text
            elif dtype == "bool":
                if not text.isin(["True", "False"]).all():
                    raise ValueError("expected True or False")
                table[column] = text == "True"
            elif dtype == "float64":
                table[column] = np.array([math.nan if value == "" else float(value) for value in text])
            elif dtype == "Int32":
                table[column] = pd.array(
                    [pd.NA if value == "" else _whole_number(value, np.int32) for value in text], dtype="Int32"
                )
            else:
                table[column] = np.array([_whole_number(value, dtype) for value in text], dtype=dtype)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{where}column {column!r} is not {dtype} ({error})") from error
    return table


def _whole_number(text: str, dtype) -> int:
    """*text* as an integer that fits *dtype*, refusing fractions, infinities and overflow."""
    value = float(text)
    info = np.iinfo(dtype)
    if not math.isfinite(value) or value != math.floor(value) or not info.min <= value <= info.max:
        raise ValueError(f"{text!r} is not a whole number that fits {np.dtype(dtype).name}")
    return int(value)


def _check_tower_table(table: pd.DataFrame, path: Path) -> None:
    """Raise unless *table* holds the tower table's invariants."""
    towers = table["tower"].tolist()
    if towers != sorted(set(towers)):
        raise ValueError(f"{path}: tower is repeated or not ascending")
    matched = table["site_id"].notna()
    if not table.loc[matched, "site_id"].between(1, 8000).all():
        raise ValueError(f"{path}: a site_id is outside 1-8000")
    if not table.loc[matched, "match_basis"].isin(MATCH_BASES).all():
        raise ValueError(f"{path}: a matched tower has a match_basis outside {MATCH_BASES}")
    if (table.loc[~matched, "match_basis"] != "").any():
        raise ValueError(f"{path}: an unmatched tower has a match_basis")
    if (table.loc[~matched, "excluded_reason"] == "").any():
        raise ValueError(f"{path}: an unmatched tower has no excluded_reason")
    primary = table[table["primary"]]
    if primary["site_id"].isna().any() or (primary["excluded_reason"] != "").any():
        raise ValueError(f"{path}: a primary tower is unmatched or excluded")
    if primary["site_id"].duplicated().any():
        raise ValueError(f"{path}: a site has two primary towers")
    offsets = table["utc_offset_hours"]
    if offsets[table["primary"]].isna().any():
        raise ValueError(f"{path}: a primary tower has no utc_offset_hours")
    known = offsets.dropna()
    if (~np.isfinite(known)).any() or ((known * 2) != np.round(known * 2)).any():
        raise ValueError(f"{path}: a utc_offset_hours is not a whole number of half hours")
    if not table["resolution_minutes"].isin([30, 60]).all():
        raise ValueError(f"{path}: resolution_minutes is not 30 or 60")
