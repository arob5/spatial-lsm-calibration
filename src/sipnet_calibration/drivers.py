"""The data model for the meteorological drivers, read from the raw files.

Overview
--------
This module defines how the ERA5 driver ensemble is represented -- its
dimensions, coordinates, variable names, units and dtype -- and provides the
functions that read it into that form. It is the single description of that
layout: the tests, the plotting layer and anything else that wants driver data
get the schema from here rather than restating it.

Unlike the site table and the annual constraints, the drivers have **no
processed file**. SIPNET reads the raw ``.clim`` text directly (pySIPNET
symlinks it into the run directory), and the full ensemble is 80,000 files, so
rewriting it into a store would create a large cache that the model never
reads. Instead :func:`load_drivers` parses the raw files for the sites a caller
names and returns the canonical form in memory. The dependency still runs one
way::

    raw/drivers/ERA5_<site>_<member>/ERA5.<member>.<start>.<end>.clim
      -> this module          load_drivers() -> xarray.Dataset

with ``processed/sites/sites.csv`` joined on for the site coordinates. Anyone
wanting a cached subset writes it themselves,
``load_drivers(...).to_zarr(path)``, and reads it back with ``xarray``; nothing
here depends on such a cache existing. ``data/README.md`` documents the source
data and the open questions about it.

Input data
----------
``data/raw/drivers/``
    One directory per site and ensemble member, ``ERA5_<site>_<member>``,
    holding one file ``ERA5.<member>.<start>.<end>.clim``. ``<site>`` is the
    1-8000 site identifier and ``<member>`` the source's 1-based member index.
    :func:`default_drivers_root` says where the directory is expected to be.

    A ``.clim`` file is the 14-column SIPNET climate format: tab-delimited text
    with space-padded fields and no header, one row per timestep, the columns
    of :data:`CLIM_FILE_COLUMNS` in that order. The files are 3-hourly. Three
    columns are constants, asserted at :data:`CLIM_FILE_CONSTANTS`; only
    ``length`` is kept, as the ``timestep_days`` attribute. The ``time`` column
    is a drifting hour-of-day label that must not be used as a timestamp
    (issue #9), so :func:`read_clim_file` uses it only to identify a row's slot
    within its day.

``data/processed/sites/sites.csv``
    The site table, for the ``lon``/``lat`` coordinates and to confirm that the
    requested sites exist. Read through
    :func:`sipnet_calibration.sites.load_sites`; only its ``site_id``, ``lon``
    and ``lat`` columns are used, and ``site_id`` must be unique.

Data model
----------
:func:`load_drivers` returns an ``xarray.Dataset`` shaped as follows.

**Dimensions**: ``member``, ``site``, ``time``.

**Data variables**, all ``float64`` on ``(member, site, time)``, one per
consumed ``.clim`` column, named as :data:`DRIVER_VARIABLES`:

==================== ====================== ========== =============
Processed name       Source column          Units      Aggregation
==================== ====================== ========== =============
``air_temperature``  ``tair``               deg C      mean
``soil_temperature`` ``tsoil``              deg C      mean
``par``              ``par``                mol m-2    sum
``precipitation``    ``precip``             mm         sum
``vpd``              ``vpd``                Pa         mean
``soil_vpd``         ``vpd_soil``           Pa         mean
``vapor_pressure``   ``vpress``             Pa         mean
``wind_speed``       ``wspd``               m s-1      mean
==================== ====================== ========== =============

``par`` and ``precipitation`` are totals over the timestep, which is why they
sum; the rest are means over it. Each variable carries ``units``,
``long_name``, ``source_name`` and ``aggregation`` from
:data:`DRIVER_VARIABLE_ATTRS`, plus ``units_status`` and ``units_provenance``:
the units are the ones the ``.clim`` format documents and SIPNET assumes when
it reads the column, not units confirmed by the producer of these files.

``par`` and ``precipitation`` also carry ``n_values_below_zero``, and ``vpd``,
``soil_vpd`` and ``wind_speed`` carry ``n_values_not_positive``. The source
files hold small negative excursions around zero and exact zeros where SIPNET
would clamp; they are read through unchanged and counted, so that nobody has to
rediscover that ``par > 0`` is not a daylight test.

With ``allow_missing=True`` there is one more variable, ``bool`` on
``(member, site)``::

    driver_present(member, site)    whether a file existed for the pair

and the eight drivers are ``NaN`` where it is ``False``. Without the flag every
requested pair must exist, so the variable is not written.

**Coordinates**

======================= ============ ==========================================
Name                    Dims         Meaning
======================= ============ ==========================================
``member``              ``member``   0-based ``int16``, in ascending order
``source_member_index`` ``member``   the 1-based index in the directory name
``site``                ``site``     handed-down ``int32`` site id, ascending
``lon``, ``lat``        ``site``     from the site table, ``float64``
``time``                ``time``     ``datetime64[ns]``, 3-hourly, see below
======================= ============ ==========================================

**Time.** Labels are the nominal ``year``/``day``/``3 * slot`` instants, built
by :func:`sipnet_calibration.obs_ops.sipnet_time_index` and never from the
``time`` column's value. The coordinate carries ``long_name``,
``time_zone = "UTC"``, ``time_label = "interval_end"``, ``time_label_note``,
``clock_status`` and ``clock_provenance``: the value in the row labeled hour
``h`` covers the interval ``(h - 3, h]`` on a clock consistent with UTC. That
is inferred from the data, not confirmed by the producer, which is what the
status attribute says.

**Attributes** on the dataset: ``title``, ``source_root``, ``source_layout``,
``timestep_days``, ``member_source = "met"``, ``member_correspondence``,
``n_sites``, ``n_members`` and ``coverage`` (``"complete"`` or ``"gaps"``).

**Missing values.** There are none in the source. A ``NaN`` appears only under
``allow_missing=True``, for a whole ``(member, site)`` pair whose file is
absent, and ``driver_present`` says which.

Functions
---------
:func:`load_drivers`
    Read the drivers for the sites named, checking every file on the way, and
    return the Dataset above.

:func:`driver_fields`
    Split the Dataset into canonical fields -- one ``DataArray`` per variable
    with dims ``(member, site, time)`` and its own units. This is the view the
    plotting layer wants.

:func:`read_clim_file`
    Parse one ``.clim`` file exactly, in source column names, and run the
    per-file checks. The building block :func:`load_drivers` is made of, public
    so that tests and one-off surveys apply the same checks the reader does.

:func:`available_members`
    Which member indices have a directory for a given site.

:func:`driver_file`
    The path of the one ``.clim`` file for a site and member.

:func:`default_drivers_root`
    Where the raw directory is expected to be, honoring
    ``$SIPNET_CALIBRATION_DATA``.

Notes
-----
**Why a reader and not a store.** SIPNET consumes the raw text, so a store
would be a second copy that only the analysis side reads, and the full
ensemble is hundreds of gigabytes of text. Reading direct means what is
plotted is parsed from the exact file the model ran on. The cost is that reads
are site-major only: a site's whole record is one file, but one timestep across
the pool means parsing every file. Calibration and the per-site figures need
the former.

**Why the time column is not the timestamp.** The ``time`` column is hour-of-
day from a whole-year ``linspace`` reduced modulo 24 with an off-by-one
endpoint (issue #9). It drifts by up to two hours within a year and is not
monotone within a year. Its drift is always non-negative and below one step,
so ``floor(time / 3)`` identifies the slot in every row, and the drift itself
is asserted so that a corrected upstream file is noticed rather than silently
accepted.

**Why end-of-interval labels.** A PAR-phase test across two sites 54 degrees
apart puts the drivers on a longitude-tracking clock consistent with UTC, with
each row's PAR accumulated over the three hours *ending* at its nominal label.
Keeping the nominal labels and recording ``time_label = "interval_end"`` means
a daily resample groups exactly the eight rows SIPNET itself calls one day,
the same :func:`~sipnet_calibration.obs_ops.sipnet_time_index` applies
unchanged to SIPNET output, and no row acquires a 2011 date. Anything that
needs interval-start semantics reads the label and shifts.

**Why float64.** There is no disk to save, and the text carries up to eight
significant figures in some columns, which float32 does not hold.

**Member indices.** ``member`` is 0-based to match every other product;
``source_member_index`` keeps the 1-based file index beside it so the mapping
to a directory is never guesswork. Whether driver member *i* corresponds to
initial condition member *i* is not established (open question 12 in
``data/README.md``), and ``member_correspondence`` says so.

Usage
-----
Name the sites, get the canonical form::

    from sipnet_calibration.drivers import driver_fields, load_drivers
    from sipnet_calibration.sites import load_sites, select_sites

    sites = select_sites(load_sites(), bbox=(-125, 24, -66, 50), sample=20, seed=0)
    drivers = load_drivers(sites["site_id"])          # every member present

    drivers["air_temperature"].dims                   # ('member', 'site', 'time')
    drivers["par"].attrs["aggregation"]               # 'sum'
    drivers["time"].attrs["time_label"]               # 'interval_end'

    # Two members only, and tolerate sites that lack a file for one of them.
    partial = load_drivers([1, 27], members=[1, 2], allow_missing=True)
    partial["driver_present"].values

For plotting, take the per-variable view::

    fields = driver_fields(drivers)
    fields["par"].attrs["units"]                      # 'mol m-2'

A cached subset, if a workflow wants one, is the caller's business::

    drivers.to_zarr(path)
    import xarray as xr
    xr.open_zarr(path)
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.obs_ops import sipnet_time_index
from sipnet_calibration.sites import DATA_ROOT_ENV_VAR, load_sites

__all__ = [
    "CLIM_FILE_COLUMNS",
    "CLIM_FILE_CONSTANTS",
    "CLOCK_PROVENANCE",
    "CLOCK_STATUS",
    "DRIVER_DIRECTORY_TEMPLATE",
    "DRIVER_FILE_GLOB",
    "DRIVER_PRESENT",
    "DRIVER_VARIABLE_ATTRS",
    "DRIVER_VARIABLES",
    "MEMBER_SOURCE",
    "NEGATIVE_TOLERANCE",
    "SOURCE_VARIABLE_NAMES",
    "STEPS_PER_DAY",
    "TIME_LABEL",
    "TIME_ZONE",
    "TIMESTEP_HOURS",
    "UNITS_PROVENANCE",
    "UNITS_STATUS",
    "available_members",
    "default_drivers_root",
    "driver_fields",
    "driver_file",
    "load_drivers",
    "read_clim_file",
]

#: The 14 columns of a ``.clim`` file, in file order, under the names pySIPNET
#: gives them. These are *source* names; :data:`SOURCE_VARIABLE_NAMES` maps the
#: eight value columns onto processed names.
CLIM_FILE_COLUMNS = (
    "loc",
    "year",
    "day",
    "time",
    "length",
    "tair",
    "tsoil",
    "par",
    "precip",
    "vpd",
    "vpd_soil",
    "vpress",
    "wspd",
    "soil_wetness",
)

#: Columns that must hold one value in every row of every file, and the value.
#: ``loc`` is a location index SIPNET only checks for constancy, ``length`` is
#: the timestep in days, ``soil_wetness`` is a legacy column SIPNET discards.
#: None of the three becomes a variable; ``length`` becomes the
#: ``timestep_days`` attribute once it has been asserted.
CLIM_FILE_CONSTANTS = {"loc": 0, "length": 0.125, "soil_wetness": 0.6}

#: The timestep, in hours, implied by :data:`CLIM_FILE_CONSTANTS`.
TIMESTEP_HOURS = 24 * CLIM_FILE_CONSTANTS["length"]

#: Rows per day, implied by the timestep.
STEPS_PER_DAY = int(round(24 / TIMESTEP_HOURS))

#: Source column -> processed variable name, for the eight columns that become
#: variables. Applied by :func:`load_drivers`; :func:`read_clim_file` keeps the
#: source names because a ``.clim`` row is positional and carries no identity
#: of its own.
SOURCE_VARIABLE_NAMES = {
    "tair": "air_temperature",
    "tsoil": "soil_temperature",
    "par": "par",
    "precip": "precipitation",
    "vpd": "vpd",
    "vpd_soil": "soil_vpd",
    "vpress": "vapor_pressure",
    "wspd": "wind_speed",
}

#: The driver variables, by processed name, in file column order.
DRIVER_VARIABLES = tuple(SOURCE_VARIABLE_NAMES.values())

#: What is and is not settled about the units below.
UNITS_STATUS = "format_documented"

#: Why. Recorded on every variable so that no consumer can take the units as
#: confirmed.
UNITS_PROVENANCE = (
    "The units the SIPNET climate-file format documents for this column "
    "(pySIPNET climate.py, and the conversions in sipnet.c), which is what "
    "SIPNET assumes when it reads the file. Whether the producer wrote the "
    "values in these units has not been confirmed; the magnitudes are "
    "consistent with them, which is evidence and not confirmation."
)

#: Per-variable metadata, by processed name. ``aggregation`` is the temporal
#: rule the variable's kind implies: ``par`` and ``precipitation`` are totals
#: over the timestep and sum; the rest are means over it and average.
DRIVER_VARIABLE_ATTRS = {
    "air_temperature": {
        "units": "deg C",
        "long_name": "Mean air temperature over the timestep",
        "source_name": "tair",
        "aggregation": "mean",
    },
    "soil_temperature": {
        "units": "deg C",
        "long_name": "Mean soil temperature over the timestep",
        "source_name": "tsoil",
        "aggregation": "mean",
    },
    "par": {
        "units": "mol m-2",
        "long_name": "Photosynthetically active radiation, total over the timestep",
        "source_name": "par",
        "aggregation": "sum",
    },
    "precipitation": {
        "units": "mm",
        "long_name": "Precipitation, total over the timestep",
        "source_name": "precip",
        "aggregation": "sum",
    },
    "vpd": {
        "units": "Pa",
        "long_name": "Vapor pressure deficit",
        "source_name": "vpd",
        "aggregation": "mean",
    },
    "soil_vpd": {
        "units": "Pa",
        "long_name": "Soil-to-air vapor pressure deficit",
        "source_name": "vpd_soil",
        "aggregation": "mean",
    },
    "vapor_pressure": {
        "units": "Pa",
        "long_name": "Vapor pressure in the canopy airspace",
        "source_name": "vpress",
        "aggregation": "mean",
    },
    "wind_speed": {
        "units": "m s-1",
        "long_name": "Mean wind speed over the timestep",
        "source_name": "wspd",
        "aggregation": "mean",
    },
}

#: The clock the time labels are on, and what a label marks.
TIME_ZONE = "UTC"
TIME_LABEL = "interval_end"

#: How well the clock is established.
CLOCK_STATUS = "inferred"

#: From what. Recorded on the ``time`` coordinate.
CLOCK_PROVENANCE = (
    "The diurnal PAR phase moves with longitude between site 1 (24.6 W) and "
    "site 27 (78.6 W) by the amount a UTC clock requires, which excludes a "
    "fixed local clock; and a PAR-centroid test at both sites places each "
    "row's total over the three hours ending at its nominal label. 'UTC with "
    "end-of-interval labels' and 'UTC-3 with start-of-interval labels' name "
    "the same intervals and cannot be told apart. Not confirmed by the "
    "producer. See issues #8 and #9."
)

#: Which ensemble the ``member`` coordinate indexes. Member indices are
#: meaningful only within one source.
MEMBER_SOURCE = "met"

#: Name of the presence variable written under ``allow_missing=True``.
DRIVER_PRESENT = "driver_present"

#: Per-site-and-member directory under the drivers root, and the file inside
#: it, ``ERA5.<member>.<start>.<end>.clim``. The glob accepts any member and
#: any dates so that a file whose name disagrees with its directory is reported
#: as the mismatch it is rather than as a missing file; the reader checks both
#: against the directory and the data.
DRIVER_DIRECTORY_TEMPLATE = "ERA5_{site}_{member}"
DRIVER_FILE_GLOB = "ERA5.*.clim"

#: How far below zero ``par`` and ``precip`` may go before a file is refused.
#: The source holds excursions of order 1e-5 and 1e-15 that read as generator
#: noise around zero; anything larger is a different problem.
NEGATIVE_TOLERANCE = 1e-4


def default_drivers_root() -> Path:
    """Where the raw driver directory is expected to be.

    ``$SIPNET_CALIBRATION_DATA/raw/drivers`` when that variable is set, and
    otherwise ``data/raw/drivers`` under this checkout. Experiments name their
    paths in ``config.py``.
    """
    root = os.environ.get(DATA_ROOT_ENV_VAR)
    data_root = Path(root) if root else Path(__file__).resolve().parents[2] / "data"
    return data_root / "raw" / "drivers"


def driver_file(root: Path | str, site: int, member: int) -> Path:
    """The ``.clim`` file for one site and one source member index.

    Parameters
    ----------
    root:
        The drivers root, laid out as :data:`DRIVER_DIRECTORY_TEMPLATE`.
    site:
        Site identifier, 1-8000.
    member:
        The source's 1-based member index, as in the directory name.

    Returns
    -------
    pathlib.Path
        The single file matching :data:`DRIVER_FILE_GLOB` in the pair's
        directory.

    Raises
    ------
    FileNotFoundError
        If the directory, or a file matching the glob inside it, is absent.
    ValueError
        If more than one file matches, since the layout promises exactly one.
    """
    name = DRIVER_DIRECTORY_TEMPLATE.format(site=int(site), member=int(member))
    directory = Path(root) / name
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no driver directory for site {site} member {member}: {directory}"
        )
    matches = sorted(directory.glob(DRIVER_FILE_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"{directory} holds no file matching {DRIVER_FILE_GLOB!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{directory} holds {len(matches)} files matching {DRIVER_FILE_GLOB!r}; "
            f"the layout promises one: {[m.name for m in matches]}"
        )
    return matches[0]


def available_members(root: Path | str, site: int) -> tuple[int, ...]:
    """The source member indices that have a directory for *site*.

    Parameters
    ----------
    root:
        The drivers root.
    site:
        Site identifier.

    Returns
    -------
    tuple of int
        1-based member indices in ascending order, possibly empty. Only the
        directory's existence is consulted; whether the file inside it is
        present and well formed is :func:`driver_file` and
        :func:`read_clim_file`'s business. A directory whose name is not
        exactly the template for its numbers, ``ERA5_3_01`` say, is ignored,
        since :func:`driver_file` could not find it either.
    """
    root = Path(root)
    members = []
    pattern = DRIVER_DIRECTORY_TEMPLATE.format(site=int(site), member="*")
    for directory in root.glob(pattern):
        parsed = _site_member_from_directory(directory.name)
        if parsed is None or not directory.is_dir():
            continue
        canonical = DRIVER_DIRECTORY_TEMPLATE.format(site=parsed[0], member=parsed[1])
        if parsed[0] == int(site) and directory.name == canonical:
            members.append(parsed[1])
    return tuple(sorted(members))


def read_clim_file(path: Path | str) -> pd.DataFrame:
    """Parse one ``.clim`` file exactly and check it.

    Parameters
    ----------
    path:
        The file to read.

    Returns
    -------
    pandas.DataFrame
        The 14 columns of :data:`CLIM_FILE_COLUMNS` under their *source*
        names, one row per timestep in file order. ``year`` and ``day`` are
        ``int32``; every other column is ``float64``, parsed with
        ``float_precision="round_trip"`` so the value is exactly the text.

    Raises
    ------
    ValueError
        If any per-file check fails: an empty file, a row without 14 fields, a
        field that is not a number, a non-finite value, a ``year`` or ``day``
        that is not a whole number, a constant column off its value, a day
        without exactly :data:`STEPS_PER_DAY` rows, days not running
        ``1..n_days`` within each year, years not contiguous or not in
        ascending order, a ``time`` column that does not follow the
        drifting-label model of issue #9, or ``par``/``precip`` further below
        zero than :data:`NEGATIVE_TOLERANCE`. The message names the file and
        the invariant.

    Notes
    -----
    The checks live here rather than in :func:`load_drivers` so that a file is
    checked wherever it is parsed, and so that a one-off survey over the whole
    ensemble applies exactly the checks the reader does.

    The drift model is asserted, not merely tolerated, so that a corrected
    upstream regeneration is noticed. When that happens the check, not the
    caller, is what needs changing.
    """
    path = Path(path)
    try:
        raw = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            dtype=np.float64,
            float_precision="round_trip",
            # A field reading "NA" must stay a parse failure, not become a
            # quiet null; and no field may be absorbed into an index.
            keep_default_na=False,
            na_values=[],
            index_col=False,
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{path}: holds no rows") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"{path}: could not be parsed as a .clim file: {error}") from error
    except ValueError as error:
        raise ValueError(
            f"{path}: a field could not be read as a number ({error}). A row with "
            "fewer than 14 fields shows up here as an empty field."
        ) from error

    _check_column_count(raw, path)
    raw.columns = list(CLIM_FILE_COLUMNS)
    _check_no_missing_values(raw, path)

    frame = raw
    for column in ("year", "day"):
        values = frame[column].to_numpy()
        if np.any(values != np.floor(values)):
            raise ValueError(f"{path}: {column} holds non-integer values")
        frame[column] = values.astype(np.int32)

    _check_constant_columns(frame, path)
    _check_day_structure(frame, path)
    _check_time_column_follows_drift_model(frame, path)
    _check_negative_excursions_bounded(frame, path)
    return frame


def load_drivers(
    sites: Iterable[int],
    *,
    members: Iterable[int] | None = None,
    root: Path | str | None = None,
    sites_table: pd.DataFrame | None = None,
    allow_missing: bool = False,
) -> xr.Dataset:
    """Read the drivers for the given sites into the canonical form.

    Parameters
    ----------
    sites:
        Site identifiers to read, any iterable of integers. Returned in
        ascending order whatever order they are given in, duplicates dropped.
        Every one must be in the site table.
    members:
        Source member indices (1-based, as in the directory names) to read,
        any iterable of integers, likewise sorted and de-duplicated. ``None``
        means every member that has a directory for any of the requested
        sites.
    root:
        The drivers root. Defaults to :func:`default_drivers_root`.
    sites_table:
        The site table, as :func:`sipnet_calibration.sites.load_sites` returns
        it. Loaded from its default location when ``None``. Only ``site_id``,
        ``lon`` and ``lat`` are read, and ``site_id`` must be unique.
    allow_missing:
        What to do about a ``(site, member)`` pair with no file. ``False``, the
        default, raises, because a missing driver member that became ``NaN``
        would propagate silently through any statistic over members. ``True``
        fills the pair with ``NaN`` and adds :data:`DRIVER_PRESENT`. At least
        one requested pair must have a file either way.

    Returns
    -------
    xarray.Dataset
        The `Data model`_ described in the module docstring: the eight
        :data:`DRIVER_VARIABLES` on ``(member, site, time)``, ``float64``, with
        ``lon``/``lat`` on ``site`` and ``source_member_index`` on ``member``.

    Raises
    ------
    FileNotFoundError
        If the root does not exist; if *members* is ``None`` and no requested
        site has a driver directory; if no requested pair has a file at all;
        or if a requested pair has no file and *allow_missing* is ``False``.
    ValueError
        If *sites* or *members* is empty, or holds anything but positive whole
        numbers; if the site table lacks ``site_id``, ``lon`` or ``lat`` or
        repeats a ``site_id``; if a site is not in the site table; if a pair's
        directory holds more than one ``.clim`` file; if a file fails
        :func:`read_clim_file`'s checks, its name does not follow the
        template, the directory and file-name members disagree, or the dates
        in the file name do not match its first and last day; or if two files
        do not share one ``(year, day, time)`` grid, since the ``time``
        coordinate is built once and applied to every file.

    Notes
    -----
    Parsing costs about a tenth of a second per file, so ten sites at ten
    members take about ten seconds and two hundred sites a few minutes; the
    Notes in the module docstring say why this is preferred to a store. Memory
    is about 2.4 MB per site-member.

    Values are read through unchanged: negative excursions of ``par`` and
    ``precipitation`` around zero, and zeros of ``vpd``, ``soil_vpd`` and
    ``wind_speed`` that SIPNET would clamp, are counted into the variable
    attributes rather than altered.
    """
    root = Path(root) if root is not None else default_drivers_root()
    if not root.is_dir():
        raise FileNotFoundError(f"drivers root {root} is not a directory")

    site_ids = _site_ids(sites)
    table = sites_table if sites_table is not None else load_sites()
    _check_site_table_is_usable(table)
    _check_sites_are_in_the_site_table(site_ids, table)

    member_ids = _member_ids(members, root=root, sites=site_ids)

    paths, present = _locate_files(root, sites=site_ids, members=member_ids)
    if not present.any():
        raise FileNotFoundError(
            f"no driver files under {root} for sites {site_ids.tolist()} and "
            f"members {member_ids.tolist()}"
        )
    if not allow_missing:
        _check_members_complete(present, sites=site_ids, members=member_ids, root=root)

    arrays, time = _read_all(paths, present)
    dataset = _assemble(
        arrays,
        present=present,
        time=time,
        sites=site_ids,
        members=member_ids,
        table=table,
        root=root,
        allow_missing=allow_missing,
    )
    return dataset


def driver_fields(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """One ``DataArray`` per driver variable, in :data:`DRIVER_VARIABLES` order.

    Each field has dims ``(member, site, time)``, is named for its variable,
    carries that variable's attributes from :data:`DRIVER_VARIABLE_ATTRS`
    with the units caveat attached, and keeps ``lon``/``lat`` and
    ``source_member_index`` as non-dimension coordinates -- the canonical
    field shape, which the Dataset already is per variable. This is the view
    facet-by-variable consumes, matching
    :func:`sipnet_calibration.constraints.constraint_fields`.

    Parameters
    ----------
    dataset:
        As returned by :func:`load_drivers`.

    Returns
    -------
    dict
        Keyed by processed variable name. :data:`DRIVER_PRESENT`, if present,
        is not a field and is left out.

    Raises
    ------
    ValueError
        If any of :data:`DRIVER_VARIABLES` is absent from *dataset*.
    """
    missing = [name for name in DRIVER_VARIABLES if name not in dataset.data_vars]
    if missing:
        raise ValueError(
            f"dataset is missing driver variables {missing}; found "
            f"{sorted(dataset.data_vars)}"
        )
    fields = {}
    for name in DRIVER_VARIABLES:
        field = dataset[name].copy(deep=False)
        field.attrs = {**_variable_attrs(name), **dataset[name].attrs}
        fields[name] = field
    return fields


# ── supporting helpers ────────────────────────────────────────────────────────

_DIRECTORY_PATTERN = re.compile(r"^ERA5_(\d+)_(\d+)$")
_FILE_PATTERN = re.compile(r"^ERA5\.(\d+)\.(\d{4}-\d{2}-\d{2})\.(\d{4}-\d{2}-\d{2})\.clim$")

#: Source columns whose sub-zero values are counted, and the attribute name.
_COUNT_BELOW_ZERO = {"par": "n_values_below_zero", "precip": "n_values_below_zero"}
_COUNT_NOT_POSITIVE = {
    "vpd": "n_values_not_positive",
    "vpd_soil": "n_values_not_positive",
    "wspd": "n_values_not_positive",
}


def _site_member_from_directory(name: str) -> tuple[int, int] | None:
    """``(site, member)`` from an ``ERA5_<site>_<member>`` name, else ``None``."""
    match = _DIRECTORY_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _dates_from_file_name(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The ``<start>`` and ``<end>`` dates embedded in a ``.clim`` file name."""
    match = _FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(
            f"{path}: file name does not follow ERA5.<member>.<start>.<end>.clim"
        )
    try:
        return pd.Timestamp(match.group(2)), pd.Timestamp(match.group(3))
    except ValueError as error:
        raise ValueError(f"{path}: file name carries an invalid date: {error}") from error


def _time_axis(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """The nominal timestamps of one parsed file.

    Built by :func:`sipnet_calibration.obs_ops.sipnet_time_index` from
    ``year``, ``day`` and the slot the ``time`` column identifies. Every file
    read in one :func:`load_drivers` call must produce the same axis.
    """
    return sipnet_time_index(
        year=frame["year"].to_numpy(),
        day_of_year=frame["day"].to_numpy(),
        hours_since_midnight=frame["time"].to_numpy(),
        timestep_hours=TIMESTEP_HOURS,
    )


def _variable_attrs(name: str) -> dict[str, str]:
    """Attributes for one variable, with the units caveat attached."""
    return {
        **DRIVER_VARIABLE_ATTRS[name],
        "units_status": UNITS_STATUS,
        "units_provenance": UNITS_PROVENANCE,
    }


def _time_attrs() -> dict[str, str]:
    """Attributes for the ``time`` coordinate: clock, label, and their status."""
    return {
        "long_name": "Nominal timestamp of the timestep",
        "time_zone": TIME_ZONE,
        "time_label": TIME_LABEL,
        "time_label_note": (
            "The value in the row labeled hour h covers the interval (h - 3, h]. "
            "Labels are the nominal year/day/3*slot instants; the source's own "
            "time column drifts and is not used (issue #9)."
        ),
        "clock_status": CLOCK_STATUS,
        "clock_provenance": CLOCK_PROVENANCE,
    }


def _site_ids(sites: Iterable[int]) -> np.ndarray:
    """Requested sites as a sorted, de-duplicated ``int32`` array."""
    return _positive_integers(sites, name="site identifiers", dtype=np.int32)


def _member_ids(
    members: Iterable[int] | None, *, root: Path, sites: np.ndarray
) -> np.ndarray:
    """Requested members as a sorted ``int16`` array, discovered when ``None``."""
    if members is None:
        found: set[int] = set()
        for site in sites:
            found.update(available_members(root, int(site)))
        if not found:
            raise FileNotFoundError(
                f"no driver directories under {root} for sites {sites.tolist()}"
            )
        return np.array(sorted(found), dtype=np.int16)
    return _positive_integers(
        members,
        name="member indices (the source's 1-based directory indices)",
        dtype=np.int16,
    )


def _positive_integers(values: Iterable[int], *, name: str, dtype) -> np.ndarray:
    """*values* as a sorted, de-duplicated array of *dtype*, or a clear error.

    Strings, booleans, non-whole floats, non-finite values, non-positive
    values and anything that would wrap when narrowed to *dtype* are refused,
    since each would otherwise resolve to a plausible-looking wrong directory.
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(
            f"{name} must be an iterable of integers, got {type(values).__name__}"
        )
    array = np.asarray(list(values))
    if array.size == 0:
        raise ValueError(f"no {name} requested")
    if array.dtype.kind == "f":
        if np.any(~np.isfinite(array)) or np.any(array != np.floor(array)):
            raise ValueError(f"{name} must be whole numbers, found non-integer values")
    elif array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integers, got {array.dtype}")
    array = np.unique(array.astype(np.int64))
    limit = np.iinfo(dtype).max
    if np.any(array < 1) or np.any(array > limit):
        bad = array[(array < 1) | (array > limit)]
        raise ValueError(f"{name} must lie within 1..{limit}, found {bad[:5].tolist()}")
    return array.astype(dtype)


def _locate_files(
    root: Path, *, sites: np.ndarray, members: np.ndarray
) -> tuple[dict[tuple[int, int], Path], np.ndarray]:
    """Paths for every ``(member, site)`` pair that has one, and a presence mask."""
    present = np.zeros((members.size, sites.size), dtype=bool)
    paths: dict[tuple[int, int], Path] = {}
    for j, site in enumerate(sites):
        for i, member in enumerate(members):
            try:
                paths[(i, j)] = driver_file(root, int(site), int(member))
            except FileNotFoundError:
                continue
            present[i, j] = True
    return paths, present


def _read_all(
    paths: dict[tuple[int, int], Path], present: np.ndarray
) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex]:
    """Parse every located file into ``(member, site, time)`` arrays.

    The first file read fixes the time axis; every later file is checked to
    share its grid before its values are copied in. Cells with no file stay
    ``NaN``.
    """
    reference: pd.DataFrame | None = None
    reference_path: Path | None = None
    arrays: dict[str, np.ndarray] = {}
    time: pd.DatetimeIndex | None = None

    for (i, j), path in sorted(paths.items(), key=lambda item: (item[0][1], item[0][0])):
        frame = read_clim_file(path)
        site, member = _site_member_from_directory(path.parent.name)
        _check_file_name_matches_contents(path, frame, site=site, member=member)
        if reference is None:
            reference, reference_path = frame, path
            time = _time_axis(frame)
            shape = present.shape + (len(frame),)
            arrays = {source: np.full(shape, np.nan) for source in SOURCE_VARIABLE_NAMES}
        else:
            _check_grids_identical(reference, frame, reference_path=reference_path, path=path)
        for source in SOURCE_VARIABLE_NAMES:
            arrays[source][i, j, :] = frame[source].to_numpy()
    assert time is not None
    return arrays, time


def _assemble(
    arrays: dict[str, np.ndarray],
    *,
    present: np.ndarray,
    time: pd.DatetimeIndex,
    sites: np.ndarray,
    members: np.ndarray,
    table: pd.DataFrame,
    root: Path,
    allow_missing: bool,
) -> xr.Dataset:
    """Put the arrays into the Dataset the module docstring describes."""
    dims = ("member", "site", "time")
    coordinates = table.set_index("site_id").loc[sites]
    data_vars = {}
    for source, name in SOURCE_VARIABLE_NAMES.items():
        values = arrays[source]
        attrs = _variable_attrs(name)
        observed = values[present]
        if source in _COUNT_BELOW_ZERO:
            attrs[_COUNT_BELOW_ZERO[source]] = int(np.count_nonzero(observed < 0))
        if source in _COUNT_NOT_POSITIVE:
            attrs[_COUNT_NOT_POSITIVE[source]] = int(np.count_nonzero(observed <= 0))
        data_vars[name] = xr.DataArray(values, dims=dims, attrs=attrs)
    if allow_missing:
        data_vars[DRIVER_PRESENT] = xr.DataArray(
            present,
            dims=("member", "site"),
            attrs={
                "long_name": "Whether a driver file existed for the member and site",
                "comment": "The eight driver variables are NaN where this is False.",
            },
        )

    dataset = xr.Dataset(
        data_vars,
        coords={
            "member": np.arange(members.size, dtype=np.int16),
            "source_member_index": ("member", members.astype(np.int16)),
            "site": sites.astype(np.int32),
            "lon": ("site", coordinates["lon"].to_numpy(np.float64)),
            "lat": ("site", coordinates["lat"].to_numpy(np.float64)),
            "time": time,
        },
    )
    dataset["time"].attrs = _time_attrs()
    dataset["member"].attrs = {
        "long_name": "Ensemble member",
        "comment": (
            "0-based, meaningful only within this source; source_member_index "
            "is the 1-based index in the directory name."
        ),
    }
    dataset["source_member_index"].attrs = {
        "long_name": "Member index in the source directory name (1-based)"
    }
    dataset["site"].attrs = {
        "long_name": "Model site identifier",
        "comment": "The handed-down 1-8000 identifier; never renumbered.",
    }
    dataset.attrs = {
        "title": "ERA5 meteorological drivers in SIPNET climate-file form",
        "source_root": str(root),
        "source_layout": f"{DRIVER_DIRECTORY_TEMPLATE}/{DRIVER_FILE_GLOB}",
        "timestep_days": CLIM_FILE_CONSTANTS["length"],
        "member_source": MEMBER_SOURCE,
        "member_correspondence": (
            "Not established. Whether driver member i corresponds to "
            "initial condition or NEE member i is open question 12 in "
            "data/README.md; nothing here assumes it does."
        ),
        "n_sites": int(sites.size),
        "n_members": int(members.size),
        "coverage": "complete" if present.all() else "gaps",
    }
    return dataset


# ── checks ────────────────────────────────────────────────────────────────────


def _check_column_count(raw: pd.DataFrame, path: Path) -> None:
    """Every row has exactly the 14 fields of :data:`CLIM_FILE_COLUMNS`.

    The parser has already refused a row with more fields than the first and
    an empty file; what remains is a first row of the wrong width, which is a
    different layout.
    """
    expected = len(CLIM_FILE_COLUMNS)
    if raw.shape[1] != expected:
        raise ValueError(
            f"{path}: expected {expected} fields per row, the first row has "
            f"{raw.shape[1]}. Only the 14-column SIPNET climate layout is read."
        )


def _check_no_missing_values(frame: pd.DataFrame, path: Path) -> None:
    """No value is missing or non-finite; SIPNET requires complete drivers.

    The parse itself refuses an empty or non-numeric field, so what reaches
    this check is a field that parsed to an infinity.
    """
    finite = np.isfinite(frame.to_numpy())
    if not finite.all():
        rows, columns = np.nonzero(~finite)
        raise ValueError(
            f"{path}: {rows.size} non-finite value(s); first at data row "
            f"{int(rows[0])}, column {frame.columns[int(columns[0])]!r}"
        )


def _check_constant_columns(frame: pd.DataFrame, path: Path) -> None:
    """``loc``, ``length`` and ``soil_wetness`` hold :data:`CLIM_FILE_CONSTANTS`.

    ``length`` is the one that matters: the slot arithmetic and the
    ``timestep_days`` attribute both assume it, so a file with a different
    timestep must be refused rather than mislabeled.
    """
    for column, expected in CLIM_FILE_CONSTANTS.items():
        values = frame[column].to_numpy()
        if not np.all(values == expected):
            seen = np.unique(values[values != expected])
            raise ValueError(
                f"{path}: {column} must be {expected!r} in every row; found "
                f"{seen[:5].tolist()}"
                + (
                    ". A different length is a different timestep, which this "
                    "reader does not handle."
                    if column == "length"
                    else ""
                )
            )


def _check_day_structure(frame: pd.DataFrame, path: Path) -> None:
    """Exactly :data:`STEPS_PER_DAY` rows per day and days ``1..n_days`` per year.

    ``n_days`` must be 365 or 366 according to the year, and the years must be
    contiguous. This is the structure the time axis is built from, so any
    departure would produce a wrong axis rather than an error downstream.
    """
    year = frame["year"].to_numpy()
    day = frame["day"].to_numpy()
    years = np.unique(year)
    if np.any(np.diff(years) != 1):
        raise ValueError(f"{path}: years are not contiguous: {years.tolist()}")
    # Rows must be grouped by year in ascending order for the per-year slices
    # below to be the years they claim to be.
    if np.any(np.diff(year) < 0):
        raise ValueError(f"{path}: rows are not in ascending year order")

    for value in years:
        rows = day[year == value]
        n_days = 366 if pd.Timestamp(int(value), 1, 1).is_leap_year else 365
        expected = np.repeat(np.arange(1, n_days + 1), STEPS_PER_DAY)
        if rows.size != expected.size:
            raise ValueError(
                f"{path}: year {value} has {rows.size} rows, expected "
                f"{expected.size} ({n_days} days x {STEPS_PER_DAY} steps)"
            )
        if not np.array_equal(rows, expected):
            first = int(np.flatnonzero(rows != expected)[0])
            raise ValueError(
                f"{path}: year {value} does not run 1..{n_days} with "
                f"{STEPS_PER_DAY} rows per day; first departure at row {first} "
                f"of the year (day {int(rows[first])}, expected {int(expected[first])})"
            )


def _check_time_column_follows_drift_model(frame: pd.DataFrame, path: Path) -> None:
    """The ``time`` column is the modulo-24 ``linspace`` of issue #9.

    For a year of ``n_days``, ``linspace(0, 24 * n_days - 1, STEPS_PER_DAY *
    n_days) % 24`` must reproduce the column to within 1e-5 h. The label is
    not used for anything, so this check exists only so that a file *without*
    the artifact is noticed: it would mean the generator was corrected, and
    the drift model documented here would then be wrong.
    """
    year = frame["year"].to_numpy()
    time = frame["time"].to_numpy()
    for value in np.unique(year):
        labels = time[year == value]
        n_days = labels.size // STEPS_PER_DAY
        model = np.linspace(0, 24 * n_days - 1, labels.size) % 24
        worst = float(np.max(np.abs(labels - model)))
        if worst > 1e-5:
            raise ValueError(
                f"{path}: the time column in {value} departs from the "
                f"modulo-24 linspace model of issue #9 by up to {worst:.3g} h. "
                "Either the file was regenerated without the artifact, in which "
                "case this check and the documentation need updating, or it is "
                "not a file this reader understands."
            )


def _check_negative_excursions_bounded(frame: pd.DataFrame, path: Path) -> None:
    """``par`` and ``precip`` never fall below ``-NEGATIVE_TOLERANCE``.

    Small negatives are known and read through; a large one would be a
    different kind of problem and is refused.
    """
    for column in ("par", "precip"):
        values = frame[column].to_numpy()
        low = values < -NEGATIVE_TOLERANCE
        if low.any():
            raise ValueError(
                f"{path}: {int(low.sum())} {column} value(s) below "
                f"-{NEGATIVE_TOLERANCE:g}, the lowest {values.min():.4g}. Small "
                "negative excursions around zero are known; these are not small."
            )


def _check_file_name_matches_contents(
    path: Path, frame: pd.DataFrame, *, site: int, member: int
) -> None:
    """The directory's site and member agree with the file name and the data.

    The member index appears in both the directory and the file name and the
    two must agree; the ``<start>`` and ``<end>`` dates in the file name must
    be the first and last day the data covers.
    """
    match = _FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(
            f"{path}: file name does not follow ERA5.<member>.<start>.<end>.clim"
        )
    if int(match.group(1)) != member:
        raise ValueError(
            f"{path}: the file name says member {int(match.group(1))}, the "
            f"directory says member {member}"
        )
    start, end = _dates_from_file_name(path)
    first = pd.Timestamp(int(frame["year"].iloc[0]), 1, 1) + pd.Timedelta(
        days=int(frame["day"].iloc[0]) - 1
    )
    last = pd.Timestamp(int(frame["year"].iloc[-1]), 1, 1) + pd.Timedelta(
        days=int(frame["day"].iloc[-1]) - 1
    )
    if (start, end) != (first, last):
        raise ValueError(
            f"{path}: the file name covers {start.date()} to {end.date()} but the "
            f"data runs {first.date()} to {last.date()}"
        )


def _check_grids_identical(
    reference: pd.DataFrame, frame: pd.DataFrame, *, reference_path: Path, path: Path
) -> None:
    """Two files share one ``(year, day, time)`` grid, value for value.

    The ``time`` coordinate is built from the first file read and applied to
    all of them, which is sound only if the grids are the same.
    """
    if len(frame) != len(reference):
        raise ValueError(
            f"{path} has {len(frame)} rows where {reference_path} has "
            f"{len(reference)}; every file read together must share one grid"
        )
    for column in ("year", "day", "time"):
        a = reference[column].to_numpy()
        b = frame[column].to_numpy()
        if not np.array_equal(a, b):
            first = int(np.flatnonzero(a != b)[0])
            raise ValueError(
                f"{path}: {column} differs from {reference_path} first at data "
                f"row {first} ({b[first]!r} against {a[first]!r}); every file "
                "read together must share one (year, day, time) grid"
            )


def _check_site_table_is_usable(table: pd.DataFrame) -> None:
    """The site table has the columns read here, and one row per site."""
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"sites_table must be a DataFrame, got {type(table).__name__}")
    missing = [column for column in ("site_id", "lon", "lat") if column not in table.columns]
    if missing:
        raise ValueError(
            f"the site table lacks column(s) {missing}; pass it as load_sites() "
            "returns it, with site_id as a column rather than the index"
        )
    duplicated = table["site_id"][table["site_id"].duplicated()]
    if not duplicated.empty:
        raise ValueError(
            f"the site table repeats site id(s) {sorted(set(duplicated.tolist()))[:5]}"
        )


def _check_sites_are_in_the_site_table(sites: np.ndarray, table: pd.DataFrame) -> None:
    """Every requested site exists in the site table, so it has coordinates."""
    unknown = sorted(set(sites.tolist()) - set(table["site_id"].tolist()))
    if unknown:
        raise ValueError(
            f"{len(unknown)} requested site(s) are not in the site table, for "
            f"example {unknown[:10]}"
        )


def _check_members_complete(
    present: np.ndarray, *, sites: np.ndarray, members: np.ndarray, root: Path
) -> None:
    """Every requested ``(site, member)`` pair has a file, unless gaps are allowed.

    The message lists the missing pairs, and says that ``allow_missing=True``
    reads the rest with ``NaN`` in their place.
    """
    if present.all():
        return
    missing = [
        (int(sites[j]), int(members[i]))
        for i, j in zip(*np.nonzero(~present), strict=True)
    ]
    shown = ", ".join(f"site {s} member {m}" for s, m in missing[:10])
    more = f", and {len(missing) - 10} more" if len(missing) > 10 else ""
    raise FileNotFoundError(
        f"{len(missing)} requested (site, member) pair(s) have no driver file "
        f"under {root}: {shown}{more}. Pass allow_missing=True to read the rest "
        "with NaN in their place and a driver_present array saying which."
    )
