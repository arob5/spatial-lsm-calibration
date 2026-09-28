"""Observed net ecosystem exchange: the eddy-covariance towers, their clocks,
their pool sites, and the series built from them.

Overview
--------
Net ecosystem exchange is observed at eddy-covariance towers, and each
AmeriFlux tower is, or sits in the cell of, one site of the 8000-site pool.
This package owns that data from AmeriFlux's FLUXNET files to the processed
files the calibration reads. Three modules own a stored artifact -- the raw
files, the tower table, the processed files -- and hold that artifact's schema, writer, reader
and checks; the others hold what those share.

========================  ==================================================
``names``                 the raw files' dims, the two time axes, file locations
``source_files``          AmeriFlux's FULLSET CSVs: the format, and the parser
``raw``                   the converted raw netCDFs: build, encode, read
``towers``                the tower table: sites, clocks, exclusions
``specs``                 what each series is; the registry
``sources``               source readers, raw file to tower series
``processed``             the processed files: build, encode, read, fields
========================  ==================================================

Everything below is re-exported here, so a caller imports from
``sipnet_calibration.net_ecosystem_exchange`` and never names a module.

The data flows one way. The first two arrows are taken once, on the SCC and
wherever the raw files are; the third anywhere::

    AMF_<tower>_FLUXNET_FULLSET_{HH,HR}_<years>_<version>.csv, one per tower  (the download)
      -> scripts/raw_sources/convert_ameriflux_nee.py
      -> raw/net_ecosystem_exchange/ameriflux_nee_{half_hourly,hourly}.nc    not tracked
      -> scripts/raw_sources/build_ameriflux_towers.py    + ameri_sites.tsv, Unmatched_Sites.csv,
      -> raw/net_ecosystem_exchange/ameriflux_towers.csv  tracked             the site table
      -> scripts/ingest_net_ecosystem_exchange.py    one file per series
      -> processed/net_ecosystem_exchange/<name>.nc
      -> this package                                load_net_ecosystem_exchange(),
                                                     net_ecosystem_exchange_fields()

``data/README.md`` documents the source and the open questions;
``data/raw/net_ecosystem_exchange/provenance.md`` records the conversion.

Input data
----------
``data/raw/net_ecosystem_exchange/ameriflux_nee_half_hourly.nc`` and ``..._hourly.nc``
    Every tower's kept FULLSET columns on one local-standard-time axis per
    resolution, in the source's names and values; not tracked. :func:`read_raw`
    checks one.

``data/raw/net_ecosystem_exchange/ameriflux_towers.csv``
    One row per tower: its pool site and how it was matched, whether it is the
    site's primary tower, its UTC offset, and why it is excluded, if it is.
    Tracked. :func:`read_tower_table` checks it.

``data/raw/net_ecosystem_exchange/Unmatched_Sites.csv`` and ``ameri_sites.tsv``
    The reanalysis's list of towers added to the pool (tracked) and AmeriFlux's
    site listing (not tracked); read by :func:`read_pool_input_list` and
    :func:`read_ameriflux_site_list` when the tower table is built.

``data/processed/sites/sites.csv``
    The site table, for the ``lon``/``lat`` of each site.

Data model
----------
:func:`load_net_ecosystem_exchange` returns one processed file as an
``xarray.Dataset``, checked on load against its spec. A processed file holds
one **series**: a single NEE estimate from a single source at a single
resolution, one of :data:`NET_ECOSYSTEM_EXCHANGE`.

**Dimensions**: ``site``, ``time``, ``bounds`` (2).

**Data variables**. ``value`` is always present; the others are present when
the series' source has them, as every AmeriFlux series does.

========================== ================= ========= ===========================================
Name                       Dims              Dtype     Meaning; missing
========================== ================= ========= ===========================================
``value``                  ``(site, time)``  float64   NEE, mean rate over the step; ``NaN``
``quality_flag``           ``(site, time)``  int8      0 measured, 1-3 gap-fill quality; ``-1``
``random_uncertainty``     ``(site, time)``  float64   ONEFlux random uncertainty; ``NaN``
``joint_uncertainty``      ``(site, time)``  float64   random and u*-filtering uncertainty; ``NaN``
``night``                  ``(site, time)``  int8      1 at night, 0 by day; ``-1``
========================== ================= ========= ===========================================

Units are the source's, ``umol m-2 s-1`` with ``constituent = "CO2"``; the
observation operator converts SIPNET's ``g m-2`` of C per step into them.

**Coordinates**

====================== ================== ===================================================
Name                   Dims               Meaning
====================== ================== ===================================================
``site``               ``site``           ``int32``, the pool sites with a primary tower, ascending
``lon``, ``lat``       ``site``           ``float64``, from the site table (``sites.site_coordinates``)
``ameriflux_site_id``  ``site``           the primary tower
``tower_lon``,         ``site``           the tower's own coordinates, from AmeriFlux
``tower_lat``
``match_basis``        ``site``           how the tower was matched (see ``towers``)
``utc_offset``         ``site``           hours; local standard time is ``time + utc_offset``
``doi``, ``site_version`` ``site``        the site's dataset DOI and its FULLSET version
``time``               ``time``           ``datetime64[ns]``, UTC, the **end** of each step
``time_bounds``        ``(time, bounds)`` the step's start and ``time``; the value is the mean
                                          rate over ``(start, time]``
====================== ================== ===================================================

``time`` runs over 2012-2024 in UTC at the series' step, labeled at the end of
each step as pySIPNET labels its output. A field from
:func:`net_ecosystem_exchange_fields` carries ``time_bounds`` as the window
coordinates ``window_start`` and ``window_end`` on ``time``, the right-closed
interval each value covers, which the observation operators read.

**Attributes** follow CF-1.11. ``value`` carries the spec's ``units``,
``long_name``, ``description``, ``upstream_product``, ``source_file``,
``source_column``, ``kind`` (``timestep_mean``, so aggregation averages),
``constituent``, ``sign_convention``, ``time_reference`` and
``units_provenance``; the companions carry their own. The dataset carries
``Conventions``, ``title``, ``net_ecosystem_exchange`` (the series' name),
``upstream_product``, ``source_file``,
``resolution``, ``towers_not_primary``, ``towers_excluded``,
``towers_without_the_series``, ``tower_table``, ``acknowledgement``,
``history`` and ``created``.

**Values are the source's, unchanged.** No quality filter, no preferred
estimate, no fallback from one estimate to another, no aggregation: which
steps count, at what resolution, is the experiment's choice.

Functions
---------
Each is documented where it is defined.

**The processed files.** :func:`load_net_ecosystem_exchange` reads one;
:func:`net_ecosystem_exchange_fields` returns the ``value`` of several, one
field per series, optionally for some sites, and
:func:`net_ecosystem_exchange_quality_flags`,
:func:`net_ecosystem_exchange_random_uncertainties` and
:func:`net_ecosystem_exchange_joint_uncertainties` do the same for the
companions. :func:`resolve_net_ecosystem_exchange` and :func:`describe` give
a spec.

**Building them.** :func:`read_source_file` parses one FULLSET file,
:func:`build_raw` and :func:`read_raw` make and read a raw file,
:func:`build_tower_table` and :func:`read_tower_table` the tower table, a reader
of :data:`SOURCE_READERS` makes a tower series, and
:func:`build_net_ecosystem_exchange` turns it into a processed Dataset.

Notes
-----
**Why the source is AmeriFlux's own files.** Every earlier derivative of them
in this project's inputs converted the stamps to UTC through a time zone with
daylight saving, one of them twice, leaving values up to 5 hours from their
labels; at 3-hourly resolution that cannot be undone, because the bins
themselves are out of phase. ``data/README.md`` has the account.

**Why one processed file per series.** Each is then a plain ``(site, time)``
array, ready to be an observation, and a new source is one reader and one spec
rather than a new file layout.

**Why the time labels are UTC.** The one exception to keeping the source's
time labels, and a relabeling rather than an alignment: each tower's stamps
move by its fixed offset, a whole number of steps, so every value keeps its
own step and ``utc_offset`` records the shift.

Usage
-----
::

    from sipnet_calibration.net_ecosystem_exchange import (
        net_ecosystem_exchange_fields, net_ecosystem_exchange_quality_flags,
    )
    from sipnet_calibration.observation.time_alignment import aggregate_time

    name = "ameriflux_nee_half_hourly_ustar_variable"
    nee = net_ecosystem_exchange_fields([name], sites=[4705])[name]      # (site, time)
    flag = net_ecosystem_exchange_quality_flags([name], sites=[4705])[name]
    measured = nee.where(flag == 0)
    three_hourly = aggregate_time(measured, "3h")                         # a mean, by kind
"""

from sipnet_calibration.net_ecosystem_exchange.names import (
    HALF_HOURLY,
    HOURLY,
    PROCESSED_END,
    PROCESSED_START,
    RAW_END,
    RAW_START,
    RESOLUTIONS,
    TIME_INDEX,
    TOWER,
    Resolution,
    ameriflux_site_list_path,
    default_raw_directory,
    default_source_root,
    pool_input_list_path,
    raw_path,
    resolve_resolution,
    tower_table_path,
)
from sipnet_calibration.net_ecosystem_exchange.processed import (
    JOINT_UNCERTAINTY,
    NIGHT,
    QUALITY_FLAG,
    RANDOM_UNCERTAINTY,
    TOWER_SITE_COORDINATES,
    VALUE,
    build_net_ecosystem_exchange,
    default_net_ecosystem_exchange_directory,
    load_net_ecosystem_exchange,
    net_ecosystem_exchange_fields,
    net_ecosystem_exchange_joint_uncertainties,
    net_ecosystem_exchange_path,
    net_ecosystem_exchange_quality_flags,
    net_ecosystem_exchange_random_uncertainties,
    netcdf_encoding,
)
from sipnet_calibration.net_ecosystem_exchange.raw import TOWER_COORDINATES, build_raw, raw_encoding, read_raw
from sipnet_calibration.net_ecosystem_exchange.source_files import (
    SOURCE,
    SourceColumn,
    SourceFile,
    SourceFormat,
    discover_source_files,
    parse_file_name,
    read_source_file,
)
from sipnet_calibration.net_ecosystem_exchange.sources import (
    SOURCE_READERS,
    TOWER_SERIES_VARIABLES,
    read_ameriflux_tower_series,
)
from sipnet_calibration.net_ecosystem_exchange.specs import (
    NET_ECOSYSTEM_EXCHANGE,
    NET_ECOSYSTEM_EXCHANGE_NAMES,
    SOURCES,
    NetEcosystemExchangeSpec,
    describe,
    resolve_net_ecosystem_exchange,
)
from sipnet_calibration.net_ecosystem_exchange.towers import (
    MATCH_BASES,
    MAXIMUM_SHORTWAVE_LAG,
    MINIMUM_OFFSET_SEPARATION,
    TOWER_COLUMN_DTYPES,
    TOWER_COLUMNS,
    UTC_OFFSET_RECOVERED,
    OffsetFit,
    build_tower_table,
    match_towers,
    read_ameriflux_site_list,
    read_pool_input_list,
    read_tower_table,
    recover_utc_offset,
    shortwave_lag_steps,
    summarize_raw,
)

__all__ = [
    # Names, axes and paths.
    "HALF_HOURLY",
    "HOURLY",
    "PROCESSED_END",
    "PROCESSED_START",
    "RAW_END",
    "RAW_START",
    "RESOLUTIONS",
    "Resolution",
    "TIME_INDEX",
    "TOWER",
    "ameriflux_site_list_path",
    "default_net_ecosystem_exchange_directory",
    "default_raw_directory",
    "default_source_root",
    "net_ecosystem_exchange_path",
    "pool_input_list_path",
    "raw_path",
    "resolve_resolution",
    "tower_table_path",
    # The source files.
    "SOURCE",
    "SourceColumn",
    "SourceFile",
    "SourceFormat",
    "discover_source_files",
    "parse_file_name",
    "read_source_file",
    # The raw files.
    "TOWER_COORDINATES",
    "build_raw",
    "raw_encoding",
    "read_raw",
    # The tower table.
    "MATCH_BASES",
    "MAXIMUM_SHORTWAVE_LAG",
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
    # The series.
    "NET_ECOSYSTEM_EXCHANGE",
    "NET_ECOSYSTEM_EXCHANGE_NAMES",
    "NetEcosystemExchangeSpec",
    "SOURCES",
    "describe",
    "resolve_net_ecosystem_exchange",
    "SOURCE_READERS",
    "TOWER_SERIES_VARIABLES",
    "read_ameriflux_tower_series",
    # The processed files.
    "JOINT_UNCERTAINTY",
    "NIGHT",
    "QUALITY_FLAG",
    "RANDOM_UNCERTAINTY",
    "TOWER_SITE_COORDINATES",
    "VALUE",
    "build_net_ecosystem_exchange",
    "load_net_ecosystem_exchange",
    "net_ecosystem_exchange_fields",
    "net_ecosystem_exchange_joint_uncertainties",
    "net_ecosystem_exchange_quality_flags",
    "net_ecosystem_exchange_random_uncertainties",
    "netcdf_encoding",
]
