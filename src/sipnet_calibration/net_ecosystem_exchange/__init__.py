"""Observed net ecosystem exchange: the eddy-covariance towers, their clocks,
their pool sites, and the products built from them.

Overview
--------
Net ecosystem exchange is observed at eddy-covariance towers, and each
AmeriFlux tower is, or sits in the cell of, one site of the 8000-site pool.
This package owns that data from AmeriFlux's FLUXNET files to the products the
calibration reads. Three modules own a stored artifact -- the raw files, the
tower table, the products -- and hold that artifact's schema, writer, reader
and checks; the others hold what those share.

========================  ==================================================
``names``                 dimension names, the two time axes, file locations
``source_files``          AmeriFlux's FULLSET CSVs: the format, and the parser
``raw``                   the converted raw netCDFs: build, encode, read
``towers``                the tower table: sites, clocks, exclusions
``specs``                 what each product is; the registry
``sources``               source readers, raw file to tower series
``processed``             the products: build, encode, read, value arrays
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
      -> scripts/ingest_net_ecosystem_exchange.py    one run per product
      -> processed/net_ecosystem_exchange/<name>.nc
      -> this package                                load_net_ecosystem_exchange(),
                                                     net_ecosystem_exchange_values()

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
:func:`load_net_ecosystem_exchange` returns one product as an
``xarray.Dataset``, checked on load against its spec. A product is one
**series**: a single NEE estimate from a single source at a single resolution,
one of :data:`NET_ECOSYSTEM_EXCHANGE`.

**Dimensions**: ``site``, ``time``, ``bounds`` (2), and ``member`` only for a
product from an ensemble source.

**Data variables**. ``value`` is always present; the others are present when
the product's source has them, as every AmeriFlux product does.

========================== ============================ ========= ===========================================
Name                       Dims                         Dtype     Meaning; missing
========================== ============================ ========= ===========================================
``value``                  ``([member,] site, time)``   float64   NEE, mean rate over the step; ``NaN``
``quality_flag``           ``(site, time)``             int8      0 measured, 1-3 gap-fill quality; ``-1``
``random_uncertainty``     ``(site, time)``             float64   ONEFlux random uncertainty; ``NaN``
``joint_uncertainty``      ``(site, time)``             float64   random and u*-filtering uncertainty; ``NaN``
``night``                  ``(site, time)``             int8      1 at night, 0 by day; ``-1``
========================== ============================ ========= ===========================================

Units are the source's, ``umol m-2 s-1`` with ``constituent = "CO2"``; the
observation operator converts SIPNET's ``g m-2`` of C per step into them.

**Coordinates**

====================== ================== ===================================================
Name                   Dims               Meaning
====================== ================== ===================================================
``site``               ``site``           ``int32``, the pool sites with a primary tower, ascending
``lon``, ``lat``       ``site``           ``float64``, from the site table
``ameriflux_site_id``  ``site``           the primary tower
``tower_lon``,         ``site``           the tower's own coordinates, from AmeriFlux
``tower_lat``
``match_basis``        ``site``           how the tower was matched (see ``towers``)
``utc_offset``         ``site``           hours; local standard time is ``time + utc_offset``
``doi``, ``site_version`` ``site``        the site's dataset DOI and its FULLSET version
``time``               ``time``           ``datetime64[ns]``, UTC, the **end** of each step
``time_step_start``    ``time``           the step's start
``time_step_length``   ``time``           ``timedelta64[ns]``, 30 or 60 minutes
``time_bounds``        ``(time, bounds)`` ``[time_step_start, time]``
``member``             ``member``         ensemble sources only
====================== ================== ===================================================

``time`` runs over 2012-2024 in UTC at the product's step. The time
coordinates are pySIPNET's names and convention for its output, so
:func:`sipnet_calibration.obs_ops.aggregate_time` bins a product and a model
field the same way.

**Attributes** follow CF-1.11. ``value`` carries the spec's ``units``,
``long_name``, ``description``, ``product``, ``source_file``,
``source_column``, ``kind`` (``timestep_mean``, so aggregation averages),
``constituent``, ``sign_convention``, ``time_reference`` and
``units_provenance``; the companions carry their own. The dataset carries
``Conventions``, ``title``, ``product``, ``product_name``, ``source_file``,
``resolution``, ``towers_not_primary``, ``towers_excluded``,
``towers_without_the_series``, ``tower_table``, ``acknowledgement``,
``history`` and ``created``.

**Values are the source's, unchanged.** No quality filter, no preferred
estimate, no fallback from one estimate to another, no aggregation: which
steps count, at what resolution, is the experiment's choice.

Functions
---------
Each is documented where it is defined.

**The products.** :func:`load_net_ecosystem_exchange` reads one;
:func:`net_ecosystem_exchange_values` returns the ``value`` of several, one
array per product, optionally for some sites, and
:func:`net_ecosystem_exchange_quality_flags`,
:func:`net_ecosystem_exchange_random_uncertainties` and
:func:`net_ecosystem_exchange_joint_uncertainties` do the same for the
companions. :func:`resolve_net_ecosystem_exchange` and :func:`describe` give
a spec.

**Building them.** :func:`read_source_file` parses one FULLSET file,
:func:`build_raw` and :func:`read_raw` make and read a raw file,
:func:`build_tower_table` and :func:`read_tower_table` the tower table, a reader
of :data:`SOURCE_READERS` makes a tower series, and
:func:`build_net_ecosystem_exchange` turns it into a product.

Notes
-----
**Why the source is AmeriFlux's own files.** Every earlier derivative of them
in this project's inputs converted the stamps to UTC through a time zone with
daylight saving, one of them twice, leaving values up to 5 hours from their
labels; at 3-hourly resolution that cannot be undone, because the bins
themselves are out of phase. ``data/README.md`` has the account.

**Why one product per series.** Each is then a plain ``(site, time)`` array,
ready to be an observation, and a new source is one reader and one spec rather than a
new product layout.

Usage
-----
::

    from sipnet_calibration.net_ecosystem_exchange import (
        load_net_ecosystem_exchange, net_ecosystem_exchange_values,
        net_ecosystem_exchange_quality_flags,
    )
    from sipnet_calibration.obs_ops import aggregate_time

    name = "ameriflux_nee_half_hourly_ustar_variable"
    nee = net_ecosystem_exchange_values(name, sites=[4705])[name]      # (site, time)
    flag = net_ecosystem_exchange_quality_flags(name, sites=[4705])[name]
    measured = nee.where(flag == 0)
    three_hourly = aggregate_time(measured, "3h")                       # a mean, by kind
"""

from sipnet_calibration.net_ecosystem_exchange.names import (
    BOUNDS,
    HALF_HOURLY,
    HOURLY,
    MEMBER,
    RESOLUTIONS,
    SITE,
    TIME,
    TIME_BOUNDS,
    TIME_INDEX,
    TIME_STEP_LENGTH,
    TIME_STEP_START,
    TOWER,
    Resolution,
    ameriflux_site_list_path,
    default_product_dir,
    default_raw_dir,
    default_source_root,
    pool_input_list_path,
    product_path,
    raw_path,
    resolve_resolution,
    tower_table_path,
)
from sipnet_calibration.net_ecosystem_exchange.processed import (
    SITE_COORDINATES,
    build_net_ecosystem_exchange,
    load_net_ecosystem_exchange,
    net_ecosystem_exchange_joint_uncertainties,
    net_ecosystem_exchange_quality_flags,
    net_ecosystem_exchange_random_uncertainties,
    net_ecosystem_exchange_values,
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
    MAXIMUM_SHORTWAVE_LAG_MINUTES,
    MINIMUM_OFFSET_SEPARATION,
    TOWER_COLUMN_DTYPES,
    TOWER_COLUMNS,
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
    "BOUNDS",
    "HALF_HOURLY",
    "HOURLY",
    "MATCH_BASES",
    "MAXIMUM_SHORTWAVE_LAG_MINUTES",
    "MEMBER",
    "MINIMUM_OFFSET_SEPARATION",
    "NET_ECOSYSTEM_EXCHANGE",
    "NET_ECOSYSTEM_EXCHANGE_NAMES",
    "NetEcosystemExchangeSpec",
    "OffsetFit",
    "RESOLUTIONS",
    "Resolution",
    "SITE",
    "SITE_COORDINATES",
    "SOURCE",
    "SOURCES",
    "SOURCE_READERS",
    "SourceColumn",
    "SourceFile",
    "SourceFormat",
    "TIME",
    "TIME_BOUNDS",
    "TIME_INDEX",
    "TIME_STEP_LENGTH",
    "TIME_STEP_START",
    "TOWER",
    "TOWER_COLUMNS",
    "TOWER_COLUMN_DTYPES",
    "TOWER_COORDINATES",
    "TOWER_SERIES_VARIABLES",
    "ameriflux_site_list_path",
    "build_net_ecosystem_exchange",
    "build_raw",
    "build_tower_table",
    "default_product_dir",
    "default_raw_dir",
    "default_source_root",
    "describe",
    "discover_source_files",
    "load_net_ecosystem_exchange",
    "match_towers",
    "net_ecosystem_exchange_joint_uncertainties",
    "net_ecosystem_exchange_quality_flags",
    "net_ecosystem_exchange_random_uncertainties",
    "net_ecosystem_exchange_values",
    "netcdf_encoding",
    "parse_file_name",
    "pool_input_list_path",
    "product_path",
    "raw_encoding",
    "raw_path",
    "read_ameriflux_site_list",
    "read_ameriflux_tower_series",
    "read_pool_input_list",
    "read_raw",
    "read_source_file",
    "read_tower_table",
    "recover_utc_offset",
    "resolve_net_ecosystem_exchange",
    "resolve_resolution",
    "shortwave_lag_steps",
    "summarize_raw",
    "tower_table_path",
]
