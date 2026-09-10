"""The data model for the annual biomass, leaf area and soil constraints.

Overview
--------
This module defines how the annual constraint data is represented -- its
dimensions, coordinates, variable names, units and dtypes -- and provides the
functions that read it and reshape it. It is the single description of that
layout: everything else in the project, the ingest script included, gets the
schema from here rather than restating it.

It sits at the end of the pipeline that builds the data, and the dependency runs
one way::

    raw/constraints/sda_8k_site_rdata/obs.{mean,cov}.Rdata
      -> scripts/export_constraints.R     long table + manifest
      -> scripts/ingest_constraints.py    processed/constraints_annual.nc
      -> this module                      load_constraints() -> xarray.Dataset

``ingest_constraints.py`` imports the schema constants from here, and its
round-trip check reads its own output back through :func:`load_constraints`, so
the writer is verified against the same description every consumer uses.
``data/README.md`` documents the source data and the open questions about it.

Input data
----------
``data/processed/constraints_annual.nc``
    The product, read by :func:`load_constraints`, whose layout is the
    `Data model`_ below. :func:`default_constraints_path` says where it is
    expected to be.

The intermediate long table
    A CSV of one row per observed ``(snapshot, site, variable)`` triple, written
    by ``export_constraints.R`` and read by :func:`read_long_table`, which
    parses it exactly. Its ``variable`` column holds *source* names.

Data model
----------
:func:`load_constraints` returns an ``xarray.Dataset`` shaped as follows. The
data variables, dims, coordinate set and ``variable`` order are checked on
load; the dtypes below are what the writer produces, not something the reader
enforces.

**Dimensions**: ``site``, ``time``, ``variable``.

**Data variables**, both ``float64``, ``NaN`` where a site-snapshot-variable
was not observed::

    observation_mean(site, time, variable)      the observation
    observation_variance(site, time, variable)  its error variance

**Coordinates**

================== ============ ===============================================
Name               Dims         Meaning
================== ============ ===============================================
``site``           ``site``     handed-down integer site id, strictly ascending
``time``           ``time``     annual snapshot key, ``datetime64``
``variable``       ``variable`` processed variable name, sorted
``lon``, ``lat``   ``site``     non-dimension coordinates, from the site table
================== ============ ===============================================

The ``site`` axis is the whole site pool, not only the observed sites.

**Variable names.** The source names are prescribed by the input data; the
processed ones follow the project convention of lower case with underscores and
no abbreviation beyond the universal. The rename happens in the ingest script,
via :data:`SOURCE_VARIABLE_NAMES`.

======================= ===========================
Source                  Processed
======================= ===========================
``AbvGrndWood``         ``aboveground_wood_carbon``
``LAI``                 ``lai``
``SoilMoistFrac``       ``soil_moisture_percent``
``TotSoilCarb``         ``total_soil_carbon``
======================= ===========================

**Attributes.** Each variable's unit is a dataset attribute,
``variable_<name>_units``, alongside ``_long_name`` and ``_source_name`` --
netCDF has nowhere to hang attributes off a coordinate value.
:func:`constraint_fields` puts the right unit on each field.

The two data variables span four variables with different units, so their own
``units`` attribute is a pointer to those dataset attributes rather than a unit
string. They also carry ``units_status`` and ``units_provenance``: the units are
documented for the reanalysis output rather than for these observation inputs,
so they are recorded but flagged.

The dataset also carries ``title``, ``source_mean_file``, ``source_cov_file``,
``source_resolution``, ``history``, ``exported_at``,
``covariances_all_diagonal`` and ``n_observed_triples``. ``time`` carries
``time_zone`` and ``time_label``, the latter being ``"nominal"``: the snapshot
keys are the source product's annual bookkeeping convention, not observation
dates.

Functions
---------
:func:`load_constraints`
    Read the product and check it against the data model above. Raises rather
    than returning something subtly wrong.

:func:`constraint_fields`
    Split the stored form into canonical fields -- one ``DataArray`` per
    variable with dims ``(site, time)``, carrying its own units -- for either
    the means or the variances. This is the view the plotting layer wants.

:func:`read_long_table`
    Read the intermediate long table exactly, for the ingest script.

:func:`snapshot_dates`
    Build the source's annual snapshot keys for given years.

:func:`default_constraints_path`
    Where the product is expected to be, honoring
    ``$SIPNET_CALIBRATION_DATA``.

Notes
-----
**Only variances are carried, not covariance matrices.** Every source
covariance is exactly diagonal, so the matrices hold nothing the diagonal does
not. That is asserted in R at every export, where the off-diagonal is still
visible, and the result is recorded in a manifest so this side can confirm the
check ran. If a future release carries genuine cross-variable
covariance, this product would need a
``(site, time, variable, variable)`` array instead.

**``variable`` is a dimension, not one array per variable.** The canonical field
convention wants dims a subset of ``(member, site, time)``, which this stored
form is not. It is stored this way because the observation operator indexes
observations by exactly ``(site, variable, time)``, so flattening to the
observation vector is a stack rather than a join, and because the variables
share one ``(site, time)`` grid here. :func:`constraint_fields` is what serves
the consumers that want the canonical form instead.

**Missingness.** Unobserved cells are ``NaN`` in a dense array; a ragged
encoding buys nothing at this size. A ``NaN`` means not observed; a zero is an
observation.

Usage
-----
Load the product, then select from it with ordinary xarray::

    from sipnet_calibration.constraints import (
        constraint_fields,
        load_constraints,
        snapshot_dates,
    )

    constraints = load_constraints()        # or load_constraints(path)

    # One variable, over every site and snapshot: dims (site, time).
    lai = constraints["observation_mean"].sel(variable="lai")

    # One site's whole record: dims (time, variable).
    site_1 = constraints.sel(site=1)

    # One snapshot. snapshot_dates builds the keys from years, so the
    # July-15 convention is not written out at the call site.
    (key,) = snapshot_dates([2015])
    in_2015 = constraints.sel(time=key)

    # A subset of sites, returned in the order given.
    subset = constraints.sel(site=[4102, 4113, 5584])

    # An observation beside its error variance.
    mean = constraints["observation_mean"].sel(variable="total_soil_carbon")
    variance = constraints["observation_variance"].sel(variable="total_soil_carbon")

For plotting, take the canonical per-variable view. It drops the ``variable``
dimension and gives each field its own units, so a plotter needs to know nothing
about this product's layout::

    fields = constraint_fields(constraints)                       # observations
    variances = constraint_fields(constraints, statistic="variance")

    fields["lai"].dims                      # ('site', 'time')
    fields["lai"].attrs["units"]            # 'm2 m-2'
    variances["total_soil_carbon"].attrs["units"]     # '(kg C m-2)2'

For the likelihood, flatten to an observation vector, keeping only what was
observed. The ``(site, variable, time)`` index that falls out is the labeling
the observation operator uses, and unstacking it is the inverse::

    observed = (
        constraints["observation_mean"]
        .stack(observation=("site", "variable", "time"))
        .dropna("observation")
    )
    observed.indexes["observation"].names   # ['site', 'variable', 'time']
    observed.unstack("observation").dims    # ('site', 'variable', 'time')

Selecting the matching error variances is the same expression against
``observation_variance``, and the two indexes align because both arrays are
``NaN`` in exactly the same places.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.sites import DATA_ROOT_ENV_VAR

__all__ = [
    "CONSTRAINT_VARIABLES",
    "CONSTRAINT_VARIABLE_ATTRS",
    "LONG_COLUMNS",
    "LONG_COLUMN_DTYPES",
    "OBSERVATION_MEAN",
    "OBSERVATION_VARIANCE",
    "SNAPSHOT_MONTH_DAY",
    "SOURCE_VARIABLE_NAMES",
    "UNITS_PROVENANCE",
    "UNITS_STATUS",
    "constraint_fields",
    "default_constraints_path",
    "load_constraints",
    "read_long_table",
    "snapshot_dates",
]

#: Name of the mean array in the processed file.
OBSERVATION_MEAN = "observation_mean"

#: Name of the variance array in the processed file.
OBSERVATION_VARIANCE = "observation_variance"

#: Source variable name -> processed variable name.
#:
#: Applied by ``scripts/ingest_constraints.py``.
SOURCE_VARIABLE_NAMES = {
    "AbvGrndWood": "aboveground_wood_carbon",
    "LAI": "lai",
    "SoilMoistFrac": "soil_moisture_percent",
    "TotSoilCarb": "total_soil_carbon",
}

#: The constrained variables, by processed name, in the order the ``variable``
#: coordinate carries them.
CONSTRAINT_VARIABLES = tuple(sorted(SOURCE_VARIABLE_NAMES.values()))

#: Month and day of the source's annual snapshot key.
#:
#: The source product's annual bookkeeping convention, not observation dates:
#: see ``data/README.md``. Nothing should read them as the instant an
#: observation was taken.
SNAPSHOT_MONTH_DAY = (7, 15)

#: What is and is not settled about the units below.
UNITS_STATUS = "unconfirmed"

#: Why. Recorded next to every unit string in the written file, so that no
#: consumer can take these as checked.
UNITS_PROVENANCE = (
    "Documented in the NALCR dataset guide for the corresponding variables of "
    "the reanalysis *output*. These files are the observation *inputs* to that "
    "reanalysis. The variable names and the snapshot keys agree between the "
    "two, so they very likely share definitions, but this has not "
    "been confirmed by the producer. See open question 9 in data/README.md."
)

#: Per-variable metadata written into the processed file, by processed name.
#:
#: The units are unconfirmed; :data:`UNITS_STATUS` and
#: :data:`UNITS_PROVENANCE` travel with every one of them.
CONSTRAINT_VARIABLE_ATTRS = {
    "aboveground_wood_carbon": {
        "units": "Mg C ha-1",
        "long_name": "Aboveground woody biomass carbon",
        "source_name": "AbvGrndWood",
    },
    "lai": {
        "units": "m2 m-2",
        "long_name": "Leaf area index",
        "source_name": "LAI",
    },
    "soil_moisture_percent": {
        "units": "percent",
        "long_name": "Soil moisture percent",
        "source_name": "SoilMoistFrac",
    },
    "total_soil_carbon": {
        "units": "kg C m-2",
        "long_name": "Total soil carbon",
        "source_name": "TotSoilCarb",
    },
}

#: Columns of the long table that ``export_constraints.R`` writes. Its
#: ``variable`` column holds *source* names.
LONG_COLUMNS = ("snapshot_date", "site_id", "variable", "mean", "variance")

#: Dtype per long-table column, as :func:`read_long_table` returns them, and
#: what it hands pandas at the read. Declaring ``mean`` and ``variance`` is not
#: on its own enough to parse them exactly; see :func:`read_long_table`.
LONG_COLUMN_DTYPES = {
    "snapshot_date": str,
    "site_id": np.int32,
    "variable": str,
    "mean": np.float64,
    "variance": np.float64,
}


def default_constraints_path() -> Path:
    """Where the processed constraint file is expected to be.

    ``$SIPNET_CALIBRATION_DATA/processed/constraints_annual.nc`` when that
    variable is set, and otherwise the ``data/`` directory of this checkout.
    Experiments name their paths in ``config.py``.
    """
    root = os.environ.get(DATA_ROOT_ENV_VAR)
    data_root = Path(root) if root else Path(__file__).resolve().parents[2] / "data"
    return data_root / "processed" / "constraints_annual.nc"


def read_long_table(path: Path | str) -> pd.DataFrame:
    """Read the long table that ``export_constraints.R`` writes.

    Parameters
    ----------
    path:
        The CSV to read.

    Returns
    -------
    pandas.DataFrame
        The columns of :data:`LONG_COLUMNS` with the dtypes of
        :data:`LONG_COLUMN_DTYPES`, one row per observed
        ``(snapshot, site, variable)`` triple. ``variable`` holds *source*
        names; the ingest script renames them.

    Raises
    ------
    ValueError
        If the columns are not exactly :data:`LONG_COLUMNS`, the file holds no
        rows, or any variable name is not a key of
        :data:`SOURCE_VARIABLE_NAMES`.

    Notes
    -----
    ``float_precision="round_trip"`` is required for an exact parse. The R side
    writes the doubles with ``%.17g``, which uniquely determines a float64, but
    pandas' default C parser is not correctly rounding and moves tens of
    thousands of the real table's values in the last bits.

    ``keep_default_na=False`` keeps a variable named ``NA`` from becoming a
    null, for the same reason the site table needs it. No such variable exists
    today; the setting costs nothing and removes the possibility.
    """
    frame = pd.read_csv(
        path,
        # site_id is read wide and narrowed after checking, as load_sites does:
        # reading straight into int32 wraps silently, and a site_id of
        # 4294967297 becomes 1 and then satisfies every check downstream.
        # mean/variance are declared rather than inferred, because pandas infers
        # int64 for an all-integer column and float_precision then does not
        # apply.
        dtype={**LONG_COLUMN_DTYPES, "site_id": np.int64},
        float_precision="round_trip",
        keep_default_na=False,
        na_values=[],
        # Without this, a row with surplus leading fields is absorbed into an
        # index and the column check still passes.
        index_col=False,
    )

    if tuple(frame.columns) != LONG_COLUMNS:
        raise ValueError(
            f"{path}: expected columns {LONG_COLUMNS}, found {tuple(frame.columns)}. "
            "Regenerate it with scripts/export_constraints.R."
        )
    if frame.empty:
        raise ValueError(f"{path}: holds no rows")

    unknown = sorted(set(frame["variable"]) - set(SOURCE_VARIABLE_NAMES))
    if unknown:
        raise ValueError(
            f"{path}: source variable names not in SOURCE_VARIABLE_NAMES: "
            f"{unknown}. A new variable in the source is a schema change, not a "
            "new row."
        )

    _check_site_ids_fit_dtype(frame, path)
    return frame.astype({"site_id": LONG_COLUMN_DTYPES["site_id"]})


def load_constraints(path: Path | str | None = None) -> xr.Dataset:
    """Read the processed annual constraints.

    Parameters
    ----------
    path:
        The netCDF file to read. Defaults to :func:`default_constraints_path`.

    Returns
    -------
    xarray.Dataset
        :data:`OBSERVATION_MEAN` and :data:`OBSERVATION_VARIANCE`, both with
        dims ``(site, time, variable)``, ``lon`` and ``lat`` as non-dimension
        coordinates on ``site``, and ``NaN`` where a site-snapshot-variable was
        not observed.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with the commands that produce it.
    ValueError
        If the variables, dims, coordinates or ``variable`` order are not the
        schema this module defines.

    Notes
    -----
    ``decode_times`` is left on: unlike the initial condition files, whose units
    attribute is an unsubstituted template, this file's time encoding is written
    by this project and is decodable.
    """
    path = Path(path) if path is not None else default_constraints_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Produce it with:\n"
            "  Rscript scripts/export_constraints.R --out <long.csv> "
            "--manifest <manifest.json>\n"
            "  python scripts/ingest_constraints.py --long-table <long.csv> "
            "--manifest <manifest.json>"
        )

    dataset = xr.open_dataset(path, engine="h5netcdf")
    try:
        _check_schema(dataset, path)
    except Exception:
        dataset.close()
        raise
    return dataset


def constraint_fields(
    dataset: xr.Dataset, *, statistic: str = "mean"
) -> dict[str, xr.DataArray]:
    """One ``DataArray`` per variable, for either the means or the variances.

    Each field has dims ``(site, time)``, is named for its variable, carries
    that variable's units and long name, and keeps ``lon``/``lat`` as
    non-dimension coordinates on ``site`` -- the canonical field shape, which
    the stored form is not.

    Parameters
    ----------
    dataset:
        As returned by :func:`load_constraints`.
    statistic:
        ``"mean"`` for the observations, ``"variance"`` for their error
        variances.

    Returns
    -------
    dict
        Keyed by processed variable name, in :data:`CONSTRAINT_VARIABLES` order.
        This is the shape multi-variable adapters return and what
        facet-by-variable consumes.

    Raises
    ------
    ValueError
        If *statistic* is neither ``"mean"`` nor ``"variance"``.
    """
    if statistic not in ("mean", "variance"):
        raise ValueError(f"statistic must be 'mean' or 'variance', got {statistic!r}")

    array = dataset[OBSERVATION_MEAN if statistic == "mean" else OBSERVATION_VARIANCE]
    fields = {}
    for name in CONSTRAINT_VARIABLES:
        field = array.sel(variable=name, drop=True).rename(name)
        field.attrs = _field_attrs(name, statistic)
        fields[name] = field
    return fields


def snapshot_dates(years: list[int] | tuple[int, ...]) -> pd.DatetimeIndex:
    """The source's annual snapshot keys for the given years, in that order.

    The month and day come from :data:`SNAPSHOT_MONTH_DAY`.
    """
    month, day = SNAPSHOT_MONTH_DAY
    return pd.DatetimeIndex([pd.Timestamp(year, month, day) for year in years])


# ── supporting helpers ────────────────────────────────────────────────────────


def _field_attrs(name: str, statistic: str) -> dict[str, str]:
    """Attributes for one canonical field, with the units caveat attached."""
    source = CONSTRAINT_VARIABLE_ATTRS[name]
    if statistic == "mean":
        units, long_name = source["units"], source["long_name"]
    else:
        units = f"({source['units']})2"
        long_name = f"{source['long_name']}: observation error variance"
    return {
        "units": units,
        "long_name": long_name,
        "source_name": source["source_name"],
        "units_status": UNITS_STATUS,
        "units_provenance": UNITS_PROVENANCE,
    }


# ── checks ────────────────────────────────────────────────────────────────────


def _check_site_ids_fit_dtype(frame: pd.DataFrame, path: Path) -> None:
    """Raise unless every site id survives narrowing to the stored width."""
    stored = np.dtype(LONG_COLUMN_DTYPES["site_id"])
    info = np.iinfo(stored)
    site_id = frame["site_id"].to_numpy()
    outside = (site_id < info.min) | (site_id > info.max)
    if outside.any():
        offenders = sorted(set(site_id[outside].tolist()))[:10]
        raise ValueError(
            f"{path}: site ids outside the range of {stored}: {offenders}. "
            "Narrowing them would wrap to a different, valid-looking site."
        )
    if (site_id < 1).any():
        offenders = sorted(set(site_id[site_id < 1].tolist()))[:10]
        raise ValueError(f"{path}: site ids below 1: {offenders}")


def _check_schema(dataset: xr.Dataset, path: Path) -> None:
    """Raise unless *dataset* matches the schema this module defines."""
    missing = {OBSERVATION_MEAN, OBSERVATION_VARIANCE} - set(dataset.data_vars)
    if missing:
        raise ValueError(
            f"{path}: missing data variables {sorted(missing)}; found "
            f"{sorted(dataset.data_vars)}"
        )

    expected_dims = ("site", "time", "variable")
    for name in (OBSERVATION_MEAN, OBSERVATION_VARIANCE):
        if dataset[name].dims != expected_dims:
            raise ValueError(
                f"{path}: {name} has dims {dataset[name].dims}, expected "
                f"{expected_dims}"
            )

    for coordinate in ("site", "time", "variable", "lon", "lat"):
        if coordinate not in dataset.coords:
            raise ValueError(f"{path}: missing the {coordinate!r} coordinate")

    for coordinate in ("lon", "lat"):
        if dataset[coordinate].dims != ("site",):
            raise ValueError(
                f"{path}: {coordinate} must be a non-dimension coordinate on "
                f"site, has dims {dataset[coordinate].dims}"
            )

    stored = tuple(str(name) for name in dataset["variable"].values)
    if stored != CONSTRAINT_VARIABLES:
        raise ValueError(
            f"{path}: variable coordinate is {stored}, expected "
            f"{CONSTRAINT_VARIABLES}"
        )

    site = dataset["site"].values
    if site.size == 0:
        raise ValueError(f"{path}: holds no sites")
    if np.any(np.diff(site) <= 0):
        raise ValueError(f"{path}: site is not strictly ascending")

    time = dataset["time"].values
    if time.size == 0:
        raise ValueError(f"{path}: holds no snapshots")
    if np.any(np.diff(time) <= np.timedelta64(0, "ns")):
        raise ValueError(
            f"{path}: time is not strictly ascending. A repeated snapshot key "
            "would make sel(time=...) return more than one snapshot."
        )
