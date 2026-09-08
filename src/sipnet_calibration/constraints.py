"""The annual biomass, leaf area and soil constraints, and their schema.

The source is a pair of R data files holding the observations assimilated by the
North American Land Carbon Reanalysis: ``obs.mean.Rdata`` and ``obs.cov.Rdata``,
each nesting as snapshot date, then site, then variable.
``scripts/export_constraints.R`` flattens them to a long CSV and
``scripts/ingest_constraints.py`` pivots that into
``data/processed/constraints_annual.nc``. See ``data/README.md`` for the source
format and the open questions.

This module holds the schema constants and the reader, so that the writer and
the reader of the processed file cannot drift apart. That is the same division
:mod:`sipnet_calibration.sites` uses, and the reason is the same: the two
settings that keep a product loadable are not something to re-derive per
notebook.

Two aspects of the layout are deliberate and worth stating.

**The observation error covariances are diagonal, so only variances are
carried.** Every one of the 103,047 covariance matrices in the source is exactly
diagonal -- checked over all 13 snapshots and all 8000 sites, not sampled -- so
the full matrices hold nothing the diagonal does not. ``export_constraints.R``
asserts diagonality at the source, where the off-diagonal is still visible, and
records the largest off-diagonal element it saw in its manifest so the Python
side can check that the assertion really ran. If a future release carries
genuine cross-variable covariance, this product gains a
``(site, time, variable, variable)`` array and :data:`OBS_VAR` becomes a view of
its diagonal.

**``variable`` is a dimension, not one array per variable.** The canonical field
convention in ``logs/2026-08-28_Plotting Design Spec.md`` asks for dims a subset
of ``(member, site, time)``, which this stored form is not. It is stored this way
because the observation operator indexes observations by exactly
``(site, variable, time)``, so flattening to the observation vector is a stack
rather than a join, and because the four variables genuinely share one
``(site, time)`` grid here. :func:`constraint_fields` produces the canonical
per-variable view for plotting, so both consumers are served without either
having to reshape the other's form.

Missingness is dense ``NaN``: the source is ragged over site, snapshot and
variable, and a dense array of 416,000 cells costs 3 MB per statistic, which is
cheaper than any ragged encoding is to reason about.
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
    "OBS_MEAN",
    "OBS_VAR",
    "SNAPSHOT_MONTH_DAY",
    "UNITS_PROVENANCE",
    "UNITS_STATUS",
    "constraint_fields",
    "default_constraints_path",
    "load_constraints",
    "read_long_table",
    "snapshot_dates",
]

#: Name of the mean array in the processed file.
OBS_MEAN = "obs_mean"

#: Name of the variance array in the processed file.
OBS_VAR = "obs_var"

#: The four constrained variables, in the alphabetical order the source uses.
#:
#: The order is load-bearing rather than cosmetic. ``obs.cov`` carries **no
#: dimension names**, so the only thing that says which row of a covariance
#: matrix belongs to which variable is the column order of the corresponding
#: ``obs.mean`` entry -- which is alphabetical. Keeping this tuple alphabetical
#: means the ``variable`` coordinate of the processed file is in the same order
#: the source used, so a mistake in the pairing shows up as a variable named
#: wrongly rather than as a silently transposed matrix.
CONSTRAINT_VARIABLES = ("AbvGrndWood", "LAI", "SoilMoistFrac", "TotSoilCarb")

#: Month and day of the source's annual snapshot key.
#:
#: The July 15 dates are the source product's annual bookkeeping convention, not
#: observation dates: see ``data/README.md``. Nothing should read them as the
#: instant an observation was taken.
SNAPSHOT_MONTH_DAY = (7, 15)

#: What is and is not settled about the units below.
UNITS_STATUS = "unconfirmed"

#: Why. Recorded next to every unit string in the written file, so that no
#: consumer can take these as checked.
UNITS_PROVENANCE = (
    "Documented in the NALCR dataset guide for the corresponding variables of "
    "the reanalysis *output*. These files are the observation *inputs* to that "
    "reanalysis. All four variable names and all thirteen snapshot keys agree "
    "between the two, so they very likely share definitions, but this has not "
    "been confirmed by the producer. See open question 9 in data/README.md."
)

#: Per-variable metadata written into the processed file.
#:
#: ``units`` is recorded because omitting it would be worse -- the canonical
#: field convention requires it, and a consumer with no unit at all has less to
#: go on than one with an unconfirmed unit and a status flag saying so. Both
#: :data:`UNITS_STATUS` and :data:`UNITS_PROVENANCE` travel with it.
CONSTRAINT_VARIABLE_ATTRS = {
    "AbvGrndWood": {
        "units": "Mg C ha-1",
        "long_name": "Above ground woody biomass",
    },
    "LAI": {
        "units": "m2 m-2",
        "long_name": "Leaf area index",
    },
    "SoilMoistFrac": {
        "units": "percent",
        "long_name": "Soil moisture fraction",
    },
    "TotSoilCarb": {
        "units": "kg C m-2",
        "long_name": "Total soil carbon",
    },
}

#: Columns of the long CSV that ``export_constraints.R`` writes.
LONG_COLUMNS = ("snapshot_date", "site_id", "variable", "mean", "variance")

#: Dtype per long-CSV column. ``mean`` and ``variance`` are read as float64 with
#: ``float_precision="round_trip"`` in :func:`read_long_table` rather than being
#: declared here, because the dtype alone does not make the parse exact.
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
    Experiments name their paths in ``config.py``; this exists so that tests,
    notebooks and the ingest script agree on one default.
    """
    root = os.environ.get(DATA_ROOT_ENV_VAR)
    data_root = Path(root) if root else Path(__file__).resolve().parents[2] / "data"
    return data_root / "processed" / "constraints_annual.nc"


def read_long_table(path: Path | str) -> pd.DataFrame:
    """Read the long CSV that ``export_constraints.R`` writes.

    Parameters
    ----------
    path:
        The CSV to read.

    Returns
    -------
    pandas.DataFrame
        The columns of :data:`LONG_COLUMNS`, one row per observed
        ``(snapshot, site, variable)`` triple.

    Raises
    ------
    ValueError
        If the columns are not exactly :data:`LONG_COLUMNS`, the file holds no
        rows, or any variable name is not in :data:`CONSTRAINT_VARIABLES`.

    Notes
    -----
    ``float_precision="round_trip"`` is load-bearing. The R side writes the
    doubles with ``%.17g``, which uniquely determines a float64, but pandas'
    default C parser is not correctly rounding and moves some of those values in
    the last bits -- the same failure that cost the site table its coordinates
    before it was caught. The round-trip parser is exact.

    ``keep_default_na=False`` keeps a variable named ``NA`` from becoming a null,
    for the same reason the site table needs it. No such variable exists today;
    the setting costs nothing and removes the possibility.
    """
    frame = pd.read_csv(
        path,
        dtype={"snapshot_date": str, "site_id": np.int32, "variable": str},
        float_precision="round_trip",
        keep_default_na=False,
        na_values=[],
    )

    if tuple(frame.columns) != LONG_COLUMNS:
        raise ValueError(
            f"{path}: expected columns {LONG_COLUMNS}, found {tuple(frame.columns)}. "
            "Regenerate it with scripts/export_constraints.R."
        )
    if frame.empty:
        raise ValueError(f"{path}: holds no rows")

    unknown = sorted(set(frame["variable"]) - set(CONSTRAINT_VARIABLES))
    if unknown:
        raise ValueError(
            f"{path}: variable names not in CONSTRAINT_VARIABLES: {unknown}. "
            "A new variable in the source is a schema change, not a new row."
        )

    for column in ("mean", "variance"):
        frame[column] = frame[column].astype(np.float64)
    return frame


def load_constraints(path: Path | str | None = None) -> xr.Dataset:
    """Read the processed annual constraints.

    Parameters
    ----------
    path:
        The netCDF file to read. Defaults to :func:`default_constraints_path`.

    Returns
    -------
    xarray.Dataset
        :data:`OBS_MEAN` and :data:`OBS_VAR`, both with dims
        ``(site, time, variable)``, ``lon`` and ``lat`` as non-dimension
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
    ``decode_times`` is left on: unlike the initial-condition files, whose units
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
    _validate_constraints(dataset, path)
    return dataset


def constraint_fields(
    dataset: xr.Dataset, *, statistic: str = "mean"
) -> dict[str, xr.DataArray]:
    """Split the stored form into canonical per-variable fields.

    The stored form carries ``variable`` as a dimension, which the canonical
    field convention does not allow. This is the view that does: one
    ``DataArray`` per variable with dims ``(site, time)``, named for the
    variable, carrying its units and long name, and keeping ``lon``/``lat`` as
    non-dimension coordinates on ``site``.

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
        Keyed by variable name, in :data:`CONSTRAINT_VARIABLES` order. This is
        the shape multi-variable adapters return and what facet-by-variable
        consumes.

    Raises
    ------
    ValueError
        If *statistic* is neither ``"mean"`` nor ``"variance"``.
    """
    if statistic not in ("mean", "variance"):
        raise ValueError(
            f"statistic must be 'mean' or 'variance', got {statistic!r}"
        )

    array = dataset[OBS_MEAN if statistic == "mean" else OBS_VAR]
    fields = {}
    for name in CONSTRAINT_VARIABLES:
        field = array.sel(variable=name, drop=True).rename(name)
        field.attrs = _field_attrs(name, statistic)
        fields[name] = field
    return fields


def snapshot_dates(years: list[int] | tuple[int, ...]) -> pd.DatetimeIndex:
    """The source's annual snapshot keys for the given years.

    Kept here rather than written out at each call site so that the month and
    day come from :data:`SNAPSHOT_MONTH_DAY` alone.
    """
    month, day = SNAPSHOT_MONTH_DAY
    return pd.DatetimeIndex([pd.Timestamp(year, month, day) for year in years])


# ── private helpers ───────────────────────────────────────────────────────────


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
        "units_status": UNITS_STATUS,
        "units_provenance": UNITS_PROVENANCE,
    }


def _validate_constraints(dataset: xr.Dataset, path: Path) -> None:
    """Raise unless *dataset* matches the schema this module defines."""
    missing = {OBS_MEAN, OBS_VAR} - set(dataset.data_vars)
    if missing:
        raise ValueError(
            f"{path}: missing data variables {sorted(missing)}; found "
            f"{sorted(dataset.data_vars)}"
        )

    expected_dims = ("site", "time", "variable")
    for name in (OBS_MEAN, OBS_VAR):
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
            f"{CONSTRAINT_VARIABLES}. The order is what pairs a variance with "
            "its variable; see the module docstring."
        )

    site = dataset["site"].values
    if site.size == 0:
        raise ValueError(f"{path}: holds no sites")
    if np.any(np.diff(site) <= 0):
        raise ValueError(f"{path}: site is not strictly ascending")
