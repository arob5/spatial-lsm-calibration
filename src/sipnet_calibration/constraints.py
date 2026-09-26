"""The constraint observations: what each is, how its raw file is read, and the
processed file it becomes.

Overview
--------
Five observation sources constrain the calibration: LandTrendr and GEDI
aboveground biomass, MODIS leaf area index, SMAP soil moisture and SoilGrids
soil organic carbon. Each arrives as one gzipped CSV under
``data/raw/constraints/`` whose columns carry no units, no time semantics and
no provenance. This module holds one :class:`ConstraintSpec` per constraint -- the
single description of what the quantity is, in what units, on what time
structure, and which columns of the raw file carry it -- and the functions that
turn the raw file into a processed netCDF and read it back. The spec's fields
are written into the netCDF as attributes, so the processed file needs no
description beyond itself.

The dependency runs one way::

    raw/constraints/<name>.csv.gz
      -> scripts/ingest_constraints.py    read_raw(), the checks, build_constraint()
      -> processed/constraints/<name>.nc  one file per constraint
      -> this module                      load_constraint() -> xarray.Dataset

``data/README.md`` documents the source files and the open questions about
them; ``data/raw/constraints/provenance.md`` records where each was copied
from and how to detect drift.

Input data
----------
``data/raw/constraints/<spec.raw_file>``
    A gzipped CSV with a header row, one row per observation record, addressed
    by ``site_id`` (the 1-8000 site identifier) and, unless static, by the
    spec's ``time_column``. Missing values are the literal ``NA``; the three
    files serialized from R carry 17 significant digits. :func:`read_raw` is
    the only reader and parses them exactly.

``data/processed/sites/sites.csv``
    The site table, for the site pool and the ``lon``/``lat`` coordinates,
    read through :func:`sipnet_calibration.sites.load_sites`.

Data model
----------
:func:`load_constraint` returns an ``xarray.Dataset`` shaped as follows. The
data variables, dims, coordinates and units are checked on load against the
spec; the dtypes are what the writer produces.

**Dimensions**: ``site``, and ``time`` unless the spec's structure is
``STATIC``, and ``bounds`` (of length 2) when ``time_bounds`` is present.

**Data variables**, both ``float64``, ``NaN`` where a site (and time) was not
observed, ``NaN`` at the same elements of both::

    value(site[, time])               the observation, in the spec's units
    standard_deviation(site[, time])  its reported standard deviation, same units

**Coordinates**

================ ================== =============================================
Name             Dims               Meaning
================ ================== =============================================
``site``         ``site``           ``int32``, the whole 1-8000 pool, ascending
``lon``, ``lat`` ``site``           ``float64``, from the site table
``time``         ``time``           ``datetime64[ns]``; see below
``time_bounds``  ``(time, bounds)`` the half-open interval a value covers,
                                    only when the structure documents one
================ ================== =============================================

**Time.** What the ``time`` label means depends on the spec's
:class:`TimeStructure` and is written on the coordinate in words:

* ``ANNUAL``: one value per calendar year. ``time`` is January 1 of the year,
  a key rather than an acquisition time, and ``time_bounds`` is the calendar
  year ``[Jan 1, next Jan 1)``.
* ``DATED``: one value per source date, carried exactly as the source wrote
  it, with no bounds: what the label marks is described on the variable, not
  encoded.
* ``STATIC``: no time dimension. The raw file's yearly copies are checked to
  be identical and collapsed to one value per site.

**Attributes** follow the Climate and Forecast conventions (CF-1.11), as
pySIPNET's model output does, so the two sides read alike. ``time`` carries
``standard_name``, ``axis`` and, when present, ``bounds``; ``lon`` and ``lat``
carry ``standard_name`` and ``units``; no coordinate is encoded with a
``_FillValue``. ``value`` carries the spec's ``units``, ``long_name``,
``description``, ``upstream_product``, ``source_file``, ``source_column``,
``time_reference``, ``units_provenance`` and, when set, ``constituent``,
``sign_convention`` and ``comment``. No constraint carries ``cell_methods``:
CF has no vocabulary for "the nearest composite" or "an annual map", and the
words are in ``time_reference`` and ``comment`` instead. The dataset carries
``Conventions``, ``title``, ``constraint``, ``upstream_product``, ``source_file``,
``time_structure``, ``rows_read``, ``rows_dropped_by_quality_flag``,
``rows_collapsed_as_copies``, ``history`` and ``created``.

**Units** are the raw file's units, unchanged. The ingest changes structure,
never values; converting an observation into model units, or the reverse, is
the observation operator's job.

**Missing values.** ``NaN`` means not observed. A zero is an observation.

Functions
---------
:func:`resolve_constraint`
    The spec for a constraint name, raising if there is none.

:func:`load_constraint`
    Read one constraint's processed file and check it against its spec.

:func:`constraint_fields`, :func:`constraint_standard_deviations`
    The ``value`` or ``standard_deviation`` arrays of several constraints, one
    field per constraint, optionally for a subset of sites. An annual
    constraint's field carries its windows, read from ``time_bounds``, as the
    one-dimensional coordinates ``window_start`` and ``window_end`` on
    ``time``.

:func:`read_raw`
    Parse a raw file exactly, in its source column names.

:func:`build_constraint`
    Turn a raw frame into the processed Dataset above. Pure; the ingest script
    wraps it with the checks and the write.

:func:`netcdf_encoding`
    The on-disk encoding the ingest script writes with.

:func:`describe`
    A spec rendered as a paragraph.

Notes
-----
**One spec, no separate processed schema.** The spec plays the role
pySIPNET's ``VariableSpec`` plays for model output: one flat record per
variable from which everything else is derived. Its ``xarray_attributes()``
is what makes the netCDF self-describing, so there is nothing to keep in
step between a raw description and a processed one.

**One processed file per constraint.** The five sources have three time
structures and no shared grid; a single dense file would re-impose the
assembler's alignment onto July 15 keys. Each constraint is stored at its
source's own
resolution, and how an observation is placed against model time is decided by
its observation operator.

**No unit conversion at ingest.** SoilGrids soil carbon is stored in the
source's ``Mg ha-1`` rather than the ``kg m-2`` of the assembled files it was
once compared against; the factor is Pint's to supply where it is needed.

**Dropping quality-flagged rows.** MODIS rows with ``qc == "001"`` fail the
producer's quality test and are equivalent to ``sd > 20``; their ``lai == 0``
values carry the upstream product's fill standard deviation (248 x 0.1 = 24.8). They
are dropped and counted at ingest rather than carried as observations. The
raw file keeps them.

Usage
-----
::

    from sipnet_calibration.constraints import (
        constraint_fields,
        constraint_standard_deviations,
        describe,
        load_constraint,
        resolve_constraint,
    )

    lai = load_constraint("modis_leaf_area_index")       # Dataset: value, standard_deviation
    lai["value"].sel(site=4102).dropna("time")            # one site's composites

    fields = constraint_fields(sites=[4102, 4113])        # every constraint, two sites
    fields["smap_soil_moisture"].dims                     # ('site', 'time')
    fields["soilgrids_soil_organic_carbon"].dims          # ('site',)
    fields["landtrendr_aboveground_biomass"].attrs["units"]   # 'Mg ha-1'

    standard_deviations = constraint_standard_deviations(["modis_leaf_area_index"])
    variance = standard_deviations["modis_leaf_area_index"] ** 2

    print(describe(resolve_constraint("smap_soil_moisture")))
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.dataset import BOUNDS_DIMENSION
from pysipnet.units import validate_units

from sipnet_calibration.conventions import (
    CF_CONVENTIONS,
    LAT,
    LON,
    NAME_PATTERN,
    SITE,
    SITE_ID,
    TIME,
    TIME_BOUNDS,
    WINDOW_END,
    WINDOW_START,
    FrozenMapping,
    data_root,
)
from sipnet_calibration.io import utc_timestamp
from sipnet_calibration.sites import check_site_table_lists_the_sites, site_coordinates
from sipnet_calibration.validation import as_names, as_site_ids, truncated

__all__ = [
    "CALENDAR",
    "CONSTRAINTS",
    "CONSTRAINT_NAMES",
    "MISSING_TOKEN",
    "PRODUCER_UNCONFIRMED",
    "STANDARD_DEVIATION",
    "TIME_REFERENCE_FOR_STRUCTURE",
    "TIME_UNITS",
    "VALUE",
    "ConstraintSpec",
    "TimeStructure",
    "build_constraint",
    "constraint_fields",
    "constraint_path",
    "constraint_standard_deviations",
    "default_constraints_dir",
    "default_raw_dir",
    "describe",
    "load_constraint",
    "netcdf_encoding",
    "read_raw",
    "resolve_constraint",
]


# ── the spec ──────────────────────────────────────────────────────────────────


class TimeStructure(StrEnum):
    """How a constraint's records are placed in time."""

    STATIC = "static"
    """One value per site with no time. A year column in the raw file, if any,
    is an artifact of the assembly and is checked to be constant and collapsed."""

    ANNUAL = "annual"
    """One value per calendar year. The raw time column holds the year; the
    processed ``time`` is January 1 of that year, with ``time_bounds`` spanning
    the calendar year."""

    DATED = "dated"
    """One value per source date. The raw time column holds an ISO date, which
    is carried as written with no bounds; what it marks is described in words."""


@dataclass(frozen=True)
class ConstraintSpec:
    """Everything a consumer needs to know about one constraint observation.

    One instance per raw file. The fields describe the quantity, the raw file
    that carries it and how it sits in time; :meth:`xarray_attributes` is what
    the processed netCDF stores, so the file describes itself.
    """

    name: str
    """Processed name: the raw file's stem, the registry key and the output file's stem."""

    long_label: str
    """Plot-ready name without units, e.g. ``"Leaf area index"``."""

    units: str
    """UDUNITS-style unit string, physical units only, validated by :mod:`pysipnet.units`."""

    constituent: str
    """Substance the unit refers to, ``"C"`` for carbon, or ``""``."""

    description: str
    """What the quantity is and how the producer constructed it, with the citation."""

    upstream_product: str
    """The producer's own product name, e.g. ``"MODIS MCD15A3H v061"``."""

    time_structure: TimeStructure
    """How the records sit in time; see :class:`TimeStructure`."""

    raw_file: str
    """File name under ``data/raw/constraints/``."""

    raw_columns: tuple[str, ...]
    """The raw file's header, in order; :func:`read_raw` refuses any other."""

    value_column: str
    """Raw column holding the observation."""

    sd_column: str
    """Raw column holding the observation's standard deviation."""

    time_column: str | None
    """Raw column holding the year (``ANNUAL``, ``STATIC``) or the ISO date
    (``DATED``); ``None`` only for a static table with no time column."""

    quality_column: str | None = None
    """Raw column holding a quality flag; rows not equal to *quality_pass* are dropped."""

    quality_pass: str = ""
    """The flag value of a row that passes; required with *quality_column*."""

    units_provenance: str = ""
    """Where the unit comes from and how firm it is, in a sentence."""

    sign_convention: str = ""
    """Which direction is positive, when that is not obvious."""

    comment: str = ""
    """Anything else a reader must know; the CF ``comment`` attribute."""

    def __post_init__(self) -> None:
        if not NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Constraint name {self.name!r} is not lower_case_with_underscores."
            )
        validate_units(self.units)
        if not self.description or not self.long_label or not self.upstream_product:
            raise ValueError(
                f"Constraint {self.name!r} needs a description, long_label and upstream_product."
            )
        if len(set(self.raw_columns)) != len(self.raw_columns):
            raise ValueError(f"Constraint {self.name!r}: raw_columns repeats a column.")
        if SITE_ID not in self.raw_columns:
            raise ValueError(f"Constraint {self.name!r}: raw_columns lacks {SITE_ID!r}.")
        for role, column in self._named_columns().items():
            if column not in self.raw_columns:
                raise ValueError(
                    f"Constraint {self.name!r}: {role} {column!r} is not in raw_columns "
                    f"{self.raw_columns}."
                )
        if self.time_column is None and self.time_structure is not TimeStructure.STATIC:
            raise ValueError(
                f"Constraint {self.name!r}: a {self.time_structure.value} constraint needs a time_column."
            )
        if (self.quality_column is None) != (self.quality_pass == ""):
            raise ValueError(
                f"Constraint {self.name!r}: quality_column and quality_pass go together."
            )

    @property
    def time_reference(self) -> str:
        """In words, what the ``time`` label of the processed file marks."""
        return TIME_REFERENCE_FOR_STRUCTURE[self.time_structure]

    @property
    def has_time_bounds(self) -> bool:
        """Whether the processed file carries ``time_bounds``."""
        return self.time_structure is TimeStructure.ANNUAL

    @property
    def dims(self) -> tuple[str, ...]:
        """The dims of the processed data variables."""
        if self.time_structure is TimeStructure.STATIC:
            return (SITE,)
        return (SITE, TIME)

    def xarray_attributes(self) -> dict[str, Any]:
        """Attributes for the ``value`` array of the processed file.

        Keys follow the Climate and Forecast conventions where one exists
        (``units``, ``long_name``, ``comment``); the rest are spelled out.
        """
        attrs: dict[str, Any] = {
            "units": self.units,
            "long_name": self.long_label,
            "description": self.description,
            "upstream_product": self.upstream_product,
            "source_file": self.raw_file,
            "source_column": self.value_column,
            "time_reference": self.time_reference,
            "units_provenance": self.units_provenance,
        }
        if self.constituent:
            attrs["constituent"] = self.constituent
        if self.sign_convention:
            attrs["sign_convention"] = self.sign_convention
        if self.comment:
            attrs["comment"] = self.comment
        return attrs

    def _named_columns(self) -> dict[str, str]:
        columns = {"value_column": self.value_column, "sd_column": self.sd_column}
        if self.time_column is not None:
            columns["time_column"] = self.time_column
        if self.quality_column is not None:
            columns["quality_column"] = self.quality_column
        return columns


#: How a raw file writes a missing value.
MISSING_TOKEN = "NA"

#: In words, what the ``time`` label of a constraint with each structure marks.
TIME_REFERENCE_FOR_STRUCTURE: Mapping[TimeStructure, str] = FrozenMapping(
    {
        TimeStructure.STATIC: (
            "a static map with no time dimension. The source repeated one value into "
            "every year; the copies were checked to be identical and collapsed."
        ),
        TimeStructure.ANNUAL: (
            "the value attributed to the calendar year given by time_bounds; the "
            "January 1 label is a key, not an acquisition time."
        ),
        TimeStructure.DATED: (
            "the source's own date label, carried as written. What instant or "
            "interval it marks is stated in the comment, not encoded."
        ),
    }
)

#: The sentence every unit provenance ends with, because it is true of every one.
PRODUCER_UNCONFIRMED = "Not confirmed by the producer; see data/README.md, open question 9."


# ── the registry ──────────────────────────────────────────────────────────────

CONSTRAINTS: tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        name="landtrendr_aboveground_biomass",
        long_label="Aboveground biomass",
        units="Mg ha-1",
        constituent="C",
        description=(
            "That calendar year's LandTrendr annual Landsat biomass map at the site, "
            "2012-2023, as extracted by PEcAn Landtrendr_AGB_prep.R. Means are whole "
            "numbers. Standard deviations for 2012-2017 are LandTrendr's own and are "
            "whole numbers; 2018-2023 come from a separate object beside a random-forest "
            "model and appear to be predicted. Coverage is US land only. The season "
            "within the year that an annual value represents is not documented."
        ),
        upstream_product="LandTrendr annual Landsat biomass",
        time_structure=TimeStructure.ANNUAL,
        raw_file="landtrendr_aboveground_biomass.csv.gz",
        raw_columns=("site_id", "year", "agb_mean", "agb_sd"),
        value_column="agb_mean",
        sd_column="agb_sd",
        time_column="year",
        units_provenance=(
            "Documented as Mg C ha-1 for the reanalysis output. PEcAn's prep applies no "
            "biomass-to-carbon factor and LandTrendr's native product is dry biomass, so "
            "the constituent is unconfirmed by about a factor of two. " + PRODUCER_UNCONFIRMED
        ),
        comment=(
            "929 records carry a standard deviation of exactly zero, 925 of them with a "
            "mean of zero; they are written through unchanged."
        ),
    ),
    ConstraintSpec(
        name="gedi_aboveground_biomass",
        long_label="Aboveground biomass",
        units="Mg ha-1",
        constituent="",
        description=(
            "An annual GEDI aboveground biomass value at the site, 2019-2024. The upstream "
            "product, "
            "its version, and how footprint retrievals were aggregated to the 1 km site "
            "are not documented. Independent of LandTrendr and not part of the set the "
            "reanalysis assimilated."
        ),
        upstream_product="GEDI",
        time_structure=TimeStructure.ANNUAL,
        raw_file="gedi_aboveground_biomass.csv.gz",
        raw_columns=("year", "site_id", "agb", "sd"),
        value_column="agb",
        sd_column="sd",
        time_column="year",
        units_provenance=(
            "Not established. Mg ha-1 is assumed as the native GEDI biomass unit; whether "
            "the values are carbon or dry biomass is unknown, so no constituent is recorded. "
            + PRODUCER_UNCONFIRMED
        ),
    ),
    ConstraintSpec(
        name="modis_leaf_area_index",
        long_label="Leaf area index",
        units="m2 m-2",
        constituent="",
        description=(
            "Lai_500m and LaiStdDev_500m of each 4-day MODIS composite at the site, June "
            "through August of 2011-2024, as extracted by PEcAn MODIS_LAI_prep.R. Rows "
            "flagged qc '001' fail the producer's quality test, are equivalent to sd > 20, "
            "and are dropped at ingest; their sd of 24.8 is the upstream product's fill "
            "value 248 x 0.1. LAI is one-sided green leaf area per unit ground area in broadleaf "
            "canopies and half the total needle area in conifers."
        ),
        upstream_product="MODIS MCD15A3H v061",
        time_structure=TimeStructure.DATED,
        raw_file="modis_leaf_area_index.csv.gz",
        raw_columns=("date", "site_id", "lat", "lon", "lai", "sd", "qc"),
        value_column="lai",
        sd_column="sd",
        time_column="date",
        quality_column="qc",
        quality_pass="000",
        units_provenance=(
            "The upstream product's documented unit and 0.1 scale factor. "
            + PRODUCER_UNCONFIRMED
        ),
        comment=(
            "The date is the composite's label as the extraction returned it. The "
            "compositing period is 4 days; whether the label marks its first day is not "
            "confirmed, so no time_bounds variable is written. Many unflagged records "
            "carry a standard deviation of exactly zero; they are written through "
            "unchanged."
        ),
    ),
    ConstraintSpec(
        name="smap_soil_moisture",
        long_label="Soil moisture",
        units="percent",
        constituent="",
        description=(
            "SMAP Level 4 sm_profile_analysis at the site, multiplied by 100, on the "
            "July 15 key of each year 2015-2024, as extracted by PEcAn SMAP_SMP_prep.R. "
            "The standard deviation is either a fixed 4 or the L4 ensemble standard "
            "deviation x 100; which produced this file is not documented. The source "
            "grid is coarser than the site grid, so neighboring sites can share a value."
        ),
        upstream_product="SMAP Level 4 soil moisture",
        time_structure=TimeStructure.DATED,
        raw_file="smap_soil_moisture.csv.gz",
        raw_columns=("date", "site_id", "lat", "lon", "smp", "sd"),
        value_column="smp",
        sd_column="sd",
        time_column="date",
        units_provenance=(
            "A fraction multiplied by 100 in the prep code. What the fraction is of "
            "(volumetric water, saturation, holding capacity) and over what depth is not "
            "established. " + PRODUCER_UNCONFIRMED
        ),
        comment=(
            "The date is the assembler's July 15 snapshot key, not an acquisition time. "
            "Documented as a single SMAP L4 value on that day; the time of day is not "
            "confirmed."
        ),
    ),
    ConstraintSpec(
        name="soilgrids_soil_organic_carbon",
        long_label="Soil organic carbon",
        units="Mg ha-1",
        constituent="C",
        description=(
            "SoilGrids250m soil organic carbon at the site, integrated over 0-200 cm "
            "(established by correlation, not by an attribute), as extracted by PEcAn "
            "Soilgrids_SoilC_prep.R. A static map: the assembler wrote the same value "
            "into every year 2012-2024."
        ),
        upstream_product="SoilGrids250m v2.0",
        time_structure=TimeStructure.STATIC,
        raw_file="soilgrids_soil_organic_carbon.csv.gz",
        raw_columns=("site_id", "soc", "sd", "year"),
        value_column="soc",
        sd_column="sd",
        time_column="year",
        units_provenance=(
            "Inferred: the values are exactly ten times those of the assembled files, "
            "which are declared kg C m-2 on the same unconfirmed basis. " + PRODUCER_UNCONFIRMED
        ),
    ),
)

#: The constraint names, in registry order.
CONSTRAINT_NAMES: tuple[str, ...] = tuple(spec.name for spec in CONSTRAINTS)


def resolve_constraint(name: str) -> ConstraintSpec:
    """The spec named *name*, or a ``KeyError`` listing the names that exist."""
    for spec in CONSTRAINTS:
        if spec.name == name:
            return spec
    raise KeyError(f"No constraint named {name!r}. Known: {list(CONSTRAINT_NAMES)}")


# ── the processed file ────────────────────────────────────────────────────────

#: Name of the observation array in the processed file.
VALUE = "value"

#: Name of the standard-deviation array in the processed file.
STANDARD_DEVIATION = "standard_deviation"


#: On-disk time encoding. Written explicitly so nothing is inherited from a default.
TIME_UNITS = "days since 2000-01-01"
CALENDAR = "proleptic_gregorian"


def default_raw_dir() -> Path:
    """Where the raw constraint files are expected: ``data/raw/constraints/``.

    ``$SIPNET_CALIBRATION_DATA`` replaces ``data/`` when set.
    """
    return _data_root() / "raw" / "constraints"


def default_constraints_dir() -> Path:
    """Where the processed files are expected: ``data/processed/constraints/``.

    ``$SIPNET_CALIBRATION_DATA`` replaces ``data/`` when set.
    """
    return _data_root() / "processed" / "constraints"


def constraint_path(
    constraint: str | ConstraintSpec, directory: Path | str | None = None
) -> Path:
    """The processed file of a constraint: ``<directory>/<name>.nc``."""
    name = constraint if isinstance(constraint, str) else constraint.name
    base = Path(directory) if directory is not None else default_constraints_dir()
    return base / f"{name}.nc"


def load_constraint(
    constraint: str | ConstraintSpec, path: Path | str | None = None
) -> xr.Dataset:
    """Read one processed constraint and check it against its spec.

    Parameters
    ----------
    constraint:
        A constraint name from :data:`CONSTRAINT_NAMES`, or a spec.
    path:
        The netCDF to read. Defaults to :func:`constraint_path`.

    Returns
    -------
    xarray.Dataset
        :data:`VALUE` and :data:`STANDARD_DEVIATION` on the spec's dims, with
        ``lon`` and ``lat`` on ``site``, and ``NaN`` where not observed.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with the command that produces it.
    ValueError
        If the file does not match the spec's data model.
    """
    spec = constraint if isinstance(constraint, ConstraintSpec) else resolve_constraint(constraint)
    path = Path(path) if path is not None else constraint_path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Produce it with:\n"
            f"  python scripts/ingest_constraints.py --constraint {spec.name}"
        )
    dataset = xr.open_dataset(path, engine="h5netcdf")
    try:
        _check_processed_file_matches_the_spec(dataset, spec, path)
    except Exception:
        dataset.close()
        raise
    return dataset


def constraint_fields(
    names: Sequence[str] | None = None,
    *,
    sites: Iterable[int] | None = None,
    directory: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The observations of several constraints, one field each.

    Parameters
    ----------
    names:
        Constraint names, a sequence, in the order the result should carry
        them. Defaults to every constraint in :data:`CONSTRAINT_NAMES`.
    sites:
        Site ids to keep, a sequence, in the order given, each once. Defaults
        to the whole pool.
    directory:
        Where the processed files are. Defaults to
        :func:`default_constraints_dir`.

    Returns
    -------
    dict
        Constraint name to its ``value`` array, renamed to the constraint,
        with dims ``(site, time)`` or ``(site,)`` and the array's attributes.
        An annual constraint's field also carries its windows, read from the
        CF ``time_bounds``, as the one-dimensional coordinates
        ``window_start`` and ``window_end`` on ``time``
        (:data:`~sipnet_calibration.conventions.WINDOW_START`,
        :data:`~sipnet_calibration.conventions.WINDOW_END`).

    Raises
    ------
    TypeError
        If *names* or *sites* is one value, a string or a set; if a name is
        not a string; or if a site id is a boolean, a float or not a number.
    ValueError
        If a site id is not from 1 to the largest ``int32``, is asked for
        twice, or *sites* is a two-dimensional array.
    KeyError
        If a name is not a constraint, or a requested site is not in the
        processed file.
    """
    return _fields(VALUE, names, sites, directory)


def constraint_standard_deviations(
    names: Sequence[str] | None = None,
    *,
    sites: Iterable[int] | None = None,
    directory: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The reported standard deviations, as :func:`constraint_fields` does the values."""
    return _fields(STANDARD_DEVIATION, names, sites, directory)


# ── raw to processed ──────────────────────────────────────────────────────────


def read_raw(spec: ConstraintSpec, root: Path | str | None = None) -> pd.DataFrame:
    """Parse a constraint's raw file exactly, in its source column names.

    Parameters
    ----------
    spec:
        Which constraint.
    root:
        The directory holding the raw files. Defaults to
        :func:`default_raw_dir`.

    Returns
    -------
    pandas.DataFrame
        The columns of ``spec.raw_columns``, in order: ``site_id`` as
        ``int64``, the value and standard deviation as ``float64`` with
        ``NaN`` where the file says ``NA``, a ``DATED`` time column and any
        quality column as strings, a year column as ``int64``.

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    ValueError
        If the header is not ``spec.raw_columns`` or the file has no rows.

    Notes
    -----
    ``float_precision="round_trip"`` is what makes the read exact: three of the
    files were written from R at 17 significant digits, which pandas' default
    parser does not reproduce. ``keep_default_na=False`` with ``na_values``
    set to the literal ``NA`` keeps the quality flag ``"000"`` a string and
    lets nothing else become missing by accident.
    """
    path = (Path(root) if root is not None else default_raw_dir()) / spec.raw_file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see data/raw/constraints/provenance.md")

    try:
        with warnings.catch_warnings():
            # A row with more fields than the header only warns by default and
            # loses its trailing field; here that is a malformed file.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(
                path,
                dtype=_raw_dtypes(spec),
                float_precision="round_trip",
                keep_default_na=False,
                na_values=[MISSING_TOKEN],
                index_col=False,
            )
    except (ValueError, OverflowError, pd.errors.ParserWarning) as error:
        raise ValueError(f"{path}: could not be parsed as its spec declares: {error}") from error
    if tuple(frame.columns) != spec.raw_columns:
        raise ValueError(
            f"{path}: header is {tuple(frame.columns)}, expected {spec.raw_columns}. "
            "A changed raw file is a spec change, not a new row."
        )
    if frame.empty:
        raise ValueError(f"{path}: holds no rows")
    return frame


def build_constraint(
    spec: ConstraintSpec, frame: pd.DataFrame, site_table: pd.DataFrame
) -> xr.Dataset:
    """Turn a raw frame into the processed Dataset the data model describes.

    Parameters
    ----------
    spec:
        Which constraint.
    frame:
        As :func:`read_raw` returns it.
    site_table:
        The site table from :func:`sipnet_calibration.sites.load_sites`; its
        ``site_id`` is the pool and its ``lon``/``lat`` the coordinates.

    Returns
    -------
    xarray.Dataset
        Dense over the whole pool (and every time label the kept rows carry),
        ``NaN`` where not observed, with every attribute the data model lists.

    Raises
    ------
    KeyError
        If a row's site is not in the site table.
    ValueError
        If two kept rows share a ``(site, time)``, or if a static
        constraint's copies differ.

    Notes
    -----
    This is pure: it neither reads nor writes files, so the ingest script and
    the tests call it on the same frames. The friendlier, earlier checks live
    in the script; the guards here are the ones that would otherwise let a
    fancy-indexed assignment silently overwrite an element.
    """
    kept, n_dropped = _apply_quality_filter(spec, frame)
    site = np.sort(site_table[SITE_ID].to_numpy(np.int64))

    row_site = kept[SITE_ID].to_numpy(np.int64)
    check_site_table_lists_the_sites(
        site_table, np.unique(row_site).tolist(), message_name=f"{spec.name}: site(s)"
    )
    site_index = np.searchsorted(site, row_site)

    n_collapsed = 0
    if spec.time_structure is TimeStructure.STATIC:
        arrays = _static_arrays(spec, kept, site_index, site.size)
        n_collapsed = len(kept) - len(np.unique(site_index))
        coords: dict[str, Any] = {}
    else:
        time = _time_labels(spec, kept)
        arrays = _dated_arrays(spec, kept, site_index, time, site.size)
        coords = _time_coords(spec, time)

    coords.update(site_coordinates(site.tolist(), site_table))
    dataset = xr.Dataset(
        {
            VALUE: (spec.dims, arrays[0], spec.xarray_attributes()),
            STANDARD_DEVIATION: (spec.dims, arrays[1], _sd_attributes(spec)),
        },
        coords=coords,
        attrs=_dataset_attributes(
            spec, n_read=len(frame), n_dropped=n_dropped, n_collapsed=n_collapsed
        ),
    )
    return dataset


def netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """The on-disk encoding for a constraint built by :func:`build_constraint`.

    Both data arrays are compressed with ``NaN`` as the fill value; ``time``
    and ``time_bounds`` are integer days on :data:`TIME_UNITS`; and no
    coordinate carries a ``_FillValue``, as CF requires.
    """
    encoding: dict[str, dict[str, Any]] = {
        VALUE: {"zlib": True, "complevel": 4, "_FillValue": np.nan},
        STANDARD_DEVIATION: {"zlib": True, "complevel": 4, "_FillValue": np.nan},
    }
    for name in dataset.coords:
        encoding[str(name)] = {"_FillValue": None}
    if TIME in dataset.coords:
        encoding[TIME].update({"units": TIME_UNITS, "calendar": CALENDAR, "dtype": "int32"})
    if TIME_BOUNDS in dataset.coords:
        encoding[TIME_BOUNDS].update(
            {"units": TIME_UNITS, "calendar": CALENDAR, "dtype": "int32"}
        )
    return encoding


def describe(spec: ConstraintSpec) -> str:
    """A constraint spec as a paragraph, for ``--describe`` and the run log."""
    units = f"{spec.units} {spec.constituent}".strip()
    lines = [
        f"{spec.name}: {spec.long_label} ({units}), from {spec.upstream_product}.",
        f"  raw file   {spec.raw_file}",
        f"  columns    value {spec.value_column!r}, sd {spec.sd_column!r}"
        + (f", time {spec.time_column!r}" if spec.time_column else "")
        + (
            f"; rows kept where {spec.quality_column!r} == {spec.quality_pass!r}"
            if spec.quality_column
            else ""
        ),
        f"  time       {spec.time_structure.value}: {spec.time_reference}",
        f"  units      {spec.units_provenance}",
        f"  what       {spec.description}",
    ]
    if spec.comment:
        lines.append(f"  comment    {spec.comment}")
    return "\n".join(lines)


# ── supporting helpers ────────────────────────────────────────────────────────

def _data_root() -> Path:
    return data_root()


def _raw_dtypes(spec: ConstraintSpec) -> dict[str, Any]:
    """What to hand pandas per column, so nothing is inferred."""
    # site_id is read wide and checked before narrowing, as load_sites does:
    # reading straight into int32 wraps silently. The value and sd are declared
    # float64 because pandas infers int64 for an all-integer column and
    # float_precision then does not apply.
    dtypes: dict[str, Any] = {
        SITE_ID: np.int64,
        spec.value_column: np.float64,
        spec.sd_column: np.float64,
    }
    if spec.time_column is not None:
        dtypes[spec.time_column] = (
            str if spec.time_structure is TimeStructure.DATED else np.int64
        )
    if spec.quality_column is not None:
        dtypes[spec.quality_column] = str
    for column in spec.raw_columns:
        dtypes.setdefault(column, np.float64)
    return dtypes


def _apply_quality_filter(spec: ConstraintSpec, frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if spec.quality_column is None:
        return frame, 0
    passes = frame[spec.quality_column] == spec.quality_pass
    return frame[passes], int((~passes).sum())


def _time_labels(spec: ConstraintSpec, frame: pd.DataFrame) -> pd.DatetimeIndex:
    """The processed ``time`` value of every row, in row order."""
    column = frame[spec.time_column]
    if spec.time_structure is TimeStructure.ANNUAL:
        stamps = pd.to_datetime(column.astype(np.int64).astype(str), format="%Y")
    else:
        stamps = pd.to_datetime(column, format="%Y-%m-%d")
    return pd.DatetimeIndex(stamps).as_unit("ns")


def _dated_arrays(
    spec: ConstraintSpec,
    frame: pd.DataFrame,
    site_index: np.ndarray,
    row_time: pd.DatetimeIndex,
    n_sites: int,
) -> tuple[np.ndarray, np.ndarray]:
    time = pd.DatetimeIndex(sorted(row_time.unique())).as_unit("ns")
    time_index = time.get_indexer(row_time)
    _check_no_duplicate_site_time_keys(site_index, time_index, spec)

    value = np.full((n_sites, time.size), np.nan)
    sd = np.full((n_sites, time.size), np.nan)
    value[site_index, time_index] = frame[spec.value_column].to_numpy(np.float64)
    sd[site_index, time_index] = frame[spec.sd_column].to_numpy(np.float64)
    return value, sd


def _static_arrays(
    spec: ConstraintSpec, frame: pd.DataFrame, site_index: np.ndarray, n_sites: int
) -> tuple[np.ndarray, np.ndarray]:
    _check_static_copies_agree(spec, frame)
    first = ~pd.Series(site_index).duplicated().to_numpy()
    value = np.full(n_sites, np.nan)
    sd = np.full(n_sites, np.nan)
    value[site_index[first]] = frame[spec.value_column].to_numpy(np.float64)[first]
    sd[site_index[first]] = frame[spec.sd_column].to_numpy(np.float64)[first]
    return value, sd


def _time_coords(spec: ConstraintSpec, row_time: pd.DatetimeIndex) -> dict[str, Any]:
    time = pd.DatetimeIndex(sorted(row_time.unique())).as_unit("ns")
    attrs = {
        "standard_name": "time",
        "axis": "T",
        "long_name": _TIME_LONG_NAME[spec.time_structure],
        "comment": spec.time_reference,
    }
    coords: dict[str, Any] = {TIME: (TIME, time.to_numpy(), attrs)}
    if spec.has_time_bounds:
        attrs["bounds"] = TIME_BOUNDS
        start = time.to_numpy()
        end = (time + pd.DateOffset(years=1)).as_unit("ns").to_numpy()
        coords[TIME_BOUNDS] = (
            (TIME, BOUNDS_DIMENSION),
            np.stack([start, end], axis=1),
            {
                "long_name": "Calendar year the value is attributed to",
                "comment": "The half-open interval [time, time + 1 year), in the CF bounds form.",
            },
        )
    return coords


_TIME_LONG_NAME = FrozenMapping(
    {
        TimeStructure.ANNUAL: "Calendar year key",
        TimeStructure.DATED: "Source date label",
    }
)


def _sd_attributes(spec: ConstraintSpec) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "units": spec.units,
        "long_name": f"{spec.long_label}: reported standard deviation",
        "description": (
            f"The standard deviation the source reports beside each {spec.value_column!r}; "
            "what it measures is the producer's to say."
        ),
        "source_file": spec.raw_file,
        "source_column": spec.sd_column,
    }
    if spec.constituent:
        attrs["constituent"] = spec.constituent
    return attrs


def _dataset_attributes(
    spec: ConstraintSpec, *, n_read: int, n_dropped: int, n_collapsed: int
) -> dict[str, Any]:
    return {
        "Conventions": CF_CONVENTIONS,
        "title": f"{spec.long_label} constraint from {spec.upstream_product}",
        "constraint": spec.name,
        "upstream_product": spec.upstream_product,
        "source_file": spec.raw_file,
        "time_structure": spec.time_structure.value,
        "rows_read": n_read,
        "rows_dropped_by_quality_flag": n_dropped,
        "rows_collapsed_as_copies": n_collapsed,
        "history": (
            f"scripts/ingest_constraints.py: read data/raw/constraints/{spec.raw_file}"
            + (
                f", kept rows with {spec.quality_column} == {spec.quality_pass!r}"
                if spec.quality_column
                else ""
            )
            + (
                ", collapsed the identical yearly copies"
                if spec.time_structure is TimeStructure.STATIC
                else ""
            )
            + ", placed the records on the site pool"
        ),
        "created": utc_timestamp(),
    }


def _fields(
    array: str,
    names: Sequence[str] | None,
    sites: Iterable[int] | None,
    directory: Path | str | None,
) -> dict[str, xr.DataArray]:
    names = CONSTRAINT_NAMES if names is None else as_names(names, message_name="names")
    wanted = None if sites is None else list(as_site_ids(sites, message_name="sites"))
    fields: dict[str, xr.DataArray] = {}
    for name in names:
        spec = resolve_constraint(name)
        dataset = load_constraint(spec, constraint_path(spec, directory))
        field = dataset[array].rename(name)
        if TIME_BOUNDS in dataset.coords:
            field = field.assign_coords(_window_coords(dataset))
        if wanted is not None:
            check_constraint_holds_the_sites(dataset, wanted, name=name)
            field = field.sel({SITE: wanted})
        fields[name] = field
    return fields


def _window_coords(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """CF ``time_bounds`` as the two one-dimensional window coordinates on ``time``.

    A ``DataArray`` cannot carry the ``(time, bounds)`` variable, its
    ``bounds`` dimension being none of the array's, so the pair rides along
    as :data:`~sipnet_calibration.conventions.WINDOW_START` and
    :data:`~sipnet_calibration.conventions.WINDOW_END`, the way pySIPNET's
    model output carries ``time_step_start`` beside ``time``.
    """
    bounds = dataset[TIME_BOUNDS]
    comment = "One edge of the CF time_bounds of the value at this label."
    start_name = "Start of the interval the value is attributed to"
    end_name = "End of the interval the value is attributed to"
    return {
        WINDOW_START: xr.DataArray(
            bounds.isel({BOUNDS_DIMENSION: 0}).values,
            dims=TIME,
            attrs={"long_name": start_name, "comment": comment},
        ),
        WINDOW_END: xr.DataArray(
            bounds.isel({BOUNDS_DIMENSION: 1}).values,
            dims=TIME,
            attrs={"long_name": end_name, "comment": comment},
        ),
    }


# ── checks ────────────────────────────────────────────────────────────────────


def check_constraint_holds_the_sites(
    dataset: xr.Dataset, site_ids: Sequence[int], *, name: str
) -> None:
    """A constraint's processed file holds every site asked of it."""
    held = set(dataset[SITE].values.tolist())
    missing = [site for site in site_ids if site not in held]
    if missing:
        raise KeyError(
            f"{name}: site(s) {truncated(missing)} are not in the processed file; ask only for "
            "sites of the site table it was built on."
        )


def _check_no_duplicate_site_time_keys(
    site_index: np.ndarray, time_index: np.ndarray, spec: ConstraintSpec
) -> None:
    keys = pd.MultiIndex.from_arrays([site_index, time_index])
    if keys.has_duplicates:
        n = int(keys.duplicated().sum())
        raise ValueError(
            f"{spec.name}: {n} rows share a (site, time) with another row. Two records "
            "for one (site, time) key would silently overwrite each other."
        )


def _check_static_copies_agree(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Raise unless every site carries one value across the raw time column."""
    distinct = frame.groupby(SITE_ID)[[spec.value_column, spec.sd_column]].nunique(
        dropna=False
    )
    varying = distinct[(distinct > 1).any(axis=1)]
    if not varying.empty:
        raise ValueError(
            f"{spec.name}: {len(varying)} sites carry different values in different "
            f"years (first: {varying.index[:5].tolist()}). A static constraint must be "
            "constant; if the source now varies in time its time_structure is wrong."
        )


def _check_processed_file_matches_the_spec(
    dataset: xr.Dataset, spec: ConstraintSpec, path: Path
) -> None:
    """Raise unless *dataset* is the processed file the spec describes."""
    missing = {VALUE, STANDARD_DEVIATION} - set(dataset.data_vars)
    if missing:
        raise ValueError(f"{path}: missing data variables {sorted(missing)}")
    for name in (VALUE, STANDARD_DEVIATION):
        if dataset[name].dims != spec.dims:
            raise ValueError(f"{path}: {name} has dims {dataset[name].dims}, expected {spec.dims}")
        if dataset[name].attrs.get("units") != spec.units:
            raise ValueError(
                f"{path}: {name} has units {dataset[name].attrs.get('units')!r}, the spec "
                f"says {spec.units!r}"
            )
    if dataset.attrs.get("constraint") != spec.name:
        raise ValueError(
            f"{path}: written for constraint {dataset.attrs.get('constraint')!r}, not {spec.name!r}"
        )

    for coordinate in (SITE, LON, LAT, *spec.dims):
        if coordinate not in dataset.coords:
            raise ValueError(f"{path}: missing the {coordinate!r} coordinate")
    for coordinate in (LON, LAT):
        if dataset[coordinate].dims != (SITE,):
            raise ValueError(
                f"{path}: {coordinate} must be on site, has dims {dataset[coordinate].dims}"
            )
    if (TIME_BOUNDS in dataset.coords) != spec.has_time_bounds:
        raise ValueError(
            f"{path}: time_bounds {'present' if TIME_BOUNDS in dataset.coords else 'absent'}, "
            f"but a {spec.time_structure.value} constraint "
            f"{'carries' if spec.has_time_bounds else 'does not carry'} them"
        )

    site = dataset[SITE].values
    if site.size == 0 or np.any(np.diff(site) <= 0):
        raise ValueError(f"{path}: site is empty or not strictly ascending")
    if TIME in dataset.dims:
        time = dataset[TIME].values
        if time.size == 0 or np.any(np.diff(time) <= np.timedelta64(0, "ns")):
            raise ValueError(f"{path}: time is empty or not strictly ascending")

    observed = np.isfinite(dataset[VALUE].values)
    if not np.array_equal(observed, np.isfinite(dataset[STANDARD_DEVIATION].values)):
        raise ValueError(
            f"{path}: value and standard_deviation are missing at different elements"
        )
