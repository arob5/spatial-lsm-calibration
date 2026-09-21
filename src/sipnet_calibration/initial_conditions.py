"""The initial condition ensemble: variable specs and processing code.

Overview
--------
The reanalysis behind the 8000-site pool started every ensemble member from a
set of initial state values drawn by a PEcAn script, one netCDF per site and
member. This module holds one :class:`InitialConditionSpec` per variable in
those files -- what the quantity is, where it came from, which SIPNET initial
parameter PEcAn fed it into and how -- and the functions that read the source
files, the converted raw file and the processed product. The spec's fields are
written into the processed netCDF as attributes, so it needs no description
beyond itself; the raw file keeps the source files' own attribute strings.

The dependency runs one way, and the first arrow is taken once, on the SCC::

    <site>/IC_site_<site>_<member>.nc  x 800,000      (the PEcAn source files)
      -> scripts/raw_sources/convert_initial_conditions.py
      -> raw/initial_conditions/pecan_pool_initial_conditions.nc   tracked
      -> scripts/ingest_initial_conditions.py    read_raw(), build_initial_conditions()
      -> processed/initial_conditions.nc
      -> this module                              load_initial_conditions()

``data/README.md`` documents the source files and the open questions about
them; ``data/raw/initial_conditions/provenance.md`` records the conversion.

Input data
----------
``data/raw/initial_conditions/files/<site>/IC_site_<site>_<member>.nc``
    The source files, present only on the SCC. netCDF-3 classic, no
    global attributes, one unlimited ``time`` dimension of length 1 whose
    variable carries an undecodable units template, and three to five scalar
    ``float64`` variables named as :data:`SOURCE_NAMES`, each declaring
    ``_FillValue = -999.0``, ``long_name`` and ``units``.
    :func:`read_source_file` parses one exactly and refuses anything else.

``data/raw/initial_conditions/pecan_pool_initial_conditions.nc``
    The values of the same 800,000 files as one array, in the source files'
    variable names, units strings and 1-based member index; the raw input
    everything else reads. :func:`read_raw` checks it against the specs.

``data/processed/sites/sites.csv``
    The site table, for the pool and the ``lon``/``lat`` coordinates.

Data model
----------
:func:`load_initial_conditions` returns an ``xarray.Dataset`` shaped as
follows. Variables, dims, coordinates and units are checked on load against
the specs; the dtypes are what the writer produces.

**Dimensions**: ``member``, ``site``. There is no ``time``: the source's is a
length-1 record dimension whose units attribute is an unsubstituted template,
and what it claimed is kept verbatim in the dataset attributes.

**Data variables**, one per spec, all ``float64`` on ``(member, site)``,
``NaN`` where no source file for the site carries the variable. Units follow
pySIPNET's convention of a physical ``units`` string plus a ``constituent``
attribute, so ``attrs["units"]`` is ``"kg m-2"`` and ``attrs["constituent"]``
is ``"C"``, never ``"kg C m-2"``::

    initial_aboveground_biomass_carbon    kg m-2, C   source AbvGrndWood
    initial_wood_carbon                   kg m-2, C   source wood_carbon_content
    initial_leaf_carbon                   kg m-2, C   source leaf_carbon_content
    initial_soil_organic_carbon           kg m-2, C   source soil_organic_carbon_content
    initial_soil_moisture_saturation      percent     source SoilMoistFrac

**Coordinates**

=================== ============ ===================================================
Name                Dims         Meaning
=================== ============ ===================================================
``member``          ``member``   ``int16``, 0-based, ascending; the project convention
``source_member``   ``member``   ``int16``, the 1-based index in the source file name
``site``            ``site``     ``int32``, the whole 1-8000 pool, ascending
``lon``, ``lat``    ``site``     ``float64``, from the site table
=================== ============ ===================================================

**Attributes** follow CF-1.11 as the constraint products do. Each variable
carries the spec's ``units``, ``long_name``, ``description``, ``product``,
``source_name``, ``source_units``, ``source_long_name``,
``sipnet_initial_condition``, ``pecan_conversion``, ``units_provenance`` and,
when set, ``constituent`` and ``comment``. The dataset carries
``Conventions``, ``title``, ``product``, ``source_file``, ``source_root``,
``source_script``, ``source_script_note``, ``nominal_date``,
``nominal_date_provenance``,
``source_time_units``, ``source_time_long_name``, ``source_time_value``,
``member_source``, ``member_correspondence``, ``n_sites``, ``n_members``,
``history`` and ``created``.

**Values are the source files', unchanged.** No unit conversion, no masking:
negative wood and leaf carbon are written through and counted in the ingest
report, because dropping or flooring them is a modeling decision (see the
``comment`` of those two specs for what PEcAn itself did).

**Missing values.** ``NaN`` has one meaning: none of the site's 100 source
files carries the variable. That absence is
a property of the site, checked to be identical across members, and no file
holds an explicit fill; both are asserted at conversion and on load.

Functions
---------
:func:`resolve_initial_condition`
    The spec for a processed name, raising if there is none.

:func:`load_initial_conditions`
    Read the processed product and check it against the specs.

:func:`initial_condition_fields`
    One canonical ``(member, site)`` field per spec, optionally for a subset
    of sites.

:func:`read_source_file`, :func:`read_source_directory`
    Parse one source file exactly and run the per-file checks; or
    every file of one site's directory, refusing anything else in it.

:func:`site_member_from_file_name`
    The ``(site, member)`` a source file name encodes.

:func:`build_raw`
    Assemble parsed files into the raw Dataset the conversion writes.

:func:`read_raw`
    Read the converted raw file and check it against the specs.

:func:`build_initial_conditions`
    Turn the raw Dataset into the processed one. Pure; the ingest script wraps
    it with the checks and the write.

:func:`raw_encoding`, :func:`netcdf_encoding`
    The on-disk encodings of the two files.

:func:`describe`
    A spec rendered as a paragraph.

:func:`default_source_root`, :func:`default_raw_dir`, :func:`raw_path`,
:func:`default_product_path`
    Where the source tree, the raw file and the product are expected to
    be, all honoring ``$SIPNET_CALIBRATION_DATA``.

Notes
-----
**One spec, no separate schema.** As ``ConstraintSpec`` does for the
observations, the spec plays the role pySIPNET's ``VariableSpec`` plays for
model output: one flat record per variable from which the product's attributes
are derived. ``sipnet_initial_condition`` names a field of
``pysipnet.parameters.InitialConditions`` and is checked against it at
import, so the two vocabularies cannot drift.

**Why the files are converted and the conversion tracked.** SIPNET never
reads these files; they are a PEcAn intermediate, 816 MB and 800,000 inodes
on the SCC for 32 MB of values, with one file open per ``(site, member)``
cell. The conversion changes structure only -- bit-exact values, the source
names and attribute strings, the source member index -- and the result is
small enough to live in version control, which is the only form in which the
ensemble exists off the SCC.

**Why ``initial_`` names.** The product is the model's starting state --
PEcAn calls the format ``pool_initial_conditions`` -- and the prefix keeps
every name distinct from the constraint products' without inventing a product
prefix. ``biomass`` rather than the file's ``woody`` for the first variable
because the Spawn and Gibbs product is total aboveground biomass carbon.

**Why no state-to-parameter conversion.** Three of the four SIPNET initial
parameters these feed depend on parameters the calibration proposes (the root
fractions for ``plantWoodInit``, the specific leaf weight for ``laiInit``, the
water holding capacity for ``soilWFracInit``), so the conversion is evaluated
per proposed parameter vector in the experiment layer. Each spec records the
formula PEcAn applied in ``pecan_conversion``.

Usage
-----
::

    from sipnet_calibration.initial_conditions import (
        initial_condition_fields, load_initial_conditions, describe,
        resolve_initial_condition,
    )

    ic = load_initial_conditions()                       # Dataset, (member, site)
    ic["initial_soil_organic_carbon"].sel(site=4102)     # one site's 100 members
    ic["initial_wood_carbon"].mean("member")             # a map

    fields = initial_condition_fields(sites=[4102, 4113])
    fields["initial_leaf_carbon"].dims                   # ('member', 'site')

    print(describe(resolve_initial_condition("initial_soil_moisture_saturation")))
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.parameters import InitialConditions
from pysipnet.units import validate_units
from scipy.io import netcdf_file

from sipnet_calibration.sites import DATA_ROOT_ENV_VAR

__all__ = [
    "CF_CONVENTIONS",
    "INITIAL_CONDITIONS",
    "INITIAL_CONDITION_NAMES",
    "InitialConditionSpec",
    "MEMBER",
    "NAME_PATTERN",
    "NOMINAL_DATE",
    "SOURCE_SCRIPT",
    "SOURCE_SCRIPT_NOTE",
    "PRODUCT_FILE",
    "RAW_FILE",
    "SITE",
    "SOURCE_FILE_TEMPLATE",
    "SOURCE_FILL_VALUE",
    "SOURCE_LONG_NAMES",
    "SOURCE_MEMBER",
    "SOURCE_NAMES",
    "SOURCE_TIME_LONG_NAME",
    "SOURCE_TIME_UNITS",
    "SOURCE_TIME_VALUE",
    "SOURCE_UNITS",
    "SourceFile",
    "build_initial_conditions",
    "build_raw",
    "default_product_path",
    "default_raw_dir",
    "default_source_root",
    "describe",
    "initial_condition_fields",
    "load_initial_conditions",
    "netcdf_encoding",
    "raw_encoding",
    "raw_path",
    "read_raw",
    "read_source_directory",
    "read_source_file",
    "resolve_initial_condition",
    "site_member_from_file_name",
]


# ── the spec ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InitialConditionSpec:
    """Everything the rest of the project needs to know about one initial condition
    variable.

    One instance per variable of the source files. :meth:`xarray_attributes`
    is what the processed netCDF stores, so the file describes itself.
    """

    name: str
    """Processed name: the variable's name in the product and the registry key."""

    source_name: str
    """The variable's name in the source files and in the raw file."""

    long_label: str
    """Plot-ready name without units, e.g. ``"Initial leaf carbon"``."""

    units: str
    """UDUNITS-style unit string, physical only, validated by :mod:`pysipnet.units`.
    The unit the values are actually in, which for soil moisture is not the
    string the files carry; see *units_provenance*."""

    constituent: str
    """Substance the unit refers to, ``"C"`` for carbon, or ``""``."""

    description: str
    """What the quantity is and how it was prepared, with the citation."""

    product: str
    """The upstream data product the ensemble was drawn from."""

    sipnet_initial_condition: str
    """The field of ``pysipnet.parameters.InitialConditions`` PEcAn fed this
    variable into, or ``""`` when PEcAn does not use it."""

    pecan_conversion: str
    """In words, the formula PEcAn's ``write.config.SIPNET`` applied to reach
    that parameter, and what it does when the value is unusable."""

    units_provenance: str
    """Where the unit comes from and how firm it is, in a sentence or two."""

    comment: str = ""
    """Anything else a reader must know; the CF ``comment`` attribute."""

    def __post_init__(self) -> None:
        if not NAME_PATTERN.match(self.name):
            raise ValueError(f"Name {self.name!r} is not lower_case_with_underscores.")
        validate_units(self.units)
        if not self.description or not self.long_label or not self.product:
            raise ValueError(f"{self.name!r} needs a description, long_label and product.")
        if not self.pecan_conversion or not self.units_provenance:
            raise ValueError(f"{self.name!r} needs pecan_conversion and units_provenance.")
        if self.source_name not in SOURCE_UNITS:
            raise ValueError(
                f"{self.name!r}: source_name {self.source_name!r} is not a variable the "
                f"source files carry: {sorted(SOURCE_UNITS)}"
            )
        if self.sipnet_initial_condition and (
            self.sipnet_initial_condition not in _SIPNET_INITIAL_CONDITION_FIELDS
        ):
            raise ValueError(
                f"{self.name!r}: {self.sipnet_initial_condition!r} is not a field of "
                f"pysipnet.parameters.InitialConditions: "
                f"{sorted(_SIPNET_INITIAL_CONDITION_FIELDS)}"
            )

    @property
    def source_units(self) -> str:
        """The ``units`` string the source files carry for this variable."""
        return SOURCE_UNITS[self.source_name]

    @property
    def source_long_name(self) -> str:
        """The ``long_name`` string the source files carry for this variable."""
        return SOURCE_LONG_NAMES[self.source_name]

    def xarray_attributes(self) -> dict[str, Any]:
        """Attributes for the variable's array in the processed product.

        Keys follow the Climate and Forecast conventions where one exists
        (``units``, ``long_name``, ``comment``); the rest are spelled out.
        """
        attrs: dict[str, Any] = {
            "units": self.units,
            "long_name": self.long_label,
            "description": self.description,
            "product": self.product,
            "source_name": self.source_name,
            "source_units": self.source_units,
            "source_long_name": self.source_long_name,
            "sipnet_initial_condition": self.sipnet_initial_condition or "none",
            "pecan_conversion": self.pecan_conversion,
            "units_provenance": self.units_provenance,
        }
        if self.constituent:
            attrs["constituent"] = self.constituent
        if self.comment:
            attrs["comment"] = self.comment
        return attrs


#: What a processed name must look like: lower case words joined by underscores.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

#: The ``units`` attribute each variable carries in the source files. The
#: keys are the variables those files can hold; a file with any other variable
#: is refused. These strings are PEcAn's ``standard_vars.csv`` entries and are
#: recorded, not interpreted: soil moisture is a 0-100 percentage despite ``(-)``.
SOURCE_UNITS: dict[str, str] = {
    "AbvGrndWood": "kg C m-2",
    "wood_carbon_content": "kg C m-2",
    "leaf_carbon_content": "kg C m-2",
    "soil_organic_carbon_content": "kg C m-2",
    "SoilMoistFrac": "(-)",
}

#: The ``long_name`` attribute each variable carries in the source files.
SOURCE_LONG_NAMES: dict[str, str] = {
    "AbvGrndWood": "Above ground woody biomass",
    "wood_carbon_content": "Wood Carbon Content",
    "leaf_carbon_content": "Leaf Carbon Content",
    "soil_organic_carbon_content": "Soil Organic Carbon Content by Layer",
    "SoilMoistFrac": "Average Layer Fraction of Saturation",
}

#: The source variable names, in the order the specs and the raw file carry them.
SOURCE_NAMES: tuple[str, ...] = tuple(SOURCE_UNITS)

#: The fill value every source variable declares. None is present in the
#: ensemble, and :func:`read_source_file` refuses a file that holds one.
SOURCE_FILL_VALUE = -999.0

#: What the source's degenerate ``time`` variable carries: an unsubstituted
#: template no calendar library can parse (issue #3), PEcAn's standard long
#: name for the dimension, and the value 1.0. Asserted identical in every file.
SOURCE_TIME_UNITS = "days since [year]-01-01 00:00:00 UTC"
SOURCE_TIME_LONG_NAME = "Time middle averaging period"
SOURCE_TIME_VALUE = 1.0

#: The source file layout under the source root: one directory per site
#: holding one file per member, ``<member>`` being the 1-based member index.
SOURCE_FILE_TEMPLATE = "{site}/IC_site_{site}_{member}.nc"

#: The PEcAn script that draws the ensemble and writes the source files, and
#: the caveat every reader must see beside it.
SOURCE_SCRIPT = (
    "/projectnb/dietzelab/dongchen/anchorSites/IC_prep_anchorSites.R (Dongchen Zhang, "
    "2024-03-27); the same code is modules/assim.sequential/inst/anchor/"
    "IC_prep_anchorSites.Rmd on PEcAn develop"
)
SOURCE_SCRIPT_NOTE = (
    "That script targets the 343 anchor sites. The 8000-site files were written on "
    "2025-07-23 by a run whose script was not found; they match the script's "
    "construction exactly (five variables, wood = biomass - leaf bitwise, soil "
    "moisture in percent), so this is the template for that run, not a confirmed "
    "record. Open question 24 in data/README.md."
)

#: The date the PEcAn script sampled the source products at, from its own code
#: (``time_poimt <- as.Date("2011-07-15")``, the variable name spelled as the
#: script spells it). The files carry no date.
NOMINAL_DATE = "2011-07-15"

#: The sentence every units provenance ends with, because it is true of every one.
_UNCONFIRMED = "Unconfirmed; see data/README.md, open question 24."

_SIPNET_INITIAL_CONDITION_FIELDS: frozenset[str] = frozenset(InitialConditions.model_fields)


# ── the registry ──────────────────────────────────────────────────────────────

INITIAL_CONDITIONS: tuple[InitialConditionSpec, ...] = (
    InitialConditionSpec(
        name="initial_aboveground_biomass_carbon",
        source_name="AbvGrndWood",
        long_label="Initial aboveground biomass carbon",
        units="kg m-2",
        constituent="C",
        description=(
            "Aboveground biomass carbon density at the site from the 300 m Spawn and "
            "Gibbs (2020) map for 2010, sampled by PEcAn Prep_AGB_IC_from_2010_global: "
            "one draw per member from a normal with the pixel's mean and its "
            "uncertainty layer as standard deviation (an uncertainty of 0 replaced by "
            "0.1), negatives set to 0, then Mg ha-1 converted to kg m-2. Carbon by the "
            "product's own definition. Present at every site."
        ),
        product="Spawn and Gibbs (2020), Global Aboveground and Belowground Biomass Carbon "
        "Density Maps for the Year 2010, ORNL DAAC, doi:10.3334/ORNLDAAC/1763",
        sipnet_initial_condition="",
        pecan_conversion=(
            "Not used. PEcAn.data.land::prepare_pools takes the wood pool from "
            "wood_carbon_content, which every file carries, and would use AbvGrndWood "
            "only together with a coarse-root pool, which no file carries."
        ),
        units_provenance=(
            "The target of the PEcAn script's ud_convert(x, 'Mg ha-1', 'kg m-2') and the "
            "files' own units attribute, kg C m-2. " + _UNCONFIRMED
        ),
        comment=(
            "Where leaf_carbon_content is absent, wood_carbon_content is bitwise equal to "
            "this variable; elsewhere it is this variable minus leaf_carbon_content. "
            "Asserted at ingest."
        ),
    ),
    InitialConditionSpec(
        name="initial_wood_carbon",
        source_name="wood_carbon_content",
        long_label="Initial wood carbon",
        units="kg m-2",
        constituent="C",
        description=(
            "The derived wood pool: the member's aboveground biomass carbon "
            "draw minus its leaf carbon draw where a leaf draw exists, and the biomass "
            "draw itself where it does not. Present at every site. Negative wherever "
            "the leaf draw exceeds the biomass draw, which it does at a substantial "
            "share of the members that have leaf carbon; the ingest report counts them."
        ),
        product="Spawn and Gibbs (2020) biomass carbon minus MODIS-derived leaf carbon, "
        "computed by the PEcAn script",
        sipnet_initial_condition="total_wood_carbon",
        pecan_conversion=(
            "plantWoodInit = 1000 x wood_carbon_content / (1 - fineRootFrac - "
            "coarseRootFrac) in PEcAn since commit 913dcec66 (2025-09-02); plantWoodInit "
            "= 1000 x wood_carbon_content before it. Which version ran the 8000-site "
            "reanalysis is not established. A negative value fails prepare_pools' "
            "is.valid and the member keeps the template default."
        ),
        units_provenance=(
            "Inherits the biomass draw's kg C m-2; the leaf draw subtracted from it is "
            "nominally kg C m-2 (see initial_leaf_carbon). " + _UNCONFIRMED
        ),
        comment=(
            "Negative values are written through unchanged and counted in the ingest "
            "report; whether to drop, floor or replace them is the experiment's decision."
        ),
    ),
    InitialConditionSpec(
        name="initial_leaf_carbon",
        source_name="leaf_carbon_content",
        long_label="Initial leaf carbon",
        units="kg m-2",
        constituent="C",
        description=(
            "Leaf carbon from MODIS MCD15A3H leaf area index: the composite nearest "
            "2011-07-15 within 30 days at the site (the same extraction as the "
            "modis_leaf_area_index constraint, rows with sd >= 20 dropped), one normal "
            "draw per member from its LAI and standard deviation, divided by one draw "
            "from the site PFT's 100 specific leaf area samples. Absent at the sites, "
            "mostly high latitude, where no composite passed."
        ),
        product="MODIS MCD15A3H v061 leaf area index via PEcAn MODIS_LAI_prep, and the "
        "PFT specific leaf area samples of the reanalysis (samples.Rdata)",
        sipnet_initial_condition="leaf_area_index",
        pecan_conversion=(
            "laiInit = leaf_carbon_content x SLA, SLA being the run's own specific leaf "
            "area draw, so the round trip to leaf carbon holds only for the same draw; "
            "then laiInit = 0 if the PFT is deciduous (fracLeafFall > 0.5) and the run "
            "starts outside leaf-on. A negative value fails prepare_pools' is.valid and "
            "the member keeps the template default."
        ),
        units_provenance=(
            "LAI (m2 m-2) divided by SLA in m2 per kg leaf mass, so the values are kg "
            "leaf m-2 labeled kg C m-2; the leaf carbon fraction (about 0.48) is not "
            "applied. " + _UNCONFIRMED
        ),
        comment=(
            "Negative at some members, all at grassland sites, matching the negative "
            "values in that PFT's specific leaf area sample. Written through unchanged "
            "and counted in the ingest report."
        ),
    ),
    InitialConditionSpec(
        name="initial_soil_organic_carbon",
        source_name="soil_organic_carbon_content",
        long_label="Initial soil organic carbon",
        units="kg m-2",
        constituent="C",
        description=(
            "Soil organic carbon stock sampled with replacement, one draw per member, "
            "from the 200 ISCN profile stocks PEcAn holds for the site's CEC level-2 "
            "ecoregion (PEcAn.data.land::iscn_soc, from ISCN_ALL_DATA_DATASET_1-1; "
            "ecoregion by point-in-polygon), g cm-2 converted to kg m-2. The "
            "integration depth of the ISCN stocks is not documented. Present at every "
            "site."
        ),
        product="ISCN (International Soil Carbon Network) generation 3 database, by CEC "
        "level-2 ecoregion, via PEcAn IC_ISCN_SOC",
        sipnet_initial_condition="soil_carbon",
        pecan_conversion="soilInit = 1000 x soil_organic_carbon_content.",
        units_provenance=(
            "The target of the PEcAn script's ud_convert(x, 'g cm-2', 'kg m-2') and the "
            "files' own units attribute, kg C m-2. The files' long name says 'by "
            "Layer' but each holds one scalar. " + _UNCONFIRMED
        ),
        comment=(
            "Far fewer distinct values than members, because each site's members are "
            "drawn with replacement from a 200-value pool."
        ),
    ),
    InitialConditionSpec(
        name="initial_soil_moisture_saturation",
        source_name="SoilMoistFrac",
        long_label="Initial surface soil moisture, percent of saturation",
        units="percent",
        constituent="",
        description=(
            "Surface (2-5 cm) soil moisture as percent of saturation from the C3S/ESA "
            "CCI active-microwave climate data record (v202212, daily, 0.25 degree): the "
            "first day with a retrieval from 2011-07-15 forward within 30 days, one "
            "normal draw per member from the retrieval and its uncertainty, negatives "
            "set to 0, by PEcAn extract_SM_CDS. Absent at the sites, mostly high "
            "latitude or densely vegetated, where the record has no retrieval."
        ),
        product="Copernicus C3S / ESA CCI Soil moisture gridded data from 1978 to present, "
        "active sensor, CDR v202212, doi:10.24381/cds.d7782f18",
        sipnet_initial_condition="soil_wetness_fraction",
        pecan_conversion=(
            "soilWFracInit = SoilMoistFrac / 100. SIPNET defines soilWFracInit as a "
            "fraction of the water holding capacity of its whole soil bucket, so the "
            "mapping equates two different fractions."
        ),
        units_provenance=(
            "The CCI variable's own attributes (units 'percent', long name 'Percent of "
            "Saturation Soil Moisture', valid range 0-100) and the values, which run 0 "
            "to 100 with a median of 60. The files' units attribute is PEcAn's "
            "standard_vars string '(-)'. " + _UNCONFIRMED
        ),
    ),
)

#: The processed names, in registry order.
INITIAL_CONDITION_NAMES: tuple[str, ...] = tuple(spec.name for spec in INITIAL_CONDITIONS)


def resolve_initial_condition(name: str) -> InitialConditionSpec:
    """The spec named *name*, or a ``KeyError`` listing the names that exist."""
    for spec in INITIAL_CONDITIONS:
        if spec.name == name:
            return spec
    raise KeyError(f"No initial condition named {name!r}. Known: {list(INITIAL_CONDITION_NAMES)}")


# ── the files ─────────────────────────────────────────────────────────────────

#: The converted raw file and the processed product, under ``data/``.
RAW_FILE = "pecan_pool_initial_conditions.nc"
PRODUCT_FILE = "initial_conditions.nc"

#: Dimension and coordinate names.
SITE = "site"
MEMBER = "member"
SOURCE_MEMBER = "source_member"

#: The metadata conventions the product follows, as the constraint products do.
CF_CONVENTIONS = "CF-1.11"


@dataclass(frozen=True)
class SourceFile:
    """One source file, parsed.

    Attributes
    ----------
    site, member:
        The numbers the file name encodes; ``member`` is the source's 1-based
        index.
    values:
        Source variable name to the file's one value, for the variables the
        file carries. Every value is finite and not the fill.
    """

    site: int
    member: int
    values: Mapping[str, float]


def default_source_root() -> Path:
    """Where the source file tree is expected: ``data/raw/initial_conditions/files``.

    Present only on the SCC, as a symlink. ``$SIPNET_CALIBRATION_DATA``
    replaces ``data/`` when set.
    """
    return default_raw_dir() / "files"


def default_raw_dir() -> Path:
    """Where the converted raw file lives: ``data/raw/initial_conditions/``."""
    return _data_root() / "raw" / "initial_conditions"


def raw_path(directory: Path | str | None = None) -> Path:
    """The converted raw file: ``<directory>/pecan_pool_initial_conditions.nc``."""
    base = Path(directory) if directory is not None else default_raw_dir()
    return base / RAW_FILE


def default_product_path() -> Path:
    """Where the processed product is expected: ``data/processed/initial_conditions.nc``."""
    return _data_root() / "processed" / PRODUCT_FILE


def site_member_from_file_name(name: str) -> tuple[int, int] | None:
    """The ``(site, member)`` a file name encodes, or ``None`` if it is not one.

    Only a name that is exactly ``IC_site_<site>_<member>.nc`` with plain
    positive integers counts; ``IC_site_01_5.nc`` does not, since it would not
    round-trip through :data:`SOURCE_FILE_TEMPLATE`.
    """
    match = _SOURCE_FILE_NAME.match(name)
    if match is None:
        return None
    return int(match.group("site")), int(match.group("member"))


def read_source_file(path: Path | str) -> SourceFile:
    """Parse one source file exactly and run the per-file checks.

    Parameters
    ----------
    path:
        The file, named as :data:`SOURCE_FILE_TEMPLATE` within its site
        directory.

    Returns
    -------
    SourceFile
        The file's variables under their source names.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the name is not the template or disagrees with its directory; the
        file is not netCDF-3 classic; it has a global attribute, a dimension
        other than an unlimited length-1 ``time``, or a ``time`` variable
        whose attributes or value differ from the source template; a data
        variable is not a scalar ``float64`` on ``("time",)``, is not one of
        :data:`SOURCE_NAMES`, or carries attributes other than exactly the
        expected ``_FillValue``, ``long_name`` and ``units``; a value is the
        fill or not finite; or the file carries no data variable.

    Notes
    -----
    The checks live here so that a file is checked wherever it is parsed and
    the conversion applies exactly one set of rules to all 800,000. Read with
    ``scipy.io.netcdf_file``, which reads netCDF-3 directly and needs no
    library beyond the project's; the files cannot be opened by ``h5netcdf``
    at all, and the ``time`` units are undecodable, so the generic xarray path
    would need two workarounds for nothing the parser wants.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such initial condition file: {path}")
    parsed = site_member_from_file_name(path.name)
    if parsed is None:
        raise ValueError(f"{path}: name is not IC_site_<site>_<member>.nc")
    site, member = parsed
    if path.parent.name != str(site):
        raise ValueError(
            f"{path}: the file name says site {site} but the directory is "
            f"{path.parent.name!r}; the layout is {SOURCE_FILE_TEMPLATE}"
        )
    try:
        handle = netcdf_file(str(path), "r", mmap=False, maskandscale=False)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"{path}: not readable as netCDF-3 classic ({error})") from error
    with handle:
        _check_source_file_is_classic_with_no_global_attributes(handle, path)
        _check_source_time_is_the_degenerate_template(handle, path)
        values = _check_and_read_source_variables(handle, path)
    return SourceFile(site=site, member=member, values=values)


def read_source_directory(root: Path | str, site: int) -> list[SourceFile]:
    """Parse every file of one site's directory, refusing anything else in it.

    Parameters
    ----------
    root:
        The source tree.
    site:
        The site whose directory ``<root>/<site>`` to read.

    Returns
    -------
    list of SourceFile
        One record per file, in file-name order.

    Raises
    ------
    ValueError
        If the directory is missing or empty, or holds an entry that is not
        an ``IC_site_<site>_<member>.nc`` file (hidden files such as
        ``.DS_Store`` are skipped as filesystem debris), plus whatever
        :func:`read_source_file` raises for a file.

    Notes
    -----
    The unit of parallel work in ``scripts/convert_initial_conditions.py``,
    which is why it lives here: a worker process has to be able to import it.
    """
    directory = Path(root) / str(int(site))
    if not directory.is_dir():
        raise ValueError(f"{directory}: no such site directory")
    records = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        if site_member_from_file_name(path.name) is None:
            raise ValueError(
                f"{path}: not an IC_site_<site>_<member>.nc file; a source site "
                "directory holds nothing else"
            )
        records.append(read_source_file(path))
    if not records:
        raise ValueError(f"{directory}: holds no files")
    return records


def build_raw(
    files: Iterable[SourceFile],
    *,
    source_root: str,
    conversion_script: str,
) -> xr.Dataset:
    """Assemble parsed source files into the raw Dataset the conversion writes.

    Parameters
    ----------
    files:
        Every parsed file of the tree, in any order.
    source_root:
        The directory they were read from, for the attributes.
    conversion_script:
        The script doing the conversion, for the attributes.

    Returns
    -------
    xarray.Dataset
        One ``float64`` variable per :data:`SOURCE_NAMES` entry on
        ``(site, member)``, in source names with the source ``units`` and
        ``long_name`` strings, ``NaN`` where a file lacks the variable;
        ``site`` ``int32`` and ``member`` ``int16`` (the 1-based source index),
        both ascending; the source's time metadata and the counts as
        attributes.

    Raises
    ------
    ValueError
        If two files claim the same ``(site, member)``, if the member set
        differs between sites, if a variable's presence differs between the
        members of one site, or if no file was given.

    Notes
    -----
    Pure: it neither reads nor writes files, so the conversion script and the
    tests call it on the same records. The guards here are the ones that a
    fancy-indexed assignment would otherwise turn into a silent overwrite.
    """
    records = list(files)
    if not records:
        raise ValueError("no source files to assemble")
    sites = np.array(sorted({record.site for record in records}), dtype=np.int64)
    members = np.array(sorted({record.member for record in records}), dtype=np.int64)
    site_index = {int(site): i for i, site in enumerate(sites)}
    member_index = {int(member): j for j, member in enumerate(members)}

    shape = (sites.size, members.size)
    seen = np.zeros(shape, dtype=bool)
    arrays = {name: np.full(shape, np.nan) for name in SOURCE_NAMES}
    for record in records:
        i, j = site_index[record.site], member_index[record.member]
        if seen[i, j]:
            raise ValueError(f"two files for site {record.site} member {record.member}")
        seen[i, j] = True
        for name, value in record.values.items():
            if name not in arrays:
                raise ValueError(
                    f"site {record.site} member {record.member}: variable {name!r} is not "
                    f"one of {list(SOURCE_NAMES)}"
                )
            arrays[name][i, j] = value
    _check_every_site_has_every_member(seen, sites, members)
    _check_presence_is_uniform_over_members(arrays, sites)

    _check_site_ids_fit_dtype(sites, np.int32, "site")
    _check_site_ids_fit_dtype(members, np.int16, "member")
    dataset = xr.Dataset(
        {
            name: (
                (SITE, MEMBER),
                arrays[name],
                {
                    "units": SOURCE_UNITS[name],
                    "long_name": SOURCE_LONG_NAMES[name],
                    "source_fill_value": SOURCE_FILL_VALUE,
                    "comment": (
                        "The source file's value, bit for bit; NaN where none of the site's "
                        "files carries the variable."
                    ),
                },
            )
            for name in SOURCE_NAMES
        },
        coords={
            SITE: (SITE, sites.astype(np.int32), _SITE_ATTRS),
            MEMBER: (
                MEMBER,
                members.astype(np.int16),
                {
                    "long_name": "Ensemble member index in the source file name",
                    "comment": "The 1-based <member> of the source file name.",
                },
            ),
        },
        attrs={
            "title": "PEcAn pool initial conditions for the 8000-site pool, as one array",
            "source_root": source_root,
            "source_layout": SOURCE_FILE_TEMPLATE,
            "source_format": "NETCDF3_CLASSIC, one file per (site, member)",
            "source_fill_value": SOURCE_FILL_VALUE,
            "source_time_units": SOURCE_TIME_UNITS,
            "source_time_long_name": SOURCE_TIME_LONG_NAME,
            "source_time_value": SOURCE_TIME_VALUE,
            "n_source_files": len(records),
            "n_sites": int(sites.size),
            "n_members": int(members.size),
            "conversion_script": conversion_script,
            "history": (
                f"{conversion_script}: read every {SOURCE_FILE_TEMPLATE} under the source "
                "root, checked each against the source template, and laid the values on "
                "(site, member) unchanged, in the source files' names and units strings"
            ),
            "converted": _now(),
        },
    )
    return dataset


def raw_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """The on-disk encoding of the raw file: compressed, ``NaN`` as the fill."""
    encoding: dict[str, dict[str, Any]] = {
        name: {"zlib": True, "complevel": 4, "_FillValue": np.nan} for name in SOURCE_NAMES
    }
    for name in dataset.coords:
        encoding[str(name)] = {"_FillValue": None}
    return encoding


def read_raw(path: Path | str | None = None) -> xr.Dataset:
    """Read the converted raw file and check it against the specs.

    Parameters
    ----------
    path:
        The netCDF to read. Defaults to :func:`raw_path`.

    Returns
    -------
    xarray.Dataset
        As :func:`build_raw` returns it.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with where it comes from.
    ValueError
        If the variables are not exactly :data:`SOURCE_NAMES` on
        ``(site, member)`` with the source attribute strings, the coordinates
        are not ascending, a value is non-finite without being ``NaN``, or a
        variable is present for some members of a site and not others.
    """
    path = Path(path) if path is not None else raw_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file. The raw file is tracked in version control; if it "
            "is missing from a checkout, regenerate it on the SCC with "
            "scripts/raw_sources/convert_initial_conditions.py (see "
            "data/raw/initial_conditions/provenance.md)."
        )
    dataset = xr.open_dataset(path, engine="h5netcdf")
    try:
        _check_raw(dataset, path)
    except Exception:
        dataset.close()
        raise
    return dataset


def build_initial_conditions(raw: xr.Dataset, sites: pd.DataFrame) -> xr.Dataset:
    """Turn the raw Dataset into the processed product the data model describes.

    Parameters
    ----------
    raw:
        As :func:`read_raw` returns it.
    sites:
        The site table from :func:`sipnet_calibration.sites.load_sites`; its
        ``site_id`` is the pool and its ``lon``/``lat`` the coordinates.

    Returns
    -------
    xarray.Dataset
        The five specs' variables on ``(member, site)`` with their attributes,
        ``member`` 0-based with ``source_member`` beside it, ``site`` the pool
        with ``lon``/``lat``, and the dataset attributes of the data model.

    Raises
    ------
    ValueError
        If the raw file's sites are not exactly the site table's pool.

    Notes
    -----
    Pure, and structural only: the values are the raw file's, transposed. The
    member axis is renumbered from the source files' 1-based index to the
    project's 0-based one, and the source index is kept as a coordinate so a
    source file name can always be recovered.
    """
    pool = np.sort(sites["site_id"].to_numpy(np.int64))
    raw_sites = raw[SITE].values.astype(np.int64)
    if not np.array_equal(raw_sites, pool):
        extra = sorted(set(raw_sites.tolist()) - set(pool.tolist()))[:10]
        missing = sorted(set(pool.tolist()) - set(raw_sites.tolist()))[:10]
        raise ValueError(
            f"raw file sites are not the site table's pool: not in the table {extra}, "
            f"not in the file {missing}"
        )
    coordinates = sites.set_index("site_id").loc[pool, ["lon", "lat"]]
    source_member = raw[MEMBER].values.astype(np.int16)

    data_vars = {
        spec.name: (
            (MEMBER, SITE),
            np.ascontiguousarray(raw[spec.source_name].values.T),
            spec.xarray_attributes(),
        )
        for spec in INITIAL_CONDITIONS
    }
    coords = {
        MEMBER: (
            MEMBER,
            np.arange(source_member.size, dtype=np.int16),
            {
                "long_name": "Ensemble member",
                "comment": (
                    "0-based, meaningful only within this product; whether member i "
                    "corresponds to member i of another source is not established "
                    "(member_correspondence)."
                ),
            },
        ),
        SOURCE_MEMBER: (
            MEMBER,
            source_member,
            {
                "long_name": "Ensemble member index in the source file name",
                "comment": "The 1-based <member> of the source file name.",
            },
        ),
        SITE: (SITE, pool.astype(np.int32), _SITE_ATTRS),
        "lon": (SITE, coordinates["lon"].to_numpy(np.float64), _LON_ATTRS),
        "lat": (SITE, coordinates["lat"].to_numpy(np.float64), _LAT_ATTRS),
    }
    return xr.Dataset(data_vars, coords=coords, attrs=_product_attributes(raw))


def netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """The on-disk encoding of the product: compressed, ``NaN`` as the fill,
    no ``_FillValue`` on any coordinate, as CF requires."""
    encoding: dict[str, dict[str, Any]] = {
        name: {"zlib": True, "complevel": 4, "_FillValue": np.nan}
        for name in INITIAL_CONDITION_NAMES
    }
    for name in dataset.coords:
        encoding[str(name)] = {"_FillValue": None}
    return encoding


def load_initial_conditions(path: Path | str | None = None) -> xr.Dataset:
    """Read the processed product and check it against the specs.

    Parameters
    ----------
    path:
        The netCDF to read. Defaults to :func:`default_product_path`.

    Returns
    -------
    xarray.Dataset
        The data model above.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with the command that produces it.
    ValueError
        If the file does not match the data model.
    """
    path = Path(path) if path is not None else default_product_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file. Produce it with:\n  python scripts/ingest_initial_conditions.py"
        )
    dataset = xr.open_dataset(path, engine="h5netcdf")
    try:
        _check_product(dataset, path)
    except Exception:
        dataset.close()
        raise
    return dataset


def initial_condition_fields(
    names: Sequence[str] | None = None,
    *,
    sites: Iterable[int] | None = None,
    path: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The initial conditions as canonical fields, one per variable.

    Parameters
    ----------
    names:
        Processed names, in the order the result should carry them. Defaults
        to every one of :data:`INITIAL_CONDITION_NAMES`.
    sites:
        Site ids to keep, in the order given. Defaults to the whole pool.
    path:
        The product to read. Defaults to :func:`default_product_path`.

    Returns
    -------
    dict
        Name to its ``(member, site)`` array, with the variable's attributes
        and ``lon``/``lat`` on ``site``.

    Raises
    ------
    ValueError
        If a requested site is not in the pool.
    """
    if isinstance(names, str):
        names = [names]
    wanted_names = list(names) if names is not None else list(INITIAL_CONDITION_NAMES)
    for name in wanted_names:
        resolve_initial_condition(name)
    if isinstance(sites, (int, np.integer)):
        sites = [sites]
    wanted_sites = None if sites is None else [int(site) for site in sites]

    dataset = load_initial_conditions(path)
    if wanted_sites is not None:
        missing = sorted(set(wanted_sites) - set(dataset[SITE].values.tolist()))
        if missing:
            raise ValueError(f"sites not in the pool: {missing[:10]}")
        dataset = dataset.sel({SITE: wanted_sites})
    return {name: dataset[name] for name in wanted_names}


def describe(spec: InitialConditionSpec) -> str:
    """A spec as a paragraph, for ``--describe`` and the run log."""
    units = f"{spec.units} {spec.constituent}".strip()
    lines = [
        f"{spec.name}: {spec.long_label} ({units}), from {spec.product}.",
        f"  source     {spec.source_name!r}, units {spec.source_units!r}, "
        f"long name {spec.source_long_name!r}",
        f"  sipnet     {spec.sipnet_initial_condition or 'none'}: {spec.pecan_conversion}",
        f"  units      {spec.units_provenance}",
        f"  what       {spec.description}",
    ]
    if spec.comment:
        lines.append(f"  comment    {spec.comment}")
    return "\n".join(lines)


# ── supporting helpers ────────────────────────────────────────────────────────

_SOURCE_FILE_NAME = re.compile(r"^IC_site_(?P<site>[1-9]\d*)_(?P<member>[1-9]\d*)\.nc$")

_SITE_ATTRS = {
    "long_name": "Model site identifier",
    "comment": "The handed-down 1-8000 identifier of the site table; never renumbered.",
}
_LON_ATTRS = {"standard_name": "longitude", "long_name": "Longitude", "units": "degrees_east"}
_LAT_ATTRS = {"standard_name": "latitude", "long_name": "Latitude", "units": "degrees_north"}

#: Attribute names each source data variable must carry, exactly.
_SOURCE_VARIABLE_ATTRIBUTES = frozenset({"_FillValue", "long_name", "units"})


def _data_root() -> Path:
    root = os.environ.get(DATA_ROOT_ENV_VAR)
    return Path(root) if root else Path(__file__).resolve().parents[2] / "data"


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _decode(value: Any) -> Any:
    """A netCDF-3 attribute as a Python string or float."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return float(value.ravel()[0]) if value.size == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attributes(obj: Any) -> dict[str, Any]:
    """The attributes of a ``scipy.io.netcdf_file`` or one of its variables."""
    return {str(key): _decode(value) for key, value in obj._attributes.items()}


def _product_attributes(raw: xr.Dataset) -> dict[str, Any]:
    return {
        "Conventions": CF_CONVENTIONS,
        "title": "Initial condition ensemble for the 8000-site pool",
        "product": (
            "PEcAn pool initial conditions drawn for the North American reanalysis; one "
            "spec per variable in sipnet_calibration.initial_conditions"
        ),
        "source_file": RAW_FILE,
        "source_root": str(raw.attrs.get("source_root", "")),
        "source_script": SOURCE_SCRIPT,
        "source_script_note": SOURCE_SCRIPT_NOTE,
        "nominal_date": NOMINAL_DATE,
        "nominal_date_provenance": (
            "The sampling date in the PEcAn script; the source files themselves carry no "
            "date. Biomass is a 2010 annual map and soil carbon is undated."
        ),
        "source_time_units": SOURCE_TIME_UNITS,
        "source_time_long_name": SOURCE_TIME_LONG_NAME,
        "source_time_value": SOURCE_TIME_VALUE,
        "member_source": "ic",
        "member_correspondence": (
            "Not established: whether member i here corresponds to member i of the "
            "drivers or of any other ensemble is unknown, and xarray aligns integer "
            "member labels silently."
        ),
        "n_sites": int(raw.sizes[SITE]),
        "n_members": int(raw.sizes[MEMBER]),
        "history": (
            f"scripts/ingest_initial_conditions.py: read {RAW_FILE}, renamed the source "
            "variables to the spec names, renumbered member from 1-based to 0-based "
            "keeping the source index as source_member, placed the sites on the site "
            "table's pool with lon/lat, and wrote the spec fields as attributes; values "
            "unchanged"
        ),
        "created": _now(),
    }


# ── checks ────────────────────────────────────────────────────────────────────


def _check_source_file_is_classic_with_no_global_attributes(handle: Any, path: Path) -> None:
    if handle.version_byte != 1:
        raise ValueError(
            f"{path}: netCDF-3 version byte is {handle.version_byte}, expected 1 (classic)"
        )
    attrs = _attributes(handle)
    if attrs:
        raise ValueError(
            f"{path}: carries global attributes {sorted(attrs)}; source files carry none"
        )
    if set(handle.dimensions) != {"time"}:
        raise ValueError(
            f"{path}: dimensions are {sorted(handle.dimensions)}, expected exactly ['time']"
        )
    if handle.dimensions["time"] is not None:
        raise ValueError(f"{path}: the time dimension is not the unlimited record dimension")
    if handle._recs != 1:
        raise ValueError(f"{path}: time has {handle._recs} records, expected 1")


def _check_source_time_is_the_degenerate_template(handle: Any, path: Path) -> None:
    if "time" not in handle.variables:
        raise ValueError(f"{path}: has no time variable")
    time = handle.variables["time"]
    attrs = _attributes(time)
    expected = {"units": SOURCE_TIME_UNITS, "long_name": SOURCE_TIME_LONG_NAME}
    if attrs != expected:
        raise ValueError(
            f"{path}: time attributes are {attrs}, expected {expected}. A substituted year "
            "would mean issue #3 was fixed upstream; notice it rather than average it away."
        )
    value = np.asarray(time.data, dtype=np.float64).ravel()
    if value.size != 1 or value[0] != SOURCE_TIME_VALUE:
        raise ValueError(f"{path}: time value is {value.tolist()}, expected [{SOURCE_TIME_VALUE}]")


def _check_and_read_source_variables(handle: Any, path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, variable in handle.variables.items():
        if name == "time":
            continue
        if name not in SOURCE_UNITS:
            raise ValueError(
                f"{path}: variable {name!r} is not one the source files carry "
                f"({sorted(SOURCE_UNITS)}). A new variable is a spec change, not a new column."
            )
        if variable.dimensions != ("time",):
            raise ValueError(
                f"{path}: {name} has dims {variable.dimensions}, expected ('time',); a "
                "layer-resolved variable would look like this"
            )
        if variable.data.dtype.newbyteorder("=") != np.dtype(np.float64):
            raise ValueError(f"{path}: {name} is {variable.data.dtype}, expected float64")
        attrs = _attributes(variable)
        expected = {
            "_FillValue": SOURCE_FILL_VALUE,
            "long_name": SOURCE_LONG_NAMES[name],
            "units": SOURCE_UNITS[name],
        }
        if attrs != expected:
            raise ValueError(
                f"{path}: {name} attributes are {attrs}, expected exactly {expected}. An "
                "unexpected scale_factor or add_offset would silently rescale the value."
            )
        data = np.asarray(variable.data, dtype=np.float64).ravel()
        if data.size != 1:
            raise ValueError(f"{path}: {name} holds {data.size} values, expected 1")
        value = float(data[0])
        if value == SOURCE_FILL_VALUE:
            raise ValueError(
                f"{path}: {name} holds the fill value {SOURCE_FILL_VALUE}. No file in the "
                "ensemble does, and the raw file has no representation for an explicit "
                "fill distinct from an absent variable."
            )
        if not np.isfinite(value):
            raise ValueError(f"{path}: {name} holds the non-finite value {value!r}")
        values[name] = value
    if not values:
        raise ValueError(f"{path}: carries no data variable")
    return values


def _check_every_site_has_every_member(seen: np.ndarray, sites: np.ndarray, members: np.ndarray) -> None:
    if seen.all():
        return
    gaps = np.argwhere(~seen)
    examples = [f"site {sites[i]} member {members[j]}" for i, j in gaps[:5]]
    raise ValueError(
        f"{gaps.shape[0]} (site, member) pairs have no file, for example {examples}. The "
        "ensemble is a complete rectangle; a gap means the tree is incomplete."
    )


def _check_presence_is_uniform_over_members(arrays: Mapping[str, np.ndarray], sites: np.ndarray) -> None:
    for name, array in arrays.items():
        present = np.isfinite(array)
        mixed = present.any(axis=1) & ~present.all(axis=1)
        if mixed.any():
            raise ValueError(
                f"{name}: present for some members and absent for others at "
                f"{int(mixed.sum())} sites, for example {sites[mixed][:5].tolist()}. "
                "Presence is a property of the site in this ensemble."
            )


def _check_site_ids_fit_dtype(values: np.ndarray, dtype: type, what: str) -> None:
    info = np.iinfo(dtype)
    if values.min() < max(info.min, 1) or values.max() > info.max:
        raise ValueError(f"{what} values {values.min()}-{values.max()} do not fit {dtype.__name__}")


def _check_raw(dataset: xr.Dataset, path: Path) -> None:
    """Raise unless *dataset* is the raw file :func:`build_raw` describes."""
    if set(dataset.data_vars) != set(SOURCE_NAMES):
        raise ValueError(
            f"{path}: variables are {sorted(dataset.data_vars)}, expected {sorted(SOURCE_NAMES)}"
        )
    for name in SOURCE_NAMES:
        array = dataset[name]
        if array.dims != (SITE, MEMBER):
            raise ValueError(f"{path}: {name} has dims {array.dims}, expected ('site', 'member')")
        if array.dtype != np.float64:
            raise ValueError(f"{path}: {name} is {array.dtype}, expected float64")
        for key, table in (("units", SOURCE_UNITS), ("long_name", SOURCE_LONG_NAMES)):
            if array.attrs.get(key) != table[name]:
                raise ValueError(
                    f"{path}: {name} has {key} {array.attrs.get(key)!r}, the source string is "
                    f"{table[name]!r}"
                )
        values = array.values
        if np.isinf(values).any():
            raise ValueError(f"{path}: {name} holds an infinite value")
    for coordinate, dtype in ((SITE, np.int32), (MEMBER, np.int16)):
        values = dataset[coordinate].values
        if values.size == 0 or np.any(np.diff(values) <= 0):
            raise ValueError(f"{path}: {coordinate} is empty or not strictly ascending")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{path}: {coordinate} is {values.dtype}, expected an integer type")
        # The product narrows these with astype, which wraps silently.
        _check_site_ids_fit_dtype(values.astype(np.int64), dtype, f"{path}: {coordinate}")
    _check_presence_is_uniform_over_members(
        {name: dataset[name].values for name in SOURCE_NAMES}, dataset[SITE].values
    )
    for key in ("source_time_units", "source_time_long_name", "source_time_value", "n_source_files"):
        if key not in dataset.attrs:
            raise ValueError(f"{path}: missing the {key!r} attribute")


def _check_product(dataset: xr.Dataset, path: Path) -> None:
    """Raise unless *dataset* is the product the data model describes."""
    if set(dataset.data_vars) != set(INITIAL_CONDITION_NAMES):
        raise ValueError(
            f"{path}: variables are {sorted(dataset.data_vars)}, expected "
            f"{sorted(INITIAL_CONDITION_NAMES)}"
        )
    for spec in INITIAL_CONDITIONS:
        array = dataset[spec.name]
        if array.dims != (MEMBER, SITE):
            raise ValueError(f"{path}: {spec.name} has dims {array.dims}, expected ('member', 'site')")
        if array.attrs.get("units") != spec.units:
            raise ValueError(
                f"{path}: {spec.name} has units {array.attrs.get('units')!r}, the spec says "
                f"{spec.units!r}"
            )
        if array.attrs.get("source_name") != spec.source_name:
            raise ValueError(
                f"{path}: {spec.name} was written from {array.attrs.get('source_name')!r}, "
                f"the spec says {spec.source_name!r}"
            )
        if array.attrs.get("long_name") != spec.long_label:
            raise ValueError(f"{path}: {spec.name} lacks the spec's long_name")
        if np.isinf(array.values).any():
            raise ValueError(f"{path}: {spec.name} holds an infinite value")
    for coordinate in (MEMBER, SOURCE_MEMBER, SITE, "lon", "lat"):
        if coordinate not in dataset.coords:
            raise ValueError(f"{path}: missing the {coordinate!r} coordinate")
    for coordinate in ("lon", "lat"):
        if dataset[coordinate].dims != (SITE,):
            raise ValueError(f"{path}: {coordinate} must be on site, has dims {dataset[coordinate].dims}")
    if dataset[SOURCE_MEMBER].dims != (MEMBER,):
        raise ValueError(f"{path}: source_member must be on member")
    member = dataset[MEMBER].values
    if member.size == 0 or not np.array_equal(member, np.arange(member.size)):
        raise ValueError(f"{path}: member is not 0..n-1")
    source_member = dataset[SOURCE_MEMBER].values
    if source_member.min() < 1 or np.any(np.diff(source_member) <= 0):
        raise ValueError(
            f"{path}: source_member is not strictly ascending from 1 or more; a source "
            "file name could not be recovered from it"
        )
    site = dataset[SITE].values
    if site.size == 0 or np.any(np.diff(site) <= 0):
        raise ValueError(f"{path}: site is empty or not strictly ascending")
    lon, lat = dataset["lon"].values, dataset["lat"].values
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError(f"{path}: lon or lat holds a non-finite value")
    if np.abs(lon).max() > 180 or np.abs(lat).max() > 90:
        raise ValueError(f"{path}: lon or lat is outside the geographic range; are they swapped?")
    _check_presence_is_uniform_over_members(
        {name: dataset[name].values.T for name in INITIAL_CONDITION_NAMES}, site
    )
    if dataset.attrs.get("Conventions") != CF_CONVENTIONS:
        raise ValueError(f"{path}: Conventions is {dataset.attrs.get('Conventions')!r}, expected {CF_CONVENTIONS!r}")
