"""The data model for the initial condition ensemble, and its reader.

Overview
--------
This module defines how the initial condition ensemble is represented -- its
dimensions, coordinates, variable names, units and dtype -- and provides the
functions that read it. It is the single description of that layout: the ingest
script, the tests, the plotting layer and the experiment layer's
state-to-parameter conversion all get the schema from here rather than
restating it.

Unlike the drivers, the initial conditions get an **ingested product**; the
Notes say why. The dependency runs one way::

    raw/initial_conditions/<site>/IC_site_<site>_<member>.nc
      -> scripts/ingest_ic.py -> processed/ic.nc
      -> this module            load_initial_conditions() -> xarray.Dataset

with ``processed/sites/sites.csv`` joined on for the site coordinates.
``data/README.md`` documents the source data and the open questions about it.

Input data
----------
``data/processed/ic.nc``
    The product :func:`load_initial_conditions` reads, written by
    ``scripts/ingest_ic.py``. :func:`default_ic_path` says where it is expected
    to be.

``data/raw/initial_conditions/``
    The source, read by the ingest script through :func:`read_ic_file`, which
    lives here so that the file-format checks travel with the schema.
    :func:`default_ic_root` says where it is expected to be.

    One netCDF-3 classic file per site and ensemble member, laid out as
    ``<site>/IC_site_<site>_<member>.nc``, where ``<site>`` is the 1-8000 site
    identifier and ``<member>`` the source's 1-based member index. Each file
    holds a single unlimited ``time`` dimension of length 1 and one ``float64``
    scalar variable on ``("time",)`` per initial condition field, each
    declaring ``_FillValue = -999.0`` and a ``units`` attribute. The ``time``
    coordinate is undecodable (issue #3), so the files must be opened with
    ``decode_times=False``; they are netCDF-3, so they must be opened with
    ``engine="scipy"`` -- ``h5netcdf``, which this project writes its own
    products with, cannot open them at all.

Data model
----------
:func:`load_initial_conditions` returns an ``xarray.Dataset`` shaped as
follows.

**Dimensions**: ``member``, ``site``, and ``variable`` for the presence
companion only.

**Data variables**, one per initial condition field, all ``float64`` on
``(member, site)``, named as :data:`IC_VARIABLES`:

=================================== =============================== ========
Processed name                      Source variable                 Units
=================================== =============================== ========
``initial_aboveground_wood_carbon`` ``AbvGrndWood``                 kg C m-2
``initial_wood_carbon``             ``wood_carbon_content``         kg C m-2
``initial_soil_organic_carbon``     ``soil_organic_carbon_content`` kg C m-2
=================================== =============================== ========

Each carries ``units``, ``long_name``, ``source_name``, ``source_long_name``
and ``aggregation`` from :data:`IC_VARIABLE_ATTRS`, plus ``units_status`` and
``units_provenance``: the units are the ``units`` attribute the source files
carry, asserted identical across every file read, and nothing more than that.
Each also carries ``n_explicit_fills`` and ``n_values_not_positive``, counted
rather than altered.

Two variables carry ``related_constraint_variable``,
``related_constraint_unit_factor``, ``related_constraint_status`` and
``related_constraint_note`` from
:data:`RELATED_CONSTRAINT_VARIABLES`, because a reader will want to compare
them against ``constraints_annual.nc`` and the two products are neither in the
same units nor established to be the same quantity. See `Notes`_.

**Presence companions**, both ``bool``, written unconditionally::

    ic_present(member, site)                  a file existed for the pair
    variable_present(member, site, variable)  the file carried the variable

**Coordinates**

======================= ============== ========================================
Name                    Dims           Meaning
======================= ============== ========================================
``member``              ``member``     0-based ``int16``, in ascending order
``source_member_index`` ``member``     1-based ``int16``, the file name's index
``site``                ``site``       handed-down ``int32`` site id, ascending
``lon``, ``lat``        ``site``       from the site table, ``float64``
``variable``            ``variable``   the :data:`IC_VARIABLES` names, in order
======================= ============== ========================================

**No time dimension.** The source's ``time`` is length 1, is the unlimited
record dimension, and its ``units`` attribute is an unsubstituted template, so
it is dropped. What the source claimed is kept verbatim in the dataset
attributes ``source_time_units``, ``source_time_long_name`` and
``source_time_value``, beside ``time_status`` and ``time_note``. See `Notes`_
for what that costs.

**Attributes** on the dataset: ``title``, ``source_root``, ``source_layout``,
``source_fill_value``, ``history``, ``member_source = "ic"``,
``member_correspondence``, ``n_sites``, ``n_members``, ``coverage``
(``"complete"`` or ``"gaps"``), a ``coverage_note`` when ``coverage`` is
``"gaps"``, and the five time attributes above.

**Missing values.** Every kind of absence is ``NaN`` in a data variable, and
the two presence companions are what tell them apart:

======================================== ======= ============== ====================
Kind                                     Value   ``ic_present`` ``variable_present``
======================================== ======= ============== ====================
No file for the ``(site, member)`` pair  ``NaN`` ``False``      ``False``
File exists, does not carry the variable ``NaN`` ``True``       ``False``
File carries an explicit ``-999.0`` fill ``NaN`` ``True``       ``True``
======================================== ======= ============== ====================

So ``variable_present & isnan(value)`` **is** the explicit-fill indicator, and
that identity is what :func:`load_initial_conditions` checks.

Functions
---------
:func:`load_initial_conditions`
    Read the product and check it against the data model above. Raises rather
    than returning something subtly wrong.

:func:`initial_condition_fields`
    Split the Dataset into canonical fields -- one ``DataArray`` per variable
    with dims ``(member, site)`` and its own units. This is the view the
    plotting layer wants, and it leaves the presence companions out.

:func:`read_ic_file`
    Parse one raw file exactly, in source variable names, and run the per-file
    checks. The building block ``scripts/ingest_ic.py`` is made of.

:func:`available_sites`
    Which site identifiers have a directory under a raw root.

:func:`available_members`
    Which member indices have a file for a given site.

:func:`ic_file`
    The path of the one file for a site and member.

:func:`default_ic_root`, :func:`default_ic_path`
    Where the raw directory and the written product are expected to be, both
    honoring ``$SIPNET_CALIBRATION_DATA``.

:func:`site_member_from_file_name`
    The ``(site, member)`` a source file name encodes, or ``None`` when the
    name is not exactly the layout template.

:func:`variable_attrs`, :func:`time_attrs`
    The attribute sets the product carries, for one variable and for the
    dropped ``time`` coordinate.

Notes
-----
**Why a product and not a reader.** The drivers got a reader on two grounds
and neither holds here. SIPNET consumes the raw ``.clim`` text, so a driver
store would be a copy the model never reads; SIPNET has no netCDF reader and
never reads these files at all -- initial conditions reach it as parameters.
And a driver store would have been hundreds of gigabytes, where this product
is ``n_variables x n_members x n_sites x 8`` bytes plus a byte per presence
cell, tens of megabytes at the ensemble sizes ``data/README.md`` records.
What the raw form costs is one file open per ``(site, member)`` cell, with no
axis along which that is cheap, which is exactly what a store removes; and the
store is the only form in which this ensemble exists off the SCC.

**Why the state-to-parameter conversion is not applied.** These files hold
initial state -- carbon pool sizes per unit ground area -- while SIPNET is
initialized from parameters, and computing three of the four parameters from
the corresponding pool needs parameters we calibrate
(``sipnet.c:1885-1924``):

* ``envi.plantWoodC = (1 - coarseRootFrac - fineRootFrac) * plantWoodInit``,
  so ``plantWoodInit`` depends on the two root fractions;
* ``envi.plantLeafC = laiInit * leafCSpWt``, so ``laiInit`` depends on the
  specific leaf weight;
* ``envi.soilWater = soilWFracInit * soilWHC``, so ``soilWFracInit`` depends
  on the water holding capacity.

Only ``soilInit`` is free of them, and even there the conversion is purely a
unit change: the parameter *is* the pool, and the factor of 1000 is
kg C m-2 to the g C m-2 that ``sipnet/docs/parameters.md`` documents. So the
conversion changes with every proposed parameter vector, is evaluated per
ensemble member at run time, and belongs to the experiment layer. This module
writes the source values in their source units, unchanged.

**Why the processed names are namespaced with ``initial_``.** These are not
observations: they are the starting state of a different analysis, at an
instant nobody has established (issue #3). ``constraints.py`` already maps a
source variable named ``AbvGrndWood`` to ``aboveground_wood_carbon`` in
``Mg C ha-1``, while the initial condition variable of that same source name is
in ``kg C m-2``. Since the ``VARIABLES`` registry holds one canonical unit per
processed name, sharing the name would force one of the two units to be wrong,
and would assert an identity nobody has confirmed -- for the soil variable
there is evidence *against* it, recorded in ``data/README.md``. The
``initial_`` prefix also reads with SIPNET's own ``plantWoodInit`` /
``soilInit`` / ``laiInit``.

The cost is that comparing initial wood against the observed biomass
constraint, which is a figure someone will want, is the caller's to get right
and fails by a factor of ten with no visual cue.
:data:`RELATED_CONSTRAINT_VARIABLES` puts the counterpart, the factor and how
well the correspondence is established into the product itself, so that the
factor lives here rather than in someone's head.

**Why the fill is masked but other non-finite values are refused.** ``-999.0``
is the *declared* ``_FillValue`` of every source variable, so masking it is the
file's own instruction rather than a judgment -- unlike the driver reader,
which reads non-physical values through because nothing declared them missing.
But masking makes every absence look alike, so :func:`read_ic_file` reads
unmasked, refuses any other non-finite value, and reports which variables held
a fill, and the product records the count.

**Why both presence companions are written unconditionally.** A schema whose
shape depends on the data means every consumer branches on whether a variable
exists. The two arrays are one byte per cell before compression and nearly
constant, so they cost almost nothing after it.

**Why these helpers are public.** The writer is a separate script, and the
attribute sets and the file-name parse are schema: the reader validates
against them, so the same code has to build them and parse names the way
discovery does. :func:`read_ic_file` is public so that a survey over the whole
ensemble applies exactly the checks the ingest applies.

**Member indices.** ``member`` is 0-based to match every other product, and
``source_member_index`` keeps the 1-based file index beside it, which matters
here because a partial read is normal: a checkout holding a few files gives a
short member axis whose source indices are whatever the file names carry.
Whether initial condition member *i* corresponds to driver member *i* is not
established -- open question 12 in ``data/README.md`` -- and the ensembles are
different sizes, which argues against it. Nothing here assumes a pairing;
``member_source`` is written so that a guard can refuse one.

**What dropping time costs.** Nothing in the calibration, since the dimension
is degenerate. But the instant the initial state describes is unrecoverable,
so these values cannot be checked against a *dated* observation and cannot
initialize a run at a known date without an external decision. In particular
there is nothing here to align against the thirteen annual constraint
snapshots.

**The variable set is not settled**, so a variable outside
:data:`SOURCE_VARIABLE_NAMES` is refused rather than dropped or guessed at, and
a survey of the full ensemble stops on one with the evidence needed to specify
it. See :data:`UNSPECIFIED_VARIABLES` and open question 6 in
``data/README.md``.

Usage
-----
Read the product, take the canonical per-variable view::

    from sipnet_calibration.initial_conditions import (
        initial_condition_fields,
        load_initial_conditions,
    )

    ic = load_initial_conditions()
    ic.sizes                                        # member, site, variable
    ic["initial_wood_carbon"].dims                  # ('member', 'site')
    ic.attrs["member_source"]                       # 'ic'

    fields = initial_condition_fields(ic)
    fields["initial_soil_organic_carbon"].attrs["units"]        # 'kg C m-2'

Tell an explicit fill from a variable the file never carried::

    filled = ic["variable_present"].sel(variable="initial_wood_carbon") & (
        ic["initial_wood_carbon"].isnull()
    )

Parse one raw file, with every per-file check applied::

    from sipnet_calibration.initial_conditions import (
        default_ic_root,
        ic_file,
        read_ic_file,
    )

    contents = read_ic_file(ic_file(default_ic_root(), site=1, member=1))
    contents.values["AbvGrndWood"]              # in source names
    contents.explicit_fills                     # source names that held -999.0
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from sipnet_calibration.sites import DATA_ROOT_ENV_VAR

__all__ = [
    "IC_FILE_GLOB",
    "IC_FILE_TEMPLATE",
    "IC_PRESENT",
    "IC_VARIABLE_ATTRS",
    "IC_VARIABLES",
    "IcFileContents",
    "MEMBER_SOURCE",
    "RELATED_CONSTRAINT_VARIABLES",
    "SOURCE_FILL_VALUE",
    "SOURCE_TIME_LONG_NAME",
    "SOURCE_TIME_UNITS",
    "SOURCE_TIME_VALUE",
    "SOURCE_VARIABLE_NAMES",
    "TIME_STATUS",
    "UNITS_PROVENANCE",
    "UNITS_STATUS",
    "UNSPECIFIED_VARIABLES",
    "VARIABLE_PRESENT",
    "available_members",
    "available_sites",
    "default_ic_path",
    "default_ic_root",
    "ic_file",
    "initial_condition_fields",
    "load_initial_conditions",
    "read_ic_file",
    "site_member_from_file_name",
    "time_attrs",
    "variable_attrs",
]

#: Source variable name -> processed variable name.
#:
#: Applied by ``scripts/ingest_ic.py``. Only the three variables confirmed by
#: inspecting real files are here; see :data:`UNSPECIFIED_VARIABLES` for the
#: two that are reported but unregistered, and the module Notes for why the
#: names are namespaced rather than shared with
#: :data:`sipnet_calibration.constraints.SOURCE_VARIABLE_NAMES`.
SOURCE_VARIABLE_NAMES = {
    "AbvGrndWood": "initial_aboveground_wood_carbon",
    "wood_carbon_content": "initial_wood_carbon",
    "soil_organic_carbon_content": "initial_soil_organic_carbon",
}

#: The initial condition variables, by processed name, in source file order.
IC_VARIABLES = tuple(SOURCE_VARIABLE_NAMES.values())

#: Source variables that are reported to appear in files not available here,
#: and are therefore **not** part of the schema.
#:
#: Their units, long names and shapes are unknown, so registering them would
#: mean inventing a unit. :func:`read_ic_file` refuses them like any other
#: unregistered variable but names this blocker in the message, so that a
#: survey of the full ensemble stops with the evidence needed to specify them.
#: Open question 6 in ``data/README.md``.
UNSPECIFIED_VARIABLES = ("leaf_carbon_content", "SoilMoistFrac")

#: What is and is not settled about the units below.
UNITS_STATUS = "source_attribute"

#: Why. Recorded on every variable so that no consumer can take the units as
#: confirmed.
UNITS_PROVENANCE = (
    "The 'units' attribute the source netCDF file carries for this variable, "
    "asserted identical across every file read. Not confirmed by the "
    "producer, and not checked against an independent definition. The "
    "constraint files carry a variable of the same source name in a different "
    "unit for AbvGrndWood, and a variable of the same declared unit whose "
    "values do not correspond for the soil carbon; see open questions 6 and 9 "
    "in data/README.md."
)

#: Per-variable metadata, by processed name.
#:
#: ``long_name`` is ours, following the project naming convention;
#: ``source_long_name`` keeps the file's own so the correspondence is never
#: guesswork. ``aggregation`` records that these are stocks rather than
#: per-timestep totals. It is inert in this product, which has no time
#: dimension, but it is a property of the quantity and the ``VARIABLES``
#: registry asks for it.
IC_VARIABLE_ATTRS = {
    "initial_aboveground_wood_carbon": {
        "units": "kg C m-2",
        "long_name": "Initial aboveground woody biomass carbon",
        "source_name": "AbvGrndWood",
        "source_long_name": "Above ground woody biomass",
        "aggregation": "instantaneous",
    },
    "initial_wood_carbon": {
        "units": "kg C m-2",
        "long_name": "Initial wood carbon",
        "source_name": "wood_carbon_content",
        "source_long_name": "Wood Carbon Content",
        "aggregation": "instantaneous",
    },
    "initial_soil_organic_carbon": {
        "units": "kg C m-2",
        "long_name": "Initial soil organic carbon",
        "source_name": "soil_organic_carbon_content",
        "source_long_name": "Soil Organic Carbon Content by Layer",
        "aggregation": "instantaneous",
    },
}

#: Initial-condition variables that have a counterpart among the annual
#: constraints, the factor from this product's unit to that one's, and how well
#: the correspondence is established.
#:
#: This exists because the two products deliberately do not share processed
#: names, so nothing else would stop a reader plotting them on one axis in
#: units that differ by a factor of ten. ``status`` is the load-bearing field:
#: ``"unconfirmed"`` means the source names agree and the quantities plausibly
#: match, and ``"contradicted"`` means the values do not correspond at the
#: sites checked. Neither is a license to convert. See ``data/README.md``.
RELATED_CONSTRAINT_VARIABLES = {
    "initial_aboveground_wood_carbon": {
        "variable": "aboveground_wood_carbon",
        "unit_factor": 10.0,
        "status": "unconfirmed",
        "note": (
            "Same source name, AbvGrndWood, in both sources, but this product "
            "is kg C m-2 from the file's own attribute and the constraint "
            "product is Mg C ha-1 documented for the reanalysis output. "
            "1 kg C m-2 = 10 Mg C ha-1. Whether the two are the same quantity "
            "is not established."
        ),
    },
    "initial_soil_organic_carbon": {
        "variable": "total_soil_carbon",
        "unit_factor": 1.0,
        "status": "contradicted",
        "note": (
            "Both are declared kg C m-2, but the values do not correspond at "
            "the sites that could be checked, and the source names differ "
            "(soil_organic_carbon_content against TotSoilCarb). Do not treat "
            "these as the same quantity."
        ),
    },
}

#: The ``_FillValue`` every source data variable declares.
SOURCE_FILL_VALUE = -999.0

#: What the source's degenerate ``time`` coordinate claims, kept verbatim.
#:
#: The ``units`` string is an unsubstituted template that no calendar library
#: can parse, which is issue #3; both strings trace to PEcAn's
#: ``standard_vars.csv``. Asserted identical across every file read.
SOURCE_TIME_UNITS = "days since [year]-01-01 00:00:00 UTC"
SOURCE_TIME_LONG_NAME = "Time middle averaging period"
SOURCE_TIME_VALUE = 1.0

#: What became of that coordinate in this product.
TIME_STATUS = "dropped"

#: Which ensemble the ``member`` coordinate indexes. Member indices are
#: meaningful only within one source.
MEMBER_SOURCE = "ic"

#: Names of the two presence companions.
IC_PRESENT = "ic_present"
VARIABLE_PRESENT = "variable_present"

#: The one file for a site and member, under the site's directory, and the glob
#: that finds every member's file for a site. The glob wildcards the *site*,
#: so a file whose embedded site disagrees with its directory is found and
#: reported as the mismatch it is rather than passed over as absent.
IC_FILE_TEMPLATE = "IC_site_{site}_{member}.nc"
IC_FILE_GLOB = "IC_site_*.nc"


@dataclass(frozen=True)
class IcFileContents:
    """One parsed initial condition file, in *source* variable names.

    Attributes
    ----------
    values:
        Source variable name -> the file's value, already ``NaN`` where the
        file held :data:`SOURCE_FILL_VALUE`. Only variables the file carries
        appear, so the keys are what answers "which variables does this file
        have".
    explicit_fills:
        The source names whose value was the declared fill. A subset of
        ``values``' keys, and what distinguishes an explicit fill from a
        variable the file never carried.
    units, long_names:
        Source variable name -> the file's ``units`` and ``long_name``
        attributes, kept so that the ingest can assert they agree across files
        and with :data:`IC_VARIABLE_ATTRS`.
    time_units, time_long_name, time_value:
        What the file's degenerate ``time`` coordinate claimed, kept for the
        same reason.
    """

    values: dict[str, float]
    explicit_fills: frozenset[str]
    units: dict[str, str]
    long_names: dict[str, str]
    time_units: str
    time_long_name: str
    time_value: float



def default_ic_root() -> Path:
    """Where the raw initial condition directory is expected to be.

    ``$SIPNET_CALIBRATION_DATA/raw/initial_conditions`` when that variable is
    set, and otherwise ``data/raw/initial_conditions`` under this checkout.
    Experiments name their paths in ``config.py``.
    """
    return _data_root() / "raw" / "initial_conditions"


def default_ic_path() -> Path:
    """Where the written product is expected to be.

    ``$SIPNET_CALIBRATION_DATA/processed/ic.nc`` when that variable is set, and
    otherwise the ``data/`` directory of this checkout.
    """
    return _data_root() / "processed" / "ic.nc"


def ic_file(root: Path | str, site: int, member: int) -> Path:
    """The initial condition file for one site and one source member index.

    Parameters
    ----------
    root:
        The raw root, holding one directory per site.
    site:
        Site identifier, 1-8000.
    member:
        The source's 1-based member index, as in the file name.

    Returns
    -------
    pathlib.Path
        ``<root>/<site>/IC_site_<site>_<member>.nc``, which exists.

    Raises
    ------
    FileNotFoundError
        If the site's directory, or the file inside it, is absent.
    """
    site, member = int(site), int(member)
    directory = Path(root) / str(site)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no initial condition directory for site {site}: {directory}"
        )
    path = directory / IC_FILE_TEMPLATE.format(site=site, member=member)
    if not path.is_file():
        raise FileNotFoundError(
            f"no initial condition file for site {site} member {member}: {path}"
        )
    return path


def available_sites(root: Path | str) -> tuple[int, ...]:
    """The site identifiers that have a directory under *root*.

    Parameters
    ----------
    root:
        The raw root.

    Returns
    -------
    tuple of int
        Site identifiers in ascending order, possibly empty. Only a directory
        whose name is exactly a positive integer counts; anything else is
        ignored rather than reported, because the source tree carries
        filesystem debris. Whether the directory holds any file is
        :func:`available_members`' business.
    """
    root = Path(root)
    if not root.is_dir():
        return ()
    sites = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        # str.isdigit() accepts unicode digits that int() also accepts but
        # that would not round-trip through the path template, so compare the
        # canonical form instead.
        if entry.name.isdigit() and str(int(entry.name)) == entry.name:
            site = int(entry.name)
            if site > 0:
                sites.append(site)
    return tuple(sorted(sites))


def available_members(root: Path | str, site: int) -> tuple[int, ...]:
    """The source member indices that have a file for *site*.

    Parameters
    ----------
    root:
        The raw root.
    site:
        Site identifier.

    Returns
    -------
    tuple of int
        1-based member indices in ascending order, possibly empty. Taken from
        the file names matching :data:`IC_FILE_GLOB`, and only from names that
        are exactly :data:`IC_FILE_TEMPLATE` for their numbers. A file whose
        embedded site disagrees with its directory is *not* filtered out here:
        it is returned so that ``scripts/ingest_ic.py`` can report the
        mismatch rather than silently skipping a file.
    """
    directory = Path(root) / str(int(site))
    if not directory.is_dir():
        return ()
    members = []
    for path in directory.glob(IC_FILE_GLOB):
        parsed = site_member_from_file_name(path.name)
        if parsed is None or not path.is_file():
            continue
        members.append(parsed[1])
    return tuple(sorted(members))


def read_ic_file(path: Path | str) -> IcFileContents:
    """Parse one initial condition file exactly and check it.

    Parameters
    ----------
    path:
        The file to read.

    Returns
    -------
    IcFileContents
        The file's variables under their *source* names, with the declared fill
        masked to ``NaN`` and recorded, together with the units, long names and
        the degenerate time metadata.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any per-file check fails: a file that is not readable as netCDF-3;
        no ``time`` variable, or a ``time`` whose length is not 1; a data
        variable whose dims are not exactly ``("time",)``, which is what a
        layer-resolved soil variable would look like; a variable outside
        :data:`SOURCE_VARIABLE_NAMES`, with the message naming
        :data:`UNSPECIFIED_VARIABLES` and open question 6 when it is one of
        those; no data variable at all; a ``_FillValue`` that is not
        :data:`SOURCE_FILL_VALUE`; a ``units`` attribute that is missing or
        disagrees with :data:`IC_VARIABLE_ATTRS`; or a non-finite value that is
        not the declared fill. The message names the file and the invariant.

    Notes
    -----
    The checks live here rather than in the ingest script so that a file is
    checked wherever it is parsed, and so that a one-off survey over the whole
    ensemble applies exactly the checks the ingest does. Checks that span
    files, or that are about the assembled product, belong to
    ``scripts/ingest_ic.py`` instead.

    The file is opened with ``decode_times=False`` and ``engine="scipy"``,
    both of which are load-bearing: the ``time`` units attribute is
    undecodable (issue #3), and these are netCDF-3 classic files that
    ``h5netcdf`` cannot open. It is read with ``mask_and_scale=False`` so that
    the declared fill is visible and can be told apart from any other
    non-finite value, which is then refused.
    """
    path = Path(path)
    try:
        dataset = xr.open_dataset(
            path, decode_times=False, engine="scipy", mask_and_scale=False
        )
    except FileNotFoundError:
        raise
    except Exception as error:
        raise ValueError(
            f"{path}: could not be read as a netCDF-3 initial condition file "
            f"({type(error).__name__}: {error})"
        ) from error

    with dataset:
        _check_time_variable_is_degenerate(dataset, path)
        _check_no_unexpected_variables(dataset, path)
        _check_variables_are_scalar_on_time(dataset, path)
        _check_variables_are_float64(dataset, path)
        _check_no_unhandled_cf_attributes(dataset, path)
        _check_fill_values_are_the_expected_sentinel(dataset, path)
        _check_units_match_the_registered_units(dataset, path)
        _check_only_declared_fills_are_non_finite(dataset, path)

        values: dict[str, float] = {}
        fills: set[str] = set()
        units: dict[str, str] = {}
        long_names: dict[str, str] = {}
        for source in dataset.data_vars:
            array = dataset[source]
            value = float(np.ravel(array.values)[0])
            if value == SOURCE_FILL_VALUE:
                fills.add(str(source))
                value = float("nan")
            values[str(source)] = value
            units[str(source)] = str(array.attrs["units"])
            long_names[str(source)] = str(array.attrs.get("long_name", ""))

        time = dataset["time"]
        return IcFileContents(
            values=values,
            explicit_fills=frozenset(fills),
            units=units,
            long_names=long_names,
            time_units=str(time.attrs.get("units", "")),
            time_long_name=str(time.attrs.get("long_name", "")),
            time_value=float(np.ravel(time.values)[0]),
        )


def load_initial_conditions(path: Path | str | None = None) -> xr.Dataset:
    """Read the initial condition product and check it against the schema.

    Parameters
    ----------
    path:
        The product to read. Defaults to :func:`default_ic_path`.

    Returns
    -------
    xarray.Dataset
        The data model described in this module's docstring: the
        :data:`IC_VARIABLES` on ``(member, site)`` as ``float64``, the two
        presence companions, ``lon``/``lat`` on ``site``, and
        ``source_member_index`` on ``member``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is not the product this module describes: a missing or
        extra data variable, a wrong dtype, wrong dims, a missing coordinate, a
        ``member`` that is not ``0..n-1``, a ``site`` or
        ``source_member_index`` that is not ascending, a missing per-variable
        or dataset attribute, a ``variable`` coordinate that is not
        :data:`IC_VARIABLES` in order, or a presence companion inconsistent
        with the values -- a variable present where no file was, or a value
        that is finite where ``variable_present`` is ``False``.

    Notes
    -----
    This is the reader every consumer uses, including the round-trip check in
    ``scripts/ingest_ic.py``, so that the writer cannot drift from the
    schema. It raises rather than returning something subtly wrong, because
    the failure this guards against -- a ``NaN`` whose origin is unknown -- is
    invisible downstream.
    """
    path = Path(path) if path is not None else default_ic_path()
    if not path.exists():
        raise FileNotFoundError(
            f"no initial condition product at {path}. Build it with "
            "scripts/ingest_ic.py."
        )
    dataset = xr.open_dataset(path, engine="h5netcdf")
    try:
        _check_dataset_matches_the_schema(dataset, path)
        _check_attributes_are_complete(dataset, path)
        _check_presence_companions_agree_with_the_values(dataset, path)
    except Exception:
        dataset.close()
        raise
    return dataset


def initial_condition_fields(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """One ``DataArray`` per variable, in :data:`IC_VARIABLES` order.

    Each field has dims ``(member, site)``, is named for its variable, carries
    that variable's attributes from :data:`IC_VARIABLE_ATTRS` with the units
    caveat attached, and keeps ``lon``/``lat`` and ``source_member_index`` as
    non-dimension coordinates -- the canonical field shape, which the Dataset
    already is per variable. This is the view facet-by-variable consumes,
    matching :func:`sipnet_calibration.drivers.driver_fields` and
    :func:`sipnet_calibration.constraints.constraint_fields`.

    Parameters
    ----------
    dataset:
        As returned by :func:`load_initial_conditions`.

    Returns
    -------
    dict
        Keyed by processed variable name. The presence companions are not
        fields and are left out; a caller that wants them reads them off the
        Dataset.

    Raises
    ------
    ValueError
        If any of :data:`IC_VARIABLES` is absent from *dataset*.
    """
    missing = [name for name in IC_VARIABLES if name not in dataset.data_vars]
    if missing:
        raise ValueError(
            f"dataset is missing initial condition variables {missing}; found "
            f"{sorted(dataset.data_vars)}"
        )
    fields = {}
    for name in IC_VARIABLES:
        field = dataset[name].copy(deep=False)
        field.attrs = {**variable_attrs(name), **dataset[name].attrs}
        fields[name] = field
    return fields


def variable_attrs(name: str) -> dict[str, object]:
    """Attributes for one variable, with the units caveat and any counterpart.

    :data:`IC_VARIABLE_ATTRS` for *name*, plus ``units_status`` and
    ``units_provenance``, plus the four ``related_constraint_*`` entries when
    :data:`RELATED_CONSTRAINT_VARIABLES` has one. The runtime counts
    ``n_explicit_fills`` and ``n_values_not_positive`` are added by the ingest
    script, which is what can count them.
    """
    attrs: dict[str, object] = {
        **IC_VARIABLE_ATTRS[name],
        "units_status": UNITS_STATUS,
        "units_provenance": UNITS_PROVENANCE,
    }
    related = RELATED_CONSTRAINT_VARIABLES.get(name)
    if related is not None:
        attrs["related_constraint_variable"] = related["variable"]
        attrs["related_constraint_unit_factor"] = related["unit_factor"]
        attrs["related_constraint_status"] = related["status"]
        attrs["related_constraint_note"] = related["note"]
    return attrs


def time_attrs() -> dict[str, object]:
    """Dataset attributes recording the source's dropped ``time`` coordinate.

    ``source_time_units``, ``source_time_long_name``, ``source_time_value``,
    ``time_status`` and a ``time_note`` giving the reason and the issue number.
    """
    return {
        "source_time_units": SOURCE_TIME_UNITS,
        "source_time_long_name": SOURCE_TIME_LONG_NAME,
        "source_time_value": SOURCE_TIME_VALUE,
        "time_status": TIME_STATUS,
        "time_note": (
            "The source files carry a length-1 unlimited 'time' dimension "
            "whose units attribute is an unsubstituted template that no "
            "calendar library can parse (issue #3), so it is dropped and what "
            "it claimed is recorded here. These values describe a single "
            "instant that has not been established, so nothing here is "
            "time-aware and there is nothing to align against the annual "
            "constraint snapshots."
        ),
    }


def site_member_from_file_name(name: str) -> tuple[int, int] | None:
    """``(site, member)`` from an ``IC_site_<site>_<member>.nc`` name.

    ``None`` when the name is not exactly the template for its numbers, so
    that :func:`available_members` ignores debris rather than failing on it.
    """
    match = _FILE_PATTERN.match(name)
    if match is None:
        return None
    site, member = match.group(1), match.group(2)
    if IC_FILE_TEMPLATE.format(site=int(site), member=int(member)) != name:
        return None
    return int(site), int(member)


# ── supporting helpers ────────────────────────────────────────────────────────


#: Exactly ``IC_site_<site>_<member>.nc``, with no leading zeros or extra
#: parts, so that a name off the template is skipped rather than half-read.
_FILE_PATTERN = re.compile(r"^IC_site_(\d+)_(\d+)\.nc$")

#: Per-variable source attributes this reader understands and acts on.
#: Anything else is refused by :func:`_check_no_unhandled_cf_attributes`,
#: because the read disables CF scaling and an unhandled attribute would
#: change what the value means.
_HANDLED_VARIABLE_ATTRS = frozenset({"units", "long_name", "_FillValue"})

#: Dataset attributes the product must carry, checked on read.
_REQUIRED_DATASET_ATTRS = (
    "title",
    "source_root",
    "source_layout",
    "source_fill_value",
    "history",
    "member_source",
    "member_correspondence",
    "n_sites",
    "n_members",
    "coverage",
    "source_time_units",
    "source_time_long_name",
    "source_time_value",
    "time_status",
    "time_note",
)

#: Per-variable attributes the product must carry, checked on read. The two
#: counts are added by the ingest, which is what can count them.
_REQUIRED_VARIABLE_ATTRS = (
    "units",
    "long_name",
    "source_name",
    "source_long_name",
    "aggregation",
    "units_status",
    "units_provenance",
    "n_explicit_fills",
    "n_values_not_positive",
)


def _data_root() -> Path:
    """The ``data`` directory, honoring ``$SIPNET_CALIBRATION_DATA``."""
    root = os.environ.get(DATA_ROOT_ENV_VAR)
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[2] / "data"


# ── checks ────────────────────────────────────────────────────────────────────


def _check_time_variable_is_degenerate(dataset: xr.Dataset, path: Path) -> None:
    """The file has a ``time`` variable of length exactly 1.

    A length other than 1 would mean these are not static initial conditions,
    which is a different product and must not be quietly averaged away.
    """
    if "time" not in dataset.variables:
        raise ValueError(
            f"{path}: has no 'time' variable. Every initial condition file "
            "carries one, of length 1."
        )
    length = int(dataset.sizes.get("time", 0))
    if length != 1:
        raise ValueError(
            f"{path}: 'time' has length {length}, expected 1. A longer time "
            "dimension would mean these are not static initial conditions, "
            "which is a different product; it must not be averaged away."
        )


def _check_variables_are_scalar_on_time(dataset: xr.Dataset, path: Path) -> None:
    """Every data variable has dims exactly ``("time",)``.

    This is the check that catches a layer-resolved variable. The soil
    variable's own long name says "by Layer", so a file carrying a layer
    dimension is a live possibility, and it would otherwise be flattened into
    a single cell silently.
    """
    for source, array in dataset.data_vars.items():
        if tuple(array.dims) != ("time",):
            raise ValueError(
                f"{path}: variable {source!r} has dims {tuple(array.dims)}, "
                "expected ('time',). A variable resolved over layers or any "
                "other dimension cannot be stored as one value per site and "
                "member, and must not be flattened silently."
            )


def _check_variables_are_float64(dataset: xr.Dataset, path: Path) -> None:
    """Every data variable is ``float64``, as the source format is.

    The product advertises ``float64`` and its contract is bit-exact
    passthrough, so a ``float32`` source would lose about eight significant
    digits between the file and the product with nothing to signal it, and an
    integer source would make the declared fill ambiguous.
    """
    for source, array in dataset.data_vars.items():
        if array.dtype != np.float64:
            raise ValueError(
                f"{path}: variable {source!r} is {array.dtype}, expected "
                "float64. The product stores float64 and promises the source "
                "value unchanged; a narrower source dtype would be upcast "
                "silently and the promise would be false."
            )


def _check_no_unexpected_variables(dataset: xr.Dataset, path: Path) -> None:
    """Every data variable is in :data:`SOURCE_VARIABLE_NAMES`, and there is one.

    An unregistered variable has no processed name and no units, so it can
    only be dropped or guessed at; both are worse than stopping. When the
    variable is one of :data:`UNSPECIFIED_VARIABLES` the message says so and
    points at open question 6, because that case is expected and is what a
    survey of the full ensemble exists to resolve.
    """
    found = [str(name) for name in dataset.data_vars]
    if not found:
        raise ValueError(
            f"{path}: holds no data variable. Expected at least one of "
            f"{sorted(SOURCE_VARIABLE_NAMES)}."
        )
    unknown = [name for name in found if name not in SOURCE_VARIABLE_NAMES]
    if not unknown:
        return
    unspecified = [name for name in unknown if name in UNSPECIFIED_VARIABLES]
    other = [name for name in unknown if name not in UNSPECIFIED_VARIABLES]
    if unspecified:
        raise ValueError(
            f"{path}: carries {unspecified}, which are reported to appear in "
            "the ensemble but are not specified: no file carrying either was "
            "available when the schema was written, so their units and long "
            "names are unknown and registering them would have meant "
            "inventing a unit. This is the expected blocker, not a defect in "
            "the file. Record the units and long names this file carries, add "
            "them to IC_VARIABLE_ATTRS and SOURCE_VARIABLE_NAMES, and rerun. "
            "See open question 6 in data/README.md."
        )
    raise ValueError(
        f"{path}: carries unregistered variable(s) {other}. Only "
        f"{sorted(SOURCE_VARIABLE_NAMES)} have a processed name and a unit. A "
        "new variable needs both before it can be stored, so this stops "
        "rather than dropping it."
    )


def _check_no_unhandled_cf_attributes(dataset: xr.Dataset, path: Path) -> None:
    """No data variable carries a CF attribute this reader does not act on.

    The file is read with ``mask_and_scale=False`` so that the declared fill
    stays visible, but that also switches off ``scale_factor`` and
    ``add_offset`` decoding. A packed variable would then be stored as its raw
    storage value, rescaled by an arbitrary factor, with nothing to signal it
    -- the same failure :func:`_check_units_match_the_registered_units` exists
    to prevent. ``missing_value`` is a CF alias for ``_FillValue`` and would
    likewise go unmasked. None of the available files carries any of these, so
    refusing them costs nothing and keeps the "source values unchanged" claim
    true.
    """
    for source, array in dataset.data_vars.items():
        unhandled = sorted(set(array.attrs) - _HANDLED_VARIABLE_ATTRS)
        if unhandled:
            raise ValueError(
                f"{path}: variable {source!r} carries {unhandled}, which this "
                "reader does not act on. It reads with mask_and_scale=False, "
                "so scale_factor, add_offset and missing_value would be "
                "ignored rather than applied and the stored value would not "
                "be the physical one. Handle the attribute explicitly before "
                "admitting a file that uses it."
            )


def _check_fill_values_are_the_expected_sentinel(
    dataset: xr.Dataset, path: Path
) -> None:
    """Every data variable declares :data:`SOURCE_FILL_VALUE`.

    Compared numerically rather than by dtype: the real files carry the
    attribute as ``float64`` and a netCDF-3 writer given a Python float may
    write ``float32``, which is not a difference worth failing on.
    """
    for source, array in dataset.data_vars.items():
        if "_FillValue" not in array.attrs:
            raise ValueError(
                f"{path}: variable {source!r} declares no _FillValue. The "
                f"schema relies on {SOURCE_FILL_VALUE} marking a missing "
                "value, and a variable without it cannot be read that way."
            )
        declared = float(np.ravel(array.attrs["_FillValue"])[0])
        if declared != SOURCE_FILL_VALUE:
            raise ValueError(
                f"{path}: variable {source!r} declares _FillValue "
                f"{declared!r}, expected {SOURCE_FILL_VALUE}. A different "
                "sentinel would be read as a real value."
            )


def _check_units_match_the_registered_units(dataset: xr.Dataset, path: Path) -> None:
    """Every data variable's ``units`` attribute is the registered one.

    The product keeps one unit string per variable, so a file that disagreed
    would be invisible afterwards. This is also what makes the recorded
    ``units_status`` true: the unit is the source's, asserted across files.
    """
    for source, array in dataset.data_vars.items():
        processed = SOURCE_VARIABLE_NAMES[str(source)]
        expected = IC_VARIABLE_ATTRS[processed]["units"]
        if "units" not in array.attrs:
            raise ValueError(
                f"{path}: variable {source!r} has no units attribute. The "
                f"schema records {expected!r} for it, taken from the source "
                "files, and a file that states nothing cannot confirm it."
            )
        found = str(array.attrs["units"])
        if found != expected:
            raise ValueError(
                f"{path}: variable {source!r} is in {found!r}, but the schema "
                f"registers {expected!r} for {processed!r}. Nothing here "
                "converts units, so a file on a different scale would be "
                "stored as though it were on this one."
            )


def _check_only_declared_fills_are_non_finite(
    dataset: xr.Dataset, path: Path
) -> None:
    """The only non-finite or sentinel value is :data:`SOURCE_FILL_VALUE`.

    Run on the unmasked read. A ``NaN`` or infinity in the source would be
    indistinguishable from a fill once masked, and the product's whole account
    of missingness rests on that distinction.
    """
    for source, array in dataset.data_vars.items():
        values = np.ravel(array.values)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{path}: variable {source!r} holds a non-finite value "
                f"({values.tolist()}). Missing values are marked with "
                f"{SOURCE_FILL_VALUE}; a NaN or infinity would be "
                "indistinguishable from that once masked, and the product "
                "tells the two kinds of absence apart."
            )


def _check_dataset_matches_the_schema(dataset: xr.Dataset, path: Path) -> None:
    """The product has exactly the variables, dims, dtypes and coords declared.

    The structural half of :func:`load_initial_conditions`' validation: the
    :data:`IC_VARIABLES` and both presence companions and nothing else, the
    dims and dtypes of each, the five coordinates, ``member`` equal to
    ``0..n-1``, ``site`` and ``source_member_index`` ascending, and
    ``variable`` equal to :data:`IC_VARIABLES` in order.
    """
    expected_vars = set(IC_VARIABLES) | {IC_PRESENT, VARIABLE_PRESENT}
    found = set(map(str, dataset.data_vars))
    if found != expected_vars:
        raise ValueError(
            f"{path}: data variables are {sorted(found)}, expected "
            f"{sorted(expected_vars)}."
        )

    for name in IC_VARIABLES:
        array = dataset[name]
        if tuple(array.dims) != ("member", "site"):
            raise ValueError(
                f"{path}: {name!r} has dims {tuple(array.dims)}, expected "
                "('member', 'site')."
            )
        if array.dtype != np.float64:
            raise ValueError(
                f"{path}: {name!r} is {array.dtype}, expected float64."
            )

    for name, dims in ((IC_PRESENT, ("member", "site")),
                       (VARIABLE_PRESENT, ("member", "site", "variable"))):
        array = dataset[name]
        if tuple(array.dims) != dims:
            raise ValueError(
                f"{path}: {name!r} has dims {tuple(array.dims)}, expected "
                f"{dims}."
            )
        if array.dtype != np.bool_:
            raise ValueError(f"{path}: {name!r} is {array.dtype}, expected bool.")

    for coord in ("member", "source_member_index", "site", "lon", "lat", "variable"):
        if coord not in dataset.coords:
            raise ValueError(
                f"{path}: coordinate {coord!r} is missing; the product "
                "promises member, source_member_index, site, lon, lat and "
                "variable."
            )

    member = dataset["member"].values
    if not np.array_equal(member, np.arange(member.size)):
        raise ValueError(
            f"{path}: 'member' is not 0..{member.size - 1}. Member labels are "
            "a 0-based index into this source's ensemble; the source's own "
            "1-based index is source_member_index."
        )
    if dataset["member"].dtype != np.int16:
        raise ValueError(
            f"{path}: 'member' is {dataset['member'].dtype}, expected int16."
        )
    if dataset["site"].dtype != np.int32:
        raise ValueError(
            f"{path}: 'site' is {dataset['site'].dtype}, expected int32."
        )
    if dataset["source_member_index"].dtype != np.int16:
        raise ValueError(
            f"{path}: 'source_member_index' is "
            f"{dataset['source_member_index'].dtype}, expected int16, to "
            "match the member axis it sits on."
        )

    for coord in ("site", "source_member_index"):
        values = dataset[coord].values
        if np.any(np.diff(values) <= 0):
            raise ValueError(
                f"{path}: {coord!r} is not strictly ascending. Both axes are "
                "sorted and unique so that a selection cannot silently pick "
                "the wrong cell."
            )

    variable = [str(name) for name in dataset["variable"].values]
    if tuple(variable) != IC_VARIABLES:
        raise ValueError(
            f"{path}: 'variable' is {variable}, expected {list(IC_VARIABLES)} "
            "in that order. variable_present is indexed by it, so an "
            "out-of-order axis would attribute one variable's presence to "
            "another."
        )

    for coord in ("lon", "lat"):
        if tuple(dataset[coord].dims) != ("site",):
            raise ValueError(
                f"{path}: {coord!r} has dims {tuple(dataset[coord].dims)}, "
                "expected ('site',) as a non-dimension coordinate."
            )


def _check_presence_companions_agree_with_the_values(
    dataset: xr.Dataset, path: Path
) -> None:
    """The presence arrays and the values tell the same story.

    ``variable_present`` is ``False`` wherever ``ic_present`` is, no value is
    finite where ``variable_present`` is ``False``, and every ``NaN`` is
    accounted for by one of the three kinds of absence in the module
    docstring. Without this the ``variable_present & isnan(value)``
    fill indicator would be a claim rather than a fact.
    """
    ic_present = dataset[IC_PRESENT].transpose("member", "site").values
    variable_present = dataset[VARIABLE_PRESENT].transpose(
        "member", "site", "variable"
    ).values

    absent_file = ~ic_present
    if np.any(variable_present[absent_file]):
        raise ValueError(
            f"{path}: {VARIABLE_PRESENT} is True where {IC_PRESENT} is False. "
            "A file that does not exist cannot carry a variable."
        )

    for index, name in enumerate(IC_VARIABLES):
        values = dataset[name].transpose("member", "site").values
        present = variable_present[:, :, index]
        # isnan, not isfinite: the data model says every absence is NaN, and an
        # infinity is not NaN, so isfinite would let one through as an absence
        # and a consumer's nanmean would then propagate it.
        if np.any(~np.isnan(values[~present])):
            raise ValueError(
                f"{path}: {name!r} is not NaN where {VARIABLE_PRESENT} is "
                "False. A value the source never carried cannot have one, and "
                "the fill indicator variable_present & isnan(value) would be "
                "wrong."
            )
        recorded = dataset[name].attrs.get("n_explicit_fills")
        if recorded is not None:
            counted = int(np.count_nonzero(present & np.isnan(values)))
            if int(recorded) != counted:
                raise ValueError(
                    f"{path}: {name!r} records n_explicit_fills "
                    f"{int(recorded)} but the arrays hold {counted} cells "
                    f"where {VARIABLE_PRESENT} is True and the value is NaN. "
                    "That count and that indicator are two statements of the "
                    "same fact, so they cannot disagree."
                )


def _check_attributes_are_complete(dataset: xr.Dataset, path: Path) -> None:
    """Every attribute the data model promises is present.

    Per variable: ``units``, ``long_name``, ``source_name``,
    ``source_long_name``, ``aggregation``, ``units_status``,
    ``units_provenance``, ``n_explicit_fills``, ``n_values_not_positive``, and
    the four ``related_constraint_*`` entries where one is registered. On the
    dataset: ``member_source``, ``member_correspondence``, ``coverage``, the
    five time attributes, and the source and provenance strings. These are
    what make the product self-describing, so a missing one is a defect rather
    than a cosmetic gap.
    """
    missing = [key for key in _REQUIRED_DATASET_ATTRS if key not in dataset.attrs]
    if missing:
        raise ValueError(
            f"{path}: dataset attributes {missing} are missing. They are what "
            "make the product self-describing -- the units caveat, the member "
            "source, the coverage and what the dropped time coordinate "
            "claimed."
        )
    if dataset.attrs["member_source"] != MEMBER_SOURCE:
        raise ValueError(
            f"{path}: member_source is {dataset.attrs['member_source']!r}, "
            f"expected {MEMBER_SOURCE!r}. Member labels are meaningful only "
            "within one source, and this attribute is what a cross-source "
            "guard reads."
        )
    if dataset.attrs["coverage"] not in ("complete", "gaps"):
        raise ValueError(
            f"{path}: coverage is {dataset.attrs['coverage']!r}, expected "
            "'complete' or 'gaps'."
        )

    for name in IC_VARIABLES:
        attrs = dataset[name].attrs
        missing = [key for key in _REQUIRED_VARIABLE_ATTRS if key not in attrs]
        related = RELATED_CONSTRAINT_VARIABLES.get(name)
        if related is not None:
            missing += [
                key
                for key in (
                    "related_constraint_variable",
                    "related_constraint_unit_factor",
                    "related_constraint_status",
                    "related_constraint_note",
                )
                if key not in attrs
            ]
        if missing:
            raise ValueError(
                f"{path}: variable {name!r} is missing attributes {missing}."
            )
        expected_units = IC_VARIABLE_ATTRS[name]["units"]
        if attrs["units"] != expected_units:
            raise ValueError(
                f"{path}: variable {name!r} records units {attrs['units']!r}, "
                f"expected {expected_units!r}. Nothing converts units, so the "
                "stored string has to be the source's."
            )
