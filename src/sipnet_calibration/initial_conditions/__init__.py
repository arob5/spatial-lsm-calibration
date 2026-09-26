"""The initial condition ensemble: the variables, the readers, and the
conversion to SIPNET parameters.

Overview
--------
Every member of the 8000-site ensemble starts SIPNET from a drawn initial
state -- soil organic carbon, wood carbon, leaf carbon and surface soil
moisture, one value per site and member. This package owns that data from
PEcAn's files through to the parameters SIPNET reads. Three of the modules own
a stored artifact -- the source tree, the raw netCDF, the product -- and each
holds everything about its own: the schema, how it is written, how it is read
back, and the checks both sides are held to. The other three hold what those
share: the names, the variable specs, and the conversion to SIPNET.

========================  ==================================================
``names``                 dimension names, file locations
``source_files``          PEcAn's source netCDFs: the format, and the parser
``specs``                 what each variable is; the registry
``raw``                   the tracked raw netCDF: build, encode, read
``processed``             the product: build, encode, read, fields
``sipnet_parameters``     the conversion to SIPNET's initial parameters
========================  ==================================================

Everything below is re-exported here, so a caller imports from
``sipnet_calibration.initial_conditions`` and never names a module.

The data flows one way, and the first arrow is taken once, on the SCC::

    <site>/IC_site_<site>_<member>.nc  x 800,000      (the PEcAn source files)
      -> scripts/raw_sources/convert_initial_conditions.py
      -> raw/initial_conditions/pecan_pool_initial_conditions.nc   tracked
      -> scripts/ingest_initial_conditions.py    read_raw(), build_initial_conditions()
      -> processed/initial_conditions.nc
      -> this package                            load_initial_conditions()

``data/README.md`` documents the source files and the open questions about
them; ``data/raw/initial_conditions/provenance.md`` records the conversion.

Input data
----------
``data/raw/initial_conditions/files/<site>/IC_site_<site>_<member>.nc``
    The source files, present only on the SCC. netCDF-3 classic, no
    global attributes, one unlimited ``time`` dimension of length 1 whose
    variable carries an undecodable units template, and three to five scalar
    ``float64`` variables named as ``SOURCE.names``, each declaring
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

**Dimensions**: ``initial_condition_member``
(:data:`~sipnet_calibration.conventions.INITIAL_CONDITION_MEMBER`,
the ensemble's own members, a batch dim named for its source so that it
crosses rather than pairs with the samples or the drivers), ``site``. There
is no ``time``: the source's is a
length-1 record dimension whose units attribute is an unsubstituted template,
and what it claimed is kept verbatim in the dataset attributes.

**Data variables**, one per spec, all ``float64`` on
``(initial_condition_member, site)``,
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

============================ ============================ ===============================
Name                         Dims                         Meaning
============================ ============================ ===============================
``initial_condition_member`` ``initial_condition_member`` ``int64``, 0-based: the
                                                          member's identity,
                                                          ``source_index - 1``
``source_index``             ``initial_condition_member`` ``int64``, the 1-based index in
                                                          the source file name
``site``                     ``site``                     ``int32``, the whole pool,
                                                          ascending
``lon``, ``lat``             ``site``                     ``float64``, from the site table
============================ ============================ ===============================

The tracked raw file keeps its own ``member`` dim, the source index, since raw
data is never edited; :func:`build_initial_conditions` renames it.

**Attributes** follow CF-1.11 as the constraint products do. Each variable
carries the spec's ``units``, ``long_name``, ``description``, ``product``,
``source_name``, ``source_units``, ``source_long_name``,
``sipnet_initial_condition``, ``pecan_conversion``, ``units_provenance`` and,
when set, ``constituent`` and ``comment``. The dataset carries
``Conventions``, ``title``, ``product``, ``source_file``, ``source_root``,
``source_script``, ``source_script_note``, ``nominal_date``,
``nominal_date_provenance``,
``source_time_units``, ``source_time_long_name``, ``source_time_value``,
``n_sites``, ``n_initial_condition_members``, ``history`` and ``created``.

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
Each is documented where it is defined.

**The variables.** :func:`resolve_initial_condition` looks a spec up by its
processed name; :func:`describe` renders one as a paragraph for a run log.

**Reading the source files.** :func:`read_source_file` parses one file and
applies every per-file check; :func:`read_source_directory` does one site's
directory; :func:`site_member_from_file_name` decodes a file name.

**The raw file.** :func:`build_raw` assembles parsed files into the Dataset
the conversion writes, and :func:`read_raw` reads it back and checks it.

**The processed product.** :func:`build_initial_conditions` turns the raw
Dataset into the product, :func:`load_initial_conditions` reads and checks it,
and :func:`initial_condition_fields` returns it as one field per
variable, optionally for a subset of sites.

**The conversion.** :func:`to_sipnet_initial_conditions` converts one
member; :func:`to_sipnet_initial_conditions_table` converts a whole
``(initial_condition_member, site)`` ensemble to a table of SIPNET field
values.

**Paths and encodings.** :func:`default_source_root`, :func:`default_raw_dir`,
:func:`raw_path` and :func:`default_product_path` say where each file is
expected, all honoring ``$SIPNET_CALIBRATION_DATA``; :func:`raw_encoding` and
:func:`netcdf_encoding` give the two files' on-disk encodings.

Notes
-----
**One spec, no separate schema.** As ``ConstraintSpec`` does for the
observations, the spec plays the role pySIPNET's ``VariableSpec`` plays for
model output: one flat record per variable from which the product's attributes
are derived. ``sipnet_initial_condition`` names a field of
``pysipnet.parameters.InitialConditions`` and is checked against it at
import, so a spec cannot name a field that does not exist.

**Why the files are converted and the conversion tracked.** SIPNET never
reads these files; they are a PEcAn intermediate, one small file per
``(site, member)`` cell, whose inode count on the SCC dwarfs the size of the
values in it. The conversion changes structure only -- bit-exact values, the source
names and attribute strings, the source member index -- and the result is
small enough to live in version control, which is the only form in which the
ensemble exists off the SCC.

**Why ``initial_`` names.** The product is the model's starting state --
PEcAn calls the format ``pool_initial_conditions`` -- and the prefix keeps
every name distinct from the constraint products' without inventing a product
prefix. ``biomass`` rather than the file's ``woody`` for the first variable
because the Spawn and Gibbs product is total aboveground biomass carbon.

**Why the product stores state and not parameters.** Two of the four
conversions read a parameter the calibration proposes -- the root fractions for
``plantWoodInit``, the specific leaf weight for ``laiInit`` -- so the ingest
applies none of them and the product holds the state in its own units. (A
third, ``soilWFracInit``, takes no proposed parameter but is a fraction of a
water holding capacity the calibration also proposes, so what it *means*
moves too.) The conversion is a function of a state and a parameter vector,
:func:`to_sipnet_initial_conditions`, evaluated per proposal; each spec also
records the formula PEcAn applied, in ``pecan_conversion``.

**Why the conversion refuses rather than repairs.** Wood carbon is negative
over much of the ensemble and two variables are absent at some sites, so the
product does not convert unfiltered. Choosing what to do about that is the
job of the prior on initial conditions, not of a unit conversion; see the
Notes of :func:`to_sipnet_initial_conditions`.

Usage
-----
::

    from sipnet_calibration.initial_conditions import (
        initial_condition_fields, load_initial_conditions, describe,
        resolve_initial_condition,
    )

    ic = load_initial_conditions()          # Dataset, (initial_condition_member, site)
    ic["initial_soil_organic_carbon"].sel(site=4102)     # one site's 100 members
    ic["initial_wood_carbon"].mean("initial_condition_member")  # a map

    fields = initial_condition_fields(sites=[4102, 4113])
    fields["initial_leaf_carbon"].dims      # ('initial_condition_member', 'site')

    print(describe(resolve_initial_condition("initial_soil_moisture_saturation")))

    from sipnet_calibration.initial_conditions import (
        to_sipnet_initial_conditions, to_sipnet_initial_conditions_table,
    )

    conditions = to_sipnet_initial_conditions(           # one member, one site
        initial_soil_organic_carbon=13.085,
        initial_wood_carbon=0.058,
        initial_leaf_carbon=0.121,
        initial_soil_moisture_saturation=60.0,
        leaf_carbon_per_area=32.0,
        fine_root_fraction=0.2,
        coarse_root_fraction=0.2,
        deciduous=False,
    )
    conditions.soil_carbon                               # 13085.0 g C m-2

    # Which members to run is the prior's decision, not this module's, and the
    # conversion refuses a negative pool rather than choosing for you. Pick the
    # members first -- a whole (initial_condition_member, site) rectangle at a
    # time, since a DataArray cannot be ragged.
    site = {name: field.sel(site=4102) for name, field in fields.items()}
    usable = np.flatnonzero(site["initial_wood_carbon"].values >= 0)
    table = to_sipnet_initial_conditions_table(          # one row per member
        {name: field.isel(initial_condition_member=usable) for name, field in site.items()},
        leaf_carbon_per_area=32.0,                       # scalar or per member
        fine_root_fraction=0.2,
        coarse_root_fraction=0.2,
        deciduous=True,                                  # scalar or per site
    )
    table.iloc[0]                                        # one cell's six fields
"""

from __future__ import annotations

from sipnet_calibration.initial_conditions.names import (
    PRODUCT_FILE,
    RAW_FILE,
    RAW_MEMBER,
    default_product_path,
    default_raw_dir,
    default_source_root,
    raw_path,
)
from sipnet_calibration.initial_conditions.processed import (
    build_initial_conditions,
    initial_condition_fields,
    load_initial_conditions,
    netcdf_encoding,
)
from sipnet_calibration.initial_conditions.raw import build_raw, raw_encoding, read_raw
from sipnet_calibration.initial_conditions.sipnet_parameters import (
    CONVERTED_SIPNET_FIELDS,
    to_sipnet_initial_conditions,
    to_sipnet_initial_conditions_table,
)
from sipnet_calibration.initial_conditions.source_files import (
    NOMINAL_DATE,
    SOURCE,
    SOURCE_SCRIPT,
    SOURCE_SCRIPT_NOTE,
    SourceFile,
    SourceFormat,
    SourceVariable,
    read_source_directory,
    read_source_file,
    site_member_from_file_name,
)
from sipnet_calibration.initial_conditions.specs import (
    INITIAL_CONDITION_NAMES,
    INITIAL_CONDITIONS,
    InitialConditionSpec,
    describe,
    resolve_initial_condition,
)

__all__ = [
    # Names and paths.
    "PRODUCT_FILE",
    "RAW_FILE",
    "RAW_MEMBER",
    "default_product_path",
    "default_raw_dir",
    "default_source_root",
    "raw_path",
    # What the source files contain, and their provenance.
    "NOMINAL_DATE",
    "SOURCE",
    "SOURCE_SCRIPT",
    "SOURCE_SCRIPT_NOTE",
    "SourceFormat",
    "SourceVariable",
    # The variables.
    "INITIAL_CONDITIONS",
    "INITIAL_CONDITION_NAMES",
    "InitialConditionSpec",
    "describe",
    "resolve_initial_condition",
    # Reading the source files.
    "SourceFile",
    "read_source_directory",
    "read_source_file",
    "site_member_from_file_name",
    # The raw file and the processed product.
    "build_initial_conditions",
    "build_raw",
    "initial_condition_fields",
    "load_initial_conditions",
    "netcdf_encoding",
    "raw_encoding",
    "read_raw",
    # The conversion to SIPNET parameters.
    "CONVERTED_SIPNET_FIELDS",
    "to_sipnet_initial_conditions",
    "to_sipnet_initial_conditions_table",
]
