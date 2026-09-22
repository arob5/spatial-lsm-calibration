"""What each initial condition variable is.

One :class:`InitialConditionSpec` per variable of the source files: what the
quantity is, which upstream product PEcAn drew it from and how, which SIPNET
initial parameter it feeds and by what formula, and how firm its units are.

These specs are the only description of the ensemble there is. The processed
netCDF stores a spec's fields as the variable's attributes, so the file
describes itself, and its reader checks it against the same specs -- which is
why nothing else in the package carries a schema of its own.

Contents
--------
:class:`InitialConditionSpec`
    The record, one per variable.
:data:`INITIAL_CONDITIONS`, :data:`INITIAL_CONDITION_NAMES`
    The registry and its processed names, in order.
:func:`resolve_initial_condition`
    A spec by its processed name.
:func:`describe`
    One spec as a paragraph, for a run log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pysipnet.parameters import InitialConditions
from pysipnet.units import validate_units

from sipnet_calibration.initial_conditions.source_files import SOURCE

__all__ = [
    "INITIAL_CONDITIONS",
    "INITIAL_CONDITION_NAMES",
    "InitialConditionSpec",
    "NAME_PATTERN",
    "describe",
    "resolve_initial_condition",
]


#: What a processed name must look like: lower case words joined by underscores.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


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
        if self.source_name not in SOURCE.variables:
            raise ValueError(
                f"{self.name!r}: source_name {self.source_name!r} is not a variable the "
                f"source files carry: {sorted(SOURCE.variables)}"
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
        return SOURCE.variables[self.source_name].units

    @property
    def source_long_name(self) -> str:
        """The ``long_name`` string the source files carry for this variable."""
        return SOURCE.variables[self.source_name].long_name

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


#: The caveat every spec's units_provenance ends with.
_UNCONFIRMED = "Unconfirmed; see data/README.md, open question 24."

#: Checked by InitialConditionSpec, so a spec cannot name a pySIPNET initial
#: condition field that does not exist.
_SIPNET_INITIAL_CONDITION_FIELDS: frozenset[str] = frozenset(InitialConditions.model_fields)

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
