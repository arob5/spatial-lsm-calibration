"""Variable specs: canonical units, aggregation rule, plotting style.

Overview
--------
This module holds the one description of every variable the project handles:
the canonical unit its processed form is held in, what one stored value of it
means over its own timestep and therefore how several combine, the sign
convention it follows, and the display metadata a plot needs.

**A :class:`VarSpec` describes the processed form of a variable, not the
source form.** The processed form is what an ingest script writes, what a
reader such as :mod:`sipnet_calibration.drivers` returns, and what an adapter
in :mod:`sipnet_calibration.fields` converts a new source into: one variable,
under the project's processed name, in one unit. Raw source names and units
are not described here at all, and neither are the intermediate forms an
ingest passes through. So ``units`` says what an adapter must produce and what
everything downstream may assume, not what any particular file happens to
contain.

Where it sits in the pipeline
-----------------------------
The registry is downstream of nothing: it is a literal mapping, with no file,
environment or import dependency, and it imports nothing from
:mod:`sipnet_calibration.plotting`. It sits in the data layer rather than
under ``plotting/`` because
:func:`sipnet_calibration.observation_operators.aggregate_time` reads it and
will be imported by the observation operator; under ``plotting/`` that would
pull ``matplotlib.pyplot`` into the likelihood.

**Its relationship to the data processing scripts is one of agreement, not of
use.** No ingest script imports this module, and this module reads nothing
they write. Each product's source-to-processed rename and its own unit strings
live with that product's reader: ``SOURCE_VARIABLE_NAMES`` and
``DRIVER_VARIABLE_ATTRS`` in :mod:`sipnet_calibration.drivers`, and
``SOURCE_VARIABLE_NAMES`` and ``CONSTRAINT_VARIABLE_ATTRS`` in
:mod:`sipnet_calibration.constraints`. They live there because a reader has to
write those attributes into the product it produces. What this module adds is the
statement that those units are *canonical*: the ones every other product and
adapter must match. That the two agree is asserted in
``tests/test_variable_registry.py`` rather than arranged by an import, so
neither side can be changed alone. Issue #24 removes the remaining
duplication by having the driver reader take its units and rules from here.

The one consumer today is
:func:`~sipnet_calibration.observation_operators.aggregate_time`, which reads
``agg``. ``units`` is for ``fields.validate_field`` and ``label``, ``cmap``,
``center`` and ``transform`` for the plotting layer; neither reads the
registry yet, so those fields are recorded and not yet consulted.

Data model
----------
:data:`VARIABLES` maps a processed variable name to a :class:`VarSpec`. The
keys follow the project's processed naming convention -- lower case with
underscores, no abbreviation that is not universal -- so ``lai`` and ``par``
are keys but ``AbvGrndWood`` is not.

======================= ==================================================
:class:`VarSpec` field  Meaning
======================= ==================================================
``label``               Short display name, for a panel title or a legend
                        entry (for axis labels use the long name)
``units``               The canonical unit. Adapters convert into it and
                        nothing downstream reconciles units.
``agg``                 What one stored value means over its own timestep,
                        and hence the rule for combining several. One of
                        :data:`AGGREGATION_RULES`.
``cmap``                Matplotlib colormap name for a spatial panel.
``center``              The value a diverging colormap is centered on, or
                        ``None`` for a quantity with no meaningful center.
``sign``                The direction convention, for a signed quantity;
                        ``None`` where the question does not arise.
``transform``           The axis scale a plotter should use --- ``None``,
                        ``"log"`` or ``"symlog"``. ``None`` throughout at
                        present; carried because the design names it.
======================= ==================================================

**The aggregation rules.** ``agg`` is not a record of an aggregation that has
already happened, and not an instruction to aggregate. It says what a single
stored value already represents over the timestep it sits on, which is a fixed
property of the variable; the rule for combining several values follows from
that, and is what
:func:`~sipnet_calibration.observation_operators.aggregate_time` applies when
a caller asks for a coarser period. :data:`AGGREGATION_RULES` holds the five
values ``agg`` may take:

``"sum"``
    The value is a quantity **accumulated over its timestep** -- a flux total,
    a radiation total, a precipitation depth. It is extensive: it scales with
    the length of the interval, so the value for a coarser period is the sum
    of the finer ones. ``nee``, ``par`` and ``precipitation``.
``"mean"``
    The value is **intensive state**, describing conditions during its
    timestep rather than accumulating over it, so its magnitude does not
    depend on how long the interval is. A coarser period takes the mean.
    The temperatures, the vapor pressures and ``wind_speed``.
``"last"``
    The value is **already cumulative** -- a running total since some origin
    -- so the value at the end of a period already covers the whole period and
    the last value is the period's value. No registered variable carries it;
    it is what a ``cum_nee`` would.
``"first"``
    The mirror of ``"last"``, for a series read against its opening value.
    No registered variable carries it either.
:data:`INSTANTANEOUS`
    The value is a **level at an instant** -- a stock -- rather than anything
    accumulated or averaged over an interval, so it has no aggregation rule at
    all: neither a sum nor a mean of two levels is a level. This is a refusal
    rather than a method. ``aggregate_time`` raises on a variable carrying it
    instead of silently taking a period's first or last value, and a caller
    who does want one of those says so with ``how=``. It is not among the
    methods ``how=`` accepts. The four annual constraints carry it.

**Included variables.** The eight meteorological drivers, the four annual
constraints, and ``nee``. The first twelve have readers, so their units are
checked against what those readers write, and the drivers' aggregation rules
with them. ``nee`` is here because its canonical unit is a decision this
module has to record, though no reader produces it yet.

**Excluded variables.** Most SIPNET outputs -- ``gpp``, the carbon pools, the
cumulative fluxes -- and the initial-condition variables. pySIPNET names them
and documents their units in the docstrings of ``SIPNETResult``, but it
carries no machine-readable unit and no aggregation rule, so nothing here
could be derived from it; what is missing is their *processed* names, which
belong to the SIPNET adapter and to the initial-condition reader, neither
written yet. A guess made here would have to be renamed. :func:`variable_spec`
raises on a name that is absent rather than guessing a rule for it.

Functions
---------
:func:`variable_spec`
    One variable's :class:`VarSpec`, raising a message that says what to do
    when the name is not registered.

Constants
---------
:data:`VARIABLES`
    The registry itself.
:data:`AGGREGATION_RULES`
    The five values ``agg`` may take, defined above.
:data:`INSTANTANEOUS`
    The one of them that is a refusal rather than a reduction.
:data:`GRAMS_CARBON_PER_MICROMOLE_CO2`
    Grams of carbon per micromole of CO2, for the observed-NEE conversion
    below.

Notes
-----
**Canonical NEE is a per-timestep total,** ``g C m-2``, matching what SIPNET
reports, so that the likelihood compares totals and the model side needs no
conversion. The observed product is a rate, ``umol CO2 m-2 s-1``, confirmed by
its producer. The conversion into the canonical form, which belongs in the NEE
adapter, is::

    g C m-2 per timestep
        = rate[umol CO2 m-2 s-1] * GRAMS_CARBON_PER_MICROMOLE_CO2
          * timestep_length_days * 86400

One micromole of CO2 carries one micromole of carbon, so the factor is
carbon's molar mass scaled to micromoles. The timestep length must come from
the ``.clim`` ``length`` column, whose value is
``drivers.CLIM_FILE_CONSTANTS["length"]`` and is checked against every file
the reader parses, rather than from an assumed three hours. The producer of
the gap-filled product documents the same conversion as
``kg C m-2 s-1 = umol CO2 m-2 s-1 * 12e-9``, which is this calculation with
the molar mass rounded to 12; the two differ by 0.09%, and the unrounded value
is used here because nothing depends on matching the producer's arithmetic
digit for digit.

The cost of choosing the total over the rate is that ``g C m-2`` does not name
the timestep, so a 3-hourly field and a daily one carry the same ``units``
while differing by a factor of eight.
:func:`~sipnet_calibration.observation_operators.aggregate_time` records
``aggregation_applied`` and ``aggregation_freq`` in the attributes of what it
returns, which is what tells the two apart.

**A driver variable's rule appears twice**, here and in
``drivers.DRIVER_VARIABLE_ATTRS``, which writes an ``aggregation`` attribute
onto every driver field. This module is the authority: ``aggregate_time``
reads the registry and never the attribute. The two are asserted equal in
``tests/test_variable_registry.py``, so neither can be changed alone, and
issue #24 removes the duplication by having the reader derive its attribute
from here.

**``center = 0.0`` for a signed flux is correctness, not cosmetics.** A
sequential colormap on a quantity whose sign is the thing being read is a
genuinely misleading map. It is set on ``nee``, and on the two temperatures,
where zero is the phase boundary rather than an arbitrary origin.

Usage
-----
::

    from sipnet_calibration.variable_registry import VARIABLES, variable_spec

    variable_spec("nee").agg              # 'sum'
    variable_spec("air_temperature").agg  # 'mean'
    VARIABLES["par"].units                # 'mol m-2'
    sorted(VARIABLES)                     # every registered variable
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AGGREGATION_RULES",
    "GRAMS_CARBON_PER_MICROMOLE_CO2",
    "INSTANTANEOUS",
    "VARIABLES",
    "VarSpec",
    "variable_spec",
]

#: The aggregation rule for a stock, and a refusal rather than a reduction.
INSTANTANEOUS = "instantaneous"

#: What :attr:`VarSpec.agg` may be. Defined in the module docstring.
AGGREGATION_RULES: tuple[str, ...] = ("sum", "mean", "last", "first", INSTANTANEOUS)

#: Grams of carbon per micromole of CO2. The module docstring gives the
#: observed-NEE conversion it belongs to.
GRAMS_CARBON_PER_MICROMOLE_CO2 = 12.011e-6


@dataclass(frozen=True)
class VarSpec:
    """The processed form of one variable: its unit, its rule and its style.

    One entry describes a variable as the project stores and passes it
    around, not as any source file holds it. Nothing here refers to a raw
    column name, a source unit, or a particular file.

    Attributes
    ----------
    label:
        Short display name, for a panel title or a legend entry.
    units:
        The unit the variable's processed form is in. Adapters convert into
        it, ``fields.validate_field`` checks a field's ``attrs["units"]``
        against it, and nothing downstream reconciles units.
    agg:
        What one stored value means over the timestep it sits on -- an
        interval total, interval-average state, a running total, or a level
        at an instant -- and hence how several combine into a coarser
        period. One of :data:`AGGREGATION_RULES`, each defined in the module
        docstring.
    cmap:
        Matplotlib colormap name for a spatial panel.
    center:
        The value a diverging colormap is centered on, or ``None``.
    sign:
        The direction convention of a signed quantity, as free text, or
        ``None`` where the question does not arise.
    transform:
        The axis scale a plotter should use: ``None``, ``"log"`` or
        ``"symlog"``.

    Notes
    -----
    Frozen, so an entry cannot be mutated by a consumer that holds one.
    """

    label: str
    units: str
    agg: str
    cmap: str = "viridis"
    center: float | None = None
    sign: str | None = None
    transform: str | None = None


#: Processed variable name to :class:`VarSpec`. The module docstring says
#: which variables are here, which are deliberately absent, and why.
VARIABLES: dict[str, VarSpec] = {
    # ── meteorological drivers ────────────────────────────────────────────
    "air_temperature": VarSpec(
        label="Air temperature",
        units="deg C",
        agg="mean",
        cmap="RdBu_r",
        center=0.0,
    ),
    "soil_temperature": VarSpec(
        label="Soil temperature",
        units="deg C",
        agg="mean",
        cmap="RdBu_r",
        center=0.0,
    ),
    "par": VarSpec(label="PAR", units="mol m-2", agg="sum", cmap="viridis"),
    "precipitation": VarSpec(
        label="Precipitation", units="mm", agg="sum", cmap="Blues"
    ),
    "vpd": VarSpec(label="VPD", units="Pa", agg="mean", cmap="viridis"),
    "soil_vpd": VarSpec(label="Soil VPD", units="Pa", agg="mean", cmap="viridis"),
    "vapor_pressure": VarSpec(
        label="Vapor pressure", units="Pa", agg="mean", cmap="viridis"
    ),
    "wind_speed": VarSpec(
        label="Wind speed", units="m s-1", agg="mean", cmap="viridis"
    ),
    # ── annual constraints ────────────────────────────────────────────────
    "aboveground_wood_carbon": VarSpec(
        label="Aboveground wood C",
        units="Mg C ha-1",
        agg=INSTANTANEOUS,
        cmap="viridis",
    ),
    "lai": VarSpec(label="LAI", units="m2 m-2", agg=INSTANTANEOUS, cmap="YlGn"),
    "soil_moisture_percent": VarSpec(
        label="Soil moisture", units="percent", agg=INSTANTANEOUS, cmap="Blues"
    ),
    "total_soil_carbon": VarSpec(
        label="Total soil C", units="kg C m-2", agg=INSTANTANEOUS, cmap="viridis"
    ),
    # ── fluxes ────────────────────────────────────────────────────────────
    "nee": VarSpec(
        label="NEE",
        units="g C m-2",
        agg="sum",
        cmap="RdBu_r",
        center=0.0,
        sign="+ to atmosphere",
    ),
}


def variable_spec(name: str) -> VarSpec:
    """The :class:`VarSpec` for *name*.

    Parameters
    ----------
    name:
        A processed variable name, a key of :data:`VARIABLES`.

    Returns
    -------
    VarSpec
        The registered specification.

    Raises
    ------
    ValueError
        If *name* is not a string, or is not registered. The message lists the
        registered names and says to add an entry, since inferring a rule for
        an unregistered variable is what this module exists to prevent.
    """
    if not isinstance(name, str):
        raise ValueError(
            f"a variable name must be a string, got {name!r}. An array whose "
            "name is None has usually been through an xarray operation that "
            "does not carry the name forward, such as arithmetic between two "
            "arrays"
        )
    try:
        return VARIABLES[name]
    except KeyError:
        raise ValueError(
            f"{name!r} is not in the variable registry, so its canonical unit "
            "and aggregation rule are not known. Add an entry to VARIABLES in "
            "sipnet_calibration.variable_registry. The registered variables "
            f"are {sorted(VARIABLES)}"
        ) from None
