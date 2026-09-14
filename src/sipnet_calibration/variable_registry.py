"""Variable specs: canonical units, aggregation rule, plotting style.

Overview
--------
This module holds the one description of every variable the project handles.
This includes the canonical unit for the stored variable, temporal aggregation
rule, and the sign convention. Also stores variable information that is 
utilized by the plotting code for proper formatting, labels, etc.

Data model
----------
:data:`VARIABLES` maps a processed variable name to a :class:`VarSpec`.

======================= ==================================================
:class:`VarSpec` field  Meaning
======================= ==================================================
``label``               Short display name, for a panel title or a legend
                        entry (for axis labels use the long name)
``units``               The canonical unit. Adapters convert into it and
                        nothing downstream reconciles units.
``agg``                 The temporal aggregation rule, one of
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

**Included Variables.** This registry includes meteorological drivers and
constraint data.

**Excluded Variables.** Most SIPNET outputs (GPP, carbon pools, cumulative
fluxes) and the initial condition variables. These are already handled 
by pySIPNET and ``initial_conditions.py``. :func:`variable_spec` raises on 
a name that is absent rather than guessing a rule for it.

Functions
---------
:func:`variable_spec`
    One variable's :class:`VarSpec`, raising a message that says what to do
    when the name is not registered.

Notes
-----
**Canonical NEE is a per-timestep total,** ``g C m-2``, matching what SIPNET
reports, so that the likelihood compares totals and the model side needs no
conversion. The observed product is a rate, ``umol CO2 m-2 s-1``, confirmed by
its producer; :data:`GRAMS_CARBON_PER_MICROMOLE_CO2` and the timestep length
convert it, and that conversion belongs in the NEE adapter.

The cost of that choice is that ``g C m-2`` does not name the timestep, so a
3-hourly field and a daily one carry the same ``units`` while differing by a
factor of eight. :func:`~sipnet_calibration.observation_operators.aggregate_time`
records ``aggregation_applied`` and ``aggregation_freq`` in the attributes of
what it returns, which is what tells the two apart.

**A driver variable's rule appears twice**, here and in
``drivers.DRIVER_VARIABLE_ATTRS``, which writes an ``aggregation`` attribute
onto every driver field. This module is the authority (issue #24 removes the
duplication):
``aggregate_time`` reads the registry and never the attribute. The two are
asserted equal in ``tests/test_variable_registry.py``, so neither can be
changed alone; removing the duplication means the reader deriving its
attribute from here, which is possible now that the registry is in the data
layer.

**``center = 0.0`` for a signed flux is correctness, not cosmetics.** A
sequential colormap on a quantity whose sign is the thing being read is a
genuinely misleading map. It is set on ``nee``, and on the two temperatures,
where zero is the phase boundary rather than an arbitrary origin.

Usage
-----
::

    from sipnet_calibration.variable_registry import VARIABLES, variable_spec

    variable_spec("nee").agg            # 'sum'
    variable_spec("air_temperature").agg  # 'mean'
    VARIABLES["par"].units              # 'mol m-2'
    sorted(VARIABLES)                   # every registered variable
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

#: The rule for a stock: a value that is a level at an instant rather than a
#: quantity accumulated over an interval, and so has no aggregation rule of
#: its own. ``aggregate_time`` refuses a variable carrying it rather than
#: silently taking the first or the last value of each period.
INSTANTANEOUS = "instantaneous"

#: What :attr:`VarSpec.agg` may be.
#:
#: ``"sum"`` for a quantity accumulated over the timestep -- a flux total, a
#: radiation total, a precipitation depth -- which is extensive, so a coarser
#: period is the sum of the finer ones. ``"mean"`` for intensive state, whose
#: value does not depend on the length of the interval it describes.
#: ``"last"`` for an already-cumulative series, whose value at the end of a
#: period is the period's value. ``"first"`` for symmetry. :data:`INSTANTANEOUS`
#: is a refusal rather than a method, and is not among the methods
#: ``aggregate_time`` accepts as an explicit ``how``.
AGGREGATION_RULES: tuple[str, ...] = ("sum", "mean", "last", "first", INSTANTANEOUS)

#: Grams of carbon per micromole of CO2, for the observed-NEE conversion.
#:
#: One micromole of CO2 carries one micromole of carbon, so the factor is the
#: molar mass of carbon scaled to micromoles. The conversion from the observed
#: rate to the canonical per-timestep total is::
#:
#:     g C m-2 per timestep
#:         = rate[umol CO2 m-2 s-1] * GRAMS_CARBON_PER_MICROMOLE_CO2
#:           * timestep_length_days * 86400
#:
#: The timestep length must come from the ``.clim`` ``length`` column, whose
#: value is ``drivers.CLIM_FILE_CONSTANTS["length"]`` and is checked against
#: every file the reader parses, rather than from an assumed three hours.
#:
#: The producer of the gap-filled product documents the conversion as
#: ``kg C m-2 s-1 = umol CO2 m-2 s-1 * 12e-9``, which is the same calculation
#: with carbon's molar mass rounded to 12. The two differ by 0.09%; the
#: unrounded value is used here because nothing depends on matching the
#: producer's arithmetic digit for digit, and 12.011 is the right mass.
GRAMS_CARBON_PER_MICROMOLE_CO2 = 12.011e-6


@dataclass(frozen=True)
class VarSpec:
    """What one variable is: its canonical unit, its rules and its style.

    Attributes
    ----------
    label:
        Short display name, for a panel title or a legend entry.
    units:
        The canonical unit, as an adapter must produce it and as
        ``fields.validate_field`` checks ``attrs["units"]`` against.
    agg:
        The temporal aggregation rule, one of :data:`AGGREGATION_RULES`.
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


#: Processed variable name to :class:`VarSpec`.
#:
#: The eight meteorological drivers, whose units and rules agree with
#: ``drivers.DRIVER_VARIABLE_ATTRS``; the four annual constraints, whose units
#: agree with ``constraints.CONSTRAINT_VARIABLE_ATTRS``; and ``nee``. The
#: module docstring says which variables are deliberately absent and why.
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
