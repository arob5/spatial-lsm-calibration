"""What each variable *is*: its canonical unit, its aggregation rule, its style.

Overview
--------
This module holds the one description of every variable the project handles.
Three of its fields are correctness rather than presentation -- the canonical
unit a variable is held in, the temporal aggregation rule it obeys, and the
sign convention it follows -- and they are here because they are properties of
the variable. Every place that rediscovers one of them is a place it can be
rediscovered wrongly, and two of the three fail silently when they are:
aggregating a per-timestep flux with a mean is wrong by the number of steps in
the period, and plotting a rate against a total is wrong by orders of
magnitude. Neither looks wrong on a figure.

It is also what replaces a function per variable -- ``plot_nee``, ``plot_gpp``,
``plot_lai`` -- which is the combinatorial trap the plotting design exists to
avoid.

Where it sits
-------------
This is a **data-layer** module, not a plotting one, and it imports nothing
from :mod:`sipnet_calibration.plotting`. That is deliberate and load-bearing:
:func:`sipnet_calibration.observation_operators.aggregate_time` reads the
registry and is imported by the observation operator, so a registry under
``plotting/`` would make the likelihood import ``matplotlib.pyplot``, and
would invert the dependency direction that
:mod:`sipnet_calibration`'s own docstring states. The plotting layer reads
this module; this module reads nothing.

The dependency runs::

    variable_registry  ->  observation_operators  ->  the likelihood
                       ->  fields.validate_field
                       ->  plotting/

What it reads
-------------
Nothing. It is a literal mapping, with no file, environment or import
dependency of its own.

The data model
--------------
:data:`VARIABLES` maps a **processed** variable name to a :class:`VarSpec`.
The keys follow the project's naming convention -- lower case with
underscores, and no abbreviation that is not universal -- so ``lai`` and
``par`` are keys but ``AbvGrndWood`` is not; the source-name correspondence
lives with each reader, in ``drivers.SOURCE_VARIABLE_NAMES`` and
``constraints.SOURCE_VARIABLE_NAMES``.

======================= ==================================================
:class:`VarSpec` field  Meaning
======================= ==================================================
``label``               Short display name, for a panel title or a legend
                        entry. Not an axis label: an axis wants the field's
                        own ``long_name``, which is longer and says what
                        the value is a summary of.
``units``               The **canonical** unit. Adapters convert into it and
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

**Which variables are here.** The eight meteorological drivers, the four
annual constraints, and ``nee``. The first twelve have readers, so their units
and rules are checked against what those readers write. ``nee`` is here
because its canonical unit is a decision this module has to record; no reader
produces it yet.

**Which are deliberately absent.** SIPNET's other outputs -- ``gpp``, the
carbon pools, the cumulative fluxes -- and the initial-condition variables.
Their *processed names* are the SIPNET adapter's and
``initial_conditions.py``'s to settle, and a guess made here would have to be
renamed later. :func:`variable_spec` raises on a name that is absent rather
than guessing a rule for it.

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
onto every driver field. This module is the authority:
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
#: The timestep length must come from the ``.clim`` ``length`` column
#: (``drivers.CLIM_FILE_CONSTANTS["length"]``, asserted per file), never from
#: an assumed three hours. Agrees with the factor the producer of the
#: gap-filled product documents, ``kg C m-2 s-1 = umol CO2 m-2 s-1 * 12e-9``.
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
