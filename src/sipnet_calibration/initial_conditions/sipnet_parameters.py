"""Turning a member's initial state into the parameters SIPNET reads.

The product stores the initial state in the source's own units because the
SIPNET parameters it feeds depend on parameters the calibration proposes: the
root fractions for ``plantWoodInit`` and the specific leaf weight for
``laiInit``. The conversion is therefore a function of a state *and* a
parameter vector, applied per proposal rather than once at ingest, and it
lives here.

Contents
--------
:func:`to_sipnet_initial_conditions`
    One ``(member, site)`` cell to a ``pysipnet.parameters.InitialConditions``.
:func:`to_sipnet_initial_conditions_table`
    A whole ensemble to a table of the same field values, one row per cell.
:data:`CONVERTED_SIPNET_FIELDS`
    The fields both of them set.

Both refuse state that is not physically valid rather than flooring or
substituting it. :func:`to_sipnet_initial_conditions`'s Notes say why, and
record what the conversion does and does not reproduce of PEcAn's arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.parameters import InitialConditions

from sipnet_calibration.initial_conditions.names import MEMBER, SITE
from sipnet_calibration.initial_conditions.specs import resolve_initial_condition

__all__ = [
    "CONVERTED_SIPNET_FIELDS",
    "to_sipnet_initial_conditions",
    "to_sipnet_initial_conditions_table",
]


#: The fields of ``pysipnet.parameters.InitialConditions`` the conversion sets,
#: in the class's own order. ``litter_carbon`` and ``snow_water_equivalent`` are
#: left at pySIPNET's defaults: nothing in the ensemble informs them, and PEcAn
#: passed them nothing either.
CONVERTED_SIPNET_FIELDS: tuple[str, ...] = (
    "total_wood_carbon",
    "leaf_area_index",
    "soil_carbon",
    "soil_wetness_fraction",
    "fine_root_fraction",
    "coarse_root_fraction",
)


def to_sipnet_initial_conditions(
    *,
    initial_soil_organic_carbon: float,
    initial_wood_carbon: float,
    initial_leaf_carbon: float,
    initial_soil_moisture_saturation: float,
    leaf_carbon_per_area: float,
    fine_root_fraction: float,
    coarse_root_fraction: float,
    deciduous: bool,
) -> InitialConditions:
    """One member's initial state as pySIPNET initial conditions.

    The state comes from the product, the parameters the mapping needs from the
    vector being proposed.

    The mapping is the one in the "How PEcAn used them" table of
    ``data/README.md``::

        soil_carbon           = 1000 x initial_soil_organic_carbon
        total_wood_carbon     = 1000 x initial_wood_carbon
                                / (1 - fine_root_fraction - coarse_root_fraction)
        leaf_area_index       = 1000 x initial_leaf_carbon / leaf_carbon_per_area,
                                or 0 for a deciduous PFT, since the run starts
                                outside leaf-on
        soil_wetness_fraction = initial_soil_moisture_saturation / 100

    Two of the four read a proposed parameter, which is why the product stores
    the state in its own units and this is applied per proposal.

    Parameters
    ----------
    initial_soil_organic_carbon, initial_wood_carbon, initial_leaf_carbon:
        The member's pools in the product's units, ``kg m-2`` of carbon.
    initial_soil_moisture_saturation:
        The member's surface soil moisture in the product's units, percent of
        saturation, so between 0 and 100.
    leaf_carbon_per_area:
        ``leafCSpWt``, g C m-2 of leaf, from the same parameter vector. Must be
        at least SIPNET's ``TINY`` of 1e-6, which it is silently floored at.
    fine_root_fraction, coarse_root_fraction:
        ``fineRootFrac`` and ``coarseRootFrac``, the shares of the total wood
        pool SIPNET splits off as roots. Their sum must leave at least a
        hundredth of the pool above ground.
    deciduous:
        Whether the site's PFT drops its leaves (PEcAn's ``fracLeafFall >
        0.5``). Our runs start on January 1, outside every leaf-on window, so a
        deciduous PFT starts with no leaves and *initial_leaf_carbon* is not
        read at all.

    Returns
    -------
    pysipnet.parameters.InitialConditions
        The six fields of :data:`CONVERTED_SIPNET_FIELDS`, with
        ``litter_carbon`` and ``snow_water_equivalent`` at pySIPNET's defaults.

    Raises
    ------
    ValueError
        If a pool the mapping reads is negative, NaN or infinite; if
        *initial_soil_moisture_saturation* is above 100; if
        *leaf_carbon_per_area* is below SIPNET's own floor; if either root
        fraction is outside ``[0, 1]``; or if the two leave less than a
        hundredth of the wood pool above ground.
    TypeError
        If *deciduous* is not a boolean.

    Notes
    -----
    **Two formulas are PEcAn's only up to a parameter.** PEcAn wrote the leaf
    row as ``leaf x SLA``, using the run's own specific leaf area draw; this
    uses SIPNET's ``leafCSpWt`` instead. The two agree only if
    ``leaf_carbon_per_area = 1000 / SLA``, which the leaf carbon note below
    explains they do not. The wood divisor entered PEcAn in September 2025, and
    which version produced the reanalysis is open question 24 of
    ``data/README.md``; the divided form is what this implements.

    **Two guards are ours, and both are against SIPNET running nonsense
    quietly.** Its initial wood pool is
    ``total_wood_carbon x (1 - fine - coarse)``, so a root-fraction sum of 1 or
    more divides by zero or flips the pool's sign; pySIPNET validates the two
    fractions separately but not their sum, and SIPNET runs a negative wood
    pool to completion -- exit 0, a full output file, empty stderr (verified,
    and filed upstream as TARPS-group/pySIPNET#39). The floor is on the
    remainder rather than the sum, because a sum just below 1 is finite and
    just as wrong: at ``1 - 1e-16`` the aboveground pool is multiplied by
    1e16. And ``setupModel`` silently raises ``leafCSpWt`` to its ``TINY`` of
    1e-6, so below that the initial leaf carbon SIPNET recovers as
    ``laiInit x leafCSpWt`` is not the one converted here; that is refused too.

    **The soil moisture has an upper end as well as a lower one.** It is a
    percent of saturation, documented over 0 to 100 by its source, and
    dividing by 100 is what makes ``soilWFracInit`` a fraction. The other end
    cannot be guarded: a value already given as a 0-1 fraction is a valid
    percentage too and converts to a hundredth of what was meant, which only a
    declared ``units`` attribute on the input catches.

    **Physically valid input only.** A negative or missing pool is refused, not
    floored or substituted. Wood carbon is negative wherever PEcAn's leaf draw
    exceeded its biomass draw, and two variables are absent at the sites whose
    source files omit them (the specs' ``description`` fields and the ingest
    report say where), so the product does not convert unfiltered. Which
    members to use is a question for the prior on initial conditions. PEcAn's
    own answer was to skip the pool and leave SIPNET's template default in
    place, without saying so.

    **The leaf carbon is only nominally carbon.** PEcAn built
    ``initial_leaf_carbon`` as a MODIS LAI draw divided by a specific leaf area
    draw expressed per kilogram of *leaf mass*, never applying the leaf carbon
    fraction of about 0.48; ``leaf_carbon_per_area`` is per m2 of *carbon*. LAI
    therefore converts back to LAI only under the same SLA draw with the carbon
    fraction applied consistently. With SIPNET's template value the forested
    sites start at an implausible leaf area index.

    **The LAI is entangled with leaf_carbon_per_area.** Under the default model
    flags SIPNET admits an exact invariance: for any ``gamma > 0``, scaling
    ``leafCSpWt`` and ``attenuation`` by ``gamma`` and ``laiInit`` by
    ``1 / gamma`` leaves the initial leaf carbon ``laiInit x leafCSpWt``
    unchanged, and with it the product ``attenuation x leaf_carbon /
    leafCSpWt`` that the light response depends on, so every carbon and water
    output is unchanged and only the LAI diagnostic moves. (Exactly so in real
    arithmetic; in float64 the rescaled products differ by an ulp often enough
    to matter to a bitwise comparison, though not at output precision.) Under NEE,
    biomass, soil carbon and soil water alone, therefore, only the ratio
    ``attenuation / leafCSpWt`` is identified. LAI observations are the only
    constraint in the planned set that breaks the degeneracy.

    **Soil wetness equates two different fractions.** The product is a percent
    of *saturation* of a satellite retrieval's 2-5 cm surface layer;
    ``soilWFracInit`` is a fraction of the water holding capacity of SIPNET's
    single soil bucket. Dividing by 100 converts the units, not the definition,
    and whether it is the intended correspondence is open question 24. PEcAn
    did the same, and the parameter is transient: SIPNET reads it once and the
    soil water pool equilibrates within weeks.
    """
    _check_arguments_are_scalar(
        initial_soil_organic_carbon=initial_soil_organic_carbon,
        initial_wood_carbon=initial_wood_carbon,
        initial_leaf_carbon=initial_leaf_carbon,
        initial_soil_moisture_saturation=initial_soil_moisture_saturation,
        leaf_carbon_per_area=leaf_carbon_per_area,
        fine_root_fraction=fine_root_fraction,
        coarse_root_fraction=coarse_root_fraction,
        deciduous=deciduous,
    )
    converted = _sipnet_fields_from_state(
        initial_soil_organic_carbon=np.array([initial_soil_organic_carbon]),
        initial_wood_carbon=np.array([initial_wood_carbon]),
        initial_leaf_carbon=np.array([initial_leaf_carbon]),
        initial_soil_moisture_saturation=np.array([initial_soil_moisture_saturation]),
        leaf_carbon_per_area=np.array([leaf_carbon_per_area]),
        fine_root_fraction=np.array([fine_root_fraction]),
        coarse_root_fraction=np.array([coarse_root_fraction]),
        deciduous=np.array([deciduous]),
        index=None,
    )
    return InitialConditions(**{name: float(values[0]) for name, values in converted.items()})


def to_sipnet_initial_conditions_table(
    state: xr.Dataset | Mapping[str, xr.DataArray],
    *,
    leaf_carbon_per_area: float | xr.DataArray,
    fine_root_fraction: float | xr.DataArray,
    coarse_root_fraction: float | xr.DataArray,
    deciduous: bool | xr.DataArray,
) -> pd.DataFrame:
    """The conversion over a whole ``(member, site)`` ensemble, as a table.

    :func:`to_sipnet_initial_conditions` cell by cell: the same formulas and
    the same refusals, one row per cell. The prior predictive needs a parameter
    set for every member of every site it runs, and a table is what the
    ensemble layer feeds them from.

    Parameters
    ----------
    state:
        The initial conditions as a ``Dataset`` or as the ``dict`` of fields
        :func:`sipnet_calibration.initial_conditions.processed.initial_condition_fields`
        returns, in the product's units. Must
        carry ``initial_soil_organic_carbon``, ``initial_wood_carbon``,
        ``initial_leaf_carbon`` and ``initial_soil_moisture_saturation``; any
        other variable is ignored. Where a variable declares ``units``, they
        are checked against the spec.
    leaf_carbon_per_area, fine_root_fraction, coarse_root_fraction, deciduous:
        As in :func:`to_sipnet_initial_conditions`, each either a scalar or a
        ``DataArray`` over any subset of the dims of *state*, so that a
        parameter drawn per member and a PFT property held per site both
        broadcast. Where both sides label a dim, the labels must match exactly;
        nothing is filled or dropped. A dim carrying no coordinate is matched
        by position, as everywhere else in xarray, so label a parameter whose
        order you are not certain of.

    Returns
    -------
    pandas.DataFrame
        One row per cell, indexed by the dims the inputs broadcast to and
        always ordered ``(member, site)``, with
        :data:`CONVERTED_SIPNET_FIELDS` as columns. For any cell,
        ``InitialConditions(**table.loc[cell])`` equals what
        :func:`to_sipnet_initial_conditions` returns for it, so every row
        here also passes pySIPNET's own field validation.

    Raises
    ------
    KeyError
        If *state* lacks one of the four variables.
    TypeError
        If a value of *state* is not a ``DataArray``, or *deciduous* is not
        boolean.
    ValueError
        For the refusals of :func:`to_sipnet_initial_conditions`, naming the
        offending cells; if the inputs broadcast to dims other than ``member``
        and ``site``; if their indexes do not match, or they were selected for
        different members or sites; or if a variable's ``units`` are not the
        product's.

    Notes
    -----
    The whole ensemble does not convert. ``initial_wood_carbon`` is negative
    over much of it and ``initial_leaf_carbon`` is absent at some sites, so the
    product passed unfiltered is refused and the members to run have to be
    chosen first. See the Notes of :func:`to_sipnet_initial_conditions`.
    """
    arrays = {name: _state_variable(state, name) for name in _STATE_VARIABLES}
    arrays["leaf_carbon_per_area"] = _as_data_array(leaf_carbon_per_area)
    arrays["fine_root_fraction"] = _as_data_array(fine_root_fraction)
    arrays["coarse_root_fraction"] = _as_data_array(coarse_root_fraction)
    arrays["deciduous"] = _as_data_array(deciduous)

    _check_scalar_coordinates_agree(arrays)
    broadcast = xr.broadcast(*xr.align(*arrays.values(), join="exact"))
    _check_cells_are_member_and_site(broadcast[0])
    order = [dim for dim in (MEMBER, SITE) if dim in broadcast[0].dims]
    broadcast = [array.transpose(*order) for array in broadcast]
    template = broadcast[0]
    index = _cell_index(template)

    converted = _sipnet_fields_from_state(
        index=index,
        **{name: array.values.ravel() for name, array in zip(arrays, broadcast)},
    )
    table = pd.DataFrame(
        {name: converted[name] for name in CONVERTED_SIPNET_FIELDS},
        index=index if index is not None else pd.RangeIndex(1),
    )
    _check_cells_are_addressable(table)
    return table


#: Grams in a kilogram: the ensemble's carbon pools are kg m-2 and SIPNET's g m-2.
_KILOGRAM_IN_GRAMS = 1000.0

#: Percent to fraction, for the soil moisture.
_PERCENT_IN_ONE = 100.0

#: The least share of ``total_wood_carbon`` that may remain above ground once
#: the roots are split off. The conversion divides by that remainder, so a
#: remainder near zero turns a plausible aboveground pool into an implausible
#: total: at 0.01 the factor is already 100, against the 0.4 or so the
#: reanalysis PFTs use. Below this the result is not a wood pool any more, and
#: refusing it is cheaper than discovering it in a likelihood.
_MINIMUM_WOOD_FRACTION = 0.01

#: SIPNET's ``TINY`` (``sipnet/src/common/util.h``), which ``setupModel``
#: silently floors ``leafCSpWt`` at to avoid dividing by zero. A proposal below
#: it would be converted with one value and run with another.
_SIPNET_TINY = 1e-6

#: The four product variables the conversion reads. Its keyword arguments
#: carry the same names, so a caller's state maps onto them without a lookup.
_STATE_VARIABLES: tuple[str, ...] = (
    "initial_soil_organic_carbon",
    "initial_wood_carbon",
    "initial_leaf_carbon",
    "initial_soil_moisture_saturation",
)


def _sipnet_fields_from_state(
    *,
    initial_soil_organic_carbon: np.ndarray,
    initial_wood_carbon: np.ndarray,
    initial_leaf_carbon: np.ndarray,
    initial_soil_moisture_saturation: np.ndarray,
    leaf_carbon_per_area: np.ndarray,
    fine_root_fraction: np.ndarray,
    coarse_root_fraction: np.ndarray,
    deciduous: np.ndarray,
    index: pd.Index | None,
) -> dict[str, np.ndarray]:
    """Convert one flat array per input into one flat array per SIPNET field.

    Every input array holds one entry per cell and they are all the same
    length; every returned array is that length, in that order. This is where
    the formulas and the refusals live, and both public functions call it: the
    single-member form passes arrays of length one, the table form passes the
    broadcast ensemble.

    Parameters
    ----------
    index:
        Labels for the cells, used only to say which ones a refusal is about.
        ``None`` for a single cell with no label, as the single-member form
        passes.

    Returns
    -------
    dict
        :data:`CONVERTED_SIPNET_FIELDS` to its values.
    """
    soil = np.asarray(initial_soil_organic_carbon, dtype=float)
    wood = np.asarray(initial_wood_carbon, dtype=float)
    leaf = np.asarray(initial_leaf_carbon, dtype=float)
    wetness = np.asarray(initial_soil_moisture_saturation, dtype=float)
    leaf_carbon = np.asarray(leaf_carbon_per_area, dtype=float)
    fine = np.asarray(fine_root_fraction, dtype=float)
    coarse = np.asarray(coarse_root_fraction, dtype=float)
    deciduous = np.asarray(deciduous)

    _check_deciduous_is_boolean(deciduous)
    _check_state_is_physical(
        {
            "initial_soil_organic_carbon": soil,
            "initial_wood_carbon": wood,
            "initial_soil_moisture_saturation": wetness,
        },
        index,
    )
    # The leaf carbon is read only where the PFT keeps its leaves, so only there
    # does it have to be valid. Requiring a value the mapping never reads would
    # refuse the sites whose source files carry no leaf carbon at all, and the
    # members whose specific leaf area draw was negative -- every one of which
    # is at a grassland site, though most grassland members are unaffected.
    evergreen = ~deciduous
    if evergreen.any():
        _check_state_is_physical(
            {"initial_leaf_carbon": leaf[evergreen]},
            None if index is None else index[evergreen],
            population="cells whose PFT keeps its leaves",
        )
    if evergreen.any():
        _check_leaf_carbon_per_area_is_usable(
            leaf_carbon[evergreen], None if index is None else index[evergreen]
        )
    _check_soil_moisture_is_a_percentage(wetness, index)
    _check_root_fractions_leave_wood(fine, coarse, index)

    # Only the evergreen cells are computed: a deciduous cell's leaf carbon is
    # deliberately not validated, so it must not reach the arithmetic either.
    #
    # errstate holds the whole formula block so that what a caller sees does
    # not depend on their numpy error state: an overflow is reported by
    # _check_converted_values_are_finite as the documented ValueError, rather
    # than escaping as a RuntimeWarning or, under np.seterr(all="raise"), as a
    # FloatingPointError from whichever formula happened to overflow first.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        leaf_area_index = np.zeros(np.shape(leaf), dtype=float)
        leaf_area_index[evergreen] = (
            _KILOGRAM_IN_GRAMS * leaf[evergreen] / leaf_carbon[evergreen]
        )
        converted = {
            "total_wood_carbon": _KILOGRAM_IN_GRAMS * wood / (1.0 - fine - coarse),
            "leaf_area_index": leaf_area_index,
            "soil_carbon": _KILOGRAM_IN_GRAMS * soil,
            "soil_wetness_fraction": wetness / _PERCENT_IN_ONE,
            "fine_root_fraction": fine,
            "coarse_root_fraction": coarse,
        }
    _check_converted_values_are_finite(converted, index)
    return converted


def _state_variable(state: xr.Dataset | Mapping[str, xr.DataArray], name: str) -> xr.DataArray:
    """The named variable of *state*, checked to be a ``DataArray`` in the
    product's units."""
    try:
        array = state[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not in the state; the conversion reads {list(_STATE_VARIABLES)}."
        ) from None
    if not isinstance(array, xr.DataArray):
        raise TypeError(
            f"{name} is a {type(array).__name__}, not a DataArray. The table form "
            "converts an ensemble; use to_sipnet_initial_conditions for one member."
        )
    _check_units_are_the_products(array, name)
    return array


def _as_data_array(value: Any) -> xr.DataArray:
    """A parameter as a ``DataArray``: a scalar becomes a zero-dimensional one."""
    return value if isinstance(value, xr.DataArray) else xr.DataArray(value)


def _cell_index(array: xr.DataArray) -> pd.Index | None:
    """The table's row index, in the order ``array.values.ravel()`` produces.

    ``None`` when the inputs broadcast to no dimensions at all, which is one
    cell with nothing to label it by.
    """
    if not array.dims:
        return None
    levels = [
        array.coords[dim].values if dim in array.coords else np.arange(array.sizes[dim])
        for dim in array.dims
    ]
    names = [str(dim) for dim in array.dims]
    if len(levels) == 1:
        return pd.Index(levels[0], name=names[0])
    return pd.MultiIndex.from_product(levels, names=names)


def _abbreviate(labels: list[Any]) -> str:
    """A coordinate's labels for an error message, shortened past a handful."""
    if not labels:
        return "unlabeled"
    shown = ", ".join(str(label) for label in labels[:5])
    return shown if len(labels) <= 5 else f"{shown}, ... ({len(labels)} in all)"


def _offending_cells(
    index: pd.Index | None, bad: np.ndarray, values: np.ndarray, population: str = "cells"
) -> str:
    """The end of a refusal message: which cells are bad, or the bad value.

    Parameters
    ----------
    index:
        The cell labels, or ``None`` for a single unlabeled cell, in which case
        the message carries the offending value instead of a position.
    bad:
        Boolean mask over the cells *population* describes.
    population:
        What *bad* was computed over. Not always every cell: the leaf carbon is
        checked only where the PFT keeps its leaves, and a count against the
        whole ensemble would misstate how much of it was examined.
    """
    if index is None:
        return f" (value {values[bad][0]})"
    return (
        f", at {int(np.count_nonzero(bad))} of {bad.size} {population}, for example "
        f"{index[bad][:5].tolist()}"
    )


def _check_arguments_are_scalar(**arguments: Any) -> None:
    for name, value in arguments.items():
        if np.ndim(value) != 0:
            raise TypeError(
                f"{name} has {np.ndim(value)} dimensions; this form converts one "
                "member at one site. Use to_sipnet_initial_conditions_table for "
                "an ensemble."
            )


def _check_scalar_coordinates_agree(arrays: Mapping[str, xr.DataArray]) -> None:
    """Refuse inputs that were selected down to different members or sites.

    ``xr.align`` compares the indexes of dimensions, and ``.sel(member=0)``
    leaves ``member`` as a scalar coordinate on no dimension, which alignment
    therefore ignores. Two things have to be refused here, and broadcasting
    turns both into a full, plausible table:

    * two inputs selected to *different* single labels, which would be
      converted against each other;
    * one input selected to a single label while another still carries that
      dimension, which would replicate the selected cell across every label of
      the other and index the result by labels its state never came from.
    """
    scalars: dict[str, tuple[str, Any]] = {}
    dimensioned: dict[str, tuple[str, list[Any]]] = {}
    for name, array in arrays.items():
        for coordinate in (MEMBER, SITE):
            if coordinate in array.dims:
                labels = (
                    array.coords[coordinate].values.tolist()
                    if coordinate in array.coords
                    else []
                )
                dimensioned.setdefault(coordinate, (name, labels))
            elif coordinate in array.coords and array.coords[coordinate].ndim == 0:
                value = array.coords[coordinate].item()
                held_by, held = scalars.setdefault(coordinate, (name, value))
                if held != value:
                    raise ValueError(
                        f"{held_by} is for {coordinate} {held} and {name} for "
                        f"{coordinate} {value}. Selecting a single {coordinate} with "
                        "`.sel` leaves it as a scalar coordinate, which alignment does "
                        "not compare, so these would otherwise have been converted "
                        "against each other."
                    )

    for coordinate, (scalar_name, value) in scalars.items():
        if coordinate not in dimensioned:
            continue
        dim_name, labels = dimensioned[coordinate]
        if labels == [value]:
            continue
        raise ValueError(
            f"{scalar_name} is for {coordinate} {value} alone, but {dim_name} still has "
            f"a {coordinate} dimension ({_abbreviate(labels)}). Broadcasting would "
            f"repeat {coordinate} {value} across every one of them and label the rows "
            f"with the others, so select both sides to the same {coordinate}s, or "
            "neither."
        )


def _check_deciduous_is_boolean(values: np.ndarray) -> None:
    if values.dtype != np.bool_:
        raise TypeError(
            f"deciduous is {values.dtype}, expected boolean. It says whether the site's "
            "PFT drops its leaves; casting a numeric or object array to bool would make "
            "every non-zero value, NaN included, deciduous and silently zero the LAI."
        )


def _check_state_is_physical(
    values: Mapping[str, np.ndarray], index: pd.Index | None, population: str = "cells"
) -> None:
    for name, array in values.items():
        bad = ~np.isfinite(array) | (array < 0.0)
        if not bad.any():
            continue
        raise ValueError(
            f"{name} is negative, NaN or infinite"
            f"{_offending_cells(index, bad, array, population)}. "
            "The conversion takes physically valid state only. The ensemble's negative "
            "wood and leaf members and the sites where a variable is absent are for the "
            "initial condition prior to resolve; a unit conversion may not floor, "
            "substitute or drop them."
        )


def _check_leaf_carbon_per_area_is_usable(values: np.ndarray, index: pd.Index | None) -> None:
    bad = ~np.isfinite(values) | (values < _SIPNET_TINY)
    if bad.any():
        raise ValueError(
            f"leaf_carbon_per_area is not finite and at least {_SIPNET_TINY:g}"
            f"{_offending_cells(index, bad, values, 'cells whose PFT keeps its leaves')}. "
            "It divides the leaf carbon to give the initial LAI, and SIPNET's leafCSpWt "
            "is positive by definition. The floor is SIPNET's own TINY, which setupModel "
            "silently raises leafCSpWt to: below it the run recovers "
            "laiInit x leafCSpWt with the floored value, so the leaf carbon SIPNET "
            "starts from is not the one converted here. A specific leaf area draw that "
            "small has to be excluded by the prior rather than absorbed here."
        )


def _check_soil_moisture_is_a_percentage(values: np.ndarray, index: pd.Index | None) -> None:
    bad = values > _PERCENT_IN_ONE
    if bad.any():
        raise ValueError(
            "initial_soil_moisture_saturation is above 100"
            f"{_offending_cells(index, bad, values)}. The product holds a percent of "
            "saturation, whose source is documented over 0 to 100, and dividing by 100 "
            "is what makes soilWFracInit a fraction. A value above 100 is either a "
            "different quantity or a unit that is not percent.\n"
            "Note that the other end cannot be checked: a value already expressed as a "
            "0-1 fraction is a valid percentage too, and converts to a hundredth of "
            "what was meant. Declaring `units` on the input is what catches that."
        )


def _check_root_fractions_leave_wood(
    fine: np.ndarray, coarse: np.ndarray, index: pd.Index | None
) -> None:
    for name, values in (("fine_root_fraction", fine), ("coarse_root_fraction", coarse)):
        bad = ~np.isfinite(values) | (values < 0.0) | (values > 1.0)
        if bad.any():
            raise ValueError(
                f"{name} is outside [0, 1] or not finite"
                f"{_offending_cells(index, bad, values)}. It is a share of the total wood "
                "pool."
            )
    total = fine + coarse
    bad = (1.0 - fine - coarse) < _MINIMUM_WOOD_FRACTION
    if bad.any():
        raise ValueError(
            "fine_root_fraction + coarse_root_fraction must be below "
            f"{1.0 - _MINIMUM_WOOD_FRACTION:g}"
            f"{_offending_cells(index, bad, total)}. SIPNET's initial wood pool is "
            "total_wood_carbon x (1 - fine - coarse), and the conversion divides by that "
            "remainder. At a sum of 1 or more it is zero or negative, and SIPNET runs a "
            "negative wood pool to completion: exit code 0, a full output file, empty "
            "stderr. Just below 1 it is finite but absurd -- a sum of 1 - 1e-16 "
            "multiplies the aboveground pool by 1e16 -- which nothing downstream would "
            "catch either, so the guard is a floor on the remainder rather than on the "
            "sum. pySIPNET validates the two fractions separately and not their sum at "
            "all (TARPS-group/pySIPNET#39)."
        )


def _check_cells_are_member_and_site(array: xr.DataArray) -> None:
    extra = [str(dim) for dim in array.dims if dim not in (MEMBER, SITE)]
    if extra:
        raise ValueError(
            f"the inputs broadcast to dims {[str(dim) for dim in array.dims]}, but the "
            f"conversion is over (member, site) cells and {extra} is not among them. A "
            "parameter varying over anything else has to be selected down first."
        )


def _check_cells_are_addressable(table: pd.DataFrame) -> None:
    if table.index.is_unique:
        return
    repeated = table.index[table.index.duplicated()].unique().tolist()
    raise ValueError(
        f"the inputs repeat {_abbreviate(repeated)}, so the table's rows cannot be "
        "addressed one cell at a time: `table.loc[cell]` would return several rows and "
        "the documented InitialConditions(**table.loc[cell]) would fail. Select each "
        "member and site once."
    )


def _check_units_are_the_products(array: xr.DataArray, name: str) -> None:
    spec = resolve_initial_condition(name)
    units = array.attrs.get("units")
    if units is not None and units != spec.units:
        raise ValueError(
            f"{name} carries units {units!r}, not the product's {spec.units!r}. The "
            "conversion applies the change to SIPNET's own units itself, so values "
            "converted already would be scaled twice."
        )


def _check_converted_values_are_finite(
    values: Mapping[str, np.ndarray], index: pd.Index | None
) -> None:
    for name, array in values.items():
        bad = ~np.isfinite(array)
        if not bad.any():
            continue
        raise ValueError(
            f"the conversion produced a {name} that is not finite"
            f"{_offending_cells(index, bad, array)}. The inputs were all finite, so the "
            "overflow is in the formula -- a pool large enough that the factor of 1000 "
            "leaves the float range, or a divisor small enough to push past it. This "
            "catches only what overflows to infinity: a root-fraction sum just below 1 "
            "yields a finite but absurd wood pool, which passes here and which only a "
            "prior on the fractions can exclude. pySIPNET would refuse a non-finite "
            "value, and a table may not carry a row the single-member form would not "
            "return."
        )
