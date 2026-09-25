"""Observation operators: from one site's model output to a prediction of one
observed quantity, on that quantity's own grid.

Where this sits
---------------
An operator reads what SIPNET wrote (through
:func:`sipnet_calibration.fields.label_run`, a labeled ``xr.Dataset``) and,
where needed, the SIPNET parameter values the run used (a row of a SIPNET
table from :class:`~sipnet_calibration.parameter_vector.ParameterVector`),
and returns what the instrument would have read. The
:class:`~sipnet_calibration.observation.vector.ObservationVector` calls it,
converts the result into the observation's units and checks it; a
predictive-check figure calls it directly. In the notation of the inference
design it is the output-to-observation map, one per observed product.

The contract
------------
:class:`ObservationOperator` is a protocol. An operator declares
``output_variable_names``, the pySIPNET output variables it reads (so a worker
reads only those columns), and ``sipnet_parameter_names``, the SIPNET
parameters it reads (so the parameter vector can be checked to set them). Its
call is::

    operator(model_output, observed_values, *, sipnet_parameters=None) -> DataArray

* ``model_output`` is an ``xr.Dataset`` of pySIPNET-named variables on
  ``(time,)`` with a scalar ``site``, on ``(site, time)``, or on
  ``(member, site, time)``, carrying pySIPNET's time coordinates and
  attributes.
* ``observed_values`` is the observed array of one product, ``(site[, time])``,
  read for its ``site`` and ``time`` coordinates, its time bounds where
  present, and nothing else; never for its values.
* ``sipnet_parameters`` is the SIPNET table for these ``(member, site)``, a
  ``(site,)`` table, or a mapping of scalars for one run.
* The result is on ``observed_values``' ``site`` and ``time`` grid, with the
  model output's ``member`` if any, and carries ``units`` and, where the
  quantity has one, ``constituent`` attributes saying what it is. It need not
  be in the observation's units.
* The operator is **pointwise in site and member**: applied to a stack it
  equals itself applied to each slice. That is what lets it run on the worker.
  :func:`check_operator` tests it.

The library ships four operators, each a frozen dataclass named for what it
does, and one default binding, :data:`DEFAULT_OBS_OPS`, which holds only the
product whose construction is documented. How to read the model for the other
products is a modeling decision the experiment writes in its ``config.py``.

Usage
-----
::

    from sipnet_calibration.observation import DEFAULT_OBS_OPS, SelectTimestep
    from pysipnet.units import convert_dataarray_units

    lai = DEFAULT_OBS_OPS["modis_leaf_area_index"]
    predicted = lai(model_output, observed_lai, sipnet_parameters=table)
    predicted = convert_dataarray_units(predicted, to_units=observed_lai.attrs["units"])

    wood = SelectTimestep("wood_carbon")          # the state at each observed label
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from pysipnet.parameters.model import (
    PARAMETER_SPECS,
    SIPNET_PARAMS_BY_GROUP,
    resolve_parameter_name,
)
from pysipnet.variables import resolve_output_variable

from sipnet_calibration.observation.alignment import (
    WINDOW_REDUCTIONS,
    reduce_windows,
    run_window,
    select_timestep_at,
    windows_from_time_bounds,
)
from sipnet_calibration.observation.units import divide

__all__ = [
    "DEFAULT_OBS_OPS",
    "ComputeLeafAreaIndex",
    "ObservationOperator",
    "ReduceOverRun",
    "ReduceOverTimeBounds",
    "SelectTimestep",
    "check_operator",
    "select_sites",
    "sipnet_parameter_array",
    "sipnet_parameter_spec",
]

SITE = "site"
MEMBER = "member"
TIME = "time"


@runtime_checkable
class ObservationOperator(Protocol):
    """Model output to a prediction of one observed quantity, on that quantity's grid."""

    output_variable_names: tuple[str, ...]
    sipnet_parameter_names: tuple[str, ...]

    def __call__(
        self,
        model_output: xr.Dataset,
        observed_values: xr.DataArray,
        *,
        sipnet_parameters: xr.Dataset | Mapping[str, Any] | None = None,
    ) -> xr.DataArray: ...


@dataclass(frozen=True)
class SelectTimestep:
    """The value of one model variable at the timestep containing each observed label.

    For each label in ``observed_values.time`` the model timestep whose
    interval ``(time_step_start, time]`` contains it is read: the state at
    the end of that step for a pool, the mean over it for a step mean or a
    rate. The label comes only from the observation. A per-step total is
    refused; make it a rate first.
    """

    output_variable_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_variable_name", _output_name(self.output_variable_name))

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return (self.output_variable_name,)

    @property
    def sipnet_parameter_names(self) -> tuple[str, ...]:
        return ()

    def __call__(self, model_output, observed_values, *, sipnet_parameters=None) -> xr.DataArray:
        variable = select_sites(model_output[self.output_variable_name], observed_values)
        return select_timestep_at(variable, observed_values[TIME])


@dataclass(frozen=True)
class ReduceOverTimeBounds:
    """One model variable reduced over each observation's own time bounds.

    Requires ``observed_values`` to carry ``time_bounds_start`` and
    ``time_bounds_end``, which an annual product's array does. ``how`` is one
    of the window reductions, checked against the variable's kind.
    """

    output_variable_name: str
    how: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_variable_name", _output_name(self.output_variable_name))
        _check_how(self.how)

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return (self.output_variable_name,)

    @property
    def sipnet_parameter_names(self) -> tuple[str, ...]:
        return ()

    def __call__(self, model_output, observed_values, *, sipnet_parameters=None) -> xr.DataArray:
        variable = select_sites(model_output[self.output_variable_name], observed_values)
        windows = windows_from_time_bounds(observed_values)
        return reduce_windows(variable, windows, self.how, labels=observed_values[TIME])


@dataclass(frozen=True)
class ReduceOverRun:
    """One model variable reduced over the whole run, for a static observation.

    ``observed_values`` is ``(site,)``: it documents no time, so the model is
    reduced over its entire record and the result has no ``time`` dimension.
    """

    output_variable_name: str
    how: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_variable_name", _output_name(self.output_variable_name))
        _check_how(self.how)

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return (self.output_variable_name,)

    @property
    def sipnet_parameter_names(self) -> tuple[str, ...]:
        return ()

    def __call__(self, model_output, observed_values, *, sipnet_parameters=None) -> xr.DataArray:
        if TIME in observed_values.dims:
            raise ValueError(
                f"ReduceOverRun reads a static observation, and {observed_values.name!r} "
                f"has a {TIME!r} dimension; use ReduceOverTimeBounds or SelectTimestep."
            )
        variable = select_sites(model_output[self.output_variable_name], observed_values)
        reduced = reduce_windows(variable, run_window(variable), self.how)
        return reduced.isel({TIME: 0}, drop=True)


@dataclass(frozen=True)
class ComputeLeafAreaIndex:
    """Leaf carbon over leaf carbon per area, at the timestep containing each label.

    SIPNET tracks leaf carbon, not leaf area; its own leaf area index is
    ``plantLeafC / leafCSpWt`` (``sipnet.c``), so the prediction depends on
    the parameter ``leaf_carbon_per_area`` the run used. The ratio is
    dimensionless and carries no constituent; the observation's ``m2 m-2``
    converts from it by a factor of one.
    """

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return ("leaf_carbon",)

    @property
    def sipnet_parameter_names(self) -> tuple[str, ...]:
        return ("leaf_carbon_per_area",)

    def __call__(self, model_output, observed_values, *, sipnet_parameters=None) -> xr.DataArray:
        leaf_carbon = select_sites(model_output["leaf_carbon"], observed_values)
        per_area = sipnet_parameter_array(sipnet_parameters, "leaf_carbon_per_area", leaf_carbon)
        lai = divide(leaf_carbon, per_area)
        lai.name = "leaf_area_index"
        return select_timestep_at(lai, observed_values[TIME])


#: The library's default operator per product: only the bindings whose
#: construction is documented (see ``data/README.md`` and the constraint
#: specs). How to read the model for the other products is a modeling
#: decision, and an experiment binds its own in ``config.py``.
DEFAULT_OBS_OPS: Mapping[str, ObservationOperator] = {
    "modis_leaf_area_index": ComputeLeafAreaIndex(),
}


def select_sites(variable: xr.DataArray, observed_values: xr.DataArray) -> xr.DataArray:
    """*variable* at the observation's sites, raising if any is missing.

    A model array with a ``site`` dimension is selected down to the
    observation's sites, in the observation's order. One with a scalar
    ``site`` coordinate must be the observation's single site. One with no
    site at all is refused: the correspondence would be a guess.
    """
    wanted = np.asarray(observed_values[SITE].values).ravel()
    who = _name_of(observed_values)
    if len(set(wanted.tolist())) != wanted.size:
        raise ValueError(f"{who} repeats a site; an observation names each site once.")
    if SITE in variable.dims:
        have = set(variable[SITE].values.tolist())
        missing = [int(s) for s in wanted if s not in have]
        if missing:
            raise ValueError(
                f"the model output has no site(s) {missing[:10]} that {who} observes; "
                "run the model at every observed site, or select the observations to "
                "the sites that were run."
            )
        return variable.sel({SITE: wanted})
    if SITE in variable.coords:
        site = int(variable[SITE].values)
        if wanted.size != 1 or int(wanted[0]) != site:
            raise ValueError(
                f"the model output is one run at site {site}, and {who} observes "
                f"site(s) {wanted.tolist()[:10]}; select the observation to that one site."
            )
        return variable
    raise ValueError(
        f"the model output carries no {SITE!r} coordinate, so it cannot be matched to "
        f"the sites {who} observes; label the run with fields.label_run(site=...)."
    )


def sipnet_parameter_array(
    sipnet_parameters: xr.Dataset | Mapping[str, Any] | None, name: str, like: xr.DataArray
) -> xr.DataArray:
    """One SIPNET parameter's values as a ``DataArray`` with pySIPNET's units.

    From a SIPNET table (a ``Dataset`` on ``(member, site)`` or ``(site,)``,
    selected to *like*'s sites), or from a mapping of scalars for one run.
    The result carries ``units`` and ``constituent`` from
    ``pysipnet.parameters.model.PARAMETER_SPECS``, so the arithmetic verbs can
    combine it with a model variable.
    """
    spec = sipnet_parameter_spec(name)
    attrs = {"units": spec.units, "long_name": spec.long_label}
    if spec.constituent:
        attrs["constituent"] = spec.constituent
    if sipnet_parameters is None:
        raise ValueError(
            f"this operator reads the SIPNET parameter {name!r}; pass sipnet_parameters= "
            "(a SIPNET table or a mapping of the run's values)."
        )
    if isinstance(sipnet_parameters, xr.Dataset):
        if name not in sipnet_parameters.data_vars:
            raise ValueError(
                f"the SIPNET table has no variable {name!r}, which this operator reads; "
                f"it has {list(sipnet_parameters.data_vars)[:10]}."
            )
        array = sipnet_parameters[name]
        selectors = {}
        for dim in (SITE, MEMBER):
            if dim in array.dims and dim in like.coords:
                wanted = np.asarray(like[dim].values).ravel()
                missing = [x for x in wanted.tolist() if x not in set(array[dim].values.tolist())]
                if missing:
                    raise ValueError(
                        f"the SIPNET table's {name!r} has no {dim} label(s) {missing[:10]} "
                        f"that the model output has; the table and the runs must cover the "
                        f"same {dim}s."
                    )
                selectors[dim] = wanted
        if selectors:
            array = array.sel(selectors)
        array = array.copy()
        array.attrs = attrs
        array.name = name
        return array
    flat = resolve_parameter_name(name)
    value = None
    for key in (name, flat, spec.sipnet_name):
        try:
            value = sipnet_parameters[key]
            break
        except (KeyError, TypeError):
            continue
    if value is None:
        raise ValueError(f"sipnet_parameters has no entry {name!r}, which this operator reads.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(
            f"sipnet_parameters[{name!r}] must be a number for one run, got "
            f"{type(value).__name__}; pass a SIPNET table for several runs."
        )
    return xr.DataArray(float(value), name=name, attrs=attrs)


def sipnet_parameter_spec(name: str) -> Any:
    """pySIPNET's ``ParameterSpec`` for a flat parameter name or alias."""
    flat = resolve_parameter_name(name)
    for group, names in SIPNET_PARAMS_BY_GROUP.items():
        if flat in names:
            return PARAMETER_SPECS[f"{group}.{flat}"]
    raise ValueError(f"{name!r} resolves to {flat!r}, which no parameter group holds.")


def check_operator(
    operator: ObservationOperator,
    model_output: xr.Dataset,
    observed_values: xr.DataArray,
    *,
    sipnet_parameters: xr.Dataset | Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """Check an operator against its contract on real inputs, and return its result.

    Checks that the declared names resolve in pySIPNET's registries and are
    present in *model_output*; that the result is on *observed_values*' grid
    and carries ``units``; and, when *model_output* has ``site`` or ``member``
    dimensions, that the operator is pointwise: its value on the stack equals
    its value on each slice. Raises ``ValueError`` naming the first broken
    rule.
    """
    _check_declarations(operator, model_output, sipnet_parameters)
    result = operator(model_output, observed_values, sipnet_parameters=sipnet_parameters)
    _check_result_grid(result, observed_values, model_output, operator)
    _check_pointwise(operator, model_output, observed_values, sipnet_parameters, result)
    return result


# ── supporting helpers ────────────────────────────────────────────────────────


def _name_of(observed_values: xr.DataArray) -> str:
    """How an observed array is called in a message."""
    return repr(observed_values.name) if observed_values.name is not None else "the observation"


def _output_name(name: str) -> str:
    try:
        return resolve_output_variable(name).name
    except KeyError as error:
        raise ValueError(
            f"{name!r} is not a pySIPNET output variable or alias: {error}"
        ) from None


def _check_how(how: str) -> None:
    if how not in WINDOW_REDUCTIONS:
        raise ValueError(f"how must be one of {list(WINDOW_REDUCTIONS)}, got {how!r}.")


def _check_declarations(operator: Any, model_output: xr.Dataset, sipnet_parameters: Any) -> None:
    for attribute in ("output_variable_names", "sipnet_parameter_names"):
        names = getattr(operator, attribute, None)
        if not isinstance(names, tuple) or not all(isinstance(n, str) for n in names):
            raise ValueError(
                f"{type(operator).__name__}.{attribute} must be a tuple of names, got {names!r}."
            )
    for name in operator.output_variable_names:
        if resolve_output_variable(name).name != name:
            raise ValueError(
                f"{type(operator).__name__} declares {name!r}, which is an alias; declare "
                f"the registry name {resolve_output_variable(name).name!r}."
            )
        if name not in model_output.data_vars:
            raise ValueError(
                f"{type(operator).__name__} reads {name!r}, which the model output does "
                f"not carry; it has {list(model_output.data_vars)[:10]}."
            )
    for name in operator.sipnet_parameter_names:
        resolve_parameter_name(name)
    if operator.sipnet_parameter_names and sipnet_parameters is None:
        raise ValueError(
            f"{type(operator).__name__} reads SIPNET parameters "
            f"{list(operator.sipnet_parameter_names)}; pass sipnet_parameters=."
        )


def _check_result_grid(
    result: Any, observed_values: xr.DataArray, model_output: xr.Dataset, operator: Any
) -> None:
    who = type(operator).__name__
    if not isinstance(result, xr.DataArray):
        raise ValueError(f"{who} returned {type(result).__name__}, not a DataArray.")
    if not isinstance(result.attrs.get("units"), str):
        raise ValueError(f"{who}'s result carries no 'units' attribute.")
    if MEMBER in result.dims and MEMBER not in model_output.dims:
        raise ValueError(f"{who}'s result has a member dimension the model output lacks.")
    wanted_sites = np.asarray(observed_values[SITE].values).ravel().tolist()
    if SITE in result.dims:
        if result[SITE].values.tolist() != wanted_sites:
            raise ValueError(f"{who}'s result is not on the observation's sites, in order.")
    elif SITE in result.coords:
        if [int(result[SITE].values)] != wanted_sites:
            raise ValueError(f"{who}'s result is not at the observation's site.")
    else:
        raise ValueError(f"{who}'s result carries no site.")
    if TIME in observed_values.dims:
        if TIME not in result.dims or not np.array_equal(
            result[TIME].values, observed_values[TIME].values
        ):
            raise ValueError(f"{who}'s result is not on the observation's time labels.")
    elif TIME in result.dims:
        raise ValueError(f"{who}'s result has a time dimension for a static observation.")


def _check_pointwise(
    operator: Any,
    model_output: xr.Dataset,
    observed_values: xr.DataArray,
    sipnet_parameters: Any,
    result: xr.DataArray,
) -> None:
    observed_sites = np.asarray(observed_values[SITE].values).ravel().tolist()
    for dim in (MEMBER, SITE):
        if dim not in model_output.dims or model_output.sizes[dim] < 2:
            continue
        if dim == SITE:
            if len(observed_sites) < 2 or SITE not in observed_values.dims:
                continue
            label = observed_sites[-1]
            slice_observed = observed_values.sel({SITE: [label]})
        else:
            label = model_output[dim].values[-1]
            slice_observed = observed_values
        slice_output = model_output.sel({dim: label})
        slice_parameters = sipnet_parameters
        if isinstance(sipnet_parameters, xr.Dataset) and dim in sipnet_parameters.dims:
            slice_parameters = sipnet_parameters.sel({dim: label})
        expected = operator(slice_output, slice_observed, sipnet_parameters=slice_parameters)
        got = result.sel({dim: label})
        same = np.allclose(
            np.asarray(got.values, float), np.asarray(expected.values, float), equal_nan=True
        )
        if not same:
            raise ValueError(
                f"{type(operator).__name__} is not pointwise in {dim!r}: its value on the "
                f"stack differs from its value on the {dim}={label!r} slice alone."
            )
