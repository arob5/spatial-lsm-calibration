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
predictive-check figure calls it directly. It is the map from model output to
one observed product.

The contract
------------
:class:`ObservationOperator` is a protocol. An operator declares
``output_variable_names``, the pySIPNET output variables it reads (so a worker
reads only those columns), and ``sipnet_parameter_names``, the SIPNET
parameters it reads (so the forward model can supply them: from the run's
values, or from the base parameter set where the parameter vector does not set
them). Its call is::

    operator(model_output, observed_values, *, sipnet_parameters=None) -> DataArray

* ``model_output`` is an ``xr.Dataset`` of pySIPNET-named variables on
  ``(time,)`` with a scalar ``site``, on ``(site, time)``, or on
  ``(*batch, site, time)`` with any batch dims (``sample``, a data source's
  ``<source>_member``), carrying pySIPNET's time coordinates and attributes.
* ``observed_values`` is the observed array of one product, ``(site[, time])``,
  read for its ``site`` and ``time`` coordinates, its windows where
  present, and nothing else; never for its values.
* ``sipnet_parameters`` is the SIPNET table for these ``(*batch, site)``, a
  ``(site,)`` table, or a mapping of scalars for one run.
* The result is on ``observed_values``' ``site`` and ``time`` grid, with the
  model output's batch dims if any and no dim that neither the model output nor
  the observation has, and carries ``units`` and, where the quantity has one,
  ``constituent`` attributes saying what it is. It need not be in the
  observation's units.
* The operator is **pointwise in site and in every batch dim**: applied to a
  stack it equals itself applied to each slice. That is what lets it run on
  the worker. :func:`check_operator` tests it.

The library ships four operators, each a frozen dataclass named for what it
does, and one default binding, :data:`DEFAULT_OBS_OPS`, which holds only the
product whose construction is established from a primary source. How to read
the model for the other products is a modeling decision the experiment writes
in its ``config.py``.

The checks at the bottom of this module are the contract's, and
:class:`~sipnet_calibration.observation.vector.ObservationVector` applies the
same ones: :func:`check_operator_declares_names`,
:func:`check_model_output_carries_what_is_read` and
:func:`check_result_is_on_the_observation_grid`.

Usage
-----
::

    from sipnet_calibration.observation import DEFAULT_OBS_OPS, SelectTimestep
    from pysipnet.units import convert_dataarray_units

    lai = DEFAULT_OBS_OPS["modis_leaf_area_index"]
    predicted = lai(model_output, observed_lai, sipnet_parameters=sipnet_table)
    predicted = convert_dataarray_units(predicted, to_units=observed_lai.attrs["units"])

    wood = SelectTimestep("wood_carbon")          # the state at each observed label
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from pysipnet.arithmetic import divide_with_units
from pysipnet.parameters.model import parameter_dataarray, resolve_parameter_name
from pysipnet.variables import resolve_output_variable

from sipnet_calibration.conventions import SITE, TIME, FrozenMapping
from sipnet_calibration.fields import (
    STACKED_LABEL_SUFFIX,
    batch_dims,
    coordinate_labels,
    field_label,
    missing_labels,
    recorded_stacked_dims,
    scalar_batch_labels,
)
from sipnet_calibration.observation.time_alignment import (
    check_how_is_a_window_reduction,
    check_run_spans_the_windows,
    reduce_windows,
    run_window,
    select_timestep_at,
    windows_from_observed_values,
)
from sipnet_calibration.validation import check_site_ids_are_unique

__all__ = [
    "DEFAULT_OBS_OPS",
    "ComputeLeafAreaIndex",
    "ObservationOperator",
    "ReduceOverRun",
    "ReduceOverWindows",
    "SelectTimestep",
    "check_model_output_carries_what_is_read",
    "check_operator",
    "check_operator_declares_names",
    "check_result_is_on_the_observation_grid",
    "extract_sipnet_parameter_at_coords",
    "select_observed_sites",
]

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

    Parameters
    ----------
    output_variable_name:
        The pySIPNET output variable to read: a registry name or an alias,
        stored as the registry name.

    Raises
    ------
    ValueError
        On construction, if *output_variable_name* is not a pySIPNET output
        variable. On a call, if the observation has no ``time`` dimension
        (a static product is read by :class:`ReduceOverRun`), or for any
        refusal of :func:`select_observed_sites` or
        :func:`~sipnet_calibration.observation.time_alignment.select_timestep_at`.
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
        check_observation_is_dated(observed_values, type(self).__name__)
        variable = select_observed_sites(model_output[self.output_variable_name], observed_values)
        return select_timestep_at(variable, observed_values[TIME])


@dataclass(frozen=True)
class ReduceOverWindows:
    """One model variable reduced over each observation's own window.

    Requires ``observed_values`` to carry ``window_start`` and
    ``window_end``, which an annual constraint's field does.

    Parameters
    ----------
    output_variable_name:
        The pySIPNET output variable to read: a registry name or an alias,
        stored as the registry name.
    how:
        One of
        :data:`~sipnet_calibration.observation.time_alignment.WINDOW_REDUCTIONS`,
        checked against the variable's kind on a call.

    Raises
    ------
    TypeError
        On construction, if *how* is not a string.
    ValueError
        On construction, if *output_variable_name* is not a pySIPNET output
        variable or *how* is not a window reduction. On a call, if the
        observed values carry no windows; if a window reaches a step or more
        beyond the model record, which would reduce over part of it
        (:func:`~sipnet_calibration.observation.time_alignment.check_run_spans_the_windows`);
        or for any refusal of
        :func:`select_observed_sites` or
        :func:`~sipnet_calibration.observation.time_alignment.reduce_windows`.
    """

    output_variable_name: str
    how: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_variable_name", _output_name(self.output_variable_name))
        check_how_is_a_window_reduction(self.how, type(self).__name__)

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return (self.output_variable_name,)

    @property
    def sipnet_parameter_names(self) -> tuple[str, ...]:
        return ()

    def __call__(self, model_output, observed_values, *, sipnet_parameters=None) -> xr.DataArray:
        variable = select_observed_sites(model_output[self.output_variable_name], observed_values)
        windows = windows_from_observed_values(observed_values)
        observation = field_label(observed_values, "the observation")
        check_run_spans_the_windows(variable, windows, f"{type(self).__name__} on {observation}")
        return reduce_windows(variable, windows, self.how, labels=observed_values[TIME])


@dataclass(frozen=True)
class ReduceOverRun:
    """One model variable reduced over the whole run, for a static observation.

    ``observed_values`` is ``(site,)``: it documents no time, so the model is
    reduced over its entire record and the result has no ``time`` dimension.

    Parameters
    ----------
    output_variable_name:
        The pySIPNET output variable to read: a registry name or an alias,
        stored as the registry name.
    how:
        One of
        :data:`~sipnet_calibration.observation.time_alignment.WINDOW_REDUCTIONS`,
        checked against the variable's kind on a call.

    Raises
    ------
    TypeError
        On construction, if *how* is not a string.
    ValueError
        On construction, if *output_variable_name* is not a pySIPNET output
        variable or *how* is not a window reduction. On a call, if the
        observation has a ``time`` dimension, or for any refusal of
        :func:`select_observed_sites` or
        :func:`~sipnet_calibration.observation.time_alignment.reduce_windows`.
    """

    output_variable_name: str
    how: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_variable_name", _output_name(self.output_variable_name))
        check_how_is_a_window_reduction(self.how, type(self).__name__)

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return (self.output_variable_name,)

    @property
    def sipnet_parameter_names(self) -> tuple[str, ...]:
        return ()

    def __call__(self, model_output, observed_values, *, sipnet_parameters=None) -> xr.DataArray:
        check_observation_is_static(observed_values, type(self).__name__)
        variable = select_observed_sites(model_output[self.output_variable_name], observed_values)
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

    Raises
    ------
    TypeError
        On a call, if a mapping of SIPNET parameters holds a value that is
        not a number.
    ValueError
        On a call, if the observation has no ``time`` dimension, or for any
        refusal of :func:`select_observed_sites`,
        :func:`extract_sipnet_parameter_at_coords` or
        :func:`~sipnet_calibration.observation.time_alignment.select_timestep_at`.
    """

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return ("leaf_carbon",)

    @property
    def sipnet_parameter_names(self) -> tuple[str, ...]:
        return ("leaf_carbon_per_area",)

    def __call__(self, model_output, observed_values, *, sipnet_parameters=None) -> xr.DataArray:
        check_observation_is_dated(observed_values, type(self).__name__)
        leaf_carbon = select_observed_sites(model_output["leaf_carbon"], observed_values)
        per_area = extract_sipnet_parameter_at_coords(
            sipnet_parameters, "leaf_carbon_per_area", leaf_carbon
        )
        lai = divide_with_units(leaf_carbon, per_area)
        lai.name = "leaf_area_index"
        return select_timestep_at(lai, observed_values[TIME])


#: The library's default operator per product. It binds only the
#: constructions established from a primary source: today MODIS leaf area
#: index, as SIPNET's own ``plantLeafC / leafCSpWt`` (``sipnet.c``). How to
#: read the model for the other products is a modeling decision, and an
#: experiment binds its own in ``config.py``.
DEFAULT_OBS_OPS: Mapping[str, ObservationOperator] = FrozenMapping(
    {"modis_leaf_area_index": ComputeLeafAreaIndex()}
)


def select_observed_sites(
    source_field: xr.DataArray, target_field: xr.DataArray
) -> xr.DataArray:
    """Restrict a field to the sites another field is on.

    The operators call this first, with an output variable as the source and
    the observed values as the target, so that everything after it works on
    the observation's sites, in the observation's order. A source with a
    ``site`` dimension is selected down to the target's sites; a source from
    one run, carrying ``site`` as a scalar coordinate, is returned unchanged
    when it is the target's one site. Only the two fields' ``site``
    coordinates are read.

    Parameters
    ----------
    source_field:
        The field to restrict: typically one output variable of a model
        output, ``model_output[name]``, on ``(time,)`` with a scalar ``site``
        coordinate for one run, or on ``(site, time)`` or
        ``(*batch, site, time)`` for a stack. Its ``site`` labels are the
        site ids.
    target_field:
        The field whose sites the result is on: typically the observed values
        of one product, ``(site[, time])``.

    Returns
    -------
    xarray.DataArray
        *source_field* at the target's sites, in the order
        *target_field* lists them, with every other dimension, coordinate
        and attribute unchanged. For a one-run source, *source_field*
        itself.

    Raises
    ------
    ValueError
        If *target_field* lists a site twice; if *source_field* lacks a
        site the observation observes; if *source_field* is one run at a
        site other than the observation's one site, or the observation
        observes several sites; or if *source_field* carries no ``site``
        coordinate at all.

    Notes
    -----
    A source with no ``site`` is refused rather than assumed to be the
    observed site: an unlabeled run could be any site, and matching it by
    position would be a guess. :func:`sipnet_calibration.fields.label_run`
    is what gives a run its site.
    """
    wanted = coordinate_labels(target_field[SITE])
    message_name = field_label(target_field, "the observation")
    check_site_ids_are_unique(wanted, message_name=message_name)
    if SITE in source_field.dims:
        check_model_output_has_the_observed_sites(source_field, wanted, message_name)
        return source_field.sel({SITE: wanted})
    check_run_is_labeled_with_a_site(source_field, message_name)
    check_run_is_at_the_observed_site(source_field, wanted, message_name)
    return source_field


def extract_sipnet_parameter_at_coords(
    sipnet_parameters: xr.Dataset | Mapping[str, Any] | None,
    sipnet_parameter_name: str,
    target_field: xr.DataArray,
) -> xr.DataArray:
    """One SIPNET parameter's values at a field's site and batch coordinates.

    An operator that reads a SIPNET parameter, as the leaf area index
    operator reads ``leaf_carbon_per_area``, calls this to get the
    parameter's values lined up with the output variable it combines them
    with. The result is labeled by pySIPNET's
    :func:`~pysipnet.parameters.model.parameter_dataarray`, so
    :mod:`pysipnet.arithmetic` can combine it with the variable and keep the
    attributes right.

    Parameters
    ----------
    sipnet_parameters:
        The values the runs used, in either of two forms. A SIPNET table: an
        ``xr.Dataset`` with one variable per SIPNET parameter, under
        pySIPNET's flat names, on ``(*batch, site)`` or ``(site,)``, as
        :meth:`~sipnet_calibration.parameter_vector.ParameterVector.sipnet_table`
        returns it. Or, for one run, a mapping from SIPNET parameter name to
        a number, as :func:`~sipnet_calibration.parameter_vector.sipnet_overrides`
        returns it; a key may be the flat name, an alias, or SIPNET's own
        name.
    sipnet_parameter_name:
        The SIPNET parameter to select: pySIPNET's flat name
        (``"leaf_carbon_per_area"``), an alias of it, or SIPNET's own name.
    target_field:
        The field the values will be combined with, typically an output
        variable. Only its coordinate labels are read, for each dim of the
        table (``site`` and every batch dim), whether they are dimensions or
        scalar coordinates on the target: from a SIPNET table the values are
        taken at those labels, in that order, a scalar label selecting without
        keeping the dimension. A table dimension the target has no coordinate
        for is refused, since using it whole would broadcast it into the run,
        as is a scalar table label (``site`` or an integer batch label) the
        target's labels disagree with.

    Returns
    -------
    xarray.DataArray
        Named by the parameter's flat name, with the attributes
        ``parameter_dataarray`` gives it: ``units``, ``long_name``,
        ``description``, ``sipnet_name`` and, where pySIPNET declares one,
        ``constituent``. From a SIPNET table, the parameter's variable at
        *target_field*'s site and batch labels; from a mapping, a 0-d array of
        the run's value, which broadcasts against *target_field*.

    Raises
    ------
    TypeError
        If a mapping's entry for the parameter is not a number.
    ValueError
        If *sipnet_parameters* is ``None``; if a SIPNET table has no variable
        for the parameter, has a dimension the target has no coordinate for,
        lacks a label of it that *target_field* has, or carries a scalar label
        the target's disagree with (a stacked target's ``<dim>_label``
        coordinates included, or lost); if a batch dim of *target_field* is a
        stack of a dim the table has
        (:func:`~sipnet_calibration.fields.stack_batch_dims`), whose labels are
        not the table's; if a mapping has no entry for the parameter under any
        of its names; or if a value is not finite or lies outside the
        parameter's pySIPNET domain.
    KeyError
        If *sipnet_parameter_name* is not a pySIPNET parameter name or alias.
    """
    name = resolve_parameter_name(sipnet_parameter_name)
    if sipnet_parameters is None:
        raise ValueError(
            f"this operator reads the SIPNET parameter {name!r}; pass sipnet_parameters= "
            "(a SIPNET table or a mapping of the run's values)."
        )
    if isinstance(sipnet_parameters, xr.Dataset):
        return parameter_dataarray(name, _table_values_at(sipnet_parameters, name, target_field))
    return parameter_dataarray(name, _mapping_value(sipnet_parameters, name))


def check_operator(
    operator: ObservationOperator,
    model_output: xr.Dataset,
    observed_values: xr.DataArray,
    *,
    sipnet_parameters: xr.Dataset | Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """Check an operator against its contract on real inputs, and return its result.

    Checks that the declared names are pySIPNET registry names, not aliases,
    and that *model_output* carries the variables and *sipnet_parameters* is
    given where parameters are read; that the result is on *observed_values*'
    grid and carries ``units``; and, when *model_output* has a ``site`` dim
    or batch dims of two labels or more, that the operator is pointwise: its
    value on the stack equals, label by label, its value on the last slice of
    each of those dims alone.

    Parameters
    ----------
    operator:
        The operator to check.
    model_output, observed_values, sipnet_parameters:
        What the operator is called with, as in the contract above.

    Returns
    -------
    xarray.DataArray
        The operator's result on the whole of *model_output*, unconverted.

    Raises
    ------
    TypeError
        If the operator is not callable or its declarations are not tuples of
        strings, *model_output* is not a ``Dataset``, or the result is not a
        ``DataArray``.
    ValueError
        Naming the first rule broken: a declaration that names an alias; a
        variable the model output lacks, or parameters read and not given; a
        result without ``units``, off the observation's sites or time labels,
        or with a dim that neither the model output nor the observation has;
        or a result that is not pointwise. The operator's own refusals pass
        through.
    KeyError
        If a declared name is not in pySIPNET's registries.
    """
    message_name = type(operator).__name__
    check_operator_declares_names(operator)
    check_model_output_carries_what_is_read(
        model_output,
        output_variable_names=operator.output_variable_names,
        sipnet_parameter_names=operator.sipnet_parameter_names,
        sipnet_parameters=sipnet_parameters,
        message_name=message_name,
    )
    result = operator(model_output, observed_values, sipnet_parameters=sipnet_parameters)
    check_result_is_on_the_observation_grid(result, observed_values, model_output, message_name)
    check_operator_is_pointwise(operator, model_output, observed_values, sipnet_parameters, result)
    return result


# ── supporting helpers ────────────────────────────────────────────────────────


def _output_name(name: str) -> str:
    try:
        return resolve_output_variable(name).name
    except KeyError as error:
        raise ValueError(
            f"{name!r} is not a pySIPNET output variable or alias: {error}"
        ) from None


def _table_values_at(
    table: xr.Dataset, name: str, target_field: xr.DataArray
) -> xr.DataArray:
    """A SIPNET table's variable *name* at *target_field*'s labels on each of its dims.

    Every dim of the table's variable -- ``site`` and each batch dim, whatever
    it is named -- is selected at the target's labels or refused, so no table
    dim is ever broadcast into a run.
    """
    check_table_has_the_parameter(table, name)
    values = table[name]
    check_target_is_not_a_stack_of_table_dims(target_field, values, name)
    selectors: dict[str, Any] = {}
    for dim in map(str, values.dims):
        check_target_has_a_coordinate_for(target_field, dim, name)
        wanted = coordinate_labels(target_field[dim])
        check_table_has_the_labels(values, dim, wanted, name)
        # A scalar label selects without keeping the dimension, as the
        # target, one run, has none.
        selectors[dim] = wanted[0] if target_field[dim].ndim == 0 else wanted
    scalar_site = [SITE] if SITE in values.coords and values[SITE].ndim == 0 else []
    for dim in [*scalar_site, *scalar_batch_labels(values)]:
        check_scalar_table_label_agrees(values, target_field, dim, name)
    return values.sel(selectors) if selectors else values


def _is_stacked_into(target_field: xr.DataArray, dim: str, stacked_dim: str) -> bool:
    """Whether *dim* was stacked into the target's batch dim *stacked_dim*.

    The stack record on *stacked_dim*'s coordinate says so, and so does a
    ``<dim>_label`` coordinate on *stacked_dim*, whatever the record says,
    unless *dim* is *stacked_dim* itself.
    """
    # A restack records only the dim it stacked, and carries the labels of
    # the dims stacked before it on the new dim; an operation may also drop
    # the record. A label coordinate on a dim does not make a stack of itself.
    record = recorded_stacked_dims(target_field[stacked_dim])
    if record is not None and dim in record:
        return True
    label = f"{dim}{STACKED_LABEL_SUFFIX}"
    return (
        dim != stacked_dim
        and label in target_field.coords
        and target_field[label].dims == (stacked_dim,)
    )


def _labels_of(target_field: xr.DataArray, dim: str) -> xr.DataArray | None:
    """The target's labels of *dim*: its coordinate, or the ``<dim>_label`` of a stack of it."""
    if dim in target_field.coords:
        return target_field[dim]
    label = f"{dim}{STACKED_LABEL_SUFFIX}"
    return target_field[label] if label in target_field.coords else None


def _mapping_value(sipnet_parameters: Mapping[str, Any], name: str) -> float:
    """One run's value of *name* from a mapping keyed by any of its names."""
    value = None
    for key, candidate in sipnet_parameters.items():
        try:
            if resolve_parameter_name(str(key)) == name:
                value = candidate
                break
        except KeyError:
            continue
    if value is None:
        raise ValueError(
            f"sipnet_parameters has no entry {name!r}, which this operator reads; add it "
            "under its flat name, an alias or SIPNET's own name."
        )
    check_mapping_value_is_a_number(value, name)
    return float(value)


def _pointwise_slices(
    model_output: xr.Dataset, observed_values: xr.DataArray
) -> list[tuple[str, Any, xr.DataArray]]:
    """The ``(dim, label, observed slice)`` for the last slice of each batch dim and ``site``.

    A dimension is sliced only where there are two or more labels to tell
    apart; ``site`` also needs the observation to have a ``site`` dimension
    of two or more.
    """
    slices = []
    for dim in batch_dims(model_output):
        if model_output.sizes[dim] >= 2:
            slices.append((dim, model_output[dim].values[-1], observed_values))
    observed_sites = coordinate_labels(observed_values[SITE])
    if (
        SITE in model_output.dims
        and model_output.sizes[SITE] >= 2
        and SITE in observed_values.dims
        and len(observed_sites) >= 2
    ):
        label = observed_sites[-1]
        slices.append((SITE, label, observed_values.sel({SITE: [label]})))
    return slices


def _on_a_dimension(array: xr.DataArray, dim: str, label: Any) -> xr.DataArray:
    """*array* with *dim* as a dimension of one label, whatever form it had it in."""
    if dim in array.dims:
        return array
    if dim in array.coords:
        return array.expand_dims(dim)
    return array.expand_dims({dim: [label]})


def _agree_by_label(got: xr.DataArray, expected: xr.DataArray) -> bool:
    """Whether two arrays hold the same values at the same labels, in any dim order."""
    if set(got.dims) != set(expected.dims):
        return False
    try:
        got, expected = xr.align(got, expected.transpose(*got.dims), join="exact")
    except ValueError:
        return False
    return bool(
        np.allclose(
            np.asarray(got.values, float), np.asarray(expected.values, float), equal_nan=True
        )
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_operator_declares_names(operator: Any, message_name: str | None = None) -> None:
    """The operator is callable and declares pySIPNET names for what it reads.

    ``output_variable_names`` and ``sipnet_parameter_names`` must each be a
    tuple of strings; every output variable name must be a pySIPNET registry
    name and every SIPNET parameter name pySIPNET's flat name, not an alias,
    since those are what the model output and a SIPNET table carry.

    Parameters
    ----------
    operator:
        The operator to check.
    message_name:
        What the message calls the operator's owner, such as the product it
        predicts; prefixes the message when given.

    Raises
    ------
    TypeError
        If *operator* is not callable, or a declaration is not a tuple of
        strings.
    ValueError
        If a declaration names an alias.
    KeyError
        If a declared name is not in pySIPNET's registries.
    """
    who = f"{message_name}: the operator" if message_name else type(operator).__name__
    if not callable(operator):
        raise TypeError(
            f"{who} must be callable, got {type(operator).__name__}; pass an "
            "ObservationOperator such as SelectTimestep('wood_carbon')."
        )
    for attribute in ("output_variable_names", "sipnet_parameter_names"):
        names = getattr(operator, attribute, None)
        if not isinstance(names, tuple) or not all(isinstance(n, str) for n in names):
            raise TypeError(
                f"{who} must declare {attribute} as a tuple of names, got {names!r}; "
                "declare e.g. ('wood_carbon',), or () for none."
            )
    for name in operator.output_variable_names:
        registry_name = resolve_output_variable(name).name
        if registry_name != name:
            raise ValueError(
                f"{who} declares {name!r}, which is an alias; declare the registry name "
                f"{registry_name!r}, which is what the model output carries."
            )
    for name in operator.sipnet_parameter_names:
        flat_name = resolve_parameter_name(name)
        if flat_name != name:
            raise ValueError(
                f"{who} declares {name!r}, which is an alias; declare pySIPNET's flat "
                f"name {flat_name!r}, which is what a SIPNET table carries."
            )


def check_model_output_carries_what_is_read(
    model_output: Any,
    *,
    output_variable_names: Sequence[str],
    sipnet_parameter_names: Sequence[str],
    sipnet_parameters: Any,
    message_name: str = "the operators",
) -> None:
    """The model output carries what is read, and the parameters are given if read.

    Parameters
    ----------
    model_output:
        What the operators will be called with.
    output_variable_names, sipnet_parameter_names:
        What they read: one operator's declarations, or the union over a
        vector's.
    sipnet_parameters:
        What they will be given.
    message_name:
        What the message calls the reader.

    Raises
    ------
    TypeError
        If *model_output* is not an ``xr.Dataset``.
    ValueError
        If it lacks a variable that is read, or SIPNET parameters are read and
        *sipnet_parameters* is ``None``.
    """
    if not isinstance(model_output, xr.Dataset):
        raise TypeError(
            f"model_output must be an xarray Dataset of pySIPNET variables, got "
            f"{type(model_output).__name__}; label a run with fields.label_run."
        )
    missing = [n for n in output_variable_names if n not in model_output.data_vars]
    if missing:
        raise ValueError(
            f"the model output lacks {missing}, which {message_name} read; it has "
            f"{list(model_output.data_vars)[:10]}. Select those variables from each run."
        )
    if sipnet_parameter_names and sipnet_parameters is None:
        raise ValueError(
            f"{message_name} read SIPNET parameters {list(sipnet_parameter_names)}; pass "
            "sipnet_parameters=."
        )


def check_result_is_on_the_observation_grid(
    result: Any, observed_values: xr.DataArray, model_output: xr.Dataset, message_name: str
) -> None:
    """An operator's result is a labeled array on the observation's grid.

    Parameters
    ----------
    result:
        What the operator returned.
    observed_values:
        The observation it was called with; its ``site`` may be a dimension
        or a scalar coordinate.
    model_output:
        The model output it was called with, for its dims.
    message_name:
        What the message calls the producer of *result*: an operator's type
        name, or the product it predicts.

    Raises
    ------
    TypeError
        If *result* is not a ``DataArray``.
    ValueError
        If it carries no ``units`` attribute; has a dim that neither the
        model output nor the observation has; is not on the observation's
        sites, in order, whether ``site`` is a dimension or a scalar; or is
        not on the observation's ``time`` labels (compared as instants,
        whatever their datetime units), or has a ``time`` dimension for a
        static observation.
    """
    if not isinstance(result, xr.DataArray):
        raise TypeError(
            f"{message_name}: the operator returned {type(result).__name__}, not a "
            "DataArray; return the labeled array the time-alignment verbs give."
        )
    if not isinstance(result.attrs.get("units"), str):
        raise ValueError(
            f"{message_name}: the operator's result carries no 'units' attribute, so it "
            "cannot be converted into the observation's units. Write the operator with "
            "pysipnet.arithmetic, which labels its results, or set attrs['units'] on "
            "its result."
        )
    added = [
        str(d) for d in result.dims if d not in model_output.dims and d not in observed_values.dims
    ]
    if added:
        raise ValueError(
            f"{message_name}: the result has dim(s) {added} that neither the model output "
            "nor the observation has; an operator keeps the model output's batch dims and "
            "adds none."
        )
    check_result_is_at_the_observed_sites(result, observed_values, message_name)
    check_result_is_on_the_observed_time_labels(result, observed_values, message_name)


def check_result_is_at_the_observed_sites(
    result: xr.DataArray, observed_values: xr.DataArray, message_name: str
) -> None:
    """The result's ``site``, a dimension or a scalar, is the observation's, in order."""
    wanted_sites = coordinate_labels(observed_values[SITE])
    if SITE not in result.coords:
        raise ValueError(
            f"{message_name}: the result carries no site; select the model output with "
            "select_observed_sites, which keeps the observation's site labels."
        )
    if coordinate_labels(result[SITE]) != wanted_sites:
        where = "on the observation's sites, in order" if SITE in result.dims else (
            "at the observation's site"
        )
        raise ValueError(
            f"{message_name}: the result is not {where} "
            f"({coordinate_labels(result[SITE])[:10]} for {wanted_sites[:10]}); select "
            "the model output with select_observed_sites, which follows the observation."
        )


def check_result_is_on_the_observed_time_labels(
    result: xr.DataArray, observed_values: xr.DataArray, message_name: str
) -> None:
    """A dated observation's result is on its ``time`` labels; a static one's has none."""
    if TIME in observed_values.dims:
        # numpy compares datetime64 across units, so a label in seconds
        # matches the same instant in nanoseconds.
        if TIME not in result.dims or not np.array_equal(
            result[TIME].values, observed_values[TIME].values
        ):
            raise ValueError(
                f"{message_name}: the result is not on the observation's time labels; "
                "read the model at observed_values['time'], as select_timestep_at and "
                "reduce_windows(labels=...) do."
            )
    elif TIME in result.dims:
        raise ValueError(
            f"{message_name}: the result has a time dimension for a static observation; "
            "reduce the model over the run, as ReduceOverRun does."
        )


def check_operator_is_pointwise(
    operator: Any,
    model_output: xr.Dataset,
    observed_values: xr.DataArray,
    sipnet_parameters: Any,
    result: xr.DataArray,
) -> None:
    """The operator on the stack equals, by label, the operator on the last slice of each dim."""
    for dim, label, slice_observed in _pointwise_slices(model_output, observed_values):
        slice_output = model_output.sel({dim: label})
        slice_parameters = sipnet_parameters
        if isinstance(sipnet_parameters, xr.Dataset) and dim in sipnet_parameters.dims:
            slice_parameters = sipnet_parameters.sel({dim: label})
        expected = operator(slice_output, slice_observed, sipnet_parameters=slice_parameters)
        # A result without the dim reduced over it, which the comparison shows.
        got = result.sel({dim: [label]}) if dim in result.dims else result
        got, expected = _on_a_dimension(got, dim, label), _on_a_dimension(expected, dim, label)
        if not _agree_by_label(got, expected):
            raise ValueError(
                f"{type(operator).__name__} is not pointwise in {dim!r}: its value on the "
                f"stack differs from its value on the {dim}={label!r} slice alone. An "
                f"operator must compute each {dim} from that {dim}'s model output only."
            )


def check_observation_is_dated(observed_values: xr.DataArray, message_name: str) -> None:
    if TIME not in observed_values.dims:
        raise ValueError(
            f"{message_name} reads the model at each observed time label, and "
            f"{field_label(observed_values, 'the observation')} has no {TIME!r} "
            "dimension; a static observation is read over the whole run, with "
            "ReduceOverRun."
        )


def check_observation_is_static(observed_values: xr.DataArray, message_name: str) -> None:
    if TIME in observed_values.dims:
        raise ValueError(
            f"{message_name} reads a static observation, and "
            f"{field_label(observed_values, 'the observation')} has a {TIME!r} dimension; "
            "use ReduceOverWindows or SelectTimestep."
        )


def check_model_output_has_the_observed_sites(
    source_field: xr.DataArray, wanted: Sequence[Any], message_name: str
) -> None:
    missing = missing_labels(source_field, SITE, wanted)
    if missing:
        raise ValueError(
            f"the model output has no site(s) {missing[:10]} that {message_name} observes; "
            "run the model at every observed site, or select the observations to the "
            "sites that were run."
        )


def check_run_is_labeled_with_a_site(source_field: xr.DataArray, message_name: str) -> None:
    if SITE not in source_field.coords:
        raise ValueError(
            f"the model output carries no {SITE!r} coordinate, so it cannot be matched to "
            f"the sites {message_name} observes; label the run with fields.label_run(site=...)."
        )


def check_run_is_at_the_observed_site(
    source_field: xr.DataArray, wanted: Sequence[Any], message_name: str
) -> None:
    site = int(source_field[SITE].values)
    if list(wanted) != [site]:
        raise ValueError(
            f"the model output is one run at site {site}, and {message_name} observes "
            f"site(s) {list(wanted)[:10]}; select the observation to that one site."
        )


def check_table_has_the_parameter(table: xr.Dataset, name: str) -> None:
    if name not in table.data_vars:
        raise ValueError(
            f"the SIPNET table has no variable {name!r}, which this operator reads; "
            f"it has {list(table.data_vars)[:10]}. Build the table with every parameter "
            "the operators declare."
        )


def check_table_has_the_labels(
    values: xr.DataArray, dim: str, wanted: Sequence[Any], name: str
) -> None:
    missing = missing_labels(values, dim, wanted)
    if missing:
        raise ValueError(
            f"the SIPNET table's {name!r} has no {dim} label(s) {missing[:10]} "
            f"that the model output has; the table and the runs must cover the "
            f"same {dim}s."
        )


def check_target_is_not_a_stack_of_table_dims(
    target_field: xr.DataArray, values: xr.DataArray, name: str
) -> None:
    """No batch dim of the target is a stack of a dim the table has."""
    # A stack's 0..n-1 labels are not the labels of the dims stacked into it,
    # whatever the stacked dim is called, so the table cannot be read at them.
    for dim in batch_dims(target_field):
        stacked = [str(d) for d in values.dims if _is_stacked_into(target_field, str(d), dim)]
        if stacked:
            raise ValueError(
                f"the model output's {dim!r} is a stack of {stacked}, which the SIPNET table's "
                f"{name!r} is on; its labels are not the table's. Select from the table by "
                "the original labels, or unstack the model output with "
                "fields.unstack_batch_dims first."
            )


def check_target_has_a_coordinate_for(target_field: xr.DataArray, dim: str, name: str) -> None:
    if dim not in target_field.coords:
        raise ValueError(
            f"the SIPNET table's {name!r} is on {dim!r}, and the model output carries no "
            f"{dim} coordinate to select it at; using it whole would broadcast every "
            f"{dim} of the table into one run. Label the run with fields.label_run, or "
            f"select the table to the run's {dim}."
        )


def check_scalar_table_label_agrees(
    values: xr.DataArray, target_field: xr.DataArray, dim: str, name: str
) -> None:
    """A table selected to one label of *dim* is for the target's labels of *dim* alone."""
    # A stacked target carries dim's labels as <dim>_label on the stacked dim.
    labels = _labels_of(target_field, dim)
    if labels is None:
        check_a_stack_of_the_dim_carries_its_labels(target_field, dim, name)
        return
    label = values[dim].values.item()
    target_labels = coordinate_labels(labels)
    if any(t != label for t in target_labels):
        raise ValueError(
            f"the SIPNET table's {name!r} is for {dim} {label!r} alone, and the model "
            f"output is at {dim}(s) {target_labels[:10]}; pass the table for every {dim} "
            "the runs were."
        )


def check_a_stack_of_the_dim_carries_its_labels(
    target_field: xr.DataArray, dim: str, name: str
) -> None:
    """A batch dim of the target that records a stack of *dim* carries its labels."""
    label = f"{dim}{STACKED_LABEL_SUFFIX}"
    for stacked_dim in batch_dims(target_field):
        record = recorded_stacked_dims(target_field[stacked_dim])
        if record is not None and dim in record:
            raise ValueError(
                f"the model output's {stacked_dim!r} is a stack of {dim!r} with no "
                f"{label!r} coordinate, so which {dim} each run was is lost, and the SIPNET "
                f"table's {name!r} cannot be checked against it. Stack the model output "
                "again from its unstacked form, which keeps the labels."
            )


def check_mapping_value_is_a_number(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(
            f"sipnet_parameters[{name!r}] must be a number for one run, got "
            f"{type(value).__name__}; pass a SIPNET table for several runs."
        )
