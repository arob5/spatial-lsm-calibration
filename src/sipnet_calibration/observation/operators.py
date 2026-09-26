"""Observation operators: from one site's model output to a prediction of one
observed quantity, on that quantity's own grid.

Where this sits
---------------
An operator reads what SIPNET wrote (through
:func:`sipnet_calibration.fields.to_model_output`, a labeled ``xr.Dataset``) and,
where needed, the SIPNET parameter values the run used (SIPNET parameter
fields from :class:`~sipnet_calibration.parameter_vector.ParameterVector`),
and returns what the instrument would have read. The
:class:`~sipnet_calibration.observation.vector.ObservationVector` calls it,
converts the result into the observation source's units and checks it; a
predictive-check figure calls it directly. It is the map from model output to
one observation source.

The contract
------------
:class:`ObservationOperator` is a protocol. An operator declares
``output_variable_names``, the pySIPNET output variables it reads (so a worker
reads only those columns), and ``sipnet_parameter_names_read``, the SIPNET
parameters it reads (so the forward model can supply them, from the run's own
``SIPNETResult.parameters``). Its call is::

    operator(model_output, observed_values, *, sipnet_parameter_fields=None) -> Field

* ``model_output`` is a
  :data:`~sipnet_calibration.fields.ModelOutput`: pySIPNET-named variables on
  ``(time,)`` with a scalar ``site``, on ``(site, time)``, or on
  ``(*batch, site, time)`` with any batch dims (``sample``, a data source's
  ``<source>_member``), carrying pySIPNET's time coordinates and attributes.
* ``observed_values`` is the
  :data:`~sipnet_calibration.observation.source.ObservedValues` of one
  observation source, ``(site[, time])``, read for its ``site`` and ``time``
  coordinates, its windows where present, and nothing else; never for its
  values.
* ``sipnet_parameter_fields`` is the
  :data:`~sipnet_calibration.fields.SIPNETParameterFields` the runs
  used: on ``(*batch, site)`` or ``(site,)`` for a stack, or with no dim and a
  scalar ``site`` for one run.
* The result is on ``observed_values``' ``site`` and ``time`` grid, with the
  model output's batch dims if any and no dim that neither the model output nor
  the observed values have, and carries ``units`` and, where the quantity has
  one, ``constituent`` attributes saying what it is. It need not be in the
  observed values' units.
* The operator is **pointwise in site and in every batch dim**: applied to a
  stack it equals itself applied to each slice. That is what lets it run on
  the worker. :func:`check_operator` tests it.

The library ships four operators, each a frozen dataclass named for what it
does, and one default binding, :data:`DEFAULT_OBS_OPS`, which holds only the
observation source whose construction is established from a primary source.
How to read the model for the other observation sources is a modeling decision
the experiment writes in its ``config.py``.

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
    predicted = lai(model_output, observed_lai, sipnet_parameter_fields=sipnet_parameter_fields)
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

from sipnet_calibration import fields
from sipnet_calibration.conventions import LAT, LON, SITE, TIME, FrozenMapping
from sipnet_calibration.fields import (
    STACKED_LABEL_SUFFIX,
    Field,
    ModelOutput,
    SIPNETParameterFields,
    batch_dims,
    check_sipnet_parameter_name_is_a_flat_name,
    coordinate_labels,
    missing_labels,
    recorded_stacked_dims,
    scalar_batch_labels,
    validate_model_output,
    validate_sipnet_parameter_fields,
)
from sipnet_calibration.observation.source import ObservedValues, validate_observed_values
from sipnet_calibration.observation.time_alignment import (
    check_how_is_a_window_reduction,
    check_run_spans_the_windows,
    reduce_windows,
    run_window,
    select_timestep_at,
    windows_from_observed_values,
)

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
    "restrict_to_observed_sites",
]

@runtime_checkable
class ObservationOperator(Protocol):
    """Model output to a prediction of one observed quantity, on that quantity's grid."""

    output_variable_names: tuple[str, ...]
    sipnet_parameter_names_read: tuple[str, ...]

    def __call__(
        self,
        model_output: ModelOutput,
        observed_values: ObservedValues,
        *,
        sipnet_parameter_fields: SIPNETParameterFields | None = None,
    ) -> Field: ...


@dataclass(frozen=True)
class SelectTimestep:
    """The value of one model variable at the timestep containing each observed label.

    For each label in ``observed_values.time`` the model timestep whose
    interval ``(time_step_start, time]`` contains it is read: the state at
    the end of that step for a pool, the mean over it for a step mean or a
    rate. The label comes only from the observed values. A per-step total is
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
        variable. On a call, if the observed values have no ``time``
        dimension (a static observation source is read by
        :class:`ReduceOverRun`), or for any
        refusal of :func:`restrict_to_observed_sites` or
        :func:`~sipnet_calibration.observation.time_alignment.select_timestep_at`.
    """

    output_variable_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_variable_name", _output_name(self.output_variable_name))

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return (self.output_variable_name,)

    @property
    def sipnet_parameter_names_read(self) -> tuple[str, ...]:
        return ()

    def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None) -> Field:
        check_observed_values_are_dated(observed_values, type(self).__name__)
        variable = restrict_to_observed_sites(
            model_output[self.output_variable_name], observed_values
        )
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
        :func:`restrict_to_observed_sites` or
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
    def sipnet_parameter_names_read(self) -> tuple[str, ...]:
        return ()

    def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None) -> Field:
        variable = restrict_to_observed_sites(
            model_output[self.output_variable_name], observed_values
        )
        windows = windows_from_observed_values(observed_values)
        observed = fields.message_name(observed_values, "the observation source")
        check_run_spans_the_windows(variable, windows, f"{type(self).__name__} on {observed}")
        return reduce_windows(variable, windows, self.how, labels=observed_values[TIME])


@dataclass(frozen=True)
class ReduceOverRun:
    """One model variable reduced over the whole run, for a static observation source.

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
        observed values have a ``time`` dimension, or for any refusal of
        :func:`restrict_to_observed_sites` or
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
    def sipnet_parameter_names_read(self) -> tuple[str, ...]:
        return ()

    def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None) -> Field:
        check_observed_values_are_static(observed_values, type(self).__name__)
        variable = restrict_to_observed_sites(
            model_output[self.output_variable_name], observed_values
        )
        reduced = reduce_windows(variable, run_window(variable), self.how)
        return reduced.isel({TIME: 0}, drop=True)


@dataclass(frozen=True)
class ComputeLeafAreaIndex:
    """Leaf carbon over leaf carbon per area, at the timestep containing each label.

    SIPNET tracks leaf carbon, not leaf area; its own leaf area index is
    ``plantLeafC / leafCSpWt`` (``sipnet.c``), so the prediction depends on
    the parameter ``leaf_carbon_per_area`` the run used. The ratio is
    dimensionless and carries no constituent; the observed values' ``m2 m-2``
    converts from it by a factor of one.

    Raises
    ------
    TypeError
        On a call, if the SIPNET parameter fields are not a Dataset.
    ValueError
        On a call, if the observed values have no ``time`` dimension, or for any
        refusal of :func:`restrict_to_observed_sites`,
        :func:`extract_sipnet_parameter_at_coords` or
        :func:`~sipnet_calibration.observation.time_alignment.select_timestep_at`.
    """

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return ("leaf_carbon",)

    @property
    def sipnet_parameter_names_read(self) -> tuple[str, ...]:
        return ("leaf_carbon_per_area",)

    def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None) -> Field:
        check_observed_values_are_dated(observed_values, type(self).__name__)
        leaf_carbon = restrict_to_observed_sites(model_output["leaf_carbon"], observed_values)
        per_area = extract_sipnet_parameter_at_coords(
            sipnet_parameter_fields, "leaf_carbon_per_area", leaf_carbon
        )
        lai = divide_with_units(leaf_carbon, per_area)
        lai.name = "leaf_area_index"
        return select_timestep_at(lai, observed_values[TIME])


#: The library's default operator per observation source. It binds only the
#: constructions established from a primary source: today MODIS leaf area
#: index, as SIPNET's own ``plantLeafC / leafCSpWt`` (``sipnet.c``). How to
#: read the model for the other observation sources is a modeling decision,
#: and an experiment binds its own in ``config.py``.
DEFAULT_OBS_OPS: Mapping[str, ObservationOperator] = FrozenMapping(
    {"modis_leaf_area_index": ComputeLeafAreaIndex()}
)


def restrict_to_observed_sites(
    model_field: xr.DataArray, observed_values: xr.DataArray
) -> xr.DataArray:
    """Restrict a model field to the sites observed values are on.

    The operators call this first, with an output variable as the model field,
    so that everything after it works on the observed sites, in the observed
    values' order. A model field with a ``site`` dimension is selected down to
    the observed sites; a model field from one run, carrying ``site`` as a
    scalar coordinate, is returned unchanged when it is the one observed site.
    Only the two fields' ``site`` coordinates are read.

    Parameters
    ----------
    model_field:
        The field to restrict: typically one output variable of a model
        output, ``model_output[name]``, on ``(time,)`` with a scalar ``site``
        coordinate for one run, or on ``(site, time)`` or
        ``(*batch, site, time)`` for a stack. Its ``site`` labels are the
        site ids.
    observed_values:
        The observed values whose sites the result is on: those of one
        observation source, ``(site[, time])``.

    Returns
    -------
    xarray.DataArray
        *model_field* at the observed sites, in the order *observed_values*
        lists them, with every other dimension, coordinate and attribute
        unchanged. For one run, *model_field* itself.

    Raises
    ------
    TypeError
        If *observed_values* is not a ``DataArray``.
    ValueError
        If *observed_values* are not observed values
        (:func:`~sipnet_calibration.observation.source.validate_observed_values`:
        unique ``int32`` sites with ``lon``/``lat``, ``units``, among the
        rest); if *model_field* lacks an observed site; if *model_field* is
        one run at a site other than the one observed site, or several sites
        are observed; or if *model_field* carries no ``site`` coordinate at
        all.

    Notes
    -----
    A model field with no ``site`` is refused rather than assumed to be at the
    observed site: an unlabeled run could be any site, and matching it by
    position would be a guess. :func:`sipnet_calibration.fields.to_model_output`
    is what gives a run its site.
    """
    check_observed_values_are_valid(observed_values)
    wanted = coordinate_labels(observed_values[SITE])
    message_name = fields.message_name(observed_values, "the observation source")
    if SITE in model_field.dims:
        check_model_output_has_the_observed_sites(model_field, wanted, message_name)
        return model_field.sel({SITE: wanted})
    check_run_is_labeled_with_a_site(model_field, message_name)
    check_run_is_at_the_observed_site(model_field, wanted, message_name)
    return model_field


def extract_sipnet_parameter_at_coords(
    sipnet_parameter_fields: SIPNETParameterFields | None,
    sipnet_parameter_name: str,
    target_field: Field,
) -> Field:
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
    sipnet_parameter_fields:
        The values the runs used, as SIPNET parameter fields
        (:data:`~sipnet_calibration.fields.SIPNETParameterFields`):
        one variable per SIPNET parameter under pySIPNET's flat names, on
        ``(*batch, site)`` or ``(site,)`` as
        :meth:`~sipnet_calibration.parameter_vector.ParameterVector.sipnet_parameter_fields`
        returns them, or with no dim for one run, as the forward model's
        worker builds them from the run's own ``SIPNETResult.parameters``.
    sipnet_parameter_name:
        The SIPNET parameter to select: pySIPNET's flat name
        (``"leaf_carbon_per_area"``), an alias of it, or SIPNET's own name.
    target_field:
        The field the values will be combined with, typically an output
        variable. Only its coordinate labels are read, for each dim of the
        SIPNET parameter fields (``site`` and every batch dim), whether they
        are dimensions or scalar coordinates on the target: the values are
        taken at those labels, in that order, a scalar label selecting
        without keeping the dimension. A dimension of the SIPNET parameter
        fields the target has no coordinate for is refused, since using it
        whole would broadcast it into the run, as is a scalar label of theirs
        (``site`` or an integer batch label) the target's labels disagree
        with.

    Returns
    -------
    Field
        Named by the parameter's flat name, with the attributes
        ``parameter_dataarray`` gives it: ``units``, ``long_name``,
        ``description``, ``sipnet_name`` and, where pySIPNET declares one,
        ``constituent``; the parameter's variable at *target_field*'s site
        and batch labels, without the SIPNET parameter fields' ``lon``/``lat``,
        so that it combines with the target at the target's.

    Raises
    ------
    TypeError
        If *sipnet_parameter_fields* is not an ``xr.Dataset``.
    ValueError
        If *sipnet_parameter_fields* is ``None``, or are not SIPNET parameter
        fields
        (:func:`~sipnet_calibration.fields.validate_sipnet_parameter_fields`);
        if they have no variable for the parameter, have a dimension the
        target has no coordinate for, lack a label of it that *target_field*
        has, or carry a scalar label the target's disagree with (a stacked
        target's ``<dim>_label`` coordinates included, or lost); if a batch
        dim of *target_field* is a stack of a dim the SIPNET parameter fields
        have (:func:`~sipnet_calibration.fields.stack_batch_dims`), whose
        labels are not theirs; or if a value is not finite or lies outside
        the parameter's pySIPNET domain.
    KeyError
        If *sipnet_parameter_name* is not a pySIPNET parameter name or alias.
    """
    name = resolve_parameter_name(sipnet_parameter_name)
    check_sipnet_parameter_fields_are_given(sipnet_parameter_fields, name)
    validate_sipnet_parameter_fields(sipnet_parameter_fields)
    values = _sipnet_parameter_values_at(sipnet_parameter_fields, name, target_field)
    return parameter_dataarray(name, values)


def check_operator(
    operator: ObservationOperator,
    model_output: ModelOutput,
    observed_values: ObservedValues,
    *,
    sipnet_parameter_fields: SIPNETParameterFields | None = None,
) -> Field:
    """Check an operator against its contract on real inputs, and return its result.

    Checks that the declared names are pySIPNET registry names, not aliases,
    that *model_output* is a model output carrying the variables, and that
    *sipnet_parameter_fields* are given where parameters are read; that the
    result is on *observed_values*' grid and carries ``units``; and, when *model_output* has a ``site`` dim
    or batch dims of two labels or more, that the operator is pointwise: its
    value on the stack equals, label by label, its value on the last slice of
    each of those dims alone.

    Parameters
    ----------
    operator:
        The operator to check.
    model_output, observed_values, sipnet_parameter_fields:
        What the operator is called with, as in the contract above.

    Returns
    -------
    xarray.DataArray
        The operator's result on the whole of *model_output*, unconverted.

    Raises
    ------
    TypeError
        If the operator is not callable or its declarations are not tuples of
        strings, *model_output* or *sipnet_parameter_fields* is not a
        ``Dataset``, or the result is not a ``DataArray``.
    ValueError
        Naming the first rule broken: a declaration that names an alias; a
        model output that is not one
        (:func:`~sipnet_calibration.fields.validate_model_output`), or lacks
        a variable read; parameters read and not given, or not SIPNET
        parameter fields; a
        result without ``units``, off the observed values' sites or time
        labels, or with a dim that neither the model output nor the observed
        values have;
        or a result that is not pointwise. The operator's own refusals pass
        through.
    KeyError
        If a declared name is not in pySIPNET's registries.
    """
    message_name = type(operator).__name__
    check_operator_declares_names(operator)
    check_observed_values_are_valid(observed_values)
    check_model_output_carries_what_is_read(
        model_output,
        output_variable_names=operator.output_variable_names,
        sipnet_parameter_names_read=operator.sipnet_parameter_names_read,
        sipnet_parameter_fields=sipnet_parameter_fields,
        message_name=message_name,
    )
    result = operator(
        model_output, observed_values, sipnet_parameter_fields=sipnet_parameter_fields
    )
    check_result_is_on_the_observation_grid(result, observed_values, model_output, message_name)
    check_operator_is_pointwise(
        operator, model_output, observed_values, sipnet_parameter_fields, result
    )
    return result


# ── supporting helpers ────────────────────────────────────────────────────────


def _output_name(name: str) -> str:
    try:
        return resolve_output_variable(name).name
    except KeyError as error:
        raise ValueError(
            f"{name!r} is not a pySIPNET output variable or alias: {error}"
        ) from None


def _sipnet_parameter_values_at(
    sipnet_parameter_fields: xr.Dataset, name: str, target_field: xr.DataArray
) -> xr.DataArray:
    """Variable *name* of the SIPNET parameter fields at *target_field*'s labels.

    Every dim of the variable -- ``site`` and each batch dim, whatever it is
    named -- is selected at the target's labels or refused, so none of its
    dims is ever broadcast into a run.
    """
    check_sipnet_parameter_fields_hold_the_parameter(sipnet_parameter_fields, name)
    values = sipnet_parameter_fields[name]
    check_target_is_not_a_stack_of_sipnet_parameter_fields_dims(target_field, values, name)
    selectors: dict[str, Any] = {}
    for dim in map(str, values.dims):
        check_target_has_a_coordinate_for(target_field, dim, name)
        wanted = coordinate_labels(target_field[dim])
        check_sipnet_parameter_fields_have_the_labels(values, dim, wanted, name)
        # A scalar label selects without keeping the dimension, as the
        # target, one run, has none.
        selectors[dim] = wanted[0] if target_field[dim].ndim == 0 else wanted
    scalar_site = [SITE] if SITE in values.coords and values[SITE].ndim == 0 else []
    for dim in [*scalar_site, *scalar_batch_labels(values)]:
        check_scalar_sipnet_parameter_fields_label_agrees(values, target_field, dim, name)
    selected = values.sel(selectors) if selectors else values
    # The target's locations are the ones the values are combined at; the
    # SIPNET parameter fields' own, from another site table perhaps, would
    # clash with them in pysipnet.arithmetic.
    return selected.drop_vars([LON, LAT], errors="ignore")


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


def _pointwise_slices(
    model_output: xr.Dataset, observed_values: xr.DataArray
) -> list[tuple[str, Any, xr.DataArray]]:
    """The ``(dim, label, observed slice)`` for the last slice of each batch dim and ``site``.

    A dimension is sliced only where there are two or more labels to tell
    apart; ``site`` also needs the observed values to have a ``site``
    dimension of two or more.
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


def check_observed_values_are_valid(observed_values: Any) -> None:
    """What an operator reads as observed values are, ``site`` a dim or a scalar.

    Runs
    :func:`~sipnet_calibration.observation.source.validate_observed_values`
    on them laid out as a field (:func:`~sipnet_calibration.fields.in_field_layout`),
    so one site's observed values with a scalar ``site``, which the operator
    contract allows, are accepted.
    """
    if isinstance(observed_values, xr.DataArray):
        observed_values = fields.in_field_layout(observed_values)
    validate_observed_values(observed_values)


def check_operator_declares_names(operator: Any, message_name: str | None = None) -> None:
    """The operator is callable and declares pySIPNET names for what it reads.

    ``output_variable_names`` and ``sipnet_parameter_names_read`` must each be a
    tuple of strings; every output variable name must be a pySIPNET registry
    name and every SIPNET parameter name pySIPNET's flat name, not an alias,
    since those are what the model output and SIPNET parameter fields carry.

    Parameters
    ----------
    operator:
        The operator to check.
    message_name:
        What the message calls the operator's owner, such as the observation
        source it predicts; prefixes the message when given.

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
    for attribute in ("output_variable_names", "sipnet_parameter_names_read"):
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
    for name in operator.sipnet_parameter_names_read:
        check_sipnet_parameter_name_is_a_flat_name(name, f"{who}'s sipnet_parameter_names_read")


def check_model_output_carries_what_is_read(
    model_output: Any,
    *,
    output_variable_names: Sequence[str],
    sipnet_parameter_names_read: Sequence[str],
    sipnet_parameter_fields: Any,
    message_name: str = "the operators",
) -> None:
    """The model output is one and carries what is read, with the parameters read.

    Runs :func:`~sipnet_calibration.fields.validate_model_output`,
    :func:`check_model_output_has_the_variables`, and, when SIPNET parameters
    are read, :func:`check_sipnet_parameter_fields_are_given` and
    :func:`~sipnet_calibration.fields.validate_sipnet_parameter_fields`.

    Parameters
    ----------
    model_output:
        What the operators will be called with.
    output_variable_names, sipnet_parameter_names_read:
        What they read: one operator's declarations, or the union over a
        vector's.
    sipnet_parameter_fields:
        What they will be given.
    message_name:
        What the message calls the reader.

    Raises
    ------
    TypeError
        If *model_output* is not an ``xr.Dataset``, or SIPNET parameters are
        read and *sipnet_parameter_fields* is not one.
    ValueError
        If *model_output* is not a model output or lacks a variable that is
        read; or if SIPNET parameters are read and *sipnet_parameter_fields*
        is ``None`` or are not SIPNET parameter fields.
    """
    validate_model_output(model_output)
    check_model_output_has_the_variables(model_output, output_variable_names, message_name)
    if sipnet_parameter_names_read:
        check_sipnet_parameter_fields_are_given(
            sipnet_parameter_fields, list(sipnet_parameter_names_read), message_name
        )
        validate_sipnet_parameter_fields(sipnet_parameter_fields)


def check_model_output_has_the_variables(
    model_output: xr.Dataset, output_variable_names: Sequence[str], message_name: str
) -> None:
    """The model output carries every output variable that is read."""
    missing = [n for n in output_variable_names if n not in model_output.data_vars]
    if missing:
        raise ValueError(
            f"the model output lacks {missing}, which {message_name} read; it has "
            f"{list(model_output.data_vars)[:10]}. Select those variables from each run."
        )


def check_sipnet_parameter_fields_are_given(
    sipnet_parameter_fields: Any, sipnet_parameter_names: Any, message_name: str = "this operator"
) -> None:
    """SIPNET parameter fields are given where SIPNET parameters are read."""
    if sipnet_parameter_fields is None:
        raise ValueError(
            f"{message_name} read the SIPNET parameters {sipnet_parameter_names!r}; pass "
            "sipnet_parameter_fields= (ParameterVector.sipnet_parameter_fields, or one run's "
            "values from its SIPNETResult.parameters)."
        )


def check_result_is_on_the_observation_grid(
    result: Any, observed_values: xr.DataArray, model_output: xr.Dataset, message_name: str
) -> None:
    """An operator's result is a field on the observed values' grid.

    Parameters
    ----------
    result:
        What the operator returned.
    observed_values:
        The observed values it was called with; their ``site`` may be a
        dimension or a scalar coordinate.
    model_output:
        The model output it was called with, for its dims.
    message_name:
        What the message calls the producer of *result*: an operator's type
        name, or the observation source it predicts.

    Raises
    ------
    TypeError
        If *result* is not a ``DataArray``.
    ValueError
        If it has a dim that neither the model output nor the observed values
        have; lacks a batch dim of the model output, or carries other labels
        on it (a mean over ``sample``, a selection, a relabeling); is not on
        the observed values' sites, in order, whether ``site`` is a dimension or a scalar;
        or is not on the observed values' ``time`` labels (compared as
        instants, whatever their datetime units), or has a ``time`` dimension
        for static observed values; or if, laid out as a field
        (:func:`~sipnet_calibration.fields.in_field_layout`, which leaves the
        operator its own dim order and a scalar ``site``), it is not one
        (:func:`~sipnet_calibration.fields.validate_field`).
    """
    check_result_is_a_dataarray(result, message_name)
    check_result_adds_no_dim(result, observed_values, model_output, message_name)
    check_result_keeps_the_batch_dims(result, model_output, message_name)
    check_result_is_at_the_observed_sites(result, observed_values, message_name)
    check_result_is_on_the_observed_time_labels(result, observed_values, message_name)
    fields.validate_field(
        fields.in_field_layout(result), message_name=f"{message_name}: the operator's result"
    )


def check_result_is_a_dataarray(result: Any, message_name: str) -> None:
    """An operator's result is a ``DataArray``."""
    if not isinstance(result, xr.DataArray):
        raise TypeError(
            f"{message_name}: the operator returned {type(result).__name__}, not a "
            "DataArray; return the labeled array the time-alignment verbs give."
        )


def check_result_adds_no_dim(
    result: xr.DataArray, observed_values: xr.DataArray, model_output: xr.Dataset, message_name: str
) -> None:
    """An operator's result adds no dim to the model output's and the observed values'."""
    added = [
        str(d) for d in result.dims if d not in model_output.dims and d not in observed_values.dims
    ]
    if added:
        raise ValueError(
            f"{message_name}: the result has dim(s) {added} that neither the model output "
            "nor the observed values have; an operator keeps the model output's batch dims and "
            "adds none."
        )


def check_result_keeps_the_batch_dims(
    result: xr.DataArray, model_output: xr.Dataset, message_name: str
) -> None:
    """An operator's result keeps the model output's batch dims and their labels."""
    for dim in batch_dims(model_output):
        if dim not in result.dims:
            raise ValueError(
                f"{message_name}: the result has no {dim!r} dim, which the model output has; "
                f"an operator is pointwise in {dim}, so compute each {dim} from its own model "
                "output and keep the dim, rather than reducing or selecting it away."
            )
        if not np.array_equal(result[dim].values, model_output[dim].values):
            raise ValueError(
                f"{message_name}: the result's {dim} labels "
                f"{coordinate_labels(result[dim])[:10]} are not the model output's "
                f"{coordinate_labels(model_output[dim])[:10]}; an operator keeps the model "
                "output's labels, in its order."
            )


def check_result_is_at_the_observed_sites(
    result: xr.DataArray, observed_values: xr.DataArray, message_name: str
) -> None:
    """The result's ``site``, a dim or a scalar, is the observed values', in order."""
    wanted_sites = coordinate_labels(observed_values[SITE])
    if SITE not in result.coords:
        raise ValueError(
            f"{message_name}: the result carries no site; select the model output with "
            "restrict_to_observed_sites, which keeps the observed values' site labels."
        )
    if coordinate_labels(result[SITE]) != wanted_sites:
        where = "on the observed values' sites, in order" if SITE in result.dims else (
            "at the observed values' site"
        )
        raise ValueError(
            f"{message_name}: the result is not {where} "
            f"({coordinate_labels(result[SITE])[:10]} for {wanted_sites[:10]}); select "
            "the model output with restrict_to_observed_sites, which follows the observed values."
        )


def check_result_is_on_the_observed_time_labels(
    result: xr.DataArray, observed_values: xr.DataArray, message_name: str
) -> None:
    """The result is on dated observed values' ``time`` labels, and else on none."""
    if TIME in observed_values.dims:
        # numpy compares datetime64 across units, so a label in seconds
        # matches the same instant in nanoseconds.
        if TIME not in result.dims or not np.array_equal(
            result[TIME].values, observed_values[TIME].values
        ):
            raise ValueError(
                f"{message_name}: the result is not on the observed values' time labels; "
                "read the model at observed_values['time'], as select_timestep_at and "
                "reduce_windows(labels=...) do."
            )
    elif TIME in result.dims:
        raise ValueError(
            f"{message_name}: the result has a time dimension for static observed values; "
            "reduce the model over the run, as ReduceOverRun does."
        )


def check_operator_is_pointwise(
    operator: Any,
    model_output: xr.Dataset,
    observed_values: xr.DataArray,
    sipnet_parameter_fields: Any,
    result: xr.DataArray,
) -> None:
    """The operator on the stack equals, by label, the operator on the last slice of each dim."""
    for dim, label, slice_observed in _pointwise_slices(model_output, observed_values):
        slice_output = model_output.sel({dim: label})
        slice_parameters = sipnet_parameter_fields
        if isinstance(sipnet_parameter_fields, xr.Dataset) and dim in sipnet_parameter_fields.dims:
            slice_parameters = sipnet_parameter_fields.sel({dim: label})
        expected = operator(
            slice_output, slice_observed, sipnet_parameter_fields=slice_parameters
        )
        # A result without the dim reduced over it, which the comparison shows.
        got = result.sel({dim: [label]}) if dim in result.dims else result
        got, expected = _on_a_dimension(got, dim, label), _on_a_dimension(expected, dim, label)
        if not _agree_by_label(got, expected):
            raise ValueError(
                f"{type(operator).__name__} is not pointwise in {dim!r}: its value on the "
                f"stack differs from its value on the {dim}={label!r} slice alone. An "
                f"operator must compute each {dim} from that {dim}'s model output only."
            )


def check_observed_values_are_dated(observed_values: xr.DataArray, message_name: str) -> None:
    if TIME not in observed_values.dims:
        raise ValueError(
            f"{message_name} reads the model at each observed time label, and "
            f"{fields.message_name(observed_values, 'the observation source')} has no "
            f"{TIME!r} dimension; a static observation source is read over the whole run, with "
            "ReduceOverRun."
        )


def check_observed_values_are_static(observed_values: xr.DataArray, message_name: str) -> None:
    if TIME in observed_values.dims:
        raise ValueError(
            f"{message_name} reads a static observation source, and "
            f"{fields.message_name(observed_values, 'the observation source')} has a "
            f"{TIME!r} dimension; use ReduceOverWindows or SelectTimestep."
        )


def check_model_output_has_the_observed_sites(
    model_field: xr.DataArray, wanted: Sequence[Any], message_name: str
) -> None:
    missing = missing_labels(model_field, SITE, wanted)
    if missing:
        raise ValueError(
            f"the model output has no site(s) {missing[:10]} that {message_name} observes; "
            "run the model at every observed site, or select the observed values to the "
            "sites that were run."
        )


def check_run_is_labeled_with_a_site(model_field: xr.DataArray, message_name: str) -> None:
    if SITE not in model_field.coords:
        raise ValueError(
            f"the model output carries no {SITE!r} coordinate, so it cannot be matched to "
            f"the sites {message_name} observes; label the run with its site, "
            "fields.to_model_output(run_output, site=...)."
        )


def check_run_is_at_the_observed_site(
    model_field: xr.DataArray, wanted: Sequence[Any], message_name: str
) -> None:
    site = int(model_field[SITE].values)
    if list(wanted) != [site]:
        raise ValueError(
            f"the model output is one run at site {site}, and {message_name} observes "
            f"site(s) {list(wanted)[:10]}; select the observed values to that one site."
        )


def check_sipnet_parameter_fields_hold_the_parameter(
    sipnet_parameter_fields: xr.Dataset, name: str
) -> None:
    if name not in sipnet_parameter_fields.data_vars:
        raise ValueError(
            f"the SIPNET parameter fields have no variable {name!r}, which this operator "
            f"reads; they have {list(sipnet_parameter_fields.data_vars)[:10]}. Build them "
            "with every parameter the operators declare."
        )


def check_sipnet_parameter_fields_have_the_labels(
    values: xr.DataArray, dim: str, wanted: Sequence[Any], name: str
) -> None:
    missing = missing_labels(values, dim, wanted)
    if missing:
        raise ValueError(
            f"the SIPNET parameter fields' {name!r} has no {dim} label(s) {missing[:10]} "
            f"that the model output has; the SIPNET parameter fields and the runs must "
            f"cover the same {dim}s."
        )


def check_target_is_not_a_stack_of_sipnet_parameter_fields_dims(
    target_field: xr.DataArray, values: xr.DataArray, name: str
) -> None:
    """No batch dim of the target stacks a dim the SIPNET parameter fields have."""
    # A stack's 0..n-1 labels are not the labels of the dims stacked into it,
    # whatever the stacked dim is called, so the SIPNET parameter fields cannot
    # be read at them.
    for dim in batch_dims(target_field):
        stacked = [str(d) for d in values.dims if _is_stacked_into(target_field, str(d), dim)]
        if stacked:
            raise ValueError(
                f"the model output's {dim!r} is a stack of {stacked}, which the SIPNET "
                f"parameter fields' {name!r} is on; its labels are not theirs. Select from "
                "the SIPNET parameter fields by the original labels, or unstack the model "
                "output with "
                "fields.unstack_batch_dims first."
            )


def check_target_has_a_coordinate_for(target_field: xr.DataArray, dim: str, name: str) -> None:
    if dim not in target_field.coords:
        raise ValueError(
            f"the SIPNET parameter fields' {name!r} is on {dim!r}, and the model output "
            f"carries no {dim} coordinate to select it at; using it whole would broadcast "
            f"every {dim} of the SIPNET parameter fields into one run. Label the run with "
            f"fields.to_model_output, or select the SIPNET parameter fields to the run's {dim}."
        )


def check_scalar_sipnet_parameter_fields_label_agrees(
    values: xr.DataArray, target_field: xr.DataArray, dim: str, name: str
) -> None:
    """SIPNET parameter fields selected to one *dim* label serve only that label."""
    # A stacked target carries dim's labels as <dim>_label on the stacked dim.
    labels = _labels_of(target_field, dim)
    if labels is None:
        check_a_stack_of_the_dim_carries_its_labels(target_field, dim, name)
        return
    label = values[dim].values.item()
    target_labels = coordinate_labels(labels)
    if any(t != label for t in target_labels):
        raise ValueError(
            f"the SIPNET parameter fields' {name!r} is for {dim} {label!r} alone, and the "
            f"model output is at {dim}(s) {target_labels[:10]}; pass the SIPNET parameter "
            f"fields for every {dim} the runs were."
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
                f"parameter fields' {name!r} cannot be checked against it. Stack the model output "
                "again from its unstacked form, which keeps the labels."
            )
