"""Placing a field on an observation source's time grid.

The time verbs on fields: combining timesteps into calendar cells or into
arbitrary windows, reading the timestep that contains an instant, building
the windows, and counting what went into each cell or window. The
observation operators are written with them, and a caller aggregates a field
with the same :func:`aggregate_time` before plotting it, so a
predictive-check figure cannot disagree with what the likelihood consumed.
Combining by calendar cells is pySIPNET's :func:`pysipnet.resample.resample`;
this module adds a default method, windows and instants.

Provided:

* :func:`aggregate_time` and :func:`aggregation_counts`: calendar cells of a
  pandas frequency (``"1D"``, ``"MS"``, ``"YS"``).
* :func:`reduce_windows` and :func:`window_counts`: arbitrary
  non-overlapping windows, as an observation's own support.
* :func:`select_timestep_at`: the model timestep whose interval contains
  each label.
* :func:`windows_from_observed_values` and :func:`run_window`: the two ways
  windows are built, each a ``pandas.IntervalIndex``.
* :func:`check_run_spans_the_windows`, :func:`check_how_is_a_window_reduction`
  and :func:`check_frequency_is_an_offset_alias`: the checks an operator or
  the forward model makes before reading the model.

How a method is chosen
----------------------
**Which methods mean anything is a property of the variable; which of them is
wanted is the caller's.** pySIPNET settles the first half: every model and
driver variable has a ``kind``, and
``pysipnet.variables.RESAMPLING_METHODS_FOR_KIND`` says what may be done with
it. Every function here refuses a method the kind does not admit, in
pySIPNET's own words (the window extremes excepted).

For the second half :func:`aggregate_time` supplies a default, which
pySIPNET's ``resample`` deliberately does not: the one method that leaves the
variable the kind it already is (:data:`DEFAULT_METHOD_FOR_KIND`). A total
sums, a step mean or a rate means, a pool or a running total takes its last
value; ``how=`` asks for something else, such as the time-weighted mean of a
pool. :func:`reduce_windows` has no default: what an observation wants of the
model over its window is the operator's decision to state.

SIPNET's ``net_ecosystem_exchange`` is a per-timestep total, so 3-hourly to
daily is a **sum**; a mean is wrong by a factor of 8 and looks entirely
plausible. That is the error the default exists to make impossible to reach by
omission.

Aggregation is a verb the caller applies, never a plotter keyword::

    plot_time_series(aggregate_time(nee, "1D"), ax=ax)   # yes
    plot_time_series(nee, temporal_agg="1D")             # no

Steps, cells and windows
------------------------
A model field carries pySIPNET's interval coordinates: ``time`` is the end of
each step, ``timestep_start`` its start and ``timestep_length`` its declared
length. A step belongs to the calendar cell or the window that contains its
**end**, and a mean is weighted by ``timestep_length``. A cell or window
holding a ``NaN`` is ``NaN`` for every method, and the counts say how many
values went in. The padding a stack of runs on different time axes leaves (a
``NaT`` interval and no value) is not a step, and every function here drops it
first, by pySIPNET's :func:`~pysipnet.resample.drop_padding`.

Nothing here converts between clocks: the labels observed values supply are
taken to be on the model field's clock, and where they are not the operator
shifts them first.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, get_args

import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict
from pysipnet.arithmetic import step_length
from pysipnet.resample import (
    STEP_LENGTH_RESAMPLED,
    check_frequency,
    check_resampling_method,
    drop_padding,
    resample,
    resampled_attributes,
)
from pysipnet.variables import (
    RESAMPLED_KIND,
    TIME_REFERENCE_FOR_KIND,
    ResamplingMethod,
    VariableKind,
    variable_kind,
)

from sipnet_calibration import fields
from sipnet_calibration.conventions import (
    SIPNET_ROW_LABEL_NAMES,
    TIME,
    TIMESTEP_LENGTH,
    TIMESTEP_START,
    WINDOW_END,
    WINDOW_START,
)
from sipnet_calibration.fields import without_stale_time_attributes

__all__ = [
    "DEFAULT_METHOD_FOR_KIND",
    "RESAMPLING_METHODS",
    "SELECTED_STEP_COORD",
    "WINDOW_REDUCTIONS",
    "aggregate_time",
    "aggregation_counts",
    "check_frequency_is_an_offset_alias",
    "check_how_is_a_window_reduction",
    "check_run_spans_the_windows",
    "reduce_windows",
    "run_window",
    "select_timestep_at",
    "window_counts",
    "windows_from_observed_values",
]

#: The three ways pySIPNET combines consecutive steps, under pySIPNET's names;
#: what :func:`aggregate_time` admits.
RESAMPLING_METHODS: tuple[str, ...] = get_args(ResamplingMethod)

#: What :func:`reduce_windows` admits: pySIPNET's three, and the extremes and
#: the leading edge of a window, which are meaningful for a state, a step mean
#: or a rate and refused for a total or a running total.
WINDOW_REDUCTIONS: tuple[str, ...] = (*RESAMPLING_METHODS, "min", "max", "first")

#: The coordinate :func:`select_timestep_at` and :func:`reduce_windows` add,
#: saying which model step end each label read: the step containing the label,
#: or the last step combined into the window.
SELECTED_STEP_COORD = "selected_timestep_end"

#: What :func:`aggregate_time` does when ``how`` is not given: the one method
#: per kind that leaves a variable the kind it already is, read off pySIPNET's
#: ``RESAMPLED_KIND`` rather than written down. That exactly one method
#: preserves each kind is checked against pySIPNET's table by the tests.
DEFAULT_METHOD_FOR_KIND: Mapping[VariableKind, str] = frozendict(
    {kind: method for (kind, method), resulting in RESAMPLED_KIND.items() if resulting == kind}
)


def aggregate_time(
    field: xr.DataArray, freq: str, *, how: str | None = None
) -> xr.DataArray:
    """Combine a field's timesteps into coarser ones, by pySIPNET's ``resample``.

    Parameters
    ----------
    field:
        A field with a ``time`` dim; its other dims are carried through. A
        field with pySIPNET's interval coordinates (model output, drivers) is
        combined by its steps, one with neither of them (observed values) on
        calendar cells alone.
    freq:
        A pandas offset alias for the coarser step: ``"1D"``, ``"7D"``,
        ``"MS"`` for calendar months, ``"YS"`` for calendar years.
    how:
        ``"sum"``, ``"mean"`` or ``"last"``; by default the method in
        :data:`DEFAULT_METHOD_FOR_KIND` for the variable's kind.

    Returns
    -------
    xarray.DataArray
        :func:`pysipnet.resample.resample`'s result, with no SIPNET row labels
        the field did not carry, and with ``last`` ``NaN`` in a cell holding
        a ``NaN`` anywhere, as ``sum`` and ``mean`` already are. A field that
        declares no kind keeps its own attributes, less ``output_decimals``,
        and gains a ``resampling`` attribute.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``, or *freq* is not a string.
    ValueError
        If *field* is not a field with a ``time`` dim or has no steps once
        its padding is dropped; if *how* is omitted and the kind is unknown
        or has no default; or if pySIPNET's ``resample`` refuses the field,
        *freq* or *how*.

    Notes
    -----
    pySIPNET's ``last`` reads only a cell's last value, so a gap earlier in
    the cell is found by counting. Aggregating half a day of a gappy record
    into a number that looks like a whole day is how a gap stops being
    visible.

    Cells on calendar cells alone carry nothing about their coverage, so the
    first and last cells of such a record may be partial with nothing to say
    so; mask on :func:`aggregation_counts` where a full cell's count is
    known.
    """
    field = _as_steps(field)
    check_frequency_is_an_offset_alias(freq)
    kind = variable_kind(field, default=None)
    method = _resampling_method_for(field, kind, how)
    if kind is None and not _has_interval_coords(field):
        result = _resample_without_a_kind(field, freq, method)
    else:
        result = resample(field, freq, how=method)
    # pySIPNET's resample writes SIPNET's row labels onto its result; a field
    # that did not carry them does not gain them (fields' model output).
    added = [n for n in SIPNET_ROW_LABEL_NAMES if n in result.coords and n not in field.coords]
    result = result.drop_vars(added)
    if method == "last":
        gaps = _steps_per_cell(field, field.isnull(), freq)
        result = result.where(xr.DataArray(gaps.values == 0, dims=gaps.dims))
    return result


def reduce_windows(
    field: xr.DataArray,
    windows: pd.IntervalIndex,
    how: str,
    *,
    labels: Any = None,
) -> xr.DataArray:
    """Combine a field's timesteps into arbitrary windows.

    A step belongs to the window containing its ``time`` label, the step's
    end; steps outside every window are ignored.

    Parameters
    ----------
    field:
        As :func:`aggregate_time` takes it.
    windows:
        A ``pandas.IntervalIndex`` of datetimes, non-overlapping, increasing,
        naive or in the field's time zone. Its ``closed`` side decides which of
        two adjacent windows a step ending on their shared edge belongs to;
        :func:`windows_from_observed_values` and :func:`run_window` build
        right-closed ones, as end-labeled steps want.
    how:
        One of :data:`WINDOW_REDUCTIONS`, checked against the variable's kind
        as :func:`aggregate_time` checks it; ``min``, ``max`` and ``first``
        only for a state, a step mean or a rate.
    labels:
        The result's ``time`` coordinate, one label per window, strictly
        increasing, a ``DataArray``'s attributes kept. Defaults to each
        window's right edge.

    Returns
    -------
    xarray.DataArray
        One value per window on *labels*, the field's other dims untouched. A
        window holding no step, or a ``NaN``, is ``NaN``. A field with
        pySIPNET's interval coordinates gives them for the steps each window
        combined (the earliest start, the summed length), with
        :data:`SELECTED_STEP_COORD` the latest end. ``kind``,
        ``time_reference`` and ``cell_methods`` are rewritten as
        :func:`aggregate_time` rewrites them; for ``min``, ``max`` and
        ``first`` the kind is kept, ``time_reference`` names the reading, and
        only a pool keeps a ``cell_methods``. A ``reduction`` attribute says
        what was done.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``, *windows* is not a
        ``pandas.IntervalIndex``, or *how* is not a string.
    ValueError
        If the field fails the checks of :func:`aggregate_time`; if the kind
        does not admit *how*; if *windows* are not non-empty, disjoint,
        increasing datetime intervals on the field's clock; or if *labels*
        are not one strictly increasing timestamp per window.
    """
    field = _as_steps(field)
    kind = variable_kind(field, default=None)
    how = _window_reduction_for(field, kind, how)
    windows = _as_windows(windows, _time_index(field))
    labels, label_attrs = _as_window_labels(labels, windows)

    weights = _step_weights(field) if how == "mean" else None
    reduced = _reduce_by_window(_without_interval_coords(field), windows, how, weights)
    reduced = reduced.assign_coords({TIME: labels})
    reduced[TIME].attrs = label_attrs
    reduced = reduced.assign_coords(_window_interval_coords(field, windows))
    reduced.name = field.name
    weighted = how == "mean" and TIMESTEP_LENGTH in field.coords
    reduced.attrs = _window_reduction_attrs(
        field.attrs, kind, how, weighted=weighted, name=fields.message_name(field, quoted=False)
    )
    return reduced


def select_timestep_at(field: xr.DataArray, times: Any) -> xr.DataArray:
    """For each label, the value of the model timestep whose interval contains it.

    The step with ``timestep_start < t <= time`` is the one running at ``t``:
    its value is the state at its end (a pool) or the mean over it (a step
    mean or a rate). A label exactly at a step end reads that step.

    Parameters
    ----------
    field:
        A model field carrying pySIPNET's interval coordinates; its other
        dims are carried through.
    times:
        The labels to read at, strictly increasing: a ``DataArray``, whose
        attributes are kept, or any datetime array-like.

    Returns
    -------
    xarray.DataArray
        The field at the selected steps on ``time`` equal to *times*, with
        :data:`SELECTED_STEP_COORD` the step end each label read, without
        the interval coordinates, and with ``time_reference`` saying which
        step the value is of.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``.
    ValueError
        If the field fails the checks of :func:`aggregate_time`, has no
        interval coordinates or has overlapping steps; if its kind has no
        value at an instant (a per-step total, a running total); or if
        *times* are not strictly increasing timestamps on the field's clock,
        each inside a step.
    """
    field = _as_steps(field)
    message_name = fields.message_name(field)
    check_field_has_interval_coords(field, message_name="select_timestep_at")
    kind = variable_kind(field, default=None)
    check_kind_has_an_instant_value(kind, message_name=message_name)
    labels, label_attrs = _as_instants(times, message_name="times")

    ends = _time_index(field)
    check_labels_are_on_the_fields_clock(labels, ends, message_name=message_name)
    steps = _step_intervals(field)
    check_steps_do_not_overlap(steps, message_name=message_name)
    # Compared in nanoseconds, so a label finer than the axis is not truncated
    # onto the axis's resolution before the containment test.
    position = steps.get_indexer(labels.as_unit("ns"))
    check_every_label_is_in_a_step(labels, position, steps, message_name=message_name)

    selected = _without_interval_coords(field.isel({TIME: position}))
    selected = selected.assign_coords(
        {
            TIME: xr.DataArray(labels, dims=TIME, attrs=label_attrs),
            SELECTED_STEP_COORD: xr.DataArray(
                ends.values[position],
                dims=TIME,
                attrs={
                    "long_name": "End of the model timestep that was read",
                    "comment": (
                        "The step whose (timestep_start, time] interval contains the label."
                    ),
                },
            ),
        }
    )
    selected.attrs = _selection_attrs(field.attrs, kind)
    return selected


def windows_from_observed_values(observed_values: xr.DataArray) -> pd.IntervalIndex:
    """The windows of observed values, as :func:`reduce_windows` takes them.

    Read from :data:`~sipnet_calibration.conventions.WINDOW_START` and
    :data:`~sipnet_calibration.conventions.WINDOW_END`, which
    :func:`sipnet_calibration.constraints.constraint_fields` puts on an annual
    constraint's ``time``.

    Parameters
    ----------
    observed_values:
        Observed values carrying the two window coordinates on ``time``.

    Returns
    -------
    pandas.IntervalIndex
        One right-closed window per ``time`` label, in order, so a model step
        ending on the shared edge of two years belongs to the year that
        ended.

    Raises
    ------
    TypeError
        If *observed_values* is not a ``DataArray``.
    ValueError
        If *observed_values* is not a field; if it carries no windows (a dated
        or static constraint documents none: read it with
        :func:`select_timestep_at` or :func:`run_window`); or if a window
        edge is not a datetime, is ``NaT``, or does not follow its start.
    """
    fields.validate_field(observed_values)
    message_name = fields.message_name(observed_values, "the observation source")
    check_observed_values_have_windows(observed_values, message_name=message_name)
    start = pd.DatetimeIndex(observed_values[WINDOW_START].values)
    end = pd.DatetimeIndex(observed_values[WINDOW_END].values)
    check_window_edges_are_present(start, end, message_name=message_name)
    check_window_ends_follow_their_starts(start, end, message_name=message_name)
    return pd.IntervalIndex.from_arrays(start, end, closed="right")


def run_window(field: xr.DataArray) -> pd.IntervalIndex:
    """One right-closed window spanning a model field's whole record.

    From the first step's ``timestep_start`` to the last step's ``time``, so
    every step belongs to it: the window an operator over a static
    observation source reduces over. Padding does not widen it.

    Parameters
    ----------
    field:
        A model field carrying pySIPNET's interval coordinates.

    Returns
    -------
    pandas.IntervalIndex
        One right-closed window.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``.
    ValueError
        If the field fails the checks of :func:`aggregate_time`, or carries no
        interval coordinates.
    """
    field = _as_steps(field)
    check_field_has_interval_coords(field, message_name="run_window")
    start = _start_index(field)
    end = _time_index(field)
    return pd.IntervalIndex.from_arrays([start.min()], [end.max()], closed="right")


def aggregation_counts(field: xr.DataArray, freq: str) -> xr.DataArray:
    """How many values each cell of :func:`aggregate_time` is formed from.

    Parameters
    ----------
    field, freq:
        As for :func:`aggregate_time`.

    Returns
    -------
    xarray.DataArray
        ``int64``, the values that are not missing per cell, zero where a
        cell held only missing values, on the ``time`` coordinates
        :func:`aggregate_time` gives the same field, so
        ``aggregate_time(field, freq).where(counts >= n)`` masks by label;
        unnamed and without attributes.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``, or *freq* is not a string.
    ValueError
        If the field or *freq* fails the checks of :func:`aggregate_time`.
    """
    field = _as_steps(field)
    check_frequency_is_an_offset_alias(freq)
    counts = _steps_per_cell(field, field.notnull(), freq).astype(np.int64)
    counts.name = None
    counts.attrs = {}
    return counts


def window_counts(
    field: xr.DataArray, windows: pd.IntervalIndex, *, labels: Any = None
) -> xr.DataArray:
    """How many values each window of :func:`reduce_windows` is formed from.

    Parameters
    ----------
    field, windows, labels:
        As for :func:`reduce_windows`.

    Returns
    -------
    xarray.DataArray
        ``int64``, the values that are not missing per window, zero where a
        window held nothing, on the ``time`` labels :func:`reduce_windows`
        gives the same arguments; unnamed and without attributes.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray`` or *windows* is not a
        ``pandas.IntervalIndex``.
    ValueError
        If the field, *windows* or *labels* fail the checks of
        :func:`reduce_windows`.
    """
    field = _as_steps(field)
    windows = _as_windows(windows, _time_index(field))
    labels, label_attrs = _as_window_labels(labels, windows)
    counts = _count_by_window(_without_interval_coords(field), windows)
    counts = counts.assign_coords({TIME: labels})
    counts[TIME].attrs = label_attrs
    return counts


# ── private helpers ───────────────────────────────────────────────────────────

#: The kinds for which ``min``, ``max`` and ``first`` over a window mean
#: something: the value is a level, so its extremes and its first reading are
#: levels too. A per-step total or a running total has no such reading.
_LEVEL_KINDS: frozenset[VariableKind] = frozenset(
    {VariableKind.TIMESTEP_END_STATE, VariableKind.TIMESTEP_MEAN, VariableKind.DAILY_RATE}
)

#: The CF ``cell_methods`` of a pool's window extreme or leading edge, which a
#: value at a step end makes literally true.
_CELL_METHODS_OF_A_READING: Mapping[str, str] = frozendict(
    {
        "min": "time: minimum",
        "max": "time: maximum",
        "first": "time: point",
    }
)

#: The kind whose values a method combines as they are, lent to a field that
#: declares none so that pySIPNET's ``resample`` will combine it.
_STAND_IN_KIND_FOR_METHOD: Mapping[str, VariableKind] = frozendict(
    {
        "sum": VariableKind.TIMESTEP_TOTAL,
        "mean": VariableKind.TIMESTEP_MEAN,
        "last": VariableKind.TIMESTEP_END_STATE,
    }
)

#: How the steps a window combines are summarized in its interval
#: coordinates: the earliest start, the latest end and the summed length.
_SPAN_OF_STEPS: Mapping[str, str] = frozendict(
    {
        TIMESTEP_START: "min",
        TIME: "max",
        TIMESTEP_LENGTH: "sum",
    }
)


def _as_steps(field: Any) -> xr.DataArray:
    """*field*, checked to be a field on a time axis, without its padding.

    The field, and so its time axis, is checked before the padding can be
    found, and only once the padding is gone can the steps left be counted.
    """
    check_field_is_on_a_time_axis(field)
    field = drop_padding(field)
    check_field_has_a_timestep(field, message_name=fields.message_name(field))
    return field


def _resample_without_a_kind(field: xr.DataArray, freq: str, method: str) -> xr.DataArray:
    """*field*, which declares no kind and has no interval coordinates, on calendar cells.

    pySIPNET's ``resample`` checks a method against the kind, so it is lent
    the kind *method* leaves unchanged, and the result keeps *field*'s own
    attributes, which say nothing of a kind.
    """
    stand_in = field.assign_attrs(kind=_STAND_IN_KIND_FOR_METHOD[method].value)
    result = resample(stand_in, freq, how=method)
    result.attrs = {key: value for key, value in field.attrs.items() if key != "output_decimals"}
    result.attrs["resampling"] = f"{method} over {freq}"
    return result


def _steps_per_cell(field: xr.DataArray, steps: xr.DataArray, freq: str) -> xr.DataArray:
    """How many of the steps *steps* marks fall in each cell pySIPNET forms of *field*.

    *steps* is a boolean per step, on *field*'s coordinates. It is summed by
    pySIPNET's ``resample`` as a per-step total, so the cells are labeled as
    the aggregate of *field* is.
    """
    indicator = steps.astype(np.float64)
    indicator.attrs = {"kind": VariableKind.TIMESTEP_TOTAL.value}
    return resample(indicator.rename("steps"), freq, how="sum")


def _resampling_method_for(
    field: xr.DataArray, kind: VariableKind | None, how: Any
) -> str:
    """The method :func:`aggregate_time` applies: *how* checked, or the kind's default."""
    message_name = fields.message_name(field)
    if how is None:
        check_kind_is_known(kind, message_name=message_name)
        check_kind_has_a_default_method(kind, message_name=message_name)
        return DEFAULT_METHOD_FOR_KIND[kind]
    if kind is None:
        check_how_is_a_resampling_method(how, message_name=message_name)
    else:
        check_resampling_method(kind, how, name=fields.message_name(field, quoted=False))
    return how


def _window_reduction_for(field: xr.DataArray, kind: VariableKind | None, how: Any) -> str:
    """*how* checked against :data:`WINDOW_REDUCTIONS` and the variable's kind."""
    check_how_is_a_window_reduction(how, fields.message_name(field))
    if how in RESAMPLING_METHODS:
        return _resampling_method_for(field, kind, how)
    check_kind_has_a_level(kind, how, message_name=fields.message_name(field))
    return how


def _has_interval_coords(field: xr.DataArray) -> bool:
    """Whether *field* carries both of pySIPNET's interval coordinates."""
    return TIMESTEP_START in field.coords and TIMESTEP_LENGTH in field.coords


def _without_interval_coords(field: xr.DataArray) -> xr.DataArray:
    """*field* without the coordinates on ``time`` other than ``time`` itself."""
    return field.drop_vars(
        [str(name) for name in field.coords if name != TIME and TIME in field[name].dims]
    )


def _time_index(field: xr.DataArray) -> pd.DatetimeIndex:
    """The field's ``time`` coordinate as a pandas index."""
    return pd.DatetimeIndex(field.coords[TIME].to_index())


def _start_index(field: xr.DataArray) -> pd.DatetimeIndex:
    """The field's ``timestep_start`` coordinate as a pandas index."""
    return pd.DatetimeIndex(field[TIMESTEP_START].to_index())


def _step_spacing(field: xr.DataArray) -> np.ndarray:
    """The gaps between consecutive ``time`` labels, in integer nanoseconds."""
    return np.diff(_time_index(field).as_unit("ns").asi8)


def _step_intervals(field: xr.DataArray) -> pd.IntervalIndex:
    """Each step's ``(timestep_start, time]``, in nanoseconds."""
    return pd.IntervalIndex.from_arrays(
        _start_index(field).as_unit("ns"), _time_index(field).as_unit("ns"), closed="right"
    )


def _as_step_array(field: xr.DataArray, values: np.ndarray) -> xr.DataArray:
    """*values*, one per timestep, as a ``DataArray`` on the field's ``time``."""
    return xr.DataArray(values, dims=TIME, coords={TIME: field[TIME]})


def _step_weights(field: xr.DataArray) -> xr.DataArray:
    """Step lengths in days, for a length-weighted mean; equal weights without them."""
    if TIMESTEP_LENGTH in field.coords:
        return _as_step_array(field, step_length(field, "d").values)
    check_steps_are_equally_spaced(field, message_name=fields.message_name(field))
    return _as_step_array(field, np.ones(field.sizes[TIME]))


def _steps_table(field: xr.DataArray) -> pd.DataFrame:
    """Each step's start, end and length, in nanoseconds, indexed by its end."""
    return pd.DataFrame(
        {
            TIMESTEP_START: field[TIMESTEP_START].values.astype("datetime64[ns]"),
            TIME: field[TIME].values.astype("datetime64[ns]"),
            TIMESTEP_LENGTH: field[TIMESTEP_LENGTH].values.astype("timedelta64[ns]"),
        },
        index=_time_index(field),
    )


def _window_reduction_attrs(
    attrs: Mapping[str, Any], kind: VariableKind | None, how: str, *, weighted: bool, name: str
) -> dict[str, Any]:
    """The variable's attributes, rewritten for a value formed over a window.

    *name* is the variable a refusal names.
    """
    if kind is not None and how in RESAMPLING_METHODS:
        out = resampled_attributes(attrs, kind, how, name=name)
    else:
        out = {key: value for key, value in attrs.items() if key != "output_decimals"}
    if how not in RESAMPLING_METHODS:
        if kind is not None:
            out["time_reference"] = f"the {how} of the {kind.value} values over the window"
        if kind is VariableKind.TIMESTEP_END_STATE:
            out["cell_methods"] = _CELL_METHODS_OF_A_READING[how]
        else:
            # The extreme of step means is no CF cell method: the input's
            # "time: mean" would be false of it, and time_reference says it.
            out.pop("cell_methods", None)
    weighting = f", weighted by {TIMESTEP_LENGTH}" if weighted else ""
    out["reduction"] = f"{how} over the window the label names{weighting}"
    return out


def _selection_attrs(attrs: Mapping[str, Any], kind: VariableKind | None) -> dict[str, Any]:
    """The variable's attributes, rewritten for a value read at a label."""
    out = dict(attrs)
    if kind is not None:
        out["time_reference"] = (
            f"{TIME_REFERENCE_FOR_KIND[kind]}, for the model timestep whose interval "
            "contains the label"
        )
    out["selection"] = (
        "the model timestep whose (timestep_start, time] interval contains the label"
    )
    return out


def _window_codes(field: xr.DataArray, windows: pd.IntervalIndex) -> np.ndarray:
    """For each step, the position of the window holding its end, or ``-1``."""
    return windows.get_indexer(_time_index(field))


def _reduce_by_window(
    bare: xr.DataArray, windows: pd.IntervalIndex, how: str, weights: xr.DataArray | None
) -> xr.DataArray:
    """*bare* reduced per window; ``NaN`` where a window has no step or a gap.

    *weights*, one per step, are what a ``mean`` is weighted by, and are read
    by it alone.
    """
    if not (_window_codes(bare, windows) >= 0).any():
        # groupby_bins refuses a binning that no value falls in.
        return _windows_filled_with(bare, len(windows), np.nan, float)

    def by_window(obj: xr.DataArray) -> Any:
        return obj.groupby_bins(TIME, windows)

    if how == "mean":
        reduced = by_window(bare * weights).sum(skipna=False) / by_window(weights).sum()
    elif how == "sum":
        reduced = by_window(bare).sum(skipna=False)
    else:
        reduced = getattr(by_window(bare), how)()
    # Every method: a window holding a gap is a gap. sum and the weighted mean
    # already propagate it; first, last, min and max skip missing values. An
    # empty window's "all" is NaN, and its value NaN already.
    complete = by_window(bare.notnull()).all() == 1
    return _drop_bin_labels(reduced.where(complete), bare.dims)


def _count_by_window(bare: xr.DataArray, windows: pd.IntervalIndex) -> xr.DataArray:
    """Values that are not missing, per window, as ``int64``."""
    if not (_window_codes(bare, windows) >= 0).any():
        return _windows_filled_with(bare, len(windows), 0, np.int64)
    counts = bare.notnull().groupby_bins(TIME, windows).sum()
    counts = _drop_bin_labels(counts, bare.dims).fillna(0).astype(np.int64)
    counts.name = None
    counts.attrs = {}
    return counts


def _drop_bin_labels(reduced: xr.DataArray, dims: Any) -> xr.DataArray:
    """A ``groupby_bins`` result put on ``time``, one entry per window, without labels."""
    bins = f"{TIME}_bins"
    return reduced.rename({bins: TIME}).drop_vars(TIME).transpose(*dims)


def _windows_filled_with(
    field: xr.DataArray, n_windows: int, fill: Any, dtype: Any
) -> xr.DataArray:
    """One *fill* per window, for when no timestep falls in any of them."""
    shape = [n_windows if dim == TIME else field.sizes[dim] for dim in field.dims]
    coords = {name: coord for name, coord in field.coords.items() if TIME not in coord.dims}
    return xr.DataArray(np.full(shape, fill, dtype=dtype), dims=field.dims, coords=coords)


def _window_interval_coords(
    field: xr.DataArray, windows: pd.IntervalIndex
) -> dict[str, xr.DataArray]:
    """The span of steps each window combined, where the field declares its steps."""
    if not _has_interval_coords(field):
        return {}
    codes = _window_codes(field, windows)
    inside = codes >= 0
    spans = (
        _steps_table(field)[inside]
        .groupby(codes[inside])
        # A plain dict: pandas rebuilds the mapping it is given as its own type
        # and fills it in place, which a frozendict refuses.
        .agg(dict(_SPAN_OF_STEPS))
        .reindex(np.arange(len(windows)))
    )
    return {
        TIMESTEP_START: xr.DataArray(
            spans[TIMESTEP_START].to_numpy("datetime64[ns]"),
            dims=TIME,
            attrs=without_stale_time_attributes(field[TIMESTEP_START].attrs),
        ),
        TIMESTEP_LENGTH: xr.DataArray(
            spans[TIMESTEP_LENGTH].to_numpy("timedelta64[ns]"),
            dims=TIME,
            attrs={
                **without_stale_time_attributes(field[TIMESTEP_LENGTH].attrs),
                "source": STEP_LENGTH_RESAMPLED,
            },
        ),
        SELECTED_STEP_COORD: xr.DataArray(
            spans[TIME].to_numpy("datetime64[ns]"),
            dims=TIME,
            attrs={"long_name": "End of the last model timestep combined into the window"},
        ),
    }


def _as_windows(windows: Any, stamps: pd.DatetimeIndex) -> pd.IntervalIndex:
    """*windows* checked, and put in the datetime resolution of *stamps*.

    pandas refuses to index one datetime resolution with another, and the two
    routinely differ: pySIPNET's axis is nanoseconds where a window built from
    dates is microseconds.
    """
    check_windows_are_valid(windows, stamps)
    left, right = pd.DatetimeIndex(windows.left), pd.DatetimeIndex(windows.right)
    unit = stamps.unit
    return pd.IntervalIndex.from_arrays(
        left.as_unit(unit), right.as_unit(unit), closed=windows.closed
    )


def _as_window_labels(labels: Any, windows: pd.IntervalIndex) -> tuple[pd.DatetimeIndex, dict]:
    """The result's ``time`` coordinate and its attributes.

    *labels* checked against *windows*, keeping a ``DataArray``'s attributes;
    or the windows' right edges, which :func:`_as_windows` has already made
    strictly increasing.
    """
    if labels is None:
        attrs = {
            "long_name": "Right edge of the window",
            "comment": "Each label is the right edge of its window.",
        }
        return pd.DatetimeIndex(windows.right), attrs
    index, attrs = _as_instants(labels, message_name="labels")
    check_labels_are_one_per_window(index, windows)
    return index, attrs


def _as_instants(times: Any, *, message_name: str) -> tuple[pd.DatetimeIndex, dict]:
    """*times* as a strictly increasing ``DatetimeIndex``, with a DataArray's attrs."""
    values = np.asarray(times.values if isinstance(times, xr.DataArray) else times).ravel()
    check_labels_are_timestamps(values, message_name=message_name)
    index = pd.DatetimeIndex(values)
    check_labels_are_given(index, message_name=message_name)
    check_labels_increase_strictly(index, message_name=message_name)
    attrs = dict(times.attrs) if isinstance(times, xr.DataArray) else {}
    return index, attrs


# ── checks ────────────────────────────────────────────────────────────────────


def check_run_spans_the_windows(
    field: xr.DataArray, windows: pd.IntervalIndex, message_name: str
) -> None:
    """Each window lies within the model field's record, less up to a step at either end."""
    field = _as_steps(field)
    check_field_has_interval_coords(field, message_name=message_name)
    ends = _time_index(field)
    check_windows_lie_within_the_record(
        _as_windows(windows, ends), _start_index(field), ends, message_name=message_name
    )


def check_how_is_a_window_reduction(how: Any, message_name: str | None = None) -> None:
    """*how* is one of :data:`WINDOW_REDUCTIONS`."""
    for_what = f" for {message_name}" if message_name else ""
    if not isinstance(how, str):
        raise TypeError(
            f"how must be a string, one of {list(WINDOW_REDUCTIONS)}{for_what}, got "
            f"{type(how).__name__}; pass one of them."
        )
    if how not in WINDOW_REDUCTIONS:
        raise ValueError(
            f"how must be one of {list(WINDOW_REDUCTIONS)}{for_what}, got {how!r}; write a "
            "reduction that is not one of them as an operator of its own."
        )


def check_frequency_is_an_offset_alias(freq: Any) -> None:
    """*freq* is a string pandas reads as a positive offset (pySIPNET's ``check_frequency``)."""
    check_frequency_is_a_string(freq)
    check_frequency(freq)


def check_frequency_is_a_string(freq: Any) -> None:
    """*freq* is a string, as a pandas offset alias is."""
    if not isinstance(freq, str):
        raise TypeError(
            f"freq must be a pandas offset alias such as '1D', 'MS' or 'YS', got "
            f"{type(freq).__name__} {freq!r}; pass the alias as a string."
        )


def check_how_is_a_resampling_method(how: Any, *, message_name: str) -> None:
    """*how* is one of :data:`RESAMPLING_METHODS`, for a variable of no known kind."""
    if not isinstance(how, str):
        raise TypeError(
            f"how must be a string, one of {list(RESAMPLING_METHODS)}, for {message_name}, got "
            f"{type(how).__name__}; pass one of them."
        )
    if how not in RESAMPLING_METHODS:
        raise ValueError(
            f"unknown resampling method {how!r} for {message_name}; choose from "
            f"{list(RESAMPLING_METHODS)}."
        )


def check_kind_is_known(kind: VariableKind | None, *, message_name: str) -> None:
    """The variable's kind is known, so a default method can be taken from it."""
    if kind is None:
        raise ValueError(
            f"{message_name} carries no 'kind' attribute and is not a SIPNET output or "
            "climate variable, so there is no default method to take from it; pass "
            "how='sum', 'mean' or 'last'."
        )


def check_kind_has_a_default_method(kind: VariableKind, *, message_name: str) -> None:
    """Some method leaves the variable's kind unchanged, and so is its default."""
    if kind not in DEFAULT_METHOD_FOR_KIND:
        raise ValueError(
            f"{message_name} is of kind {kind.value!r}, which no method leaves unchanged, so "
            "there is no default; pass how= explicitly."
        )


def check_kind_has_a_level(
    kind: VariableKind | None, how: str, *, message_name: str
) -> None:
    """A window's extreme or first reading is asked of a level, where the kind is known."""
    if kind is not None and kind not in _LEVEL_KINDS:
        raise ValueError(
            f"only a level (a pool, a step mean or a rate) has an extreme or a first reading "
            f"over a window, and {message_name} is of kind {kind.value!r}, so its {how!r} "
            "means nothing; sum a total, or take 'last' of a running total."
        )


def check_field_is_on_a_time_axis(field: Any) -> None:
    """*field* is a field with a ``time`` dim."""
    fields.validate_field(field)
    check_field_has_a_time_dim(field, message_name=fields.message_name(field))


def check_field_has_a_time_dim(field: xr.DataArray, *, message_name: str) -> None:
    """A field aligned in time has a ``time`` dim."""
    if TIME not in field.dims:
        raise ValueError(
            f"aligning in time needs a {TIME!r} dimension, and {message_name} has dims "
            f"{tuple(str(d) for d in field.dims)}; a static field has nothing to align, so "
            "use it as it is."
        )


def check_field_has_a_timestep(field: xr.DataArray, *, message_name: str) -> None:
    """At least one timestep is left once the padding is dropped."""
    if field[TIME].values.size == 0:
        raise ValueError(
            f"{message_name} has no timesteps left to aggregate, as a selection that matched "
            "nothing leaves it; select a period or a site the record covers."
        )


def check_steps_are_equally_spaced(field: xr.DataArray, *, message_name: str) -> None:
    """A field with no declared step lengths can be averaged only with equal weights."""
    spacing = _step_spacing(field)
    if spacing.size and (spacing != spacing[0]).any():
        raise ValueError(
            f"a mean over steps with no {TIMESTEP_LENGTH!r} weighs them equally, and the "
            f"steps of {message_name} are not all the same length; attach the step lengths, "
            "or reduce a field that carries them."
        )


def check_field_has_interval_coords(field: xr.DataArray, *, message_name: str) -> None:
    """*field* carries pySIPNET's step start and length, which the reader *message_name* reads."""
    missing = [c for c in (TIMESTEP_START, TIMESTEP_LENGTH) if c not in field.coords]
    if missing:
        raise ValueError(
            f"{message_name} reads the interval each step covers, and "
            f"{fields.message_name(field)} carries no {missing} coordinate; pass model output "
            "from pySIPNET, which carries both, not observed values."
        )


def check_observed_values_have_windows(
    observed_values: xr.DataArray, *, message_name: str
) -> None:
    """*observed_values* carries both window coordinates, as datetimes."""
    missing = [c for c in (WINDOW_START, WINDOW_END) if c not in observed_values.coords]
    if missing:
        raise ValueError(
            f"{message_name} carries no {missing} coordinate, so it documents no interval "
            "to reduce the model over; for a dated or static constraint, read an instant "
            "with select_timestep_at or the whole run with run_window."
        )
    for name in (WINDOW_START, WINDOW_END):
        dtype = observed_values[name].dtype
        if not pd.api.types.is_datetime64_any_dtype(dtype):
            raise ValueError(
                f"{message_name}: {name} must hold datetimes, got dtype {dtype}; convert "
                "it with pandas.to_datetime."
            )


def check_window_edges_are_present(
    start: pd.DatetimeIndex, end: pd.DatetimeIndex, *, message_name: str
) -> None:
    """Every window of some observed values has both edges."""
    if start.hasnans or end.hasnans:
        raise ValueError(
            f"{message_name}: a window edge is missing (NaT); drop the labels whose "
            "support is unknown, or read them at an instant with select_timestep_at."
        )


def check_window_ends_follow_their_starts(
    start: pd.DatetimeIndex, end: pd.DatetimeIndex, *, message_name: str
) -> None:
    """Each window of some observed values ends after it starts."""
    if not (end > start).all():
        raise ValueError(
            f"{message_name}: every window_end must follow its window_start, and some "
            "windows are reversed or empty; check how they were read."
        )


def check_windows_are_valid(windows: Any, stamps: pd.DatetimeIndex) -> None:
    """*windows* are non-empty, disjoint, increasing datetime intervals on the field's clock."""
    check_windows_are_an_interval_index(windows)
    check_windows_are_not_empty(windows)
    check_windows_hold_datetimes(windows)
    check_windows_have_both_edges(windows)
    check_every_window_holds_an_instant(windows)
    check_windows_do_not_overlap(windows)
    check_windows_increase(windows)
    check_windows_are_on_the_fields_clock(windows, stamps)


def check_windows_are_an_interval_index(windows: Any) -> None:
    """*windows* is a ``pandas.IntervalIndex``."""
    if not isinstance(windows, pd.IntervalIndex):
        raise TypeError(
            f"windows must be a pandas.IntervalIndex, got {type(windows).__name__}; build "
            "them with windows_from_observed_values(observed_values) or "
            "pd.IntervalIndex.from_arrays(starts, ends, closed='right')."
        )


def check_windows_are_not_empty(windows: pd.IntervalIndex) -> None:
    """There is at least one window."""
    if len(windows) == 0:
        raise ValueError("windows is empty, so there is nothing to reduce into; pass one or more.")


def check_windows_hold_datetimes(windows: pd.IntervalIndex) -> None:
    """The windows are intervals of datetimes."""
    if not pd.api.types.is_datetime64_any_dtype(windows.left.dtype):
        raise ValueError(
            f"windows must be intervals of datetimes, got {windows.left.dtype}; build "
            "them from timestamps."
        )


def check_windows_have_both_edges(windows: pd.IntervalIndex) -> None:
    """No window edge is ``NaT``."""
    if pd.DatetimeIndex(windows.left).hasnans or pd.DatetimeIndex(windows.right).hasnans:
        raise ValueError("windows hold a missing edge (NaT); drop the windows that lack one.")


def check_every_window_holds_an_instant(windows: pd.IntervalIndex) -> None:
    """No window is empty, which no step could fall in."""
    if np.asarray(windows.is_empty).any():
        raise ValueError(
            "a window holds no instant (its edges are equal and not both closed), so no "
            "step can fall in it; drop the empty windows."
        )


def check_windows_do_not_overlap(windows: pd.IntervalIndex) -> None:
    """No two windows share an instant, which would put a step in both."""
    if windows.is_overlapping:
        raise ValueError(
            "windows overlap, so a step could belong to two of them; reduce into "
            "non-overlapping windows."
        )


def check_windows_increase(windows: pd.IntervalIndex) -> None:
    """The windows are in increasing order."""
    if not pd.DatetimeIndex(windows.left).is_monotonic_increasing:
        raise ValueError("windows must be in increasing order; sort them first.")


def check_windows_are_on_the_fields_clock(
    windows: pd.IntervalIndex, stamps: pd.DatetimeIndex
) -> None:
    """The windows and the field's ``time`` are both naive or in one time zone."""
    left = pd.DatetimeIndex(windows.left)
    if left.tz != stamps.tz:
        raise ValueError(
            f"windows must be on the field's clock, and they are in time zone {left.tz} "
            f"where the field's time is in {stamps.tz}, so pandas would match no label; "
            "localize or convert one of them first."
        )


def check_windows_lie_within_the_record(
    windows: pd.IntervalIndex,
    starts: pd.DatetimeIndex,
    ends: pd.DatetimeIndex,
    *,
    message_name: str,
) -> None:
    """No window reaches a whole step or more beyond the record at either end."""
    first_step, last_step = ends[0] - starts[0], ends[-1] - starts[-1]
    before = (starts[0] - windows.left) >= first_step
    after = (windows.right - ends[-1]) >= last_step
    short = np.flatnonzero(np.asarray(before) | np.asarray(after))
    if short.size:
        window = windows[int(short[0])]
        raise ValueError(
            f"{message_name}: the window {window} reaches beyond the model record "
            f"({starts[0]} to {ends[-1]}) by a step or more, so a reduction over it "
            f"would cover only part of it ({short.size} of {len(windows)} windows); run "
            "the model over every observed window, or select the observations to the "
            "record (ObservationVector.select(time=...))."
        )


def check_kind_has_an_instant_value(kind: VariableKind | None, *, message_name: str) -> None:
    """The variable has a value at an instant, as a total or a running total does not."""
    if kind in (VariableKind.TIMESTEP_TOTAL, VariableKind.CUMULATIVE):
        fix = (
            "make it a rate first, with divide_with_units(field, step_length(field)) "
            "from pysipnet.arithmetic"
            if kind is VariableKind.TIMESTEP_TOTAL
            else "take its last value over a window with reduce_windows instead"
        )
        raise ValueError(
            f"{message_name} is of kind {kind.value!r}, which has no value at an instant; "
            f"{fix}."
        )
    if kind is VariableKind.TIMESTEP_START_COORDINATE:
        raise ValueError(
            f"{message_name} is a time coordinate, not a variable to read; read a model "
            "output variable instead."
        )


def check_labels_are_timestamps(values: np.ndarray, *, message_name: str) -> None:
    """The labels are timestamps, not numbers pandas would read as nanoseconds."""
    if values.dtype.kind in "iufb":
        raise ValueError(
            f"{message_name} must be timestamps, got dtype {values.dtype}, which would be "
            "read as nanoseconds since 1970; convert them with pandas.to_datetime first."
        )


def check_labels_are_given(index: pd.DatetimeIndex, *, message_name: str) -> None:
    """There is at least one label to read at."""
    if len(index) == 0:
        raise ValueError(
            f"no labels were given as {message_name}, so there is nothing to read at; pass "
            "at least one."
        )


def check_labels_increase_strictly(index: pd.DatetimeIndex, *, message_name: str) -> None:
    """The labels strictly increase, with no ``NaT``."""
    if index.hasnans or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(
            f"{message_name} must be strictly increasing timestamps with no NaT; sort them "
            "and drop the repeats and the missing ones."
        )


def check_labels_are_one_per_window(labels: pd.DatetimeIndex, windows: pd.IntervalIndex) -> None:
    """There is one label per window."""
    if len(labels) != len(windows):
        raise ValueError(
            f"labels has {len(labels)} entries for {len(windows)} windows; pass one label "
            "per window, or omit labels to take each window's right edge."
        )


def check_labels_are_on_the_fields_clock(
    labels: pd.DatetimeIndex, stamps: pd.DatetimeIndex, *, message_name: str
) -> None:
    """The labels and the field's ``time`` are both naive or in one time zone."""
    if labels.tz != stamps.tz:
        raise ValueError(
            f"the labels must be on the clock of {message_name}, and they are in time zone "
            f"{labels.tz} where its time is in {stamps.tz}; localize or convert one of them "
            "first."
        )


def check_steps_do_not_overlap(steps: pd.IntervalIndex, *, message_name: str) -> None:
    """No two steps' ``(timestep_start, time]`` intervals share an instant."""
    if steps.is_overlapping:
        raise ValueError(
            f"{message_name} has steps whose (timestep_start, time] intervals overlap, so a "
            "label could lie in two of them; rebuild the interval coordinates from the step "
            "ends, as pySIPNET's own axes are."
        )


def check_every_label_is_in_a_step(
    labels: pd.DatetimeIndex,
    position: np.ndarray,
    steps: pd.IntervalIndex,
    *,
    message_name: str,
) -> None:
    """Every label lies inside one step's ``(timestep_start, time]``."""
    outside = position < 0
    if outside.any():
        bad = labels[outside]
        raise ValueError(
            f"{len(bad)} label(s) fall in no timestep of {message_name}, the first being "
            f"{bad[0]}: the record covers ({steps.left[0]}, {steps.right[-1]}], and a label "
            "must lie inside one step's (timestep_start, time] interval; select the "
            "observations within the run, or run the model over the observed period."
        )
