"""Placing model output on an observation source's time grid.

The functions here put a model field on an observation source's time grid, build the
windows a reduction reads over, and count what went into each cell. The
observation operators are written with them, and a caller aggregates a field
with the same :func:`aggregate_time` before plotting it, so a predictive-check
figure cannot disagree with what the likelihood consumed.

Provided:

* ``aggregate_time(field, freq, *, how=None)`` -- combine timesteps into
  calendar cells (``"1D"``, ``"MS"``, ``"YS"``), the method taken from the
  variable's kind unless named.
* ``reduce_windows(field, windows, how, *, labels=None)`` -- combine
  timesteps into arbitrary non-overlapping windows, labeled as the caller
  asks; the form an observation operator uses to read the model over an
  observation's own support.
* ``select_timestep_at(field, times)`` -- for each label, the value of the
  model timestep whose interval contains it.
* ``windows_from_observed_values(observed_values)`` and ``run_window(field)``
  --
  the two ways windows are usually built; both return a
  ``pandas.IntervalIndex``.
* ``aggregation_counts(field, freq)`` and ``window_counts(field, windows)``
  -- how many values each cell or window was formed from, as integer arrays.
* ``check_run_spans_the_windows``, ``check_how_is_a_window_reduction`` and
  ``check_frequency_is_an_offset_alias`` -- the checks an operator or the
  forward model applies before reading the model.

How a method is chosen
----------------------
**Which methods mean anything is a property of the variable; which of them is
wanted is the caller's.** pySIPNET settles the first half: every model and
driver variable has a ``kind``, and its
``pysipnet.variables.RESAMPLING_METHODS_FOR_KIND`` says what may be done with
it. A total over a step adds; a pool at the end of a step does not, because
adding end-of-step values counts the same stock once per step; a rate or a
step mean averages, weighted by step length, because SIPNET's steps are not
all the same length. Every function here refuses a method the kind does not
admit, in pySIPNET's own words (the window extremes excepted).

For the second half :func:`aggregate_time` supplies a default, which
pySIPNET's ``resample`` deliberately does not: the one method that leaves the
variable the kind it already is (:data:`DEFAULT_METHOD_FOR_KIND`, derived from
pySIPNET's ``RESAMPLED_KIND``). A total sums, a step mean or a rate means, a
pool or a running total takes its last value. ``how=`` overrides the default,
and taking the time-weighted mean of a pool -- a different quantity, and a
different kind -- is exactly what it is for. :func:`reduce_windows` has no
default: a window is an observation's support, and what the observation wants
of the model over it is the operator's decision to state.

SIPNET's ``net_ecosystem_exchange`` is a per-timestep total, so 3-hourly to
daily is a **sum**; a mean is wrong by a factor of 8 and looks entirely
plausible. That is the error the default exists to make impossible to reach by
omission.

Aggregation is a verb the caller applies, never a plotter keyword::

    plot_time_series(aggregate_time(nee, "1D"))   # yes
    plot_time_series(nee, temporal_agg="1D")      # no

That keeps a real subtlety at the call site: quantile-of-daily-mean is not
daily-mean-of-quantile, and which one is wanted is a modeling choice.

Steps, cells and windows
------------------------
A model field carries pySIPNET's interval coordinates: ``time`` is the end of
each step, ``timestep_start`` its start and ``timestep_length`` its declared
length. Every function here reads the interval, not only the label. A step
belongs to the calendar cell or the window that contains its **end**; a
window edge that falls inside a step moves that whole step to the side its
end is on, and the result's ``timestep_length``, the sum of the steps
combined, shows it. A mean is weighted by ``timestep_length``. A cell or
window holding a ``NaN`` is ``NaN`` for every method, and the counts
functions say how many values went in. :func:`select_timestep_at` reads the
step whose interval ``(timestep_start, time]`` contains the label: for a
pool, the state at the end of that step; for a step mean or a rate, the mean
over it. A label exactly at a step end reads that step.

A time label whose interval coordinates are ``NaT`` and whose values are all
missing is the padding xarray leaves when runs on different time axes are
stacked and one of them is then selected. It is not a step, and every function
here drops it before anything is combined; a time label with a value and a
``NaT`` interval is refused, since which step its value covers is unknown.

Nothing here converts between clocks. The model field's ``time`` carries
pySIPNET's ``time_zone`` attribute, and the labels observed values supply are
taken to be on that clock; where they are not, the operator shifts them
first.
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

from sipnet_calibration.conventions import (
    SIPNET_ROW_LABEL_NAMES,
    TIME,
    TIMESTEP_LENGTH,
    TIMESTEP_START,
    WINDOW_END,
    WINDOW_START,
)
from sipnet_calibration import fields
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

#: The three ways pySIPNET's tables combine consecutive steps, under
#: pySIPNET's names; what :func:`aggregate_time` admits.
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
    field = _checked_steps(field)
    check_frequency_is_an_offset_alias(freq)
    kind = variable_kind(field, default=None)
    method = _method_for(field, kind, how)
    if kind is None and not _has_interval_coords(field):
        result = _resample_without_a_kind(field, freq, method)
    else:
        result = resample(field, freq, how=method)
    # pySIPNET's resample writes SIPNET's row labels onto its result; a field
    # that did not carry them does not gain them (fields' model output).
    added = [n for n in SIPNET_ROW_LABEL_NAMES if n in result.coords and n not in field.coords]
    result = result.drop_vars(added)
    if method == "last":
        # pySIPNET's last reads only a cell's last value, so a gap before it
        # is found by counting.
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

    The general form of :func:`aggregate_time`: the periods are any set of
    non-overlapping intervals rather than a calendar frequency, and the result
    is labeled as the caller asks. A step belongs to the window containing its
    ``time`` label, the step's end; steps outside every window are ignored.

    Parameters
    ----------
    field:
        As :func:`aggregate_time`: a ``time`` dimension, pySIPNET's interval
        coordinates where it is model output, and a ``kind`` attribute or a
        registry name where the variable is pySIPNET's.
    windows:
        A ``pandas.IntervalIndex`` of datetimes, non-overlapping, increasing,
        naive or in the field's time zone. Its ``closed`` side decides which of
        two adjacent windows a step ending on their shared edge belongs to;
        ``"right"`` is the natural choice for end-labeled steps, and is what
        :func:`windows_from_observed_values` and :func:`run_window` build.
    how:
        One of :data:`WINDOW_REDUCTIONS`. Required: a window is an
        observation's support, and what the observation wants of the model over
        it is the caller's statement. Checked against the variable's kind
        exactly as :func:`aggregate_time` checks, with ``min``, ``max`` and
        ``first`` admitted for a state, a step mean or a rate only.
    labels:
        The ``time`` coordinate of the result, one label per window, strictly
        increasing. Defaults to each window's right edge. An observation
        operator passes the observed values' own ``time`` coordinate, whose
        attributes are kept.

    Returns
    -------
    xarray.DataArray
        One value per window, with the field's other dimensions and their
        coordinates untouched and its ``time`` coordinate replaced by
        *labels*. A window holding no step, or holding a ``NaN``, is ``NaN``.
        Where the field carries pySIPNET's interval coordinates, the result
        carries them too, describing the steps each window actually combined:
        the earliest start, the sum of the lengths, and
        :data:`SELECTED_STEP_COORD` holding the latest end. Comparing that
        span with the window says how much of it the record covered. The
        variable's ``kind``, ``time_reference`` and ``cell_methods`` are
        rewritten as :func:`aggregate_time` rewrites them for ``sum``,
        ``mean`` and ``last``. For ``min``, ``max`` and ``first`` the kind is
        kept and ``time_reference`` says which reading was taken; only a
        pool keeps a ``cell_methods``, since CF has no word for the extreme
        of step means. A ``reduction`` attribute records what was done.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``, *windows* is not a
        ``pandas.IntervalIndex``, or *how* is not a string.
    ValueError
        If the field fails the checks of :func:`aggregate_time`; if *how* is
        not in :data:`WINDOW_REDUCTIONS`, or the variable's kind does not admit
        it; if *windows* is not of datetimes, is empty, holds ``NaT`` or an
        empty window, overlaps, is not increasing, or is in a different time
        zone from the field; or if *labels* are not timestamps, not one per
        window, or not strictly increasing.
    """
    field = _checked_steps(field)
    kind = variable_kind(field, default=None)
    how = _window_method_for(field, kind, how)
    windows = _checked_windows(windows, _time_index(field))
    labels, label_attrs = _checked_labels(labels, windows)

    weights = _step_weights(field) if how == "mean" else None
    reduced = _reduce_by_window(_without_interval_coords(field), windows, how, weights)
    reduced = reduced.assign_coords({TIME: labels})
    reduced[TIME].attrs = label_attrs
    reduced = reduced.assign_coords(_window_interval_coords(field, windows))
    reduced.name = field.name
    weighted = how == "mean" and TIMESTEP_LENGTH in field.coords
    reduced.attrs = _window_attrs(
        field.attrs, kind, how, weighted=weighted, name=fields.message_name(field, quoted=False)
    )
    return reduced


def select_timestep_at(field: xr.DataArray, times: Any) -> xr.DataArray:
    """For each label, the value of the model timestep whose interval contains it.

    The instant reading of a model field: the step with
    ``timestep_start < t <= time`` is the one that was running at ``t``, and
    its value is the state at the end of that step (for a pool), or the mean
    over it (for a step mean or a rate). A label exactly at a step end reads
    that step, so a snapshot at ``00:00`` reads the step ending at midnight.

    Parameters
    ----------
    field:
        A model field carrying pySIPNET's interval coordinates; any other
        dimensions are carried through.
    times:
        The labels to read at: a ``DataArray`` (observed values' ``time``
        coordinate, whose attributes are kept) or any datetime array-like,
        strictly increasing.

    Returns
    -------
    xarray.DataArray
        The field at the selected steps, on a ``time`` coordinate equal to
        *times*, with :data:`SELECTED_STEP_COORD` saying which step end each
        label read and the interval coordinates dropped. The attributes are the
        field's, with ``time_reference`` rewritten to say the value is the one
        of the step containing the label.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``.
    ValueError
        If the field fails the checks of :func:`aggregate_time`, has no
        interval coordinates, or has steps that overlap; if a label lies
        outside the record; if the variable is a per-step total or a running
        total (neither has a value *at* an instant; divide a total by the step
        length first); if *times* are not strictly increasing datetimes; or if
        *times* and the field's ``time`` are in different time zones.
    """
    field = _checked_steps(field)
    check_has_interval_coords(field, "select_timestep_at")
    kind = variable_kind(field, default=None)
    check_kind_has_an_instant_value(field, kind)
    labels, label_attrs = _checked_instants(times)

    ends = _time_index(field)
    check_same_clock(labels, ends, field)
    steps = _step_intervals(field)
    check_steps_do_not_overlap(field, steps)
    # Compared in nanoseconds, so a label finer than the axis is not truncated
    # onto the axis's resolution before the containment test.
    position = steps.get_indexer(labels.as_unit("ns"))
    check_every_label_is_in_a_step(field, labels, position, steps)

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
    selected.attrs = _selected_attrs(field.attrs, kind)
    return selected


def windows_from_observed_values(observed_values: xr.DataArray) -> pd.IntervalIndex:
    """The windows of observed values, as :func:`reduce_windows` takes them.

    Reads :data:`~sipnet_calibration.conventions.WINDOW_START` and
    :data:`~sipnet_calibration.conventions.WINDOW_END`, the
    one-dimensional form of CF ``time_bounds`` that
    :func:`sipnet_calibration.constraints.constraint_fields` puts on an annual
    constraint's ``time``. The windows are right-closed, so a model step ending
    on the shared edge of two years belongs to the year that ended.

    Parameters
    ----------
    observed_values:
        Observed values carrying the two window coordinates on ``time``.

    Returns
    -------
    pandas.IntervalIndex
        One right-closed window per ``time`` label, in the field's order, from
        its ``window_start`` to its ``window_end``.

    Raises
    ------
    TypeError
        If *observed_values* is not a ``DataArray``.
    ValueError
        If *observed_values* is not a field
        (:func:`sipnet_calibration.fields.validate_field`); if it carries no
        windows (a dated or static constraint
        documents no interval, and an operator over it reads an instant with
        :func:`select_timestep_at` or the run with :func:`run_window`
        instead); if a window edge is not a datetime or is ``NaT``; or if a
        window's end does not follow its start.
    """
    fields.validate_field(observed_values)
    message_name = fields.message_name(observed_values, "the observation source")
    check_has_windows(observed_values, message_name)
    start = pd.DatetimeIndex(observed_values[WINDOW_START].values)
    end = pd.DatetimeIndex(observed_values[WINDOW_END].values)
    check_windows_are_complete_and_ordered(start, end, message_name)
    return pd.IntervalIndex.from_arrays(start, end, closed="right")


def run_window(field: xr.DataArray) -> pd.IntervalIndex:
    """One right-closed window spanning a model field's whole record.

    From the first step's ``timestep_start`` to the last step's ``time``, so
    that every step belongs to it. For a static observation source, which documents no
    time at all, this is the window an operator reduces over.

    Parameters
    ----------
    field:
        A model field carrying pySIPNET's interval coordinates.

    Returns
    -------
    pandas.IntervalIndex
        One right-closed window, from the earliest step start to the latest
        step end. Padding a stack left on the field is not a step, and does
        not widen it.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``.
    ValueError
        If the field fails the checks of :func:`aggregate_time`, or carries no
        interval coordinates.
    """
    field = _checked_steps(field)
    check_has_interval_coords(field, "run_window")
    start = _start_index(field)
    end = _time_index(field)
    return pd.IntervalIndex.from_arrays([start.min()], [end.max()], closed="right")


def aggregation_counts(field: xr.DataArray, freq: str) -> xr.DataArray:
    """How many values each cell of :func:`aggregate_time` is formed from.

    The cells are the ones :func:`aggregate_time` forms, empty ones dropped,
    under the same labels, so ``aggregate_time(field, freq).where(counts >= n)``
    masks the aggregate by label.

    Parameters
    ----------
    field, freq:
        As for :func:`aggregate_time`.

    Returns
    -------
    xarray.DataArray
        ``int64``, the count of values that are not missing per cell, zero
        where a cell held only missing values, on the ``time`` coordinates
        :func:`aggregate_time` gives the same field; unnamed and without
        attributes, being a count of a field rather than a variable.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``, or *freq* is not a string.
    ValueError
        If the field or *freq* fails the checks of :func:`aggregate_time`.
    """
    field = _checked_steps(field)
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
        ``int64``, the count of values that are not missing per window, zero
        where a window held nothing, on the ``time`` labels
        :func:`reduce_windows` gives the same arguments; unnamed and without
        attributes.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray`` or *windows* is not a
        ``pandas.IntervalIndex``.
    ValueError
        If the field, *windows* or *labels* fail the checks of
        :func:`reduce_windows`.
    """
    field = _checked_steps(field)
    windows = _checked_windows(windows, _time_index(field))
    labels, label_attrs = _checked_labels(labels, windows)
    counts = _count_by_window(_without_interval_coords(field), windows)
    counts = counts.assign_coords({TIME: labels})
    counts[TIME].attrs = label_attrs
    return counts


# ── supporting helpers ────────────────────────────────────────────────────────

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

#: How the steps a cell or window combines are summarized in its interval
#: coordinates: the earliest start, the latest end and the summed length.
_SPAN_OF_STEPS: Mapping[str, str] = frozendict(
    {
        TIMESTEP_START: "min",
        TIME: "max",
        TIMESTEP_LENGTH: "sum",
    }
)


def _checked_steps(field: Any) -> xr.DataArray:
    """*field*, checked to be a field of steps, without the padding that is not.

    What every public function here starts with. The order matters: the
    field, and so its time axis, must be checked before the padding can be
    found, and only once the padding is gone can the remaining timesteps be
    counted.
    """
    check_field_is_on_a_time_axis(field)
    field = drop_padding(field)
    check_the_steps_are_aggregable(field)
    return field


def _resample_without_a_kind(field: xr.DataArray, freq: str, method: str) -> xr.DataArray:
    """*field*, which declares no kind and has no interval coordinates, on calendar cells.

    pySIPNET's ``resample`` checks a method against the kind, so it is given
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

    *steps* is a boolean per step, on *field*'s coordinates, and *field* has
    no padding. It is summed by pySIPNET's own ``resample`` as a per-step
    total, so the cells are labeled as the aggregate of *field* is.
    """
    indicator = steps.astype(np.float64)
    indicator.attrs = {"kind": VariableKind.TIMESTEP_TOTAL.value}
    return resample(indicator.rename("steps"), freq, how="sum")


def _method_for(field: xr.DataArray, kind: VariableKind | None, how: str | None) -> str:
    """The resampling method to apply, checked against what the kind admits."""
    if how is None:
        if kind is None:
            raise ValueError(
                f"{fields.message_name(field)} carries no 'kind' attribute and is not a SIPNET "
                "output or climate variable, so there is no rule to take the "
                "aggregation from. Pass how='sum', 'mean' or 'last'."
            )
        if kind not in DEFAULT_METHOD_FOR_KIND:
            raise ValueError(
                f"{fields.message_name(field)} is of kind {kind.value!r}, which no method leaves "
                "unchanged, so there is no default. Pass how= explicitly."
            )
        return DEFAULT_METHOD_FOR_KIND[kind]

    if kind is None:
        check_how_is_a_resampling_method(how, fields.message_name(field))
    else:
        check_resampling_method(kind, how, name=fields.message_name(field, quoted=False))
    return how


def _window_method_for(field: xr.DataArray, kind: VariableKind | None, how: Any) -> str:
    """*how* checked against :data:`WINDOW_REDUCTIONS` and the variable's kind."""
    check_how_is_a_window_reduction(how, fields.message_name(field))
    if how in RESAMPLING_METHODS:
        return _method_for(field, kind, how)
    check_kind_has_a_level(field, kind, how)
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


def _step_spacing_ns(field: xr.DataArray) -> np.ndarray:
    """The gaps between consecutive ``time`` labels, in integer nanoseconds."""
    return np.diff(_time_index(field).as_unit("ns").asi8)


def _step_intervals(field: xr.DataArray) -> pd.IntervalIndex:
    """Each step's ``(timestep_start, time]``, in nanoseconds."""
    return pd.IntervalIndex.from_arrays(
        _start_index(field).as_unit("ns"), _time_index(field).as_unit("ns"), closed="right"
    )


def _on_time(field: xr.DataArray, values: np.ndarray) -> xr.DataArray:
    """*values*, one per timestep, as a resamplable field on the field's ``time``."""
    return xr.DataArray(values, dims=TIME, coords={TIME: field[TIME]})


def _step_weights(field: xr.DataArray) -> xr.DataArray:
    """Step lengths in days, for a length-weighted mean; equal weights without them."""
    if TIMESTEP_LENGTH in field.coords:
        return _on_time(field, step_length(field, "d").values)
    check_steps_are_equally_spaced(field)
    return _on_time(field, np.ones(field.sizes[TIME]))


def _steps_frame(field: xr.DataArray) -> pd.DataFrame:
    """Each step's start, end and length, in nanoseconds, indexed by its end."""
    return pd.DataFrame(
        {
            TIMESTEP_START: field[TIMESTEP_START].values.astype("datetime64[ns]"),
            TIME: field[TIME].values.astype("datetime64[ns]"),
            TIMESTEP_LENGTH: field[TIMESTEP_LENGTH].values.astype("timedelta64[ns]"),
        },
        index=_time_index(field),
    )


def _window_attrs(
    attrs: Mapping[str, Any], kind: VariableKind | None, how: str, *, weighted: bool, name: str
) -> dict[str, Any]:
    """The variable's attributes, rewritten for a value formed over a window."""
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


def _selected_attrs(attrs: Mapping[str, Any], kind: VariableKind | None) -> dict[str, Any]:
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
    """*bare* reduced per window; ``NaN`` where a window has no step or a gap."""
    if not (_window_codes(bare, windows) >= 0).any():
        # groupby_bins refuses a binning that no value falls in.
        return _empty_windows(bare, len(windows), np.nan, float)

    def by_window(obj: xr.DataArray) -> Any:
        return obj.groupby_bins(TIME, windows)

    if how == "mean":
        assert weights is not None
        reduced = by_window(bare * weights).sum(skipna=False) / by_window(weights).sum()
    elif how == "sum":
        reduced = by_window(bare).sum(skipna=False)
    else:
        reduced = getattr(by_window(bare), how)()
    # Every method: a window holding a gap is a gap. sum and the weighted mean
    # already propagate it; first, last, min and max skip missing values. An
    # empty window's "all" is NaN, and its value NaN already.
    complete = by_window(bare.notnull()).all() == 1
    return _on_windows(reduced.where(complete), bare.dims)


def _count_by_window(bare: xr.DataArray, windows: pd.IntervalIndex) -> xr.DataArray:
    """Values that are not missing, per window, as ``int64``."""
    if not (_window_codes(bare, windows) >= 0).any():
        return _empty_windows(bare, len(windows), 0, np.int64)
    counts = bare.notnull().groupby_bins(TIME, windows).sum()
    counts = _on_windows(counts, bare.dims).fillna(0).astype(np.int64)
    counts.name = None
    counts.attrs = {}
    return counts


def _on_windows(reduced: xr.DataArray, dims: Any) -> xr.DataArray:
    """A ``groupby_bins`` result on ``time``, one entry per window, without labels."""
    bins = f"{TIME}_bins"
    return reduced.rename({bins: TIME}).drop_vars(TIME).transpose(*dims)


def _empty_windows(field: xr.DataArray, n_windows: int, fill: Any, dtype: Any) -> xr.DataArray:
    """One value per window when no timestep falls in any of them."""
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
        _steps_frame(field)[inside]
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


def _checked_windows(windows: Any, stamps: pd.DatetimeIndex) -> pd.IntervalIndex:
    """*windows* checked, and put in the datetime resolution of *stamps*.

    pandas refuses to index one datetime resolution with another, and the two
    routinely differ: pySIPNET's axis is nanoseconds where a window built from
    dates is microseconds.
    """
    check_windows_are_datetime_intervals(windows)
    left, right = pd.DatetimeIndex(windows.left), pd.DatetimeIndex(windows.right)
    check_windows_are_ordered_and_disjoint(windows, left, right)
    check_windows_are_on_the_fields_clock(left, stamps)
    unit = stamps.unit
    return pd.IntervalIndex.from_arrays(
        left.as_unit(unit), right.as_unit(unit), closed=windows.closed
    )


def _checked_labels(labels: Any, windows: pd.IntervalIndex) -> tuple[pd.DatetimeIndex, dict]:
    """The result's ``time`` coordinate and its attributes.

    *labels* checked against *windows*, keeping a ``DataArray``'s attributes;
    or the windows' right edges, which :func:`_checked_windows` has already
    made strictly increasing.
    """
    if labels is None:
        attrs = {
            "long_name": "Right edge of the window",
            "comment": "Each label is the right edge of its window.",
        }
        return pd.DatetimeIndex(windows.right), attrs
    index, attrs = _checked_instants(labels)
    if len(index) != len(windows):
        raise ValueError(
            f"labels has {len(index)} entries for {len(windows)} windows; pass one label "
            "per window, or omit labels to take each window's right edge."
        )
    return index, attrs


def _checked_instants(times: Any) -> tuple[pd.DatetimeIndex, dict]:
    """*times* as a strictly increasing ``DatetimeIndex``, with a DataArray's attrs."""
    values = np.asarray(times.values if isinstance(times, xr.DataArray) else times).ravel()
    if values.dtype.kind in "iufb":
        raise ValueError(
            f"the labels must be timestamps, got dtype {values.dtype}; numbers would be "
            "read as nanoseconds since 1970. Convert them with pandas.to_datetime first."
        )
    index = pd.DatetimeIndex(values)
    if len(index) == 0:
        raise ValueError("no labels were given; there is nothing to read at. Pass at least one.")
    if index.hasnans or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(
            "the labels must be strictly increasing timestamps with no NaT; sort them "
            "and drop the repeats and the missing ones."
        )
    attrs = dict(times.attrs) if isinstance(times, xr.DataArray) else {}
    return index, attrs


# ── checks ────────────────────────────────────────────────────────────────────


def check_run_spans_the_windows(
    field: xr.DataArray, windows: pd.IntervalIndex, message_name: str
) -> None:
    """Each window lies within the model field's record, less a step at each edge.

    A window the run covers only in part would be reduced over the part it
    covers, and an annual total over one month of model output would pass for
    the year's. Less than one step missing at either end of the record is
    allowed, so a window edge falling inside a step, or a record ending a few
    hours short of the window, is not refused. Padding a stack left on the
    field is dropped first, as :func:`reduce_windows` drops it, so the record
    is the field's own.

    Parameters
    ----------
    field:
        A model field carrying pySIPNET's interval coordinates.
    windows:
        The windows it is to be reduced over, as
        :func:`windows_from_observed_values` builds them.
    message_name:
        What the messages call the reader, such as the operator and the
        observation it predicts.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray`` or *windows* not an ``IntervalIndex``.
    ValueError
        Naming the first window that reaches a whole step or more beyond the
        record, and the record's span; or if the field or the windows fail
        the checks of :func:`reduce_windows`, or the field carries no
        interval coordinates.
    """
    field = _checked_steps(field)
    check_has_interval_coords(field, message_name)
    starts, ends = _start_index(field), _time_index(field)
    windows = _checked_windows(windows, ends)
    first_step, last_step = ends[0] - starts[0], ends[-1] - starts[-1]
    before = (starts[0] - windows.left) >= first_step
    after = (windows.right - ends[-1]) >= last_step
    short = np.flatnonzero(np.asarray(before) | np.asarray(after))
    if short.size:
        window = windows[int(short[0])]
        raise ValueError(
            f"{message_name}: the window {window} reaches beyond the model record "
            f"({starts[0]} to {ends[-1]}) by a step or more, so a reduction over it "
            f"would cover only part of it ({short.size} of {len(windows)} windows). "
            "Run the model over every observed window, or select the observations "
            "to the record (ObservationVector.select(time=...))."
        )


def check_how_is_a_window_reduction(how: Any, message_name: str | None = None) -> None:
    """*how* is one of :data:`WINDOW_REDUCTIONS`.

    Parameters
    ----------
    how:
        The reduction asked for.
    message_name:
        What the message says the reduction is for, if anything.

    Raises
    ------
    TypeError
        If *how* is not a string.
    ValueError
        If it is not one of :data:`WINDOW_REDUCTIONS`.
    """
    for_what = f" for {message_name}" if message_name else ""
    if not isinstance(how, str):
        raise TypeError(
            f"how must be a string, one of {list(WINDOW_REDUCTIONS)}{for_what}, got "
            f"{type(how).__name__}."
        )
    if how not in WINDOW_REDUCTIONS:
        raise ValueError(
            f"how must be one of {list(WINDOW_REDUCTIONS)}{for_what}, got {how!r}; a "
            "reduction that is not one of them is written as an operator of its own."
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


def check_how_is_a_resampling_method(how: Any, message_name: str) -> None:
    """*how* is one of :data:`RESAMPLING_METHODS`, for a field whose kind is unknown."""
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


def check_kind_has_a_level(field: xr.DataArray, kind: VariableKind | None, how: str) -> None:
    """A window's extreme or first reading is asked of a level, where the kind is known."""
    if kind is not None and kind not in _LEVEL_KINDS:
        raise ValueError(
            f"cannot take the {how!r} of {fields.message_name(field)} over a window: it is of kind "
            f"{kind.value!r}, and only a level (a pool, a step mean or a rate) has an "
            "extreme or a first reading. A total sums; a running total takes 'last'."
        )


def check_field_is_on_a_time_axis(field: Any) -> None:
    """*field* is a field with a ``time`` dim."""
    fields.validate_field(field)
    check_field_has_a_time_dim(field)


def check_field_has_a_time_dim(field: xr.DataArray) -> None:
    """A field aligned in time has a ``time`` dim."""
    if TIME not in field.dims:
        raise ValueError(
            f"aligning in time needs a {TIME!r} dimension; {fields.message_name(field)} has "
            f"dims {tuple(str(d) for d in field.dims)}. A static field has nothing to "
            "align."
        )


def check_the_steps_are_aggregable(field: xr.DataArray) -> None:
    """There is at least one step left once the padding is dropped.

    That the time labels strictly increase, so no two steps share a label and
    are summed as though they were consecutive, is the field contract's,
    which :func:`check_field_is_on_a_time_axis` checks first.
    """
    if field[TIME].values.size == 0:
        raise ValueError(
            f"{fields.message_name(field)} has no timesteps left to aggregate. An empty "
            f"{TIME!r} comes from a selection that matched nothing, or from "
            "a site of a stacked ensemble with no record of its own; select a "
            "period or a site the record covers."
        )


def check_steps_are_equally_spaced(field: xr.DataArray) -> None:
    """A field with no declared step lengths can be averaged only with equal weights."""
    spacing = _step_spacing_ns(field)
    if spacing.size and (spacing != spacing[0]).any():
        raise ValueError(
            f"{fields.message_name(field)} has no {TIMESTEP_LENGTH!r} coordinate and its steps "
            "are not all the same length, so a mean over them has no defined "
            "weighting. Attach the step lengths, or aggregate a field that "
            "carries them."
        )


def check_has_interval_coords(field: xr.DataArray, message_name: str) -> None:
    """*field* carries pySIPNET's step start and length."""
    missing = [c for c in (TIMESTEP_START, TIMESTEP_LENGTH) if c not in field.coords]
    if missing:
        raise ValueError(
            f"{message_name} reads the interval each step covers, and {fields.message_name(field)} "
            f"carries no {missing} coordinate. Model output from pySIPNET carries both; "
            "observed values do not, and are not what this reads."
        )


def check_has_windows(observed_values: xr.DataArray, message_name: str) -> None:
    """*observed_values* carries both window coordinates, as datetimes."""
    missing = [c for c in (WINDOW_START, WINDOW_END) if c not in observed_values.coords]
    if missing:
        raise ValueError(
            f"{message_name} carries no {missing} coordinate, so it documents no interval "
            "to reduce the model over. Only an annual constraint has windows; for a "
            "dated or static one read an instant with select_timestep_at, or the "
            "whole run with run_window."
        )
    for name in (WINDOW_START, WINDOW_END):
        dtype = observed_values[name].dtype
        if not pd.api.types.is_datetime64_any_dtype(dtype):
            raise ValueError(
                f"{message_name}: {name} must hold datetimes, got dtype {dtype}; convert "
                "it with pandas.to_datetime."
            )


def check_windows_are_complete_and_ordered(
    start: pd.DatetimeIndex, end: pd.DatetimeIndex, message_name: str
) -> None:
    """Every window edge is present, and each window's end follows its start."""
    if start.hasnans or end.hasnans:
        raise ValueError(
            f"{message_name}: a window edge is missing (NaT); drop the labels whose "
            "support is unknown, or read them at an instant with select_timestep_at."
        )
    if not (end > start).all():
        raise ValueError(
            f"{message_name}: every window_end must follow its window_start; the "
            "windows are reversed or empty, so check how they were read."
        )


def check_windows_are_datetime_intervals(windows: Any) -> None:
    """*windows* is a non-empty ``IntervalIndex`` of datetimes."""
    if not isinstance(windows, pd.IntervalIndex):
        raise TypeError(
            "windows must be a pandas.IntervalIndex, for instance from "
            "windows_from_observed_values(observed_values) or "
            "pd.IntervalIndex.from_arrays(starts, ends, closed='right'); got "
            f"{type(windows).__name__}."
        )
    if len(windows) == 0:
        raise ValueError("windows is empty, so there is nothing to reduce into; pass one or more.")
    if not pd.api.types.is_datetime64_any_dtype(windows.left.dtype):
        raise ValueError(
            f"windows must be intervals of datetimes, got {windows.left.dtype}; build "
            "them from timestamps."
        )


def check_windows_are_ordered_and_disjoint(
    windows: pd.IntervalIndex, left: pd.DatetimeIndex, right: pd.DatetimeIndex
) -> None:
    """The windows are complete, non-empty, non-overlapping and increasing."""
    if left.hasnans or right.hasnans:
        raise ValueError("windows hold a missing edge (NaT); drop the windows that lack one.")
    if np.asarray(windows.is_empty).any():
        raise ValueError(
            "a window holds no instant (its edges are equal and not both closed), so no "
            "step can fall in it; drop the empty windows."
        )
    if windows.is_overlapping:
        raise ValueError(
            "windows overlap, so a step could belong to two of them; reduce into "
            "non-overlapping windows."
        )
    if not left.is_monotonic_increasing:
        raise ValueError("windows must be in increasing order; sort them first.")


def check_windows_are_on_the_fields_clock(left: pd.DatetimeIndex, stamps: pd.DatetimeIndex) -> None:
    """The windows and the field's ``time`` are both naive or in one time zone."""
    if left.tz != stamps.tz:
        raise ValueError(
            f"windows are in time zone {left.tz} and the field's time coordinate in "
            f"{stamps.tz}; pandas matches no time labels across that difference. Localize or "
            "convert one of them first."
        )


def check_kind_has_an_instant_value(field: xr.DataArray, kind: VariableKind | None) -> None:
    """A total or a running total has no value at an instant."""
    if kind in (VariableKind.TIMESTEP_TOTAL, VariableKind.CUMULATIVE):
        fix = (
            "make it a rate first, with divide_with_units(field, step_length(field)) "
            "from pysipnet.arithmetic"
            if kind is VariableKind.TIMESTEP_TOTAL
            else "take its last value over a window with reduce_windows instead"
        )
        raise ValueError(
            f"{fields.message_name(field)} is of kind {kind.value!r}, which has no value at an "
            f"instant; {fix}."
        )
    if kind is VariableKind.TIMESTEP_START_COORDINATE:
        raise ValueError(
            f"{fields.message_name(field)} is a time coordinate, not a variable to read; read a "
            "model output variable instead."
        )


def check_same_clock(
    labels: pd.DatetimeIndex, ends: pd.DatetimeIndex, field: xr.DataArray
) -> None:
    """The labels and the field's ``time`` are both naive or in one time zone."""
    if labels.tz != ends.tz:
        raise ValueError(
            f"the labels are in time zone {labels.tz} and {fields.message_name(field)}'s time "
            f"coordinate in {ends.tz}; localize or convert one of them first."
        )


def check_steps_do_not_overlap(field: xr.DataArray, steps: pd.IntervalIndex) -> None:
    """No two steps' ``(timestep_start, time]`` intervals share an instant."""
    if steps.is_overlapping:
        raise ValueError(
            f"{fields.message_name(field)} has steps whose (timestep_start, time] intervals "
            "overlap, so a label could lie in two of them. pySIPNET's own axes never "
            "overlap; rebuild the interval coordinates from the step ends."
        )


def check_every_label_is_in_a_step(
    field: xr.DataArray,
    labels: pd.DatetimeIndex,
    position: np.ndarray,
    steps: pd.IntervalIndex,
) -> None:
    """Every label lies inside one step's ``(timestep_start, time]``."""
    outside = position < 0
    if outside.any():
        bad = labels[outside]
        raise ValueError(
            f"{len(bad)} label(s) fall in no timestep of {fields.message_name(field)}, the first "
            f"being {bad[0]}: the record covers ({steps.left[0]}, {steps.right[-1]}], and "
            "a label must lie inside one step's (timestep_start, time] interval. Select "
            "the observations within the run, or run the model over the observed period."
        )
