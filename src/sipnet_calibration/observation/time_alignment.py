"""Placing model output on an observation's time grid.

The functions here put a model field on an observation's time grid, build the
windows a reduction reads over, and count what went into each cell. The observation
operators are written with them, and the plotting layer aggregates with the
same :func:`aggregate_time`, so a predictive-check figure cannot disagree with
what the likelihood consumed.

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
* ``windows_from_time_bounds(observed_values)`` and ``run_window(field)`` --
  the two ways windows are usually built; both return a
  ``pandas.IntervalIndex``.
* ``aggregation_counts(field, freq)`` and ``window_counts(field, windows)``
  -- how many values each cell or window was formed from, as integer arrays.

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
admit, in pySIPNET's own words.

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
each step, ``time_step_start`` its start and ``time_step_length`` its declared
length. Every function here reads the interval, not only the label. A step
belongs to the calendar cell or the window that contains its **end**; a
window edge that falls inside a step moves that whole step to the side its
end is on, and the result's ``time_step_length``, the sum of the steps
combined, shows it. A mean is weighted by ``time_step_length``. A cell or
window holding a ``NaN`` is ``NaN`` for every method, and the counts
functions say how many values went in. :func:`select_timestep_at` reads the
step whose interval ``(time_step_start, time]`` contains the label: for a
pool, the state at the end of that step; for a step mean or a rate, the mean
over it. A label exactly at a step end reads that step.

Nothing here converts between clocks. The model field's ``time`` carries
pySIPNET's ``time_zone`` attribute, and the labels an observation supplies are
taken to be on that clock; where they are not, the operator shifts them
first.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, get_args

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.arithmetic import step_length
from pysipnet.dataset import TIME_DIMENSION, TIME_ZONE_UNDECLARED, assemble_time_coords
from pysipnet.resample import STEP_LENGTH_RESAMPLED
from pysipnet.resample import resample as pysipnet_resample
from pysipnet.variables import (
    CELL_METHODS_FOR_KIND,
    RESAMPLED_KIND,
    RESAMPLING_METHODS_FOR_KIND,
    TIME_REFERENCE_FOR_KIND,
    ResamplingMethod,
    VariableKind,
    resolve_climate_variable,
    resolve_output_variable,
)

from sipnet_calibration.conventions import TIME_BOUNDS_END, TIME_BOUNDS_START
from sipnet_calibration.fields import field_label

__all__ = [
    "DEFAULT_METHOD_FOR_KIND",
    "LENGTH_COORD",
    "RESAMPLING_METHODS",
    "SELECTED_STEP_COORD",
    "STALE_ON_A_COARSER_STEP",
    "START_COORD",
    "TIME_BOUNDS_END",
    "TIME_BOUNDS_START",
    "TIME_DIM",
    "WINDOW_REDUCTIONS",
    "aggregate_time",
    "aggregation_counts",
    "reduce_windows",
    "run_window",
    "select_timestep_at",
    "window_counts",
    "windows_from_time_bounds",
]

#: The dimension timesteps are combined along, pySIPNET's.
TIME_DIM = TIME_DIMENSION

#: pySIPNET's coordinates for the interval a row covers. ``time`` is its end,
#: :data:`START_COORD` its start and :data:`LENGTH_COORD` its declared
#: duration. A field carrying them is aggregated exactly as pySIPNET's own
#: ``resample`` would; one that does not gets calendar cell edges and, for a
#: mean, equal weights.
START_COORD = "time_step_start"
LENGTH_COORD = "time_step_length"

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

#: ``time`` attributes that describe the *source's* step and are false of a
#: coarser one. ``bounds`` names pySIPNET's two-dimensional ``time_bounds``
#: variable, which a field cannot carry (see
#: :mod:`sipnet_calibration.fields`).
STALE_ON_A_COARSER_STEP: tuple[str, ...] = ("bounds",)

#: What :func:`aggregate_time` does when ``how`` is not given: the one method
#: per kind that leaves a variable the kind it already is, read off pySIPNET's
#: ``RESAMPLED_KIND`` rather than written down. That exactly one method
#: preserves each kind is checked against pySIPNET's table by the tests.
DEFAULT_METHOD_FOR_KIND: dict[VariableKind, str] = {
    kind: method
    for (kind, method), resulting in RESAMPLED_KIND.items()
    if resulting == kind
}


def aggregate_time(
    field: xr.DataArray, freq: str, *, how: str | None = None
) -> xr.DataArray:
    """Combine a field's timesteps into coarser ones.

    Parameters
    ----------
    field:
        A field with a ``time`` dimension; any other dimensions are
        carried through untouched. A field produced by
        :func:`sipnet_calibration.fields.from_sipnet_output` or by
        :func:`sipnet_calibration.drivers.driver_fields` carries the ``kind``
        attribute this reads; so does any field taken from a pySIPNET Dataset.
    freq:
        A pandas offset alias for the coarser step: ``"1D"``, ``"7D"``,
        ``"MS"`` for calendar months, ``"YS"`` for calendar years.
    how:
        ``"sum"``, ``"mean"`` or ``"last"``. Omit it to take the method from
        the variable's kind, which is what makes a figure and a likelihood
        agree by default; pass it to ask for something else, such as the
        time-weighted mean of a pool.

    Returns
    -------
    xarray.DataArray
        The same name, dimensions and non-time coordinates, on a coarser
        ``time``. ``kind``, ``time_reference`` and ``cell_methods`` describe
        what the values now are, and a ``resampling`` attribute says how they
        were made. Cells no step falls in are dropped.

        A field carrying :data:`START_COORD` and :data:`LENGTH_COORD` keeps
        them, covering the span the cell's steps actually covered rather than
        the calendar cell: the start is the earliest step start, the ``time``
        the latest step end, and the length the sum of the declared lengths, so
        a cell the record only partly fills can be told from a full one by
        comparing the two. A field without them -- an observation, say -- gets
        calendar cell edges and **carries nothing about cell coverage**, so the
        first and last cells of such a record are partial with nothing to say
        so. For an extensive variable that is a fraction of a period reported
        in the units of a whole one; until this is settled (see the Notes) a
        caller comparing such daily totals against anything should drop the
        boundary cells itself, or mask on :func:`aggregation_counts`.

        ``time`` keeps the attributes that are still true of it and loses
        :data:`STALE_ON_A_COARSER_STEP`, which describe the step it had before.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``.
    ValueError
        If *field* has no datetime ``time`` coordinate free of ``NaT``, has no
        steps left after the padding is dropped, or timestamps that do not
        strictly increase; if *freq* is not a pandas offset alias, or is finer
        than the field's own steps, which would interpolate rather than
        aggregate; if its interval coordinates are not one-dimensional on
        ``time``, which is what stacking runs on different time axes leaves;
        if it declares a ``kind`` that is not one of pySIPNET's; if *how* is
        not one of :data:`RESAMPLING_METHODS`; if the variable's kind does not
        admit *how*, with pySIPNET's own explanation and the methods that
        would work; if *how* is omitted and the variable's kind cannot be
        determined; or if a mean is asked for on unequal steps that carry no
        :data:`LENGTH_COORD` to weight by.

    Notes
    -----
    Cells are right-closed and right-labeled, as pySIPNET's ``resample`` makes
    them, because ``time`` is the **end** of a step: a step ending at midnight
    belongs to the day that ended, so ``"1D"`` cells are ``(00:00, 24:00]`` and
    a daily record resamples to itself.

    A mean is weighted by the step length, which matters wherever the steps are
    not all equal, as SIPNET's own alternating day and night steps are not. A
    field with no :data:`LENGTH_COORD` is
    weighted equally, and refused if its steps are not equally spaced, rather
    than quietly averaging steps of different lengths.

    A cell holding a ``NaN`` is ``NaN``, for every method. Aggregating half a
    day of a gappy record into a number that looks like a whole day is how a
    gap stops being visible.

    A timestamp that is not a step of the field -- the padding xarray's
    alignment leaves when runs of different lengths are stacked and one site is
    then selected -- is dropped before anything is combined, rather than
    treated as a gap. Leaving it in would turn every cell it fell in to
    ``NaN``, interior cells included, because pySIPNET snaps each interior
    step's end onto the next step's start while a truncated run's last end is
    its declared length, so the odd timestamp lands mid-record.

    What is **not** solved is the boundary cell of a field carrying no declared
    step lengths; see Returns. :func:`aggregation_counts` says how many values
    went into each cell, which tells a partial cell from a full one only where
    the caller knows how many values a full cell holds.
    """
    field = _checked_steps(field)
    check_frequency(freq)
    check_not_upsampling(field, freq)
    kind = _variable_kind(field)
    method = _method_for(field, kind, how)
    weights = _step_weights(field) if method == "mean" else None

    values = _combine(_without_interval_coords(field), freq, method, weights)
    keep = _nonempty_cells(field, freq)
    values = values.isel({TIME_DIM: keep})

    result = values.rename(field.name) if field.name is not None else values
    result = result.assign_coords(_aggregated_time_coords(field, freq, keep))
    result[TIME_DIM].attrs = _without_stale_interval_attrs(result[TIME_DIM].attrs)
    weighted = method == "mean" and LENGTH_COORD in field.coords
    result.attrs = _aggregated_attrs(field.attrs, kind, method, freq, weighted)
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
        :func:`windows_from_time_bounds` and :func:`run_window` build.
    how:
        One of :data:`WINDOW_REDUCTIONS`. Required: a window is an
        observation's support, and what the observation wants of the model over
        it is the caller's statement. Checked against the variable's kind
        exactly as :func:`aggregate_time` checks, with ``min``, ``max`` and
        ``first`` admitted for a state, a step mean or a rate only.
    labels:
        The ``time`` coordinate of the result, one label per window, strictly
        increasing. Defaults to each window's right edge. An observation
        operator passes the observation's own ``time`` coordinate, whose
        attributes are kept.

    Returns
    -------
    xarray.DataArray
        One value per window, with the field's other dimensions and their
        coordinates untouched and its ``time`` coordinate replaced by
        *labels*. A window holding no step, or holding a ``NaN``, is ``NaN``.
        Where the field carries :data:`START_COORD` and :data:`LENGTH_COORD`,
        the result carries them too, describing the steps each window
        actually combined: the earliest start, the sum of the lengths, and
        :data:`SELECTED_STEP_COORD` holding the latest end. Comparing that
        span with the window says how much of it the record covered. The
        variable's ``kind``, ``time_reference`` and ``cell_methods`` are
        rewritten as :func:`aggregate_time` rewrites them for ``sum``,
        ``mean`` and ``last``; for ``min``, ``max`` and ``first`` the kind is
        kept and ``time_reference`` says which reading was taken. A
        ``reduction`` attribute records what was done.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``.
    ValueError
        If the field fails the checks of :func:`aggregate_time`; if *how* is
        not in :data:`WINDOW_REDUCTIONS`, or the variable's kind does not admit
        it; if *windows* is not a datetime ``IntervalIndex``, is empty, holds
        ``NaT``, overlaps, is not increasing, or is in a different time zone
        from the field; or if *labels* are not timestamps, not one per window,
        or not strictly increasing.
    """
    field = _checked_steps(field)
    kind = _variable_kind(field)
    how = _window_method_for(field, kind, how)
    windows = _checked_windows(windows, _time_index(field))
    labels, label_attrs = _checked_labels(labels, windows)

    weights = _step_weights(field) if how == "mean" else None
    reduced = _reduce_by_window(_without_interval_coords(field), windows, how, weights)
    reduced = reduced.assign_coords({TIME_DIM: labels})
    reduced[TIME_DIM].attrs = label_attrs
    reduced = reduced.assign_coords(_window_interval_coords(field, windows))
    reduced.name = field.name
    weighted = how == "mean" and LENGTH_COORD in field.coords
    reduced.attrs = _window_attrs(field.attrs, kind, how, weighted=weighted)
    return reduced


def select_timestep_at(field: xr.DataArray, times: Any) -> xr.DataArray:
    """For each label, the value of the model timestep whose interval contains it.

    The instant reading of a model field: the step with
    ``time_step_start < t <= time`` is the one that was running at ``t``, and
    its value is the state at the end of that step (for a pool), or the mean
    over it (for a step mean or a rate). A label exactly at a step end reads
    that step, so a snapshot at ``00:00`` reads the step ending at midnight.

    Parameters
    ----------
    field:
        A model field carrying pySIPNET's :data:`START_COORD` and
        :data:`LENGTH_COORD`; any other dimensions are carried through.
    times:
        The labels to read at: a ``DataArray`` (an observation's ``time``
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
        If the field fails the checks of :func:`aggregate_time` or has no
        interval coordinates; if a label lies outside the record; if the
        variable is a per-step total or a running total (neither has a value
        *at* an instant; divide a total by the step length first); if *times*
        are not strictly increasing datetimes; or if *times* and the field's
        ``time`` are in different time zones.
    """
    field = _checked_steps(field)
    check_has_interval_coords(field, "select_timestep_at")
    kind = _variable_kind(field)
    check_kind_has_an_instant_value(field, kind)
    labels, label_attrs = _checked_instants(times)

    ends = _time_index(field)
    starts = pd.DatetimeIndex(field[START_COORD].to_index())
    check_same_clock(labels, ends, field)
    position = _containing_steps(labels, starts, ends)
    check_every_label_is_in_a_step(field, labels, position, starts, ends)

    selected = _without_interval_coords(field.isel({TIME_DIM: position}))
    selected = selected.assign_coords(
        {
            TIME_DIM: xr.DataArray(labels, dims=TIME_DIM, attrs=label_attrs),
            SELECTED_STEP_COORD: xr.DataArray(
                ends.values[position],
                dims=TIME_DIM,
                attrs={
                    "long_name": "End of the model timestep that was read",
                    "comment": (
                        "The step whose (time_step_start, time] interval contains the label."
                    ),
                },
            ),
        }
    )
    selected.attrs = _selected_attrs(field.attrs, kind)
    return selected


def windows_from_time_bounds(observed_values: xr.DataArray) -> pd.IntervalIndex:
    """An observation field's attribution intervals, as windows for :func:`reduce_windows`.

    Reads :data:`TIME_BOUNDS_START` and :data:`TIME_BOUNDS_END`, the
    one-dimensional form of CF ``time_bounds`` that
    :func:`sipnet_calibration.constraints.constraint_fields` puts on an annual
    product's ``time``. The windows are right-closed, so a model step ending
    on the shared edge of two years belongs to the year that ended.

    Parameters
    ----------
    observed_values:
        An observation field carrying the two bounds coordinates on ``time``.

    Returns
    -------
    pandas.IntervalIndex
        One right-closed window per ``time`` label, in the field's order, from
        its ``time_bounds_start`` to its ``time_bounds_end``.

    Raises
    ------
    ValueError
        If the field carries no bounds (a dated or static product documents no
        interval, and an operator over it reads an instant with
        :func:`select_timestep_at` or the run with :func:`run_window`
        instead); if a bound is not a datetime or is ``NaT``; or if a
        window's end does not follow its start.
    """
    who = field_label(observed_values, "the observation")
    check_has_time_bounds(observed_values, who)
    start = pd.DatetimeIndex(observed_values[TIME_BOUNDS_START].values)
    end = pd.DatetimeIndex(observed_values[TIME_BOUNDS_END].values)
    if start.hasnans or end.hasnans:
        raise ValueError(f"{who}: a time bound is missing (NaT).")
    if not (end > start).all():
        raise ValueError(f"{who}: every time_bounds_end must follow its start.")
    return pd.IntervalIndex.from_arrays(start, end, closed="right")


def run_window(field: xr.DataArray) -> pd.IntervalIndex:
    """One right-closed window spanning a model field's whole record.

    From the first step's :data:`START_COORD` to the last step's ``time``, so
    that every step belongs to it. For a static observation, which documents no
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
    start = pd.DatetimeIndex(field[START_COORD].to_index())
    end = _time_index(field)
    return pd.IntervalIndex.from_arrays([start.min()], [end.max()], closed="right")


def aggregation_counts(field: xr.DataArray, freq: str) -> xr.DataArray:
    """How many values each cell of :func:`aggregate_time` is formed from.

    The cells are the ones :func:`aggregate_time` forms (right-closed, empty
    ones dropped), so the counts align with the aggregate by position and can
    mask it.

    Parameters
    ----------
    field, freq:
        As for :func:`aggregate_time`.

    Returns
    -------
    xarray.DataArray
        ``int64``, the count of values that are not missing per cell, zero
        where a cell held only missing values; unnamed and without attributes,
        being a count of a field rather than a variable. The cells are labeled
        as ``resample`` labels them, which differs from the relabeling
        :func:`aggregate_time` gives a field carrying pySIPNET's interval
        coordinates, so compare by position.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``.
    ValueError
        If the field fails the checks of :func:`aggregate_time`, or *freq* is
        not a pandas offset alias or is finer than the field's steps.
    """
    field = _checked_steps(field)
    check_frequency(freq)
    check_not_upsampling(field, freq)
    counts = _grouped(field.notnull(), freq).sum()
    counts = counts.fillna(0).astype(np.int64)
    counts = counts.isel({TIME_DIM: _nonempty_cells(field, freq)})
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
        If *field* is not a ``DataArray``.
    ValueError
        If the field, *windows* or *labels* fail the checks of
        :func:`reduce_windows`.
    """
    field = _checked_steps(field)
    windows = _checked_windows(windows, _time_index(field))
    labels, label_attrs = _checked_labels(labels, windows)
    counts = _count_by_window(_without_interval_coords(field), windows)
    counts = counts.assign_coords({TIME_DIM: labels})
    counts[TIME_DIM].attrs = label_attrs
    return counts


# ── supporting helpers ────────────────────────────────────────────────────────

#: The kinds for which ``min``, ``max`` and ``first`` over a window mean
#: something: the value is a level, so its extremes and its first reading are
#: levels too. A per-step total or a running total has no such reading.
_LEVEL_KINDS: frozenset[VariableKind] = frozenset(
    {VariableKind.TIMESTEP_END_STATE, VariableKind.TIMESTEP_MEAN, VariableKind.DAILY_RATE}
)

#: Names the refusal probe's own coordinates occupy.
_PROBE_COORD_NAMES = frozenset({TIME_DIM, START_COORD, LENGTH_COORD, "time_bounds"})

#: How the steps a cell or window combines are summarized in its interval
#: coordinates: the earliest start, the latest end and the summed length.
_SPAN_OF_STEPS: dict[str, str] = {START_COORD: "min", TIME_DIM: "max", LENGTH_COORD: "sum"}


def _checked_steps(field: Any) -> xr.DataArray:
    """*field*, checked to be a field of steps, without the padding that is not.

    What every public function here that reads a model field starts with. The
    order matters: the time axis must be readable before the padding can be
    found, and only once the padding is gone can the remaining labels be
    checked to increase.
    """
    check_the_time_axis(field)
    check_interval_coords_are_one_dimensional(field)
    field = _only_real_steps(field)
    check_the_steps_are_aggregable(field)
    return field


def _variable_kind(field: xr.DataArray) -> VariableKind | None:
    """The field's pySIPNET kind, from its attributes or the registries, or ``None``."""
    declared = field.attrs.get("kind")
    if declared is not None:
        try:
            return VariableKind(declared)
        except ValueError as error:
            raise ValueError(
                f"{field_label(field)} declares kind={declared!r}, which is not one of "
                f"{[k.value for k in VariableKind]}."
            ) from error
    if field.name is None:
        return None
    for resolve in (resolve_output_variable, resolve_climate_variable):
        try:
            return resolve(str(field.name)).kind
        except KeyError:
            continue
    return None


def _method_for(field: xr.DataArray, kind: VariableKind | None, how: str | None) -> str:
    """The resampling method to apply, checked against what the kind admits."""
    if how is None:
        if kind is None:
            raise ValueError(
                f"{field_label(field)} carries no 'kind' attribute and is not a SIPNET "
                "output or climate variable, so there is no rule to take the "
                "aggregation from. Pass how='sum', 'mean' or 'last'."
            )
        if kind not in DEFAULT_METHOD_FOR_KIND:
            raise ValueError(
                f"{field_label(field)} is of kind {kind.value!r}, which no method leaves "
                "unchanged, so there is no default. Pass how= explicitly."
            )
        return DEFAULT_METHOD_FOR_KIND[kind]

    if how not in RESAMPLING_METHODS:
        raise ValueError(
            f"Unknown resampling method {how!r} for {field_label(field)}; choose from "
            f"{list(RESAMPLING_METHODS)}."
        )
    if kind is not None and how not in RESAMPLING_METHODS_FOR_KIND[kind]:
        _refuse(field, kind, how)
    return how


def _window_method_for(field: xr.DataArray, kind: VariableKind | None, how: Any) -> str:
    """*how* checked against :data:`WINDOW_REDUCTIONS` and the variable's kind."""
    if not isinstance(how, str) or how not in WINDOW_REDUCTIONS:
        raise ValueError(
            f"how must be one of {list(WINDOW_REDUCTIONS)} for {field_label(field)}, "
            f"got {how!r}."
        )
    if how in RESAMPLING_METHODS:
        return _method_for(field, kind, how)
    if kind is not None and kind not in _LEVEL_KINDS:
        raise ValueError(
            f"Cannot take the {how!r} of {field_label(field)} over a window: it is of kind "
            f"{kind.value!r}, and only a level (a pool, a step mean or a rate) has an "
            f"extreme or a first reading. A total sums; a running total takes 'last'."
        )
    return how


def _refuse(field: xr.DataArray, kind: VariableKind, method: str) -> None:
    """Raise pySIPNET's own explanation of why *method* is meaningless for *kind*.

    Obtained by putting the pair to ``pysipnet.resample.resample``, on two rows
    that exist only to be refused, rather than by restating a reason that would
    then be this project's to keep in step with pySIPNET's.
    """
    label = field_label(field)
    # The probe cannot hold a variable named after one of the time coordinates
    # it must carry, so a field with such a name, or none, is put to pySIPNET
    # under a stand-in and named properly again in the message.
    name = None if field.name is None else str(field.name)
    stand_in = name if name is not None and name not in _PROBE_COORD_NAMES else "the_field"
    try:
        pysipnet_resample(_refusal_probe(stand_in, kind), "1D", how=method)
    except ValueError as refusal:
        raise ValueError(str(refusal).replace(repr(stand_in), label, 1)) from None
    raise ValueError(
        f"Cannot aggregate {label} with {method!r}: it is of kind {kind.value!r}, which "
        f"admits only {sorted(RESAMPLING_METHODS_FOR_KIND[kind])}."
    )


def _refusal_probe(name: str, kind: VariableKind) -> xr.Dataset:
    """Two rows with pySIPNET's time layout, holding one variable of *kind*."""
    start = np.array(["2000-01-01T00:00", "2000-01-01T12:00"], dtype="datetime64[ns]")
    length = np.array([12, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
    coords = assemble_time_coords(
        start=start,
        end=start + length,
        length=length,
        attributes_for=lambda _name: {},
        length_source="a probe, to obtain pySIPNET's refusal",
        time_zone=TIME_ZONE_UNDECLARED,
    )
    return xr.Dataset({name: (TIME_DIM, np.zeros(2), {"kind": kind.value})}, coords=coords)


def _only_real_steps(field: xr.DataArray) -> xr.DataArray:
    """*field* without the timestamps that are not steps of it.

    Selecting one site out of a stack of runs on different time axes leaves the
    union of those axes, so a site's shorter record carries timestamps whose
    interval coordinates are ``NaT``. They are not steps: leaving them in would
    turn every cell that holds one into ``NaN``, boundary and interior alike,
    and a ``NaT`` length casts to the ``int64`` sentinel rather than to a
    missing value, which is a step of minus 292 years. Dropping them is the
    only treatment that leaves the cells they fall in meaning what they say.
    """
    mask = np.ones(field.sizes[TIME_DIM], dtype=bool)
    for name in (START_COORD, LENGTH_COORD):
        if name in field.coords:
            mask &= ~np.isnat(field[name].values)
    return field if mask.all() else field.isel({TIME_DIM: mask})


def _without_interval_coords(field: xr.DataArray) -> xr.DataArray:
    """*field* without the coordinates on ``time`` other than ``time`` itself."""
    return field.drop_vars(
        [str(name) for name in field.coords if name != TIME_DIM and TIME_DIM in field[name].dims]
    )


def _time_index(field: xr.DataArray) -> pd.DatetimeIndex:
    """The field's ``time`` coordinate as a pandas index."""
    return pd.DatetimeIndex(field.coords[TIME_DIM].to_index())


def _on_time(field: xr.DataArray, values: np.ndarray) -> xr.DataArray:
    """*values*, one per timestep, as a resamplable field on the field's ``time``."""
    return xr.DataArray(values, dims=TIME_DIM, coords={TIME_DIM: field[TIME_DIM]})


def _step_weights(field: xr.DataArray) -> xr.DataArray:
    """Step lengths in days, for a length-weighted mean; equal weights without them."""
    if LENGTH_COORD in field.coords:
        return _on_time(field, step_length(field, "d").values)
    check_steps_are_equally_spaced(field)
    return _on_time(field, np.ones(field.sizes[TIME_DIM]))


def _steps_frame(field: xr.DataArray) -> pd.DataFrame:
    """Each step's start, end and length, in nanoseconds, indexed by its end."""
    return pd.DataFrame(
        {
            START_COORD: field[START_COORD].values.astype("datetime64[ns]"),
            TIME_DIM: field[TIME_DIM].values.astype("datetime64[ns]"),
            LENGTH_COORD: field[LENGTH_COORD].values.astype("timedelta64[ns]"),
        },
        index=_time_index(field),
    )


def _grouped(obj: xr.DataArray, freq: str) -> Any:
    """Cells of *freq*, right-closed and right-labeled because ``time`` is a step end."""
    return obj.resample({TIME_DIM: freq}, closed="right", label="right")


def _combine(
    bare: xr.DataArray, freq: str, method: str, weights: xr.DataArray | None
) -> xr.DataArray:
    """*bare* reduced onto cells of *freq* by *method*."""
    if method == "sum":
        return _grouped(bare, freq).sum(skipna=False)
    if method == "last":
        return _grouped(bare, freq).last(skipna=False)
    assert weights is not None
    return _grouped(bare * weights, freq).sum(skipna=False) / _grouped(weights, freq).sum()


def _nonempty_cells(field: xr.DataArray, freq: str) -> np.ndarray:
    """Which cells of *freq* at least one step falls in.

    Resampling produces a cell for every period the record spans, including the
    ones no step lands in; an empty cell sums to zero, which is not a
    measurement.
    """
    ones = _on_time(field, np.ones(field.sizes[TIME_DIM]))
    return np.asarray(_grouped(ones, freq).sum().values > 0)


def _aggregated_time_coords(
    field: xr.DataArray, freq: str, keep: np.ndarray
) -> dict[str, xr.DataArray]:
    """The ``time`` coordinates of the coarser field.

    A field carrying pySIPNET's interval coordinates gets them back, describing
    the span its steps covered and built by pySIPNET's own
    ``assemble_time_coords`` so the wording cannot drift. One that does not
    keeps its own ``time`` attributes on the calendar cell edges. Either way
    the caller drops :data:`STALE_ON_A_COARSER_STEP`.
    """
    if START_COORD not in field.coords or LENGTH_COORD not in field.coords:
        return {}
    # pandas bins exactly as xarray's resample does, and sums the lengths as
    # integer nanoseconds rather than through a float.
    cells = _steps_frame(field).resample(freq, closed="right", label="right")
    spans = cells.agg(_SPAN_OF_STEPS).iloc[np.flatnonzero(keep)]
    built = assemble_time_coords(
        start=spans[START_COORD].to_numpy("datetime64[ns]"),
        end=spans[TIME_DIM].to_numpy("datetime64[ns]"),
        length=spans[LENGTH_COORD].to_numpy("timedelta64[ns]"),
        attributes_for=lambda name: dict(field[name].attrs) if name in field.coords else {},
        length_source=STEP_LENGTH_RESAMPLED,
        time_zone=field[TIME_DIM].attrs.get("time_zone", TIME_ZONE_UNDECLARED),
    )
    return {
        name: xr.DataArray(values, dims=TIME_DIM, attrs=_without_stale_interval_attrs(attrs))
        for name, (_dims, values, attrs) in built.items()
        if name in (TIME_DIM, START_COORD, LENGTH_COORD)
    }


def _without_stale_interval_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """*attrs* less the ones that describe the interval before it was coarsened."""
    return {key: value for key, value in attrs.items() if key not in STALE_ON_A_COARSER_STEP}


def _aggregated_attrs(
    attrs: Mapping[str, Any],
    kind: VariableKind | None,
    method: str,
    freq: str,
    weighted: bool,
) -> dict[str, Any]:
    """The variable's attributes, rewritten to describe what the values now are."""
    out = _with_resampled_kind(attrs, kind, method)
    weighting = f", weighted by {LENGTH_COORD}" if weighted else ""
    of_kind = f" of {kind.value} values" if kind is not None else ""
    out["resampling"] = f"{method}{of_kind} over {freq}{weighting}"
    return out


def _window_attrs(
    attrs: Mapping[str, Any], kind: VariableKind | None, how: str, *, weighted: bool
) -> dict[str, Any]:
    """The variable's attributes, rewritten for a value formed over a window."""
    if how in RESAMPLING_METHODS:
        out = _with_resampled_kind(attrs, kind, how)
    else:
        out = _with_resampled_kind(attrs, None, how)
        if kind is not None:
            out["time_reference"] = f"the {how} of the {kind.value} values over the window"
            method = {"min": "minimum", "max": "maximum"}.get(how, "point")
            out["cell_methods"] = f"time: {method}"
    weighting = f", weighted by {LENGTH_COORD}" if weighted else ""
    out["reduction"] = f"{how} over the window the label names{weighting}"
    return out


def _with_resampled_kind(
    attrs: Mapping[str, Any], kind: VariableKind | None, method: str
) -> dict[str, Any]:
    """*attrs* with ``kind``, ``time_reference`` and ``cell_methods`` for *method*.

    Unchanged but for ``output_decimals`` where *kind* is ``None``. The source
    file's printf precision no longer describes a combined value, so it goes
    either way.
    """
    out = dict(attrs)
    out.pop("output_decimals", None)
    if kind is None:
        return out
    new_kind = RESAMPLED_KIND[(kind, method)]
    out["kind"] = new_kind.value
    out["time_reference"] = TIME_REFERENCE_FOR_KIND[new_kind]
    cell_methods = CELL_METHODS_FOR_KIND[new_kind]
    if cell_methods is None:
        out.pop("cell_methods", None)
    else:
        out["cell_methods"] = cell_methods
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
        "the model timestep whose (time_step_start, time] interval contains the label"
    )
    return out


def _containing_steps(
    labels: pd.DatetimeIndex, starts: pd.DatetimeIndex, ends: pd.DatetimeIndex
) -> np.ndarray:
    """For each label, the position of the step containing it, or ``-1``.

    Compared in nanoseconds, so a label finer than the axis is not truncated
    onto the axis's resolution before the containment test.
    """
    ends, starts, labels = ends.as_unit("ns"), starts.as_unit("ns"), labels.as_unit("ns")
    position = np.searchsorted(ends.asi8, labels.asi8, side="left")
    inside = position < len(ends)
    inside[inside] = starts.asi8[position[inside]] < labels.asi8[inside]
    return np.where(inside, position, -1)


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
        return obj.groupby_bins(TIME_DIM, windows)

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
    counts = bare.notnull().groupby_bins(TIME_DIM, windows).sum()
    counts = _on_windows(counts, bare.dims).fillna(0).astype(np.int64)
    counts.name = None
    counts.attrs = {}
    return counts


def _on_windows(reduced: xr.DataArray, dims: Any) -> xr.DataArray:
    """A ``groupby_bins`` result on ``time``, one entry per window, without labels."""
    bins = f"{TIME_DIM}_bins"
    return reduced.rename({bins: TIME_DIM}).drop_vars(TIME_DIM).transpose(*dims)


def _empty_windows(field: xr.DataArray, n_windows: int, fill: Any, dtype: Any) -> xr.DataArray:
    """One entry per window when no row falls in any of them."""
    shape = [n_windows if dim == TIME_DIM else field.sizes[dim] for dim in field.dims]
    coords = {name: coord for name, coord in field.coords.items() if TIME_DIM not in coord.dims}
    return xr.DataArray(np.full(shape, fill, dtype=dtype), dims=field.dims, coords=coords)


def _window_interval_coords(
    field: xr.DataArray, windows: pd.IntervalIndex
) -> dict[str, xr.DataArray]:
    """The span of steps each window combined, where the field declares its steps."""
    if START_COORD not in field.coords or LENGTH_COORD not in field.coords:
        return {}
    codes = _window_codes(field, windows)
    inside = codes >= 0
    spans = (
        _steps_frame(field)[inside]
        .groupby(codes[inside])
        .agg(_SPAN_OF_STEPS)
        .reindex(np.arange(len(windows)))
    )
    return {
        START_COORD: xr.DataArray(
            spans[START_COORD].to_numpy("datetime64[ns]"),
            dims=TIME_DIM,
            attrs=_without_stale_interval_attrs(field[START_COORD].attrs),
        ),
        LENGTH_COORD: xr.DataArray(
            spans[LENGTH_COORD].to_numpy("timedelta64[ns]"),
            dims=TIME_DIM,
            attrs={
                **_without_stale_interval_attrs(field[LENGTH_COORD].attrs),
                "source": STEP_LENGTH_RESAMPLED,
            },
        ),
        SELECTED_STEP_COORD: xr.DataArray(
            spans[TIME_DIM].to_numpy("datetime64[ns]"),
            dims=TIME_DIM,
            attrs={"long_name": "End of the last model timestep combined into the window"},
        ),
    }


def _is_datetime(dtype: Any) -> bool:
    """Whether *dtype* is a datetime one, time-zone-aware ones included."""
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return isinstance(dtype, pd.DatetimeTZDtype)
    return bool(np.issubdtype(dtype, np.datetime64))


def _checked_windows(windows: Any, stamps: pd.DatetimeIndex) -> pd.IntervalIndex:
    """*windows* checked, and put in the datetime resolution of *stamps*.

    pandas refuses to index one datetime resolution with another, and the two
    routinely differ: pySIPNET's axis is nanoseconds where a window built from
    dates is microseconds.
    """
    if not isinstance(windows, pd.IntervalIndex):
        raise ValueError(
            "windows must be a pandas.IntervalIndex, for instance from "
            "windows_from_time_bounds(observed_values) or "
            "pd.IntervalIndex.from_arrays(starts, ends, closed='right'); got "
            f"{type(windows).__name__}."
        )
    if len(windows) == 0:
        raise ValueError("windows is empty, so there is nothing to reduce into.")
    if not _is_datetime(windows.left.dtype):
        raise ValueError(f"windows must be intervals of datetimes, got {windows.left.dtype}.")
    left, right = pd.DatetimeIndex(windows.left), pd.DatetimeIndex(windows.right)
    if left.hasnans or right.hasnans:
        raise ValueError("windows hold a missing edge (NaT).")
    if left.tz != stamps.tz:
        raise ValueError(
            f"windows are in time zone {left.tz} and the field's time coordinate in "
            f"{stamps.tz}; pandas matches no rows across that difference. Localize or "
            "convert one of them first."
        )
    if windows.is_overlapping:
        raise ValueError(
            "windows overlap, so a step could belong to two of them; reduce into "
            "non-overlapping windows."
        )
    if not left.is_monotonic_increasing:
        raise ValueError("windows must be in increasing order.")
    unit = stamps.unit
    return pd.IntervalIndex.from_arrays(
        left.as_unit(unit), right.as_unit(unit), closed=windows.closed
    )


def _checked_labels(labels: Any, windows: pd.IntervalIndex) -> tuple[pd.DatetimeIndex, dict]:
    """The result's ``time`` coordinate and its attributes.

    *labels* checked against *windows*, keeping a ``DataArray``'s attributes;
    or the windows' right edges.
    """
    if labels is None:
        index = pd.DatetimeIndex(windows.right)
        attrs = {
            "long_name": "Right edge of the window",
            "comment": "Each label is the right edge of its window.",
        }
        what = "the windows' right edges"
    else:
        index, attrs = _checked_instants(labels)
        what = "labels"
    if len(index) != len(windows):
        raise ValueError(
            f"labels has {len(index)} entries for {len(windows)} windows; one label per "
            "window."
        )
    if index.hasnans or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(
            f"{what} must be strictly increasing timestamps with no NaT; pass labels= to "
            "name the windows otherwise."
        )
    return index, attrs


def _checked_instants(times: Any) -> tuple[pd.DatetimeIndex, dict]:
    """*times* as a strictly increasing ``DatetimeIndex``, with a DataArray's attrs."""
    values = np.asarray(times.values if isinstance(times, xr.DataArray) else times).ravel()
    if values.dtype.kind in "iufb":
        raise ValueError(
            f"the labels must be timestamps, got dtype {values.dtype}; numbers would be "
            "read as nanoseconds since 1970."
        )
    index = pd.DatetimeIndex(values)
    if len(index) == 0:
        raise ValueError("no labels were given; there is nothing to read at.")
    if index.hasnans or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("the labels must be strictly increasing timestamps with no NaT.")
    attrs = dict(times.attrs) if isinstance(times, xr.DataArray) else {}
    return index, attrs


# ── checks ────────────────────────────────────────────────────────────────────


def check_the_time_axis(field: Any) -> None:
    """*field* is a DataArray with a datetime ``time`` coordinate and no NaT."""
    if not isinstance(field, xr.DataArray):
        advice = (
            " A Dataset holds several variables, whose kinds differ; align one at a time."
            if isinstance(field, xr.Dataset)
            else ""
        )
        raise TypeError(f"expected an xarray.DataArray, got {type(field).__name__}.{advice}")
    if TIME_DIM not in field.dims:
        raise ValueError(
            f"aligning in time needs a {TIME_DIM!r} dimension; {field_label(field)} has "
            f"dims {tuple(str(d) for d in field.dims)}."
        )
    if TIME_DIM not in field.coords:
        raise ValueError(
            f"{field_label(field)} has a {TIME_DIM!r} dimension but no {TIME_DIM!r} "
            "coordinate, so there is nothing to place its rows by."
        )
    dtype = field.coords[TIME_DIM].dtype
    if not _is_datetime(dtype):
        raise ValueError(
            f"the {TIME_DIM!r} coordinate of {field_label(field)} has dtype {dtype}, and "
            "alignment needs datetimes."
        )
    if pd.isna(field.coords[TIME_DIM].values).any():
        raise ValueError(
            f"the {TIME_DIM!r} coordinate of {field_label(field)} holds a missing timestamp "
            "(NaT), so its rows cannot be placed. Drop those rows first."
        )


def check_interval_coords_are_one_dimensional(field: xr.DataArray) -> None:
    """The interval coordinates must describe the whole field, not one slice of it.

    :func:`sipnet_calibration.fields.stack_sipnet_outputs` gives them a ``site``
    or ``member`` dimension when the runs it stacked ran over different time
    axes, and a coarser step then has no single span or length.
    """
    offenders = [
        name
        for name in (START_COORD, LENGTH_COORD)
        if name in field.coords and field[name].dims != (TIME_DIM,)
    ]
    if offenders:
        raise ValueError(
            f"{field_label(field)} has {offenders} on dims "
            f"{[tuple(str(d) for d in field[n].dims) for n in offenders]} rather "
            f"than on {TIME_DIM!r} alone, which happens when runs on different "
            "time axes are stacked together. Select one site, or drop those "
            "coordinates to aggregate on calendar cells with equal weights."
        )


def check_the_steps_are_aggregable(field: xr.DataArray) -> None:
    """There is at least one step, and no two of them share or reverse a label.

    Duplicate labels would be summed together as though they were consecutive
    steps, which is how one record counted twice comes back looking like a
    larger flux.
    """
    times = field[TIME_DIM].values
    if times.size == 0:
        raise ValueError(
            f"{field_label(field)} has no timesteps left to aggregate. An empty "
            f"{TIME_DIM!r} comes from a selection that matched nothing, or from "
            "a site of a stacked ensemble with no record of its own."
        )
    steps = np.diff(times.astype("datetime64[ns]").astype("int64"))
    if (steps <= 0).any():
        where = int(np.flatnonzero(steps <= 0)[0]) + 1
        raise ValueError(
            f"{field_label(field)} has timestamps that do not increase: row {where} "
            f"({times[where]}) does not follow row {where - 1} "
            f"({times[where - 1]}). Sort the field on {TIME_DIM!r}, and drop or "
            "combine the duplicates; two rows sharing a label would be added "
            "together as though they were consecutive steps."
        )


def check_frequency(freq: Any) -> None:
    """Raise unless *freq* is a pandas offset alias naming a positive period."""
    bad = f"freq must be a pandas offset alias such as '1D', 'MS' or 'YS', got {freq!r}"
    if not isinstance(freq, str):
        raise ValueError(bad)
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except Exception as error:
        raise ValueError(bad) from error
    if offset is None:
        raise ValueError(bad)
    if offset.n <= 0:
        raise ValueError(
            f"freq={freq!r} names a period of {offset.n} steps, which cannot group "
            "anything; pass a positive frequency."
        )


def check_not_upsampling(field: xr.DataArray, freq: str) -> None:
    """Raise if every period of *freq* is shorter than the field's own spacing.

    Upsampling returns a field that is mostly ``NaN`` and raises nothing. The
    comparison is the smallest gap in the time axis against the longest period
    *freq* produces on it, so a sparse or gapped record at its own cadence
    passes, and calendar periods of varying length are measured rather than
    assumed.
    """
    stamps = _time_index(field)
    if len(stamps) < 2:
        return
    ones = _on_time(field, np.ones(field.sizes[TIME_DIM]))
    labels = pd.DatetimeIndex(_grouped(ones, freq).sum().coords[TIME_DIM].to_index())
    offset = pd.tseries.frequencies.to_offset(freq)
    edges = labels.append(pd.DatetimeIndex([labels[-1] + offset]))
    spacing = int(np.diff(stamps.as_unit("ns").asi8).min())
    period = int(np.diff(edges.as_unit("ns").asi8).max())
    if spacing > period:
        raise ValueError(
            f"freq={freq!r} produces periods of at most {pd.Timedelta(period, 'ns')} on a "
            f"record whose steps are at least {pd.Timedelta(spacing, 'ns')} apart, so this "
            "would interpolate rather than aggregate and return a field that is mostly "
            "missing. Pass a coarser frequency."
        )


def check_steps_are_equally_spaced(field: xr.DataArray) -> None:
    """A field with no declared step lengths can be averaged only with equal weights."""
    spacing = np.diff(field[TIME_DIM].values.astype("datetime64[ns]").astype("int64"))
    if spacing.size and (spacing != spacing[0]).any():
        raise ValueError(
            f"{field_label(field)} has no {LENGTH_COORD!r} coordinate and its steps are "
            "not all the same length, so a mean over them has no defined "
            "weighting. Attach the step lengths, or aggregate a field that "
            "carries them."
        )


def check_has_interval_coords(field: xr.DataArray, what: str) -> None:
    """*field* carries pySIPNET's step start and length."""
    missing = [c for c in (START_COORD, LENGTH_COORD) if c not in field.coords]
    if missing:
        raise ValueError(
            f"{what} reads the interval each step covers, and {field_label(field)} carries "
            f"no {missing} coordinate. Model output from pySIPNET carries both; an "
            "observation field does not, and is not what this reads."
        )


def check_has_time_bounds(observed_values: xr.DataArray, who: str) -> None:
    """*observed_values* carries both time-bounds coordinates, as datetimes."""
    missing = [c for c in (TIME_BOUNDS_START, TIME_BOUNDS_END) if c not in observed_values.coords]
    if missing:
        raise ValueError(
            f"{who} carries no {missing} coordinate, so it documents no interval to "
            "reduce the model over. Only an annual product has time bounds; for a "
            "dated or static one read an instant with select_timestep_at, or the "
            "whole run with run_window."
        )
    for name in (TIME_BOUNDS_START, TIME_BOUNDS_END):
        if not _is_datetime(observed_values[name].dtype):
            raise ValueError(
                f"{who}: {name} must hold datetimes, got dtype {observed_values[name].dtype}."
            )


def check_kind_has_an_instant_value(field: xr.DataArray, kind: VariableKind | None) -> None:
    """A total or a running total has no value at an instant."""
    if kind in (VariableKind.TIMESTEP_TOTAL, VariableKind.CUMULATIVE):
        fix = (
            "divide it by pysipnet.arithmetic.step_length() first, with "
            "pysipnet.arithmetic.divide_with_units, which makes it a rate that does"
            if kind is VariableKind.TIMESTEP_TOTAL
            else "take its last value over a window with reduce_windows instead"
        )
        raise ValueError(
            f"{field_label(field)} is of kind {kind.value!r}, which has no value at an "
            f"instant; {fix}."
        )
    if kind is VariableKind.TIMESTEP_START_COORDINATE:
        raise ValueError(f"{field_label(field)} is a time coordinate, not a variable to read.")


def check_same_clock(
    labels: pd.DatetimeIndex, ends: pd.DatetimeIndex, field: xr.DataArray
) -> None:
    """The labels and the field's ``time`` are both naive or in one time zone."""
    if labels.tz != ends.tz:
        raise ValueError(
            f"the labels are in time zone {labels.tz} and {field_label(field)}'s time "
            f"coordinate in {ends.tz}; localize or convert one of them first."
        )


def check_every_label_is_in_a_step(
    field: xr.DataArray,
    labels: pd.DatetimeIndex,
    position: np.ndarray,
    starts: pd.DatetimeIndex,
    ends: pd.DatetimeIndex,
) -> None:
    """Every label lies inside one step's ``(time_step_start, time]``."""
    outside = position < 0
    if outside.any():
        bad = labels[outside]
        raise ValueError(
            f"{len(bad)} label(s) fall in no timestep of {field_label(field)}, the first "
            f"being {bad[0]}: the record covers ({starts[0]}, {ends[-1]}], and a label "
            "must lie inside one step's (time_step_start, time] interval. Select the "
            "observations within the run, or run the model over the observed period."
        )
