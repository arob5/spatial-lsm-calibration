"""Placing model output on an observation's time grid.

The functions here put a model field on an observation's time grid, build the
windows that takes, and count what went into each cell. The observation
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

#: The ways consecutive steps may be combined, as pySIPNET names them; what
#: :func:`aggregate_time` admits, because each is checked against a kind and
#: these are the three pySIPNET's tables define.
RESAMPLING_METHODS: tuple[str, ...] = get_args(ResamplingMethod)

#: What :func:`reduce_windows` admits: pySIPNET's three, and the extremes and
#: the leading edge of a window, which are meaningful for a state, a step mean
#: or a rate and refused for a total or a running total.
WINDOW_REDUCTIONS: tuple[str, ...] = (*RESAMPLING_METHODS, "min", "max", "first")

#: The kinds for which ``min``, ``max`` and ``first`` over a window mean
#: something: the value is a level, so its extremes and its first reading are
#: levels too. A per-step total or a running total has no such reading.
_LEVEL_KINDS: frozenset[VariableKind] = frozenset(
    {VariableKind.TIMESTEP_END_STATE, VariableKind.TIMESTEP_MEAN, VariableKind.DAILY_RATE}
)

#: The coordinates an observation field carries for the interval each of its
#: values is attributed to, one-dimensional on ``time``: the CF ``time_bounds``
#: pair, which a ``DataArray`` cannot carry two-dimensionally.
TIME_BOUNDS_START = "time_bounds_start"
TIME_BOUNDS_END = "time_bounds_end"

#: The coordinate :func:`select_timestep_at` adds, saying which model step end
#: each label read.
SELECTED_STEP_COORD = "selected_timestep_end"

#: ``time`` attributes that describe the *source's* step and are false of a
#: coarser one. ``bounds`` names pySIPNET's two-dimensional ``time_bounds``
#: variable, which a field cannot carry (see
#: :mod:`sipnet_calibration.fields`).
STALE_ON_A_COARSER_STEP: tuple[str, ...] = ("bounds",)


#: The coordinate :func:`reduce_windows` groups by, internal to one call.
_WINDOW = "_window"


def _kind_preserving_methods() -> dict[VariableKind, str]:
    """The one method per kind that leaves a variable the kind it already is.

    Read off pySIPNET's ``RESAMPLED_KIND`` rather than written down, so a
    change there is either carried through or raises here.
    """
    defaults: dict[VariableKind, str] = {}
    for (kind, method), resulting in RESAMPLED_KIND.items():
        if resulting != kind:
            continue
        if kind in defaults:
            raise RuntimeError(
                f"pySIPNET admits both {defaults[kind]!r} and {method!r} as "
                f"kind-preserving for {kind.value!r}, so there is no one "
                "default; aggregate_time must be given an explicit rule."
            )
        defaults[kind] = method
    return defaults


#: What :func:`aggregate_time` does when ``how`` is not given.
DEFAULT_METHOD_FOR_KIND: dict[VariableKind, str] = _kind_preserving_methods()


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
        boundary cells itself. Until then, mask on :func:`aggregation_counts`.

        ``time`` keeps the attributes that are still true of it and loses
        :data:`STALE_ON_A_COARSER_STEP`, which describe the step it had before.

    Raises
    ------
    ValueError
        If *field* is not a ``DataArray`` with a datetime ``time`` coordinate
        free of ``NaT``, has no steps left after the padding is dropped, or
        timestamps that do not strictly increase; if *freq* is not a pandas
        offset alias, or is finer than the field's own steps, which would
        interpolate rather than aggregate; if its interval
        coordinates are not one-dimensional on ``time``, which is what stacking
        runs on different time axes leaves; if it declares a ``kind`` that is
        not one of pySIPNET's; if *how* is not one of
        :data:`RESAMPLING_METHODS`; if the variable's kind does not admit
        *how*, with pySIPNET's own explanation and the methods that would
        work; if *how* is omitted and the variable's kind cannot be
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
    _check_the_time_axis(field)
    _check_frequency(freq)
    _check_interval_coords_are_one_dimensional(field)
    field = _only_real_steps(field)
    _check_the_steps_are_aggregable(field)
    _check_not_upsampling(field, freq)
    kind = _variable_kind(field)
    method = _method_for(field, kind, how)
    weights = _step_weights(field) if method == "mean" else None

    bare = field.drop_vars(
        [str(name) for name in field.coords if _is_on_time(field, name) and name != TIME_DIM]
    )
    values = _combine(bare, freq, method, weights)
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
    ValueError
        If the field fails the checks of :func:`aggregate_time`; if *how* is
        not in :data:`WINDOW_REDUCTIONS`, or the variable's kind does not admit
        it; if *windows* is not a datetime ``IntervalIndex``, is empty, holds
        ``NaT``, overlaps, is not increasing, or is in a different time zone
        from the field; or if *labels* are not timestamps, not one per window,
        or not strictly increasing.
    """
    _check_the_time_axis(field)
    _check_interval_coords_are_one_dimensional(field)
    field = _only_real_steps(field)
    _check_the_steps_are_aggregable(field)
    kind = _variable_kind(field)
    how = _window_method_for(field, kind, how)
    stamps = _time_index(field)
    windows = _checked_windows(windows, stamps)
    labels, label_attrs = _checked_labels(labels, windows)

    membership = windows.get_indexer(stamps)
    weights = _step_weights(field) if how == "mean" else None
    bare = field.drop_vars(
        [str(name) for name in field.coords if _is_on_time(field, name) and name != TIME_DIM]
    )
    reduced = _reduce_by_window(bare, membership, how, len(windows), weights)
    reduced = reduced.assign_coords({TIME_DIM: labels})
    reduced[TIME_DIM].attrs = label_attrs
    reduced = reduced.assign_coords(_window_interval_coords(field, membership, len(windows)))
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
    ValueError
        If the field has no interval coordinates, a label lies outside the
        record, the variable is a per-step total or a running total (neither
        has a value *at* an instant; divide a total by the step length first),
        or *times* are not strictly increasing datetimes.
    """
    _check_the_time_axis(field)
    _check_interval_coords_are_one_dimensional(field)
    field = _only_real_steps(field)
    _check_the_steps_are_aggregable(field)
    _check_has_interval_coords(field, "select_timestep_at")
    kind = _variable_kind(field)
    _check_kind_has_an_instant_value(field, kind)
    labels, label_attrs = _checked_instants(times)

    ends = pd.DatetimeIndex(field[TIME_DIM].to_index())
    starts = pd.DatetimeIndex(field[START_COORD].to_index())
    _check_same_clock(labels, ends, field)
    # Compared in nanoseconds, so a label finer than the axis is not truncated
    # onto the axis's resolution before the containment test.
    ends, starts, labels = ends.as_unit("ns"), starts.as_unit("ns"), labels.as_unit("ns")
    position = np.searchsorted(ends.asi8, labels.asi8, side="left")
    outside = (position >= len(ends)) | (
        starts.asi8[np.minimum(position, len(ends) - 1)] >= labels.asi8
    )
    if outside.any():
        bad = labels[outside]
        raise ValueError(
            f"{len(bad)} label(s) fall in no timestep of {_field_label(field)}, the first being "
            f"{bad[0]}: the record covers ({starts[0]}, {ends[-1]}], and a label must "
            "lie inside one step's (time_step_start, time] interval. Select the "
            "observations within the run, or run the model over the observed period."
        )

    selected = field.isel({TIME_DIM: position})
    drop = [str(n) for n in selected.coords if _is_on_time(selected, n)]
    selected = selected.drop_vars(drop)
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

    Raises
    ------
    ValueError
        If the field carries no bounds. A dated or static product documents no
        interval, and an operator over it reads an instant
        (:func:`select_timestep_at`) or the run (:func:`run_window`) instead.
    """
    missing = [c for c in (TIME_BOUNDS_START, TIME_BOUNDS_END) if c not in observed_values.coords]
    if missing:
        raise ValueError(
            f"{observed_values.name!r} carries no {missing} coordinate, so it documents "
            "no interval to reduce the model over. Only an annual product has time "
            "bounds; for a dated or static one read an instant with "
            "select_timestep_at, or the whole run with run_window."
        )
    for name in (TIME_BOUNDS_START, TIME_BOUNDS_END):
        if not _is_datetime(observed_values[name].dtype):
            raise ValueError(
                f"{observed_values.name!r}: {name} must hold datetimes, got dtype "
                f"{observed_values[name].dtype}."
            )
    start = pd.DatetimeIndex(observed_values[TIME_BOUNDS_START].values)
    end = pd.DatetimeIndex(observed_values[TIME_BOUNDS_END].values)
    if start.hasnans or end.hasnans:
        raise ValueError(f"{observed_values.name!r}: a time bound is missing (NaT).")
    if not (end > start).all():
        raise ValueError(f"{observed_values.name!r}: every time_bounds_end must follow its start.")
    return pd.IntervalIndex.from_arrays(start, end, closed="right")


def run_window(field: xr.DataArray) -> pd.IntervalIndex:
    """One right-closed window spanning a model field's whole record.

    From the first step's :data:`START_COORD` to the last step's ``time``, so
    that every step belongs to it. For a static observation, which documents no
    time at all, this is the window an operator reduces over.
    """
    _check_the_time_axis(field)
    _check_has_interval_coords(field, "run_window")
    field = _only_real_steps(field)
    start = pd.DatetimeIndex(field[START_COORD].to_index())
    end = pd.DatetimeIndex(field[TIME_DIM].to_index())
    return pd.IntervalIndex.from_arrays([start.min()], [end.max()], closed="right")


def aggregation_counts(field: xr.DataArray, freq: str) -> xr.DataArray:
    """How many values each cell of :func:`aggregate_time` is formed from.

    The count of values that are not missing, per cell, on the same cells
    :func:`aggregate_time` forms (right-closed, empty ones dropped), so it
    aligns with the aggregate by position and can mask it. Zero where a cell
    held only missing values. No attributes: it is a count of a field, not a
    variable. The cells are labeled as ``resample`` labels them, which differs
    from the relabeling :func:`aggregate_time` gives a field carrying
    pySIPNET's interval coordinates, so compare by position.
    """
    _check_the_time_axis(field)
    _check_frequency(freq)
    field = _only_real_steps(field)
    _check_the_steps_are_aggregable(field)
    _check_not_upsampling(field, freq)
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

    Aligned with what :func:`reduce_windows` returns for the same *field*,
    *windows* and *labels*; zero where a window held nothing; no attributes.
    """
    _check_the_time_axis(field)
    field = _only_real_steps(field)
    _check_the_steps_are_aggregable(field)
    stamps = _time_index(field)
    windows = _checked_windows(windows, stamps)
    labels, label_attrs = _checked_labels(labels, windows)
    counts = _count_by_window(field, windows.get_indexer(stamps), len(windows))
    counts = counts.assign_coords({TIME_DIM: labels})
    counts[TIME_DIM].attrs = label_attrs
    return counts


# ── supporting helpers ────────────────────────────────────────────────────────


def _check_interval_coords_are_one_dimensional(field: xr.DataArray) -> None:
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
            f"{_field_label(field)} has {offenders} on dims "
            f"{[tuple(str(d) for d in field[n].dims) for n in offenders]} rather "
            f"than on {TIME_DIM!r} alone, which happens when runs on different "
            "time axes are stacked together. Select one site, or drop those "
            "coordinates to aggregate on calendar cells with equal weights."
        )


def _check_the_steps_are_aggregable(field: xr.DataArray) -> None:
    """There is at least one step, and no two of them share or reverse a label.

    Duplicate labels would be summed together as though they were consecutive
    steps, which is how one record counted twice comes back looking like a
    larger flux.
    """
    times = field[TIME_DIM].values
    if times.size == 0:
        raise ValueError(
            f"{_field_label(field)} has no timesteps left to aggregate. An empty "
            f"{TIME_DIM!r} comes from a selection that matched nothing, or from "
            "a site of a stacked ensemble with no record of its own."
        )
    steps = np.diff(times.astype("datetime64[ns]").astype("int64"))
    if (steps <= 0).any():
        where = int(np.flatnonzero(steps <= 0)[0]) + 1
        raise ValueError(
            f"{_field_label(field)} has timestamps that do not increase: row {where} "
            f"({times[where]}) does not follow row {where - 1} "
            f"({times[where - 1]}). Sort the field on {TIME_DIM!r}, and drop or "
            "combine the duplicates; two rows sharing a label would be added "
            "together as though they were consecutive steps."
        )


def _field_label(field: xr.DataArray) -> str:
    """How a field is called in a message: its name, or else its derivation.

    A result of :mod:`pysipnet.arithmetic` is unnamed and records what it was
    computed from in its ``derivation`` attribute.
    """
    if field.name is not None:
        return repr(field.name)
    derivation = field.attrs.get("derivation")
    return repr(derivation) if derivation else "the field"

def _variable_kind(field: xr.DataArray) -> VariableKind | None:
    """The field's pySIPNET kind, from its attributes or the registries, or ``None``."""
    declared = field.attrs.get("kind")
    if declared is not None:
        try:
            return VariableKind(declared)
        except ValueError as error:
            raise ValueError(
                f"{_field_label(field)} declares kind={declared!r}, which is not one of "
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
                f"{_field_label(field)} carries no 'kind' attribute and is not a SIPNET "
                "output or climate variable, so there is no rule to take the "
                "aggregation from. Pass how='sum', 'mean' or 'last'."
            )
        if kind not in DEFAULT_METHOD_FOR_KIND:
            raise ValueError(
                f"{_field_label(field)} is of kind {kind.value!r}, which no method leaves "
                "unchanged, so there is no default. Pass how= explicitly."
            )
        return DEFAULT_METHOD_FOR_KIND[kind]

    if how not in RESAMPLING_METHODS:
        raise ValueError(
            f"Unknown resampling method {how!r} for {_field_label(field)}; choose from "
            f"{list(RESAMPLING_METHODS)}."
        )
    if kind is not None and how not in RESAMPLING_METHODS_FOR_KIND[kind]:
        _refuse(field.name, kind, how)
    return how


def _refuse(name: Any, kind: VariableKind, method: str) -> None:
    """Raise pySIPNET's own explanation of why *method* is meaningless for *kind*.

    Obtained by putting the pair to ``pysipnet.resample.resample``, on two rows
    that exist only to be refused, rather than by restating a reason that would
    then be this project's to keep in step with pySIPNET's.
    """
    label = str(name) if name is not None else "the field"
    # The probe cannot hold a variable named after one of the time coordinates
    # it must carry, so a field with such a name is put to pySIPNET under a
    # stand-in and named properly again in the message.
    stand_in = label if label not in _PROBE_COORD_NAMES else "the_field"
    try:
        pysipnet_resample(_refusal_probe(stand_in, kind), "1D", how=method)
    except ValueError as refusal:
        raise ValueError(str(refusal).replace(repr(stand_in), repr(label), 1)) from None
    raise ValueError(
        f"Cannot aggregate {label!r} with {method!r}: it is of kind "
        f"{kind.value!r}, which admits only "
        f"{sorted(RESAMPLING_METHODS_FOR_KIND[kind])}."
    )


#: Names the refusal probe's own coordinates occupy.
_PROBE_COORD_NAMES = frozenset({TIME_DIM, START_COORD, LENGTH_COORD, "time_bounds"})


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


def _is_on_time(field: xr.DataArray, name: Any) -> bool:
    """Whether the coordinate *name* varies along ``time``."""
    return TIME_DIM in field[name].dims


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


def _step_days(field: xr.DataArray) -> np.ndarray:
    """Declared step lengths in days. Never ``NaT``: those are not steps."""
    nanoseconds = field[LENGTH_COORD].values.astype("timedelta64[ns]").astype("int64")
    return nanoseconds / 86_400e9


def _step_weights(field: xr.DataArray) -> xr.DataArray:
    """Step lengths in days, for a length-weighted mean."""
    if LENGTH_COORD in field.coords:
        return _on_time(field, _step_days(field))

    spacing = np.diff(field[TIME_DIM].values.astype("datetime64[ns]").astype("int64"))
    if spacing.size and (spacing != spacing[0]).any():
        raise ValueError(
            f"{_field_label(field)} has no {LENGTH_COORD!r} coordinate and its steps are "
            "not all the same length, so a mean over them has no defined "
            "weighting. Attach the step lengths, or aggregate a field that "
            "carries them."
        )
    return xr.DataArray(
        np.ones(field.sizes[TIME_DIM]), dims=TIME_DIM, coords={TIME_DIM: field[TIME_DIM]}
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

    start = _grouped(_on_time(field, field[START_COORD].values), freq).min()
    end = _grouped(_on_time(field, field[TIME_DIM].values), freq).max()
    length = _grouped(_on_time(field, _step_days(field)), freq).sum()

    built = assemble_time_coords(
        start=start.values[keep].astype("datetime64[ns]"),
        end=end.values[keep].astype("datetime64[ns]"),
        length=np.rint(length.values[keep] * 86_400e9)
        .astype("int64")
        .view("timedelta64[ns]"),
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


def _on_time(field: xr.DataArray, values: np.ndarray) -> xr.DataArray:
    """*values*, one per timestep, as a resamplable field on the field's ``time``."""
    return xr.DataArray(values, dims=TIME_DIM, coords={TIME_DIM: field[TIME_DIM]})


def _aggregated_attrs(
    attrs: Mapping[str, Any],
    kind: VariableKind | None,
    method: str,
    freq: str,
    weighted: bool,
) -> dict[str, Any]:
    """The variable's attributes, rewritten to describe what the values now are."""
    out = dict(attrs)
    if kind is not None:
        new_kind = RESAMPLED_KIND[(kind, method)]
        out["kind"] = new_kind.value
        out["time_reference"] = TIME_REFERENCE_FOR_KIND[new_kind]
        cell_methods = CELL_METHODS_FOR_KIND[new_kind]
        if cell_methods is None:
            out.pop("cell_methods", None)
        else:
            out["cell_methods"] = cell_methods
    weighting = f", weighted by {LENGTH_COORD}" if weighted else ""
    of_kind = f" of {kind.value} values" if kind is not None else ""
    out["resampling"] = f"{method}{of_kind} over {freq}{weighting}"
    # The source file's printf precision no longer describes a combined value.
    out.pop("output_decimals", None)
    return out


def _check_the_time_axis(field: Any) -> None:
    """*field* is a DataArray with a datetime ``time`` coordinate and no NaT."""
    if not isinstance(field, xr.DataArray):
        advice = (
            " A Dataset holds several variables, whose kinds differ; align one at a time."
            if isinstance(field, xr.Dataset)
            else ""
        )
        raise ValueError(f"expected an xarray.DataArray, got {type(field).__name__}.{advice}")
    if TIME_DIM not in field.dims:
        raise ValueError(
            f"aligning in time needs a {TIME_DIM!r} dimension; {_field_label(field)} has "
            f"dims {tuple(str(d) for d in field.dims)}."
        )
    if TIME_DIM not in field.coords:
        raise ValueError(
            f"{_field_label(field)} has a {TIME_DIM!r} dimension but no {TIME_DIM!r} "
            "coordinate, so there is nothing to place its rows by."
        )
    dtype = field.coords[TIME_DIM].dtype
    if not _is_datetime(dtype):
        raise ValueError(
            f"the {TIME_DIM!r} coordinate of {_field_label(field)} has dtype {dtype}, and "
            "alignment needs datetimes."
        )
    if pd.isna(field.coords[TIME_DIM].values).any():
        raise ValueError(
            f"the {TIME_DIM!r} coordinate of {_field_label(field)} holds a missing timestamp "
            "(NaT), so its rows cannot be placed. Drop those rows first."
        )


def _check_frequency(freq: Any) -> None:
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


def _check_not_upsampling(field: xr.DataArray, freq: str) -> None:
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


def _check_has_interval_coords(field: xr.DataArray, what: str) -> None:
    """*field* carries pySIPNET's step start and length."""
    missing = [c for c in (START_COORD, LENGTH_COORD) if c not in field.coords]
    if missing:
        raise ValueError(
            f"{what} reads the interval each step covers, and {_field_label(field)} carries no "
            f"{missing} coordinate. Model output from pySIPNET carries both; an "
            "observation field does not, and is not what this reads."
        )


def _check_kind_has_an_instant_value(field: xr.DataArray, kind: VariableKind | None) -> None:
    """A total or a running total has no value at an instant."""
    if kind in (VariableKind.TIMESTEP_TOTAL, VariableKind.CUMULATIVE):
        fix = (
            "divide it by pysipnet.arithmetic.step_length() first, with "
            "pysipnet.arithmetic.divide_with_units, which makes it a rate that does"
            if kind is VariableKind.TIMESTEP_TOTAL
            else "take its last value over a window with reduce_windows instead"
        )
        raise ValueError(
            f"{_field_label(field)} is of kind {kind.value!r}, which has no value at an "
            f"instant; {fix}."
        )
    if kind is VariableKind.TIMESTEP_START_COORDINATE:
        raise ValueError(f"{_field_label(field)} is a time coordinate, not a variable to read.")


def _check_same_clock(
    labels: pd.DatetimeIndex, ends: pd.DatetimeIndex, field: xr.DataArray
) -> None:
    if labels.tz != ends.tz:
        raise ValueError(
            f"the labels are in time zone {labels.tz} and {_field_label(field)}'s time "
            f"coordinate in {ends.tz}; localize or convert one of them first."
        )


def _window_method_for(field: xr.DataArray, kind: VariableKind | None, how: Any) -> str:
    """*how* checked against :data:`WINDOW_REDUCTIONS` and the variable's kind."""
    if not isinstance(how, str) or how not in WINDOW_REDUCTIONS:
        raise ValueError(
            f"how must be one of {list(WINDOW_REDUCTIONS)} for {_field_label(field)}, got {how!r}."
        )
    if kind is None:
        return how
    if how in RESAMPLING_METHODS:
        if how not in RESAMPLING_METHODS_FOR_KIND[kind]:
            _refuse(field.name, kind, how)
        return how
    if kind not in _LEVEL_KINDS:
        raise ValueError(
            f"Cannot take the {how!r} of {_field_label(field)} over a window: it is of kind "
            f"{kind.value!r}, and only a level (a pool, a step mean or a rate) has an "
            f"extreme or a first reading. A total sums; a running total takes 'last'."
        )
    return how


def _reduce_by_window(
    bare: xr.DataArray,
    membership: np.ndarray,
    how: str,
    n_windows: int,
    weights: xr.DataArray | None,
) -> xr.DataArray:
    """*bare* reduced per window; ``NaN`` where a window has no step or a gap."""
    if not (membership >= 0).any():
        return _empty_windows(bare, n_windows, np.nan, float)
    rows = _members(bare, membership)
    if how == "mean":
        assert weights is not None
        weighted = _members(bare * weights, membership).groupby(_WINDOW).sum(skipna=False)
        total = _members(weights, membership).groupby(_WINDOW).sum()
        reduced = weighted / total
    elif how == "sum":
        reduced = rows.groupby(_WINDOW).sum(skipna=False)
    else:
        reduced = getattr(rows.groupby(_WINDOW), how)()
    # Every method: a window holding a gap is a gap. sum and the weighted mean
    # already propagate it; first, last, min and max skip missing values.
    complete = rows.notnull().groupby(_WINDOW).all()
    reduced = reduced.where(complete)
    return _by_window(reduced, n_windows, bare.dims)


def _count_by_window(
    field: xr.DataArray, membership: np.ndarray, n_windows: int
) -> xr.DataArray:
    """Values that are not missing, per window, as ``int64``."""
    bare = field.drop_vars(
        [str(name) for name in field.coords if _is_on_time(field, name) and name != TIME_DIM]
    )
    if not (membership >= 0).any():
        return _empty_windows(bare, n_windows, 0, np.int64)
    counts = _members(bare, membership).notnull().groupby(_WINDOW).sum()
    counts = _by_window(counts, n_windows, bare.dims).fillna(0).astype(np.int64)
    counts.name = None
    counts.attrs = {}
    return counts


def _members(field: xr.DataArray, membership: np.ndarray) -> xr.DataArray:
    """The rows inside some window, with the window index as a coordinate."""
    inside = np.flatnonzero(membership >= 0)
    rows = field.isel({TIME_DIM: inside})
    return rows.assign_coords({_WINDOW: (TIME_DIM, membership[inside])})


def _empty_windows(field: xr.DataArray, n_windows: int, fill: Any, dtype: Any) -> xr.DataArray:
    """One entry per window when no row falls in any of them."""
    shape = [n_windows if dim == TIME_DIM else field.sizes[dim] for dim in field.dims]
    coords = {name: coord for name, coord in field.coords.items() if TIME_DIM not in coord.dims}
    return xr.DataArray(np.full(shape, fill, dtype=dtype), dims=field.dims, coords=coords)


def _by_window(reduced_groups: xr.DataArray, n_windows: int, dims: Any) -> xr.DataArray:
    """One entry per window, in the field's dimension order, without labels."""
    full = reduced_groups.reindex({_WINDOW: np.arange(n_windows)})
    full = full.rename({_WINDOW: TIME_DIM}).drop_vars(TIME_DIM, errors="ignore")
    return full.transpose(*dims)


def _window_interval_coords(
    field: xr.DataArray, membership: np.ndarray, n_windows: int
) -> dict[str, xr.DataArray]:
    """The span of steps each window combined, where the field declares its steps."""
    if START_COORD not in field.coords or LENGTH_COORD not in field.coords:
        return {}
    inside = membership >= 0
    if not inside.any():
        nat = np.full(n_windows, np.datetime64("NaT", "ns"))
        return {
            START_COORD: xr.DataArray(nat, dims=TIME_DIM, attrs=dict(field[START_COORD].attrs)),
            LENGTH_COORD: xr.DataArray(
                np.full(n_windows, np.timedelta64("NaT", "ns")),
                dims=TIME_DIM,
                attrs=dict(field[LENGTH_COORD].attrs),
            ),
            SELECTED_STEP_COORD: xr.DataArray(nat, dims=TIME_DIM),
        }
    window = pd.Index(membership[inside], name=_WINDOW)
    start = pd.Series(field[START_COORD].values[inside], index=window).groupby(level=0).min()
    end = pd.Series(field[TIME_DIM].values[inside], index=window).groupby(level=0).max()
    length = pd.Series(_step_days(field)[inside], index=window).groupby(level=0).sum()
    every = np.arange(n_windows)
    return {
        START_COORD: xr.DataArray(
            start.reindex(every).values.astype("datetime64[ns]"),
            dims=TIME_DIM,
            attrs=_without_stale_interval_attrs(field[START_COORD].attrs),
        ),
        LENGTH_COORD: xr.DataArray(
            _days_to_timedelta(length.reindex(every).values),
            dims=TIME_DIM,
            attrs={
                **_without_stale_interval_attrs(field[LENGTH_COORD].attrs),
                "source": STEP_LENGTH_RESAMPLED,
            },
        ),
        SELECTED_STEP_COORD: xr.DataArray(
            end.reindex(every).values.astype("datetime64[ns]"),
            dims=TIME_DIM,
            attrs={"long_name": "End of the last model timestep combined into the window"},
        ),
    }


def _days_to_timedelta(days: np.ndarray) -> np.ndarray:
    """Fractional days to ``timedelta64[ns]``, ``NaT`` where missing."""
    out = np.full(days.shape, np.timedelta64("NaT", "ns"))
    finite = np.isfinite(days)
    out[finite] = np.rint(days[finite] * 86_400e9).astype("int64").astype("timedelta64[ns]")
    return out


def _window_attrs(
    attrs: Mapping[str, Any], kind: VariableKind | None, how: str, *, weighted: bool
) -> dict[str, Any]:
    """The variable's attributes, rewritten for a value formed over a window."""
    out = dict(attrs)
    if kind is not None and how in RESAMPLING_METHODS:
        new_kind = RESAMPLED_KIND[(kind, how)]
        out["kind"] = new_kind.value
        out["time_reference"] = TIME_REFERENCE_FOR_KIND[new_kind]
        cell_methods = CELL_METHODS_FOR_KIND[new_kind]
        if cell_methods is None:
            out.pop("cell_methods", None)
        else:
            out["cell_methods"] = cell_methods
    elif kind is not None:
        out["time_reference"] = f"the {how} of the {kind.value} values over the window"
        method = {"min": "minimum", "max": "maximum"}.get(how, "point")
        out["cell_methods"] = f"time: {method}"
    weighting = f", weighted by {LENGTH_COORD}" if weighted else ""
    out["reduction"] = f"{how} over the window the label names{weighting}"
    out.pop("output_decimals", None)
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


def _is_datetime(dtype: Any) -> bool:
    """Whether *dtype* is a datetime one, time-zone-aware ones included."""
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return isinstance(dtype, pd.DatetimeTZDtype)
    return bool(np.issubdtype(dtype, np.datetime64))


def _time_index(field: xr.DataArray) -> pd.DatetimeIndex:
    """The field's ``time`` coordinate as a pandas index."""
    return pd.DatetimeIndex(field.coords[TIME_DIM].to_index())


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
