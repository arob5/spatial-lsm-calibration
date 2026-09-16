"""Time indexing and temporal aggregation for the likelihood and the plots.

Overview
--------
SIPNET labels its rows with ``year``, ``day`` and ``time`` columns rather
than timestamps, and the model runs at a finer step than most observations
are made at. This module builds the timestamps and reduces a field in time,
either into regular periods (days, months, years) or into arbitrary windows
such as an observation's own intervals. The observation operator applies
these before the likelihood sees a residual, and the plotting layer applies
the same functions before a predictive check is drawn, so the two cannot
disagree.

Functions
---------
:func:`sipnet_time_index`
    Timestamps for rows labeled the way SIPNET labels them. The ``time``
    column drifts (issue #9) and is used only to identify a row's slot within
    its day, never as the timestamp itself.

:func:`aggregate_time`
    Reduce a field along ``time`` into regular periods, by the rule the
    field's own ``aggregation`` attribute names or by an explicit ``how``.

:func:`aggregation_counts`
    How many values each period of such an aggregation was formed from, for
    a caller imposing its own completeness rule.

:func:`reduce_windows`
    Reduce a field into arbitrary, possibly irregular, windows given as a
    ``pandas.IntervalIndex``, labeling the result as the caller asks.

:func:`window_counts`
    The counts for :func:`reduce_windows`.

Aggregation is a verb the caller applies, never a plotter keyword::

    plot_time_series(aggregate_time(nee, "1D"))     # yes
    plot_time_series(nee, temporal_agg="1D")        # no

Notes
-----
**Where the rule comes from.** :func:`aggregate_time` has two sources for its
rule and no default: the ``how`` argument, and the field's ``aggregation``
attribute. pySIPNET writes that attribute onto every model output field from
the variable's kind (a ``flux`` is a total over the step and sums; a ``state``
or a step ``mean`` averages), and :mod:`sipnet_calibration.drivers` writes it
onto the driver fields it reads.
A field carrying neither is refused, because a wrong default is silent:
SIPNET's ``net_ecosystem_exchange`` is ``g m-2`` of carbon per timestep, so
3-hourly to daily is a sum, and a mean is wrong by a factor of eight while
looking plausible. Observed fields carry no ``aggregation`` attribute: how an
observation is placed in time is decided by its observation operator.

**Units are not converted here.** Turning a per-step total into a rate, or
a model unit into an observation's, is the observation operator's job.

**What a period's label means.** Periods are left-closed, and pandas decides
where the label goes: a start-anchored frequency such as ``"1D"``, ``"MS"``
or ``"YS"`` labels the period's start, an end-anchored one such as ``"ME"``
or ``"YE"`` labels its end. The drivers and SIPNET's output label the **end**
of each timestep (:data:`sipnet_calibration.drivers.TIME_LABEL`), so a day's
eight rows labeled 00:00 to 21:00 group under that day, which is the grouping
SIPNET's own ``day`` column gives. :func:`reduce_windows` likewise assigns a
row to a window by the row's label; an alignment that wants midpoint
membership shifts the labels by half a step before calling it.

Usage
-----
::

    from sipnet_calibration.observation_operators import (
        aggregate_time,
        aggregation_counts,
        reduce_windows,
        sipnet_time_index,
    )

    daily_par = aggregate_time(par, "1D")     # par carries aggregation="sum"
    daily_tair = aggregate_time(tair, "1D")   # tair carries aggregation="mean"
    daily_obs = aggregate_time(nee_obs, "1D", how="mean")   # an observation says

    # Only whole days.
    counts = aggregation_counts(par, "1D")
    whole_days = aggregate_time(par, "1D").where(counts == 8)

    # Irregular windows, labeled with the observation's own keys.
    windows = pd.IntervalIndex.from_arrays(starts, ends, closed="right")
    at_obs = reduce_windows(soil_carbon, windows, "mean", labels=obs["time"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.time_conventions import TIME_LABEL_ATTR, TimeLabel

__all__ = [
    "AGGREGATION_ATTR",
    "REDUCTIONS",
    "TIME_DIM",
    "aggregate_time",
    "aggregation_counts",
    "reduce_windows",
    "sipnet_time_index",
    "window_counts",
]

#: The dimension aggregated along.
TIME_DIM = "time"

#: The attribute a field carries to name its own aggregation rule. pySIPNET's
#: ``VariableSpec.xarray_attributes()`` writes it from the variable's kind, and
#: :mod:`sipnet_calibration.drivers` writes it onto the driver fields.
AGGREGATION_ATTR = "aggregation"

#: What ``how`` may be, and what the ``aggregation`` attribute may name.
#: ``"sum"`` for a quantity accumulated over its step, ``"mean"`` for one
#: describing its step, ``"first"``/``"last"`` for a running total or a level
#: at a period's edge, ``"min"``/``"max"`` for a period's extreme.
REDUCTIONS: tuple[str, ...] = ("sum", "mean", "min", "max", "first", "last")

#: The one value pySIPNET writes that names no reduction: its ``COORDINATE``
#: kind, for the ``year``/``day_of_year``/``hour_of_day`` columns.
_NO_REDUCTION = "none"

#: The coordinate :func:`reduce_windows` groups by, internal to one call.
_WINDOW = "_window"


def sipnet_time_index(
    year, day_of_year, hours_since_midnight, *, timestep_hours: float = 3.0
) -> pd.DatetimeIndex:
    """Timestamps for rows labeled the way SIPNET labels them.

    SIPNET's climate files and its output give each row a year, an integer day
    of year with 1 being January 1, and a fractional hour of the day -- the
    columns SIPNET calls ``year``, ``day`` and ``time``. This builds the
    nominal timestamp of each row as::

        year-01-01  +  (day_of_year - 1) days  +  slot * timestep_hours

    where ``slot = floor(hours_since_midnight / timestep_hours)`` is the row's
    position within its day. The hour value itself is used for nothing else.

    Parameters
    ----------
    year, day_of_year, hours_since_midnight:
        Array-likes of equal length: SIPNET's ``year``, ``day`` and ``time``
        columns under clearer names. ``year`` and ``day_of_year`` are integers
        (or floats that are whole numbers); ``hours_since_midnight`` is hours
        since midnight of that day.
    timestep_hours:
        Length of one row's timestep in hours; must be finite and divide 24
        into a whole number of steps. The default is the 3-hourly drivers. A
        daily file passes ``24.0``.

    Returns
    -------
    pandas.DatetimeIndex
        Naive timestamps, ``datetime64[ns]``, one per row, in row order. The
        index says nothing about the clock the labels are on or whether a
        label marks an interval's start or end; the caller records that, as
        :mod:`sipnet_calibration.drivers` does in the ``time`` attributes.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional or differ in length;
        ``timestep_hours`` is not a finite number dividing 24; a ``year`` or
        ``day_of_year`` is not numeric or not a whole number; a
        ``day_of_year`` is outside ``1..366``, or is 366 in a non-leap year;
        an ``hours_since_midnight`` is not finite or is outside ``[0, 24)``;
        or the resulting index is not strictly increasing.

    Notes
    -----
    ``floor`` rather than ``round`` because the ``.clim`` ``time`` column
    drifts late by up to two hours within a year (issue #9): the row for
    nominal hour 21 is labeled ``23.00`` on 31 December, and rounding would
    put it in a ninth slot. The drift is always non-negative and always below
    one step, so ``floor`` identifies the slot in every row. SIPNET copies the
    same column verbatim into its output, so this applies to output as well
    as to drivers.

    The slot division is exact for the timesteps SIPNET is run at (3, 1, 0.5
    and 24 hours). A step such as ``0.1`` divides 24 but is not exactly
    representable, so a label sitting exactly on a slot boundary can floor
    into the slot below.
    """
    year = np.asarray(year)
    day = np.asarray(day_of_year)
    hours = np.asarray(hours_since_midnight, dtype=np.float64)
    if not (year.shape == day.shape == hours.shape) or year.ndim != 1:
        raise ValueError(
            "year, day_of_year and hours_since_midnight must be one-dimensional "
            f"and the same length; got shapes {year.shape}, {day.shape}, {hours.shape}"
        )

    try:
        timestep_hours = float(timestep_hours)
    except (TypeError, ValueError) as error:
        raise ValueError(f"timestep_hours must be a number, got {timestep_hours!r}") from error
    if not (np.isfinite(timestep_hours) and timestep_hours > 0):
        raise ValueError(f"timestep_hours must divide 24, got {timestep_hours!r}")
    steps_per_day = 24.0 / timestep_hours
    if abs(steps_per_day - round(steps_per_day)) > 1e-9:
        raise ValueError(f"timestep_hours must divide 24, got {timestep_hours!r}")
    steps_per_day = int(round(steps_per_day))

    year = _whole_numbers(year, name="year")
    day = _whole_numbers(day, name="day_of_year")

    if year.size == 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")

    if np.any(day < 1) or np.any(day > 366):
        bad = day[(day < 1) | (day > 366)]
        raise ValueError(f"day_of_year must be within 1..366, found {bad[:5].tolist()}")
    is_leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    if np.any((day == 366) & ~is_leap):
        bad_years = np.unique(year[(day == 366) & ~is_leap])
        raise ValueError(f"day_of_year 366 in non-leap year(s) {bad_years[:5].tolist()}")

    if np.any(~np.isfinite(hours)) or np.any(hours < 0) or np.any(hours >= 24):
        bad = hours[~((hours >= 0) & (hours < 24))]
        raise ValueError(
            f"hours_since_midnight must lie within [0, 24), found {bad[:5].tolist()}"
        )

    slot = np.floor(hours / timestep_hours).astype(np.int64)
    # An hour just below 24 with a step that divides 24 always floors below
    # steps_per_day; the guard is against floating-point noise at the edge.
    slot = np.minimum(slot, steps_per_day - 1)

    year_start = pd.to_datetime(pd.Series(year), format="%Y").to_numpy()
    offset = (day - 1).astype("timedelta64[D]") + (
        (slot * timestep_hours * 3600.0).round().astype(np.int64).astype("timedelta64[s]")
    )
    index = pd.DatetimeIndex(year_start + offset).as_unit("ns")

    if not index.is_monotonic_increasing or index.has_duplicates:
        where = int(np.flatnonzero(np.diff(index.asi8) <= 0)[0]) + 1
        raise ValueError(
            "the timestamps are not strictly increasing: row "
            f"{where} ({index[where]}) does not follow row {where - 1} "
            f"({index[where - 1]}). Rows out of order, or a time label that "
            "drifted into the next slot."
        )
    return index


def aggregate_time(
    field: xr.DataArray,
    freq: str,
    *,
    how: str | None = None,
    min_count: int = 1,
) -> xr.DataArray:
    """Aggregate *field* along ``time`` into regular periods.

    The rule comes from *how*, or from the field's own ``aggregation``
    attribute when *how* is not given. There is no other default.

    Parameters
    ----------
    field:
        A canonical field with a ``time`` dimension carrying a datetime
        coordinate. Any other dimensions -- ``member``, ``site`` -- are
        untouched, as are the coordinates on them, so ``lon``/``lat`` survive.
    freq:
        The target period, as a pandas offset alias: ``"1D"``, ``"MS"``,
        ``"YS"``. Must be at least the field's own spacing; see Raises.
    how:
        One of :data:`REDUCTIONS`. ``None`` takes the rule from
        ``field.attrs["aggregation"]``, which pySIPNET writes on every model
        output field from the variable's kind and ``load_drivers`` on every
        driver field; a field without that attribute, an observation say, has
        to be told.
    min_count:
        The fewest values a period may be formed from and still produce a
        value. A period with fewer than *min_count* values that are not
        missing comes back as ``NaN``. At least 1, which is the default: no
        observations gives missing, never zero.

    Returns
    -------
    xarray.DataArray
        The aggregated field, with the same dimensions and the same name.
        ``attrs`` are carried through, with ``aggregation_applied`` (the
        method used) and ``aggregation_freq`` (*freq*) added, so that two
        aggregations of one variable are distinguishable: a per-timestep total
        carries the same ``units`` at every resolution. The ``time``
        coordinate's ``time_label`` says whether the labels mark the start of
        each period (start-anchored frequencies such as ``"1D"``) or its end
        (``"ME"``, ``"YE"``); the source's clock attributes are kept.

    Raises
    ------
    ValueError
        If *field* is not a ``DataArray``, has no ``time`` dimension, has no
        ``time`` coordinate, or that coordinate is not a datetime one and so
        cannot be resampled; if the ``time`` coordinate is not increasing; if
        *how* is given and is not in :data:`REDUCTIONS`; if *how* is not
        given and the field carries no ``aggregation`` attribute, or one that
        names no reduction; if *min_count* is not an integer of at least 1;
        or if *freq* names periods shorter than the field's own spacing, which
        would interpolate rather than aggregate.

    Notes
    -----
    A period with no values that are not missing comes back ``NaN``, never
    zero, under every method; ``.resample(...).sum()`` on its own would
    return zero for an all-missing day, which reads as zero flux.

    A partial period is returned as the partial total or mean it is, never
    scaled up: scaling assumes the absent timesteps resemble the present
    ones, which for a diurnal flux is false. A caller wanting only whole
    periods says so with ``min_count=`` or by masking on
    :func:`aggregation_counts`. The project's records are contiguous within
    their covered spans, so a partial period arises only at a record edge,
    which is why ``min_count=1`` is the default.
    """
    _check_frequency(freq)
    _check_min_count(min_count)
    _check_aggregatable(field)
    method = _resolved_method(field, how)

    _check_not_upsampling(field, freq)

    counts = _count_by_period(field, freq)
    aggregated = _reduce(field.resample({TIME_DIM: freq}), method)
    # One mask for every method; xarray aligns `counts` by dimension name.
    aggregated = aggregated.where(counts >= int(min_count))

    # Set explicitly: the reductions' keep_attrs defaults vary across xarray
    # versions, and the copy keeps the caller's field free of this provenance.
    aggregated.name = field.name
    aggregated.attrs = dict(field.attrs)
    aggregated.attrs["aggregation_applied"] = method
    aggregated.attrs["aggregation_freq"] = freq
    aggregated[TIME_DIM].attrs = _period_time_attrs(field, freq)
    return aggregated


def aggregation_counts(field: xr.DataArray, freq: str) -> xr.DataArray:
    """How many values each period of :func:`aggregate_time` is formed from.

    The count of values that are not missing, per period, which is what a
    completeness rule stricter than :func:`aggregate_time`'s default needs.

    Parameters
    ----------
    field:
        As :func:`aggregate_time`. Its attributes are not consulted, so a
        field with no ``aggregation`` attribute can be counted.
    freq:
        As :func:`aggregate_time`.

    Returns
    -------
    xarray.DataArray
        Integer counts, with the same dimensions as the aggregate and the same
        ``time`` axis, so it aligns with what :func:`aggregate_time` returns
        and can mask it directly. Zero where a period held nothing. It carries
        no attributes: it is a count of the field, not a field.

    Raises
    ------
    ValueError
        As :func:`aggregate_time`, for the ``time`` axis, *freq* and the
        upsampling check.
    """
    _check_frequency(freq)
    _check_aggregatable(field)
    _check_not_upsampling(field, freq)
    return _count_by_period(field, freq)


def reduce_windows(
    field: xr.DataArray,
    windows: pd.IntervalIndex,
    how: str,
    *,
    labels=None,
    min_count: int = 1,
) -> xr.DataArray:
    """Reduce *field* along ``time`` into the given windows.

    The general form of :func:`aggregate_time`: the periods are any set of
    non-overlapping intervals rather than a regular frequency, and the result
    is labeled as the caller asks. A row belongs to the window that contains
    its ``time`` label; rows outside every window are ignored.

    Parameters
    ----------
    field:
        As :func:`aggregate_time`.
    windows:
        A ``pandas.IntervalIndex`` of datetimes, non-overlapping, in
        increasing order, naive or in the same time zone as the field. Its
        ``closed`` side decides which of two adjacent windows a label on their
        shared edge belongs to; ``"right"`` matches end-labeled rows,
        ``"left"`` start-labeled ones.
    how:
        One of :data:`REDUCTIONS`. Required: a window has no frequency to
        infer anything from, and the field's ``aggregation`` attribute is
        about its own steps, not about what an observation wants.
    labels:
        The ``time`` coordinate of the result, one label per window, strictly
        increasing. Defaults to each window's right edge, labeled
        ``time_label = "interval_end"``. An observation operator passes the
        observation's own ``time`` coordinate, whose attributes are kept.
    min_count:
        As :func:`aggregate_time`: a window formed from fewer values that are
        not missing comes back as ``NaN``.

    Returns
    -------
    xarray.DataArray
        One value per window, with *field*'s other dimensions and their
        coordinates untouched and its ``time`` coordinate replaced by
        *labels*. ``attrs`` are carried through, with ``aggregation_applied``
        added. A window holding no rows at all is ``NaN``.

    Raises
    ------
    ValueError
        If *field* fails the checks of :func:`aggregate_time`; if *how* is not
        in :data:`REDUCTIONS`; if *windows* is not a datetime
        ``IntervalIndex``, is empty, holds ``NaT``, overlaps, is not
        increasing, or is in a different time zone from the field; if
        *labels* are not timestamps, not the length of *windows* or not
        strictly increasing; or if *min_count* is not an integer of at
        least 1.

    Notes
    -----
    Membership is by the row's label, not by the interval the row covers, so
    a right-closed window ``(a, b]`` collects the end-labeled steps that end
    in it. For midpoint membership, shift the labels by half a step first.
    """
    how = _check_reduction(how)
    _check_min_count(min_count)
    _check_aggregatable(field)
    stamps = _time_index(field)
    windows = _checked_windows(windows, stamps)
    labels, label_attrs = _checked_labels(labels, windows)

    membership = windows.get_indexer(stamps)
    counts = _count_by_window(field, membership, len(windows))
    reduced = _reduce_by_window(field, membership, how, len(windows))
    reduced = reduced.where(counts >= int(min_count))

    reduced = reduced.assign_coords({TIME_DIM: labels})
    reduced[TIME_DIM].attrs = label_attrs
    reduced.name = field.name
    reduced.attrs = dict(field.attrs)
    reduced.attrs["aggregation_applied"] = how
    return reduced


def window_counts(field: xr.DataArray, windows: pd.IntervalIndex, *, labels=None) -> xr.DataArray:
    """How many values each window of :func:`reduce_windows` is formed from.

    Parameters
    ----------
    field, windows, labels:
        As :func:`reduce_windows`.

    Returns
    -------
    xarray.DataArray
        Integer counts aligned with what :func:`reduce_windows` returns, zero
        where a window held nothing, carrying no attributes.

    Raises
    ------
    ValueError
        As :func:`reduce_windows`, for the field, the windows and the labels.
    """
    _check_aggregatable(field)
    stamps = _time_index(field)
    windows = _checked_windows(windows, stamps)
    labels, label_attrs = _checked_labels(labels, windows)
    membership = windows.get_indexer(stamps)
    counts = _count_by_window(field, membership, len(windows))
    counts = counts.assign_coords({TIME_DIM: labels})
    counts[TIME_DIM].attrs = label_attrs
    return counts


# ── supporting helpers ────────────────────────────────────────────────────────


def _whole_numbers(values: np.ndarray, *, name: str) -> np.ndarray:
    """*values* as ``int64``, raising if any is not a whole number."""
    if values.dtype.kind in "iu":
        return values.astype(np.int64)
    if values.dtype.kind == "f":
        if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must hold whole numbers")
        return values.astype(np.int64)
    if values.dtype.kind == "b":
        raise ValueError(f"{name} must be numeric, got booleans")
    try:
        return _whole_numbers(values.astype(np.float64), name=name)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def _time_index(field: xr.DataArray) -> pd.DatetimeIndex:
    """The ``time`` coordinate as a ``DatetimeIndex``, time zone kept."""
    return pd.DatetimeIndex(field.coords[TIME_DIM].to_index())


def _period_time_attrs(field: xr.DataArray, freq: str) -> dict:
    """Attributes for the ``time`` coordinate of a regular aggregation.

    pandas labels start-anchored frequencies by the period's start and
    end-anchored ones by its end. The source's clock attributes still hold
    and are kept; its label attributes describe the source's steps and are
    replaced.
    """
    source = field.coords[TIME_DIM].attrs
    label = pd.Grouper(freq=freq).label
    edge = TimeLabel.INTERVAL_START if label == "left" else TimeLabel.INTERVAL_END
    attrs = {key: source[key] for key in ("time_zone", "clock_status", "clock_provenance") if key in source}
    attrs["long_name"] = f"Period label ({freq})"
    attrs[TIME_LABEL_ATTR] = edge.value
    attrs["time_label_note"] = (
        f"Each label marks the {'start' if label == 'left' else 'end'} of a "
        f"period of {freq}, aggregated from a source whose labels were "
        f"{source.get(TIME_LABEL_ATTR, 'unrecorded')}."
    )
    return attrs


def _resolved_method(field: xr.DataArray, how: str | None) -> str:
    """The reduction to apply, from *how* or from the field's attribute."""
    if how is not None:
        return _check_reduction(how)

    rule = field.attrs.get(AGGREGATION_ATTR)
    what = f"{field.name!r}" if field.name is not None else "this unnamed array"
    if rule is None:
        raise ValueError(
            f"{what} carries no {AGGREGATION_ATTR!r} attribute, so there is no "
            "rule to aggregate it by; pass how=. pySIPNET writes the attribute "
            "onto model output and load_drivers onto the drivers; arithmetic "
            "between arrays that disagree on it drops it. An observation never "
            "has one: how it is placed in time is its observation operator's "
            "decision."
        )
    if rule == _NO_REDUCTION:
        raise ValueError(
            f"{what} carries {AGGREGATION_ATTR}={rule!r}, which is what pySIPNET "
            "writes on a time coordinate such as year or day_of_year; it "
            "cannot be aggregated"
        )
    if rule not in REDUCTIONS:
        raise ValueError(
            f"{what} carries {AGGREGATION_ATTR}={rule!r}, which is not one of "
            f"{list(REDUCTIONS)}; pass how= to say what is meant"
        )
    return str(rule)


def _reduce(grouped, method: str) -> xr.DataArray:
    """Apply *method* to a resample or groupby object.

    ``skipna`` makes ``first``/``last`` the first/last *observed* value of a
    period rather than its first/last row.
    """
    if method == "sum":
        return grouped.sum()
    if method == "mean":
        return grouped.mean()
    if method == "min":
        return grouped.min()
    if method == "max":
        return grouped.max()
    if method == "last":
        return grouped.last(skipna=True)
    if method == "first":
        return grouped.first(skipna=True)
    raise ValueError(f"unhandled reduction {method!r}")


def _is_datetime(dtype) -> bool:
    """Whether *dtype* is a datetime one, timezone-aware ones included."""
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return isinstance(dtype, pd.DatetimeTZDtype)
    return np.issubdtype(dtype, np.datetime64)


def _rows_per_period(field: xr.DataArray, freq: str) -> xr.DataArray:
    """How many rows of *field*'s time axis fall in each period, missing or not."""
    ones = xr.DataArray(
        np.ones(field.sizes[TIME_DIM]),
        dims=TIME_DIM,
        coords={TIME_DIM: field.coords[TIME_DIM]},
    )
    return ones.resample({TIME_DIM: freq}).sum().fillna(0)


def _count_by_period(field: xr.DataArray, freq: str) -> xr.DataArray:
    """Values that are not missing, per period, as ``int64``."""
    # An empty period sums to NaN; fill before the cast, which is otherwise
    # platform-dependent (0 on arm64, INT64_MIN on x86-64).
    counts = field.notnull().resample({TIME_DIM: freq}).sum()
    counts = counts.fillna(0).astype(np.int64)
    counts.name = None
    counts.attrs = {}
    return counts


def _members(field: xr.DataArray, membership: np.ndarray) -> xr.DataArray:
    """The rows inside some window, with the window index as a coordinate."""
    inside = np.flatnonzero(membership >= 0)
    rows = field.isel({TIME_DIM: inside})
    return rows.assign_coords({_WINDOW: (TIME_DIM, membership[inside])})


def _by_window(reduced_groups: xr.DataArray, n_windows: int, dims) -> xr.DataArray:
    """One entry per window, in the field's dimension order, without labels."""
    full = reduced_groups.reindex({_WINDOW: np.arange(n_windows)})
    full = full.rename({_WINDOW: TIME_DIM}).drop_vars(TIME_DIM, errors="ignore")
    return full.transpose(*dims)


def _reduce_by_window(
    field: xr.DataArray, membership: np.ndarray, how: str, n_windows: int
) -> xr.DataArray:
    """*field* reduced per window, ``NaN`` where a window has no rows."""
    if not (membership >= 0).any():
        shape = [n_windows if dim == TIME_DIM else field.sizes[dim] for dim in field.dims]
        coords = {name: coord for name, coord in field.coords.items() if TIME_DIM not in coord.dims}
        return xr.DataArray(np.full(shape, np.nan), dims=field.dims, coords=coords)
    reduced = _reduce(_members(field, membership).groupby(_WINDOW), how)
    return _by_window(reduced, n_windows, field.dims)


def _count_by_window(field: xr.DataArray, membership: np.ndarray, n_windows: int) -> xr.DataArray:
    """Values that are not missing, per window, as ``int64``."""
    if not (membership >= 0).any():
        shape = [n_windows if dim == TIME_DIM else field.sizes[dim] for dim in field.dims]
        coords = {name: coord for name, coord in field.coords.items() if TIME_DIM not in coord.dims}
        return xr.DataArray(np.zeros(shape, dtype=np.int64), dims=field.dims, coords=coords)
    counts = _members(field, membership).notnull().groupby(_WINDOW).sum()
    counts = _by_window(counts, n_windows, field.dims).fillna(0).astype(np.int64)
    counts.name = None
    counts.attrs = {}
    return counts


# ── checks ────────────────────────────────────────────────────────────────────


def _check_frequency(freq: str) -> None:
    """Raise unless *freq* is a pandas offset alias naming a positive period.

    ``to_offset(None)`` returns ``None`` rather than raising, hence the
    explicit check.
    """
    if not isinstance(freq, str):
        raise ValueError(
            "freq must be a pandas offset alias such as '1D', 'MS' or 'YS', "
            f"got {freq!r}"
        )
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except Exception as error:
        raise ValueError(
            "freq must be a pandas offset alias such as '1D', 'MS' or 'YS', "
            f"got {freq!r}"
        ) from error
    if offset is None:
        raise ValueError(
            "freq must be a pandas offset alias such as '1D', 'MS' or 'YS', "
            f"got {freq!r}"
        )
    if offset.n <= 0:
        raise ValueError(
            f"freq={freq!r} names a period of {offset.n} steps, which cannot "
            "group anything; pass a positive frequency"
        )


def _check_reduction(how) -> str:
    """Raise unless *how* is one of :data:`REDUCTIONS`; return it as a plain string."""
    if not isinstance(how, str) or how not in REDUCTIONS:
        raise ValueError(f"how must be one of {list(REDUCTIONS)}, got {how!r}")
    return str(how)


def _check_aggregatable(field: xr.DataArray) -> None:
    """Raise unless *field* has a strictly increasing datetime ``time`` axis."""
    if not isinstance(field, xr.DataArray):
        advice = (
            " A Dataset holds several variables, whose aggregation rules "
            "differ; aggregate one field at a time."
            if isinstance(field, xr.Dataset)
            else ""
        )
        raise ValueError(
            f"expected an xarray.DataArray, got {type(field).__name__}.{advice}"
        )
    if TIME_DIM not in field.dims:
        raise ValueError(
            f"the array has dimensions {list(field.dims)} and needs "
            f"{TIME_DIM!r} to be aggregated in time"
        )
    if TIME_DIM not in field.coords:
        raise ValueError(
            f"the array has a {TIME_DIM!r} dimension but no {TIME_DIM!r} "
            "coordinate, so there is nothing to group its rows by"
        )
    times = field.coords[TIME_DIM]
    if not _is_datetime(times.dtype):
        raise ValueError(
            f"the {TIME_DIM!r} coordinate has dtype {times.dtype}, and "
            "aggregation needs datetimes. SIPNET's output and the .clim "
            "drivers carry year, day and hour columns instead; convert them "
            "with sipnet_time_index first."
        )
    stamps = _time_index(field)
    if len(stamps) == 0:
        raise ValueError(
            f"the {TIME_DIM!r} axis is empty, so there is nothing to "
            "aggregate. A selection that matched no timestamps is the usual "
            "cause."
        )
    if stamps.hasnans:
        raise ValueError(
            f"the {TIME_DIM!r} coordinate holds a missing timestamp (NaT), so "
            "its rows cannot be grouped into periods. Drop those rows, or "
            "rebuild the axis with sipnet_time_index."
        )
    if not stamps.is_monotonic_increasing or stamps.has_duplicates:
        backwards = np.flatnonzero(stamps[1:] <= stamps[:-1])
        where = int(backwards[0]) + 1
        raise ValueError(
            f"the {TIME_DIM!r} coordinate is not strictly increasing: entry "
            f"{where} ({stamps[where]}) does not follow entry {where - 1} "
            f"({stamps[where - 1]}). Two sources concatenated out of order "
            "group into overlapping periods, which is wrong rather than empty."
        )


def _check_not_upsampling(field: xr.DataArray, freq: str) -> None:
    """Raise if every period of *freq* is shorter than the field's spacing.

    Upsampling returns a field that is mostly ``NaN`` with no error. The
    comparison is between the smallest gap in the field's time axis and the
    longest period *freq* produces on it, so a sparse or gapped field at its
    own cadence or coarser passes, and calendar periods of varying length
    (365 or 366 days, 28 to 31) are measured rather than assumed.
    """
    stamps = _time_index(field)
    if len(stamps) < 2:
        return
    labels = pd.DatetimeIndex(_rows_per_period(field, freq).coords[TIME_DIM].to_index())
    # The label differences give every period's length but the last one's, so
    # the boundary after the last label is appended.
    offset = pd.tseries.frequencies.to_offset(freq)
    edges = labels.append(pd.DatetimeIndex([labels[-1] + offset]))
    spacing = np.diff(stamps.as_unit("ns").asi8).min()
    period = np.diff(edges.as_unit("ns").asi8).max()
    if spacing > period:
        raise ValueError(
            f"freq={freq!r} produces periods of at most "
            f"{pd.Timedelta(int(period), 'ns')} on a field whose rows are at "
            f"least {pd.Timedelta(int(spacing), 'ns')} apart, so this would "
            "interpolate rather than aggregate and would return a field that "
            "is mostly missing. Pass a coarser frequency."
        )


def _check_min_count(min_count: int) -> None:
    """Raise unless *min_count* is an integer of at least 1."""
    if isinstance(min_count, bool) or not isinstance(min_count, (int, np.integer)):
        raise ValueError(
            f"min_count must be an integer, got {min_count!r}. It counts "
            "values, so a fraction of one has no meaning."
        )
    if int(min_count) < 1:
        raise ValueError(
            f"min_count must be at least 1, got {min_count!r}. Zero would let "
            "a period formed from no observations produce a value, which for "
            "a sum is the zero this argument exists to prevent."
        )


def _checked_windows(windows, stamps: pd.DatetimeIndex) -> pd.IntervalIndex:
    """*windows* checked and put in the time unit of *stamps*.

    Raises unless it is a non-empty, increasing, non-overlapping
    ``IntervalIndex`` of datetimes without ``NaT``, naive or in the time zone
    of *stamps*.
    """
    if not isinstance(windows, pd.IntervalIndex):
        raise ValueError(
            "windows must be a pandas.IntervalIndex, for instance from "
            f"pd.IntervalIndex.from_arrays(starts, ends, closed='right'); got "
            f"{type(windows).__name__}"
        )
    if len(windows) == 0:
        raise ValueError("windows is empty, so there is nothing to reduce into")
    if not _is_datetime(windows.left.dtype):
        raise ValueError(
            f"windows must be intervals of datetimes, got {windows.left.dtype}"
        )
    left, right = pd.DatetimeIndex(windows.left), pd.DatetimeIndex(windows.right)
    if left.hasnans or right.hasnans:
        raise ValueError("windows hold a missing edge (NaT)")
    if left.tz != stamps.tz:
        raise ValueError(
            f"windows are in time zone {left.tz} and the field's time "
            f"coordinate in {stamps.tz}; pandas matches no rows across that "
            "difference. Localize or convert one of them first."
        )
    if windows.is_overlapping:
        raise ValueError(
            "windows overlap, so a row could belong to two of them; reduce "
            "into non-overlapping windows"
        )
    if not left.is_monotonic_increasing:
        raise ValueError("windows must be in increasing order")
    # pandas refuses to index one datetime resolution with another.
    unit = stamps.unit
    return pd.IntervalIndex.from_arrays(left.as_unit(unit), right.as_unit(unit), closed=windows.closed)


def _checked_labels(labels, windows: pd.IntervalIndex) -> tuple[pd.DatetimeIndex, dict]:
    """The result's ``time`` coordinate and its attributes.

    *labels* checked against *windows*, keeping a ``DataArray``'s attributes;
    or the windows' right edges, labeled as interval ends.
    """
    if labels is None:
        index = pd.DatetimeIndex(windows.right)
        attrs = {
            TIME_LABEL_ATTR: TimeLabel.INTERVAL_END.value,
            "time_label_note": "Each label is the right edge of its window.",
        }
        what = "the windows' right edges"
    else:
        values = labels.values if isinstance(labels, xr.DataArray) else np.asarray(labels).ravel()
        if values.dtype.kind in "iufb":
            raise ValueError(
                f"labels must be timestamps, got dtype {values.dtype}; numbers "
                "would be read as nanoseconds since 1970"
            )
        index = pd.DatetimeIndex(values)
        attrs = dict(labels.attrs) if isinstance(labels, xr.DataArray) else {}
        what = "labels"
    if len(index) != len(windows):
        raise ValueError(
            f"labels has {len(index)} entries for {len(windows)} windows; "
            "one label per window"
        )
    if index.hasnans or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(
            f"{what} must be strictly increasing timestamps with no NaT; pass "
            "labels= to name the windows otherwise"
        )
    return index, attrs
