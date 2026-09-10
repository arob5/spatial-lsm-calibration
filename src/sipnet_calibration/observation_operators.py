"""Observation-space operations shared by the likelihood and the plots.

Overview
--------
Two things this project does happen in observation space rather than in model
space: turning SIPNET's own row labeling into timestamps, and aggregating a
series in time. Both belong to the observation operator, which applies them
before the likelihood sees a residual, and both are wanted again by the
plotting layer before a predictive check is drawn. This module holds the one
implementation of each, for both to import.

Functions
---------
:func:`sipnet_time_index`
    Timestamps for rows labeled the way SIPNET labels them, with ``year``,
    ``day`` and ``time`` columns rather than a datetime index. The ``time``
    column drifts (issue #9) and is used only to identify a row's slot within
    its day, never as the timestamp itself.

:func:`aggregate_time`
    Aggregate a canonical field along ``time``, taking the rule from
    :data:`~sipnet_calibration.variable_registry.VARIABLES`.

:func:`aggregation_counts`
    How many values each period of such an aggregation was formed from, so a
    caller can impose its own completeness rule.

Aggregation is a verb the caller applies, never a plotter keyword::

    plot_time_series(aggregate_time(nee, "1D"))     # yes
    plot_time_series(nee, temporal_agg="1D")        # no

Besides keeping signatures small, that keeps a real subtlety at the call site:
a quantile of daily means is not the daily mean of a quantile, and which one
is wanted is a modeling choice rather than a default.

Notes
-----
**Why one implementation and not two.** A posterior predictive figure drawn
at a different aggregation than the likelihood used looks fine, is wrong, and
takes a day to diagnose. Sharing the code path removes the possibility. If a
second aggregation path ever seems wanted for plotting, that is a sign the
design is wrong rather than that the rule needs an exception.

**The rule is a property of the variable, not of the call site.** ``how``
defaults to ``VARIABLES[field.name].agg``; pass it only to override
deliberately. SIPNET's ``nee`` is ``g C m-2`` per timestep -- extensive -- so
3-hourly to daily is a **sum**, and a mean is wrong by a factor of eight while
looking entirely plausible. ``air_temperature`` and ``vpd`` are intensive
(mean), ``par`` and ``precipitation`` are per-timestep totals (sum), and the
carbon and leaf-area stocks have no rule at all and are refused. The registry
is the authority; the ``aggregation`` attribute that
:mod:`sipnet_calibration.drivers` writes onto its fields is not read here.

**Nothing here converts between a rate and a total.** Where an adapter must --
observed NEE is a rate and SIPNET's is a per-timestep total -- the timestep
length comes from the ``.clim`` ``length`` column rather than an assumed three
hours, and the factor is
:data:`~sipnet_calibration.variable_registry.GRAMS_CARBON_PER_MICROMOLE_CO2`.

**What a period's label means.** Periods are left-closed, and pandas decides
where the label goes: a start-anchored frequency such as ``"1D"``, ``"MS"``
or ``"YS"`` labels the period's start, while an end-anchored one such as
``"ME"`` or ``"YE"`` labels its end. Nothing here overrides that, so the
label the result carries is the one the frequency asked for.

The drivers and SIPNET's output label the **end** of each timestep
(``drivers.TIME_LABEL``), so a day's eight rows labeled 00:00 to 21:00 span
the interval from 21:00 the previous day to 21:00 on the day they are labeled
with. That is not corrected here, deliberately: grouping by the nominal label
reproduces SIPNET's own ``day`` column exactly, which is the grouping the
model's own daily output uses and therefore the one a comparison against it
needs. Observed NEE is on a different clock again (issue #8), and reconciling
the two is the NEE adapter's problem, not this module's.

**Planned, and not implemented.** ``obs_index`` will give the
``(site, variable, time)`` labeling of the flat observation vector, which the
observation operator and ``fields.from_eki_predictions`` must share or the
predictions come back mislabeled against the observations. It has to
enumerate the triples actually observed rather than a product of the axes,
since neither the NEE record nor the annual constraints is rectangular.
Deferred to issue #23, which carries the signature and the reasoning.

Usage
-----
::

    from sipnet_calibration.observation_operators import (
        aggregate_time,
        aggregation_counts,
        sipnet_time_index,
    )

    daily_par = aggregate_time(par, "1D")     # summed: par is a total
    daily_tair = aggregate_time(tair, "1D")   # meaned: it is intensive

    # A stricter completeness rule than the default, applied identically
    # wherever it is wanted.
    counts = aggregation_counts(par, "1D")
    whole_days = aggregate_time(par, "1D").where(counts == 8)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.variable_registry import (
    AGGREGATION_RULES,
    INSTANTANEOUS,
    variable_spec,
)

__all__ = [
    "AGGREGATION_METHODS",
    "TIME_DIM",
    "aggregate_time",
    "aggregation_counts",
    "sipnet_time_index",
]

#: The dimension aggregated along.
TIME_DIM = "time"

#: What ``how`` may be: every rule of
#: :data:`~sipnet_calibration.variable_registry.AGGREGATION_RULES` that names
#: a reduction. :data:`~sipnet_calibration.variable_registry.INSTANTANEOUS` is
#: a refusal rather than a method, so it is not among them and cannot be
#: passed as an override.
AGGREGATION_METHODS: tuple[str, ...] = tuple(
    rule for rule in AGGREGATION_RULES if rule != INSTANTANEOUS
)


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
        or the resulting index is not strictly increasing. SIPNET writes rows
        in order, so the last condition is what a label drifting by one full
        step or more turns into: the row lands in the next row's slot and the
        two collide.

    Notes
    -----
    ``floor`` rather than ``round`` because the ``.clim`` ``time`` column
    drifts late by up to two hours within a year (issue #9): the row for
    nominal hour 21 is labeled ``23.00`` on 31 December, and rounding would
    put it in a ninth slot. The drift is always non-negative and always below
    one step, so ``floor`` identifies the slot in every row. SIPNET copies the
    same column verbatim into its output, which is why this lives here rather
    than in a driver-specific module.

    A drift of less than one step is invisible to a single label by design;
    :mod:`sipnet_calibration.drivers` asserts the drift model of the source
    separately, on whole files.

    The slot division is exact for the timesteps SIPNET is run at (3, 1, 0.5
    and 24 hours). A step such as ``0.1`` divides 24 but is not exactly
    representable, so a label sitting exactly on a slot boundary can floor
    into the slot below.

    Nothing about the result depends on an interval convention. The nominal
    label ``slot * timestep_hours`` is what the source wrote, and
    ``resample`` on it groups a day's rows exactly as SIPNET's own ``day``
    column does.
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
    """Aggregate *field* along ``time``, at the variable's own rule.

    The rule comes from the registry, not from the call site:
    ``VARIABLES[field.name].agg`` decides whether a coarser period is the sum
    of the finer ones or their mean. Pass *how* only to override it
    deliberately.

    Parameters
    ----------
    field:
        A canonical field with a ``time`` dimension carrying a datetime
        coordinate. Any other dimensions -- ``member``, ``site`` -- are
        untouched, as are the coordinates on them, so ``lon``/``lat`` survive.
        Its ``name`` must be a key of
        :data:`~sipnet_calibration.variable_registry.VARIABLES` unless *how*
        is given.
    freq:
        The target period, as a pandas offset alias: ``"1D"``, ``"MS"``,
        ``"YS"``. Must be at least the field's own spacing; see Raises.
    how:
        One of :data:`AGGREGATION_METHODS`, overriding the variable's rule.
        ``None`` takes the rule from the registry. Given explicitly, the
        registry is not consulted at all, so a field whose variable is not
        registered -- an observation error variance, say -- can still be
        aggregated by saying how.
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
        aggregations of one variable are distinguishable: a canonical
        per-timestep total carries the same ``units`` at every resolution.

    Raises
    ------
    ValueError
        If *field* is not a ``DataArray``, has no ``time`` dimension, has no
        ``time`` coordinate, or that coordinate is not a datetime one and so
        cannot be resampled; if the ``time`` coordinate is not increasing; if
        *how* is given and is not in :data:`AGGREGATION_METHODS`; if *how* is
        not given and the field has no name or a name that is not registered;
        if the variable's rule is
        :data:`~sipnet_calibration.variable_registry.INSTANTANEOUS`; if
        *min_count* is not an integer of at least 1; or if *freq* names a
        period shorter than the field's own spacing.

    Notes
    -----
    **Empty periods do not become zero.** ``.resample(...).sum()`` returns 0
    for an all-missing group, so a day with no observations would otherwise
    read as zero flux rather than as unobserved. With NEE about 55% missing
    over site and time that is the common case, not an edge case. The guard is
    one mechanism for every method: the period's count of values that are not
    missing is computed once, and the reduction is masked where it falls below
    *min_count*. ``min_count=`` is deliberately not also passed to ``.sum()``;
    two mechanisms doing one job.

    **Partial periods are not scaled, and are not dropped by default.**
    Summing three of a day's eight timesteps gives a partial total, and
    scaling it up assumes the absent timesteps resemble the present ones. For
    a diurnal flux that is false in the worst direction: a day missing its
    night hours would scale to a strongly negative fake. So the value returned
    is the partial total, and a caller wanting only whole periods says so,
    with ``min_count=`` or by masking on :func:`aggregation_counts`. The rule
    is the caller's to choose and belongs in the experiment's config, where
    the figure and the likelihood read the same one.

    ``min_count=1`` is a safe default for the data this project has rather
    than in general. Both products it is applied to are contiguous in time:
    the gap-filled NEE record runs unbroken within each site's covered years,
    so its missingness is structural rather than punctured and a partial
    period can only arise at a record edge, and the ``.clim`` driver files
    hold no missing values at all. (A driver *field* does carry ``NaN``,
    where :func:`~sipnet_calibration.drivers.load_drivers` was asked for a
    member and site whose file is absent; that is a whole pair missing, not
    an interior gap, and ``driver_present`` says which.) ``data/README.md``
    records the measurements behind that.

    **A stock is refused rather than guessed at.** ``lai`` and the two soil
    and biomass constraints are levels at an instant, not quantities
    accumulated over an interval, so neither a sum nor a mean is their
    aggregation, and picking ``first`` or ``last`` silently would be a choice
    made in the wrong place. Those are also the variables that arrive at
    their source resolution already, where aggregation would be a no-op. A
    caller who does want the year-end value says ``how="last"``.
    """
    # The argument checks come first, so that a mistake in the call is
    # reported as itself rather than as whatever the data then runs into.
    _check_frequency(freq)
    _check_min_count(min_count)
    _check_aggregatable(field)
    method = _resolved_method(field, how)

    _check_not_upsampling(field, freq)

    counts = _count_by_period(field, freq)
    aggregated = _reduce(field.resample({TIME_DIM: freq}), method)
    # One guard for every method, rather than sum's own min_count plus a mask
    # for the rest: a period with nothing in it is missing under all four, and
    # two mechanisms doing one job means either can be deleted unnoticed.
    # xarray aligns `counts` by dimension name, so this is correct whatever
    # order the field stores its dimensions in.
    aggregated = aggregated.where(counts >= int(min_count))

    # The attributes are set here rather than left to the reduction's
    # keep_attrs: its default varies between xarray versions and between
    # reductions, and a dropped `units` breaks the axis label of every plot
    # downstream. Copied, so that aggregating does not add provenance to the
    # field the caller keeps.
    aggregated.name = field.name
    aggregated.attrs = dict(field.attrs)
    aggregated.attrs["aggregation_applied"] = method
    aggregated.attrs["aggregation_freq"] = freq
    return aggregated


def aggregation_counts(field: xr.DataArray, freq: str) -> xr.DataArray:
    """How many values each period of :func:`aggregate_time` is formed from.

    The count of values that are not missing, per period, which is what a
    completeness rule stricter than :func:`aggregate_time`'s default needs.

    Parameters
    ----------
    field:
        As :func:`aggregate_time`. Its ``name`` is not consulted, so a field
        whose variable is not registered can be counted.
    freq:
        As :func:`aggregate_time`.

    Returns
    -------
    xarray.DataArray
        Integer counts, with the same dimensions as the aggregate and the same
        ``time`` axis, so it aligns with what :func:`aggregate_time` returns
        and can mask it directly. Zero where a period held nothing. It carries
        no ``units`` or ``long_name``, being a count of the field rather than
        a field, and so is not a canonical field and cannot be plotted by
        :func:`~sipnet_calibration.plotting.series.plot_time_series`.

    Raises
    ------
    ValueError
        As :func:`aggregate_time`, for the ``time`` axis, *freq* and the
        upsampling check. The variable's rule is not consulted, so a stock is
        counted rather than refused.
    """
    _check_frequency(freq)
    _check_aggregatable(field)
    _check_not_upsampling(field, freq)
    return _count_by_period(field, freq)


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


def _resolved_method(field: xr.DataArray, how: str | None) -> str:
    """The reduction to apply, from *how* or from the variable's rule.

    Raises
    ------
    ValueError
        If *how* is given and is not in :data:`AGGREGATION_METHODS`; if it is
        not given and *field* has no name, or a name the registry does not
        hold; or if the variable's rule is
        :data:`~sipnet_calibration.variable_registry.INSTANTANEOUS`. Each
        message says what to pass instead.
    """
    if how is not None:
        if how == INSTANTANEOUS:
            raise ValueError(
                f"how={how!r} names a refusal rather than a reduction: it is "
                "what a stock's registry entry carries, and it says to choose "
                f"a reduction here. The methods are {list(AGGREGATION_METHODS)}"
            )
        if how not in AGGREGATION_METHODS:
            raise ValueError(
                f"how must be one of {list(AGGREGATION_METHODS)}, got {how!r}"
            )
        return how

    spec = variable_spec(field.name)
    if spec.agg == INSTANTANEOUS:
        raise ValueError(
            f"{field.name!r} is a stock -- a level at an instant rather than a "
            "quantity accumulated over an interval -- so it has no "
            "aggregation rule, and neither a sum nor a mean is right for it. "
            'Say which you want, for instance how="last" for the value at the '
            "end of each period"
        )
    return spec.agg


def _reduce(resampled, method: str) -> xr.DataArray:
    """Apply *method* to a resample object.

    Attributes are not preserved here; :func:`aggregate_time` sets them on the
    result. ``skipna`` is what makes ``last`` the last *observed* value of a
    period rather than its last row, which is ``NaN`` whenever a record ends
    part-way through one.
    """
    if method == "sum":
        return resampled.sum()
    if method == "mean":
        return resampled.mean()
    if method == "last":
        return resampled.last(skipna=True)
    if method == "first":
        return resampled.first(skipna=True)
    raise ValueError(f"unhandled aggregation method {method!r}")


def _is_datetime(dtype) -> bool:
    """Whether *dtype* is a datetime one, timezone-aware ones included.

    ``numpy.issubdtype`` raises a ``TypeError`` on a pandas extension dtype
    such as ``datetime64[us, UTC]``, which xarray resamples correctly, so the
    pandas predicate is asked first.
    """
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return isinstance(dtype, pd.DatetimeTZDtype)
    return np.issubdtype(dtype, np.datetime64)


def _rows_per_period(field: xr.DataArray, freq: str) -> xr.DataArray:
    """How many rows of *field*'s time axis fall in each period.

    Rows, not values, so it says nothing about missingness: it is the shape
    of the grouping itself, which is what :func:`_check_not_upsampling` asks
    about. One dimension, whatever the field's own shape.
    """
    ones = xr.DataArray(
        np.ones(field.sizes[TIME_DIM]),
        dims=TIME_DIM,
        coords={TIME_DIM: field.coords[TIME_DIM]},
    )
    return ones.resample({TIME_DIM: freq}).sum().fillna(0)


def _count_by_period(field: xr.DataArray, freq: str) -> xr.DataArray:
    """Values that are not missing, per period, as ``int64``.

    The one place the count is computed, so :func:`aggregate_time`'s guard and
    :func:`aggregation_counts` cannot disagree about what a period holds.
    """
    # A period holding no rows at all comes back as NaN from the sum, so the
    # fill is what makes the result an integer count rather than an undefined
    # NaN-to-int64 cast -- which saturates to 0 on arm64 and to INT64_MIN on
    # x86-64, and would make an empty period read as fully observed there.
    counts = field.notnull().resample({TIME_DIM: freq}).sum()
    counts = counts.fillna(0).astype(np.int64)
    counts.name = None
    counts.attrs = {}
    return counts


# ── checks ────────────────────────────────────────────────────────────────────


def _check_frequency(freq: str) -> None:
    """Raise unless *freq* is a pandas offset alias naming a positive period.

    Notes
    -----
    ``to_offset(None)`` returns ``None`` rather than raising, so ``freq=None``
    would otherwise reach the resample and fail as a ``TypeError`` naming
    neither the argument nor the array.
    """
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


def _check_aggregatable(field: xr.DataArray) -> None:
    """Raise unless *field* is an array that can be aggregated along time.

    Checks that it is a ``DataArray``, that it has a ``time`` dimension with a
    datetime coordinate on it, and that the coordinate increases. A field
    without a ``time`` coordinate cannot be resampled at all, and one whose
    coordinate is integer-valued fails inside pandas with a message that does
    not say which array was at fault.
    """
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
    # Through a DatetimeIndex rather than the raw values, because a
    # timezone-aware coordinate comes back as an object array of Timestamps,
    # which numpy will not compare against a timedelta64.
    stamps = pd.DatetimeIndex(times.values)
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
    """Raise if *freq* would group nothing and invent empty periods besides.

    Notes
    -----
    Upsampling returns a field that is mostly ``NaN`` with no error, and the
    emptiness reads as missing data rather than as a mistake.

    The test is on what the grouping *does*, not on how long a period lasts,
    because a calendar offset has no single length: a year is 365 days or
    366, a quarter 90 to 92, a month 28 to 31. Comparing one measured period
    against the source's spacing therefore picks the wrong side for an annual
    series that happens to span a leap year, which is exactly what the
    constraint product is -- and it depends on which probe date the period
    was measured from, which is arbitrary.

    Two conditions have to hold together. Every row alone in its own period
    means the aggregation grouped nothing; empty periods besides means it
    invented rows that were never observed. Either on its own is legitimate:
    a daily field aggregated to ``"1D"`` groups nothing and is a lawful
    no-op, and a record with a gap in it produces empty periods while still
    grouping its rows eight at a time.
    """
    rows = _rows_per_period(field, freq)
    n_periods = int(rows.sizes[TIME_DIM])
    n_occupied = int((rows > 0).sum())
    n_rows = int(field.sizes[TIME_DIM])
    if n_occupied == n_rows and n_periods > n_occupied:
        raise ValueError(
            f"freq={freq!r} puts each of the {n_rows} timestamps in a period "
            f"of its own and adds {n_periods - n_occupied} empty ones, so "
            "this would interpolate rather than aggregate and would return a "
            "field that is mostly missing. Pass a coarser frequency."
        )


def _check_min_count(min_count: int) -> None:
    """Raise unless *min_count* is an integer of at least 1.

    Zero would mean a period formed from nothing still produces a value,
    which for a sum is the zero this function exists to prevent.
    """
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
