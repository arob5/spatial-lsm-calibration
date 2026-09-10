"""Observation-space operations shared by the likelihood and the plots.

This module exists so that temporal aggregation and observation indexing each
have exactly one implementation. It is imported by both the observation operator
H and the plotting layer, so a predictive-check figure cannot silently disagree
with what the likelihood consumed.

Provided:

* ``sipnet_time_index(year, day_of_year, hours_since_midnight, *,
  timestep_hours) -> DatetimeIndex`` -- SIPNET output and ``.clim`` drivers
  carry ``year``, ``day`` and ``time`` columns, not a datetime index. The
  ``time`` column drifts (issue #9) and is used only to identify a row's slot
  within its day, never as the timestamp.

Planned (issue #6):

* ``aggregate_time(field, freq, *, how=None)``
* ``obs_index(sites, variables, times) -> pd.MultiIndex`` -- the
  ``(site, variable, time)`` labeling of the flat observation vector. The same
  object must be used to flatten observations into ``y`` and to unstack EKI's
  ``(J, N)`` predictions back into canonical fields, or the two will mislabel
  relative to each other. (Note: this was originally expected to come from an
  ``index`` layer in pyEKI. That layer does not exist -- it is this module's job.)

**The aggregation rule is a property of the variable, not of the call site.**
When it lands, ``how`` will default to ``VARIABLES[field.name].agg``; pass it
only to deliberately override. SIPNET's ``nee`` is ``g C m-2 per timestep`` -- extensive -- so
3-hourly to daily is a **sum**, and a mean is wrong by a factor of 8 while
looking entirely plausible. ``tair``/``vpd`` are intensive (mean),
``par``/``precip`` are per-timestep totals (sum), and carbon pools are stocks
(instantaneous).

Any rate-vs-total conversion must use the ``.clim`` ``length`` column (timestep
length in days) rather than assuming a fixed timestep.

Aggregation is a verb the caller applies, never a plotter keyword::

    plot_time_series(agg(nee, "1D"))          # yes
    plot_time_series(nee, temporal_agg="1D")  # no

That keeps a real subtlety at the call site: quantile-of-daily-mean is not
daily-mean-of-quantile, and which one is wanted is a modeling choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["sipnet_time_index"]


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
