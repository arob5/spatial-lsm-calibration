"""Observation-space operations shared by the likelihood and the plots.

This module exists so that temporal aggregation and observation indexing each
have exactly one implementation. It is imported by both the observation operator
H and the plotting layer, so a predictive-check figure cannot silently disagree
with what the likelihood consumed.

Provided:

* ``sipnet_time_index(year, day, time, *, timestep_hours) -> DatetimeIndex`` --
  SIPNET output and ``.clim`` drivers carry ``year``, ``day``, ``time`` columns,
  not a datetime index. The ``time`` column drifts (issue #9) and is used only
  to identify a row's slot within its day, never as the timestamp.

Planned (issue #6):

* ``aggregate_time(field, freq, *, how=None)``
* ``obs_index(sites, variables, times) -> pd.MultiIndex`` -- the
  ``(site, variable, time)`` labeling of the flat observation vector. The same
  object must be used to flatten observations into ``y`` and to unstack EKI's
  ``(J, N)`` predictions back into canonical fields, or the two will mislabel
  relative to each other. (Note: this was originally expected to come from an
  ``index`` layer in pyEKI. That layer does not exist -- it is this module's job.)

**The aggregation rule is a property of the variable, not of the call site.**
``how`` defaults to ``VARIABLES[field.name].agg``; pass it only to deliberately
override. SIPNET's ``nee`` is ``g C m-2 per timestep`` -- extensive -- so
3-hourly to daily is a **sum**, and a mean is wrong by a factor of 8 while
looking entirely plausible. ``tair``/``vpd`` are intensive (mean),
``par``/``precip`` are per-timestep totals (sum), and carbon pools are stocks
(instantaneous).

Any rate-vs-total conversion must use the ``.clim`` ``length`` column (timestep
length in days) rather than assuming a fixed timestep.

Aggregation is a verb the caller applies, never a plotter keyword::

    series_panel(agg(nee, "1D"))          # yes
    series_panel(nee, temporal_agg="1D")  # no

That keeps a real subtlety at the call site: quantile-of-daily-mean is not
daily-mean-of-quantile, and which one is wanted is a modeling choice.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["sipnet_time_index"]


def sipnet_time_index(year, day, time, *, timestep_hours: float = 3.0) -> pd.DatetimeIndex:
    """Timestamps for rows labeled the way SIPNET labels them.

    SIPNET's climate files and its output give each row a ``year``, an integer
    ``day`` of year with 1 being January 1, and a fractional-hour ``time``. This
    builds the nominal timestamp of each row as::

        year-01-01  +  (day - 1) days  +  slot * timestep_hours

    where ``slot = floor(time / timestep_hours)`` is the row's position within
    its day. The ``time`` value itself is used for nothing else.

    Parameters
    ----------
    year, day, time:
        Array-likes of equal length. ``year`` and ``day`` are integers (or
        floats that are whole numbers); ``time`` is hours since midnight of
        ``day``.
    timestep_hours:
        Length of one row's timestep in hours; must divide 24. The default is
        the 3-hourly drivers. A daily file passes ``24.0``.

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
        If the lengths differ; ``timestep_hours`` does not divide 24; a
        ``day`` is outside ``1..366``, or is 366 in a non-leap year; a ``time``
        is outside ``[0, 24)``; or a ``time`` label does not sit inside its
        slot, meaning ``time - slot * timestep_hours`` is not in
        ``[0, timestep_hours)``. That last condition is what a label with a
        drift of one full step or more would violate.

    Notes
    -----
    ``floor`` rather than ``round`` because the ``.clim`` ``time`` column
    drifts late by up to two hours within a year (issue #9): the row for
    nominal hour 21 is labeled ``23.00`` on 31 December, and rounding would
    put it in a ninth slot. The drift is always non-negative and always below
    one step, so ``floor`` identifies the slot in every row. SIPNET copies the
    same column verbatim into its output, which is why this lives here rather
    than in a driver-specific module.

    Nothing about the result depends on an interval convention. The nominal
    label ``slot * timestep_hours`` is what the source wrote, and
    ``resample`` on it groups a day's rows exactly as SIPNET's own ``day``
    column does.
    """
    raise NotImplementedError
