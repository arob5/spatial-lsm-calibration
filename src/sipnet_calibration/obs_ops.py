"""Observation-space operations shared by the likelihood and the plots.

This module exists so that temporal aggregation and observation indexing each
have exactly one implementation. It is imported by both the observation operator
H and the plotting layer, so a predictive-check figure cannot silently disagree
with what the likelihood consumed.

Planned contents:

* ``aggregate_time(field, freq, *, how=None)``
* ``sipnet_time_index(year, day, time) -> DatetimeIndex`` -- SIPNET output and
  ``.clim`` drivers carry ``year``, ``day``, ``time`` columns, not a datetime index.
* ``obs_index(sites, variables, times) -> pd.MultiIndex`` -- the
  ``(site, variable, time)`` labelling of the flat observation vector. The same
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
daily-mean-of-quantile, and which one is wanted is a modelling choice.
"""
