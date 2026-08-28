"""Observation-space operations shared by the likelihood and the plots.

This module exists so that temporal aggregation has exactly one implementation.
It is imported by both the observation operator H and the plotting layer, so a
predictive-check figure cannot silently disagree with what the likelihood
consumed.

Aggregation is a verb the caller applies, never a plotter keyword::

    series_panel(agg(nee, "1D"), obs=agg(nee_obs, "1D"))   # yes
    series_panel(nee, temporal_agg="1D", obs=nee_obs)       # no

That keeps a real subtlety at the call site: quantile-of-daily-mean is not
daily-mean-of-quantile, and which one is wanted is a modelling choice.

Also home to ``sipnet_time_index``: SIPNET output and .clim drivers carry
``year``, ``day``, ``time`` columns rather than a datetime index.
"""
