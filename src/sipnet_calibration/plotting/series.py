"""L2 -- time-series panels.

``series_panel(field, ax=None, *, show="fan"|"spaghetti"|"line", obs=None,
role=..., levels=...)`` covers prior predictive, posterior predictive, a driver
ensemble, the NEE observation ensemble, and a single deterministic run -- one
function, because it branches on presence of the ``member`` dim rather than on a
mode keyword.

Aggregation is *not* a keyword here; callers apply
:func:`sipnet_calibration.obs_ops.aggregate_time` first. Overlaying several
aggregations on raw data is then three calls onto one ``ax``.

Lines must be NaN-aware: never interpolate across gaps in observations.
"""
