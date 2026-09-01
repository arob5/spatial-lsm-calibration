"""L2 -- time-series panels.

``series_panel(field, ax=None, *, show="auto", role="posterior",
levels=(0.5, 0.9), **style) -> Axes``

``show="auto"`` resolves to ``"fan"`` when the ``member`` dim is present and
``"line"`` when it is not. That single branch is what lets one function cover
prior predictive, posterior predictive, a driver ensemble, the NEE observation
ensemble, and a single deterministic run.

**There is no ``obs=`` parameter.** Observations are themselves often an
ensemble (the NEE product has 25 members), so an ``obs=`` keyword would need a
rule for rendering "obs" differently from "model" that no single default gets
right. Because every plotter takes ``ax`` and returns ``Axes``, overlay is just a
second call::

    ax = series_panel(pred, role="posterior")
    series_panel(nee_obs, ax=ax, role="obs")

The same pattern gives truth lines and several aggregations on one panel, with no
extra machinery.

Aggregation is not a keyword here either; callers apply
``sipnet_calibration.obs_ops.aggregate_time`` first, and it defaults to the
variable's own rule from the registry (``nee`` sums, ``tair`` means).

Lines must be NaN-aware: never interpolate across gaps in observations.
"""
