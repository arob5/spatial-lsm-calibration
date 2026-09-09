"""Plotting suite: matplotlib, ensemble- and site-aware.

Five layers, dependencies strictly upward:

* **L1** :mod:`primitives` -- signature always ``(ax, plain numpy, **style) -> artist``.
  No pandas, no xarray, no figure creation.
* **L2** :mod:`series`, :mod:`maps` -- one variable, one Axes, canonical field in.
* **L3** :mod:`facet` -- the one generic facet function; owns figure and axes
  construction, shared limits, legend de-duplication.
* **L4** experiment reports -- in ``experiments/<task>/plots.py``, *never* here.
  Anything that knows a task name is not core.
* **L5** :mod:`diagnostics` -- inference diagnostics (EKI history, marginals,
  coverage, per-site parameter maps).

Invariants, for every plotter in this package:

* takes ``ax``, returns ``Axes``;
* never calls ``plt.show()`` or ``savefig``, and never creates a figure
  implicitly (that is :mod:`facet`'s job);
* never accepts a ``SIPNETResult``, a DataFrame, or a path (that is an adapter's
  job, in :mod:`sipnet_calibration.fields`);
* takes style from :mod:`registry` and :mod:`style`, not from a dozen keywords.

Three rules cut across the layers and are stated once here:

* **``time`` is the x-axis of a series panel, and every other dim present is a
  sample dim.** A field shaped ``(site, time)`` draws a curve per site exactly
  as ``(member, time)`` draws a curve per member. See :mod:`series`.
* **Lines and bands keep** ``NaN`` **so the gap shows; points drop it** so the
  artist holds exactly what was observed. See :mod:`primitives`.
* **Style precedence is primitive default, then role, then explicit keyword.**
  See :mod:`style`.

Aggregation is not part of this package. It is a verb the caller applies with
:func:`sipnet_calibration.obs_ops.aggregate_time`, which defaults to the
variable's own rule, so that a predictive-check figure cannot disagree with
what the likelihood consumed.

Interactive single-run inspection is out of scope: ``pysipnet.viz.dashboard``
already owns it.

The spatial layer -- :mod:`maps`, and the ``basemap``, ``map_points`` and
``map_raster`` primitives -- is not implemented; it is blocked on the
projection decision in issue #4.
"""

from sipnet_calibration.plotting.facet import by_site, by_variable, facet
from sipnet_calibration.plotting.primitives import band, fan, line, points, spaghetti
from sipnet_calibration.plotting.series import series_panel
from sipnet_calibration.plotting.style import (
    BAND_ALPHAS,
    CURVE_COLORS,
    RC_PARAMS,
    ROLES,
    axis_label,
    role_style,
    use_project_style,
)

__all__ = [
    "BAND_ALPHAS",
    "CURVE_COLORS",
    "RC_PARAMS",
    "ROLES",
    "axis_label",
    "band",
    "by_site",
    "by_variable",
    "facet",
    "fan",
    "line",
    "points",
    "role_style",
    "series_panel",
    "spaghetti",
    "use_project_style",
]
