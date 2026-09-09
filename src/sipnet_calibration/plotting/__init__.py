"""Plotting for the project's canonical fields: matplotlib, ensemble-aware.

A canonical field is an ``xarray.DataArray`` whose dimensions are a subset of
``(member, site, time)``, described in
:mod:`sipnet_calibration.fields`. Everything here takes one, or a mapping of
several, and draws it.

==========================  ================================================
:mod:`primitives`           ``(ax, numpy arrays, **style) -> artist``
:mod:`series`               a time-series panel on one ``Axes``
:mod:`maps`                 a spatial panel on one ``Axes`` (not implemented)
:mod:`facet`                a grid of panels, and the figure around it
:mod:`diagnostics`          inference diagnostics (not implemented)
:mod:`style`                roles, colors and matplotlib settings
:mod:`registry`             per-variable display metadata (not implemented)
==========================  ================================================

Every panel takes an ``Axes`` and returns it. None of them creates a figure
except when no ``Axes`` is given, and none calls ``show`` or ``savefig``. None
accepts a ``SIPNETResult``, a ``DataFrame`` or a path; converting those to a
canonical field is the job of an adapter in
:mod:`sipnet_calibration.fields`. Reports that know an experiment's name
belong in ``experiments/<task>/plots.py`` rather than here.

Three conventions run through the package:

* In a time-series panel, ``time`` is the x-axis and every other dimension
  present is a sample dimension, so a field shaped ``(site, time)`` draws a
  curve per site as ``(member, time)`` draws one per member.
* Curves and bands keep ``NaN``, so gaps show; scattered points drop it.
* Style is resolved as the drawing function's default, then the role, then any
  keyword given explicitly.

Temporal aggregation is not done here. It is applied by the caller with
:func:`sipnet_calibration.obs_ops.aggregate_time`, which the observation
operator also uses.

Interactive inspection of a single run is out of scope; ``pysipnet.viz``
covers it.
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
