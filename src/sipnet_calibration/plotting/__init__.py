"""Plotting for the project's model output, drivers and observations.

The figures this project needs are ensemble figures over many sites: a
predictive spread against an observation, a driver ensemble at a handful of
sites, a calibrated parameter across the site pool. This package draws them
from ``xarray`` arrays, so that a plot is asked for in terms of the data
rather than in terms of matplotlib.

==================  ==================================================
:mod:`primitives`   adds one element to an ``Axes``
:mod:`series`       time series plots
:mod:`maps`         maps of values or classes at sites, and of rasters
:mod:`basemap`      the coastlines, boundaries and graticule under a map
:mod:`facet`        grids of panels, and the figure around them
:mod:`diagnostics`  inference diagnostics (not implemented)
:mod:`style`        roles, colors and matplotlib settings
==================  ==================================================

The panel functions (:func:`plot_time_series`, :func:`plot_map`) draw onto an
``Axes`` they are given and return it; :func:`animate_map` draws onto one too
and returns the animation. Only :mod:`facet` creates a figure, returning it
with its axes, and nothing here calls ``show`` or ``savefig``: saving is the
job of the report that wanted the figure, in ``experiments/<task>/plots.py``,
which is also where anything that knows an experiment's name belongs. Nothing
here accepts a ``SIPNETResult``, a ``DataFrame`` or a path; turning those into
fields is the job of the module that owns the data:
:func:`sipnet_calibration.fields.to_model_output` for model output, and the
data-source modules' readers (``driver_fields``, ``constraint_fields``,
``site_labels_field``, ``initial_condition_fields``) for the rest.

Two conventions run across the package. Curves and bands keep ``NaN``, so gaps
in the data show as gaps in the figure, while scattered points drop it. Style
is resolved as the drawing function's default, then the role asked for, then
any keyword given explicitly.

Temporal aggregation is not done here. It is applied by the caller with
:func:`sipnet_calibration.observation.time_alignment.aggregate_time`, which
the observation operator also uses.

Interactive inspection of a single run is out of scope; ``pysipnet.viz``
covers it.
"""

from sipnet_calibration.plotting.facet import (
    build_plot_grid,
    plot_by_site,
    plot_by_variable,
    plot_map_by,
    plot_map_grid,
    plot_map_quantiles,
)
from sipnet_calibration.plotting.maps import (
    Cells,
    Points,
    Triangles,
    animate_map,
    plot_map,
    summarize_batch,
)
from sipnet_calibration.plotting.primitives import band, fan, line, points, spaghetti
from sipnet_calibration.plotting.series import plot_time_series
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
    "Cells",
    "Points",
    "Triangles",
    "animate_map",
    "CURVE_COLORS",
    "RC_PARAMS",
    "ROLES",
    "axis_label",
    "band",
    "build_plot_grid",
    "fan",
    "line",
    "plot_by_site",
    "plot_by_variable",
    "plot_map",
    "plot_map_by",
    "plot_map_grid",
    "plot_map_quantiles",
    "plot_time_series",
    "points",
    "role_style",
    "spaghetti",
    "summarize_batch",
    "use_project_style",
]
