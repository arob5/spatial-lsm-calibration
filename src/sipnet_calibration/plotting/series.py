"""Time-series panels over the project's canonical fields.

:func:`series_panel` draws one field against time onto one ``Axes``. It covers
a single deterministic run, an ensemble drawn as a fan or as individual
curves, and scattered observations with error bars, according to the shape of
the field it is given and the role it is asked for.

The panel takes an ``Axes`` and returns it. It sets no title and never saves
or shows a figure.

How the shape of a field is read
--------------------------------
``time`` is the x-axis, and every other dimension present is a sample
dimension. ``show="auto"`` draws a fan when there is at least one sample
dimension and a single curve when there is none:

==========================  ================================================
Dimensions of the field     ``show="auto"`` draws
==========================  ================================================
``(time,)``                 one curve
``(member, time)``          a fan over members
``(site, time)``            a fan over sites
``(member, site, time)``    a fan over the member-site curves together
==========================  ================================================

To draw a fan over members for one site, select the site first. To draw one
panel per site, use :func:`sipnet_calibration.plotting.facet.by_site`.

Drawing several things on one panel
-----------------------------------
Every panel takes ``ax`` and returns it, so an overlay is a second call::

    ax = series_panel(predicted, role="posterior")
    series_panel(observed, ax=ax, role="obs")

That covers observations, a truth line, and several aggregations of the same
variable on one panel. An observation ensemble is drawn the same way; only the
role differs. An observation error given as a variance or a standard deviation
is a second field on the same grid, passed as ``variance=`` or
``standard_deviation=``.

Aggregating first
-----------------
Aggregation is applied by the caller, with
:func:`sipnet_calibration.obs_ops.aggregate_time`, which takes the rule from
the variable::

    series_panel(aggregate_time(nee, "1D"))

Note that the order matters: the quantiles of a daily sum are not the daily
sum of the quantiles, and which is wanted is a modeling choice.

Usage
-----
::

    from sipnet_calibration.drivers import driver_fields, load_drivers
    from sipnet_calibration.plotting import series_panel

    tair = driver_fields(load_drivers([1, 27]))["air_temperature"]

    series_panel(tair.sel(site=1))                        # a fan over members
    series_panel(tair.sel(site=1), show="spaghetti")      # the members as curves
    series_panel(tair.isel(member=0), show="spaghetti", label_by="site")

    # An observation with its error variance, over a model panel.
    ax = series_panel(predicted_wood.sel(site=s))
    series_panel(observed_wood.sel(site=s), ax=ax, role="obs", show="points",
                 variance=wood_variance.sel(site=s))
"""

from __future__ import annotations

from typing import Any

import xarray as xr
from matplotlib.axes import Axes

__all__ = ["SHOW_KINDS", "series_panel"]

#: What ``show`` may be. ``"auto"`` is resolved from the field's dimensions.
SHOW_KINDS: tuple[str, ...] = ("auto", "line", "spaghetti", "fan", "points")


def series_panel(
    field: xr.DataArray,
    ax: Axes | None = None,
    *,
    show: str = "auto",
    role: str = "posterior",
    levels: tuple[float, ...] = (0.5, 0.9),
    n_max: int = 25,
    label: str | None = None,
    label_by: str | None = None,
    variance: xr.DataArray | None = None,
    standard_deviation: xr.DataArray | None = None,
    n_sigma: float = 1.0,
    **style: Any,
) -> Axes:
    """Draw *field* against time on one ``Axes``.

    Parameters
    ----------
    field:
        A canonical field whose dimensions are a subset of
        ``(member, site, time)`` and include ``time``. Every dimension other
        than ``time`` is a sample dimension.
    ax:
        The axes to draw on. If ``None``, a figure and axes are created with
        ``matplotlib.pyplot.subplots``.
    show:
        One of :data:`SHOW_KINDS`. ``"auto"`` draws a fan when the field has a
        sample dimension and a single curve when it does not. Asking for
        ``"line"`` on a field that has a sample dimension, or for ``"fan"`` or
        ``"spaghetti"`` on one that does not, is an error.
    role:
        A key of :data:`.style.ROLES`, deciding color, line style and marker.
    levels:
        Interval widths for ``show="fan"``; see :func:`.primitives.fan`.
    n_max:
        The most curves ``show="spaghetti"`` draws; see
        :func:`.primitives.spaghetti`.
    label:
        The legend entry. ``None`` uses the value of *role*. Pass
        ``"_nolegend_"`` for no entry.
    label_by:
        The name of a coordinate on the sample dimension, with
        ``show="spaghetti"``. Each curve is then labeled with that
        coordinate's value and colored from :data:`.style.CURVE_COLORS`
        instead of from the role. *label* is ignored when this is given.
    variance, standard_deviation:
        The observation error, as a field aligned with *field*, with
        ``show="points"``. At most one of the two may be given.
    n_sigma:
        Multiplies the standard deviation to give each error bar's
        half-length.
    **style:
        Passed through to the drawing function, overriding the role's
        keywords.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on, which is *ax* itself when it was given. Its y label
        is set from :func:`.style.axis_label`; the x axis and the title are
        left alone.

    Raises
    ------
    ValueError
        If *field* is not a ``DataArray``, or its dimensions are not a subset
        of ``(member, site, time)`` or do not include ``time``; if *show* is
        not in :data:`SHOW_KINDS` or does not suit the field's dimensions; if
        *label_by* is given without ``show="spaghetti"`` or names a coordinate
        that is not on a sample dimension; if both *variance* and
        *standard_deviation* are given, either is given without
        ``show="points"``, either does not align with *field*, or a variance
        is negative; or if *n_sigma* is not finite and positive.

    Notes
    -----
    ``show="fan"`` also draws the median as a curve, and the legend entry goes
    on that curve rather than on a band.

    Quantiles ignore missing values and are taken over all sample dimensions
    at once, so a field with both ``member`` and ``site`` is summarized over
    the whole set of curves rather than in two stages.
    """
    raise NotImplementedError
