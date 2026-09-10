"""Time series plots.

Nearly everything this project produces is indexed by time: SIPNET output, the
meteorological drivers that force it, and the observations it is calibrated
against. This module plots them, whether the quantity is a single run, an
ensemble, or a set of observations carrying error estimates.

:func:`plot_time_series` is the one function. It draws onto an ``Axes`` it is
given and returns it, so several quantities can be layered on one panel by
calling it repeatedly, and a grid of panels is built by
:mod:`sipnet_calibration.plotting.facet`. The elements it draws come from
:mod:`sipnet_calibration.plotting.primitives`, and its colors and line styles
from :mod:`sipnet_calibration.plotting.style`.

Temporal aggregation is applied by the caller before plotting, using
:func:`sipnet_calibration.obs_ops.aggregate_time`, which takes the rule from
the variable. The same function is used by the observation operator, so a
predictive check is drawn at the aggregation the likelihood consumed.

Usage
-----
::

    from sipnet_calibration.drivers import driver_fields, load_drivers
    from sipnet_calibration.plotting import plot_time_series

    tair = driver_fields(load_drivers([1, 27]))["air_temperature"]

    plot_time_series(tair.sel(site=1))                     # quantile bands
    plot_time_series(tair.sel(site=1), show="spaghetti")   # members as curves

    # Observations, with their error variance, over a model panel.
    ax = plot_time_series(predicted_wood.sel(site=s))
    plot_time_series(observed_wood.sel(site=s), ax=ax, role="obs",
                     show="points", variance=wood_variance.sel(site=s))
"""

from __future__ import annotations

from typing import Any

import xarray as xr
from matplotlib.axes import Axes

__all__ = ["SHOW_KINDS", "plot_time_series"]

#: What ``show`` may be. ``"auto"`` is resolved from the data's dimensions.
SHOW_KINDS: tuple[str, ...] = ("auto", "line", "spaghetti", "fan", "points")


def plot_time_series(
    data: xr.DataArray,
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
    """Plot *data* against time on one ``Axes``.

    ``time`` is the x axis and every other dimension is treated as a sample
    dimension, so the same call covers a single run and an ensemble, and an
    ensemble over sites is drawn the way an ensemble over members is. With
    ``show="auto"``:

    ========================  =============================================
    Dimensions of *data*      what is drawn
    ========================  =============================================
    ``(time,)``               one curve
    ``(member, time)``        quantile bands over the members
    ``(site, time)``          quantile bands over the sites
    ``(member, site, time)``  quantile bands over the member-site curves
    ========================  =============================================

    To summarize members at one site, select the site first. To draw one panel
    per site, use :func:`sipnet_calibration.plotting.facet.plot_by_site`.

    Parameters
    ----------
    data:
        An ``xarray.DataArray`` holding one variable, which must satisfy:

        * ``time`` is one of its dimensions;
        * its other dimension names, if any, are among ``member`` and
          ``site``;
        * ``attrs`` carries ``units`` and ``long_name``, which become the y
          axis label.

        The readers in this project produce arrays that satisfy this;
        see :mod:`sipnet_calibration.fields` for the wider convention they
        follow, of which this function uses only the three points above.
    ax:
        The axes to draw on. If ``None``, a figure and axes are created with
        ``matplotlib.pyplot.subplots``.
    show:
        One of :data:`SHOW_KINDS`. ``"auto"`` draws quantile bands when *data*
        has a sample dimension and a single curve when it does not. Asking for
        ``"line"`` when there is a sample dimension, or for ``"fan"`` or
        ``"spaghetti"`` when there is not, is an error rather than a silent
        reduction.
    role:
        A key of :data:`.style.ROLES`, deciding color, line style and marker.
    levels:
        Widths of the quantile bands for ``show="fan"``; see
        :func:`.primitives.fan`.
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
        instead of from the role, which is how a panel with one curve per site
        is made readable. *label* is ignored when this is given.
    variance, standard_deviation:
        The observation error, as a ``DataArray`` aligned with *data*, with
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
        left alone, a panel title being the grid's to set.

    Raises
    ------
    ValueError
        If *data* is not a ``DataArray``, has no ``time`` dimension, or has a
        dimension other than ``member`` and ``site`` beside it; if *show* is
        not in :data:`SHOW_KINDS` or does not suit the dimensions; if
        *label_by* is given without ``show="spaghetti"`` or names a coordinate
        that is not on a sample dimension; if both *variance* and
        *standard_deviation* are given, either is given without
        ``show="points"``, either does not align with *data*, or a variance is
        negative; or if *n_sigma* is not finite and positive.

    Notes
    -----
    Restricting the dimension names to ``member``, ``site`` and ``time`` is a
    guard rather than a requirement: the sample-dimension rule would work on
    any name. It is checked because a further dimension is usually a mistake.
    Passing the stored form of the annual constraints, which carries a
    ``variable`` dimension, would otherwise draw quantile bands across four
    variables with four different units -- a plausible-looking figure of
    nothing.

    ``show="fan"`` also draws the median as a curve, and the legend entry goes
    on that curve rather than on a band.

    Quantiles ignore missing values and are taken over all sample dimensions
    at once, so data with both ``member`` and ``site`` is summarized over the
    whole set of curves rather than in two stages.
    """
    raise NotImplementedError
