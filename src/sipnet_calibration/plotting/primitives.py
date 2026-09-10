"""Low-level plotting primitives.

The base plotting helpers used throughout this project, responsible for adding
elements to existing matplotlib ``Axes`` objects. The supported elements are
the ones that come up repeatedly when visualizing land surface modeling data
and results: a single time series, an ensemble of them drawn as separate
curves, an ensemble summarized as nested intervals, and observations with
their error bars.

Each function takes an ``Axes`` and plain numpy arrays, adds one element to
it, and returns what it added. They accept no pandas and no xarray, they
create no figure, and they know nothing about roles or about variables. Style
keywords are passed straight through to matplotlib.

====================  ==============================================
:func:`line`          one curve
:func:`spaghetti`     an ensemble as individual curves
:func:`band`          one filled interval between explicit bounds
:func:`fan`           nested central intervals computed from samples
:func:`points`        scattered values, with optional error bars
====================  ==============================================

Missing values
--------------
Curves and bands keep ``NaN``, so a gap in the data is a gap in the drawing:
:func:`line` and :func:`spaghetti` break the curve there, and :func:`band` and
:func:`fan` draw one region per run of finite values rather than spanning the
gap. :func:`points` instead drops the entries that are not finite, so what is
drawn is exactly the values that were present.

Usage
-----
::

    from sipnet_calibration.plotting import fan, line, points

    line(ax, time, values, color="k")
    fan(ax, time, ensemble, levels=(0.5, 0.9), color="#0072B2")
    points(ax, years, observed, yerr=standard_errors, marker="o")
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

__all__ = ["band", "fan", "line", "points", "spaghetti"]


def line(ax: Axes, x: np.ndarray, y: np.ndarray, **style: Any) -> Line2D:
    """Draw one curve on *ax*.

    Parameters
    ----------
    ax:
        The axes to draw on.
    x:
        One-dimensional, length ``n``. May be ``datetime64``, which matplotlib
        converts to its own date numbers; read them back with
        ``matplotlib.dates.num2date``.
    y:
        One-dimensional, length ``n``. ``NaN`` breaks the curve.
    **style:
        Passed to ``Axes.plot``.

    Returns
    -------
    matplotlib.lines.Line2D
        The curve.

    Raises
    ------
    ValueError
        If *x* or *y* is not one-dimensional, or they differ in length.
    """
    raise NotImplementedError


def spaghetti(
    ax: Axes, x: np.ndarray, samples: np.ndarray, *, n_max: int = 25, **style: Any
) -> list[Line2D]:
    """Draw an ensemble as individual curves, at most *n_max* of them.

    Parameters
    ----------
    ax:
        The axes to draw on.
    x:
        One-dimensional, length ``n``.
    samples:
        Two-dimensional, ``(n_samples, n)``. ``NaN`` breaks a curve.
    n_max:
        The most curves to draw. If ``n_samples`` exceeds it, the curves drawn
        are the ``n_max`` evenly spaced samples, the first and the last
        included. Pass ``n_samples`` to draw them all.
    **style:
        Passed to ``Axes.plot`` for every curve, so they share one color. Only
        the first curve keeps a ``label``; the others are given
        ``"_nolegend_"``, so an ensemble contributes one legend entry.

    Returns
    -------
    list of matplotlib.lines.Line2D
        One per curve drawn, in the order drawn.

    Raises
    ------
    ValueError
        If *samples* is not two-dimensional, its second axis differs in length
        from *x*, or *n_max* is not a positive integer.

    Notes
    -----
    Decimation changes what the figure shows. To summarize a large ensemble
    rather than sample it, use :func:`fan`.
    """
    raise NotImplementedError


def band(
    ax: Axes, x: np.ndarray, lower: np.ndarray, upper: np.ndarray, **style: Any
) -> PolyCollection:
    """Fill the interval between *lower* and *upper* over *x*.

    Parameters
    ----------
    ax:
        The axes to draw on.
    x:
        One-dimensional, length ``n``.
    lower, upper:
        One-dimensional, length ``n``. Where either is not finite the fill is
        interrupted rather than spanning the gap.
    **style:
        Passed to ``Axes.fill_between``.

    Returns
    -------
    matplotlib.collections.PolyCollection
        The filled region, with one path per run of finite bounds.

    Raises
    ------
    ValueError
        If the three arrays are not one-dimensional and the same length.
    """
    raise NotImplementedError


def fan(
    ax: Axes,
    x: np.ndarray,
    samples: np.ndarray,
    *,
    levels: tuple[float, ...] = (0.5, 0.9),
    **style: Any,
) -> list[PolyCollection]:
    """Draw central intervals of *samples* as nested translucent bands.

    Parameters
    ----------
    ax:
        The axes to draw on.
    x:
        One-dimensional, length ``n``.
    samples:
        Two-dimensional, ``(n_samples, n)``. Quantiles are taken along the
        first axis, ignoring ``NaN``.
    levels:
        Interval widths rather than raw quantiles: ``0.5`` is the band from
        the 25th to the 75th percentile, ``0.9`` the band from the 5th to the
        95th. Any number of levels is allowed, each within ``(0, 1)``.
    **style:
        Passed to :func:`band` for every band, so they share one color.
        ``alpha`` is set per band from :data:`.style.BAND_ALPHAS`; passing it
        explicitly gives every band the same opacity.

    Returns
    -------
    list of matplotlib.collections.PolyCollection
        One per level, widest first, so narrower bands are drawn on top. Only
        the narrowest band keeps a ``label``, so a fan contributes one legend
        entry.

    Raises
    ------
    ValueError
        If *samples* is not two-dimensional or its second axis differs in
        length from *x*; or if *levels* is empty, holds a duplicate, or holds
        a value outside ``(0, 1)``.

    Notes
    -----
    A column of *samples* in which every value is missing produces a gap in
    the bands, and no warning: the ``All-NaN slice encountered`` warning
    numpy would otherwise raise per quantile per such column is suppressed
    around the quantile calculation.
    """
    raise NotImplementedError


def points(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    yerr: np.ndarray | None = None,
    **style: Any,
) -> ErrorbarContainer:
    """Draw scattered values, with error bars where *yerr* is given.

    Entries in which *x*, *y* or *yerr* is not finite are dropped together
    before drawing.

    Parameters
    ----------
    ax:
        The axes to draw on.
    x:
        One-dimensional, length ``n``.
    y:
        One-dimensional, length ``n``.
    yerr:
        One-dimensional, length ``n``, or ``None``. The **half-length** of
        each bar, in the units of *y*.
    **style:
        Passed to ``Axes.errorbar``.

    Returns
    -------
    matplotlib.container.ErrorbarContainer
        The container, whether or not *yerr* was given.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional and the same length.
    """
    raise NotImplementedError
