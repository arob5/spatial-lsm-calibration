"""L1 -- axes-level primitives.

Overview
--------
Every function here has the signature ``(ax, <plain numpy>, **style) ->
artist``. No pandas, no xarray, no figure creation, no knowledge of roles or
of the variable registry. That is what makes them reusable everywhere and
trivial to test: a primitive's whole contract is the artist it puts on the
axes and the data that artist holds.

Provided, for time series:

==================  =========================================================
:func:`line`        one curve
:func:`spaghetti`   an ensemble as individual curves, decimated to ``n_max``
:func:`band`        one filled interval between explicit bounds
:func:`fan`         nested central intervals computed from samples
:func:`points`      scattered observations, optionally with error bars
==================  =========================================================

The spatial primitives the design calls for -- ``map_points``,
``map_raster`` and ``basemap`` -- are not here yet. They are blocked on the
projection decision (issue #4) and belong with :mod:`.maps`.

Missing values
--------------
The rule across this module is that **lines and bands keep** ``NaN`` **so the
gap shows, and points drop it so the artist holds exactly what was
observed.** A curve is a claim about what happened between its vertices, and
interpolating across a gap in observations is a false claim;
:func:`line` and :func:`spaghetti` therefore pass ``NaN`` through, and
matplotlib breaks the curve there. :func:`band` and :func:`fan` do the same:
``fill_between`` emits one path per contiguous run of finite bounds, so a
band does not bridge a gap either. :func:`points`, by contrast, draws a set
of observations rather than a curve, where a ``NaN`` is an absence and not a
gap, so it drops the non-finite entries.

Notes
-----
:func:`fan` takes *symmetric interval levels* -- ``(0.5, 0.9)`` -- rather than
four raw quantiles: it generalizes to any number of bands, reads correctly in
a legend, and cannot be handed a non-monotone tuple.

:func:`spaghetti` takes ``n_max`` and decimates. A single site is 38,000
timesteps by 25 members; :func:`fan` collapses the ensemble through quantiles
but :func:`spaghetti` does not, and the guardrail belongs here rather than at
every call site.

Usage
-----
::

    from sipnet_calibration.plotting import primitives as p

    p.line(ax, time, values, color="k")
    p.fan(ax, time, ensemble, levels=(0.5, 0.9), color="#0072B2")
    p.points(ax, years, observed, yerr=standard_errors, marker="o")
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
        The axes to draw on. Never created here.
    x:
        One-dimensional, length ``n``. May be ``datetime64``, which matplotlib
        converts to its own date numbers; a caller asserting on the artist
        reads them back with ``matplotlib.dates.num2date``.
    y:
        One-dimensional, length ``n``. ``NaN`` is kept and breaks the curve.
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
        Two-dimensional, ``(n_samples, n)``. ``NaN`` is kept and breaks a
        curve, as in :func:`line`.
    n_max:
        The most curves to draw. If ``n_samples`` exceeds it, the curves drawn
        are at ``numpy.linspace(0, n_samples - 1, n_max)`` rounded to integers
        -- evenly spaced, deterministic, and including the first and the last.
        The default matches the 25-member observation ensemble, so that
        ensemble is drawn whole.
    **style:
        Passed to ``Axes.plot`` for every curve, so they share one color. Only
        the first curve keeps a ``label``; the rest are given
        ``"_nolegend_"``, so an ensemble contributes one legend entry rather
        than ``n_max`` of them.

    Returns
    -------
    list of matplotlib.lines.Line2D
        In the order drawn, one per curve actually drawn.

    Raises
    ------
    ValueError
        If *samples* is not two-dimensional, its second axis differs in length
        from *x*, or *n_max* is not a positive integer.

    Notes
    -----
    Decimation changes what the figure shows, which is why it is explicit and
    documented rather than a silent guard. A caller that must see every member
    passes ``n_max=n_samples``; a caller that wants the ensemble summarized
    rather than sampled wants :func:`fan`.
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
        One-dimensional, length ``n``. Where either is not finite the band is
        interrupted: ``fill_between`` emits one path per contiguous run, so a
        gap in the data is a gap in the band rather than a straight edge
        drawn across it.
    **style:
        Passed to ``Axes.fill_between``. Callers that want a fan should use
        :func:`fan`, which manages the opacity ramp.

    Returns
    -------
    matplotlib.collections.PolyCollection
        The filled region, whose ``get_paths()`` has one entry per contiguous
        run.

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
    """Draw nested central intervals of *samples* as translucent bands.

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
        Central interval widths, not raw quantiles: ``0.5`` is the band from
        the 25th to the 75th percentile and ``0.9`` the band from the 5th to
        the 95th. Any number of levels is allowed. Each must lie in
        ``(0, 1)``.
    **style:
        Passed to :func:`band` for every band, so they share one color.
        ``alpha`` is set here and an explicit ``alpha`` overrides the whole
        ramp for every band, which is rarely what is wanted.

    Returns
    -------
    list of matplotlib.collections.PolyCollection
        One per level, **widest first**, so that narrower bands layer on top.
        Opacity is spaced linearly across :data:`.style.BAND_ALPHAS` from the
        widest band to the narrowest, so a more probable interval reads as
        more solid. Only the narrowest band keeps a ``label``, so a fan
        contributes one legend entry.

    Raises
    ------
    ValueError
        If *samples* is not two-dimensional or its second axis differs in
        length from *x*; if *levels* is empty, holds a duplicate, or holds a
        value outside ``(0, 1)``.

    Notes
    -----
    A column of *samples* that is entirely ``NaN`` -- every member missing at
    that timestep -- yields ``NaN`` bounds and therefore a gap, which is the
    intended reading. ``numpy.nanquantile`` warns once per quantile per such
    column while doing so, and this function suppresses that warning around
    the quantile call only. It is not information: with NEE about 55 percent
    missing over site and time, a single panel would otherwise emit tens of
    thousands of identical warnings and bury anything real.
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
    """Draw scattered observations, with error bars where *yerr* is given.

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
        each bar, in the units of *y*; a caller working from a variance
        converts and scales before calling. Non-finite entries are dropped
        with their point.
    **style:
        Passed to ``Axes.errorbar``.

    Returns
    -------
    matplotlib.container.ErrorbarContainer
        The container, whatever *yerr* was: ``errorbar`` returns one even with
        no error bars, so the return type does not depend on the arguments.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional and the same length.

    Notes
    -----
    Entries where *x*, *y* or *yerr* is not finite are **dropped**, together,
    before drawing. Points are a set of observations rather than a curve: a
    missing entry is an absence, not a gap to be shown. Dropping also makes
    the artist's data exactly the observed set, which is what the annual
    constraints -- ragged over site, year and variable -- need.
    """
    raise NotImplementedError
