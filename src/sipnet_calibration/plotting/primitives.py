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

import warnings
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

from sipnet_calibration.plotting.style import BAND_ALPHAS

__all__ = [
    "band",
    "fan",
    "line",
    "nanquantile",
    "points",
    "spaghetti",
    "thinned_indices",
]


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
    x, y = np.asarray(x), np.asarray(y)
    _check_same_length(x=x, y=y)
    (drawn,) = ax.plot(x, y, **style)
    return drawn


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
    x, samples = np.asarray(x), np.asarray(samples)
    _check_samples(x, samples)
    if not isinstance(n_max, (int, np.integer)) or int(n_max) < 1:
        raise ValueError(f"n_max must be a positive integer, got {n_max!r}")

    label = style.pop("label", None)
    drawn = []
    for position, index in enumerate(thinned_indices(len(samples), int(n_max))):
        keep_label = position == 0 and label is not None
        drawn.append(
            line(
                ax,
                x,
                samples[index],
                label=label if keep_label else "_nolegend_",
                **style,
            )
        )
    return drawn


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
    x, lower, upper = np.asarray(x), np.asarray(lower), np.asarray(upper)
    _check_same_length(x=x, lower=lower, upper=upper)
    return ax.fill_between(x, lower, upper, **style)


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
    x, samples = np.asarray(x), np.asarray(samples)
    _check_samples(x, samples)
    levels = _checked_levels(levels)

    widest_first = sorted(levels, reverse=True)
    given_alpha = style.pop("alpha", None)
    alphas = (
        np.full(len(widest_first), given_alpha)
        if given_alpha is not None
        else np.linspace(BAND_ALPHAS[0], BAND_ALPHAS[1], len(widest_first))
    )
    label = style.pop("label", None)

    wanted = []
    for level in widest_first:
        tail = (1.0 - level) / 2.0
        wanted.extend((tail, 1.0 - tail))
    limits = nanquantile(samples, wanted)

    drawn = []
    for position, alpha in enumerate(alphas):
        is_narrowest = position == len(widest_first) - 1
        keep_label = is_narrowest and label is not None
        drawn.append(
            band(
                ax,
                x,
                limits[2 * position],
                limits[2 * position + 1],
                alpha=alpha,
                label=label if keep_label else "_nolegend_",
                **style,
            )
        )
    return drawn


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
    x, y = np.asarray(x), np.asarray(y)
    arrays = {"x": x, "y": y}
    if yerr is not None:
        yerr = np.asarray(yerr)
        arrays["yerr"] = yerr
    _check_same_length(**arrays)

    keep = _is_finite(x) & _is_finite(y)
    if yerr is not None:
        keep = keep & _is_finite(yerr)
    return ax.errorbar(
        x[keep], y[keep], yerr=None if yerr is None else yerr[keep], **style
    )


# ── shared utilities ────────────────────────────────────────────────────────


def nanquantile(samples: np.ndarray, quantiles) -> np.ndarray:
    """Quantiles along the first axis, without the all-missing-slice warning.

    A column in which every sample is missing yields ``NaN``, which is what
    the caller wants; ``numpy`` announces it once per quantile per such
    column, which the caller does not need.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        return np.nanquantile(samples, quantiles, axis=0)


def thinned_indices(n_samples: int, n_max: int) -> np.ndarray:
    """Indices of the samples to draw: all of them, or *n_max* evenly spaced.

    When thinning happens the spacing is at least one, so the indices are
    distinct; they run from the first sample to the last, which for
    ``n_max`` of one means the first alone.
    """
    if n_samples <= n_max:
        return np.arange(n_samples)
    return np.linspace(0, n_samples - 1, n_max).round().astype(int)


# ── supporting helpers ────────────────────────────────────────────────────────


def _is_finite(values: np.ndarray) -> np.ndarray:
    """Which entries are usable, for datetimes as well as for numbers."""
    if np.issubdtype(values.dtype, np.datetime64):
        return ~np.isnat(values)
    if values.dtype.kind in "iub":
        return np.ones(values.shape, dtype=bool)
    return np.isfinite(values)


def _check_same_length(**arrays: np.ndarray) -> None:
    """Raise unless every array is one-dimensional and they share a length."""
    wrong = {name: a.ndim for name, a in arrays.items() if a.ndim != 1}
    if wrong:
        detail = ", ".join(f"{n} has {d} dimensions" for n, d in wrong.items())
        raise ValueError(f"expected one-dimensional arrays; {detail}")
    lengths = {name: a.size for name, a in arrays.items()}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{n} of {s}" for n, s in lengths.items())
        raise ValueError(f"expected arrays of the same length; got {detail}")


def _check_samples(x: np.ndarray, samples: np.ndarray) -> None:
    """Raise unless *samples* is shaped ``(n_samples, len(x))``."""
    if x.ndim != 1:
        raise ValueError(f"x must be one-dimensional; it has {x.ndim} dimensions")
    if samples.ndim != 2:
        raise ValueError(
            f"samples must be two-dimensional, (n_samples, {x.size}); it has "
            f"{samples.ndim}. A single series is drawn by line()."
        )
    if samples.shape[1] != x.size:
        raise ValueError(
            f"each sample has {samples.shape[1]} values and x has {x.size}; "
            "they must match"
        )


def _checked_levels(levels) -> tuple[float, ...]:
    """*levels* as a tuple, raising unless they are widths within ``(0, 1)``."""
    if isinstance(levels, (int, float)):
        raise ValueError(
            f"levels must be a sequence of interval widths, got the single "
            f"number {levels!r}; pass ({levels},) to draw one band"
        )
    levels = tuple(float(level) for level in levels)
    if not levels:
        raise ValueError("levels must name at least one interval width")
    if len(set(levels)) != len(levels):
        raise ValueError(f"levels must not repeat a width; got {levels}")
    outside = [level for level in levels if not 0.0 < level < 1.0]
    if outside:
        raise ValueError(f"every level must lie within (0, 1); got {outside}")
    return levels
