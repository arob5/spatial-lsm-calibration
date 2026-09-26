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
:func:`sipnet_calibration.observation.time_alignment.aggregate_time`, which
takes the rule from the variable. The same function is used by the observation
operator, so a predictive check is drawn at the aggregation the likelihood
consumed.

Usage
-----
::

    from sipnet_calibration.drivers import driver_fields, load_drivers
    from sipnet_calibration.plotting import plot_time_series

    tair = driver_fields(load_drivers([1, 27]))["air_temperature"]

    plot_time_series(tair.sel(site=1))                     # quantile bands
    plot_time_series(tair.sel(site=1), show="spaghetti")   # driver members as curves

    # Observations, with their error variance, over a model panel.
    ax = plot_time_series(predicted_wood.sel(site=s))
    plot_time_series(observed_wood.sel(site=s), ax=ax, role="obs",
                     show="points", variance=wood_variance.sel(site=s))
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from sipnet_calibration.conventions import SITE, SPATIAL_DIM_NAMES, TIME
from sipnet_calibration.fields import batch_dims, validate_field
from sipnet_calibration.plotting import primitives
from sipnet_calibration.plotting.style import CURVE_COLORS, axis_label, role_style
from sipnet_calibration.validation import as_positive_integer

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

    ``time`` is the x axis and every batch dim is summarized, so the same
    call covers a single run and an ensemble, whatever its batch dims are
    named. With ``show="auto"``:

    ===================================  ====================================
    Dimensions of *data*                 what is drawn
    ===================================  ====================================
    ``(time,)``                          one curve
    ``(sample, time)``                   quantile bands over the samples
    ``(sample, driver_member, time)``    quantile bands over every curve of
                                         both batch dims at once
    ===================================  ====================================

    A ``site`` dim is refused rather than summarized: sites are not
    replicates of one another. Select one site first, or draw one panel per
    site with :func:`sipnet_calibration.plotting.facet.plot_by_site`.

    Parameters
    ----------
    data:
        A field (:func:`sipnet_calibration.fields.validate_field`) with a
        ``time`` dim and no spatial dim, whose ``units`` and ``long_name``
        become the y axis label; a scalar ``site`` coordinate is fine.
    ax:
        The axes to draw on. If ``None``, a figure and axes are created with
        ``matplotlib.pyplot.subplots``.
    show:
        One of :data:`SHOW_KINDS`. ``"auto"`` draws quantile bands when *data*
        has a sample dimension and a single curve when it does not. Asking for
        ``"line"`` or ``"points"`` when there is a sample dimension, or for
        ``"fan"`` or ``"spaghetti"`` when there is not, is an error rather
        than a silent reduction.
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
        The name of a coordinate on the batch dims, with
        ``show="spaghetti"``. Each curve is then labeled with that
        coordinate's value and colored from :data:`.style.CURVE_COLORS`
        instead of from the role, which is how a panel with a few labeled
        curves is made readable. *label* is ignored when this is given.
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
    TypeError
        If *data* is not a ``DataArray``.
    ValueError
        If *data* is not a field, has no ``time`` dimension, or has a spatial
        dim (a ``site`` dim among them); if *show* is
        not in :data:`SHOW_KINDS` or does not suit the dimensions; if
        *label_by* is given without ``show="spaghetti"`` or names a coordinate
        that is not on a batch dim; if both *variance* and
        *standard_deviation* are given, either is given without
        ``show="points"``, either does not align with *data*, or a variance is
        negative; or if *n_sigma* is not finite and positive.

    Notes
    -----
    Only batch dims are summarized, and the field contract refuses any dim
    that is not a batch dim, a spatial dim or ``time``: an array that stacks
    several variables along a ``variable`` dim would otherwise draw quantile
    bands across variables with different units -- a plausible-looking figure
    of nothing.

    ``show="fan"`` also draws the median as a curve, and the legend entry goes
    on that curve rather than on a band.

    Quantiles ignore missing values and are taken over all batch dims at once,
    so data with two is summarized over the whole set of curves rather than in
    two stages.
    """
    validate_field(data)
    _check_plottable(data)
    batch = batch_dims(data)
    show = _resolved_show(show, batch)
    _check_label_by(label_by, show, data, batch)
    yerr = _error_bar_lengths(data, variance, standard_deviation, n_sigma, show)

    if ax is None:
        _, ax = plt.subplots()
    if label is None:
        label = role
    x = (
        data.coords[TIME].values
        if TIME in data.coords
        else np.arange(data.sizes[TIME])
    )

    if show == "line":
        primitives.line(
            ax, x, data.values, label=label, **role_style(role, "line", **style)
        )
    elif show == "points":
        primitives.points(
            ax,
            x,
            data.values,
            yerr=yerr,
            label=label,
            **role_style(role, "points", **style),
        )
    else:
        curves = _curves_over_the_batch(data, batch)
        if show == "spaghetti" and label_by is not None:
            _draw_labeled_curves(
                ax, x, curves, data, batch, label_by, n_max, style
            )
        elif show == "spaghetti":
            primitives.spaghetti(
                ax,
                x,
                curves,
                n_max=n_max,
                label=label,
                **role_style(role, "line", **style),
            )
        else:
            primitives.fan(
                ax, x, curves, levels=levels, **role_style(role, "band", **style)
            )
            primitives.line(
                ax,
                x,
                primitives.nanquantile(curves, 0.5),
                label=label,
                **role_style(role, "line", **style),
            )

    ax.set_ylabel(axis_label(data))
    return ax


# ── supporting helpers ────────────────────────────────────────────────────────


def _resolved_show(show: str, batch: tuple[str, ...]) -> str:
    """*show* with ``"auto"`` resolved, raising if it does not suit the data."""
    if show not in SHOW_KINDS:
        raise ValueError(f"show must be one of {list(SHOW_KINDS)}, got {show!r}")
    if show == "auto":
        return "fan" if batch else "line"
    if show in ("fan", "spaghetti") and not batch:
        raise ValueError(
            f"show={show!r} summarizes several curves, but the data has only "
            f"{TIME!r}. Use show='line' or show='points'."
        )
    if show in ("line", "points") and batch:
        raise ValueError(
            f"show={show!r} draws one curve, but the data also has "
            f"{list(batch)}. Select or reduce first, or use show='fan' "
            "or show='spaghetti'."
        )
    return show


def _curves_over_the_batch(
    data: xr.DataArray, batch: tuple[str, ...]
) -> np.ndarray:
    """*data* as ``(n_curves, n_time)``, the batch dims flattened together."""
    ordered = data.transpose(*batch, TIME)
    return ordered.values.reshape(-1, ordered.sizes[TIME])


def _curve_labels(
    data: xr.DataArray, batch: tuple[str, ...], label_by: str
) -> list[str]:
    """One label per stacked curve, from the *label_by* coordinate's values.

    The order matches :func:`_curves_over_the_batch`, which flattens the batch dims
    in the order they are given.
    """
    coordinate = data.coords[label_by]
    sizes = tuple(data.sizes[dim] for dim in batch)
    labels = []
    for position in np.ndindex(*sizes):
        chosen = {
            dim: index
            for dim, index in zip(batch, position)
            if dim in coordinate.dims
        }
        labels.append(f"{label_by} {coordinate.isel(chosen).values}")
    return labels


def _draw_labeled_curves(
    ax, x, curves, data, batch, label_by, n_max, style
):
    """Draw each curve in its own color, labeled by a coordinate's value."""
    labels = _curve_labels(data, batch, label_by)
    chosen = primitives.thinned_indices(
        len(curves), as_positive_integer(n_max, message_name="n_max")
    )
    for position, index in enumerate(chosen):
        keywords = {
            "color": CURVE_COLORS[position % len(CURVE_COLORS)],
            **style,
        }
        primitives.line(ax, x, curves[index], label=labels[index], **keywords)


def _error_bar_lengths(
    data: xr.DataArray,
    variance: xr.DataArray | None,
    standard_deviation: xr.DataArray | None,
    n_sigma: float,
    show: str,
) -> np.ndarray | None:
    """Half-length of each error bar, or ``None`` when no error was given."""
    given = [
        (name, array)
        for name, array in (
            ("variance", variance),
            ("standard_deviation", standard_deviation),
        )
        if array is not None
    ]
    if not given:
        return None
    if len(given) == 2:
        raise ValueError(
            "pass variance or standard_deviation, not both; they are two ways "
            "of stating the same error"
        )
    if show != "points":
        raise ValueError(
            f"error bars are drawn by show='points', not show={show!r}"
        )
    if not (np.isfinite(n_sigma) and n_sigma > 0):
        raise ValueError(f"n_sigma must be finite and positive, got {n_sigma!r}")

    name, error = given[0]
    _check_aligned(data, error, name)
    # show == "points" here, so the data has no sample dimension and both
    # arrays are one-dimensional over time; no reordering is possible.
    values = error.values
    if name == "variance":
        if np.any(values[np.isfinite(values)] < 0):
            raise ValueError("a variance cannot be negative")
        values = np.sqrt(values)
    return n_sigma * values


# ── checks ────────────────────────────────────────────────────────────────────


def _check_plottable(data: xr.DataArray) -> None:
    """Raise unless the field has ``time`` and no spatial dim, which a series needs."""
    if TIME not in data.dims:
        raise ValueError(
            f"the array has dimensions {list(data.dims)} and needs {TIME!r} "
            "to be plotted against time. Select or aggregate first, or use a "
            "spatial plot."
        )
    spatial = [dim for dim in data.dims if dim in SPATIAL_DIM_NAMES]
    if spatial:
        advice = (
            f"select one site with .sel({SITE}=...), or draw one panel per site with "
            "facet.plot_by_site(field)"
            if SITE in spatial
            else "select one location first"
        )
        raise ValueError(
            f"the array has the spatial dim(s) {spatial}, which a time series does not "
            f"summarize: sites are not replicates of one another. {advice[0].upper()}"
            f"{advice[1:]}."
        )


def _check_label_by(
    label_by: str | None,
    show: str,
    data: xr.DataArray,
    batch: tuple[str, ...],
) -> None:
    """Raise unless *label_by* names a coordinate that can label the curves."""
    if label_by is None:
        return
    if show != "spaghetti":
        raise ValueError(
            f"label_by labels individual curves and needs show='spaghetti', "
            f"not show={show!r}"
        )
    if label_by not in data.coords:
        raise ValueError(
            f"label_by={label_by!r} is not a coordinate; the coordinates are "
            f"{sorted(data.coords)}"
        )
    dims = data.coords[label_by].dims
    if not dims or not set(dims) <= set(batch):
        raise ValueError(
            f"label_by={label_by!r} is on {list(dims)}, which is not among the "
            f"batch dims {list(batch)}; it cannot name a curve"
        )


def _check_aligned(data: xr.DataArray, error: xr.DataArray, name: str) -> None:
    """Raise unless the error array covers exactly the same points as *data*."""
    if not isinstance(error, xr.DataArray):
        raise ValueError(f"{name} must be an xarray.DataArray")
    if set(error.dims) != set(data.dims):
        raise ValueError(
            f"{name} has dimensions {list(error.dims)} and the data has "
            f"{list(data.dims)}; they must match"
        )
    try:
        xr.align(data, error, join="exact")
    except ValueError as mismatch:
        raise ValueError(
            f"{name} is not aligned with the data: {mismatch}"
        ) from mismatch
