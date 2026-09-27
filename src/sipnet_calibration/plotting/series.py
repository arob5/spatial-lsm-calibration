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

    import matplotlib.pyplot as plt

    from sipnet_calibration.drivers import driver_fields, load_drivers
    from sipnet_calibration.plotting import plot_time_series

    tair = driver_fields(load_drivers([1, 27]))["air_temperature"]

    figure, ax = plt.subplots()
    plot_time_series(tair.sel(site=1), ax)                     # quantile bands
    plot_time_series(tair.sel(site=1), ax, show="spaghetti")   # driver members as curves

    # Observations, with their error variance, over a model panel.
    plot_time_series(predicted_wood.sel(site=s), ax)
    plot_time_series(observed_wood.sel(site=s), ax, role="observation",
                     show="points", variance=wood_variance.sel(site=s))
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from sipnet_calibration.conventions import SITE, SPATIAL_DIM_NAMES, TIME
from sipnet_calibration.fields import batch_dims, message_name, validate_field
from sipnet_calibration.plotting import primitives
from sipnet_calibration.plotting.style import (
    CURVE_COLORS,
    axis_label,
    check_key_is_known,
    check_number_is_finite,
    check_number_is_positive,
    check_option_is_known,
    check_value_is_a_number,
    role_style,
)
from sipnet_calibration.validation import as_positive_integer

__all__ = ["SHOW_OPTIONS", "plot_time_series"]

#: What ``show`` may be. ``"auto"`` is resolved from the field's dimensions.
SHOW_OPTIONS: tuple[str, ...] = ("auto", "line", "spaghetti", "fan", "points")


def plot_time_series(
    field: xr.DataArray,
    ax: Axes,
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
    """Plot *field* against time on one ``Axes``.

    ``time`` is the x axis and every batch dim is summarized, so the same
    call covers a single run and an ensemble, whatever its batch dims are
    named. With ``show="auto"``:

    ===================================  ====================================
    Dimensions of *field*                 what is drawn
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
    field:
        A field (:func:`sipnet_calibration.fields.validate_field`) with a
        ``time`` dim and no spatial dim, whose ``units`` and ``long_name``
        become the y axis label; a scalar ``site`` coordinate is fine.
    ax:
        The axes to draw on.
    show:
        One of :data:`SHOW_OPTIONS`. ``"auto"`` draws quantile bands when
        *field* has a batch dim and a single curve when it does not. Asking
        for ``"line"`` or ``"points"`` when there is a batch dim, or for
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
        The observation error, as a ``DataArray`` aligned with *field*, with
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
        *ax*, drawn on. Its y label is set from :func:`.style.axis_label`;
        the x axis and the title are left alone, a panel title being the
        grid's to set.

    Raises
    ------
    TypeError
        If an argument is of the wrong type: *field* or an error not a
        ``DataArray``, *ax* not an ``Axes``, *n_sigma* not a number.
    KeyError
        If *role* or *label_by* names something that is not there.
    ValueError
        If *field* is not a field or not a time series with a label, or an
        argument does not suit it; each message says which, and what to do.

    Notes
    -----
    ``show="fan"`` also draws the median as a curve, and the legend entry goes
    on that curve rather than on a band.

    Quantiles ignore missing values and are taken over all batch dims at
    once, so a field with two is summarized over the whole set of curves
    rather than in two stages.
    """
    validate_field(field)
    primitives.check_ax_is_an_axes(ax)
    name = message_name(field)
    check_field_is_a_time_series(field, message_name=name)
    batch = batch_dims(field)
    check_option_is_known(show, SHOW_OPTIONS, message_name="show")
    show = _show_for_the_batch(show, batch)
    check_show_suits_the_batch(show, batch, message_name=name)
    if label_by is not None:
        check_label_by_names_the_curves(label_by, show, field, batch, message_name=name)
    yerr = _error_bar_lengths(field, variance, standard_deviation, n_sigma, show)
    y_label = axis_label(field)

    if label is None:
        label = role
    x = field[TIME].values

    if show == "line":
        primitives.line(
            ax, x, field.values, label=label, **role_style(role, "line", **style)
        )
    elif show == "points":
        primitives.points(
            ax,
            x,
            field.values,
            yerr=yerr,
            label=label,
            **role_style(role, "points", **style),
        )
    else:
        curves = _curves_over_the_batch(field, batch)
        if show == "spaghetti" and label_by is not None:
            _draw_labeled_curves(
                ax, x, curves, _curve_labels(field, batch, label_by), n_max, style
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

    ax.set_ylabel(y_label)
    return ax


# ── private helpers ───────────────────────────────────────────────────────────


def _show_for_the_batch(show: str, batch: tuple[str, ...]) -> str:
    """*show*, with ``"auto"`` resolved from whether there is a batch dim."""
    if show != "auto":
        return show
    return "fan" if batch else "line"


def _curves_over_the_batch(
    field: xr.DataArray, batch: tuple[str, ...]
) -> np.ndarray:
    """*field* as ``(n_curves, n_time)``, the batch dims flattened together."""
    ordered = field.transpose(*batch, TIME)
    return ordered.values.reshape(-1, ordered.sizes[TIME])


def _curve_labels(
    field: xr.DataArray, batch: tuple[str, ...], label_by: str
) -> list[str]:
    """One label per curve, from the *label_by* coordinate's values.

    The order matches :func:`_curves_over_the_batch`, which flattens the
    batch dims in the order they are given.
    """
    coordinate = field.coords[label_by]
    sizes = tuple(field.sizes[dim] for dim in batch)
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
    ax: Axes,
    x: np.ndarray,
    curves: np.ndarray,
    labels: list[str],
    n_max: int,
    style: dict[str, Any],
) -> None:
    """Draw each curve in its own color, labeled by a coordinate's value."""
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
    field: xr.DataArray,
    variance: xr.DataArray | None,
    standard_deviation: xr.DataArray | None,
    n_sigma: float,
    show: str,
) -> np.ndarray | None:
    """Half-length of each error bar, or ``None`` when no error was given."""
    given = {
        name: array
        for name, array in (
            ("variance", variance),
            ("standard_deviation", standard_deviation),
        )
        if array is not None
    }
    if not given:
        return None
    check_error_is_given_once(given)
    check_error_bars_are_drawn_as_points(show)
    check_value_is_a_number(n_sigma, message_name="n_sigma")
    check_number_is_finite(n_sigma, message_name="n_sigma")
    check_number_is_positive(n_sigma, message_name="n_sigma")
    ((name, error),) = given.items()
    check_error_is_aligned_with_the_field(field, error, message_name=name)
    # show == "points" here, so the field has no batch dim and both arrays
    # are one-dimensional over time; no reordering is possible.
    values = error.values
    if name == "variance":
        check_variance_is_not_negative(values)
        values = np.sqrt(values)
    return n_sigma * values


# ── checks ────────────────────────────────────────────────────────────────────


def check_field_is_a_time_series(field: xr.DataArray, *, message_name: str) -> None:
    """A field to plot as a series has ``time`` and no spatial dim."""
    check_field_has_a_time_dim(field, message_name=message_name)
    check_field_has_no_spatial_dim(field, message_name=message_name)


def check_field_has_a_time_dim(field: xr.DataArray, *, message_name: str) -> None:
    """A field to plot against time has a ``time`` dim."""
    if TIME not in field.dims:
        raise ValueError(
            f"{message_name}: a time series needs {TIME!r} to plot against, and the "
            f"field's dims are {list(field.dims)}; select or aggregate to a time axis "
            "first, or draw a map with maps.plot_map."
        )


def check_field_has_no_spatial_dim(field: xr.DataArray, *, message_name: str) -> None:
    """A time series summarizes no spatial dim: sites are not replicates of one another."""
    spatial = [dim for dim in field.dims if dim in SPATIAL_DIM_NAMES]
    if spatial:
        advice = (
            f"select one site with .sel({SITE}=...), or draw one panel per site with "
            "facet.plot_by_site(field)"
            if SITE in spatial
            else "select one location first"
        )
        raise ValueError(
            f"{message_name}: a time series does not summarize the spatial dim(s) "
            f"{spatial}, since sites are not replicates of one another; {advice}."
        )


def check_show_suits_the_batch(show: str, batch: tuple[str, ...], *, message_name: str) -> None:
    """A summary of several curves is drawn over batch dims, one curve without them."""
    if show in ("fan", "spaghetti") and not batch:
        raise ValueError(
            f"{message_name}: show={show!r} summarizes several curves, and the field has "
            f"only {TIME!r}; use show='line' or show='points'."
        )
    if show in ("line", "points") and batch:
        raise ValueError(
            f"{message_name}: show={show!r} draws one curve, and the field also has the "
            f"batch dim(s) {list(batch)}; select or reduce them first, or use show='fan' "
            "or show='spaghetti'."
        )


def check_label_by_names_the_curves(
    label_by: str,
    show: str,
    field: xr.DataArray,
    batch: tuple[str, ...],
    *,
    message_name: str,
) -> None:
    """*label_by* names a coordinate on the batch dims of a spaghetti plot."""
    check_label_by_is_drawn_as_spaghetti(show)
    check_key_is_known(label_by, field.coords, message_name="coordinate")
    check_label_by_is_on_the_batch_dims(label_by, field, batch, message_name=message_name)


def check_label_by_is_drawn_as_spaghetti(show: str) -> None:
    """Curves are labeled one by one only when drawn one by one."""
    if show != "spaghetti":
        raise ValueError(
            f"label_by labels individual curves, which only show='spaghetti' draws, not "
            f"show={show!r}; pass show='spaghetti'."
        )


def check_label_by_is_on_the_batch_dims(
    label_by: str, field: xr.DataArray, batch: tuple[str, ...], *, message_name: str
) -> None:
    """The *label_by* coordinate lies on the batch dims, so it can name a curve."""
    dims = field.coords[label_by].dims
    if not dims or not set(dims) <= set(batch):
        raise ValueError(
            f"{message_name}: label_by={label_by!r} is on {list(dims)}, which is not among "
            f"the batch dims {list(batch)}, so it cannot name a curve; pass a coordinate "
            "on the batch dims."
        )


def check_error_is_given_once(given: dict[str, xr.DataArray]) -> None:
    """At most one of the variance and the standard deviation is given."""
    if len(given) > 1:
        raise ValueError(
            "variance and standard_deviation are two ways of stating the same error, and "
            "both were given; pass variance or standard_deviation, not both."
        )


def check_error_bars_are_drawn_as_points(show: str) -> None:
    """Error bars are drawn on scattered points."""
    if show != "points":
        raise ValueError(
            f"error bars are drawn by show='points', not show={show!r}; pass "
            "show='points' with an error."
        )


def check_error_is_aligned_with_the_field(
    field: xr.DataArray, error: Any, *, message_name: str
) -> None:
    """The error is a ``DataArray`` on exactly the field's points."""
    check_error_is_a_dataarray(error, message_name=message_name)
    check_error_has_the_fields_dims(field, error, message_name=message_name)
    check_error_has_the_fields_labels(field, error, message_name=message_name)


def check_error_is_a_dataarray(error: Any, *, message_name: str) -> None:
    """The error is labeled: a ``DataArray``."""
    if not isinstance(error, xr.DataArray):
        raise TypeError(
            f"{message_name} must be an xarray.DataArray aligned with the field, got "
            f"{type(error).__name__}; pass it labeled as the field is."
        )


def check_error_has_the_fields_dims(
    field: xr.DataArray, error: xr.DataArray, *, message_name: str
) -> None:
    """The error has the field's dims."""
    if set(error.dims) != set(field.dims):
        raise ValueError(
            f"{message_name} has dimensions {list(error.dims)} and the field has "
            f"{list(field.dims)}; they must match, so select the error as the field was."
        )


def check_error_has_the_fields_labels(
    field: xr.DataArray, error: xr.DataArray, *, message_name: str
) -> None:
    """The error has exactly the field's labels on each dim."""
    try:
        xr.align(field, error, join="exact")
    except ValueError as mismatch:
        raise ValueError(
            f"{message_name} is not aligned with the field ({mismatch}); pass the error on "
            "the field's own labels."
        ) from mismatch


def check_variance_is_not_negative(values: np.ndarray) -> None:
    """A variance is never negative."""
    if np.any(values[np.isfinite(values)] < 0):
        raise ValueError(
            "variance: a variance cannot be negative; pass the square of a standard "
            "deviation, or standard_deviation itself."
        )
