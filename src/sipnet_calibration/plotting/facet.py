"""Grids of plot panels.

Multi-panel figures are the normal way to look at this project's data: one
panel per site, or one per variable, with the panels sharing a scale so that
they can be read against each other. This module builds them. It creates the
figure and the grid of axes, calls a plotting function once per panel, and
handles the panel titles, the shared axis limits and the single figure legend.

:func:`build_plot_grid` is the general form and accepts any sequence of items.
:func:`plot_by_site` and :func:`plot_by_variable` cover the two cases that come
up constantly for time series and are written over it.

Grids of maps are written over it too, and differ in sharing a frame and,
by default, one color scale with a single colorbar:

========================  ====================================================
:func:`plot_map_grid`     one map per entry of a ``dict`` of fields
:func:`plot_map_by`       one map per member, or per time step
:func:`plot_map_quantiles`  one map per quantile over the members
========================  ====================================================

This is the only part of the package that creates a figure. The plotting
functions it calls draw onto an ``Axes`` they are given, which is what lets the
same function serve a single panel and a grid of them.

Arguments beyond the item being plotted are bound with ``functools.partial``
rather than passed through this module::

    from functools import partial
    plot_by_site(tair, partial(plot_time_series, show="spaghetti"))

Usage
-----
::

    from sipnet_calibration.plotting import build_plot_grid, plot_by_site

    # One panel per site, each a driver ensemble, on one y scale.
    figure, axes = plot_by_site(air_temperature, sites=six_sites, share="y")

    # One map per quantile over the members, on one color scale.
    figure, axes = plot_map_quantiles(wood, extent="CONUS", log=True)

    # Mean and standard deviation, each on its own scale.
    figure, axes = plot_map_grid({
        "mean": member_summary(wood, "mean"),
        "standard deviation": member_summary(wood, "standard_deviation"),
    })

    # The general form.
    figure, axes = build_plot_grid(
        six_sites,
        lambda ax, site: plot_time_series(field.sel(site=site), ax=ax),
        labels=lambda site: f"site {site}",
    )
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sipnet_calibration.plotting import maps
from sipnet_calibration.plotting.primitives import thinned_indices
from sipnet_calibration.plotting.series import plot_time_series
from sipnet_calibration.plotting.style import axis_label

__all__ = [
    "LEGEND_MODES",
    "SCALE_MODES",
    "SHARE_MODES",
    "SITE_DIM",
    "build_plot_grid",
    "plot_by_site",
    "plot_by_variable",
    "plot_map_by",
    "plot_map_grid",
    "plot_map_quantiles",
]

#: What ``share`` may be, matching ``pyplot.subplots``' ``sharex``/``sharey``.
SHARE_MODES: tuple[str, ...] = ("none", "x", "y", "both")

#: What ``legend`` may be.
LEGEND_MODES: tuple[str, ...] = ("dedup", "each", "none")

#: The dimension :func:`plot_by_site` splits on.
SITE_DIM = "site"

#: What ``scale`` may be for a grid of maps: one color scale for every panel,
#: or one per panel.
SCALE_MODES: tuple[str, ...] = ("shared", "each")


def build_plot_grid(
    items: Sequence[Any],
    panel_fn: Callable[[Axes, Any], None],
    *,
    ncol: int = 3,
    share: str = "none",
    labels: Sequence[str] | Callable[[Any], str] | None = None,
    panel_size: tuple[float, float] = (3.2, 2.4),
    legend: str = "dedup",
) -> tuple[Figure, np.ndarray]:
    """Build a figure of panels, one per element of *items*.

    Parameters
    ----------
    items:
        Any sequence. One panel is drawn per element, in order, filling rows
        left to right.
    panel_fn:
        Called as ``panel_fn(ax, item)`` once per element, with the axes for
        that panel. Its return value is ignored, so a plotting function that
        returns its ``Axes`` can be passed directly.
    ncol:
        Panels per row. The number of rows follows from ``len(items)``, and
        the unused axes of the last row are hidden.
    share:
        One of :data:`SHARE_MODES`, passed to ``pyplot.subplots`` as
        ``sharex`` and ``sharey``. ``"y"`` puts every panel on one y scale.
    labels:
        Panel titles: a sequence as long as *items*, a callable applied to
        each element, or ``None`` for no titles.
    panel_size:
        Width and height of one panel, in inches. The figure is sized from
        this and the shape of the grid, and laid out with matplotlib's
        constrained layout, which is what keeps the legend clear of the
        panels.
    legend:
        ``"dedup"`` collects the handles and labels of every panel, keeps the
        first occurrence of each label, and places one legend on the figure.
        ``"each"`` gives every panel its own. ``"none"`` draws none.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        The figure, and a one-dimensional object array of the axes that were
        drawn on, in the order of *items*. The hidden axes are not included.

    Raises
    ------
    ValueError
        If *items* is empty; if *ncol* is not a positive integer; if *share*
        is not in :data:`SHARE_MODES`; if *legend* is not ``"dedup"``,
        ``"each"`` or ``"none"``; or if *labels* is a sequence of a different
        length from *items*.
    """
    items = list(items)
    if not items:
        raise ValueError("items is empty; there is nothing to draw")
    if not isinstance(ncol, (int, np.integer)) or int(ncol) < 1:
        raise ValueError(f"ncol must be a positive integer, got {ncol!r}")
    if share not in SHARE_MODES:
        raise ValueError(f"share must be one of {list(SHARE_MODES)}, got {share!r}")
    if legend not in LEGEND_MODES:
        raise ValueError(f"legend must be one of {list(LEGEND_MODES)}, got {legend!r}")
    titles = _panel_titles(items, labels)

    ncol = min(int(ncol), len(items))
    nrow = math.ceil(len(items) / ncol)
    figure, grid = plt.subplots(
        nrow,
        ncol,
        figsize=(ncol * panel_size[0], nrow * panel_size[1]),
        sharex=share in ("x", "both"),
        sharey=share in ("y", "both"),
        squeeze=False,
        layout="constrained",
    )
    every_axes = grid.ravel()
    for spare in every_axes[len(items) :]:
        spare.set_visible(False)

    axes = np.empty(len(items), dtype=object)
    try:
        for position, item in enumerate(items):
            axes[position] = every_axes[position]
            panel_fn(every_axes[position], item)
            if titles is not None:
                every_axes[position].set_title(titles[position])
        _add_legend(figure, axes, legend)
    except BaseException:
        plt.close(figure)
        raise
    return figure, axes


def _panel_titles(items, labels) -> list[str] | None:
    """One title per item, or ``None`` when no titles were asked for."""
    if labels is None:
        return None
    if callable(labels):
        return [str(labels(item)) for item in items]
    if isinstance(labels, str):
        raise ValueError(
            f"labels is the single string {labels!r}, which would title the "
            "panels one character each; pass one label per panel, or a callable"
        )
    titles = list(labels)
    if len(titles) != len(items):
        raise ValueError(
            f"labels has {len(titles)} entries and there are {len(items)} "
            "panels; they must match"
        )
    return [str(title) for title in titles]


def _add_legend(figure: Figure, axes: np.ndarray, legend: str) -> None:
    """Place the legend asked for, if there is anything to put in it."""
    if legend == "none":
        return
    if legend == "each":
        for ax in axes:
            if ax.get_legend_handles_labels()[1]:
                ax.legend()
        return
    unique: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
    if unique:
        figure.legend(list(unique.values()), list(unique), loc="outside upper right")


def plot_by_site(
    data: xr.DataArray,
    panel_fn: Callable[..., Any] | None = None,
    *,
    sites: Sequence[int] | None = None,
    **grid_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One panel per site, each drawing that site's slice of *data*.

    Parameters
    ----------
    data:
        A ``DataArray`` with a ``site`` dimension.
    panel_fn:
        Called as ``panel_fn(data.sel(site=s), ax=ax)`` for each site.
        ``None`` uses
        :func:`sipnet_calibration.plotting.series.plot_time_series`.
    sites:
        The site ids to draw, in that order. ``None`` draws every site in
        *data*, which for a whole-pool field is 8000 panels.
    **grid_kwargs:
        Passed to :func:`build_plot_grid`. ``labels`` defaults to
        ``"site <id>"``.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        As :func:`build_plot_grid`, with one entry per site.

    Raises
    ------
    ValueError
        If *data* has no ``site`` dimension, or *sites* names an id that is
        not in it.
    """
    if SITE_DIM not in data.dims:
        raise ValueError(
            f"the array has dimensions {list(data.dims)} and needs {SITE_DIM!r} "
            "to be split by site"
        )
    if SITE_DIM not in data.coords:
        raise ValueError(
            f"the array has a {SITE_DIM!r} dimension but no {SITE_DIM!r} "
            "coordinate, so its panels cannot be named or selected"
        )
    available = list(data.coords[SITE_DIM].values)
    if sites is None:
        chosen = available
    else:
        if isinstance(sites, (str, bytes)) or not hasattr(sites, "__iter__"):
            raise ValueError(f"sites must be a sequence of site ids, got {sites!r}")
        chosen = list(sites)
        missing = [site for site in chosen if site not in available]
        if missing:
            raise ValueError(
                f"no such site(s) in the data: {missing}. It holds "
                f"{len(available)} site(s), starting {available[:5]}"
            )

    panel_fn = plot_time_series if panel_fn is None else panel_fn
    grid_kwargs.setdefault("labels", lambda site: f"site {site}")
    return build_plot_grid(
        chosen,
        lambda ax, site: panel_fn(data.sel({SITE_DIM: site}), ax=ax),
        **grid_kwargs,
    )


def plot_by_variable(
    data: dict[str, xr.DataArray],
    panel_fn: Callable[..., Any] | None = None,
    **grid_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One panel per variable, in the order *data* gives them.

    Parameters
    ----------
    data:
        Variable name to ``DataArray``, as
        :func:`sipnet_calibration.drivers.driver_fields` and
        :func:`sipnet_calibration.constraints.constraint_fields` return. The
        variables need not share a time axis.
    panel_fn:
        Called as ``panel_fn(array, ax=ax)`` for each variable. ``None`` uses
        :func:`sipnet_calibration.plotting.series.plot_time_series`.
    **grid_kwargs:
        Passed to :func:`build_plot_grid`. ``labels`` defaults to each
        variable's ``long_name``, falling back to its name. ``share`` defaults
        to ``"none"``, since the variables have different units.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        As :func:`build_plot_grid`, with one entry per variable.

    Raises
    ------
    ValueError
        If *data* is empty.
    """
    if not data:
        raise ValueError("data is empty; there is nothing to draw")
    panel_fn = plot_time_series if panel_fn is None else panel_fn
    names = list(data)
    grid_kwargs.setdefault(
        "labels", [data[name].attrs.get("long_name", name) for name in names]
    )
    return build_plot_grid(
        names, lambda ax, name: panel_fn(data[name], ax=ax), **grid_kwargs
    )


def plot_map_grid(
    fields: Mapping[str, xr.DataArray],
    *,
    scale: str = "each",
    extent: Any = None,
    ncol: int = 3,
    panel_size: tuple[float, float] | None = None,
    colorbar_label: str | None = None,
    **map_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One map per entry of *fields*, all in one frame.

    Parameters
    ----------
    fields:
        Panel title to field, in panel order. Each field is a map in the sense
        of :func:`sipnet_calibration.plotting.maps.plot_map`.
    scale:
        ``"shared"`` resolves one color scale over every panel's values in the
        frame and draws one colorbar, or one legend, for the figure. ``"each"``
        gives each panel its own, which suits panels of different quantities,
        such as a mean beside a standard deviation.
    extent:
        As :func:`~sipnet_calibration.plotting.maps.plot_map` takes it.
        ``None`` fits one frame to every panel's data, so the panels line up.
    ncol:
        Panels per row.
    panel_size:
        Width and height of one panel, in inches. ``None`` sizes it from the
        frame's shape.
    colorbar_label:
        The shared colorbar's label, in place of the first field's.
    **map_kwargs:
        Passed to :func:`~sipnet_calibration.plotting.maps.plot_map` for every
        panel. With ``scale="shared"`` the color keywords are resolved once,
        over all of them.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        As :func:`build_plot_grid`, with one entry per field.

    Raises
    ------
    ValueError
        If *fields* is empty or *scale* is not in :data:`SCALE_MODES`, and
        whatever :func:`~sipnet_calibration.plotting.maps.plot_map` raises for a
        panel.
    """
    if not fields:
        raise ValueError("fields is empty; there is nothing to draw")
    if scale not in SCALE_MODES:
        raise ValueError(f"scale must be one of {list(SCALE_MODES)}, got {scale!r}")
    arrays = list(fields.values())
    bounds = maps.map_bounds(arrays, extent)
    color = {k: map_kwargs.pop(k) for k in maps.COLOR_KEYWORDS if k in map_kwargs}
    shared = maps.color_scale(arrays, bounds=bounds, **color) if scale == "shared" else None
    if panel_size is None:
        panel_size = _map_panel_size(bounds, own_key=shared is None)

    def draw(ax: Axes, name: str) -> None:
        if shared is None:
            maps.plot_map(fields[name], ax, extent=bounds, **color, **map_kwargs)
        else:
            maps.plot_map(fields[name], ax, extent=bounds, scale=shared, colorbar=False, **map_kwargs)

    figure, axes = build_plot_grid(
        list(fields), draw, ncol=ncol, labels=list(fields), panel_size=panel_size, legend="none"
    )
    if shared is not None:
        _add_shared_key(figure, axes, shared, arrays, bounds, colorbar_label)
    return figure, axes


def plot_map_by(
    field: xr.DataArray,
    dim: str,
    *,
    values: Sequence[Any] | None = None,
    n_max: int = 12,
    scale: str = "shared",
    **grid_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One map per value of *dim*: per ensemble member, or per time step.

    Parameters
    ----------
    field:
        A field that is a map at each value of *dim*.
    dim:
        The dimension to split on, such as ``"member"`` or ``"time"``.
    values:
        The coordinate values to draw, in order. ``None`` draws them all, or
        *n_max* evenly spaced ones, first and last included, if there are more.
    n_max:
        The most panels drawn when *values* is ``None``.
    scale:
        As :func:`plot_map_grid` takes it; shared by default, since every panel
        is the same quantity.
    **grid_kwargs:
        Passed to :func:`plot_map_grid`.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        As :func:`build_plot_grid`, one entry per value, titled by it.

    Raises
    ------
    ValueError
        If *field* has no *dim*, *values* names one it does not hold, or
        *n_max* is not a positive integer.
    """
    if not isinstance(field, xr.DataArray) or dim not in field.dims:
        dims = list(getattr(field, "dims", ()))
        raise ValueError(f"plot_map_by needs a DataArray with a {dim!r} dimension; got {dims}")
    if not isinstance(n_max, (int, np.integer)) or int(n_max) < 1:
        raise ValueError(f"n_max must be a positive integer, got {n_max!r}")
    available = field[dim].values
    if values is None:
        chosen = available[thinned_indices(len(available), int(n_max))]
    else:
        chosen = np.asarray(getattr(values, "values", values))
        missing = [v for v in chosen if v not in available]
        if missing:
            raise ValueError(f"no such {dim} value(s) in the field: {missing[:5]}")
    panels = {maps.coordinate_label(dim, value): field.sel({dim: value}) for value in chosen}
    if len(panels) < len(chosen):
        panels = {f"{dim} {value}": field.sel({dim: value}) for value in chosen}
    return plot_map_grid(panels, scale=scale, **grid_kwargs)


def plot_map_quantiles(
    field: xr.DataArray,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    *,
    dim: str = "member",
    scale: str = "shared",
    **grid_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One map per quantile of *field* over its ensemble dimension.

    Parameters
    ----------
    field:
        A continuous field that is a map at each value of *dim*.
    quantiles:
        The quantiles, each in ``(0, 1)``. The default is the 90% central
        interval and the median, the outer band of
        :func:`~sipnet_calibration.plotting.primitives.fan`.
    dim:
        The ensemble dimension.
    scale:
        As :func:`plot_map_grid` takes it; shared by default, so the panels
        read against each other.
    **grid_kwargs:
        Passed to :func:`plot_map_grid`.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        As :func:`build_plot_grid`, one entry per quantile, titled
        ``"5th percentile"``, ``"median"`` and so on.

    Raises
    ------
    ValueError
        If *quantiles* is empty, and whatever
        :func:`~sipnet_calibration.plotting.maps.member_summary` raises.
    """
    quantiles = [float(q) for q in quantiles]
    if not quantiles:
        raise ValueError("quantiles must name at least one quantile")
    panels = {maps.quantile_label(q): maps.member_summary(field, q, dim=dim) for q in quantiles}
    grid_kwargs.setdefault("colorbar_label", axis_label(field))
    return plot_map_grid(panels, scale=scale, **grid_kwargs)


def _map_panel_size(bounds: maps.ProjectedBounds, *, own_key: bool) -> tuple[float, float]:
    """A panel size matching the frame's shape, with room for a colorbar."""
    width = 3.4
    aspect = (bounds.y_max - bounds.y_min) / (bounds.x_max - bounds.x_min)
    height = float(np.clip(width * aspect, 1.6, 5.0)) + 0.35
    return (width + (1.2 if own_key else 0.0), height)


def _add_shared_key(figure, axes, scale, fields, bounds, label) -> None:
    """One colorbar, or one legend of the classes present, for the figure."""
    if scale.categories is None:
        figure.colorbar(
            scale.mappable(), ax=list(axes), shrink=0.8, label=label or scale.label
        )
        return
    figure.legend(
        handles=scale.legend_handles(scale.classes_in(fields, bounds)),
        title=label or scale.label,
        loc="outside right center",
    )
