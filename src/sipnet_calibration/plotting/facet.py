"""Grids of plot panels.

Multi-panel figures are the normal way to look at this project's data: one
panel per site, or one per variable, with the panels sharing a scale so that
they can be read against each other. This module builds them. It creates the
figure and the grid of axes, calls a plotting function once per panel, and
handles the panel titles, the shared axis limits and the single figure legend.

:func:`build_plot_grid` is the general form and accepts any sequence of items.
:func:`plot_by_site` and :func:`plot_by_variable` cover the two cases that come
up constantly and are written over it.

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

    # The general form.
    figure, axes = build_plot_grid(
        six_sites,
        lambda ax, site: plot_time_series(field.sel(site=site), ax=ax),
        labels=lambda site: f"site {site}",
    )
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sipnet_calibration.plotting.series import plot_time_series

__all__ = [
    "LEGEND_MODES",
    "SHARE_MODES",
    "SITE_DIM",
    "build_plot_grid",
    "plot_by_site",
    "plot_by_variable",
]

#: What ``share`` may be, matching ``pyplot.subplots``' ``sharex``/``sharey``.
SHARE_MODES: tuple[str, ...] = ("none", "x", "y", "both")

#: What ``legend`` may be.
LEGEND_MODES: tuple[str, ...] = ("dedup", "each", "none")

#: The dimension :func:`plot_by_site` splits on.
SITE_DIM = "site"


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
        this and the shape of the grid.
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
    )
    every_axes = grid.ravel()
    for spare in every_axes[len(items) :]:
        spare.set_visible(False)

    axes = np.empty(len(items), dtype=object)
    for position, item in enumerate(items):
        axes[position] = every_axes[position]
        panel_fn(every_axes[position], item)
        if titles is not None:
            every_axes[position].set_title(titles[position])

    _add_legend(figure, axes, legend)
    return figure, axes


def _panel_titles(items, labels) -> list[str] | None:
    """One title per item, or ``None`` when no titles were asked for."""
    if labels is None:
        return None
    if callable(labels):
        return [str(labels(item)) for item in items]
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
    available = (
        list(data.coords[SITE_DIM].values) if SITE_DIM in data.coords else []
    )
    if sites is None:
        chosen = available
    else:
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
