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

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["SHARE_MODES", "build_plot_grid", "plot_by_site", "plot_by_variable"]

#: What ``share`` may be, matching ``pyplot.subplots``' ``sharex``/``sharey``.
SHARE_MODES: tuple[str, ...] = ("none", "x", "y", "both")


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
