"""Grids of panels: figure and axes construction, shared limits, legends.

:func:`facet` draws one panel per element of a sequence, and owns everything
around the panels -- the figure, the grid of axes, axis sharing, panel titles
and the legend. It is why the panel functions never create a figure of their
own.

:func:`by_site` and :func:`by_variable` are the two cases that come up
constantly, written over :func:`facet`.

Callbacks
---------
:func:`facet` takes ``panel_fn(ax, item)``, called once per element, and
ignores what it returns. The two wrappers instead take a panel function of the
form ``panel_fn(field, ax=ax)`` -- such as
:func:`sipnet_calibration.plotting.series.series_panel` -- and build the
selection themselves.

Bind any further arguments with ``functools.partial``::

    from functools import partial
    by_site(tair, partial(series_panel, role="prior", show="spaghetti"))

Usage
-----
::

    from sipnet_calibration.plotting import by_site, facet, series_panel

    # One panel per site, each a driver ensemble fan, on one y scale.
    fig, axes = by_site(air_temperature, sites=six_sites, share="y", ncol=3)

    # The general form.
    fig, axes = facet(six_sites,
                      lambda ax, s: series_panel(field.sel(site=s), ax=ax),
                      labels=lambda s: f"site {s}")
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["SHARE_MODES", "by_site", "by_variable", "facet"]

#: What ``share`` may be, matching ``pyplot.subplots``' ``sharex``/``sharey``.
SHARE_MODES: tuple[str, ...] = ("none", "x", "y", "both")


def facet(
    items: Sequence[Any],
    panel_fn: Callable[[Axes, Any], None],
    *,
    ncol: int = 3,
    share: str = "none",
    labels: Sequence[str] | Callable[[Any], str] | None = None,
    panel_size: tuple[float, float] = (3.2, 2.4),
    legend: str = "dedup",
) -> tuple[Figure, np.ndarray]:
    """Build a grid of panels, one per element of *items*.

    Parameters
    ----------
    items:
        Any sequence. One panel is drawn per element, in order, filling rows
        left to right.
    panel_fn:
        Called as ``panel_fn(ax, item)`` once per element. Its return value is
        ignored, so a panel function that returns its ``Axes`` may be used
        directly.
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
        drawn on, aligned with *items*. The hidden axes are not in it.

    Raises
    ------
    ValueError
        If *items* is empty; if *ncol* is not a positive integer; if *share*
        is not in :data:`SHARE_MODES`; if *legend* is not ``"dedup"``,
        ``"each"`` or ``"none"``; or if *labels* is a sequence of a different
        length from *items*.
    """
    raise NotImplementedError


def by_site(
    field: xr.DataArray,
    panel_fn: Callable[..., Any] | None = None,
    *,
    sites: Sequence[int] | None = None,
    **facet_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One panel per site, each drawing that site's slice of *field*.

    Parameters
    ----------
    field:
        A canonical field with a ``site`` dimension.
    panel_fn:
        Called as ``panel_fn(field.sel(site=s), ax=ax)``. ``None`` uses
        :func:`sipnet_calibration.plotting.series.series_panel`.
    sites:
        The site ids to draw, in that order. ``None`` draws every site on
        *field*, which for a whole-pool field is 8000 panels.
    **facet_kwargs:
        Passed to :func:`facet`. ``labels`` defaults to ``"site <id>"``.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        As :func:`facet`, with one entry per site.

    Raises
    ------
    ValueError
        If *field* has no ``site`` dimension, or *sites* names an id that is
        not on it.
    """
    raise NotImplementedError


def by_variable(
    fields: dict[str, xr.DataArray],
    panel_fn: Callable[..., Any] | None = None,
    **facet_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One panel per variable, in the order *fields* gives them.

    Parameters
    ----------
    fields:
        Variable name to field, as
        :func:`sipnet_calibration.drivers.driver_fields` and
        :func:`sipnet_calibration.constraints.constraint_fields` return. The
        fields need not share a time axis.
    panel_fn:
        Called as ``panel_fn(field, ax=ax)``. ``None`` uses
        :func:`sipnet_calibration.plotting.series.series_panel`.
    **facet_kwargs:
        Passed to :func:`facet`. ``labels`` defaults to each field's
        ``long_name``, falling back to its name. ``share`` defaults to
        ``"none"``, since the variables have different units.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        As :func:`facet`, with one entry per variable.

    Raises
    ------
    ValueError
        If *fields* is empty.
    """
    raise NotImplementedError
