"""L3 -- the one generic facet function.

Overview
--------
:func:`facet` owns figure and axes construction, shared limits, panel titles
and legend deduplication, for every grid of panels in the project. It is the
reason an L2 panel never creates its own figure: most of the repetitive figure
code in a project like this is figure, axes, limit and legend plumbing, and
centralizing it here is where the bulk of the saving comes from.

Two thin wrappers cover the cases that come up constantly --
:func:`by_site`, one panel per site, and :func:`by_variable`, one panel per
variable. Both are a few lines over :func:`facet` and exist because writing
the closure and the titles by hand at every call site is exactly the
repetition this layer removes.

Callbacks
---------
The two layers take callbacks of different shapes, and each reads naturally
for what it is:

* :func:`facet` takes ``panel_fn(ax, item) -> None``. *items* is any sequence
  and the callback does whatever it likes with an element; its return is
  ignored.
* The wrappers take a **panel composer**, ``panel_fn(field, ax=ax)`` --
  :func:`sipnet_calibration.plotting.series.series_panel` and anything with
  its signature. The wrapper builds the selection and the closure.

Extra arguments are bound with ``functools.partial`` rather than threaded
through this module::

    from functools import partial
    by_site(tair, partial(series_panel, role="prior", show="spaghetti"))

Notes
-----
**Shared limits come from matplotlib's own axis sharing**, through ``share``,
rather than from a separate pass that computes common limits and applies
them. The design spec carried both a ``share`` and a ``common_lims``
parameter; they are the same request, and ``sharex``/``sharey`` gets it right
through autoscaling, including as later artists are added to a panel.

**The return is** ``(Figure, axes)``, not the ``Figure`` alone that the spec
proposed. ``fig.axes`` also holds the hidden filler axes that pad the last
row, so a caller wanting to annotate panel *k* would otherwise have to redo
the grid arithmetic. The returned array is aligned one-to-one with *items*.

**No style is applied.** :func:`.style.use_project_style` is an experiment
report's call, not this module's; a facet grid that restyled the session
would be as surprising as an import that did.

Usage
-----
::

    from sipnet_calibration.plotting import by_site, facet, series_panel

    # One panel per site, each a driver ensemble fan, shared y-limits.
    fig, axes = by_site(air_temperature, sites=six_sites, share="y", ncol=3)

    # The general form.
    fig, axes = facet(six_sites, lambda ax, s: series_panel(f.sel(site=s), ax=ax),
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
        ignored, so an L2 panel that returns its ``Axes`` can be passed
        directly.
    ncol:
        Panels per row. The number of rows follows from ``len(items)``; the
        unused axes of the last row are hidden rather than left as empty
        frames.
    share:
        One of :data:`SHARE_MODES`, passed to ``pyplot.subplots`` as
        ``sharex``/``sharey``. ``"y"`` is what gives a grid of panels one
        common y-scale, which is usually the point of drawing them together.
    labels:
        Panel titles: a sequence the same length as *items*, or a callable
        applied to each element, or ``None`` for no titles.
    panel_size:
        Width and height of one panel in inches. The figure is sized from this
        and the grid, so a six-panel grid is not the same size as a
        twenty-panel one.
    legend:
        ``"dedup"`` collects the handles and labels of every panel, keeps the
        first occurrence of each label, and places one figure-level legend --
        the usual case, where each panel draws the same few roles.
        ``"each"`` gives every panel its own legend. ``"none"`` draws none.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        The figure, and a one-dimensional object array of the axes actually
        drawn on, aligned with *items*. Hidden filler axes are not in it.

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
        A canonical field with a ``site`` dim.
    panel_fn:
        A panel composer called as ``panel_fn(field.sel(site=s), ax=ax)``.
        ``None`` means
        :func:`sipnet_calibration.plotting.series.series_panel`. Bind extra
        arguments with ``functools.partial``.
    sites:
        The site ids to draw, in that order. ``None`` means every site on
        *field*, which is a grid of 8000 panels for a whole-pool field --
        select first.
    **facet_kwargs:
        Passed to :func:`facet`. ``labels`` defaults to ``"site <id>"``.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        As :func:`facet`, with one entry per site.

    Raises
    ------
    ValueError
        If *field* has no ``site`` dim, or *sites* names an id that is not on
        it.
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
        Variable name to canonical field, as the multi-variable adapters
        return -- :func:`sipnet_calibration.drivers.driver_fields` and
        :func:`sipnet_calibration.constraints.constraint_fields`. The fields
        need not share a time axis, which is the reason those adapters return
        a mapping rather than a ``Dataset``.
    panel_fn:
        A panel composer called as ``panel_fn(field, ax=ax)``. ``None`` means
        :func:`sipnet_calibration.plotting.series.series_panel`.
    **facet_kwargs:
        Passed to :func:`facet`. ``labels`` defaults to each field's
        ``long_name``, falling back to the variable's name. ``share`` defaults
        to ``"none"`` and should stay there: the variables have different
        units, so a shared y-axis would be meaningless.

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
