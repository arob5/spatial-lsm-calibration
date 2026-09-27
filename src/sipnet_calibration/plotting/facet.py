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

===========================  =================================================
:func:`plot_map_grid`        one map per entry of a ``dict`` of fields
:func:`plot_map_by`          one map per batch label, or per time label
:func:`plot_map_quantiles`   one map per quantile over a batch dim
===========================  =================================================

This is the only part of the package that creates a figure. The plotting
functions it calls draw onto an ``Axes`` they are given, which is what lets the
same function serve a single panel and a grid of them.

Arguments beyond the item being plotted are bound with ``functools.partial``
rather than passed through this module::

    from functools import partial
    plot_by_site(air_temperature, partial(plot_time_series, show="spaghetti"))

Usage
-----
::

    from sipnet_calibration.drivers import driver_fields, load_drivers
    from sipnet_calibration.initial_conditions import initial_condition_fields
    from sipnet_calibration.plotting import (
        build_plot_grid,
        plot_by_site,
        plot_map_grid,
        plot_map_quantiles,
        plot_time_series,
        summarize_batch,
    )

    air_temperature = driver_fields(load_drivers([1, 27]))["air_temperature"]
    wood = initial_condition_fields(["initial_wood_carbon"])["initial_wood_carbon"]

    # One panel per site, each a driver ensemble, on one y scale.
    figure, axes = plot_by_site(air_temperature, sites=[27, 1], share="y")

    # One map per quantile over the initial conditions' members, on one scale.
    batch_dim = "initial_condition_member"
    figure, axes = plot_map_quantiles(wood, batch_dim=batch_dim, extent="CONUS", log=True)

    # Mean and standard deviation, each on its own scale.
    figure, axes = plot_map_grid({
        "mean": summarize_batch(wood, "mean", batch_dim=batch_dim),
        "standard deviation": summarize_batch(wood, "standard_deviation", batch_dim=batch_dim),
    })

    # The general form.
    figure, axes = build_plot_grid(
        [1, 27],
        lambda ax, site: plot_time_series(air_temperature.sel(site=site), ax),
        labels=lambda site: f"site {site}",
    )
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sipnet_calibration.conventions import SAMPLE, SITE
from sipnet_calibration.fields import message_name, missing_labels, validate_field
from sipnet_calibration.plotting import maps
from sipnet_calibration.plotting.primitives import thinned_indices
from sipnet_calibration.plotting.series import plot_time_series
from sipnet_calibration.plotting.style import (
    axis_label,
    check_keywords_are_not_retired,
    check_option_is_known,
    as_real_number,
)
from sipnet_calibration.validation import (
    as_positive_integer,
    as_sequence,
    as_site_ids,
    truncated,
)

__all__ = [
    "LEGEND_OPTIONS",
    "SCALE_OPTIONS",
    "SHARE_OPTIONS",
    "build_plot_grid",
    "plot_by_site",
    "plot_by_variable",
    "plot_map_by",
    "plot_map_grid",
    "plot_map_quantiles",
]

#: What ``share`` may be, matching ``pyplot.subplots``' ``sharex``/``sharey``.
SHARE_OPTIONS: tuple[str, ...] = ("none", "x", "y", "both")

#: What ``legend`` may be.
LEGEND_OPTIONS: tuple[str, ...] = ("dedup", "each", "none")

#: What ``scale`` may be for a grid of maps: one color scale for every panel,
#: or one per panel.
SCALE_OPTIONS: tuple[str, ...] = ("shared", "each")


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
        One of :data:`SHARE_OPTIONS`, passed to ``pyplot.subplots`` as
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
        One of :data:`LEGEND_OPTIONS`. ``"dedup"`` collects the handles and
        labels of every panel, keeps the first occurrence of each label, and
        places one legend on the figure. ``"each"`` gives every panel its
        own. ``"none"`` draws none.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        The figure, and a one-dimensional object array of the axes that were
        drawn on, in the order of *items*. The hidden axes are not included.

    Raises
    ------
    TypeError
        If *ncol* is a boolean or not an integer, or *labels* is one string.
    ValueError
        If *items* is empty; if *ncol* is less than 1; if *share* or *legend*
        is not one of its options; or if *labels* is a sequence of a
        different length from *items*.
    """
    items = list(items)
    check_items_are_given(items)
    ncol = as_positive_integer(ncol, message_name="ncol")
    check_option_is_known(share, SHARE_OPTIONS, message_name="share")
    check_option_is_known(legend, LEGEND_OPTIONS, message_name="legend")
    titles = _panel_titles(items, labels)

    ncol = min(ncol, len(items))
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


def plot_by_site(
    field: xr.DataArray,
    panel_fn: Callable[..., Any] | None = None,
    *,
    sites: Sequence[int] | None = None,
    **grid_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One panel per site, each drawing that site's slice of *field*.

    Parameters
    ----------
    field:
        A field with a ``site`` dimension.
    panel_fn:
        Called as ``panel_fn(field.sel(site=s), ax=ax)`` for each site.
        ``None`` uses
        :func:`sipnet_calibration.plotting.series.plot_time_series`.
    sites:
        The site ids to draw, a sequence, in that order, each once. ``None``
        draws every site in *field*, which for a whole-pool field is a panel
        per site of the pool.
    **grid_kwargs:
        Passed to :func:`build_plot_grid`. ``labels`` defaults to
        ``"site <id>"``.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        As :func:`build_plot_grid`, with one entry per site.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray`` (a Dataset is split into fields
        first); if *sites* is one id, a string or a set, or holds a boolean, a
        float or a value that is not a number.
    ValueError
        If *field* is not a field
        (:func:`sipnet_calibration.fields.validate_field`) or has no ``site``
        dimension, or *sites* names a site twice or holds a value that is not
        a site id.
    KeyError
        If *sites* names a site that is not in *field*.
    """
    validate_field(field)
    name = message_name(field)
    check_field_has_a_site_dim(field, message_name=name)
    chosen = (
        field.coords[SITE].values.tolist()
        if sites is None
        else list(as_site_ids(sites, message_name="sites"))
    )
    check_field_holds_the_labels(field, SITE, chosen, message_name=name)

    panel_fn = plot_time_series if panel_fn is None else panel_fn
    grid_kwargs.setdefault("labels", lambda site: f"site {site}")
    return build_plot_grid(
        chosen,
        lambda ax, site: panel_fn(field.sel({SITE: site}), ax=ax),
        **grid_kwargs,
    )


def plot_by_variable(
    fields_by_name: dict[str, xr.DataArray],
    panel_fn: Callable[..., Any] | None = None,
    **grid_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One panel per variable, in the order *fields_by_name* gives them.

    Parameters
    ----------
    fields_by_name:
        Variable name to field, as
        :func:`sipnet_calibration.drivers.driver_fields` and
        :func:`sipnet_calibration.constraints.constraint_fields` return. The
        variables need not share a time axis.
    panel_fn:
        Called as ``panel_fn(field, ax=ax)`` for each variable. ``None`` uses
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
        If *fields_by_name* is empty.
    """
    check_fields_are_given(fields_by_name, example="{name: field}", message_name="fields_by_name")
    panel_fn = plot_time_series if panel_fn is None else panel_fn
    names = list(fields_by_name)
    grid_kwargs.setdefault(
        "labels", [fields_by_name[name].attrs.get("long_name", name) for name in names]
    )
    return build_plot_grid(
        names, lambda ax, name: panel_fn(fields_by_name[name], ax=ax), **grid_kwargs
    )


def plot_map_grid(
    fields_by_title: Mapping[str, xr.DataArray],
    *,
    scale: str = "each",
    extent: Any = None,
    ncol: int = 3,
    panel_size: tuple[float, float] | None = None,
    colorbar_label: str | None = None,
    **map_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One map per entry of *fields_by_title*, all in one frame.

    Parameters
    ----------
    fields_by_title:
        Panel title to field, in panel order. Each field is a map in the sense
        of :func:`sipnet_calibration.plotting.maps.plot_map`.
    scale:
        One of :data:`SCALE_OPTIONS`. ``"shared"`` resolves one color scale
        over every panel's values in the frame and draws one colorbar, or one
        legend, for the figure. ``"each"`` gives each panel its own, which
        suits panels of different quantities, such as a mean beside a
        standard deviation.
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
        If *fields_by_title* is empty or *scale* is not in
        :data:`SCALE_OPTIONS`, and whatever
        :func:`~sipnet_calibration.plotting.maps.check_field_is_a_map` raises
        for a panel, every panel checked before any is drawn.
    """
    check_fields_are_a_mapping(fields_by_title, message_name="fields_by_title")
    check_fields_are_given(
        fields_by_title, example="{title: field}", message_name="fields_by_title"
    )
    check_option_is_known(scale, SCALE_OPTIONS, message_name="scale")
    panel_fields = list(fields_by_title.values())
    # Every panel is checked before the frame and a shared scale read their
    # values, which a panel that is not a map would break with a raw error.
    for panel_field in panel_fields:
        maps.check_field_is_a_map(panel_field)
    bounds = maps.map_bounds(panel_fields, extent)
    color, map_kwargs = maps.split_color_keywords(map_kwargs)
    shared = maps.color_scale(panel_fields, bounds=bounds, **color) if scale == "shared" else None
    if panel_size is None:
        panel_size = _map_panel_size(bounds, own_key=shared is None)

    def draw(ax: Axes, title: str) -> None:
        if shared is None:
            maps.plot_map(fields_by_title[title], ax, extent=bounds, **color, **map_kwargs)
        else:
            maps.plot_map(
                fields_by_title[title], ax, extent=bounds, scale=shared, colorbar=False,
                **map_kwargs,
            )

    titles = list(fields_by_title)
    figure, axes = build_plot_grid(
        titles, draw, ncol=ncol, labels=titles, panel_size=panel_size, legend="none"
    )
    if shared is not None:
        _add_shared_key(figure, axes, shared, panel_fields, bounds, colorbar_label)
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
    """One map per value of *dim*: per batch label, or per time label.

    Parameters
    ----------
    field:
        A field that is a map at each value of *dim*.
    dim:
        The dimension to split on, such as ``"sample"`` or ``"time"``.
    values:
        The coordinate labels to draw, in order. ``None`` draws them all, or
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
    TypeError
        If *field* is not a ``DataArray``, or *n_max* is a boolean or not an
        integer.
    KeyError
        If *values* names a label *dim* does not hold.
    ValueError
        If *field* is not a field
        (:func:`sipnet_calibration.fields.validate_field`); if it has no
        *dim*, or a batch dim other than *dim* (with advice on each); if
        *n_max* is less than 1; and whatever :func:`plot_map_grid` raises for
        a panel.
    """
    validate_field(field)
    name = message_name(field)
    maps.check_field_has_the_dim(field, dim, message_name=name)
    maps.check_field_has_no_batch_dim_besides(field, dim, message_name=name)
    n_max = as_positive_integer(n_max, message_name="n_max")
    available = field[dim].values
    if values is None:
        chosen = available[thinned_indices(len(available), n_max)]
    else:
        chosen = _labels_asked_for(values)
        check_field_holds_the_labels(field, dim, chosen, message_name=name)
    # Titled by the label as the field holds it, so a Timestamp asked for is
    # titled as the date it selects.
    selected = [field.sel({dim: value}) for value in chosen]
    panels = {maps.coordinate_label(dim, panel[dim].values): panel for panel in selected}
    if len(panels) < len(selected):
        panels = {_full_title(dim, panel[dim].values): panel for panel in selected}
    return plot_map_grid(panels, scale=scale, **grid_kwargs)


def plot_map_quantiles(
    field: xr.DataArray,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    *,
    batch_dim: str = SAMPLE,
    scale: str = "shared",
    **grid_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """One map per quantile of *field* over one of its batch dims.

    Parameters
    ----------
    field:
        A continuous field that is a map at each label of *batch_dim*.
    quantiles:
        The quantiles, each in ``(0, 1)``. The default is the 90% central
        interval and the median, the outer band of
        :func:`~sipnet_calibration.plotting.primitives.fan`.
    batch_dim:
        The batch dim the quantiles are taken over.
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
    TypeError
        If *dim=* is passed: the batch dim is named with *batch_dim*.
    ValueError
        If *quantiles* is empty; if *batch_dim* is not a batch dim of
        *field*, or *field* has one other than *batch_dim* (with advice on
        each); and whatever
        :func:`~sipnet_calibration.plotting.maps.summarize_batch` raises.
    """
    check_keywords_are_not_retired(
        grid_kwargs, {"dim": "batch_dim="}, message_name="plot_map_quantiles"
    )
    quantiles = as_sequence(quantiles, message_name="quantiles")
    check_quantiles_are_given(quantiles)
    quantiles = [
        as_real_number(q, fix="pass quantiles in (0, 1)", message_name="quantiles")
        for q in quantiles
    ]
    for quantile in quantiles:
        maps.check_quantile_is_in_range(quantile, message_name="quantiles")
    validate_field(field)
    name = message_name(field)
    maps.check_batch_dim_is_the_fields(field, batch_dim, message_name=name)
    maps.check_field_has_no_batch_dim_besides(field, batch_dim, message_name=name)
    panels = {
        maps.quantile_label(q): maps.summarize_batch(field, q, batch_dim=batch_dim)
        for q in quantiles
    }
    grid_kwargs.setdefault("colorbar_label", axis_label(field))
    return plot_map_grid(panels, scale=scale, **grid_kwargs)


# ── private helpers ───────────────────────────────────────────────────────────


def _panel_titles(
    items: list[Any], labels: Sequence[str] | Callable[[Any], str] | None
) -> list[str] | None:
    """One title per item, or ``None`` when no titles were asked for."""
    if labels is None:
        return None
    if callable(labels):
        return [str(labels(item)) for item in items]
    check_labels_are_not_one_string(labels)
    titles = [str(title) for title in as_sequence(labels, message_name="labels")]
    check_labels_match_the_items(titles, items)
    return titles


def _labels_asked_for(values: Any) -> np.ndarray:
    """The labels of a ``values=`` argument, a sequence, as an array of the caller's elements."""
    if isinstance(values, Iterator):
        values = tuple(values)  # a generator can be read only once
    # as_sequence checks the form alone: it gives datetime64 elements back as
    # integers or dates, which no longer match a time coordinate.
    as_sequence(values, message_name="values")
    if not hasattr(values, "__array__"):
        values = tuple(values)
    return np.asarray(getattr(values, "values", values))


def _full_title(dim: str, label: Any) -> str:
    """A panel title giving *label* in full, for labels a shorter title does not tell apart."""
    if np.issubdtype(np.asarray(label).dtype, np.datetime64):
        return f"{dim} {pd.Timestamp(label)}"
    return f"{dim} {label}"


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


def _map_panel_size(bounds: maps.ProjectedBounds, *, own_key: bool) -> tuple[float, float]:
    """A panel size matching the frame's shape, with room for a colorbar."""
    width = 3.4
    aspect = (bounds.y_max - bounds.y_min) / (bounds.x_max - bounds.x_min)
    height = float(np.clip(width * aspect, 1.6, 5.0)) + 0.35
    return (width + (1.2 if own_key else 0.0), height)


def _add_shared_key(
    figure: Figure,
    axes: np.ndarray,
    scale: maps.ColorScale,
    panel_fields: Sequence[xr.DataArray],
    bounds: maps.ProjectedBounds,
    label: str | None,
) -> None:
    """One colorbar, or one legend of the classes present, for the figure."""
    if scale.class_names is None:
        figure.colorbar(
            scale.mappable(), ax=list(axes), shrink=0.8, label=label or scale.label
        )
        return
    figure.legend(
        handles=scale.legend_handles(scale.classes_in(panel_fields, bounds)),
        title=label or scale.label,
        loc="outside right center",
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_items_are_given(items: list[Any]) -> None:
    """A grid is given at least one item to draw a panel for."""
    if not items:
        raise ValueError("items is empty, so there is nothing to draw; pass one item per panel.")


def check_labels_match_the_items(titles: list[str], items: list[Any]) -> None:
    """There is one panel title per item."""
    if len(titles) != len(items):
        raise ValueError(
            f"labels has {len(titles)} entries and there are {len(items)} panels; they "
            "must match, one title per panel."
        )


def check_fields_are_a_mapping(fields: Any, *, message_name: str) -> None:
    """A grid of maps is given a mapping of panel title to field."""
    if not isinstance(fields, Mapping):
        raise TypeError(
            f"{message_name} must be a mapping of panel title to field, got "
            f"{type(fields).__name__}; pass {{title: field}}."
        )


def check_labels_are_not_one_string(labels: Any) -> None:
    """Panel titles are a sequence or a callable, not one string read a character at a time."""
    if isinstance(labels, str):
        raise TypeError(
            f"labels is the one string {labels!r}, which would title the panels a character "
            "each; pass one title per panel, or a callable."
        )


def check_fields_are_given(fields: Mapping[str, Any], *, example: str, message_name: str) -> None:
    """A grid of fields is given at least one; *example* is the mapping's form."""
    if not fields:
        raise ValueError(
            f"{message_name} is empty, so there is nothing to draw; pass {example}."
        )


def check_field_has_a_site_dim(field: xr.DataArray, *, message_name: str) -> None:
    """The field drawn one panel per site has a ``site`` dim."""
    if SITE not in field.dims:
        raise ValueError(
            f"{message_name}: panels are drawn one per site, and the field has no {SITE!r} "
            f"dim (its dims are {list(field.dims)}); draw it on one panel instead."
        )


def check_field_holds_the_labels(
    field: xr.DataArray, dim: str, labels: Sequence[Any], *, message_name: str
) -> None:
    """Every label asked for is on the field's *dim* coordinate."""
    missing = missing_labels(field, dim, labels)
    if missing:
        held = field.indexes[dim].tolist()  # as the field holds them: Timestamps, not ints
        raise KeyError(
            f"{message_name}: no such {dim} label(s) in the field: {truncated(missing)}; it "
            f"holds {len(held)}, {truncated(held)}, so ask for those."
        )


def check_quantiles_are_given(quantiles: Sequence[float]) -> None:
    """At least one quantile is asked for."""
    if not quantiles:
        raise ValueError(
            "quantiles is empty, so there is no map to draw; pass at least one, such as "
            "(0.05, 0.5, 0.95)."
        )


