"""Maps.

Much of what this project calibrates varies over space: initial conditions,
observations, per-site parameters, the site labels a prior pools over. This
module draws one such quantity as a map over the site pool or a region of it,
whether it is a value per site, a class per site, or a surface on a
longitude/latitude grid.

:func:`plot_map` is the one-panel function. Like
:func:`~sipnet_calibration.plotting.series.plot_time_series`, it draws onto an
``Axes`` it is given and returns it. Grids of maps -- one per batch label,
per quantile, per timestep -- are built by
:mod:`sipnet_calibration.plotting.facet`, and :func:`animate_map` plays a map
through time.

What it draws
-------------
What map is drawn follows from the data, never from a mode keyword:

====================  ===================  =====================================
Map                   Dimensions           Recognized by
====================  ===================  =====================================
values at sites       ``(site,)``          ``site`` dim, numeric values
classes at sites      ``(site,)``          :func:`~sipnet_calibration.fields.is_categorical`
raster                ``(lat, lon)``       ``lat`` and ``lon`` are dimensions
====================  ===================  =====================================

Every map is of a field (:func:`sipnet_calibration.fields.validate_field`)
with a ``long_name``, which labels its colorbar or legend. A raster's ``lat``
and ``lon`` are one-dimensional and strictly monotonic, in degrees.

**Categorical fields** are those :func:`~sipnet_calibration.fields.is_categorical`
accepts. An optional ``flag_display_names``, a tuple aligned with
``flag_meanings``, is what the legend shows in their place; it is this
project's attribute, not CF's.
:func:`sipnet_calibration.site_labels.site_labels_field` makes one from a
site-labels data source. A class keeps its color in every map of the same source,
because colors are keyed by the class's position in ``flag_meanings`` and not
by which classes a map happens to show. Without ``flag_meanings`` the classes
are the ``flag_values`` codes, and ``false`` and ``true`` for a boolean field
(``run_succeeded``, ``driver_present``).

**Missing values.** A site that is in the field with a missing value is drawn
as missing: no marker, or a transparent cell. A site that is not in the field
is simply absent. To map a subset, select it; to show missingness, keep the
``NaN``.

How sites are drawn
-------------------
``render`` chooses, and each is a :class:`SiteRenderer`:

==================  =============================================================
:class:`Points`     one marker per site. The default.
:class:`Cells`      each pixel takes its nearest site's value, if that site is
                    within ``radius``; otherwise it is left blank.
:class:`Triangles`  linear interpolation over the Delaunay triangulation,
                    omitting triangles with an edge over ``max_edge``.
                    Continuous fields only.
==================  =============================================================

Neither :class:`Points` nor :class:`Cells` draws a value that is not some
site's value, and a sparse set of sites looks sparse under both.
:class:`Triangles` interpolates, and is there for when that is wanted.

The frame and the color scale
-----------------------------
``extent`` is ``None`` for a frame fitted to what is drawn, a name in
:data:`sipnet_calibration.sites.EXTENTS`, a ``(west, south, east, north)`` box
in degrees, or a :class:`ProjectedBounds`. Everything is drawn in
:data:`~sipnet_calibration.projection.SITE_PROJECTION`, on an equal aspect, with
the basemap and graticule of :mod:`sipnet_calibration.plotting.basemap`.

The color scale is taken from the values **inside the frame**. ``center``
makes it diverging and symmetric about a value, ``log`` makes it logarithmic,
and ``robust`` clips it to the 2nd-98th percentiles. The colormap is not chosen
from the sign of the data: a signed quantity needs ``center=0.0`` asked for.

Functions
---------
:func:`plot_map`
    One field, one map.
:func:`summarize_batch`
    Reduce a batch dim to one statistic per site, keeping the attributes a
    map needs for its label.
:func:`animate_map`
    Play a field through one of its dimensions, on a fixed color scale.
:func:`map_bounds`, :func:`color_scale`
    The frame and the color scale for a set of fields, which the grids in
    :mod:`~sipnet_calibration.plotting.facet` share across panels.
:func:`check_field_is_a_map`
    The check that a field is one of the maps above, which the grids and
    :func:`animate_map` run on every panel before a shared scale reads it.

Notes
-----
There is no ``stat`` keyword reducing the ensemble inside :func:`plot_map`: a
map of an array with a batch dim or a ``time`` dimension is refused, as
:func:`~sipnet_calibration.plotting.series.plot_time_series` refuses a
reduction it was not asked for. Averaging an ensemble is a choice, and it
should be visible where the map is asked for.

A Gaussian process, or any other model of a surface, is not fitted here. Its
predictions are data -- a raster, or values at sites -- and are mapped like
any other field.

Usage
-----
::

    import matplotlib.pyplot as plt

    from sipnet_calibration.plotting import animate_map, plot_map, summarize_batch
    from sipnet_calibration.site_labels import site_labels_field

    figure, ax = plt.subplots(layout="constrained")
    plot_map(site_labels_field("reanalysis_3pft"), ax)              # classes
    plot_map(wood.isel(initial_condition_member=0), ax, extent="CONUS", log=True)
    plot_map(summarize_batch(wood, "median", batch_dim="initial_condition_member"),
             ax, render="cells")                                    # a mosaic
    plot_map(residual, ax, center=0.0, extent=(-90, 35, -75, 45))   # a region

    animation = animate_map(monthly_nee, "time", ax=ax, center=0.0)
    animation.save("nee.gif", writer="pillow")
"""

from __future__ import annotations

import functools
import textwrap
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Protocol, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, LogNorm, Normalize
from matplotlib.patches import Patch

from sipnet_calibration.conventions import LAT, LON, SAMPLE, SITE, TIME
from sipnet_calibration.fields import batch_dims, is_categorical, message_name, validate_field
from sipnet_calibration.plotting import primitives
from sipnet_calibration.plotting.basemap import (
    DEFAULT_LAYER_NAMES,
    MAX_ANGULAR_DISTANCE,
    draw_basemap,
    draw_graticule,
)
from sipnet_calibration.plotting.style import axis_label, category_colors
from sipnet_calibration.projection import SITE_PROJECTION
from sipnet_calibration.sites import EXTENTS
from sipnet_calibration.validation import truncated

__all__ = [
    "COLOR_KEYWORDS",
    "RENDERERS",
    "Cells",
    "ColorScale",
    "Points",
    "ProjectedBounds",
    "SiteRenderer",
    "Triangles",
    "animate_map",
    "batch_dim_advice",
    "check_batch_dim_is_the_fields",
    "check_field_has_no_batch_dim_besides",
    "check_field_has_the_dim",
    "check_field_is_a_map",
    "color_scale",
    "coordinate_label",
    "map_bounds",
    "plot_map",
    "quantile_label",
    "split_color_keywords",
    "summarize_batch",
]

#: The keywords of :func:`plot_map` that decide the color scale, which the
#: grids and :func:`animate_map` resolve once for every panel or frame.
COLOR_KEYWORDS: tuple[str, ...] = (
    "cmap", "vmin", "vmax", "center", "log", "robust", "norm", "colors",
)


def plot_map(
    field: xr.DataArray,
    ax: Axes,
    *,
    render: str | SiteRenderer | None = None,
    extent: str | tuple[float, float, float, float] | ProjectedBounds | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    log: bool = False,
    robust: bool = False,
    norm: Normalize | None = None,
    colors: Mapping[str, str] | None = None,
    scale: ColorScale | None = None,
    colorbar: bool = True,
    basemap: bool | Sequence[str] = True,
    graticule: bool = True,
    **style: Any,
) -> Axes:
    """Draw *field* as a map on one ``Axes``.

    Parameters
    ----------
    field:
        A field holding the data of one of the maps in the module docstring:
        values or classes on ``(site,)``, or a raster on ``(lat, lon)``.
    ax:
        The axes to draw on.
    render:
        How to draw a site field: ``"points"`` (the default when ``None``),
        ``"cells"``, ``"triangles"``, or any :class:`SiteRenderer`, such as
        ``Cells(radius=25e3)``. Must be ``None`` for a raster.
    extent:
        The frame: ``None`` fits it to what is drawn; a key of
        :data:`~sipnet_calibration.sites.EXTENTS`; a ``(west, south, east,
        north)`` box in degrees; or a :class:`ProjectedBounds`.
    cmap:
        The colormap for a continuous field. Defaults to ``"viridis"``, or
        ``"RdBu_r"`` when *center* is given.
    vmin, vmax:
        Color limits. Default to the range of the values inside the frame.
    center:
        Make the scale diverging, symmetric about this value.
    log:
        A logarithmic scale; every value inside the frame must be positive.
    robust:
        Take the default limits from the 2nd and 98th percentiles.
    norm:
        A matplotlib ``Normalize``, which overrides *vmin*, *vmax*, *center*,
        *log* and *robust*.
    colors:
        For a categorical field, class name to color, overriding
        :func:`~sipnet_calibration.plotting.style.category_colors` for those
        classes.
    scale:
        A :class:`ColorScale` from :func:`color_scale`, shared with other maps.
        It overrides every color keyword above.
    colorbar:
        Add a colorbar, or for a categorical field a legend of the classes
        present in the frame.
    basemap:
        ``True`` draws every basemap layer, ``False`` none, and a sequence of
        names from :data:`~sipnet_calibration.plotting.basemap.BASEMAP_LAYERS`
        those layers.
    graticule:
        Draw meridians and parallels, labeled on the bottom and left edges.
    **style:
        Passed to the renderer's drawing function, and so to matplotlib.

    Returns
    -------
    matplotlib.axes.Axes
        *ax*, drawn on, with its limits set to the frame, an equal aspect,
        and no ticks. The title is left alone.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``, *ax* is not an ``Axes``, *render*
        is neither a renderer name nor a :class:`SiteRenderer`, or *basemap*
        is one layer name rather than a sequence of them.
    KeyError
        If *render* or *extent* is a name that is not a key of
        :data:`RENDERERS` or :data:`~sipnet_calibration.sites.EXTENTS`, or
        *basemap* names a layer that is not in
        :data:`~sipnet_calibration.plotting.basemap.BASEMAP_LAYERS`.
    ValueError
        If *field* is not a field or holds the data of none of the maps above
        -- in particular if it has a batch dim or a ``time`` dimension, where
        the message names the functions that draw those; if *render* is given
        for a raster, or interpolates a categorical field; if *extent* is not
        a valid box; or if *log* is asked for with a nonpositive value in the
        frame.
    """
    _draw_map(
        field, ax, render=render, extent=extent,
        color={"cmap": cmap, "vmin": vmin, "vmax": vmax, "center": center, "log": log,
               "robust": robust, "norm": norm, "colors": colors},
        colorbar=colorbar, basemap=basemap, graticule=graticule, style=style, scale=scale,
    )
    return ax


def summarize_batch(
    field: xr.DataArray, stat: str | float, *, batch_dim: str = SAMPLE
) -> xr.DataArray:
    """One statistic of *field* over one of its batch dims, attributes kept.

    Parameters
    ----------
    field:
        A continuous field with the batch dim *batch_dim*.
    stat:
        ``"mean"``, ``"median"``, ``"standard_deviation"``, or a quantile in
        ``(0, 1)``. Missing values are skipped.
    batch_dim:
        The batch dim to reduce, such as ``"sample"`` or
        ``"initial_condition_member"``.

    Returns
    -------
    xarray.DataArray
        *field* without *batch_dim*, keeping its name and ``units``, with a
        ``long_name`` saying what was taken, for example ``"Aboveground wood
        carbon, 5th percentile over initial_condition_member"``.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``.
    ValueError
        If *field* is not a field, *batch_dim* is not one of its batch dims,
        *field* is categorical, or *stat* is not one of the above.
    """
    validate_field(field)
    name = message_name(field)
    check_batch_dim_is_the_fields(field, batch_dim, message_name=name)
    check_field_is_continuous(field, message_name=name)
    check_stat_is_a_summary(stat)
    if isinstance(stat, str):
        reducers = {
            "mean": field.mean,
            "median": field.median,
            "standard_deviation": field.std,
        }
        summary = reducers[stat](batch_dim, keep_attrs=True)
        description = stat.replace("_", " ")
    else:
        quantile = float(stat)
        summary = field.quantile(quantile, batch_dim, keep_attrs=True).drop_vars("quantile")
        description = quantile_label(quantile)
    long_name = field.attrs.get("long_name", field.name or "value")
    summary.attrs["long_name"] = f"{long_name}, {description} over {batch_dim}"
    summary.name = field.name
    return summary


def animate_map(
    field: xr.DataArray,
    dim: str = TIME,
    *,
    ax: Axes,
    interval: float = 0.25,
    **map_kwargs: Any,
) -> FuncAnimation:
    """Play *field* through *dim*, one map per step, on one color scale.

    Parameters
    ----------
    field:
        A field that is a map, in the sense of :func:`plot_map`, at each value
        of *dim*.
    dim:
        The dimension to play through.
    ax:
        The axes to draw on, whose figure the animation plays in.
    interval:
        Seconds between frames.
    **map_kwargs:
        Passed to :func:`plot_map`. The color keywords are resolved once, over
        every frame, and ``extent=None`` fits the frame to all of them.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation. Keep a reference to it while it plays. Save it with
        ``animation.save("map.gif", writer="pillow")``, or show it in a
        notebook with ``IPython.display.HTML(animation.to_jshtml())``.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray`` or *ax* is not an ``Axes``.
    ValueError
        If *field* is not a field; if it has no *dim*, a *dim* of length zero,
        or a batch dim other than *dim*; or if a single step of it is not a
        map (:func:`check_field_is_a_map`), checked before any frame is drawn.
    """
    validate_field(field)
    primitives.check_ax_is_an_axes(ax)
    name = message_name(field)
    check_field_has_the_dim(field, dim, message_name=name)
    check_dim_has_steps_to_play(field, dim, message_name=name)
    check_field_has_no_batch_dim_besides(field, dim, message_name=name)
    frames = [field.isel({dim: i}) for i in range(field.sizes[dim])]
    # Every frame has the first's dims, so checking it checks them all. It has
    # to come before the shared scale and frame read their values, which
    # fail in xarray's words on a frame that is not a map.
    check_field_is_a_map(frames[0])
    color, rest = split_color_keywords(map_kwargs)
    bounds = map_bounds(frames, rest.pop("extent", None))
    scale = color_scale(frames, bounds=bounds, **color)

    artist, renderer = _draw_map(
        frames[0], ax, render=rest.pop("render", None), extent=bounds, scale=scale,
        colorbar=rest.pop("colorbar", True), basemap=rest.pop("basemap", True),
        graticule=rest.pop("graticule", True), style=rest,
    )
    titles = [coordinate_label(dim, value) for value in field[dim].values]
    ax.set_title(titles[0])
    state = {"artist": artist}

    def draw_frame(position: int):
        values = _plotted_values(frames[position], scale)
        if renderer is None:
            state["artist"].set_array(np.ma.masked_invalid(values))
        else:
            state["artist"] = renderer.update(state["artist"], values)
        ax.set_title(titles[position])
        return (state["artist"],)

    return FuncAnimation(
        ax.figure, draw_frame, frames=len(frames), interval=interval * 1e3, blit=False
    )


# ── renderers ─────────────────────────────────────────────────────────────────


class SiteRenderer(Protocol):
    """How values at sites become an artist.

    ``interpolates`` is ``True`` for a renderer that draws values between
    sites, which :func:`plot_map` refuses for a categorical field. ``draw``
    takes projected coordinates and returns the artist; ``update`` puts new
    values at the same sites, for a frame of :func:`animate_map`, and returns
    the artist to keep, which may be a new one.
    """

    interpolates: bool

    def draw(self, ax: Axes, x, y, values, *, bounds: ProjectedBounds, **style: Any) -> Any: ...

    def update(self, artist: Any, values) -> Any: ...


@dataclass(frozen=True)
class Points:
    """One marker per site.

    Parameters
    ----------
    size:
        Marker area in points squared. ``None`` sizes the markers from the axes
        area and the number of sites in the frame, so that a dense pool reads
        as a field and a sparse set stays legible.
    """

    size: float | None = None
    interpolates = False

    def draw(self, ax, x, y, values, *, bounds, **style):
        size = self.size if self.size is not None else _automatic_marker_area(ax, x, y, values, bounds)
        style.setdefault("s", size)
        artist = primitives.site_points(ax, x, y, values, **style)
        artist._site_xy = np.column_stack([x, y])
        return artist

    def update(self, artist, values):
        values = np.asarray(values, dtype=float)
        keep = np.isfinite(values)
        artist.set_offsets(artist._site_xy[keep])
        artist.set_array(values[keep])
        return artist


@dataclass(frozen=True)
class Cells:
    """Each pixel takes the value of its nearest site within ``radius``.

    Parameters
    ----------
    radius:
        The farthest a colored pixel may be from its site, in meters measured
        in the projection. The projection's anisotropy stays under 1.3
        over the site pool (``tests/test_projection.py``), so this is within
        that factor of ground distance.
    pixels:
        Pixels across the frame.
    """

    radius: float = 50e3
    pixels: int = 800
    interpolates = False

    def draw(self, ax, x, y, values, *, bounds, **style):
        return primitives.site_cells(
            ax, x, y, values, radius=self.radius, bounds=tuple(bounds),
            pixels=self.pixels, **style,
        )

    def update(self, artist, values):
        artist.set_data(primitives.cells_from_index(artist.site_index, values))
        return artist


@dataclass(frozen=True)
class Triangles:
    """Linear interpolation between sites, for continuous fields.

    Parameters
    ----------
    max_edge:
        Triangles with an edge longer than this, in projected meters, are not
        drawn, so no fill spans a wider gap between sites.
    shading:
        ``"gouraud"`` interpolates linearly; ``"flat"`` colors each triangle by
        the mean of its corners.
    """

    max_edge: float = 150e3
    shading: str = "gouraud"
    interpolates = True

    def draw(self, ax, x, y, values, *, bounds, **style):
        artist = primitives.site_triangles(
            ax, x, y, values, max_edge=self.max_edge, shading=self.shading, **style
        )
        artist._redraw = functools.partial(self.draw, ax, x, y, bounds=bounds, **style)
        return artist

    def update(self, artist, values):
        # Which triangles are masked depends on which values are missing, so a
        # new frame is a new triangulation rather than new colors on the old one.
        redraw = artist._redraw
        artist.remove()
        return redraw(values)


#: The renderers a string ``render`` names.
RENDERERS: Mapping[str, SiteRenderer] = frozendict(
    {
        "points": Points(),
        "cells": Cells(),
        "triangles": Triangles(),
    }
)


# ── frames and color scales ───────────────────────────────────────────────────


class ProjectedBounds(NamedTuple):
    """A frame in projected meters, ``(x_min, y_min, x_max, y_max)``.

    A distinct type so that ``extent`` can tell it from a box in degrees.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class ColorScale:
    """A resolved color scale: what every panel drawn on it shares.

    ``class_names`` and ``colors`` are set for a categorical scale and
    ``None`` otherwise; ``label`` is the colorbar label or the legend title.
    ``display_names``, where set, is what the legend shows for each class.
    """

    cmap: Colormap
    norm: Normalize
    label: str
    class_names: tuple[str, ...] | None = None
    colors: tuple[str, ...] | None = None
    display_names: tuple[str, ...] | None = None

    def mappable(self) -> ScalarMappable:
        """A ``ScalarMappable`` on this scale, for a colorbar."""
        return ScalarMappable(norm=self.norm, cmap=self.cmap)

    def classes_in(self, fields: Sequence[xr.DataArray], bounds: ProjectedBounds) -> list[int]:
        """Positions of the classes some field in *fields* has inside *bounds*."""
        return _present_classes([_geometry(field, self) for field in fields], bounds)

    def legend_handles(self, present: Sequence[int] | None = None) -> list[Patch]:
        """One patch per class, or per class in *present* (positions)."""
        positions = range(len(self.class_names)) if present is None else sorted(present)
        names = self.display_names or self.class_names
        return [Patch(facecolor=self.colors[i], label=names[i]) for i in positions]


def map_bounds(
    fields: Sequence[xr.DataArray],
    extent: str | tuple[float, float, float, float] | ProjectedBounds | None = None,
) -> ProjectedBounds:
    """The frame for *fields*, in projected meters.

    Parameters
    ----------
    fields:
        Map fields, as :func:`plot_map` takes them.
    extent:
        As :func:`plot_map` takes it. ``None`` fits the frame to every site
        and every drawable raster cell of every field, with a margin.

    Returns
    -------
    ProjectedBounds

    Raises
    ------
    KeyError
        If *extent* is a name that is not a key of
        :data:`~sipnet_calibration.sites.EXTENTS`.
    ValueError
        If *extent* is not a valid box, or ``None`` with nothing to fit to.
    """
    if isinstance(extent, ProjectedBounds):
        return extent
    if isinstance(extent, str):
        check_extent_name_is_known(extent)
        return ProjectedBounds(*SITE_PROJECTION.projected_bounds(EXTENTS[extent]))
    if extent is not None:
        return ProjectedBounds(*SITE_PROJECTION.projected_bounds(tuple(extent)))

    xs, ys = [], []
    for field in fields:
        geometry = _geometry(field)
        drawn = np.isfinite(geometry.values) if geometry.is_raster else slice(None)
        xs.append(np.ravel(geometry.x[drawn]))
        ys.append(np.ravel(geometry.y[drawn]))
    x, y = np.concatenate(xs), np.concatenate(ys)
    check_frame_has_something_to_fit(x)
    margin = max(_FRAME_MARGIN * max(np.ptp(x), np.ptp(y)), _MINIMUM_MARGIN)
    return ProjectedBounds(x.min() - margin, y.min() - margin, x.max() + margin, y.max() + margin)


def color_scale(
    fields: Sequence[xr.DataArray],
    *,
    bounds: ProjectedBounds,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    log: bool = False,
    robust: bool = False,
    norm: Normalize | None = None,
    colors: Mapping[str, str] | None = None,
) -> ColorScale:
    """One color scale for every field in *fields*, from their values in *bounds*.

    Parameters
    ----------
    fields:
        Map fields, all categorical over the same classes or all continuous.
    bounds:
        The frame; only values drawn inside it set the limits.
    cmap, vmin, vmax, center, log, robust, norm, colors:
        As :func:`plot_map` takes them.

    Returns
    -------
    ColorScale

    Raises
    ------
    ValueError
        If the fields mix categorical and continuous, or categorical fields
        disagree on their classes; if *colors* names an unknown class; if
        *log* is combined with *center* or meets a nonpositive value.
    """
    categorical = [is_categorical(field) for field in fields]
    check_fields_are_all_categorical_or_all_continuous(categorical)
    if all(categorical) and fields:
        return _categorical_scale(fields, colors)
    return _continuous_scale(
        fields, bounds, cmap=cmap, vmin=vmin, vmax=vmax, center=center, log=log,
        robust=robust, norm=norm,
    )


# ── titles, advice and keywords ───────────────────────────────────────────────


def quantile_label(quantile: float) -> str:
    """``"median"`` for 0.5, else the ordinal percentile: ``"5th percentile"``."""
    if quantile == 0.5:
        return "median"
    percent = round(quantile * 100, 6)
    number = f"{percent:g}"
    whole = int(percent) if float(percent).is_integer() else None
    if whole is not None and 10 <= whole % 100 <= 20:
        suffix = "th"
    else:
        suffix = {"1": "st", "2": "nd", "3": "rd"}.get(number[-1], "th")
    return f"{number}{suffix} percentile"


def coordinate_label(dim: str, value: Any) -> str:
    """A panel or frame title for one value of *dim*: a date, or ``"sample 3"``."""
    if np.issubdtype(np.asarray(value).dtype, np.datetime64):
        stamp = pd.Timestamp(value)
        return stamp.strftime("%Y-%m-%d") if stamp == stamp.normalize() else stamp.strftime("%Y-%m-%d %H:%M")
    return f"{dim} {value}"


def batch_dim_advice(dims: Sequence[str]) -> list[str]:
    """What to do with each batch dim in *dims* before mapping, one sentence each."""
    return [
        f"for the batch dim {dim!r}, draw one map per label with "
        f"facet.plot_map_by(field, {dim!r}), quantile maps with "
        f"facet.plot_map_quantiles(field, batch_dim={dim!r}), select one with "
        f"field.isel({dim}=0), or reduce first with "
        f"maps.summarize_batch(field, stat, batch_dim={dim!r})"
        for dim in dims
    ]


def split_color_keywords(keywords: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """*keywords* split into the :data:`COLOR_KEYWORDS` and the rest."""
    color = {k: v for k, v in keywords.items() if k in COLOR_KEYWORDS}
    rest = {k: v for k, v in keywords.items() if k not in COLOR_KEYWORDS}
    return color, rest


# ── private helpers ───────────────────────────────────────────────────────────

#: Margin around a fitted frame, as a fraction of its larger side, and the
#: least margin in meters, for a frame around a single site.
_FRAME_MARGIN = 0.03
_MINIMUM_MARGIN = 100e3

#: Percentiles giving the limits of a ``robust`` scale.
_ROBUST_PERCENTILES = (2.0, 98.0)

#: Characters per line of a panel's colorbar label, which runs along a map's
#: height and would otherwise overrun it.
_LABEL_WIDTH = 40

#: Where data artists sit: above the graticule, below the basemap.
_DATA_ZORDER = 1.5

#: Automatic marker area, in points squared: the share of the axes the markers
#: cover in total, and the smallest and largest a marker may be.
_MARKER_COVERAGE = 0.35
_MARKER_AREA_LIMITS = (2.0, 30.0)

#: The statistics :func:`summarize_batch` takes by name.
_SUMMARY_NAMES: tuple[str, ...] = ("mean", "median", "standard_deviation")

#: The classes of a boolean field, in the order of their codes.
_BOOLEAN_CLASSES: tuple[str, ...] = ("false", "true")


@dataclass(frozen=True)
class _Geometry:
    """A field's drawable form: projected positions and the values to color.

    For a site field ``x``, ``y`` and ``values`` are per site. For a raster they
    are per cell, ``(m, n)``, with the cell corners ``(m + 1, n + 1)`` beside
    them. Categorical values are class positions, as floats with ``NaN``.
    """

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    x_corners: np.ndarray | None = None
    y_corners: np.ndarray | None = None

    @property
    def is_raster(self) -> bool:
        return self.x_corners is not None

    def in_frame(self, bounds: ProjectedBounds) -> np.ndarray:
        return (
            (self.x >= bounds.x_min) & (self.x <= bounds.x_max)
            & (self.y >= bounds.y_min) & (self.y <= bounds.y_max)
        )


def _draw_map(
    field: xr.DataArray,
    ax: Axes,
    *,
    render: str | SiteRenderer | None,
    extent: str | tuple[float, float, float, float] | ProjectedBounds | None,
    colorbar: bool,
    basemap: bool | Sequence[str],
    graticule: bool,
    style: dict[str, Any],
    color: dict[str, Any] | None = None,
    scale: ColorScale | None = None,
) -> tuple[Any, SiteRenderer | None]:
    """Draw *field* on *ax*; return the data artist and the renderer used.

    The scale is *scale* when given, as the grids and animations pass it, and
    otherwise resolved from *color* over this field alone.
    """
    check_field_is_a_map(field)
    primitives.check_ax_is_an_axes(ax)
    is_raster = SITE not in field.dims
    renderer = None if is_raster else _renderer_for(render)
    if is_raster:
        check_render_is_not_given_for_a_raster(render)
    if is_categorical(field) and renderer is not None:
        check_renderer_draws_only_site_values(renderer)

    bounds = map_bounds([field], extent)
    if scale is None:
        scale = color_scale([field], bounds=bounds, **(color or {}))
    geometry = _geometry(field, scale)

    _frame_axes(ax, bounds)
    if graticule:
        draw_graticule(ax)
    keywords = {"cmap": scale.cmap, "norm": scale.norm, "zorder": _DATA_ZORDER, **style}
    if is_raster:
        artist = primitives.raster(ax, geometry.x_corners, geometry.y_corners, geometry.values, **keywords)
    else:
        artist = renderer.draw(ax, geometry.x, geometry.y, geometry.values, bounds=bounds, **keywords)
    if basemap:
        draw_basemap(ax, layer_names=DEFAULT_LAYER_NAMES if basemap is True else basemap)
    if colorbar:
        _add_scale_key(ax, scale, geometry, bounds)
    # Drawing an image or a mesh can move the limits; the frame is the frame.
    _frame_axes(ax, bounds)
    return artist, renderer


def _renderer_for(render: str | SiteRenderer | None) -> SiteRenderer:
    """The renderer *render* names or is; points when ``None``."""
    if render is None:
        return RENDERERS["points"]
    if isinstance(render, str):
        check_renderer_name_is_known(render)
        return RENDERERS[render]
    check_render_is_a_renderer(render)
    return render


def _frame_axes(ax: Axes, bounds: ProjectedBounds) -> None:
    """Set *ax* to the frame: its limits, an equal aspect, no ticks, a thin border."""
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)


def _add_scale_key(ax: Axes, scale: ColorScale, geometry: _Geometry, bounds: ProjectedBounds) -> None:
    """A colorbar for a continuous scale, a legend of present classes otherwise."""
    if scale.class_names is None:
        # An inset rather than a stolen slice of the panel, so the bar
        # matches the map's height after the equal aspect has shrunk it.
        colorbar_axes = ax.inset_axes([1.03, 0.0, 0.035, 1.0])
        ax.figure.colorbar(
            scale.mappable(), cax=colorbar_axes, label=textwrap.fill(scale.label, _LABEL_WIDTH)
        )
        return
    ax.legend(
        handles=scale.legend_handles(_present_classes([geometry], bounds)),
        title=scale.label, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )


def _present_classes(geometries: Sequence[_Geometry], bounds: ProjectedBounds) -> list[int]:
    """The class positions some geometry has inside *bounds*, ascending."""
    present: set[int] = set()
    for geometry in geometries:
        values = geometry.values[geometry.in_frame(bounds)]
        present.update(int(v) for v in np.unique(values[np.isfinite(values)]))
    return sorted(present)


def _geometry(field: xr.DataArray, scale: ColorScale | None = None) -> _Geometry:
    """*field* projected, with values as the scale colors them."""
    if SITE in field.dims:
        x, y = SITE_PROJECTION.forward(field[LON].values, field[LAT].values)
        return _Geometry(np.atleast_1d(x), np.atleast_1d(y), _plotted_values(field, scale))
    return _raster_geometry(field, _plotted_values(field, scale))


def _plotted_values(field: xr.DataArray, scale: ColorScale | None) -> np.ndarray:
    """The numbers a map colors: the values, or each class's position."""
    if SITE not in field.dims:
        field = field.transpose(LAT, LON)
    if not is_categorical(field):
        values = np.asarray(field.values, dtype=float)
    else:
        class_names = (
            scale.class_names if scale is not None and scale.class_names else _class_names_of(field)
        )
        values = _class_positions(field, class_names)
    if SITE not in field.dims:
        values = np.where(_raster_drawable(field), values, np.nan)
    return values


def _raster_geometry(field: xr.DataArray, values: np.ndarray) -> _Geometry:
    """A raster's projected cell centers and corners, with *values* per cell."""
    lat, lon = field[LAT].values.astype(float), field[LON].values.astype(float)
    lon_corners, lat_corners = np.meshgrid(_cell_edges(lon), np.clip(_cell_edges(lat), -90, 90))
    check_raster_avoids_the_antipode(lon_corners, lat_corners)
    x_corners, y_corners = SITE_PROJECTION.forward(lon_corners, lat_corners)
    lon_centers, lat_centers = np.meshgrid(lon, lat)
    x, y = SITE_PROJECTION.forward(lon_centers, lat_centers)
    return _Geometry(x, y, values, x_corners, y_corners)


def _raster_drawable(field: xr.DataArray) -> np.ndarray:
    """Which cells of a ``(lat, lon)`` raster are near enough the center to draw."""
    lon_centers, lat_centers = np.meshgrid(field[LON].values, field[LAT].values)
    return SITE_PROJECTION.angular_distance(lon_centers, lat_centers) <= MAX_ANGULAR_DISTANCE


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges for monotonic *centers*: midpoints, and half a step beyond each end."""
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    middle = (centers[:-1] + centers[1:]) / 2
    return np.concatenate([[2 * centers[0] - middle[0]], middle, [2 * centers[-1] - middle[-1]]])


def _class_names_of(field: xr.DataArray) -> tuple[str, ...]:
    """The classes of a categorical field, in the order their colors are keyed by."""
    if "flag_meanings" in field.attrs:
        return tuple(str(field.attrs["flag_meanings"]).split())
    if "flag_values" in field.attrs:
        return tuple(str(code) for code in np.asarray(field.attrs["flag_values"]).ravel())
    return _BOOLEAN_CLASSES


def _display_names_of(
    field: xr.DataArray, class_names: tuple[str, ...]
) -> tuple[str, ...] | None:
    """The field's ``flag_display_names``, one per class, or ``None``."""
    if "flag_display_names" not in field.attrs:
        return None
    names = tuple(str(name) for name in field.attrs["flag_display_names"])
    check_display_names_pair_with_the_classes(names, class_names, message_name=message_name(field))
    return names


def _class_positions(field: xr.DataArray, class_names: tuple[str, ...]) -> np.ndarray:
    """Each value's position in *class_names*, as floats, ``NaN`` where missing."""
    values = np.asarray(field.values)
    if "flag_values" not in field.attrs:
        return values.astype(float)  # a boolean mask: false 0, true 1
    name = message_name(field)
    codes = np.asarray(field.attrs["flag_values"], dtype=float).ravel()
    check_flag_values_pair_with_the_classes(codes, class_names, message_name=name)
    numeric = values.astype(float)
    positions = np.full(numeric.shape, np.nan)
    known = np.isfinite(numeric)
    matches = numeric[known][:, None] == codes[None, :]
    check_codes_are_declared(matches, numeric[known], codes, message_name=name)
    positions[known] = matches.argmax(axis=1)
    return positions


def _categorical_scale(
    fields: Sequence[xr.DataArray], colors: Mapping[str, str] | None
) -> ColorScale:
    """The one categorical scale *fields* share, with *colors* overriding the palette."""
    class_names = _class_names_of(fields[0])
    display_names = _display_names_of(fields[0], class_names)
    for field in fields[1:]:
        check_fields_share_the_classes(class_names, _class_names_of(field))
        check_fields_share_the_display_names(
            display_names, _display_names_of(field, class_names)
        )
    palette = category_colors(len(class_names)) if len(class_names) else []
    for class_name, class_color in (colors or {}).items():
        check_color_names_a_class(class_name, class_names)
        palette[class_names.index(class_name)] = class_color
    cmap = ListedColormap(palette) if palette else ListedColormap(["#999999"])
    count = max(len(class_names), 1)
    norm = BoundaryNorm(np.arange(count + 1) - 0.5, count)
    label = fields[0].attrs.get("long_name") or (fields[0].name or "class")
    return ColorScale(cmap, norm, str(label), tuple(class_names), tuple(palette), display_names)


def _continuous_scale(
    fields: Sequence[xr.DataArray],
    bounds: ProjectedBounds,
    *,
    cmap: str | Colormap | None,
    vmin: float | None,
    vmax: float | None,
    center: float | None,
    log: bool,
    robust: bool,
    norm: Normalize | None,
) -> ColorScale:
    """The one continuous scale *fields* share, from their values in *bounds*."""
    label = axis_label(fields[0])
    if norm is not None:
        return ColorScale(plt.get_cmap(cmap or "viridis"), norm, label)
    check_log_scale_has_no_center(log, center)
    values = _values_in_frame(fields, bounds)
    if log:
        check_values_are_positive_for_a_log_scale(values)
    if values.size:
        low, high = (np.percentile(values, _ROBUST_PERCENTILES) if robust
                     else (values.min(), values.max()))
    else:
        low, high = (1.0, 10.0) if log else (0.0, 1.0)
    low = float(low if vmin is None else vmin)
    high = float(high if vmax is None else vmax)
    if log:
        return ColorScale(plt.get_cmap(cmap or "viridis"), LogNorm(low, high), label)
    if center is not None:
        half = max(abs(low - center), abs(high - center)) or 1.0
        return ColorScale(plt.get_cmap(cmap or "RdBu_r"), Normalize(center - half, center + half), label)
    return ColorScale(plt.get_cmap(cmap or "viridis"), Normalize(low, high), label)


def _values_in_frame(fields: Sequence[xr.DataArray], bounds: ProjectedBounds) -> np.ndarray:
    """The finite values of every field drawn inside *bounds*, flattened."""
    kept = []
    for field in fields:
        geometry = _geometry(field)
        values = geometry.values[geometry.in_frame(bounds)]
        kept.append(values[np.isfinite(values)])
    return np.concatenate(kept) if kept else np.empty(0)


def _automatic_marker_area(ax, x, y, values, bounds) -> float:
    """A marker area such that the sites in the frame cover a fixed share of the axes."""
    figure_width, figure_height = ax.figure.get_size_inches()
    position = ax.get_position()
    area = (position.width * figure_width * 72) * (position.height * figure_height * 72)
    inside = (
        np.isfinite(np.asarray(values, dtype=float))
        & (x >= bounds.x_min) & (x <= bounds.x_max) & (y >= bounds.y_min) & (y <= bounds.y_max)
    )
    count = max(int(inside.sum()), 1)
    return float(np.clip(_MARKER_COVERAGE * area / count, *_MARKER_AREA_LIMITS))


# ── checks ────────────────────────────────────────────────────────────────────


def check_field_is_a_map(field: Any) -> None:
    """*field* is a map: a field on ``site`` alone, or a ``(lat, lon)`` raster."""
    check_field_is_a_dataarray_to_map(field)
    validate_field(field)
    name = message_name(field)
    check_field_has_a_spatial_dim_to_map(field, message_name=name)
    check_map_has_only_its_spatial_dims(field, message_name=name)
    check_raster_coordinates_are_monotonic(field, message_name=name)


def check_field_is_a_dataarray_to_map(field: Any) -> None:
    """A map is drawn from a ``DataArray``."""
    if not isinstance(field, xr.DataArray):
        raise TypeError(
            f"a map is drawn from a field, an xarray.DataArray, got {type(field).__name__}; "
            "convert a table or a file with an adapter, such as "
            "sipnet_calibration.site_labels.site_labels_field."
        )


def check_field_has_a_spatial_dim_to_map(field: xr.DataArray, *, message_name: str) -> None:
    """A map has a ``site`` dim, or ``lat`` and ``lon`` dims."""
    if SITE not in field.dims and not {LAT, LON} <= set(field.dims):
        raise ValueError(
            f"{message_name}: a map needs a {SITE!r} dimension, or {LAT!r} and {LON!r} "
            f"dimensions for a raster, and the field has {list(field.dims)}; select a "
            "batch or time label away, or map each one with facet.plot_map_by."
        )


def check_map_has_only_its_spatial_dims(field: xr.DataArray, *, message_name: str) -> None:
    """A map has no dim beyond its spatial ones, with advice for each extra dim."""
    allowed = {SITE} if SITE in field.dims else {LAT, LON}
    extra = [dim for dim in field.dims if dim not in allowed]
    if not extra:
        return
    advice = batch_dim_advice([d for d in batch_dims(field) if d in extra])
    if TIME in extra:
        advice.append(
            "for 'time', select a step with field.sel(time=...), aggregate with "
            "observation.time_alignment.aggregate_time, draw panels with "
            "facet.plot_map_by(field, 'time'), or play it with maps.animate_map(field, ax=ax)"
        )
    raise ValueError(
        f"{message_name}: a map draws one value per {' and '.join(sorted(allowed))}, and the "
        f"field also has {extra}" + ("; " + "; ".join(advice) if advice else "; select them away")
        + "."
    )


def check_raster_coordinates_are_monotonic(field: xr.DataArray, *, message_name: str) -> None:
    """A raster's ``lat`` and ``lon`` are one-dimensional and strictly monotonic."""
    if SITE in field.dims:
        return
    for name in (LAT, LON):
        values = np.asarray(field[name].values, dtype=float)
        steps = np.diff(values)
        if values.ndim != 1 or not (np.all(steps > 0) or np.all(steps < 0)):
            raise ValueError(
                f"{message_name}: a raster's {name!r} coordinate must be one-dimensional "
                "and strictly monotonic; sort it with .sortby, or regrid onto a regular "
                "lat/lon grid."
            )


def check_field_has_the_dim(field: xr.DataArray, dim: str, *, message_name: str) -> None:
    """The field that maps are drawn over *dim* of has *dim*."""
    if dim not in field.dims:
        raise ValueError(
            f"{message_name}: maps are drawn over {dim!r}, and the field has no such dim "
            f"(its dims are {list(field.dims)}); pass one it has, such as 'time' or a "
            "batch dim."
        )


def check_dim_has_steps_to_play(field: xr.DataArray, dim: str, *, message_name: str) -> None:
    """The dim an animation plays through has at least one step."""
    if field.sizes[dim] == 0:
        raise ValueError(
            f"{message_name}: there are no {dim!r} steps to play, the field's {dim!r} having "
            "length 0; select a non-empty range of it."
        )


def check_field_has_no_batch_dim_besides(
    field: xr.DataArray, dim: str, *, message_name: str
) -> None:
    """Maps drawn over *dim* are each one map: the field has no other batch dim."""
    others = [d for d in batch_dims(field) if d != dim]
    if others:
        raise ValueError(
            f"{message_name}: maps are drawn over {dim!r}, and the field also has the batch "
            f"dim(s) {others}, which a map does not draw; " + "; ".join(batch_dim_advice(others))
            + "."
        )


def check_batch_dim_is_the_fields(
    field: xr.DataArray, batch_dim: str, *, message_name: str
) -> None:
    """*batch_dim* is one of the field's batch dims."""
    have = list(batch_dims(field))
    if batch_dim not in have:
        advice = (
            f"pass batch_dim= naming one of {have}"
            if have
            else "a field without one is one map already; draw it with maps.plot_map(field, ax)"
        )
        raise ValueError(
            f"{message_name}: {batch_dim!r} is not a batch dim of the field (its batch dims "
            f"are {have}, its dims {list(field.dims)}); {advice}."
        )


def check_field_is_continuous(field: xr.DataArray, *, message_name: str) -> None:
    """A field summarized over a batch dim holds a quantity, not classes."""
    if is_categorical(field):
        raise ValueError(
            f"{message_name}: a categorical field has no mean, median or quantiles of its "
            "codes; map one label of it, or count its classes yourself."
        )


def check_stat_is_a_summary(stat: str | float) -> None:
    """*stat* is a statistic :func:`summarize_batch` takes: a name, or a quantile in (0, 1)."""
    if isinstance(stat, str):
        if stat not in _SUMMARY_NAMES:
            raise ValueError(
                f"stat must be one of {list(_SUMMARY_NAMES)} or a quantile in (0, 1), got "
                f"{stat!r}; pass one of them."
            )
    elif not 0.0 < float(stat) < 1.0:
        raise ValueError(f"stat: a quantile lies in (0, 1), got {stat!r}; pass one such as 0.05.")


def check_renderer_name_is_known(render: str) -> None:
    """A renderer named by a string is a key of :data:`RENDERERS`."""
    if render not in RENDERERS:
        raise KeyError(
            f"render must be one of {list(RENDERERS)} or a SiteRenderer, got {render!r}; "
            "pass one of them."
        )


def check_render_is_a_renderer(render: Any) -> None:
    """*render* is a :class:`SiteRenderer`: it has ``draw`` and ``update``."""
    if not (hasattr(render, "draw") and hasattr(render, "update")):
        raise TypeError(
            f"render must be one of {list(RENDERERS)} or a SiteRenderer, got "
            f"{type(render).__name__}; pass a name, or an object with draw and update."
        )


def check_render_is_not_given_for_a_raster(render: Any) -> None:
    """A raster is drawn cell by cell, never by a site renderer."""
    if render is not None:
        raise ValueError(
            "render applies to site fields, and a raster is drawn cell by cell; "
            "leave render=None."
        )


def check_renderer_draws_only_site_values(renderer: SiteRenderer) -> None:
    """A categorical field is drawn by a renderer that does not interpolate."""
    if renderer.interpolates:
        raise ValueError(
            "a categorical field cannot be interpolated between sites; use "
            "render='points' or render='cells'."
        )


def check_extent_name_is_known(extent: str) -> None:
    """A named extent is a key of :data:`~sipnet_calibration.sites.EXTENTS`."""
    if extent not in EXTENTS:
        raise KeyError(
            f"unknown extent {extent!r}; pass one of the named extents {list(EXTENTS)}, "
            "a (west, south, east, north) box, or a ProjectedBounds."
        )


def check_frame_has_something_to_fit(x: np.ndarray) -> None:
    """A fitted frame has at least one site or raster cell to fit around."""
    if x.size == 0:
        raise ValueError("there is nothing to fit a frame to; pass extent.")


def check_fields_are_all_categorical_or_all_continuous(categorical: Sequence[bool]) -> None:
    """Fields sharing a color scale are all categorical or all continuous."""
    if any(categorical) and not all(categorical):
        raise ValueError(
            "a color scale cannot be shared by categorical and continuous fields; "
            "map them on separate scales (scale='each')."
        )


def check_fields_share_the_classes(
    class_names: tuple[str, ...], other: tuple[str, ...]
) -> None:
    """Categorical fields sharing a color scale have the same classes, in order."""
    if other != class_names:
        raise ValueError(
            "categorical fields sharing a color scale must have the same classes, in the "
            f"same order, and they have {truncated(class_names)} and {truncated(other)}; "
            "map them on separate scales (scale='each')."
        )


def check_fields_share_the_display_names(
    display_names: tuple[str, ...] | None, other: tuple[str, ...] | None
) -> None:
    """Categorical fields sharing a color scale show their classes by the same names."""
    if other != display_names:
        raise ValueError(
            "categorical fields sharing a color scale must have the same "
            f"flag_display_names, and they have {truncated(display_names or ())} and "
            f"{truncated(other or ())}; map them on separate scales (scale='each')."
        )


def check_color_names_a_class(class_name: str, class_names: tuple[str, ...]) -> None:
    """A color override is keyed by one of the field's classes."""
    if class_name not in class_names:
        raise ValueError(
            f"colors names {class_name!r}, which is not a class; key colors by the classes "
            f"{truncated(class_names)}."
        )


def check_log_scale_has_no_center(log: bool, center: float | None) -> None:
    """A logarithmic scale is not also diverging about a center."""
    if log and center is not None:
        raise ValueError("a logarithmic scale has no center; pass log or center, not both.")


def check_values_are_positive_for_a_log_scale(values: np.ndarray) -> None:
    """Every value on a logarithmic scale is positive."""
    nonpositive = int(np.count_nonzero(values <= 0))
    if nonpositive:
        raise ValueError(
            f"a logarithmic scale needs positive values, and {nonpositive} value(s) in the "
            f"frame are zero or negative (the least is {values.min():g}); mask them with "
            ".where(field > 0), or drop log=True."
        )


def check_raster_avoids_the_antipode(lon: np.ndarray, lat: np.ndarray) -> None:
    """No raster cell reaches the antipode of the projection center."""
    if np.any(SITE_PROJECTION.angular_distance(lon, lat) > 179.0):
        raise ValueError(
            "the raster reaches the antipode of the projection center, which cannot be "
            "projected; crop it to the region being mapped first, for example "
            "raster.sel(lat=slice(0, 90), lon=slice(-180, -10))."
        )


def check_flag_values_pair_with_the_classes(
    codes: np.ndarray, class_names: tuple[str, ...], *, message_name: str
) -> None:
    """``flag_values`` holds one code per class of ``flag_meanings``."""
    if len(codes) != len(class_names):
        raise ValueError(
            f"{message_name}: flag_values has {len(codes)} codes and flag_meanings "
            f"{len(class_names)} classes; they must pair up, one code per class."
        )


def check_display_names_pair_with_the_classes(
    names: tuple[str, ...], class_names: tuple[str, ...], *, message_name: str
) -> None:
    """``flag_display_names`` holds one name per class of ``flag_meanings``."""
    if len(names) != len(class_names):
        raise ValueError(
            f"{message_name}: flag_display_names has {len(names)} names and flag_meanings "
            f"{len(class_names)} classes; they must pair up, one name per class."
        )


def check_codes_are_declared(
    matches: np.ndarray, values: np.ndarray, codes: np.ndarray, *, message_name: str
) -> None:
    """Every code a categorical field holds is one of its ``flag_values``."""
    undeclared = ~matches.any(axis=1)
    if undeclared.any():
        raise ValueError(
            f"{message_name}: code(s) {truncated(sorted(set(values[undeclared].tolist())))} "
            f"are not among flag_values {codes.tolist()}; declare every code the field holds."
        )
