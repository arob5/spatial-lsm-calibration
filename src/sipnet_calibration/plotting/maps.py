"""Maps.

Much of what this project calibrates varies over space: initial conditions,
observations, per-site parameters, the site labels a prior pools over. This
module draws one such quantity as a map over the site pool or a region of it,
whether it is a value per site, a class per site, or a surface on a
longitude/latitude grid.

:func:`plot_map` is the one-panel function. Like
:func:`~sipnet_calibration.plotting.series.plot_time_series`, it draws onto an
``Axes`` it is given and returns it. Grids of maps -- one per ensemble member,
per quantile, per time step -- are built by
:mod:`sipnet_calibration.plotting.facet`, and :func:`animate_map` plays a map
through time.

What it draws
-------------
The kind of map follows from the data, never from a mode keyword:

====================  ===================  =====================================
Kind                  Dimensions           Recognized by
====================  ===================  =====================================
values at sites       ``(site,)``          ``site`` dim, numeric values
classes at sites      ``(site,)``          a ``flag_meanings`` attribute, or
                                           string values
raster                ``(lat, lon)``       ``lat`` and ``lon`` are dimensions
====================  ===================  =====================================

A site map needs ``lon`` and ``lat`` as coordinates on ``site``, which the
readers in this project provide. A raster needs ``lat`` and ``lon`` as
one-dimensional, monotonic coordinates in degrees. Both need ``units`` and
``long_name`` in ``attrs`` unless they are categorical, which need only
``long_name``.

**Categorical fields** follow CF: integer codes, with ``flag_values`` and a
space-separated ``flag_meanings`` naming each code's class. An optional
``flag_display_names``, a tuple aligned with ``flag_meanings``, is what the
legend shows in their place; it is this project's attribute, not CF's.
:func:`sipnet_calibration.site_labels.site_labels_field` makes one from a
site-labels product. A class keeps its color in every map of the same product,
because colors are keyed by the class's position in ``flag_meanings`` and not
by which classes a map happens to show.

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
                    within ``radius_km``; otherwise it is left blank.
:class:`Triangles`  linear interpolation over the Delaunay triangulation,
                    omitting triangles with an edge over ``max_edge_km``.
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
:func:`member_summary`
    Reduce an ensemble to one statistic per site, keeping the attributes a map
    needs for its label.
:func:`animate_map`
    Play a field through one of its dimensions, on a fixed color scale.
:func:`map_bounds`, :func:`color_scale`
    The frame and the color scale for a set of fields, which the grids in
    :mod:`~sipnet_calibration.plotting.facet` share across panels.

Notes
-----
There is no ``stat`` keyword reducing the ensemble inside :func:`plot_map`: a
map of an array with a ``member`` or ``time`` dimension is refused, as
:func:`~sipnet_calibration.plotting.series.plot_time_series` refuses a
reduction it was not asked for. Averaging an ensemble is a choice, and it
should be visible where the map is asked for.

A Gaussian process, or any other model of a surface, is not fitted here. Its
predictions are data -- a raster, or values at sites -- and are mapped like
any other field.

Usage
-----
::

    from sipnet_calibration.plotting import plot_map, member_summary
    from sipnet_calibration.site_labels import site_labels_field

    plot_map(site_labels_field("reanalysis_3pft"))              # classes
    plot_map(wood.sel(member=0), extent="CONUS", log=True)      # one member
    plot_map(member_summary(wood, "median"), render="cells")    # a mosaic
    plot_map(residual, center=0.0, extent=(-90, 35, -75, 45))   # a region

    animation = animate_map(monthly_nee, "time", center=0.0)
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
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, LogNorm, Normalize
from matplotlib.patches import Patch

from sipnet_calibration.plotting import primitives
from sipnet_calibration.plotting.basemap import (
    DEFAULT_LAYERS,
    MAX_ANGULAR_DISTANCE,
    draw_basemap,
    draw_graticule,
)
from sipnet_calibration.plotting.style import axis_label, category_colors
from sipnet_calibration.projection import SITE_PROJECTION
from sipnet_calibration.sites import EXTENTS

__all__ = [
    "COLOR_KEYWORDS",
    "Cells",
    "ColorScale",
    "Points",
    "ProjectedBounds",
    "RENDERERS",
    "SITE_DIM",
    "SiteRenderer",
    "Triangles",
    "animate_map",
    "color_scale",
    "coordinate_label",
    "map_bounds",
    "member_summary",
    "plot_map",
    "quantile_label",
]

#: The dimension a site map is drawn over.
SITE_DIM = "site"

#: The keywords of :func:`plot_map` that decide the color scale, which the
#: grids and :func:`animate_map` resolve once for every panel or frame.
COLOR_KEYWORDS: tuple[str, ...] = (
    "cmap", "vmin", "vmax", "center", "log", "robust", "norm", "colors",
)


class ProjectedBounds(NamedTuple):
    """A frame in projected meters, ``(x_min, y_min, x_max, y_max)``.

    A distinct type so that ``extent`` can tell it from a box in degrees.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float


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
    """Each pixel takes the value of its nearest site within ``radius_km``.

    Parameters
    ----------
    radius_km:
        The farthest a colored pixel may be from its site, in kilometers
        measured in the projection. The projection's anisotropy stays under 1.3
        over the site pool (``tests/test_projection.py``), so this is within
        that factor of ground distance.
    pixels:
        Pixels across the frame.
    """

    radius_km: float = 50.0
    pixels: int = 800
    interpolates = False

    def draw(self, ax, x, y, values, *, bounds, **style):
        return primitives.site_cells(
            ax, x, y, values, radius=self.radius_km * 1e3, bounds=tuple(bounds),
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
    max_edge_km:
        Triangles with an edge longer than this, in projected kilometers, are
        not drawn, so no fill spans a wider gap between sites.
    shading:
        ``"gouraud"`` interpolates linearly; ``"flat"`` colors each triangle by
        the mean of its corners.
    """

    max_edge_km: float = 150.0
    shading: str = "gouraud"
    interpolates = True

    def draw(self, ax, x, y, values, *, bounds, **style):
        artist = primitives.site_triangles(
            ax, x, y, values, max_edge=self.max_edge_km * 1e3, shading=self.shading, **style
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
RENDERERS: Mapping[str, SiteRenderer] = {
    "points": Points(),
    "cells": Cells(),
    "triangles": Triangles(),
}


@dataclass(frozen=True)
class ColorScale:
    """A resolved color scale: what every panel drawn on it shares.

    ``categories`` and ``colors`` are set for a categorical scale and ``None``
    otherwise; ``label`` is the colorbar label or the legend title.
    ``display_names``, where set, is what the legend shows for each category.
    """

    cmap: Colormap
    norm: Normalize
    label: str
    categories: tuple[str, ...] | None = None
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
        positions = range(len(self.categories)) if present is None else sorted(present)
        names = self.display_names or self.categories
        return [Patch(facecolor=self.colors[i], label=names[i]) for i in positions]


def plot_map(
    field: xr.DataArray,
    ax: Axes | None = None,
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
        An ``xarray.DataArray`` of one of the kinds in the module docstring:
        values or classes on ``(site,)`` with ``lon``/``lat`` on ``site``, or a
        raster on ``(lat, lon)``.
    ax:
        The axes to draw on. If ``None``, a figure and axes are created.
    render:
        How to draw a site field: ``"points"`` (the default when ``None``),
        ``"cells"``, ``"triangles"``, or any :class:`SiteRenderer`, such as
        ``Cells(radius_km=25)``. Must be ``None`` for a raster.
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
        The axes drawn on, with its limits set to the frame, an equal aspect,
        and no ticks. The title is left alone.

    Raises
    ------
    ValueError
        If *field* is not one of the kinds above -- in particular if it has a
        ``member`` or ``time`` dimension, where the message names the functions
        that draw those; if *render* is unknown, is given for a raster, or
        interpolates a categorical field; if *extent* is not a known name or a
        valid box; or if *log* is asked for with a nonpositive value in the
        frame.
    """
    ax, _, _ = _draw_map(
        field, ax, render=render, extent=extent,
        color={"cmap": cmap, "vmin": vmin, "vmax": vmax, "center": center, "log": log,
               "robust": robust, "norm": norm, "colors": colors},
        colorbar=colorbar, basemap=basemap, graticule=graticule, style=style, scale=scale,
    )
    return ax


def member_summary(field: xr.DataArray, stat: str | float, *, dim: str = "member") -> xr.DataArray:
    """One statistic of *field* over its ensemble dimension, attributes kept.

    Parameters
    ----------
    field:
        A continuous field with a *dim* dimension.
    stat:
        ``"mean"``, ``"median"``, ``"standard_deviation"``, or a quantile in
        ``(0, 1)``. Missing values are skipped.
    dim:
        The dimension to reduce.

    Returns
    -------
    xarray.DataArray
        *field* without *dim*, keeping its name and ``units``, with a
        ``long_name`` saying what was taken, for example ``"Aboveground wood
        carbon, 5th percentile over members"``.

    Raises
    ------
    ValueError
        If *field* has no *dim*, is categorical, or *stat* is not one of the
        above.
    """
    if not isinstance(field, xr.DataArray) or dim not in field.dims:
        dims = list(getattr(field, "dims", ()))
        raise ValueError(f"member_summary needs a DataArray with a {dim!r} dimension; got {dims}")
    if _is_categorical(field):
        raise ValueError("a categorical field has no mean, median or quantiles of its codes")
    if isinstance(stat, str):
        reducers = {
            "mean": lambda: field.mean(dim, keep_attrs=True),
            "median": lambda: field.median(dim, keep_attrs=True),
            "standard_deviation": lambda: field.std(dim, keep_attrs=True),
        }
        if stat not in reducers:
            raise ValueError(
                f"stat must be one of {list(reducers)} or a quantile in (0, 1); got {stat!r}"
            )
        summary, description = reducers[stat](), stat.replace("_", " ")
    else:
        quantile = float(stat)
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"a quantile must lie in (0, 1); got {stat!r}")
        summary = field.quantile(quantile, dim, keep_attrs=True).drop_vars("quantile")
        description = quantile_label(quantile)
    long_name = field.attrs.get("long_name", field.name or "value")
    summary.attrs["long_name"] = f"{long_name}, {description} over {dim}s"
    summary.name = field.name
    return summary


def animate_map(
    field: xr.DataArray,
    dim: str = "time",
    *,
    ax: Axes | None = None,
    interval_ms: int = 250,
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
        The axes to draw on. If ``None``, a figure and axes are created.
    interval_ms:
        Milliseconds between frames.
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
    ValueError
        If *field* has no *dim*, or a single step of it is not a map.
    """
    if not isinstance(field, xr.DataArray) or dim not in field.dims:
        dims = list(getattr(field, "dims", ()))
        raise ValueError(f"animate_map needs a DataArray with a {dim!r} dimension; got {dims}")
    frames = [field.isel({dim: i}) for i in range(field.sizes[dim])]
    color, rest = _split_color_keywords(map_kwargs)
    bounds = map_bounds(frames, rest.pop("extent", None))
    scale = color_scale(frames, bounds=bounds, **color)

    ax, artist, renderer = _draw_map(
        frames[0], ax, render=rest.pop("render", None), extent=bounds, scale=scale,
        colorbar=rest.pop("colorbar", True), basemap=rest.pop("basemap", True),
        graticule=rest.pop("graticule", True), style=rest,
    )
    labels = [coordinate_label(dim, value) for value in field[dim].values]
    ax.set_title(labels[0])
    state = {"artist": artist}

    def show(position: int):
        values = _plotted_values(frames[position], scale)
        if renderer is None:
            state["artist"].set_array(np.ma.masked_invalid(values))
        else:
            state["artist"] = renderer.update(state["artist"], values)
        ax.set_title(labels[position])
        return (state["artist"],)

    return FuncAnimation(ax.figure, show, frames=len(frames), interval=interval_ms, blit=False)


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
    ValueError
        If *extent* is an unknown name or not a valid box, or ``None`` with
        nothing to fit to.
    """
    if isinstance(extent, ProjectedBounds):
        return extent
    if isinstance(extent, str):
        if extent not in EXTENTS:
            raise ValueError(f"unknown extent {extent!r}; the named extents are {list(EXTENTS)}")
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
    if x.size == 0:
        raise ValueError("there is nothing to fit a frame to; pass extent")
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
    categorical = [_is_categorical(field) for field in fields]
    if any(categorical) and not all(categorical):
        raise ValueError("a color scale cannot be shared by categorical and continuous fields")
    if all(categorical) and fields:
        return _categorical_scale(fields, colors)
    return _continuous_scale(
        fields, bounds, cmap=cmap, vmin=vmin, vmax=vmax, center=center, log=log,
        robust=robust, norm=norm,
    )


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
    """A panel or frame title for one value of *dim*: a date, or ``"member 3"``."""
    if np.issubdtype(np.asarray(value).dtype, np.datetime64):
        stamp = pd.Timestamp(value)
        return stamp.strftime("%Y-%m-%d") if stamp == stamp.normalize() else stamp.strftime("%Y-%m-%d %H:%M")
    return f"{dim} {value}"


# ── supporting helpers ────────────────────────────────────────────────────────

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


def _draw_map(field, ax, *, render, extent, colorbar, basemap, graticule, style,
              color=None, scale=None):
    """Draw *field*; return the axes, the data artist, and the renderer used.

    The scale is *scale* when given, as the grids and animations pass it, and
    otherwise resolved from *color* over this field alone.
    """
    is_raster = _check_map_field(field)
    renderer = None if is_raster else _resolved_renderer(render)
    if is_raster and render is not None:
        raise ValueError("render applies to site fields; a raster is drawn cell by cell")
    categorical = _is_categorical(field)
    if categorical and renderer is not None and renderer.interpolates:
        raise ValueError(
            "a categorical field cannot be interpolated between sites; use "
            "render='points' or render='cells'"
        )

    bounds = map_bounds([field], extent)
    if scale is None:
        scale = color_scale([field], bounds=bounds, **(color or {}))
    geometry = _geometry(field, scale)

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    _frame_axes(ax, bounds)
    if graticule:
        draw_graticule(ax)
    keywords = {"cmap": scale.cmap, "norm": scale.norm, "zorder": _DATA_ZORDER, **style}
    if is_raster:
        artist = primitives.raster(ax, geometry.x_corners, geometry.y_corners, geometry.values, **keywords)
    else:
        artist = renderer.draw(ax, geometry.x, geometry.y, geometry.values, bounds=bounds, **keywords)
    if basemap:
        draw_basemap(ax, layers=DEFAULT_LAYERS if basemap is True else basemap)
    if colorbar:
        _add_scale_key(ax, scale, geometry, bounds)
    # Drawing an image or a mesh can move the limits; the frame is the frame.
    _frame_axes(ax, bounds)
    return ax, artist, renderer


def _resolved_renderer(render) -> SiteRenderer:
    if render is None:
        return RENDERERS["points"]
    if isinstance(render, str):
        if render not in RENDERERS:
            raise ValueError(f"render must be one of {list(RENDERERS)} or a SiteRenderer; got {render!r}")
        return RENDERERS[render]
    if not (hasattr(render, "draw") and hasattr(render, "update")):
        raise ValueError(f"render must be one of {list(RENDERERS)} or a SiteRenderer; got {render!r}")
    return render


def _frame_axes(ax: Axes, bounds: ProjectedBounds) -> None:
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
    if scale.categories is None:
        # An inset rather than a stolen slice of the grid cell, so the bar
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
    present: set[int] = set()
    for geometry in geometries:
        values = geometry.values[geometry.in_frame(bounds)]
        present.update(int(v) for v in np.unique(values[np.isfinite(values)]))
    return sorted(present)


def _geometry(field: xr.DataArray, scale: ColorScale | None = None) -> _Geometry:
    """*field* projected, with values as the scale colors them."""
    if SITE_DIM in field.dims:
        x, y = SITE_PROJECTION.forward(field["lon"].values, field["lat"].values)
        return _Geometry(np.atleast_1d(x), np.atleast_1d(y), _plotted_values(field, scale))
    return _raster_geometry(field, _plotted_values(field, scale))


def _plotted_values(field: xr.DataArray, scale: ColorScale | None) -> np.ndarray:
    """The numbers a map colors: the values, or each class's position."""
    if SITE_DIM not in field.dims:
        field = field.transpose("lat", "lon")
    if not _is_categorical(field):
        values = np.asarray(field.values, dtype=float)
    else:
        categories = scale.categories if scale is not None and scale.categories else _categories(field)
        values = _class_positions(field, categories)
    if SITE_DIM not in field.dims:
        values = np.where(_raster_drawable(field), values, np.nan)
    return values


def _raster_geometry(field: xr.DataArray, values: np.ndarray) -> _Geometry:
    lat, lon = field["lat"].values.astype(float), field["lon"].values.astype(float)
    lon_corners, lat_corners = np.meshgrid(_cell_edges(lon), np.clip(_cell_edges(lat), -90, 90))
    _check_raster_avoids_antipode(lon_corners, lat_corners)
    x_corners, y_corners = SITE_PROJECTION.forward(lon_corners, lat_corners)
    lon_centers, lat_centers = np.meshgrid(lon, lat)
    x, y = SITE_PROJECTION.forward(lon_centers, lat_centers)
    return _Geometry(x, y, values, x_corners, y_corners)


def _raster_drawable(field: xr.DataArray) -> np.ndarray:
    """Which cells of a ``(lat, lon)`` raster are near enough the center to draw."""
    lon_centers, lat_centers = np.meshgrid(field["lon"].values, field["lat"].values)
    return SITE_PROJECTION.angular_distance(lon_centers, lat_centers) <= MAX_ANGULAR_DISTANCE


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges for monotonic *centers*: midpoints, and half a step beyond each end."""
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    middle = (centers[:-1] + centers[1:]) / 2
    return np.concatenate([[2 * centers[0] - middle[0]], middle, [2 * centers[-1] - middle[-1]]])


def _is_categorical(field: xr.DataArray) -> bool:
    return "flag_meanings" in field.attrs or np.asarray(field.values).dtype.kind in "OUS"


def _categories(field: xr.DataArray) -> tuple[str, ...]:
    if "flag_meanings" in field.attrs:
        return tuple(str(field.attrs["flag_meanings"]).split())
    values = np.asarray(field.values).ravel()
    present = {str(v) for v in values if isinstance(v, str) and v != ""}
    return tuple(sorted(present))


def _display_names(field: xr.DataArray, categories: tuple[str, ...]) -> tuple[str, ...] | None:
    if "flag_display_names" not in field.attrs:
        return None
    names = tuple(str(name) for name in field.attrs["flag_display_names"])
    _check_display_names_match(names, categories)
    return names


def _class_positions(field: xr.DataArray, categories: tuple[str, ...]) -> np.ndarray:
    """Each value's position in *categories*, as floats, ``NaN`` where missing."""
    values = np.asarray(field.values)
    if "flag_meanings" not in field.attrs:
        lookup = {name: position for position, name in enumerate(categories)}
        flat = [lookup.get(str(v), np.nan) if isinstance(v, str) and v != "" else np.nan
                for v in values.ravel()]
        return np.asarray(flat, dtype=float).reshape(values.shape)
    codes = np.asarray(field.attrs.get("flag_values", np.arange(len(categories))), dtype=float).ravel()
    _check_flags_match(codes, categories)
    numeric = values.astype(float)
    positions = np.full(numeric.shape, np.nan)
    known = np.isfinite(numeric)
    matches = numeric[known][:, None] == codes[None, :]
    _check_codes_are_declared(matches, numeric[known], codes)
    positions[known] = matches.argmax(axis=1)
    return positions


def _categorical_scale(fields, colors) -> ColorScale:
    categories = _categories(fields[0])
    display_names = _display_names(fields[0], categories)
    for field in fields[1:]:
        if _categories(field) != categories:
            raise ValueError(
                "categorical fields sharing a color scale must have the same classes, in the "
                f"same order; got {categories} and {_categories(field)}"
            )
        if _display_names(field, categories) != display_names:
            raise ValueError(
                "categorical fields sharing a color scale must have the same "
                f"flag_display_names; got {display_names} and "
                f"{_display_names(field, categories)}"
            )
    palette = category_colors(len(categories)) if len(categories) else []
    for name, color in (colors or {}).items():
        if name not in categories:
            raise ValueError(f"colors names {name!r}, which is not a class; the classes are {categories}")
        palette[categories.index(name)] = color
    cmap = ListedColormap(palette) if palette else ListedColormap(["#999999"])
    count = max(len(categories), 1)
    norm = BoundaryNorm(np.arange(count + 1) - 0.5, count)
    label = fields[0].attrs.get("long_name") or (fields[0].name or "class")
    return ColorScale(cmap, norm, str(label), tuple(categories), tuple(palette), display_names)


def _continuous_scale(fields, bounds, *, cmap, vmin, vmax, center, log, robust, norm) -> ColorScale:
    label = axis_label(fields[0])
    if norm is not None:
        return ColorScale(plt.get_cmap(cmap or "viridis"), norm, label)
    if log and center is not None:
        raise ValueError("a logarithmic scale has no center; pass log or center, not both")
    values = _values_in_frame(fields, bounds)
    if log:
        _check_positive_for_log(values)
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


def _values_in_frame(fields, bounds) -> np.ndarray:
    kept = []
    for field in fields:
        geometry = _geometry(field)
        values = geometry.values[geometry.in_frame(bounds)]
        kept.append(values[np.isfinite(values)])
    return np.concatenate(kept) if kept else np.empty(0)


def _split_color_keywords(keywords: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    color = {k: v for k, v in keywords.items() if k in COLOR_KEYWORDS}
    rest = {k: v for k, v in keywords.items() if k not in COLOR_KEYWORDS}
    return color, rest


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


def _check_map_field(field: xr.DataArray) -> bool:
    """Raise unless *field* is a map; return whether it is a raster."""
    if not isinstance(field, xr.DataArray):
        raise ValueError(
            f"expected an xarray.DataArray, got {type(field).__name__}. Tables and "
            "files are converted by an adapter, such as "
            "sipnet_calibration.site_labels.site_labels_field."
        )
    dims = set(field.dims)
    if SITE_DIM in dims:
        _check_only(field, {SITE_DIM})
        for name in ("lon", "lat"):
            if name not in field.coords or field.coords[name].dims != (SITE_DIM,):
                raise ValueError(
                    f"a site map needs {name!r} as a coordinate on 'site'. The readers in "
                    "sipnet_calibration add it; for an array built by hand, join it from "
                    "sipnet_calibration.sites.load_sites()."
                )
        return False
    if {"lat", "lon"} <= dims:
        _check_only(field, {"lat", "lon"})
        for name in ("lat", "lon"):
            values = np.asarray(field[name].values, dtype=float)
            steps = np.diff(values)
            if values.ndim != 1 or not (np.all(steps > 0) or np.all(steps < 0)):
                raise ValueError(f"a raster's {name!r} coordinate must be one-dimensional and strictly monotonic")
        return True
    raise ValueError(
        f"a map needs a 'site' dimension, or 'lat' and 'lon' dimensions for a raster; "
        f"the array has {list(field.dims)}"
    )


def _check_only(field: xr.DataArray, allowed: set[str]) -> None:
    extra = [dim for dim in field.dims if dim not in allowed]
    if not extra:
        return
    advice = []
    if "member" in extra:
        advice.append(
            "for 'member', draw one map per member with facet.plot_map_by(field, "
            "'member'), quantile maps with facet.plot_map_quantiles(field), or reduce "
            "first with maps.member_summary(field, stat)"
        )
    if "time" in extra:
        advice.append(
            "for 'time', select a step with field.sel(time=...), aggregate with "
            "observation.time_alignment.aggregate_time, draw panels with facet.plot_map_by(field, 'time'), "
            "or play it with maps.animate_map(field)"
        )
    raise ValueError(
        f"a map draws one value per {' and '.join(sorted(allowed))}, but the array also "
        f"has {extra}" + ("; " + "; ".join(advice) if advice else "")
    )


def _check_raster_avoids_antipode(lon: np.ndarray, lat: np.ndarray) -> None:
    if np.any(SITE_PROJECTION.angular_distance(lon, lat) > 179.0):
        raise ValueError(
            "the raster reaches the antipode of the projection center, which cannot be "
            "projected; crop it to the region being mapped first, for example "
            "raster.sel(lat=slice(0, 90), lon=slice(-180, -10))"
        )


def _check_positive_for_log(values: np.ndarray) -> None:
    nonpositive = int(np.count_nonzero(values <= 0))
    if nonpositive:
        raise ValueError(
            f"a logarithmic scale needs positive values, but {nonpositive} value(s) in the "
            f"frame are zero or negative (the least is {values.min():g})"
        )


def _check_flags_match(codes: np.ndarray, categories: tuple[str, ...]) -> None:
    if len(codes) != len(categories):
        raise ValueError(
            f"flag_values has {len(codes)} codes and flag_meanings {len(categories)} "
            "classes; they must pair up"
        )


def _check_display_names_match(names: tuple[str, ...], categories: tuple[str, ...]) -> None:
    if len(names) != len(categories):
        raise ValueError(
            f"flag_display_names has {len(names)} names and flag_meanings "
            f"{len(categories)} classes; they must pair up"
        )


def _check_codes_are_declared(matches: np.ndarray, values: np.ndarray, codes: np.ndarray) -> None:
    undeclared = ~matches.any(axis=1)
    if undeclared.any():
        raise ValueError(
            f"code(s) {sorted(set(values[undeclared].tolist()))[:5]} are not among "
            f"flag_values {codes.tolist()}"
        )
