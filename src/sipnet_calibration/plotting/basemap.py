"""The geography drawn under a map: coastlines, boundaries, lakes, graticule.

Overview
--------
A spatial panel in :mod:`sipnet_calibration.plotting.maps` is drawn on a plain
``Axes`` in projected meters, so the reference geography has to be drawn onto
it too. This module owns that geography -- which Natural Earth layers are
used, how they are clipped, the file they are stored in -- and draws it, along
with a graticule of meridians and parallels.

It sits downstream of two scripts, and the dependency runs one way::

    naciscdn.org (Natural Earth 1:50m)
      -> scripts/raw_sources/download_natural_earth.py  data/raw/natural_earth/*.zip
      -> scripts/build_basemap.py                       plotting/basemap_data/natural_earth_50m.npz
      -> this module                                    load_basemap(), draw_basemap()

Both the archives and the built file are tracked, so neither script runs in a
normal working copy.

Input data
----------
``src/sipnet_calibration/plotting/basemap_data/natural_earth_50m.npz``
    The built basemap, read by :func:`load_basemap`, in the layout below.
    :func:`basemap_path` says where it is. It ships inside the package, so the
    plotting layer reads nothing under ``data/``.

Data model
----------
The file is a NumPy ``.npz`` archive. For each layer named in
:data:`BASEMAP_LAYERS`:

======================= ============= ==========================================
Array                   Dtype, shape  Meaning
======================= ============= ==========================================
``<layer>_vertices``    float32 (n,2) longitude, latitude in degrees, WGS 84
``<layer>_offsets``     int64 (p+1,)  part *i* is ``vertices[offsets[i]:offsets[i+1]]``
``<layer>_source_md5``  str ()        md5 of the Natural Earth archive it came from
======================= ============= ==========================================

and, once, the clip it was built under:

========================== ============= =======================================
Array                      Dtype, shape  Meaning
========================== ============= =======================================
``center``                 float64 (2,)  lon, lat of the projection center
``max_angular_distance``   float64 ()    :data:`MAX_ANGULAR_DISTANCE` at build
========================== ============= =======================================

Every part is an open polyline of at least two vertices, every vertex of which
lies within ``max_angular_distance`` degrees of ``center``. A polygon layer
(lakes) is stored as its rings, so it is drawn as an outline. Nothing is
missing: a part cut by the clip becomes two parts, never a ``NaN``.

:func:`load_basemap` refuses a file built for a different center or clip than
the current :data:`~sipnet_calibration.projection.SITE_PROJECTION` and
:data:`MAX_ANGULAR_DISTANCE`, naming the command that rebuilds it.

Functions
---------
:func:`draw_basemap`
    Add the chosen layers to an ``Axes`` as projected line collections.
:func:`draw_graticule`
    Add meridians and parallels over the current axes limits, labeled where
    they cross the bottom, left and top edges.
:func:`load_basemap`, :func:`write_basemap`, :func:`basemap_path`
    Read and write the file above, and say where it is.
:func:`clip_to_drawable`
    Cut polylines down to the parts near enough the center to draw; the build
    script and the graticule share it.
:func:`graticule_spacing`
    The graticule interval for a frame of a given size.

Notes
-----
**Why clip by angle from the center.** The one point the projection cannot
take is the antipode of its center, and a world layer reaches it. Keeping only
what lies within :data:`MAX_ANGULAR_DISTANCE` of the center stays far clear of
it while still covering every frame of the site pool, whose farthest site is
well inside that radius. The same rule crops the graticule.

**The layers are drawn above the data**, at :data:`BASEMAP_ZORDER`, so a site
cell that extends past a coast is still read against the coastline. The
graticule is drawn below it, at :data:`GRATICULE_ZORDER`.

**There is no north arrow**: projected north rotates by about 150 degrees
across the domain (:meth:`~sipnet_calibration.projection.Projection.factors`),
so the graticule is the honest orientation cue.

Usage
-----
::

    from sipnet_calibration.plotting.basemap import draw_basemap, draw_graticule

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)   # projected meters
    draw_graticule(ax)
    draw_basemap(ax, layers=("coastline", "borders"))
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from sipnet_calibration.projection import SITE_PROJECTION

__all__ = [
    "BASEMAP_LAYERS",
    "BASEMAP_ZORDER",
    "BasemapLayer",
    "DEFAULT_LAYERS",
    "GRATICULE_ZORDER",
    "MAX_ANGULAR_DISTANCE",
    "basemap_path",
    "clip_to_drawable",
    "draw_basemap",
    "draw_graticule",
    "graticule_spacing",
    "load_basemap",
    "write_basemap",
]


@dataclass(frozen=True)
class BasemapLayer:
    """One layer of the basemap: its Natural Earth source and how it is drawn.

    Parameters
    ----------
    name:
        The layer's key in :data:`BASEMAP_LAYERS` and its prefix in the file.
    source_file:
        The Natural Earth archive under ``data/raw/natural_earth/`` it is built
        from.
    color, linewidth:
        Its default line style, which keywords to :func:`draw_basemap`
        override.
    """

    name: str
    source_file: str
    color: str
    linewidth: float


#: The layers the basemap holds, in drawing order.
BASEMAP_LAYERS: Mapping[str, BasemapLayer] = MappingProxyType(
    {
        layer.name: layer
        for layer in (
            BasemapLayer("coastline", "ne_50m_coastline.zip", "0.25", 0.6),
            BasemapLayer("lakes", "ne_50m_lakes.zip", "0.35", 0.4),
            BasemapLayer("borders", "ne_50m_admin_0_boundary_lines_land.zip", "0.35", 0.5),
            BasemapLayer("states", "ne_50m_admin_1_states_provinces_lines.zip", "0.6", 0.3),
        )
    }
)

#: What :func:`draw_basemap` draws when not told otherwise: all of it.
DEFAULT_LAYERS: tuple[str, ...] = tuple(BASEMAP_LAYERS)

#: Degrees of arc from the projection center beyond which nothing is kept.
MAX_ANGULAR_DISTANCE = 100.0

#: Where the basemap and the graticule sit relative to the data, which
#: matplotlib draws at zorder 1 to 2.
BASEMAP_ZORDER = 3.0
GRATICULE_ZORDER = 0.5


def basemap_path() -> Path:
    """Where the built basemap is: inside the package, beside this module."""
    return Path(__file__).with_name("basemap_data") / "natural_earth_50m.npz"


def load_basemap(path: Path | str | None = None) -> dict[str, list[np.ndarray]]:
    """Read the built basemap.

    Parameters
    ----------
    path:
        The file; ``None`` reads :func:`basemap_path`, and caches it.

    Returns
    -------
    dict
        Layer name to a list of parts, each a ``float64`` array of shape
        ``(n, 2)`` holding longitude and latitude in degrees.

    Raises
    ------
    ValueError
        If a layer of :data:`BASEMAP_LAYERS` is missing from the file, or the
        file was built for a different center or clip than the current ones.
    """
    if path is None:
        return _default_basemap()
    return _read_basemap(Path(path))


def write_basemap(
    parts: Mapping[str, Sequence[np.ndarray]],
    source_md5: Mapping[str, str],
    path: Path | str,
) -> None:
    """Write a basemap in the layout of the module's data model.

    Parameters
    ----------
    parts:
        Layer name to its parts, each ``(n, 2)`` longitude and latitude, already
        clipped with :func:`clip_to_drawable`. Every layer of
        :data:`BASEMAP_LAYERS` must be present.
    source_md5:
        Layer name to the md5 of the archive it was read from.
    path:
        Where to write. The caller is responsible for writing to a partial path
        and renaming, as ``scripts/build_basemap.py`` does.

    Raises
    ------
    ValueError
        If a layer is missing, or a part is not ``(n, 2)`` with ``n >= 2``.
    """
    missing = [name for name in BASEMAP_LAYERS if name not in parts or name not in source_md5]
    if missing:
        raise ValueError(f"no parts or md5 for basemap layer(s) {missing}")
    arrays: dict[str, np.ndarray] = {
        "center": np.array([SITE_PROJECTION.lon_0, SITE_PROJECTION.lat_0]),
        "max_angular_distance": np.array(MAX_ANGULAR_DISTANCE),
    }
    for name in BASEMAP_LAYERS:
        layer_parts = [np.asarray(part, dtype=float) for part in parts[name]]
        bad = [p.shape for p in layer_parts if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2]
        if bad:
            raise ValueError(f"layer {name!r} has parts that are not (n >= 2, 2): {bad[:3]}")
        lengths = [len(part) for part in layer_parts]
        arrays[f"{name}_vertices"] = (
            np.concatenate(layer_parts).astype(np.float32)
            if layer_parts
            else np.empty((0, 2), np.float32)
        )
        arrays[f"{name}_offsets"] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        arrays[f"{name}_source_md5"] = np.array(source_md5[name])
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def clip_to_drawable(lon: np.ndarray, lat: np.ndarray) -> list[np.ndarray]:
    """The runs of a polyline that lie within the drawable distance of the center.

    Parameters
    ----------
    lon, lat:
        One polyline, as one-dimensional arrays of degrees.

    Returns
    -------
    list of numpy.ndarray
        Each run of consecutive vertices within :data:`MAX_ANGULAR_DISTANCE` of
        the projection center, as ``(n, 2)`` longitude and latitude, dropping
        runs of fewer than two vertices.
    """
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    keep = SITE_PROJECTION.angular_distance(lon, lat) <= MAX_ANGULAR_DISTANCE
    if not keep.any():
        return []
    # Boundaries of the runs of True in keep.
    edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))
    return [
        np.column_stack([lon[start:stop], lat[start:stop]])
        for start, stop in zip(edges[0::2], edges[1::2])
        if stop - start >= 2
    ]


def draw_basemap(
    ax: Axes, *, layers: Sequence[str] = DEFAULT_LAYERS, **style: Any
) -> list[LineCollection]:
    """Draw basemap layers onto *ax*, in projected meters.

    Parameters
    ----------
    ax:
        The axes, whose data coordinates are
        :data:`~sipnet_calibration.projection.SITE_PROJECTION` meters.
    layers:
        Names from :data:`BASEMAP_LAYERS`, drawn in the order given.
    **style:
        Passed to every ``LineCollection``, overriding each layer's own color
        and line width.

    Returns
    -------
    list of matplotlib.collections.LineCollection
        One per layer, in the order drawn.

    Raises
    ------
    ValueError
        If a name is not in :data:`BASEMAP_LAYERS`.

    Notes
    -----
    Drawing does not change the axes limits, so it can come before or after
    they are set.
    """
    if isinstance(layers, str):
        layers = (layers,)
    unknown = [name for name in layers if name not in BASEMAP_LAYERS]
    if unknown:
        raise ValueError(f"unknown basemap layer(s) {unknown}; the layers are {list(BASEMAP_LAYERS)}")
    drawn = []
    for name in layers:
        layer = BASEMAP_LAYERS[name]
        keywords = {
            "colors": layer.color,
            "linewidths": layer.linewidth,
            "zorder": BASEMAP_ZORDER,
            **style,
        }
        collection = LineCollection(_projected_layer(name), **keywords)
        ax.add_collection(collection, autolim=False)
        drawn.append(collection)
    return drawn


def draw_graticule(
    ax: Axes,
    *,
    spacing: float | None = None,
    labels: bool = True,
    **style: Any,
) -> LineCollection:
    """Draw meridians and parallels over *ax*'s current limits.

    Parameters
    ----------
    ax:
        The axes, in projected meters, with its limits already set: the
        spacing and the label positions are worked out from them.
    spacing:
        Degrees between lines. ``None`` takes :func:`graticule_spacing` of the
        frame.
    labels:
        Label each meridian where it crosses the bottom edge and each parallel
        where it crosses the left edge, or the top edge if it misses the left.
    **style:
        Passed to the ``LineCollection``.

    Returns
    -------
    matplotlib.collections.LineCollection
        The lines. The labels are ``Text`` artists added to *ax*.

    Raises
    ------
    ValueError
        If *spacing* is not finite and positive.
    """
    x_min, x_max = sorted(ax.get_xlim())
    y_min, y_max = sorted(ax.get_ylim())
    if spacing is None:
        spacing = graticule_spacing(max(x_max - x_min, y_max - y_min))
    if not (np.isfinite(spacing) and spacing > 0):
        raise ValueError(f"spacing must be finite and positive, got {spacing!r}")

    meridians = _graticule_lines(float(spacing), meridians=True)
    parallels = _graticule_lines(float(spacing), meridians=False)
    keywords = {"colors": "0.82", "linewidths": 0.4, "zorder": GRATICULE_ZORDER, **style}
    collection = LineCollection(
        [line for _, parts in meridians + parallels for line in parts], **keywords
    )
    ax.add_collection(collection, autolim=False)

    if labels:
        _label_crossings(
            ax, meridians, side="bottom", at=y_min, bounds=(x_min, x_max), longitude=True
        )
        on_left = _label_crossings(
            ax, parallels, side="left", at=x_min, bounds=(y_min, y_max), longitude=False
        )
        # Over a wide frame the northern parallels are arcs that leave through
        # the top edge rather than the left, and would otherwise go unlabeled.
        _label_crossings(
            ax, [line for line in parallels if line[0] not in on_left], side="top",
            at=y_max, bounds=(x_min, x_max), longitude=False,
        )
    return collection


def graticule_spacing(frame_size: float) -> float:
    """The graticule interval, in degrees, for a frame *frame_size* meters across."""
    for threshold, spacing in ((6.0e6, 20.0), (2.5e6, 10.0), (1.0e6, 5.0), (4.0e5, 2.0)):
        if frame_size > threshold:
            return spacing
    return 1.0


# ── supporting helpers ────────────────────────────────────────────────────────


@functools.cache
def _default_basemap() -> dict[str, list[np.ndarray]]:
    return _read_basemap(basemap_path())


def _read_basemap(path: Path) -> dict[str, list[np.ndarray]]:
    with np.load(path) as stored:
        _check_built_for_this_projection(stored, path)
        layers = {}
        for name in BASEMAP_LAYERS:
            if f"{name}_vertices" not in stored:
                raise ValueError(
                    f"{path} has no layer {name!r}; rebuild it with scripts/build_basemap.py"
                )
            vertices = stored[f"{name}_vertices"].astype(float)
            offsets = stored[f"{name}_offsets"]
            layers[name] = [vertices[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    return layers


@functools.cache
def _projected_layer(name: str) -> list[np.ndarray]:
    """A layer of the default basemap, projected once per process."""
    projected = []
    for part in _default_basemap()[name]:
        x, y = SITE_PROJECTION.forward(part[:, 0], part[:, 1])
        projected.append(np.column_stack([x, y]))
    return projected


#: Sampling interval along a graticule line, in degrees.
_GRATICULE_STEP = 0.25


@functools.cache
def _graticule_lines(spacing: float, *, meridians: bool) -> list[tuple[float, list[np.ndarray]]]:
    """Each graticule line's value, and its drawable parts in projected meters."""
    if meridians:
        values = np.arange(-180.0, 180.0, spacing)
        along = np.arange(-90.0, 90.0 + _GRATICULE_STEP / 2, _GRATICULE_STEP)
    else:
        values = np.arange(-90.0 + spacing, 90.0, spacing)
        along = np.arange(-180.0, 180.0 + _GRATICULE_STEP / 2, _GRATICULE_STEP)
    lines = []
    for value in values:
        constant = np.full_like(along, value)
        lon, lat = (constant, along) if meridians else (along, constant)
        parts = []
        for part in clip_to_drawable(lon, lat):
            x, y = SITE_PROJECTION.forward(part[:, 0], part[:, 1])
            parts.append(np.column_stack([x, y]))
        if parts:
            lines.append((float(value), parts))
    return lines


def _label_crossings(
    ax, lines, *, side: str, at: float, bounds: tuple[float, float], longitude: bool
) -> set[float]:
    """Label each line where it crosses one edge of the frame; return those labeled.

    *side* is ``"bottom"`` or ``"top"``, for crossings of the horizontal line
    ``y = at`` placed by their x, or ``"left"``, for crossings of ``x = at``
    placed by their y.
    """
    across, along = (0, 1) if side == "left" else (1, 0)
    offset_points, alignment = {
        "bottom": ((0, -3), {"ha": "center", "va": "top"}),
        "top": ((0, 3), {"ha": "center", "va": "bottom"}),
        "left": ((-3, 0), {"ha": "right", "va": "center"}),
    }[side]
    minimum_gap = 0.04 * (bounds[1] - bounds[0])
    placed: list[float] = []
    labeled: set[float] = set()
    for value, parts in lines:
        for part in parts:
            offset = part[:, across] - at
            for i in np.flatnonzero(offset[:-1] * offset[1:] < 0):
                t = offset[i] / (offset[i] - offset[i + 1])
                position = part[i, along] + t * (part[i + 1, along] - part[i, along])
                if not bounds[0] <= position <= bounds[1]:
                    continue
                if any(abs(position - p) < minimum_gap for p in placed):
                    continue
                placed.append(position)
                labeled.add(value)
                point = (at, position) if side == "left" else (position, at)
                ax.annotate(
                    _format_degrees(value, longitude=longitude),
                    point, xytext=offset_points, textcoords="offset points", fontsize=7,
                    annotation_clip=False, **alignment,
                )
    return labeled


def _format_degrees(value: float, *, longitude: bool) -> str:
    """``120°W``, ``45°N``, ``0°``, ``180°``."""
    magnitude = f"{abs(value):g}°"
    if value == 0 or (longitude and abs(value) == 180):
        return magnitude
    if longitude:
        return magnitude + ("E" if value > 0 else "W")
    return magnitude + ("N" if value > 0 else "S")


# ── checks ────────────────────────────────────────────────────────────────────


def _check_built_for_this_projection(stored, path: Path) -> None:
    """Raise unless *stored* was clipped for the current center and radius."""
    center = tuple(float(v) for v in stored["center"])
    expected = (SITE_PROJECTION.lon_0, SITE_PROJECTION.lat_0)
    radius = float(stored["max_angular_distance"])
    if center != expected or radius != MAX_ANGULAR_DISTANCE:
        raise ValueError(
            f"{path} was built for center {center} and a {radius}-degree clip, but the "
            f"projection is centered at {expected} and the clip is "
            f"{MAX_ANGULAR_DISTANCE} degrees. Rebuild it with scripts/build_basemap.py."
        )
