"""The display projection for spatial figures, and the transform onto it.

Overview
--------
Site coordinates are geographic -- longitude and latitude on WGS 84 -- and are
not plottable as they stand: a cell of the site grid is about 921 m wide at the
south of the pool and a small fraction of that at the north
(``data/README.md``), so plotting degrees directly stretches the Arctic by more
than sevenfold and makes any density or heatmap panel misleading.

This module owns the one projection the project's spatial figures use. It holds
the parameters, builds a :class:`pyproj.CRS` from them, and provides the
transform, the axes bounds and the distortion measures that a spatial panel
needs. PROJ does the projection arithmetic; what lives here is the project's
choice of projection and the shape in which the rest of the code consumes it.

It sits between the site table and the plotting layer, and the dependency runs
one way::

    data/processed/sites/sites.csv
      -> sipnet_calibration.sites          load_sites() -> lon, lat
      -> this module                       SITE_PROJECTION.forward() -> x, y
      -> sipnet_calibration.plotting.maps  the spatial panels

Nothing here knows about matplotlib, an axes, or a figure. Equally, nothing in
``plotting/`` defines projection parameters: a panel that needs projected
coordinates calls :meth:`Projection.forward`, and a panel that needs axes limits
for a named region calls :meth:`Projection.projected_bounds` on a box from
:data:`sipnet_calibration.sites.EXTENTS`.

Input data
----------
This module reads nothing. The parameters are in code, for the same reason
:data:`~sipnet_calibration.sites.SITE_GRID` is: the definition and the transform
that implements it must not be able to disagree.

Output data
-----------
It writes two files, the interchange form for anything that is not this package
-- a colleague's QGIS session, an R script, a figure caption::

    src/sipnet_calibration/projections/north_america_laea.projjson
    src/sipnet_calibration/projections/north_america_laea.projstring

Both are **serialized by PROJ** from :data:`SITE_PROJECTION`, not hand-written,
so they are authoritative rather than a transcription. They are tracked, so a
fresh checkout has them, and :func:`check_definitions` compares them against
what the parameters serialize to now; the test suite calls it. Editing them by
hand is therefore a test failure, not a way to change the projection.

Data model
----------
:class:`Projection` is a projected CRS, as the parameters EPSG names for it:

======================== ============= ======================================
Field                    Type          Meaning
======================== ============= ======================================
``name``                 ``str``       CRS name, as written to the files
``lat_0``, ``lon_0``     ``float``     latitude/longitude of natural origin
``false_easting``        ``float``     meters added to x
``false_northing``       ``float``     meters added to y
``base_crs``             ``str``       CRS of the input coordinates
======================== ============= ======================================

:data:`SITE_PROJECTION` is the project's projection: a Lambert Azimuthal Equal
Area centered at 50 N, 100 W on WGS 84, in meters, with no false origin.

The method is not a field. Only Lambert Azimuthal Equal Area is expressible
here, named by :data:`LAEA_METHOD` and :data:`LAEA_METHOD_CODE`, because that is
the projection the project chose; a second method means a second class, not a
different argument. :data:`DEFINITION_STEM` is the file name stem of the two
interchange files.

**Units and axis order.** :meth:`Projection.forward` takes longitude first and
latitude second, which is the traditional GDAL and PROJ ordering rather than the
axis order EPSG:4326 formally declares, matching how the site table stores them
and what ``pyproj``'s ``always_xy=True`` selects. It returns easting and
northing in meters, in that order.

Functions
---------
:meth:`Projection.forward`
    Longitude and latitude in degrees to easting and northing in meters.

:meth:`Projection.projected_bounds`
    The projected bounding box of a longitude/latitude box, for axes limits.

:meth:`Projection.factors`
    Local distortion at given points: the scale factors, the angular
    deformation, the areal scale, and the rotation of projected north.

:meth:`Projection.crs`, :meth:`Projection.proj_string`, :meth:`Projection.projjson`
    The projection as a :class:`pyproj.CRS`, and its two serializations.

:attr:`Projection.antipode`
    The one point :meth:`Projection.forward` cannot project.

:func:`write_definitions` and :func:`check_definitions`
    Write the interchange files, and verify the tracked ones still match the
    parameters. ``python -m sipnet_calibration.projection --write`` regenerates
    them.

:func:`default_definition_dir` and :func:`definition_paths`
    Where those files live, and their paths by format.

Notes
-----
**Why this projection.** The choice, the alternatives, and the distortion
measured over the real 8000 sites are recorded in
`issue #4 <https://github.com/arob5/spatial-lsm-calibration/issues/4>`_. In
summary: every candidate considered is equal-area, which is the property the
plotting design requires, so the choice turns on shape. The site pool spans 7 N
to 82.5 N and 178.8 W to 20.0 W, which is outside the domain of use of any
Albers Equal Area Conic -- Snyder's guidance puts Albers at regions of
predominant east-west expanse -- and ESRI:102003, which the published reanalysis
figures used, reaches a maximum angular deformation of 107 degrees and a 9:1
local anisotropy at the northernmost sites. This projection holds angular
deformation under 14 degrees and anisotropy under 1.3 over the whole pool, which
``tests/test_projection.py`` asserts against the real site table.

**Anisotropy is not only cosmetic here.** The spatial renderer planned in
``plotting/maps.py`` is to triangulate *after* projecting, and a Delaunay
triangulation is not affine-invariant, so a strong local anisotropy would make
the mesh an artifact of the projection rather than of where the sites are. Its
long-edge mask threshold is a projected length too, which means one ground
distance only where the local scale is close to isotropic; :meth:`Projection.factors`
is how a caller turns one into the other.

**No datum transformation is involved.** The base CRS is WGS 84, matching the
site coordinates, so nothing is shifted. A NAD83-based definition, such as the
ESRI codes, would have raised the question, and the answer would have been that
it does not matter: NAD 83 and WGS 84 differ by about 2 m, and the full-domain
extent is 16,000 km across, so the shift is under a thousandth of a pixel at any
figure size anyone would render.

**North is not up, and not by a little.** Over the site pool the rotation of
projected north runs from -71 degrees on the Chukchi coast to +75 in northeast
Greenland, a spread of about 146 degrees; across the whole
``NORTH_AMERICA`` extent it runs from -78 to +79, and the two extremes are the
northwest and *northeast* corners rather than opposite ends of a diagonal. A
single north arrow is therefore not merely imprecise but wrong nearly
everywhere on such a figure; a graticule is the honest indicator.
:meth:`Projection.factors` reports the rotation at a point as
``meridian_convergence``, and ``tests/test_projection.py`` pins those extremes.
Mind its sign convention, which is PROJ's: see that method's Notes.

**There is no inverse transform**, because nothing in the spatial panels as
specified needs one: a graticule, an extent and a site marker are all forward.
PROJ has one, so adding it is a two-line method the day something wants to
report an axes position back in longitude and latitude, label the parallel that
leaves the left spine, or drive a raster renderer.

**A point outside the projection's domain raises**, rather than coming back as
``inf``. PROJ's default is to return infinity, which propagates into an axes
limit or into a triangulation as a silently dropped point, so the transform is
run with ``errcheck=True`` and the error is re-raised as a ``ValueError`` naming
what was wrong. The only such point here is the antipode of the center, at
:attr:`Projection.antipode`; no site is within 125 degrees of it.

**A longitude/latitude box does not project to a rectangle**, so axes limits
come from :meth:`Projection.projected_bounds` rather than from projecting the
four corners, which understate a box by hundreds of kilometers. That method
documents why, and ``tests/test_projection.py`` measures it.

Usage
-----
Project site coordinates::

    from sipnet_calibration.projection import SITE_PROJECTION
    from sipnet_calibration.sites import EXTENTS, load_sites

    sites = load_sites()
    x, y = SITE_PROJECTION.forward(sites["lon"].to_numpy(), sites["lat"].to_numpy())

Axes limits for a named region, in the same projected meters. Set the aspect
ratio to equal, or the equal-area property does not survive to the page::

    x_min, y_min, x_max, y_max = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

Turn a ground distance into the projected length a mask should use::

    factors = SITE_PROJECTION.factors(lon, lat)
    projected_threshold = 300e3 * factors.tissot_semimajor.max()

The definition, for anything outside this package::

    print(SITE_PROJECTION.proj_string())
    # +proj=laea +lat_0=50 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m ...

Regenerate the tracked interchange files after changing a parameter::

    python -m sipnet_calibration.projection --write
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyproj
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import LambertAzimuthalEqualAreaConversion

__all__ = [
    "DEFINITION_STEM",
    "LAEA_METHOD",
    "LAEA_METHOD_CODE",
    "Projection",
    "SITE_PROJECTION",
    "check_definitions",
    "default_definition_dir",
    "definition_paths",
    "write_definitions",
]

#: The EPSG coordinate operation method this class expresses, as the registry
#: names and numbers it. Carried so that the definition files say which method
#: their parameters belong to, and so that a reader of this module does not have
#: to infer it from ``+proj=laea``.
LAEA_METHOD = "Lambert Azimuthal Equal Area"
LAEA_METHOD_CODE = 9820


# ── the projection ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Projection:
    """A Lambert Azimuthal Equal Area projected CRS, and the transform onto it.

    Parameters
    ----------
    name:
        CRS name, written into the definition files. It does **not** name those
        files: :func:`definition_paths` builds them from :data:`DEFINITION_STEM`
        instead, so two projections written to one directory with the default
        stem overwrite each other.
    lat_0, lon_0:
        Latitude and longitude of the natural origin, in degrees. The point
        ``(lon_0, lat_0)`` maps to ``(false_easting, false_northing)``.
    false_easting, false_northing:
        Constants added to the projected coordinates, in meters.
    base_crs:
        The CRS the input coordinates are on, in any form
        :meth:`pyproj.CRS.from_user_input` accepts. Must be a geographic CRS,
        since :meth:`forward` takes degrees.

    Notes
    -----
    Frozen and hashable, so ``dataclasses.replace`` gives a variant and the
    built CRS and transformer can be cached per instance.
    """

    name: str
    lat_0: float
    lon_0: float
    false_easting: float = 0.0
    false_northing: float = 0.0
    base_crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat_0 <= 90.0:
            raise ValueError(f"lat_0 must be in [-90, 90], got {self.lat_0}")
        if not -360.0 <= self.lon_0 <= 360.0:
            raise ValueError(f"lon_0 must be in [-360, 360], got {self.lon_0}")
        # The false origin is added to every projected coordinate and written
        # into the definition files, so a NaN here would put "+x_0=nan" in a
        # file meant to be authoritative.
        for field, value in (
            ("false_easting", self.false_easting),
            ("false_northing", self.false_northing),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite, got {value}")
        # Checked here rather than left to first use. An unhashable base_crs
        # would otherwise fail in the transformer cache with "unhashable type",
        # and a projected one would fail inside PROJ with several kilobytes of
        # JSON -- neither of which names the field.
        # Hashability first, and explicitly: a dict is a form
        # CRS.from_user_input accepts, and a geographic one at that, so it
        # passes every check below and then fails in the cache with
        # "unhashable type: 'dict'" at the first transform.
        try:
            hash(self.base_crs)
        except TypeError as error:
            raise ValueError(
                f"base_crs must be hashable, since the built CRS is cached on it; "
                f"got {type(self.base_crs).__name__}. Pass an EPSG string, a PROJ "
                "string or a pyproj.CRS"
            ) from error
        try:
            base = pyproj.CRS.from_user_input(self.base_crs)
        except pyproj.exceptions.CRSError as error:
            raise ValueError(f"base_crs is not a CRS PROJ recognizes: {error}") from error
        if not base.is_geographic:
            raise ValueError(
                f"base_crs must be geographic, since forward() takes degrees; "
                f"{self.base_crs} is {base.type_name}"
            )

    # ── the transform ────────────────────────────────────────────────────────

    def crs(self) -> pyproj.CRS:
        """This projection as a :class:`pyproj.CRS`.

        Built from the parameters through PROJ's own conversion class, so the
        method and parameter codes in the serializations come from PROJ rather
        than from anything written here.
        """
        return _crs(self)

    def forward(self, lon, lat):
        """Longitude and latitude in degrees to easting and northing in meters.

        Parameters
        ----------
        lon, lat:
            Degrees, scalar or array-like, longitude first. Broadcast against
            each other. Longitudes are not wrapped: -190 and 170 are the same
            meridian and both are accepted.

        Returns
        -------
        tuple
            ``(x, y)`` in meters. Scalars in, scalars out; otherwise arrays of
            the broadcast shape, ``float64``.

        Raises
        ------
        ValueError
            If any coordinate is not finite, any latitude is outside
            ``[-90, 90]``, or any point lies outside the projection's domain --
            which for this projection means the antipode of its center, at
            :attr:`antipode`.

        Notes
        -----
        PROJ returns infinity rather than raising for a point it cannot project,
        and infinity propagates into an axes limit or into a triangulation as a
        silently dropped point. So the transform runs with ``errcheck=True`` and
        the resulting ``ProjError`` is re-raised as a ``ValueError``.
        """
        longitude, latitude, scalar = _check_coordinates(lon, lat)
        # Built outside the try: pyproj's CRSError is a subclass of ProjError,
        # so a definition that will not build would otherwise be re-raised as a
        # complaint about an input point that was fine.
        transformer = _transformer(self)
        try:
            x, y = transformer.transform(longitude, latitude, errcheck=True)
        except pyproj.exceptions.ProjError as error:
            raise ValueError(_outside_domain_message(self, error)) from error
        if scalar:
            return float(x), float(y)
        return np.asarray(x), np.asarray(y)

    def factors(self, lon, lat):
        """Local distortion at the given points.

        Parameters
        ----------
        lon, lat:
            Degrees, longitude first, as :meth:`forward` takes them.

        Returns
        -------
        pyproj.proj.Factors
            PROJ's own factors. The ones this project uses are
            ``tissot_semimajor`` and ``tissot_semiminor``, the extreme linear
            scale factors; ``angular_distortion``; ``areal_scale``, which is 1
            everywhere for this projection and is worth asserting rather than
            assuming; and ``meridian_convergence``, the rotation of projected
            north in degrees.

        Raises
        ------
        ValueError
            On the same inputs :meth:`forward` rejects, and additionally where
            PROJ cannot compute factors. That region is **wider than the point
            ``forward`` refuses**: it extends about a degree around the
            antipode, where PROJ returns infinity for every factor. Left
            unchecked, a single such point turns
            ``300e3 * factors.tissot_semimajor.max()`` into ``inf`` and a
            long-edge mask into one that masks nothing.

        Notes
        -----
        This is what turns a ground distance into a projected one and back --
        the conversion the long-edge triangle mask in ``plotting/maps.py``
        needs, and which cannot be done from the projection parameters alone.

        PROJ's conventions, which are its own rather than this module's:
        ``angular_distortion`` is in degrees; ``meridian_convergence`` is in
        degrees and is the *negative* of the clockwise rotation of the image of
        true north, so a caller rotating a label takes its sign as PROJ gives
        it rather than as the eye reads it; ``tissot_semimajor`` and
        ``tissot_semiminor`` multiply to 1 here, since the projection is
        equal-area.
        """
        longitude, latitude, _ = _check_coordinates(lon, lat)
        if longitude.size == 0:
            # PROJ raises "longitude and latitude must be same size" on empty
            # input, which is both false and a different exception type from
            # everything else this module raises.
            raise ValueError("factors() needs at least one point, got an empty array")
        try:
            return _proj(self).get_factors(longitude, latitude, radians=False, errcheck=True)
        except pyproj.exceptions.ProjError as error:
            raise ValueError(_outside_domain_message(self, error)) from error

    @property
    def antipode(self) -> tuple[float, float]:
        """``(lon, lat)`` of the point :meth:`forward` cannot project.

        Exposed because a vendored world coastline reaches it, and clipping to
        avoid it needs somewhere to clip around. Longitude comes back in
        ``[-180, 180)`` whatever :attr:`lon_0` was given as.
        """
        return (self.lon_0 + 180.0 + 180.0) % 360.0 - 180.0, -self.lat_0

    def projected_bounds(self, bbox, *, samples_per_edge: int = 256):
        """The projected bounding box of a longitude/latitude box.

        Parameters
        ----------
        bbox:
            ``(west, south, east, north)`` in degrees, as
            :data:`sipnet_calibration.sites.EXTENTS` holds them and
            :func:`~sipnet_calibration.sites.select_sites` accepts them.
        samples_per_edge:
            How many points to place along each edge of the box. The default is
            far more than these extents need and costs microseconds.

        Returns
        -------
        tuple
            ``(x_min, y_min, x_max, y_max)`` in meters, in the same order as
            *bbox*, so it drops into ``ax.set_xlim`` and ``ax.set_ylim`` as
            ``bounds[0::2]`` and ``bounds[1::2]``.

        Raises
        ------
        ValueError
            If *bbox* is not four finite numbers, west is east of east, south is
            north of north, *samples_per_edge* is not an integer of at least 2,
            the box contains :attr:`antipode`, or any sampled point fails
            :meth:`forward`.

        Notes
        -----
        The edges of a longitude/latitude box project to curves, so the bound is
        taken over samples along the whole boundary rather than over the four
        corners. The edge farthest from the projection center bows away from it
        between its corners, and corners alone miss that: on the CONUS box,
        whose southern edge is the far one from a center at 50 N, they
        understate it by hundreds of kilometers, which is a visible clip.

        The boundary is enough, and the interior needs no sampling: the forward
        transform is a local diffeomorphism everywhere it is *defined*, so its
        components have no interior critical point and each extreme is attained
        on the edge of the box. The qualifier is why a box containing the
        antipode is refused rather than answered: there the interior holds a
        singularity, the boundary bound is not a bound, and the returned box
        would be wrong without being obviously wrong.
        """
        west, south, east, north = _check_bbox(bbox)
        if not isinstance(samples_per_edge, (int, np.integer)) or isinstance(
            samples_per_edge, bool
        ):
            raise ValueError(f"samples_per_edge must be an integer, got {samples_per_edge!r}")
        if samples_per_edge < 2:
            raise ValueError(f"samples_per_edge must be at least 2, got {samples_per_edge}")
        self._check_bbox_excludes_antipode(west, south, east, north)

        along = np.linspace(0.0, 1.0, samples_per_edge)
        lons = west + along * (east - west)
        lats = south + along * (north - south)
        constant = np.ones_like(along)
        boundary_lon = np.concatenate([lons, lons, west * constant, east * constant])
        boundary_lat = np.concatenate([south * constant, north * constant, lats, lats])

        x, y = self.forward(boundary_lon, boundary_lat)
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    # ── the definition, in interchange form ──────────────────────────────────

    def proj_string(self) -> str:
        """The definition as a PROJ string, on one line.

        The shortest thing to paste into a colleague's session, and what PROJ,
        GDAL and R's ``sf`` all accept directly. It carries less than the
        PROJJSON does -- no CRS name, and the base CRS by datum rather than by
        EPSG code -- which is why both are stored.

        Notes
        -----
        PROJ warns that converting a CRS to a PROJ string loses information,
        which is true and is the reason the PROJJSON is stored beside it. The
        warning is suppressed here because it describes a deliberate choice
        rather than a mistake a caller can act on.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self.crs().to_proj4().strip()

    def projjson(self) -> dict:
        """The definition as PROJJSON, as a ``dict`` ready for :mod:`json`.

        PROJ's own serialization, so the EPSG method and parameter codes, the
        base CRS and the axis definitions are the registry's rather than this
        module's transcription of them.
        """
        return json.loads(self.crs().to_json())

    # ── helpers ──────────────────────────────────────────────────────────────

    def _check_bbox_excludes_antipode(self, west, south, east, north) -> None:
        """Raise if :attr:`antipode` lies inside the box.

        :meth:`projected_bounds` takes its bound over the boundary, which is
        valid only where the transform is defined throughout the box.
        """
        antipode_lon, antipode_lat = self.antipode
        # How far east of the box's western edge the antipode lies, on the
        # circle, since the box's own longitudes need not lie in [-180, 180].
        # Reduced to [0, 360) rather than to [-180, 180): the symmetric range
        # cannot express an offset above 180, so a box wider than that would
        # appear to end before the antipode it actually contains.
        offset = (antipode_lon - west) % 360.0
        if offset <= (east - west) and south <= antipode_lat <= north:
            raise ValueError(
                f"the box contains ({antipode_lon}, {antipode_lat}), the antipode of the "
                "projection center, where this projection is undefined; the bounds of "
                "such a box are unbounded, not merely large"
            )


#: The project's display projection: Lambert Azimuthal Equal Area centered at
#: 50 N, 100 W, on WGS 84, in meters, with no false origin.
#:
#: Chosen over the ESRI:102003 Albers of the published reanalysis figures, and
#: over the other candidates, on measured distortion across the real site pool;
#: see the module Notes and issue #4. It is the project default, and one
#: projection serving both the full-domain and the CONUS figures is what keeps
#: panels comparable -- but it is not a prohibition. A caller wanting a
#: different center builds its own :class:`Projection`, or
#: ``dataclasses.replace``s this one.
SITE_PROJECTION = Projection(
    name="North America LAEA (SIPNET calibration display projection)",
    lat_0=50.0,
    lon_0=-100.0,
)


# ── the interchange files ─────────────────────────────────────────────────────

#: File name stem of the tracked interchange files, without an extension.
DEFINITION_STEM = "north_america_laea"


def default_definition_dir() -> Path:
    """Where the tracked interchange files live: ``projections/`` beside this module.

    Package data rather than anything under ``data/``, because
    ``data/processed/`` is regenerable and absent on a fresh clone, whereas this
    definition must always be present.
    """
    return Path(__file__).resolve().parent / "projections"


def definition_paths(
    directory: Path | str | None = None, *, stem: str = DEFINITION_STEM
) -> dict[str, Path]:
    """The paths of the interchange files, keyed by format.

    Parameters
    ----------
    directory:
        Where they live. Defaults to :func:`default_definition_dir`.
    stem:
        File name stem. Defaults to :data:`DEFINITION_STEM`.

    Returns
    -------
    dict
        ``{"projjson": path, "projstring": path}``.

    Raises
    ------
    ValueError
        If *stem* is empty or is a path rather than a bare file name.
    """
    if not stem or Path(stem).name != stem:
        raise ValueError(f"stem must be a bare file name, not a path, got {stem!r}")
    root = Path(directory) if directory is not None else default_definition_dir()
    return {suffix: root / f"{stem}.{suffix}" for suffix in ("projjson", "projstring")}


def write_definitions(
    directory: Path | str | None = None,
    *,
    projection: Projection = SITE_PROJECTION,
    stem: str = DEFINITION_STEM,
) -> dict[str, Path]:
    """Write *projection*'s interchange files, overwriting them.

    Parameters
    ----------
    directory, stem:
        As :func:`definition_paths`.
    projection:
        The projection to serialize. Defaults to :data:`SITE_PROJECTION`.

    Returns
    -------
    dict
        The paths written, keyed as :func:`definition_paths` keys them.

    Notes
    -----
    The parameters are the source of truth and these files are their output, so
    this is the only thing that should ever write them.
    :func:`check_definitions` makes a hand-edit a test failure.

    Both files are staged beside their destinations and moved into place only
    once every one of them is on disk, so an interrupted or failed run cannot
    leave one file describing this projection and the other describing the last
    one.
    """
    paths = definition_paths(directory, stem=stem)
    contents = _definition_contents(projection)
    staged: dict[Path, Path] = {}
    try:
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".partial")
            partial.write_text(contents[key], encoding="utf-8")
            staged[path] = partial
        for path, partial in staged.items():
            os.replace(partial, path)
    finally:
        for partial in staged.values():
            partial.unlink(missing_ok=True)
    return paths


def check_definitions(
    directory: Path | str | None = None,
    *,
    projection: Projection = SITE_PROJECTION,
    stem: str = DEFINITION_STEM,
) -> None:
    """Raise unless the stored interchange files match *projection* exactly.

    Parameters
    ----------
    directory, projection, stem:
        As :func:`write_definitions`.

    Raises
    ------
    FileNotFoundError
        If a file is missing, naming the command that writes it.
    ValueError
        If a file's content is not what *projection* serializes to, naming the
        command that regenerates it.

    Notes
    -----
    This is the anti-drift device: the parameters live in the dataclass, the
    files are generated, and the test suite calls this, so a parameter change
    that skips the regeneration fails rather than shipping a definition that
    disagrees with the transform. It is the same arrangement as the site table's
    round-trip assertion in ``scripts/ingest_sites.py``.

    It also catches a PROJ upgrade that changes the serialization, which is a
    real event: the files carry a PROJJSON schema version and PROJ's spelling of
    the parameters, neither of which this project chooses.
    """
    paths = definition_paths(directory, stem=stem)
    contents = _definition_contents(projection)
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"no {key} definition at {path}; write it with {_WRITE_COMMAND}"
            )
        if path.read_text(encoding="utf-8") != contents[key]:
            raise ValueError(
                f"{path} is not what {projection.name} serializes to, so the stored "
                f"definition and the transform disagree; regenerate it with "
                f"{_WRITE_COMMAND}, and if the file was edited by hand, make the "
                "change in the dataclass instead"
            )


# ── helpers ───────────────────────────────────────────────────────────────────
#
# Private: building the CRS, checking arguments, and the command line.

_WRITE_COMMAND = "`python -m sipnet_calibration.projection --write`"

#: Bound on the CRS, transformer and Proj caches. The documented way to get a
#: variant is ``dataclasses.replace``, so a caller scanning centers would
#: otherwise leak two PROJ objects per variant for the life of the process.
#: A handful is plenty: one projection is the norm and a facet of a few is the
#: most anyone has proposed.
_CACHE_SIZE = 32


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _crs(projection: Projection) -> pyproj.CRS:
    """*projection* as a :class:`pyproj.CRS`, built once per instance.

    Cached because building a CRS parses a definition through PROJ, which is far
    more expensive than the transform itself, and because
    :data:`SITE_PROJECTION` is a module-level constant that every panel uses.
    """
    return ProjectedCRS(
        conversion=LambertAzimuthalEqualAreaConversion(
            latitude_natural_origin=projection.lat_0,
            longitude_natural_origin=projection.lon_0,
            false_easting=projection.false_easting,
            false_northing=projection.false_northing,
        ),
        name=projection.name,
        geodetic_crs=pyproj.CRS.from_user_input(projection.base_crs),
    )


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _transformer(projection: Projection) -> pyproj.Transformer:
    """The transformer from *projection*'s base CRS onto it, built once.

    ``always_xy=True`` selects longitude-then-latitude regardless of the axis
    order the base CRS declares, which is the order the site table stores and
    the order :meth:`Projection.forward` documents.
    """
    return pyproj.Transformer.from_crs(
        pyproj.CRS.from_user_input(projection.base_crs), _crs(projection), always_xy=True
    )


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _proj(projection: Projection) -> pyproj.Proj:
    """*projection* as a :class:`pyproj.Proj`, for :meth:`Projection.factors`.

    Cached for the same reason as :func:`_crs`: constructing it parses the
    definition through PROJ, which costs more than the query it serves.
    """
    return pyproj.Proj(_crs(projection))


def _outside_domain_message(projection: Projection, error: Exception) -> str:
    """What to tell a caller whose point PROJ would not take."""
    antipode_lon, antipode_lat = projection.antipode
    return (
        f"{error}. This projection is undefined at the antipode of its center, "
        f"({antipode_lon}, {antipode_lat}); distortion factors are unavailable for "
        "about a degree around it, though the transform itself is not"
    )


def _check_coordinates(lon, lat):
    """The broadcast, finite, in-range ``(lon, lat)`` pair, as ``float64`` arrays.

    PROJ would reject an out-of-range latitude itself, but with a message about
    an internal error rather than about the argument, and it treats a NaN as a
    point it simply cannot project. Checking here means the message names the
    problem, and the counts are taken before broadcasting so they say how many
    values the caller passed that are bad rather than how large the result would
    have been.
    """
    longitude = _as_float_array(lon)
    latitude = _as_float_array(lat)
    scalar = longitude.ndim == 0 and latitude.ndim == 0

    bad_lon = int(np.count_nonzero(~np.isfinite(longitude)))
    bad_lat = int(np.count_nonzero(~np.isfinite(latitude)))
    if bad_lon or bad_lat:
        raise ValueError(
            f"{bad_lon} longitude(s) and {bad_lat} latitude(s) are not finite, so they "
            "cannot be projected"
        )
    outside = np.abs(latitude) > 90.0
    if np.any(outside):
        worst = float(np.max(np.abs(latitude[outside])))
        raise ValueError(
            f"{int(np.count_nonzero(outside))} latitude(s) are outside [-90, 90], the "
            f"worst being {worst}; arguments are (lon, lat), longitude first"
        )
    # Longitudes are deliberately not wrapped, but they are bounded: PROJ
    # accepts -9999 as a longitude and projects it to a real-looking point.
    wild = np.abs(longitude) > 360.0
    if np.any(wild):
        worst = float(np.max(np.abs(longitude[wild])))
        raise ValueError(
            f"{int(np.count_nonzero(wild))} longitude(s) are outside [-360, 360], the "
            f"worst being {worst}; longitudes need not be wrapped to [-180, 180], but a "
            "value this large is a fill value or a swapped argument"
        )

    longitude, latitude = np.broadcast_arrays(longitude, latitude)
    return longitude, latitude, scalar


def _as_float_array(values):
    """*values* as a ``float64`` array, with any masked entries as ``NaN``.

    ``np.asarray`` drops a mask silently, so a masked array would project its
    fill values to real-looking coordinates. Filling with ``NaN`` instead routes
    them into the non-finite guard.
    """
    if np.ma.isMaskedArray(values):
        return np.ma.filled(values.astype(float), np.nan)
    return np.asarray(values, dtype=float)


def _check_bbox(bbox):
    """The four floats of a well-formed ``(west, south, east, north)`` box.

    Everything unusable raises :class:`ValueError`, including the cases that
    would otherwise surface as a ``TypeError`` from unpacking or as a bare numpy
    message, so a caller has one exception type to handle and a message that
    names the parameter.
    """
    try:
        values = tuple(bbox)
    except TypeError:
        raise ValueError(f"bbox must be (west, south, east, north), got {bbox!r}") from None
    if len(values) != 4:
        raise ValueError(
            f"bbox must be (west, south, east, north), got {len(values)} value(s): {bbox!r}"
        )
    try:
        west, south, east, north = (float(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError(f"bbox values must be numbers, got {bbox!r}") from None
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError(f"bbox values must be finite, got {bbox!r}")
    if west > east:
        raise ValueError(
            f"bbox west {west} is east of east {east}; this does not wrap the "
            "antimeridian, matching sipnet_calibration.sites.select_sites"
        )
    if south > north:
        raise ValueError(f"bbox south {south} is north of north {north}")
    return west, south, east, north


def _definition_contents(projection: Projection) -> dict[str, str]:
    """The text of each interchange file, keyed as :func:`definition_paths` keys them."""
    return {
        "projjson": projection.crs().to_json(pretty=True, indentation=2) + "\n",
        "projstring": projection.proj_string() + "\n",
    }


def _main(argv: list[str] | None = None) -> int:
    """``python -m sipnet_calibration.projection [--write | --check]``."""
    parser = argparse.ArgumentParser(
        prog="python -m sipnet_calibration.projection",
        description=(
            "Write or verify the interchange files for the project's display "
            "projection. The parameters live in the dataclass; these files are "
            "serialized from it by PROJ."
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--write", action="store_true", help="regenerate the files, overwriting them"
    )
    action.add_argument(
        "--check",
        action="store_true",
        help="verify the stored files match the parameters; the default",
    )
    parser.add_argument(
        "--directory",
        default=None,
        help="where the files live; defaults to the package's projections/ directory",
    )
    arguments = parser.parse_args(argv)

    if arguments.write:
        try:
            written = write_definitions(arguments.directory)
        except OSError as error:
            print(f"error: {error}")
            return 1
        for key, path in written.items():
            print(f"wrote {key}: {path}")
        return 0

    try:
        check_definitions(arguments.directory)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}")
        return 1
    print(f"the stored definition matches {SITE_PROJECTION.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
