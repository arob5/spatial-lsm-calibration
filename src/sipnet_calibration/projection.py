"""The display projection for spatial figures, and the forward transform onto it.

Overview
--------
Site coordinates are geographic -- longitude and latitude on WGS 84 -- and are
not plottable as they stand: a degree of longitude is about 921 m at the south
of the site pool and 121 m at the north, so plotting degrees directly stretches
the Arctic by a factor of eight and makes any density or heatmap panel
misleading. This module owns the one projection the project's spatial figures
use, holds its complete definition, and provides the forward transform from
longitude and latitude to projected meters.

It sits between the site table and the plotting layer, and the dependency runs
one way::

    data/processed/sites/sites.csv
      -> sipnet_calibration.sites          load_sites() -> lon, lat
      -> this module                       SITE_PROJECTION.forward() -> x, y
      -> sipnet_calibration.plotting.maps  the spatial panels

Nothing here knows about matplotlib, an axes, or a figure. Equally, nothing in
``plotting/`` defines projection parameters: a panel that needs projected
coordinates calls :meth:`Projection.forward`, and a panel that needs axes
limits for a named region calls :meth:`Projection.projected_bounds` on a box
from :data:`sipnet_calibration.sites.EXTENTS`.

Input data
----------
This module reads nothing. The parameters are in code, for the same reason
:data:`~sipnet_calibration.sites.SITE_GRID` is: the definition and the transform
that implements it must not be able to disagree.

It *writes* two files, which are the interchange form for tools that are not
this package -- PROJ, GDAL, R, a colleague's QGIS session::

    src/sipnet_calibration/projections/north_america_laea.projjson
    src/sipnet_calibration/projections/north_america_laea.projstring

Both are generated from :data:`SITE_PROJECTION` by :func:`write_definitions`,
are tracked in version control so that a fresh checkout has them, and are
checked against the dataclass by :func:`check_definitions`, which the test suite
calls. Editing them by hand is therefore a test failure, not a way to change the
projection.

Data model
----------
:class:`Ellipsoid` is a reference ellipsoid: a name, a semi-major axis in
meters, an inverse flattening, and the authority code it is registered under.
:data:`WGS84` is the one instance this project uses, being the ellipsoid of
EPSG:4326, which is the CRS the site coordinates are on.

:class:`Projection` is a complete projected CRS definition:

======================== ============== =====================================
Field                    Type           Meaning
======================== ============== =====================================
``name``                 ``str``        CRS name, as written to the files
``method``               ``str``        EPSG method name
``method_code``          ``int``        EPSG method code
``lat_0``, ``lon_0``     ``float``      latitude/longitude of natural origin
``false_easting``        ``float``      meters added to x
``false_northing``       ``float``      meters added to y
``ellipsoid``            ``Ellipsoid``  the ellipsoid the formulas use
``base_crs_name``        ``str``        geographic CRS of the input coordinates
``base_crs_code``        ``int``        its EPSG code
======================== ============== =====================================

:data:`SITE_PROJECTION` is the project's projection: a Lambert Azimuthal Equal
Area centered at 50 N, 100 W on WGS 84, with no false origin, in meters.

**Units and axis order.** ``forward`` takes longitude first and latitude second,
which is the traditional GDAL and PROJ ordering rather than the axis order
EPSG:4326 formally declares, matching how the site table stores them and what
``pyproj``'s ``always_xy=True`` selects. It returns easting and northing in
meters, in that order.

Functions
---------
:meth:`Projection.forward`
    Longitude and latitude in degrees to easting and northing in meters.

:meth:`Projection.projected_bounds`
    The projected bounding box of a longitude/latitude box, for axes limits.

:meth:`Projection.proj_string` and :meth:`Projection.projjson`
    The definition in the two interchange forms, derived from the fields.

:func:`write_definitions` and :func:`check_definitions`
    Write the interchange files, and verify the tracked ones still match the
    dataclass. ``python -m sipnet_calibration.projection --write`` is the
    command that regenerates them.

Notes
-----
**Why this projection.** The choice, the alternatives, and the distortion
measured over the real 8000 sites are recorded in
`issue #4 <https://github.com/arob5/spatial-lsm-calibration/issues/4>`_. In
summary: every candidate considered is equal-area, which is the property the
plotting design requires, so the choice turns on shape. The site pool spans 75
degrees of latitude and 159 of longitude, which is outside the domain of use of
any Albers Equal Area Conic -- Snyder's guidance puts Albers at regions of
predominant east-west expanse -- and ESRI:102003, which the published
reanalysis figures used, reaches a maximum angular deformation of 107 degrees
and a 9:1 local anisotropy at the northernmost sites. This projection holds
angular deformation under 14 degrees and anisotropy under 1.3 over the whole
pool, which ``tests/test_projection.py`` asserts against the real site table.

**Anisotropy is not only cosmetic here.** ``TriRenderer`` triangulates *after*
projecting, and a Delaunay triangulation is not affine-invariant, so a strong
local anisotropy makes the mesh an artifact of the projection rather than of
where the sites are; and the long-edge mask threshold that renderer needs is a
projected length, which under a projection whose scale varies ninefold cannot
mean one ground distance.

**No datum transformation is involved.** The base CRS is WGS 84, matching the
site coordinates, so nothing is shifted. A NAD83-based definition, such as the
ESRI codes, would have raised the question, and the answer would have been that
it does not matter: NAD 83 and WGS 84 differ by about 2 m, which is 1.9e-4 of a
pixel on a 1000 px wide axes of the full domain.

**Equal-areaness does not validate an implementation.** It follows from the
functional form even when the parameters are wrong -- a wrong cone constant
placed coordinates thousands of km out while still preserving area exactly, in
the analysis on issue #4. The tests therefore validate against published
coordinates: Snyder's Appendix A worked example, and PROJ's own regression
values. Equal-areaness is asserted as a property, downstream of that.

**The exact antipode of the center is not projectable**, and fails badly rather
than loudly: the ellipsoidal formula yields infinity times ``sin(180 deg)``,
which is a finite, plausible-looking number. :meth:`Projection.forward` raises
there instead. PROJ rejects the same input. Nothing in this project comes near
it -- the farthest site is 51.1 degrees from the center -- but a vendored
coastline handed to the same function might.

**A longitude/latitude box does not project to a rectangle.** Its edges become
curves, so :meth:`Projection.projected_bounds` samples along them rather than
projecting the four corners; corners alone clip the top of a CONUS box by tens
of kilometers.

Usage
-----
Project site coordinates::

    from sipnet_calibration.projection import SITE_PROJECTION
    from sipnet_calibration.sites import EXTENTS, load_sites

    sites = load_sites()
    x, y = SITE_PROJECTION.forward(sites["lon"].to_numpy(), sites["lat"].to_numpy())

Axes limits for a named region, in the same projected meters::

    x_min, y_min, x_max, y_max = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])

The definition, for anything outside this package::

    print(SITE_PROJECTION.proj_string())
    # +proj=laea +lat_0=50 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m ...

Regenerate the tracked interchange files after changing a parameter::

    python -m sipnet_calibration.projection --write
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DEFINITION_STEM",
    "Ellipsoid",
    "LAEA_METHOD",
    "LAEA_METHOD_CODE",
    "Projection",
    "SITE_PROJECTION",
    "WGS84",
    "check_definitions",
    "default_definition_dir",
    "definition_paths",
    "write_definitions",
]


# ── ellipsoids ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Ellipsoid:
    """A reference ellipsoid, as the projection formulas need it.

    Parameters
    ----------
    name:
        The ellipsoid's registered name, written into the definition files.
    semi_major:
        Equatorial radius in meters.
    inverse_flattening:
        ``1/f``. Held rather than the eccentricity because it is what the EPSG
        registry and PROJJSON carry.
    authority_code:
        EPSG code of the ellipsoid.
    """

    name: str
    semi_major: float
    inverse_flattening: float
    authority_code: int

    @classmethod
    def from_eccentricity_squared(
        cls, name: str, semi_major: float, eccentricity_squared: float, authority_code: int = 0
    ) -> Ellipsoid:
        """An ellipsoid given ``e**2`` instead of ``1/f``.

        Older sources, Snyder's worked examples among them, tabulate ``e**2``.
        The conversion is exact in both directions to floating-point rounding,
        so nothing is lost by storing the inverse flattening.
        """
        raise NotImplementedError

    @property
    def flattening(self) -> float:
        """``f``, the flattening."""
        raise NotImplementedError

    @property
    def eccentricity_squared(self) -> float:
        """``e**2 = 2f - f**2``."""
        raise NotImplementedError

    @property
    def eccentricity(self) -> float:
        """``e``."""
        raise NotImplementedError


#: The ellipsoid of EPSG:4326, which is the CRS the site coordinates are on, so
#: it is the ellipsoid the projection is defined against. Parameters as
#: registered: ``a = 6378137 m`` exactly and ``1/f = 298.257223563``.
#:
#: GRS 80, the ellipsoid of the NAD83-based ESRI codes, shares the semi-major
#: axis exactly and differs in inverse flattening by 1.4e-9, which displaces a
#: site by at most 0.08 mm. The choice between them is therefore about which
#: definition is internally consistent, not about accuracy.
WGS84 = Ellipsoid(
    name="WGS 84",
    semi_major=6378137.0,
    inverse_flattening=298.257223563,
    authority_code=7030,
)


# ── the projection ────────────────────────────────────────────────────────────

#: The one coordinate operation method implemented here, as the EPSG registry
#: names and numbers it. Carried explicitly so that the definition files say
#: which method their parameters belong to, and so that a projection declaring
#: some other method cannot be silently transformed by these formulas.
LAEA_METHOD = "Lambert Azimuthal Equal Area"
LAEA_METHOD_CODE = 9820


@dataclass(frozen=True, slots=True)
class Projection:
    """A projected CRS: its complete definition, and the transform onto it.

    Only the Lambert Azimuthal Equal Area method is implemented, since it is the
    one this project uses; :attr:`method_code` is carried so that the definition
    files say which method the parameters belong to, and so that a second method
    added later cannot be silently transformed by this one's formulas.

    Parameters
    ----------
    name:
        CRS name, written into the definition files and used for their file
        names via :func:`definition_paths`.
    lat_0, lon_0:
        Latitude and longitude of the natural origin, in degrees. The point
        ``(lon_0, lat_0)`` maps to ``(false_easting, false_northing)``.
    false_easting, false_northing:
        Constants added to the projected coordinates, in meters.
    ellipsoid:
        The ellipsoid the formulas use. Must be the ellipsoid of
        :attr:`base_crs_code`, or the definition contradicts itself.
    base_crs_name, base_crs_code:
        The geographic CRS the input coordinates are on.
    method, method_code:
        EPSG method name and code.
    """

    name: str
    lat_0: float
    lon_0: float
    false_easting: float = 0.0
    false_northing: float = 0.0
    ellipsoid: Ellipsoid = WGS84
    base_crs_name: str = "WGS 84"
    base_crs_code: int = 4326
    method: str = LAEA_METHOD
    method_code: int = LAEA_METHOD_CODE

    def __post_init__(self) -> None:
        # Implemented here rather than left raising, unlike the rest of this
        # skeleton, because SITE_PROJECTION is a module-level constant and so
        # has to be constructible for the module to import at all.
        if self.method_code != LAEA_METHOD_CODE:
            raise ValueError(
                f"{self.name} names EPSG method {self.method_code}; only "
                f"{LAEA_METHOD_CODE} ({LAEA_METHOD}) is implemented"
            )
        if not -90.0 <= self.lat_0 <= 90.0:
            raise ValueError(f"lat_0 must be in [-90, 90], got {self.lat_0}")
        # A pole exactly, to within a nanodegree: about 0.1 mm of latitude.
        if abs(abs(self.lat_0) - 90.0) < 1e-9:
            raise ValueError(
                f"lat_0 of {self.lat_0} is the polar aspect, which needs its own "
                "formulas; the oblique ones implemented here divide by cos(lat_0)"
            )
        if not -360.0 <= self.lon_0 <= 360.0:
            raise ValueError(f"lon_0 must be in [-360, 360], got {self.lon_0}")

    # ── the transform ────────────────────────────────────────────────────────

    def forward(self, lon, lat):
        """Longitude and latitude in degrees to easting and northing in meters.

        Parameters
        ----------
        lon, lat:
            Degrees, scalar or array-like, longitude first. Broadcast against
            each other. Longitudes are not wrapped: -190 and 170 are the same
            meridian and both are accepted.
        lat:
            Degrees, in ``[-90, 90]``.

        Returns
        -------
        tuple
            ``(x, y)`` in meters. Scalars in, scalars out; otherwise arrays of
            the broadcast shape, ``float64``.

        Raises
        ------
        ValueError
            If any coordinate is not finite, any latitude is outside
            ``[-90, 90]``, or any point is at the antipode of the projection
            center, where the projection is undefined.

        Notes
        -----
        Snyder (1987) equations 3-11, 3-12 and 24-1 to 24-6, the oblique
        ellipsoidal case, which also covers the equatorial one. The polar
        aspects need their own formulas and are rejected by
        :meth:`__post_init__` rather than approximated here.

        The ellipsoidal form works through the authalic latitude, so the
        distortion pattern is that of a spherical LAEA on the authalic sphere,
        which is why the closed-form spherical scale factors are a valid check
        on the output and are used as one in the tests.
        """
        raise NotImplementedError

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
            far more than is needed for the extents this project uses and costs
            microseconds.

        Returns
        -------
        tuple
            ``(x_min, y_min, x_max, y_max)`` in meters, in the same order as
            *bbox*, so it drops into ``ax.set_xlim`` and ``ax.set_ylim`` as
            ``bounds[0::2]`` and ``bounds[1::2]``.

        Raises
        ------
        ValueError
            If *bbox* is not four values, west is east of east, south is north
            of north, or any of them fails :meth:`forward`.

        Notes
        -----
        The edges of a longitude/latitude box project to curves, so the bound is
        taken over samples along the whole boundary rather than over the four
        corners. Corners alone would clip the poleward edge of a wide box: the
        top edge of the CONUS box bows away from the projection center by tens
        of kilometers between its corners.
        """
        raise NotImplementedError

    # ── the definition, in interchange form ──────────────────────────────────

    def proj_string(self) -> str:
        """The definition as a PROJ string, on one line.

        The form PROJ, GDAL and R's ``sf`` all accept directly, and the shortest
        thing to paste into a colleague's session.
        """
        raise NotImplementedError

    def projjson(self) -> dict:
        """The definition as PROJJSON, as a ``dict`` ready for :mod:`json`.

        A ``ProjectedCRS`` object against the v0.7 schema, carrying the base
        geographic CRS with its ellipsoid, the conversion with its EPSG method
        and parameter codes, and the Cartesian coordinate system with its two
        axes in meters.

        Notes
        -----
        No ``id`` is claimed for the projected CRS itself, because it is not
        registered with any authority -- that is precisely why the definition is
        stored in this repository. The method and parameters do carry their EPSG
        codes, since those are registered.

        The base CRS is written with a ``datum`` rather than the
        ``datum_ensemble`` that the EPSG registry now uses for WGS 84, following
        the ``ProjectedCRS`` example in the PROJ documentation. The distinction
        is about which realization of WGS 84 is meant, at the 2 m level, and is
        immaterial at the scale of these figures.
        """
        raise NotImplementedError


#: The project's display projection: Lambert Azimuthal Equal Area centered at
#: 50 N, 100 W, on WGS 84, in meters, with no false origin.
#:
#: Chosen over the ESRI:102003 Albers of the published reanalysis figures, and
#: over the other candidates, on measured distortion across the real site pool;
#: see the module Notes and issue #4. One projection serves both the
#: full-domain and the CONUS figures, so that panels are comparable.
SITE_PROJECTION = Projection(
    name="North America LAEA (SIPNET calibration display projection)",
    lat_0=50.0,
    lon_0=-100.0,
)

#: File name stem of the tracked interchange files, without an extension.
DEFINITION_STEM = "north_america_laea"


# ── the interchange files ─────────────────────────────────────────────────────


def default_definition_dir() -> Path:
    """Where the tracked interchange files live: ``projections/`` beside this module.

    Package data rather than anything under ``data/``, because ``data/`` is
    regenerable and absent on a fresh clone, whereas this definition must always
    be present.
    """
    raise NotImplementedError


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
    """
    raise NotImplementedError


def write_definitions(
    projection: Projection = SITE_PROJECTION,
    directory: Path | str | None = None,
    *,
    stem: str = DEFINITION_STEM,
) -> dict[str, Path]:
    """Write *projection*'s interchange files, overwriting them.

    Parameters
    ----------
    projection:
        The projection to serialize. Defaults to :data:`SITE_PROJECTION`.
    directory, stem:
        As :func:`definition_paths`.

    Returns
    -------
    dict
        The paths written, keyed as :func:`definition_paths` keys them.

    Notes
    -----
    The dataclass is the source of truth and these files are its output, so this
    is the only thing that should ever write them. :func:`check_definitions`
    makes a hand-edit a test failure.
    """
    raise NotImplementedError


def check_definitions(
    projection: Projection = SITE_PROJECTION,
    directory: Path | str | None = None,
    *,
    stem: str = DEFINITION_STEM,
) -> None:
    """Raise unless the stored interchange files match *projection* exactly.

    Parameters
    ----------
    projection, directory, stem:
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
    """
    raise NotImplementedError


# ── helpers ───────────────────────────────────────────────────────────────────
#
# Private: the projection formulas, and the argument checking they need.


def _authalic_q(lat_radians, eccentricity_squared):
    """Snyder's ``q``, equation 3-12: ``2`` times the authalic sine, up to ``q_p``."""
    raise NotImplementedError


def _check_coordinates(lon, lat):
    """The broadcast, finite, in-range ``(lon, lat)`` pair, as ``float64`` arrays."""
    raise NotImplementedError


def _main(argv: list[str] | None = None) -> int:
    """``python -m sipnet_calibration.projection [--write | --check]``."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
