"""The display projection for spatial figures, and the forward transform onto it.

Overview
--------
Site coordinates are geographic -- longitude and latitude on WGS 84 -- and are
not plottable as they stand: a cell of the site grid is about 921 m wide at the
south of the pool and 121 m at the north, so plotting degrees directly stretches
the Arctic by more than sevenfold and makes any density or heatmap panel
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

Output data
-----------
It writes two files, the interchange form for tools that are not this package --
PROJ, GDAL, R, a colleague's QGIS session::

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
``base_datum_name``      ``str``        its geodetic datum, by name
======================== ============== =====================================

:data:`SITE_PROJECTION` is the project's projection: a Lambert Azimuthal Equal
Area centered at 50 N, 100 W on WGS 84, with no false origin, in meters.

Two constants name the one EPSG method implemented, :data:`LAEA_METHOD` and
:data:`LAEA_METHOD_CODE`, which construction enforces. Three describe the
serialization: :data:`PROJJSON_SCHEMA` is the schema version the PROJJSON
declares, :data:`PROJ_ELLIPSOID_NAMES` holds PROJ's built-in ellipsoid tokens
with the parameters PROJ substitutes for them, and :data:`DEFINITION_STEM` is
the file name stem of the two interchange files.

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

:func:`default_definition_dir` and :func:`definition_paths`
    Where those files live, and their paths by format.

Notes
-----
**Why this projection.** The choice, the alternatives, and the distortion
measured over the real 8000 sites are recorded in
`issue #4 <https://github.com/arob5/spatial-lsm-calibration/issues/4>`_. In
summary: every candidate considered is equal-area, which is the property the
plotting design requires, so the choice turns on shape. The site pool spans 76
degrees of latitude and 159 of longitude, which is outside the domain of use of
any Albers Equal Area Conic -- Snyder's guidance puts Albers at regions of
predominant east-west expanse -- and ESRI:102003, which the published
reanalysis figures used, reaches a maximum angular deformation of 107 degrees
and a 9:1 local anisotropy at the northernmost sites. This projection holds
angular deformation under 14 degrees and anisotropy under 1.3 over the whole
pool, which ``tests/test_projection.py`` asserts against the real site table.

**Anisotropy is not only cosmetic here.** The spatial renderer planned in
``plotting/maps.py`` is to triangulate *after* projecting, and a Delaunay
triangulation is not affine-invariant, so a strong local anisotropy would make
the mesh an artifact of the projection rather than of where the sites are. Its
long-edge mask threshold is a projected length too, which means one ground
distance only where the local scale is close to isotropic -- and under 102003 it
is anisotropic ninefold.

**No datum transformation is involved.** The base CRS is WGS 84, matching the
site coordinates, so nothing is shifted. A NAD83-based definition, such as the
ESRI codes, would have raised the question, and the answer would have been that
it does not matter: NAD 83 and WGS 84 differ by about 2 m, and the full-domain
extent is 16,000 km across, so the shift is under a thousandth of a pixel at any
figure size anyone would render.

**Equal-areaness does not validate an implementation.** It follows from the
functional form even when the parameters are wrong -- a wrong cone constant
placed coordinates thousands of km out while still preserving area exactly, in
the analysis on issue #4. The tests therefore validate against published
coordinates: Snyder's Appendix A worked example, and PROJ's own regression
values. Equal-areaness is asserted as a property, downstream of that.

**The exact antipode of the center is not projectable**, and fails badly rather
than loudly. What it fails with depends on the last bit of the arithmetic: the
quantity that should be zero there lands just negative, just positive or exactly
zero, giving ``NaN``, a fraction of a meter from the projection center, or
infinity. For :data:`SITE_PROJECTION` it is ``NaN``. None of the three is an
error, so :meth:`Projection.forward` raises instead. PROJ rejects the same
input. Nothing in this project comes near it -- no site is more than 55 degrees
of arc from the center, against 180 for the antipode -- but a vendored coastline
handed to the same function might.

**There is no inverse transform, by decision rather than oversight.** Nothing
in the spatial panels as specified needs one: a graticule, an extent and a site
marker are all forward. It would be needed to report an axes position back in
longitude and latitude, to label the parallel that leaves the left spine, and
for a raster renderer that has to know where each cell sits, so if the plotting
work wants any of those, the inverse comes with it. A caller cannot reasonably
supply it, since the ellipsoidal inverse needs the authalic-to-geodetic step
this module already owns.

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

Axes limits for a named region, in the same projected meters::

    x_min, y_min, x_max, y_max = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])

The definition, for anything outside this package::

    print(SITE_PROJECTION.proj_string())
    # +proj=laea +lat_0=50 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m ...

Regenerate the tracked interchange files after changing a parameter::

    python -m sipnet_calibration.projection --write
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

__all__ = [
    "DEFINITION_STEM",
    "Ellipsoid",
    "LAEA_METHOD",
    "LAEA_METHOD_CODE",
    "PROJJSON_SCHEMA",
    "PROJ_ELLIPSOID_NAMES",
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

    def __post_init__(self) -> None:
        if not math.isfinite(self.semi_major) or self.semi_major <= 0:
            raise ValueError(f"semi_major must be finite and positive, got {self.semi_major}")
        # Zero is the sphere. Otherwise the flattening must be under 1, i.e.
        # 1/f above 1: below 0.5 the eccentricity squared goes negative and its
        # square root is NaN.
        if not math.isfinite(self.inverse_flattening) or not (
            self.inverse_flattening == 0.0 or self.inverse_flattening > 1.0
        ):
            raise ValueError(
                f"inverse_flattening must be 0 (a sphere) or greater than 1, got "
                f"{self.inverse_flattening}"
            )
        # The condition that actually matters is on the derived eccentricity,
        # and it is not the same condition: at 1/f = 1 + 1e-9 the flattening is
        # 1 - 1e-9, so e**2 is 1 - 1e-18, which *is* 1.0 in float64. Then q is
        # arctanh(1) = inf and every projected coordinate comes back NaN with
        # nothing to say why -- and nothing downstream catches it, since the
        # coordinate checks see finite input and the antipode check compares a
        # NaN and finds it false. So the eccentricity is checked, not inferred.
        if self.inverse_flattening != 0.0 and not self.eccentricity_squared < 1.0:
            raise ValueError(
                f"inverse_flattening of {self.inverse_flattening} is so close to 1 that "
                f"the eccentricity is 1 to floating point, where the authalic latitude "
                f"is infinite; it must exceed 1 by more than about 1e-8"
            )

    @classmethod
    def from_eccentricity_squared(
        cls, name: str, semi_major: float, eccentricity_squared: float, authority_code: int = 0
    ) -> Ellipsoid:
        """An ellipsoid given ``e**2`` instead of ``1/f``.

        Older sources, Snyder's worked examples among them, tabulate ``e**2``.

        Parameters
        ----------
        name:
            The ellipsoid's name.
        semi_major:
            Equatorial radius in meters.
        eccentricity_squared:
            ``e**2``, in ``[0, 1)``. Zero gives a sphere.
        authority_code:
            EPSG code, or 0 for an ellipsoid that is not registered. Zero is
            not merely a placeholder: :func:`_proj_ellipsoid` looks the code up
            in :data:`PROJ_ELLIPSOID_NAMES`, so 0 is what makes the PROJ string
            spell the parameters out rather than claim a ``+ellps=`` token.

        Returns
        -------
        Ellipsoid
            With the equivalent inverse flattening, 0 for a sphere.

        Raises
        ------
        ValueError
            If *eccentricity_squared* is outside ``[0, 1)``.

        Notes
        -----
        The flattening is taken as ``e**2 / (1 + sqrt(1 - e**2))`` rather than
        the textbook ``1 - sqrt(1 - e**2)``. They are equal in exact arithmetic
        but not in floating point: the subtraction cancels as ``e**2`` goes to
        zero, costing four significant digits by ``e**2 = 1e-12``. The form used
        here round-trips to a couple of ulp at every eccentricity.
        """
        if not 0.0 <= eccentricity_squared < 1.0:
            raise ValueError(
                f"eccentricity_squared must be in [0, 1), got {eccentricity_squared}"
            )
        flattening = eccentricity_squared / (1.0 + math.sqrt(1.0 - eccentricity_squared))
        return cls(
            name=name,
            semi_major=semi_major,
            # Zero is how a sphere is written, here and in WKT alike, since 1/f
            # is undefined for one.
            inverse_flattening=0.0 if flattening == 0.0 else 1.0 / flattening,
            authority_code=authority_code,
        )

    @property
    def flattening(self) -> float:
        """``f``, the flattening. Zero for a sphere."""
        if self.inverse_flattening == 0.0:
            return 0.0
        return 1.0 / self.inverse_flattening

    @property
    def eccentricity_squared(self) -> float:
        """``e**2 = 2f - f**2``."""
        flattening = self.flattening
        return 2.0 * flattening - flattening * flattening

    @property
    def eccentricity(self) -> float:
        """``e``."""
        return math.sqrt(self.eccentricity_squared)


#: The ellipsoid of EPSG:4326, which is the CRS the site coordinates are on, so
#: it is the ellipsoid the projection is defined against. Parameters as
#: registered: ``a = 6378137 m`` exactly and ``1/f = 298.257223563``.
#:
#: GRS 80, the ellipsoid of the NAD83-based ESRI codes, shares the semi-major
#: axis exactly and differs in inverse flattening by 1.5e-6, five parts in a
#: billion, which displaces a site by well under a millimeter. The choice
#: between them is therefore about which definition is internally consistent,
#: not about accuracy.
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

    Only the Lambert Azimuthal Equal Area method is implemented; a projection
    naming any other is refused at construction.

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
    ellipsoid:
        The ellipsoid the formulas use. Must be the ellipsoid of
        :attr:`base_crs_code`, or the definition contradicts itself.
    base_crs_name, base_crs_code:
        The geographic CRS the input coordinates are on.
    base_datum_name:
        Its geodetic datum, by name. No authority code is claimed for it: EPSG
        registers 6326 as the WGS 84 *ensemble*, and the definition written here
        carries a single reference frame instead; see :meth:`projjson`.
    method, method_code:
        EPSG method name and code; see :data:`LAEA_METHOD_CODE`.
    """

    name: str
    lat_0: float
    lon_0: float
    false_easting: float = 0.0
    false_northing: float = 0.0
    ellipsoid: Ellipsoid = WGS84
    base_crs_name: str = "WGS 84"
    base_crs_code: int = 4326
    base_datum_name: str = "World Geodetic System 1984"
    method: str = LAEA_METHOD
    method_code: int = LAEA_METHOD_CODE

    def __post_init__(self) -> None:
        if (self.method, self.method_code) != (LAEA_METHOD, LAEA_METHOD_CODE):
            raise ValueError(
                f"{self.name} names EPSG method {self.method_code} ({self.method}); "
                f"only {LAEA_METHOD_CODE} ({LAEA_METHOD}) is implemented"
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
        # The false origin is added to every projected coordinate and written
        # into the definition files, so a NaN here defeats the guard in
        # forward() and puts "+x_0=nan" in a file meant to be authoritative.
        for field, value in (
            ("false_easting", self.false_easting),
            ("false_northing", self.false_northing),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite, got {value}")

    # ── the transform ────────────────────────────────────────────────────────

    def forward(self, lon, lat):
        """Longitude and latitude in degrees to easting and northing in meters.

        Parameters
        ----------
        lon, lat:
            Degrees, scalar or array-like, longitude first; latitude in
            ``[-90, 90]`` and longitude in ``[-360, 360]``. Broadcast against
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
            ``[-90, 90]``, any longitude is outside ``[-360, 360]``, or any
            point is at the antipode of the projection center, where the
            projection is undefined.

        Notes
        -----
        Snyder (1987), the oblique ellipsoidal case of section 24 -- equations
        3-11 to 3-13 for the authalic latitude and radius, 14-15 for ``m1``, and
        the 24-17 to 24-20 group for ``B``, ``D``, ``x`` and ``y``. It covers
        the equatorial aspect too. The polar aspects need their own formulas and
        are rejected by :meth:`__post_init__` rather than approximated here.

        The ellipsoidal form works through the authalic latitude, so the
        distortion is that of a spherical LAEA on the authalic sphere to within
        ``O(e**2)``. That is what makes the closed-form spherical scale factors
        a check on the output, and it is why the test that uses them exactly
        does so on a sphere: on the ellipsoid the authalic map contributes shape
        distortion of its own, which over this site pool reaches 9e-4 relative
        in each scale factor and 2e-3 in their ratio.
        """
        longitude, latitude, scalar = _check_coordinates(lon, lat)
        e2 = self.ellipsoid.eccentricity_squared
        semi_major = self.ellipsoid.semi_major

        q_pole = _authalic_q(math.pi / 2, e2)
        authalic_radius = semi_major * math.sqrt(q_pole / 2)
        origin = math.radians(self.lat_0)
        # beta is the authalic latitude; the ratio can exceed 1 by an epsilon at
        # the poles, where q(lat) is q_pole up to rounding.
        beta_0 = math.asin(min(1.0, max(-1.0, _authalic_q(origin, e2) / q_pole)))
        parallel_radius = math.cos(origin) / math.sqrt(1 - e2 * math.sin(origin) ** 2)
        stretch = semi_major * parallel_radius / (authalic_radius * math.cos(beta_0))

        # The clip is insurance, not arithmetic this relies on: the ratio is
        # exactly +-1 at the poles for every eccentricity tested, which is a
        # property of the arctanh form in _authalic_q rather than of arcsin.
        # A ratio over 1 by one ulp would return NaN, so the guard stays.
        beta = np.arcsin(np.clip(_authalic_q(np.radians(latitude), e2) / q_pole, -1.0, 1.0))
        delta_lon = np.radians(longitude - self.lon_0)
        cos_delta_lon = np.cos(delta_lon)

        # 1 + cos(angular distance from the center), on the authalic sphere.
        span = (
            1.0
            + math.sin(beta_0) * np.sin(beta)
            + math.cos(beta_0) * np.cos(beta) * cos_delta_lon
        )
        _check_not_antipodal(span, self)

        scale = authalic_radius * np.sqrt(2.0 / span)
        x = self.false_easting + scale * stretch * np.cos(beta) * np.sin(delta_lon)
        y = self.false_northing + (scale / stretch) * (
            math.cos(beta_0) * np.sin(beta) - math.sin(beta_0) * np.cos(beta) * cos_delta_lon
        )
        if scalar:
            return float(x), float(y)
        return x, y

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
            If *bbox* is not four finite numbers, west is east of east, south is
            north of north, *samples_per_edge* is not an integer of at least 2,
            the box contains the antipode of the projection center, or any
            sampled point fails :meth:`forward`.

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
        antipode of the projection center is refused rather than answered: there
        the interior holds a singularity, the boundary bound is not a bound, and
        the returned box would be wrong without being obviously wrong.
        """
        west, south, east, north = _check_bbox(bbox)
        if not isinstance(samples_per_edge, (int, np.integer)) or isinstance(
            samples_per_edge, bool
        ):
            raise ValueError(f"samples_per_edge must be an integer, got {samples_per_edge!r}")
        if samples_per_edge < 2:
            raise ValueError(f"samples_per_edge must be at least 2, got {samples_per_edge}")
        _check_bbox_excludes_antipode(self, west, south, east, north)

        along = np.linspace(0.0, 1.0, samples_per_edge)
        lons = west + along * (east - west)
        lats = south + along * (north - south)
        constant = np.ones_like(along)
        boundary_lon = np.concatenate([lons, lons, west * constant, east * constant])
        boundary_lat = np.concatenate([south * constant, north * constant, lats, lats])

        x, y = self.forward(boundary_lon, boundary_lat)
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    @property
    def antipode(self) -> tuple[float, float]:
        """``(lon, lat)`` of the point :meth:`forward` cannot project.

        Exposed because the module tells callers to keep away from it -- a
        vendored world coastline reaches it -- and clipping to avoid it needs
        somewhere to clip around. Longitude comes back in ``[-180, 180)``
        whatever :attr:`lon_0` was given as.
        """
        return (self.lon_0 + 180.0 + 180.0) % 360.0 - 180.0, -self.lat_0

    # ── the definition, in interchange form ──────────────────────────────────

    def proj_string(self) -> str:
        """The definition as a PROJ string, on one line.

        The form PROJ, GDAL and R's ``sf`` all accept directly, and the shortest
        thing to paste into a colleague's session.

        Notes
        -----
        The ellipsoid is written as ``+ellps=`` when :data:`PROJ_ELLIPSOID_NAMES`
        holds a token whose parameters match this one's to the last digit, and as
        explicit ``+a=`` and ``+rf=`` otherwise. The named token is preferred
        because it carries the ellipsoid's identity, but it is a claim about what
        PROJ will substitute, so it is only made when the substitution is known
        to be this ellipsoid.
        """
        return " ".join(
            [
                "+proj=laea",
                f"+lat_0={_number(self.lat_0)}",
                f"+lon_0={_number(self.lon_0)}",
                f"+x_0={_number(self.false_easting)}",
                f"+y_0={_number(self.false_northing)}",
                _proj_ellipsoid(self.ellipsoid),
                "+units=m",
                "+no_defs",
                "+type=crs",
            ]
        )

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
        return {
            "$schema": PROJJSON_SCHEMA,
            "type": "ProjectedCRS",
            "name": self.name,
            "base_crs": {
                "type": "GeographicCRS",
                "name": self.base_crs_name,
                "datum": {
                    "type": "GeodeticReferenceFrame",
                    "name": self.base_datum_name,
                    "ellipsoid": {
                        "name": self.ellipsoid.name,
                        "semi_major_axis": self.ellipsoid.semi_major,
                        "inverse_flattening": self.ellipsoid.inverse_flattening,
                        "id": {"authority": "EPSG", "code": self.ellipsoid.authority_code},
                    },
                },
                "coordinate_system": {
                    "subtype": "ellipsoidal",
                    # The axis order EPSG:4326 declares, which is not the order
                    # forward() takes its arguments in; see the module docstring.
                    "axis": [
                        {
                            "name": "Geodetic latitude",
                            "abbreviation": "Lat",
                            "direction": "north",
                            "unit": "degree",
                        },
                        {
                            "name": "Geodetic longitude",
                            "abbreviation": "Lon",
                            "direction": "east",
                            "unit": "degree",
                        },
                    ],
                },
                "id": {"authority": "EPSG", "code": self.base_crs_code},
            },
            "conversion": {
                "name": f"{self.name} conversion",
                "method": {
                    "name": self.method,
                    "id": {"authority": "EPSG", "code": self.method_code},
                },
                # "metre" and "Cartesian" are the spellings the EPSG registry
                # and the PROJJSON schema use. They are not ours to Americanize
                # or to case-correct, whatever the writing conventions say.
                "parameters": [
                    {
                        "name": "Latitude of natural origin",
                        "value": self.lat_0,
                        "unit": "degree",
                        "id": {"authority": "EPSG", "code": 8801},
                    },
                    {
                        "name": "Longitude of natural origin",
                        "value": self.lon_0,
                        "unit": "degree",
                        "id": {"authority": "EPSG", "code": 8802},
                    },
                    {
                        "name": "False easting",
                        "value": self.false_easting,
                        "unit": "metre",
                        "id": {"authority": "EPSG", "code": 8806},
                    },
                    {
                        "name": "False northing",
                        "value": self.false_northing,
                        "unit": "metre",
                        "id": {"authority": "EPSG", "code": 8807},
                    },
                ],
            },
            "coordinate_system": {
                "subtype": "Cartesian",
                "axis": [
                    {
                        "name": "Easting",
                        "abbreviation": "E",
                        "direction": "east",
                        "unit": "metre",
                    },
                    {
                        "name": "Northing",
                        "abbreviation": "N",
                        "direction": "north",
                        "unit": "metre",
                    },
                ],
            },
        }


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

#: PROJJSON schema the emitted definition declares itself against.
PROJJSON_SCHEMA = "https://proj.org/schemas/v0.7/projjson.schema.json"

#: PROJ's built-in ellipsoid tokens, by EPSG code, with the parameters PROJ
#: substitutes for each. Held with the parameters rather than as bare names so
#: that :meth:`Projection.proj_string` can refuse to write ``+ellps=WGS84`` for
#: an ellipsoid that is not the one PROJ means by it, and fall back to explicit
#: ``+a`` and ``+rf`` instead. From the PROJ ellipsoid documentation.
PROJ_ELLIPSOID_NAMES = MappingProxyType(
    {
        7030: ("WGS84", 6378137.0, 298.257223563),
        7019: ("GRS80", 6378137.0, 298.257222101),
    }
)


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
    """
    if not stem or Path(stem).name != stem:
        raise ValueError(f"stem must be a bare file name, not a path, got {stem!r}")
    root = Path(directory) if directory is not None else default_definition_dir()
    return {suffix: root / f"{stem}.{suffix}" for suffix in ("projjson", "projstring")}


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
    paths = definition_paths(directory, stem=stem)
    contents = _definition_contents(projection)
    # Both files are staged beside their destinations and moved into place only
    # once every one of them is on disk, so an interrupted or failed run cannot
    # leave one file describing this projection and the other describing the
    # last one. os.replace is atomic within a filesystem.
    staged = {}
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
# Private: the projection formulas, and the argument checking they need.


#: How close to the antipode of the center :meth:`Projection.forward` refuses to
#: go, as a floor on ``1 + cos(angular distance)``. That quantity is about
#: ``(pi - c)**2 / 2``, so the guard bites within 8e-4 degrees of the antipode.
#:
#: The floor is not there to stop coordinates growing without bound -- an
#: equal-area azimuthal projection maps the whole ellipsoid into a disk of
#: radius ``2 R``, about two Earth radii, so nothing here diverges. It is there
#: because that disk's rim is where the formula loses its meaning: ``B`` grows
#: like ``1/sqrt(span)`` while ``sin(delta_lon)`` vanishes, and ``span`` itself
#: is a cancelling difference that ends up on either side of zero. Their
#: product at the antipode is therefore ``NaN``, or infinity, or a fraction of a
#: meter from the center -- for a point that belongs on the rim, 12,700 km out.
_ANTIPODE_FLOOR = 1e-10

_WRITE_COMMAND = "`python -m sipnet_calibration.projection --write`"


def _authalic_q(lat_radians, eccentricity_squared):
    """Snyder's ``q``, equation 3-12.

    ``q`` is ``q_p * sin(beta)`` for the authalic latitude ``beta``, and reduces
    to ``2 sin(lat)`` on a sphere.

    Notes
    -----
    Snyder *subtracts* a term ``ln((1 - e sin) / (1 + e sin)) / (2e)``, which is
    itself ``-arctanh(e sin) / e``, so the two minus signs cancel and this adds
    ``arctanh(e sin) / e``. The ``arctanh`` form is used for two
    reasons. The logarithm's argument is ``1 - 2 e sin`` near the equator, where
    taking its logarithm cancels; that costs only nanometers on the ground, but
    the identity is exact and the form is no longer. More usefully, ``arctanh``
    is exactly odd in its argument, so ``q(-90) / q_p`` comes out as exactly
    ``-1`` where the logarithm gives ``-0.9999999999999996``. ``arcsin``
    amplifies that shortfall near ``-1``, which would otherwise spread the south
    pole over a meter of easting with the meridian it is approached along.
    """
    sin_lat = np.sin(lat_radians)
    if eccentricity_squared == 0.0:
        return 2.0 * sin_lat
    eccentricity = math.sqrt(eccentricity_squared)
    return (1 - eccentricity_squared) * (
        sin_lat / (1 - eccentricity_squared * sin_lat * sin_lat)
        + np.arctanh(eccentricity * sin_lat) / eccentricity
    )


def _check_coordinates(lon, lat):
    """The broadcast, finite, in-range ``(lon, lat)`` pair, as ``float64`` arrays.

    The counts in the messages are taken before broadcasting, so they say how
    many values the caller passed that are bad rather than how large the result
    would have been. One NaN against 8000 latitudes is one bad coordinate, and
    reporting 8000 would hide which input is at fault.
    """
    longitude = _as_float_array(lon)
    latitude = _as_float_array(lat)
    scalar = longitude.ndim == 0 and latitude.ndim == 0

    # Checked before the range test, which NaN passes: every comparison against
    # NaN is False, so a NaN latitude would go on to project to NaN and then
    # into an axes limit or a triangulation as a silently dropped point.
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
    # Longitudes are deliberately not wrapped, but they are bounded: nothing
    # would otherwise reject a fill value, and -9999 projects to a real-looking
    # point.
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


def _check_not_antipodal(span, projection):
    """Raise if any point is at the antipode of *projection*'s center.

    *span* is ``1 + cos(c)`` for the authalic angular distance ``c`` from the
    center. The check is not decoration: at the antipode the formula evaluates
    to infinity times ``sin(180 deg)``, which is a finite, plausible-looking
    number rather than an error or a NaN.
    """
    too_close = np.asarray(span) < _ANTIPODE_FLOOR
    if np.any(too_close):
        antipode_lon, antipode_lat = projection.antipode
        raise ValueError(
            f"{int(np.count_nonzero(too_close))} point(s) are at the antipode of the "
            f"projection center, near ({antipode_lon}, {antipode_lat}), where "
            "this projection is undefined"
        )


def _check_bbox_excludes_antipode(projection, west, south, east, north):
    """Raise if the antipode of *projection*'s center lies inside the box.

    :meth:`Projection.projected_bounds` takes its bound over the boundary, which
    is valid only where the transform is defined throughout the box.
    """
    antipode_lon, antipode_lat = projection.antipode
    # How far east of the box's western edge the antipode lies, on the circle,
    # since the box's own longitudes need not lie in [-180, 180]. Reduced to
    # [0, 360) rather than to [-180, 180): the symmetric range cannot express
    # an offset above 180, so a box wider than that would appear to end before
    # the antipode it actually contains.
    offset = (antipode_lon - west) % 360.0
    if offset <= (east - west) and south <= antipode_lat <= north:
        raise ValueError(
            f"the box contains ({antipode_lon}, {antipode_lat}), the antipode of the "
            "projection center, where this projection is undefined; the bounds of such "
            "a box are unbounded, not merely large"
        )


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


def _number(value: float) -> str:
    """A parameter for the PROJ string: shortest exact form, no trailing zeros.

    ``repr`` of a float is the shortest decimal that reads back as the same
    float, so the PROJ string reproduces the dataclass exactly. A fixed number
    of decimal places would not: twelve is short of the seventeen significant
    digits a float can need, so a parameter would round on its way into the file
    that is supposed to be authoritative for it.
    """
    number = float(value)
    if number == 0.0:  # also folds -0.0, which is not a distinct parameter
        return "0"
    text = repr(number)
    # repr switches to exponent notation below 1e-4 and at 1e16 and above, so a
    # parameter of an extreme magnitude is emitted as, say, "1e-13". PROJ reads
    # parameters with strtod and accepts that; no parameter of this projection
    # is anywhere near either threshold.
    return text[:-2] if text.endswith(".0") else text


def _proj_ellipsoid(ellipsoid: Ellipsoid) -> str:
    """``+ellps=`` where PROJ's token means this exact ellipsoid, else the parameters.

    A sphere is written ``+R=``, which is PROJ's parameter for one. It cannot be
    written ``+rf=0``: ``rf`` is the *reverse* flattening, so zero there asks
    PROJ for a flattening of ``1/0``, even though zero is exactly how a sphere
    is spelled in this module, in WKT and in PROJJSON alike.
    """
    known = PROJ_ELLIPSOID_NAMES.get(ellipsoid.authority_code)
    if known is not None:
        name, semi_major, inverse_flattening = known
        if (semi_major, inverse_flattening) == (
            ellipsoid.semi_major,
            ellipsoid.inverse_flattening,
        ):
            return f"+ellps={name}"
    if ellipsoid.inverse_flattening == 0.0:
        return f"+R={_number(ellipsoid.semi_major)}"
    return f"+a={_number(ellipsoid.semi_major)} +rf={_number(ellipsoid.inverse_flattening)}"


def _definition_contents(projection: Projection) -> dict[str, str]:
    """The text of each interchange file, keyed as :func:`definition_paths` keys them."""
    return {
        "projjson": json.dumps(projection.projjson(), indent=2) + "\n",
        "projstring": projection.proj_string() + "\n",
    }


def _main(argv: list[str] | None = None) -> int:
    """``python -m sipnet_calibration.projection [--write | --check]``."""
    parser = argparse.ArgumentParser(
        prog="python -m sipnet_calibration.projection",
        description=(
            "Write or verify the interchange files for the project's display "
            "projection. The parameters live in the dataclass; these files are "
            "generated from it."
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--write", action="store_true", help="regenerate the files, overwriting them"
    )
    action.add_argument(
        "--check",
        action="store_true",
        help="verify the stored files match the dataclass; the default",
    )
    parser.add_argument(
        "--directory",
        default=None,
        help="where the files live; defaults to the package's projections/ directory",
    )
    arguments = parser.parse_args(argv)

    if arguments.write:
        try:
            written = write_definitions(directory=arguments.directory)
        except OSError as error:
            print(f"error: {error}")
            return 1
        for key, path in written.items():
            print(f"wrote {key}: {path}")
        return 0
    try:
        check_definitions(directory=arguments.directory)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}")
        return 1
    print(f"the stored definition matches {SITE_PROJECTION.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
