"""Tests for the display projection and its forward transform.

The transform is validated against **published coordinates**, not against its
own properties. Equal-areaness is not evidence that an implementation is right:
it follows from the functional form even when the parameters are wrong, which is
how the wrong cone constant in the analysis on issue #4 survived an equal-area
check while placing coordinates thousands of km out. So the cases in
``TestForwardAgainstPublishedCoordinates`` come first, from two independent
sources, and the property tests are downstream of them.

Two references, both offline:

- **Snyder** (1987), *Map Projections: A Working Manual*, USGS Professional
  Paper 1395, Appendix A: the oblique ellipsoidal Lambert Azimuthal Equal-Area
  worked example, on Clarke 1866. It exercises the authalic-latitude path with
  a center away from the equator, and the book's seven-figure intermediates
  bound the agreement at a few millimeters.
- **PROJ**, ``test/gie/builtins.gie``, the ``+proj=laea +ellps=GRS80`` block:
  PROJ's own regression values, to its own 0.1 mm tolerance. Four of its five
  points are 2.24 degrees of arc from the origin, where every azimuthal
  projection agrees to third order in the angular distance, so the fifth at
  (150, 50) is what actually pins the functional form; there is a test that
  says so. The same block documents that PROJ rejects the antipode, though only
  this implementation's refusal is asserted here -- PROJ's own cannot be run on
  this machine.

The scale factors are pinned separately, on a sphere, where the authalic
latitude is the geodetic one and the closed-form spherical values are therefore
exact. That is what fixes the *shape* of the distortion field, which an area
check cannot see.

The cases over the real site pool read ``data/processed/sites/sites.csv`` and
are skipped when it is absent. They are what pin the distortion figures the
projection was chosen on, so that a change to the parameters that quietly makes
the Arctic worse fails here rather than in a figure.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from sipnet_calibration.projection import (
    LAEA_METHOD_CODE,
    SITE_PROJECTION,
    WGS84,
    Ellipsoid,
    Projection,
    check_definitions,
    default_definition_dir,
    definition_paths,
    write_definitions,
)
from sipnet_calibration.projection import _main as projection_main
from sipnet_calibration.projection import _number
from sipnet_calibration.sites import EXTENTS, SITE_GRID, default_sites_path, load_sites, select_sites

# Snyder, Appendix A, oblique ellipsoidal Lambert Azimuthal Equal-Area.
# Clarke 1866 is given there as a and e**2 rather than as an inverse flattening.
SNYDER_CLARKE_1866 = {"semi_major": 6378206.4, "eccentricity_squared": 0.00676866}
SNYDER_CENTER = {"lat_0": 40.0, "lon_0": -100.0}
SNYDER_POINT = (-110.0, 30.0)
SNYDER_EXPECTED = (-965932.11, -1056814.93)
#: The book carries seven significant figures, so its own rounding is worth a
#: few millimeters of the result; nothing here is asserted tighter than that.
SNYDER_TOLERANCE_M = 0.02

# PROJ test/gie/builtins.gie, operation "+proj=laea +ellps=GRS80", forward.
# Defaults apply, so the center is (0, 0): the equatorial aspect, which the
# oblique formulas must also cover.
PROJ_LAEA_CASES = [
    ((2.0, 1.0), (222602.471450095, 110589.827224410)),
    ((2.0, -1.0), (222602.471450095, -110589.827224409)),
    ((-2.0, 1.0), (-222602.471450095, 110589.827224410)),
    ((-2.0, -1.0), (-222602.471450095, -110589.827224409)),
    ((150.0, 50.0), (4372597.1888, 10352365.4614)),
]
PROJ_TOLERANCE_M = 1e-4

# Measured over data/processed/sites/sites.csv under SITE_PROJECTION, and the
# reason this projection was chosen over the alternatives; see issue #4. Held as
# ceilings rather than as the measured values, since a slightly better number is
# not a regression.
SITE_OMEGA_CEILING_DEG = 14.0
SITE_ANISOTROPY_CEILING = 1.3
SITE_AREA_TOLERANCE = 1e-6


class TestEllipsoid:
    def test_wgs84_matches_the_registered_parameters(self):
        """``a = 6378137`` exactly, ``1/f = 298.257223563``, EPSG 7030."""
        assert WGS84.semi_major == 6378137.0
        assert WGS84.inverse_flattening == 298.257223563
        assert WGS84.authority_code == 7030

    def test_eccentricity_squared_follows_from_the_inverse_flattening(self):
        """``e**2 = 2f - f**2``, giving the documented 6.69437999e-3 for WGS 84."""
        flattening = 1.0 / 298.257223563
        assert WGS84.flattening == pytest.approx(flattening, rel=1e-15)
        assert WGS84.eccentricity_squared == pytest.approx(
            2 * flattening - flattening**2, rel=1e-15
        )
        assert WGS84.eccentricity_squared == pytest.approx(0.00669437999014, rel=1e-12)
        assert WGS84.eccentricity == pytest.approx(math.sqrt(WGS84.eccentricity_squared))

    def test_from_eccentricity_squared_round_trips(self):
        """Snyder's ``e**2`` for Clarke 1866 comes back unchanged through ``1/f``."""
        clarke = Ellipsoid.from_eccentricity_squared("Clarke 1866", **SNYDER_CLARKE_1866)
        assert clarke.eccentricity_squared == pytest.approx(
            SNYDER_CLARKE_1866["eccentricity_squared"], rel=1e-15
        )
        assert clarke.inverse_flattening == pytest.approx(294.9786, rel=1e-6)
        # 0 is not a placeholder: it is what makes proj_string spell the
        # parameters out instead of claiming a +ellps= token.
        assert clarke.authority_code == 0

    def test_from_eccentricity_squared_writes_a_sphere_as_zero_inverse_flattening(self):
        """``1/f`` is undefined for a sphere, so zero stands for one, and the
        eccentricity comes back as zero rather than as a division by it."""
        sphere = Ellipsoid.from_eccentricity_squared("Sphere", 6371007.0, 0.0)
        assert sphere.inverse_flattening == 0.0
        assert sphere.flattening == 0.0
        assert sphere.eccentricity_squared == 0.0

    def test_from_eccentricity_squared_rejects_an_impossible_shape(self):
        for bad in (-1e-9, 1.0, 2.0):
            with pytest.raises(ValueError, match="eccentricity_squared"):
                Ellipsoid.from_eccentricity_squared("bad", 6378137.0, bad)

    def test_from_eccentricity_squared_round_trips_at_a_tiny_eccentricity(self):
        """The textbook ``1 - sqrt(1 - e**2)`` cancels here, losing four
        significant digits by ``e**2 = 1e-12``; the form used does not."""
        for eccentricity_squared in (1e-12, 1e-9, 1e-6, 1e-3):
            ellipsoid = Ellipsoid.from_eccentricity_squared(
                "tiny", 6378137.0, eccentricity_squared
            )
            assert ellipsoid.eccentricity_squared == pytest.approx(
                eccentricity_squared, rel=1e-14
            )

    def test_rejects_a_figure_that_is_not_an_ellipsoid(self):
        """A flattening of 1 or more gives a NaN eccentricity, and a negative
        radius silently point-reflects the whole map."""
        for semi_major in (0.0, -6378137.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="semi_major"):
                Ellipsoid("bad", semi_major, 298.257223563, 0)
        for inverse_flattening in (0.5, 1.0 - 1e-9, -298.0, float("nan")):
            with pytest.raises(ValueError, match="inverse_flattening"):
                Ellipsoid("bad", 6378137.0, inverse_flattening, 0)
        # Zero is the sphere, and 1 exactly is the degenerate limit that is
        # allowed through as a shape: neither raises.
        assert Ellipsoid("sphere", 6378137.0, 0.0, 0).eccentricity_squared == 0.0

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            WGS84.semi_major = 1.0  # type: ignore[misc]


class TestProjectionDefinition:
    def test_site_projection_is_the_agreed_laea(self):
        """LAEA, center 50 N/100 W, no false origin, WGS 84, EPSG method 9820."""
        assert SITE_PROJECTION.lat_0 == 50.0
        assert SITE_PROJECTION.lon_0 == -100.0
        assert SITE_PROJECTION.false_easting == 0.0
        assert SITE_PROJECTION.false_northing == 0.0
        assert SITE_PROJECTION.ellipsoid == WGS84
        assert SITE_PROJECTION.base_crs_code == 4326
        assert SITE_PROJECTION.method_code == LAEA_METHOD_CODE == 9820

    def test_rejects_a_method_it_cannot_transform(self):
        """A projection naming another EPSG method must not be constructible."""
        with pytest.raises(ValueError, match="9822"):
            Projection(name="Albers", lat_0=50.0, lon_0=-100.0, method_code=9822)

    def test_rejects_the_polar_aspect(self):
        """``lat_0`` at a pole needs its own formulas, so it raises rather than
        dividing by ``cos(lat_0)``."""
        for pole in (90.0, -90.0):
            with pytest.raises(ValueError, match="polar aspect"):
                Projection(name="polar", lat_0=pole, lon_0=0.0)

    def test_rejects_an_out_of_range_origin(self):
        for bad_lat in (100.0, -100.0):
            with pytest.raises(ValueError, match=r"lat_0 must be in \[-90, 90\]"):
                Projection(name="bad", lat_0=bad_lat, lon_0=0.0)
        for bad_lon in (400.0, -400.0):
            with pytest.raises(ValueError, match=r"lon_0 must be in \[-360, 360\]"):
                Projection(name="bad", lat_0=50.0, lon_0=bad_lon)

    def test_rejects_a_non_finite_false_origin(self):
        """It is added to every projected coordinate and written into the
        definition files, so a NaN here defeats the guard in ``forward`` and
        puts ``+x_0=nan`` in a file meant to be authoritative."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="false_easting must be finite"):
                Projection(name="bad", lat_0=50.0, lon_0=-100.0, false_easting=bad)
            with pytest.raises(ValueError, match="false_northing must be finite"):
                Projection(name="bad", lat_0=50.0, lon_0=-100.0, false_northing=bad)

    def test_accepts_a_near_polar_center_and_an_unwrapped_origin(self):
        """The guards reject the polar aspect and out-of-range values, not a
        legitimate high-latitude center -- an Alaska panel is an obvious future
        caller -- nor the documented ``[-360, 360]`` longitude range."""
        assert Projection(name="alaska", lat_0=85.0, lon_0=-154.0).lat_0 == 85.0
        assert Projection(name="wrapped", lat_0=50.0, lon_0=260.0).lon_0 == 260.0
        # The polar guard must reject a pole and nothing else: a center a
        # hundredth of a degree short of one is still an oblique projection,
        # and it projects.
        near_polar = Projection(name="near polar", lat_0=89.99, lon_0=0.0)
        assert near_polar.forward(0.0, 89.99) == pytest.approx((0.0, 0.0), abs=1e-9)
        assert all(math.isfinite(value) for value in near_polar.forward(45.0, 60.0))

    def test_rejects_a_method_name_that_disagrees_with_the_code(self):
        """The name is written into PROJJSON beside the code, so the two must
        not be allowed to describe different methods."""
        with pytest.raises(ValueError, match="Albers"):
            Projection(name="mislabeled", lat_0=50.0, lon_0=-100.0, method="Albers Equal Area")

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            SITE_PROJECTION.lat_0 = 0.0  # type: ignore[misc]


class TestForwardAgainstPublishedCoordinates:
    def test_snyders_oblique_example(self):
        """Snyder Appendix A: Clarke 1866, center 40 N/100 W, point 30 N/110 W."""
        projection = Projection(
            name="Snyder Appendix A",
            ellipsoid=Ellipsoid.from_eccentricity_squared("Clarke 1866", **SNYDER_CLARKE_1866),
            **SNYDER_CENTER,
        )
        x, y = projection.forward(*SNYDER_POINT)
        assert x == pytest.approx(SNYDER_EXPECTED[0], abs=SNYDER_TOLERANCE_M)
        assert y == pytest.approx(SNYDER_EXPECTED[1], abs=SNYDER_TOLERANCE_M)

    @pytest.mark.parametrize(("lonlat", "expected"), PROJ_LAEA_CASES)
    def test_projs_regression_values(self, lonlat, expected):
        """``+proj=laea +ellps=GRS80`` from PROJ's own test suite, to 0.1 mm."""
        x, y = _proj_reference_projection().forward(*lonlat)
        assert x == pytest.approx(expected[0], abs=PROJ_TOLERANCE_M)
        assert y == pytest.approx(expected[1], abs=PROJ_TOLERANCE_M)

    def test_the_far_field_case_is_what_pins_the_functional_form(self):
        """Why PROJ's (150, 50) case earns its place beside four near-origin
        ones. On a sphere the radius from the center must be ``2R sin(c/2)``,
        the equal-area law, and every azimuthal projection agrees with that to
        third order in ``c``: an azimuthal *equidistant* radius, ``R c``,
        differs from it by 6.3e-5 at PROJ's near-origin points and by 22
        percent at its far one. So the near-origin cases barely constrain the
        law and the far one does."""
        radius = 6371007.0
        spherical = Projection(
            name="spherical",
            lat_0=0.0,
            lon_0=0.0,
            ellipsoid=Ellipsoid.from_eccentricity_squared("Sphere", radius, 0.0),
        )
        for lon, lat in [(2.0, 1.0), (150.0, 50.0)]:
            distance = float(_angular_distance(spherical, lon, lat))
            x, y = spherical.forward(lon, lat)
            equal_area = 2 * radius * math.sin(distance / 2)
            equidistant = radius * distance
            assert math.hypot(x, y) == pytest.approx(equal_area, rel=1e-12)
            discrimination = abs(equidistant - equal_area) / equal_area
            if distance < math.radians(5):
                assert discrimination < 1e-4
            else:
                assert discrimination > 0.1


class TestForwardBehavior:
    def test_the_center_maps_to_the_false_origin(self):
        """``(lon_0, lat_0)`` goes to ``(false_easting, false_northing)``."""
        assert SITE_PROJECTION.forward(
            SITE_PROJECTION.lon_0, SITE_PROJECTION.lat_0
        ) == pytest.approx((0.0, 0.0), abs=1e-9)

    def test_false_origin_translates_and_nothing_else(self):
        """A projection differing only in its false origin differs only by that
        constant offset, at every site."""
        shifted = Projection(
            name="shifted",
            lat_0=SITE_PROJECTION.lat_0,
            lon_0=SITE_PROJECTION.lon_0,
            false_easting=4321000.0,
            false_northing=3210000.0,
        )
        lon = np.array([-160.0, -100.0, -30.0, -70.0])
        lat = np.array([65.0, 10.0, 80.0, 45.0])
        x, y = SITE_PROJECTION.forward(lon, lat)
        shifted_x, shifted_y = shifted.forward(lon, lat)
        assert shifted_x == pytest.approx(x + 4321000.0, abs=1e-6)
        assert shifted_y == pytest.approx(y + 3210000.0, abs=1e-6)

    def test_scalars_in_scalars_out(self):
        x, y = SITE_PROJECTION.forward(-90.0, 45.0)
        # `type` rather than `isinstance`: np.float64 is a subclass of float,
        # and is not a scalar for json.dumps or for a formatted axis label.
        assert type(x) is float
        assert type(y) is float

    def test_broadcasts_and_preserves_shape(self):
        """A scalar longitude against an array of latitudes, and 2-D input."""
        x, y = SITE_PROJECTION.forward(-100.0, np.array([40.0, 50.0, 60.0]))
        assert x.shape == y.shape == (3,)
        # A scalar longitude at the central meridian puts every point on x = 0.
        assert x == pytest.approx(np.zeros(3), abs=1e-9)

        grid_lon, grid_lat = np.meshgrid(np.linspace(-140, -60, 4), np.linspace(20, 70, 3))
        x, y = SITE_PROJECTION.forward(grid_lon, grid_lat)
        assert x.shape == y.shape == grid_lon.shape

    def test_returns_float64_for_integer_input(self):
        x, y = SITE_PROJECTION.forward(np.array([-100, -90]), np.array([50, 40]))
        assert x.dtype == y.dtype == np.float64

    def test_longitudes_are_not_wrapped(self):
        """-190 and 170 are the same meridian and must project the same, since a
        vendored coastline can carry either."""
        assert SITE_PROJECTION.forward(-190.0, 10.0) == pytest.approx(
            SITE_PROJECTION.forward(170.0, 10.0), abs=1e-6
        )

    def test_rejects_non_finite_coordinates(self):
        """``NaN`` cannot be allowed through: it propagates into axes limits and
        into the triangulation as a silently dropped point."""
        for lon, lat in [(np.nan, 50.0), (-100.0, np.nan), (np.inf, 50.0)]:
            with pytest.raises(ValueError, match="not finite"):
                SITE_PROJECTION.forward(lon, lat)
        with pytest.raises(ValueError, match="2 longitude"):
            SITE_PROJECTION.forward(np.array([-100.0, np.nan, np.nan]), 50.0)

    def test_counts_bad_values_before_broadcasting(self):
        """One bad scalar against 8000 latitudes is one bad coordinate. The
        count is the whole diagnostic value of the message -- it says whether
        one site is wrong or the whole table is -- so it must not report the
        broadcast size."""
        with pytest.raises(ValueError, match="1 longitude\\(s\\) and 0 latitude"):
            SITE_PROJECTION.forward(np.nan, np.full(8000, 50.0))

    def test_rejects_a_masked_coordinate_rather_than_projecting_its_fill(self):
        """``np.asarray`` drops a mask silently, so a masked entry would project
        its fill value to a finite, plausible coordinate."""
        masked_lon = np.ma.masked_array([-100.0, 1e30], mask=[False, True])
        masked_lat = np.ma.masked_array([50.0, 50.0], mask=[False, True])
        with pytest.raises(ValueError, match="not finite"):
            SITE_PROJECTION.forward(masked_lon, masked_lat)

    def test_rejects_a_longitude_fill_value(self):
        """Longitudes are deliberately not wrapped, but they are bounded:
        nothing would otherwise reject -9999, which projects to a real-looking
        point."""
        for bad in (-9999.0, 1e20):
            with pytest.raises(ValueError, match=r"outside \[-360, 360\]"):
                SITE_PROJECTION.forward(bad, 50.0)

    def test_rejects_latitudes_outside_the_poles(self):
        with pytest.raises(ValueError, match="outside"):
            SITE_PROJECTION.forward(-100.0, 90.5)
        # One corrupt row among good ones must fail too: at 999 degrees the
        # projection returns a finite -11,581 km of northing.
        with pytest.raises(ValueError, match="1 latitude"):
            SITE_PROJECTION.forward(
                np.array([-100.0, -95.0, -90.0]), np.array([45.0, 999.0, 50.0])
            )
        # The message says which order the arguments go in, because swapping
        # them is the way this is usually reached.
        with pytest.raises(ValueError, match="longitude first"):
            SITE_PROJECTION.forward(50.0, -100.0)

    def test_accepts_the_poles_themselves(self):
        """The authalic latitude at a pole is the ratio of two quantities that
        are equal up to rounding, so the pole must not fall out of ``arcsin``."""
        for pole in (90.0, -90.0):
            x, y = SITE_PROJECTION.forward(-100.0, pole)
            assert math.isfinite(x) and math.isfinite(y)

        # A strongly flattened ellipsoid, where a formula that was not exactly
        # odd in latitude would put the ratio outside [-1, 1] and arcsin would
        # return NaN.
        flattened = Projection(
            name="flattened",
            lat_0=50.0,
            lon_0=-100.0,
            ellipsoid=Ellipsoid.from_eccentricity_squared("e2 = 0.3", 6378137.0, 0.3),
        )
        for pole in (90.0, -90.0):
            x, y = flattened.forward(-100.0, pole)
            assert math.isfinite(x) and math.isfinite(y)

    def test_a_pole_is_one_point_whatever_meridian_it_is_approached_along(self):
        """What keeps this true is that ``_authalic_q`` is exactly odd in
        latitude. Written with Snyder's logarithm instead of the equivalent
        ``arctanh``, ``q(-90) / q_p`` falls short of -1 by 4e-16, ``arcsin``
        amplifies that near -1, and the south pole spreads over a meter of
        easting with the meridian it is approached along."""
        for pole in (90.0, -90.0):
            x, y = SITE_PROJECTION.forward(np.array([-180.0, -100.0, -10.0, 179.9]), pole)
            assert x == pytest.approx(np.zeros(4), abs=1e-6)
            assert y == pytest.approx(np.full(4, y[0]), abs=1e-6)

    def test_rejects_the_antipode_of_the_center(self):
        """The formula there is infinity times ``sin(180 deg)``, which is a
        finite plausible number rather than an error. PROJ rejects it too."""
        antipode = (SITE_PROJECTION.lon_0 + 180.0, -SITE_PROJECTION.lat_0)
        with pytest.raises(ValueError, match="antipode"):
            SITE_PROJECTION.forward(*antipode)
        with pytest.raises(ValueError, match="antipode"):
            SITE_PROJECTION.forward(np.array([-100.0, antipode[0]]), np.array([50.0, antipode[1]]))

    def test_accepts_a_point_far_from_the_center_but_short_of_the_antipode(self):
        """The guard must not narrow the projection's domain: PROJ accepts 124
        degrees of arc, and so must this."""
        x, y = _proj_reference_projection().forward(150.0, 50.0)
        assert math.isfinite(x) and math.isfinite(y)
        # 179 degrees of arc from the center is still projectable.
        x, y = SITE_PROJECTION.forward(SITE_PROJECTION.lon_0 + 179.0, -SITE_PROJECTION.lat_0)
        assert math.isfinite(x) and math.isfinite(y)


class TestEqualArea:
    def test_area_scale_is_unity_across_the_domain(self):
        """A property of the method, asserted downstream of the published-value
        cases rather than as evidence for them: the areal scale factor is 1 at
        the center, at the far south and at the far north alike."""
        lon = np.array([-100.0, -60.0, -170.0, -25.0])
        lat = np.array([50.0, 7.5, 68.5, 82.5])
        area, _, _ = _tissot(SITE_PROJECTION, lon, lat)
        assert area == pytest.approx(np.ones(4), abs=1e-6)

    def test_scale_factors_match_the_closed_form_spherical_values(self):
        """For an ellipsoidal LAEA the distortion is that of a spherical LAEA on
        the authalic sphere, so on a sphere the radial and tangential scale
        factors must be exactly ``cos(c/2)`` and ``sec(c/2)`` for angular
        distance ``c``. This is what pins the shape of the distortion field,
        where the area check alone cannot: an area check passes for any
        equal-area map, right parameters or wrong."""
        radius = 6371007.0
        spherical = Projection(
            name="spherical",
            lat_0=SITE_PROJECTION.lat_0,
            lon_0=SITE_PROJECTION.lon_0,
            ellipsoid=Ellipsoid.from_eccentricity_squared("Sphere", radius, 0.0),
        )
        lon = np.array([-140.0, -60.0, -100.0, -20.0])
        lat = np.array([30.0, 70.0, 7.0, 60.0])
        distance = _angular_distance(spherical, lon, lat)

        area, scale_max, scale_min = _tissot(spherical, lon, lat, radius=radius)
        assert area == pytest.approx(np.ones(4), abs=1e-6)
        assert scale_max == pytest.approx(1.0 / np.cos(distance / 2), rel=1e-6)
        assert scale_min == pytest.approx(np.cos(distance / 2), rel=1e-6)


@pytest.mark.skipif(
    not default_sites_path().exists(), reason="processed/sites/sites.csv is not present"
)
class TestSiteDomainDistortion:
    def test_every_site_projects_to_a_finite_coordinate(self):
        x, y = SITE_PROJECTION.forward(*_site_coordinates())
        assert np.isfinite(x).all()
        assert np.isfinite(y).all()

    def test_angular_deformation_stays_under_the_ceiling(self):
        """No site exceeds 14 degrees, against 107 for the ESRI:102003 that the
        published reanalysis figures used; this is the measurement the choice of
        projection rests on."""
        assert _site_metrics()["omega"].max() < SITE_OMEGA_CEILING_DEG

    def test_anisotropy_stays_under_the_ceiling(self):
        """No site exceeds 1.3:1, against 9.2:1 for ESRI:102003. This one is
        load-bearing beyond appearance: the Delaunay triangulation is computed
        after projecting, and the long-edge mask is a projected length."""
        assert _site_metrics()["anisotropy"].max() < SITE_ANISOTROPY_CEILING

    def test_area_is_preserved_over_the_whole_pool(self):
        area = _site_metrics()["area"]
        assert np.abs(area - 1.0).max() < SITE_AREA_TOLERANCE

    def test_the_most_distorted_site_is_the_one_farthest_from_the_center(self):
        """Distortion grows monotonically with angular distance from the center,
        so the extremes of the pool are where the ceilings are tested, and a
        change of center moves both together."""
        metrics = _site_metrics()
        assert int(np.argmax(metrics["omega"])) == int(np.argmax(metrics["distance"]))
        assert int(np.argmax(metrics["anisotropy"])) == int(np.argmax(metrics["distance"]))


class TestProjectedBounds:
    def test_bounds_of_a_named_extent(self):
        """``EXTENTS["CONUS"]`` gives axes limits in the same projected meters as
        ``forward`` returns."""
        x_min, y_min, x_max, y_max = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])
        assert x_min < x_max
        assert y_min < y_max
        # The box straddles the central meridian, so it spans x = 0.
        assert x_min < 0.0 < x_max
        # Every corner of the box is inside the bounds it produced.
        west, south, east, north = EXTENTS["CONUS"]
        x, y = SITE_PROJECTION.forward(
            np.array([west, east, west, east]), np.array([south, south, north, north])
        )
        assert x.min() >= x_min and x.max() <= x_max
        assert y.min() >= y_min and y.max() <= y_max

    def test_boundary_sampling_beats_the_four_corners(self):
        """A longitude/latitude box projects to a curved quadrilateral, so the
        true bound is outside the corner-only one. The far edge from the
        projection center is where the corners miss it, which for a center at
        50 N and the CONUS box is its southern edge."""
        west, south, east, north = EXTENTS["CONUS"]
        corner_x, corner_y = SITE_PROJECTION.forward(
            np.array([west, east, west, east]), np.array([south, south, north, north])
        )
        _, y_min, _, _ = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])
        # Measured at 392 km; asserted well below that, but far enough above
        # zero that only real boundary sampling passes.
        assert corner_y.min() - y_min > 300e3

    def test_contains_every_site_inside_the_box(self):
        """Whatever ``select_sites(bbox=...)`` returns must project inside the
        limits the same box gives, or a figure clips its own data."""
        if not default_sites_path().exists():
            pytest.skip("processed/sites/sites.csv is not present")
        sites = select_sites(load_sites(), bbox=EXTENTS["CONUS"])
        x, y = SITE_PROJECTION.forward(sites["lon"].to_numpy(), sites["lat"].to_numpy())
        x_min, y_min, x_max, y_max = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])
        assert x.min() >= x_min and x.max() <= x_max
        assert y.min() >= y_min and y.max() <= y_max

    def test_rejects_a_malformed_box(self):
        """Four values, west of east, south of north -- the same contract as
        ``select_sites``."""
        with pytest.raises(ValueError, match="west, south, east, north"):
            SITE_PROJECTION.projected_bounds((-125.0, 24.0, -66.0))
        with pytest.raises(ValueError, match="antimeridian"):
            SITE_PROJECTION.projected_bounds((-66.0, 24.0, -125.0, 50.0))
        with pytest.raises(ValueError, match="north of north"):
            SITE_PROJECTION.projected_bounds((-125.0, 50.0, -66.0, 24.0))
        with pytest.raises(ValueError, match="samples_per_edge"):
            SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"], samples_per_edge=1)
        with pytest.raises(ValueError, match="samples_per_edge must be an integer"):
            SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"], samples_per_edge=8.0)
        # Five values is as wrong as three, and both must say so rather than
        # silently using the first four.
        with pytest.raises(ValueError, match="5 value"):
            SITE_PROJECTION.projected_bounds((-125.0, 24.0, -66.0, 50.0, 0.0))
        # Everything unusable is a ValueError, including what would otherwise
        # surface as a TypeError from unpacking or a bare numpy message.
        for bad in (None, 5, {"west": -125.0}, (-125.0, 24.0, -66.0, np.nan)):
            with pytest.raises(ValueError, match="bbox"):
                SITE_PROJECTION.projected_bounds(bad)

    def test_coerces_numeric_strings_the_way_it_documents(self):
        assert SITE_PROJECTION.projected_bounds(("-125", "24", "-66", "50")) == pytest.approx(
            SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])
        )

    def test_refuses_a_box_containing_the_antipode_of_the_center(self):
        """The boundary-sampling argument holds only where the transform is
        defined throughout the box. With the antipode inside, the interior holds
        a singularity and the boundary bound is not a bound -- an interior point
        lands 48 km outside it, growing without limit toward the antipode."""
        antipode_box = (70.0, -60.0, 90.0, -40.0)
        with pytest.raises(ValueError, match="antipode"):
            SITE_PROJECTION.projected_bounds(antipode_box)

    def test_more_samples_do_not_change_the_answer_materially(self):
        """The default is past convergence for these extents. The tolerance is
        in meters against figures thousands of kilometers across, so it is
        several orders of magnitude below one rendered pixel."""
        for name, box in EXTENTS.items():
            default = np.array(SITE_PROJECTION.projected_bounds(box))
            fine = np.array(SITE_PROJECTION.projected_bounds(box, samples_per_edge=8192))
            assert np.abs(default - fine).max() < 100.0, name


class TestInterchangeFiles:
    def test_proj_string_carries_every_parameter(self):
        """``+proj=laea``, both origin parameters, the false origin, the
        ellipsoid and the units, so that pasting it elsewhere reproduces this
        projection and not a defaulted one."""
        assert SITE_PROJECTION.proj_string() == (
            "+proj=laea +lat_0=50 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 "
            "+units=m +no_defs +type=crs"
        )

    def test_proj_string_spells_out_an_ellipsoid_proj_does_not_name(self):
        """``+ellps=`` is a claim about what PROJ will substitute, so it is only
        made for an ellipsoid whose parameters are the ones PROJ means."""
        projection = Projection(
            name="Snyder Appendix A",
            ellipsoid=Ellipsoid.from_eccentricity_squared("Clarke 1866", **SNYDER_CLARKE_1866),
            **SNYDER_CENTER,
        )
        assert "+ellps=" not in projection.proj_string()
        assert "+a=6378206.4" in projection.proj_string()
        assert "+rf=294.978610787262" in projection.proj_string()

    def test_proj_string_names_grs80_when_the_parameters_are_grs80s(self):
        """The lookup arm, which nothing else reaches: the Clarke 1866 fixture
        misses the table entirely on its authority code of 0."""
        assert "+ellps=GRS80" in _proj_reference_projection().proj_string()

    def test_proj_string_refuses_the_ellps_token_for_a_mismatched_ellipsoid(self):
        """EPSG 7030 with parameters that are not PROJ's WGS84 must not claim
        the token. Holding the parameters in ``PROJ_ELLIPSOID_NAMES`` rather
        than bare names is the whole point, and this is the arm that uses them."""
        impostor = Projection(
            name="impostor",
            lat_0=50.0,
            lon_0=-100.0,
            ellipsoid=Ellipsoid("WGS 84 (modified)", 6378137.0, 298.0, 7030),
        )
        assert "+ellps=" not in impostor.proj_string()
        assert "+a=6378137 +rf=298" in impostor.proj_string()

    def test_proj_string_writes_a_sphere_as_a_radius(self):
        """``+rf`` is the *reverse* flattening, so ``+rf=0`` asks PROJ for a
        flattening of 1/0, even though zero is how this module, WKT and
        PROJJSON all spell a sphere. PROJ's parameter for one is ``+R``."""
        spherical = Projection(
            name="spherical",
            lat_0=0.0,
            lon_0=0.0,
            ellipsoid=Ellipsoid.from_eccentricity_squared("Sphere", 6371007.0, 0.0),
        )
        assert "+R=6371007" in spherical.proj_string()
        assert "+rf=" not in spherical.proj_string()
        assert spherical.projjson()["base_crs"]["datum"]["ellipsoid"][
            "inverse_flattening"
        ] == 0.0

    def test_number_formatting_round_trips_exactly(self):
        """A fixed number of decimal places is short of the seventeen
        significant digits a float can need, so a parameter would round on its
        way into the file that is supposed to be authoritative for it."""
        for value in (50.0, -100.0, 0.0, 6378137.0, 298.257223563, 294.978610787262, 1 / 3):
            assert float(_number(value)) == value
        assert _number(-0.0) == "0"
        assert _number(0.0) == "0"
        assert "e" not in _number(6378137.0)

    def test_projjson_declares_the_epsg_method_and_parameter_codes(self):
        """Method 9820, parameters 8801, 8802, 8806 and 8807, and the base CRS
        with the ellipsoid the transform actually used.

        Every value here is a literal rather than a comparison against the
        dataclass field it came from. The tracked definition file cannot serve
        as the check on its own: it is generated from this same source, and the
        workflow this module documents regenerates it, so a wrong unit or a
        swapped axis order would survive both.
        """
        document = SITE_PROJECTION.projjson()
        assert document["$schema"] == "https://proj.org/schemas/v0.7/projjson.schema.json"
        assert document["type"] == "ProjectedCRS"

        conversion = document["conversion"]
        assert conversion["method"]["name"] == "Lambert Azimuthal Equal Area"
        assert conversion["method"]["id"] == {"authority": "EPSG", "code": 9820}
        assert [parameter["id"]["code"] for parameter in conversion["parameters"]] == [
            8801,
            8802,
            8806,
            8807,
        ]
        # Two degrees then two meters. A unit swapped here is a definition that
        # reads without error and puts the data thousands of km away.
        assert [parameter["unit"] for parameter in conversion["parameters"]] == [
            "degree",
            "degree",
            "metre",
            "metre",
        ]

        base = document["base_crs"]
        assert base["name"] == "WGS 84"
        assert base["id"] == {"authority": "EPSG", "code": 4326}
        assert base["datum"]["type"] == "GeodeticReferenceFrame"
        assert base["datum"]["name"] == "World Geodetic System 1984"
        ellipsoid = base["datum"]["ellipsoid"]
        assert ellipsoid["name"] == "WGS 84"
        assert ellipsoid["semi_major_axis"] == 6378137.0
        assert ellipsoid["inverse_flattening"] == 298.257223563
        assert ellipsoid["id"] == {"authority": "EPSG", "code": 7030}
        # The base CRS declares the axis order EPSG:4326 does, which is not the
        # order forward() takes its arguments in.
        assert base["coordinate_system"]["subtype"] == "ellipsoidal"
        assert [
            axis["abbreviation"] for axis in base["coordinate_system"]["axis"]
        ] == ["Lat", "Lon"]

        # The projected CRS itself is registered with nobody, which is why the
        # definition is stored in this repository at all.
        assert "id" not in document
        assert document["coordinate_system"]["subtype"] == "Cartesian"
        axes = [axis["name"] for axis in document["coordinate_system"]["axis"]]
        assert axes == ["Easting", "Northing"]
        assert {axis["unit"] for axis in document["coordinate_system"]["axis"]} == {"metre"}

    def test_projjson_parameter_values_equal_the_dataclass_fields(self):
        """The serialization is derived, so nothing can be stale in one place
        and current in the other."""
        values = {
            parameter["name"]: parameter["value"]
            for parameter in SITE_PROJECTION.projjson()["conversion"]["parameters"]
        }
        assert values == {
            "Latitude of natural origin": SITE_PROJECTION.lat_0,
            "Longitude of natural origin": SITE_PROJECTION.lon_0,
            "False easting": SITE_PROJECTION.false_easting,
            "False northing": SITE_PROJECTION.false_northing,
        }

    def test_serializers_distinguish_the_two_false_origin_parameters(self):
        """``SITE_PROJECTION`` has both at zero, so it cannot tell them apart.
        Copy-paste between two adjacent four-line parameter dicts is how they
        get swapped, and the swap is invisible to every other test here."""
        offset = Projection(
            name="offset",
            lat_0=52.0,
            lon_0=10.0,
            false_easting=4321000.0,
            false_northing=3210000.0,
        )
        assert "+x_0=4321000 +y_0=3210000" in offset.proj_string()
        values = {
            parameter["name"]: parameter["value"]
            for parameter in offset.projjson()["conversion"]["parameters"]
        }
        assert values["False easting"] == 4321000.0
        assert values["False northing"] == 3210000.0
        assert values["Latitude of natural origin"] == 52.0
        assert values["Longitude of natural origin"] == 10.0

    def test_projjson_is_valid_json_and_stable_across_calls(self):
        text = json.dumps(SITE_PROJECTION.projjson(), indent=2)
        assert json.loads(text) == SITE_PROJECTION.projjson()
        assert SITE_PROJECTION.projjson() == SITE_PROJECTION.projjson()

    def test_the_tracked_files_match_the_dataclass(self):
        """``check_definitions`` passes on the files in the repository. This is
        the anti-drift device: a parameter change that skips the regeneration
        fails here."""
        check_definitions()
        for path in definition_paths().values():
            assert path.is_file()
            assert path.parent == default_definition_dir()

    def test_check_definitions_rejects_an_edited_file(self, tmp_path):
        """A hand-edited definition is a test failure, not a way to change the
        projection."""
        paths = write_definitions(directory=tmp_path)
        check_definitions(directory=tmp_path)
        paths["projstring"].write_text(
            paths["projstring"].read_text().replace("+lat_0=50", "+lat_0=45")
        )
        with pytest.raises(ValueError, match="disagree"):
            check_definitions(directory=tmp_path)

    def test_check_definitions_detects_a_changed_parameter(self, tmp_path):
        """The same failure from the other side: the files are current and the
        dataclass has moved on."""
        write_definitions(directory=tmp_path)
        moved = Projection(name=SITE_PROJECTION.name, lat_0=45.0, lon_0=-100.0)
        with pytest.raises(ValueError, match="regenerate"):
            check_definitions(moved, directory=tmp_path)

    def test_check_definitions_names_the_regeneration_command(self, tmp_path):
        """A missing or stale file has to say what to run."""
        with pytest.raises(FileNotFoundError, match="--write"):
            check_definitions(directory=tmp_path)

    def test_write_definitions_is_idempotent(self, tmp_path):
        first = {
            key: path.read_text() for key, path in write_definitions(directory=tmp_path).items()
        }
        second = {
            key: path.read_text() for key, path in write_definitions(directory=tmp_path).items()
        }
        assert first == second

    def test_write_definitions_overwrites_a_stale_file(self, tmp_path):
        """Idempotence alone is satisfied by a function that writes nothing the
        second time, and this is the only escape hatch when ``check_definitions``
        fails: if it declined to overwrite, a parameter change could never be
        brought back into agreement."""
        paths = write_definitions(directory=tmp_path)
        paths["projstring"].write_text("+proj=laea +lat_0=45\n")
        write_definitions(directory=tmp_path)
        assert paths["projstring"].read_text() == SITE_PROJECTION.proj_string() + "\n"
        check_definitions(directory=tmp_path)

    def test_write_definitions_creates_a_missing_directory(self, tmp_path):
        paths = write_definitions(directory=tmp_path / "nested" / "dir")
        assert all(path.is_file() for path in paths.values())

    def test_write_definitions_leaves_no_partial_files(self, tmp_path):
        """The files are staged beside their destinations and moved into place
        together, so a failed run cannot leave one file describing this
        projection and the other describing the last one."""
        directory = tmp_path / "staged"
        write_definitions(directory=directory)
        assert list(directory.glob("*.partial")) == []

    def test_definition_paths_rejects_a_stem_that_is_a_path(self, tmp_path):
        for bad in ("../escaped", "a/b", "/absolute", ""):
            with pytest.raises(ValueError, match="bare file name"):
                definition_paths(tmp_path, stem=bad)

    def test_files_end_in_a_newline(self, tmp_path):
        """So that they are well-formed text files, and so that an editor adding
        one is not mistaken for a drifted definition."""
        for path in write_definitions(directory=tmp_path).values():
            assert path.read_text().endswith("\n")

    def test_module_main_writes_and_checks(self, tmp_path, capsys):
        """``python -m sipnet_calibration.projection --write`` and ``--check``."""
        assert projection_main(["--check", "--directory", str(tmp_path)]) == 1
        assert "error" in capsys.readouterr().out
        assert projection_main(["--write", "--directory", str(tmp_path)]) == 0
        assert projection_main(["--check", "--directory", str(tmp_path)]) == 0
        assert "matches" in capsys.readouterr().out

        # A file that exists but has drifted is the other error arm, and it has
        # to name the command that fixes it.
        definition_paths(tmp_path)["projstring"].write_text("+proj=laea\n")
        assert projection_main(["--check", "--directory", str(tmp_path)]) == 1
        assert "regenerate" in capsys.readouterr().out

    def test_module_main_defaults_to_checking_the_tracked_directory(self, capsys):
        """``--check`` is documented as the default, and the mutually exclusive
        group is not required, so the no-flag path is real behavior -- a default
        that silently did nothing would also return 0."""
        assert projection_main([]) == 0
        assert "matches" in capsys.readouterr().out


class TestExtents:
    def test_named_extents_are_well_formed_boxes(self):
        """Four values each, west of east and south of north, so every one is
        accepted by both ``select_sites`` and ``projected_bounds``."""
        for name, box in EXTENTS.items():
            assert len(box) == 4, name
            west, south, east, north = box
            assert west < east, name
            assert south < north, name
            assert -180.0 <= west and east <= 180.0, name
            assert -90.0 <= south and north <= 90.0, name
            SITE_PROJECTION.projected_bounds(box)

    def test_north_america_is_the_grid_extent(self):
        """So that it contains every site by construction rather than by a bound
        anyone chose."""
        assert EXTENTS["NORTH_AMERICA"] == (
            SITE_GRID.west,
            SITE_GRID.south,
            SITE_GRID.east,
            SITE_GRID.north,
        )

    def test_alaska_is_the_registered_area_of_use_clipped_at_the_grid_edge(self):
        """The EPSG extent for "United States (USA) - Alaska", registered for
        EPSG:3338, runs west across the antimeridian; the grid does not, and
        neither does ``select_sites``."""
        west, south, east, north = EXTENTS["ALASKA"]
        assert west == SITE_GRID.west
        assert (south, east, north) == (51.3, -129.99, 71.4)

    @pytest.mark.skipif(
        not default_sites_path().exists(), reason="processed/sites/sites.csv is not present"
    )
    def test_the_extents_select_the_documented_subsets(self):
        """CONUS is the 3640 sites ``data/README.md`` records, and
        NORTH_AMERICA is all of them."""
        sites = load_sites()
        assert len(select_sites(sites, bbox=EXTENTS["NORTH_AMERICA"])) == len(sites)
        assert len(select_sites(sites, bbox=EXTENTS["CONUS"])) == 3640
        assert 0 < len(select_sites(sites, bbox=EXTENTS["ALASKA"])) < len(sites)


# ── helpers ───────────────────────────────────────────────────────────────────


def _proj_reference_projection() -> Projection:
    """The CRS of PROJ's ``+proj=laea +ellps=GRS80`` test block: GRS 80, center (0, 0)."""
    return Projection(
        name="PROJ builtins.gie reference",
        lat_0=0.0,
        lon_0=0.0,
        ellipsoid=Ellipsoid("GRS 1980", 6378137.0, 298.257222101, 7019),
    )


def _angular_distance(projection: Projection, lon, lat):
    """Great-circle distance from *projection*'s center, in radians."""
    center = math.radians(projection.lat_0)
    latitude = np.radians(lat)
    delta_lon = np.radians(np.asarray(lon, dtype=float) - projection.lon_0)
    return np.arccos(
        np.clip(
            math.sin(center) * np.sin(latitude)
            + math.cos(center) * np.cos(latitude) * np.cos(delta_lon),
            -1.0,
            1.0,
        )
    )


def _tissot(projection: Projection, lon, lat, *, radius: float | None = None, step: float = 1e-5):
    """Areal scale and the two extreme linear scale factors, at each point.

    The Jacobian is taken by central differences with respect to *true distance
    on the ellipsoid*, using the meridional and prime-vertical radii of
    curvature, so its singular values are the scale factors directly and no
    further normalization is needed. Independent of the projection formulas, so
    it is a check on them rather than a restatement.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    semi_major = radius if radius is not None else projection.ellipsoid.semi_major
    e2 = projection.ellipsoid.eccentricity_squared

    latitude = np.radians(lat)
    w = 1 - e2 * np.sin(latitude) ** 2
    prime_vertical = semi_major / np.sqrt(w)
    meridional = semi_major * (1 - e2) / w**1.5
    east_per_degree = np.radians(1.0) * prime_vertical * np.cos(latitude)
    north_per_degree = np.radians(1.0) * meridional

    east_x1, east_y1 = projection.forward(lon + step, lat)
    east_x0, east_y0 = projection.forward(lon - step, lat)
    north_x1, north_y1 = projection.forward(lon, lat + step)
    north_x0, north_y0 = projection.forward(lon, lat - step)

    dx_de = (east_x1 - east_x0) / (2 * step * east_per_degree)
    dy_de = (east_y1 - east_y0) / (2 * step * east_per_degree)
    dx_dn = (north_x1 - north_x0) / (2 * step * north_per_degree)
    dy_dn = (north_y1 - north_y0) / (2 * step * north_per_degree)

    trace = dx_de**2 + dy_de**2 + dx_dn**2 + dy_dn**2
    area = np.abs(dx_de * dy_dn - dx_dn * dy_de)
    spread = np.sqrt(np.maximum(trace**2 - 4 * area**2, 0.0))
    scale_max = np.sqrt((trace + spread) / 2)
    scale_min = np.sqrt(np.maximum((trace - spread) / 2, 0.0))
    return area, scale_max, scale_min


def _site_coordinates():
    """The real site coordinates, longitude first."""
    sites = load_sites()
    return sites["lon"].to_numpy(), sites["lat"].to_numpy()


def _site_metrics():
    """Distortion of :data:`SITE_PROJECTION` at every site in the pool."""
    lon, lat = _site_coordinates()
    area, scale_max, scale_min = _tissot(SITE_PROJECTION, lon, lat)
    omega = np.degrees(2 * np.arcsin(np.clip((scale_max - scale_min) / (scale_max + scale_min), 0, 1)))
    return {
        "area": area,
        "anisotropy": scale_max / scale_min,
        "omega": omega,
        "distance": _angular_distance(SITE_PROJECTION, lon, lat),
    }
