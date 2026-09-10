"""Tests for the display projection.

PROJ does the projection arithmetic, so there is nothing here that checks a
formula: testing PROJ against PROJ would say nothing. What is tested is
everything the project actually decided —

- that :data:`SITE_PROJECTION` is the projection that was chosen, with the
  parameters that were chosen, on the base CRS the site coordinates are on;
- that the distortion over the **real 8000-site pool** stays inside the
  ceilings the choice was made on, which is the one measurement the decision
  rests on and the one thing a future parameter change could quietly break;
- that the stored interchange files still say what the parameters say, so the
  definition a colleague reads cannot drift from the transform this package
  applies — including across a PROJ upgrade, which changes the serialization
  without anyone here touching a parameter;
- the boundary behavior that is this module's own rather than PROJ's: the
  argument guards, the refusal to return ``inf``, and the boundary sampling in
  ``projected_bounds``.

The site-pool cases read ``data/processed/sites/sites.csv`` and skip when it is
absent, which on a fresh clone it is: it is untracked, and a git worktree does
not have it at all unless ``$SIPNET_CALIBRATION_DATA`` points at a checkout that
does. Those are the tests that matter most, so run them deliberately rather than
trusting a green suite.
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings

import numpy as np
import pyproj
import pytest
from types import MappingProxyType

from sipnet_calibration.projection import (
    DEFINITION_STEM,
    LAEA_METHOD,
    LAEA_METHOD_CODE,
    SITE_PROJECTION,
    Projection,
    check_definitions,
    default_definition_dir,
    definition_paths,
    write_definitions,
)
from sipnet_calibration.projection import _main as projection_main
from sipnet_calibration.sites import EXTENTS, SITE_GRID, default_sites_path, load_sites, select_sites

# Measured over data/processed/sites/sites.csv under SITE_PROJECTION, and the
# reason this projection was chosen over the alternatives; see issue #4. Held as
# ceilings rather than as the measured values, since a slightly better number is
# not a regression. For contrast, the ESRI:102003 Albers that the published
# reanalysis figures used reaches 107 degrees and 9.2:1 over the same sites.
SITE_OMEGA_CEILING_DEG = 14.0
SITE_ANISOTROPY_CEILING = 1.3
SITE_AREA_TOLERANCE = 1e-6

#: The projected bounds of each named extent, in meters, to the nearest
#: kilometer. Literal rather than recomputed, so that a change in the extent, in
#: the projection, or in the boundary sampling has to be noticed and re-blessed
#: rather than silently agreeing with itself.
EXPECTED_BOUNDS_KM = {
    "CONUS": (-2564.9, -2861.6, 3436.4, 546.3),
    "NORTH_AMERICA": (-7968.8, -4657.9, 8031.0, 4265.3),
    "ALASKA": (-4289.3, 561.0, -1046.0, 3823.3),
}

needs_site_table = pytest.mark.skipif(
    not default_sites_path().exists(),
    reason="processed/sites/sites.csv is not present; set $SIPNET_CALIBRATION_DATA",
)


class TestTheProjectionThatWasChosen:
    def test_site_projection_is_the_agreed_laea(self):
        """A Lambert Azimuthal Equal Area centered at 50 N, 100 W, on WGS 84,
        in meters, with no false origin."""
        assert SITE_PROJECTION.lat_0 == 50.0
        assert SITE_PROJECTION.lon_0 == -100.0
        assert SITE_PROJECTION.false_easting == 0.0
        assert SITE_PROJECTION.false_northing == 0.0
        assert SITE_PROJECTION.base_crs == "EPSG:4326"

    def test_the_built_crs_is_what_the_parameters_say(self):
        """The parameters reach PROJ intact. Literals rather than comparisons
        against the fields they came from, since the latter would pass however
        the conversion was built."""
        document = SITE_PROJECTION.projjson()
        assert document["type"] == "ProjectedCRS"
        assert document["name"] == SITE_PROJECTION.name

        conversion = document["conversion"]
        assert conversion["method"]["name"] == LAEA_METHOD == "Lambert Azimuthal Equal Area"
        assert conversion["method"]["id"] == {
            "authority": "EPSG",
            "code": LAEA_METHOD_CODE,
        } == {"authority": "EPSG", "code": 9820}
        assert [
            (parameter["name"], parameter["value"], parameter["unit"])
            for parameter in conversion["parameters"]
        ] == [
            ("Latitude of natural origin", 50, "degree"),
            ("Longitude of natural origin", -100, "degree"),
            ("False easting", 0, "metre"),
            ("False northing", 0, "metre"),
        ]

        assert document["base_crs"]["id"] == {"authority": "EPSG", "code": 4326}
        assert document["base_crs"]["name"] == "WGS 84"
        assert [axis["name"] for axis in document["coordinate_system"]["axis"]] == [
            "Easting",
            "Northing",
        ]
        assert {axis["unit"] for axis in document["coordinate_system"]["axis"]} == {"metre"}

    def test_the_crs_is_equal_area(self):
        """The property the whole choice rests on, asked of PROJ rather than
        inferred from the method name."""
        factors = SITE_PROJECTION.factors(
            np.array([-100.0, -60.0, -170.0, -25.0]), np.array([50.0, 7.5, 68.5, 82.5])
        )
        assert factors.areal_scale == pytest.approx(np.ones(4), abs=SITE_AREA_TOLERANCE)

    def test_the_crs_transformer_and_proj_are_built_once_and_reused(self):
        """Each parses a definition through PROJ, which costs far more than the
        query it serves, and every panel uses this projection. The transformer
        is the most expensive of the three, since it resolves an operation as
        well as building the CRS."""
        from sipnet_calibration.projection import _proj, _transformer

        assert SITE_PROJECTION.crs() is SITE_PROJECTION.crs()
        assert _transformer(SITE_PROJECTION) is _transformer(SITE_PROJECTION)
        assert _proj(SITE_PROJECTION) is _proj(SITE_PROJECTION)
        # And the cache is keyed on the projection, not shared across variants.
        variant = dataclasses.replace(SITE_PROJECTION, lat_0=45.0)
        assert _transformer(SITE_PROJECTION) is not _transformer(variant)

    def test_a_variant_is_a_one_liner_and_revalidates(self):
        """The projection is the project default, not a prohibition, so a caller
        wanting a different center must be able to say so — and must not be able
        to say something invalid."""
        alaska = dataclasses.replace(SITE_PROJECTION, name="Alaska LAEA", lat_0=64.0, lon_0=-150.0)
        assert alaska.forward(-150.0, 64.0) == pytest.approx((0.0, 0.0), abs=1e-6)
        with pytest.raises(ValueError, match=r"lat_0 must be in \[-90, 90\]"):
            dataclasses.replace(SITE_PROJECTION, lat_0=100.0)

    def test_rejects_an_out_of_range_origin(self):
        for bad_lat in (100.0, -100.0):
            with pytest.raises(ValueError, match=r"lat_0 must be in \[-90, 90\]"):
                Projection(name="bad", lat_0=bad_lat, lon_0=0.0)
        for bad_lon in (400.0, -400.0):
            with pytest.raises(ValueError, match=r"lon_0 must be in \[-360, 360\]"):
                Projection(name="bad", lat_0=50.0, lon_0=bad_lon)

    def test_rejects_a_non_finite_false_origin(self):
        """It is added to every projected coordinate and written into the
        definition files, so a NaN here would put ``+x_0=nan`` in a file meant
        to be authoritative."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="false_easting must be finite"):
                Projection(name="bad", lat_0=50.0, lon_0=-100.0, false_easting=bad)
            with pytest.raises(ValueError, match="false_northing must be finite"):
                Projection(name="bad", lat_0=50.0, lon_0=-100.0, false_northing=bad)

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            SITE_PROJECTION.lat_0 = 0.0  # type: ignore[misc]


class TestBaseCrs:
    def test_rejects_a_projected_base_crs(self):
        """``forward`` takes degrees, so a projected base CRS is a silent
        wrong answer -- and PROJ's own complaint is several kilobytes of JSON
        that never names the field."""
        with pytest.raises(ValueError, match="base_crs must be geographic"):
            Projection(name="bad", lat_0=50.0, lon_0=-100.0, base_crs="EPSG:3857")

    def test_rejects_a_base_crs_proj_does_not_recognize(self):
        with pytest.raises(ValueError, match="not a CRS PROJ recognizes"):
            Projection(name="bad", lat_0=50.0, lon_0=-100.0, base_crs="nonsense")

    def test_rejects_an_unhashable_base_crs(self):
        """A ``dict`` is a form ``CRS.from_user_input`` accepts, but it cannot
        key the cache, and the failure would otherwise be an unhashable-type
        ``TypeError`` at first use rather than a ``ValueError`` at
        construction."""
        with pytest.raises(ValueError, match="base_crs must be hashable"):
            Projection(
                name="bad",
                lat_0=50.0,
                lon_0=-100.0,
                base_crs={"proj": "longlat", "datum": "WGS84"},
            )

    def test_accepts_another_geographic_base_crs(self):
        """NAD83 is geographic too, so it is admissible -- the site coordinates
        are simply not on it."""
        nad83 = Projection(name="on NAD83", lat_0=50.0, lon_0=-100.0, base_crs="EPSG:4269")
        x, y = nad83.forward(-100.0, 50.0)
        assert math.isfinite(x) and math.isfinite(y)


class TestForward:
    def test_the_center_maps_to_the_false_origin(self):
        assert SITE_PROJECTION.forward(
            SITE_PROJECTION.lon_0, SITE_PROJECTION.lat_0
        ) == pytest.approx((0.0, 0.0), abs=1e-6)

    def test_false_origin_translates_and_nothing_else(self):
        shifted = dataclasses.replace(
            SITE_PROJECTION, false_easting=4321000.0, false_northing=3210000.0
        )
        lon = np.array([-160.0, -100.0, -30.0, -70.0])
        lat = np.array([65.0, 10.0, 80.0, 45.0])
        x, y = SITE_PROJECTION.forward(lon, lat)
        shifted_x, shifted_y = shifted.forward(lon, lat)
        assert shifted_x == pytest.approx(x + 4321000.0, abs=1e-6)
        assert shifted_y == pytest.approx(y + 3210000.0, abs=1e-6)

    def test_scalars_in_scalars_out(self):
        x, y = SITE_PROJECTION.forward(-90.0, 45.0)
        # `type` rather than `isinstance`: np.float64 is a subclass of float and
        # is not a scalar for json.dumps or for a formatted axis label.
        assert type(x) is float
        assert type(y) is float

    def test_broadcasts_and_preserves_shape(self):
        x, y = SITE_PROJECTION.forward(-100.0, np.array([40.0, 50.0, 60.0]))
        assert x.shape == y.shape == (3,)
        # A scalar longitude at the central meridian puts every point on x = 0.
        assert x == pytest.approx(np.zeros(3), abs=1e-6)

        grid_lon, grid_lat = np.meshgrid(np.linspace(-140, -60, 4), np.linspace(20, 70, 3))
        x, y = SITE_PROJECTION.forward(grid_lon, grid_lat)
        assert x.shape == y.shape == grid_lon.shape

    def test_returns_float64_for_integer_input(self):
        x, y = SITE_PROJECTION.forward(np.array([-100, -90]), np.array([50, 40]))
        assert x.dtype == y.dtype == np.float64

    def test_longitudes_are_not_wrapped(self):
        """-190 and 170 are the same meridian, since a vendored coastline can
        carry either."""
        assert SITE_PROJECTION.forward(-190.0, 10.0) == pytest.approx(
            SITE_PROJECTION.forward(170.0, 10.0), abs=1e-6
        )

    def test_a_bad_base_crs_is_not_reported_as_a_bad_input_point(self):
        """``CRSError`` is a subclass of ``ProjError``, so guarding the
        transformer's construction would blame the antipode for a definition
        that will not build. Construction is now validated up front."""
        with pytest.raises(ValueError, match="base_crs"):
            Projection(name="bad", lat_0=50.0, lon_0=-100.0, base_crs="EPSG:3857").forward(
                -90.0, 40.0
            )

    def test_refuses_a_point_outside_the_domain_rather_than_returning_inf(self):
        """PROJ's default is to return infinity, which propagates into an axes
        limit or into a triangulation as a silently dropped point. The message
        has to name the antipode, since that is the only such point here."""
        raw = pyproj.Transformer.from_crs(
            "EPSG:4326", SITE_PROJECTION.crs(), always_xy=True
        ).transform(*SITE_PROJECTION.antipode)
        assert not np.isfinite(raw).all(), "PROJ no longer returns inf; the guard's premise"

        # The coordinates matter: they say which point in an 8000-row array to
        # look for.
        with pytest.raises(ValueError, match=r"antipode of its center, \(80\.0, -50\.0\)"):
            SITE_PROJECTION.forward(*SITE_PROJECTION.antipode)
        with pytest.raises(ValueError, match="antipode of its center"):
            SITE_PROJECTION.forward(
                np.array([-100.0, SITE_PROJECTION.antipode[0]]),
                np.array([50.0, SITE_PROJECTION.antipode[1]]),
            )

    def test_accepts_a_point_far_from_the_center_but_short_of_the_antipode(self):
        """The guard must not narrow the projection's domain: everything but the
        antipode itself projects, including 179 degrees of arc away."""
        antipode_lon, antipode_lat = SITE_PROJECTION.antipode
        x, y = SITE_PROJECTION.forward(antipode_lon - 1.0, antipode_lat)
        assert math.isfinite(x) and math.isfinite(y)

    def test_accepts_the_poles(self):
        for pole in (90.0, -90.0):
            x, y = SITE_PROJECTION.forward(-100.0, pole)
            assert math.isfinite(x) and math.isfinite(y)

    def test_rejects_non_finite_coordinates(self):
        for lon, lat in [(np.nan, 50.0), (-100.0, np.nan), (np.inf, 50.0)]:
            with pytest.raises(ValueError, match="not finite"):
                SITE_PROJECTION.forward(lon, lat)
        with pytest.raises(ValueError, match="2 longitude"):
            SITE_PROJECTION.forward(np.array([-100.0, np.nan, np.nan]), 50.0)

    def test_counts_bad_values_before_broadcasting(self):
        """One bad scalar against 8000 latitudes is one bad coordinate. The
        count says whether one site is wrong or the whole table is, so it must
        not report the broadcast size."""
        with pytest.raises(ValueError, match=r"1 longitude\(s\) and 0 latitude"):
            SITE_PROJECTION.forward(np.nan, np.full(8000, 50.0))

    def test_rejects_a_masked_coordinate_rather_than_projecting_its_fill(self):
        masked_lon = np.ma.masked_array([-100.0, 1e30], mask=[False, True])
        masked_lat = np.ma.masked_array([50.0, 50.0], mask=[False, True])
        with pytest.raises(ValueError, match="not finite"):
            SITE_PROJECTION.forward(masked_lon, masked_lat)

    def test_rejects_latitudes_outside_the_poles(self):
        with pytest.raises(ValueError, match=r"outside \[-90, 90\]"):
            SITE_PROJECTION.forward(-100.0, 90.5)
        # The message says which order the arguments go in, because swapping
        # them is the way this is usually reached.
        with pytest.raises(ValueError, match="longitude first"):
            SITE_PROJECTION.forward(50.0, -100.0)
        # One corrupt row among good ones must fail too.
        with pytest.raises(ValueError, match="1 latitude"):
            SITE_PROJECTION.forward(
                np.array([-100.0, -95.0, -90.0]), np.array([45.0, 999.0, 50.0])
            )

    def test_rejects_a_longitude_fill_value(self):
        """PROJ accepts -9999 as a longitude and projects it to a real-looking
        point, so the bound is this module's to impose."""
        for bad in (-9999.0, 1e20):
            with pytest.raises(ValueError, match=r"outside \[-360, 360\]"):
                SITE_PROJECTION.forward(bad, 50.0)


class TestFactors:
    def test_reports_the_rotation_of_projected_north(self):
        """North is not up, and the sign is PROJ's rather than the one the eye
        reads: at the northwest of the domain ``meridian_convergence`` is
        *negative*. It is zero on the central meridian by construction."""
        factors = SITE_PROJECTION.factors(
            np.array([-170.0, -100.0, -20.0]), np.array([60.0, 50.0, 20.0])
        )
        assert factors.meridian_convergence == pytest.approx([-57.91, 0.0, 42.09], abs=0.01)

    @needs_site_table
    def test_the_convergence_spread_over_the_pool_is_what_the_docs_claim(self):
        """The figure that justifies refusing a north arrow, pinned so the prose
        cannot drift from it again: it was wrong in sign, in magnitude and in
        location before this test existed."""
        sites = load_sites()
        convergence = SITE_PROJECTION.factors(
            sites["lon"].to_numpy(), sites["lat"].to_numpy()
        ).meridian_convergence
        assert convergence.min() == pytest.approx(-70.6, abs=0.2)
        assert convergence.max() == pytest.approx(75.1, abs=0.2)
        assert convergence.max() - convergence.min() == pytest.approx(146.0, abs=1.0)

    def test_factors_refuses_the_region_where_proj_returns_infinity(self):
        """Wider than the point ``forward`` refuses: about a degree around the
        antipode. Unchecked, one such point turns the documented
        ``300e3 * tissot_semimajor.max()`` into ``inf``, and a long-edge mask
        into one that masks nothing."""
        antipode_lon, antipode_lat = SITE_PROJECTION.antipode
        with pytest.raises(ValueError, match="undefined at the antipode"):
            SITE_PROJECTION.factors(antipode_lon, antipode_lat + 0.001)
        with pytest.raises(ValueError, match="undefined at the antipode"):
            SITE_PROJECTION.factors(
                np.array([-100.0, antipode_lon]), np.array([50.0, antipode_lat + 0.001])
            )

    def test_factors_rejects_empty_input_in_this_module_s_own_terms(self):
        """PROJ raises "longitude and latitude must be same size" on empty
        input, which is false and is a different exception type from everything
        else here."""
        with pytest.raises(ValueError, match="at least one point"):
            SITE_PROJECTION.factors(np.array([]), np.array([]))

    def test_factors_reuses_one_proj_object(self):
        """Constructing it parses the definition through PROJ, which costs more
        than the query it serves."""
        from sipnet_calibration.projection import _proj

        assert _proj(SITE_PROJECTION) is _proj(SITE_PROJECTION)

    def test_reports_the_scale_factors_a_ground_distance_needs(self):
        """The long-edge triangle mask is a projected length and wants a ground
        distance; this is the conversion, and it cannot be had from the
        projection parameters alone."""
        factors = SITE_PROJECTION.factors(-170.0, 60.0)
        assert factors.tissot_semimajor == pytest.approx(1.0626, abs=1e-3)
        assert factors.tissot_semiminor == pytest.approx(0.9411, abs=1e-3)
        # Equal-area means the product is 1, which is what makes one scale
        # factor the reciprocal of the other.
        assert factors.tissot_semimajor * factors.tissot_semiminor == pytest.approx(1.0, abs=1e-6)

    def test_is_unity_and_unrotated_at_the_center(self):
        factors = SITE_PROJECTION.factors(SITE_PROJECTION.lon_0, SITE_PROJECTION.lat_0)
        assert factors.tissot_semimajor == pytest.approx(1.0, abs=1e-9)
        assert factors.tissot_semiminor == pytest.approx(1.0, abs=1e-9)
        assert factors.meridian_convergence == pytest.approx(0.0, abs=1e-9)
        assert factors.angular_distortion == pytest.approx(0.0, abs=1e-6)

    def test_reports_the_variant_projection_it_is_called_on(self):
        """Every other case here goes through ``SITE_PROJECTION``, so ``factors``
        could ignore ``self`` entirely and still pass. A variant reports unity
        and no rotation at *its* center, where ``SITE_PROJECTION`` reports 1.035
        and -43 degrees."""
        variant = dataclasses.replace(
            SITE_PROJECTION, name="Alaska LAEA", lat_0=64.0, lon_0=-150.0
        )
        own = variant.factors(-150.0, 64.0)
        # PROJ computes factors by finite differences, so the center is unity to
        # about 1e-8 rather than exactly.
        assert own.tissot_semimajor == pytest.approx(1.0, abs=1e-6)
        assert own.tissot_semiminor == pytest.approx(1.0, abs=1e-6)
        assert own.meridian_convergence == pytest.approx(0.0, abs=1e-6)

        default = SITE_PROJECTION.factors(-150.0, 64.0)
        assert default.tissot_semimajor == pytest.approx(1.0347, abs=1e-3)
        assert default.meridian_convergence == pytest.approx(-43.26, abs=0.01)

    def test_checks_its_arguments_the_way_forward_does(self):
        with pytest.raises(ValueError, match="not finite"):
            SITE_PROJECTION.factors(np.nan, 50.0)
        with pytest.raises(ValueError, match="longitude first"):
            SITE_PROJECTION.factors(50.0, -100.0)


@needs_site_table
class TestSiteDomainDistortion:
    """The measurement the choice of projection rests on, over the real pool."""

    @staticmethod
    def _factors():
        sites = load_sites()
        return SITE_PROJECTION.factors(sites["lon"].to_numpy(), sites["lat"].to_numpy())

    def test_every_site_projects_to_a_finite_coordinate(self):
        sites = load_sites()
        x, y = SITE_PROJECTION.forward(sites["lon"].to_numpy(), sites["lat"].to_numpy())
        assert np.isfinite(x).all()
        assert np.isfinite(y).all()

    def test_angular_deformation_stays_under_the_ceiling(self):
        """No site exceeds 14 degrees, against 107 for the ESRI:102003 that the
        published reanalysis figures used."""
        assert self._factors().angular_distortion.max() < SITE_OMEGA_CEILING_DEG

    def test_anisotropy_stays_under_the_ceiling(self):
        """No site exceeds 1.3:1, against 9.2:1 for ESRI:102003. Load-bearing
        beyond appearance: the triangulation is computed after projecting, and
        the long-edge mask is a projected length."""
        factors = self._factors()
        anisotropy = factors.tissot_semimajor / factors.tissot_semiminor
        assert anisotropy.max() < SITE_ANISOTROPY_CEILING

    def test_area_is_preserved_over_the_whole_pool(self):
        assert np.abs(self._factors().areal_scale - 1.0).max() < SITE_AREA_TOLERANCE

    def test_the_worst_distortion_is_at_the_site_farthest_from_the_center(self):
        """Distortion grows with angular distance from the center, so the
        extremes of the pool are where the ceilings are tested and a change of
        center moves both together."""
        sites = load_sites()
        lon = sites["lon"].to_numpy()
        lat = sites["lat"].to_numpy()
        center = math.radians(SITE_PROJECTION.lat_0)
        distance = np.arccos(
            np.clip(
                math.sin(center) * np.sin(np.radians(lat))
                + math.cos(center) * np.cos(np.radians(lat))
                * np.cos(np.radians(lon - SITE_PROJECTION.lon_0)),
                -1.0,
                1.0,
            )
        )
        assert int(np.argmax(self._factors().angular_distortion)) == int(np.argmax(distance))
        assert np.degrees(distance.max()) == pytest.approx(54.8, abs=0.1)


class TestProjectedBounds:
    @pytest.mark.parametrize("name", sorted(EXPECTED_BOUNDS_KM))
    def test_bounds_of_each_named_extent(self, name):
        """Pinned to literal kilometers, so the answer cannot drift with the
        extent, the projection or the sampling without being re-blessed. A
        containment assertion alone would pass for an arbitrarily wide box, and
        an over-wide box is a figure whose data fills half the axes."""
        bounds_km = tuple(v / 1000.0 for v in SITE_PROJECTION.projected_bounds(EXTENTS[name]))
        assert bounds_km == pytest.approx(EXPECTED_BOUNDS_KM[name], abs=0.1)

    def test_boundary_sampling_beats_the_four_corners(self):
        """A longitude/latitude box projects to a curved quadrilateral, so the
        true bound lies outside the corner-only one. The far edge from the
        projection center is where the corners miss it, which for a center at
        50 N and the CONUS box is its southern edge. Two-sided, because the
        one-sided form gets easier to satisfy the looser the bound becomes."""
        west, south, east, north = EXTENTS["CONUS"]
        corner_x, corner_y = SITE_PROJECTION.forward(
            np.array([west, east, west, east]), np.array([south, south, north, north])
        )
        _, y_min, _, _ = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])
        assert 350e3 < corner_y.min() - y_min < 450e3
        # The other three edges are attained at corners, so they must agree.
        assert corner_x.min() == pytest.approx(
            SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])[0], abs=1.0
        )

    @needs_site_table
    def test_contains_every_site_inside_the_box(self):
        """Whatever ``select_sites(bbox=...)`` returns must project inside the
        limits the same box gives, or a figure clips its own data."""
        sites = select_sites(load_sites(), bbox=EXTENTS["CONUS"])
        x, y = SITE_PROJECTION.forward(sites["lon"].to_numpy(), sites["lat"].to_numpy())
        x_min, y_min, x_max, y_max = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])
        assert x.min() >= x_min and x.max() <= x_max
        assert y.min() >= y_min and y.max() <= y_max

    def test_rejects_a_malformed_box(self):
        with pytest.raises(ValueError, match="3 value"):
            SITE_PROJECTION.projected_bounds((-125.0, 24.0, -66.0))
        with pytest.raises(ValueError, match="5 value"):
            SITE_PROJECTION.projected_bounds((-125.0, 24.0, -66.0, 50.0, 0.0))
        with pytest.raises(ValueError, match="antimeridian"):
            SITE_PROJECTION.projected_bounds((-66.0, 24.0, -125.0, 50.0))
        with pytest.raises(ValueError, match="north of north"):
            SITE_PROJECTION.projected_bounds((-125.0, 50.0, -66.0, 24.0))
        with pytest.raises(ValueError, match="must be numbers"):
            SITE_PROJECTION.projected_bounds({"west": -125.0, "s": 1, "e": 2, "n": 3})
        with pytest.raises(ValueError, match="must be finite"):
            SITE_PROJECTION.projected_bounds((-125.0, 24.0, -66.0, np.nan))
        with pytest.raises(ValueError, match=r"\(west, south, east, north\)"):
            SITE_PROJECTION.projected_bounds(None)
        with pytest.raises(ValueError, match="samples_per_edge must be at least 2"):
            SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"], samples_per_edge=1)
        with pytest.raises(ValueError, match="samples_per_edge must be an integer"):
            SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"], samples_per_edge=8.0)

    def test_coerces_numeric_strings_the_way_it_documents(self):
        assert SITE_PROJECTION.projected_bounds(("-125", "24", "-66", "50")) == pytest.approx(
            SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])
        )

    def test_samples_per_edge_actually_controls_the_sampling(self):
        """Nothing else observes the parameter taking effect: comparing the
        default against a finer run passes trivially for an implementation that
        discards it. Two samples per edge is the corners-only bound the method
        exists to avoid, so it is the one call that distinguishes the two."""
        west, south, east, north = EXTENTS["CONUS"]
        corner_x, corner_y = SITE_PROJECTION.forward(
            np.array([west, east, west, east]), np.array([south, south, north, north])
        )
        coarse = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"], samples_per_edge=2)
        assert coarse == pytest.approx(
            (corner_x.min(), corner_y.min(), corner_x.max(), corner_y.max()), abs=1.0
        )
        default_y_min = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])[1]
        assert coarse[1] - default_y_min > 350e3

    def test_more_samples_do_not_change_the_answer_materially(self):
        """The tolerance is meters against figures thousands of kilometers
        across, several orders of magnitude below one rendered pixel."""
        for name, box in EXTENTS.items():
            default = np.array(SITE_PROJECTION.projected_bounds(box))
            fine = np.array(SITE_PROJECTION.projected_bounds(box, samples_per_edge=8192))
            assert np.abs(default - fine).max() < 100.0, name

    @pytest.mark.parametrize(
        "box",
        [
            (70.0, -60.0, 90.0, -40.0),  # the antipode, snugly
            (-180.0, -90.0, 180.0, 90.0),  # the whole globe
            (-200.0, -60.0, 100.0, -40.0),  # 300 degrees wide, unwrapped
            (0.0, -60.0, 300.0, -40.0),  # 300 degrees wide, positive
            (60.0, -55.0, 80.0, -45.0),  # antipode exactly on the east edge
            (80.0, -55.0, 100.0, -45.0),  # exactly on the west edge
            (70.0, -50.0, 90.0, -45.0),  # exactly on the south edge
            (70.0, -55.0, 90.0, -50.0),  # exactly on the north edge
        ],
    )
    def test_refuses_a_box_containing_the_antipode(self, box):
        """The boundary-sampling argument holds only where the transform is
        defined throughout the box. A box wider than 180 degrees is the case a
        symmetric reduction of the longitude offset cannot express: an offset of
        280 degrees comes back as -80, the box looks as though it ends before
        the antipode, and the bounds returned are quietly not bounds. Measured
        on the whole globe, an interior point sat 12,699 km outside a bound
        whose x_max came back as 0."""
        with pytest.raises(ValueError, match="unbounded, not merely large"):
            SITE_PROJECTION.projected_bounds(box)

    @pytest.mark.parametrize(
        "box",
        [
            (-200.0, -30.0, 100.0, -10.0),  # spans the antipode's meridian, not its latitude
            (-125.0, -60.0, -66.0, -40.0),  # spans its latitude, not its meridian
        ],
    )
    def test_accepts_a_box_that_misses_the_antipode_in_one_coordinate(self, box):
        """Both halves of the guard's conjunction have to be there: with either
        one dropped, a box that misses the antipode in the other coordinate
        would be refused for no reason."""
        x_min, y_min, x_max, y_max = SITE_PROJECTION.projected_bounds(box)
        assert x_min < x_max and y_min < y_max


class TestAntipode:
    def test_antipode_is_the_point_forward_refuses(self):
        assert SITE_PROJECTION.antipode == (80.0, -50.0)
        with pytest.raises(ValueError, match="antipode"):
            SITE_PROJECTION.forward(*SITE_PROJECTION.antipode)

    @pytest.mark.parametrize("lon_0", [-100.0, 0.0, 100.0, 260.0, -350.0, 179.9])
    def test_antipode_longitude_is_wrapped_whatever_the_origin_was_given_as(self, lon_0):
        projection = Projection(name="wherever", lat_0=10.0, lon_0=lon_0)
        antipode_lon, antipode_lat = projection.antipode
        assert -180.0 <= antipode_lon < 180.0
        assert antipode_lat == -10.0
        with pytest.raises(ValueError, match="antipode"):
            projection.forward(antipode_lon, antipode_lat)


class TestInterchangeFiles:
    def test_proj_string_carries_every_parameter(self):
        assert SITE_PROJECTION.proj_string() == (
            "+proj=laea +lat_0=50 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 "
            "+units=m +no_defs +type=crs"
        )

    def test_proj_string_does_not_warn(self):
        """PROJ warns that a PROJ string loses information, which is true and is
        why the PROJJSON is stored beside it -- but it describes a deliberate
        choice, not something a caller can act on."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            assert SITE_PROJECTION.proj_string().startswith("+proj=laea")

    def test_the_two_serializations_describe_the_same_crs(self):
        """They carry different amounts — the PROJ string has no CRS name and
        gives the base CRS by datum rather than by EPSG code — so the test is
        that they project identically, not that they are textually equal."""
        from_json = pyproj.CRS.from_user_input(SITE_PROJECTION.projjson())
        from_string = pyproj.CRS.from_user_input(SITE_PROJECTION.proj_string())
        lon = np.linspace(-179.0, -20.0, 50)
        lat = np.linspace(7.0, 84.0, 50)
        first = pyproj.Transformer.from_crs("EPSG:4326", from_json, always_xy=True).transform(lon, lat)
        second = pyproj.Transformer.from_crs(
            "EPSG:4326", from_string, always_xy=True
        ).transform(lon, lat)
        assert np.hypot(first[0] - second[0], first[1] - second[1]).max() == 0.0

    def test_projjson_is_valid_json_and_stable_across_calls(self):
        text = json.dumps(SITE_PROJECTION.projjson(), indent=2)
        assert json.loads(text) == SITE_PROJECTION.projjson()
        assert SITE_PROJECTION.projjson() == SITE_PROJECTION.projjson()

    def test_serializations_distinguish_the_two_false_origin_parameters(self):
        """``SITE_PROJECTION`` has both at zero, so it cannot tell them apart."""
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

    def test_the_tracked_files_match_the_parameters(self):
        """The anti-drift device: a parameter change that skips the
        regeneration fails here, and so does a PROJ upgrade that changes the
        serialization."""
        check_definitions()
        for path in definition_paths().values():
            assert path.is_file()
            assert path.parent == default_definition_dir()

    def test_the_tracked_projjson_is_what_proj_emits_now(self):
        """Compared as parsed JSON rather than as text, so the failure says
        which field moved rather than that two blobs differ."""
        stored = json.loads(definition_paths()["projjson"].read_text())
        assert stored == SITE_PROJECTION.projjson()

    def test_check_definitions_rejects_an_edited_file(self, tmp_path):
        paths = write_definitions(tmp_path)
        check_definitions(tmp_path)
        paths["projstring"].write_text(
            paths["projstring"].read_text().replace("+lat_0=50", "+lat_0=45")
        )
        with pytest.raises(ValueError, match="disagree"):
            check_definitions(tmp_path)

    def test_check_definitions_detects_a_changed_parameter(self, tmp_path):
        write_definitions(tmp_path)
        moved = dataclasses.replace(SITE_PROJECTION, lat_0=45.0)
        with pytest.raises(ValueError, match="regenerate"):
            check_definitions(tmp_path, projection=moved)

    def test_check_definitions_names_the_regeneration_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--write"):
            check_definitions(tmp_path)

    def test_write_definitions_is_idempotent(self, tmp_path):
        first = {key: path.read_text() for key, path in write_definitions(tmp_path).items()}
        second = {key: path.read_text() for key, path in write_definitions(tmp_path).items()}
        assert first == second

    def test_write_definitions_overwrites_a_stale_file(self, tmp_path):
        """Idempotence alone is satisfied by a function that writes nothing the
        second time, and this is the only escape hatch when
        ``check_definitions`` fails."""
        paths = write_definitions(tmp_path)
        paths["projstring"].write_text("+proj=laea +lat_0=45\n")
        write_definitions(tmp_path)
        assert paths["projstring"].read_text() == SITE_PROJECTION.proj_string() + "\n"
        check_definitions(tmp_path)

    def test_write_definitions_creates_a_missing_directory(self, tmp_path):
        paths = write_definitions(tmp_path / "nested" / "dir")
        assert all(path.is_file() for path in paths.values())

    def test_a_failed_write_leaves_the_previous_pair_intact(self, tmp_path, monkeypatch):
        """What the staging is for. With the second move failing, neither file
        may be half-updated and no partial may be left behind — otherwise a
        colleague pastes a definition that describes neither projection."""
        write_definitions(tmp_path)
        moved = dataclasses.replace(SITE_PROJECTION, lat_0=45.0)

        import sipnet_calibration.projection as module

        real_replace = module.os.replace
        calls = {"n": 0}

        def failing_replace(src, dst):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk full")
            return real_replace(src, dst)

        monkeypatch.setattr(module.os, "replace", failing_replace)
        with pytest.raises(OSError, match="disk full"):
            write_definitions(tmp_path, projection=moved)

        assert list(tmp_path.glob("*.partial")) == []
        # The first file did move, so the pair is inconsistent -- which is
        # exactly what check_definitions is for, and it must say so.
        with pytest.raises(ValueError, match="regenerate"):
            check_definitions(tmp_path)

    def test_files_end_in_a_newline(self, tmp_path):
        for path in write_definitions(tmp_path).values():
            assert path.read_text().endswith("\n")

    def test_definition_paths_rejects_a_stem_that_is_a_path(self, tmp_path):
        for bad in ("../escaped", "a/b", "/absolute", ""):
            with pytest.raises(ValueError, match="bare file name"):
                definition_paths(tmp_path, stem=bad)

    def test_definition_stem_names_the_tracked_files(self):
        assert DEFINITION_STEM == "north_america_laea"
        assert definition_paths()["projjson"].name == "north_america_laea.projjson"

    def test_module_main_writes_and_checks(self, tmp_path, capsys):
        assert projection_main(["--check", "--directory", str(tmp_path)]) == 1
        assert "error" in capsys.readouterr().out
        assert projection_main(["--write", "--directory", str(tmp_path)]) == 0
        assert projection_main(["--check", "--directory", str(tmp_path)]) == 0
        assert "matches" in capsys.readouterr().out

        # A file that exists but has drifted is the other error arm.
        definition_paths(tmp_path)["projstring"].write_text("+proj=laea\n")
        assert projection_main(["--check", "--directory", str(tmp_path)]) == 1
        assert "regenerate" in capsys.readouterr().out

    def test_module_main_defaults_to_checking(self, tmp_path, capsys):
        """``--check`` is the documented default and the group is not required,
        so the no-flag path is real behavior. It must *fail* on a bad directory,
        not merely succeed on the good one -- a default that silently did
        nothing would also return 0."""
        assert projection_main(["--directory", str(tmp_path)]) == 1
        assert "--write" in capsys.readouterr().out
        assert projection_main([]) == 0
        assert "matches" in capsys.readouterr().out

    def test_module_main_reports_a_write_failure(self, tmp_path, capsys):
        blocked = tmp_path / "a-file"
        blocked.write_text("not a directory\n")
        assert projection_main(["--write", "--directory", str(blocked / "sub")]) == 1
        assert "Not a directory" in capsys.readouterr().out


class TestExtents:
    def test_named_extents_are_well_formed_boxes(self):
        for name, box in EXTENTS.items():
            assert len(box) == 4, name
            west, south, east, north = box
            assert west < east, name
            assert south < north, name
            assert -180.0 <= west and east <= 180.0, name
            assert -90.0 <= south and north <= 90.0, name

    def test_extents_cannot_be_mutated(self):
        """A figure and the site subset it plots are supposed to agree on what a
        region means, so a caller must not be able to reassign an entry."""
        assert isinstance(EXTENTS, MappingProxyType)
        with pytest.raises(TypeError):
            EXTENTS["CONUS"] = (0.0, 0.0, 1.0, 1.0)  # type: ignore[index]
        with pytest.raises(TypeError):
            del EXTENTS["ALASKA"]  # type: ignore[attr-defined]

    def test_north_america_is_the_grid_extent(self):
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

    @needs_site_table
    def test_the_extents_select_the_documented_subsets(self):
        sites = load_sites()
        assert len(select_sites(sites, bbox=EXTENTS["NORTH_AMERICA"])) == len(sites)
        assert len(select_sites(sites, bbox=EXTENTS["CONUS"])) == 3640
        assert 0 < len(select_sites(sites, bbox=EXTENTS["ALASKA"])) < len(sites)
