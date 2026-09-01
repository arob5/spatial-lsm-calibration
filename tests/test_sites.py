"""Tests for the site grid.

The real-data cases use coordinates taken from ``data/raw/sites/pts.shp``, which
is tracked, so they are reproducible without reading the shapefile here. Reading
it is the job of ``scripts/ingest_sites.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sipnet_calibration.sites import SITE_GRID, Grid

# (site_id, lon, lat, lon_idx, lat_idx) from data/raw/sites/pts.shp
REAL_SITES = [
    (1, -24.5625010172526, 82.54583435058593, 18532, 9065),
    (731, -178.75416768391926, 68.5125010172526, 29, 7381),
    (6104, -84.32916768391927, 35.92916768391927, 11360, 3471),
    (8000, -71.42083435058593, 7.012501017252603, 12909, 1),
]

# The stored coordinates depart from exact cell centres by up to this much,
# consistent with 32-bit storage upstream. See SITE_GRID's documentation.
STORED_COORD_TOLERANCE_DEG = 1.02e-6


class TestGridGeometry:
    def test_site_grid_matches_the_documented_extent(self):
        assert SITE_GRID.west == -179.0
        assert SITE_GRID.south == 7.0
        assert SITE_GRID.east == pytest.approx(-20.0)
        assert SITE_GRID.north == pytest.approx(85.0)
        assert SITE_GRID.shape == (9360, 19080)

    def test_step_is_thirty_arcseconds(self):
        assert SITE_GRID.step_arcsec == pytest.approx(30.0)
        assert SITE_GRID.step == pytest.approx(1 / 120)

    def test_extent_is_consistent_with_dimensions(self):
        # (east - west) and (north - south) must be whole numbers of cells
        assert (SITE_GRID.east - SITE_GRID.west) * SITE_GRID.cells_per_degree == pytest.approx(
            SITE_GRID.n_lon
        )
        assert (SITE_GRID.north - SITE_GRID.south) * SITE_GRID.cells_per_degree == pytest.approx(
            SITE_GRID.n_lat
        )

    def test_rejects_degenerate_construction(self):
        with pytest.raises(ValueError, match="positive extent"):
            Grid(west=0.0, south=0.0, n_lon=0, n_lat=10, cells_per_degree=120)
        with pytest.raises(ValueError, match="cells_per_degree"):
            Grid(west=0.0, south=0.0, n_lon=10, n_lat=10, cells_per_degree=0)

    def test_is_immutable(self):
        with pytest.raises(Exception):
            SITE_GRID.west = 0.0  # type: ignore[misc]


class TestIndexToLonLat:
    def test_first_cell_centre_is_half_a_step_in(self):
        lon, lat = SITE_GRID.index_to_lonlat(0, 0)
        assert lon == pytest.approx(-179.0 + 0.5 / 120)
        assert lat == pytest.approx(7.0 + 0.5 / 120)

    def test_last_cell_centre_is_half_a_step_short_of_the_far_edge(self):
        lon, lat = SITE_GRID.index_to_lonlat(SITE_GRID.n_lon - 1, SITE_GRID.n_lat - 1)
        assert lon == pytest.approx(-20.0 - 0.5 / 120)
        assert lat == pytest.approx(85.0 - 0.5 / 120)

    def test_scalars_in_scalars_out(self):
        lon, lat = SITE_GRID.index_to_lonlat(5, 7)
        assert isinstance(lon, float) and isinstance(lat, float)

    def test_arrays_in_arrays_out(self):
        lon, lat = SITE_GRID.index_to_lonlat([0, 1, 2], [0, 1, 2])
        assert lon.shape == (3,) and lat.shape == (3,)

    @pytest.mark.parametrize(
        "lon_idx, lat_idx",
        [(-1, 0), (0, -1), (19080, 0), (0, 9360)],
    )
    def test_rejects_out_of_range(self, lon_idx, lat_idx):
        with pytest.raises(ValueError, match="outside"):
            SITE_GRID.index_to_lonlat(lon_idx, lat_idx)

    def test_rejects_fractional_indices(self):
        with pytest.raises(ValueError, match="must be integers"):
            SITE_GRID.index_to_lonlat(0.5, 0)


class TestLonLatToIndex:
    @pytest.mark.parametrize("site_id, lon, lat, lon_idx, lat_idx", REAL_SITES)
    def test_real_site_coordinates_resolve_to_their_indices(
        self, site_id, lon, lat, lon_idx, lat_idx
    ):
        assert SITE_GRID.lonlat_to_index(lon, lat) == (lon_idx, lat_idx)

    @pytest.mark.parametrize("site_id, lon, lat, lon_idx, lat_idx", REAL_SITES)
    def test_reconstruction_is_within_the_stored_coordinate_tolerance(
        self, site_id, lon, lat, lon_idx, lat_idx
    ):
        back_lon, back_lat = SITE_GRID.index_to_lonlat(lon_idx, lat_idx)
        assert abs(back_lon - lon) <= STORED_COORD_TOLERANCE_DEG
        assert abs(back_lat - lat) <= STORED_COORD_TOLERANCE_DEG

    def test_round_trip_over_the_whole_grid(self):
        rng = np.random.default_rng(0)
        j = rng.integers(0, SITE_GRID.n_lon, size=5000)
        k = rng.integers(0, SITE_GRID.n_lat, size=5000)
        lon, lat = SITE_GRID.index_to_lonlat(j, k)
        j2, k2 = SITE_GRID.lonlat_to_index(lon, lat)
        assert np.array_equal(j, j2)
        assert np.array_equal(k, k2)

    def test_scalars_in_scalars_out(self):
        j, k = SITE_GRID.lonlat_to_index(-179.0 + 0.5 / 120, 7.0 + 0.5 / 120)
        assert isinstance(j, int) and isinstance(k, int)

    def test_rejects_a_point_that_is_not_on_the_grid(self):
        # a cell edge rather than a centre: half a step away from any centre
        with pytest.raises(ValueError, match="not on the grid"):
            SITE_GRID.lonlat_to_index(-179.0, 7.0)

    def test_tolerance_is_honoured(self):
        lon, lat = SITE_GRID.index_to_lonlat(100, 100)
        nudged = lon + 5e-4
        with pytest.raises(ValueError, match="not on the grid"):
            SITE_GRID.lonlat_to_index(nudged, lat)
        assert SITE_GRID.lonlat_to_index(nudged, lat, tol=1e-3) == (100, 100)

    def test_rejects_coordinates_outside_the_grid(self):
        lon, lat = SITE_GRID.index_to_lonlat(0, 0)
        with pytest.raises(ValueError, match="longitude outside"):
            SITE_GRID.lonlat_to_index(lon - 1.0, lat)
        with pytest.raises(ValueError, match="latitude outside"):
            SITE_GRID.lonlat_to_index(lon, lat - 1.0)

    def test_the_stored_offset_does_not_shift_any_index(self):
        # every site is within STORED_COORD_TOLERANCE_DEG of a centre, which is
        # three orders of magnitude below half a cell, so rounding is unambiguous
        half_cell = 0.5 / SITE_GRID.cells_per_degree
        assert STORED_COORD_TOLERANCE_DEG < half_cell / 100
