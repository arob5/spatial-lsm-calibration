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

# The stored coordinates depart from exact cell centers by up to this much,
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
    def test_first_cell_center_is_half_a_step_in(self):
        lon, lat = SITE_GRID.index_to_lonlat(0, 0)
        assert lon == pytest.approx(-179.0 + 0.5 / 120)
        assert lat == pytest.approx(7.0 + 0.5 / 120)

    def test_last_cell_center_is_half_a_step_short_of_the_far_edge(self):
        lon, lat = SITE_GRID.index_to_lonlat(SITE_GRID.n_lon - 1, SITE_GRID.n_lat - 1)
        assert lon == pytest.approx(-20.0 - 0.5 / 120)
        assert lat == pytest.approx(85.0 - 0.5 / 120)

    def test_scalars_in_scalars_out(self):
        lon, lat = SITE_GRID.index_to_lonlat(5, 7)
        # `type(...) is float`, not isinstance: np.float64 is a float subclass,
        # so isinstance would pass without the conversion this asserts.
        assert type(lon) is float and type(lat) is float

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
        # a cell edge rather than a center: half a step away from any center
        with pytest.raises(ValueError, match="not on the grid"):
            SITE_GRID.lonlat_to_index(-179.0, 7.0)

    def test_tolerance_is_honored(self):
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
        # every site is within STORED_COORD_TOLERANCE_DEG of a center, which is
        # three orders of magnitude below half a cell, so rounding is unambiguous
        half_cell = 0.5 / SITE_GRID.cells_per_degree
        assert STORED_COORD_TOLERANCE_DEG < half_cell / 100


# ── the site table and the ingest script ─────────────────────────────────────
#
# These read the tracked shapefile directly. It is the only input under
# data/raw/ that is in version control, which is what makes the ingest script
# testable end to end here rather than only on the SCC.

import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import shapefile

from sipnet_calibration.sites import (
    SITE_COLUMN_DTYPES,
    SITE_COLUMNS,
    load_sites,
    select_sites,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_SITES = REPO_ROOT / "data" / "raw" / "sites"
SHAPEFILE = RAW_SITES / "pts.shp"
SITE_ID_MAP = REPO_ROOT / "data" / "site_id_map.csv"

N_SITES = 8000

# The two records carrying non-ASCII bytes. Under latin-1 both decode to
# plausible-looking strings rather than raising, which is why the encoding is
# asserted rather than left to a default.
UTF8_SITES = {
    7176: "Rayón (MX-Ray)",
    7813: "Estación Experimental Forestal Horizontes",
}

# Sites named literally "NA", which a default read_csv turns into nulls.
NA_NAMED_SITES = [3392, 7484, 7542, 7589, 7595, 7607, 7616, 7617]


def _load_ingest_module():
    """Import ``scripts/ingest_sites.py``, which is a script, not a package."""
    path = REPO_ROOT / "scripts" / "ingest_sites.py"
    spec = importlib.util.spec_from_file_location("ingest_sites", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ingest_sites"] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest_module()


def _shapefile_coordinates():
    """``(lon, lat)`` float64 arrays straight from the shapefile geometry."""
    with shapefile.Reader(str(SHAPEFILE)) as reader:
        shapes = reader.shapes()
    lon = np.array([shape.points[0][0] for shape in shapes], dtype=np.float64)
    lat = np.array([shape.points[0][1] for shape in shapes], dtype=np.float64)
    return lon, lat


def _digest_raw_inputs() -> dict[str, str]:
    """SHA-256 of every file the script reads, to prove it wrote none of them."""
    paths = sorted(RAW_SITES.iterdir()) + [SITE_ID_MAP]
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    """Run the ingest script into a temporary directory and load the result.

    Scoped to the module because the run reads 8000 records and the tests all
    interrogate the same output. Writes nowhere near ``data/processed/``.
    """
    out = tmp_path_factory.mktemp("processed") / "sites" / "sites.csv"
    before = _digest_raw_inputs()
    status = ingest.main(
        ["--shapefile", str(SHAPEFILE), "--site-id-map", str(SITE_ID_MAP), "--out", str(out)]
    )
    assert status == 0
    return {
        "path": out,
        "table": load_sites(out),
        "raw_digests_before": before,
    }


class TestIngestScript:
    def test_help_works(self, capsys):
        with pytest.raises(SystemExit) as caught:
            ingest.parse_args(["--help"])
        assert caught.value.code == 0
        assert "sites.csv" in capsys.readouterr().out

    def test_writes_eight_thousand_rows(self, ingested):
        assert len(ingested["table"]) == N_SITES

    def test_creates_its_output_directory(self, ingested):
        # The fixture's --out named a directory that did not exist; the run had
        # to create it, as it must on a fresh clone where data/processed/ is
        # absent.
        assert ingested["path"].is_file()
        assert ingested["path"].parent.name == "sites"

    def test_leaves_its_inputs_untouched(self, ingested):
        assert _digest_raw_inputs() == ingested["raw_digests_before"]

    def test_columns_are_the_agreed_set_in_order(self, ingested):
        assert tuple(ingested["table"].columns) == SITE_COLUMNS

    def test_there_is_no_pft_column(self, ingested):
        # A PFT labeling is an experimental choice and must not be baked into
        # the shared key; it is its own product keyed on site_id.
        assert "pft" not in ingested["table"].columns


class TestSiteIdentifiers:
    def test_site_id_is_one_to_eight_thousand_in_record_order(self, ingested):
        # Every other product joins on this and the identifiers are not ours to
        # renumber, so a permutation would be as much a failure as a gap.
        assert np.array_equal(
            ingested["table"]["site_id"].to_numpy(), np.arange(1, N_SITES + 1)
        )

    def test_the_check_rejects_a_permuted_site_id(self):
        permuted = np.arange(1, N_SITES + 1)
        permuted[[0, 1]] = permuted[[1, 0]]
        with pytest.raises(ingest.IngestError, match="record order"):
            ingest.check_site_ids_are_the_full_range(permuted)

    def test_the_check_rejects_a_gap(self):
        with pytest.raises(ingest.IngestError, match="7999 value"):
            ingest.check_site_ids_are_the_full_range(np.arange(1, N_SITES))

    def test_ameriflux_identifiers_cover_the_mapped_sites_only(self, ingested):
        mapped = ingested["table"]["ameriflux_site_id"] != ""
        assert int(mapped.sum()) == 185
        expected = pd.read_csv(SITE_ID_MAP)
        joined = ingested["table"].set_index("site_id")["ameriflux_site_id"]
        for row in expected.itertuples():
            assert joined.loc[row.index] == row.Site_ID

    def test_duplicate_ameriflux_identifiers_are_rejected(self):
        with pytest.raises(ingest.IngestError, match="duplicate Ameriflux"):
            ingest.check_ameriflux_map_is_usable(
                {1: "US-Ha1", 2: "US-Ha1"}, site_ids=np.array([1, 2])
            )

    def test_an_ameriflux_row_naming_an_unknown_site_is_rejected(self):
        with pytest.raises(ingest.IngestError, match="not in the shapefile"):
            ingest.check_ameriflux_map_is_usable(
                {99999: "US-Ha1"}, site_ids=np.array([1, 2])
            )


class TestCoordinateRoundTrip:
    """The point of the exercise: CSV is where this table can lose precision."""

    def test_coordinates_are_bitwise_equal_to_the_shapefile(self, ingested):
        lon, lat = _shapefile_coordinates()
        table = ingested["table"]
        # Exact equality on float64, not approx: any difference at all is a
        # loss, and 1632 longitudes moved under float_format="%.17g".
        assert np.array_equal(table["lon"].to_numpy(), lon)
        assert np.array_equal(table["lat"].to_numpy(), lat)

    def test_every_coordinate_is_reproduced_exactly_by_repr(self, ingested):
        # The written form is repr, so this is the property the file relies on.
        for value in ingested["table"]["lon"].to_numpy().tolist():
            assert float(repr(value)) == value

    @pytest.mark.parametrize("float_format", [None, "%.17g"])
    def test_the_reader_setting_is_what_makes_the_round_trip_exact(
        self, tmp_path, float_format
    ):
        # The default parser is inexact for BOTH write formats -- 1496 of 8000
        # longitudes from repr, 1632 from %.17g -- and float_precision fixes
        # both. So the guarantee lives in load_sites' reader setting, not in
        # FLOAT_FORMAT. If a future pandas makes the default parser exact, this
        # fails, which is the right way to find out.
        lon, _ = _shapefile_coordinates()
        path = tmp_path / "coords.csv"
        pd.DataFrame({"lon": lon}).to_csv(
            path, index=False, float_format=float_format
        )
        loose = pd.read_csv(path)["lon"].to_numpy()
        exact = pd.read_csv(path, float_precision="round_trip")["lon"].to_numpy()
        assert int((loose != lon).sum()) > 1000
        assert np.array_equal(exact, lon)

    def test_load_sites_reads_exactly_whatever_format_was_written(self, tmp_path):
        # The claim above, through the real loader rather than pandas directly.
        lon, _ = _shapefile_coordinates()
        for float_format in (None, "%.17g"):
            path = tmp_path / f"t{float_format}.csv"
            frame = pd.DataFrame({
                "site_id": np.arange(1, len(lon) + 1, dtype=np.int32),
                "lon": lon,
                "lat": lon,
                "lon_idx": np.zeros(len(lon), dtype=np.int32),
                "lat_idx": np.zeros(len(lon), dtype=np.int32),
                "site_name": ["x"] * len(lon),
                "site_order": np.zeros(len(lon), dtype=np.int32),
                "cluster": np.ones(len(lon), dtype=np.int8),
                "landcover": np.ones(len(lon), dtype=np.int8),
                "ameriflux_site_id": [""] * len(lon),
            })[list(SITE_COLUMNS)]
            frame.to_csv(path, index=False, float_format=float_format)
            assert np.array_equal(load_sites(path)["lon"].to_numpy(), lon)

    def test_grid_indices_resolve_the_stored_coordinates(self, ingested):
        table = ingested["table"]
        lon_idx, lat_idx = SITE_GRID.lonlat_to_index(
            table["lon"].to_numpy(), table["lat"].to_numpy()
        )
        assert np.array_equal(lon_idx, table["lon_idx"].to_numpy())
        assert np.array_equal(lat_idx, table["lat_idx"].to_numpy())

    def test_grid_index_pairs_are_distinct(self, ingested):
        pairs = ingested["table"][["lon_idx", "lat_idx"]].to_numpy()
        assert np.unique(pairs, axis=0).shape[0] == N_SITES

    def test_the_distinctness_check_rejects_a_shared_cell(self):
        with pytest.raises(ingest.IngestError, match="share a grid cell"):
            ingest.check_index_pairs_are_distinct(
                np.array([5, 5]), np.array([7, 7])
            )

    def test_indices_reconstruct_the_coordinates_to_the_stored_tolerance(self, ingested):
        table = ingested["table"]
        lon, lat = SITE_GRID.index_to_lonlat(
            table["lon_idx"].to_numpy(), table["lat_idx"].to_numpy()
        )
        assert np.abs(lon - table["lon"].to_numpy()).max() <= STORED_COORD_TOLERANCE_DEG
        assert np.abs(lat - table["lat"].to_numpy()).max() <= STORED_COORD_TOLERANCE_DEG


class TestTextRoundTrip:
    def test_the_two_utf8_site_names_survive(self, ingested):
        names = ingested["table"].set_index("site_id")["site_name"]
        for site_id, expected in UTF8_SITES.items():
            assert names.loc[site_id] == expected

    def test_those_names_are_not_ascii(self, ingested):
        # Guards the test above: if the expected strings were ever replaced with
        # their latin-1 misreadings the assertions would still pass, and this
        # would not.
        names = ingested["table"].set_index("site_id")["site_name"]
        for site_id in UTF8_SITES:
            assert not names.loc[site_id].isascii()
        assert sum(not name.isascii() for name in ingested["table"]["site_name"]) == 2

    @pytest.mark.filterwarnings("ignore:Specified encoding:UserWarning")
    def test_latin_one_corrupts_them_quietly(self):
        # The failure this asserts about is silent: neither read raises.
        with shapefile.Reader(str(SHAPEFILE), encoding="utf-8") as reader:
            correct = reader.record(7175)["site_names"]
        with shapefile.Reader(str(SHAPEFILE), encoding="latin-1") as reader:
            wrong = reader.record(7175)["site_names"]
        assert correct == UTF8_SITES[7176]
        # The exact mojibake, not merely "differs": the module docstring's
        # argument is that this particular string looks like a real site label.
        assert wrong == "RayÃ³n (MX-Ray)"

    @pytest.mark.filterwarnings("ignore:Specified encoding:UserWarning")
    def test_reading_as_latin_one_is_rejected(self):
        contents = ingest.read_shapefile(SHAPEFILE, encoding="latin-1")
        with pytest.raises(ingest.IngestError, match="was read as 'latin-1'"):
            ingest.check_encoding_is_utf8(contents, encoding_used="latin-1")

    def test_a_cpg_declaring_something_else_is_rejected(self):
        # The other branch of the same check, which "declares" also matched, so
        # nothing exercised it.
        contents = ingest.ShapefileContents(
            declared_encoding="latin-1",
            field_names=(),
            records=(),
            shape_types=(),
            points=(),
        )
        with pytest.raises(ingest.IngestError, match=r"\.cpg declares 'latin-1'"):
            ingest.check_encoding_is_utf8(contents, encoding_used="utf-8")

    def test_a_missing_cpg_is_rejected(self):
        contents = ingest.ShapefileContents(
            declared_encoding=None,
            field_names=(),
            records=(),
            shape_types=(),
            points=(),
        )
        with pytest.raises(ingest.IngestError, match="undeclared"):
            ingest.check_encoding_is_utf8(contents, encoding_used="utf-8")

    def test_the_cpg_declares_utf8(self):
        assert ingest.read_declared_encoding(SHAPEFILE) == "utf-8"

    def test_sites_named_na_are_names_and_not_nulls(self, ingested):
        names = ingested["table"].set_index("site_id")["site_name"]
        for site_id in NA_NAMED_SITES:
            assert names.loc[site_id] == "NA"
        assert ingested["table"]["site_name"].notna().all()

    def test_a_default_read_csv_would_null_them(self, ingested):
        # Why load_sites passes keep_default_na=False. Not a property of our
        # code, so it is asserted against pandas rather than against us.
        naive = pd.read_csv(ingested["path"])
        assert naive["site_name"].isna().sum() == len(NA_NAMED_SITES)


class TestDbfNumerics:
    """``cluster``, ``landcover`` and ``site_order`` arrive as floats."""

    def test_they_are_stored_as_declared_decimals_in_the_dbf(self):
        with shapefile.Reader(str(SHAPEFILE)) as reader:
            declared = {field[0]: field for field in reader.fields[1:]}
            record = reader.record(0)
        for name in ("cluster", "landcover", "site_order"):
            assert declared[name][3] == 15, "the .dbf still declares 15 decimals"
            assert isinstance(record[name], float), "so pyshp still returns floats"

    def test_they_are_written_as_integers(self, ingested):
        table = ingested["table"]
        for name in ("cluster", "landcover", "site_order"):
            assert np.issubdtype(table[name].dtype, np.integer)

    def test_their_ranges_are_the_documented_ones(self, ingested):
        table = ingested["table"]
        assert sorted(table["cluster"].unique()) == list(range(1, 7))
        assert sorted(table["landcover"].unique()) == list(range(1, 9))

    def test_site_order_is_zero_or_a_rank(self, ingested):
        site_order = ingested["table"]["site_order"].to_numpy()
        assert int((site_order == 0).sum()) == 6907
        assert np.array_equal(np.sort(site_order[site_order != 0]), np.arange(1, 1094))

    def test_a_non_integral_value_is_rejected(self):
        with pytest.raises(ingest.IngestError, match="non-integral"):
            ingest.check_values_are_integral(np.array([1.0, 2.5]), name="cluster")

    def test_a_repeated_rank_is_rejected(self):
        with pytest.raises(ingest.IngestError, match="permutation"):
            ingest.check_site_order_is_a_permutation(np.array([1, 1, 0]))


class TestShapeChecks:
    def test_every_shape_is_a_single_point(self, ingested):
        contents = ingest.read_shapefile(SHAPEFILE, encoding="utf-8")
        ingest.check_shapes_are_single_points(contents)
        ingest.check_record_count(contents)
        assert len(contents.points) == N_SITES
        assert all(len(points) == 1 for points in contents.points)

    def test_a_multipoint_shape_is_rejected(self):
        contents = ingest.ShapefileContents(
            declared_encoding="utf-8",
            field_names=(),
            records=(),
            shape_types=(int(shapefile.POINT),),
            points=(((0.0, 0.0), (1.0, 1.0)),),
        )
        with pytest.raises(ingest.IngestError, match="exactly one point"):
            ingest.check_shapes_are_single_points(contents)

    def test_a_non_point_shape_is_rejected(self):
        contents = ingest.ShapefileContents(
            declared_encoding="utf-8",
            field_names=(),
            records=(),
            shape_types=(int(shapefile.POLYGON),),
            points=(((0.0, 0.0),),),
        )
        with pytest.raises(ingest.IngestError, match="not POINT"):
            ingest.check_shapes_are_single_points(contents)

    def test_a_short_record_count_is_rejected(self):
        contents = ingest.ShapefileContents(
            declared_encoding="utf-8",
            field_names=("site_id",),
            records=({"site_id": 1},),
            shape_types=(int(shapefile.POINT),),
            points=(((0.0, 0.0),),),
        )
        with pytest.raises(ingest.IngestError, match=f"expected {N_SITES} records"):
            ingest.check_record_count(contents)

    def test_a_missing_dbf_field_is_rejected(self):
        contents = ingest.ShapefileContents(
            declared_encoding="utf-8",
            field_names=("site_id",),
            records=(),
            shape_types=(),
            points=(),
        )
        with pytest.raises(ingest.IngestError, match="missing field"):
            ingest.check_fields_are_present(
                contents, required=("site_id", "site_names")
            )


class TestLoadSites:
    def test_dtypes_are_the_declared_ones(self, ingested):
        table = ingested["table"]
        for column, dtype in SITE_COLUMN_DTYPES.items():
            if dtype is str:
                # pandas returns object or StringDtype depending on its version;
                # what matters is that every value is a str.
                assert all(isinstance(value, str) for value in table[column])
            else:
                assert table[column].dtype == np.dtype(dtype)

    def test_a_missing_file_names_the_script_that_builds_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ingest_sites.py"):
            load_sites(tmp_path / "absent.csv")

    def test_a_table_with_the_wrong_columns_is_rejected(self, tmp_path):
        path = tmp_path / "wrong.csv"
        pd.DataFrame({"site_id": [1], "lon": [0.0]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_sites(path)

    def test_a_duplicate_site_id_is_rejected(self, ingested, tmp_path):
        path = tmp_path / "duplicated.csv"
        table = ingested["table"].head(3).copy()
        table.loc[2, "site_id"] = 1
        table.to_csv(path, index=False)
        with pytest.raises(ValueError, match="duplicate site_id"):
            load_sites(path)

    def test_an_unsorted_table_is_rejected(self, ingested, tmp_path):
        path = tmp_path / "unsorted.csv"
        ingested["table"].head(3).iloc[::-1].to_csv(path, index=False)
        with pytest.raises(ValueError, match="ascending site_id"):
            load_sites(path)

    def test_the_default_path_honors_the_environment_variable(self, monkeypatch, tmp_path):
        from sipnet_calibration.sites import DATA_ROOT_ENV_VAR, default_sites_path

        monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path))
        assert default_sites_path() == tmp_path / "processed" / "sites" / "sites.csv"

    def test_the_default_path_falls_back_to_the_checkout(self, monkeypatch):
        from sipnet_calibration.sites import DATA_ROOT_ENV_VAR, default_sites_path

        monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
        assert default_sites_path() == (
            REPO_ROOT / "data" / "processed" / "sites" / "sites.csv"
        )


class TestSelectSites:
    def test_no_filters_returns_everything(self, ingested):
        assert len(select_sites(ingested["table"])) == N_SITES

    def test_ids_are_returned_in_the_order_given(self, ingested):
        chosen = select_sites(ingested["table"], ids=[8000, 1, 4000])
        assert chosen["site_id"].tolist() == [8000, 1, 4000]

    def test_an_unknown_id_raises(self, ingested):
        with pytest.raises(KeyError, match="not in the table"):
            select_sites(ingested["table"], ids=[1, 99999])

    def test_a_repeated_id_raises(self, ingested):
        with pytest.raises(ValueError, match="duplicate site ids"):
            select_sites(ingested["table"], ids=[1, 1])

    def test_bbox_selects_the_conterminous_us(self, ingested):
        # 3640 of the 8000 sites, per data/README.md.
        conus = select_sites(ingested["table"], bbox=(-125, 24, -66, 50))
        assert len(conus) == 3640
        assert conus["lon"].between(-125, -66).all()
        assert conus["lat"].between(24, 50).all()

    def test_bbox_includes_its_edges(self, ingested):
        table = ingested["table"]
        row = table.iloc[0]
        exact = select_sites(table, bbox=(row.lon, row.lat, row.lon, row.lat))
        assert row.site_id in exact["site_id"].tolist()

    def test_a_positive_longitude_bbox_selects_nothing(self, ingested):
        # The whole pool is in the western hemisphere; this is the mistake the
        # docstring warns about, and it should return empty rather than raise.
        assert len(select_sites(ingested["table"], bbox=(66, 24, 125, 50))) == 0

    def test_an_inverted_bbox_raises(self, ingested):
        with pytest.raises(ValueError, match="east of"):
            select_sites(ingested["table"], bbox=(-66, 24, -125, 50))
        with pytest.raises(ValueError, match="north of"):
            select_sites(ingested["table"], bbox=(-125, 50, -66, 24))

    def test_where_filters_on_any_column(self, ingested):
        mapped = select_sites(
            ingested["table"], where=lambda t: t["ameriflux_site_id"] != ""
        )
        assert len(mapped) == 185

    def test_where_filters_on_a_joined_column(self, ingested):
        # The replacement for pft=: a labeling is joined on by the caller.
        table = ingested["table"].head(10).copy()
        labeling = pd.DataFrame(
            {"site_id": table["site_id"], "pft": ["DBF"] * 4 + ["ENF"] * 6}
        )
        labeled = table.merge(labeling, on="site_id")
        assert len(select_sites(labeled, where=lambda t: t["pft"] == "DBF")) == 4

    def test_where_must_return_a_boolean_mask(self, ingested):
        with pytest.raises(ValueError, match="boolean mask"):
            select_sites(ingested["table"], where=lambda t: t["site_id"])

    def test_where_must_return_one_value_per_row(self, ingested):
        with pytest.raises(ValueError, match="shape"):
            select_sites(ingested["table"], where=lambda t: np.array([True, False]))

    def test_sample_is_reproducible_under_a_seed(self, ingested):
        first = select_sites(ingested["table"], sample=20, seed=0)
        second = select_sites(ingested["table"], sample=20, seed=0)
        assert first["site_id"].tolist() == second["site_id"].tolist()
        assert len(first) == 20

    def test_a_different_seed_gives_a_different_sample(self, ingested):
        first = select_sites(ingested["table"], sample=50, seed=0)
        second = select_sites(ingested["table"], sample=50, seed=1)
        assert first["site_id"].tolist() != second["site_id"].tolist()

    def test_sample_is_in_ascending_site_id_order(self, ingested):
        drawn = select_sites(ingested["table"], sample=100, seed=3)
        assert drawn["site_id"].is_monotonic_increasing

    def test_sample_draws_without_replacement(self, ingested):
        drawn = select_sites(ingested["table"], sample=200, seed=4)
        assert drawn["site_id"].nunique() == 200

    def test_an_oversized_sample_raises_rather_than_truncating(self, ingested):
        with pytest.raises(ValueError, match="from 8000 site"):
            select_sites(ingested["table"], sample=N_SITES + 1)

    def test_filters_compose_with_sample_applied_last(self, ingested):
        chosen = select_sites(
            ingested["table"],
            bbox=(-125, 24, -66, 50),
            where=lambda t: t["ameriflux_site_id"] != "",
            sample=20,
            seed=0,
        )
        assert len(chosen) == 20
        assert (chosen["ameriflux_site_id"] != "").all()
        assert chosen["lon"].between(-125, -66).all()

    def test_the_input_table_is_not_modified(self, ingested):
        # A private copy, not the module-scoped fixture: ~20 earlier tests share
        # that frame, so damage done by any of them would already be in `before`
        # and a schema change would go unnoticed.
        table = ingested["table"].copy()
        before = table.copy()
        select_sites(table, bbox=(-125, 24, -66, 50), sample=10, seed=0)
        pd.testing.assert_frame_equal(table, before)
        assert list(table.columns) == list(before.columns)

    def test_the_index_is_reset(self, ingested):
        chosen = select_sites(ingested["table"], bbox=(-125, 24, -66, 50))
        assert chosen.index.tolist() == list(range(len(chosen)))


class TestRoundTripCheckItself:
    """The script's central assertion, which had no test of its own."""

    def _table(self, ingested):
        return ingested["table"].head(20).copy()

    def test_it_passes_on_a_faithful_write(self, ingested, tmp_path):
        table = self._table(ingested)
        path = tmp_path / "ok.csv"
        ingest.write_site_table(table, path)
        ingest.check_csv_round_trip(table, path)  # must not raise

    def test_it_catches_a_lossy_coordinate(self, ingested, tmp_path):
        table = self._table(ingested)
        path = tmp_path / "lossy.csv"
        table.to_csv(path, index=False, float_format="%.6f")
        with pytest.raises(ingest.IngestError, match="lon did not survive"):
            ingest.check_csv_round_trip(table, path)

    def test_it_catches_a_mangled_site_name(self, ingested, tmp_path):
        table = self._table(ingested)
        path = tmp_path / "name.csv"
        corrupted = table.copy()
        corrupted.loc[0, "site_name"] = "not the real name"
        ingest.write_site_table(corrupted, path)
        with pytest.raises(ingest.IngestError, match="site_name did not survive"):
            ingest.check_csv_round_trip(table, path)

    def test_it_catches_a_dropped_row(self, ingested, tmp_path):
        table = self._table(ingested)
        path = tmp_path / "short.csv"
        ingest.write_site_table(table.head(19), path)
        with pytest.raises(ingest.IngestError, match="row count"):
            ingest.check_csv_round_trip(table, path)


class TestMainExitCodes:
    def test_missing_shapefile_exits_two(self, tmp_path, capsys):
        status = ingest.main(
            ["--shapefile", str(tmp_path / "absent.shp"),
             "--site-id-map", str(SITE_ID_MAP),
             "--out", str(tmp_path / "out.csv")]
        )
        assert status == 2
        assert "is not a file" in capsys.readouterr().err

    def test_missing_site_id_map_exits_two(self, tmp_path, capsys):
        status = ingest.main(
            ["--shapefile", str(SHAPEFILE),
             "--site-id-map", str(tmp_path / "absent.csv"),
             "--out", str(tmp_path / "out.csv")]
        )
        assert status == 2
        assert "is not a file" in capsys.readouterr().err

    @pytest.mark.filterwarnings("ignore:Specified encoding:UserWarning")
    def test_a_failed_check_exits_one_with_a_message_not_a_traceback(
        self, tmp_path, capsys
    ):
        status = ingest.main(
            ["--shapefile", str(SHAPEFILE),
             "--site-id-map", str(SITE_ID_MAP),
             "--encoding", "latin-1",
             "--out", str(tmp_path / "out.csv")]
        )
        assert status == 1
        err = capsys.readouterr().err
        # A one-line diagnosis, not a traceback.
        assert "Traceback" not in err
        assert "error: the shapefile was read as 'latin-1'" in err


class TestSchemaIsPinnedToALiteral:
    """Guards against SITE_COLUMNS and the code drifting together.

    Every other schema test compares the table against ``SITE_COLUMNS``, which
    both the writer and the reader order by -- so editing the constant moves
    them in step and nothing notices. These name the contract outright.
    """

    def test_column_names_and_order(self):
        assert SITE_COLUMNS == (
            "site_id",
            "lon",
            "lat",
            "lon_idx",
            "lat_idx",
            "site_name",
            "site_order",
            "cluster",
            "landcover",
            "ameriflux_site_id",
        )

    def test_column_dtypes(self):
        assert SITE_COLUMN_DTYPES == {
            "site_id": np.int32,
            "lon": np.float64,
            "lat": np.float64,
            "lon_idx": np.int32,
            "lat_idx": np.int32,
            "site_name": str,
            "site_order": np.int32,
            "cluster": np.int8,
            "landcover": np.int8,
            "ameriflux_site_id": str,
        }


class TestLoaderNormalizesTheFile:
    def test_columns_come_back_in_canonical_order_however_the_file_is_ordered(
        self, ingested, tmp_path
    ):
        # load_sites reorders to SITE_COLUMNS. The ingest script already writes
        # in that order, so nothing else exercises the reorder.
        shuffled = list(reversed(SITE_COLUMNS))
        path = tmp_path / "shuffled.csv"
        ingested["table"].head(5)[shuffled].to_csv(path, index=False)
        assert tuple(load_sites(path).columns) == SITE_COLUMNS

    def test_the_write_format_is_the_shortest_round_tripping_one(self):
        # Intent, not behavior: with load_sites reading at round_trip precision
        # the written format cannot change any value, so no behavioral test can
        # pin this. repr is chosen so the file is also correct for readers that
        # are not ours. See the FLOAT_FORMAT comment in the ingest script.
        assert ingest.FLOAT_FORMAT is None


class TestIntegerCastsCannotWrap:
    """Range, not just integrality. A narrowing cast wraps in silence."""

    def test_ingest_rejects_a_cluster_too_large_for_int8(self):
        with pytest.raises(ingest.IngestError, match="outside the range of int8"):
            ingest.check_values_fit_dtype(
                np.array([1.0, 200.0]), name="cluster", dtype=np.int8
            )

    @pytest.mark.parametrize("value", [128.0, -129.0])
    def test_ingest_rejects_either_int8_boundary(self, value):
        with pytest.raises(ingest.IngestError, match="wrap silently"):
            ingest.check_values_fit_dtype(
                np.array([value]), name="cluster", dtype=np.int8
            )

    def test_ingest_accepts_the_real_ranges(self, ingested):
        for column in ("site_order", "cluster", "landcover", "lon_idx", "lat_idx"):
            values = ingested["table"][column].to_numpy().astype(np.float64)
            dtype = SITE_COLUMN_DTYPES[column]
            ingest.check_values_fit_dtype(values, name=column, dtype=dtype)

    def test_ingest_catches_a_value_that_would_saturate_int64(self):
        # Integral and finite, so check_values_are_integral passes it.
        with pytest.raises(ingest.IngestError, match="outside the range"):
            ingest._as_integer([1e300], name="cluster", dtype=np.int8)

    def _one_row(self, tmp_path, **overrides):
        fields = {
            "site_id": 1, "lon": -100.0, "lat": 40.0, "lon_idx": 9480,
            "lat_idx": 3960, "site_name": "x", "site_order": 0, "cluster": 1,
            "landcover": 1, "ameriflux_site_id": "",
        }
        fields.update(overrides)
        path = tmp_path / "one.csv"
        path.write_text(
            ",".join(SITE_COLUMNS) + "\n"
            + ",".join(str(fields[c]) for c in SITE_COLUMNS) + "\n"
        )
        return path

    def test_loader_rejects_a_site_id_that_would_wrap_to_a_valid_one(self, tmp_path):
        # 4294967297 wraps to 1 in int32 and would then pass every other check.
        path = self._one_row(tmp_path, site_id=4294967297)
        with pytest.raises(ValueError, match="outside the range of int32"):
            load_sites(path)

    def test_loader_rejects_an_out_of_range_landcover(self, tmp_path):
        path = self._one_row(tmp_path, landcover=200)
        with pytest.raises(ValueError, match="outside the range of int8"):
            load_sites(path)

    def test_loader_still_returns_the_narrow_dtypes(self, ingested):
        for column, dtype in SITE_COLUMN_DTYPES.items():
            if dtype is not str:
                assert ingested["table"][column].dtype == np.dtype(dtype)


class TestPredicateMaskAlignment:
    def test_a_reordered_series_mask_selects_the_same_rows(self, ingested):
        table = ingested["table"]
        straight = select_sites(table, where=lambda t: t["lat"] > 40)
        reordered = select_sites(
            table, where=lambda t: (t["lat"] > 40).sort_index(ascending=False)
        )
        assert straight["site_id"].tolist() == reordered["site_id"].tolist()

    def test_it_aligns_after_an_earlier_filter_left_a_gappy_index(self, ingested):
        table = ingested["table"]
        both = select_sites(
            table, bbox=(-125, 24, -66, 50), where=lambda t: t["lat"] > 40
        )
        manual = table[
            (table.lon >= -125) & (table.lon <= -66)
            & (table.lat >= 24) & (table.lat <= 50) & (table.lat > 40)
        ]
        assert both["site_id"].tolist() == manual["site_id"].tolist()

    def test_a_series_with_a_foreign_index_is_rejected(self, ingested):
        table = ingested["table"].head(10)
        with pytest.raises(ValueError, match="cannot be aligned"):
            select_sites(
                table,
                where=lambda t: pd.Series(
                    [True] * 10, index=t["site_id"].to_numpy() + 10_000
                ),
            )

    def test_a_nullable_boolean_mask_without_missing_values_works(self, ingested):
        table = ingested["table"].head(4)
        chosen = select_sites(
            table,
            where=lambda t: pd.Series(
                pd.array([True, False, True, False], dtype="boolean"), index=t.index
            ),
        )
        assert len(chosen) == 2

    def test_a_nullable_boolean_mask_with_missing_values_says_what_to_do(
        self, ingested
    ):
        # The module's own documented pattern, on a labeling with a gap.
        table = ingested["table"].head(4).copy()
        table["pft"] = pd.array(["DBF", None, "ENF", "DBF"], dtype="string")
        with pytest.raises(ValueError, match="fillna"):
            select_sites(table, where=lambda t: t["pft"] == "DBF")

    def test_the_documented_join_pattern_works_once_na_is_resolved(self, ingested):
        table = ingested["table"].head(4).copy()
        table["pft"] = pd.array(["DBF", None, "ENF", "DBF"], dtype="string")
        chosen = select_sites(table, where=lambda t: (t["pft"] == "DBF").fillna(False))
        assert len(chosen) == 2


class TestSelectByIdShape:
    def test_a_duplicated_table_is_rejected_rather_than_multiplying_rows(
        self, ingested
    ):
        doubled = pd.concat([ingested["table"].head(3)] * 2, ignore_index=True)
        with pytest.raises(ValueError, match="repeated site id"):
            select_sites(doubled, ids=[1, 2])

    def test_it_preserves_dtype_and_column_order(self, ingested):
        table = ingested["table"]
        by_id = select_sites(table, ids=[1, 2])
        by_bbox = select_sites(table, bbox=(-125, 24, -66, 50))
        assert by_id.dtypes["site_id"] == by_bbox.dtypes["site_id"] == np.int32
        assert list(by_id.columns) == list(by_bbox.columns) == list(SITE_COLUMNS)

    def test_it_preserves_column_order_on_a_joined_table(self, ingested):
        labeling = pd.DataFrame({"pft": ["A", "B", "C"], "site_id": [1, 2, 3]})
        joined = labeling.merge(ingested["table"], on="site_id")
        assert list(select_sites(joined, ids=[1, 2]).columns) == list(joined.columns)

    def test_float_ids_are_rejected_rather_than_truncated(self, ingested):
        with pytest.raises(ValueError, match="whole numbers"):
            select_sites(ingested["table"], ids=[5.9, 1.2])


class TestIngestPublishesAtomically:
    def test_a_failed_check_leaves_the_previous_table_in_place(
        self, ingested, tmp_path, monkeypatch
    ):
        # Make the write lossy, so the round-trip check fails on a real file.
        def lossy(table, path):
            table.to_csv(path, index=False, float_format="%.4f")

        monkeypatch.setattr(ingest, "write_site_table", lossy)
        out = tmp_path / "sites.csv"
        out.write_text("previous contents\n")
        with pytest.raises(ingest.IngestError, match="did not survive"):
            ingest.write_checked_site_table(ingested["table"].head(5), out)
        # The canonical path still holds what it held before.
        assert out.read_text() == "previous contents\n"
        assert not list(tmp_path.glob("*.partial"))

    def test_a_good_run_replaces_the_file_and_leaves_no_partial(
        self, ingested, tmp_path
    ):
        out = tmp_path / "sites.csv"
        out.write_text("previous contents\n")
        ingest.write_checked_site_table(ingested["table"].head(5), out)
        assert len(load_sites(out)) == 5
        assert not list(tmp_path.glob("*.partial"))


class TestAmerifluxMapRows:
    def _map(self, tmp_path, text):
        path = tmp_path / "map.csv"
        path.write_text(text)
        return path

    def test_a_repeated_site_id_is_rejected_not_collapsed(self, tmp_path):
        path = self._map(tmp_path, "Site_ID,index\nUS-AAA,10\nUS-BBB,10\nUS-CCC,11\n")
        with pytest.raises(ingest.IngestError, match="more than once"):
            ingest.read_ameriflux_map(path)

    def test_a_blank_identifier_is_rejected_not_read_as_missing(self, tmp_path):
        path = self._map(tmp_path, "Site_ID,index\n,10\nUS-CCC,12\n")
        with pytest.raises(ingest.IngestError, match="blank Site_ID"):
            ingest.read_ameriflux_map(path)

    def test_the_real_map_passes(self):
        assert len(ingest.read_ameriflux_map(SITE_ID_MAP)) == 185


class TestMainReportsRatherThanTracebacks:
    @pytest.mark.filterwarnings("ignore:Specified encoding:UserWarning")
    def test_an_unknown_encoding_is_a_message_not_a_traceback(self, tmp_path, capsys):
        status = ingest.main(
            ["--shapefile", str(SHAPEFILE), "--site-id-map", str(SITE_ID_MAP),
             "--encoding", "not-a-codec", "--out", str(tmp_path / "o.csv")]
        )
        assert status == 1
        err = capsys.readouterr().err
        assert "Traceback" not in err and "error: LookupError" in err

    def test_an_unwritable_output_is_a_message_not_a_traceback(
        self, tmp_path, capsys
    ):
        directory = tmp_path / "out.csv"
        directory.mkdir()
        status = ingest.main(
            ["--shapefile", str(SHAPEFILE), "--site-id-map", str(SITE_ID_MAP),
             "--out", str(directory)]
        )
        assert status == 1
        err = capsys.readouterr().err
        assert "Traceback" not in err and err.startswith("Reading")
