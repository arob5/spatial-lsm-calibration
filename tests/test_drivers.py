"""Tests for the driver reader over the raw ``.clim`` files.

Most cases run against small synthetic ``.clim`` files written to ``tmp_path``
in the real layout -- whole years of 3-hourly rows, 14 columns, tabs between
space-padded fields -- so that every check has a file that trips it and the
expected answer can be written out by hand. Everything about one file is
pySIPNET's, and its own tests cover its parsing and validation; the cases here
are about what this module adds: the directory layout, the stacking into
``(driver_member, site, time)``, the checks pySIPNET does not make, and that the time
axis and the variable attributes are pySIPNET's, unchanged.

The cases at the end run against the local files under ``data/raw/drivers/``
and are skipped when they are absent: that pySIPNET refuses them as they
stand, and that their values, read with regular hour labels, land on the same
axis as a SIPNET run on them.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysipnet.climate import ClimateDrivers
from pysipnet.variables import CLIMATE_COLUMN_NAMES

from conftest import DRIVERS_ROOT, LOCAL_DRIVER_PAIRS, REPOSITORY, site_table_of
from sipnet_calibration.conventions import (
    DATA_ROOT_ENV_VAR,
    LAT_ATTRIBUTES,
    LON_ATTRIBUTES,
    SITE_ATTRIBUTES,
    TIME_COORD_NAMES,
)
from sipnet_calibration.drivers import (
    DRIVER_PRESENT,
    DRIVER_VARIABLES,
    NEGATIVE_TOLERANCE,
    UNITS_PROVENANCE,
    available_members,
    default_drivers_root,
    driver_fields,
    driver_file,
    load_drivers,
    read_driver_file,
)
from sipnet_calibration.observation.time_alignment import aggregate_time
from sipnet_calibration.sites import default_sites_path, load_sites

#: The 14 fields of a legacy-layout row, under SIPNET's own names.
FILE_COLUMNS = (
    "loc", "year", "day", "time", "length", "tair", "tsoil", "par", "precip",
    "vpd", "vpd_soil", "vpress", "wspd", "soil_wetness",
)

#: The value columns, in file order, under SIPNET's names and pySIPNET's.
VALUE_COLUMNS = dict(zip(FILE_COLUMNS[5:13], DRIVER_VARIABLES, strict=True))


# ── synthetic files ───────────────────────────────────────────────────────────


def synthetic_rows(years=(2013,), *, seed=0) -> pd.DataFrame:
    """One whole year of 3-hourly rows per entry of *years*, in the 14-column layout."""
    rng = np.random.default_rng(seed)
    frames = []
    for year in years:
        n_days = 366 if pd.Timestamp(year, 1, 1).is_leap_year else 365
        n = 8 * n_days
        frames.append(
            pd.DataFrame(
                {
                    "loc": 0,
                    "year": year,
                    "day": np.repeat(np.arange(1, n_days + 1), 8),
                    "time": np.tile(np.arange(8) * 3.0, n_days),
                    "length": 0.125,
                    "tair": rng.normal(5, 10, n).round(3),
                    "tsoil": rng.normal(4, 6, n).round(3),
                    "par": np.abs(rng.normal(3, 2, n)).round(4),
                    "precip": np.abs(rng.normal(0, 0.5, n)).round(4),
                    "vpd": np.abs(rng.normal(300, 100, n)).round(2) + 1.0,
                    "vpd_soil": np.abs(rng.normal(200, 100, n)).round(2),
                    "vpress": np.abs(rng.normal(800, 200, n)).round(2) + 1.0,
                    "wspd": np.abs(rng.normal(3, 1, n)).round(3) + 0.1,
                    "soil_wetness": 0.6,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[list(FILE_COLUMNS)]


def write_rows(path: Path, rows: pd.DataFrame) -> Path:
    """Write *rows* the way the source does: tabs between space-padded fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    formats = {"year": "{:d}", "day": "{:3d}", "time": "{:9.6f}", "loc": "{:d}"}
    lines = []
    for record in rows.itertuples(index=False):
        fields = []
        for column, value in zip(FILE_COLUMNS, record, strict=True):
            fields.append(formats.get(column, "{}").format(value))
        lines.append("\t".join(fields))
    path.write_text("\n".join(lines) + "\n")
    return path


def file_name(rows: pd.DataFrame, member: int) -> str:
    first = pd.Timestamp(int(rows["year"].iloc[0]), 1, 1) + pd.Timedelta(days=int(rows["day"].iloc[0]) - 1)
    last = pd.Timestamp(int(rows["year"].iloc[-1]), 1, 1) + pd.Timedelta(days=int(rows["day"].iloc[-1]) - 1)
    return f"ERA5.{member}.{first.date()}.{last.date()}.clim"


def write_pair(root: Path, site: int, member: int, rows: pd.DataFrame | None = None, **kwargs) -> Path:
    """A synthetic file in the real layout, returning its path."""
    if rows is None:
        rows = synthetic_rows(seed=site * 100 + member, **kwargs)
    return write_rows(root / f"ERA5_{site}_{member}" / file_name(rows, member), rows)


@pytest.fixture
def sites_table() -> pd.DataFrame:
    """A site table holding what ``load_drivers`` reads from it."""
    ids = np.arange(1, 11)
    return site_table_of(*ids, lon=-100.0 + ids, lat=40.0 + 0.5 * ids)


@pytest.fixture
def root(tmp_path) -> Path:
    """Two sites with two members each, all present."""
    for site in (3, 7):
        for member in (1, 2):
            write_pair(tmp_path, site, member)
    return tmp_path


# ── schema constants ──────────────────────────────────────────────────────────


class TestSchemaConstants:
    def test_the_variables_are_pysipnets_value_columns_in_file_order(self):
        assert DRIVER_VARIABLES == (
            "air_temperature",
            "soil_temperature",
            "photosynthetically_active_radiation",
            "precipitation",
            "vapor_pressure_deficit",
            "soil_vapor_pressure_deficit",
            "vapor_pressure",
            "wind_speed",
        )
        assert set(DRIVER_VARIABLES) < set(CLIMATE_COLUMN_NAMES)


# ── paths ─────────────────────────────────────────────────────────────────────


class TestPaths:
    def test_default_root_honors_the_data_root_variable(self, monkeypatch, tmp_path):
        monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path))
        assert default_drivers_root() == tmp_path / "raw" / "drivers"
        monkeypatch.delenv(DATA_ROOT_ENV_VAR)
        assert default_drivers_root() == REPOSITORY / "data" / "raw" / "drivers"

    def test_driver_file_finds_the_one_file(self, root):
        path = driver_file(root, 3, 2)
        assert path.parent.name == "ERA5_3_2"
        assert path.name == "ERA5.2.2013-01-01.2013-12-31.clim"

    def test_driver_file_raises_when_the_directory_is_absent(self, root):
        with pytest.raises(FileNotFoundError, match="no driver directory"):
            driver_file(root, 3, 9)

    def test_driver_file_raises_when_the_directory_is_empty(self, root):
        (root / "ERA5_3_4").mkdir()
        with pytest.raises(FileNotFoundError, match="holds no file"):
            driver_file(root, 3, 4)

    def test_driver_file_raises_when_two_files_match(self, root):
        extra = root / "ERA5_3_1" / "ERA5.1.2014-01-01.2014-12-31.clim"
        extra.write_text("")
        with pytest.raises(ValueError, match="promises one"):
            driver_file(root, 3, 1)

    def test_available_members_lists_the_directories_in_order(self, root):
        (root / "ERA5_3_10").mkdir()
        (root / "ERA5_30_5").mkdir()  # another site that shares a prefix
        (root / "ERA5_3_notanumber").mkdir()
        assert available_members(root, 3) == (1, 2, 10)
        assert available_members(root, 30) == (5,)
        assert available_members(root, 4) == ()


# ── read_driver_file ──────────────────────────────────────────────────────────


class TestReadDriverFile:
    def test_returns_pysipnets_reading_of_the_file(self, tmp_path):
        rows = synthetic_rows()
        path = write_rows(tmp_path / "a.clim", rows)
        climate = read_driver_file(path)
        assert isinstance(climate, ClimateDrivers)
        assert climate.n_columns == 14
        # pySIPNET parses with pandas' default float parser, which can land a
        # unit in the last place away from the nearest double to the text.
        for source, name in VALUE_COLUMNS.items():
            np.testing.assert_allclose(
                climate.pandas[name].to_numpy(), rows[source].to_numpy(), rtol=1e-15, atol=0
            )

    def test_a_file_pysipnet_refuses_is_refused_with_its_path_and_pysipnets_reason(self, tmp_path):
        """The drift of ``data/README.md`` Note 15: each label 2.47 s later
        than the one before it plus the declared length."""
        rows = synthetic_rows()
        rows["time"] = np.linspace(0, 24 * 365 - 1, len(rows)) % 24
        path = write_rows(tmp_path / "a.clim", rows)
        with pytest.raises(ValueError, match=r"a\.clim: pySIPNET refused the file: .*drift"):
            read_driver_file(path)

    def test_a_label_that_steps_backwards_is_refused(self, tmp_path):
        rows = synthetic_rows()
        rows.loc[100, "time"] -= 1.0
        path = write_rows(tmp_path / "a.clim", rows)
        with pytest.raises(ValueError, match="pySIPNET refused the file"):
            read_driver_file(path)

    def test_the_declared_clock_is_passed_to_pysipnet(self, tmp_path):
        path = write_rows(tmp_path / "a.clim", synthetic_rows())
        assert read_driver_file(path).time_zone is None
        assert read_driver_file(path, time_zone="UTC").time_zone == "UTC"

    @pytest.mark.parametrize("column", ["par", "precip"])
    def test_rejects_totals_far_below_zero(self, tmp_path, column):
        rows = synthetic_rows()
        rows.loc[7, column] = -0.01
        with pytest.raises(ValueError, match=f"{VALUE_COLUMNS[column]} value\\(s\\) below"):
            read_driver_file(write_rows(tmp_path / "a.clim", rows))

    def test_negative_tolerance_is_the_exact_bound(self, tmp_path):
        rows = synthetic_rows()
        rows.loc[7, "par"] = -NEGATIVE_TOLERANCE
        read_driver_file(write_rows(tmp_path / "a.clim", rows))
        rows.loc[7, "par"] = -NEGATIVE_TOLERANCE * 1.01
        with pytest.raises(ValueError, match="photosynthetically_active_radiation value"):
            read_driver_file(write_rows(tmp_path / "b.clim", rows))

    def test_reads_small_negatives_through_unchanged(self, tmp_path):
        rows = synthetic_rows()
        rows.loc[7, "par"] = -1.374e-05
        rows.loc[8, "precip"] = -NEGATIVE_TOLERANCE / 2
        frame = read_driver_file(write_rows(tmp_path / "a.clim", rows)).pandas
        assert frame["photosynthetically_active_radiation"].iloc[7] == -1.374e-05
        assert frame["precipitation"].iloc[8] == -NEGATIVE_TOLERANCE / 2


# ── load_drivers ──────────────────────────────────────────────────────────────


class TestLoadDrivers:
    def test_returns_the_documented_dims_coords_and_dtypes(self, root, sites_table):
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table)
        assert tuple(dataset.data_vars) == DRIVER_VARIABLES
        for name in DRIVER_VARIABLES:
            assert dataset[name].dims == ("driver_member", "site", "time")
            assert dataset[name].dtype == np.float64
        assert dataset.sizes == {"driver_member": 2, "site": 2, "time": 2920, "bounds": 2}
        assert set(dataset.coords) == {
            "driver_member", "source_index", "site", "lon", "lat",
            "time", "time_step_start", "time_step_length", "time_bounds",
        }
        assert dataset["driver_member"].dtype == np.int64
        assert dataset["source_index"].dtype == np.int64
        assert dataset["site"].dtype == np.int32
        assert dataset["lon"].dims == ("site",) and dataset["lat"].dims == ("site",)
        np.testing.assert_array_equal(dataset["lon"].values, [-97.0, -93.0])
        np.testing.assert_array_equal(dataset["lat"].values, [41.5, 43.5])
        assert dataset["source_index"].dims == ("driver_member",)
        assert dataset["time_bounds"].dims == ("time", "bounds")
        assert "member_source" not in dataset.attrs
        assert "member_correspondence" not in dataset.attrs
        assert dataset.attrs["coverage"] == "complete"

    def test_values_are_the_files_values(self, root, sites_table):
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table)
        frame = read_driver_file(driver_file(root, 7, 2)).pandas
        for name in DRIVER_VARIABLES:
            np.testing.assert_array_equal(
                dataset[name].sel(site=7, driver_member=1).values, frame[name].to_numpy()
            )

    def test_the_time_axis_is_pysipnets(self, root, sites_table):
        """Every time coordinate, values and attributes, is what pySIPNET's
        own Dataset for the file holds: ``time`` at the step end."""
        dataset = load_drivers([3], root=root, sites_table=sites_table)
        own = read_driver_file(driver_file(root, 3, 1)).xarray
        for name in (*TIME_COORD_NAMES, "time_bounds"):
            np.testing.assert_array_equal(dataset[name].values, own[name].values)
            assert dataset[name].attrs == own[name].attrs, name
        starts = pd.date_range("2013-01-01", periods=2920, freq="3h").as_unit("ns")
        np.testing.assert_array_equal(dataset["time_step_start"].values, starts.to_numpy())
        np.testing.assert_array_equal(
            dataset["time"].values, (starts + pd.Timedelta(hours=3)).to_numpy()
        )

    def test_the_clock_is_undeclared_unless_the_caller_declares_it(self, root, sites_table):
        undeclared = load_drivers([3], root=root, sites_table=sites_table)
        assert undeclared["time"].attrs["time_zone"] == "undeclared"
        assert undeclared.attrs["time_zone"] == "undeclared"
        declared = load_drivers([3], root=root, sites_table=sites_table, time_zone="UTC")
        assert declared["time"].attrs["time_zone"] == "UTC"
        assert declared.attrs["time_zone"] == "UTC"

    def test_the_variable_attributes_are_pysipnets_plus_the_units_caveat(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table)
        own = read_driver_file(driver_file(root, 3, 1)).xarray
        for name in DRIVER_VARIABLES:
            attrs = dict(dataset[name].attrs)
            assert attrs.pop("units_provenance") == UNITS_PROVENANCE
            attrs.pop("n_values_below_zero", None)
            attrs.pop("n_values_not_positive", None)
            assert attrs == own[name].attrs, name
        assert dataset["precipitation"].attrs["kind"] == "timestep_total"
        assert dataset["air_temperature"].attrs["kind"] == "timestep_mean"

    def test_the_dataset_attributes_are_pysipnets_plus_the_ensembles(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table)
        own = read_driver_file(driver_file(root, 3, 1)).xarray
        for key, value in own.attrs.items():
            assert dataset.attrs[key] == value, key
        assert dataset.attrs["Conventions"] == "CF-1.11"
        assert dataset.attrs["n_sites"] == 1 and dataset.attrs["n_driver_members"] == 2

    def test_sites_come_back_in_the_order_given(self, root, sites_table):
        dataset = load_drivers([7, 3], root=root, sites_table=sites_table)
        np.testing.assert_array_equal(dataset["site"].values, [7, 3])
        np.testing.assert_array_equal(
            dataset["lon"].values, sites_table.set_index("site_id").loc[[7, 3], "lon"].values
        )
        ascending = load_drivers([3, 7], root=root, sites_table=sites_table)
        xr.testing.assert_identical(
            dataset["air_temperature"].sel(site=[3, 7]), ascending["air_temperature"]
        )

    def test_driver_member_is_zero_based_and_source_index_is_the_file_index(self, root, sites_table):
        write_pair(root, 3, 5)
        write_pair(root, 7, 5)
        dataset = load_drivers([3, 7], source_indices=[5, 1], root=root, sites_table=sites_table)
        np.testing.assert_array_equal(dataset["driver_member"].values, [4, 0])
        np.testing.assert_array_equal(dataset["source_index"].values, [5, 1])
        frame = read_driver_file(driver_file(root, 3, 5)).pandas
        np.testing.assert_array_equal(
            dataset["photosynthetically_active_radiation"].sel(site=3, driver_member=4).values,
            frame["photosynthetically_active_radiation"].to_numpy(),
        )

    def test_source_indices_keep_the_order_given_as_sites_do(self, root, sites_table):
        write_pair(root, 3, 5)
        forward = load_drivers([3], source_indices=[1, 5, 2], root=root, sites_table=sites_table)
        backward = load_drivers([3], source_indices=[2, 5, 1], root=root, sites_table=sites_table)
        np.testing.assert_array_equal(forward["source_index"].values, [1, 5, 2])
        np.testing.assert_array_equal(backward["source_index"].values, [2, 5, 1])
        np.testing.assert_array_equal(
            forward["air_temperature"].values[::-1], backward["air_temperature"].values
        )

    def test_a_source_index_named_twice_is_refused(self, root, sites_table):
        with pytest.raises(ValueError, match=r"source index\(es\) \[1\] more than once"):
            load_drivers([3], source_indices=[1, 2, 1], root=root, sites_table=sites_table)

    def test_source_indices_none_means_every_member_found(self, root, sites_table):
        write_pair(root, 3, 5)
        write_pair(root, 7, 5)
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table)
        np.testing.assert_array_equal(dataset["source_index"].values, [1, 2, 5])

    def test_rejects_a_non_positive_source_index(self, root, sites_table):
        with pytest.raises(ValueError, match=r"source_indices\[0\] must be at least 1"):
            load_drivers([3], source_indices=[0], root=root, sites_table=sites_table)

    @pytest.mark.parametrize("indices", [[1.5], [2.0], "12", 2, ["1", "2"], [True], [np.inf]])
    def test_rejects_source_indices_that_are_not_integers(self, root, sites_table, indices):
        with pytest.raises(TypeError, match="source_indices"):
            load_drivers([3], source_indices=indices, root=root, sites_table=sites_table)

    def test_a_source_index_beyond_int64_is_a_value_error(self, root, sites_table):
        """``2**70`` raised numpy's raw OverflowError."""
        with pytest.raises(ValueError, match=r"source_indices\[0\] must be"):
            load_drivers([3], source_indices=[2**70], root=root, sites_table=sites_table)

    def test_the_member_coordinates_carry_their_attributes(self, root, sites_table):
        from sipnet_calibration.conventions import (
            DATA_SOURCE_MEMBER_ATTRIBUTES,
            SOURCE_INDEX_ATTRIBUTES,
        )

        dataset = load_drivers([3, 7], root=root, sites_table=sites_table)
        assert dict(dataset["driver_member"].attrs) == dict(DATA_SOURCE_MEMBER_ATTRIBUTES)
        assert dict(dataset["source_index"].attrs) == dict(SOURCE_INDEX_ATTRIBUTES)

    def test_rejects_no_source_indices(self, root, sites_table):
        with pytest.raises(ValueError, match="no source indices requested"):
            load_drivers([3], source_indices=[], root=root, sites_table=sites_table)

    def test_a_source_index_beyond_int16_is_a_directory_like_any_other(self, root, sites_table):
        with pytest.raises(FileNotFoundError, match="site 3 source index 40000"):
            load_drivers([3], source_indices=[1, 40000], root=root, sites_table=sites_table)

    @pytest.mark.parametrize("sites", [3, ["3"], "3", [True], {3, 7}, [np.inf], [1.5], [3.0]])
    def test_rejects_sites_that_are_not_integers(self, root, sites_table, sites):
        with pytest.raises(TypeError, match="sites"):
            load_drivers(sites, root=root, sites_table=sites_table)

    @pytest.mark.parametrize("sites", [[2**31], [0], [], [3, 3]])
    def test_rejects_sites_that_are_not_positive_whole_numbers_named_once(
        self, root, sites_table, sites
    ):
        with pytest.raises(ValueError, match="site"):
            load_drivers(sites, root=root, sites_table=sites_table)

    def test_rejects_an_unusable_site_table(self, root, sites_table):
        with pytest.raises(ValueError, match="no 'site_id' column or index"):
            load_drivers([3], root=root, sites_table=sites_table.drop(columns=["site_id"]))
        for column in ("lon", "lat"):
            with pytest.raises(ValueError, match=f"no \\['{column}'\\] column"):
                load_drivers([3], root=root, sites_table=sites_table.drop(columns=[column]))
        with pytest.raises(TypeError, match="must be a DataFrame"):
            load_drivers([3], root=root, sites_table=sites_table.to_dict())
        duplicated = pd.concat([sites_table, sites_table.head(3)], ignore_index=True)
        with pytest.raises(ValueError, match="more than once"):
            load_drivers([3], root=root, sites_table=duplicated)

    def test_a_site_table_keyed_on_site_id_is_accepted(self, root, sites_table):
        keyed = load_drivers([3], root=root, sites_table=sites_table.set_index("site_id"))
        plain = load_drivers([3], root=root, sites_table=sites_table)
        assert keyed.identical(plain)

    def test_lon_and_lat_carry_their_cf_attributes(self, root, sites_table):
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table)
        assert dataset["lon"].attrs == dict(LON_ATTRIBUTES)
        assert dataset["lat"].attrs == dict(LAT_ATTRIBUTES)
        expected = sites_table.set_index("site_id").loc[[3, 7]]
        np.testing.assert_array_equal(dataset["lon"].values, expected["lon"].to_numpy())
        field = driver_fields(dataset)["air_temperature"]
        assert field["lat"].attrs == dict(LAT_ATTRIBUTES)

    def test_site_carries_the_shared_site_attributes(self, root, sites_table):
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table)
        assert dataset["site"].attrs == dict(SITE_ATTRIBUTES)
        assert dataset["site"].dtype == np.int32

    def test_a_directory_off_the_template_is_ignored_by_discovery(self, root, sites_table):
        """``ERA5_3_04`` names member 4 but is not the directory ``driver_file``
        would look in, so discovery does not report it."""
        write_pair(root, 3, 4)
        (root / "ERA5_3_4").rename(root / "ERA5_3_04")
        assert available_members(root, 3) == (1, 2)
        dataset = load_drivers([3], root=root, sites_table=sites_table)
        np.testing.assert_array_equal(dataset["source_index"].values, [1, 2])

    def test_non_physical_values_are_counted_not_altered(self, tmp_path, sites_table):
        rows = synthetic_rows(seed=1)
        rows.loc[[0, 1, 2], "par"] = -1e-6
        rows.loc[[20, 21, 22, 23, 24], "par"] = 0.0  # zeros are not below zero
        rows.loc[[3], "precip"] = -1e-15
        rows.loc[[25, 26], "precip"] = 0.0
        rows.loc[[4, 5], "vpd"] = 0.0
        rows.loc[[6], "wspd"] = 0.0
        rows.loc[[7, 8, 9, 10], "vpd_soil"] = 0.0
        write_pair(tmp_path, 3, 1, rows)
        with warnings.catch_warnings():
            # pySIPNET warns about the zero vpd and wind speed; they are the point.
            warnings.simplefilter("ignore")
            dataset = load_drivers([3], root=tmp_path, sites_table=sites_table)
        par = dataset["photosynthetically_active_radiation"]
        assert par.attrs["n_values_below_zero"] == 3
        assert dataset["precipitation"].attrs["n_values_below_zero"] == 1
        assert dataset["vapor_pressure_deficit"].attrs["n_values_not_positive"] == 2
        assert dataset["wind_speed"].attrs["n_values_not_positive"] == 1
        assert dataset["soil_vapor_pressure_deficit"].attrs["n_values_not_positive"] == 4
        assert par.values[0, 0, 0] == -1e-6
        assert "n_values_below_zero" not in dataset["vapor_pressure_deficit"].attrs

    def test_counts_are_over_every_file_present_and_nothing_else(self, tmp_path, sites_table):
        for member, n_negative in ((1, 2), (2, 5)):
            rows = synthetic_rows(seed=member)
            rows.loc[list(range(n_negative)), "par"] = -1e-6
            write_pair(tmp_path, 3, member, rows)
        write_pair(tmp_path, 7, 1)
        dataset = load_drivers(
            [3, 7], source_indices=[1, 2], root=tmp_path, sites_table=sites_table, allow_missing=True
        )
        assert not dataset[DRIVER_PRESENT].values.all()
        attrs = dataset["photosynthetically_active_radiation"].attrs
        assert attrs["n_values_below_zero"] == 7

    def test_a_discovered_member_beyond_int16_is_read_like_any_other(self, root, sites_table):
        write_pair(root, 3, 40000)
        dataset = load_drivers([3], root=root, sites_table=sites_table)
        assert dataset["source_index"].values.tolist() == [1, 2, 40000]

    @pytest.mark.parametrize("time_zone", ["America/Denver", "EST"])
    def test_an_invalid_clock_is_refused_before_any_file_is_read(
        self, root, sites_table, time_zone
    ):
        with pytest.raises(ValueError, match="time_zone must be") as refusal:
            load_drivers([3], root=root, sites_table=sites_table, time_zone=time_zone)
        assert ".clim" not in str(refusal.value)

    def test_aggregation_keeps_the_declared_clock(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table, time_zone="UTC")
        field = driver_fields(dataset)["air_temperature"].isel(site=0, driver_member=0)
        assert aggregate_time(field, "1D")["time"].attrs["time_zone"] == "UTC"

    # pySIPNET's Dataset declares no units encoding for time and time_bounds.
    @pytest.mark.filterwarnings("ignore:Variable time has datetime type:UserWarning")
    def test_no_coordinate_is_written_with_a_fill_value(self, root, sites_table, tmp_path):
        """CF forbids ``_FillValue`` on a coordinate."""
        path = tmp_path / "drivers.nc"
        load_drivers([3], root=root, sites_table=sites_table).to_netcdf(path)
        with xr.open_dataset(path, decode_cf=False) as written:
            for name in written.coords:
                assert "_FillValue" not in written[name].attrs, name

    def test_missing_pair_raises_by_default(self, root, sites_table):
        write_pair(root, 3, 5)
        with pytest.raises(FileNotFoundError, match=r"\(site, source index\) pair\(s\).*site 7 source index 5.*allow_missing=True"):
            load_drivers([3, 7], root=root, sites_table=sites_table)

    def test_allow_missing_fills_nan_and_writes_driver_present(self, root, sites_table):
        write_pair(root, 3, 5)
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table, allow_missing=True)
        present = dataset[DRIVER_PRESENT]
        assert present.dims == ("driver_member", "site") and present.dtype == bool
        np.testing.assert_array_equal(present.values, [[True, True], [True, True], [True, False]])
        par = dataset["photosynthetically_active_radiation"]
        assert np.isnan(par.sel(site=7, driver_member=4).values).all()
        assert np.isfinite(par.sel(site=3, driver_member=4).values).all()
        assert dataset.attrs["coverage"] == "gaps"
        assert present.attrs["long_name"] == (
            "Whether a driver file existed for the driver member and site"
        )

    def test_driver_present_is_mapped(self, root, sites_table):
        """It validated as a field and was refused by the map for want of units."""
        import matplotlib.pyplot as plt

        from sipnet_calibration.plotting import plot_map

        write_pair(root, 3, 5)
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table, allow_missing=True)
        figure, ax = plt.subplots()
        try:
            plot_map(dataset[DRIVER_PRESENT].sel(driver_member=4), ax=ax)
        finally:
            plt.close(figure)

    def test_two_loads_of_different_members_align_member_for_member(self, root, sites_table):
        """A member's label is its identity: ``source_index - 1`` in every load."""
        write_pair(root, 3, 5)
        both = load_drivers([3], source_indices=[1, 5], root=root, sites_table=sites_table)
        one = load_drivers([3], source_indices=[5], root=root, sites_table=sites_table)
        assert one["driver_member"].values.tolist() == [4]
        difference = both["air_temperature"] - one["air_temperature"]
        assert difference["driver_member"].values.tolist() == [4]
        assert float(abs(difference).max()) == 0.0
        xr.testing.assert_identical(
            one["air_temperature"], both["air_temperature"].sel(driver_member=[4])
        )

    def test_driver_present_is_all_true_when_nothing_is_missing(self, root, sites_table):
        """The variable's presence follows the flag, not the data, so a caller
        that passed ``allow_missing=True`` can always index it."""
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table, allow_missing=True)
        assert dataset[DRIVER_PRESENT].values.all()
        assert dataset.attrs["coverage"] == "complete"
        assert DRIVER_PRESENT not in load_drivers([3, 7], root=root, sites_table=sites_table)

    def test_raises_when_no_file_exists_at_all(self, root, sites_table):
        with pytest.raises(FileNotFoundError, match="no driver directories"):
            load_drivers([4], root=root, sites_table=sites_table)
        with pytest.raises(FileNotFoundError, match="no driver files"):
            load_drivers([3], source_indices=[9], root=root, sites_table=sites_table, allow_missing=True)

    def test_rejects_a_missing_root(self, tmp_path, sites_table):
        with pytest.raises(FileNotFoundError, match="not a directory"):
            load_drivers([3], root=tmp_path / "nowhere", sites_table=sites_table)

    def test_rejects_a_site_not_in_the_site_table(self, root, sites_table):
        write_pair(root, 11, 1)
        with pytest.raises(KeyError, match="not in the site table"):
            load_drivers([11], root=root, sites_table=sites_table)

    def test_rejects_a_directory_member_that_disagrees_with_the_file_name(self, root, sites_table):
        path = driver_file(root, 3, 2)
        path.rename(path.with_name(path.name.replace("ERA5.2.", "ERA5.4.")))
        with pytest.raises(ValueError, match="file name says member 4, the directory says member 2"):
            load_drivers([3], root=root, sites_table=sites_table)

    @pytest.mark.parametrize(
        "name", ["ERA5.2.2013-01-01.2014-12-31.clim", "ERA5.2.2013-01-02.2013-12-31.clim"]
    )
    def test_rejects_file_name_dates_that_do_not_match_the_data(self, root, sites_table, name):
        path = driver_file(root, 3, 2)
        path.rename(path.with_name(name))
        start, end = name.split(".")[2:4]
        with pytest.raises(ValueError, match=f"file name covers {start} to {end}"):
            load_drivers([3], root=root, sites_table=sites_table)

    def test_rejects_a_file_name_off_the_template(self, root, sites_table):
        path = driver_file(root, 3, 2)
        path.rename(path.with_name("ERA5.2.clim"))
        with pytest.raises(ValueError, match="does not follow"):
            load_drivers([3], root=root, sites_table=sites_table)

    def test_rejects_two_files_on_different_time_axes(self, root, sites_table):
        write_pair(root, 7, 3, synthetic_rows(years=(2014,)))
        write_pair(root, 3, 3, synthetic_rows(years=(2013,)))
        with pytest.raises(ValueError, match="time_step_start differs from"):
            load_drivers([3, 7], source_indices=[3], root=root, sites_table=sites_table)
        write_pair(root, 7, 4, synthetic_rows(years=(2013, 2014)))
        write_pair(root, 3, 4, synthetic_rows(years=(2013,)))
        with pytest.raises(ValueError, match="steps where"):
            load_drivers([3, 7], source_indices=[4], root=root, sites_table=sites_table)
        # One label moved by a fraction of a second, well inside what pySIPNET
        # accepts as rounding: still a different axis.
        rows = synthetic_rows(years=(2013,))
        write_pair(root, 3, 6, rows)
        rows.loc[100, "time"] += 5e-6
        write_pair(root, 7, 6, rows)
        with pytest.raises(ValueError, match="time_step_start differs from"):
            load_drivers([3, 7], source_indices=[6], root=root, sites_table=sites_table)
        # The same starts, and a last step half as long: a different axis too.
        rows = synthetic_rows(years=(2013,))
        write_pair(root, 3, 8, rows)
        rows.loc[len(rows) - 1, "length"] = 0.0625
        write_pair(root, 7, 8, rows)
        with pytest.raises(ValueError, match="time_step_length differs from"):
            load_drivers([3, 7], source_indices=[8], root=root, sites_table=sites_table)

    def test_round_trips_through_zarr(self, root, sites_table, tmp_path):
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table)
        store = tmp_path / "cache.zarr"
        dataset.to_zarr(store, consolidated=False)
        back = xr.open_zarr(store, consolidated=False).load()
        xr.testing.assert_identical(back, dataset)


# ── driver_fields ─────────────────────────────────────────────────────────────


class TestDriverFields:
    def test_one_field_per_variable_in_order(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table, allow_missing=True)
        fields = driver_fields(dataset)
        assert tuple(fields) == DRIVER_VARIABLES
        for name, field in fields.items():
            assert field.name == name
            assert field.dims == ("driver_member", "site", "time")
            assert set(field.coords) == {
                "driver_member", "source_index", "site", "lon", "lat", *TIME_COORD_NAMES
            }

    def test_nothing_points_at_a_bounds_variable_a_field_cannot_carry(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table)
        assert dataset["time"].attrs["bounds"] == "time_bounds"
        for field in driver_fields(dataset).values():
            assert "bounds" not in field["time"].attrs
        # The Dataset itself keeps it.
        assert dataset["time"].attrs["bounds"] == "time_bounds"

    def test_fields_carry_the_variable_attributes(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table)
        fields = driver_fields(dataset)
        for name, field in fields.items():
            assert field.attrs == dataset[name].attrs
        assert fields["photosynthetically_active_radiation"].attrs["units"] == "mol m-2"

    def test_driver_present_is_not_a_field(self, root, sites_table):
        write_pair(root, 3, 5)
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table, allow_missing=True)
        assert DRIVER_PRESENT in dataset
        assert DRIVER_PRESENT not in driver_fields(dataset)

    def test_rejects_a_dataset_missing_a_variable(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table).drop_vars("wind_speed")
        with pytest.raises(ValueError, match=r"missing driver variables \['wind_speed'\]"):
            driver_fields(dataset)


# ── the real files ────────────────────────────────────────────────────────────


#: The drivers root the real-file cases read, the fixtures' own.
REAL_ROOT = DRIVERS_ROOT


def real_pairs() -> list[tuple[int, int]]:
    """The pairs of :data:`LOCAL_DRIVER_PAIRS` that have a directory here."""
    return [
        (site, member)
        for site, member in LOCAL_DRIVER_PAIRS
        if (REAL_ROOT / f"ERA5_{site}_{member}").is_dir()
    ]


needs_real_files = pytest.mark.skipif(
    not real_pairs(), reason="the local driver files are not present in this checkout"
)
needs_site_table = pytest.mark.skipif(
    not default_sites_path().exists(), reason="processed/sites/sites.csv is not present"
)


@needs_real_files
class TestRealFiles:
    """Against ``data/raw/drivers/``; skipped when the files are absent."""

    def test_pysipnet_refuses_the_local_files_for_their_drifting_labels(self):
        """``data/README.md`` Note 15. Correcting the files is a separate piece
        of work; until then this is the answer every one of them gives."""
        for pair in real_pairs():
            with pytest.raises(ValueError, match="pySIPNET refused the file: The labels drift"):
                read_driver_file(driver_file(REAL_ROOT, *pair))

    @needs_site_table
    def test_the_local_files_do_not_load(self):
        sites = sorted({site for site, _ in real_pairs()})
        members = sorted({member for _, member in real_pairs()})
        with pytest.raises(ValueError, match="pySIPNET refused the file"):
            load_drivers(
                sites, source_indices=members, root=REAL_ROOT, sites_table=load_sites(),
                allow_missing=True,
            )

    def test_with_regular_labels_the_local_files_load(self, real_drivers):
        present = real_drivers[DRIVER_PRESENT]
        for site, member in real_pairs():
            assert bool(present.sel(site=site, source_index=member).values)
        assert int(present.sum()) == len(real_pairs())
        assert real_drivers.attrs["coverage"] == "gaps"
        assert str(real_drivers["time_step_start"].values[0]) == "2012-01-01T00:00:00.000000000"
        assert str(real_drivers["time"].values[-1]) == "2025-01-01T00:00:00.000000000"

    def test_a_run_and_its_drivers_share_one_time_axis(self, real_drivers, site_1_result):
        """The run's output and the drivers it ran on are aligned by
        construction, not by any code here: both axes are pySIPNET's."""
        from sipnet_calibration.fields import from_sipnet_output

        nee = from_sipnet_output(site_1_result, ["nee"])["net_ecosystem_exchange"]
        par = driver_fields(real_drivers)["photosynthetically_active_radiation"]
        head = par.sel(site=1, source_index=1).isel(time=slice(0, nee.sizes["time"]))
        for name in TIME_COORD_NAMES:
            np.testing.assert_array_equal(nee[name].values, head[name].values)
