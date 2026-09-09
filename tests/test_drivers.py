"""Tests for the driver schema and the reader over the raw ``.clim`` files.

Most cases run against small synthetic ``.clim`` files written to ``tmp_path``
in the real layout -- whole years of rows, the drifting ``time`` column
generated the way the source generates it, the three constant columns -- so
that every check has a file that trips it and the expected answer can be
written out by hand. The traps worth a file each: a ``time`` column that does
*not* drift, a day with seven rows, a directory whose member disagrees with its
file name, a pair with no file at all.

The cases at the end run against the three real files under
``data/raw/drivers/`` and are skipped when they are absent. Those are the ones
that pin the drift model, the clock inference and the counts of non-physical
values to the actual data.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sipnet_calibration.drivers import (
    CLIM_FILE_COLUMNS,
    CLIM_FILE_CONSTANTS,
    CLOCK_STATUS,
    DRIVER_PRESENT,
    DRIVER_VARIABLE_ATTRS,
    DRIVER_VARIABLES,
    MEMBER_SOURCE,
    NEGATIVE_TOLERANCE,
    SOURCE_VARIABLE_NAMES,
    STEPS_PER_DAY,
    TIME_LABEL,
    TIME_ZONE,
    TIMESTEP_HOURS,
    UNITS_STATUS,
    available_members,
    default_drivers_root,
    driver_fields,
    driver_file,
    load_drivers,
    read_clim_file,
)
from sipnet_calibration.sites import DATA_ROOT_ENV_VAR, default_sites_path, load_sites

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = REPO_ROOT / "data" / "raw" / "drivers"

#: The eight value columns, in file order.
VALUE_COLUMNS = tuple(SOURCE_VARIABLE_NAMES)


# ── synthetic files ───────────────────────────────────────────────────────────


def drifting_time_column(n_days: int) -> np.ndarray:
    """The ``time`` column the source generator writes for a year of *n_days*."""
    return np.linspace(0, 24 * n_days - 1, STEPS_PER_DAY * n_days) % 24


def synthetic_rows(years=(2013,), *, seed=0, drift=True) -> pd.DataFrame:
    """One whole year per entry of *years*, in the 14-column layout."""
    rng = np.random.default_rng(seed)
    frames = []
    for year in years:
        n_days = 366 if pd.Timestamp(year, 1, 1).is_leap_year else 365
        n = STEPS_PER_DAY * n_days
        time = drifting_time_column(n_days) if drift else np.tile(np.arange(8) * 3.0, n_days)
        frames.append(
            pd.DataFrame(
                {
                    "loc": CLIM_FILE_CONSTANTS["loc"],
                    "year": year,
                    "day": np.repeat(np.arange(1, n_days + 1), STEPS_PER_DAY),
                    "time": time,
                    "length": CLIM_FILE_CONSTANTS["length"],
                    "tair": rng.normal(5, 10, n).round(3),
                    "tsoil": rng.normal(4, 6, n).round(3),
                    "par": np.abs(rng.normal(3, 2, n)).round(4),
                    "precip": np.abs(rng.normal(0, 0.5, n)).round(4),
                    "vpd": np.abs(rng.normal(300, 100, n)).round(2) + 1.0,
                    "vpd_soil": np.abs(rng.normal(200, 100, n)).round(2),
                    "vpress": np.abs(rng.normal(800, 200, n)).round(2) + 1.0,
                    "wspd": np.abs(rng.normal(3, 1, n)).round(3) + 0.1,
                    "soil_wetness": CLIM_FILE_CONSTANTS["soil_wetness"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[list(CLIM_FILE_COLUMNS)]


def write_rows(path: Path, rows: pd.DataFrame) -> Path:
    """Write *rows* the way the source does: tabs between space-padded fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    formats = {"year": "{:d}", "day": "{:3d}", "time": "{:9.6f}", "loc": "{:d}"}
    lines = []
    for record in rows.itertuples(index=False):
        fields = []
        for column, value in zip(CLIM_FILE_COLUMNS, record, strict=True):
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
    ids = np.arange(1, 11, dtype=np.int32)
    return pd.DataFrame(
        {"site_id": ids, "lon": -100.0 + ids, "lat": 40.0 + 0.5 * ids}
    )


@pytest.fixture
def root(tmp_path) -> Path:
    """Two sites with two members each, all present."""
    for site in (3, 7):
        for member in (1, 2):
            write_pair(tmp_path, site, member)
    return tmp_path


# ── schema constants ──────────────────────────────────────────────────────────


class TestSchemaConstants:
    def test_every_source_column_maps_to_a_processed_name_in_file_order(self):
        value_columns = [c for c in CLIM_FILE_COLUMNS if c not in ("loc", "year", "day", "time", "length", "soil_wetness")]
        assert list(SOURCE_VARIABLE_NAMES) == value_columns
        assert DRIVER_VARIABLES == tuple(SOURCE_VARIABLE_NAMES[c] for c in value_columns)
        assert len(set(DRIVER_VARIABLES)) == len(DRIVER_VARIABLES)

    def test_processed_names_follow_the_naming_convention(self):
        for name in DRIVER_VARIABLES:
            assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name
        assert "tair" not in DRIVER_VARIABLES and "wspd" not in DRIVER_VARIABLES

    def test_every_variable_has_units_long_name_source_name_and_aggregation(self):
        assert set(DRIVER_VARIABLE_ATTRS) == set(DRIVER_VARIABLES)
        for name, attrs in DRIVER_VARIABLE_ATTRS.items():
            assert set(attrs) == {"units", "long_name", "source_name", "aggregation"}
            assert SOURCE_VARIABLE_NAMES[attrs["source_name"]] == name

    def test_totals_sum_and_means_average(self):
        for name, attrs in DRIVER_VARIABLE_ATTRS.items():
            expected = "sum" if name in ("par", "precipitation") else "mean"
            assert attrs["aggregation"] == expected, name

    def test_timestep_constants_agree(self):
        assert TIMESTEP_HOURS == 24 * CLIM_FILE_CONSTANTS["length"] == 3.0
        assert STEPS_PER_DAY == 8


# ── paths ─────────────────────────────────────────────────────────────────────


class TestPaths:
    def test_default_root_honors_the_data_root_variable(self, monkeypatch, tmp_path):
        monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path))
        assert default_drivers_root() == tmp_path / "raw" / "drivers"
        monkeypatch.delenv(DATA_ROOT_ENV_VAR)
        assert default_drivers_root() == REPO_ROOT / "data" / "raw" / "drivers"

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


# ── read_clim_file ────────────────────────────────────────────────────────────


class TestReadClimFile:
    def test_parses_the_fourteen_columns_exactly(self, tmp_path):
        rows = synthetic_rows()
        path = write_rows(tmp_path / "a.clim", rows)
        frame = read_clim_file(path)
        assert tuple(frame.columns) == CLIM_FILE_COLUMNS
        assert frame["year"].dtype == np.int32 and frame["day"].dtype == np.int32
        # Exact: every parsed double re-formats to the text on disk. The
        # writer rounds ``time`` to six decimals, so the comparison is against
        # the text rather than the pre-rounding doubles.
        text = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, names=CLIM_FILE_COLUMNS)
        for column in VALUE_COLUMNS + ("time",):
            assert frame[column].dtype == np.float64
            assert [float(s) for s in text[column]] == frame[column].tolist()
        for column in VALUE_COLUMNS:
            np.testing.assert_array_equal(frame[column].to_numpy(), rows[column].to_numpy())

    def test_rejects_a_row_with_the_wrong_field_count(self, tmp_path):
        path = write_rows(tmp_path / "a.clim", synthetic_rows())
        lines = path.read_text().splitlines()
        lines[10] = lines[10] + "\t1.0"
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="could not be parsed"):
            read_clim_file(path)

    def test_rejects_a_short_row(self, tmp_path):
        path = write_rows(tmp_path / "a.clim", synthetic_rows())
        lines = path.read_text().splitlines()
        lines[10] = "\t".join(lines[10].split("\t")[:-1])
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="could not be read as a number.*fewer than 14 fields"):
            read_clim_file(path)

    def test_rejects_a_thirteen_column_layout(self, tmp_path):
        rows = synthetic_rows().drop(columns="loc")
        path = tmp_path / "a.clim"
        rows.to_csv(path, sep="\t", header=False, index=False)
        with pytest.raises(ValueError, match="expected 14 fields"):
            read_clim_file(path)

    def test_rejects_a_nan_field(self, tmp_path):
        """The text ``nan`` is not admitted as a number, so it fails the parse
        rather than becoming a quiet null."""
        path = write_rows(tmp_path / "a.clim", synthetic_rows())
        text = path.read_text().splitlines()
        fields = text[5].split("\t")
        fields[7] = "nan"
        text[5] = "\t".join(fields)
        path.write_text("\n".join(text) + "\n")
        with pytest.raises(ValueError, match="could not be read as a number"):
            read_clim_file(path)

    def test_rejects_an_infinite_value(self, tmp_path):
        path = write_rows(tmp_path / "a.clim", synthetic_rows())
        text = path.read_text().splitlines()
        fields = text[5].split("\t")
        fields[7] = "inf"
        text[5] = "\t".join(fields)
        path.write_text("\n".join(text) + "\n")
        with pytest.raises(ValueError, match="non-finite.*column 'par'"):
            read_clim_file(path)

    def test_rejects_a_word_where_a_number_belongs(self, tmp_path):
        path = write_rows(tmp_path / "a.clim", synthetic_rows())
        text = path.read_text().splitlines()
        fields = text[5].split("\t")
        fields[7] = "NA"
        text[5] = "\t".join(fields)
        path.write_text("\n".join(text) + "\n")
        with pytest.raises(ValueError, match="could not be read as a number"):
            read_clim_file(path)

    @pytest.mark.parametrize("column", ["loc", "length", "soil_wetness"])
    def test_rejects_a_constant_column_off_its_value(self, tmp_path, column):
        rows = synthetic_rows()
        rows.loc[100, column] = 0.25 if column == "length" else 1
        path = write_rows(tmp_path / "a.clim", rows)
        with pytest.raises(ValueError, match=f"{column} must be"):
            read_clim_file(path)

    def test_rejects_a_day_without_eight_rows(self, tmp_path):
        rows = synthetic_rows().drop(index=100).reset_index(drop=True)
        path = write_rows(tmp_path / "a.clim", rows)
        with pytest.raises(ValueError, match="has 2919 rows, expected 2920"):
            read_clim_file(path)

    def test_rejects_days_out_of_sequence(self, tmp_path):
        rows = synthetic_rows()
        rows.loc[rows["day"] == 40, "day"] = 41
        rows.loc[(rows["day"] == 41)].index  # two days now claim to be 41
        rows.loc[rows.index[8 * 40 : 8 * 41], "day"] = 40  # swap them
        path = write_rows(tmp_path / "a.clim", rows)
        with pytest.raises(ValueError, match="does not run 1..365"):
            read_clim_file(path)

    def test_rejects_a_leap_year_with_365_days(self, tmp_path):
        rows = synthetic_rows(years=(2013,))
        rows["year"] = 2012
        path = write_rows(tmp_path / "a.clim", rows)
        with pytest.raises(ValueError, match="year 2012 has 2920 rows, expected 2928"):
            read_clim_file(path)

    def test_rejects_non_contiguous_years(self, tmp_path):
        rows = synthetic_rows(years=(2013, 2015))
        path = write_rows(tmp_path / "a.clim", rows)
        with pytest.raises(ValueError, match="years are not contiguous"):
            read_clim_file(path)

    def test_rejects_a_time_column_without_the_drift(self, tmp_path):
        path = write_rows(tmp_path / "a.clim", synthetic_rows(drift=False))
        with pytest.raises(ValueError, match="departs from the modulo-24 linspace model"):
            read_clim_file(path)

    def test_accepts_the_drifting_time_column(self, tmp_path):
        rows = synthetic_rows(years=(2012, 2013))
        frame = read_clim_file(write_rows(tmp_path / "a.clim", rows))
        # The last slot of each year is labeled two hours late, and passes.
        assert frame["time"].iloc[8 * 366 - 1] == pytest.approx(23.0)
        assert frame["time"].iloc[-1] == pytest.approx(23.0)

    def test_rejects_par_far_below_zero(self, tmp_path):
        rows = synthetic_rows()
        rows.loc[7, "par"] = -0.01
        with pytest.raises(ValueError, match="par value\\(s\\) below"):
            read_clim_file(write_rows(tmp_path / "a.clim", rows))

    def test_reads_small_negative_par_through_unchanged(self, tmp_path):
        rows = synthetic_rows()
        rows.loc[7, "par"] = -1.374e-05
        rows.loc[8, "precip"] = -NEGATIVE_TOLERANCE / 2
        frame = read_clim_file(write_rows(tmp_path / "a.clim", rows))
        assert frame["par"].iloc[7] == -1.374e-05
        assert frame["precip"].iloc[8] == -NEGATIVE_TOLERANCE / 2


# ── load_drivers ──────────────────────────────────────────────────────────────


class TestLoadDrivers:
    def test_returns_the_documented_dims_coords_and_dtypes(self, root, sites_table):
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table)
        assert set(dataset.data_vars) == set(DRIVER_VARIABLES)
        for name in DRIVER_VARIABLES:
            assert dataset[name].dims == ("member", "site", "time")
            assert dataset[name].dtype == np.float64
        assert dataset.sizes == {"member": 2, "site": 2, "time": 2920}
        assert dataset["member"].dtype == np.int16
        assert dataset["site"].dtype == np.int32
        assert dataset["lon"].dims == ("site",) and dataset["lat"].dims == ("site",)
        np.testing.assert_array_equal(dataset["lon"].values, [-97.0, -93.0])
        np.testing.assert_array_equal(dataset["lat"].values, [41.5, 43.5])
        assert dataset["source_member_index"].dims == ("member",)
        assert dataset["time"].dtype == np.dtype("datetime64[ns]")
        assert dataset.attrs["member_source"] == MEMBER_SOURCE
        assert dataset.attrs["coverage"] == "complete"
        assert dataset.attrs["timestep_days"] == CLIM_FILE_CONSTANTS["length"]

    def test_values_are_the_files_values(self, root, sites_table):
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table)
        frame = read_clim_file(driver_file(root, 7, 2))
        for source, name in SOURCE_VARIABLE_NAMES.items():
            np.testing.assert_array_equal(
                dataset[name].sel(site=7, member=1).values, frame[source].to_numpy()
            )

    def test_sites_come_back_ascending_whatever_order_is_given(self, root, sites_table):
        dataset = load_drivers([7, 3, 7], root=root, sites_table=sites_table)
        np.testing.assert_array_equal(dataset["site"].values, [3, 7])

    def test_member_is_zero_based_and_source_member_index_is_the_file_index(self, root, sites_table):
        write_pair(root, 3, 5)
        write_pair(root, 7, 5)
        dataset = load_drivers([3, 7], members=[5, 1], root=root, sites_table=sites_table)
        np.testing.assert_array_equal(dataset["member"].values, [0, 1])
        np.testing.assert_array_equal(dataset["source_member_index"].values, [1, 5])
        frame = read_clim_file(driver_file(root, 3, 5))
        np.testing.assert_array_equal(
            dataset["par"].sel(site=3, member=1).values, frame["par"].to_numpy()
        )

    def test_members_none_means_every_member_found(self, root, sites_table):
        write_pair(root, 3, 5)
        write_pair(root, 7, 5)
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table)
        np.testing.assert_array_equal(dataset["source_member_index"].values, [1, 2, 5])

    def test_rejects_a_non_positive_member_index(self, root, sites_table):
        with pytest.raises(ValueError, match="1-based"):
            load_drivers([3], members=[0], root=root, sites_table=sites_table)

    def test_time_axis_is_the_nominal_three_hourly_grid(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table)
        expected = pd.date_range("2013-01-01", periods=2920, freq="3h").as_unit("ns")
        pd.testing.assert_index_equal(dataset.indexes["time"], expected, check_names=False)

    def test_time_attributes_record_clock_label_and_status(self, root, sites_table):
        attrs = load_drivers([3], root=root, sites_table=sites_table)["time"].attrs
        assert attrs["time_zone"] == TIME_ZONE == "UTC"
        assert attrs["time_label"] == TIME_LABEL == "interval_end"
        assert attrs["clock_status"] == CLOCK_STATUS == "inferred"
        assert "clock_provenance" in attrs and "time_label_note" in attrs

    def test_variable_attributes_carry_units_and_the_units_caveat(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table)
        for name in DRIVER_VARIABLES:
            attrs = dataset[name].attrs
            for key, value in DRIVER_VARIABLE_ATTRS[name].items():
                assert attrs[key] == value
            assert attrs["units_status"] == UNITS_STATUS
            assert "units_provenance" in attrs

    def test_non_physical_values_are_counted_not_altered(self, tmp_path, sites_table):
        rows = synthetic_rows(seed=1)
        rows.loc[[0, 1, 2], "par"] = -1e-6
        rows.loc[[3], "precip"] = -1e-15
        rows.loc[[4, 5], "vpd"] = 0.0
        rows.loc[[6], "wspd"] = 0.0
        rows.loc[[7, 8, 9, 10], "vpd_soil"] = 0.0
        write_pair(tmp_path, 3, 1, rows)
        dataset = load_drivers([3], root=tmp_path, sites_table=sites_table)
        assert dataset["par"].attrs["n_values_below_zero"] == 3
        assert dataset["precipitation"].attrs["n_values_below_zero"] == 1
        assert dataset["vpd"].attrs["n_values_not_positive"] == 2
        assert dataset["wind_speed"].attrs["n_values_not_positive"] == 1
        assert dataset["soil_vpd"].attrs["n_values_not_positive"] == 4
        assert dataset["par"].values[0, 0, 0] == -1e-6
        assert "n_values_below_zero" not in dataset["vpd"].attrs

    def test_missing_pair_raises_by_default(self, root, sites_table):
        write_pair(root, 3, 5)
        with pytest.raises(FileNotFoundError, match="site 7 member 5.*allow_missing=True"):
            load_drivers([3, 7], root=root, sites_table=sites_table)

    def test_allow_missing_fills_nan_and_writes_driver_present(self, root, sites_table):
        write_pair(root, 3, 5)
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table, allow_missing=True)
        present = dataset[DRIVER_PRESENT]
        assert present.dims == ("member", "site") and present.dtype == bool
        np.testing.assert_array_equal(present.values, [[True, True], [True, True], [True, False]])
        assert np.isnan(dataset["par"].sel(site=7, member=2).values).all()
        assert np.isfinite(dataset["par"].sel(site=3, member=2).values).all()
        assert dataset.attrs["coverage"] == "gaps"

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
            load_drivers([3], members=[9], root=root, sites_table=sites_table, allow_missing=True)

    def test_rejects_a_missing_root(self, tmp_path, sites_table):
        with pytest.raises(FileNotFoundError, match="not a directory"):
            load_drivers([3], root=tmp_path / "nowhere", sites_table=sites_table)

    def test_rejects_a_site_not_in_the_site_table(self, root, sites_table):
        write_pair(root, 11, 1)
        with pytest.raises(ValueError, match="not in the site table"):
            load_drivers([11], root=root, sites_table=sites_table)

    def test_rejects_a_directory_member_that_disagrees_with_the_file_name(self, root, sites_table):
        path = driver_file(root, 3, 2)
        path.rename(path.with_name(path.name.replace("ERA5.2.", "ERA5.4.")))
        with pytest.raises(ValueError, match="file name says member 4, the directory says member 2"):
            load_drivers([3], root=root, sites_table=sites_table)

    def test_rejects_file_name_dates_that_do_not_match_the_data(self, root, sites_table):
        path = driver_file(root, 3, 2)
        path.rename(path.with_name("ERA5.2.2013-01-01.2014-12-31.clim"))
        with pytest.raises(ValueError, match="file name covers 2013-01-01 to 2014-12-31"):
            load_drivers([3], root=root, sites_table=sites_table)

    def test_rejects_a_file_name_off_the_template(self, root, sites_table):
        path = driver_file(root, 3, 2)
        path.rename(path.with_name("ERA5.2.clim"))
        with pytest.raises(ValueError, match="does not follow"):
            load_drivers([3], root=root, sites_table=sites_table)

    def test_rejects_two_files_on_different_grids(self, root, sites_table):
        write_pair(root, 7, 3, synthetic_rows(years=(2014,)))
        write_pair(root, 3, 3, synthetic_rows(years=(2013,)))
        with pytest.raises(ValueError, match="year differs from"):
            load_drivers([3, 7], members=[3], root=root, sites_table=sites_table)
        write_pair(root, 7, 4, synthetic_rows(years=(2013, 2014)))
        write_pair(root, 3, 4, synthetic_rows(years=(2013,)))
        with pytest.raises(ValueError, match="rows where"):
            load_drivers([3, 7], members=[4], root=root, sites_table=sites_table)

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
            assert field.dims == ("member", "site", "time")
            assert set(field.coords) >= {"lon", "lat", "source_member_index", "time", "site", "member"}

    def test_fields_carry_the_variable_attributes(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table)
        fields = driver_fields(dataset)
        assert fields["par"].attrs["units"] == "mol m-2"
        assert fields["par"].attrs["aggregation"] == "sum"
        assert fields["par"].attrs["units_status"] == UNITS_STATUS
        assert fields["vpd"].attrs["n_values_not_positive"] == 0

    def test_driver_present_is_not_a_field(self, root, sites_table):
        write_pair(root, 3, 5)
        dataset = load_drivers([3, 7], root=root, sites_table=sites_table, allow_missing=True)
        assert DRIVER_PRESENT in dataset
        assert DRIVER_PRESENT not in driver_fields(dataset)

    def test_rejects_a_dataset_missing_a_variable(self, root, sites_table):
        dataset = load_drivers([3], root=root, sites_table=sites_table).drop_vars("vpd")
        with pytest.raises(ValueError, match=r"missing driver variables \['vpd'\]"):
            driver_fields(dataset)


# ── the real files ────────────────────────────────────────────────────────────


def real_pairs() -> list[tuple[int, int]]:
    if not REAL_ROOT.is_dir():
        return []
    pairs = []
    for directory in REAL_ROOT.iterdir():
        match = re.fullmatch(r"ERA5_(\d+)_(\d+)", directory.name)
        if match:
            pairs.append((int(match.group(1)), int(match.group(2))))
    return sorted(pairs)


needs_real_files = pytest.mark.skipif(
    not real_pairs(), reason="data/raw/drivers/ is not present in this checkout"
)
needs_site_table = pytest.mark.skipif(
    not default_sites_path().exists(), reason="processed/sites/sites.csv is not present"
)


@pytest.fixture(scope="module")
def frames() -> dict[tuple[int, int], pd.DataFrame]:
    """The three real files, parsed once."""
    return {pair: read_clim_file(driver_file(REAL_ROOT, *pair)) for pair in real_pairs()}


@needs_real_files
class TestRealFiles:
    """Against ``data/raw/drivers/``; skipped when the files are absent."""

    def test_the_local_files_parse_and_pass_every_check(self, frames):
        for frame in frames.values():
            assert len(frame) == 37992
            assert frame["year"].iloc[0] == 2012 and frame["year"].iloc[-1] == 2024

    def test_the_drift_model_holds_to_five_in_ten_million_hours(self, frames):
        for frame in frames.values():
            for year, group in frame.groupby("year"):
                n_days = len(group) // STEPS_PER_DAY
                model = drifting_time_column(n_days)
                assert np.abs(group["time"].to_numpy() - model).max() < 6e-7
            # And the label is two hours late by the last slot of the year.
            assert frame["time"].iloc[-1] == pytest.approx(23.0, abs=1e-6)

    @needs_site_table
    def test_the_local_files_are_non_rectangular_and_load_with_allow_missing(self):
        pairs = real_pairs()
        sites = sorted({s for s, _ in pairs})
        members = sorted({m for _, m in pairs})
        if len(pairs) == len(sites) * len(members):
            pytest.skip("the local files happen to form a rectangle")
        with pytest.raises(FileNotFoundError, match="allow_missing=True"):
            load_drivers(sites, root=REAL_ROOT, sites_table=load_sites())
        dataset = load_drivers(sites, root=REAL_ROOT, sites_table=load_sites(), allow_missing=True)
        present = dataset[DRIVER_PRESENT]
        for site, member in pairs:
            assert bool(present.sel(site=site, source_member_index=member).values) is True  # noqa: E712
        assert int(present.sum()) == len(pairs)
        assert dataset.attrs["coverage"] == "gaps"
        assert str(dataset["time"].values[0]) == "2012-01-01T00:00:00.000000000"
        assert str(dataset["time"].values[-1]) == "2024-12-31T21:00:00.000000000"

    @needs_site_table
    def test_par_phase_moves_with_longitude_as_a_utc_clock_requires(self, frames):
        """The first-harmonic PAR phase shifts about 3.6 h between sites 1 and
        27, not 0, which is the evidence behind ``clock_status``."""
        by_site = {}
        for (site, _member), frame in frames.items():
            by_site.setdefault(site, frame)
        if len(by_site) < 2:
            pytest.skip("the clock test needs two sites")
        table = load_sites().set_index("site_id")

        def phase_hours(frame: pd.DataFrame) -> float:
            summer = frame[(frame["day"] >= 120) & (frame["day"] <= 240)]
            slot = np.floor(summer["time"].to_numpy() / TIMESTEP_HOURS)
            mean_by_slot = summer.groupby(slot)["par"].mean()
            angle = 2 * np.pi * (TIMESTEP_HOURS * mean_by_slot.index.to_numpy()) / 24
            phase = np.arctan2((mean_by_slot * np.sin(angle)).sum(), (mean_by_slot * np.cos(angle)).sum())
            return float((phase % (2 * np.pi)) * 24 / (2 * np.pi))

        sites = sorted(by_site)
        west, east = min(sites, key=lambda s: table.loc[s, "lon"]), max(sites, key=lambda s: table.loc[s, "lon"])
        shift = phase_hours(by_site[west]) - phase_hours(by_site[east])
        utc_expects = (table.loc[east, "lon"] - table.loc[west, "lon"]) / 15
        assert utc_expects > 1.0, "the two sites are too close in longitude to discriminate"
        assert shift == pytest.approx(utc_expects, abs=0.6)

    def test_non_physical_value_counts_match_the_files(self, frames):
        for frame in frames.values():
            assert (frame["par"] < 0).sum() > 0, "the source is known to hold negative par"
            assert frame["par"].min() >= -NEGATIVE_TOLERANCE
            assert frame["precip"].min() >= -NEGATIVE_TOLERANCE
            assert (frame["wspd"] > 0).all()
            assert (frame["vpress"] > 0).all()
            assert (frame["vpd_soil"] == 0).mean() > 0.2
