"""Tests for the constraint specs, the processed files they describe, and the ingest.

Three layers. The specs are checked for internal consistency and against the
real raw files' headers. The conversion is exercised on small synthetic raw
tables, one per time structure, where the expected processed file can be written out
by hand and every refusal can be provoked. Finally the real files are built
and the results compared against the assembled file the reanalysis used,
``processed/constraints_annual.nc``, which the new processed files must reproduce
exactly under the assembler's own rules: those rules -- the July 15 selection
for LAI with its 30-day window and earlier-date tie-break, the 0.66 floor on
the LAI standard deviation, the factor of ten on soil carbon -- live in this
file and nowhere in the processed files.

The real-data cases skip when the raw files, the site table or the assembled
file are absent from the working copy.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import REPOSITORY, load_script, write_site_table_csv
from sipnet_calibration import constraints as module
from sipnet_calibration.constraints import (
    CONSTRAINT_NAMES,
    CONSTRAINTS,
    STANDARD_DEVIATION,
    VALUE,
    ConstraintSpec,
    TimeStructure,
    build_constraint,
    constraint_fields,
    constraint_path,
    constraint_standard_deviations,
    describe,
    load_constraint,
    netcdf_encoding,
    read_raw,
    resolve_constraint,
)
from sipnet_calibration.conventions import CF_CONVENTIONS, data_root
from sipnet_calibration.sites import N_SITES, default_sites_path, load_sites

#: The tracked raw files, found from the repository rather than the data root.
RAW_DIR = REPOSITORY / "data" / "raw" / "constraints"
ASSEMBLED = data_root() / "processed" / "constraints_annual.nc"

#: Old assembled name -> new constraint, for the reproduction tests.
ASSEMBLED_NAMES = {
    "aboveground_wood_carbon": "landtrendr_aboveground_biomass",
    "lai": "modis_leaf_area_index",
    "soil_moisture_percent": "smap_soil_moisture",
    "total_soil_carbon": "soilgrids_soil_organic_carbon",
}


ingest = load_script("scripts/ingest_constraints.py")


# ── synthetic fixtures ────────────────────────────────────────────────────────

SYNTHETIC_SITES = [1, 2, 3, 4]
SYNTHETIC_COORDS = {1: (-100.0, 40.0), 2: (-101.0, 41.0), 3: (-102.0, 42.0), 4: (-103.0, 43.0)}


def _write_sites(path: Path, site_ids=SYNTHETIC_SITES) -> Path:
    """A minimal site table that ``load_sites`` accepts."""
    return write_site_table_csv(
        path,
        site_ids,
        lon=[SYNTHETIC_COORDS[site][0] for site in site_ids],
        lat=[SYNTHETIC_COORDS[site][1] for site in site_ids],
    )


def _write_raw(root: Path, spec: ConstraintSpec, rows: list[dict]) -> Path:
    """Write rows as the raw files are written: gzipped CSV, ``NA``, ``%.17g``."""
    lines = [",".join(spec.raw_columns)]
    for row in rows:
        csv_fields = []
        for column in spec.raw_columns:
            value = row[column]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                csv_fields.append("NA")
            elif isinstance(value, float):
                csv_fields.append(f"{value:.17g}")
            else:
                csv_fields.append(str(value))
        lines.append(",".join(csv_fields))
    root.mkdir(parents=True, exist_ok=True)
    path = root / spec.raw_file
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _spec(**overrides) -> ConstraintSpec:
    """A small annual spec, with fields overridden per test."""
    fields = dict(
        name="test_annual",
        long_label="Test quantity",
        units="Mg ha-1",
        constituent="C",
        description="A synthetic annual constraint.",
        upstream_product="test",
        time_structure=TimeStructure.ANNUAL,
        raw_file="test_annual.csv.gz",
        raw_columns=("site_id", "year", "mean", "sd"),
        value_column="mean",
        sd_column="sd",
        time_column="year",
    )
    fields.update(overrides)
    return ConstraintSpec(**fields)


ANNUAL = _spec()
DATED = _spec(
    name="test_dated",
    time_structure=TimeStructure.DATED,
    raw_file="test_dated.csv.gz",
    raw_columns=("date", "site_id", "lat", "lon", "obs", "sd", "qc"),
    value_column="obs",
    time_column="date",
    quality_column="qc",
    quality_pass="000",
    units="m2 m-2",
    constituent="",
)
STATIC = _spec(
    name="test_static",
    time_structure=TimeStructure.STATIC,
    raw_file="test_static.csv.gz",
    raw_columns=("site_id", "soc", "sd", "year"),
    value_column="soc",
)

ANNUAL_ROWS = [
    dict(site_id=1, year=2012, mean=10.0, sd=2.0),
    dict(site_id=1, year=2013, mean=11.0, sd=0.0),
    dict(site_id=2, year=2012, mean=0.1, sd=0.66),
    dict(site_id=2, year=2013, mean=np.nan, sd=np.nan),
    dict(site_id=4, year=2013, mean=40.0, sd=4.0),
]

DATED_ROWS = [
    dict(date="2012-07-11", site_id=1, lat=40.0, lon=-100.0, obs=1.5, sd=0.1, qc="000"),
    dict(date="2012-07-15", site_id=1, lat=40.0, lon=-100.0, obs=0.0, sd=24.8, qc="001"),
    dict(date="2012-07-15", site_id=2, lat=41.0, lon=-101.0, obs=2.5, sd=0.2, qc="000"),
    dict(date="2013-07-16", site_id=2, lat=41.0, lon=-101.0, obs=2.7, sd=0.0, qc="000"),
]

STATIC_ROWS = [
    dict(site_id=1, soc=742.8590087890625, sd=522.8154296875, year=2012),
    dict(site_id=1, soc=742.8590087890625, sd=522.8154296875, year=2013),
    dict(site_id=2, soc=np.nan, sd=np.nan, year=2012),
    dict(site_id=2, soc=np.nan, sd=np.nan, year=2013),
    dict(site_id=3, soc=100.5, sd=10.25, year=2012),
    dict(site_id=3, soc=100.5, sd=10.25, year=2013),
]


@pytest.fixture
def site_table(tmp_path) -> pd.DataFrame:
    return load_sites(_write_sites(tmp_path / "sites" / "sites.csv"))


@pytest.fixture
def raw_root(tmp_path) -> Path:
    return tmp_path / "raw"


def _ingest(spec, rows, raw_root, site_table, out_dir) -> xr.Dataset:
    _write_raw(raw_root, spec, rows)
    return ingest.ingest(spec, raw_root, site_table, out_dir)


def _dates(*days: str) -> np.ndarray:
    return np.array(days, dtype="datetime64[ns]")


# ── the specs ─────────────────────────────────────────────────────────────────


def test_every_registered_spec_names_its_file_and_columns_consistently():
    names = [spec.name for spec in CONSTRAINTS]
    assert names == list(CONSTRAINT_NAMES)
    assert len(set(names)) == len(names)
    assert len({spec.raw_file for spec in CONSTRAINTS}) == len(CONSTRAINTS)
    for spec in CONSTRAINTS:
        assert spec.raw_file == f"{spec.name}.csv.gz"
        assert spec.units_provenance, spec.name


def test_static_specs_have_no_bounds_and_annual_specs_do():
    for spec in CONSTRAINTS:
        assert spec.has_time_bounds == (spec.time_structure is TimeStructure.ANNUAL)
        assert spec.dims == (("site",) if spec.time_structure is TimeStructure.STATIC else ("site", "time"))


def test_resolve_constraint_names_the_known_constraints_on_a_miss():
    assert resolve_constraint("modis_leaf_area_index").quality_pass == "000"
    with pytest.raises(KeyError, match="modis_leaf_area_index"):
        resolve_constraint("lai")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"name": "Bad Name"}, "lower_case_with_underscores"),
        ({"units": "Mg C ha-1"}, "substance"),
        ({"value_column": "nope"}, "not in raw_columns"),
        ({"time_column": None}, "needs a time_column"),
        ({"quality_column": "sd"}, "go together"),
        ({"raw_columns": ("site_id", "year", "mean", "sd", "sd")}, "repeats"),
        ({"raw_columns": ("id", "year", "mean", "sd")}, "site_id"),
        ({"description": ""}, "needs a description"),
    ],
)
def test_an_inconsistent_spec_is_refused_at_construction(overrides, message):
    with pytest.raises(ValueError, match=message):
        _spec(**overrides)


def test_xarray_attributes_carry_the_spec():
    attrs = resolve_constraint("landtrendr_aboveground_biomass").xarray_attributes()
    assert attrs["units"] == "Mg ha-1"
    assert attrs["constituent"] == "C"
    assert attrs["source_column"] == "agb_mean"
    assert "calendar year" in attrs["time_reference"]
    assert "constituent" not in resolve_constraint("modis_leaf_area_index").xarray_attributes()


def test_the_processed_file_carries_exactly_the_documented_attributes(raw_root, site_table, tmp_path):
    """The data model's attribute lists, as written, with no extras and none missing."""
    processed = _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, tmp_path / "out")
    assert set(processed[VALUE].attrs) == {
        "units", "constituent", "long_name", "description", "upstream_product", "source_file",
        "source_column", "time_reference", "units_provenance",
    }
    assert set(processed[STANDARD_DEVIATION].attrs) == {
        "units", "constituent", "long_name", "description", "source_file", "source_column",
    }
    assert set(processed.attrs) == {
        "Conventions", "title", "constraint", "upstream_product", "source_file", "time_structure",
        "rows_read", "rows_dropped_by_quality_flag", "rows_collapsed_as_copies", "history",
        "created",
    }
    assert processed.attrs["time_structure"] == "annual"
    assert processed["time"].attrs["standard_name"] == "time"
    assert processed["time"].attrs["axis"] == "T"
    assert processed["lon"].attrs == {
        "standard_name": "longitude", "long_name": "Longitude", "units": "degrees_east"
    }
    assert processed["lat"].attrs["standard_name"] == "latitude"
    assert processed["lat"].attrs["units"] == "degrees_north"
    assert processed[VALUE].attrs["upstream_product"] == "test"
    assert "calendar year" in processed[VALUE].attrs["time_reference"]

    dated = _ingest(DATED, DATED_ROWS, raw_root, site_table, tmp_path / "out")
    assert "constituent" not in dated[VALUE].attrs
    assert dated.attrs["time_structure"] == "dated"


def test_describe_names_the_file_the_columns_and_the_filter():
    text = describe(resolve_constraint("modis_leaf_area_index"))
    assert "modis_leaf_area_index.csv.gz" in text
    assert "'lai'" in text and "'qc' == '000'" in text
    assert "dated" in text


# ── the conversion, on synthetic tables ───────────────────────────────────────


def test_an_annual_table_becomes_a_dense_dataset_with_calendar_year_bounds(
    raw_root, site_table, tmp_path
):
    processed = _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, tmp_path / "out")

    assert processed[VALUE].dims == ("site", "time")
    assert processed["site"].values.tolist() == SYNTHETIC_SITES
    assert processed["site"].dtype == np.int32
    assert np.array_equal(processed["time"].values, _dates("2012-01-01", "2013-01-01"))
    assert processed["time"].attrs["bounds"] == "time_bounds"
    assert processed["time_bounds"].dims == ("time", "bounds")
    assert np.array_equal(processed["time_bounds"].values[0], _dates("2012-01-01", "2013-01-01"))
    assert np.array_equal(processed["time_bounds"].values[1], _dates("2013-01-01", "2014-01-01"))

    value = processed[VALUE].values
    assert value[0].tolist() == [10.0, 11.0]  # site 1
    assert value[1, 0] == 0.1 and np.isnan(value[1, 1])  # site 2: 0.1 exact, then NA
    assert np.isnan(value[2]).all()  # site 3 never observed
    assert np.isnan(value[3, 0]) and value[3, 1] == 40.0
    assert processed[STANDARD_DEVIATION].values[0, 1] == 0.0  # written through
    assert processed["lon"].values.tolist() == [-100.0, -101.0, -102.0, -103.0]

    assert processed.attrs["Conventions"] == "CF-1.11" == CF_CONVENTIONS
    assert processed.attrs["rows_read"] == 5
    assert processed.attrs["rows_dropped_by_quality_flag"] == 0
    assert processed.attrs["rows_collapsed_as_copies"] == 0
    assert processed[VALUE].attrs["units"] == "Mg ha-1"


def test_a_dated_table_drops_the_flagged_rows_and_counts_them(raw_root, site_table, tmp_path):
    processed = _ingest(DATED, DATED_ROWS, raw_root, site_table, tmp_path / "out")

    assert processed[VALUE].dims == ("site", "time")
    assert "time_bounds" not in processed.coords
    assert "bounds" not in processed["time"].attrs
    assert np.array_equal(
        processed["time"].values, _dates("2012-07-11", "2012-07-15", "2013-07-16")
    )
    value = processed[VALUE].values
    assert value[0].tolist()[:1] == [1.5] and np.isnan(value[0, 1])  # flagged row gone
    assert value[1, 1] == 2.5 and value[1, 2] == 2.7
    assert processed.attrs["rows_dropped_by_quality_flag"] == 1
    assert processed.attrs["rows_read"] == 4


def test_a_static_table_collapses_to_one_value_per_site(raw_root, site_table, tmp_path):
    processed = _ingest(STATIC, STATIC_ROWS, raw_root, site_table, tmp_path / "out")

    assert processed[VALUE].dims == ("site",)
    assert "time" not in processed.coords
    assert processed[VALUE].values[0] == 742.8590087890625
    assert processed[STANDARD_DEVIATION].values[2] == 10.25
    assert np.isnan(processed[VALUE].values[1]) and np.isnan(processed[VALUE].values[3])
    assert "collapsed" in processed.attrs["history"]
    assert processed.attrs["rows_collapsed_as_copies"] == 3


def test_the_written_file_reads_back_through_the_loader_identically(
    raw_root, site_table, tmp_path
):
    out_dir = tmp_path / "out"
    built = _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, out_dir)
    with load_constraint(ANNUAL, constraint_path(ANNUAL, out_dir)) as read:
        read = read.load()
    assert read.identical(built)
    assert not (out_dir / f"{ANNUAL.name}.nc.partial").exists()


def test_no_coordinate_is_encoded_with_a_fill_value(raw_root, site_table, tmp_path):
    processed = _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, tmp_path / "out")
    encoding = netcdf_encoding(processed)
    for coordinate in processed.coords:
        assert encoding[str(coordinate)]["_FillValue"] is None
    assert np.isnan(encoding[VALUE]["_FillValue"])

    raw = xr.open_dataset(constraint_path(ANNUAL.name, tmp_path / "out"), decode_cf=False)
    with raw:
        assert raw.attrs["Conventions"] == "CF-1.11"
        assert raw["time"].attrs["units"] == "days since 2000-01-01"
        assert raw["time"].attrs["calendar"] == "proleptic_gregorian"
        assert raw["time"].dtype == np.int32 and raw["time_bounds"].dtype == np.int32
        assert raw["site"].dtype == np.int32
        assert "_FillValue" not in raw["time"].attrs
        assert "time_bounds" not in raw[VALUE].attrs.get("coordinates", "")


def test_read_raw_parses_seventeen_digit_doubles_exactly(raw_root):
    value = 0.1 + 0.2  # 0.30000000000000004, which a lossy parse rounds
    _write_raw(raw_root, ANNUAL, [dict(site_id=1, year=2012, mean=value, sd=1.0)])
    frame = read_raw(ANNUAL, raw_root)
    assert frame["mean"].iloc[0] == value
    assert frame["site_id"].dtype == np.int64
    assert frame["year"].dtype == np.int64


def test_read_raw_keeps_the_quality_flag_a_string(raw_root):
    _write_raw(raw_root, DATED, DATED_ROWS)
    frame = read_raw(DATED, raw_root)
    assert frame["qc"].tolist() == ["000", "001", "000", "000"]
    assert frame["date"].dtype == object or str(frame["date"].dtype).startswith("str")


def test_read_raw_refuses_a_changed_header(raw_root):
    other = _spec(raw_columns=("site_id", "year", "agb", "sd"), value_column="agb")
    _write_raw(raw_root, other, [dict(site_id=1, year=2012, agb=1.0, sd=1.0)])
    with pytest.raises(ValueError, match="header"):
        read_raw(ANNUAL, raw_root)


def test_read_raw_names_the_missing_file(raw_root):
    with pytest.raises(FileNotFoundError, match="provenance"):
        read_raw(ANNUAL, raw_root)


# ── the refusals ──────────────────────────────────────────────────────────────


def _refused(spec, rows, raw_root, site_table, tmp_path, message):
    with pytest.raises((ingest.IngestError, ValueError, KeyError), match=message):
        _ingest(spec, rows, raw_root, site_table, tmp_path / "out")


def test_a_duplicate_key_is_refused(raw_root, site_table, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=1, year=2012, mean=99.0, sd=1.0)]
    _refused(ANNUAL, rows, raw_root, site_table, tmp_path, "repeat")


def test_a_site_outside_the_pool_is_refused(raw_root, site_table, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=9, year=2012, mean=1.0, sd=1.0)]
    _refused(ANNUAL, rows, raw_root, site_table, tmp_path, r"site\(s\) \[9\] are not in the site table")


def test_a_site_id_below_one_is_refused(raw_root, site_table, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=0, year=2012, mean=1.0, sd=1.0)]
    _refused(ANNUAL, rows, raw_root, site_table, tmp_path, "must be integer site ids from 1 to")


def test_coordinates_disagreeing_with_the_site_table_are_refused(raw_root, site_table, tmp_path):
    rows = [dict(row) for row in DATED_ROWS]
    rows[0]["lat"] = 82.5  # site 1 of the other pool
    _refused(DATED, rows, raw_root, site_table, tmp_path, "disagrees with the site table")


def test_a_value_without_a_standard_deviation_is_refused(raw_root, site_table, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=3, year=2012, mean=5.0, sd=np.nan)]
    _refused(ANNUAL, rows, raw_root, site_table, tmp_path, "missing in different places")


def test_a_negative_standard_deviation_is_refused(raw_root, site_table, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=3, year=2012, mean=5.0, sd=-1.0)]
    _refused(ANNUAL, rows, raw_root, site_table, tmp_path, "negative")


def test_a_static_table_whose_copies_differ_is_refused(raw_root, site_table, tmp_path):
    rows = [dict(row) for row in STATIC_ROWS]
    rows[1]["soc"] = 743.0
    _refused(STATIC, rows, raw_root, site_table, tmp_path, "Either the source changed")
    frame = read_raw(STATIC, raw_root)
    with pytest.raises(ValueError, match="static"):
        build_constraint(STATIC, frame, site_table)


def test_a_quality_column_where_nothing_passes_is_refused(raw_root, site_table, tmp_path):
    rows = [dict(row, qc="001") for row in DATED_ROWS]
    _refused(DATED, rows, raw_root, site_table, tmp_path, "no row has")


def test_a_malformed_date_is_refused(raw_root, site_table, tmp_path):
    rows = [dict(row) for row in DATED_ROWS]
    rows[0]["date"] = "2012-13-01"
    _refused(DATED, rows, raw_root, site_table, tmp_path, "ISO date")


def test_build_constraint_refuses_a_duplicate_site_time_key_itself(site_table):
    frame = pd.DataFrame(ANNUAL_ROWS + [dict(site_id=1, year=2012, mean=99.0, sd=1.0)])
    with pytest.raises(ValueError, match="share a"):
        build_constraint(ANNUAL, frame, site_table)


# ── the readers ───────────────────────────────────────────────────────────────


def test_constraint_fields_and_sds_select_sites_in_the_order_given(
    raw_root, site_table, tmp_path, monkeypatch
):
    out_dir = tmp_path / "out"
    _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, out_dir)
    # The registry does not know the synthetic spec, so resolve it for the test.
    monkeypatch.setattr(module, "CONSTRAINTS", (ANNUAL,))
    monkeypatch.setattr(module, "CONSTRAINT_NAMES", (ANNUAL.name,))

    fields = constraint_fields(sites=[4, 1], directory=out_dir)
    field = fields[ANNUAL.name]
    assert field.name == ANNUAL.name
    assert field["site"].values.tolist() == [4, 1]
    assert field.attrs["units"] == "Mg ha-1"
    assert field.sel(site=1).values.tolist() == [10.0, 11.0]

    sds = constraint_standard_deviations([ANNUAL.name], sites=[1], directory=out_dir)[ANNUAL.name]
    assert sds.values.tolist() == [[2.0, 0.0]]
    assert "standard deviation" in sds.attrs["long_name"]

    with pytest.raises(KeyError, match=r"site\(s\) \[7\] are not in the processed file"):
        constraint_fields(sites=[1, 7], directory=out_dir)


def test_constraint_fields_refuses_one_string_of_sites_rather_than_reading_its_characters(
    raw_root, site_table, tmp_path, monkeypatch
):
    """``sites="14"`` once read as sites 1 and 4, one character per site."""
    out_dir = tmp_path / "out"
    _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, out_dir)
    monkeypatch.setattr(module, "CONSTRAINTS", (ANNUAL,))
    monkeypatch.setattr(module, "CONSTRAINT_NAMES", (ANNUAL.name,))

    with pytest.raises(TypeError, match="one string '14'"):
        constraint_fields(sites="14", directory=out_dir)
    with pytest.raises(TypeError, match="one string '14'"):
        constraint_standard_deviations(sites="14", directory=out_dir)
    with pytest.raises(ValueError, match="more than once"):
        constraint_fields(sites=[1, 1], directory=out_dir)
    with pytest.raises(TypeError, match="sequence of site ids"):
        constraint_fields(sites=4, directory=out_dir)
    with pytest.raises(TypeError, match="no order to keep"):
        constraint_fields(sites={4, 1}, directory=out_dir)
    with pytest.raises(TypeError, match="one string"):
        constraint_fields(ANNUAL.name, directory=out_dir)


def test_constraint_fields_takes_sites_as_any_array_like(raw_root, site_table, tmp_path, monkeypatch):
    """A field's own site coordinate, or a JAX array, is a sequence of site ids."""
    import jax.numpy as jnp

    out_dir = tmp_path / "out"
    _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, out_dir)
    monkeypatch.setattr(module, "CONSTRAINTS", (ANNUAL,))
    monkeypatch.setattr(module, "CONSTRAINT_NAMES", (ANNUAL.name,))

    field = constraint_fields(sites=[4, 1], directory=out_dir)[ANNUAL.name]
    for given in (field["site"], jnp.array([4, 1]), np.array([4, 1]), {4: 0, 1: 0}.keys()):
        again = constraint_fields(sites=given, directory=out_dir)[ANNUAL.name]
        assert again["site"].values.tolist() == [4, 1]


def test_a_missing_processed_file_names_the_command_that_builds_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest_constraints.py --constraint"):
        load_constraint("smap_soil_moisture", tmp_path / "absent.nc")


def test_the_loader_refuses_a_file_written_for_another_constraint(raw_root, site_table, tmp_path):
    out_dir = tmp_path / "out"
    _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, out_dir)
    other = _spec(name="test_other", raw_file="test_other.csv.gz")
    with pytest.raises(ValueError, match="written for constraint"):
        load_constraint(other, constraint_path(ANNUAL, out_dir))



# ── the loader's checks ───────────────────────────────────────────────────────


def _perturbations():
    def drop_sd(ds):
        return ds.drop_vars(STANDARD_DEVIATION)

    def wrong_dims(ds):
        return ds.assign({VALUE: ds[VALUE].isel(time=0, drop=True)})

    def wrong_units(ds):
        ds[VALUE].attrs["units"] = "kg m-2"
        return ds

    def drop_lon(ds):
        return ds.drop_vars("lon")

    def lon_on_time(ds):
        return ds.drop_vars("lon").assign_coords(lon=("time", np.zeros(ds.sizes["time"])))

    def drop_bounds(ds):
        return ds.drop_vars("time_bounds")

    def reversed_site(ds):
        return ds.isel(site=slice(None, None, -1))

    def reversed_time(ds):
        return ds.isel(time=slice(None, None, -1))

    def orphan_sd(ds):
        sd = ds[STANDARD_DEVIATION].values.copy()
        sd[0, 0] = np.nan
        return ds.assign({STANDARD_DEVIATION: (ds[VALUE].dims, sd, ds[STANDARD_DEVIATION].attrs)})

    return [
        pytest.param(drop_sd, "missing data variables", id="missing-variable"),
        pytest.param(wrong_dims, "has dims", id="wrong-dims"),
        pytest.param(wrong_units, "has units", id="wrong-units"),
        pytest.param(drop_lon, "missing the 'lon'", id="missing-lon"),
        pytest.param(lon_on_time, "must be on site", id="lon-off-site"),
        pytest.param(drop_bounds, "time_bounds absent", id="missing-bounds"),
        pytest.param(reversed_site, "site is empty or not strictly ascending", id="site-order"),
        pytest.param(reversed_time, "time is empty or not strictly ascending", id="time-order"),
        pytest.param(orphan_sd, "missing at different elements", id="nan-mismatch"),
    ]


@pytest.mark.parametrize("perturb, message", _perturbations())
def test_the_loader_refuses_a_processed_file_that_departs_from_the_data_model(
    raw_root, site_table, tmp_path, perturb, message
):
    processed = _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, tmp_path / "out").load()
    broken = perturb(processed.copy(deep=True))
    path = tmp_path / "broken.nc"
    broken.to_netcdf(path, engine="h5netcdf", encoding={"time": {"units": module.TIME_UNITS}})
    with pytest.raises(ValueError, match=message):
        load_constraint(ANNUAL, path)


def test_a_dated_processed_file_with_bounds_is_refused(raw_root, site_table, tmp_path):
    processed = _ingest(DATED, DATED_ROWS, raw_root, site_table, tmp_path / "out").load()
    bounds = np.stack([processed["time"].values, processed["time"].values], axis=1)
    path = tmp_path / "broken.nc"
    processed.assign_coords(time_bounds=(("time", "bounds"), bounds)).to_netcdf(path)
    with pytest.raises(ValueError, match="time_bounds present"):
        load_constraint(DATED, path)


def test_a_failed_constraint_round_trip_keeps_the_partial_and_never_writes_the_processed_file(
    raw_root, site_table, tmp_path, monkeypatch, capsys
):
    """The .partial design: a check that fails after the write must not rename."""
    out_dir = tmp_path / "out"

    def read_back_differently(spec, path=None):
        dataset = load_constraint(spec, path)
        dataset.attrs["rows_read"] = -1
        return dataset

    monkeypatch.setattr(ingest, "load_constraint", read_back_differently)
    with pytest.raises(ingest.IngestError, match="does not read back identical"):
        _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, out_dir)
    assert not constraint_path(ANNUAL, out_dir).exists()
    assert (out_dir / f"{ANNUAL.name}.nc.partial").exists()
    assert f"{ANNUAL.name}.nc.partial" in capsys.readouterr().err


def test_a_time_label_that_is_not_a_whole_day_is_refused(raw_root, site_table, tmp_path, monkeypatch):
    """xarray would silently switch the on-disk units to hours."""
    original = module.build_constraint

    def shift_labels(spec, frame, site_table):
        processed = original(spec, frame, site_table)
        return processed.assign_coords(time=processed["time"] + np.timedelta64(12, "h"))

    monkeypatch.setattr(ingest, "build_constraint", shift_labels)
    with pytest.warns(UserWarning, match="hours since"):
        with pytest.raises(ingest.IngestError, match="not a whole day"):
            _ingest(DATED, DATED_ROWS, raw_root, site_table, tmp_path / "out")


# ── the parser's guards ───────────────────────────────────────────────────────


@pytest.mark.parametrize("token", ["", "null", "NaN", "nan"])
def test_only_the_literal_na_is_missing(raw_root, token):
    path = _write_raw(raw_root, ANNUAL, [dict(site_id=1, year=2012, mean=1.0, sd=1.0)])
    with gzip.open(path, "rt") as handle:
        text = handle.read().replace("1,2012,1,1", f"1,2012,{token},1")
    with gzip.open(path, "wt") as handle:
        handle.write(text)
    with pytest.raises(ValueError, match="could not be parsed"):
        read_raw(ANNUAL, raw_root)


def test_a_header_only_file_is_refused(raw_root):
    _write_raw(raw_root, ANNUAL, [])
    with pytest.raises(ValueError, match="holds no rows"):
        read_raw(ANNUAL, raw_root)


def test_a_row_with_a_surplus_field_is_refused(raw_root):
    path = _write_raw(raw_root, ANNUAL, [dict(site_id=1, year=2012, mean=1.0, sd=1.0)])
    with gzip.open(path, "at") as handle:
        handle.write("2,2012,2,2,7\n")
    with pytest.raises(ValueError, match="could not be parsed"):
        read_raw(ANNUAL, raw_root)


def test_a_missing_year_names_the_file(raw_root):
    path = _write_raw(raw_root, ANNUAL, [dict(site_id=1, year=2012, mean=1.0, sd=1.0)])
    with gzip.open(path, "at") as handle:
        handle.write("2,NA,2,2\n")
    with pytest.raises(ValueError, match="test_annual.csv.gz: could not be parsed"):
        read_raw(ANNUAL, raw_root)


# ── the remaining refusals ────────────────────────────────────────────────────


def test_an_empty_date_is_refused(raw_root, site_table, tmp_path):
    rows = [dict(row) for row in DATED_ROWS]
    rows[0]["date"] = ""
    _refused(DATED, rows, raw_root, site_table, tmp_path, "not dates")


def test_a_year_outside_the_plausible_range_is_refused(raw_root, site_table, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=3, year=20120, mean=1.0, sd=1.0)]
    _refused(ANNUAL, rows, raw_root, site_table, tmp_path, "does not hold years")


def test_an_infinite_value_is_refused(raw_root, site_table, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=3, year=2012, mean=float("inf"), sd=1.0)]
    _refused(ANNUAL, rows, raw_root, site_table, tmp_path, "infinite")


def test_a_missing_coordinate_is_refused(raw_root, site_table, tmp_path):
    rows = [dict(row) for row in DATED_ROWS]
    rows[0]["lat"] = np.nan
    _refused(DATED, rows, raw_root, site_table, tmp_path, "missing or not finite")


def test_a_quality_column_of_nothing_but_na_is_refused_cleanly(raw_root, site_table, tmp_path):
    rows = [dict(row, qc=None) for row in DATED_ROWS]
    _refused(DATED, rows, raw_root, site_table, tmp_path, "no row has")


def test_a_file_with_no_observed_rows_leaves_the_existing_processed_file_alone(
    raw_root, site_table, tmp_path
):
    out_dir = tmp_path / "out"
    good = _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, out_dir)
    before = constraint_path(ANNUAL, out_dir).stat().st_mtime_ns
    rows = [dict(site_id=1, year=2012, mean=np.nan, sd=np.nan)]
    _refused(ANNUAL, rows, raw_root, site_table, tmp_path, "no row carries an observed value")
    assert constraint_path(ANNUAL, out_dir).stat().st_mtime_ns == before
    with load_constraint(ANNUAL, constraint_path(ANNUAL, out_dir)) as kept:
        assert kept.load().identical(good)


def test_the_report_counts_what_was_written(raw_root, site_table, tmp_path):
    out_dir = tmp_path / "out"
    processed = _ingest(DATED, DATED_ROWS, raw_root, site_table, out_dir)
    report = ingest.describe_processed_file(processed, constraint_path(DATED, out_dir))
    assert "observed 3 of 12 elements" in report
    assert "dropped by quality flag 1" in report
    assert "standard deviations of zero: 1" in report
    assert "2012-07-11 .. 2013-07-16" in report


# ── the command line ──────────────────────────────────────────────────────────


def test_describe_exits_zero_without_touching_data(capsys):
    assert ingest.main(["--describe", "--raw-root", "/nonexistent"]) == 0
    assert "modis_leaf_area_index.csv.gz" in capsys.readouterr().out


def test_a_missing_raw_root_is_a_reported_error_not_a_traceback(tmp_path, capsys):
    site_table_path = _write_sites(tmp_path / "sites" / "sites.csv")
    code = ingest.main(
        ["--raw-root", str(tmp_path / "absent"), "--site-table", str(site_table_path),
         "--out-dir", str(tmp_path / "out"), "--constraint", "smap_soil_moisture"]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ") and "provenance" in captured.err
    assert captured.out == ""


def test_a_successful_run_exits_zero_and_reports(raw_root, tmp_path, monkeypatch, capsys):
    site_table_path = _write_sites(tmp_path / "sites" / "sites.csv")
    _write_raw(raw_root, ANNUAL, ANNUAL_ROWS)
    monkeypatch.setattr(module, "CONSTRAINTS", (ANNUAL,))
    monkeypatch.setattr(module, "CONSTRAINT_NAMES", (ANNUAL.name,))
    monkeypatch.setattr(ingest, "CONSTRAINT_NAMES", (ANNUAL.name,))
    code = ingest.main(
        ["--raw-root", str(raw_root), "--site-table", str(site_table_path), "--out-dir", str(tmp_path / "out")]
    )
    assert code == 0
    assert "observed 4 of 8 elements" in capsys.readouterr().out
    assert constraint_path(ANNUAL, tmp_path / "out").exists()


def test_constraint_fields_keeps_the_order_of_names_given(
    raw_root, site_table, tmp_path, monkeypatch
):
    out_dir = tmp_path / "out"
    other = _spec(name="test_other", raw_file="test_other.csv.gz")
    _ingest(ANNUAL, ANNUAL_ROWS, raw_root, site_table, out_dir)
    _ingest(other, ANNUAL_ROWS[:2], raw_root, site_table, out_dir)
    monkeypatch.setattr(module, "CONSTRAINTS", (ANNUAL, other))
    monkeypatch.setattr(module, "CONSTRAINT_NAMES", (ANNUAL.name, other.name))

    fields = constraint_fields([other.name, ANNUAL.name], sites=[4], directory=out_dir)
    assert list(fields) == [other.name, ANNUAL.name]
    assert fields[other.name]["site"].values.tolist() == [4]
    assert list(constraint_fields(directory=out_dir)) == [ANNUAL.name, other.name]
    assert list(constraint_fields([ANNUAL.name], directory=out_dir)) == [ANNUAL.name]


# ── the real files ────────────────────────────────────────────────────────────


def _real_files_present() -> bool:
    return default_sites_path().exists() and all(
        (RAW_DIR / spec.raw_file).exists() for spec in CONSTRAINTS
    )


needs_real_files = pytest.mark.skipif(
    not _real_files_present(), reason="raw constraint files or site table absent"
)


@pytest.mark.parametrize("spec", CONSTRAINTS, ids=lambda spec: spec.name)
@needs_real_files
def test_each_real_file_has_the_header_its_spec_declares(spec):
    with gzip.open(RAW_DIR / spec.raw_file, "rt") as handle:
        header = handle.readline().rstrip("\n").replace('"', "")
    assert tuple(header.split(",")) == spec.raw_columns


@pytest.fixture(scope="session")
def real_processed_constraints(tmp_path_factory) -> dict[str, xr.Dataset]:
    """Every constraint built from the real files into a temporary directory."""
    if not _real_files_present():
        pytest.skip("raw constraint files or site table absent")
    out_dir = tmp_path_factory.mktemp("constraints")
    site_table = load_sites()
    return {
        spec.name: ingest.ingest(spec, RAW_DIR, site_table, out_dir).load() for spec in CONSTRAINTS
    }


@pytest.mark.slow
@needs_real_files
def test_every_real_constraint_is_dense_over_the_pool(real_processed_constraints):
    for name, processed in real_processed_constraints.items():
        assert processed.sizes["site"] == N_SITES, name
        observed = np.isfinite(processed[VALUE].values)
        assert observed.any(), name
        assert np.array_equal(observed, np.isfinite(processed[STANDARD_DEVIATION].values)), name
        assert processed.attrs["constraint"] == name


@pytest.mark.slow
@needs_real_files
def test_the_modis_constraint_has_no_flagged_rows_left(real_processed_constraints):
    processed = real_processed_constraints["modis_leaf_area_index"]
    assert processed.attrs["rows_dropped_by_quality_flag"] > 0
    sd = processed[STANDARD_DEVIATION].values
    assert np.nanmax(sd) <= 20.0
    assert np.nanmin(processed[VALUE].values) > 0.0


# ── reproduction of the assembled file ────────────────────────────────────────


@pytest.fixture(scope="session")
def assembled() -> xr.Dataset:
    """The obsolete assembled file, as the reference the new ones must match."""
    if not ASSEMBLED.exists():
        pytest.skip("the assembled constraints_annual.nc is not in this working copy")
    with xr.open_dataset(ASSEMBLED, engine="h5netcdf") as dataset:
        return dataset.load()


def _assembled_records(assembled: xr.Dataset, old_name: str) -> tuple[pd.DataFrame, pd.Series]:
    mean = assembled["observation_mean"].sel(variable=old_name).to_series().dropna()
    variance = assembled["observation_variance"].sel(variable=old_name).to_series()
    frame = mean.rename("mean").reset_index()
    frame["year"] = frame["time"].dt.year
    return frame, variance


def _observed_records(processed: xr.Dataset) -> pd.DataFrame:
    frame = processed[VALUE].to_series().dropna().rename("value").reset_index()
    frame[STANDARD_DEVIATION] = processed[STANDARD_DEVIATION].to_series().loc[
        pd.MultiIndex.from_frame(frame[list(processed[VALUE].dims)])
    ].to_numpy()
    if "time" in frame:
        frame["year"] = frame["time"].dt.year
    return frame


@pytest.mark.slow
@needs_real_files
@pytest.mark.parametrize(
    "old_name, scale",
    [("aboveground_wood_carbon", 1.0), ("soil_moisture_percent", 1.0)],
)
def test_annual_and_snapshot_constraints_reproduce_the_assembled_values(
    real_processed_constraints, assembled, old_name, scale
):
    reference, variance = _assembled_records(assembled, old_name)
    ours = _observed_records(real_processed_constraints[ASSEMBLED_NAMES[old_name]])
    merged = reference.merge(ours, on=["site", "year"], how="left", validate="one_to_one")
    assert merged["value"].notna().all(), "an assembled observation has no counterpart"
    assert len(ours) == len(reference), "the processed file carries observations the assembler did not"
    np.testing.assert_array_equal(merged["value"] * scale, merged["mean"])
    expected_sd = np.sqrt(
        variance.loc[list(zip(merged["site"], merged["time_x"], strict=True))].to_numpy()
    )
    np.testing.assert_allclose(merged[STANDARD_DEVIATION] * scale, expected_sd, rtol=0, atol=1e-9)


@pytest.mark.slow
@needs_real_files
def test_the_static_soil_carbon_is_the_assembled_value_times_ten(real_processed_constraints, assembled):
    reference, variance = _assembled_records(assembled, "total_soil_carbon")
    processed = real_processed_constraints["soilgrids_soil_organic_carbon"]
    per_site = reference.groupby("site")["mean"].agg(["nunique", "first"])
    assert (per_site["nunique"] == 1).all(), "the assembled values were not constant in time"
    ours = processed[VALUE].to_series().dropna()
    assert sorted(ours.index) == sorted(per_site.index)
    np.testing.assert_allclose(ours.loc[per_site.index] / 10.0, per_site["first"], rtol=0, atol=1e-12)


@pytest.mark.slow
@needs_real_files
def test_the_lai_selection_rule_reproduces_the_assembled_file(real_processed_constraints, assembled):
    """Nearest passing composite to July 15 within 30 days, earlier date on a tie.

    The assembler's rule, reverse-engineered and verified on 2026-09-17: it
    reproduces every assembled mean, and ``max(sd, 0.66)`` every assembled
    variance. The later-date tie-break does not. The rule lives here and in
    the observation layer, never in the processed file.
    """
    processed = real_processed_constraints["modis_leaf_area_index"]
    records = _observed_records(processed)
    records = records[records["year"] >= 2012]
    key = pd.to_datetime(records["year"].astype(str) + "-07-15")
    records = records.assign(distance=(records["time"] - key).dt.days.abs())
    records = records[records["distance"] <= 30]
    selected = (
        records.sort_values(["site", "year", "distance", "time"])
        .drop_duplicates(["site", "year"])
        .set_index(["site", "year"])
    )

    reference, variance = _assembled_records(assembled, "lai")
    reference = reference.set_index(["site", "year"])
    assert set(selected.index) == set(reference.index)
    aligned = selected.loc[reference.index]
    np.testing.assert_array_equal(aligned["value"].to_numpy(), reference["mean"].to_numpy())
    expected_sd = np.sqrt(
        variance.loc[list(zip(reference.index.get_level_values("site"), reference["time"], strict=True))]
        .to_numpy()
    )
    np.testing.assert_allclose(
        np.maximum(aligned[STANDARD_DEVIATION].to_numpy(), 0.66), expected_sd, rtol=0, atol=1e-9
    )
