"""Tests for the constraint specs, the products they describe, and the ingest.

Three layers. The specs are checked for internal consistency and against the
real raw files' headers. The conversion is exercised on small synthetic raw
tables, one per time structure, where the expected product can be written out
by hand and every refusal can be provoked. Finally the real files are built
and the results compared against the assembled product the reanalysis used,
``processed/constraints_annual.nc``, which the new products must reproduce
exactly under the assembler's own rules: those rules -- the July 15 selection
for LAI with its 30-day window and earlier-date tie-break, the 0.66 floor on
the LAI standard deviation, the factor of ten on soil carbon -- live in this
file and nowhere in the products.

The real-data cases skip when the raw files, the site table or the assembled
product are absent from the working copy.
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sipnet_calibration import constraints as module
from sipnet_calibration.constraints import (
    CF_CONVENTIONS,
    CONSTRAINT_NAMES,
    CONSTRAINTS,
    STANDARD_DEVIATION,
    VALUE,
    ConstraintSpec,
    TimeStructure,
    build_constraint,
    constraint_fields,
    constraint_path,
    constraint_sds,
    describe,
    load_constraint,
    netcdf_encoding,
    read_raw,
    resolve_constraint,
)
from sipnet_calibration.sites import SITE_COLUMNS, default_sites_path, load_sites

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "constraints"
ASSEMBLED = REPO_ROOT / "data" / "processed" / "constraints_annual.nc"

#: Old assembled name -> new constraint, for the reproduction tests.
ASSEMBLED_NAMES = {
    "aboveground_wood_carbon": "landtrendr_aboveground_biomass",
    "lai": "modis_leaf_area_index",
    "soil_moisture_percent": "smap_soil_moisture",
    "total_soil_carbon": "soilgrids_soil_organic_carbon",
}


def _load_ingest_module():
    """Import ``scripts/ingest_constraints.py``, which is a script."""
    path = REPO_ROOT / "scripts" / "ingest_constraints.py"
    spec = importlib.util.spec_from_file_location("ingest_constraints", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["ingest_constraints"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


ingest = _load_ingest_module()


# ── synthetic fixtures ────────────────────────────────────────────────────────

SYNTHETIC_SITES = [1, 2, 3, 4]
SYNTHETIC_COORDS = {1: (-100.0, 40.0), 2: (-101.0, 41.0), 3: (-102.0, 42.0), 4: (-103.0, 43.0)}


def _write_sites(path: Path, site_ids=SYNTHETIC_SITES) -> Path:
    """A minimal site table that ``load_sites`` accepts."""
    frame = pd.DataFrame(
        {
            "site_id": np.array(site_ids, dtype=np.int32),
            "lon": [SYNTHETIC_COORDS[site][0] for site in site_ids],
            "lat": [SYNTHETIC_COORDS[site][1] for site in site_ids],
            "lon_index": np.arange(len(site_ids), dtype=np.int32) + 1000,
            "lat_index": np.arange(len(site_ids), dtype=np.int32) + 2000,
            "site_name": [f"site {site}" for site in site_ids],
            "site_order": np.zeros(len(site_ids), dtype=np.int32),
            "cluster": np.ones(len(site_ids), dtype=np.int8),
            "landcover": np.ones(len(site_ids), dtype=np.int8),
            "ameriflux_site_id": [""] * len(site_ids),
        }
    )
    assert tuple(frame.columns) == SITE_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_raw(root: Path, spec: ConstraintSpec, rows: list[dict]) -> Path:
    """Write rows as the raw files are written: gzipped CSV, ``NA``, ``%.17g``."""
    lines = [",".join(spec.raw_columns)]
    for row in rows:
        cells = []
        for column in spec.raw_columns:
            cell = row[column]
            if cell is None or (isinstance(cell, float) and np.isnan(cell)):
                cells.append("NA")
            elif isinstance(cell, float):
                cells.append(f"{cell:.17g}")
            else:
                cells.append(str(cell))
        lines.append(",".join(cells))
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
        product="test",
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
def sites(tmp_path) -> pd.DataFrame:
    return load_sites(_write_sites(tmp_path / "sites" / "sites.csv"))


@pytest.fixture
def raw_root(tmp_path) -> Path:
    return tmp_path / "raw"


def _ingest(spec, rows, raw_root, sites, out_dir) -> xr.Dataset:
    _write_raw(raw_root, spec, rows)
    return ingest.ingest(spec, raw_root, sites, out_dir)


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


def test_describe_names_the_file_the_columns_and_the_filter():
    text = describe(resolve_constraint("modis_leaf_area_index"))
    assert "modis_leaf_area_index.csv.gz" in text
    assert "'lai'" in text and "'qc' == '000'" in text
    assert "dated" in text


# ── the conversion, on synthetic tables ───────────────────────────────────────


def test_an_annual_table_becomes_a_dense_product_with_calendar_year_bounds(
    raw_root, sites, tmp_path
):
    product = _ingest(ANNUAL, ANNUAL_ROWS, raw_root, sites, tmp_path / "out")

    assert product[VALUE].dims == ("site", "time")
    assert product["site"].values.tolist() == SYNTHETIC_SITES
    assert product["site"].dtype == np.int32
    assert np.array_equal(product["time"].values, _dates("2012-01-01", "2013-01-01"))
    assert product["time"].attrs["bounds"] == "time_bounds"
    assert product["time_bounds"].dims == ("time", "bounds")
    assert np.array_equal(product["time_bounds"].values[0], _dates("2012-01-01", "2013-01-01"))
    assert np.array_equal(product["time_bounds"].values[1], _dates("2013-01-01", "2014-01-01"))

    value = product[VALUE].values
    assert value[0].tolist() == [10.0, 11.0]  # site 1
    assert value[1, 0] == 0.1 and np.isnan(value[1, 1])  # site 2: 0.1 exact, then NA
    assert np.isnan(value[2]).all()  # site 3 never observed
    assert np.isnan(value[3, 0]) and value[3, 1] == 40.0
    assert product[STANDARD_DEVIATION].values[0, 1] == 0.0  # written through
    assert product["lon"].values.tolist() == [-100.0, -101.0, -102.0, -103.0]

    assert product.attrs["Conventions"] == CF_CONVENTIONS
    assert product.attrs["rows_read"] == 5
    assert product.attrs["rows_dropped_by_quality_flag"] == 0
    assert product[VALUE].attrs["units"] == "Mg ha-1"


def test_a_dated_table_drops_the_flagged_rows_and_counts_them(raw_root, sites, tmp_path):
    product = _ingest(DATED, DATED_ROWS, raw_root, sites, tmp_path / "out")

    assert product[VALUE].dims == ("site", "time")
    assert "time_bounds" not in product.coords
    assert "bounds" not in product["time"].attrs
    assert np.array_equal(
        product["time"].values, _dates("2012-07-11", "2012-07-15", "2013-07-16")
    )
    value = product[VALUE].values
    assert value[0].tolist()[:1] == [1.5] and np.isnan(value[0, 1])  # flagged row gone
    assert value[1, 1] == 2.5 and value[1, 2] == 2.7
    assert product.attrs["rows_dropped_by_quality_flag"] == 1
    assert product.attrs["rows_read"] == 4


def test_a_static_table_collapses_to_one_value_per_site(raw_root, sites, tmp_path):
    product = _ingest(STATIC, STATIC_ROWS, raw_root, sites, tmp_path / "out")

    assert product[VALUE].dims == ("site",)
    assert "time" not in product.coords
    assert product[VALUE].values[0] == 742.8590087890625
    assert product[STANDARD_DEVIATION].values[2] == 10.25
    assert np.isnan(product[VALUE].values[1]) and np.isnan(product[VALUE].values[3])
    assert "collapsed" in product.attrs["history"]


def test_the_written_file_reads_back_through_the_loader_identically(
    raw_root, sites, tmp_path
):
    out_dir = tmp_path / "out"
    built = _ingest(ANNUAL, ANNUAL_ROWS, raw_root, sites, out_dir)
    with load_constraint(ANNUAL, constraint_path(ANNUAL, out_dir)) as read:
        read = read.load()
    assert read.identical(built)
    assert not (out_dir / f"{ANNUAL.name}.nc.partial").exists()


def test_no_coordinate_is_encoded_with_a_fill_value(raw_root, sites, tmp_path):
    product = _ingest(ANNUAL, ANNUAL_ROWS, raw_root, sites, tmp_path / "out")
    encoding = netcdf_encoding(product)
    for coordinate in product.coords:
        assert encoding[str(coordinate)]["_FillValue"] is None
    assert np.isnan(encoding[VALUE]["_FillValue"])

    raw = xr.open_dataset(constraint_path(ANNUAL.name, tmp_path / "out"), decode_cf=False)
    with raw:
        assert raw.attrs["Conventions"] == CF_CONVENTIONS
        assert raw["time"].attrs["units"] == module.TIME_UNITS
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


def _refused(spec, rows, raw_root, sites, tmp_path, message):
    with pytest.raises((ingest.IngestError, ValueError), match=message):
        _ingest(spec, rows, raw_root, sites, tmp_path / "out")


def test_a_duplicate_key_is_refused(raw_root, sites, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=1, year=2012, mean=99.0, sd=1.0)]
    _refused(ANNUAL, rows, raw_root, sites, tmp_path, "repeat")


def test_a_site_outside_the_pool_is_refused(raw_root, sites, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=9, year=2012, mean=1.0, sd=1.0)]
    _refused(ANNUAL, rows, raw_root, sites, tmp_path, "not in the site table")


def test_a_site_id_below_one_is_refused(raw_root, sites, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=0, year=2012, mean=1.0, sd=1.0)]
    _refused(ANNUAL, rows, raw_root, sites, tmp_path, "site ids outside")


def test_coordinates_disagreeing_with_the_site_table_are_refused(raw_root, sites, tmp_path):
    rows = [dict(row) for row in DATED_ROWS]
    rows[0]["lat"] = 82.5  # site 1 of the other pool
    _refused(DATED, rows, raw_root, sites, tmp_path, "disagrees with the site table")


def test_a_value_without_a_standard_deviation_is_refused(raw_root, sites, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=3, year=2012, mean=5.0, sd=np.nan)]
    _refused(ANNUAL, rows, raw_root, sites, tmp_path, "missing in different places")


def test_a_negative_standard_deviation_is_refused(raw_root, sites, tmp_path):
    rows = ANNUAL_ROWS + [dict(site_id=3, year=2012, mean=5.0, sd=-1.0)]
    _refused(ANNUAL, rows, raw_root, sites, tmp_path, "negative")


def test_a_static_table_whose_copies_differ_is_refused(raw_root, sites, tmp_path):
    rows = [dict(row) for row in STATIC_ROWS]
    rows[1]["soc"] = 743.0
    _refused(STATIC, rows, raw_root, sites, tmp_path, "different values in different years")
    frame = read_raw(STATIC, raw_root)
    with pytest.raises(ValueError, match="static"):
        build_constraint(STATIC, frame, sites)


def test_a_quality_column_where_nothing_passes_is_refused(raw_root, sites, tmp_path):
    rows = [dict(row, qc="001") for row in DATED_ROWS]
    _refused(DATED, rows, raw_root, sites, tmp_path, "no row has")


def test_a_malformed_date_is_refused(raw_root, sites, tmp_path):
    rows = [dict(row) for row in DATED_ROWS]
    rows[0]["date"] = "2012-13-01"
    _refused(DATED, rows, raw_root, sites, tmp_path, "ISO date")


def test_build_constraint_refuses_a_duplicate_cell_itself(sites):
    frame = pd.DataFrame(ANNUAL_ROWS + [dict(site_id=1, year=2012, mean=99.0, sd=1.0)])
    with pytest.raises(ValueError, match="share a"):
        build_constraint(ANNUAL, frame, sites)


# ── the readers ───────────────────────────────────────────────────────────────


def test_constraint_fields_and_sds_select_sites_in_the_order_given(
    raw_root, sites, tmp_path, monkeypatch
):
    out_dir = tmp_path / "out"
    _ingest(ANNUAL, ANNUAL_ROWS, raw_root, sites, out_dir)
    # The registry does not know the synthetic spec, so resolve it for the test.
    monkeypatch.setattr(module, "CONSTRAINTS", (ANNUAL,))
    monkeypatch.setattr(module, "CONSTRAINT_NAMES", (ANNUAL.name,))

    fields = constraint_fields(sites=[4, 1], directory=out_dir)
    field = fields[ANNUAL.name]
    assert field.name == ANNUAL.name
    assert field["site"].values.tolist() == [4, 1]
    assert field.attrs["units"] == "Mg ha-1"
    assert field.sel(site=1).values.tolist() == [10.0, 11.0]

    sds = constraint_sds([ANNUAL.name], sites=[1], directory=out_dir)[ANNUAL.name]
    assert sds.values.tolist() == [[2.0, 0.0]]
    assert "standard deviation" in sds.attrs["long_name"]

    with pytest.raises(ValueError, match="not in the pool"):
        constraint_fields(sites=[1, 7], directory=out_dir)


def test_a_missing_product_names_the_command_that_builds_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest_constraints.py --constraint"):
        load_constraint("smap_soil_moisture", tmp_path / "absent.nc")


def test_the_loader_refuses_a_file_written_for_another_constraint(raw_root, sites, tmp_path):
    out_dir = tmp_path / "out"
    _ingest(ANNUAL, ANNUAL_ROWS, raw_root, sites, out_dir)
    other = _spec(name="test_other", raw_file="test_other.csv.gz")
    with pytest.raises(ValueError, match="written for constraint"):
        load_constraint(other, constraint_path(ANNUAL, out_dir))


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
def real_products(tmp_path_factory) -> dict[str, xr.Dataset]:
    """Every constraint built from the real files into a temporary directory."""
    if not _real_files_present():
        pytest.skip("raw constraint files or site table absent")
    out_dir = tmp_path_factory.mktemp("constraints")
    sites = load_sites()
    return {
        spec.name: ingest.ingest(spec, RAW_DIR, sites, out_dir).load() for spec in CONSTRAINTS
    }


@pytest.mark.slow
@needs_real_files
def test_every_real_product_is_dense_over_the_pool(real_products):
    for name, product in real_products.items():
        assert product.sizes["site"] == 8000, name
        observed = np.isfinite(product[VALUE].values)
        assert observed.any(), name
        assert np.array_equal(observed, np.isfinite(product[STANDARD_DEVIATION].values)), name
        assert product.attrs["constraint"] == name


@pytest.mark.slow
@needs_real_files
def test_the_modis_product_has_no_flagged_rows_left(real_products):
    product = real_products["modis_leaf_area_index"]
    assert product.attrs["rows_dropped_by_quality_flag"] > 0
    sd = product[STANDARD_DEVIATION].values
    assert np.nanmax(sd) <= 20.0
    assert np.nanmin(product[VALUE].values) > 0.0


# ── reproduction of the assembled product ─────────────────────────────────────


@pytest.fixture(scope="session")
def assembled() -> xr.Dataset:
    """The obsolete assembled product, as the reference the new ones must match."""
    if not ASSEMBLED.exists():
        pytest.skip("the assembled constraints_annual.nc is not in this working copy")
    with xr.open_dataset(ASSEMBLED, engine="h5netcdf") as dataset:
        return dataset.load()


def _assembled_cells(assembled: xr.Dataset, old_name: str) -> tuple[pd.DataFrame, pd.Series]:
    mean = assembled["observation_mean"].sel(variable=old_name).to_series().dropna()
    variance = assembled["observation_variance"].sel(variable=old_name).to_series()
    frame = mean.rename("mean").reset_index()
    frame["year"] = frame["time"].dt.year
    return frame, variance


def _product_cells(product: xr.Dataset) -> pd.DataFrame:
    frame = product[VALUE].to_series().dropna().rename("value").reset_index()
    frame[STANDARD_DEVIATION] = product[STANDARD_DEVIATION].to_series().loc[
        pd.MultiIndex.from_frame(frame[list(product[VALUE].dims)])
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
def test_annual_and_snapshot_products_reproduce_the_assembled_values(
    real_products, assembled, old_name, scale
):
    reference, variance = _assembled_cells(assembled, old_name)
    ours = _product_cells(real_products[ASSEMBLED_NAMES[old_name]])
    merged = reference.merge(ours, on=["site", "year"], how="left", validate="one_to_one")
    assert merged["value"].notna().all(), "an assembled cell has no counterpart"
    assert len(ours) == len(reference), "the product carries cells the assembler did not"
    np.testing.assert_array_equal(merged["value"] * scale, merged["mean"])
    expected_sd = np.sqrt(
        variance.loc[list(zip(merged["site"], merged["time_x"], strict=True))].to_numpy()
    )
    np.testing.assert_allclose(merged[STANDARD_DEVIATION] * scale, expected_sd, rtol=0, atol=1e-9)


@pytest.mark.slow
@needs_real_files
def test_the_static_soil_carbon_is_the_assembled_value_times_ten(real_products, assembled):
    reference, variance = _assembled_cells(assembled, "total_soil_carbon")
    product = real_products["soilgrids_soil_organic_carbon"]
    per_site = reference.groupby("site")["mean"].agg(["nunique", "first"])
    assert (per_site["nunique"] == 1).all(), "the assembled values were not constant in time"
    ours = product[VALUE].to_series().dropna()
    assert sorted(ours.index) == sorted(per_site.index)
    np.testing.assert_allclose(ours.loc[per_site.index] / 10.0, per_site["first"], rtol=0, atol=1e-12)


@pytest.mark.slow
@needs_real_files
def test_the_lai_selection_rule_reproduces_the_assembled_product(real_products, assembled):
    """Nearest passing composite to July 15 within 30 days, earlier date on a tie.

    The assembler's rule, reverse-engineered and verified on 2026-09-17: it
    reproduces every assembled mean, and ``max(sd, 0.66)`` every assembled
    variance. The later-date tie-break does not. The rule lives here and in
    the observation layer, never in the product.
    """
    product = real_products["modis_leaf_area_index"]
    cells = _product_cells(product)
    cells = cells[cells["year"] >= 2012]
    key = pd.to_datetime(cells["year"].astype(str) + "-07-15")
    cells = cells.assign(distance=(cells["time"] - key).dt.days.abs())
    cells = cells[cells["distance"] <= 30]
    selected = (
        cells.sort_values(["site", "year", "distance", "time"])
        .drop_duplicates(["site", "year"])
        .set_index(["site", "year"])
    )

    reference, variance = _assembled_cells(assembled, "lai")
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
