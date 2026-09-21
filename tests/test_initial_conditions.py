"""Tests for the initial condition specs, the file parser, the conversion and
the ingest.

Three layers. The specs are checked for internal consistency and against
pySIPNET's ``InitialConditions`` fields. The parser, the conversion and the
ingest are exercised on a small synthetic tree written in the producer's exact
format -- netCDF-3 classic, the ``[year]`` template on ``time``, the
``_FillValue`` triple on every variable -- where every refusal can be
provoked and the expected product written out by hand. Finally the three real
files in a local checkout and the tracked raw file are read, when present, and
the properties the ingest relies on are checked on them.

The real-data cases skip when the files are absent from the working copy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysipnet.parameters import InitialConditions
from scipy.io import netcdf_file

from sipnet_calibration import initial_conditions as module
from sipnet_calibration.initial_conditions import (
    INITIAL_CONDITION_NAMES,
    INITIAL_CONDITIONS,
    MEMBER,
    SITE,
    SOURCE_FILL_VALUE,
    SOURCE_LONG_NAMES,
    SOURCE_MEMBER,
    SOURCE_NAMES,
    SOURCE_TIME_LONG_NAME,
    SOURCE_TIME_UNITS,
    SOURCE_UNITS,
    InitialConditionSpec,
    SourceFile,
    build_initial_conditions,
    build_raw,
    describe,
    initial_condition_fields,
    load_initial_conditions,
    netcdf_encoding,
    raw_encoding,
    read_raw,
    read_source_file,
    resolve_initial_condition,
    site_member_from_file_name,
)
from sipnet_calibration.sites import SITE_COLUMNS, load_sites

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SOURCE_ROOT = REPO_ROOT / "data" / "raw" / "initial_conditions" / "files"
TRACKED_RAW = REPO_ROOT / "data" / "raw" / "initial_conditions" / module.RAW_FILE
SITES_CSV = REPO_ROOT / "data" / "processed" / "sites" / "sites.csv"


def _load_script(name: str):
    """Import ``scripts/<name>.py``, which is a script."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


convert = _load_script("convert_initial_conditions")
ingest = _load_script("ingest_initial_conditions")


# ── synthetic fixtures ────────────────────────────────────────────────────────

SYNTHETIC_SITES = [1, 2, 3]
SYNTHETIC_COORDS = {1: (-100.0, 40.0), 2: (-101.0, 41.0), 3: (-102.0, 42.0), 4: (-103.0, 43.0)}

#: Per site, per member, the producer's values. Site 1 has every variable
#: with a negative wood draw in member 2; site 2 lacks leaf and soil
#: moisture, so its wood equals its biomass; site 3 lacks soil moisture only.
SYNTHETIC_VALUES = {
    1: {
        1: {
            "AbvGrndWood": 0.5,
            "leaf_carbon_content": 0.125,
            "wood_carbon_content": 0.375,
            "soil_organic_carbon_content": 12.5,
            "SoilMoistFrac": 61.25,
        },
        2: {
            "AbvGrndWood": 0.25,
            "leaf_carbon_content": 0.75,
            "wood_carbon_content": -0.5,
            "soil_organic_carbon_content": 20.0,
            "SoilMoistFrac": 40.0,
        },
    },
    2: {
        1: {
            "AbvGrndWood": 3.0,
            "wood_carbon_content": 3.0,
            "soil_organic_carbon_content": 55.7,
        },
        2: {
            "AbvGrndWood": 2.5,
            "wood_carbon_content": 2.5,
            "soil_organic_carbon_content": 13.085454307591759,
        },
    },
    3: {
        1: {
            "AbvGrndWood": 1.0,
            "leaf_carbon_content": 0.25,
            "wood_carbon_content": 0.75,
            "soil_organic_carbon_content": 8.0,
        },
        2: {
            "AbvGrndWood": 1.5,
            "leaf_carbon_content": 0.5,
            "wood_carbon_content": 1.0,
            "soil_organic_carbon_content": 9.0,
        },
    },
}


def _write_source_file(
    root: Path,
    site: int,
    member: int,
    values: dict[str, float],
    *,
    directory: int | None = None,
    name: str | None = None,
    time_attrs: dict[str, str] | None = None,
    time_values: list[float] | None = None,
    extra_attrs: dict[str, dict] | None = None,
    attribute_overrides: dict[str, dict] | None = None,
    global_attrs: dict[str, str] | None = None,
) -> Path:
    """Write one file as ``pool_ic_list2netcdf`` does, with optional defects."""
    directory_path = root / str(directory if directory is not None else site)
    directory_path.mkdir(parents=True, exist_ok=True)
    path = directory_path / (name or f"IC_site_{site}_{member}.nc")
    handle = netcdf_file(str(path), "w", version=1)
    for key, value in (global_attrs or {}).items():
        setattr(handle, key, value)
    handle.createDimension("time", None)
    time = handle.createVariable("time", "f8", ("time",))
    time[:] = np.asarray(time_values if time_values is not None else [1.0])
    for key, value in (
        time_attrs
        if time_attrs is not None
        else {"units": SOURCE_TIME_UNITS, "long_name": SOURCE_TIME_LONG_NAME}
    ).items():
        setattr(time, key, value)
    for variable, value in values.items():
        array = handle.createVariable(variable, "f8", ("time",))
        array[:] = np.asarray([value] * len(time_values if time_values is not None else [1.0]))
        attrs = {
            "units": SOURCE_UNITS.get(variable, "kg C m-2"),
            "_FillValue": SOURCE_FILL_VALUE,
            "long_name": SOURCE_LONG_NAMES.get(variable, variable),
        }
        attrs.update((attribute_overrides or {}).get(variable, {}))
        attrs.update((extra_attrs or {}).get(variable, {}))
        for key, attribute in attrs.items():
            setattr(array, key, attribute)
    handle.close()
    return path


def _write_tree(root: Path, values=SYNTHETIC_VALUES) -> Path:
    for site, members in values.items():
        for member, record in members.items():
            _write_source_file(root, site, member, record)
    return root


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


def _records(values=SYNTHETIC_VALUES) -> list[SourceFile]:
    return [
        SourceFile(site=site, member=member, values=record)
        for site, members in values.items()
        for member, record in members.items()
    ]


@pytest.fixture
def tree(tmp_path) -> Path:
    return _write_tree(tmp_path / "files")


@pytest.fixture
def sites_csv(tmp_path) -> Path:
    return _write_sites(tmp_path / "sites.csv")


@pytest.fixture
def raw(tree, tmp_path) -> Path:
    """The synthetic tree converted through the script, as a path."""
    out = tmp_path / "raw" / module.RAW_FILE
    sites_csv = _write_sites(tmp_path / "sites.csv")
    assert convert.main(["--root", str(tree), "--out", str(out), "--sites", str(sites_csv), "--jobs", "1"]) == 0
    return out


# ── the specs ─────────────────────────────────────────────────────────────────


def test_specs_are_one_per_source_variable_with_distinct_names():
    assert len({spec.name for spec in INITIAL_CONDITIONS}) == len(INITIAL_CONDITIONS)
    assert {spec.source_name for spec in INITIAL_CONDITIONS} == set(SOURCE_NAMES)
    assert INITIAL_CONDITION_NAMES == tuple(spec.name for spec in INITIAL_CONDITIONS)


def test_specs_name_real_sipnet_initial_conditions():
    fields = set(InitialConditions.model_fields)
    used = {spec.sipnet_initial_condition for spec in INITIAL_CONDITIONS} - {""}
    assert used <= fields
    assert {"total_wood_carbon", "leaf_area_index", "soil_carbon", "soil_wetness_fraction"} == used


def test_spec_attributes_carry_the_source_strings():
    spec = resolve_initial_condition("initial_soil_moisture_saturation")
    attrs = spec.xarray_attributes()
    assert attrs["units"] == "percent"
    assert attrs["source_units"] == "(-)"
    assert attrs["source_long_name"] == "Average Layer Fraction of Saturation"
    assert attrs["sipnet_initial_condition"] == "soil_wetness_fraction"
    assert "constituent" not in attrs
    wood = resolve_initial_condition("initial_wood_carbon").xarray_attributes()
    assert wood["constituent"] == "C"
    assert "comment" in wood


def test_spec_refuses_a_bad_name_unit_source_or_sipnet_field():
    good = dict(
        name="initial_thing",
        source_name="AbvGrndWood",
        long_label="Thing",
        units="kg m-2",
        constituent="C",
        description="d",
        product="p",
        sipnet_initial_condition="",
        pecan_conversion="c",
        units_provenance="u",
    )
    InitialConditionSpec(**good)
    with pytest.raises(ValueError, match="lower_case"):
        InitialConditionSpec(**{**good, "name": "InitialThing"})
    with pytest.raises(ValueError):
        InitialConditionSpec(**{**good, "units": "kg C m-2"})
    with pytest.raises(ValueError, match="source_name"):
        InitialConditionSpec(**{**good, "source_name": "TotSoilCarb"})
    with pytest.raises(ValueError, match="InitialConditions"):
        InitialConditionSpec(**{**good, "sipnet_initial_condition": "plantWoodInit"})
    with pytest.raises(ValueError, match="pecan_conversion"):
        InitialConditionSpec(**{**good, "pecan_conversion": ""})


def test_resolve_and_describe():
    spec = resolve_initial_condition("initial_leaf_carbon")
    assert spec.source_name == "leaf_carbon_content"
    text = describe(spec)
    assert "initial_leaf_carbon" in text and "leaf_carbon_content" in text and "laiInit" in text
    with pytest.raises(KeyError, match="initial_soil_organic_carbon"):
        resolve_initial_condition("soil")


# ── the parser ────────────────────────────────────────────────────────────────


def test_file_name_parsing():
    assert site_member_from_file_name("IC_site_4102_100.nc") == (4102, 100)
    for bad in ("IC_site_01_5.nc", "IC_site_1_0.nc", "IC_site_1_5.nc.partial", "ERA5_1_1.clim", "IC_site_1.nc"):
        assert site_member_from_file_name(bad) is None


def test_read_source_file_returns_the_values_bit_for_bit(tree):
    record = read_source_file(tree / "2" / "IC_site_2_2.nc")
    assert record.site == 2 and record.member == 2
    assert record.values == SYNTHETIC_VALUES[2][2]
    assert record.values["soil_organic_carbon_content"] == 13.085454307591759


def test_read_source_file_refuses_a_directory_or_name_mismatch(tmp_path):
    root = tmp_path / "files"
    path = _write_source_file(root, 1, 1, SYNTHETIC_VALUES[2][1], directory=7)
    with pytest.raises(ValueError, match="directory"):
        read_source_file(path)
    path = _write_source_file(root, 1, 1, SYNTHETIC_VALUES[2][1], name="ic_1_1.nc")
    with pytest.raises(ValueError, match="IC_site"):
        read_source_file(path)
    with pytest.raises(FileNotFoundError):
        read_source_file(root / "1" / "IC_site_1_99.nc")


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        (dict(time_attrs={"units": "days since 2011-01-01 00:00:00 UTC", "long_name": SOURCE_TIME_LONG_NAME}), "issue #3"),
        (dict(time_values=[1.0, 2.0]), "records"),
        (dict(global_attrs={"title": "x"}), "global attributes"),
        (dict(extra_attrs={"AbvGrndWood": {"scale_factor": 0.1}}), "scale_factor"),
        (dict(attribute_overrides={"AbvGrndWood": {"units": "Mg ha-1"}}), "attributes"),
    ],
)
def test_read_source_file_refuses_a_file_off_the_template(tmp_path, defect, message):
    path = _write_source_file(tmp_path / "files", 1, 1, SYNTHETIC_VALUES[2][1], **defect)
    with pytest.raises(ValueError, match=message):
        read_source_file(path)


def test_read_source_file_refuses_fills_unknown_variables_and_empty_files(tmp_path):
    root = tmp_path / "files"
    path = _write_source_file(root, 1, 1, {**SYNTHETIC_VALUES[2][1], "AbvGrndWood": SOURCE_FILL_VALUE})
    with pytest.raises(ValueError, match="fill value"):
        read_source_file(path)
    path = _write_source_file(root, 1, 2, {**SYNTHETIC_VALUES[2][1], "TotSoilCarb": 1.0})
    with pytest.raises(ValueError, match="TotSoilCarb"):
        read_source_file(path)
    path = _write_source_file(root, 1, 3, {})
    with pytest.raises(ValueError, match="no data variable"):
        read_source_file(path)
    path = _write_source_file(root, 1, 4, {**SYNTHETIC_VALUES[2][1], "AbvGrndWood": float("nan")})
    with pytest.raises(ValueError, match="non-finite"):
        read_source_file(path)


# ── build_raw and the conversion ──────────────────────────────────────────────


def test_build_raw_lays_values_on_site_member_with_nan_for_absent_variables():
    raw = build_raw(_records(), source_root="here", conversion_script="test")
    assert raw["AbvGrndWood"].dims == (SITE, MEMBER)
    assert raw[SITE].values.tolist() == SYNTHETIC_SITES
    assert raw[SITE].dtype == np.int32 and raw[MEMBER].dtype == np.int16
    assert raw[MEMBER].values.tolist() == [1, 2]
    leaf = raw["leaf_carbon_content"].values
    assert np.isnan(leaf[1]).all() and np.isfinite(leaf[[0, 2]]).all()
    assert raw["wood_carbon_content"].sel(site=1, member=2).item() == -0.5
    assert raw["SoilMoistFrac"].attrs["units"] == "(-)"
    assert raw.attrs["n_source_files"] == 6
    assert raw.attrs["source_time_units"] == SOURCE_TIME_UNITS


def test_build_raw_refuses_gaps_duplicates_and_mixed_presence():
    records = _records()
    with pytest.raises(ValueError, match="no file"):
        build_raw(records[:-1], source_root="", conversion_script="")
    with pytest.raises(ValueError, match="two files"):
        build_raw(records + [records[0]], source_root="", conversion_script="")
    mixed = {**SYNTHETIC_VALUES, 3: {1: SYNTHETIC_VALUES[3][1], 2: SYNTHETIC_VALUES[2][2]}}
    with pytest.raises(ValueError, match="property of the site"):
        build_raw(_records(mixed), source_root="", conversion_script="")
    with pytest.raises(ValueError, match="no source files"):
        build_raw([], source_root="", conversion_script="")


def test_conversion_script_writes_a_raw_file_that_reads_back(raw):
    with read_raw(raw) as dataset:
        assert set(dataset.data_vars) == set(SOURCE_NAMES)
        assert dataset["soil_organic_carbon_content"].sel(site=2, member=2).item() == 13.085454307591759
        assert dataset.attrs["n_source_files"] == 6
    assert not raw.with_suffix(".nc.partial").exists()


def test_conversion_script_refuses_strays_and_a_wrong_pool(tree, tmp_path, capsys):
    sites = _write_sites(tmp_path / "sites.csv")
    (tree / "notes.txt").write_text("x")
    assert convert.main(["--root", str(tree), "--out", str(tmp_path / "o.nc"), "--sites", str(sites), "--jobs", "1"]) == 1
    assert "not site directories" in capsys.readouterr().err
    (tree / "notes.txt").unlink()
    (tree / "1" / "README").write_text("x")
    assert convert.main(["--root", str(tree), "--out", str(tmp_path / "o.nc"), "--sites", str(sites), "--jobs", "1"]) == 1
    assert "IC_site" in capsys.readouterr().err
    (tree / "1" / "README").unlink()
    wrong = _write_sites(tmp_path / "wrong.csv", site_ids=[1, 2, 3, 4])
    assert convert.main(["--root", str(tree), "--out", str(tmp_path / "o.nc"), "--sites", str(wrong), "--jobs", "1"]) == 1
    assert "pool" in capsys.readouterr().err
    assert not (tmp_path / "o.nc").exists()


def test_read_raw_refuses_a_file_off_the_schema(raw, tmp_path):
    with read_raw(raw) as dataset:
        broken = dataset.load().copy()
    other = tmp_path / "other.nc"
    broken.drop_vars("SoilMoistFrac").to_netcdf(other, engine="h5netcdf")
    with pytest.raises(ValueError, match="variables"):
        read_raw(other)
    renamed = broken.copy()
    renamed["AbvGrndWood"].attrs["units"] = "Mg ha-1"
    renamed.to_netcdf(other, engine="h5netcdf", encoding=raw_encoding(renamed))
    with pytest.raises(ValueError, match="source string"):
        read_raw(other)
    with pytest.raises(FileNotFoundError, match="convert_initial_conditions"):
        read_raw(tmp_path / "missing.nc")


# ── the ingest ────────────────────────────────────────────────────────────────


def test_build_initial_conditions_is_the_data_model(raw, sites_csv):
    sites = load_sites(sites_csv)
    with read_raw(raw) as raw_dataset:
        product = build_initial_conditions(raw_dataset, sites)
    assert set(product.data_vars) == set(INITIAL_CONDITION_NAMES)
    for name in INITIAL_CONDITION_NAMES:
        assert product[name].dims == (MEMBER, SITE)
        assert product[name].dtype == np.float64
    assert product[MEMBER].values.tolist() == [0, 1]
    assert product[SOURCE_MEMBER].values.tolist() == [1, 2]
    assert product[SITE].values.tolist() == SYNTHETIC_SITES
    assert product["lon"].sel(site=2).item() == -101.0 and product["lat"].sel(site=3).item() == 42.0
    # values are the raw ones, transposed, negatives included
    assert product["initial_wood_carbon"].sel(member=1, site=1).item() == -0.5
    assert np.isnan(product["initial_leaf_carbon"].sel(site=2).values).all()
    assert product["initial_soil_moisture_saturation"].attrs["units"] == "percent"
    assert product.attrs["Conventions"] == module.CF_CONVENTIONS
    assert product.attrs["nominal_date"] == module.NOMINAL_DATE
    assert product.attrs["member_source"] == "ic"
    assert product.attrs["n_members"] == 2 and product.attrs["n_sites"] == 3


def test_build_initial_conditions_refuses_a_different_pool(raw, tmp_path):
    sites = load_sites(_write_sites(tmp_path / "s.csv", site_ids=[1, 2]))
    with read_raw(raw) as raw_dataset, pytest.raises(ValueError, match="pool"):
        build_initial_conditions(raw_dataset, sites)


def test_ingest_script_round_trips_and_fields_select_sites(raw, sites_csv, tmp_path):
    out = tmp_path / "processed" / module.PRODUCT_FILE
    assert ingest.main(["--raw", str(raw), "--sites", str(sites_csv), "--out", str(out)]) == 0
    assert not out.with_suffix(".nc.partial").exists()
    with load_initial_conditions(out) as product, read_raw(raw) as raw_dataset:
        for spec in INITIAL_CONDITIONS:
            assert np.array_equal(
                product[spec.name].values, raw_dataset[spec.source_name].values.T, equal_nan=True
            )
            assert product[spec.name].attrs == spec.xarray_attributes()
    fields = initial_condition_fields(["initial_soil_organic_carbon"], sites=[3, 1], path=out)
    field = fields["initial_soil_organic_carbon"]
    assert field.dims == (MEMBER, SITE) and field[SITE].values.tolist() == [3, 1]
    assert "lon" in field.coords and field.attrs["units"] == "kg m-2"
    with pytest.raises(ValueError, match="not in the pool"):
        initial_condition_fields(sites=[9], path=out)
    with pytest.raises(KeyError):
        initial_condition_fields(["soil"], path=out)


def test_ingest_checks_refuse_a_broken_wood_identity_or_member_gap(raw, sites_csv):
    sites = load_sites(sites_csv)
    with read_raw(raw) as raw_dataset:
        dataset = raw_dataset.load().copy()
    ingest.check_raw(dataset, sites)
    broken = dataset.copy(deep=True)
    broken["wood_carbon_content"].values[0, 0] += 1e-12
    with pytest.raises(ingest.IngestError, match="identity"):
        ingest.check_wood_is_biomass_minus_leaf(broken)
    gapped = dataset.assign_coords(member=np.array([1, 3], dtype=np.int16))
    with pytest.raises(ingest.IngestError, match="1..2"):
        ingest.check_members_are_contiguous_from_one(gapped)


def test_load_refuses_a_product_off_the_data_model(raw, sites_csv, tmp_path):
    sites = load_sites(sites_csv)
    with read_raw(raw) as raw_dataset:
        product = build_initial_conditions(raw_dataset, sites)
    path = tmp_path / "p.nc"
    product.drop_vars("initial_leaf_carbon").to_netcdf(path, engine="h5netcdf")
    with pytest.raises(ValueError, match="variables"):
        load_initial_conditions(path)
    wrong = product.copy()
    wrong["initial_wood_carbon"].attrs["units"] = "g m-2"
    wrong.to_netcdf(path, engine="h5netcdf", encoding=netcdf_encoding(wrong))
    with pytest.raises(ValueError, match="units"):
        load_initial_conditions(path)
    with pytest.raises(FileNotFoundError, match="ingest_initial_conditions"):
        load_initial_conditions(tmp_path / "missing.nc")


# ── the real files ────────────────────────────────────────────────────────────


def test_local_source_files_parse_to_their_known_values():
    path = LOCAL_SOURCE_ROOT / "1" / "IC_site_1_1.nc"
    if not path.exists():
        pytest.skip("the producer's files are not in this working copy")
    record = read_source_file(path)
    assert record.site == 1 and record.member == 1
    assert set(record.values) == {"AbvGrndWood", "wood_carbon_content", "soil_organic_carbon_content"}
    assert record.values["soil_organic_carbon_content"] == 13.085454307591759
    assert record.values["AbvGrndWood"] == record.values["wood_carbon_content"]


@pytest.fixture(scope="module")
def tracked_raw() -> xr.Dataset:
    if not TRACKED_RAW.exists():
        pytest.skip("the tracked raw file is not in this working copy")
    with read_raw(TRACKED_RAW) as dataset:
        return dataset.load()


def test_tracked_raw_file_is_the_full_ensemble(tracked_raw):
    assert tracked_raw.sizes == {SITE: 8000, MEMBER: 100}
    assert tracked_raw[SITE].values.tolist() == list(range(1, 8001))
    assert tracked_raw[MEMBER].values.tolist() == list(range(1, 101))
    assert tracked_raw.attrs["n_source_files"] == 800000
    for name in ("AbvGrndWood", "wood_carbon_content", "soil_organic_carbon_content"):
        assert np.isfinite(tracked_raw[name].values).all()


def test_tracked_raw_file_holds_the_producer_identities(tracked_raw):
    biomass = tracked_raw["AbvGrndWood"].values
    wood = tracked_raw["wood_carbon_content"].values
    leaf = tracked_raw["leaf_carbon_content"].values
    has_leaf = np.isfinite(leaf)
    assert np.array_equal(wood[~has_leaf], biomass[~has_leaf])
    assert np.array_equal(wood[has_leaf], (biomass - leaf)[has_leaf])
    assert (wood < 0).any(), "the negative wood draws the specs describe are present"
    moisture = tracked_raw["SoilMoistFrac"].values
    assert np.nanmin(moisture) >= 0 and np.nanmax(moisture) <= 100
    assert np.nanmedian(moisture) > 1, "a percentage, not a fraction"


def test_tracked_raw_file_ingests_onto_the_site_pool(tracked_raw):
    if not SITES_CSV.exists():
        pytest.skip("the site table is not in this working copy")
    sites = load_sites(SITES_CSV)
    ingest.check_raw(tracked_raw, sites)
    product = build_initial_conditions(tracked_raw, sites)
    assert product["initial_soil_organic_carbon"].dims == (MEMBER, SITE)
    assert product[SOURCE_MEMBER].values[0] == 1 and product[MEMBER].values[0] == 0
