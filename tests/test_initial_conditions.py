"""Tests for the initial condition specs, the file parser, the conversion and
the ingest.

Three layers. The specs are checked for internal consistency and against
pySIPNET's ``InitialConditions`` fields. The parser, the conversion and the
ingest are exercised on a small synthetic tree written in PEcAn's exact
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
    CF_CONVENTIONS,
    CONVERTED_SIPNET_FIELDS,
    SOURCE,
    INITIAL_CONDITIONS,
    INITIAL_CONDITION_NAMES,
    InitialConditionSpec,
    MEMBER,
    SITE,
    SOURCE,
    SOURCE_MEMBER,
    SourceFile,
    SourceVariable,
    build_initial_conditions,
    build_raw,
    describe,
    initial_condition_fields,
    load_initial_conditions,
    netcdf_encoding,
    raw_encoding,
    read_raw,
    read_source_directory,
    read_source_file,
    resolve_initial_condition,
    site_member_from_file_name,
    to_sipnet_initial_conditions,
    to_sipnet_initial_conditions_table,
)
import sipnet_calibration
from sipnet_calibration import sites as sites_module
from sipnet_calibration.sites import SITE_COLUMNS, load_sites

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SOURCE_ROOT = REPO_ROOT / "data" / "raw" / "initial_conditions" / "files"
TRACKED_RAW = REPO_ROOT / "data" / "raw" / "initial_conditions" / module.RAW_FILE
SITES_CSV = REPO_ROOT / "data" / "processed" / "sites" / "sites.csv"


def _load_script(name: str, package: str = "scripts"):
    """Import a script by path, since scripts are not importable modules."""
    path = REPO_ROOT / package / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


convert = _load_script("convert_initial_conditions", "scripts/raw_sources")
ingest = _load_script("ingest_initial_conditions")


# ── synthetic fixtures ────────────────────────────────────────────────────────

SYNTHETIC_SITES = [1, 2, 3]
SYNTHETIC_COORDS = {1: (-100.0, 40.0), 2: (-101.0, 41.0), 3: (-102.0, 42.0), 4: (-103.0, 43.0)}

#: Per site, per member, the source values. Site 1 has every variable
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


def _source_attribute(variable, attribute, default):
    """The source string for *variable*, or *default* for a made-up variable.

    The synthetic files include variables the real format does not carry, so
    that the parser's refusals can be provoked.
    """
    known = SOURCE.variables.get(variable)
    return default if known is None else getattr(known, attribute)


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
        else {"units": SOURCE.time_units, "long_name": SOURCE.time_long_name}
    ).items():
        setattr(time, key, value)
    for variable, value in values.items():
        array = handle.createVariable(variable, "f8", ("time",))
        array[:] = np.asarray([value] * len(time_values if time_values is not None else [1.0]))
        attrs = {
            "units": _source_attribute(variable, "units", "kg C m-2"),
            "_FillValue": SOURCE.fill_value,
            "long_name": _source_attribute(variable, "long_name", variable),
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


# ── the source format, pinned to literals ─────────────────────────────────────


def test_source_format_is_pinned_to_literals():
    """The fixtures write files *from* SOURCE and the parser checks them
    *against* SOURCE, so only literals here can catch a change to it."""
    assert SOURCE.file_template == "{site}/IC_site_{site}_{member}.nc"
    assert SOURCE.fill_value == -999.0
    assert SOURCE.time_units == "days since [year]-01-01 00:00:00 UTC"
    assert SOURCE.time_long_name == "Time middle averaging period"
    assert SOURCE.time_value == 1.0
    assert SOURCE.names == (
        "AbvGrndWood",
        "wood_carbon_content",
        "leaf_carbon_content",
        "soil_organic_carbon_content",
        "SoilMoistFrac",
    )
    assert isinstance(SOURCE.names, tuple)
    # The specs carry the same variables in the same order, which is the order
    # build_raw lays the raw file out in.
    assert tuple(spec.source_name for spec in INITIAL_CONDITIONS) == SOURCE.names
    assert SOURCE.variables["SoilMoistFrac"].units == "(-)"
    assert SOURCE.variables["AbvGrndWood"].long_name == "Above ground woody biomass"


def test_source_format_cannot_be_mutated():
    """A spec reads SOURCE at attribute-access time, so a mutable mapping here
    would let an already-built spec start reporting different source units."""
    with pytest.raises(TypeError):
        SOURCE.variables["bogus"] = SourceVariable("bogus", "kg m-2", "Bogus")
    with pytest.raises(TypeError):
        del SOURCE.variables["AbvGrndWood"]
    assert hash(SOURCE) == hash(SOURCE)


# ── reading a site directory ──────────────────────────────────────────────────


def test_read_source_directory_reads_one_site_in_file_name_order(tree):
    records = read_source_directory(tree, 1)
    assert [(r.site, r.member) for r in records] == [(1, 1), (1, 2)]


def test_read_source_directory_skips_debris_and_refuses_strays(tree, tmp_path):
    (tree / "1" / ".DS_Store").write_bytes(b"debris")
    assert len(read_source_directory(tree, 1)) == 2

    (tree / "1" / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="holds nothing else"):
        read_source_directory(tree, 1)
    (tree / "1" / "notes.txt").unlink()

    with pytest.raises(ValueError, match="no such site directory"):
        read_source_directory(tree, 999)

    empty = tmp_path / "empty"
    (empty / "1").mkdir(parents=True)
    with pytest.raises(ValueError, match="holds no files"):
        read_source_directory(empty, 1)


def test_read_source_file_refuses_a_layered_variable_and_an_unreadable_file(tmp_path):
    """The per-variable dimension guard is what a layer-resolved upstream
    variable would trip; the file-level one catches a different shape."""
    root = tmp_path / "1"
    root.mkdir()
    path = root / "IC_site_1_1.nc"
    with netcdf_file(str(path), "w") as handle:
        handle.createDimension("time", None)
        handle.createDimension("layer", 2)
        time = handle.createVariable("time", "f8", ("time",))
        time[:] = [SOURCE.time_value]
        time.units, time.long_name = SOURCE.time_units, SOURCE.time_long_name
        layered = handle.createVariable("AbvGrndWood", "f8", ("time", "layer"))
        layered[:] = np.array([[1.0, 2.0]])
        layered.units = SOURCE.variables["AbvGrndWood"].units
        layered.long_name = SOURCE.variables["AbvGrndWood"].long_name
        layered._FillValue = SOURCE.fill_value
    with pytest.raises(ValueError, match=r"dimensions"):
        read_source_file(path)

    text = tmp_path / "2" / "IC_site_2_1.nc"
    text.parent.mkdir()
    text.write_text("this is not a netCDF file at all, but it is long enough to parse into")
    with pytest.raises(ValueError, match="not readable as netCDF-3 classic"):
        read_source_file(text)


def test_read_source_file_refuses_a_truncated_file(tree, tmp_path):
    """scipy raises IndexError from inside its own reader here, which is not a
    ValueError and would escape the conversion script without naming the file.
    """
    good = (tree / "1" / "IC_site_1_1.nc").read_bytes()
    path = tmp_path / "1" / "IC_site_1_1.nc"
    path.parent.mkdir(parents=True)
    path.write_bytes(good[: len(good) // 2])
    with pytest.raises(ValueError, match="not readable as netCDF-3 classic"):
        read_source_file(path)


def test_read_source_file_refuses_a_path_that_is_not_a_regular_file(tmp_path):
    directory = tmp_path / "1" / "IC_site_1_1.nc"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="not a regular file"):
        read_source_file(directory)

    dangling = tmp_path / "2" / "IC_site_2_1.nc"
    dangling.parent.mkdir()
    dangling.symlink_to(tmp_path / "nowhere.nc")
    with pytest.raises(ValueError, match="not a regular file"):
        read_source_file(dangling)


# ── the attribute contract, against literals rather than against the writer ──

VARIABLE_ATTRIBUTES = (
    "units", "long_name", "description", "product", "source_name", "source_units",
    "source_long_name", "sipnet_initial_condition", "pecan_conversion",
    "units_provenance",
)
PRODUCT_ATTRIBUTES = (
    "Conventions", "title", "product", "source_file", "source_root", "source_script",
    "source_script_note", "nominal_date", "nominal_date_provenance",
    "source_time_units", "source_time_long_name", "source_time_value",
    "member_source", "member_correspondence", "n_sites", "n_members", "history",
    "created",
)
RAW_ATTRIBUTES = (
    "title", "source_root", "source_layout", "source_format", "source_fill_value",
    "source_time_units", "source_time_long_name", "source_time_value",
    "n_source_files", "n_sites", "n_members", "conversion_script", "history",
    "converted",
)


def test_product_attributes_are_the_documented_set(raw, sites_csv):
    """Asserting the file against spec.xarray_attributes() compares the writer
    to itself; these literals are what the package docstring promises."""
    product = build_initial_conditions(read_raw(raw), load_sites(sites_csv))
    assert tuple(product.attrs) == PRODUCT_ATTRIBUTES
    for spec in INITIAL_CONDITIONS:
        expected = VARIABLE_ATTRIBUTES + (("constituent",) if spec.constituent else ())
        expected += ("comment",) if spec.comment else ()
        assert set(product[spec.name].attrs) == set(expected), spec.name
    assert set(product["lon"].attrs) == {"standard_name", "long_name", "units"}
    assert product["lon"].attrs["standard_name"] == "longitude"
    assert product["lat"].attrs["standard_name"] == "latitude"
    assert set(product[SITE].attrs) == {"long_name", "comment"}


def test_raw_attributes_are_the_documented_set(raw):
    raw = read_raw(raw)
    assert tuple(raw.attrs) == RAW_ATTRIBUTES
    assert raw.attrs["source_fill_value"] == -999.0
    assert raw.attrs["source_layout"] == "{site}/IC_site_{site}_{member}.nc"
    assert raw.attrs["n_sites"] == raw.sizes[SITE]
    assert raw.attrs["n_members"] == raw.sizes[MEMBER]
    for name in SOURCE.names:
        assert raw[name].attrs["source_fill_value"] == -999.0


def test_product_declares_the_shared_cf_version():
    assert CF_CONVENTIONS == "CF-1.11"


# ── the checks the readers apply on load ─────────────────────────────────────


def test_readers_refuse_a_variable_present_for_some_members_only(raw, sites_csv, tmp_path):
    """NaN means "no source file for this site carries it", so it cannot vary
    across a site's members. Both readers assert it, not just build_raw."""
    dataset = read_raw(raw)
    broken_raw = dataset.copy(deep=True)
    broken_raw["SoilMoistFrac"].values[0, 0] = np.nan
    path = tmp_path / "broken_raw.nc"
    broken_raw.to_netcdf(path, engine="h5netcdf", encoding=raw_encoding(broken_raw))
    with pytest.raises(ValueError, match="property of the site"):
        read_raw(path)

    product = build_initial_conditions(dataset, load_sites(sites_csv))
    product["initial_soil_moisture_saturation"].values[0, 0] = np.nan
    out = tmp_path / "product.nc"
    product.to_netcdf(out, engine="h5netcdf", encoding=netcdf_encoding(product))
    with pytest.raises(ValueError, match="property of the site"):
        load_initial_conditions(out)


def test_product_reader_refuses_infinities_and_misplaced_coordinates(raw, sites_csv, tmp_path):
    base = build_initial_conditions(read_raw(raw), load_sites(sites_csv))

    def refused(mutate, message):
        variant = mutate(base.copy(deep=True))
        path = tmp_path / "v.nc"
        variant.to_netcdf(path, engine="h5netcdf", encoding=netcdf_encoding(variant))
        with pytest.raises(ValueError, match=message):
            load_initial_conditions(path)
        path.unlink()

    def infinite(dataset):
        dataset["initial_wood_carbon"].values[:, 0] = np.inf
        return dataset

    refused(infinite, "infinite value")
    refused(lambda d: d.assign_coords(lon=(MEMBER, d["lon"].values[: d.sizes[MEMBER]])),
            "must be on site")
    refused(lambda d: d.assign_coords(source_member=(SITE, d[SITE].values.astype(np.int16))),
            "source_member must be on member")


def test_build_raw_refuses_member_ids_that_do_not_fit_int16():
    files = [
        SourceFile(site=1, member=1, values={"AbvGrndWood": 1.0}),
        SourceFile(site=1, member=40000, values={"AbvGrndWood": 2.0}),
    ]
    with pytest.raises(ValueError, match="int16"):
        build_raw(files, source_root="r", conversion_script="s")


# ── the default paths ─────────────────────────────────────────────────────────


def test_default_paths_sit_beside_the_package_not_inside_it(monkeypatch):
    """The data root is the repository's, whatever the package's shape.

    Derived here from ``sipnet_calibration.__file__`` rather than from the
    module under test, so that a module moving deeper into the package cannot
    move the data root with it and still agree with itself.
    """
    monkeypatch.delenv(sites_module.DATA_ROOT_ENV_VAR, raising=False)
    root = Path(sipnet_calibration.__file__).resolve().parents[2] / "data"

    assert module.default_product_path() == root / "processed" / module.PRODUCT_FILE
    assert module.default_raw_dir() == root / "raw" / "initial_conditions"
    assert module.raw_path() == root / "raw" / "initial_conditions" / module.RAW_FILE
    assert module.default_source_root() == root / "raw" / "initial_conditions" / "files"


def test_every_product_reads_the_same_data_root(monkeypatch, tmp_path):
    """The root lives in sipnet_calibration.conventions so that one setting
    moves all of them. Four modules used to spell it out separately, and the
    spelling broke here the moment a module moved a directory deeper."""
    from sipnet_calibration import constraints, conventions, drivers, sites

    monkeypatch.setenv(conventions.DATA_ROOT_ENV_VAR, str(tmp_path))
    assert conventions.data_root() == tmp_path
    assert module.default_raw_dir() == tmp_path / "raw" / "initial_conditions"
    assert constraints.default_raw_dir() == tmp_path / "raw" / "constraints"
    assert sites.default_sites_path().is_relative_to(tmp_path)
    assert drivers.default_drivers_root().is_relative_to(tmp_path)

    monkeypatch.delenv(conventions.DATA_ROOT_ENV_VAR)
    root = Path(sipnet_calibration.__file__).resolve().parents[2] / "data"
    for path in (
        module.default_raw_dir(),
        constraints.default_raw_dir(),
        sites.default_sites_path(),
        drivers.default_drivers_root(),
    ):
        assert path.is_relative_to(root), path

    # sites still exports the name it used to own.
    assert sites.DATA_ROOT_ENV_VAR == conventions.DATA_ROOT_ENV_VAR


def test_default_paths_follow_the_data_root_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(sites_module.DATA_ROOT_ENV_VAR, str(tmp_path))
    assert module.default_product_path() == tmp_path / "processed" / module.PRODUCT_FILE
    assert module.default_source_root() == tmp_path / "raw" / "initial_conditions" / "files"


# ── the specs ─────────────────────────────────────────────────────────────────


def test_specs_are_one_per_source_variable_with_distinct_names():
    assert len({spec.name for spec in INITIAL_CONDITIONS}) == len(INITIAL_CONDITIONS)
    assert {spec.source_name for spec in INITIAL_CONDITIONS} == set(SOURCE.names)
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
    with pytest.raises(ValueError, match="substance"):
        InitialConditionSpec(**{**good, "units": "kg C m-2"})
    with pytest.raises(ValueError, match="source_name"):
        InitialConditionSpec(**{**good, "source_name": "TotSoilCarb"})
    with pytest.raises(ValueError, match="InitialConditions"):
        InitialConditionSpec(**{**good, "sipnet_initial_condition": "plantWoodInit"})
    with pytest.raises(ValueError, match="pecan_conversion"):
        InitialConditionSpec(**{**good, "pecan_conversion": ""})
    for empty in ("description", "long_label", "product"):
        with pytest.raises(ValueError, match="description, long_label and product"):
            InitialConditionSpec(**{**good, empty: ""})


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
        (dict(time_attrs={"units": "days since 2011-01-01 00:00:00 UTC", "long_name": SOURCE.time_long_name}), "issue #3"),
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
    path = _write_source_file(root, 1, 1, {**SYNTHETIC_VALUES[2][1], "AbvGrndWood": SOURCE.fill_value})
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
    assert raw.attrs["source_time_units"] == SOURCE.time_units


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
        assert set(dataset.data_vars) == set(SOURCE.names)
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
    with pytest.raises(KeyError, match="No initial condition named"):
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


# ── the conversion to SIPNET parameters ───────────────────────────────────────

VALID_STATE = dict(
    initial_soil_organic_carbon=13.085,
    initial_wood_carbon=0.058,
    initial_leaf_carbon=0.121,
    initial_soil_moisture_saturation=60.0,
)
VALID_PARAMETERS = dict(
    leaf_carbon_per_area=32.0,
    fine_root_fraction=0.2,
    coarse_root_fraction=0.25,
    deciduous=False,
)


def ensemble_state(leaf=(0.12, 0.13)):
    """A two-member, two-site state with the product's units on every variable."""

    def field(name, values):
        spec = resolve_initial_condition(name)
        return xr.DataArray(
            np.asarray(values, dtype=float),
            dims=(MEMBER, SITE),
            coords={MEMBER: [0, 1], SITE: [1, 27]},
            attrs={"units": spec.units},
        )

    return {
        # The product carries this one too, and the conversion never reads it.
        "initial_aboveground_biomass_carbon": field(
            "initial_aboveground_biomass_carbon", [[0.17, 6.13], [0.19, 7.13]]
        ),
        "initial_soil_organic_carbon": field(
            "initial_soil_organic_carbon", [[13.0, 20.0], [14.0, 21.0]]
        ),
        "initial_wood_carbon": field("initial_wood_carbon", [[0.05, 6.0], [0.06, 7.0]]),
        "initial_leaf_carbon": field("initial_leaf_carbon", [list(leaf), list(leaf)]),
        "initial_soil_moisture_saturation": field(
            "initial_soil_moisture_saturation", [[60.0, 50.0], [61.0, 51.0]]
        ),
    }


def test_conversion_applies_the_pecan_formulas_and_round_trips():
    conditions = to_sipnet_initial_conditions(**VALID_STATE, **VALID_PARAMETERS)

    assert conditions.soil_carbon == pytest.approx(13085.0)
    assert conditions.soil_wetness_fraction == pytest.approx(0.6)
    assert conditions.total_wood_carbon == pytest.approx(105.45454545454545)
    assert conditions.leaf_area_index == pytest.approx(3.78125)

    # The root fractions are carried through, not recomputed, and not swapped:
    # SIPNET splits the wood pool with them and they are distinct parameters.
    assert conditions.fine_root_fraction == 0.2
    assert conditions.coarse_root_fraction == 0.25

    # Back to the state: SIPNET's own splits, run the other way.
    assert conditions.soil_carbon / 1000 == pytest.approx(
        VALID_STATE["initial_soil_organic_carbon"]
    )
    assert conditions.soil_wetness_fraction * 100 == pytest.approx(
        VALID_STATE["initial_soil_moisture_saturation"]
    )
    wood = conditions.total_wood_carbon * (
        1 - conditions.fine_root_fraction - conditions.coarse_root_fraction
    )
    assert wood / 1000 == pytest.approx(VALID_STATE["initial_wood_carbon"])
    leaf = conditions.leaf_area_index * VALID_PARAMETERS["leaf_carbon_per_area"]
    assert leaf / 1000 == pytest.approx(VALID_STATE["initial_leaf_carbon"])

    # The two pools the ensemble says nothing about keep pySIPNET's defaults.
    assert conditions.litter_carbon == 0.0 and conditions.snow_water_equivalent == 0.0


@pytest.mark.parametrize("name", sorted(VALID_STATE))
def test_conversion_takes_a_pool_of_zero(name):
    """Zero is physically valid -- a site with no leaves, no wood, dry soil --
    and the refusal is of negatives, not of the boundary."""
    conditions = to_sipnet_initial_conditions(
        **{**VALID_STATE, name: 0.0}, **VALID_PARAMETERS
    )
    assert conditions.model_dump()


def test_converted_fields_are_the_ones_the_specs_name():
    assert set(CONVERTED_SIPNET_FIELDS) <= set(InitialConditions.model_fields)
    named = {spec.sipnet_initial_condition for spec in INITIAL_CONDITIONS} - {""}
    assert named < set(CONVERTED_SIPNET_FIELDS)
    assert set(CONVERTED_SIPNET_FIELDS) - named == {"fine_root_fraction", "coarse_root_fraction"}
    # The order is the class's own, and the table's columns follow it.
    assert CONVERTED_SIPNET_FIELDS == (
        "total_wood_carbon",
        "leaf_area_index",
        "soil_carbon",
        "soil_wetness_fraction",
        "fine_root_fraction",
        "coarse_root_fraction",
    )
    assert list(CONVERTED_SIPNET_FIELDS) == [
        field for field in InitialConditions.model_fields if field in CONVERTED_SIPNET_FIELDS
    ]


def test_conversion_zeroes_the_lai_of_a_deciduous_pft():
    evergreen = to_sipnet_initial_conditions(**VALID_STATE, **VALID_PARAMETERS)
    deciduous = to_sipnet_initial_conditions(
        **VALID_STATE, **{**VALID_PARAMETERS, "deciduous": True}
    )
    assert evergreen.leaf_area_index > 0
    assert deciduous.leaf_area_index == 0.0
    # Nothing else moves: the rule is about the leaves only.
    assert deciduous.total_wood_carbon == evergreen.total_wood_carbon
    assert deciduous.soil_carbon == evergreen.soil_carbon
    assert deciduous.soil_wetness_fraction == evergreen.soil_wetness_fraction

    # A deciduous PFT never reads the leaf carbon, so the members where it is
    # absent or negative -- every grassland member -- still convert.
    for leaf in (np.nan, np.inf, -0.4):
        conditions = to_sipnet_initial_conditions(
            **{**VALID_STATE, "initial_leaf_carbon": leaf},
            **{**VALID_PARAMETERS, "deciduous": True},
        )
        assert conditions.leaf_area_index == 0.0
        with pytest.raises(ValueError, match="initial_leaf_carbon is negative"):
            to_sipnet_initial_conditions(
                **{**VALID_STATE, "initial_leaf_carbon": leaf}, **VALID_PARAMETERS
            )


@pytest.mark.parametrize(
    ("fine", "coarse", "message"),
    [
        (0.6, 0.4, "must be below 0.99"),
        (0.6, 0.5, "must be below 0.99"),
        (1.0, 0.0, "must be below 0.99"),
        (0.0, 1.0, "must be below 0.99"),
        (0.3, 0.8, "must be below 0.99"),
        # Finite but absurd: the remainder is 1e-16, so the aboveground pool is
        # multiplied by 1e16. Refusing only a sum of 1 or more let this through.
        (0.5, 0.5 - 1e-16, "must be below 0.99"),
        (0.5, 0.4949, "must be below 0.99"),
        (-0.1, 0.2, "fine_root_fraction is outside"),
        (0.2, 1.5, "coarse_root_fraction is outside"),
        (np.nan, 0.2, "fine_root_fraction is outside"),
    ],
)
def test_conversion_refuses_root_fractions_that_leave_no_wood(fine, coarse, message):
    with pytest.raises(ValueError, match=message):
        to_sipnet_initial_conditions(
            **VALID_STATE,
            **{**VALID_PARAMETERS, "fine_root_fraction": fine, "coarse_root_fraction": coarse},
        )
    # The guard is ours: pySIPNET takes the same pair without complaint, and the
    # run it produces exits 0 with a negative wood pool (TARPS-group/pySIPNET#39).
    if np.isfinite(fine) and np.isfinite(coarse) and 0 <= fine <= 1 and 0 <= coarse <= 1:
        InitialConditions(
            total_wood_carbon=100.0,
            leaf_area_index=1.0,
            soil_carbon=100.0,
            soil_wetness_fraction=0.5,
            fine_root_fraction=fine,
            coarse_root_fraction=coarse,
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
@pytest.mark.parametrize(
    "name",
    ["initial_soil_organic_carbon", "initial_wood_carbon", "initial_soil_moisture_saturation"],
)
def test_conversion_refuses_state_that_is_not_physical(name, bad):
    with pytest.raises(ValueError, match=f"{name} is negative, NaN or infinite"):
        to_sipnet_initial_conditions(**{**VALID_STATE, name: bad}, **VALID_PARAMETERS)


@pytest.mark.parametrize("bad", [0.0, -32.0, np.nan, 1e-9, 1e-7])
def test_conversion_refuses_a_leaf_carbon_per_area_below_sipnets_floor(bad):
    """SIPNET's setupModel raises leafCSpWt to TINY = 1e-6 without saying so,
    so a smaller value would be converted with one number and run with
    another: laiInit x leafCSpWt recovers a different initial leaf carbon."""
    with pytest.raises(ValueError, match="leaf_carbon_per_area is not finite and at least"):
        to_sipnet_initial_conditions(
            **VALID_STATE, **{**VALID_PARAMETERS, "leaf_carbon_per_area": bad}
        )
    # At the floor itself the conversion and the run agree, so it is accepted.
    conditions = to_sipnet_initial_conditions(
        **VALID_STATE, **{**VALID_PARAMETERS, "leaf_carbon_per_area": 1e-6}
    )
    assert conditions.leaf_area_index == pytest.approx(
        1000 * VALID_STATE["initial_leaf_carbon"] / 1e-6
    )


def test_conversion_refuses_soil_moisture_above_one_hundred():
    """The product is a percent of saturation over 0 to 100; dividing by 100 is
    what makes soilWFracInit a fraction."""
    with pytest.raises(ValueError, match="initial_soil_moisture_saturation is above 100"):
        to_sipnet_initial_conditions(
            **{**VALID_STATE, "initial_soil_moisture_saturation": 101.0}, **VALID_PARAMETERS
        )
    for edge in (0.0, 100.0):
        conditions = to_sipnet_initial_conditions(
            **{**VALID_STATE, "initial_soil_moisture_saturation": edge}, **VALID_PARAMETERS
        )
        assert conditions.soil_wetness_fraction == pytest.approx(edge / 100)


@pytest.mark.parametrize("bad", [1, 1.0, np.nan, "yes"])
def test_conversion_refuses_a_deciduous_flag_that_is_not_boolean(bad):
    with pytest.raises(TypeError, match="deciduous"):
        to_sipnet_initial_conditions(**VALID_STATE, **{**VALID_PARAMETERS, "deciduous": bad})


def test_conversion_table_is_the_single_member_form_cell_by_cell():
    state = ensemble_state()
    leaf_carbon_per_area = xr.DataArray([32.0, 40.0], dims=MEMBER, coords={MEMBER: [0, 1]})
    deciduous = xr.DataArray([False, True], dims=SITE, coords={SITE: [1, 27]})

    table = to_sipnet_initial_conditions_table(
        state,
        leaf_carbon_per_area=leaf_carbon_per_area,
        fine_root_fraction=0.2,
        coarse_root_fraction=0.25,
        deciduous=deciduous,
    )

    assert list(table.columns) == list(CONVERTED_SIPNET_FIELDS)
    assert table.index.names == [MEMBER, SITE]
    assert len(table) == 4
    for member in (0, 1):
        for site in (1, 27):
            one = to_sipnet_initial_conditions(
                **{
                    name: float(state[name].sel({MEMBER: member, SITE: site}))
                    for name in VALID_STATE
                },
                leaf_carbon_per_area=float(leaf_carbon_per_area.sel({MEMBER: member})),
                fine_root_fraction=0.2,
                coarse_root_fraction=0.25,
                deciduous=bool(deciduous.sel({SITE: site})),
            )
            row = table.loc[(member, site)]
            assert InitialConditions(**row) == one
    # The per-site PFT and the per-member parameter both landed where they belong.
    assert (table.loc[(slice(None), 27), "leaf_area_index"] == 0.0).all()
    assert table.loc[(0, 1), "leaf_area_index"] != table.loc[(1, 1), "leaf_area_index"]


def test_conversion_refuses_inputs_selected_for_different_members():
    """`.sel` leaves the member as a scalar coordinate, which alignment ignores.

    Without a check of its own, a state taken for one member and a parameter
    taken for another convert against each other and the table they produce has
    no member level left to notice it in.
    """
    state = ensemble_state()
    leaf_carbon_per_area = xr.DataArray([32.0, 40.0], dims=MEMBER, coords={MEMBER: [0, 1]})
    parameters = dict(fine_root_fraction=0.2, coarse_root_fraction=0.25, deciduous=False)

    with pytest.raises(ValueError, match="for member 0 and leaf_carbon_per_area for member 1"):
        to_sipnet_initial_conditions_table(
            {name: field.sel({MEMBER: 0}) for name, field in state.items()},
            leaf_carbon_per_area=leaf_carbon_per_area.sel({MEMBER: 1}),
            **parameters,
        )
    matched = to_sipnet_initial_conditions_table(
        {name: field.sel({MEMBER: 0}) for name, field in state.items()},
        leaf_carbon_per_area=leaf_carbon_per_area.sel({MEMBER: 0}),
        **parameters,
    )
    assert matched.index.tolist() == [1, 27]


def test_conversion_table_refuses_a_value_pysipnet_would_refuse():
    """The table never builds an InitialConditions, so it checks the products
    itself; otherwise a row could carry an inf the single-member form rejects."""
    state = ensemble_state()
    state["initial_soil_organic_carbon"][0, 0] = 1e308
    with np.errstate(over="ignore"):  # the overflow is the point; the check reports it
        with pytest.raises(ValueError, match=r"soil_carbon that is not finite.*\(0, 1\)"):
            to_sipnet_initial_conditions_table(
                state, leaf_carbon_per_area=32.0, fine_root_fraction=0.2,
                coarse_root_fraction=0.25, deciduous=False,
            )


def test_conversion_table_rows_are_member_then_site():
    """Member ids and site ids overlap, so an index that came back (site,
    member) would make .loc[(5, 42)] silently return a different cell."""
    state = ensemble_state()
    # A pool held per site only, which is what would set the broadcast order.
    state["initial_soil_organic_carbon"] = xr.DataArray(
        np.array([13.0, 20.0]),
        dims=SITE,
        coords={SITE: [1, 27]},
        attrs={"units": resolve_initial_condition("initial_soil_organic_carbon").units},
    )
    table = to_sipnet_initial_conditions_table(
        state, leaf_carbon_per_area=32.0, fine_root_fraction=0.2,
        coarse_root_fraction=0.25, deciduous=False,
    )
    assert table.index.names == [MEMBER, SITE]
    assert table.loc[(1, 27), "soil_carbon"] == pytest.approx(20000.0)


def test_conversion_does_not_compute_the_branch_the_deciduous_rule_discards():
    """A deciduous cell's leaf carbon is never validated, so it must never be
    evaluated either: under np.seterr(all="raise") one such cell would abort
    the conversion of every other."""
    old = np.seterr(all="raise")
    try:
        conditions = to_sipnet_initial_conditions(
            **{**VALID_STATE, "initial_leaf_carbon": 1e306},
            **{**VALID_PARAMETERS, "deciduous": True},
        )
        assert conditions.leaf_area_index == 0.0
    finally:
        np.seterr(**old)


def test_conversion_refuses_arguments_of_the_wrong_shape_or_kind():
    with pytest.raises(TypeError, match="initial_soil_organic_carbon has 1 dimensions"):
        to_sipnet_initial_conditions(
            **{**VALID_STATE, "initial_soil_organic_carbon": [13.0, 14.0]}, **VALID_PARAMETERS
        )
    with pytest.raises(TypeError, match="initial_wood_carbon is a float, not a DataArray"):
        to_sipnet_initial_conditions_table(
            {**ensemble_state(), "initial_wood_carbon": 1.0},
            leaf_carbon_per_area=32.0, fine_root_fraction=0.2,
            coarse_root_fraction=0.25, deciduous=False,
        )


def test_conversion_table_of_scalars_is_one_unlabeled_row():
    table = to_sipnet_initial_conditions_table(
        {name: xr.DataArray(value) for name, value in VALID_STATE.items()}, **VALID_PARAMETERS
    )
    assert len(table) == 1
    assert list(table.columns) == list(CONVERTED_SIPNET_FIELDS)
    assert InitialConditions(**table.iloc[0]) == to_sipnet_initial_conditions(
        **VALID_STATE, **VALID_PARAMETERS
    )
    # With no cells to name, a refusal falls back to the offending value.
    with pytest.raises(ValueError, match=r"\(value -1.0\)"):
        to_sipnet_initial_conditions_table(
            {
                name: xr.DataArray(-1.0 if name == "initial_wood_carbon" else value)
                for name, value in VALID_STATE.items()
            },
            **VALID_PARAMETERS,
        )


def test_conversion_table_labels_an_unindexed_dim_by_position():
    """xarray matches a dim carrying no coordinate positionally, and the index
    then reports positions. Documented, and pinned here so it cannot drift into
    looking like site ids without anyone noticing."""
    state = {
        name: xr.DataArray(field.values, dims=(MEMBER, SITE), attrs=field.attrs)
        for name, field in ensemble_state().items()
    }
    table = to_sipnet_initial_conditions_table(
        state, leaf_carbon_per_area=32.0, fine_root_fraction=0.2,
        coarse_root_fraction=0.25, deciduous=False,
    )
    assert table.index.tolist() == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_conversion_table_takes_a_dataset_and_one_site():
    state = ensemble_state()
    dataset = xr.Dataset(state)
    both = to_sipnet_initial_conditions_table(
        dataset, leaf_carbon_per_area=32.0, fine_root_fraction=0.2,
        coarse_root_fraction=0.25, deciduous=False,
    )
    assert len(both) == 4

    one_site = to_sipnet_initial_conditions_table(
        dataset.sel({SITE: 1}), leaf_carbon_per_area=32.0, fine_root_fraction=0.2,
        coarse_root_fraction=0.25, deciduous=False,
    )
    assert one_site.index.name == MEMBER
    assert one_site.index.tolist() == [0, 1]
    assert one_site.loc[0].to_dict() == both.loc[(0, 1)].to_dict()


def test_conversion_table_refuses_a_bad_state_naming_the_cells():
    state = ensemble_state(leaf=(0.12, np.nan))
    parameters = dict(
        leaf_carbon_per_area=32.0, fine_root_fraction=0.2, coarse_root_fraction=0.25
    )

    with pytest.raises(ValueError, match=r"for example \[\(0, 27\), \(1, 27\)\]\."):
        to_sipnet_initial_conditions_table(state, deciduous=False, **parameters)
    # Deciduous at site 27 is where the absent leaf carbon is, so it converts.
    deciduous = xr.DataArray([False, True], dims=SITE, coords={SITE: [1, 27]})
    to_sipnet_initial_conditions_table(state, deciduous=deciduous, **parameters)

    negative = ensemble_state()
    negative["initial_wood_carbon"][1, 0] = -0.3
    with pytest.raises(
        ValueError, match=r"initial_wood_carbon.*1 of 4 cells.*for example \[\(1, 1\)\]\."
    ):
        to_sipnet_initial_conditions_table(negative, deciduous=False, **parameters)

    # The leaf carbon is checked over the evergreen cells only, so the count it
    # reports has to say so rather than claim to be the whole ensemble.
    half = ensemble_state(leaf=(np.nan, 0.13))
    deciduous_at_27 = xr.DataArray([False, True], dims=SITE, coords={SITE: [1, 27]})
    with pytest.raises(ValueError, match=r"2 of 2 cells whose PFT keeps its leaves"):
        to_sipnet_initial_conditions_table(half, deciduous=deciduous_at_27, **parameters)

    del state["initial_soil_organic_carbon"]
    with pytest.raises(KeyError, match="initial_soil_organic_carbon"):
        to_sipnet_initial_conditions_table(state, deciduous=False, **parameters)


def test_conversion_table_refuses_wrong_units_dims_and_unaligned_parameters():
    parameters = dict(
        leaf_carbon_per_area=32.0, fine_root_fraction=0.2, coarse_root_fraction=0.25,
        deciduous=False,
    )

    converted = ensemble_state()
    converted["initial_soil_organic_carbon"].attrs["units"] = "g m-2"
    with pytest.raises(ValueError, match="units 'g m-2', not the product's 'kg m-2'"):
        to_sipnet_initial_conditions_table(converted, **parameters)

    over_time = ensemble_state()
    with pytest.raises(ValueError, match=r"\['time'\] is not among them"):
        to_sipnet_initial_conditions_table(
            over_time,
            **{**parameters, "fine_root_fraction": xr.DataArray([0.2, 0.3], dims="time")},
        )

    with pytest.raises(ValueError, match="cannot align|conflicting|not equal"):
        to_sipnet_initial_conditions_table(
            ensemble_state(),
            **{
                **parameters,
                "deciduous": xr.DataArray([False, True], dims=SITE, coords={SITE: [1, 99]}),
            },
        )

    # An input without its units attribute is taken at its word, so a hand-built
    # field is usable; the check is against a contradiction, not for a label.
    bare = ensemble_state()
    for array in bare.values():
        array.attrs.clear()
    assert len(to_sipnet_initial_conditions_table(bare, **parameters)) == 4


# ── the real files ────────────────────────────────────────────────────────────


def test_local_source_files_parse_to_their_known_values():
    path = LOCAL_SOURCE_ROOT / "1" / "IC_site_1_1.nc"
    if not path.exists():
        pytest.skip("the PEcAn source files are not in this working copy")
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


def test_tracked_raw_file_holds_the_pecan_identities(tracked_raw):
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


# ── the gaps mutation testing found ───────────────────────────────────────────


def _write_raw_variant(raw: Path, tmp_path: Path, mutate) -> Path:
    """The synthetic raw file with *mutate* applied, written with the raw encoding."""
    with read_raw(raw) as dataset:
        variant = mutate(dataset.load().copy(deep=True))
    path = tmp_path / "variant.nc"
    variant.to_netcdf(path, engine="h5netcdf", encoding=raw_encoding(variant))
    return path


def test_ingest_main_reports_a_broken_identity_and_a_member_gap(raw, sites_csv, tmp_path, capsys):
    def break_wood(dataset):
        dataset["wood_carbon_content"].values[0, 0] += 1e-12
        return dataset

    def gap(dataset):
        return dataset.assign_coords(member=np.array([1, 3], dtype=np.int16))

    out = tmp_path / "p.nc"
    for mutate, message in ((break_wood, "identity"), (gap, "1..2")):
        variant = _write_raw_variant(raw, tmp_path, mutate)
        assert ingest.main(["--raw", str(variant), "--sites", str(sites_csv), "--out", str(out)]) == 1
        assert message in capsys.readouterr().err
        assert not out.exists() and not out.with_suffix(".nc.partial").exists()


def test_read_source_file_refuses_the_wrong_dtype_and_a_missing_attribute(tmp_path):
    root = tmp_path / "files"
    path = _write_source_file(root, 1, 1, SYNTHETIC_VALUES[2][1])
    handle = netcdf_file(str(root / "1" / "IC_site_1_2.nc"), "w", version=1)
    handle.createDimension("time", None)
    time = handle.createVariable("time", "f8", ("time",))
    time[:] = np.asarray([1.0])
    time.units, time.long_name = SOURCE.time_units, SOURCE.time_long_name
    single = handle.createVariable("AbvGrndWood", "f4", ("time",))
    single[:] = np.asarray([0.5], dtype="f4")
    single.units, single.long_name = SOURCE.variables["AbvGrndWood"].units, SOURCE.variables["AbvGrndWood"].long_name
    single._FillValue = SOURCE.fill_value
    handle.close()
    with pytest.raises(ValueError, match="float64"):
        read_source_file(root / "1" / "IC_site_1_2.nc")
    assert read_source_file(path).values["AbvGrndWood"] == 3.0
    # an attribute missing, not merely wrong or extra
    handle = netcdf_file(str(root / "1" / "IC_site_1_3.nc"), "w", version=1)
    handle.createDimension("time", None)
    time = handle.createVariable("time", "f8", ("time",))
    time[:] = np.asarray([1.0])
    time.units, time.long_name = SOURCE.time_units, SOURCE.time_long_name
    bare = handle.createVariable("AbvGrndWood", "f8", ("time",))
    bare[:] = np.asarray([0.5])
    bare.units = SOURCE.variables["AbvGrndWood"].units
    handle.close()
    with pytest.raises(ValueError, match="attributes"):
        read_source_file(root / "1" / "IC_site_1_3.nc")


def test_read_source_file_refuses_the_rest_of_the_template(tmp_path):
    root = tmp_path / "files"

    def write(name, *, version=1, extra_dim=False, fixed_time=False, time_value=1.0):
        handle = netcdf_file(str(root / "1" / name), "w", version=version)
        handle.createDimension("time", 1 if fixed_time else None)
        if extra_dim:
            handle.createDimension("layer", 1)
        time = handle.createVariable("time", "f8", ("time",))
        time[:] = np.asarray([time_value])
        time.units, time.long_name = SOURCE.time_units, SOURCE.time_long_name
        var = handle.createVariable("AbvGrndWood", "f8", ("time",))
        var[:] = np.asarray([0.5])
        var.units, var.long_name = SOURCE.variables["AbvGrndWood"].units, SOURCE.variables["AbvGrndWood"].long_name
        var._FillValue = SOURCE.fill_value
        handle.close()
        return root / "1" / name

    (root / "1").mkdir(parents=True)
    with pytest.raises(ValueError, match="version byte"):
        read_source_file(write("IC_site_1_1.nc", version=2))
    with pytest.raises(ValueError, match="dimensions"):
        read_source_file(write("IC_site_1_2.nc", extra_dim=True))
    with pytest.raises(ValueError, match="unlimited"):
        read_source_file(write("IC_site_1_3.nc", fixed_time=True))
    with pytest.raises(ValueError, match="time value"):
        read_source_file(write("IC_site_1_4.nc", time_value=2.0))


def test_build_raw_refuses_ids_that_do_not_fit_and_unknown_names():
    records = _records()
    with pytest.raises(ValueError, match="int32"):
        build_raw(records + [SourceFile(site=2**31, member=1, values=SYNTHETIC_VALUES[2][1]),
                             SourceFile(site=2**31, member=2, values=SYNTHETIC_VALUES[2][2])],
                  source_root="", conversion_script="")
    with pytest.raises(ValueError, match="not one of"):
        build_raw([SourceFile(site=1, member=1, values={"TotSoilCarb": 1.0})], source_root="", conversion_script="")


def test_read_raw_refuses_the_rest_of_its_schema(raw, tmp_path):
    cases = [
        (lambda d: d.assign_coords(member=np.array([0, 1], dtype=np.int16)), "member"),
        (lambda d: d.assign_coords(member=np.array([1, 40000], dtype=np.int64)), "int16"),
        (lambda d: d.assign_coords(member=np.array([1.0, 2.0])), "integer"),
        (lambda d: d.assign_coords(site=np.array([3, 2, 1], dtype=np.int32)), "ascending"),
        (lambda d: d.transpose("member", "site"), "dims"),
        (lambda d: d.assign(AbvGrndWood=d["AbvGrndWood"].astype(np.float32)), "float64"),
        (lambda d: _with_inf(d), "infinite"),
        (lambda d: _without_attr(d, "n_source_files"), "n_source_files"),
    ]
    for mutate, message in cases:
        with pytest.raises(ValueError, match=message):
            read_raw(_write_raw_variant(raw, tmp_path, mutate))


def _with_inf(dataset):
    dataset["AbvGrndWood"].values[0, 0] = np.inf
    return dataset


def _without_attr(dataset, key):
    del dataset.attrs[key]
    return dataset


def test_load_refuses_the_rest_of_the_data_model(raw, sites_csv, tmp_path):
    sites = load_sites(sites_csv)
    with read_raw(raw) as raw_dataset:
        product = build_initial_conditions(raw_dataset, sites)
    path = tmp_path / "p.nc"

    def refused(mutate, message, encode=True):
        variant = mutate(product.copy(deep=True))
        variant.to_netcdf(path, engine="h5netcdf", encoding=netcdf_encoding(variant) if encode else None)
        with pytest.raises(ValueError, match=message):
            load_initial_conditions(path)

    refused(lambda d: d.assign_coords(member=np.array([1, 2], dtype=np.int16)), "0..n-1")
    refused(lambda d: d.assign_coords(source_member=(MEMBER, np.array([1, 1], dtype=np.int16))), "source_member")
    refused(lambda d: d.assign_coords(source_member=(MEMBER, np.array([0, 1], dtype=np.int16))), "source_member")
    refused(lambda d: d.transpose(SITE, MEMBER), "dims")
    refused(lambda d: d.assign_coords(lon=(SITE, d["lat"].values * 5)), "geographic")
    refused(lambda d: d.assign_coords(lat=(SITE, np.array([np.nan, 1.0, 2.0]))), "non-finite")
    refused(lambda d: d.drop_vars("lat"), "coordinate")
    refused(lambda d: _rename_attr(d, "initial_wood_carbon", "source_name", "AbvGrndWood"), "written from")
    refused(lambda d: _rename_attr(d, "initial_wood_carbon", "long_name", ""), "long_name")
    refused(lambda d: d.assign_attrs(Conventions="CF-1.6"), "Conventions")
    refused(lambda d: d.assign_coords(site=np.array([3, 2, 1], dtype=np.int32)), "ascending")


def _rename_attr(dataset, variable, key, value):
    dataset[variable].attrs[key] = value
    return dataset


def test_coordinates_carry_no_fill_value_on_disk(raw, sites_csv, tmp_path):
    import h5netcdf

    sites = load_sites(sites_csv)
    with read_raw(raw) as raw_dataset:
        product = build_initial_conditions(raw_dataset, sites)
    path = tmp_path / "p.nc"
    product.to_netcdf(path, engine="h5netcdf", encoding=netcdf_encoding(product))
    for written in (path, raw):
        with h5netcdf.File(written, "r") as handle:
            for name in (SITE, MEMBER, SOURCE_MEMBER, "lon", "lat"):
                if name in handle.variables:
                    assert "_FillValue" not in handle.variables[name].attrs, (written, name)


def test_biomass_spec_is_not_fed_to_sipnet():
    spec = resolve_initial_condition("initial_aboveground_biomass_carbon")
    assert spec.sipnet_initial_condition == ""
    assert spec.xarray_attributes()["sipnet_initial_condition"] == "none"


def test_conversion_limit_sites_and_a_variable_absent_everywhere(tree, tmp_path, capsys):
    sites = _write_sites(tmp_path / "sites.csv")
    out = tmp_path / "trial.nc"
    assert convert.main(["--root", str(tree), "--out", str(out), "--sites", str(sites), "--jobs", "1", "--limit-sites", "2"]) == 0
    text = capsys.readouterr().out
    assert "pool check is skipped" in text and "SoilMoistFrac" in text
    with read_raw(out) as dataset:
        assert dataset.sizes[SITE] == 2
    assert convert.main(["--root", str(tree), "--out", str(out), "--sites", str(sites), "--jobs", "1", "--limit-sites", "0"]) == 1
    assert "at least 1" in capsys.readouterr().err
    # a tree where no site carries soil moisture: the report prints dashes, exit 0
    values = {site: {m: {k: v for k, v in rec.items() if k != "SoilMoistFrac"} for m, rec in members.items()}
              for site, members in SYNTHETIC_VALUES.items()}
    dry = _write_tree(tmp_path / "dry", values)
    assert convert.main(["--root", str(dry), "--out", str(tmp_path / "dry.nc"), "--sites", str(sites), "--jobs", "1"]) == 0
    assert "-            -" in capsys.readouterr().out
    product = tmp_path / "dry_product.nc"
    assert ingest.main(["--raw", str(tmp_path / "dry.nc"), "--sites", str(sites), "--out", str(product)]) == 0
    assert product.exists()


def test_conversion_refuses_a_named_site_table_that_is_absent(tree, tmp_path, capsys):
    assert convert.main(["--root", str(tree), "--out", str(tmp_path / "o.nc"), "--sites", str(tmp_path / "nope.csv"), "--jobs", "1"]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_round_trip_checks_notice_a_file_that_differs(
    raw, sites_csv, tmp_path, monkeypatch, capsys
):
    sites = load_sites(sites_csv)
    with read_raw(raw) as raw_dataset:
        dataset = raw_dataset.load().copy(deep=True)
        product = build_initial_conditions(raw_dataset, sites)
    other = _write_raw_variant(raw, tmp_path, lambda d: d.assign(AbvGrndWood=d["AbvGrndWood"] + 1))
    with pytest.raises(convert.ConversionError, match="round-trip"):
        convert.check_round_trip(dataset, other)
    changed = product.copy(deep=True)
    changed["initial_wood_carbon"].values[0, 0] += 1
    path = tmp_path / "changed.nc"
    changed.to_netcdf(path, engine="h5netcdf", encoding=netcdf_encoding(changed))
    with pytest.raises(ingest.IngestError, match="round-trip"):
        ingest.check_round_trip(product, path)
    # a failing round trip keeps the .partial for inspection and says where it is
    monkeypatch.setattr(ingest, "check_round_trip", lambda d, p: (_ for _ in ()).throw(ingest.IngestError("boom")))
    out = tmp_path / "never.nc"
    capsys.readouterr()
    assert ingest.main(["--raw", str(raw), "--sites", str(sites_csv), "--out", str(out)]) == 1
    assert not out.exists() and out.with_suffix(".nc.partial").exists()
    assert str(out.with_suffix(".nc.partial")) in capsys.readouterr().err
