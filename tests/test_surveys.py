"""Tests for the two survey scripts that assert what ``data/README.md`` records.

These scripts are diagnostics, but they are also the mechanism that keeps the
coverage numbers in the README from going stale: each carries a ``RECORDED``
table, compares its measurements against it, and exits non-zero when one no
longer holds. That makes the comparison itself worth testing, because a checker
that cannot fail is worse than no checker -- it reports success over a changed
file.

So the cases here are mostly about the *checker*: that a changed measurement is
caught, that a report round-tripped through the JSON output still compares
equal, and that the failure and success exit codes are distinct and mean what
the docstrings say. The measuring code is exercised on small synthetic inputs,
and on the real files where this working copy has them.

The real-data cases skip when the files are absent. The soil texture ensemble is
never complete locally -- the repository holds three members of two sites -- so
nothing here asserts its recorded coverage; that check belongs where the files
are, which is what ``--no-check`` exists for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
PHENOLOGY_DIR = REPO_ROOT / "data" / "raw" / "phenology"
SOIL_TEXTURE_DIR = REPO_ROOT / "data" / "raw" / "soil_texture"


def _load_script(name: str):
    """Import a file in ``scripts/``, which is not a package."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


phenology = _load_script("survey_phenology")
soil = _load_script("survey_soil_texture")


# ── synthetic fixtures ────────────────────────────────────────────────────────


def _write_phenology(path: Path, rows: list[dict]) -> Path:
    """A phenology CSV with the real header and ``NA`` for a missing day."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(phenology.COLUMNS))
    frame.to_csv(path, index=False, na_rep="NA")
    return path


def _phenology_rows(sites=(1, 2), years=(2012, 2013)) -> list[dict]:
    """A complete site-year rectangle, every day present and flagged best."""
    return [
        {
            "year": year,
            "site_id": site,
            "lat": 40.0 + site,
            "lon": -100.0 - site,
            "leafonday": 120,
            "leafoffday": 280,
            "leafon_qa": 0,
            "leafoff_qa": 0,
        }
        for site in sites
        for year in years
    ]


def _write_soil_file(path: Path, porosity=None, depths=soil.DEPTHS_METERS) -> Path:
    """One soil texture netCDF with the real variable names and depths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(depths)
    porosity = np.full(n, 0.45) if porosity is None else np.asarray(porosity)
    third = np.full(n, 1.0 / 3.0, dtype=np.float32)
    dataset = xr.Dataset(
        {
            soil.POROSITY: ("depth", porosity.astype(np.float32)),
            **{name: ("depth", third) for name in soil.TEXTURE_FRACTIONS},
        },
        coords={"depth": np.asarray(depths, dtype=np.float64)},
    )
    dataset.to_netcdf(path)
    return path


def _soil_root(tmp_path: Path, sites=(1, 2), members=(1, 2)) -> Path:
    root = tmp_path / "soil"
    for site in sites:
        for member in members:
            _write_soil_file(root / str(site) / f"Soil_params_0-{site}_{member}.nc")
    return root


# ── the phenology survey's measurements ───────────────────────────────────────


def test_a_complete_rectangle_is_recognized(tmp_path):
    path = _write_phenology(tmp_path / "p.csv", _phenology_rows())
    report = phenology.build_report(phenology.read_phenology(path), path, None)
    assert report["complete_rectangle"] is True
    assert report["rows"] == 4
    assert report["sites"] == 2


def test_a_missing_site_year_is_not_a_rectangle(tmp_path):
    path = _write_phenology(tmp_path / "p.csv", _phenology_rows()[:3])
    report = phenology.build_report(phenology.read_phenology(path), path, None)
    assert report["complete_rectangle"] is False


def test_quality_three_is_reported_as_coinciding_with_missing(tmp_path):
    rows = _phenology_rows()
    rows[0] |= {"leafonday": np.nan, "leafon_qa": 3}
    rows[0] |= {"leafoffday": np.nan, "leafoff_qa": 3}
    path = _write_phenology(tmp_path / "p.csv", rows)
    report = phenology.build_report(phenology.read_phenology(path), path, None)
    assert report["quality_three_is_exactly_missing"] is True
    assert report["missing_days"]["leafonday"] == 1


def test_a_day_present_under_quality_three_breaks_the_coincidence(tmp_path):
    rows = _phenology_rows()
    rows[0] |= {"leafon_qa": 3}  # flagged poor, but the day is still there
    path = _write_phenology(tmp_path / "p.csv", rows)
    report = phenology.build_report(phenology.read_phenology(path), path, None)
    assert report["quality_three_is_exactly_missing"] is False


def test_the_inverted_pair_is_counted(tmp_path):
    """The `yday` artifact the README documents: leaf-off in the next January."""
    rows = _phenology_rows()
    rows[0] |= {"leafonday": 300, "leafoffday": 40}
    path = _write_phenology(tmp_path / "p.csv", rows)
    report = phenology.build_report(phenology.read_phenology(path), path, None)
    assert report["inverted_rows"] == 1
    assert report["inverted_sites"] == 1
    assert report["inverted_leafoffday_median"] == 40.0


def test_a_quality_value_outside_zero_to_three_is_reported(tmp_path):
    rows = _phenology_rows()
    rows[0] |= {"leafon_qa": 7}
    path = _write_phenology(tmp_path / "p.csv", rows)
    report = phenology.build_report(phenology.read_phenology(path), path, None)
    assert report["quality_values_outside_0_3"] == [7]


def test_a_wrong_header_is_refused(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("year,site_id\n2012,1\n")
    with pytest.raises(ValueError, match="header is"):
        phenology.read_phenology(path)


def test_an_absent_file_names_the_readme_section(tmp_path):
    with pytest.raises(FileNotFoundError, match="Leaf phenology"):
        phenology.read_phenology(tmp_path / "absent.csv")


def test_an_unparseable_file_is_reported_with_its_path(tmp_path):
    path = tmp_path / "p.csv"
    path.write_bytes(b"\x88\x88\x88\x88")
    with pytest.raises(ValueError, match=str(path)):
        phenology.read_phenology(path)


# ── the phenology survey's checker ────────────────────────────────────────────


def test_a_file_with_no_recorded_entry_is_reported_not_silently_passed():
    failures = phenology.compare_with_recorded({"rows": 1}, "not_a_known_file.csv")
    assert failures and "no recorded characteristics" in failures[0]


def test_a_changed_measurement_is_caught():
    recorded = phenology.RECORDED["leaf_phenology_8k.csv"]
    report = dict(recorded) | {"rows": recorded["rows"] + 1}
    failures = phenology.compare_with_recorded(report, "leaf_phenology_8k.csv")
    assert len(failures) == 1 and failures[0].startswith("rows:")


def test_a_changed_nested_quality_count_is_caught():
    """The counts live one level down, which the key handling has to reach."""
    recorded = phenology.RECORDED["leaf_phenology_8k.csv"]
    counts = {
        column: dict(values) for column, values in recorded["quality_counts"].items()
    }
    counts["leafon_qa"][0] += 1
    report = dict(recorded) | {"quality_counts": counts}
    failures = phenology.compare_with_recorded(report, "leaf_phenology_8k.csv")
    assert len(failures) == 1 and failures[0].startswith("quality_counts:")


def test_string_and_integer_keys_compare_equal_at_every_depth():
    """A report read back from --out has string keys; it must still compare."""
    recorded = phenology.RECORDED["leaf_phenology_8k.csv"]
    through_json = json.loads(json.dumps(recorded))
    assert any(isinstance(key, int) for key in recorded["quality_counts"]["leafon_qa"])
    assert phenology.compare_with_recorded(through_json, "leaf_phenology_8k.csv") == []


def test_an_absent_measurement_is_caught_rather_than_skipped():
    recorded = phenology.RECORDED["leaf_phenology_8k.csv"]
    report = {key: value for key, value in recorded.items() if key != "inverted_rows"}
    failures = phenology.compare_with_recorded(report, "leaf_phenology_8k.csv")
    assert len(failures) == 1 and failures[0].startswith("inverted_rows:")


# ── the soil texture survey's measurements ────────────────────────────────────


def test_coverage_counts_sites_members_and_files(tmp_path):
    report = soil.survey_coverage(_soil_root(tmp_path))
    assert report["sites_with_a_directory"] == 2
    assert report["members_per_site"] == [2]
    assert report["member_range"] == [1, 2]
    assert report["total_files"] == 4
    assert report["file_names_off_template"] == 0


def test_a_file_whose_name_disagrees_with_its_directory_is_off_template(tmp_path):
    root = _soil_root(tmp_path, sites=(1,), members=(1,))
    _write_soil_file(root / "1" / "Soil_params_0-2_1.nc")
    report = soil.survey_coverage(root)
    assert report["file_names_off_template"] == 1


def test_a_zero_padded_directory_does_not_collide_with_its_plain_form(tmp_path):
    """`0027` and `27` both int() to 27; one used to overwrite the other."""
    root = _soil_root(tmp_path, sites=(27,), members=(1,))
    _write_soil_file(root / "0027" / "Soil_params_0-27_50.nc")
    report = soil.survey_coverage(root)
    assert report["sites_with_a_directory"] == 1
    assert report["file_names_off_template"] == 1
    assert report["first_off_template"] == ["0027/"]


def test_site_directories_holding_no_files_are_an_error_not_a_crash(tmp_path):
    """The shape an interrupted copy takes."""
    root = tmp_path / "soil"
    (root / "1").mkdir(parents=True)
    (root / "2").mkdir(parents=True)
    with pytest.raises(ValueError, match="none holding a file on the template"):
        soil.survey_coverage(root)


def test_an_empty_root_is_an_error(tmp_path):
    root = tmp_path / "soil"
    root.mkdir()
    with pytest.raises(ValueError, match="no site directories"):
        soil.survey_coverage(root)


def test_an_absent_root_names_the_readme_section(tmp_path):
    with pytest.raises(FileNotFoundError, match="Soil texture"):
        soil.survey_coverage(tmp_path / "absent")


def test_soil_water_holding_capacity_reproduces_pecans_formula(tmp_path):
    """Porosity times layer thickness, summed over the 2 m profile, in cm."""
    path = _write_soil_file(tmp_path / "one.nc", porosity=np.full(6, 0.5))
    with xr.open_dataset(path) as dataset:
        # Thicknesses are 0.05, 0.10, 0.15, 0.30, 0.40, 1.00 m, summing to 2 m,
        # so a uniform porosity of 0.5 gives exactly 100 cm.
        assert soil.soil_water_holding_capacity(dataset) == pytest.approx(100.0)


def test_a_missing_porosity_layer_gives_nan_as_pecan_does(tmp_path):
    """PEcAn's `sum` takes na.rm = FALSE, so one absent layer makes soilWHC NA."""
    porosity = np.full(6, 0.5)
    porosity[-1] = np.nan
    path = _write_soil_file(tmp_path / "one.nc", porosity=porosity)
    with xr.open_dataset(path) as dataset:
        assert np.isnan(soil.soil_water_holding_capacity(dataset))


def test_a_file_with_a_missing_porosity_is_counted_and_left_out_of_the_range(tmp_path):
    root = _soil_root(tmp_path, sites=(1,), members=(1,))
    porosity = np.full(6, 0.5)
    porosity[0] = np.nan
    _write_soil_file(root / "1" / "Soil_params_0-1_2.nc", porosity=porosity)
    coverage = soil.survey_coverage(root)
    content = soil.survey_content(root, coverage["sites"], sample=10, every=True)
    assert content["files_with_missing_porosity"] == 1
    # The finite file is a uniform 0.45 over 2 m, so 90 cm; the NaN one is left
    # out of the range rather than dragging it down with a partial integral.
    assert content["soil_water_holding_capacity_cm"]["min"] == pytest.approx(90.0)
    assert content["soil_water_holding_capacity_cm"]["max"] == pytest.approx(90.0)


def test_a_different_depth_profile_is_caught(tmp_path):
    """soilWHC over another profile is a different quantity, not a new value."""
    root = tmp_path / "soil"
    _write_soil_file(
        root / "1" / "Soil_params_0-1_1.nc", depths=(0.05, 0.15, 0.3, 0.6, 1.0, 3.0)
    )
    coverage = soil.survey_coverage(root)
    content = soil.survey_content(root, coverage["sites"], sample=10, every=True)
    assert content["depths_are_the_expected_profile"] is False
    assert soil.compare_with_recorded({**coverage, **content})


def test_units_disagreeing_between_files_are_all_reported(tmp_path):
    root = _soil_root(tmp_path, sites=(1,), members=(1,))
    path = root / "1" / "Soil_params_0-1_2.nc"
    _write_soil_file(path)
    with xr.open_dataset(path) as dataset:
        changed = dataset.load()
    changed[soil.POROSITY].attrs["units"] = "WRONG"
    changed.to_netcdf(path)
    coverage = soil.survey_coverage(root)
    content = soil.survey_content(root, coverage["sites"], sample=10, every=True)
    assert len(content["units"][soil.POROSITY]) == 2


def test_the_sample_is_deterministic_and_bounded():
    values = list(range(1, 101))
    assert soil._evenly_spaced(values, 5) == soil._evenly_spaced(values, 5)
    assert len(soil._evenly_spaced(values, 5)) == 5
    assert soil._evenly_spaced(values, 0) == []
    assert soil._evenly_spaced(values, 500) == values
    with pytest.raises(ValueError, match="must not be negative"):
        soil._evenly_spaced(values, -1)


def test_the_sample_spans_the_identifier_range():
    """Identifiers run north to south, so a head-of-list sample would be Arctic."""
    chosen = soil._evenly_spaced(list(range(1, 8001)), 25)
    assert chosen[0] == 1
    assert chosen[-1] == 8000


# ── the soil texture survey's checker ─────────────────────────────────────────


def test_the_recorded_coverage_is_caught_when_it_changes():
    report = dict(soil.RECORDED) | {"sites_with_a_directory": 7000}
    failures = soil.compare_with_recorded(report)
    assert len(failures) == 1 and failures[0].startswith("sites_with_a_directory:")


def test_the_recorded_coverage_passes_when_it_holds():
    assert soil.compare_with_recorded(dict(soil.RECORDED)) == []


def test_unreadable_files_fail_the_comparison():
    """A run where nothing opened used to pass on the directory listing alone."""
    report = dict(soil.RECORDED) | {"unreadable_files": 3}
    assert soil.compare_with_recorded(report)


# ── the real files ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["leaf_phenology_8k.csv", "leaf_phenology_neon.csv"]
)
def test_the_real_phenology_files_match_what_the_readme_records(name):
    path = PHENOLOGY_DIR / name
    if not path.exists():
        pytest.skip("phenology files not available in this working copy")
    frame = phenology.read_phenology(path)
    report = phenology.build_report(frame, path, None)
    failures = [
        failure
        for failure in phenology.compare_with_recorded(report, name)
        # These two need the site table, which the NEON file cannot join to;
        # the run with --sites covers them.
        if not failure.startswith(("identifiers_not_in_", "sites_of_the_pool_absent"))
    ]
    assert failures == []


def test_every_real_phenology_file_has_a_recorded_entry():
    if not PHENOLOGY_DIR.exists():
        pytest.skip("phenology files not available in this working copy")
    present = {path.name for path in PHENOLOGY_DIR.glob("*.csv")}
    assert present <= set(phenology.RECORDED)


def test_the_local_soil_texture_files_are_on_the_template_and_readable():
    """Coverage is not asserted here: the local copy is three of 769,300 files."""
    if not SOIL_TEXTURE_DIR.exists():
        pytest.skip("soil texture files not available in this working copy")
    coverage = soil.survey_coverage(SOIL_TEXTURE_DIR)
    content = soil.survey_content(SOIL_TEXTURE_DIR, coverage["sites"], 10, True)
    assert coverage["file_names_off_template"] == 0
    assert content["unreadable_files"] == 0
    assert content["depths_are_the_expected_profile"] is True
    assert content["files_with_missing_porosity"] == 0
    assert content["max_texture_fraction_residual"] < 1e-6
