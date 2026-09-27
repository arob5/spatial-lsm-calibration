"""Tests of what every script shares: its command line and its error policy.

Each script's own behavior is tested beside the library it writes for; these
pin the conventions they hold in common, and the check paths no other test
reaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import load_script
from sipnet_calibration.plotting.basemap import BASEMAP_LAYERS, write_basemap

SCRIPT_PATHS = {
    "ingest_sites": "scripts/ingest_sites.py",
    "ingest_constraints": "scripts/ingest_constraints.py",
    "ingest_initial_conditions": "scripts/ingest_initial_conditions.py",
    "ingest_site_labels": "scripts/ingest_site_labels.py",
    "build_basemap": "scripts/build_basemap.py",
    "convert_initial_conditions": "scripts/raw_sources/convert_initial_conditions.py",
    "split_site_pft_16class": "scripts/raw_sources/split_site_pft_16class.py",
    "download_natural_earth": "scripts/raw_sources/download_natural_earth.py",
    "survey_drivers": "scripts/survey_drivers.py",
    "survey_phenology": "scripts/survey_phenology.py",
    "survey_soil_texture": "scripts/survey_soil_texture.py",
}

SCRIPTS = {name: load_script(path) for name, path in SCRIPT_PATHS.items()}


# ── the command line ──────────────────────────────────────────────────────────

#: One flag each script took before its flags were renamed, with the arguments
#: its parser otherwise requires.
OLD_FLAGS = {
    "ingest_sites": ["--out", "x.csv"],
    "ingest_constraints": ["--raw-root", "raw"],
    "ingest_initial_conditions": ["--raw", "raw.nc"],
    "ingest_site_labels": ["--out-dir", "out"],
    "build_basemap": ["--raw-dir", "raw"],
    "convert_initial_conditions": ["--root", "files"],
    "split_site_pft_16class": ["--site-labels-dir", "out"],
    "download_natural_earth": ["--out-dir", "out"],
    "survey_drivers": ["--raw-directory", "drivers", "--root", "drivers"],
    "survey_phenology": ["--path", "phenology.csv"],
    "survey_soil_texture": ["--out", "report.json"],
}


@pytest.mark.parametrize("name", OLD_FLAGS)
def test_an_old_flag_is_refused_rather_than_read_as_an_abbreviation(name, capsys):
    """``allow_abbrev=False``: an old flag such as ``--out`` is not an abbreviation."""
    with pytest.raises(SystemExit) as caught:
        SCRIPTS[name].parse_args(OLD_FLAGS[name])
    assert caught.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


# ── the error policy ──────────────────────────────────────────────────────────

#: For each script, a step its main calls first, the arguments that reach it,
#: and the exit status a reported error gives.
FIRST_STEPS = {
    "ingest_sites": ("check_input_is_a_file", [], 1),
    "ingest_constraints": ("load_sites", ["--constraint", "smap_soil_moisture"], 1),
    "ingest_initial_conditions": ("load_sites", [], 1),
    "ingest_site_labels": ("load_sites", [], 1),
    "build_basemap": ("read_layers", [], 1),
    "convert_initial_conditions": ("find_site_ids", [], 1),
    "split_site_pft_16class": ("read_source", [], 1),
    "download_natural_earth": ("download", [], 1),
    "survey_drivers": ("find_driver_directories", ["--raw-directory", "drivers"], 1),
    "survey_phenology": ("read_phenology", [], 2),
    "survey_soil_texture": ("survey_coverage", [], 2),
}


def _raising(error: BaseException):
    def step(*args, **kwargs):
        raise error

    return step


@pytest.mark.parametrize("error_type", [LookupError, KeyError, TypeError, ValueError, OSError])
@pytest.mark.parametrize("name", FIRST_STEPS)
def test_an_error_the_library_raises_is_reported_not_a_traceback(
    name, error_type, monkeypatch, capsys
):
    step, argv, status = FIRST_STEPS[name]
    monkeypatch.setattr(SCRIPTS[name], step, _raising(error_type("the library refused it")))
    assert SCRIPTS[name].main(argv) == status
    assert "error: " in capsys.readouterr().err


@pytest.mark.parametrize("name", FIRST_STEPS)
def test_an_unexpected_error_is_a_traceback_not_a_report(name, monkeypatch):
    """A bug is not an input error, so it is not caught."""
    step, argv, _ = FIRST_STEPS[name]
    monkeypatch.setattr(SCRIPTS[name], step, _raising(ZeroDivisionError("a bug")))
    with pytest.raises(ZeroDivisionError):
        SCRIPTS[name].main(argv)


@pytest.mark.parametrize("name", [name for name in FIRST_STEPS if not name.startswith("survey")])
def test_the_scripts_own_error_is_a_runtime_error_and_reported(name, monkeypatch, capsys):
    script = SCRIPTS[name]
    assert issubclass(script.IngestError, RuntimeError)
    step, argv, status = FIRST_STEPS[name]
    monkeypatch.setattr(script, step, _raising(script.IngestError("an invariant broke")))
    assert script.main(argv) == status
    assert "error: an invariant broke" in capsys.readouterr().err


# ── check paths no other test reaches ─────────────────────────────────────────


def test_a_missing_time_value_is_refused():
    constraints = SCRIPTS["ingest_constraints"]
    frame = pd.DataFrame({"site_id": [1, 2], "date": ["2016-01-01", None]})
    with pytest.raises(constraints.IngestError, match="'date' has missing values"):
        constraints.check_time_column_is_complete(frame, column="date", message_name="raw.csv.gz")


def test_a_site_id_column_that_is_not_integers_is_refused():
    constraints = SCRIPTS["ingest_constraints"]
    frame = pd.DataFrame({"site_id": [1.0, 2.5]})
    with pytest.raises(constraints.IngestError, match="site_id is not integer-valued"):
        constraints.check_site_id_column_is_integer_valued(frame, message_name="raw.csv.gz")


def test_a_trial_conversion_without_an_output_is_refused(tmp_path, capsys):
    """It would overwrite the tracked raw file with a prefix of the tree."""
    convert = SCRIPTS["convert_initial_conditions"]
    (tmp_path / "1").mkdir()
    argv = ["--source-directory", str(tmp_path), "--limit-sites", "1", "--jobs", "1"]
    assert convert.main(argv) == 1
    assert "--limit-sites needs an explicit --output" in capsys.readouterr().err


def test_a_missing_source_directory_is_refused(tmp_path, capsys):
    convert = SCRIPTS["convert_initial_conditions"]
    argv = ["--source-directory", str(tmp_path / "absent"), "--output", str(tmp_path / "o.nc")]
    assert convert.main(argv) == 1
    assert "is not a directory" in capsys.readouterr().err


def test_a_source_directory_without_a_site_is_refused(tmp_path, capsys):
    convert = SCRIPTS["convert_initial_conditions"]
    argv = ["--source-directory", str(tmp_path), "--output", str(tmp_path / "o.nc")]
    assert convert.main(argv) == 1
    assert "holds no site directories" in capsys.readouterr().err


def test_a_missing_natural_earth_archive_is_refused(tmp_path, capsys):
    build = SCRIPTS["build_basemap"]
    argv = ["--raw-directory", str(tmp_path), "--output", str(tmp_path / "b.npz")]
    assert build.main(argv) == 1
    err = capsys.readouterr().err
    assert "does not exist" in err and "download_natural_earth.py" in err


def _small_basemap_parts() -> dict[str, list[np.ndarray]]:
    return {
        name: [np.array([[-100.0, 40.0], [-99.0, 41.0], [-98.0, 40.5]])]
        for name in BASEMAP_LAYERS
    }


def test_the_basemap_round_trip_passes_a_faithful_file_and_catches_a_changed_one(tmp_path):
    build = SCRIPTS["build_basemap"]
    parts_by_layer = _small_basemap_parts()
    path = tmp_path / "b.npz"
    write_basemap(parts_by_layer, {name: "0" * 32 for name in BASEMAP_LAYERS}, path)
    build.check_written_file_reads_back_identically(parts_by_layer, path)  # must not raise

    layer = next(iter(BASEMAP_LAYERS))
    more_parts = {**parts_by_layer, layer: parts_by_layer[layer] * 2}
    with pytest.raises(build.IngestError, match="read back with 1 parts, not 2"):
        build.check_written_file_reads_back_identically(more_parts, path)

    moved = {**parts_by_layer, layer: [parts_by_layer[layer][0] + 0.5]}
    with pytest.raises(build.IngestError, match="did not read back as written"):
        build.check_written_file_reads_back_identically(moved, path)
