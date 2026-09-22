"""Tests for the script that splits the 16-class assignment into two raw inputs.

The split forfeits the one check every other tracked raw input gets: neither
half can be compared against the upstream md5 once the columns are cut. What
stands in for it is the script's own assertions, so those assertions are what
these tests exercise -- each is driven to failure on a synthetic source, and
the lossless round trip is checked on the real file where this working copy
has it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_HALVES = (
    REPO_ROOT / "data" / "raw" / "labelings" / "site_pft_16class.csv",
    REPO_ROOT / "data" / "raw" / "covariates" / "site_covariates_pft_assignment.csv",
)


def _load_script():
    path = REPO_ROOT / "scripts" / "raw_sources" / "split_site_pft_16class.py"
    spec = importlib.util.spec_from_file_location("split_site_pft_16class", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["split_site_pft_16class"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


split = _load_script()


# ── synthetic fixtures ────────────────────────────────────────────────────────


def _source(rows: int = 4) -> pd.DataFrame:
    """A miniature of the producer's table: the labeling columns plus covariates."""
    frame = pd.DataFrame({column: ["x"] * rows for column in split.LABELING_COLUMNS})
    frame[split.KEY_COLUMN] = [str(site) for site in range(1, rows + 1)]
    frame[split.LABEL_COLUMN] = ["Permanent_Wetlands"] * rows
    frame["MAT"] = ["1.5"] * rows
    frame["BIOME_NAME"] = ["Tundra"] * rows
    return frame


@pytest.fixture(autouse=True)
def _small_pool(monkeypatch):
    """The real script checks against 1-8000; the fixtures are four sites."""
    monkeypatch.setattr(split, "POOL", range(1, 5))


# ── the split itself ──────────────────────────────────────────────────────────


def test_the_halves_partition_the_source():
    source = _source()
    labeling, covariates = split.split(source)
    assert set(labeling.columns) | set(covariates.columns) == set(source.columns)
    assert set(labeling.columns) & set(covariates.columns) == {split.KEY_COLUMN}


def test_both_halves_keep_the_sources_column_order():
    source = _source()
    labeling, covariates = split.split(source)
    for half in (labeling, covariates):
        order = [column for column in source.columns if column in set(half.columns)]
        assert list(half.columns) == order


def test_the_halves_rejoin_to_the_source():
    source = _source()
    labeling, covariates = split.split(source)
    split.check_the_rejoined_halves_reproduce_the_source(source, labeling, covariates)


def test_the_rejoin_check_catches_a_changed_cell():
    source = _source()
    labeling, covariates = split.split(source)
    covariates = covariates.copy()
    covariates.loc[0, "MAT"] = "999"
    with pytest.raises(split.SplitError, match="do not re-join"):
        split.check_the_rejoined_halves_reproduce_the_source(source, labeling, covariates)


def test_the_rejoin_check_catches_a_dropped_row():
    source = _source()
    labeling, covariates = split.split(source)
    with pytest.raises(split.SplitError, match="re-joining gives"):
        split.check_the_rejoined_halves_reproduce_the_source(
            source, labeling.iloc[:-1], covariates
        )


def test_a_source_missing_a_declared_column_is_refused():
    source = _source().drop(columns=["distance_margin"])
    with pytest.raises(split.SplitError, match="does not carry"):
        split.split(source)


def test_a_column_in_both_halves_is_refused():
    source = _source()
    labeling, covariates = split.split(source)
    covariates = covariates.join(labeling[[split.LABEL_COLUMN]])
    with pytest.raises(split.SplitError, match="should share only"):
        split.check_the_columns_partition_the_source(source, labeling, covariates)


def test_a_half_that_is_not_the_whole_pool_is_refused():
    source = _source()
    labeling, covariates = split.split(source)
    with pytest.raises(split.SplitError, match="not the whole site pool"):
        split.check_both_halves_are_keyed_on_the_whole_pool(labeling.iloc[:-1], covariates)


def test_a_duplicated_site_is_refused():
    source = _source()
    labeling, covariates = split.split(source)
    doubled = pd.concat([labeling.iloc[:1], labeling.iloc[:-1]], ignore_index=True)
    with pytest.raises(split.SplitError, match="repeats"):
        split.check_both_halves_are_keyed_on_the_whole_pool(doubled, covariates)


def test_an_empty_class_is_refused():
    source = _source()
    source.loc[1, split.LABEL_COLUMN] = ""
    labeling, _ = split.split(source)
    with pytest.raises(split.SplitError, match="empty final_pft"):
        split.check_the_label_column_is_complete(labeling)


def test_an_absent_source_points_at_the_provenance_record(tmp_path):
    with pytest.raises(FileNotFoundError, match="provenance.md"):
        split.read_source(tmp_path / "absent.csv")


def test_the_source_is_read_as_text_so_no_float_is_reparsed(tmp_path):
    """Reading as strings is what makes the split verbatim."""
    path = tmp_path / "s.csv"
    path.write_text(f"{split.KEY_COLUMN},MAT\n1,0.10000000000000001\n")
    frame = split.read_source(path)
    assert frame.loc[0, "MAT"] == "0.10000000000000001"


def test_writing_both_halves_round_trips(tmp_path):
    source = _source()
    labeling, covariates = split.split(source)
    written = split.write_halves(labeling, covariates, tmp_path / "l", tmp_path / "c")
    assert set(written) == {split.LABELING_FILE, split.COVARIATES_FILE}
    for name, path in written.items():
        expected = labeling if name == split.LABELING_FILE else covariates
        back = pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False)
        pd.testing.assert_frame_equal(back, expected.reset_index(drop=True))


def test_main_reports_and_exits_zero(tmp_path, capsys):
    path = tmp_path / "s.csv"
    _source().to_csv(path, index=False)
    code = split.main(
        [
            "--source", str(path),
            "--labelings-dir", str(tmp_path / "l"),
            "--covariates-dir", str(tmp_path / "c"),
        ]
    )
    assert code == 0
    assert "re-join to the source cell for cell" in capsys.readouterr().out


def test_main_reports_an_error_rather_than_a_traceback(tmp_path, capsys):
    path = tmp_path / "s.csv"
    _source().drop(columns=["distance_margin"]).to_csv(path, index=False)
    assert split.main(["--source", str(path)]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_describe_needs_no_input(capsys):
    assert split.main(["--describe"]) == 0
    out = capsys.readouterr().out
    assert split.LABELING_FILE in out and split.COVARIATES_FILE in out


# ── the real halves ───────────────────────────────────────────────────────────


def test_the_committed_halves_rejoin_losslessly(monkeypatch):
    """The two tracked files really are one table cut in two."""
    if not all(path.exists() for path in SOURCE_HALVES):
        pytest.skip("the split halves are not in this working copy")
    monkeypatch.setattr(split, "POOL", range(1, 8001))
    labeling, covariates = (
        pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False)
        for path in SOURCE_HALVES
    )
    split.check_both_halves_are_keyed_on_the_whole_pool(labeling, covariates)
    split.check_the_label_column_is_complete(labeling)
    assert set(labeling.columns) & set(covariates.columns) == {split.KEY_COLUMN}
    assert len(labeling.columns) + len(covariates.columns) - 1 == 60


def test_the_committed_labeling_half_is_the_specs_raw_columns():
    """The spec and the split script have to agree on the column set."""
    if not SOURCE_HALVES[0].exists():
        pytest.skip("the split halves are not in this working copy")
    from sipnet_calibration.labelings import resolve_labeling

    header = pd.read_csv(SOURCE_HALVES[0], nrows=0)
    assert tuple(header.columns) == resolve_labeling("pft_16class").raw_columns
