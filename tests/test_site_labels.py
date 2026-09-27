"""Tests for the site-labels specs, the processed files they describe, and the ingest.

Three layers, as the constraint tests have. The specs are checked for internal
consistency and against the real raw files' headers. The conversion is
exercised on small synthetic raw tables where the expected processed file can be
written out by hand and every refusal provoked -- including the one that
matters most, a file of the right shape holding the wrong site pool. Finally
the real file is ingested and the result checked against the site table, which
is where the exact ``landcover`` relation is confirmed over all 8000 sites.

The real-data cases skip when the raw file or the site table is absent from
the working copy.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPOSITORY, load_script, write_site_table_csv
from sipnet_calibration.conventions import (
    LAT_ATTRIBUTES,
    LON_ATTRIBUTES,
    SITE_ATTRIBUTES,
    SITE_ID,
)
from sipnet_calibration.site_labels import (
    LABEL_COLUMN,
    SITE_LABELS,
    SITE_LABELS_COLUMN_DTYPES,
    SITE_LABELS_COLUMNS,
    SITE_LABELS_NAMES,
    SiteLabelsSpec,
    _check_labels_are_flag_meanings,
    build_site_labels,
    default_raw_dir,
    default_site_labels_dir,
    describe,
    load_site_labels,
    read_raw,
    resolve_site_labels,
    site_labels_field,
    site_labels_path,
)
from sipnet_calibration.sites import default_sites_path, load_sites

#: The tracked raw files, found from the repository rather than the data root.
RAW_DIR = REPOSITORY / "data" / "raw" / "site_labels"


ingest = load_script("scripts/ingest_site_labels.py")


# ── synthetic fixtures ────────────────────────────────────────────────────────

SYNTHETIC_SITES = [1, 2, 3, 4]
SYNTHETIC_LANDCOVER = {1: 1, 2: 2, 3: 3, 4: 5}

SYNTHETIC_SPEC = SiteLabelsSpec(
    name="synthetic_3class",
    long_label="Synthetic three-class site labels",
    label_kind="plant functional type",
    labels=("conifer", "broadleaf", "grass"),
    description="Site labels that exist only in these tests.",
    upstream_product="test fixture",
    raw_file="synthetic_site_class.csv",
    raw_columns=("site", "klass"),
    site_column="site",
    label_column="klass",
    expected_rows=4,
    covers_pool=True,
    landcover_mapping={1: "conifer", 2: "conifer", 3: "broadleaf", 5: "grass"},
)

SYNTHETIC_ROWS = [
    {"site": 1, "klass": "conifer"},
    {"site": 2, "klass": "conifer"},
    {"site": 3, "klass": "broadleaf"},
    {"site": 4, "klass": "grass"},
]


def _write_sites(path: Path, site_ids=SYNTHETIC_SITES) -> Path:
    """A minimal site table that ``load_sites`` accepts."""
    return write_site_table_csv(
        path,
        site_ids,
        lon=[-100.0 - site for site in site_ids],
        lat=[40.0 + site for site in site_ids],
        landcover=[SYNTHETIC_LANDCOVER.get(site, 1) for site in site_ids],
    )


def _write_raw(root: Path, spec: SiteLabelsSpec, rows: list[dict]) -> Path:
    """Write rows as the real raw file is written: quoted CSV."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / spec.raw_file
    frame = pd.DataFrame(rows, columns=list(spec.raw_columns))
    frame.to_csv(path, index=False, quoting=1)
    return path


@pytest.fixture
def synthetic(tmp_path):
    """A raw file, a site table and an output directory that agree with each other."""
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS)
    site_table = load_sites(_write_sites(tmp_path / "sites.csv"))
    return raw_root, site_table, tmp_path / "out"


# ── the specs ─────────────────────────────────────────────────────────────────


def test_registry_names_are_unique_and_match_the_specs():
    assert SITE_LABELS_NAMES == tuple(spec.name for spec in SITE_LABELS)
    assert len(set(SITE_LABELS_NAMES)) == len(SITE_LABELS_NAMES)


def test_resolve_site_labels_names_what_exists_when_asked_for_something_else():
    with pytest.raises(KeyError, match="reanalysis_3pft"):
        resolve_site_labels("no_such_site_labels")


@pytest.mark.parametrize("spec", SITE_LABELS, ids=lambda spec: spec.name)
def test_spec_raw_file_names_a_file_in_the_raw_directory(spec):
    if not RAW_DIR.exists():
        pytest.skip("raw site labels not available in this working copy")
    assert (RAW_DIR / spec.raw_file).exists()


@pytest.mark.parametrize("spec", SITE_LABELS, ids=lambda spec: spec.name)
def test_spec_raw_columns_are_the_real_files_header(spec):
    path = RAW_DIR / spec.raw_file
    if not path.exists():
        pytest.skip("raw site labels not available in this working copy")
    header = pd.read_csv(path, nrows=0)
    assert tuple(header.columns) == spec.raw_columns


@pytest.mark.parametrize("spec", SITE_LABELS, ids=lambda spec: spec.name)
def test_landcover_mapping_sends_every_cover_class_to_a_declared_label(spec):
    if spec.landcover_mapping is None:
        pytest.skip("no landcover relation declared")
    assert set(spec.landcover_mapping.values()) <= set(spec.labels)


def test_describe_names_the_classes_and_the_relation():
    text = describe(resolve_site_labels("reanalysis_3pft"))
    assert "boreal.coniferous" in text
    assert "landcover 5-8 -> semiarid.grassland_HPDA" in text


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("name", "Reanalysis3PFT", "lower_case_with_underscores"),
        ("labels", ("only_one",), "at least two classes"),
        ("labels", ("a", "a", "b"), "repeats a class"),
        ("raw_columns", ("site", "site"), "repeats a column"),
        ("site_column", "absent", "is not in raw_columns"),
        ("expected_rows", 0, "expected_rows must be at least 1"),
        ("description", "", "needs a description"),
        ("label_kind", "", "needs a label_kind"),
    ],
)
def test_spec_refuses_an_inconsistent_field(field, value, match):
    kwargs = {
        "name": "ok_name",
        "long_label": "Fine",
        "label_kind": "plant functional type",
        "labels": ("a", "b"),
        "description": "Fine.",
        "upstream_product": "test",
        "raw_file": "f.csv",
        "raw_columns": ("site", "klass"),
        "site_column": "site",
        "label_column": "klass",
        "expected_rows": 2,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        SiteLabelsSpec(**kwargs)


def test_spec_refuses_a_landcover_mapping_onto_an_undeclared_class():
    with pytest.raises(ValueError, match="not in labels"):
        SiteLabelsSpec(
            name="ok_name",
            long_label="Fine",
            label_kind="plant functional type",
            labels=("a", "b"),
            description="Fine.",
            upstream_product="test",
            raw_file="f.csv",
            raw_columns=("site", "klass"),
            site_column="site",
            label_column="klass",
            expected_rows=2,
            landcover_mapping={1: "c"},
        )


# ── the conversion, on synthetic data ─────────────────────────────────────────


def test_ingest_writes_the_data_model(synthetic):
    raw_root, site_table, out_dir = synthetic
    site_labels = ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)

    assert tuple(site_labels.columns) == SITE_LABELS_COLUMNS
    assert site_labels[SITE_ID].dtype == np.int32
    assert site_labels[SITE_ID].tolist() == SYNTHETIC_SITES
    assert site_labels[LABEL_COLUMN].tolist() == ["conifer", "conifer", "broadleaf", "grass"]
    assert site_labels_path(SYNTHETIC_SPEC, out_dir).exists()


def test_label_is_a_categorical_over_the_specs_classes_in_order(synthetic):
    raw_root, site_table, out_dir = synthetic
    ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)
    written = load_site_labels(SYNTHETIC_SPEC, site_labels_path(SYNTHETIC_SPEC, out_dir))

    assert written[LABEL_COLUMN].dtype == pd.CategoricalDtype(
        list(SYNTHETIC_SPEC.labels), ordered=False
    )
    # Every class, in the spec's order, whether or not the file uses them all.
    assert list(written[LABEL_COLUMN].cat.categories) == list(SYNTHETIC_SPEC.labels)


def test_the_written_file_reads_back_as_what_was_built(synthetic):
    raw_root, site_table, out_dir = synthetic
    site_labels = ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)
    written = load_site_labels(SYNTHETIC_SPEC, site_labels_path(SYNTHETIC_SPEC, out_dir))
    pd.testing.assert_frame_equal(written, site_labels)


def test_rows_are_sorted_by_site_whatever_the_raw_order(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, SYNTHETIC_SPEC, list(reversed(SYNTHETIC_ROWS)))
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    site_labels = build_site_labels(SYNTHETIC_SPEC, frame)
    assert site_labels[SITE_ID].tolist() == SYNTHETIC_SITES


def test_a_class_literally_named_na_survives_the_read(tmp_path):
    """``keep_default_na=False``: the eight sites named ``NA`` taught this lesson."""
    spec = SiteLabelsSpec(
        name="na_class",
        long_label="Site labels with a class named NA",
        label_kind="cover class",
        labels=("NA", "other"),
        description="Exists to prove the null handling.",
        upstream_product="test",
        raw_file="na.csv",
        raw_columns=("site", "klass"),
        site_column="site",
        label_column="klass",
        expected_rows=2,
    )
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, spec, [{"site": 1, "klass": "NA"}, {"site": 2, "klass": "other"}])
    frame = read_raw(spec, raw_root)
    assert frame["klass"].tolist() == ["NA", "other"]


# ── the refusals ──────────────────────────────────────────────────────────────


def test_the_wrong_site_pool_is_refused_by_its_row_count(tmp_path):
    """The check that tells the 8000-site file from its 6400-site namesake."""
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS[:3])
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    with pytest.raises(ingest.IngestError, match="holds 3 rows, expected 4"):
        ingest.check_row_count_is_the_expected_pool(SYNTHETIC_SPEC, frame)


def test_the_refusal_message_points_at_the_provenance_record(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS[:3])
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    with pytest.raises(ingest.IngestError, match="provenance.md"):
        ingest.check_row_count_is_the_expected_pool(SYNTHETIC_SPEC, frame)


def test_nothing_is_written_when_a_raw_check_fails(synthetic):
    raw_root, site_table, out_dir = synthetic
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS[:3])
    with pytest.raises(ingest.IngestError):
        ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)
    assert not site_labels_path(SYNTHETIC_SPEC, out_dir).exists()
    assert not out_dir.exists() or not list(out_dir.glob("*.partial"))


def test_a_duplicate_site_is_refused(tmp_path):
    raw_root = tmp_path / "raw"
    rows = SYNTHETIC_ROWS[:3] + [{"site": 3, "klass": "grass"}]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    with pytest.raises(ingest.IngestError, match="more than once"):
        ingest.check_no_duplicate_sites(SYNTHETIC_SPEC, frame)


def test_a_site_outside_the_pool_is_refused(tmp_path, synthetic):
    raw_root, site_table, _ = synthetic
    rows = SYNTHETIC_ROWS[:3] + [{"site": 99, "klass": "grass"}]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    with pytest.raises(KeyError, match=r"site\(s\) \[99\] are not in the site table"):
        ingest.check_raw_frame(SYNTHETIC_SPEC, frame, site_table)


def test_an_unlabeled_site_is_refused_when_the_spec_covers_the_pool(tmp_path):
    raw_root = tmp_path / "raw"
    site_table = load_sites(_write_sites(tmp_path / "sites.csv", [1, 2, 3, 4, 5]))
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS)
    site_labels = build_site_labels(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    with pytest.raises(ingest.IngestError, match="unlabeled"):
        ingest.check_pool_is_completely_labeled(SYNTHETIC_SPEC, site_labels, site_table)


def test_an_undeclared_class_is_refused(tmp_path):
    raw_root = tmp_path / "raw"
    rows = SYNTHETIC_ROWS[:3] + [{"site": 4, "klass": "tundra"}]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    with pytest.raises(ValueError, match="does not declare"):
        build_site_labels(SYNTHETIC_SPEC, frame)


def test_a_declared_class_no_site_uses_is_refused(tmp_path, synthetic):
    raw_root, site_table, _ = synthetic
    rows = [dict(row, klass="conifer") for row in SYNTHETIC_ROWS]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    site_labels = build_site_labels(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    with pytest.raises(ingest.IngestError, match="that no site has"):
        ingest.check_labels_are_the_declared_set(SYNTHETIC_SPEC, site_labels)


def test_a_class_that_departs_from_the_landcover_relation_is_refused(tmp_path, synthetic):
    raw_root, site_table, _ = synthetic
    # Site 4 has landcover 5, which the mapping sends to grass.
    rows = SYNTHETIC_ROWS[:3] + [{"site": 4, "klass": "broadleaf"}]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    site_labels = build_site_labels(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    with pytest.raises(ingest.IngestError, match="landcover_mapping"):
        ingest.check_labels_match_landcover(SYNTHETIC_SPEC, site_labels, site_table)


def test_a_cover_class_the_mapping_does_not_cover_is_refused(tmp_path):
    raw_root = tmp_path / "raw"
    site_table_path = tmp_path / "sites.csv"
    _write_sites(site_table_path)
    site_table = load_sites(site_table_path)
    site_table.loc[site_table[SITE_ID] == 4, "landcover"] = np.int8(7)
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS)
    site_labels = build_site_labels(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    with pytest.raises(ingest.IngestError, match="does not cover"):
        ingest.check_labels_match_landcover(SYNTHETIC_SPEC, site_labels, site_table)


def test_a_wrong_raw_header_is_refused(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / SYNTHETIC_SPEC.raw_file).write_text("site,pft\n1,conifer\n")
    with pytest.raises(ValueError, match="header is"):
        read_raw(SYNTHETIC_SPEC, raw_root)


def test_an_absent_raw_file_points_at_the_provenance_record(tmp_path):
    with pytest.raises(FileNotFoundError, match="provenance.md"):
        read_raw(SYNTHETIC_SPEC, tmp_path / "nothing")


def test_an_absent_processed_file_names_the_command_that_makes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest_site_labels.py"):
        load_site_labels("reanalysis_3pft", tmp_path / "absent.csv")


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("site_id,label\n1,conifer\n1,grass\n", "repeats"),
        ("site_id,label\n2,conifer\n1,grass\n", "ascending"),
        ("site_id,label\n1,conifer\n2,tundra\n", "does not declare"),
        ("site_id,klass\n1,conifer\n", "header is"),
        ("site_id,label\n0,conifer\n", "site ids from 1 to 2147483647"),
    ],
)
def test_load_site_labels_refuses_a_file_off_the_data_model(tmp_path, content, match):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_site_labels(SYNTHETIC_SPEC, path)


# ── the real file ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def real_site_table() -> pd.DataFrame:
    try:
        return load_sites(default_sites_path())
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"site table not available in this working copy: {error}")


@pytest.fixture(scope="session")
def real_site_labels(real_site_table, tmp_path_factory) -> pd.DataFrame:
    """``reanalysis_3pft`` built from the tracked raw file."""
    spec = resolve_site_labels("reanalysis_3pft")
    if not (RAW_DIR / spec.raw_file).exists():
        pytest.skip("raw site labels not available in this working copy")
    out_dir = tmp_path_factory.mktemp("site_labels")
    return ingest.ingest(spec, RAW_DIR, real_site_table, out_dir)


def test_the_real_site_labels_cover_the_whole_pool(real_site_labels, real_site_table):
    assert len(real_site_labels) == len(real_site_table)
    assert real_site_labels[SITE_ID].tolist() == real_site_table[SITE_ID].tolist()


def test_the_real_site_labels_are_exactly_the_landcover_aggregation(real_site_labels, real_site_table):
    """The claim data/README.md makes under Site labels, over all 8000 sites."""
    spec = resolve_site_labels("reanalysis_3pft")
    joined = real_site_labels.merge(real_site_table[[SITE_ID, "landcover"]], on=SITE_ID)
    expected = joined["landcover"].map(dict(spec.landcover_mapping))
    assert (expected == joined[LABEL_COLUMN].astype(str)).all()
    # And the relation is onto: every class is reached from some cover class.
    assert set(expected) == set(spec.labels)


def test_every_declared_class_is_used_by_the_real_site_labels(real_site_labels):
    spec = resolve_site_labels("reanalysis_3pft")
    assert set(real_site_labels[LABEL_COLUMN].unique()) == set(spec.labels)


def test_the_real_raw_file_has_the_specs_row_count(real_site_table):
    spec = resolve_site_labels("reanalysis_3pft")
    if not (RAW_DIR / spec.raw_file).exists():
        pytest.skip("raw site labels not available in this working copy")
    assert len(read_raw(spec, RAW_DIR)) == spec.expected_rows


# ── the 16-class site labels ──────────────────────────────────────────────────


@pytest.mark.parametrize("spec", SITE_LABELS, ids=lambda spec: spec.name)
def test_display_names_cover_exactly_the_classes(spec):
    if spec.display_names is None:
        pytest.skip("no display names declared")
    assert set(spec.display_names) == set(spec.labels)


def test_the_16class_spec_declares_no_landcover_relation():
    """It comes from a different upstream cover product, so no exact relation holds."""
    assert resolve_site_labels("pft_16class").landcover_mapping is None


def test_display_names_must_cover_every_class():
    kwargs = dict(
        name="ok_name",
        long_label="Fine",
        label_kind="plant functional type",
        labels=("a", "b"),
        description="Fine.",
        upstream_product="test",
        raw_file="f.csv",
        raw_columns=("site", "klass"),
        site_column="site",
        label_column="klass",
        expected_rows=2,
    )
    with pytest.raises(ValueError, match="no entry for"):
        SiteLabelsSpec(**kwargs, display_names={"a": "A"})
    with pytest.raises(ValueError, match="not classes of these site labels"):
        SiteLabelsSpec(**kwargs, display_names={"a": "A", "b": "B", "c": "C"})


@pytest.fixture(scope="session")
def real_16class(real_site_table, tmp_path_factory) -> pd.DataFrame:
    spec = resolve_site_labels("pft_16class")
    if not (RAW_DIR / spec.raw_file).exists():
        pytest.skip("raw site labels not available in this working copy")
    return ingest.ingest(spec, RAW_DIR, real_site_table, tmp_path_factory.mktemp("l16"))


def test_the_16class_site_labels_cover_the_pool_with_all_sixteen(real_16class, real_site_table):
    spec = resolve_site_labels("pft_16class")
    assert len(real_16class) == len(real_site_table)
    assert set(real_16class[LABEL_COLUMN].unique()) == set(spec.labels)


def test_the_two_site_labels_data_sources_do_not_nest(real_site_labels, real_16class):
    """Recorded in data/README.md Note 11 and in the spec's own comment.

    Every 16-class class draws from at least two of the three reanalysis
    classes, so no class has a parent whose prior it could inherit.
    """
    joined = real_16class.merge(real_site_labels, on=SITE_ID, suffixes=("_16", "_3"))
    spread = (
        pd.crosstab(joined[f"{LABEL_COLUMN}_16"], joined[f"{LABEL_COLUMN}_3"]) > 0
    ).sum(axis=1)
    assert spread.min() >= 2
    assert (spread == 3).sum() == 12


# ── the orchestration ─────────────────────────────────────────────────────────
#
# The checks above are exercised by calling them. These exercise the wiring:
# that ``ingest`` calls each one, that ``write_processed_file`` will not rename over a
# failed round trip, and that ``main`` turns a refusal into a reported error
# with an exit code rather than a traceback. Deleting a call site is invisible
# to a test that calls the check directly.


RAW_CHECKS = (
    "check_row_count_is_the_expected_pool",
    "check_no_duplicate_sites",
    "check_site_table_lists_the_sites",
)
SITE_LABELS_CHECKS = (
    "check_labels_are_the_declared_set",
    "check_pool_is_completely_labeled",
    "check_labels_match_landcover",
)


class _Sentinel(ingest.IngestError):
    """Raised by a stubbed check, so the test can tell it apart from a real one."""


@pytest.mark.parametrize("check", RAW_CHECKS + SITE_LABELS_CHECKS + ("check_round_trip",))
def test_ingest_calls_every_check(synthetic, monkeypatch, check):
    """Each check is reached on a run whose data is otherwise valid."""
    raw_root, site_table, out_dir = synthetic

    def boom(*args, **kwargs):
        raise _Sentinel(check)

    monkeypatch.setattr(ingest, check, boom)
    with pytest.raises(_Sentinel, match=check):
        ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)


@pytest.mark.parametrize("check", RAW_CHECKS)
def test_the_raw_checks_run_before_anything_is_built(synthetic, monkeypatch, check):
    """A raw check firing must leave the output directory untouched."""
    raw_root, site_table, out_dir = synthetic

    def boom(*args, **kwargs):
        raise _Sentinel(check)

    monkeypatch.setattr(ingest, check, boom)
    with pytest.raises(_Sentinel):
        ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)
    assert not out_dir.exists()


def test_the_row_count_check_runs_first(synthetic, monkeypatch):
    """On the wrong pool every check fails; the row count is the useful message."""
    raw_root, site_table, out_dir = synthetic
    for check in RAW_CHECKS[1:] + SITE_LABELS_CHECKS:
        monkeypatch.setattr(ingest, check, lambda *a, **k: None)
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS[:3])
    with pytest.raises(ingest.IngestError, match="holds 3 rows, expected 4"):
        ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)


# ── the .partial write protocol ───────────────────────────────────────────────


def test_a_failed_site_labels_round_trip_keeps_the_partial_and_never_writes_the_processed_file(
    synthetic, monkeypatch
):
    """CLAUDE.md's rule: rename only after the checks pass."""
    raw_root, site_table, out_dir = synthetic

    def boom(*args, **kwargs):
        raise ingest.IngestError("round trip")

    monkeypatch.setattr(ingest, "check_round_trip", boom)
    with pytest.raises(ingest.IngestError):
        ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)

    out = site_labels_path(SYNTHETIC_SPEC, out_dir)
    assert not out.exists()
    assert [path.name for path in out_dir.glob("*.partial")] == [f"{out.name}.partial"]


def test_a_failed_round_trip_leaves_an_existing_processed_file_intact(synthetic, monkeypatch):
    """A failed rerun must not damage the file a previous run left behind."""
    raw_root, site_table, out_dir = synthetic
    ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)
    out = site_labels_path(SYNTHETIC_SPEC, out_dir)
    before = out.read_bytes()

    def boom(*args, **kwargs):
        raise ingest.IngestError("round trip")

    monkeypatch.setattr(ingest, "check_round_trip", boom)
    with pytest.raises(ingest.IngestError):
        ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)
    assert out.read_bytes() == before


def test_the_round_trip_check_compares_against_the_library_loader(
    synthetic, monkeypatch
):
    """Site labels that do not read back as built are refused, not renamed."""
    raw_root, site_table, out_dir = synthetic
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = site_labels_path(SYNTHETIC_SPEC, out_dir).with_suffix(".csv.partial")
    site_labels = build_site_labels(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    site_labels.iloc[:-1].to_csv(partial, index=False)
    with pytest.raises(ingest.IngestError, match="does not read back"):
        ingest.check_round_trip(SYNTHETIC_SPEC, site_labels, partial)


# ── main ──────────────────────────────────────────────────────────────────────


def _argv(raw_root, site_table_path, out_dir, *extra):
    return [
        "--site-labels", SYNTHETIC_SPEC.name,
        "--raw-root", str(raw_root),
        "--site-table", str(site_table_path),
        "--out-dir", str(out_dir),
        *extra,
    ]


@pytest.fixture
def real_argv(tmp_path):
    """A `main` invocation against the registry's own site labels and raw file."""
    # main reads both from its defaults, so check the paths it will read.
    for spec in SITE_LABELS:
        if not (default_raw_dir() / spec.raw_file).exists():
            pytest.skip("raw site labels not available in this working copy")
    if not default_sites_path().exists():
        pytest.skip("site table not available in this working copy")
    return tmp_path


@pytest.mark.parametrize("name", SITE_LABELS_NAMES)
def test_main_exits_zero_and_writes_the_processed_file(real_argv, name):
    out_dir = real_argv / "out"
    code = ingest.main(["--site-labels", name, "--out-dir", str(out_dir)])
    assert code == 0
    written = load_site_labels(name, site_labels_path(name, out_dir))
    assert len(written) == resolve_site_labels(name).expected_rows


def test_main_with_no_arguments_builds_every_site_labels_data_source(real_argv):
    out_dir = real_argv / "out"
    assert ingest.main(["--out-dir", str(out_dir)]) == 0
    assert sorted(path.stem for path in out_dir.glob("*.csv")) == sorted(SITE_LABELS_NAMES)


def test_main_honors_the_site_labels_argument(real_argv):
    """Naming one site-labels data source must not build the others."""
    out_dir = real_argv / "out"
    code = ingest.main(["--site-labels", "reanalysis_3pft", "--out-dir", str(out_dir)])
    assert code == 0
    assert [path.name for path in out_dir.glob("*.csv")] == ["reanalysis_3pft.csv"]


def test_main_reports_an_error_and_exits_one(tmp_path, capsys):
    site_table_path = _write_sites(tmp_path / "sites.csv")
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS[:3])
    # main resolves by name, so drive it through the registry's own spec with a
    # raw file of the wrong length.
    spec = resolve_site_labels("reanalysis_3pft")
    _write_raw(raw_root, spec, [{"site": 1, "pft": spec.labels[0]}])
    code = ingest.main(
        [
            "--site-labels", spec.name,
            "--raw-root", str(raw_root),
            "--site-table", str(site_table_path),
            "--out-dir", str(tmp_path / "out"),
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err
    assert not (tmp_path / "out").exists()


def test_main_describes_without_reading_data(tmp_path, capsys):
    code = ingest.main(["--describe", "--raw-root", str(tmp_path / "absent")])
    assert code == 0
    out = capsys.readouterr().out
    assert "reanalysis_3pft" in out and "pft_16class" in out


def test_main_refuses_unknown_site_labels(capsys):
    with pytest.raises(SystemExit) as exit_info:
        ingest.main(["--site-labels", "no_such_site_labels"])
    assert exit_info.value.code == 2


def test_describe_processed_file_reports_the_pool_the_right_way_round(synthetic):
    raw_root, site_table, out_dir = synthetic
    site_labels = ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)
    text = ingest.describe_processed_file(
        SYNTHETIC_SPEC, site_labels, site_table, site_labels_path(SYNTHETIC_SPEC, out_dir)
    )
    assert f"{len(site_labels)} of {len(site_table)} in the pool" in text
    for label in SYNTHETIC_SPEC.labels:
        assert label in text


def test_describe_processed_file_survives_a_class_no_site_uses(synthetic):
    """A zero-count class must not KeyError out of the run report."""
    raw_root, site_table, _ = synthetic
    spec = dataclasses.replace(SYNTHETIC_SPEC, landcover_mapping=None)
    site_labels = build_site_labels(spec, read_raw(spec, raw_root))
    site_labels = site_labels[site_labels[LABEL_COLUMN] != "grass"]
    text = ingest.describe_processed_file(spec, site_labels, site_table, Path("x.csv"))
    assert "grass" in text


# ── boundaries the refusal tests step over ────────────────────────────────────


def test_a_file_with_too_many_rows_is_refused(tmp_path):
    """`==`, not `>=`: an over-long file is as wrong as a short one."""
    raw_root = tmp_path / "raw"
    rows = SYNTHETIC_ROWS + [{"site": 5, "klass": "grass"}]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    with pytest.raises(ingest.IngestError, match="holds 5 rows, expected 4"):
        ingest.check_row_count_is_the_expected_pool(SYNTHETIC_SPEC, frame)


def test_exactly_one_unused_class_is_refused(tmp_path, synthetic):
    """The existing case leaves two classes unused; one is the boundary."""
    raw_root, site_table, _ = synthetic
    rows = [dict(row) for row in SYNTHETIC_ROWS]
    rows[3]["klass"] = "broadleaf"  # grass now unused, conifer and broadleaf are not
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    site_labels = build_site_labels(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    with pytest.raises(ingest.IngestError, match=r"\['grass'\]"):
        ingest.check_labels_are_the_declared_set(SYNTHETIC_SPEC, site_labels)


def test_partial_site_labels_are_allowed_when_the_spec_says_so(tmp_path):
    spec = dataclasses.replace(SYNTHETIC_SPEC, covers_pool=False, expected_rows=3)
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, spec, SYNTHETIC_ROWS[:3])
    site_table = load_sites(_write_sites(tmp_path / "sites.csv"))
    site_labels = build_site_labels(spec, read_raw(spec, raw_root))
    ingest.check_pool_is_completely_labeled(spec, site_labels, site_table)  # must not raise


def test_a_reordered_header_is_refused(tmp_path):
    """The spec declares an order, and `read_raw`'s docstring says it is enforced."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / SYNTHETIC_SPEC.raw_file).write_text("klass,site\nconifer,1\n")
    with pytest.raises(ValueError, match="header is"):
        read_raw(SYNTHETIC_SPEC, raw_root)


def test_a_reordered_processed_file_header_is_refused(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("label,site_id\nconifer,1\n")
    with pytest.raises(ValueError, match="header is"):
        load_site_labels(SYNTHETIC_SPEC, path)


def test_a_row_with_more_fields_than_the_header_is_refused(tmp_path):
    """The parse hardening in `load_site_labels`, which carries its own comment."""
    path = tmp_path / "bad.csv"
    path.write_text("site_id,label\n1,conifer\n2,grass,extra\n")
    with pytest.raises(ValueError, match="could not be parsed"):
        load_site_labels(SYNTHETIC_SPEC, path)


def test_an_identifier_too_large_for_int32_is_refused(tmp_path):
    """Narrowing without the check would wrap silently to a negative id."""
    path = tmp_path / "bad.csv"
    path.write_text(f"site_id,label\n{2**31},conifer\n")
    with pytest.raises(ValueError, match="site ids from 1 to 2147483647"):
        load_site_labels(SYNTHETIC_SPEC, path)


def test_a_class_named_na_survives_the_processed_file_round_trip(tmp_path):
    """`keep_default_na=False` is tested on read_raw; this is the processed side."""
    spec = dataclasses.replace(
        SYNTHETIC_SPEC,
        name="na_labels",
        labels=("NA", "other"),
        expected_rows=2,
        landcover_mapping=None,
    )
    path = tmp_path / "p.csv"
    path.write_text("site_id,label\n1,NA\n2,other\n")
    assert load_site_labels(spec, path)[LABEL_COLUMN].tolist() == ["NA", "other"]


def test_an_empty_processed_file_is_refused(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("site_id,label\n")
    with pytest.raises(ValueError, match="holds no rows"):
        load_site_labels(SYNTHETIC_SPEC, path)


def test_the_site_labels_index_is_reset_after_sorting(tmp_path):
    """Without `ignore_index` the frame keeps the raw file's row numbers."""
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, SYNTHETIC_SPEC, list(reversed(SYNTHETIC_ROWS)))
    site_labels = build_site_labels(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    assert site_labels.index.tolist() == list(range(len(SYNTHETIC_ROWS)))


# ── the registry and the schema, pinned rather than restated ──────────────────


def test_the_registry_class_orders_are_what_was_run_against():
    """`labels` is a prior's class axis; reordering it silently would move it."""
    assert resolve_site_labels("reanalysis_3pft").labels == (
        "boreal.coniferous",
        "temperate.deciduous.HPDA",
        "semiarid.grassland_HPDA",
    )
    sixteen = resolve_site_labels("pft_16class").labels
    assert sixteen[0] == "Evergreen_Needleleaf_Forest__P1"
    assert sixteen[-1] == "Permanent_Wetlands"
    assert len(sixteen) == 16


def test_the_registry_declares_what_the_readme_says_it_does():
    assert resolve_site_labels("reanalysis_3pft").covers_pool is True
    assert resolve_site_labels("pft_16class").covers_pool is True


def test_the_schema_constants_are_read_only():
    """A caller mutating these would change every later read of site labels."""
    with pytest.raises(TypeError):
        SITE_LABELS_COLUMN_DTYPES["site_id"] = np.int64
    with pytest.raises(TypeError):
        resolve_site_labels("reanalysis_3pft").landcover_mapping[1] = "x"


def test_a_spec_cannot_be_mutated_after_construction():
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolve_site_labels("reanalysis_3pft").expected_rows = 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("long_label", "", "needs a description"),
        ("upstream_product", "", "needs a description"),
        ("labels", ("a", ""), "class name is empty"),
        ("label_column", "absent", "is not in raw_columns"),
        ("site_column", "klass", "are the same"),
        ("name", "ok_name!!", "lower_case_with_underscores"),
    ],
)
def test_more_inconsistent_spec_fields_are_refused(field, value, match):
    kwargs = {
        "name": "ok_name",
        "long_label": "Fine",
        "label_kind": "plant functional type",
        "labels": ("a", "b"),
        "description": "Fine.",
        "upstream_product": "test",
        "raw_file": "f.csv",
        "raw_columns": ("site", "klass"),
        "site_column": "site",
        "label_column": "klass",
        "expected_rows": 2,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        SiteLabelsSpec(**kwargs)


# ── the categorical field ─────────────────────────────────────────────────────


def test_site_labels_field_is_cf_flag_codes_with_locations(synthetic):
    raw_root, site_table, out_dir = synthetic
    ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)
    field = site_labels_field(
        SYNTHETIC_SPEC, site_table=site_table, path=site_labels_path(SYNTHETIC_SPEC, out_dir)
    )

    assert field.dims == ("site",) and field.dtype == np.int8
    assert field["site"].values.tolist() == SYNTHETIC_SITES
    assert field.attrs["flag_meanings"] == "conifer broadleaf grass"
    assert field.attrs["flag_values"].tolist() == [0, 1, 2]
    assert "units" not in field.attrs
    meanings = field.attrs["flag_meanings"].split()
    assert [meanings[code] for code in field.values] == ["conifer", "conifer", "broadleaf", "grass"]
    np.testing.assert_array_equal(field["lon"].values, site_table["lon"].to_numpy())
    assert field["lon"].attrs == dict(LON_ATTRIBUTES)
    assert field["lat"].attrs == dict(LAT_ATTRIBUTES)
    assert field["site"].attrs == dict(SITE_ATTRIBUTES)
    assert field.name == SYNTHETIC_SPEC.name
    assert field.attrs["long_name"] == "Plant functional type (synthetic_3class)"
    assert "flag_display_names" not in field.attrs


def test_site_labels_field_carries_display_names_in_flag_order(synthetic):
    raw_root, site_table, out_dir = synthetic
    named = dataclasses.replace(
        SYNTHETIC_SPEC,
        display_names={"grass": "Grassland", "conifer": "Conifer forest", "broadleaf": "Broadleaf forest"},
    )
    ingest.ingest(named, raw_root, site_table, out_dir)
    field = site_labels_field(named, site_table=site_table, path=site_labels_path(named, out_dir))
    assert field.attrs["flag_display_names"] == ("Conifer forest", "Broadleaf forest", "Grassland")


def test_the_16class_field_labels_every_site_with_readable_names(real_16class, real_site_table, tmp_path):
    spec = resolve_site_labels("pft_16class")
    path = site_labels_path(spec, tmp_path)
    real_16class.to_csv(path, index=False)
    field = site_labels_field(spec, site_table=real_site_table, path=path)
    assert field.sizes["site"] == len(real_site_table)
    assert field.attrs["flag_meanings"].split() == list(spec.labels)
    assert field.attrs["flag_display_names"] == tuple(spec.display_names[label] for label in spec.labels)
    assert sorted(np.unique(field.values).tolist()) == list(range(len(spec.labels)))


def test_site_labels_field_refuses_a_labeled_site_the_table_lacks(synthetic):
    raw_root, site_table, out_dir = synthetic
    ingest.ingest(SYNTHETIC_SPEC, raw_root, site_table, out_dir)
    with pytest.raises(KeyError, match="not in the site table"):
        site_labels_field(
            SYNTHETIC_SPEC, site_table=site_table.iloc[:2], path=site_labels_path(SYNTHETIC_SPEC, out_dir)
        )


def test_a_class_name_with_whitespace_cannot_be_a_flag_meaning():
    spaced = dataclasses.replace(
        SYNTHETIC_SPEC, labels=("conifer", "broad leaf", "grass"),
        landcover_mapping={1: "conifer", 2: "conifer", 3: "broad leaf", 5: "grass"},
    )
    with pytest.raises(ValueError, match="whitespace"):
        _check_labels_are_flag_meanings(spaced)


# ── the default paths ─────────────────────────────────────────────────────────


def test_the_default_paths_honor_the_data_root_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SIPNET_CALIBRATION_DATA", str(tmp_path))
    assert default_raw_dir() == tmp_path / "raw" / "site_labels"
    assert default_site_labels_dir() == tmp_path / "processed" / "site_labels"


def test_site_labels_path_is_the_name_with_a_csv_suffix(tmp_path):
    three = site_labels_path("reanalysis_3pft", tmp_path)
    assert three == tmp_path / "reanalysis_3pft.csv"
    sixteen = site_labels_path(resolve_site_labels("pft_16class"), tmp_path)
    assert sixteen.name == "pft_16class.csv"


# ── the spec cannot change, and hashes ───────────────────────────────────────


@pytest.mark.parametrize("spec", SITE_LABELS, ids=lambda spec: spec.name)
def test_a_registered_spec_hashes_pickles_and_cannot_change(spec):
    import pickle

    assert {spec: spec.name}[spec] == spec.name
    assert pickle.loads(pickle.dumps(spec)) == spec
    for mapping in (spec.landcover_mapping, spec.display_names):
        if mapping is not None:
            with pytest.raises(TypeError):
                mapping[next(iter(mapping))] = "changed"


def test_a_spec_keeps_its_own_copy_of_the_mappings_it_was_given():
    display_names = {label: label.title() for label in SYNTHETIC_SPEC.labels}
    spec = dataclasses.replace(SYNTHETIC_SPEC, display_names=display_names)
    display_names[SYNTHETIC_SPEC.labels[0]] = "changed"
    assert spec.display_names[SYNTHETIC_SPEC.labels[0]] == SYNTHETIC_SPEC.labels[0].title()
    assert hash(spec) == hash(dataclasses.replace(spec))
