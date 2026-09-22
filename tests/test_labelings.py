"""Tests for the labeling specs, the products they describe, and the ingest.

Three layers, as the constraint tests have. The specs are checked for internal
consistency and against the real raw files' headers. The conversion is
exercised on small synthetic raw tables where the expected product can be
written out by hand and every refusal provoked -- including the one that
matters most, a file of the right shape holding the wrong site pool. Finally
the real file is ingested and the result checked against the site table, which
is where the exact ``landcover`` relation is confirmed over all 8000 sites.

The real-data cases skip when the raw file or the site table is absent from
the working copy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sipnet_calibration.labelings import (
    LABELING_COLUMNS,
    LABELING_NAMES,
    LABELINGS,
    LABEL_COLUMN,
    SITE_COLUMN,
    LabelingSpec,
    build_labeling,
    describe,
    label_dtype,
    labeling_path,
    load_labeling,
    read_raw,
    resolve_labeling,
)
from sipnet_calibration.sites import SITE_COLUMNS, default_sites_path, load_sites

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "labelings"


def _load_ingest_module():
    """Import ``scripts/ingest_labelings.py``, which is a script."""
    path = REPO_ROOT / "scripts" / "ingest_labelings.py"
    spec = importlib.util.spec_from_file_location("ingest_labelings", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["ingest_labelings"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


ingest = _load_ingest_module()


# ── synthetic fixtures ────────────────────────────────────────────────────────

SYNTHETIC_SITES = [1, 2, 3, 4]
SYNTHETIC_LANDCOVER = {1: 1, 2: 2, 3: 3, 4: 5}

SYNTHETIC_SPEC = LabelingSpec(
    name="synthetic_3class",
    long_label="Synthetic three-class labeling",
    label_kind="plant functional type",
    labels=("conifer", "broadleaf", "grass"),
    description="A labeling that exists only in these tests.",
    product="test fixture",
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
    frame = pd.DataFrame(
        {
            "site_id": np.array(site_ids, dtype=np.int32),
            "lon": [-100.0 - site for site in site_ids],
            "lat": [40.0 + site for site in site_ids],
            "lon_index": np.arange(len(site_ids), dtype=np.int32) + 1000,
            "lat_index": np.arange(len(site_ids), dtype=np.int32) + 2000,
            "site_name": [f"site {site}" for site in site_ids],
            "site_order": np.zeros(len(site_ids), dtype=np.int32),
            "cluster": np.ones(len(site_ids), dtype=np.int8),
            "landcover": np.array(
                [SYNTHETIC_LANDCOVER.get(site, 1) for site in site_ids], dtype=np.int8
            ),
            "ameriflux_site_id": [""] * len(site_ids),
        }
    )
    assert tuple(frame.columns) == SITE_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_raw(root: Path, spec: LabelingSpec, rows: list[dict]) -> Path:
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
    sites = load_sites(_write_sites(tmp_path / "sites.csv"))
    return raw_root, sites, tmp_path / "out"


# ── the specs ─────────────────────────────────────────────────────────────────


def test_registry_names_are_unique_and_match_the_specs():
    assert LABELING_NAMES == tuple(spec.name for spec in LABELINGS)
    assert len(set(LABELING_NAMES)) == len(LABELING_NAMES)


def test_resolve_labeling_names_what_exists_when_asked_for_something_else():
    with pytest.raises(KeyError, match="reanalysis_3pft"):
        resolve_labeling("no_such_labeling")


@pytest.mark.parametrize("spec", LABELINGS, ids=lambda spec: spec.name)
def test_spec_raw_file_names_a_file_in_the_raw_directory(spec):
    if not RAW_DIR.exists():
        pytest.skip("raw labelings not available in this working copy")
    assert (RAW_DIR / spec.raw_file).exists()


@pytest.mark.parametrize("spec", LABELINGS, ids=lambda spec: spec.name)
def test_spec_raw_columns_are_the_real_files_header(spec):
    path = RAW_DIR / spec.raw_file
    if not path.exists():
        pytest.skip("raw labelings not available in this working copy")
    header = pd.read_csv(path, nrows=0)
    assert tuple(header.columns) == spec.raw_columns


@pytest.mark.parametrize("spec", LABELINGS, ids=lambda spec: spec.name)
def test_landcover_mapping_sends_every_cover_class_to_a_declared_label(spec):
    if spec.landcover_mapping is None:
        pytest.skip("no landcover relation declared")
    assert set(spec.landcover_mapping.values()) <= set(spec.labels)


def test_describe_names_the_classes_and_the_relation():
    text = describe(resolve_labeling("reanalysis_3pft"))
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
        ("expected_rows", 0, "must be positive"),
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
        "product": "test",
        "raw_file": "f.csv",
        "raw_columns": ("site", "klass"),
        "site_column": "site",
        "label_column": "klass",
        "expected_rows": 2,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        LabelingSpec(**kwargs)


def test_spec_refuses_a_landcover_mapping_onto_an_undeclared_class():
    with pytest.raises(ValueError, match="not in labels"):
        LabelingSpec(
            name="ok_name",
            long_label="Fine",
            label_kind="plant functional type",
            labels=("a", "b"),
            description="Fine.",
            product="test",
            raw_file="f.csv",
            raw_columns=("site", "klass"),
            site_column="site",
            label_column="klass",
            expected_rows=2,
            landcover_mapping={1: "c"},
        )


# ── the conversion, on synthetic data ─────────────────────────────────────────


def test_ingest_writes_the_data_model(synthetic):
    raw_root, sites, out_dir = synthetic
    product = ingest.ingest(SYNTHETIC_SPEC, raw_root, sites, out_dir)

    assert tuple(product.columns) == LABELING_COLUMNS
    assert product[SITE_COLUMN].dtype == np.int32
    assert product[SITE_COLUMN].tolist() == SYNTHETIC_SITES
    assert product[LABEL_COLUMN].tolist() == ["conifer", "conifer", "broadleaf", "grass"]
    assert labeling_path(SYNTHETIC_SPEC, out_dir).exists()


def test_label_is_a_categorical_over_the_specs_classes_in_order(synthetic):
    raw_root, sites, out_dir = synthetic
    ingest.ingest(SYNTHETIC_SPEC, raw_root, sites, out_dir)
    written = load_labeling(SYNTHETIC_SPEC, labeling_path(SYNTHETIC_SPEC, out_dir))

    assert written[LABEL_COLUMN].dtype == label_dtype(SYNTHETIC_SPEC)
    # Every class, in the spec's order, whether or not the file uses them all.
    assert list(written[LABEL_COLUMN].cat.categories) == list(SYNTHETIC_SPEC.labels)


def test_the_written_file_reads_back_as_what_was_built(synthetic):
    raw_root, sites, out_dir = synthetic
    product = ingest.ingest(SYNTHETIC_SPEC, raw_root, sites, out_dir)
    written = load_labeling(SYNTHETIC_SPEC, labeling_path(SYNTHETIC_SPEC, out_dir))
    pd.testing.assert_frame_equal(written, product)


def test_rows_are_sorted_by_site_whatever_the_raw_order(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, SYNTHETIC_SPEC, list(reversed(SYNTHETIC_ROWS)))
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    product = build_labeling(SYNTHETIC_SPEC, frame)
    assert product[SITE_COLUMN].tolist() == SYNTHETIC_SITES


def test_a_class_literally_named_na_survives_the_read(tmp_path):
    """``keep_default_na=False``: the eight sites named ``NA`` taught this lesson."""
    spec = LabelingSpec(
        name="na_class",
        long_label="A labeling with a class named NA",
        label_kind="cover class",
        labels=("NA", "other"),
        description="Exists to prove the null handling.",
        product="test",
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


def test_nothing_is_written_when_a_check_fails(tmp_path, synthetic):
    raw_root, sites, out_dir = synthetic
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS[:3])
    with pytest.raises(ingest.IngestError):
        ingest.ingest(SYNTHETIC_SPEC, raw_root, sites, out_dir)
    assert not labeling_path(SYNTHETIC_SPEC, out_dir).exists()
    assert not out_dir.exists() or not list(out_dir.glob("*.partial"))


def test_a_duplicate_site_is_refused(tmp_path):
    raw_root = tmp_path / "raw"
    rows = SYNTHETIC_ROWS[:3] + [{"site": 3, "klass": "grass"}]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    with pytest.raises(ingest.IngestError, match="more than once"):
        ingest.check_no_duplicate_sites(SYNTHETIC_SPEC, frame)


def test_a_site_outside_the_pool_is_refused(tmp_path, synthetic):
    raw_root, sites, _ = synthetic
    rows = SYNTHETIC_ROWS[:3] + [{"site": 99, "klass": "grass"}]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    with pytest.raises(ingest.IngestError, match="not sites"):
        ingest.check_sites_are_in_the_site_table(SYNTHETIC_SPEC, frame, sites)


def test_an_unlabeled_site_is_refused_when_the_spec_covers_the_pool(tmp_path):
    raw_root = tmp_path / "raw"
    sites = load_sites(_write_sites(tmp_path / "sites.csv", [1, 2, 3, 4, 5]))
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS)
    product = build_labeling(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    with pytest.raises(ingest.IngestError, match="unlabeled"):
        ingest.check_pool_is_completely_labeled(SYNTHETIC_SPEC, product, sites)


def test_an_undeclared_class_is_refused(tmp_path):
    raw_root = tmp_path / "raw"
    rows = SYNTHETIC_ROWS[:3] + [{"site": 4, "klass": "tundra"}]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    frame = read_raw(SYNTHETIC_SPEC, raw_root)
    with pytest.raises(ValueError, match="does not declare"):
        build_labeling(SYNTHETIC_SPEC, frame)


def test_a_declared_class_no_site_uses_is_refused(tmp_path, synthetic):
    raw_root, sites, _ = synthetic
    rows = [dict(row, klass="conifer") for row in SYNTHETIC_ROWS]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    product = build_labeling(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    with pytest.raises(ingest.IngestError, match="that no site has"):
        ingest.check_labels_are_the_declared_set(SYNTHETIC_SPEC, product)


def test_a_class_that_departs_from_the_landcover_relation_is_refused(tmp_path, synthetic):
    raw_root, sites, _ = synthetic
    # Site 4 has landcover 5, which the mapping sends to grass.
    rows = SYNTHETIC_ROWS[:3] + [{"site": 4, "klass": "broadleaf"}]
    _write_raw(raw_root, SYNTHETIC_SPEC, rows)
    product = build_labeling(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    with pytest.raises(ingest.IngestError, match="landcover_mapping"):
        ingest.check_labels_match_landcover(SYNTHETIC_SPEC, product, sites)


def test_a_cover_class_the_mapping_does_not_cover_is_refused(tmp_path):
    raw_root = tmp_path / "raw"
    sites_path = tmp_path / "sites.csv"
    _write_sites(sites_path)
    sites = load_sites(sites_path)
    sites.loc[sites[SITE_COLUMN] == 4, "landcover"] = np.int8(7)
    _write_raw(raw_root, SYNTHETIC_SPEC, SYNTHETIC_ROWS)
    product = build_labeling(SYNTHETIC_SPEC, read_raw(SYNTHETIC_SPEC, raw_root))
    with pytest.raises(ingest.IngestError, match="does not cover"):
        ingest.check_labels_match_landcover(SYNTHETIC_SPEC, product, sites)


def test_a_wrong_raw_header_is_refused(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / SYNTHETIC_SPEC.raw_file).write_text("site,pft\n1,conifer\n")
    with pytest.raises(ValueError, match="header is"):
        read_raw(SYNTHETIC_SPEC, raw_root)


def test_an_absent_raw_file_points_at_the_provenance_record(tmp_path):
    with pytest.raises(FileNotFoundError, match="provenance.md"):
        read_raw(SYNTHETIC_SPEC, tmp_path / "nothing")


def test_an_absent_product_names_the_command_that_makes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest_labelings.py"):
        load_labeling("reanalysis_3pft", tmp_path / "absent.csv")


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("site_id,label\n1,conifer\n1,grass\n", "repeats"),
        ("site_id,label\n2,conifer\n1,grass\n", "ascending"),
        ("site_id,label\n1,conifer\n2,tundra\n", "does not declare"),
        ("site_id,klass\n1,conifer\n", "header is"),
        ("site_id,label\n0,conifer\n", "positive int32"),
    ],
)
def test_load_labeling_refuses_a_file_off_the_data_model(tmp_path, content, match):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_labeling(SYNTHETIC_SPEC, path)


# ── the real file ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def real_sites() -> pd.DataFrame:
    try:
        return load_sites(default_sites_path())
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"site table not available in this working copy: {error}")


@pytest.fixture(scope="session")
def real_product(real_sites, tmp_path_factory) -> pd.DataFrame:
    """``reanalysis_3pft`` built from the tracked raw file."""
    spec = resolve_labeling("reanalysis_3pft")
    if not (RAW_DIR / spec.raw_file).exists():
        pytest.skip("raw labelings not available in this working copy")
    out_dir = tmp_path_factory.mktemp("labelings")
    return ingest.ingest(spec, RAW_DIR, real_sites, out_dir)


def test_the_real_labeling_covers_the_whole_pool(real_product, real_sites):
    assert len(real_product) == len(real_sites)
    assert real_product[SITE_COLUMN].tolist() == real_sites[SITE_COLUMN].tolist()


def test_the_real_labeling_is_exactly_the_landcover_aggregation(real_product, real_sites):
    """The claim data/README.md makes under Site labelings, over all 8000 sites."""
    spec = resolve_labeling("reanalysis_3pft")
    joined = real_product.merge(real_sites[[SITE_COLUMN, "landcover"]], on=SITE_COLUMN)
    expected = joined["landcover"].map(dict(spec.landcover_mapping))
    assert (expected == joined[LABEL_COLUMN].astype(str)).all()
    # And the relation is onto: every class is reached from some cover class.
    assert set(expected) == set(spec.labels)


def test_every_declared_class_is_used_by_the_real_labeling(real_product):
    spec = resolve_labeling("reanalysis_3pft")
    assert set(real_product[LABEL_COLUMN].unique()) == set(spec.labels)


def test_the_real_raw_file_has_the_specs_row_count(real_sites):
    spec = resolve_labeling("reanalysis_3pft")
    if not (RAW_DIR / spec.raw_file).exists():
        pytest.skip("raw labelings not available in this working copy")
    assert len(read_raw(spec, RAW_DIR)) == spec.expected_rows
