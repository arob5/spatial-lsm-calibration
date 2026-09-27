#!/usr/bin/env python
"""Build the processed site labels, one CSV per site-labels data source.

Overview
--------
For each site-labels data source in
``sipnet_calibration.site_labels.SITE_LABELS``, read its raw table, check it
against the site pool and write the result as
``data/processed/site_labels/<name>.csv``. Every decision about what a source
holds -- its classes, its raw columns, how many rows the file must have, how it
relates to the site table's ``landcover`` -- is set in the source's spec in the
library; this script is the orchestration and the checks.

Input data
----------
``--raw-directory``, default :func:`sipnet_calibration.site_labels.default_raw_dir`
    One CSV per site-labels data source, named by ``spec.raw_file``, read
    exactly by :func:`sipnet_calibration.site_labels.read_raw`. See
    ``data/raw/site_labels/provenance.md`` for where each came from.

``--site-table``, default :func:`sipnet_calibration.sites.default_sites_path`
    The site table: the pool a site-labels data source must label, and the
    ``landcover`` column a ``landcover_mapping`` is checked against.

Output data
-----------
``--output-directory``, default
:func:`sipnet_calibration.site_labels.default_site_labels_dir`
    One processed file per site-labels data source, in the data model
    :mod:`sipnet_calibration.site_labels` documents: ``site_id`` and
    ``label``, ascending by ``site_id``, each label one of the spec's classes.

Notes
-----
**Two upstream files are named ``site_pft.csv``**, one directory apart, with the
same header and the same three class names; one labels the 8000-site pool and
one the older 6400-site pool. Nothing inside either announces which it is, so
``check_row_count_is_the_expected_pool`` refuses a file whose row count is not
the spec's and names the other pool in the message. It runs before anything
else, because every later check would also fail on the wrong file and would say
something less useful about why.

The ingest changes structure, never values: the class names are written exactly
as the producer wrote them, because they are the join key to the reanalysis's
per-PFT trait tables. What it does change is the column names, which are this
project's to choose, and the row order, which becomes ascending by ``site_id``.

Each file is written through :func:`sipnet_calibration.io.write_checked`, and
its check reads it back with
:func:`sipnet_calibration.site_labels.load_site_labels`, the function every
consumer uses.

Usage
-----
::

    uv run python scripts/ingest_site_labels.py              # every source
    uv run python scripts/ingest_site_labels.py --site-labels reanalysis_3pft
    uv run python scripts/ingest_site_labels.py --describe   # the specs, no I/O
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sipnet_calibration.conventions import LAT, SITE_ID
from sipnet_calibration.io import write_checked
from sipnet_calibration.site_labels import (
    LABEL_COLUMN,
    SITE_LABELS_NAMES,
    SiteLabelsSpec,
    build_site_labels,
    default_raw_dir,
    default_site_labels_dir,
    describe,
    load_site_labels,
    read_raw,
    resolve_site_labels,
    site_labels_path,
)
from sipnet_calibration.sites import (
    check_site_table_lists_the_sites,
    default_sites_path,
    load_sites,
    site_lookup,
)
from sipnet_calibration.validation import truncated

#: The site table's column a ``landcover_mapping`` is a function of.
LANDCOVER_COLUMN = "landcover"


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Build every site-labels data source asked for, or describe their specs."""
    args = parse_args(argv)
    names = args.site_labels or list(SITE_LABELS_NAMES)

    if args.describe:
        print("\n\n".join(describe(resolve_site_labels(name)) for name in names))
        return 0

    raw_directory = args.raw_directory or default_raw_dir()
    output_directory = args.output_directory or default_site_labels_dir()
    try:
        site_table = load_sites(args.site_table or default_sites_path())
        for name in names:
            spec = resolve_site_labels(name)
            site_labels = ingest(spec, raw_directory, site_table, output_directory)
            path = site_labels_path(spec, output_directory)
            print(describe_processed_file(site_labels, path, spec, site_table))
    except (IngestError, OSError, ValueError, LookupError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, as the module docstring's Usage describes it."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--site-labels",
        action="append",
        choices=SITE_LABELS_NAMES,
        metavar="NAME",
        help="A site-labels data source to build; repeatable. Default: all of "
        + ", ".join(SITE_LABELS_NAMES),
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print each site-labels data source's spec and exit without reading "
        "data.",
    )
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=None,
        help=f"Directory of the raw files. Default: {default_raw_dir()}.",
    )
    parser.add_argument(
        "--site-table",
        type=Path,
        default=None,
        help=f"The site table. Default: {default_sites_path()}.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=f"Where to write. Default: {default_site_labels_dir()}.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def ingest(
    spec: SiteLabelsSpec, raw_directory: Path, site_table: pd.DataFrame, output_directory: Path
) -> pd.DataFrame:
    """Read, check, build and write one site-labels data source; what was written."""
    frame = read_raw(spec, raw_directory)
    check_raw_frame_is_valid(spec, frame, site_table)
    site_labels = build_site_labels(spec, frame)
    check_site_labels_are_valid(spec, site_labels, site_table)
    write_processed_file(site_labels, site_labels_path(spec, output_directory), spec)
    return site_labels


def describe_processed_file(
    site_labels: pd.DataFrame, path: Path, spec: SiteLabelsSpec, site_table: pd.DataFrame
) -> str:
    """A short report of what was written, for the run log."""
    counts = site_labels[LABEL_COLUMN].value_counts().reindex(list(spec.labels), fill_value=0)
    width = max(len(label) for label in spec.labels)
    latitude = (
        site_labels.assign(
            **{LAT: site_lookup(site_table).loc[site_labels[SITE_ID], LAT].to_numpy()}
        )
        .groupby(LABEL_COLUMN, observed=False)[LAT]
        .agg(["min", "median", "max"])
    )
    lines = [
        f"{path}",
        f"  site labels           : {spec.name} ({spec.label_kind})",
        f"  sites labeled         : {len(site_labels)} of {len(site_table)} in the pool",
        f"  classes               : {len(spec.labels)}",
    ]
    for label in spec.labels:
        row = latitude.loc[label]
        lines.append(
            f"    {label:<{width}} : {counts[label]:>5} sites, "
            f"latitude {row['min']:.1f} / {row['median']:.1f} / {row['max']:.1f}"
        )
    if spec.landcover_mapping is not None:
        lines.append("  landcover relation    : holds for every site")
    return "\n".join(lines)


# ── supporting types and helpers ──────────────────────────────────────────────


class IngestError(RuntimeError):
    """A raw or written file breaks an invariant the processed file depends on."""


def write_processed_file(site_labels: pd.DataFrame, path: Path, spec: SiteLabelsSpec) -> None:
    """Write through a ``.partial`` file, moved in once it reads back identical."""
    write_checked(
        path,
        write=lambda partial: site_labels.to_csv(partial, index=False),
        check=lambda partial: check_written_file_reads_back_identically(
            spec, site_labels, partial
        ),
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_raw_frame_is_valid(
    spec: SiteLabelsSpec, frame: pd.DataFrame, site_table: pd.DataFrame
) -> None:
    """The raw rows are fit to build the site labels from."""
    # First, because on the wrong file every later check also fails and says
    # something less useful about why.
    check_row_count_is_the_expected_pool(spec, frame)
    check_raw_sites_are_each_labeled_once(spec, frame)
    check_site_table_lists_the_sites(
        site_table, frame[spec.site_column].tolist(), message_name=f"{spec.raw_file}: site(s)"
    )


def check_site_labels_are_valid(
    spec: SiteLabelsSpec, site_labels: pd.DataFrame, site_table: pd.DataFrame
) -> None:
    """The built site labels hold to their spec, before they are written."""
    check_declared_labels_are_all_used(spec, site_labels)
    check_pool_is_completely_labeled(spec, site_labels, site_table)
    check_labels_match_landcover(spec, site_labels, site_table)


def check_row_count_is_the_expected_pool(spec: SiteLabelsSpec, frame: pd.DataFrame) -> None:
    """The raw file has the spec's row count, so it labels the pool the spec means."""
    if len(frame) == spec.expected_rows:
        return
    raise IngestError(
        f"{spec.raw_file}: holds {len(frame)} rows, expected {spec.expected_rows}; check "
        "which file was copied against data/raw/site_labels/provenance.md (upstream, the "
        "source of this file has a same-named sibling one directory up covering the older "
        "6400-site pool, with the same header and the same class names), or, if the pool "
        f"itself changed, change expected_rows on the {spec.name!r} spec."
    )


def check_raw_sites_are_each_labeled_once(spec: SiteLabelsSpec, frame: pd.DataFrame) -> None:
    """No site of the raw file is labeled twice; the class is a function of the site."""
    site = frame[spec.site_column]
    duplicated = site[site.duplicated()].unique()
    if duplicated.size:
        raise IngestError(
            f"{spec.raw_file}: sites {truncated(duplicated.tolist())} appear more than "
            f"once, and a site-labels data source gives each site exactly one class; re-copy it and compare it with data/raw/site_labels/provenance.md."
        )


def check_declared_labels_are_all_used(spec: SiteLabelsSpec, site_labels: pd.DataFrame) -> None:
    """Every class the spec declares is used by some site."""
    # build_site_labels already refuses an undeclared class. What this adds is
    # the other direction: a declared class that no site has usually means the
    # spec and the file have drifted apart.
    used = set(site_labels[LABEL_COLUMN].unique())
    missing = [label for label in spec.labels if label not in used]
    if missing:
        raise IngestError(
            f"{spec.raw_file}: declares classes {truncated(missing)} that no site has; either the "
            f"file is not the one {spec.name!r} describes, or the spec's labels should no "
            "longer list them."
        )


def check_pool_is_completely_labeled(
    spec: SiteLabelsSpec, site_labels: pd.DataFrame, site_table: pd.DataFrame
) -> None:
    """Where the spec says so, every site in the pool has a class."""
    if not spec.covers_pool:
        return
    unlabeled = sorted(set(site_table[SITE_ID]) - set(site_labels[SITE_ID]))
    if unlabeled:
        raise IngestError(
            f"{spec.raw_file}: leaves {len(unlabeled)} of {len(site_table)} sites "
            f"unlabeled, {truncated(unlabeled)}; {spec.name!r} declares covers_pool, so "
            "a source that is legitimately partial should set it False, and consumers "
            "then have to handle a site with no class."
        )


def check_labels_match_landcover(
    spec: SiteLabelsSpec, site_labels: pd.DataFrame, site_table: pd.DataFrame
) -> None:
    """Where the spec records one, the class is its function of ``landcover``."""
    if spec.landcover_mapping is None:
        return
    # Every labeled site is in the site table: check_raw_frame_is_valid checked it.
    landcover = site_lookup(site_table).loc[site_labels[SITE_ID], LANDCOVER_COLUMN].to_numpy()
    joined = site_labels.assign(**{LANDCOVER_COLUMN: landcover})
    check_landcover_mapping_covers_the_pool(spec, joined)
    check_labels_follow_the_landcover_mapping(spec, joined)


def check_landcover_mapping_covers_the_pool(spec: SiteLabelsSpec, joined: pd.DataFrame) -> None:
    """The spec's ``landcover_mapping`` has an entry for every landcover class used."""
    uncovered = sorted(set(joined[LANDCOVER_COLUMN]) - set(spec.landcover_mapping))
    if uncovered:
        raise IngestError(
            f"{spec.raw_file}: the pool uses landcover classes {truncated(uncovered)}, which "
            f"{spec.name!r}'s landcover_mapping does not cover; extend the mapping, or "
            "set it to None if the relation no longer holds."
        )


def check_labels_follow_the_landcover_mapping(spec: SiteLabelsSpec, joined: pd.DataFrame) -> None:
    """Every site's class is the one its landcover maps to."""
    # The relation is measured rather than stated by the producer, so this
    # guards against a regenerated raw file quietly departing from it; it does
    # not derive the classes.
    expected = joined[LANDCOVER_COLUMN].map(spec.landcover_mapping)
    disagreeing = joined[expected != joined[LABEL_COLUMN].astype(str)]
    if not disagreeing.empty:
        detail = truncated(
            f"site {row[SITE_ID]} landcover {row[LANDCOVER_COLUMN]} -> {row[LABEL_COLUMN]}"
            for _, row in disagreeing.iterrows()
        )
        raise IngestError(
            f"{spec.raw_file}: {len(disagreeing)} of {len(joined)} sites do not follow "
            f"{spec.name!r}'s landcover_mapping, {detail}; the relation is measured, not "
            "stated by the producer, so a raw file that breaks it is either a new version "
            "of the upstream product or the wrong file (data/README.md open question 24(k)); "
            f"re-copy it and compare it with data/raw/site_labels/provenance.md."
        )


def check_written_file_reads_back_identically(
    spec: SiteLabelsSpec, site_labels: pd.DataFrame, partial: Path
) -> None:
    """The written file reads back through the library loader as what was built."""
    check_read_back_is_identical(
        site_labels, load_site_labels(spec, partial), message_name=str(partial)
    )


def check_read_back_is_identical(
    site_labels: pd.DataFrame, read_back: pd.DataFrame, *, message_name: str
) -> None:
    """The table read back is identical to the one it was written from."""
    try:
        pd.testing.assert_frame_equal(read_back, site_labels)
    except AssertionError as error:
        raise IngestError(
            f"{message_name}: the written file does not read back identical to what was "
            f"built ({error}); inspect the kept partial file."
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
