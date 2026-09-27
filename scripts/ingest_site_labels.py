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
library; this script is the orchestration and the checks, and its own round-trip
check reads each file back with
:func:`sipnet_calibration.site_labels.load_site_labels`, the same function every
consumer uses.

Input data
----------
``--raw-root``, default ``data/raw/site_labels/``
    One CSV per site-labels data source, named by ``spec.raw_file``, read
    exactly by :func:`sipnet_calibration.site_labels.read_raw`. See
    ``data/raw/site_labels/provenance.md`` for where each came from.

``--site-table``, default ``data/processed/sites/sites.csv``
    The site table: the pool a site-labels data source must label, and the
    ``landcover`` column a ``landcover_mapping`` is checked against.

Output data
-----------
``--out-dir``, default ``data/processed/site_labels/``, one processed file per
site-labels data source::

    site_id,label
    1,semiarid.grassland_HPDA
    2,semiarid.grassland_HPDA

ascending by ``site_id``, with ``label`` one of the spec's declared classes.
``sipnet_calibration.site_labels`` documents the data model.

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

Output is written to a ``.partial`` path and renamed only once it reads back
identically through the library loader, so a failed check cannot leave a
corrupt file where the canonical one belongs.
A failed check keeps the ``.partial`` file for inspection and prints its
path (:func:`sipnet_calibration.io.write_checked`).

Usage
-----
::

    python scripts/ingest_site_labels.py                       # every source
    python scripts/ingest_site_labels.py --site-labels reanalysis_3pft
    python scripts/ingest_site_labels.py --describe            # the specs, no I/O
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


class IngestError(Exception):
    """A raw file does not satisfy an invariant the processed file depends on."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.site_labels or list(SITE_LABELS_NAMES)

    if args.describe:
        print("\n\n".join(describe(resolve_site_labels(name)) for name in names))
        return 0

    raw_root = args.raw_root if args.raw_root is not None else default_raw_dir()
    out_dir = args.out_dir if args.out_dir is not None else default_site_labels_dir()
    site_table_path = args.site_table or default_sites_path()

    try:
        site_table = load_sites(site_table_path)
        for name in names:
            spec = resolve_site_labels(name)
            site_labels = ingest(spec, raw_root, site_table, out_dir)
            path = site_labels_path(spec, out_dir)
            print(describe_processed_file(spec, site_labels, site_table, path))
    except (IngestError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
        "--raw-root",
        type=Path,
        default=None,
        help="Directory of the raw files. Default: data/raw/site_labels.",
    )
    parser.add_argument(
        "--site-table",
        type=Path,
        default=None,
        help="The site table. Default: data/processed/sites/sites.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write. Default: data/processed/site_labels.",
    )
    return parser.parse_args(argv)


# ── the ingest steps, in the order main calls them ────────────────────────────


def ingest(
    spec: SiteLabelsSpec, raw_root: Path, site_table: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    """Read, check, build and write one site-labels data source."""
    frame = read_raw(spec, raw_root)
    check_raw_frame(spec, frame, site_table)

    site_labels = build_site_labels(spec, frame)
    check_site_labels_are_valid(spec, site_labels, site_table)
    write_processed_file(spec, site_labels, site_labels_path(spec, out_dir))
    return site_labels


def check_raw_frame(spec: SiteLabelsSpec, frame: pd.DataFrame, site_table: pd.DataFrame) -> None:
    """Every check on the raw rows, before anything is built from them."""
    # First, because on the wrong file every later check also fails and says
    # something less useful about why.
    check_row_count_is_the_expected_pool(spec, frame)
    check_no_duplicate_sites(spec, frame)
    check_site_table_lists_the_sites(
        site_table, frame[spec.site_column].tolist(), message_name=f"{spec.raw_file}: site(s)"
    )


def check_site_labels_are_valid(
    spec: SiteLabelsSpec, site_labels: pd.DataFrame, site_table: pd.DataFrame
) -> None:
    """Every check on the built site labels, before they are written."""
    check_labels_are_the_declared_set(spec, site_labels)
    check_pool_is_completely_labeled(spec, site_labels, site_table)
    check_labels_match_landcover(spec, site_labels, site_table)


def write_processed_file(spec: SiteLabelsSpec, site_labels: pd.DataFrame, out: Path) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename."""
    write_checked(
        out,
        write=lambda partial: site_labels.to_csv(partial, index=False),
        check=lambda partial: check_round_trip(spec, site_labels, partial),
    )


def describe_processed_file(
    spec: SiteLabelsSpec, site_labels: pd.DataFrame, site_table: pd.DataFrame, path: Path
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


# ── checks ────────────────────────────────────────────────────────────────────


def check_row_count_is_the_expected_pool(spec: SiteLabelsSpec, frame: pd.DataFrame) -> None:
    """The raw file has the spec's row count, so it labels the pool we think.

    This is the check that tells two same-named upstream files apart; see the
    script's Notes and ``data/raw/site_labels/provenance.md``.
    """
    if len(frame) == spec.expected_rows:
        return
    raise IngestError(
        f"{spec.raw_file}: holds {len(frame)} rows, expected {spec.expected_rows}. "
        "The likely cause is a file of the right shape but the wrong pool: upstream, "
        "the source of this file has a same-named sibling one directory up covering "
        "the older 6400-site pool, with the same header and the same class names. "
        "Check which file was copied against data/raw/site_labels/provenance.md, or, if "
        f"the pool itself changed, change expected_rows on the {spec.name!r} spec."
    )


def check_no_duplicate_sites(spec: SiteLabelsSpec, frame: pd.DataFrame) -> None:
    """No site is labeled twice; the class is a function of the site."""
    site = frame[spec.site_column]
    duplicated = site[site.duplicated()].unique()
    if duplicated.size:
        raise IngestError(
            f"{spec.raw_file}: sites {duplicated[:5].tolist()}"
            f"{' and more' if duplicated.size > 5 else ''} appear more than once. "
            "A site-labels data source gives each site exactly one class."
        )


def check_labels_are_the_declared_set(spec: SiteLabelsSpec, site_labels: pd.DataFrame) -> None:
    """Every class the spec declares is used, and no other class appears.

    ``build_site_labels`` already refuses an undeclared class. What this adds is
    the other direction: a declared class that no site has usually means the
    spec and the file have drifted apart.
    """
    used = set(site_labels[LABEL_COLUMN].unique())
    missing = [label for label in spec.labels if label not in used]
    if missing:
        raise IngestError(
            f"{spec.raw_file}: declares classes {missing} that no site has. "
            f"Either the file is not the one {spec.name!r} describes, or the spec's "
            "labels should no longer list them."
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
            f"{spec.raw_file}: leaves {len(unlabeled)} of {len(site_table)} sites unlabeled, "
            f"the first being {unlabeled[:5]}. {spec.name!r} declares covers_pool; a "
            "source that is legitimately partial should set it False, and consumers "
            "then have to handle a site with no class."
        )


def check_labels_match_landcover(
    spec: SiteLabelsSpec, site_labels: pd.DataFrame, site_table: pd.DataFrame
) -> None:
    """Where the spec records one, the class is that function of ``landcover``.

    The relation is measured rather than stated by the producer, so this is a
    guard against a regenerated raw file quietly departing from it, not a
    derivation of the classes.
    """
    if spec.landcover_mapping is None:
        return
    # Every labeled site is in the site table: check_raw_frame checked it.
    landcover = site_lookup(site_table).loc[site_labels[SITE_ID], "landcover"].to_numpy()
    joined = site_labels.assign(landcover=landcover)

    uncovered = sorted(set(joined["landcover"]) - set(spec.landcover_mapping))
    if uncovered:
        raise IngestError(
            f"{spec.raw_file}: the pool uses landcover classes {uncovered}, which "
            f"{spec.name!r}'s landcover_mapping does not cover. Extend the mapping, or "
            "set it to None if the relation no longer holds."
        )

    expected = joined["landcover"].map(spec.landcover_mapping)
    disagreeing = joined[expected != joined[LABEL_COLUMN].astype(str)]
    if not disagreeing.empty:
        first = disagreeing.head(5)
        detail = ", ".join(
            f"site {row[SITE_ID]} landcover {row['landcover']} -> {row[LABEL_COLUMN]}"
            for _, row in first.iterrows()
        )
        raise IngestError(
            f"{spec.raw_file}: {len(disagreeing)} of {len(joined)} sites do not follow "
            f"{spec.name!r}'s landcover_mapping ({detail}). The relation is measured, "
            "not stated by the producer, so a raw file that breaks it is either a new "
            "version of the upstream product or the wrong file; see data/README.md open "
            "question 24(k)."
        )


def check_round_trip(spec: SiteLabelsSpec, site_labels: pd.DataFrame, partial: Path) -> None:
    """The written file reads back through the library loader as what was built."""
    written = load_site_labels(spec, partial)
    try:
        pd.testing.assert_frame_equal(written, site_labels)
    except AssertionError as error:
        raise IngestError(
            f"{partial}: the written file does not read back identical to what was "
            f"built: {error}."
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
