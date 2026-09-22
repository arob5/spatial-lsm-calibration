#!/usr/bin/env python
"""Build the processed site labelings, one CSV per labeling.

Overview
--------
For each labeling in ``sipnet_calibration.labelings.LABELINGS``, read its raw
table, check it against the site pool and write the result as
``data/processed/labelings/<name>.csv``. Every decision about what a labeling
holds -- its classes, its raw columns, how many rows the file must have, how it
relates to the site table's ``landcover`` -- is a field of the labeling's spec
in the library; this script is the orchestration and the checks, and its own
round-trip check reads each file back with
:func:`sipnet_calibration.labelings.load_labeling`, the same function every
consumer uses.

Input data
----------
``--raw-root``, default ``data/raw/labelings/``
    One CSV per labeling, named by ``spec.raw_file``, read exactly by
    :func:`sipnet_calibration.labelings.read_raw`. See
    ``data/raw/labelings/provenance.md`` for where each came from.

``--sites``, default ``data/processed/sites/sites.csv``
    The site table: the pool a labeling must label, and the ``landcover``
    column a ``landcover_mapping`` is checked against.

Output data
-----------
``--out-dir``, default ``data/processed/labelings/``, one CSV per labeling::

    site_id,label
    1,semiarid.grassland_HPDA
    2,semiarid.grassland_HPDA

ascending by ``site_id``, with ``label`` one of the spec's declared classes.
``sipnet_calibration.labelings`` documents the data model.

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

Usage
-----
::

    python scripts/ingest_labelings.py                        # every labeling
    python scripts/ingest_labelings.py --labeling reanalysis_3pft
    python scripts/ingest_labelings.py --describe             # the specs, no I/O
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sipnet_calibration.labelings import (
    LABEL_COLUMN,
    LABELING_NAMES,
    SITE_COLUMN,
    LabelingSpec,
    build_labeling,
    default_labelings_dir,
    default_raw_dir,
    describe,
    labeling_path,
    load_labeling,
    read_raw,
    resolve_labeling,
)
from sipnet_calibration.sites import default_sites_path, load_sites


class IngestError(Exception):
    """A raw file does not satisfy an invariant the product depends on."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.labeling or list(LABELING_NAMES)

    if args.describe:
        print("\n\n".join(describe(resolve_labeling(name)) for name in names))
        return 0

    raw_root = args.raw_root if args.raw_root is not None else default_raw_dir()
    out_dir = args.out_dir if args.out_dir is not None else default_labelings_dir()
    sites_path = args.sites if args.sites is not None else default_sites_path()

    try:
        sites = load_sites(sites_path)
        for name in names:
            spec = resolve_labeling(name)
            product = ingest(spec, raw_root, sites, out_dir)
            print(describe_product(spec, product, sites, labeling_path(spec, out_dir)))
    except (IngestError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--labeling",
        action="append",
        choices=LABELING_NAMES,
        metavar="NAME",
        help="A labeling to build; repeatable. Default: all of " + ", ".join(LABELING_NAMES),
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print each labeling's spec and exit without reading data.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Directory of the raw files. Default: data/raw/labelings.",
    )
    parser.add_argument(
        "--sites",
        type=Path,
        default=None,
        help="The site table. Default: data/processed/sites/sites.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write. Default: data/processed/labelings.",
    )
    return parser.parse_args(argv)


# ── the ingest steps, in the order main calls them ────────────────────────────


def ingest(
    spec: LabelingSpec, raw_root: Path, sites: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    """Read, check, build and write one labeling."""
    frame = read_raw(spec, raw_root)
    check_raw_frame(spec, frame, sites)

    product = build_labeling(spec, frame)
    check_product(spec, product, sites)
    write_product(spec, product, labeling_path(spec, out_dir))
    return product


def check_raw_frame(spec: LabelingSpec, frame: pd.DataFrame, sites: pd.DataFrame) -> None:
    """Every check on the raw rows, before anything is built from them."""
    # First, because on the wrong file every later check also fails and says
    # something less useful about why.
    check_row_count_is_the_expected_pool(spec, frame)
    check_no_duplicate_sites(spec, frame)
    check_sites_are_in_the_site_table(spec, frame, sites)


def check_product(spec: LabelingSpec, product: pd.DataFrame, sites: pd.DataFrame) -> None:
    """Every check on the built product, before it is written."""
    check_labels_are_the_declared_set(spec, product)
    check_pool_is_completely_labeled(spec, product, sites)
    check_labels_match_landcover(spec, product, sites)


def write_product(spec: LabelingSpec, product: pd.DataFrame, out: Path) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    product.to_csv(partial, index=False)
    check_round_trip(spec, product, partial)
    partial.replace(out)


def describe_product(
    spec: LabelingSpec, product: pd.DataFrame, sites: pd.DataFrame, path: Path
) -> str:
    """A short report of what was written, for the run log."""
    counts = product[LABEL_COLUMN].value_counts().reindex(list(spec.labels), fill_value=0)
    width = max(len(label) for label in spec.labels)
    latitude = (
        product.merge(sites[[SITE_COLUMN, "lat"]], on=SITE_COLUMN)
        .groupby(LABEL_COLUMN, observed=False)["lat"]
        .agg(["min", "median", "max"])
    )
    lines = [
        f"{path}",
        f"  labeling              : {spec.name} ({spec.label_kind})",
        f"  sites labeled         : {len(product)} of {len(sites)} in the pool",
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


def check_row_count_is_the_expected_pool(spec: LabelingSpec, frame: pd.DataFrame) -> None:
    """The raw file has the spec's row count, so it labels the pool we think.

    This is the check that tells two same-named upstream files apart; see the
    script's Notes and ``data/raw/labelings/provenance.md``.
    """
    if len(frame) == spec.expected_rows:
        return
    raise IngestError(
        f"{spec.raw_file}: holds {len(frame)} rows, expected {spec.expected_rows}. "
        "The likely cause is a file of the right shape but the wrong pool: upstream, "
        "the source of this file has a same-named sibling one directory up labeling "
        "the older 6400-site pool, with the same header and the same class names. "
        "Check which file was copied against data/raw/labelings/provenance.md, or, if "
        f"the pool itself changed, change expected_rows on the {spec.name!r} spec."
    )


def check_no_duplicate_sites(spec: LabelingSpec, frame: pd.DataFrame) -> None:
    """No site is labeled twice; a labeling is a function of the site."""
    site = frame[spec.site_column]
    duplicated = site[site.duplicated()].unique()
    if duplicated.size:
        raise IngestError(
            f"{spec.raw_file}: sites {duplicated[:5].tolist()}"
            f"{' and more' if duplicated.size > 5 else ''} appear more than once. "
            "A labeling gives each site exactly one class."
        )


def check_sites_are_in_the_site_table(
    spec: LabelingSpec, frame: pd.DataFrame, sites: pd.DataFrame
) -> None:
    """Every identifier the raw file labels is a site in the pool."""
    unknown = sorted(set(frame[spec.site_column]) - set(sites[SITE_COLUMN]))
    if unknown:
        raise IngestError(
            f"{spec.raw_file}: labels {len(unknown)} identifiers that are not sites, "
            f"the first being {unknown[:5]}. Site identifiers are a shared key and are "
            "never renumbered, so this is the wrong site pool rather than a table to "
            "extend."
        )


def check_labels_are_the_declared_set(spec: LabelingSpec, product: pd.DataFrame) -> None:
    """Every class the spec declares is used, and no other class appears.

    ``build_labeling`` already refuses an undeclared class. What this adds is
    the other direction: a declared class that no site has usually means the
    spec and the file have drifted apart.
    """
    used = set(product[LABEL_COLUMN].unique())
    missing = [label for label in spec.labels if label not in used]
    if missing:
        raise IngestError(
            f"{spec.raw_file}: declares classes {missing} that no site has. "
            f"Either the file is not the one {spec.name!r} describes, or the spec's "
            "labels should no longer list them."
        )


def check_pool_is_completely_labeled(
    spec: LabelingSpec, product: pd.DataFrame, sites: pd.DataFrame
) -> None:
    """Where the spec says so, every site in the pool has a class."""
    if not spec.covers_pool:
        return
    unlabeled = sorted(set(sites[SITE_COLUMN]) - set(product[SITE_COLUMN]))
    if unlabeled:
        raise IngestError(
            f"{spec.raw_file}: leaves {len(unlabeled)} of {len(sites)} sites unlabeled, "
            f"the first being {unlabeled[:5]}. {spec.name!r} declares covers_pool; a "
            "labeling that is legitimately partial should set it False, and consumers "
            "then have to handle a site with no class."
        )


def check_labels_match_landcover(
    spec: LabelingSpec, product: pd.DataFrame, sites: pd.DataFrame
) -> None:
    """Where the spec records one, the class is that function of ``landcover``.

    The relation is measured rather than stated by the producer, so this is a
    guard against a regenerated raw file quietly departing from it, not a
    derivation of the classes.
    """
    if spec.landcover_mapping is None:
        return
    joined = product.merge(sites[[SITE_COLUMN, "landcover"]], on=SITE_COLUMN, how="left")

    uncovered = sorted(set(joined["landcover"]) - set(spec.landcover_mapping))
    if uncovered:
        raise IngestError(
            f"{spec.raw_file}: the pool uses landcover classes {uncovered}, which "
            f"{spec.name!r}'s landcover_mapping does not cover. Extend the mapping, or "
            "set it to None if the relation no longer holds."
        )

    expected = joined["landcover"].map(dict(spec.landcover_mapping))
    disagreeing = joined[expected != joined[LABEL_COLUMN].astype(str)]
    if not disagreeing.empty:
        first = disagreeing.head(5)
        detail = ", ".join(
            f"site {row[SITE_COLUMN]} landcover {row['landcover']} -> {row[LABEL_COLUMN]}"
            for _, row in first.iterrows()
        )
        raise IngestError(
            f"{spec.raw_file}: {len(disagreeing)} of {len(joined)} sites do not follow "
            f"{spec.name!r}'s landcover_mapping ({detail}). The relation is measured, "
            "not stated by the producer, so a raw file that breaks it is either a new "
            "version of the labeling or the wrong file; see data/README.md open "
            "question 24(k)."
        )


def check_round_trip(spec: LabelingSpec, product: pd.DataFrame, partial: Path) -> None:
    """The written file reads back through the library loader as what was built."""
    written = load_labeling(spec, partial)
    try:
        pd.testing.assert_frame_equal(written, product)
    except AssertionError as error:
        raise IngestError(
            f"{partial}: the written file does not read back identical to what was "
            f"built: {error}. The partial file is left in place for inspection."
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
