#!/usr/bin/env python
"""Split the 16-class PFT assignment into site labels and a covariate table.

Overview
--------
The producer's ``final_8000_sites_with_final_pft_v4.csv`` is one 60-column
table holding two different things: the site labels with the workings of how
each label was derived, and a set of environmental covariates assembled to
derive it. This script cuts it in two along the column partition in
:data:`SITE_LABELS_COLUMNS`, writing both halves as tracked raw inputs.

Like ``convert_initial_conditions.py`` beside it, this **creates** a raw input
rather than processing one, and is not part of the ingest pipeline. A normal
working copy never runs it: it is run once where the source is, and re-run only
if the producer issues a new version.

The split is the reason the script exists rather than a pair of shell commands.
Neither output can be compared against the upstream md5 once the columns are
cut, so what stands in for that check is this script plus the assertions it
makes: that the two column sets partition the source exactly, that both are
keyed on the same complete site set, and that re-joining them reproduces the
source cell for cell.

Input data
----------
``--source``, default the path in :data:`DEFAULT_SOURCE`
    The producer's table: 8000 rows by 60 columns, keyed on ``index``, which
    holds this project's 1-8000 site identifiers.

Output data
-----------
``--site-labels-dir``, default ``data/raw/site_labels/``
    ``site_pft_16class.csv``: ``index``, ``final_pft`` and the columns
    recording how each label was assigned, in the source's own column order
    and with its values written through unchanged.

``--covariates-dir``, default ``data/raw/covariates/``
    ``site_covariates_pft_assignment.csv``: ``index`` and every other column,
    likewise unchanged.

Both are written to a ``.partial`` path and renamed only once the round trip
has been checked, so a failed run cannot leave a corrupt file where a tracked
one belongs.

Notes
-----
**Nothing is converted, renamed, reordered or rounded.** The two halves carry
the producer's column names and the source's own text for every cell: the
source is re-read with ``dtype=str`` and ``keep_default_na=False``, so a float
is written back as the exact characters it arrived as and no precision question
arises. ``check_the_rejoined_halves_reproduce_the_source`` compares the
re-joined result against those strings, which is what makes "unchanged"
checkable rather than asserted.

**``index`` is in both halves**, and is the only column that is. It is the join
key, so duplicating it is what makes the halves independently usable; every
other column belongs to exactly one.

Usage
-----
::

    python scripts/raw_sources/split_site_pft_16class.py
    python scripts/raw_sources/split_site_pft_16class.py --source /path/to/v4.csv
    python scripts/raw_sources/split_site_pft_16class.py --describe
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

#: Where the producer's table lives on the SCC.
DEFAULT_SOURCE = Path(
    "/projectnb/dietzelab/guYANG/SIPNET_Model_Calibration/Final_PFT_assignment_v4"
    "/final_8000_sites_with_final_pft_v4.csv"
)

#: The column both halves carry, and the only one they share.
KEY_COLUMN = "index"

#: The class column, the reason the site-labels half exists.
LABEL_COLUMN = "final_pft"

#: Columns that go to the site-labels half: the key, the class, and the record of
#: how each label was arrived at. Everything else is a covariate.
#:
#: The workings travel with the label rather than with the covariates because
#: they are about the label: 363 of the 8000 sites were assigned by nearest
#: ecological profile rather than directly, and `second_nearest_final_pft` and
#: `distance_margin` are what make the sensitivity of a result to those sites
#: measurable instead of guesswork.
SITE_LABELS_COLUMNS = (
    KEY_COLUMN,
    LABEL_COLUMN,
    "final_pft_direct",
    "final_source_type",
    "source_file",
    "pam_used_for_clustering",
    "pam_k",
    "pam_cluster",
    "subpft_name",
    "dbf_original_pam_cluster",
    "cropland_final_group",
    "final_pft_assignment_method",
    "final_pft_source_final",
    "nearest_ecological_distance",
    "second_nearest_final_pft",
    "second_nearest_ecological_distance",
    "distance_margin",
    "n_vars_used_in_distance",
)

#: Output file names, which are what the provenance records name.
SITE_LABELS_FILE = "site_pft_16class.csv"
COVARIATES_FILE = "site_covariates_pft_assignment.csv"

#: The site pool the table is indexed against.
POOL = range(1, 8001)


class SplitError(Exception):
    """The source does not satisfy an invariant the split depends on."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.describe:
        print(describe())
        return 0

    try:
        source = read_source(args.source)
        site_labels, covariates = split(source)
        check_the_halves(source, site_labels, covariates)
        written = write_halves(site_labels, covariates, args.site_labels_dir, args.covariates_dir)
        print(report(args.source, source, site_labels, covariates, written))
    except (SplitError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE, help=f"Default: {DEFAULT_SOURCE}"
    )
    parser.add_argument(
        "--site-labels-dir",
        type=Path,
        default=Path("data/raw/site_labels"),
        help="Where the site-labels half goes. Default: data/raw/site_labels.",
    )
    parser.add_argument(
        "--covariates-dir",
        type=Path,
        default=Path("data/raw/covariates"),
        help="Where the covariate half goes. Default: data/raw/covariates.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the column partition and exit without reading anything.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def read_source(path: Path) -> pd.DataFrame:
    """The producer's table, every cell as the text the file holds.

    Reading as text is what lets the split be verbatim: no float is parsed, so
    none can be written back at a different precision, and no empty field
    becomes a ``NaN`` that would be written as ``""`` rather than what was
    there.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The source lives on the SCC; see "
            "data/raw/site_labels/provenance.md for the path and the md5."
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False)
    if frame.empty:
        raise SplitError(f"{path}: holds no rows")
    return frame


def split(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The source cut in two along :data:`SITE_LABELS_COLUMNS`.

    Both halves keep the source's column order, so a reader comparing either
    against the original sees the columns in the order the producer wrote them.
    """
    check_the_source_has_the_expected_columns(source)
    site_labels = [column for column in source.columns if column in set(SITE_LABELS_COLUMNS)]
    covariate = [
        column
        for column in source.columns
        if column not in set(SITE_LABELS_COLUMNS) or column == KEY_COLUMN
    ]
    return source[site_labels].copy(), source[covariate].copy()


def check_the_halves(
    source: pd.DataFrame, site_labels: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    """Every check on the split, before either half is written."""
    check_the_columns_partition_the_source(source, site_labels, covariates)
    check_both_halves_are_keyed_on_the_whole_pool(site_labels, covariates)
    check_the_label_column_is_complete(site_labels)
    check_the_rejoined_halves_reproduce_the_source(source, site_labels, covariates)


def write_halves(
    site_labels: pd.DataFrame,
    covariates: pd.DataFrame,
    site_labels_dir: Path,
    covariates_dir: Path,
) -> dict[str, Path]:
    """Write each half to a ``.partial`` path, check the round trip, rename."""
    written = {}
    for frame, directory, name in (
        (site_labels, site_labels_dir, SITE_LABELS_FILE),
        (covariates, covariates_dir, COVARIATES_FILE),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / name
        partial = out.with_suffix(out.suffix + ".partial")
        frame.to_csv(partial, index=False)
        check_the_written_file_reads_back(frame, partial)
        partial.replace(out)
        written[name] = out
    return written


def report(
    source_path: Path,
    source: pd.DataFrame,
    site_labels: pd.DataFrame,
    covariates: pd.DataFrame,
    written: dict[str, Path],
) -> str:
    """What was read and written, with the md5s the provenance records want."""
    lines = [
        f"source : {source_path}",
        f"         {len(source)} rows x {len(source.columns)} columns, "
        f"md5 {md5(source_path)}",
        "",
    ]
    for name, path in written.items():
        frame = site_labels if name == SITE_LABELS_FILE else covariates
        lines.append(
            f"wrote  : {path}\n"
            f"         {len(frame)} rows x {len(frame.columns)} columns, "
            f"{path.stat().st_size:,} bytes, md5 {md5(path)}"
        )
    lines += [
        "",
        f"columns: {len(site_labels.columns)} + {len(covariates.columns)} - 1 shared key "
        f"= {len(source.columns)}",
        "         the halves re-join to the source cell for cell",
    ]
    return "\n".join(lines)


def describe() -> str:
    """The column partition, without reading anything."""
    return (
        f"{SITE_LABELS_FILE}: {len(SITE_LABELS_COLUMNS)} columns\n  "
        + "\n  ".join(SITE_LABELS_COLUMNS)
        + f"\n\n{COVARIATES_FILE}: {KEY_COLUMN} and every other column of the source."
    )


# ── supporting helpers ────────────────────────────────────────────────────────


def md5(path: Path) -> str:
    """The md5 of a file, for the provenance record."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ── checks ────────────────────────────────────────────────────────────────────


def check_the_source_has_the_expected_columns(source: pd.DataFrame) -> None:
    """The source carries every column the site-labels half claims."""
    absent = [column for column in SITE_LABELS_COLUMNS if column not in source.columns]
    if absent:
        raise SplitError(
            f"the source does not carry {absent}, which SITE_LABELS_COLUMNS names. "
            "A new version of the producer's table is a change to that constant, "
            "not something to split around."
        )


def check_the_columns_partition_the_source(
    source: pd.DataFrame, site_labels: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    """Every source column lands in exactly one half, bar the shared key."""
    union = set(site_labels.columns) | set(covariates.columns)
    if union != set(source.columns):
        raise SplitError(
            f"the halves do not cover the source: missing "
            f"{sorted(set(source.columns) - union)}, extra {sorted(union - set(source.columns))}"
        )
    shared = set(site_labels.columns) & set(covariates.columns)
    if shared != {KEY_COLUMN}:
        raise SplitError(
            f"the halves share {sorted(shared)}; they should share only {KEY_COLUMN!r}. "
            "A column in both is a column that can drift between them."
        )


def check_both_halves_are_keyed_on_the_whole_pool(
    site_labels: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    """Both halves hold every site once, so the join cannot lose or duplicate a row."""
    for name, frame in ((SITE_LABELS_FILE, site_labels), (COVARIATES_FILE, covariates)):
        key = frame[KEY_COLUMN].astype(int)
        if key.duplicated().any():
            raise SplitError(f"{name}: {KEY_COLUMN} repeats")
        if sorted(key) != list(POOL):
            raise SplitError(
                f"{name}: {KEY_COLUMN} is not the whole site pool {POOL.start}-{POOL.stop - 1}"
            )


def check_the_label_column_is_complete(site_labels: pd.DataFrame) -> None:
    """No site is left without a class; the site labels have no unlabeled state."""
    blank = site_labels[site_labels[LABEL_COLUMN].str.strip() == ""]
    if not blank.empty:
        raise SplitError(
            f"{SITE_LABELS_FILE}: {len(blank)} sites have an empty {LABEL_COLUMN}, "
            f"the first being {blank[KEY_COLUMN].head(5).tolist()}"
        )


def check_the_rejoined_halves_reproduce_the_source(
    source: pd.DataFrame, site_labels: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    """Re-joining the halves gives the source back, cell for cell.

    This is what stands in for the md5 comparison a verbatim copy would get:
    the split cannot be compared against the upstream file once the columns are
    cut, so instead it is shown to be lossless.
    """
    # A left join from the site-labels half, which is a column slice of the source
    # and so still in its row order; an outer join sorts on the key and would
    # compare a re-ordered frame against the original. That both halves hold
    # each site exactly once is established separately, so nothing is hidden by
    # joining this way.
    rejoined = site_labels.merge(covariates, on=KEY_COLUMN, how="left", validate="1:1")
    if len(rejoined) != len(source):
        raise SplitError(
            f"re-joining gives {len(rejoined)} rows against the source's {len(source)}"
        )
    rejoined = rejoined[list(source.columns)]
    try:
        pd.testing.assert_frame_equal(rejoined, source)
    except AssertionError as error:
        raise SplitError(f"the halves do not re-join to the source: {error}") from error


def check_the_written_file_reads_back(frame: pd.DataFrame, partial: Path) -> None:
    """The file on disk parses back to exactly what was written."""
    written = pd.read_csv(partial, dtype=str, keep_default_na=False, index_col=False)
    try:
        pd.testing.assert_frame_equal(written, frame.reset_index(drop=True))
    except AssertionError as error:
        raise SplitError(
            f"{partial}: does not read back as what was written: {error}. "
            "The partial file is left in place for inspection."
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
