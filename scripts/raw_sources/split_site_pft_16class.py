#!/usr/bin/env python
"""Split the 16-class PFT assignment into site labels and a covariate table.

Overview
--------
The producer's ``final_8000_sites_with_final_pft_v4.csv`` is one table holding
two different things: the site labels with the workings of how each label was
derived, and a set of environmental covariates assembled to derive it. This
script cuts it in two along the column partition in
:data:`SITE_LABELS_COLUMN_NAMES`, writing both halves as tracked raw inputs.

Input data
----------
``--source``, default the path in :data:`DEFAULT_SOURCE`
    The producer's table: one row per site of the pool
    (:data:`sipnet_calibration.sites.N_SITES`), keyed on
    ``index``, which holds this project's site ids.

Output data
-----------
``--site-labels-directory``, default the repository's ``data/raw/site_labels/``
    ``site_pft_16class.csv``: ``index``, ``final_pft`` and the columns
    recording how each label was assigned, in the source's own column order
    and with its values written through unchanged.

``--covariates-directory``, default the repository's ``data/raw/covariates/``
    ``site_covariates_pft_assignment.csv``: ``index`` and every other column,
    likewise unchanged.

Notes
-----
**Not part of the ingest pipeline.** Like ``convert_initial_conditions.py``
beside it, this *creates* a raw input rather than processing one. A normal
working copy never runs it: it is run once where the source is, and re-run
only if the producer issues a new version.

**The split is why the script exists** rather than a pair of shell commands.
Neither output can be compared against the upstream md5 once the columns are
cut, so what stands in for that check is this script plus its checks: that
the two column sets partition the source exactly, that both are keyed on the
same complete site set, and that re-joining them reproduces the source value
for value.

**Nothing is converted, renamed, reordered or rounded.** The two halves carry
the producer's column names and the source's own text for every CSV field: the
source is read with ``dtype=str`` and ``keep_default_na=False``, so a float is
written back as the exact characters it arrived as and no precision question
arises.

**``index`` is in both halves**, and is the only column that is. It is the join
key, so duplicating it is what makes the halves independently usable; every
other column belongs to exactly one.

The two are written together through
:func:`sipnet_calibration.io.write_checked_together`, so neither is moved into
place unless both read back.

Usage
-----
::

    uv run python scripts/raw_sources/split_site_pft_16class.py
    uv run python scripts/raw_sources/split_site_pft_16class.py --source /path/to/v4.csv
    uv run python scripts/raw_sources/split_site_pft_16class.py --describe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sipnet_calibration.io import file_md5, write_checked_together
from sipnet_calibration.sites import N_SITES
from sipnet_calibration.validation import truncated

#: Where the producer's table lives on the SCC.
DEFAULT_SOURCE = Path(
    "/projectnb/dietzelab/guYANG/SIPNET_Model_Calibration/Final_PFT_assignment_v4"
    "/final_8000_sites_with_final_pft_v4.csv"
)

#: The column both halves carry, and the only one they share.
KEY_COLUMN = "index"

#: The class column, the reason the site-labels half exists.
CLASS_COLUMN = "final_pft"

#: Columns that go to the site-labels half: the key, the class, and the record of
#: how each label was arrived at. Everything else is a covariate.
#:
#: The workings travel with the label rather than with the covariates because
#: they are about the label: some sites were assigned by nearest ecological
#: profile rather than directly, and `second_nearest_final_pft` and
#: `distance_margin` are what make the sensitivity of a result to those sites
#: measurable instead of guesswork.
SITE_LABELS_COLUMN_NAMES = (
    KEY_COLUMN,
    CLASS_COLUMN,
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

#: The repository, where the tracked inputs are.
REPOSITORY = Path(__file__).resolve().parents[2]

#: Where the site-labels half goes by default: this repository's tracked raw
#: input, whatever the working directory or ``$SIPNET_CALIBRATION_DATA``.
DEFAULT_SITE_LABELS_DIRECTORY = REPOSITORY / "data" / "raw" / "site_labels"

#: Where the covariate half goes by default, found from the checkout likewise.
DEFAULT_COVARIATES_DIRECTORY = REPOSITORY / "data" / "raw" / "covariates"

#: The site-labels half's file name, which the provenance records name.
SITE_LABELS_FILE_NAME = "site_pft_16class.csv"

#: The covariate half's file name, which the provenance records name.
COVARIATES_FILE_NAME = "site_covariates_pft_assignment.csv"


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Split the source and write both halves, or describe the partition."""
    args = parse_args(argv)
    if args.describe:
        print(describe_partition())
        return 0

    try:
        source = read_source(args.source)
        site_labels, covariates = split_source(source)
        check_halves_are_valid(source, site_labels, covariates)
        written = write_halves(
            site_labels, covariates, args.site_labels_directory, args.covariates_directory
        )
        print(describe_written_halves(args.source, source, site_labels, covariates, written))
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
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"The producer's table. Default: {DEFAULT_SOURCE}."
    )
    parser.add_argument(
        "--site-labels-directory",
        type=Path,
        default=DEFAULT_SITE_LABELS_DIRECTORY,
        help=f"Where the site-labels half goes. Default: {DEFAULT_SITE_LABELS_DIRECTORY}.",
    )
    parser.add_argument(
        "--covariates-directory",
        type=Path,
        default=DEFAULT_COVARIATES_DIRECTORY,
        help=f"Where the covariate half goes. Default: {DEFAULT_COVARIATES_DIRECTORY}.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the column partition and exit without reading anything.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def describe_partition() -> str:
    """The column partition, without reading anything."""
    return (
        f"{SITE_LABELS_FILE_NAME}: {len(SITE_LABELS_COLUMN_NAMES)} columns\n  "
        + "\n  ".join(SITE_LABELS_COLUMN_NAMES)
        + f"\n\n{COVARIATES_FILE_NAME}: {KEY_COLUMN} and every other column of the source."
    )


def read_source(path: Path) -> pd.DataFrame:
    """The producer's table, every CSV field as the text the file holds."""
    # Reading as text is what lets the split be verbatim: no float is parsed, so
    # none can be written back at a different precision, and no empty field
    # becomes a NaN that would be written as "" rather than what was there.
    check_source_exists(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False)
    check_source_has_rows(frame, message_name=str(path))
    return frame


def split_source(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The site-labels half and the covariate half, each in the source's order."""
    check_source_has_the_site_labels_columns(source)
    site_labels_column_names = [
        column for column in source.columns if column in set(SITE_LABELS_COLUMN_NAMES)
    ]
    covariate_column_names = [
        column
        for column in source.columns
        if column not in set(SITE_LABELS_COLUMN_NAMES) or column == KEY_COLUMN
    ]
    return source[site_labels_column_names].copy(), source[covariate_column_names].copy()


def write_halves(
    site_labels: pd.DataFrame,
    covariates: pd.DataFrame,
    site_labels_directory: Path,
    covariates_directory: Path,
) -> dict[str, Path]:
    """Write both halves, moved in once both read back; file name -> path written."""
    halves = (
        (site_labels, site_labels_directory / SITE_LABELS_FILE_NAME),
        (covariates, covariates_directory / COVARIATES_FILE_NAME),
    )
    written = write_checked_together(
        [
            (
                path,
                lambda partial, frame=frame: frame.to_csv(partial, index=False),
                lambda partial, frame=frame: check_written_file_reads_back_identically(
                    frame, partial
                ),
            )
            for frame, path in halves
        ]
    )
    return {path.name: path for path in written}


def describe_written_halves(
    source_path: Path,
    source: pd.DataFrame,
    site_labels: pd.DataFrame,
    covariates: pd.DataFrame,
    written: dict[str, Path],
) -> str:
    """What was read and written, with the md5s the provenance records."""
    lines = [
        f"source : {source_path}",
        f"         {len(source)} rows x {len(source.columns)} columns, "
        f"md5 {file_md5(source_path)}",
        "",
    ]
    for name, path in written.items():
        frame = site_labels if name == SITE_LABELS_FILE_NAME else covariates
        lines.append(
            f"wrote  : {path}\n"
            f"         {len(frame)} rows x {len(frame.columns)} columns, "
            f"{path.stat().st_size:,} bytes, md5 {file_md5(path)}"
        )
    lines += [
        "",
        f"columns: {len(site_labels.columns)} + {len(covariates.columns)} - 1 shared key "
        f"= {len(source.columns)}",
        "         the halves re-join to the source value for value",
    ]
    return "\n".join(lines)


# ── supporting types and helpers ──────────────────────────────────────────────


class IngestError(RuntimeError):
    """The source or a written half breaks an invariant the split depends on."""


# ── checks ────────────────────────────────────────────────────────────────────


def check_source_exists(path: Path) -> None:
    """The producer's table exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; the source lives on the SCC, and "
            "data/raw/site_labels/provenance.md gives its path and md5."
        )


def check_source_has_rows(frame: pd.DataFrame, *, message_name: str) -> None:
    """The producer's table holds at least one row."""
    if frame.empty:
        raise IngestError(f"{message_name}: holds no rows; re-copy it and compare it with data/raw/site_labels/provenance.md.")


def check_source_has_the_site_labels_columns(source: pd.DataFrame) -> None:
    """The source carries every column the site-labels half claims."""
    absent = [column for column in SITE_LABELS_COLUMN_NAMES if column not in source.columns]
    if absent:
        raise IngestError(
            f"the source does not carry {truncated(absent)}, which SITE_LABELS_COLUMN_NAMES "
            "names; "
            "a new version of the producer's table is a change to that constant, not "
            "something to split around."
        )


def check_halves_are_valid(
    source: pd.DataFrame, site_labels: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    """The halves are a lossless split of the source, before either is written."""
    check_columns_partition_the_source(source, site_labels, covariates)
    check_both_halves_are_keyed_on_the_whole_pool(site_labels, covariates)
    check_class_column_is_complete(site_labels)
    check_rejoined_halves_reproduce_the_source(source, site_labels, covariates)


def check_columns_partition_the_source(
    source: pd.DataFrame, site_labels: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    """Every source column lands in exactly one half, bar the shared key."""
    check_halves_cover_the_source(source, site_labels, covariates)
    check_halves_share_only_the_key(site_labels, covariates)


def check_halves_cover_the_source(
    source: pd.DataFrame, site_labels: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    """The two halves' columns are exactly the source's."""
    union = set(site_labels.columns) | set(covariates.columns)
    if union != set(source.columns):
        raise IngestError(
            f"the halves do not cover the source: missing "
            f"{truncated(sorted(set(source.columns) - union))}, extra "
            f"{truncated(sorted(union - set(source.columns)))}; split with split_source."
        )


def check_halves_share_only_the_key(site_labels: pd.DataFrame, covariates: pd.DataFrame) -> None:
    """The halves share no column but the key."""
    shared = set(site_labels.columns) & set(covariates.columns)
    if shared != {KEY_COLUMN}:
        raise IngestError(
            f"the halves share {truncated(sorted(shared))}; they should share only {KEY_COLUMN!r}, "
            "since a column in both is a column that can drift between them."
        )


def check_both_halves_are_keyed_on_the_whole_pool(
    site_labels: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    """Both halves hold every site once, so the join cannot lose or duplicate a row."""
    for name, frame in ((SITE_LABELS_FILE_NAME, site_labels), (COVARIATES_FILE_NAME, covariates)):
        key = frame[KEY_COLUMN].astype(int)
        check_key_does_not_repeat(key, message_name=name)
        check_key_is_the_whole_pool(key, message_name=name)


def check_key_does_not_repeat(key: pd.Series, *, message_name: str) -> None:
    """No site id appears twice in a half's key."""
    repeated = sorted(key[key.duplicated()].unique().tolist())
    if repeated:
        raise IngestError(
            f"{message_name}: {KEY_COLUMN} repeats {truncated(repeated)}, and a half holds "
            f"one row per site; re-copy it and compare it with data/raw/site_labels/provenance.md."
        )


def check_key_is_the_whole_pool(key: pd.Series, *, message_name: str) -> None:
    """A half's key is exactly the site pool ``1..N_SITES``."""
    if sorted(key) != list(range(1, N_SITES + 1)):
        raise IngestError(
            f"{message_name}: {KEY_COLUMN} is not the whole site pool 1-{N_SITES}; re-copy it and compare it with data/raw/site_labels/provenance.md."
        )


def check_class_column_is_complete(site_labels: pd.DataFrame) -> None:
    """No site is left without a class; the site labels have no unlabeled state."""
    blank = site_labels[site_labels[CLASS_COLUMN].str.strip() == ""]
    if not blank.empty:
        raise IngestError(
            f"{SITE_LABELS_FILE_NAME}: {len(blank)} sites have an empty {CLASS_COLUMN}, "
            f"{truncated(blank[KEY_COLUMN].tolist())}, and every site of the pool has a "
            f"class; re-copy it and compare it with data/raw/site_labels/provenance.md."
        )


def check_rejoined_halves_reproduce_the_source(
    source: pd.DataFrame, site_labels: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    """Re-joining the halves gives the source back, value for value."""
    # This is what stands in for the md5 comparison a verbatim copy would get.
    # A left join from the site-labels half, which is a column slice of the
    # source and so still in its row order; an outer join sorts on the key and
    # would compare a re-ordered frame against the original. That both halves
    # hold each site exactly once is established separately, so nothing is
    # hidden by joining this way.
    rejoined = site_labels.merge(covariates, on=KEY_COLUMN, how="left", validate="1:1")
    if len(rejoined) != len(source):
        raise IngestError(
            f"re-joining gives {len(rejoined)} rows against the source's {len(source)}; "
            "split with split_source."
        )
    try:
        pd.testing.assert_frame_equal(rejoined[list(source.columns)], source)
    except AssertionError as error:
        raise IngestError(
            f"the halves do not re-join to the source ({error}); split with split_source."
        ) from error


def check_written_file_reads_back_identically(frame: pd.DataFrame, partial: Path) -> None:
    """The file on disk parses back to exactly what was written."""
    written = pd.read_csv(partial, dtype=str, keep_default_na=False, index_col=False)
    try:
        pd.testing.assert_frame_equal(written, frame.reset_index(drop=True))
    except AssertionError as error:
        raise IngestError(
            f"{partial}: does not read back as what was written ({error}); inspect the "
            "kept partial file."
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
