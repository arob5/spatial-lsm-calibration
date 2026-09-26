#!/usr/bin/env python
"""Survey the MODIS leaf phenology tables, and check what the README records.

Overview
--------
Read a leaf phenology CSV and report what holds across it: its shape and the
site-year rectangle it covers, how the quality flags distribute and whether they
coincide with the missing days, the day-of-year distributions, and how many
site-years carry a leaf-on day at or after their leaf-off day. Then compare the
measurements against the characteristics ``data/README.md`` records for this
file, and **exit non-zero if one no longer holds**.

That last part is what makes this more than a diagnostic. The README records
measured characteristics of raw data, which is its job, but a number written
down and nowhere else goes stale in silence. The numbers live here as
:data:`RECORDED`, the README says what the property is and points here, and a
re-copied or regenerated file that changed is refused rather than absorbed.

This is not part of the ingest pipeline: nothing is written, nothing under
``data/processed/`` depends on it, and no product is built from phenology yet.

Input data
----------
``--path``, default ``data/raw/phenology/leaf_phenology_8k.csv``
    One row per site-year with the columns of :data:`COLUMNS`. ``leafonday``
    and ``leafoffday`` are day-of-year or the literal ``NA``; the two ``_qa``
    columns are integers 0-3.

``--sites``, default ``data/processed/sites/sites.csv``
    The site table, used only to check that the file's identifiers are the
    project's site pool and that its coordinates agree with it. Skipped with
    ``--no-sites``, which is what the NEON companion file needs: it is keyed on
    BETY identifiers and joins to nothing here.

Output data
-----------
A report to stdout and, with ``--out``, the same content as JSON. Nothing is
written to ``data/``.

The exit status is 0 when every recorded characteristic still holds, 1 when one
does not, and 2 when the file could not be read at all. ``--no-check`` reports
without comparing, which is what to use on a file :data:`RECORDED` has no entry
for.

Notes
-----
**The day columns invert on about one site-year in ninety, and the file is
right to.** ``PEcAn.data.remote::extract_phenology_MODIS`` reads two MODIS bands
that are days since 1970-01-01, guards against leaf-on falling after leaf-off on
that scale, and only then converts each with ``lubridate::yday``, which discards
the year. A leaf-off that falls in the following calendar year therefore comes
back as a small day-of-year and the pair inverts, past a guard that was correct
where it ran. The count is surveyed and recorded because anything that
differences the two columns has to handle it.

Usage
-----
::

    python scripts/survey_phenology.py
    python scripts/survey_phenology.py --path data/raw/phenology/leaf_phenology_neon.csv \\
        --no-sites --no-check
    python scripts/survey_phenology.py --out phenology_survey.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sipnet_calibration import conventions
from sipnet_calibration.sites import default_sites_path, load_sites

#: The file's header, in order. Any other header is a different product.
COLUMNS = (
    "year",
    "site_id",
    "lat",
    "lon",
    "leafonday",
    "leafoffday",
    "leafon_qa",
    "leafoff_qa",
)

#: The two day-of-year columns, each with the quality column that grades it.
DAY_COLUMNS = {"leafonday": "leafon_qa", "leafoffday": "leafoff_qa"}

#: What each quality value means, from the MCD12Q2 Collection 6 user guide by
#: way of the comment in ``extract_phenology_MODIS``.
QUALITY_MEANINGS = {0: "best", 1: "good", 2: "fair", 3: "poor"}

#: The characteristics ``data/README.md`` records, per file, keyed on the file
#: name. Measured on 2026-09-21 from the copies described in the README's
#: `Leaf phenology` section. A disagreement is a changed file, not a changed
#: threshold: re-read the file, then update the README and this table together.
RECORDED: dict[str, dict[str, Any]] = {
    "leaf_phenology_8k.csv": {
        "rows": 96_000,
        "sites": 8000,
        "years": list(range(2012, 2024)),
        "complete_rectangle": True,
        "quality_counts": {
            "leafon_qa": {0: 62_475, 1: 2852, 2: 1504, 3: 29_169},
            "leafoff_qa": {0: 65_784, 1: 1393, 2: 757, 3: 28_066},
        },
        "quality_three_is_exactly_missing": True,
        "sites_with_any_day": {"leafonday": 6599, "leafoffday": 6600},
        "median_day": {"leafonday": 149.0, "leafoffday": 262.0},
        "inverted_rows": 731,
        "inverted_sites": 308,
        "inverted_leafoffday_median": 42.0,
        # Only reachable with --sites, which the NEON file cannot use.
        "identifiers_not_in_the_site_table": 0,
        "sites_of_the_pool_absent": 0,
    },
    "leaf_phenology_neon.csv": {
        "rows": 390,
        "sites": 39,
        "years": list(range(2012, 2022)),
        "complete_rectangle": True,
        "quality_counts": {
            "leafon_qa": {0: 310, 1: 4, 2: 2, 3: 74},
            "leafoff_qa": {0: 319, 1: 1, 3: 70},
        },
        "quality_three_is_exactly_missing": True,
        "sites_with_any_day": {"leafonday": 36, "leafoffday": 36},
        "median_day": {"leafonday": 126.5, "leafoffday": 271.0},
        "inverted_rows": 4,
        "inverted_sites": 1,
        "inverted_leafoffday_median": 120.5,
    },
}


def default_data_root() -> Path:
    """The repository's ``data/``, honoring ``$SIPNET_CALIBRATION_DATA``.

    Delegates to :func:`sipnet_calibration.conventions.data_root`, which finds
    it from the installed package rather than by counting parents from this
    file, so moving a script does not silently retarget every path it reads.
    """
    return conventions.data_root()


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.path or default_data_root() / "raw" / "phenology" / "leaf_phenology_8k.csv"
    try:
        frame = read_phenology(path)
        site_table = None if args.no_sites else load_sites(args.sites or default_sites_path())
        report = build_report(frame, path, site_table)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(format_report(report))

    if args.out is not None:
        try:
            args.out.write_text(json.dumps(report, indent=2, default=str))
        except OSError as error:
            print(f"error: could not write {args.out}: {error}", file=sys.stderr)
            return 2
        print(f"\nwrote {args.out}")

    if args.no_check:
        return 0
    failures = compare_with_recorded(report, path.name)
    print(format_comparison(failures, path.name))
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="The CSV to survey. Default: data/raw/phenology/leaf_phenology_8k.csv.",
    )
    parser.add_argument(
        "--sites",
        type=Path,
        default=None,
        help="The site table. Default: data/processed/sites/sites.csv.",
    )
    parser.add_argument(
        "--no-sites",
        action="store_true",
        help="Skip the site-table checks. Needed for the NEON file, which is keyed on "
        "BETY identifiers.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Report without comparing against the recorded characteristics.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Also write the report as JSON here."
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def read_phenology(path: Path) -> pd.DataFrame:
    """Read a phenology CSV exactly, refusing a header that is not :data:`COLUMNS`."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see data/README.md, Leaf phenology")
    try:
        return _read_checked(path)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(str(path)):
            raise
        raise ValueError(f"{path}: could not be parsed as a phenology table: {error}") from error


def _read_checked(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "year": np.int64,
            "site_id": np.int64,
            "lat": np.float64,
            "lon": np.float64,
            "leafonday": np.float64,
            "leafoffday": np.float64,
            "leafon_qa": np.int64,
            "leafoff_qa": np.int64,
        },
        index_col=False,
    )
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(f"{path}: header is {tuple(frame.columns)}, expected {COLUMNS}")
    if frame.empty:
        raise ValueError(f"{path}: holds no rows")
    return frame


def build_report(
    frame: pd.DataFrame, path: Path, site_table: pd.DataFrame | None
) -> dict[str, Any]:
    """Every measurement, in one dictionary."""
    report: dict[str, Any] = {
        "path": str(path),
        "rows": len(frame),
        "sites": int(frame["site_id"].nunique()),
        "years": [int(year) for year in sorted(frame["year"].unique())],
        "duplicate_site_years": int(frame.duplicated(["site_id", "year"]).sum()),
    }
    report["complete_rectangle"] = (
        report["duplicate_site_years"] == 0
        and report["rows"] == report["sites"] * len(report["years"])
    )
    report.update(survey_quality(frame))
    report.update(survey_days(frame))
    report.update(survey_inversions(frame))
    table = survey_against_site_table(frame, site_table)
    report["site_table"] = table
    # Lifted to the top level because compare_with_recorded only reaches that
    # far, and these are the properties --sites exists to establish.
    if table["checked"]:
        report["identifiers_not_in_the_site_table"] = table[
            "identifiers_not_in_the_site_table"
        ]
        report["sites_of_the_pool_absent"] = table["sites_of_the_pool_absent"]
    return report


def survey_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """The quality distributions, and whether flag 3 is exactly the missing set."""
    counts = {
        quality: {int(value): int(n) for value, n in frame[quality].value_counts().items()}
        for quality in DAY_COLUMNS.values()
    }
    unexpected = sorted(
        {value for per_column in counts.values() for value in per_column}
        - set(QUALITY_MEANINGS)
    )
    agrees = all(
        bool(((frame[quality] == 3) == frame[day].isna()).all())
        for day, quality in DAY_COLUMNS.items()
    )
    return {
        "quality_counts": counts,
        "quality_values_outside_0_3": unexpected,
        "quality_three_is_exactly_missing": agrees,
    }


def survey_days(frame: pd.DataFrame) -> dict[str, Any]:
    """Per day column: how much is missing, its extremes and median, its coverage."""
    return {
        "missing_days": {day: int(frame[day].isna().sum()) for day in DAY_COLUMNS},
        "median_day": {day: _or_none(frame[day].median()) for day in DAY_COLUMNS},
        "day_range": {
            day: [_or_none(frame[day].min()), _or_none(frame[day].max())]
            for day in DAY_COLUMNS
        },
        "sites_with_any_day": {
            day: int(frame.groupby("site_id")[day].count().gt(0).sum())
            for day in DAY_COLUMNS
        },
    }


def survey_inversions(frame: pd.DataFrame) -> dict[str, Any]:
    """Site-years whose leaf-on day is at or after their leaf-off day.

    Not an error in the file: the upstream conversion discards the year, so a
    leaf-off in the following January comes back as a small day-of-year. See
    the module Notes.
    """
    both = frame.dropna(subset=list(DAY_COLUMNS))
    inverted = both[both["leafonday"] >= both["leafoffday"]]
    return {
        "rows_with_both_days": len(both),
        "inverted_rows": len(inverted),
        "inverted_sites": int(inverted["site_id"].nunique()),
        "inverted_leafoffday_median": _or_none(inverted["leafoffday"].median()),
    }


def survey_against_site_table(
    frame: pd.DataFrame, site_table: pd.DataFrame | None
) -> dict[str, Any]:
    """Whether the file's identifiers are the project's sites, and agree on position."""
    if site_table is None:
        return {"checked": False}
    known = set(site_table["site_id"])
    unknown = sorted(set(frame["site_id"]) - known)
    joined = frame.merge(
        site_table[["site_id", "lon", "lat"]], on="site_id", how="inner", suffixes=("", "_table")
    )
    return {
        "checked": True,
        "identifiers_not_in_the_site_table": len(unknown),
        "first_unknown_identifiers": unknown[:5],
        "sites_of_the_pool_absent": len(known - set(frame["site_id"])),
        "max_coordinate_difference_degrees": _or_none(
            max(
                (joined["lat"] - joined["lat_table"]).abs().max(),
                (joined["lon"] - joined["lon_table"]).abs().max(),
            )
            if not joined.empty
            else None
        ),
    }


def format_report(report: dict[str, Any]) -> str:
    """The measurements as readable lines."""
    years = report["years"]
    lines = [
        f"{report['path']}",
        f"  rows                  : {report['rows']}",
        f"  sites                 : {report['sites']}",
        f"  years                 : {years[0]}-{years[-1]} ({len(years)})",
        f"  duplicate site-years  : {report['duplicate_site_years']}",
        f"  complete rectangle    : {_yes(report['complete_rectangle'])}",
        "",
    ]
    for day, quality in DAY_COLUMNS.items():
        counts = report["quality_counts"][quality]
        graded = ", ".join(
            f"{value} ({QUALITY_MEANINGS.get(value, '?')}) {counts[value]}"
            for value in sorted(counts)
        )
        low, high = report["day_range"][day]
        lines += [
            f"  {day}",
            f"    quality             : {graded}",
            f"    missing             : {report['missing_days'][day]}",
            f"    day of year         : {low} to {high}, median "
            f"{report['median_day'][day]}",
            f"    sites with any      : {report['sites_with_any_day'][day]}",
        ]
    lines += [
        "",
        f"  quality 3 is exactly missing : {_yes(report['quality_three_is_exactly_missing'])}",
        f"  quality values outside 0-3   : {report['quality_values_outside_0_3'] or 'none'}",
        f"  rows with both days          : {report['rows_with_both_days']}",
        f"  of those, leaf-on >= leaf-off: {report['inverted_rows']} rows over "
        f"{report['inverted_sites']} sites, leaf-off day median "
        f"{report['inverted_leafoffday_median']}",
    ]
    table = report["site_table"]
    if table["checked"]:
        lines += [
            "",
            f"  identifiers not a site       : {table['identifiers_not_in_the_site_table']}"
            + (
                f" (first {table['first_unknown_identifiers']})"
                if table["first_unknown_identifiers"]
                else ""
            ),
            f"  pool sites absent            : {table['sites_of_the_pool_absent']}",
            f"  max coordinate difference    : "
            f"{table['max_coordinate_difference_degrees']} degrees",
        ]
    else:
        lines += ["", "  site table                   : not checked (--no-sites)"]
    return "\n".join(lines)


def compare_with_recorded(report: dict[str, Any], file_name: str) -> list[str]:
    """Which recorded characteristics no longer hold, as readable lines."""
    recorded = RECORDED.get(file_name)
    if recorded is None:
        return [
            f"no recorded characteristics for {file_name!r}; add an entry to RECORDED "
            "and a paragraph to data/README.md, or pass --no-check"
        ]
    failures = []
    for key, expected in recorded.items():
        measured = report.get(key)
        if isinstance(expected, dict):
            # Recursively, because the int keys live in the per-quality-value
            # counts one level down, and a report round-tripped through --out's
            # JSON comes back with those keys as strings.
            measured = _stringify_keys({} if measured is None else measured)
            expected = _stringify_keys(expected)
        if measured != expected:
            failures.append(f"{key}: recorded {expected}, measured {measured}")
    return failures


def format_comparison(failures: list[str], file_name: str) -> str:
    """The comparison against :data:`RECORDED`, as readable lines."""
    if not failures:
        return (
            f"\nEvery characteristic data/README.md records for {file_name} still holds."
        )
    lines = [
        f"\nerror: {len(failures)} characteristic(s) data/README.md records for "
        f"{file_name} no longer hold:"
    ]
    lines += [f"  {failure}" for failure in failures]
    lines.append(
        "\nThe file has changed, or was re-copied from a different source. Establish "
        "which, then update RECORDED in this script and the Leaf phenology section of "
        "data/README.md together."
    )
    return "\n".join(lines)


# ── supporting helpers ────────────────────────────────────────────────────────


def _stringify_keys(value: Any) -> Any:
    """*value* with every mapping key a string, at every depth.

    JSON has only string keys, so a report written by ``--out`` and read back
    compares equal to one measured in this process only after this.
    """
    if isinstance(value, dict):
        return {str(key): _stringify_keys(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_stringify_keys(inner) for inner in value]
    return value


def _or_none(value: Any) -> float | None:
    """A float, or ``None`` where the measurement has no value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def _yes(value: bool) -> str:
    return "yes" if value else "NO"


if __name__ == "__main__":
    raise SystemExit(main())
