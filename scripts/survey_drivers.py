#!/usr/bin/env python3
"""Survey the ERA5 driver files.

Overview
--------
Walk a drivers root, apply :func:`sipnet_calibration.drivers.read_clim_file`
to every file, and report what holds across the whole ensemble: whether the
directory template covers every site and member, whether the ``(site, member)``
set is a complete rectangle, and which files fail which check. The three files
available locally settle the format; only the SCC can settle the coverage, and
that is what this answers.

This is a diagnostic, not an ingest: it reports what it finds and never refuses
or rewrites anything. It is a one-off, run once where the files are and then
discarded; the facts it establishes go into ``data/README.md``, the script does
not.

Input data
----------
``--root``
    A directory laid out as ``<root>/ERA5_<site>_<member>/ERA5.<member>.<start>.<end>.clim``,
    the layout :mod:`sipnet_calibration.drivers` documents. Nothing here
    assumes the site pool or the ensemble size; reporting them is the point.

Output data
-----------
A report to stdout and, with ``--out``, the same content as JSON:

* the site identifiers and member indices seen, and whether every
  ``(site, member)`` pair has a directory and a file;
* directories that do not match the template, and pairs with zero or several
  ``.clim`` files;
* per check, the files that fail it, with the message;
* the distinct ``(start, end)`` date pairs in file names;
* the distinct constant-column values seen for ``loc``, ``length`` and
  ``soil_wetness``, which are asserted per file but worth tabulating;
* per variable, the count of negative or non-positive values and the extremes,
  summed over the files that parsed;
* whether every file shares one ``(year, day, time)`` grid;
* any file that failed to parse, with the reason.

Notes
-----
Runs under the project environment rather than bare Python: the whole point is
to apply the reader's own checks, so it imports them. Parsing costs tens of
milliseconds per file and there are 80,000, so a serial run is about two hours;
``--jobs`` parallelizes over files.

Usage
-----
::

    python scripts/survey_drivers.py --root /path/to/ERA5_2012_2024
    python scripts/survey_drivers.py --root ... --jobs 16 --out drivers_survey.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sipnet_calibration.drivers import (
    CLIM_FILE_CONSTANTS,
    DRIVER_FILE_GLOB,
    SOURCE_VARIABLE_NAMES,
    read_clim_file,
)

DIRECTORY_PATTERN = re.compile(r"^ERA5_(\d+)_(\d+)$")
FILE_PATTERN = re.compile(r"^ERA5\.(\d+)\.(\d{4}-\d{2}-\d{2})\.(\d{4}-\d{2}-\d{2})\.clim$")

#: Which invariant a ``read_clim_file`` message is about, by a phrase it carries.
CHECK_PHRASES = {
    "could not be parsed": "column_count",
    "expected 14 fields": "column_count",
    "missing or non-finite": "no_missing_values",
    "must be": "constant_columns",
    "years are not contiguous": "day_structure",
    "ascending year order": "day_structure",
    "rows, expected": "day_structure",
    "does not run": "day_structure",
    "linspace model": "time_drift_model",
    "below -": "negative_excursions",
    "non-integer": "integer_year_day",
}


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 1

    directories, off_template = find_driver_directories(args.root)
    if not directories:
        print(f"error: no ERA5_<site>_<member> directories under {args.root}", file=sys.stderr)
        return 1

    results = run_survey(directories, jobs=args.jobs)
    report = build_report(results, off_template)
    print_report(report)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote {args.out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="The drivers root.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers (default 1).")
    parser.add_argument("--out", type=Path, default=None, help="Also write the report as JSON here.")
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def find_driver_directories(root: Path) -> tuple[list[Path], list[str]]:
    """Every ``ERA5_<site>_<member>`` directory under *root*, and every
    directory that does not match the template."""
    matching, off_template = [], []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if DIRECTORY_PATTERN.match(entry.name):
            matching.append(entry)
        else:
            off_template.append(entry.name)
    return matching, off_template


def run_survey(directories: list[Path], *, jobs: int) -> list[FileFacts]:
    if jobs <= 1:
        return [survey_one_file(d) for d in directories]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(survey_one_file, directories, chunksize=64))


def survey_one_file(directory: Path) -> FileFacts:
    """Locate and parse one pair's file, recording what passed and what failed."""
    site, member = (int(x) for x in DIRECTORY_PATTERN.match(directory.name).groups())
    facts = FileFacts(site=site, member=member, directory=str(directory))
    matches = sorted(directory.glob(DRIVER_FILE_GLOB))
    facts.n_files = len(matches)
    if len(matches) != 1:
        return facts
    path = matches[0]
    facts.file = path.name

    name_match = FILE_PATTERN.match(path.name)
    if name_match is None:
        facts.name_problems.append("file name off the template")
    else:
        if int(name_match.group(1)) != member:
            facts.name_problems.append(
                f"file name member {name_match.group(1)} differs from directory member {member}"
            )
        facts.name_dates = (name_match.group(2), name_match.group(3))

    try:
        frame = read_clim_file(path)
    except ValueError as error:
        facts.error = str(error)
        facts.failed_check = classify(facts.error)
        _collect_what_we_can(facts, path)
        return facts
    except Exception as error:  # noqa: BLE001 -- a survey reports, it does not stop
        facts.error = f"{type(error).__name__}: {error}"
        facts.failed_check = "unexpected"
        return facts

    facts.n_rows = len(frame)
    first = pd.Timestamp(int(frame["year"].iloc[0]), 1, 1) + pd.Timedelta(days=int(frame["day"].iloc[0]) - 1)
    last = pd.Timestamp(int(frame["year"].iloc[-1]), 1, 1) + pd.Timedelta(days=int(frame["day"].iloc[-1]) - 1)
    facts.data_dates = (str(first.date()), str(last.date()))
    if facts.name_dates is not None and facts.name_dates != facts.data_dates:
        facts.name_problems.append("file name dates differ from the data")
    facts.grid_hash = hashlib.sha1(
        np.ascontiguousarray(frame[["year", "day", "time"]].to_numpy(np.float64)).tobytes()
    ).hexdigest()
    for column in SOURCE_VARIABLE_NAMES:
        values = frame[column].to_numpy()
        facts.stats[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "n_below_zero": int((values < 0).sum()),
            "n_not_positive": int((values <= 0).sum()),
        }
    for column in CLIM_FILE_CONSTANTS:
        facts.constants[column] = [float(v) for v in np.unique(frame[column].to_numpy())]
    return facts


def build_report(results: list[FileFacts], off_template: list[str]) -> dict[str, object]:
    """Aggregate the per-file results into the report described above."""
    sites = sorted({r.site for r in results})
    members = sorted({r.member for r in results})
    have_file = {(r.site, r.member) for r in results if r.n_files == 1}
    missing_pairs = [(s, m) for s in sites for m in members if (s, m) not in have_file]

    by_check: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in results:
        if r.error:
            by_check[r.failed_check or "unknown"].append({"directory": r.directory, "message": r.error})

    parsed = [r for r in results if r.n_rows is not None]
    stats: dict[str, dict[str, float | int]] = {}
    for column in SOURCE_VARIABLE_NAMES:
        per = [r.stats[column] for r in parsed if column in r.stats]
        if per:
            stats[column] = {
                "min": min(p["min"] for p in per),
                "max": max(p["max"] for p in per),
                "n_below_zero": sum(p["n_below_zero"] for p in per),
                "n_not_positive": sum(p["n_not_positive"] for p in per),
            }
    constants = {
        column: sorted({v for r in parsed for v in r.constants.get(column, [])})
        for column in CLIM_FILE_CONSTANTS
    }

    return {
        "n_directories": len(results),
        "directories_off_template": off_template,
        "sites": {"n": len(sites), "min": sites[0] if sites else None, "max": sites[-1] if sites else None,
                  "missing_in_range": _gaps(sites)},
        "members": members,
        "rectangle_complete": not missing_pairs,
        "n_missing_pairs": len(missing_pairs),
        "missing_pairs_sample": missing_pairs[:50],
        "pairs_with_no_file": [r.directory for r in results if r.n_files == 0],
        "pairs_with_several_files": [r.directory for r in results if r.n_files > 1],
        "name_problems": [
            {"directory": r.directory, "problems": r.name_problems} for r in results if r.name_problems
        ],
        "n_parsed": len(parsed),
        "n_failed": sum(1 for r in results if r.error),
        "failures_by_check": {k: {"n": len(v), "sample": v[:20]} for k, v in by_check.items()},
        "distinct_name_dates": sorted({r.name_dates for r in results if r.name_dates}),
        "distinct_row_counts": sorted({r.n_rows for r in parsed}),
        "distinct_grid_hashes": len({r.grid_hash for r in parsed}),
        "constants_seen": constants,
        "value_stats": stats,
    }


def print_report(report: dict[str, object]) -> None:
    sites = report["sites"]
    print(f"directories        {report['n_directories']}  (off template: {len(report['directories_off_template'])})")
    print(f"sites              {sites['n']} from {sites['min']} to {sites['max']}; "
          f"{len(sites['missing_in_range'])} identifiers absent in that range")
    print(f"members            {report['members']}")
    print(f"rectangle complete {report['rectangle_complete']}  (missing pairs: {report['n_missing_pairs']})")
    print(f"pairs with no file {len(report['pairs_with_no_file'])}, with several {len(report['pairs_with_several_files'])}")
    print(f"name problems      {len(report['name_problems'])}")
    print(f"parsed {report['n_parsed']}, failed {report['n_failed']}")
    for check, entry in report["failures_by_check"].items():
        print(f"  {check:24s} {entry['n']}")
        for item in entry["sample"][:3]:
            print(f"      {item['message'][:160]}")
    print(f"file-name dates    {report['distinct_name_dates']}")
    print(f"row counts         {report['distinct_row_counts']}")
    print(f"distinct grids     {report['distinct_grid_hashes']}")
    print(f"constants          {report['constants_seen']}")
    for column, s in report["value_stats"].items():
        print(f"  {column:12s} min {s['min']:12.5g} max {s['max']:12.5g} "
              f"below zero {s['n_below_zero']:9d} not positive {s['n_not_positive']:9d}")


# ── supporting types and helpers ──────────────────────────────────────────────


@dataclass
class FileFacts:
    site: int
    member: int
    directory: str
    n_files: int = 0
    file: str | None = None
    name_dates: tuple[str, str] | None = None
    data_dates: tuple[str, str] | None = None
    name_problems: list[str] = field(default_factory=list)
    error: str | None = None
    failed_check: str | None = None
    n_rows: int | None = None
    grid_hash: str | None = None
    stats: dict[str, dict[str, float | int]] = field(default_factory=dict)
    constants: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify(message: str) -> str:
    for phrase, check in CHECK_PHRASES.items():
        if phrase in message:
            return check
    return "unknown"


def _collect_what_we_can(facts: FileFacts, path: Path) -> None:
    """Row count and constant values from a file that failed a check."""
    try:
        raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        return
    facts.n_rows = None  # not a parsed file; keep the report's meaning of n_rows
    facts.constants["_raw_row_count"] = [float(len(raw))]
    if raw.shape[1] == 14:
        for index, column in ((0, "loc"), (4, "length"), (13, "soil_wetness")):
            facts.constants[column] = sorted({float(v) for v in raw[index].unique() if _is_number(v)})


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _gaps(sites: list[int]) -> list[int]:
    if not sites:
        return []
    return sorted(set(range(sites[0], sites[-1] + 1)) - set(sites))


if __name__ == "__main__":
    raise SystemExit(main())
