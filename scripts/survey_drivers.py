#!/usr/bin/env python
"""Survey the ERA5 driver files.

Overview
--------
Walk a drivers root, apply :func:`sipnet_calibration.drivers.read_driver_file`
-- pySIPNET's reader and validation, and the loader's own value check -- to
every file, and report what holds across the whole ensemble: whether the
directory template covers every site and member, whether the ``(site, member)``
set is a complete rectangle, and which files fail which check.

Input data
----------
``--raw-directory``
    A drivers root, laid out as
    ``<raw-directory>/ERA5_<site>_<member>/ERA5.<member>.<start>.<end>.clim``, the
    layout :mod:`sipnet_calibration.drivers` documents. Nothing here assumes
    the site pool or the ensemble size; reporting them is the point.

Output data
-----------
A report to stdout and, with ``--output``, the same content as JSON. Nothing is
written to ``data/``. The report holds:

* the site ids and member indices seen, and whether every ``(site, member)``
  pair has a directory and a file;
* directories that do not match the template, and pairs with zero or several
  ``.clim`` files;
* per check, the files that fail it, with the message: ``pysipnet`` for a file
  pySIPNET refuses, ``negative_excursions`` for the loader's own check;
* the distinct ``(start, end)`` date pairs in file names;
* the distinct file layouts, ``loc`` values and step lengths seen;
* per variable, the count of negative or non-positive values and the extremes,
  summed over the files that were read;
* whether every file shares one time axis;
* any file that failed unexpectedly, with the reason.

The exit status is 0 once the report is written, and 1 when the root could not
be surveyed at all.

Notes
-----
This is a diagnostic, not an ingest: it reports what it finds and never refuses
or rewrites anything, and unlike the other two surveys it records no
characteristics to check against. The three files available locally settle
the format; only the SCC can settle the coverage, and the facts it establishes
there go into ``data/README.md``.

Runs under the project environment rather than bare Python: the whole point is
to apply the reader's own checks, so it imports them. A file pySIPNET refuses
is read no further, so the value statistics cover only the files it accepts.
pySIPNET refuses the ERA5 files as generated for their drifting hour column
(``data/README.md`` Note 15), so until they are corrected the report on them
is the refusal alone. Parsing is the cost, so a serial run over the whole
ensemble takes hours; ``--jobs`` parallelizes over files.

Usage
-----
::

    uv run python scripts/survey_drivers.py --raw-directory /path/to/ERA5_2012_2024
    uv run python scripts/survey_drivers.py --raw-directory ... --jobs 16 \\
        --output drivers_survey.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sipnet_calibration.conventions import TIMESTEP_LENGTH, TIMESTEP_START
from sipnet_calibration.drivers import (
    DRIVER_FILE_GLOB,
    DRIVER_VARIABLE_NAMES,
    read_driver_file,
)

#: A driver directory's name, ``ERA5_<site>_<member>``: the pattern
#: :mod:`sipnet_calibration.drivers` matches its directories with.
DIRECTORY_PATTERN = re.compile(r"^ERA5_(\d+)_(\d+)$")

#: A driver file's name, ``ERA5.<member>.<start>.<end>.clim``: the pattern
#: :mod:`sipnet_calibration.drivers` matches its files with.
FILE_PATTERN = re.compile(r"^ERA5\.(\d+)\.(\d{4}-\d{2}-\d{2})\.(\d{4}-\d{2}-\d{2})\.clim$")

#: Which check a ``read_driver_file`` message is about, by a phrase it
#: carries. Coupled to the reader's wording; a message no phrase matches is
#: reported as ``unknown`` rather than dropped. pySIPNET's own reasons are not
#: split further here: the message carries them.
CHECK_PHRASES = {
    "pySIPNET refused the file": "pysipnet",
    "below -": "negative_excursions",
}


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Survey every driver directory under the root and report it."""
    args = parse_args(argv)
    try:
        directories, off_template = find_driver_directories(args.raw_directory)
    except (OSError, ValueError, LookupError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    results = run_survey(directories, jobs=args.jobs)
    report = build_report(results, off_template)
    print_report(report)
    if args.output is not None:
        try:
            args.output.write_text(json.dumps(report, indent=2, default=str))
        except OSError as error:
            print(f"error: could not write {args.output}: {error}", file=sys.stderr)
            return 1
        print(f"\nWrote {args.output}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, as the module docstring's Usage describes it."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--raw-directory", type=Path, required=True, help="The drivers root.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers. Default: 1.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Also write the report as JSON here."
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def find_driver_directories(root: Path) -> tuple[list[Path], list[str]]:
    """Every ``ERA5_<site>_<member>`` directory under *root*, and the others' names."""
    check_root_is_a_directory(root, message_name="--raw-directory")
    matching, off_template = [], []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if DIRECTORY_PATTERN.match(entry.name):
            matching.append(entry)
        else:
            off_template.append(entry.name)
    check_root_holds_driver_directories(matching, message_name=str(root))
    return matching, off_template


def run_survey(directories: list[Path], *, jobs: int) -> list[DirectoryFacts]:
    """The facts of every directory, surveyed in *jobs* worker processes."""
    if jobs <= 1:
        return [survey_driver_directory(directory) for directory in directories]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(survey_driver_directory, directories, chunksize=64))


def build_report(results: list[DirectoryFacts], off_template: list[str]) -> dict[str, object]:
    """The per-directory facts aggregated into the report the module describes."""
    parsed = [facts for facts in results if facts.n_rows is not None]
    return {
        **_coverage_report(results, off_template),
        "name_problems": [
            {"directory": facts.directory, "problems": facts.name_problems}
            for facts in results
            if facts.name_problems
        ],
        "n_parsed": len(parsed),
        "n_failed": sum(1 for facts in results if facts.error),
        "failures_by_check": _failures_by_check(results),
        "distinct_name_dates": sorted({facts.name_dates for facts in results if facts.name_dates}),
        "distinct_row_counts": sorted({facts.n_rows for facts in parsed}),
        "distinct_grid_hashes": len({facts.grid_hash for facts in parsed}),
        "constants_seen": {
            column: sorted({value for facts in parsed for value in facts.constants.get(column, [])})
            for column in ("n_columns", "loc", TIMESTEP_LENGTH)
        },
        "value_stats": _summed_value_stats(parsed),
    }


def print_report(report: dict[str, object]) -> None:
    """Print the report as readable lines."""
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
    for column, stats in report["value_stats"].items():
        print(f"  {column:36s} min {stats['min']:12.5g} max {stats['max']:12.5g} "
              f"below zero {stats['n_below_zero']:9d} not positive {stats['n_not_positive']:9d}")


# ── supporting types and helpers ──────────────────────────────────────────────


def survey_driver_directory(directory: Path) -> DirectoryFacts:
    """The facts of one pair's directory: its file, what passed and what failed."""
    site, member = (int(number) for number in DIRECTORY_PATTERN.match(directory.name).groups())
    facts = DirectoryFacts(site=site, member=member, directory=str(directory))
    matches = sorted(directory.glob(DRIVER_FILE_GLOB))
    facts.n_files = len(matches)
    if len(matches) != 1:
        return facts
    path = matches[0]
    facts.file = path.name
    _record_file_name_facts(facts, path.name)

    try:
        climate = read_driver_file(path)
        frame, axis = climate.pandas, climate.xarray
    except ValueError as error:
        facts.error = str(error)
        facts.failed_check = failed_check_named_by(facts.error)
        return facts
    except Exception as error:  # noqa: BLE001 -- a survey reports, it does not stop
        facts.error = f"{type(error).__name__}: {error}"
        facts.failed_check = "unexpected"
        return facts
    _record_file_content_facts(facts, climate, frame, axis)
    return facts


@dataclass
class DirectoryFacts:
    """What the survey found for one ``(site, member)`` directory."""

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


def failed_check_named_by(message: str) -> str:
    """The check a ``read_driver_file`` message is about, or ``"unknown"``."""
    for phrase, check in CHECK_PHRASES.items():
        if phrase in message:
            return check
    return "unknown"


def _coverage_report(results: list[DirectoryFacts], off_template: list[str]) -> dict[str, object]:
    """The report's directories, sites, members and the pairs without one file."""
    sites = sorted({facts.site for facts in results})
    members = sorted({facts.member for facts in results})
    have_file = {(facts.site, facts.member) for facts in results if facts.n_files == 1}
    missing_pairs = [
        (site, member) for site in sites for member in members if (site, member) not in have_file
    ]
    return {
        "n_directories": len(results),
        "directories_off_template": off_template,
        "sites": {
            "n": len(sites),
            "min": sites[0] if sites else None,
            "max": sites[-1] if sites else None,
            "missing_in_range": _missing_site_ids(sites),
        },
        "members": members,
        "rectangle_complete": not missing_pairs,
        "n_missing_pairs": len(missing_pairs),
        "missing_pairs_sample": missing_pairs[:50],
        "pairs_with_no_file": [facts.directory for facts in results if facts.n_files == 0],
        "pairs_with_several_files": [facts.directory for facts in results if facts.n_files > 1],
    }


def _failures_by_check(results: list[DirectoryFacts]) -> dict[str, dict[str, object]]:
    """Check name -> how many files failed it, and a sample of their messages."""
    by_check: dict[str, list[dict[str, str]]] = defaultdict(list)
    for facts in results:
        if facts.error:
            by_check[facts.failed_check or "unknown"].append(
                {"directory": facts.directory, "message": facts.error}
            )
    return {
        check: {"n": len(failures), "sample": failures[:20]}
        for check, failures in by_check.items()
    }


def _record_file_name_facts(facts: DirectoryFacts, file_name: str) -> None:
    """Record on *facts* the dates in *file_name* and how it is off the template."""
    name_match = FILE_PATTERN.match(file_name)
    if name_match is None:
        facts.name_problems.append("file name off the template")
        return
    if int(name_match.group(1)) != facts.member:
        facts.name_problems.append(
            f"file name member {name_match.group(1)} differs from directory member {facts.member}"
        )
    facts.name_dates = (name_match.group(2), name_match.group(3))


def _record_file_content_facts(facts: DirectoryFacts, climate, frame: pd.DataFrame, axis) -> None:
    """Record on *facts* a file's rows, dates, time axis, statistics and constants."""
    facts.n_rows = len(frame)
    starts = pd.DatetimeIndex(axis[TIMESTEP_START].values)
    facts.data_dates = (str(starts[0].date()), str(starts[-1].date()))
    if facts.name_dates is not None and facts.name_dates != facts.data_dates:
        facts.name_problems.append("file name dates differ from the data")
    facts.grid_hash = hashlib.sha1(
        np.ascontiguousarray(axis[TIMESTEP_START].values.astype("int64")).tobytes()
        + np.ascontiguousarray(axis[TIMESTEP_LENGTH].values.astype("int64")).tobytes()
    ).hexdigest()
    for name in DRIVER_VARIABLE_NAMES:
        values = frame[name].to_numpy()
        facts.stats[name] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "n_below_zero": int((values < 0).sum()),
            "n_not_positive": int((values <= 0).sum()),
        }
    facts.constants = {
        "n_columns": [float(climate.n_columns)],
        "loc": [float(climate.loc)],
        TIMESTEP_LENGTH: [float(value) for value in np.unique(frame[TIMESTEP_LENGTH].to_numpy())],
    }


def _summed_value_stats(parsed: list[DirectoryFacts]) -> dict[str, dict[str, float | int]]:
    """Per variable, the extremes and the counts summed over the files read."""
    stats: dict[str, dict[str, float | int]] = {}
    for column in DRIVER_VARIABLE_NAMES:
        per_file = [facts.stats[column] for facts in parsed if column in facts.stats]
        if per_file:
            stats[column] = {
                "min": min(one["min"] for one in per_file),
                "max": max(one["max"] for one in per_file),
                "n_below_zero": sum(one["n_below_zero"] for one in per_file),
                "n_not_positive": sum(one["n_not_positive"] for one in per_file),
            }
    return stats


def _missing_site_ids(sites: list[int]) -> list[int]:
    """The site ids absent between the smallest and the largest of *sites*."""
    if not sites:
        return []
    return sorted(set(range(sites[0], sites[-1] + 1)) - set(sites))


# ── checks ────────────────────────────────────────────────────────────────────


def check_root_is_a_directory(root: Path, *, message_name: str) -> None:
    """The drivers root is a directory."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"{message_name} {root} is not a directory; pass the drivers root."
        )


def check_root_holds_driver_directories(directories: list[Path], *, message_name: str) -> None:
    """The drivers root holds at least one ``ERA5_<site>_<member>`` directory."""
    if not directories:
        raise ValueError(
            f"no ERA5_<site>_<member> directories under {message_name}; pass the drivers "
            "root with --raw-directory."
        )


if __name__ == "__main__":
    raise SystemExit(main())
