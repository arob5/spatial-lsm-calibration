#!/usr/bin/env python3
"""Survey the initial-condition netCDF files.

Overview
--------
Report the variable sets, ensemble shape and metadata of every
initial-condition file under a root directory. The files are not all alike --
some carry ``leaf_carbon_content`` and ``SoilMoistFrac`` beyond the three
variables seen so far -- and ``scripts/ingest_ic.py`` has to know the full set
of combinations before it can build a ``(member, site)`` array. That question
can only be answered where the files are, which is the SCC.

This is a diagnostic, not an ingest: it reports what it finds and never refuses
or rewrites anything, which is why it has no ``check_*`` helpers.

Input data
----------
``--root``
    A directory of per-site subdirectories laid out as
    ``<root>/<site>/IC_site_<site>_<member>.nc``. Each file is netCDF-3 classic
    (magic ``CDF\x01``) holding a handful of scalar carbon-pool variables on a
    length-1 record dimension named ``time``, with ``units`` and ``long_name``
    attributes and a ``_FillValue`` of -999.0. The expected pool is 8000 sites
    by 100 members, but nothing here assumes that.

    The ``time`` variable's ``units`` attribute is the literal unsubstituted
    template ``days since [year]-01-01 00:00:00 UTC``, which no calendar
    library can parse (issue #3). This script does not decode it.

Output data
-----------
A report to stdout and, with ``--out``, the same content as JSON:

* the distinct variable-set signatures and how many files carry each;
* per variable, the file count and the distinct ``units`` and ``long_name``;
* the member indices present per site, and whether the ensemble is a complete
  rectangle over sites and members;
* files whose directory and file-name site numbers disagree;
* how often the ``-999.0`` fill value appears;
* whether ``AbvGrndWood`` and ``wood_carbon_content`` are always equal, which
  would make the pair redundant;
* any file that failed to parse, with the reason.

Notes
-----
**No third-party dependencies.** The files are netCDF-3 classic, whose header
format is simple enough to parse directly, so this runs under any Python 3 on
the SCC with no modules loaded and no environment to activate. Only the header
and the handful of scalar values are read, about a kilobyte per file, so a run
is dominated by filesystem latency rather than by parsing -- which is what
``--jobs`` is for, and it is worth raising on a networked filesystem.

For a record variable, ``read_values`` returns the first record only, which is
all these files have. The record count is reported separately so that a file
with more would be visible rather than silently truncated.

Usage
-----
::

    python3 scripts/survey_ic_variables.py --root <IC root> --sample 200
    python3 scripts/survey_ic_variables.py --root <IC root> --jobs 16 \
        --out ic_survey.json
    python3 scripts/survey_ic_variables.py --help
"""

from __future__ import annotations

import argparse
import json
import random
import re
import struct
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


FILE_PATTERN = re.compile(r"^IC_site_(\d+)_(\d+)\.nc$")

FILL_VALUE = -999.0

#: Reported equal in every file seen so far; the survey checks whether that
#: holds generally, because if it does the pair is redundant.
EQUALITY_PAIR = ("AbvGrndWood", "wood_carbon_content")


class NetCDFParseError(Exception):
    """Raised when a file is not parseable netCDF-3 classic."""


# ── entry point ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.root.is_dir():
        print(f"error: --root {args.root} is not a directory", file=sys.stderr)
        return 2

    print(f"Scanning {args.root} ...", file=sys.stderr, flush=True)
    paths = find_ic_files(args.root)
    if not paths:
        print(f"error: no IC_site_*.nc files found under {args.root}", file=sys.stderr)
        return 1

    if args.sample is not None:
        sites = sorted({p.parent.name for p in paths})
        wanted = min(args.sample, len(sites))
        chosen = set(random.Random(args.seed).sample(sites, wanted))
        paths = [p for p in paths if p.parent.name in chosen]
        print(f"Sampling {len(chosen)} of {len(sites)} sites.", file=sys.stderr)

    print(
        f"Reading {len(paths)} files with {args.jobs} workers ...",
        file=sys.stderr,
        flush=True,
    )
    totals = run_survey(paths, check_values=not args.no_values, jobs=args.jobs)
    report = build_report(totals)
    print_report(report)

    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nWrote {args.out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory holding the per-site subdirectories of IC netCDF files.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Survey a random sample of N sites instead of all of them.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for --sample. Default 0."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Concurrent file reads. Worth raising on a networked filesystem.",
    )
    parser.add_argument(
        "--no-values",
        action="store_true",
        help="Read headers only; skips the fill-value and equality checks.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the full result here as JSON, in addition to printing it.",
    )
    return parser.parse_args(argv)


# ── the survey steps, in the order main calls them ───────────────────────────


def find_ic_files(root: Path) -> list[Path]:
    """Every ``IC_site_<site>_<member>.nc`` under a site directory of *root*."""
    return sorted(
        path
        for site_dir in root.iterdir()
        if site_dir.is_dir()
        for path in site_dir.glob("IC_site_*.nc")
        if FILE_PATTERN.match(path.name)
    )


def run_survey(paths: list[Path], *, check_values: bool, jobs: int) -> SurveyTotals:
    """Survey every path, accumulating into a single result."""
    totals = SurveyTotals()

    def attempt(path: Path) -> tuple[Path, FileFacts | Exception]:
        try:
            return path, survey_one_file(path, check_values=check_values)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            return path, error

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for done, (path, outcome) in enumerate(pool.map(attempt, paths), start=1):
            if isinstance(outcome, Exception):
                totals.failures.append(
                    (str(path), f"{type(outcome).__name__}: {outcome}")
                )
            else:
                accumulate(totals, outcome, path)
            if done % 20000 == 0:
                print(
                    f"  ... {done} of {len(paths)} files",
                    file=sys.stderr,
                    flush=True,
                )

    return totals


def survey_one_file(path: Path, *, check_values: bool) -> FileFacts:
    """Extract the facts of interest from a single file."""
    match = FILE_PATTERN.match(path.name)
    if match is None:  # pragma: no cover - find_ic_files filters these out
        raise NetCDFParseError(f"unexpected file name {path.name!r}")
    site, member = int(match.group(1)), int(match.group(2))

    header, raw = read_header(path)
    by_name = {variable.name: variable for variable in header.variables}
    data_names = tuple(sorted(name for name in by_name if name != "time"))

    def attribute(name: str, key: str) -> str:
        value = by_name[name].attributes.get(key, "")
        return value if isinstance(value, str) else repr(value)

    time_length = header.dimension_sizes().get("time")

    n_fill = 0
    equal: bool | None = None
    if check_values:
        values = {
            name: read_values(variable, raw)
            for name, variable in by_name.items()
            if name != "time"
        }
        n_fill = sum(v.count(FILL_VALUE) for v in values.values())
        left, right = EQUALITY_PAIR
        if left in values and right in values:
            equal = values[left] == values[right]

    return FileFacts(
        site=site,
        member=member,
        signature=data_names,
        units={name: attribute(name, "units") for name in data_names},
        long_names={name: attribute(name, "long_name") for name in data_names},
        time_length=time_length,
        n_fill_values=n_fill,
        equality_pair_equal=equal,
    )


def accumulate(totals: SurveyTotals, facts: FileFacts, path: Path) -> None:
    """Fold one file's facts into the running totals."""
    totals.n_files += 1
    totals.signatures[facts.signature] += 1
    totals.signature_example.setdefault(facts.signature, str(path))
    totals.members_by_site[facts.site].add(facts.member)
    totals.time_lengths[facts.time_length] += 1
    totals.n_fill_values += facts.n_fill_values

    for name in facts.signature:
        totals.variable_files[name] += 1
        totals.variable_units[name][facts.units[name]] += 1
        totals.variable_long_names[name][facts.long_names[name]] += 1

    if facts.equality_pair_equal is not None:
        totals.n_equality_pair_checked += 1
        if not facts.equality_pair_equal:
            totals.n_equality_pair_unequal += 1
            if len(totals.unequal_examples) < 10:
                totals.unequal_examples.append(str(path))

    if str(path.parent.name) != str(facts.site):
        totals.path_mismatches.append(str(path))


def build_report(totals: SurveyTotals) -> dict[str, object]:
    """Assemble the JSON-serializable survey result."""
    return {
        "n_files": totals.n_files,
        "n_failures": len(totals.failures),
        "failures": totals.failures[:20],
        "variable_set_signatures": [
            {
                "variables": list(signature),
                "n_files": count,
                "example": totals.signature_example[signature],
            }
            for signature, count in totals.signatures.most_common()
        ],
        "variables": {
            name: {
                "n_files": totals.variable_files[name],
                "units": dict(totals.variable_units[name]),
                "long_names": dict(totals.variable_long_names[name]),
            }
            for name in sorted(totals.variable_files)
        },
        "ensemble": describe_ensemble(totals),
        "time_dimension_lengths": {
            str(k): v for k, v in sorted(totals.time_lengths.items(), key=str)
        },
        "fill_values": {
            "n_fill_values_found": totals.n_fill_values,
            "fill_value": FILL_VALUE,
        },
        "equality_check": {
            "pair": list(EQUALITY_PAIR),
            "n_checked": totals.n_equality_pair_checked,
            "n_unequal": totals.n_equality_pair_unequal,
            "examples_unequal": totals.unequal_examples,
        },
        "path_mismatches": {
            "n": len(totals.path_mismatches),
            "examples": totals.path_mismatches[:20],
        },
    }


def print_report(report: dict[str, object]) -> None:
    """Print the survey result in a form that is readable in a terminal."""
    print(f"\nFiles surveyed: {report['n_files']}  (failures: {report['n_failures']})")
    for path, error in report["failures"]:
        print(f"  FAILED {path}: {error}")

    print("\nVariable-set signatures")
    for entry in report["variable_set_signatures"]:
        print(f"  {entry['n_files']:>8d} files : {', '.join(entry['variables'])}")
        print(f"           example : {entry['example']}")

    print("\nVariables")
    for name, info in report["variables"].items():
        units = ", ".join(f"{u!r} x{n}" for u, n in info["units"].items())
        print(f"  {name:<32s} {info['n_files']:>8d} files   units: {units}")
        for long_name, n in info["long_names"].items():
            print(f"  {'':<32s} {'':>8s}   long_name: {long_name!r} x{n}")

    ensemble = report["ensemble"]
    print("\nEnsemble")
    print(f"  sites found                : {ensemble['n_sites']}")
    print(
        f"  member indices             : {ensemble['member_index_min']}"
        f"..{ensemble['member_index_max']}"
        f" ({ensemble['n_distinct_member_indices']} distinct)"
    )
    print(f"  most common ensemble size  : {ensemble['most_common_ensemble_size']}")
    print(f"  sites with another size    : {ensemble['n_sites_with_other_size']}")
    if ensemble["example_sites_with_other_size"]:
        print(f"    examples: {ensemble['example_sites_with_other_size']}")
    print(f"  ensemble size counts       : {ensemble['ensemble_size_counts']}")

    print(f"\ntime dimension lengths       : {report['time_dimension_lengths']}")
    print(f"fill values ({FILL_VALUE}) found  : {report['fill_values']['n_fill_values_found']}")

    equality = report["equality_check"]
    print(
        f"\n{equality['pair'][0]} == {equality['pair'][1]}: "
        f"{equality['n_checked'] - equality['n_unequal']} of {equality['n_checked']} checked"
    )
    for path in equality["examples_unequal"]:
        print(f"  unequal: {path}")

    mismatches = report["path_mismatches"]
    if mismatches["n"]:
        print(f"\nWARNING: {mismatches['n']} files whose directory and name disagree")
        for path in mismatches["examples"]:
            print(f"  {path}")


def describe_ensemble(totals: SurveyTotals) -> dict[str, object]:
    """Summarize the (site, member) grid: sizes, gaps, and irregular sites."""
    sizes = Counter(len(members) for members in totals.members_by_site.values())
    all_members = sorted({m for ms in totals.members_by_site.values() for m in ms})
    common = max(sizes, key=sizes.get) if sizes else 0
    odd_sites = sorted(
        site
        for site, members in totals.members_by_site.items()
        if len(members) != common
    )
    return {
        "n_sites": len(totals.members_by_site),
        "member_index_min": all_members[0] if all_members else None,
        "member_index_max": all_members[-1] if all_members else None,
        "n_distinct_member_indices": len(all_members),
        "ensemble_size_counts": {str(k): v for k, v in sorted(sizes.items())},
        "most_common_ensemble_size": common,
        "n_sites_with_other_size": len(odd_sites),
        "example_sites_with_other_size": odd_sites[:20],
    }


# ── supporting types and helpers ─────────────────────────────────────────────


@dataclass
class FileFacts:
    """What the survey extracts from one initial-condition file."""

    site: int
    member: int
    signature: tuple[str, ...]
    units: dict[str, str]
    long_names: dict[str, str]
    time_length: int | None
    n_fill_values: int
    equality_pair_equal: bool | None


@dataclass
class SurveyTotals:
    """Everything the survey accumulates across files."""

    signatures: Counter = field(default_factory=Counter)
    signature_example: dict[tuple[str, ...], str] = field(default_factory=dict)
    variable_files: Counter = field(default_factory=Counter)
    variable_units: dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    variable_long_names: dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    members_by_site: dict[int, set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    time_lengths: Counter = field(default_factory=Counter)
    n_files: int = 0
    n_fill_values: int = 0
    n_equality_pair_checked: int = 0
    n_equality_pair_unequal: int = 0
    unequal_examples: list[str] = field(default_factory=list)
    path_mismatches: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)


# ── netCDF-3 classic header parsing ──────────────────────────────────────────
#
# Format reference: https://docs.unidata.ucar.edu/nug/current/file_format_
# specifications.html. Only what this survey needs is implemented: dimensions,
# variable names, variable attributes, and the offset of each variable's data.

_MAGIC = b"CDF"
_NC_DIMENSION, _NC_VARIABLE, _NC_ATTRIBUTE = 10, 11, 12

#: netCDF type tag -> (struct format character, size in bytes).
_NC_TYPES = {
    1: ("b", 1),  # byte
    2: ("c", 1),  # char
    3: ("h", 2),  # short
    4: ("i", 4),  # int
    5: ("f", 4),  # float
    6: ("d", 8),  # double
}


@dataclass
class NcVariable:
    """One variable of a netCDF-3 file, as far as this survey cares."""

    name: str
    dimension_ids: tuple[int, ...]
    attributes: dict[str, object]
    type_tag: int
    data_offset: int
    n_bytes: int


@dataclass
class NcHeader:
    """The parts of a netCDF-3 header this survey reads.

    A dimension whose stored size is 0 is the *record* (unlimited) dimension;
    its true length is ``n_records``, which is stored separately in the header.
    :meth:`dimension_sizes` resolves that so callers do not have to.
    """

    dimensions: list[tuple[str, int]]
    global_attributes: dict[str, object]
    variables: list[NcVariable]
    n_records: int

    def dimension_sizes(self) -> dict[str, int]:
        """Dimension name to length, with the record dimension resolved."""
        return {
            name: (self.n_records if size == 0 else size)
            for name, size in self.dimensions
        }


class _Reader:
    """A cursor over a netCDF-3 header, with the padding rules built in."""

    def __init__(self, buffer: bytes) -> None:
        self._buf = buffer
        self._pos = 0

    def take(self, n: int) -> bytes:
        if self._pos + n > len(self._buf):
            raise NetCDFParseError("header ends mid-record")
        chunk = self._buf[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def int32(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def name(self) -> str:
        """A counted string, padded to a 4-byte boundary."""
        length = self.int32()
        raw = self.take(length)
        self.take(-length % 4)
        return raw.decode("utf-8", errors="replace")

    def values(self, type_tag: int, count: int) -> object:
        """An attribute's value, padded to a 4-byte boundary."""
        if type_tag not in _NC_TYPES:
            raise NetCDFParseError(f"unknown netCDF type tag {type_tag}")
        fmt, size = _NC_TYPES[type_tag]
        raw = self.take(size * count)
        self.take(-(size * count) % 4)
        if type_tag == 2:
            return raw.decode("utf-8", errors="replace").rstrip("\x00")
        decoded = struct.unpack(f">{count}{fmt}", raw)
        return decoded[0] if count == 1 else list(decoded)


def _parse_attributes(reader: _Reader) -> dict[str, object]:
    """Read an attribute list, which may be the ABSENT marker instead."""
    tag = reader.int32()
    count = reader.int32()
    if tag == 0 and count == 0:
        return {}
    if tag != _NC_ATTRIBUTE:
        raise NetCDFParseError(f"expected an attribute list, got tag {tag}")
    return {
        reader.name(): reader.values(reader.int32(), reader.int32())
        for _ in range(count)
    }


def _parse_dimensions(reader: _Reader) -> list[tuple[str, int]]:
    """Read the dimension list, which may be the ABSENT marker instead."""
    tag = reader.int32()
    count = reader.int32()
    if tag == 0 and count == 0:
        return []
    if tag != _NC_DIMENSION:
        raise NetCDFParseError(f"expected a dimension list, got tag {tag}")
    return [(reader.name(), reader.int32()) for _ in range(count)]


def _parse_variables(reader: _Reader, offset_size: int) -> list[NcVariable]:
    """Read the variable list, which may be the ABSENT marker instead."""
    tag = reader.int32()
    count = reader.int32()
    if tag == 0 and count == 0:
        return []
    if tag != _NC_VARIABLE:
        raise NetCDFParseError(f"expected a variable list, got tag {tag}")

    variables = []
    for _ in range(count):
        name = reader.name()
        dimension_ids = tuple(reader.int32() for _ in range(reader.int32()))
        attributes = _parse_attributes(reader)
        type_tag = reader.int32()
        n_bytes = reader.int32()
        raw_offset = reader.take(offset_size)
        offset = struct.unpack(">q" if offset_size == 8 else ">i", raw_offset)[0]
        variables.append(
            NcVariable(name, dimension_ids, attributes, type_tag, offset, n_bytes)
        )
    return variables


def read_header(path: Path) -> tuple[NcHeader, bytes]:
    """Parse the header of a netCDF-3 classic file.

    Returns the header and the raw bytes that were read, so that small
    variables can be decoded without a second trip to the filesystem.
    """
    raw = path.read_bytes()
    if raw[:3] != _MAGIC:
        raise NetCDFParseError(f"not a netCDF-3 classic file (magic {raw[:4]!r})")
    version = raw[3]
    if version not in (1, 2):
        raise NetCDFParseError(f"unsupported netCDF-3 version byte {version}")

    reader = _Reader(raw[4:])
    n_records = reader.int32()
    header = NcHeader(
        dimensions=_parse_dimensions(reader),
        global_attributes=_parse_attributes(reader),
        variables=_parse_variables(reader, offset_size=8 if version == 2 else 4),
        n_records=n_records,
    )
    return header, raw


def read_values(variable: NcVariable, raw: bytes) -> list[float]:
    """Decode a numeric variable's data from the already-read file bytes.

    For a record variable this returns the first record only, which is all
    these files have; the survey reports the record count separately so that a
    file with more would be visible rather than silently truncated.
    """
    fmt, size = _NC_TYPES[variable.type_tag]
    count = variable.n_bytes // size
    chunk = raw[variable.data_offset : variable.data_offset + variable.n_bytes]
    if len(chunk) < variable.n_bytes:
        raise NetCDFParseError(f"file truncated before {variable.name!r} data")
    return list(struct.unpack(f">{count}{fmt}", chunk))


if __name__ == "__main__":
    raise SystemExit(main())
