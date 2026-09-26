#!/usr/bin/env python
"""Make the tracked raw initial condition file out of PEcAn's 800,000 netCDFs.

**Not part of the raw-to-processed pipeline.** This script sits upstream of
``data/raw/``: it *creates* a raw input rather than processing one, it needs
the SCC, where the source files are, and it ran once, in 2026-09, to produce
the file that is now in version control. Nothing in a normal working copy has
to run it. It lives under ``scripts/raw_sources/`` for that reason, beside no
other pipeline script; the ingest that reads what it wrote is
``scripts/ingest_initial_conditions.py``. Run this again only if the source
files themselves change.

Overview
--------
Read every ``<site>/IC_site_<site>_<member>.nc`` under the source root, check
each against the source template, and lay the values on ``(site, member)`` as
``data/raw/initial_conditions/pecan_pool_initial_conditions.nc``, in the source
files' variable names, units strings and 1-based member index. Values are
copied bit for bit; nothing is renamed, converted or masked.

Input data
----------
``--root``, default ``data/raw/initial_conditions/files/``
    The source tree: one directory per site, named by the 1-8000 site
    identifier, holding one netCDF-3 classic file per ensemble member. The
    file format is described in ``sipnet_calibration.initial_conditions`` and
    parsed by its ``read_source_directory``, which refuses anything outside
    the template.

``--sites``, default ``data/processed/sites/sites.csv``
    The site table, used only to check that the site directories are exactly
    the pool. Skipped with a note if the table is absent.

Output data
-----------
``--out``, default ``data/raw/initial_conditions/pecan_pool_initial_conditions.nc``
    Five ``float64`` variables on ``(site, member)``, ``NaN`` where a site's
    files lack the variable, with the source attribute strings; ``site``
    ``int32`` and ``member`` ``int16`` ascending; the source's time metadata,
    the file count and the conversion record as global attributes.
    ``sipnet_calibration.initial_conditions.read_raw`` documents and checks it.

Notes
-----
The report printed at the end -- per-variable coverage, ranges and negative
counts, the variable-set signatures, and the md5 of the written file -- is
what ``data/raw/initial_conditions/provenance.md`` records. The numbers are
printed rather than asserted because they describe the source data, not an
invariant of ours; the invariants (a complete rectangle, presence uniform over
members, the source template in every file) are the ``check_*`` functions and
the per-file checks in the library.

Output is written to a ``.partial`` path and renamed only once it reads back
bit-identical through ``read_raw``, so a failed run cannot leave a corrupt
file where the tracked one belongs.
A failed check keeps the ``.partial`` file for inspection and prints its
path (:func:`sipnet_calibration.io.write_checked`).

Usage
-----
On the SCC, from the project checkout with its venv synced::

    uv run python scripts/raw_sources/convert_initial_conditions.py --jobs 16

or through the batch system, on the group's buy-in nodes::

    qsub scripts/raw_sources/convert_initial_conditions.qsub

A quick check on a partial tree (the three files in a local checkout are not
the site pool and do not form a rectangle, so this fails at the pool check, or
at the rectangle check without a site table, by design)::

    uv run python scripts/raw_sources/convert_initial_conditions.py --root data/raw/initial_conditions/files
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import xarray as xr

from sipnet_calibration.conventions import SITE
from sipnet_calibration.initial_conditions import (
    SOURCE,
    SourceFile,
    build_raw,
    default_source_root,
    raw_encoding,
    raw_path,
    read_raw,
    read_source_directory,
)
from sipnet_calibration.io import file_md5, write_checked
from sipnet_calibration.sites import (
    check_sites_are_the_site_table,
    default_sites_path,
    load_sites,
)
from sipnet_calibration.validation import range_summary

SCRIPT = "scripts/raw_sources/convert_initial_conditions.py"


class ConversionError(Exception):
    """A check failed, or the tree is not what it claims to be."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root if args.root is not None else default_source_root()
    out = args.out if args.out is not None else raw_path()
    sites_path = args.sites if args.sites is not None else default_sites_path()

    try:
        sites = discover_sites(root)
        if args.limit_sites is not None:
            # A trial run reads a prefix of the tree, which is not the pool, so the
            # pool check is skipped and the result must not be committed.
            if args.limit_sites < 1:
                raise ConversionError("--limit-sites must be at least 1")
            if args.out is None:
                raise ConversionError(
                    "--limit-sites needs an explicit --out. Its output is a prefix of "
                    f"the tree rather than the pool, and the default path ({raw_path()}) "
                    "is the tracked raw file, which a trial run must not overwrite."
                )
            sites = sites[: args.limit_sites]
            print(f"note: --limit-sites {args.limit_sites}; the pool check is skipped", flush=True)
        else:
            check_site_directories_are_the_pool(sites, sites_path, explicit=args.sites is not None)
        print(f"{len(sites)} site directories under {root}", flush=True)

        files = read_all_files(root, sites, jobs=args.jobs)
        dataset = build_raw(files, source_root=str(root), conversion_script=SCRIPT)
        report = describe_raw(dataset, files)
        write_raw(dataset, out)
        print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, md5 {file_md5(out)})")
        print(report)
    except (ConversionError, OSError, ValueError, KeyError, BrokenProcessPool) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="The source tree. Default: data/raw/initial_conditions/files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write. Default: data/raw/initial_conditions/pecan_pool_initial_conditions.nc.",
    )
    parser.add_argument(
        "--sites",
        type=Path,
        default=None,
        help="The site table, to check the directories are the pool. Default: "
        "data/processed/sites/sites.csv; skipped if absent.",
    )
    parser.add_argument(
        "--jobs", type=int, default=8, help="Worker processes reading files. Default 8."
    )
    parser.add_argument(
        "--limit-sites",
        type=int,
        default=None,
        help="Read only the first N site directories, for a trial run; the result is "
        "not the raw input and must not be committed.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def discover_sites(root: Path) -> list[int]:
    """The site directories under *root*, ascending, refusing anything else."""
    if not root.is_dir():
        raise ConversionError(f"source root {root} is not a directory")
    sites, strays = [], []
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue  # filesystem debris such as .DS_Store
        if entry.is_dir() and entry.name.isascii() and entry.name.isdigit() and str(int(entry.name)) == entry.name:
            sites.append(int(entry.name))
        else:
            strays.append(entry.name)
    if strays:
        raise ConversionError(
            f"{root} holds entries that are not site directories: {strays[:10]}. The "
            "source tree is one numeric directory per site and nothing else."
        )
    if not sites:
        raise ConversionError(f"{root} holds no site directories")
    return sorted(sites)


def read_all_files(root: Path, sites: list[int], *, jobs: int) -> list[SourceFile]:
    """Parse every file of every site, in parallel over sites."""
    files: list[SourceFile] = []
    pool = ProcessPoolExecutor(max_workers=max(1, jobs))
    try:
        for i, batch in enumerate(
            pool.map(read_source_directory, [root] * len(sites), sites, chunksize=8), start=1
        ):
            files.extend(batch)
            if i % 500 == 0 or i == len(sites):
                print(f"  ... {i} of {len(sites)} sites, {len(files)} files", flush=True)
    except BaseException:
        # A bad file should stop the run now, not after the other 799,999 are read.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return files


def write_raw(dataset: xr.Dataset, out: Path) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename."""
    write_checked(
        out,
        write=lambda partial: dataset.to_netcdf(
            partial, engine="h5netcdf", encoding=raw_encoding(dataset)
        ),
        check=lambda partial: check_round_trip(dataset, partial),
    )


def describe_raw(dataset: xr.Dataset, files: list[SourceFile]) -> str:
    """The run report: what provenance.md records."""
    lines = [
        f"sites {dataset.sizes[SITE]}  members {dataset.sizes['member']}  files {len(files)}",
        "variable                       sites   min          median       max          negative",
    ]
    for name in SOURCE.names:
        values = dataset[name].values
        present = np.isfinite(values)
        finite = values[present]
        lines.append(f"{name:30s} {int(present.any(axis=1).sum()):5d}   {range_summary(finite)}")
    signatures = Counter(tuple(sorted(record.values)) for record in files)
    lines.append("variable sets:")
    for signature, count in signatures.most_common():
        lines.append(f"  {count:7d} files: {list(signature)}")
    return "\n".join(lines)


# ── checks ────────────────────────────────────────────────────────────────────


def check_site_directories_are_the_pool(
    sites: list[int], sites_path: Path, *, explicit: bool
) -> None:
    """Raise unless the site directories are exactly the site table's pool.

    Skipped, with a note, when the *default* site table is not present; the
    ingest repeats the comparison against the written file. A table named on
    the command line has to exist.
    """
    if not sites_path.exists():
        check_a_named_site_table_exists(sites_path, explicit=explicit)
        print(f"note: {sites_path} absent; the pool check is left to the ingest", flush=True)
        return
    check_sites_are_the_site_table(
        sites, load_sites(sites_path), message_name="the site directories"
    )


def check_a_named_site_table_exists(sites_path: Path, *, explicit: bool) -> None:
    """A site table named on the command line exists."""
    if explicit and not sites_path.exists():
        raise ConversionError(
            f"site table {sites_path} does not exist; build it with scripts/ingest_sites.py "
            "or name another with --sites."
        )


def check_round_trip(dataset: xr.Dataset, partial: Path) -> None:
    """Raise unless the written file reads back bit-identical through the library."""
    with read_raw(partial) as read_back:
        for name in SOURCE.names:
            written, back = dataset[name].values, read_back[name].values
            if not np.array_equal(written, back, equal_nan=True):
                raise ConversionError(f"{name} did not round-trip bit for bit through {partial}")
            if dict(read_back[name].attrs) != dict(dataset[name].attrs):
                raise ConversionError(f"{name}'s attributes changed on the way to disk")
        for coordinate in ("site", "member"):
            if not np.array_equal(dataset[coordinate].values, read_back[coordinate].values):
                raise ConversionError(f"{coordinate} did not round-trip through {partial}")


if __name__ == "__main__":
    sys.exit(main())
