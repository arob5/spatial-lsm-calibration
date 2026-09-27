#!/usr/bin/env python
"""Make the tracked raw initial condition file out of PEcAn's per-member netCDFs.

Overview
--------
Read every ``<site>/IC_site_<site>_<member>.nc`` under the source directory,
check each against the source template, and lay the values on
``(site, member)`` as the tracked raw file, in the source files' variable
names, units strings and 1-based member index. Values are copied bit for bit;
nothing is renamed, converted or masked.

Input data
----------
``--source-directory``, default
:func:`sipnet_calibration.initial_conditions.default_source_root`
    The source tree: one directory per site, named by its site id, holding one
    netCDF-3 classic file per ensemble member. The file format is described in
    ``sipnet_calibration.initial_conditions`` and parsed by its
    ``read_source_directory``, which refuses anything outside the template.

``--site-table``, default :func:`sipnet_calibration.sites.default_sites_path`
    The site table, used only to check that the site directories are exactly
    the pool. Skipped with a note if the default table is absent.

Output data
-----------
``--output``, default :func:`sipnet_calibration.initial_conditions.raw_path`
    Five ``float64`` variables on ``(site, member)``, ``NaN`` where a site's
    files lack the variable, with the source attribute strings; ``site``
    ``int32`` and ``member`` ``int16`` ascending; the source's time metadata,
    the file count and the conversion record as global attributes.
    ``sipnet_calibration.initial_conditions.read_raw`` documents and checks it.

Notes
-----
**Not part of the raw-to-processed pipeline.** This script sits upstream of
``data/raw/``: it *creates* a raw input rather than processing one, it needs
the SCC, where the source files are, and it ran once, in 2026-09, to produce
the file that is now in version control. Run it again only if the source files
themselves change; the ingest that reads what it wrote is
``scripts/ingest_initial_conditions.py``.

The report printed at the end -- per-variable coverage, ranges and negative
counts, the variable-set signatures, and the md5 of the written file -- is
what ``data/raw/initial_conditions/provenance.md`` records. The numbers are
printed rather than checked because they describe the source data, not an
invariant of ours; the invariants (a complete rectangle, presence uniform over
members, the source template in every file) are checked by
``initial_conditions.read_source_directory`` and ``build_raw``.

The file is written through :func:`sipnet_calibration.io.write_checked`, and
its check reads it back with ``read_raw``.

Usage
-----
On the SCC, from the project checkout with its venv synced::

    uv run python scripts/raw_sources/convert_initial_conditions.py --jobs 16

or through the batch system, on the group's buy-in nodes::

    qsub scripts/raw_sources/convert_initial_conditions.qsub

A trial run on a partial tree, which is not the pool, so its output must not
be committed::

    uv run python scripts/raw_sources/convert_initial_conditions.py \\
        --limit-sites 2 --output /tmp/trial.nc
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
    RAW_MEMBER,
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
from sipnet_calibration.validation import range_summary, truncated

#: This script, as the raw file's conversion record names it.
SCRIPT = "scripts/raw_sources/convert_initial_conditions.py"


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Read the source tree and write the raw file, or report why not."""
    args = parse_args(argv)
    source_directory = args.source_directory or default_source_root()
    output = args.output or raw_path()
    try:
        site_ids = find_site_ids(source_directory)
        site_ids = choose_site_ids(
            site_ids,
            limit=args.limit_sites,
            output_given=args.output is not None,
            site_table_path=args.site_table or default_sites_path(),
            site_table_given=args.site_table is not None,
        )
        print(f"{len(site_ids)} site directories under {source_directory}", flush=True)

        files = read_source_files(source_directory, site_ids, jobs=args.jobs)
        dataset = build_raw(files, source_root=str(source_directory), conversion_script=SCRIPT)
        report = describe_raw_file(dataset, files)
        write_raw_file(dataset, output)
        print(f"wrote {output}  ({output.stat().st_size / 1e6:.1f} MB, md5 {file_md5(output)})")
        print(report)
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
        "--source-directory",
        type=Path,
        default=None,
        help=f"The source tree. Default: {default_source_root()}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Where to write. Default: {raw_path()}.",
    )
    parser.add_argument(
        "--site-table",
        type=Path,
        default=None,
        help="The site table, to check the directories are the pool. Default: "
        f"{default_sites_path()}; skipped if absent.",
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


def find_site_ids(source_directory: Path) -> list[int]:
    """The site ids of the site directories under *source_directory*, ascending."""
    check_source_directory_exists(source_directory)
    site_ids, strays = [], []
    for entry in sorted(source_directory.iterdir()):
        if entry.name.startswith("."):
            continue  # filesystem debris such as .DS_Store
        if entry.is_dir() and _is_site_directory_name(entry.name):
            site_ids.append(int(entry.name))
        else:
            strays.append(entry.name)
    check_source_directory_holds_only_site_directories(source_directory, strays)
    check_source_directory_holds_a_site(source_directory, site_ids)
    return sorted(site_ids)


def choose_site_ids(
    site_ids: list[int],
    *,
    limit: int | None,
    output_given: bool,
    site_table_path: Path,
    site_table_given: bool,
) -> list[int]:
    """The site ids to read: the pool, checked against the site table, or a prefix."""
    if limit is not None:
        # A trial run reads a prefix of the tree, which is not the pool, so the
        # pool check is skipped and the result must not be committed.
        check_limit_is_positive(limit)
        check_trial_run_names_its_output(output_given)
        print(f"note: --limit-sites {limit}; the pool check is skipped", flush=True)
        return site_ids[:limit]
    if site_table_given:
        check_site_table_exists(site_table_path)
    if not site_table_path.exists():
        print(f"note: {site_table_path} absent; the pool check is left to the ingest", flush=True)
        return site_ids
    check_sites_are_the_site_table(
        load_sites(site_table_path), site_ids, message_name="the site directories"
    )
    return site_ids


def read_source_files(source_directory: Path, site_ids: list[int], *, jobs: int) -> list[SourceFile]:
    """Every file of every site, parsed in parallel over sites."""
    files: list[SourceFile] = []
    pool = ProcessPoolExecutor(max_workers=max(1, jobs))
    try:
        _read_into(files, pool, source_directory, site_ids)
    except BrokenProcessPool as error:
        pool.shutdown(wait=False, cancel_futures=True)
        raise IngestError(
            f"a worker process reading the source files died ({error}); rerun with fewer "
            "--jobs, or on a node with more memory."
        ) from error
    except BaseException:
        # A bad file should stop the run now, not after every other file is read.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return files


def describe_raw_file(dataset: xr.Dataset, files: list[SourceFile]) -> str:
    """The run report: what provenance.md records."""
    lines = [
        f"sites {dataset.sizes[SITE]}  members {dataset.sizes[RAW_MEMBER]}  files {len(files)}",
        "variable                       sites   min          median       max          negative",
    ]
    for name in SOURCE.names:
        values = dataset[name].values
        present = np.isfinite(values)
        lines.append(
            f"{name:30s} {int(present.any(axis=1).sum()):5d}   {range_summary(values[present])}"
        )
    signatures = Counter(tuple(sorted(record.values)) for record in files)
    lines.append("variable sets:")
    for signature, count in signatures.most_common():
        lines.append(f"  {count:7d} files: {list(signature)}")
    return "\n".join(lines)


def write_raw_file(dataset: xr.Dataset, path: Path) -> None:
    """Write through a ``.partial`` file, moved in once it reads back bit for bit."""
    write_checked(
        path,
        write=lambda partial: dataset.to_netcdf(
            partial, engine="h5netcdf", encoding=raw_encoding(dataset)
        ),
        check=lambda partial: check_written_file_reads_back_identically(dataset, partial),
    )


# ── supporting types and helpers ──────────────────────────────────────────────


class IngestError(RuntimeError):
    """The source tree or the written file breaks an invariant of the raw file."""


def _read_into(
    files: list[SourceFile],
    pool: ProcessPoolExecutor,
    source_directory: Path,
    site_ids: list[int],
) -> None:
    """Extend *files* with every file of every site, read on *pool*."""
    for count, site_files in enumerate(
        pool.map(read_source_directory, [source_directory] * len(site_ids), site_ids, chunksize=8),
        start=1,
    ):
        files.extend(site_files)
        if count % 500 == 0 or count == len(site_ids):
            print(f"  ... {count} of {len(site_ids)} sites, {len(files)} files", flush=True)


def _is_site_directory_name(name: str) -> bool:
    """Whether *name* is a site id written plainly: ASCII digits, no leading zero."""
    return name.isascii() and name.isdigit() and str(int(name)) == name


# ── checks ────────────────────────────────────────────────────────────────────


def check_source_directory_exists(source_directory: Path) -> None:
    """The source directory is a directory."""
    if not source_directory.is_dir():
        raise FileNotFoundError(
            f"source directory {source_directory} is not a directory; pass the source "
            "tree with --source-directory."
        )


def check_source_directory_holds_only_site_directories(
    source_directory: Path, strays: list[str]
) -> None:
    """The source directory holds nothing but site directories."""
    if strays:
        raise IngestError(
            f"{source_directory} holds entries that are not site directories, "
            f"{truncated(strays)}; the source tree is one numeric directory per site and "
            "nothing else, so move them out."
        )


def check_source_directory_holds_a_site(source_directory: Path, site_ids: list[int]) -> None:
    """The source directory holds at least one site directory."""
    if not site_ids:
        raise IngestError(
            f"{source_directory} holds no site directories; pass the source tree with "
            "--source-directory."
        )


def check_limit_is_positive(limit: int) -> None:
    """A trial run's ``--limit-sites`` is at least 1."""
    if limit < 1:
        raise IngestError(f"--limit-sites must be at least 1, got {limit}; pass a positive count.")


def check_trial_run_names_its_output(output_given: bool) -> None:
    """A trial run names its ``--output``, so it cannot overwrite the tracked file."""
    if not output_given:
        raise IngestError(
            "--limit-sites needs an explicit --output, since its output is a prefix of the "
            f"tree rather than the pool and the default path ({raw_path()}) is the tracked "
            "raw file; name another path with --output."
        )


def check_site_table_exists(site_table_path: Path) -> None:
    """A site table named on the command line exists."""
    if not site_table_path.exists():
        raise FileNotFoundError(
            f"site table {site_table_path} does not exist; build it with "
            "scripts/ingest_sites.py or name another with --site-table."
        )


def check_written_file_reads_back_identically(dataset: xr.Dataset, partial: Path) -> None:
    """The written file reads back through the library as what was built."""
    with read_raw(partial) as read_back:
        check_read_back_is_identical(dataset, read_back.load(), message_name=str(partial))


def check_read_back_is_identical(
    dataset: xr.Dataset, read_back: xr.Dataset, *, message_name: str
) -> None:
    """The file read back is identical to what was written, NaN for NaN."""
    if not read_back.identical(dataset):
        raise IngestError(
            f"{message_name}: the written file does not read back identical to what was "
            "built; inspect the kept partial file."
        )


if __name__ == "__main__":
    raise SystemExit(main())
