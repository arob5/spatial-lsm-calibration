#!/usr/bin/env python
"""Convert the producer's 800,000 initial condition files into one raw netCDF.

Overview
--------
Read every ``<site>/IC_site_<site>_<member>.nc`` under the source root, check
each against the source template, and lay the values on ``(site, member)`` as
``data/raw/initial_conditions/pecan_pool_initial_conditions.nc``, in the
producer's variable names, units strings and 1-based member index. Values are
copied bit for bit; nothing is renamed, converted or masked. The result is
the raw input ``ingest_initial_conditions.py`` reads, and it is tracked in
version control, so this script runs once, on the SCC, and again only if the
producer's files change.

Input data
----------
``--root``, default ``data/raw/initial_conditions/files/``
    The producer's tree: one directory per site, named by the 1-8000 site
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
printed rather than asserted because they describe the producer's data, not an
invariant of ours; the invariants (a complete rectangle, presence uniform over
members, the source template in every file) are the ``check_*`` functions and
the per-file checks in the library.

Output is written to a ``.partial`` path and renamed only once it reads back
bit-identical through ``read_raw``, so a failed run cannot leave a corrupt
file where the tracked one belongs.

Usage
-----
On the SCC, from the project checkout with its venv synced::

    uv run python scripts/convert_initial_conditions.py --jobs 16

or through the batch system, on the group's buy-in nodes::

    qsub scripts/convert_initial_conditions.qsub

A quick check on a partial tree (the three files in a local checkout do not
form a rectangle, so this fails at the rectangle check, by design)::

    uv run python scripts/convert_initial_conditions.py --root data/raw/initial_conditions/files
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

from sipnet_calibration.initial_conditions import (
    SOURCE_NAMES,
    SourceFile,
    build_raw,
    default_source_root,
    raw_encoding,
    raw_path,
    read_raw,
    read_source_directory,
)
from sipnet_calibration.sites import default_sites_path, load_sites

SCRIPT = "scripts/convert_initial_conditions.py"


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
        if args.limit_sites:
            sites = sites[: args.limit_sites]
        check_site_directories_are_the_pool(sites, sites_path)
        print(f"{len(sites)} site directories under {root}", flush=True)

        files = read_all_files(root, sites, jobs=args.jobs)
        dataset = build_raw(files, source_root=str(root), conversion_script=SCRIPT)
        write_raw(dataset, out)
        print(describe_raw(dataset, out, files))
    except (ConversionError, OSError, ValueError) as error:
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
        help="The producer's tree. Default: data/raw/initial_conditions/files.",
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
        if entry.is_dir() and entry.name.isdigit() and str(int(entry.name)) == entry.name:
            sites.append(int(entry.name))
        else:
            strays.append(entry.name)
    if strays:
        raise ConversionError(
            f"{root} holds entries that are not site directories: {strays[:10]}. The "
            "producer's tree is one numeric directory per site and nothing else."
        )
    if not sites:
        raise ConversionError(f"{root} holds no site directories")
    return sorted(sites)


def read_all_files(root: Path, sites: list[int], *, jobs: int) -> list[SourceFile]:
    """Parse every file of every site, in parallel over sites."""
    files: list[SourceFile] = []
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        for i, batch in enumerate(
            pool.map(read_source_directory, [root] * len(sites), sites, chunksize=8), start=1
        ):
            files.extend(batch)
            if i % 500 == 0 or i == len(sites):
                print(f"  ... {i} of {len(sites)} sites, {len(files)} files", flush=True)
    return files


def write_raw(dataset: xr.Dataset, out: Path) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    dataset.to_netcdf(partial, engine="h5netcdf", encoding=raw_encoding(dataset))
    check_round_trip(dataset, partial)
    partial.replace(out)


def describe_raw(dataset: xr.Dataset, out: Path, files: list[SourceFile]) -> str:
    """The run report: what provenance.md records."""
    lines = [
        f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, md5 {_md5(out)})",
        f"sites {dataset.sizes['site']}  members {dataset.sizes['member']}  files {len(files)}",
        "variable                       sites   min          median       max          negative",
    ]
    for name in SOURCE_NAMES:
        values = dataset[name].values
        present = np.isfinite(values)
        finite = values[present]
        lines.append(
            f"{name:30s} {int(present.any(axis=1).sum()):5d}   "
            f"{finite.min():<12.6g} {np.median(finite):<12.6g} {finite.max():<12.6g} "
            f"{int((finite < 0).sum())}"
        )
    signatures = Counter(tuple(sorted(record.values)) for record in files)
    lines.append("variable sets:")
    for signature, count in signatures.most_common():
        lines.append(f"  {count:7d} files: {list(signature)}")
    return "\n".join(lines)


# ── supporting helpers ────────────────────────────────────────────────────────


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── checks ────────────────────────────────────────────────────────────────────


def check_site_directories_are_the_pool(sites: list[int], sites_path: Path) -> None:
    """Raise unless the site directories are exactly the site table's pool.

    Skipped, with a note, when the site table is not present; the ingest
    repeats the comparison against the written file.
    """
    if not sites_path.exists():
        print(f"note: {sites_path} absent; the pool check is left to the ingest", flush=True)
        return
    pool = load_sites(sites_path)["site_id"].to_numpy(np.int64)
    found = np.asarray(sites, dtype=np.int64)
    if np.array_equal(found, np.sort(pool)):
        return
    missing = sorted(set(pool.tolist()) - set(sites))[:10]
    extra = sorted(set(sites) - set(pool.tolist()))[:10]
    raise ConversionError(
        f"site directories are not the site table's pool: {len(set(pool.tolist()) - set(sites))} "
        f"pool sites have no directory (first {missing}); {len(set(sites) - set(pool.tolist()))} "
        f"directories are not in the pool (first {extra})"
    )


def check_round_trip(dataset: xr.Dataset, partial: Path) -> None:
    """Raise unless the written file reads back bit-identical through the library."""
    with read_raw(partial) as read_back:
        for name in SOURCE_NAMES:
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
