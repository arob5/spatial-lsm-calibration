#!/usr/bin/env python
"""Make the raw NEE files out of AmeriFlux's FLUXNET FULLSET files.

**Not part of the raw-to-processed pipeline.** This script sits upstream of
``data/raw/``: it *creates* the raw NEE inputs rather than processing them, it
needs the SCC, where the download is, and it is run again only when the
download changes. The ingest that reads what it wrote is
``scripts/ingest_net_ecosystem_exchange.py``, after
``scripts/raw_sources/build_ameriflux_towers.py`` has made the tower table.

Overview
--------
Read every half-hourly and hourly FULLSET CSV under the source root, check each
against its zip and against the source format, and lay the kept columns of
every tower on one local-standard-time axis per resolution, as
``data/raw/net_ecosystem_exchange/ameriflux_nee_half_hourly.nc`` and
``ameriflux_nee_hourly.nc``. Values are copied unchanged, in the source's
column names; nothing is shifted to UTC, matched to a site or filtered.

Input data
----------
``--root``, default ``data/raw/net_ecosystem_exchange/fluxnet/``
    The AmeriFlux FLUXNET download: for each tower an
    ``AMF_<tower>_FLUXNET_FULLSET_<years>_<version>.zip`` and, unzipped beside
    it, ``AMF_<tower>_FLUXNET_FULLSET_{HH,HR}_<years>_<version>.csv``, one per
    tower, whose format ``sipnet_calibration.net_ecosystem_exchange`` describes
    and ``read_source_file`` enforces. Other files there are passed over.

Output data
-----------
``--out-dir``, default ``data/raw/net_ecosystem_exchange/``
    ``ameriflux_nee_half_hourly.nc`` and ``ameriflux_nee_hourly.nc``: the kept
    columns on ``(tower, time_index)`` in source names, ``float64`` values with
    ``NaN`` and ``int8`` flags with ``-1`` for missing, the shared stamps, and
    each tower's file, version, years, absent columns and md5.
    ``read_raw`` documents and checks them.

Notes
-----
Each CSV is checked against the CRC-32 its zip records before it is read, so
a CSV edited or truncated after the download is refused. ``--no-zip-check``
skips that, for a directory without the zips.

Some towers' files carry no constant-u*-threshold (CUT) columns; they are
stored as missing and named in ``absent_columns``. The report printed at the
end -- per resolution, the towers, their years and versions, and the md5 of
each written file -- is what ``provenance.md`` records.

Output is written to a ``.partial`` path and renamed only once it reads back
bit-identical through ``read_raw``.

Usage
-----
On the SCC, from a checkout with its venv synced::

    uv run python scripts/raw_sources/convert_ameriflux_nee.py --jobs 8

or through the batch system::

    qsub scripts/raw_sources/convert_ameriflux_nee.qsub
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import xarray as xr

from sipnet_calibration.net_ecosystem_exchange import (
    RESOLUTIONS,
    SOURCE,
    Resolution,
    SourceFile,
    build_raw,
    default_raw_dir,
    default_source_root,
    discover_source_files,
    parse_file_name,
    raw_encoding,
    raw_path,
    read_raw,
    read_source_file,
)

SCRIPT = "scripts/raw_sources/convert_ameriflux_nee.py"


class ConversionError(Exception):
    """A check failed, or the download is not what it claims to be."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root if args.root is not None else default_source_root()
    out_dir = args.out_dir if args.out_dir is not None else default_raw_dir()
    try:
        paths = discover_source_files(root)
        if args.towers:
            paths = select_towers(paths, args.towers, out_dir_given=args.out_dir is not None)
        print(f"{len(paths)} FULLSET files under {root}", flush=True)
        fingerprints = {path.name: _fingerprint(path) for path in paths}
        if not args.no_zip_check:
            check_every_file_matches_its_zip(paths, fingerprints)
        md5 = {name: digest for name, (_, digest) in fingerprints.items()}
        files = read_all_files(paths, jobs=args.jobs)
        for resolution in RESOLUTIONS.values():
            of_resolution = [record for record in files if record.resolution == resolution]
            if not of_resolution:
                print(f"no {resolution.name} files", flush=True)
                continue
            dataset = build_raw(
                of_resolution, resolution, md5_by_file=md5, source_root=str(root), conversion_script=SCRIPT
            )
            out = raw_path(resolution, out_dir)
            write_raw(dataset, out)
            print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, md5 {_md5(out)})")
            print(describe_raw(resolution, of_resolution))
    except (ConversionError, OSError, ValueError, BrokenProcessPool) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="The FLUXNET download. Default: data/raw/net_ecosystem_exchange/fluxnet.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write. Default: data/raw/net_ecosystem_exchange.",
    )
    parser.add_argument("--jobs", type=int, default=8, help="Worker processes reading files. Default 8.")
    parser.add_argument(
        "--towers", nargs="+", default=None,
        help="Convert only these towers, for a trial run; needs --out-dir, since the result is "
        "not the raw input.",
    )
    parser.add_argument(
        "--no-zip-check", action="store_true",
        help="Skip checking each CSV against the CRC-32 its zip records.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def select_towers(paths: list[Path], towers: list[str], *, out_dir_given: bool) -> list[Path]:
    """The files of *towers*, refusing a trial run aimed at the real raw files."""
    if not out_dir_given:
        raise ConversionError(
            "--towers needs an explicit --out-dir: its output is a subset of the download, "
            f"and the default directory ({default_raw_dir()}) holds the real raw files."
        )
    chosen = [path for path in paths if parse_file_name(path.name)[0] in set(towers)]
    missing = sorted(set(towers) - {parse_file_name(path.name)[0] for path in chosen})
    if missing:
        raise ConversionError(f"no FULLSET file for {missing}")
    return chosen


def read_all_files(paths: list[Path], *, jobs: int) -> list[SourceFile]:
    """Parse every file, in parallel."""
    files: list[SourceFile] = []
    pool = ProcessPoolExecutor(max_workers=max(1, jobs))
    try:
        for i, record in enumerate(pool.map(read_source_file, paths), start=1):
            files.append(record)
            if i % 20 == 0 or i == len(paths):
                print(f"  ... {i} of {len(paths)} files", flush=True)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return files


def write_raw(dataset: xr.Dataset, out: Path) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    try:
        dataset.to_netcdf(partial, engine="h5netcdf", encoding=raw_encoding(dataset))
        check_round_trip(dataset, partial)
        partial.replace(out)
    finally:
        partial.unlink(missing_ok=True)


def describe_raw(resolution: Resolution, files: list[SourceFile]) -> str:
    """The run report: what provenance.md records."""
    versions = Counter(record.version for record in files)
    lines = [
        f"{resolution.name}: {len(files)} towers; versions "
        + ", ".join(f"{version} x{count}" for version, count in sorted(versions.items())),
        "tower     years      version  steps with NEE_VUT_REF  absent columns",
    ]
    for record in sorted(files, key=lambda r: r.tower):
        present = int(np.isfinite(record.values["NEE_VUT_REF"]).sum())
        absent = "CUT" if record.absent_columns else ""
        lines.append(
            f"{record.tower:8s}  {record.first_year}-{record.last_year}  {record.version:7s}  "
            f"{present:21d}  {absent}"
        )
    return "\n".join(lines)


# ── supporting helpers ────────────────────────────────────────────────────────


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> tuple[int, str]:
    """The file's CRC-32 and md5, in one read."""
    crc, digest = 0, hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            crc = zlib.crc32(chunk, crc)
            digest.update(chunk)
    return crc, digest.hexdigest()


def _zip_for(path: Path) -> Path:
    """The zip a FULLSET CSV came out of: the same name without the resolution code."""
    tower, resolution, first, last, version = parse_file_name(path.name)
    return path.with_name(f"AMF_{tower}_FLUXNET_FULLSET_{first}-{last}_{version}.zip")


# ── checks ────────────────────────────────────────────────────────────────────


def check_every_file_matches_its_zip(paths: list[Path], fingerprints: dict[str, tuple[int, str]]) -> None:
    """Raise unless each CSV has the size and CRC-32 its zip records for it."""
    for path in paths:
        archive = _zip_for(path)
        if not archive.is_file():
            raise ConversionError(f"{path.name}: no zip {archive.name} beside it to check against")
        with zipfile.ZipFile(archive) as zipped:
            try:
                info = zipped.getinfo(path.name)
            except KeyError as error:
                raise ConversionError(f"{archive.name} does not hold {path.name}") from error
        if path.stat().st_size != info.file_size or fingerprints[path.name][0] != info.CRC:
            raise ConversionError(
                f"{path.name} differs from the copy in {archive.name}: the CSV was changed after "
                "the download. Re-extract it, or find out who changed it."
            )


def check_round_trip(dataset: xr.Dataset, partial: Path) -> None:
    """Raise unless the written file reads back bit-identical through the library."""
    with read_raw(partial) as read_back:
        for name in SOURCE.names:
            if not np.array_equal(dataset[name].values, read_back[name].values, equal_nan=True):
                raise ConversionError(f"{name} did not round-trip bit for bit through {partial}")
        for name in dataset.coords:
            if not np.array_equal(dataset[name].values, read_back[name].values):
                raise ConversionError(f"coordinate {name} did not round-trip through {partial}")


if __name__ == "__main__":
    sys.exit(main())
