#!/usr/bin/env python
"""Build the processed net ecosystem exchange files from the raw files.

Overview
--------
For each series -- one NEE estimate at one resolution, one
``NetEcosystemExchangeSpec`` -- read the raw file of its resolution and the
tower table, keep each primary tower's series through the source's reader,
move it to UTC by the tower's offset, place it on its pool site, and write
``data/processed/net_ecosystem_exchange/<name>.nc``. Every decision about a
series lives in its spec and every decision about a tower in the tower table;
this script is the orchestration and the checks, and its round trip reads each
file back with ``load_net_ecosystem_exchange``.

Input data
----------
``--raw-dir``, default ``data/raw/net_ecosystem_exchange/``
    ``ameriflux_nee_half_hourly.nc`` and ``ameriflux_nee_hourly.nc``, the
    converted FULLSET columns on a local-standard-time axis, read by
    ``read_raw``.

``--tower-table``, default ``data/raw/net_ecosystem_exchange/ameriflux_towers.csv`` (tracked)
    Each tower's pool site, primary status, UTC offset and exclusion, read by
    ``read_tower_table``.

``--site-table``, default ``data/processed/sites/sites.csv``
    The site table, for each site's ``lon``/``lat``.

Output data
-----------
``--out-dir``, default ``data/processed/net_ecosystem_exchange/``, one file per
series::

    value(site, time)                float64, umol m-2 s-1 CO2, the series
    quality_flag(site, time)         int8, 0 measured, 1-3 gap-fill quality, -1 none
    random_uncertainty(site, time)   float64
    joint_uncertainty(site, time)    float64
    night(site, time)                int8

``time`` is the UTC end of each step, with CF ``time_bounds``; ``site`` the
pool sites with a primary tower that carries the series.
``sipnet_calibration.net_ecosystem_exchange`` documents the data model.

Notes
-----
The ingest changes structure, never values: no quality filter, no preferred
estimate, no fallback from one estimate to another, no aggregation. The UTC
shift relabels each tower's steps by a fixed whole number of steps.

Output is written through ``io.write_checked``: to a ``.partial`` path, renamed
only once it reads back identical through the library loader.

Usage
-----
::

    python scripts/ingest_net_ecosystem_exchange.py
    python scripts/ingest_net_ecosystem_exchange.py --series ameriflux_nee_half_hourly_ustar_variable
    python scripts/ingest_net_ecosystem_exchange.py --describe     # the specs, no I/O
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.conventions import SITE, TIME
from sipnet_calibration.io import write_checked
from sipnet_calibration.net_ecosystem_exchange import (
    NET_ECOSYSTEM_EXCHANGE,
    NET_ECOSYSTEM_EXCHANGE_NAMES,
    QUALITY_FLAG,
    SOURCE_READERS,
    TOWER,
    VALUE,
    NetEcosystemExchangeSpec,
    build_net_ecosystem_exchange,
    default_net_ecosystem_exchange_directory,
    default_raw_directory,
    describe,
    load_net_ecosystem_exchange,
    net_ecosystem_exchange_path,
    netcdf_encoding,
    raw_path,
    read_raw,
    read_tower_table,
    resolve_net_ecosystem_exchange,
    tower_table_path,
)
from sipnet_calibration.sites import default_site_table_path, load_sites


class IngestError(Exception):
    """A check failed, or an input is not what it claims to be."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = (
        [resolve_net_ecosystem_exchange(name) for name in args.series]
        if args.series
        else list(NET_ECOSYSTEM_EXCHANGE)
    )
    if args.describe:
        print("\n\n".join(describe(spec) for spec in specs))
        return 0
    raw_directory = args.raw_dir if args.raw_dir is not None else default_raw_directory()
    out_dir = args.out_dir if args.out_dir is not None else default_net_ecosystem_exchange_directory()
    try:
        tower_table = read_tower_table(args.tower_table or tower_table_path())
        site_table = load_sites(args.site_table or default_site_table_path())
        for spec in specs:
            dataset = ingest(spec, raw_directory, tower_table, site_table, out_dir)
            print(describe_processed_file(dataset, spec, net_ecosystem_exchange_path(spec, out_dir)))
    except (IngestError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--describe", action="store_true", help="Print each spec and exit without reading data.")
    parser.add_argument(
        "--series",
        action="append",
        choices=NET_ECOSYSTEM_EXCHANGE_NAMES,
        metavar="NAME",
        help="A series to build; repeatable. Default: all of " + ", ".join(NET_ECOSYSTEM_EXCHANGE_NAMES),
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Default: data/raw/net_ecosystem_exchange.")
    parser.add_argument(
        "--tower-table",
        type=Path,
        default=None,
        help="Default: the tracked data/raw/net_ecosystem_exchange/ameriflux_towers.csv.",
    )
    parser.add_argument("--site-table", type=Path, default=None, help="Default: data/processed/sites/sites.csv.")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Default: data/processed/net_ecosystem_exchange."
    )
    return parser.parse_args(argv)


# ── the ingest steps, in the order main calls them ────────────────────────────


def ingest(
    spec: NetEcosystemExchangeSpec,
    raw_directory: Path,
    tower_table: pd.DataFrame,
    site_table: pd.DataFrame,
    out_dir: Path,
) -> xr.Dataset:
    """Read, check, build and write one series."""
    with read_raw(raw_path(spec.resolution, raw_directory)) as raw:
        check_raw(raw, tower_table)
        tower_series = SOURCE_READERS[spec.source](spec, raw, tower_table)
    dataset = build_net_ecosystem_exchange(spec, tower_series, tower_table, site_table)
    check_processed(dataset, spec)
    write_processed_file(dataset, net_ecosystem_exchange_path(spec, out_dir), spec)
    return dataset


def check_raw(raw: xr.Dataset, tower_table: pd.DataFrame) -> None:
    """Every check on a raw file beyond the schema ``read_raw`` enforces."""
    check_every_raw_tower_is_in_the_tower_table(raw, tower_table)
    check_every_primary_tower_is_in_its_raw_file(raw, tower_table)
    check_resolutions_agree(raw, tower_table)


def check_processed(dataset: xr.Dataset, spec: NetEcosystemExchangeSpec) -> None:
    """Every check on a built series beyond the schema the loader enforces."""
    check_the_series_has_sites(dataset, spec)
    check_quality_flag_is_set_wherever_there_is_a_value(dataset, spec)


def write_processed_file(dataset: xr.Dataset, out: Path, spec: NetEcosystemExchangeSpec) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename."""
    write_checked(
        out,
        write=lambda partial: dataset.to_netcdf(
            partial, engine="h5netcdf", encoding=netcdf_encoding(dataset)
        ),
        check=lambda partial: check_round_trip(dataset, partial, spec),
    )


def describe_processed_file(dataset: xr.Dataset, spec: NetEcosystemExchangeSpec, out: Path) -> str:
    """A short report of what was written, for the run log."""
    value = dataset[VALUE].values
    present = np.isfinite(value)
    flag = dataset[QUALITY_FLAG].values if QUALITY_FLAG in dataset else None
    measured = int(((flag == 0) & present).sum()) if flag is not None else 0
    lines = [
        f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)",
        f"  {spec.name}: {dataset.sizes[SITE]} sites, {dataset.sizes[TIME]} steps, "
        f"{present.mean():.1%} of cells with a value, {measured / max(int(present.sum()), 1):.1%} of those measured",
    ]
    if present.any():
        lines.append(
            f"  value min {np.nanmin(value):.4g}  median {np.nanmedian(value):.4g}  max {np.nanmax(value):.4g} {spec.units}"
        )
    for name in ("towers_excluded", "towers_without_the_series"):
        if dataset.attrs.get(name):
            lines.append(f"  {name}: {dataset.attrs[name]}")
    return "\n".join(lines)


# ── checks ────────────────────────────────────────────────────────────────────


def check_every_raw_tower_is_in_the_tower_table(raw: xr.Dataset, tower_table: pd.DataFrame) -> None:
    """Raise unless the tower table has a row for every tower of the raw file."""
    missing = sorted(set(raw[TOWER].values.tolist()) - set(tower_table["tower"]))
    if missing:
        raise IngestError(
            f"towers of {raw.attrs.get('resolution')} raw file not in the tower table: {missing[:10]}. "
            "Rebuild the table with scripts/raw_sources/build_ameriflux_towers.py."
        )


def check_every_primary_tower_is_in_its_raw_file(raw: xr.Dataset, tower_table: pd.DataFrame) -> None:
    """Raise unless every primary tower of the raw file's resolution is in the raw file."""
    primary = tower_table[tower_table["primary"] & (tower_table["resolution_minutes"] == raw.attrs["resolution_minutes"])]
    missing = sorted(set(primary["tower"]) - set(raw[TOWER].values.tolist()))
    if missing:
        raise IngestError(
            f"primary towers of the tower table are not in the {raw.attrs.get('resolution')} raw file: "
            f"{missing[:10]}. The table and the raw files are from different runs."
        )


def check_resolutions_agree(raw: xr.Dataset, tower_table: pd.DataFrame) -> None:
    """Raise unless the table gives each raw tower the raw file's resolution."""
    rows = tower_table.set_index("tower").loc[raw[TOWER].values.tolist()]
    wrong = rows.index[rows["resolution_minutes"] != raw.attrs["resolution_minutes"]].tolist()
    if wrong:
        raise IngestError(f"the tower table gives {wrong[:10]} another resolution than their raw file")


def check_the_series_has_sites(dataset: xr.Dataset, spec: NetEcosystemExchangeSpec) -> None:
    """Raise if no tower carries the series: an empty processed file is a broken input."""
    if dataset.sizes[SITE] == 0:
        raise IngestError(f"{spec.name}: no primary tower carries {spec.value_column}")


def check_quality_flag_is_set_wherever_there_is_a_value(
    dataset: xr.Dataset, spec: NetEcosystemExchangeSpec
) -> None:
    """Raise if a value has no quality flag: measured and filled could not be told apart."""
    if QUALITY_FLAG not in dataset:
        return
    orphaned = np.isfinite(dataset[VALUE].values) & (dataset[QUALITY_FLAG].values < 0)
    if orphaned.any():
        raise IngestError(f"{spec.name}: {int(orphaned.sum())} values have no quality flag")


def check_round_trip(dataset: xr.Dataset, partial: Path, spec: NetEcosystemExchangeSpec) -> None:
    """Raise unless the written file reads back identical through the library loader."""
    with load_net_ecosystem_exchange(spec, partial) as written:
        written = written.load()
    if not written.identical(dataset):
        raise IngestError(f"{partial}: the written file does not read back identical to what was built.")


if __name__ == "__main__":
    raise SystemExit(main())
