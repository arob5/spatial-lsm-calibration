#!/usr/bin/env python
"""Make the tracked tower table: each AmeriFlux tower's pool site and clock.

**Not part of the raw-to-processed pipeline.** Like
``convert_ameriflux_nee.py`` beside it, this script *creates* a raw input: the
tracked ``data/raw/net_ecosystem_exchange/ameriflux_towers.csv``, the one record
of which pool site each tower is, which tower a site's products carry, each
tower's UTC offset, and why a tower is left out. It needs the raw NEE files,
not the SCC, and is run again when they or the site pool change.

Overview
--------
For every tower of the raw files, recover its UTC offset from its
``SW_IN_POT`` and check its measured shortwave against it; match it to a pool
site by the exact rule of ``sipnet_calibration.net_ecosystem_exchange.towers``;
choose one tower per site; and write the table.

Input data
----------
``--raw-dir``, default ``data/raw/net_ecosystem_exchange/``
    ``ameriflux_nee_half_hourly.nc`` and ``ameriflux_nee_hourly.nc``, from
    ``convert_ameriflux_nee.py``; ``read_raw`` checks them. Also where the two
    lists below are looked for by default.

``--site-list``, default ``<raw dir>/ameri_sites.tsv``
    AmeriFlux's site listing, downloaded beside the FLUXNET files: each site's
    coordinates, IGBP class and FLUXNET DOI. Not tracked, because it carries
    contact details.

``--pool-input-list``, default ``<raw dir>/Unmatched_Sites.csv``
    The reanalysis's list of AmeriFlux towers added to the site pool, in the
    order they were placed; tracked.

``--sites``, default ``data/processed/sites/sites.csv``
    The site table: the pool's names and cells.

Output data
-----------
``--out``, default ``<raw dir>/ameriflux_towers.csv``
    One row per tower, the columns of ``TOWER_COLUMNS``; ``read_tower_table``
    documents and checks them.

Notes
-----
The printed report -- how many towers match on each basis, how many sites
have more than one tower, and every exclusion with its reason -- is for the
run log and for reviewing the diff of the table. Numbers are printed rather
than asserted: they describe the download, and the invariants are
``read_tower_table``'s checks.

Output is written to a ``.partial`` path and renamed only once it reads back
through ``read_tower_table`` identical to what was built.

Usage
-----
::

    python scripts/raw_sources/build_ameriflux_towers.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sipnet_calibration.net_ecosystem_exchange import (
    MATCH_BASES,
    RESOLUTIONS,
    build_tower_table,
    default_raw_dir,
    raw_path,
    read_ameriflux_site_list,
    read_pool_input_list,
    read_raw,
    read_tower_table,
    summarize_raw,
    tower_table_path,
)
from sipnet_calibration.net_ecosystem_exchange.names import (
    ameriflux_site_list_path,
    pool_input_list_path,
)
from sipnet_calibration.sites import default_sites_path, load_sites


class BuildError(Exception):
    """A check failed, or an input is not what it claims to be."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir if args.raw_dir is not None else default_raw_dir()
    out = args.out if args.out is not None else tower_table_path(raw_dir)
    try:
        site_list = read_ameriflux_site_list(args.site_list or ameriflux_site_list_path(raw_dir))
        pool_input_list = read_pool_input_list(args.pool_input_list or pool_input_list_path(raw_dir))
        site_table = load_sites(args.sites or default_sites_path())
        summaries = summarize_every_raw_file(raw_dir, site_list)
        table = build_tower_table(summaries, site_list, pool_input_list, site_table)
        write_tower_table(table, out)
        print(describe_table(table, out))
    except (BuildError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Default: data/raw/net_ecosystem_exchange.")
    parser.add_argument("--site-list", type=Path, default=None, help="Default: <raw dir>/ameri_sites.tsv.")
    parser.add_argument(
        "--pool-input-list", type=Path, default=None, help="Default: <raw dir>/Unmatched_Sites.csv."
    )
    parser.add_argument("--sites", type=Path, default=None, help="Default: data/processed/sites/sites.csv.")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write. Default: <raw dir>/ameriflux_towers.csv.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def summarize_every_raw_file(raw_dir: Path, site_list: pd.DataFrame) -> pd.DataFrame:
    """The per-tower summaries of every raw file present."""
    summaries = []
    for resolution in RESOLUTIONS.values():
        path = raw_path(resolution, raw_dir)
        if not path.exists():
            print(f"note: no {path.name}; its towers are not in the table", flush=True)
            continue
        with read_raw(path) as raw:
            summaries.append(summarize_raw(raw, site_list))
            print(f"summarized {raw.sizes['tower']} {resolution.name} towers", flush=True)
    if not summaries:
        raise BuildError(f"no raw NEE file under {raw_dir}")
    return pd.concat(summaries, ignore_index=True)


def write_tower_table(table: pd.DataFrame, out: Path) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    try:
        table.to_csv(partial, index=False)
        check_round_trip(table, partial)
        partial.replace(out)
    finally:
        partial.unlink(missing_ok=True)


def describe_table(table: pd.DataFrame, out: Path) -> str:
    """The run report."""
    matched = table[table["site_id"].notna()]
    lines = [
        f"wrote {out}: {len(table)} towers, {matched['site_id'].nunique()} pool sites, "
        f"{int(table['primary'].sum())} primary",
        "matched by: " + ", ".join(f"{basis} {int((matched['match_basis'] == basis).sum())}" for basis in MATCH_BASES),
        f"sites with more than one tower: {int(matched['site_id'].value_counts().gt(1).sum())}",
        f"offsets: {table['utc_offset_hours'].value_counts().sort_index().to_dict()}",
        "excluded:",
    ]
    for tower, reason in table.loc[table["excluded_reason"] != "", ["tower", "excluded_reason"]].itertuples(index=False):
        lines.append(f"  {tower:8s} {reason}")
    lines.append("comments:")
    for tower, comment in table.loc[table["comment"] != "", ["tower", "comment"]].itertuples(index=False):
        lines.append(f"  {tower:8s} {comment}")
    return "\n".join(lines)


# ── checks ────────────────────────────────────────────────────────────────────


def check_round_trip(table: pd.DataFrame, partial: Path) -> None:
    """Raise unless the written table reads back, through the library, as built."""
    read_back = read_tower_table(partial)
    try:
        pd.testing.assert_frame_equal(read_back, table.reset_index(drop=True), check_exact=True)
    except AssertionError as error:
        raise BuildError(f"the tower table did not round-trip through {partial}: {error}") from error


if __name__ == "__main__":
    sys.exit(main())
