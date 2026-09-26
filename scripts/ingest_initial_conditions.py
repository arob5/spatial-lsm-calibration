#!/usr/bin/env python
"""Build the processed initial condition product from the tracked raw file.

Overview
--------
Read ``data/raw/initial_conditions/pecan_pool_initial_conditions.nc``, check
it, rename the source variables to the spec names and the ``member`` dim to
``initial_condition_member``, renumber the members,
place the sites on the site pool and write
``data/processed/initial_conditions.nc``. Every decision about what a variable
is -- its unit, its provenance, the SIPNET parameter it feeds -- is a field of
its ``InitialConditionSpec`` in the library; this script is the orchestration
and the checks, and its round-trip check reads the file back with
``sipnet_calibration.initial_conditions.load_initial_conditions``, the same
function every reader of the product uses.

Input data
----------
``--raw``, default ``data/raw/initial_conditions/pecan_pool_initial_conditions.nc``
    The converted raw file, five variables on ``(site, member)`` in source
    names, written by ``scripts/raw_sources/convert_initial_conditions.py``
    and read exactly by
    ``read_raw``.

``--sites``, default ``data/processed/sites/sites.csv``
    The site table: the pool the product is on, and the ``lon``/``lat``
    coordinates.

Output data
-----------
``--out``, default ``data/processed/initial_conditions.nc``::

    initial_aboveground_biomass_carbon(initial_condition_member, site)  float64, kg C m-2
    initial_wood_carbon(initial_condition_member, site)                 float64, kg C m-2
    initial_leaf_carbon(initial_condition_member, site)                 float64, kg C m-2
    initial_soil_organic_carbon(initial_condition_member, site)         float64, kg C m-2
    initial_soil_moisture_saturation(initial_condition_member, site)    float64, percent

``NaN`` where no source file for the site carried the variable;
``initial_condition_member`` 0-based with ``source_index`` carrying the source
files' 1-based index; ``site`` the whole pool. ``sipnet_calibration.initial_conditions`` documents the data
model.

Notes
-----
The ingest changes structure, never values. Negative wood and leaf carbon --
a substantial share of the members with leaf carbon, in PEcAn's construction
``wood = biomass - leaf`` -- are written through and counted in the report,
because dropping or flooring them is the experiment's decision and PEcAn's own
handling (it kept the template default for such members) is recorded in the
specs.

The identity ``wood_carbon_content == AbvGrndWood - leaf_carbon_content``
(and ``== AbvGrndWood`` where leaf is absent) is asserted bit for bit: it is
how PEcAn built the wood pool, and a break means the source changed.

Output is written to a ``.partial`` path and renamed only once it reads back
bit-identical through the library loader.
A failed check keeps the ``.partial`` file for inspection and prints its
path (:func:`sipnet_calibration.io.write_checked`).

Usage
-----
::

    python scripts/ingest_initial_conditions.py
    python scripts/ingest_initial_conditions.py --describe     # the specs, no I/O
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.conventions import (
    INITIAL_CONDITION_MEMBER,
    LAT,
    LON,
    SITE,
    SOURCE_INDEX,
)
from sipnet_calibration.initial_conditions import (
    INITIAL_CONDITIONS,
    RAW_MEMBER,
    SOURCE,
    build_initial_conditions,
    default_product_path,
    describe,
    load_initial_conditions,
    netcdf_encoding,
    raw_path,
    read_raw,
)
from sipnet_calibration.io import write_checked
from sipnet_calibration.sites import (
    check_sites_are_the_site_table,
    default_sites_path,
    load_sites,
)
from sipnet_calibration.validation import range_summary


class IngestError(Exception):
    """A check failed, or an input is not what it claims to be."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.describe:
        print("\n\n".join(describe(spec) for spec in INITIAL_CONDITIONS))
        return 0

    raw = args.raw if args.raw is not None else raw_path()
    out = args.out if args.out is not None else default_product_path()
    sites_path = args.sites if args.sites is not None else default_sites_path()

    try:
        sites = load_sites(sites_path)
        with read_raw(raw) as raw_dataset:
            check_raw(raw_dataset, sites)
            dataset = build_initial_conditions(raw_dataset, sites)
        write_product(dataset, out)
        print(describe_product(dataset, out))
    except (IngestError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--describe", action="store_true", help="Print each spec and exit without reading data."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="The converted raw file. Default: data/raw/initial_conditions/"
        "pecan_pool_initial_conditions.nc.",
    )
    parser.add_argument(
        "--sites",
        type=Path,
        default=None,
        help="The site table. Default: data/processed/sites/sites.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write. Default: data/processed/initial_conditions.nc.",
    )
    return parser.parse_args(argv)


# ── the ingest steps, in the order main calls them ────────────────────────────


def check_raw(raw: xr.Dataset, sites: pd.DataFrame) -> None:
    """Every check on the raw file beyond the schema ``read_raw`` enforces."""
    check_every_source_variable_has_a_spec()
    check_sites_are_the_site_table(
        sites, raw[SITE].values.tolist(), message_name="the raw file's sites"
    )
    check_members_are_contiguous_from_one(raw)
    check_wood_is_biomass_minus_leaf(raw)


def write_product(dataset: xr.Dataset, out: Path) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename."""
    write_checked(
        out,
        write=lambda partial: dataset.to_netcdf(
            partial, engine="h5netcdf", encoding=netcdf_encoding(dataset)
        ),
        check=lambda partial: check_round_trip(dataset, partial),
    )


def describe_product(dataset: xr.Dataset, out: Path) -> str:
    """A short report of what was written, for the run log."""
    lines = [
        f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)",
        f"members {dataset.sizes[INITIAL_CONDITION_MEMBER]}  sites {dataset.sizes[SITE]}  nominal date "
        f"{dataset.attrs['nominal_date']}",
        "variable                            units     sites   min          median       max          negative",
    ]
    for spec in INITIAL_CONDITIONS:
        values = dataset[spec.name].values
        present = np.isfinite(values)
        finite = values[present]
        lines.append(
            f"{spec.name:35s} {spec.units:9s} {int(present.any(axis=0).sum()):5d}   "
            + range_summary(finite)
        )
    return "\n".join(lines)


# ── checks ────────────────────────────────────────────────────────────────────


def check_every_source_variable_has_a_spec() -> None:
    """Raise unless the specs cover exactly the variables the raw file can hold."""
    specified = {spec.source_name for spec in INITIAL_CONDITIONS}
    if specified != set(SOURCE.names):
        raise IngestError(
            f"specs cover {sorted(specified)} but the source variables are "
            f"{sorted(SOURCE.names)}; a variable without a spec would be dropped silently"
        )


def check_members_are_contiguous_from_one(raw: xr.Dataset) -> None:
    """Raise unless the source files' member index runs 1..n with no gap."""
    members = raw[RAW_MEMBER].values.astype(np.int64)
    expected = np.arange(1, members.size + 1)
    if not np.array_equal(members, expected):
        raise IngestError(
            f"source member indices are {members[:5].tolist()}... to {members[-1]}, "
            f"expected 1..{members.size}; renumbering to 0-based would hide the gap"
        )


def check_wood_is_biomass_minus_leaf(raw: xr.Dataset) -> None:
    """Raise unless PEcAn's wood identity holds bit for bit everywhere."""
    biomass = raw["AbvGrndWood"].values
    wood = raw["wood_carbon_content"].values
    leaf = raw["leaf_carbon_content"].values
    has_leaf = np.isfinite(leaf)
    both = np.isfinite(biomass) & np.isfinite(wood)
    if not both.all():
        raise IngestError("AbvGrndWood and wood_carbon_content are not present at every cell")
    expected = np.where(has_leaf, biomass - leaf, biomass)
    mismatch = wood != expected
    if mismatch.any():
        raise IngestError(
            f"wood_carbon_content differs from AbvGrndWood - leaf_carbon_content (or "
            f"AbvGrndWood where leaf is absent) at {int(mismatch.sum())} cells, first at "
            f"site {raw[SITE].values[np.argwhere(mismatch)[0][0]]}. That identity is how the "
            "PEcAn built the wood pool; a break means the source changed."
        )


def check_round_trip(dataset: xr.Dataset, partial: Path) -> None:
    """Raise unless the written file reads back bit-identical through the library."""
    with load_initial_conditions(partial) as read_back:
        for spec in INITIAL_CONDITIONS:
            if not np.array_equal(dataset[spec.name].values, read_back[spec.name].values, equal_nan=True):
                raise IngestError(f"{spec.name} did not round-trip bit for bit through {partial}")
            if dict(read_back[spec.name].attrs) != dict(dataset[spec.name].attrs):
                raise IngestError(f"{spec.name}'s attributes changed on the way to disk")
        for coordinate in (INITIAL_CONDITION_MEMBER, SOURCE_INDEX, SITE, LON, LAT):
            if not np.array_equal(dataset[coordinate].values, read_back[coordinate].values):
                raise IngestError(f"{coordinate} did not round-trip through {partial}")


if __name__ == "__main__":
    sys.exit(main())
