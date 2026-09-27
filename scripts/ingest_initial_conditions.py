#!/usr/bin/env python
"""Build the processed initial condition file from the tracked raw file.

Overview
--------
Read the tracked raw initial condition file, check it, rename the source
variables to the spec names and the ``member`` dim to
``initial_condition_member``, renumber the members from 0, place the sites on
the site pool and write the processed file. Every decision about what a
variable is -- its unit, its provenance, the SIPNET parameter it feeds -- is
set in its ``InitialConditionSpec`` in the library; this script is the
orchestration and the checks.

Input data
----------
``--raw-file``, default :func:`sipnet_calibration.initial_conditions.raw_path`
    The converted raw file, five variables on ``(site, member)`` in source
    names, written by ``scripts/raw_sources/convert_initial_conditions.py``
    and read exactly by :func:`sipnet_calibration.initial_conditions.read_raw`.

``--site-table``, default :func:`sipnet_calibration.sites.default_sites_path`
    The site table: the pool the processed file is on, and the ``lon``/``lat``
    coordinates.

Output data
-----------
``--output``, default
:func:`sipnet_calibration.initial_conditions.default_processed_path`
    One ``float64`` variable per spec on ``(initial_condition_member, site)``,
    ``NaN`` where no source file for the site carried it, in the data model
    :mod:`sipnet_calibration.initial_conditions` documents.

Notes
-----
The ingest changes structure, never values. Negative wood and leaf carbon --
a substantial share of the members with leaf carbon, in PEcAn's construction
``wood = biomass - leaf`` -- are written through and counted in the report,
because dropping or flooring them is the experiment's decision and PEcAn's own
handling (it kept the template default for such members) is recorded in the
specs.

The identity ``wood_carbon_content == AbvGrndWood - leaf_carbon_content``
(and ``== AbvGrndWood`` where leaf is absent) is checked bit for bit: it is
how PEcAn built the wood pool, and a break means the source changed.

The file is written through :func:`sipnet_calibration.io.write_checked`, and
its check reads it back with
:func:`sipnet_calibration.initial_conditions.load_initial_conditions`, the
function every reader of the processed file uses.

Usage
-----
::

    uv run python scripts/ingest_initial_conditions.py
    uv run python scripts/ingest_initial_conditions.py --describe     # the specs, no I/O
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
    default_processed_path,
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

#: The raw file's names for the three pools PEcAn's wood identity relates.
BIOMASS_SOURCE_NAME = "AbvGrndWood"
WOOD_SOURCE_NAME = "wood_carbon_content"
LEAF_SOURCE_NAME = "leaf_carbon_content"


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Build the processed initial condition file, or describe the specs."""
    args = parse_args(argv)
    if args.describe:
        print("\n\n".join(describe(spec) for spec in INITIAL_CONDITIONS))
        return 0

    raw_file = args.raw_file or raw_path()
    output = args.output or default_processed_path()
    try:
        site_table = load_sites(args.site_table or default_sites_path())
        with read_raw(raw_file) as raw:
            check_raw_file_is_valid(raw, site_table)
            dataset = build_initial_conditions(raw, site_table)
        write_processed_file(dataset, output)
        print(describe_processed_file(dataset, output))
    except (IngestError, OSError, ValueError, LookupError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, as the module docstring's Usage describes it."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--describe", action="store_true", help="Print each spec and exit without reading data."
    )
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=None,
        help="The converted raw file. Default: data/raw/initial_conditions/"
        "pecan_pool_initial_conditions.nc.",
    )
    parser.add_argument(
        "--site-table",
        type=Path,
        default=None,
        help="The site table. Default: data/processed/sites/sites.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write. Default: data/processed/initial_conditions.nc.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def write_processed_file(dataset: xr.Dataset, path: Path) -> None:
    """Write through a ``.partial`` file, moved in once it reads back bit for bit."""
    write_checked(
        path,
        write=lambda partial: dataset.to_netcdf(
            partial, engine="h5netcdf", encoding=netcdf_encoding(dataset)
        ),
        check=lambda partial: check_round_trip(dataset, partial),
    )


def describe_processed_file(dataset: xr.Dataset, path: Path) -> str:
    """A short report of what was written, for the run log."""
    lines = [
        f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)",
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


# ── supporting types and helpers ──────────────────────────────────────────────


class IngestError(RuntimeError):
    """The raw or the written file breaks an invariant the processed file depends on."""


# ── checks ────────────────────────────────────────────────────────────────────


def check_raw_file_is_valid(raw: xr.Dataset, site_table: pd.DataFrame) -> None:
    """The raw file is fit to build the processed file from, beyond what ``read_raw`` checks."""
    check_every_source_variable_has_a_spec()
    check_sites_are_the_site_table(
        site_table, raw[SITE].values.tolist(), message_name="the raw file's sites"
    )
    check_members_are_contiguous_from_one(raw)
    check_biomass_and_wood_are_everywhere(raw)
    check_wood_is_biomass_minus_leaf(raw)


def check_every_source_variable_has_a_spec() -> None:
    """The specs cover exactly the variables the raw file can hold."""
    specified = {spec.source_name for spec in INITIAL_CONDITIONS}
    if specified != set(SOURCE.names):
        raise IngestError(
            f"the specs cover {sorted(specified)} but the source variables are "
            f"{sorted(SOURCE.names)}, and a variable without a spec would be dropped "
            "silently; give every source variable a spec in INITIAL_CONDITIONS."
        )


def check_members_are_contiguous_from_one(raw: xr.Dataset) -> None:
    """The source files' member index runs ``1..n`` with no gap."""
    members = raw[RAW_MEMBER].values.astype(np.int64)
    expected = np.arange(1, members.size + 1)
    if not np.array_equal(members, expected):
        raise IngestError(
            f"source member indices are {members[:5].tolist()}... to {members[-1]}, "
            f"expected 1..{members.size}, and renumbering to 0-based would hide the gap; "
            "rebuild the raw file from the complete source tree."
        )


def check_biomass_and_wood_are_everywhere(raw: xr.Dataset) -> None:
    """Aboveground biomass and wood carbon are present at every site and member."""
    both = np.isfinite(raw[BIOMASS_SOURCE_NAME].values) & np.isfinite(raw[WOOD_SOURCE_NAME].values)
    if not both.all():
        raise IngestError(
            f"{BIOMASS_SOURCE_NAME} and {WOOD_SOURCE_NAME} are not present at every site "
            "and member; every PEcAn source file carries both, so the source changed."
        )


def check_wood_is_biomass_minus_leaf(raw: xr.Dataset) -> None:
    """PEcAn's wood identity holds bit for bit at every site and member."""
    biomass = raw[BIOMASS_SOURCE_NAME].values
    leaf = raw[LEAF_SOURCE_NAME].values
    expected = np.where(np.isfinite(leaf), biomass - leaf, biomass)
    mismatch = raw[WOOD_SOURCE_NAME].values != expected
    if mismatch.any():
        raise IngestError(
            f"{WOOD_SOURCE_NAME} differs from {BIOMASS_SOURCE_NAME} - {LEAF_SOURCE_NAME} "
            f"(or {BIOMASS_SOURCE_NAME} where leaf is absent) at {int(mismatch.sum())} "
            f"(site, member) pairs, first at site "
            f"{raw[SITE].values[np.argwhere(mismatch)[0][0]]}; that identity is how PEcAn "
            "built the wood pool, so the source changed."
        )


def check_round_trip(dataset: xr.Dataset, partial: Path) -> None:
    """The written file reads back through the library as what was built, bit for bit."""
    with load_initial_conditions(partial) as read_back:
        check_variables_read_back_bitwise(dataset, read_back, partial=partial)
        check_variable_attributes_read_back(dataset, read_back, partial=partial)
        check_coordinates_read_back(dataset, read_back, partial=partial)


def check_variables_read_back_bitwise(
    dataset: xr.Dataset, read_back: xr.Dataset, *, partial: Path
) -> None:
    """Every variable reads back bit for bit."""
    for spec in INITIAL_CONDITIONS:
        if not np.array_equal(
            dataset[spec.name].values, read_back[spec.name].values, equal_nan=True
        ):
            raise IngestError(
                f"{spec.name} did not round-trip bit for bit through {partial}; inspect "
                "the kept partial file."
            )


def check_variable_attributes_read_back(
    dataset: xr.Dataset, read_back: xr.Dataset, *, partial: Path
) -> None:
    """Every variable's attributes read back unchanged."""
    for spec in INITIAL_CONDITIONS:
        if dict(read_back[spec.name].attrs) != dict(dataset[spec.name].attrs):
            raise IngestError(
                f"{spec.name}'s attributes changed on the way to disk through {partial}; "
                "inspect the kept partial file."
            )


def check_coordinates_read_back(
    dataset: xr.Dataset, read_back: xr.Dataset, *, partial: Path
) -> None:
    """Every coordinate reads back unchanged."""
    for coordinate in (INITIAL_CONDITION_MEMBER, SOURCE_INDEX, SITE, LON, LAT):
        if not np.array_equal(dataset[coordinate].values, read_back[coordinate].values):
            raise IngestError(
                f"{coordinate} did not round-trip through {partial}; inspect the kept "
                "partial file."
            )


if __name__ == "__main__":
    raise SystemExit(main())
