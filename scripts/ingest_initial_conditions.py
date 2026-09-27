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
    uv run python scripts/ingest_initial_conditions.py --describe   # the specs, no I/O
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.conventions import INITIAL_CONDITION_MEMBER, SITE
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
from sipnet_calibration.validation import range_summary, truncated

#: The raw file's name for aboveground biomass, the whole of PEcAn's wood identity.
BIOMASS_SOURCE_NAME = "AbvGrndWood"

#: The raw file's name for wood carbon, biomass less leaf in PEcAn's wood identity.
WOOD_SOURCE_NAME = "wood_carbon_content"

#: The raw file's name for leaf carbon, the part of biomass that is not wood.
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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--describe", action="store_true", help="Print each spec and exit without reading data."
    )
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=None,
        help=f"The converted raw file. Default: {raw_path()}.",
    )
    parser.add_argument(
        "--site-table",
        type=Path,
        default=None,
        help=f"The site table. Default: {default_sites_path()}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Where to write. Default: {default_processed_path()}.",
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
        check=lambda partial: check_written_file_reads_back_identically(dataset, partial),
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
    """The raw file is fit to build from, beyond what ``read_raw`` checks."""
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
            f"the specs cover {truncated(sorted(specified))} but the source variables are "
            f"{truncated(sorted(SOURCE.names))}, and a variable without a spec would be dropped "
            "silently; give every source variable a spec in INITIAL_CONDITIONS."
        )


def check_members_are_contiguous_from_one(raw: xr.Dataset) -> None:
    """The source files' member index runs ``1..n`` with no gap."""
    members = raw[RAW_MEMBER].values.astype(np.int64)
    expected = np.arange(1, members.size + 1)
    if not np.array_equal(members, expected):
        raise IngestError(
            f"source member indices are {truncated(members.tolist())}, expected "
            f"1..{members.size}, and renumbering to 0-based would hide the gap; re-copy it and compare it with data/raw/initial_conditions/provenance.md."
        )


def check_biomass_and_wood_are_everywhere(raw: xr.Dataset) -> None:
    """Aboveground biomass and wood carbon are present at every site and member."""
    both = np.isfinite(raw[BIOMASS_SOURCE_NAME].values) & np.isfinite(raw[WOOD_SOURCE_NAME].values)
    if not both.all():
        raise IngestError(
            f"{BIOMASS_SOURCE_NAME} and {WOOD_SOURCE_NAME} are not present at every site "
            f"and member, which every PEcAn source file carries; re-copy it and compare it with data/raw/initial_conditions/provenance.md."
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
            f"{raw[SITE].values[np.argwhere(mismatch)[0][0]]}, and that identity is how "
            f"PEcAn built the wood pool; re-copy it and compare it with data/raw/initial_conditions/provenance.md."
        )


def check_written_file_reads_back_identically(dataset: xr.Dataset, partial: Path) -> None:
    """The written file reads back through the library as what was built."""
    with load_initial_conditions(partial) as read_back:
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
