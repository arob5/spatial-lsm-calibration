#!/usr/bin/env python
"""Build the processed constraint products, one netCDF per constraint.

Overview
--------
For each constraint in ``sipnet_calibration.constraints.CONSTRAINTS``, read its
raw file, check it, place its records on the site pool and write the result as
``data/processed/constraints/<name>.nc``. Every decision about what a file
holds -- columns, units, time structure, which rows to drop -- is a field of
the constraint's spec in the library; this script is the orchestration and the
checks, and its own round-trip check reads each file back with
:func:`sipnet_calibration.constraints.load_constraint`, the same function every
consumer uses.

Input data
----------
``--raw-root``, default ``data/raw/constraints/``
    One gzipped CSV per constraint, named by ``spec.raw_file``, read exactly
    by :func:`sipnet_calibration.constraints.read_raw`. See
    ``data/raw/constraints/provenance.md`` for where they came from.

``--sites``, default ``data/processed/sites/sites.csv``
    The site table: the pool the products are dense over, and the ``lon``/``lat``
    coordinates.

Output data
-----------
``--out-dir``, default ``data/processed/constraints/``, one file per constraint::

    value(site[, time])               float64, NaN where unobserved
    standard_deviation(site[, time])  float64, NaN in the same cells

with ``site`` the whole pool, ``time`` the constraint's own labels (absent for
a static constraint), ``time_bounds`` for an annual one, and every attribute
the spec provides. ``sipnet_calibration.constraints`` documents the data model.

Notes
-----
The ingest changes structure, never values: no unit conversion, no temporal
alignment, no choice of which record stands for a year. The two structural
steps that do drop or merge rows are declared by the spec and counted in the
run report: rows failing a quality flag are dropped, and a static constraint's
identical yearly copies are collapsed to one.

Output is written to a ``.partial`` path and renamed only once it reads back
identically through the library loader, so a failed check cannot leave a
corrupt file where the canonical one belongs.
A failed check keeps the ``.partial`` file for inspection and prints its
path (:func:`sipnet_calibration.io.write_checked`).

Usage
-----
::

    python scripts/ingest_constraints.py                         # every constraint
    python scripts/ingest_constraints.py --constraint modis_leaf_area_index
    python scripts/ingest_constraints.py --describe              # the specs, no I/O
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.constraints import (
    CONSTRAINT_NAMES,
    STANDARD_DEVIATION,
    TIME_UNITS,
    VALUE,
    ConstraintSpec,
    TimeStructure,
    build_constraint,
    constraint_path,
    default_constraints_dir,
    default_raw_dir,
    describe,
    load_constraint,
    netcdf_encoding,
    read_raw,
    resolve_constraint,
)
from sipnet_calibration.conventions import SITE_ID
from sipnet_calibration.io import write_checked
from sipnet_calibration.sites import default_sites_path, load_sites

#: How far a raw file's lat/lon may sit from the site table before the site
#: ids are taken to mean a different pool. The real files agree to 5e-13.
COORDINATE_TOLERANCE_DEGREES = 1e-9


class IngestError(Exception):
    """A check failed, or an input is not what it claims to be."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.constraint or list(CONSTRAINT_NAMES)

    if args.describe:
        print("\n\n".join(describe(resolve_constraint(name)) for name in names))
        return 0

    raw_root = args.raw_root if args.raw_root is not None else default_raw_dir()
    out_dir = args.out_dir if args.out_dir is not None else default_constraints_dir()
    sites_path = args.sites if args.sites is not None else default_sites_path()

    try:
        sites = load_sites(sites_path)
        for name in names:
            dataset = ingest(resolve_constraint(name), raw_root, sites, out_dir)
            print(describe_product(dataset, constraint_path(name, out_dir)))
    except (IngestError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--constraint",
        action="append",
        choices=CONSTRAINT_NAMES,
        metavar="NAME",
        help="A constraint to build; repeatable. Default: all of "
        + ", ".join(CONSTRAINT_NAMES),
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print each constraint's spec and exit without reading data.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Directory of the raw files. Default: data/raw/constraints.",
    )
    parser.add_argument(
        "--sites",
        type=Path,
        default=None,
        help="The site table. Default: data/processed/sites/sites.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write. Default: data/processed/constraints.",
    )
    return parser.parse_args(argv)


# ── the ingest steps, in the order main calls them ────────────────────────────


def ingest(spec: ConstraintSpec, raw_root: Path, sites: pd.DataFrame, out_dir: Path) -> xr.Dataset:
    """Read, check, build and write one constraint."""
    frame = read_raw(spec, raw_root)
    check_raw_frame(spec, frame, sites)

    dataset = build_constraint(spec, frame, sites)
    write_product(dataset, constraint_path(spec, out_dir), spec)
    return dataset


def check_raw_frame(spec: ConstraintSpec, frame: pd.DataFrame, sites: pd.DataFrame) -> None:
    """Every check on the raw rows, before anything is built from them."""
    check_site_ids_are_valid(spec, frame)
    check_sites_are_in_the_site_table(spec, frame, sites)
    check_coordinates_match_site_table(spec, frame, sites)
    check_key_is_unique(spec, frame)
    check_value_and_sd_missing_together(spec, frame)
    check_values_are_finite(spec, frame)
    check_sd_is_not_negative(spec, frame)
    check_quality_flag_values(spec, frame)
    check_some_rows_are_observed(spec, frame)
    check_time_column_parses(spec, frame)
    if spec.time_structure is TimeStructure.STATIC:
        check_static_copies_agree(spec, frame)


def write_product(dataset: xr.Dataset, out: Path, spec: ConstraintSpec) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename."""
    write_checked(
        out,
        write=lambda partial: dataset.to_netcdf(
            partial, engine="h5netcdf", encoding=netcdf_encoding(dataset)
        ),
        check=lambda partial: check_round_trip(dataset, partial, spec),
    )


def describe_product(dataset: xr.Dataset, path: Path) -> str:
    """A short report of what was written, for the run log."""
    value, sd = dataset[VALUE].values, dataset[STANDARD_DEVIATION].values
    observed = np.isfinite(value)
    sizes = " x ".join(f"{dataset.sizes[dim]} {dim}" for dim in dataset[VALUE].dims)
    lines = [
        f"{dataset.attrs['constraint']}  ->  {path} ({path.stat().st_size / 1e6:.1f} MB)",
        f"  {sizes}; observed {int(observed.sum())} of {observed.size} cells "
        f"({observed.mean():.1%})",
        f"  rows read {dataset.attrs['rows_read']}, dropped by quality flag "
        f"{dataset.attrs['rows_dropped_by_quality_flag']}, collapsed as copies "
        f"{dataset.attrs['rows_collapsed_as_copies']}",
    ]
    if observed.any():
        lines.append(
            f"  value range [{value[observed].min():.5g}, {value[observed].max():.5g}] "
            f"{dataset[VALUE].attrs['units']}; standard deviations of zero: "
            f"{int((sd[observed] == 0).sum())}"
        )
    if "time" in dataset.dims:
        first, last = dataset["time"].values[[0, -1]]
        lines.append(f"  time {str(first)[:10]} .. {str(last)[:10]}")
    return "\n".join(lines)


# ── checks ────────────────────────────────────────────────────────────────────


def check_site_ids_are_valid(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Raise unless every site id is a positive integer that fits the stored width."""
    site = frame[SITE_ID].to_numpy()
    if not np.issubdtype(site.dtype, np.integer):
        raise IngestError(f"{spec.raw_file}: {SITE_ID} is not integer-valued")
    info = np.iinfo(np.int32)
    bad = (site < 1) | (site > info.max)
    if bad.any():
        raise IngestError(
            f"{spec.raw_file}: site ids outside 1..{info.max}: "
            f"{sorted(set(site[bad].tolist()))[:10]}"
        )


def check_sites_are_in_the_site_table(
    spec: ConstraintSpec, frame: pd.DataFrame, sites: pd.DataFrame
) -> None:
    """Raise if a row names a site the site table does not have."""
    unknown = sorted(set(frame[SITE_ID]) - set(sites[SITE_ID]))
    if unknown:
        raise IngestError(
            f"{spec.raw_file}: {len(unknown)} site ids are not in the site table, e.g. "
            f"{unknown[:10]}. The file may belong to a different site pool."
        )


def check_coordinates_match_site_table(
    spec: ConstraintSpec, frame: pd.DataFrame, sites: pd.DataFrame
) -> None:
    """Raise if a file's own lat/lon disagree with the site table for its site ids.

    Only files carrying ``lat`` and ``lon`` are checked. The columns are
    redundant with the site table and are kept in the raw files exactly for
    this: a second, 6400-site pool exists upstream whose site 1 is elsewhere.
    """
    if not {"lat", "lon"} <= set(spec.raw_columns):
        return
    table = sites.set_index(SITE_ID).loc[frame[SITE_ID].to_numpy(), ["lon", "lat"]]
    for column in ("lon", "lat"):
        given = frame[column].to_numpy(np.float64)
        if not np.isfinite(given).all():
            raise IngestError(
                f"{spec.raw_file}: {column} is missing or not finite in "
                f"{int((~np.isfinite(given)).sum())} rows"
            )
        difference = np.abs(given - table[column].to_numpy())
        if difference.max() > COORDINATE_TOLERANCE_DEGREES:
            worst = int(np.argmax(difference))
            raise IngestError(
                f"{spec.raw_file}: {column} disagrees with the site table by up to "
                f"{difference[worst]:.3g} degrees (site {frame[SITE_ID].iloc[worst]}). "
                "The site ids do not mean what the site table means."
            )


def check_key_is_unique(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Raise if two rows share a site (and time)."""
    key = [SITE_ID] + ([spec.time_column] if spec.time_column else [])
    duplicated = frame.duplicated(key)
    if duplicated.any():
        example = frame.loc[duplicated, key].iloc[0].tolist()
        raise IngestError(
            f"{spec.raw_file}: {int(duplicated.sum())} rows repeat a {tuple(key)} key, "
            f"e.g. {example}"
        )


def check_value_and_sd_missing_together(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Raise if a row has a value without a standard deviation or the reverse."""
    value_missing = frame[spec.value_column].isna().to_numpy()
    sd_missing = frame[spec.sd_column].isna().to_numpy()
    mismatched = value_missing != sd_missing
    if mismatched.any():
        raise IngestError(
            f"{spec.raw_file}: {int(mismatched.sum())} rows have {spec.value_column!r} "
            f"and {spec.sd_column!r} missing in different places"
        )


def check_values_are_finite(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Raise on an infinite value or standard deviation; only ``NA`` may be missing."""
    for column in (spec.value_column, spec.sd_column):
        values = frame[column].to_numpy(np.float64)
        infinite = np.isinf(values)
        if infinite.any():
            raise IngestError(
                f"{spec.raw_file}: {column!r} is infinite in {int(infinite.sum())} rows"
            )


def check_some_rows_are_observed(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Raise if no row that passes the quality flag carries a value.

    A file of nothing but ``NA`` would otherwise build an all-missing product
    and replace the canonical file with it.
    """
    kept = frame
    if spec.quality_column is not None:
        kept = frame[frame[spec.quality_column] == spec.quality_pass]
    if not kept[spec.value_column].notna().any():
        raise IngestError(
            f"{spec.raw_file}: no row carries an observed value"
            + (" after the quality filter" if spec.quality_column else "")
        )


def check_sd_is_not_negative(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Raise on a negative standard deviation."""
    sd = frame[spec.sd_column].to_numpy(np.float64)
    negative = sd < 0
    if negative.any():
        raise IngestError(
            f"{spec.raw_file}: {int(negative.sum())} negative standard deviations, "
            f"smallest {sd[negative].min():.6g}"
        )


def check_quality_flag_values(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Raise if the quality column never takes the passing value.

    An unexpected extra flag value is not an error -- it is a row that fails --
    but a file where nothing passes means the spec's ``quality_pass`` is wrong.
    """
    if spec.quality_column is None:
        return
    values = frame[spec.quality_column]
    if not (values == spec.quality_pass).any():
        raise IngestError(
            f"{spec.raw_file}: no row has {spec.quality_column!r} == {spec.quality_pass!r}; "
            f"values seen: {sorted(map(str, values.unique().tolist()))[:10]}"
        )


def check_time_column_parses(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Raise if a time value is missing or is not a year or an ISO date."""
    if spec.time_column is None:
        return
    column = frame[spec.time_column]
    if column.isna().any():
        raise IngestError(f"{spec.raw_file}: {spec.time_column!r} has missing values")
    if spec.time_structure is TimeStructure.DATED:
        try:
            parsed = pd.to_datetime(column, format="%Y-%m-%d")
        except (ValueError, TypeError) as error:
            raise IngestError(
                f"{spec.raw_file}: {spec.time_column!r} is not an ISO date column: {error}"
            ) from error
        # An empty string parses to NaT without raising.
        if parsed.isna().any():
            raise IngestError(
                f"{spec.raw_file}: {spec.time_column!r} has {int(parsed.isna().sum())} "
                "values that are not dates"
            )
    else:
        years = column.to_numpy()
        if not np.issubdtype(years.dtype, np.integer) or (years < 1900).any() or (years > 2100).any():
            raise IngestError(f"{spec.raw_file}: {spec.time_column!r} does not hold years")


def check_static_copies_agree(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Raise unless a static constraint carries one value per site across the file."""
    if spec.time_column is None:
        return
    distinct = frame.groupby(SITE_ID)[[spec.value_column, spec.sd_column]].nunique(
        dropna=False
    )
    varying = distinct[(distinct > 1).any(axis=1)]
    if not varying.empty:
        raise IngestError(
            f"{spec.raw_file}: {len(varying)} sites carry different values in different "
            f"years (first: {varying.index[:5].tolist()}), but {spec.name} is declared "
            "static. Either the source changed or the spec's time_structure is wrong."
        )


def check_round_trip(dataset: xr.Dataset, partial: Path, spec: ConstraintSpec) -> None:
    """Raise unless the written file reads back identical through the library loader."""
    with load_constraint(spec, partial) as written:
        written = written.load()
    if not written.identical(dataset):
        raise IngestError(
            f"{partial}: the written file does not read back identical to what was built."
        )
    if "time" in written.coords and written["time"].encoding.get("units") != TIME_UNITS:
        # xarray silently changes the units when a label is not a whole day.
        raise IngestError(
            f"{partial}: time was encoded as {written['time'].encoding.get('units')!r}, not "
            f"{TIME_UNITS!r}; a label is not a whole day."
        )


if __name__ == "__main__":
    raise SystemExit(main())
