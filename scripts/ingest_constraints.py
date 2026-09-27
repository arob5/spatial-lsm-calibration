#!/usr/bin/env python
"""Build the constraints' processed files, one netCDF per constraint.

Overview
--------
For each constraint in ``sipnet_calibration.constraints.CONSTRAINTS``, read its
raw file, check it, place its records on the site pool and write the result as
``data/processed/constraints/<name>.nc``. Every decision about what a file
holds -- columns, units, time structure, which rows to drop -- is set in the
constraint's spec in the library; this script is the orchestration and the
checks.

Input data
----------
``--raw-directory``, default :func:`sipnet_calibration.constraints.default_raw_dir`
    One gzipped CSV per constraint, named by ``spec.raw_file``, read exactly
    by :func:`sipnet_calibration.constraints.read_raw`. See
    ``data/raw/constraints/provenance.md`` for where they came from.

``--site-table``, default :func:`sipnet_calibration.sites.default_sites_path`
    The site table: the pool the processed files are dense over, and the
    ``lon``/``lat`` coordinates.

Output data
-----------
``--output-directory``, default
:func:`sipnet_calibration.constraints.default_constraints_dir`
    One processed file per constraint, in the data model
    :mod:`sipnet_calibration.constraints` documents: ``value`` and
    ``standard_deviation`` over the whole pool and the constraint's own time
    labels, ``NaN`` where unobserved.

Notes
-----
The ingest changes structure, never values: no unit conversion, no temporal
alignment, no choice of which record stands for a year. The two structural
steps that do drop or merge rows are declared by the spec and counted in the
run report: rows failing a quality flag are dropped, and a static constraint's
identical yearly copies are collapsed to one.

Each file is written through :func:`sipnet_calibration.io.write_checked`, and
its check reads it back with
:func:`sipnet_calibration.constraints.load_constraint`, the function every
consumer uses.

Usage
-----
::

    uv run python scripts/ingest_constraints.py               # every constraint
    uv run python scripts/ingest_constraints.py --constraint modis_leaf_area_index
    uv run python scripts/ingest_constraints.py --describe    # the specs, no I/O
"""

from __future__ import annotations

import argparse
import sys
import zlib
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
from sipnet_calibration.conventions import LAT, LON, SITE_ID, TIME
from sipnet_calibration.io import write_checked
from sipnet_calibration.sites import (
    check_site_table_lists_the_sites,
    default_sites_path,
    load_sites,
    site_lookup,
)
from sipnet_calibration.validation import check_site_ids_are_in_range, truncated

#: How far, in degrees, a raw file's lat/lon may sit from the site table before
#: the site ids are taken to mean a different pool.
COORDINATE_TOLERANCE = 1e-9

#: The first and last year an annual or static constraint's time column may hold.
PLAUSIBLE_YEARS = (1900, 2100)


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Build every constraint asked for, or describe their specs."""
    args = parse_args(argv)
    names = args.constraint or list(CONSTRAINT_NAMES)

    if args.describe:
        print("\n\n".join(describe(resolve_constraint(name)) for name in names))
        return 0

    raw_directory = args.raw_directory or default_raw_dir()
    output_directory = args.output_directory or default_constraints_dir()
    try:
        site_table = load_sites(args.site_table or default_sites_path())
        for name in names:
            dataset = ingest(resolve_constraint(name), raw_directory, site_table, output_directory)
            print(describe_processed_file(dataset, constraint_path(name, output_directory)))
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
        "--raw-directory",
        type=Path,
        default=None,
        help=f"Directory of the raw files. Default: {default_raw_dir()}.",
    )
    parser.add_argument(
        "--site-table",
        type=Path,
        default=None,
        help=f"The site table. Default: {default_sites_path()}.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=f"Where to write. Default: {default_constraints_dir()}.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def ingest(
    spec: ConstraintSpec, raw_directory: Path, site_table: pd.DataFrame, output_directory: Path
) -> xr.Dataset:
    """Read, check, build and write one constraint; the dataset written."""
    frame = read_raw_frame(spec, raw_directory)
    check_raw_frame_is_valid(spec, frame, site_table)
    dataset = build_constraint(spec, frame, site_table)
    write_processed_file(dataset, constraint_path(spec, output_directory), spec)
    return dataset


def describe_processed_file(dataset: xr.Dataset, path: Path) -> str:
    """A short report of what was written, for the run log."""
    value, standard_deviation = dataset[VALUE].values, dataset[STANDARD_DEVIATION].values
    observed = np.isfinite(value)
    sizes = " x ".join(f"{dataset.sizes[dim]} {dim}" for dim in dataset[VALUE].dims)
    lines = [
        f"{dataset.attrs['constraint']}  ->  {path} ({path.stat().st_size / 1e6:.1f} MB)",
        f"  {sizes}; observed {int(observed.sum())} of {observed.size} elements "
        f"({observed.mean():.1%})",
        f"  rows read {dataset.attrs['rows_read']}, dropped by quality flag "
        f"{dataset.attrs['rows_dropped_by_quality_flag']}, collapsed as copies "
        f"{dataset.attrs['rows_collapsed_as_copies']}",
    ]
    if observed.any():
        lines.append(
            f"  value range [{value[observed].min():.5g}, {value[observed].max():.5g}] "
            f"{dataset[VALUE].attrs['units']}; standard deviations of zero: "
            f"{int((standard_deviation[observed] == 0).sum())}"
        )
    if TIME in dataset.dims:
        first, last = dataset[TIME].values[[0, -1]]
        lines.append(f"  time {str(first)[:10]} .. {str(last)[:10]}")
    return "\n".join(lines)


# ── supporting types and helpers ──────────────────────────────────────────────


class IngestError(RuntimeError):
    """A raw or written file breaks an invariant the processed file depends on."""


def read_raw_frame(spec: ConstraintSpec, raw_directory: Path) -> pd.DataFrame:
    """The constraint's raw rows, as ``constraints.read_raw`` reads them.

    Raises
    ------
    IngestError
        If the gzip stream is truncated or corrupt, which ``read_raw`` lets
        through as ``EOFError`` or ``zlib.error``.
    """
    try:
        return read_raw(spec, raw_directory)
    except (EOFError, zlib.error) as error:
        raise IngestError(
            f"{Path(raw_directory) / spec.raw_file} is a truncated or corrupt gzip file "
            f"({error}); re-copy it and compare it with data/raw/constraints/provenance.md."
        ) from error


def write_processed_file(dataset: xr.Dataset, path: Path, spec: ConstraintSpec) -> None:
    """Write through a ``.partial`` file, moved in once it reads back identical."""
    write_checked(
        path,
        write=lambda partial: dataset.to_netcdf(
            partial, engine="h5netcdf", encoding=netcdf_encoding(dataset)
        ),
        check=lambda partial: check_written_file_reads_back_identically(dataset, partial, spec),
    )


def _key_column_names(spec: ConstraintSpec) -> list[str]:
    """The columns a raw row is keyed on: the site id, and any time column."""
    return [SITE_ID, *([spec.time_column] if spec.time_column else [])]


# ── checks ────────────────────────────────────────────────────────────────────


def check_raw_frame_is_valid(
    spec: ConstraintSpec, frame: pd.DataFrame, site_table: pd.DataFrame
) -> None:
    """The raw rows are fit to build the constraint's processed file from."""
    check_site_ids_are_the_site_tables(frame, site_table, message_name=spec.raw_file)
    check_coordinates_are_the_site_tables(spec, frame, site_table)
    check_key_is_unique(frame, key=_key_column_names(spec), message_name=spec.raw_file)
    check_observed_values_are_valid(spec, frame)
    check_time_column_is_valid(spec, frame)


def check_site_ids_are_the_site_tables(
    frame: pd.DataFrame, site_table: pd.DataFrame, *, message_name: str
) -> None:
    """The raw file's site ids are integers, in range, and sites of the site table."""
    check_site_id_column_is_integer_valued(frame, message_name=message_name)
    check_site_ids_are_in_range(frame[SITE_ID].to_numpy(), message_name=f"{message_name}: {SITE_ID}")
    check_site_table_lists_the_sites(
        site_table, frame[SITE_ID].unique().tolist(), message_name=f"{message_name}: site(s)"
    )


def check_coordinates_are_the_site_tables(
    spec: ConstraintSpec, frame: pd.DataFrame, site_table: pd.DataFrame
) -> None:
    """Where the raw file carries lat/lon, they are finite and the site table's."""
    # The raw files that carry coordinates name them as the site table does.
    if not {LAT, LON} <= set(spec.raw_columns):
        return
    check_coordinates_are_finite(frame, message_name=spec.raw_file)
    check_coordinates_match_site_table(frame, site_table, message_name=spec.raw_file)


def check_observed_values_are_valid(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """The values, standard deviations and quality flags are fit to build from."""
    check_value_and_standard_deviation_missing_together(
        frame, value_column=spec.value_column, standard_deviation_column=spec.sd_column,
        message_name=spec.raw_file,
    )
    check_values_are_finite(
        frame, columns=(spec.value_column, spec.sd_column), message_name=spec.raw_file
    )
    check_standard_deviation_is_not_negative(
        frame, column=spec.sd_column, message_name=spec.raw_file
    )
    if spec.quality_column is not None:
        check_quality_flag_passes_some_row(
            frame, column=spec.quality_column, passing_value=spec.quality_pass,
            message_name=spec.raw_file,
        )
    check_some_rows_are_observed(
        frame, value_column=spec.value_column, quality_column=spec.quality_column,
        passing_value=spec.quality_pass, message_name=spec.raw_file,
    )


def check_time_column_is_valid(spec: ConstraintSpec, frame: pd.DataFrame) -> None:
    """Where the constraint has a time column, it holds what its time structure says."""
    if spec.time_column is None:
        return
    check_time_column_is_complete(frame, column=spec.time_column, message_name=spec.raw_file)
    if spec.time_structure is TimeStructure.DATED:
        check_time_column_holds_dates(frame, column=spec.time_column, message_name=spec.raw_file)
    else:
        check_time_column_holds_years(frame, column=spec.time_column, message_name=spec.raw_file)
    if spec.time_structure is TimeStructure.STATIC:
        check_static_copies_agree(
            frame, value_columns=(spec.value_column, spec.sd_column),
            message_name=spec.raw_file,
        )


def check_site_id_column_is_integer_valued(frame: pd.DataFrame, *, message_name: str) -> None:
    """The raw file's ``site_id`` column is of an integer dtype."""
    if not np.issubdtype(frame[SITE_ID].to_numpy().dtype, np.integer):
        raise IngestError(
            f"{message_name}: {SITE_ID} is not integer-valued; re-copy it and compare it with data/raw/constraints/provenance.md."
        )


def check_coordinates_are_finite(frame: pd.DataFrame, *, message_name: str) -> None:
    """Every row of a raw file carrying lat/lon has finite coordinates."""
    for column in (LON, LAT):
        given = frame[column].to_numpy(np.float64)
        if not np.isfinite(given).all():
            raise IngestError(
                f"{message_name}: {column} is missing or not finite in "
                f"{int((~np.isfinite(given)).sum())} rows; re-copy it and compare it with data/raw/constraints/provenance.md."
            )


def check_coordinates_match_site_table(
    frame: pd.DataFrame, site_table: pd.DataFrame, *, message_name: str
) -> None:
    """A raw file's own lat/lon agree with the site table for its site ids."""
    # The columns are redundant with the site table and are kept in the raw
    # files exactly for this: a second, 6400-site pool exists upstream whose
    # site 1 is elsewhere.
    table = site_lookup(site_table).loc[frame[SITE_ID].to_numpy(), [LON, LAT]]
    for column in (LON, LAT):
        given = frame[column].to_numpy(np.float64)
        difference = np.abs(given - table[column].to_numpy())
        if difference.max() > COORDINATE_TOLERANCE:
            worst = int(np.argmax(difference))
            raise IngestError(
                f"{message_name}: {column} disagrees with the site table by up to "
                f"{difference[worst]:.3g} degrees (site {frame[SITE_ID].iloc[worst]}), so "
                f"the site ids do not mean what the site table means; re-copy it and compare it with data/raw/constraints/provenance.md."
            )


def check_key_is_unique(frame: pd.DataFrame, *, key: list[str], message_name: str) -> None:
    """No two rows share a key."""
    duplicated = frame.duplicated(key)
    if duplicated.any():
        example = frame.loc[duplicated, key].iloc[0].tolist()
        raise IngestError(
            f"{message_name}: {int(duplicated.sum())} rows repeat a {tuple(key)} key, "
            f"such as {example}; re-copy it and compare it with data/raw/constraints/provenance.md."
        )


def check_value_and_standard_deviation_missing_together(
    frame: pd.DataFrame, *, value_column: str, standard_deviation_column: str, message_name: str
) -> None:
    """No row has a value without a standard deviation, or the reverse."""
    value_missing = frame[value_column].isna().to_numpy()
    standard_deviation_missing = frame[standard_deviation_column].isna().to_numpy()
    mismatched = value_missing != standard_deviation_missing
    if mismatched.any():
        raise IngestError(
            f"{message_name}: {int(mismatched.sum())} rows have {value_column!r} and "
            f"{standard_deviation_column!r} missing in different places; re-copy it and compare it with data/raw/constraints/provenance.md."
        )


def check_values_are_finite(
    frame: pd.DataFrame, *, columns: tuple[str, ...], message_name: str
) -> None:
    """No value is infinite; only ``NA`` may be missing."""
    for column in columns:
        infinite = np.isinf(frame[column].to_numpy(np.float64))
        if infinite.any():
            raise IngestError(
                f"{message_name}: {column!r} is infinite in {int(infinite.sum())} rows, "
                f"and only NA may stand for a missing value; re-copy it and compare it with data/raw/constraints/provenance.md."
            )


def check_standard_deviation_is_not_negative(
    frame: pd.DataFrame, *, column: str, message_name: str
) -> None:
    """No standard deviation is negative."""
    standard_deviations = frame[column].to_numpy(np.float64)
    negative = standard_deviations < 0
    if negative.any():
        raise IngestError(
            f"{message_name}: {int(negative.sum())} negative standard deviations, "
            f"smallest {standard_deviations[negative].min():.6g}; re-copy it and compare it with data/raw/constraints/provenance.md."
        )


def check_quality_flag_passes_some_row(
    frame: pd.DataFrame, *, column: str, passing_value: str, message_name: str
) -> None:
    """Some row carries the quality flag's passing value."""
    # An unexpected extra flag value is not an error -- it is a row that fails --
    # but a file where nothing passes means the spec's quality_pass is wrong.
    values = frame[column]
    if not (values == passing_value).any():
        raise IngestError(
            f"{message_name}: no row has {column!r} == {passing_value!r}, the values seen "
            f"being {truncated(sorted(map(str, values.unique().tolist())))}; correct the "
            "spec's quality_pass."
        )


def check_some_rows_are_observed(
    frame: pd.DataFrame,
    *,
    value_column: str,
    quality_column: str | None,
    passing_value: str | None,
    message_name: str,
) -> None:
    """Some row that passes the quality flag carries a value."""
    # A file of nothing but NA would otherwise build an all-missing processed
    # file and replace the canonical one with it.
    kept = frame if quality_column is None else frame[frame[quality_column] == passing_value]
    if not kept[value_column].notna().any():
        raise IngestError(
            f"{message_name}: no row carries an observed value"
            + (" after the quality filter" if quality_column else "")
            + "; re-copy it and compare it with data/raw/constraints/provenance.md."
        )


def check_time_column_is_complete(frame: pd.DataFrame, *, column: str, message_name: str) -> None:
    """No time value is missing."""
    if frame[column].isna().any():
        raise IngestError(
            f"{message_name}: {column!r} has missing values; re-copy it and compare it with data/raw/constraints/provenance.md."
        )


def check_time_column_holds_dates(frame: pd.DataFrame, *, column: str, message_name: str) -> None:
    """Every time value is an ISO date."""
    try:
        parsed = pd.to_datetime(frame[column], format="%Y-%m-%d")
    except (ValueError, TypeError) as error:
        raise IngestError(
            f"{message_name}: {column!r} is not an ISO date column ({error}); re-copy it and compare it with data/raw/constraints/provenance.md."
        ) from error
    # An empty string parses to NaT without raising.
    if parsed.isna().any():
        raise IngestError(
            f"{message_name}: {column!r} has {int(parsed.isna().sum())} values that are "
            f"not dates; re-copy it and compare it with data/raw/constraints/provenance.md."
        )


def check_time_column_holds_years(frame: pd.DataFrame, *, column: str, message_name: str) -> None:
    """Every time value is an integer year in :data:`PLAUSIBLE_YEARS`."""
    years = frame[column].to_numpy()
    first, last = PLAUSIBLE_YEARS
    if not np.issubdtype(years.dtype, np.integer) or (years < first).any() or (years > last).any():
        raise IngestError(
            f"{message_name}: {column!r} does not hold years from {first} to {last}; "
            "check the spec's time_column."
        )


def check_static_copies_agree(
    frame: pd.DataFrame, *, value_columns: tuple[str, ...], message_name: str
) -> None:
    """A static constraint carries one value per site across its yearly copies."""
    distinct = frame.groupby(SITE_ID)[list(value_columns)].nunique(dropna=False)
    varying = distinct[(distinct > 1).any(axis=1)]
    if not varying.empty:
        raise IngestError(
            f"{message_name}: {len(varying)} sites carry different values in different "
            f"years, {truncated(varying.index.tolist())}, but the constraint is declared "
            "static; correct the spec's time_structure if the source is right, or else "
            "re-copy it and compare it with data/raw/constraints/provenance.md."
        )


def check_written_file_reads_back_identically(
    dataset: xr.Dataset, partial: Path, spec: ConstraintSpec
) -> None:
    """The written file reads back through the library loader as what was built."""
    with load_constraint(spec, partial) as written:
        written = written.load()
    check_read_back_is_identical(dataset, written, message_name=str(partial))
    check_time_is_encoded_in_days(written, message_name=str(partial))


def check_read_back_is_identical(
    dataset: xr.Dataset, read_back: xr.Dataset, *, message_name: str
) -> None:
    """The file read back is identical to the dataset it was written from."""
    if not read_back.identical(dataset):
        raise IngestError(
            f"{message_name}: the written file does not read back identical to what was "
            "built; inspect the kept partial file."
        )


def check_time_is_encoded_in_days(written: xr.Dataset, *, message_name: str) -> None:
    """The written file's ``time`` is encoded in :data:`TIME_UNITS`."""
    # xarray silently changes the units when a label is not a whole day.
    if TIME in written.coords and written[TIME].encoding.get("units") != TIME_UNITS:
        raise IngestError(
            f"{message_name}: time was encoded as {written[TIME].encoding.get('units')!r}, "
            f"not {TIME_UNITS!r}, so a label is not a whole day; label each record by its day."
        )


if __name__ == "__main__":
    raise SystemExit(main())
