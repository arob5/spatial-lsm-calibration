#!/usr/bin/env python
"""Build the annual constraints product.

Overview
--------
Pivot the long table that ``scripts/export_constraints.R`` produces into the
dense netCDF the rest of the project reads, renaming the source variables to
the processed convention on the way. The R script does the reading and the one
check that only R can do; this script makes every schema decision and every
check that can be made from the flattened data.

``sipnet_calibration.constraints`` holds the schema and the reader. This script
is the writer, and its own round-trip check reads the file back with
:func:`sipnet_calibration.constraints.load_constraints` -- the same function
every consumer uses -- so the two cannot drift apart.

Input data
----------
``--long-table``
    CSV from ``export_constraints.R``: one row per observed
    ``(snapshot, site, variable)`` triple, columns ``snapshot_date, site_id,
    variable, mean, variance``. ``variable`` holds *source* names. Doubles are
    written with ``%.17g`` and must be read with
    ``float_precision="round_trip"``, which is why the reader lives in the
    library rather than here.

``--manifest``
    JSON from the same run, recording what R checked: per-snapshot
    per-variable row counts, the exact extremes per variable, the empty
    site-snapshots, and the largest absolute off-diagonal covariance element
    seen.

``--sites``
    ``processed/sites/sites.csv``, which supplies the site pool and the
    ``lon``/``lat`` coordinates.

Output data
-----------
``--out``, default ``data/processed/constraints_annual.nc``::

    observation_mean(site, time, variable)      float64, NaN where unobserved
    observation_variance(site, time, variable)  float64, NaN where unobserved

``site`` is the whole site pool, whether or not a site was ever observed;
``time`` the annual snapshot keys the source carries; ``variable`` the processed
variable names. ``lon`` and ``lat`` are non-dimension coordinates on ``site``.
Unobserved cells are ``NaN``. The run prints the observed cell count and the
file size, and the tests check the counts against the source.

Notes
-----
The manifest is the R script's testimony about the source, and much of the
checking here is the CSV against that manifest rather than either on trust. The
load-bearing one is diagonality: the Python side cannot see an off-diagonal
element, because by the time the CSV exists it is gone, so it checks that R
*made* the claim and refuses to write if it did not.

Output is written to a ``.partial`` path and renamed only once it reads back
bitwise through the library loader, so a failed check cannot leave a corrupt
file where the canonical one belongs.

Zero variances are written through unchanged. Some source variances are exactly
zero, mostly where the observation is zero too, which is unusable as a weight
and unusable as a prior; something downstream has to floor them. Doing it here
would hide a modeling decision inside an ingest script, so the count is
reported instead.

Usage
-----
::

    Rscript scripts/export_constraints.R --out long.csv --manifest manifest.json
    python scripts/ingest_constraints.py --long-table long.csv \\
        --manifest manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.constraints import (
    CONSTRAINT_VARIABLE_ATTRS,
    CONSTRAINT_VARIABLES,
    OBSERVATION_MEAN,
    OBSERVATION_VARIANCE,
    SOURCE_VARIABLE_NAMES,
    UNITS_PROVENANCE,
    UNITS_STATUS,
    default_constraints_path,
    load_constraints,
    read_long_table,
)
from sipnet_calibration.sites import default_sites_path, load_sites

#: Reference epoch for the stored time encoding. Written explicitly so nothing
#: is inherited from a default.
TIME_UNITS = "days since 2012-01-01"

#: Compression applied to both data arrays.
COMPRESSION = {"zlib": True, "complevel": 4, "_FillValue": np.nan}


class IngestError(Exception):
    """A check failed, or an input is not what it claims to be."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out if args.out is not None else default_constraints_path()
    sites_path = args.sites if args.sites is not None else default_sites_path()

    try:
        dataset = ingest(args.long_table, args.manifest, sites_path, out)
    except (IngestError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(describe_constraints(dataset))
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--long-table",
        type=Path,
        required=True,
        help="The long CSV written by scripts/export_constraints.R.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="The JSON manifest written alongside it.",
    )
    parser.add_argument(
        "--sites",
        type=Path,
        default=None,
        help="The site table. Defaults to data/processed/sites/sites.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write. Defaults to data/processed/constraints_annual.nc.",
    )
    return parser.parse_args(argv)


# ── the ingest steps, in the order main calls them ────────────────────────────


def ingest(
    long_table: Path, manifest_path: Path, sites_path: Path | None, out: Path
) -> xr.Dataset:
    """Read, check, build and write."""
    manifest = read_manifest(manifest_path)
    check_covariances_were_diagonal(manifest)

    table = read_long_table(long_table)
    sites = load_sites(sites_path)

    check_table_matches_manifest(table, manifest)
    check_snapshots_match_manifest(table, manifest)
    check_extremes_round_tripped(table, manifest)
    check_no_duplicate_triples(table)
    check_sites_are_in_the_site_table(table, sites)

    renamed = rename_to_processed_variables(table)
    dataset = build_dataset(build_grids(renamed, sites), manifest)
    write_dataset(dataset, out)
    return dataset


def read_manifest(path: Path) -> dict:
    """Read the manifest ``export_constraints.R`` wrote."""
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise IngestError(
            f"{path} not found. Produce it with scripts/export_constraints.R."
        ) from error
    except OSError as error:
        raise IngestError(f"{path} could not be read: {error}") from error
    except json.JSONDecodeError as error:
        raise IngestError(f"{path} is not valid JSON: {error}") from error

    if not isinstance(manifest, dict):
        raise IngestError(
            f"{path}: expected a JSON object, found {type(manifest).__name__}"
        )

    check_manifest_has_required_keys(manifest, path)
    return manifest


def rename_to_processed_variables(table: pd.DataFrame) -> pd.DataFrame:
    """Map the source variable names onto the processed ones."""
    # Safe here rather than in R because the long table names the variable on
    # every row, so a row carries its own identity and the rename cannot
    # mis-pair a variance with a variable. In the source the pairing is
    # positional, which is why R keeps the source names and the source order.
    renamed = table.copy()
    renamed["variable"] = renamed["variable"].map(SOURCE_VARIABLE_NAMES)
    check_every_variable_was_renamed(renamed, table)
    return renamed


def build_grids(table: pd.DataFrame, sites: pd.DataFrame) -> Grids:
    """Pivot the long table onto the dense ``(site, time, variable)`` grid.

    Every site of the pool gets a row whether or not it was observed, so the
    product's ``site`` axis is the pool rather than whichever sites happened to
    carry an observation.
    """
    site = sites["site_id"].to_numpy(np.int32)
    time = pd.DatetimeIndex(sorted(table["snapshot_date"].unique()))

    shape = (site.size, time.size, len(CONSTRAINT_VARIABLES))
    mean = np.full(shape, np.nan)
    variance = np.full(shape, np.nan)

    site_index = np.searchsorted(site, table["site_id"].to_numpy())
    time_index = time.get_indexer(pd.to_datetime(table["snapshot_date"]))
    variable_index = np.array(
        [CONSTRAINT_VARIABLES.index(name) for name in table["variable"]]
    )
    check_indices_resolved(time_index)

    mean[site_index, time_index, variable_index] = table["mean"].to_numpy()
    variance[site_index, time_index, variable_index] = table["variance"].to_numpy()

    return Grids(
        mean=mean,
        variance=variance,
        site=site,
        time=time,
        lon=sites["lon"].to_numpy(np.float64),
        lat=sites["lat"].to_numpy(np.float64),
    )


def build_dataset(grids: Grids, manifest: dict) -> xr.Dataset:
    """Assemble the Dataset, with the attributes that travel with it."""
    dims = ("site", "time", "variable")
    dataset = xr.Dataset(
        {
            OBSERVATION_MEAN: (dims, grids.mean),
            OBSERVATION_VARIANCE: (dims, grids.variance),
        },
        coords={
            "site": grids.site,
            "time": grids.time,
            "variable": np.asarray(CONSTRAINT_VARIABLES, dtype=object),
            "lon": ("site", grids.lon),
            "lat": ("site", grids.lat),
        },
    )
    annotate_dataset(dataset, manifest)
    return dataset


def write_dataset(dataset: xr.Dataset, out: Path) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename.

    A failed check leaves the partial file for inspection and nothing at the
    canonical path, so a later read cannot pick up a half-written product.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    dataset.to_netcdf(partial, engine="h5netcdf", encoding=netcdf_encoding())

    check_round_trip(dataset, partial)
    partial.replace(out)


def describe_constraints(dataset: xr.Dataset) -> str:
    """A short report of what was written, for the run log."""
    lines = [
        f"sites {dataset.sizes['site']}  snapshots {dataset.sizes['time']}  "
        f"variables {dataset.sizes['variable']}",
        f"snapshot keys {str(dataset['time'].values[0])[:10]} .. "
        f"{str(dataset['time'].values[-1])[:10]}",
    ]
    mean, variance = dataset[OBSERVATION_MEAN], dataset[OBSERVATION_VARIANCE]
    total = int(np.prod(mean.shape))
    observed = int(np.isfinite(mean.values).sum())
    lines.append(
        f"observed {observed} of {total} cells ({observed / total:.1%}); "
        f"the rest are NaN"
    )
    for index, name in enumerate(CONSTRAINT_VARIABLES):
        column = mean.values[:, :, index]
        variances = variance.values[:, :, index]
        finite = np.isfinite(column)
        n_zero = int((variances[np.isfinite(variances)] <= 0).sum())
        extent = (
            f"[{column[finite].min():.5g}, {column[finite].max():.5g}]"
            if finite.any()
            else "[none observed]"
        )
        lines.append(
            f"  {name:<24s} n={finite.sum():>6d}  range {extent}  "
            f"non-positive variances {n_zero}"
        )
    return "\n".join(lines)


# ── supporting types and helpers ──────────────────────────────────────────────


@dataclass(frozen=True)
class Grids:
    """The dense arrays and the axes they are indexed on."""

    mean: np.ndarray
    variance: np.ndarray
    site: np.ndarray
    time: pd.DatetimeIndex
    lon: np.ndarray
    lat: np.ndarray


def _same_extreme(formatted: str, expected: str) -> bool:
    """Whether two ``%.17g`` extremes denote the same value.

    A string comparison, because that is what detects a truncated digit. The
    one exception is the sign of zero: R's ``min``/``max`` return the first of
    tied values and numpy's return the signed one, so a column holding both
    zeros can disagree on the sign of its extreme while every value round-trips
    exactly. That is not a precision loss and must not be reported as one.
    """
    zeros = {"0", "-0"}
    if formatted in zeros and expected in zeros:
        return True
    return formatted == expected


def netcdf_encoding() -> dict:
    """Explicit on-disk encoding, so nothing is inherited from a default."""
    return {
        OBSERVATION_MEAN: dict(COMPRESSION),
        OBSERVATION_VARIANCE: dict(COMPRESSION),
        "time": {
            "units": TIME_UNITS,
            "calendar": "proleptic_gregorian",
            "dtype": "int32",
        },
    }


def annotate_dataset(dataset: xr.Dataset, manifest: dict) -> None:
    """Attach the attributes that have to travel with the product."""
    dataset[OBSERVATION_MEAN].attrs = {
        "long_name": "Observed annual constraint",
        "units": "see the per-variable units in the dataset attributes",
        "units_status": UNITS_STATUS,
        "units_provenance": UNITS_PROVENANCE,
    }
    dataset[OBSERVATION_VARIANCE].attrs = {
        "long_name": "Observation error variance",
        "units": "the square of the corresponding observation's unit",
        "units_status": UNITS_STATUS,
        "units_provenance": UNITS_PROVENANCE,
        "comment": (
            "The source covariances are exactly diagonal, so these variances "
            "are lossless. Some are exactly zero; flooring them is a modeling "
            "decision and is not done here."
        ),
    }
    for name, attrs in CONSTRAINT_VARIABLE_ATTRS.items():
        for key, value in attrs.items():
            dataset.attrs[f"variable_{name}_{key}"] = value

    dataset["site"].attrs = {
        "long_name": "Model site identifier",
        "comment": "The handed-down 1-8000 identifier; never renumbered.",
    }
    dataset["time"].attrs = {
        "long_name": "Annual snapshot key",
        "time_zone": "none (nominal annual key)",
        "time_label": "nominal",
        "time_label_note": (
            "The July 15 dates are the source product's annual bookkeeping "
            "convention, not observation dates. 'nominal' extends the "
            "start/middle/end/instant vocabulary the other products use, "
            "because this label is a key rather than a time."
        ),
    }
    dataset["variable"].attrs = {
        "long_name": "Constrained variable",
        "comment": (
            "Processed names; each variable's source name is in the "
            "variable_<name>_source_name dataset attribute."
        ),
    }
    dataset.attrs.update(
        {
            "title": "Annual biomass, leaf area and soil constraints",
            "source_mean_file": manifest.get("mean_file", ""),
            "source_cov_file": manifest.get("cov_file", ""),
            "source_resolution": "annual",
            "history": (
                "scripts/export_constraints.R -> scripts/ingest_constraints.py"
            ),
            "exported_at": manifest.get("generated_at", ""),
            "covariances_all_diagonal": "true",
            "n_observed_triples": int(manifest["n_rows"]),
        }
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_manifest_has_required_keys(manifest: dict, path: Path) -> None:
    """The manifest carries everything the later checks read."""
    required = {
        "variables",
        "snapshot_dates",
        "n_rows",
        "counts_by_snapshot_variable",
        "extremes",
        "covariances_all_diagonal",
        "max_abs_offdiagonal",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise IngestError(f"{path}: manifest is missing {missing}")


def check_covariances_were_diagonal(manifest: dict) -> None:
    """Refuse to write unless the R side confirmed every covariance diagonal.

    Storing variances rather than matrices is lossless exactly when the source
    covariances are diagonal, and that is no longer checkable from the CSV. So
    this checks the claim was made and is affirmative.
    """
    if not manifest["covariances_all_diagonal"]:
        raise IngestError(
            "the manifest reports the source covariances are not all diagonal "
            f"(max |off-diagonal| = {manifest['max_abs_offdiagonal']!r}). The "
            "processed form stores variances only, which is lossless just when "
            "they are. The schema needs revisiting before this can be written; "
            "see src/sipnet_calibration/constraints.py."
        )
    if manifest["max_abs_offdiagonal"] != 0:
        raise IngestError(
            "the manifest claims all covariances are diagonal but reports a "
            f"non-zero max |off-diagonal| of {manifest['max_abs_offdiagonal']!r}"
        )


def check_table_matches_manifest(table: pd.DataFrame, manifest: dict) -> None:
    """The long table is the one the manifest describes."""
    expected = tuple(SOURCE_VARIABLE_NAMES)
    if tuple(manifest["variables"]) != expected:
        raise IngestError(
            f"manifest variables {tuple(manifest['variables'])} do not match "
            f"the expected source names {expected}"
        )
    if len(table) != manifest["n_rows"]:
        raise IngestError(
            f"the long table has {len(table)} rows, the manifest says "
            f"{manifest['n_rows']}. They are out of step; regenerate both."
        )

    observed = table.groupby(["snapshot_date", "variable"], sort=True).size().to_dict()
    for date, counts in manifest["counts_by_snapshot_variable"].items():
        for variable, expected_count in counts.items():
            actual = observed.get((date, variable), 0)
            if actual != expected_count:
                raise IngestError(
                    f"{date} {variable}: the table has {actual} rows, the "
                    f"manifest says {expected_count}"
                )


def check_snapshots_match_manifest(table: pd.DataFrame, manifest: dict) -> None:
    """Every snapshot the source held carries rows in the table.

    The time axis is built from the snapshots that carry observations, so a
    snapshot observed nowhere would drop out of the product silently and shift
    every later snapshot's index.
    """
    expected = set(manifest["snapshot_dates"])
    observed = set(table["snapshot_date"])
    missing = sorted(expected - observed)
    if missing:
        raise IngestError(
            f"{len(missing)} snapshot(s) in the manifest carry no rows in the "
            f"long table: {missing}. They would drop out of the time axis "
            "rather than appearing as all-NaN."
        )
    unexpected = sorted(observed - expected)
    if unexpected:
        raise IngestError(
            f"snapshot(s) in the long table the manifest does not list: "
            f"{unexpected}"
        )


def check_extremes_round_tripped(table: pd.DataFrame, manifest: dict) -> None:
    """The parsed extremes re-format to exactly the strings R wrote.

    This is the truncation detector. The R side wrote every value with
    ``%.17g``, which uniquely determines a float64; if the parse or the write
    lost bits, an extreme will not re-format to the same characters.
    """
    for variable, expected in manifest["extremes"].items():
        rows = table[table["variable"] == variable]
        if rows.empty:
            raise IngestError(
                f"{variable}: the manifest has extremes, the table has no rows"
            )
        for column, keys in (
            ("mean", ("mean_min", "mean_max")),
            ("variance", ("variance_min", "variance_max")),
        ):
            values = rows[column].to_numpy()
            for key, value in zip(keys, (values.min(), values.max())):
                formatted = f"{value:.17g}"
                if not _same_extreme(formatted, expected[key]):
                    raise IngestError(
                        f"{variable} {key}: R wrote {expected[key]!r}, this "
                        f"parsed to {formatted!r}. A value lost precision "
                        "between the two."
                    )


def check_no_duplicate_triples(table: pd.DataFrame) -> None:
    """No (snapshot, site, variable) triple appears twice."""
    duplicated = table.duplicated(subset=["snapshot_date", "site_id", "variable"])
    if duplicated.any():
        examples = table.loc[duplicated, ["snapshot_date", "site_id", "variable"]]
        raise IngestError(
            f"{int(duplicated.sum())} duplicated (snapshot, site, variable) "
            f"triples, for example:\n{examples.head().to_string(index=False)}"
        )


def check_sites_are_in_the_site_table(
    table: pd.DataFrame, sites: pd.DataFrame
) -> None:
    """The source's site pool is a subset of the site table's.

    Two independent statements of the 8000-site pool -- the shapefile and the
    constraint files -- and no reason for them to disagree, so a disagreement is
    worth stopping for rather than dropping rows over.
    """
    unknown = sorted(set(table["site_id"]) - set(sites["site_id"]))
    if unknown:
        raise IngestError(
            f"{len(unknown)} site ids in the constraints are absent from the "
            f"site table, for example {unknown[:10]}. Rebuild the site table "
            "with scripts/ingest_sites.py, or the pools genuinely differ."
        )


def check_every_variable_was_renamed(
    renamed: pd.DataFrame, original: pd.DataFrame
) -> None:
    """No source name fell through the rename as a null."""
    unmapped = renamed["variable"].isna()
    if unmapped.any():
        names = sorted(set(original.loc[unmapped, "variable"]))
        raise IngestError(
            f"source variable names with no processed name: {names}. Add them "
            "to SOURCE_VARIABLE_NAMES in sipnet_calibration.constraints."
        )


def check_indices_resolved(time_index: np.ndarray) -> None:
    """Every snapshot date found a slot on the time axis."""
    if np.any(time_index < 0):
        raise IngestError("a snapshot date failed to index; this is a bug")


def check_round_trip(dataset: xr.Dataset, path: Path) -> None:
    """The written file reads back through the real loader, bitwise."""
    reloaded = load_constraints(path)
    try:
        for name in (OBSERVATION_MEAN, OBSERVATION_VARIANCE):
            written, read = dataset[name].values, reloaded[name].values
            if written.shape != read.shape:
                raise IngestError(
                    f"{name}: wrote shape {written.shape}, read {read.shape}"
                )
            same = (written == read) | (np.isnan(written) & np.isnan(read))
            if not same.all():
                differ = ~same
                numeric = differ & np.isfinite(written) & np.isfinite(read)
                detail = (
                    "worst numeric difference "
                    f"{np.abs(written[numeric] - read[numeric]).max()!r}"
                    if numeric.any()
                    else "every disagreement is NaN against a value"
                )
                raise IngestError(
                    f"{name}: {int(differ.sum())} cells changed on the round "
                    f"trip; {detail}"
                )
        for coordinate in ("site", "time", "variable", "lon", "lat"):
            if not np.array_equal(
                dataset[coordinate].values, reloaded[coordinate].values
            ):
                raise IngestError(
                    f"the {coordinate!r} coordinate changed on the round trip"
                )
    finally:
        reloaded.close()


if __name__ == "__main__":
    raise SystemExit(main())
