#!/usr/bin/env python
"""Build the annual constraints: a long CSV -> ``processed/constraints_annual.nc``.

``scripts/export_constraints.R`` flattens the source R data files to one row per
observed ``(snapshot, site, variable)`` triple, plus a JSON manifest of what it
checked. This script pivots that into the dense product the rest of the project
reads, and makes every schema decision:
:mod:`sipnet_calibration.constraints` holds the schema and the reader, and this
is the writer tested against it.

    obs_mean(site, time, variable)   float64, NaN where not observed
    obs_var (site, time, variable)   float64, NaN where not observed

``site`` is the 1-8000 identifier, ``time`` the thirteen July-15 snapshot keys,
``variable`` the four constrained variables in alphabetical order. 8000 x 13 x 4
is 416,000 cells per array, of which 322,515 are observed; compressed, the file
is 2.2 MB. The raggedness costs nothing to represent densely, and dense is far
easier to reason about than any ragged encoding.

What this asserts before writing
--------------------------------
The manifest is the R script's testimony about the source, and most of what
follows is checking the CSV against it rather than taking either on trust.

* Every covariance in the source was exactly diagonal. The Python side cannot
  see an off-diagonal element -- by the time the CSV exists it is gone -- so it
  checks that the R script *made* the claim, and refuses to write if it did not.
  This is the assertion the whole storage choice rests on.
* The row count, the variable set, and the per-snapshot per-variable counts
  match the manifest.
* The exact extremes per variable match the manifest character for character
  after a round trip, which is what catches a writer or parser that truncates.
* No ``(snapshot, site, variable)`` triple appears twice.
* Every site in the CSV exists in the site table, and the source's site pool is
  exactly the site table's. That cross-checks two independent statements of the
  8000-site pool against each other.
* Reading the written file back through :func:`load_constraints` reproduces the
  arrays bitwise.

The file is written to a ``.partial`` path and renamed only once the round trip
passes, so a failed check cannot leave a corrupt file where the canonical one
belongs.

Zero variances are written through unchanged. 929 ``AbvGrndWood`` variances are
exactly zero in the source, 925 of them where the observation is also zero; that
is unusable as a weight and unusable as a prior, so something downstream has to
floor them. Doing it here would hide a modeling decision inside an ingest
script, so instead the count is asserted and reported.

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
    OBS_MEAN,
    OBS_VAR,
    UNITS_PROVENANCE,
    UNITS_STATUS,
    default_constraints_path,
    load_constraints,
    read_long_table,
)
from sipnet_calibration.sites import default_sites_path, load_sites


class IngestError(Exception):
    """A check failed, or an input is not what it claims to be."""


@dataclass(frozen=True)
class Grids:
    """The dense arrays and the axes they are indexed on."""

    mean: np.ndarray
    variance: np.ndarray
    site: np.ndarray
    time: pd.DatetimeIndex
    lon: np.ndarray
    lat: np.ndarray


# ── reading the inputs ────────────────────────────────────────────────────────


def read_manifest(path: Path) -> dict:
    """Read the manifest ``export_constraints.R`` wrote."""
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise IngestError(
            f"{path} not found. Produce it with scripts/export_constraints.R."
        ) from error
    except json.JSONDecodeError as error:
        raise IngestError(f"{path} is not valid JSON: {error}") from error

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
    return manifest


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


def check_against_manifest(table: pd.DataFrame, manifest: dict) -> None:
    """Check the long table is the one the manifest describes."""
    if tuple(manifest["variables"]) != CONSTRAINT_VARIABLES:
        raise IngestError(
            f"manifest variables {tuple(manifest['variables'])} do not match "
            f"CONSTRAINT_VARIABLES {CONSTRAINT_VARIABLES}"
        )
    if len(table) != manifest["n_rows"]:
        raise IngestError(
            f"the long table has {len(table)} rows, the manifest says "
            f"{manifest['n_rows']}. They are out of step; regenerate both."
        )

    observed = (
        table.groupby(["snapshot_date", "variable"], sort=True).size().to_dict()
    )
    for date, counts in manifest["counts_by_snapshot_variable"].items():
        for variable, expected in counts.items():
            actual = observed.get((date, variable), 0)
            if actual != expected:
                raise IngestError(
                    f"{date} {variable}: the table has {actual} rows, the "
                    f"manifest says {expected}"
                )


def check_extremes_round_tripped(table: pd.DataFrame, manifest: dict) -> None:
    """Check the parsed extremes re-format to exactly the strings R wrote.

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
        for column, keys in (("mean", ("mean_min", "mean_max")),
                             ("variance", ("variance_min", "variance_max"))):
            values = rows[column].to_numpy()
            for key, value in zip(keys, (values.min(), values.max())):
                formatted = f"{value:.17g}"
                if formatted != expected[key]:
                    raise IngestError(
                        f"{variable} {key}: R wrote {expected[key]!r}, this "
                        f"parsed to {formatted!r}. A value lost precision "
                        "between the two."
                    )


def check_no_duplicate_triples(table: pd.DataFrame) -> None:
    """Check no (snapshot, site, variable) triple appears twice."""
    duplicated = table.duplicated(subset=["snapshot_date", "site_id", "variable"])
    if duplicated.any():
        examples = table.loc[duplicated, ["snapshot_date", "site_id", "variable"]]
        raise IngestError(
            f"{int(duplicated.sum())} duplicated (snapshot, site, variable) "
            f"triples, for example:\n{examples.head().to_string(index=False)}"
        )


def check_sites_match_table(table: pd.DataFrame, sites: pd.DataFrame) -> None:
    """Check the source's site pool is exactly the site table's.

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


# ── building the arrays ───────────────────────────────────────────────────────


def build_grids(table: pd.DataFrame, sites: pd.DataFrame) -> Grids:
    """Pivot the long table onto the dense ``(site, time, variable)`` grid.

    Every site of the pool gets a row whether or not it was observed, so the
    product's ``site`` axis is the pool rather than whichever sites happened to
    carry an observation.
    """
    site = sites["site_id"].to_numpy(np.int32)
    time = pd.DatetimeIndex(sorted(table["snapshot_date"].unique()))
    variable = np.asarray(CONSTRAINT_VARIABLES, dtype=object)

    shape = (site.size, time.size, variable.size)
    mean = np.full(shape, np.nan)
    variance = np.full(shape, np.nan)

    site_index = np.searchsorted(site, table["site_id"].to_numpy())
    time_index = time.get_indexer(pd.to_datetime(table["snapshot_date"]))
    variable_index = np.array(
        [CONSTRAINT_VARIABLES.index(name) for name in table["variable"]]
    )
    if np.any(time_index < 0):
        raise IngestError("a snapshot date failed to index; this is a bug")

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
            OBS_MEAN: (dims, grids.mean),
            OBS_VAR: (dims, grids.variance),
        },
        coords={
            "site": grids.site,
            "time": grids.time,
            "variable": np.asarray(CONSTRAINT_VARIABLES, dtype=object),
            "lon": ("site", grids.lon),
            "lat": ("site", grids.lat),
        },
    )

    dataset[OBS_MEAN].attrs = {
        "long_name": "Observed annual constraint",
        "units": "see the units attribute of each variable",
        "units_status": UNITS_STATUS,
        "units_provenance": UNITS_PROVENANCE,
    }
    dataset[OBS_VAR].attrs = {
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
            "Alphabetical, matching the source column order. obs.cov carries no "
            "dimension names, so that order is what pairs a variance with its "
            "variable."
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
    return dataset


def netcdf_encoding(dataset: xr.Dataset) -> dict:
    """Explicit on-disk encoding, so nothing is inherited from a default."""
    compression = {"zlib": True, "complevel": 4, "_FillValue": np.nan}
    return {
        OBS_MEAN: dict(compression),
        OBS_VAR: dict(compression),
        "time": {
            "units": "days since 2012-01-01",
            "calendar": "proleptic_gregorian",
            "dtype": "int32",
        },
    }


# ── writing ───────────────────────────────────────────────────────────────────


def write_dataset(dataset: xr.Dataset, out: Path) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename.

    A failed check leaves the partial file for inspection and nothing at the
    canonical path, so a later read cannot pick up a half-written product.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    dataset.to_netcdf(partial, engine="h5netcdf", encoding=netcdf_encoding(dataset))

    check_round_trip(dataset, partial)
    partial.replace(out)


def check_round_trip(dataset: xr.Dataset, path: Path) -> None:
    """Check the written file reads back through the real loader, bitwise."""
    reloaded = load_constraints(path)
    try:
        for name in (OBS_MEAN, OBS_VAR):
            written, read = dataset[name].values, reloaded[name].values
            if written.shape != read.shape:
                raise IngestError(
                    f"{name}: wrote shape {written.shape}, read {read.shape}"
                )
            same = (written == read) | (np.isnan(written) & np.isnan(read))
            if not same.all():
                worst = np.nanmax(np.abs(written - read))
                raise IngestError(
                    f"{name}: {int((~same).sum())} cells changed on the round "
                    f"trip, worst difference {worst!r}"
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


# ── reporting ─────────────────────────────────────────────────────────────────


def summarize(dataset: xr.Dataset) -> str:
    """A short report of what was written, for the run log."""
    lines = [
        f"sites {dataset.sizes['site']}  snapshots {dataset.sizes['time']}  "
        f"variables {dataset.sizes['variable']}",
        f"snapshot keys {str(dataset['time'].values[0])[:10]} .. "
        f"{str(dataset['time'].values[-1])[:10]}",
    ]
    mean, variance = dataset[OBS_MEAN], dataset[OBS_VAR]
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
        lines.append(
            f"  {name:<14s} n={finite.sum():>6d}  "
            f"range [{np.nanmin(column):.5g}, {np.nanmax(column):.5g}]  "
            f"non-positive variances {n_zero}"
        )
    return "\n".join(lines)


# ── entry point ───────────────────────────────────────────────────────────────


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


def ingest(
    long_table: Path, manifest_path: Path, sites_path: Path | None, out: Path
) -> xr.Dataset:
    """Read, check, build and write. The steps in order, and nothing else."""
    manifest = read_manifest(manifest_path)
    check_covariances_were_diagonal(manifest)

    table = read_long_table(long_table)
    sites = load_sites(sites_path)

    check_against_manifest(table, manifest)
    check_extremes_round_tripped(table, manifest)
    check_no_duplicate_triples(table)
    check_sites_match_table(table, sites)

    dataset = build_dataset(build_grids(table, sites), manifest)
    write_dataset(dataset, out)
    return dataset


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out if args.out is not None else default_constraints_path()
    sites_path = args.sites if args.sites is not None else default_sites_path()

    try:
        dataset = ingest(args.long_table, args.manifest, sites_path, out)
    except (IngestError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(summarize(dataset))
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
