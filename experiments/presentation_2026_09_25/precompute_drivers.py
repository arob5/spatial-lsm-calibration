"""Summarize the driver ensemble at every site, for the deck's maps and PFT figures.

Overview
--------
Reads the drivers of every site in the site table, for ``config.DRIVER_MEMBERS``,
and reduces each member's 13-year 3-hourly record to two small summaries: each
calendar year's value, and the mean seasonal cycle by month. The full record
is 80,000 files, so the work is split into tasks, each writing its own part,
and a final step combines the parts. Meant for the SCC, as an array job; see
``precompute_drivers.qsub``.

Input data
----------
``data/raw/drivers/ERA5_<site>_<member>/ERA5.<member>.<start>.<end>.clim``
    The raw drivers, as :mod:`sipnet_calibration.drivers` reads them. Their
    hour column is corrected in a scratch copy, exactly as
    ``relabel_drivers.py`` corrects it, before they are read; the raw files
    are never changed.
``data/processed/sites/sites.csv``
    The site table: which sites exist, and their ``lon``/``lat``.

Output data
-----------
``config.DRIVER_SUMMARY_DIR``, holding two netCDF files, each with the eight
driver variables as ``float32``, under pySIPNET's names and with the
attributes :func:`sipnet_calibration.obs_ops.aggregate_time` gives them:

``driver_annual.nc``
    On ``(member, site, year)``: each calendar year's value, a sum for
    precipitation and radiation and a time-weighted mean for the rest.
``driver_monthly_climatology.nc``
    On ``(member, site, month)``: each calendar month's value, as above,
    averaged over the years; ``month`` is 1-12.

Both carry ``lon``/``lat`` on ``site`` and ``source_member_index`` on
``member``. The per-task parts go under ``parts/`` beside them.

Notes
-----
Only the hour-label drift is corrected (``data/README.md`` Note 15); the other
known issues with these drivers are not, which the dataset attributes say.

Every year of the record must be complete, so an annual value never covers
part of a year; this is checked per site.

Usage
-----
From this directory, so that ``config`` imports. One task of an array job,
and the combining step after every task has finished::

    python precompute_drivers.py --task 3 --n-tasks 100
    python precompute_drivers.py --combine

Locally, on the files this working copy holds, with the source root named::

    python precompute_drivers.py --task 1 --n-tasks 1 --sites 1 --members 1 2 \\
        --source-root <root checkout>/data/raw/drivers
    python precompute_drivers.py --combine --sites 1
"""

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration import drivers
from sipnet_calibration.obs_ops import aggregate_time
from sipnet_calibration.sites import load_sites

import config
from relabel_drivers import with_regular_hour_column

#: The two summaries, by file stem.
SUMMARIES = ("driver_annual", "driver_monthly_climatology")

#: Attributes written on both summaries.
DATASET_ATTRIBUTES = {
    "comment": (
        "Summaries of the ERA5 driver ensemble for the 2026-09-25 progress "
        "presentation. The hour column of each .clim file was set to 3 * slot "
        "before reading, correcting the label drift of data/README.md Note 15; "
        "no other known issue with these drivers is corrected."
    ),
    "source": config.DRIVERS_SOURCE,
}


# ── entry point ──────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    table = load_sites()
    sites = args.sites if args.sites else table["site_id"].tolist()
    try:
        if args.combine:
            combine_parts(args.output_dir, sites)
        else:
            summarize_task(args, table, sites)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", type=int, help="this task's number, 1-based")
    parser.add_argument("--n-tasks", type=int, help="how many tasks share the sites")
    parser.add_argument(
        "--combine", action="store_true", help="combine every task's part into the outputs"
    )
    parser.add_argument(
        "--source-root", type=Path, default=drivers.default_drivers_root(),
        help="the raw drivers root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=config.DRIVER_SUMMARY_DIR,
        help="where the summaries go (default: %(default)s)",
    )
    parser.add_argument(
        "--sites", type=int, nargs="+", help="site ids (default: every site in the site table)"
    )
    parser.add_argument(
        "--members", type=int, nargs="+", default=list(config.DRIVER_MEMBERS),
        help="1-based member indices (default: config.DRIVER_MEMBERS)",
    )
    args = parser.parse_args(argv)
    if not args.combine and (args.task is None or args.n_tasks is None):
        parser.error("give --task and --n-tasks, or --combine")
    return args


# ── steps ────────────────────────────────────────────────────────────────────


def summarize_task(args: argparse.Namespace, table: pd.DataFrame, sites: list[int]) -> None:
    """Summarize this task's share of *sites* and write it as one part per summary."""
    check_task_number(args.task, args.n_tasks)
    share = np.array_split(np.asarray(sorted(sites)), args.n_tasks)[args.task - 1]
    summaries = {stem: [] for stem in SUMMARIES}
    for k, site in enumerate(share, start=1):
        annual, monthly = summarize_site(args.source_root, int(site), args.members, table)
        summaries["driver_annual"].append(annual)
        summaries["driver_monthly_climatology"].append(monthly)
        print(f"task {args.task}: site {site} ({k} of {len(share)})", flush=True)
    for stem, parts in summaries.items():
        _write_netcdf(xr.concat(parts, dim="site"), part_path(args.output_dir, stem, args.task))


def summarize_site(
    source_root: Path, site: int, members: Sequence[int], table: pd.DataFrame
) -> tuple[xr.Dataset, xr.Dataset]:
    """The annual values and the monthly climatology of one site's drivers."""
    with tempfile.TemporaryDirectory() as scratch:
        for member in members:
            _write_relabeled_copy(source_root, Path(scratch), site, member)
        dataset = drivers.load_drivers(
            [site], members=members, root=scratch, sites_table=table, time_zone="UTC"
        )
    fields = drivers.driver_fields(dataset)
    annual = xr.Dataset({name: annual_values(field) for name, field in fields.items()})
    monthly = xr.Dataset({name: monthly_climatology(field) for name, field in fields.items()})
    return annual, monthly


def annual_values(field: xr.DataArray) -> xr.DataArray:
    """*field* aggregated to calendar years, on a ``year`` dimension."""
    yearly = aggregate_time(field, "YS")
    check_years_are_complete(yearly)
    years = yearly["time_step_start"].dt.year.to_numpy()
    return _on_new_dim(yearly, "year", years)


def monthly_climatology(field: xr.DataArray) -> xr.DataArray:
    """*field* aggregated to calendar months, then averaged over the years."""
    monthly = aggregate_time(field, "MS")
    months = monthly["time_step_start"].dt.month.to_numpy()
    # Grouped by a coordinate on time, not a dimension of repeated labels: the
    # latter drops every other coordinate.
    by_month = monthly.assign_coords(month=("time", months)).drop_vars(_time_coords(monthly))
    climatology = by_month.groupby("month").mean("time", keep_attrs=True)
    climatology.attrs["comment"] = "Each calendar month's value, averaged over the years."
    return climatology


def combine_parts(output_dir: Path, sites: list[int]) -> None:
    """Concatenate every task's parts, check them, and write the two summaries."""
    for stem in SUMMARIES:
        paths = sorted((output_dir / "parts").glob(f"{stem}_*.nc"))
        if not paths:
            raise FileNotFoundError(f"no parts matching {stem}_*.nc under {output_dir / 'parts'}")
        combined = xr.concat([xr.load_dataset(p, engine="h5netcdf") for p in paths], dim="site")
        combined = combined.sortby("site")
        check_every_site_is_present(combined, sites, stem)
        check_no_missing_values(combined, stem)
        combined.attrs.update(DATASET_ATTRIBUTES)
        _write_netcdf(combined, output_dir / f"{stem}.nc")
        print(f"wrote {output_dir / f'{stem}.nc'}: {dict(combined.sizes)}")


def part_path(output_dir: Path, stem: str, task: int) -> Path:
    """Where one task's part of one summary goes."""
    return output_dir / "parts" / f"{stem}_{task:04d}.nc"


# ── supporting helpers ───────────────────────────────────────────────────────


def _write_relabeled_copy(source_root: Path, scratch: Path, site: int, member: int) -> None:
    """Copy one pair's file into *scratch*, in the same layout, hour column corrected."""
    source = drivers.driver_file(source_root, site, member)
    target = scratch / source.parent.name / source.name
    target.parent.mkdir(parents=True)
    target.write_text(with_regular_hour_column(source.read_text(), source=source))


def _on_new_dim(field: xr.DataArray, dim: str, values: np.ndarray) -> xr.DataArray:
    """*field* with its ``time`` dimension replaced by *dim*, labeled *values*."""
    return field.drop_vars(_time_coords(field)).rename(time=dim).assign_coords({dim: values})


def _time_coords(field: xr.DataArray) -> list[str]:
    """The coordinates of *field* on its ``time`` dimension."""
    return [name for name, coord in field.coords.items() if "time" in coord.dims]


def _write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    """Write *dataset* as ``float32`` to a ``.partial`` path, then rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    encoding = {name: {"dtype": "float32"} for name in dataset.data_vars}
    dataset.to_netcdf(partial, engine="h5netcdf", encoding=encoding)
    os.replace(partial, path)


# ── checks ───────────────────────────────────────────────────────────────────


def check_task_number(task: int, n_tasks: int) -> None:
    """The task number is between 1 and the number of tasks."""
    if not 1 <= task <= n_tasks:
        raise ValueError(f"--task must be between 1 and --n-tasks ({n_tasks}), got {task}")


def check_years_are_complete(yearly: xr.DataArray) -> None:
    """Every year's steps cover the whole calendar year."""
    starts = pd.DatetimeIndex(yearly["time_step_start"].to_numpy())
    covered = pd.to_timedelta(yearly["time_step_length"].to_numpy())
    calendar = pd.to_datetime([f"{year + 1}-01-01" for year in starts.year]) - starts
    if not (starts == starts.normalize()).all() or (covered != calendar).any():
        raise ValueError(
            f"{yearly.name}: a year of the record is incomplete, so its annual value "
            "would cover part of a year; every year must start at 00:00 on 1 January "
            "and run to the next"
        )


def check_every_site_is_present(combined: xr.Dataset, sites: list[int], stem: str) -> None:
    """The parts together hold every requested site, each once."""
    held = combined["site"].to_numpy()
    if len(np.unique(held)) != len(held):
        raise ValueError(f"{stem}: a site appears in more than one part; delete stale parts")
    missing = sorted(set(sites) - set(held.tolist()))
    if missing:
        raise ValueError(
            f"{stem}: {len(missing)} site(s) have no part, the first {missing[:5]}; "
            "rerun the tasks that failed before combining"
        )


def check_no_missing_values(combined: xr.Dataset, stem: str) -> None:
    """No summary value is missing: every site had every member's file."""
    for name, values in combined.data_vars.items():
        if values.isnull().any():
            raise ValueError(f"{stem}: {name} has missing values")


if __name__ == "__main__":
    sys.exit(main())
