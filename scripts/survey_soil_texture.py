#!/usr/bin/env python
"""Survey the soil texture ensemble, and check what the README records.

Overview
--------
Walk a soil texture root and report what holds across it: which sites have a
directory and how many members each holds, whether every file name is on the
template, what variables and depths the files carry, and what ``soilWHC`` the
PEcAn formula gives for them. Then compare the measurements against the
characteristics ``data/README.md`` records, and **exit non-zero if one no longer
holds**.

That last part is what makes this more than a diagnostic. The README records
measured characteristics of raw data, which is its job, but a number written
down and nowhere else goes stale in silence. The numbers live here as
:data:`RECORDED`, the README says what the property is and points here, and a
regenerated ensemble that changed is refused rather than absorbed.

This is not part of the ingest pipeline: nothing is written, and no processed file is
built from soil texture yet.

Input data
----------
``--root``, default ``data/raw/soil_texture``
    A directory laid out as ``<site>/Soil_params_0-<site>_<member>.nc``. Each
    file is a netCDF with a single ``depth`` dimension whose values are **layer
    bottoms in meters**, and one ``float32`` variable per soil property on it.

Output data
-----------
A report to stdout and, with ``--out``, the same content as JSON. Nothing is
written to ``data/``.

The exit status is 0 when every recorded characteristic still holds, 1 when one
does not, and 2 when the root could not be read at all.

Notes
-----
**Coverage and content are surveyed separately, because their costs differ by
four orders of magnitude.** Which sites and members exist is a directory
listing over some 770,000 files and takes seconds; opening each to read its
variables does not. So the coverage pass always runs over everything, and the
content pass runs over ``--sample`` sites chosen deterministically, or over all
of them with ``--all``. Only the coverage characteristics are recorded, since
they are the ones a partial local copy cannot establish and the SCC can.

**Only the coverage pass means anything off the SCC.** Locally the repository
holds three members of two sites, so the survey reports a coverage far short of
the recorded one. ``--no-check`` is how to run it there without the comparison
failing for that reason.

**The ``soilWHC`` computed here is PEcAn's, reproduced to show the magnitude,
not a processed file.** ``write.configs.SIPNET.R`` takes layer thickness as
``c(depth[1], diff(depth))``, converts to centimeters and sums
``volume_fraction_of_water_in_soil_at_saturation`` times thickness over the
profile. It is reported because the value it gives is six to eight times
SIPNET's 12 cm template default, which matters to anyone setting up a run; a
prior over the parameter is a later piece of work.

Usage
-----
::

    python scripts/survey_soil_texture.py --no-check            # a partial local copy
    python scripts/survey_soil_texture.py --root /path/on/scc   # the real thing
    python scripts/survey_soil_texture.py --root ... --all --out soil_survey.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from sipnet_calibration import conventions
from sipnet_calibration.sites import N_SITES

#: File name template, with the site repeated inside it. The ``0-`` prefix is
#: PEcAn's input identifier and is constant across the ensemble.
FILE_PATTERN = re.compile(r"^Soil_params_0-(\d+)_(\d+)\.nc$")

#: Layer bottoms in meters. A file on another set of depths is a different
#: data source, and ``soilWHC`` would integrate over a different profile.
DEPTHS_METERS = (0.05, 0.15, 0.3, 0.6, 1.0, 2.0)

#: The variable PEcAn integrates to get ``soilWHC``.
POROSITY = "volume_fraction_of_water_in_soil_at_saturation"

#: The three fractions that partition the mineral soil, which must sum to one.
TEXTURE_FRACTIONS = (
    "fraction_of_sand_in_soil",
    "fraction_of_silt_in_soil",
    "fraction_of_clay_in_soil",
)

#: The characteristics ``data/README.md`` records. Measured on 2026-09-21 over
#: the SCC copy named in the README's `Soil texture` section; the local copy is
#: three members of two sites and cannot reach them, which is what ``--no-check``
#: is for. A disagreement is a changed ensemble, not a changed threshold:
#: re-survey, then update the README and this table together.
RECORDED: dict[str, Any] = {
    "sites_with_a_directory": 7693,
    "sites_of_the_pool_absent": 307,
    "members_per_site": [100],
    "file_names_off_template": 0,
    "member_range": [1, 100],
    # From the content pass, so only as good as --sample; all three are
    # structural rather than distributional, and hold for every file opened at
    # any sample size. The soilWHC range is deliberately absent: it moves with
    # the sample, so it is reported and never asserted.
    "depths_are_the_expected_profile": True,
    "unreadable_files": 0,
    "files_with_missing_porosity": 0,
}


def default_data_root() -> Path:
    """The repository's ``data/``, honoring ``$SIPNET_CALIBRATION_DATA``.

    Delegates to :func:`sipnet_calibration.conventions.data_root`, which finds
    it from the installed package rather than by counting parents from this
    file, so moving a script does not silently retarget every path it reads.
    """
    return conventions.data_root()


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root or default_data_root() / "raw" / "soil_texture"
    try:
        coverage = survey_coverage(root)
        content = survey_content(root, coverage["sites"], args.sample, args.all)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report = {"root": str(root), **coverage, **content}
    report.pop("sites")
    print(format_report(report))

    if args.out is not None:
        try:
            args.out.write_text(json.dumps(report, indent=2, default=str))
        except OSError as error:
            print(f"error: could not write {args.out}: {error}", file=sys.stderr)
            return 2
        print(f"\nwrote {args.out}")

    if args.no_check:
        return 0
    failures = compare_with_recorded(report)
    print(format_comparison(failures))
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Directory of per-site subdirectories. Default: data/raw/soil_texture.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=25,
        help="Sites to open files for, evenly spaced through those present. "
        "0 opens none. Default: 25.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Open every file rather than a sample."
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Report without comparing against the recorded characteristics. Use this "
        "on a partial local copy, which cannot reach the recorded coverage.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Also write the report as JSON here."
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def survey_coverage(root: Path) -> dict[str, Any]:
    """Which sites and members exist, from directory listings alone."""
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory; see data/README.md, Soil texture")

    sites: dict[int, list[int]] = {}
    off_template: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            off_template.append(entry.name)
            continue
        if not entry.name.isdecimal() or entry.name != str(int(entry.name)):
            # isdecimal, not isdigit: the latter accepts superscripts and other
            # non-decimal digits that int() then rejects. The round-trip catches
            # "0027", which would otherwise collide with "27" and silently
            # discard one directory's files.
            off_template.append(f"{entry.name}/")
            continue
        site = int(entry.name)
        members = []
        for file in entry.iterdir():
            match = FILE_PATTERN.match(file.name)
            if match is None or int(match.group(1)) != site:
                off_template.append(f"{entry.name}/{file.name}")
                continue
            members.append(int(match.group(2)))
        sites[site] = sorted(members)

    if not sites:
        raise ValueError(f"{root}: holds no site directories on the template")

    counts = sorted({len(members) for members in sites.values()})
    all_members = sorted({member for members in sites.values() for member in members})
    if not all_members:
        raise ValueError(
            f"{root}: {len(sites)} site directories, none holding a file on the "
            "template. An interrupted copy looks like this."
        )
    return {
        "sites": sites,
        "sites_with_a_directory": len(sites),
        "sites_of_the_pool_absent": N_SITES - len(set(sites) & set(range(1, N_SITES + 1))),
        "sites_outside_the_pool": sorted(set(sites) - set(range(1, N_SITES + 1)))[:5],
        "members_per_site": counts,
        "member_range": [all_members[0], all_members[-1]],
        "members_are_contiguous": all_members == list(range(all_members[0], all_members[-1] + 1)),
        "file_names_off_template": len(off_template),
        "first_off_template": off_template[:5],
        "total_files": sum(len(members) for members in sites.values()),
    }


def survey_content(
    root: Path, sites: dict[int, list[int]], sample: int, every: bool
) -> dict[str, Any]:
    """What the files hold, over every file or an evenly spaced sample of sites."""
    chosen = sorted(sites) if every else _evenly_spaced(sorted(sites), sample)
    variables: Counter[tuple[str, ...]] = Counter()
    depth_sets: set[tuple[float, ...]] = set()
    units: dict[str, set[str]] = {}
    water_capacities: list[float] = []
    fraction_residuals: list[float] = []
    unreadable: list[str] = []
    opened = 0

    for site in chosen:
        for member in sites[site]:
            path = root / str(site) / f"Soil_params_0-{site}_{member}.nc"
            try:
                with xr.open_dataset(path, decode_times=False) as dataset:
                    opened += 1
                    variables[tuple(sorted(dataset.data_vars))] += 1
                    depth_sets.add(tuple(float(value) for value in dataset["depth"].values))
                    for name, array in dataset.data_vars.items():
                        units.setdefault(name, set()).add(
                            str(array.attrs.get("units", ""))
                        )
                    if POROSITY in dataset.data_vars:
                        water_capacities.append(soil_water_holding_capacity(dataset))
                    if all(name in dataset.data_vars for name in TEXTURE_FRACTIONS):
                        total = sum(
                            dataset[name].values.astype(np.float64)
                            for name in TEXTURE_FRACTIONS
                        )
                        fraction_residuals.append(float(np.nanmax(np.abs(total - 1.0))))
            except (OSError, KeyError, ValueError) as error:
                unreadable.append(f"{path.name}: {error}")

    finite = [value for value in water_capacities if np.isfinite(value)]
    return {
        "sites_opened": len(chosen),
        "files_opened": opened,
        "depths_are_the_expected_profile": bool(
            depth_sets and depth_sets == {tuple(float(d) for d in DEPTHS_METERS)}
        ),
        "files_with_missing_porosity": len(water_capacities) - len(finite),
        "variable_sets": [
            {"variables": list(names), "files": count}
            for names, count in variables.most_common()
        ],
        "depth_sets": sorted(depth_sets),
        "units": {name: sorted(seen) for name, seen in sorted(units.items())},
        "soil_water_holding_capacity_cm": _extremes(finite),
        "max_texture_fraction_residual": max(fraction_residuals, default=None),
        "unreadable_files": len(unreadable),
        "first_unreadable": unreadable[:5],
    }


def soil_water_holding_capacity(dataset: xr.Dataset) -> float:
    """``soilWHC`` in cm, by the formula ``write.configs.SIPNET.R`` uses.

    Layer thickness is ``c(depth[1], diff(depth))``, taking the depth values as
    layer bottoms with the first layer's top at the surface; ``soilWHC`` is
    porosity times thickness, summed over the profile and converted to
    centimeters.

    Returns ``nan`` where any layer's porosity is missing, which is what PEcAn
    does: its ``sum`` takes the default ``na.rm = FALSE``, so one absent layer
    makes the whole parameter ``NA``. Skipping the layer instead would return a
    partial-profile integral that cannot be told from a genuinely dry profile.
    """
    depths = dataset["depth"].values
    thickness = np.concatenate([[depths[0]], np.diff(depths)])
    return float(np.sum(dataset[POROSITY].values * thickness) * 100.0)


def format_report(report: dict[str, Any]) -> str:
    """The measurements as readable lines."""
    lines = [
        f"{report['root']}",
        f"  sites with a directory   : {report['sites_with_a_directory']}",
        f"  pool sites absent        : {report['sites_of_the_pool_absent']} of {N_SITES}",
        f"  members per site         : {report['members_per_site']}",
        f"  member range             : {report['member_range'][0]}-{report['member_range'][1]}"
        f", contiguous {_yes(report['members_are_contiguous'])}",
        f"  files on the template    : {report['total_files']}",
        f"  file names off template  : {report['file_names_off_template']}"
        + (f" (first {report['first_off_template']})" if report["first_off_template"] else ""),
    ]
    if report["sites_outside_the_pool"]:
        lines.append(f"  sites outside the pool   : {report['sites_outside_the_pool']}")
    lines += [
        "",
        f"  files opened             : {report['files_opened']} over "
        f"{report['sites_opened']} sites",
        f"  distinct variable sets   : {len(report['variable_sets'])}",
        f"  distinct depth sets      : {[list(one) for one in report['depth_sets']]}",
        f"  unreadable files         : {report['unreadable_files']}"
        + (f" (first {report['first_unreadable']})" if report["first_unreadable"] else ""),
        f"  depths as expected       : {_yes(report['depths_are_the_expected_profile'])}",
        f"  files missing a porosity : {report['files_with_missing_porosity']}",
    ]
    lines += _format_variable_sets(report["variable_sets"])
    capacity = report["soil_water_holding_capacity_cm"]
    if capacity is not None:
        lines.append(
            f"  soilWHC over 2 m, cm     : {capacity['min']:.1f} to {capacity['max']:.1f}, "
            f"median {capacity['median']:.1f}  (SIPNET's template default is 12)"
        )
    residual = report["max_texture_fraction_residual"]
    if residual is not None:
        lines.append(f"  max |sand+silt+clay - 1| : {residual:.3g}")
    if report["units"]:
        lines.append("")
        width = max(len(name) for name in report["units"])
        lines.append("  variables and units:")
        lines += [
            f"    {name:<{width}} : "
            + " | ".join(unit or "(none)" for unit in seen)
            + ("   <- DISAGREES ACROSS FILES" if len(seen) > 1 else "")
            for name, seen in report["units"].items()
        ]
    return "\n".join(lines)


def compare_with_recorded(report: dict[str, Any]) -> list[str]:
    """Which recorded characteristics no longer hold, as readable lines."""
    failures = []
    for key, expected in RECORDED.items():
        measured = report.get(key)
        if measured != expected:
            failures.append(f"{key}: recorded {expected}, measured {measured}")
    return failures


def format_comparison(failures: list[str]) -> str:
    """The comparison against :data:`RECORDED`, as readable lines."""
    if not failures:
        return "\nEvery characteristic data/README.md records for the ensemble still holds."
    lines = [
        f"\nerror: {len(failures)} characteristic(s) data/README.md records for the soil "
        "texture ensemble no longer hold:"
    ]
    lines += [f"  {failure}" for failure in failures]
    lines.append(
        "\nOn a partial copy this is expected and --no-check is the right answer; the "
        "recorded coverage can only be reached where the whole ensemble is. Otherwise "
        "the ensemble has been regenerated: establish what changed, then update "
        "RECORDED in this script and the Soil texture section of data/README.md "
        "together."
    )
    return "\n".join(lines)


# ── supporting helpers ────────────────────────────────────────────────────────


def _format_variable_sets(sets: list[dict[str, Any]]) -> list[str]:
    """One line per variable set, the largest first, saying what each lacks."""
    if not sets:
        return []
    fullest = set(sets[0]["variables"])
    lines = []
    for entry in sets:
        names = set(entry["variables"])
        absent = sorted(fullest - names)
        what = "all of them" if not absent else f"all but {', '.join(absent)}"
        lines.append(f"    {len(names):>3} variables, {entry['files']:>6} files : {what}")
    return ["  variable sets:", *lines]


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    """At most *count* of *values*, evenly spaced, deterministically.

    Even spacing rather than a random sample so that a rerun surveys the same
    sites, and so that a sample of an identifier-ordered pool spans it: the
    identifiers run north to south, so the first *n* would all be Arctic.
    """
    if count < 0:
        raise ValueError(f"sample must not be negative, got {count}")
    if count == 0:
        return []
    if count >= len(values):
        return values
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[position] for position in dict.fromkeys(positions.tolist())]


def _extremes(values: list[float]) -> dict[str, float] | None:
    """The min, median and max of *values*, or ``None`` where there are none."""
    if not values:
        return None
    array = np.asarray(values)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _yes(value: bool) -> str:
    return "yes" if value else "NO"


if __name__ == "__main__":
    raise SystemExit(main())
