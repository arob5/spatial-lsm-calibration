"""Write copies of the deck's driver files with their hour column corrected.

Overview
--------
A temporary workaround until the ERA5 drivers are regenerated. The hour
column of the ERA5 ``.clim`` files drifts away from the declared 3-hour steps
(``data/README.md`` Note 15), so pySIPNET refuses them. This script copies the
files for ``config.DRIVER_SITES`` x ``config.DRIVER_MEMBERS`` with that one
column set to ``3 * slot``, so the deck can run SIPNET on them and read them
with :func:`sipnet_calibration.drivers.load_drivers`.

Input data
----------
``data/raw/drivers/ERA5_<site>_<member>/ERA5.<member>.<start>.<end>.clim``
    The raw drivers, in the layout :mod:`sipnet_calibration.drivers` reads:
    tab-separated, 14 fields to a row, 3-hourly, eight rows to a day starting
    at hour 0, with the hour in the fourth field. Every requested pair must
    have a file. Only the SCC holds the full set.

Output data
-----------
``config.RELABELED_DRIVERS_DIR``, in the same layout and with the same file
names, so it can be passed as ``root=`` wherever a drivers root is taken. Each
file is the source with only its hour field rewritten, as ``3 * (k % 8)`` for
the row at position ``k``; every other field is copied as written.

Notes
-----
The drift is in the labels alone: the values sit on a regular 3-hour UTC grid,
so relabeling them is a correction rather than an approximation. Nothing else
in Note 16 is changed: radiation and precipitation still cover the step ending
at the label, and the other forcings are still instantaneous at it.

A file is rewritten only if every row's drifted hour lies in the 3-hour slot
its position gives it, so a file that does not follow the pattern is refused
rather than relabeled wrongly. Each output is written to ``.partial``, read
back through pySIPNET, and renamed; then every site is read back through
``load_drivers``, which also checks the file names and the shared time axis.

``tests/conftest.py`` makes the same correction for the tests, in
``with_regular_hour_column``.

Usage
-----
From this directory, so that ``config`` imports. On the SCC, for the deck's
sites and members::

    python relabel_drivers.py

Locally, on the files this working copy holds::

    python relabel_drivers.py --sites 1 --members 1 2
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sipnet_calibration import drivers

import config

#: Rows to a day, and hours to a step, in the ERA5 driver files.
STEPS_PER_DAY = 8
STEP_HOURS = 3

#: Tab-separated fields to a row, and the position of the hour among them.
N_FIELDS = 14
HOUR_FIELD = 3


# ── entry point ──────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        for site in args.sites:
            for member in args.members:
                relabel_file(args.source_root, args.output_root, site, member)
            check_site_reads_back(args.output_root, site, args.members)
            print(f"site {site}: {len(args.members)} member(s) written")
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(args.sites) * len(args.members)} file(s) under {args.output_root}")
    return 0


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source-root", type=Path, default=drivers.default_drivers_root(),
        help="the raw drivers root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root", type=Path, default=config.RELABELED_DRIVERS_DIR,
        help="where the corrected copies go (default: %(default)s)",
    )
    parser.add_argument(
        "--sites", type=int, nargs="+", default=list(config.DRIVER_SITES),
        help="site ids (default: config.DRIVER_SITES)",
    )
    parser.add_argument(
        "--members", type=int, nargs="+", default=list(config.DRIVER_MEMBERS),
        help="1-based member indices (default: config.DRIVER_MEMBERS)",
    )
    return parser.parse_args(argv)


# ── steps ────────────────────────────────────────────────────────────────────


def relabel_file(source_root: Path, output_root: Path, site: int, member: int) -> Path:
    """Write one pair's corrected copy, checked by pySIPNET before it is renamed."""
    source = drivers.driver_file(source_root, site, member)
    target = output_root / source.parent.name / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_text(with_regular_hour_column(source.read_text(), source=source))
    drivers.read_driver_file(partial)
    partial.replace(target)
    return target


def with_regular_hour_column(text: str, *, source: Path) -> str:
    """*text* with the hour field of the row at position ``k`` set to ``3 * (k % 8)``."""
    rows = text.splitlines()
    check_whole_days(rows, source)
    lines = []
    for k, row in enumerate(rows):
        fields = row.split("\t")
        check_row_shape(fields, k, source)
        check_hour_is_in_its_slot(fields[HOUR_FIELD], k, source)
        fields[HOUR_FIELD] = f"{STEP_HOURS * (k % STEPS_PER_DAY):9.6f}"
        lines.append("\t".join(fields))
    return "\n".join(lines) + "\n"


# ── checks ───────────────────────────────────────────────────────────────────


def check_whole_days(rows: list[str], source: Path) -> None:
    """The file holds whole days of ``STEPS_PER_DAY`` rows."""
    if not rows or len(rows) % STEPS_PER_DAY:
        raise ValueError(
            f"{source}: {len(rows)} rows is not a whole number of days of "
            f"{STEPS_PER_DAY}; this correction assumes 3-hourly files from hour 0"
        )


def check_row_shape(fields: list[str], k: int, source: Path) -> None:
    """A row has ``N_FIELDS`` tab-separated fields."""
    if len(fields) != N_FIELDS:
        raise ValueError(
            f"{source}: row {k} has {len(fields)} tab-separated fields, not {N_FIELDS}"
        )


def check_hour_is_in_its_slot(hour: str, k: int, source: Path) -> None:
    """The drifted hour of row ``k`` lies in the 3-hour slot its position gives it."""
    slot = k % STEPS_PER_DAY
    if int(float(hour) // STEP_HOURS) != slot:
        raise ValueError(
            f"{source}: row {k} is labeled hour {hour.strip()}, outside slot {slot} "
            f"({STEP_HOURS * slot}-{STEP_HOURS * (slot + 1)} h); the drift is not the "
            "one this correction assumes, so the file is left alone"
        )


def check_site_reads_back(output_root: Path, site: int, members: Sequence[int]) -> None:
    """Every member written for *site* reads back through ``load_drivers``."""
    drivers.load_drivers([site], members=members, root=output_root)


if __name__ == "__main__":
    sys.exit(main())
