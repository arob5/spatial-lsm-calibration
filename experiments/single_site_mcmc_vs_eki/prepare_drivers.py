"""Prepare the experiment's driver files: the raw ERA5 files, hour labels regularized.

Overview
--------
Copies the configured site's driver file for the configured member from the
raw drivers root into the experiment's output, with its hour column set to
``3 * slot`` and every other field kept as written, and checks that pySIPNET
reads the copy. The raw file is never edited.

Input data
----------
``config.RAW_DRIVERS_ROOT / f"ERA5_{SITE}_{DRIVER_SOURCE_INDEX}" / "ERA5.*.clim"``:
exactly one SIPNET climate file, 14 tab-separated fields a row, 3-hourly,
eight rows to a day from hour 0 (``data/README.md``, Drivers).

Output data
-----------
The same file name under ``config.PREPARED_DRIVERS_ROOT``, in a directory of
the same name, so ``drivers.load_drivers(root=PREPARED_DRIVERS_ROOT)`` reads
it as it would the raw root.

Notes
-----
The raw files' hour column drifts from the three-hourly grid it stands for,
by up to two hours within a year (``data/README.md`` Note 15, issue #9);
pySIPNET refuses a file whose labels disagree with its declared step lengths,
so the raw files cannot be run as they are. The values are exactly
three-hourly and only the labels drift, so row ``k`` of a day is labeled
``3 * (k % 8)``. This is the rewrite ``tests/conftest.py`` applies to the
local files for the tests, and it fixes the drift and nothing else: the
accumulated columns still cover the three hours ending at the label (Note 16),
which ``config.DRIVER_TIME_ZONE`` records.

Usage
-----
    uv run python experiments/single_site_mcmc_vs_eki/prepare_drivers.py
"""

import sys
from pathlib import Path

import config
from sipnet_calibration import drivers
from sipnet_calibration.io import write_checked

#: Tab-separated fields in a row of the ERA5 driver files.
FIELDS_PER_ROW = 14

#: Position of the hour-of-day field in a row.
HOUR_FIELD = 3

#: Rows in a day of three-hourly drivers.
ROWS_PER_DAY = 8

#: Hours in one step.
STEP_HOURS = 3


# ── entry point ──


def main() -> int:
    """Prepare the configured site's driver file for the configured member."""
    try:
        raw_path = find_raw_driver_file(
            config.RAW_DRIVERS_ROOT, config.SITE, config.DRIVER_SOURCE_INDEX
        )
        prepared_path = (
            config.PREPARED_DRIVERS_ROOT / raw_path.parent.name / raw_path.name
        )
        prepare_driver_file(raw_path, prepared_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {prepared_path}")
    return 0


# ── the steps ──


def find_raw_driver_file(root: Path, site: int, source_index: int) -> Path:
    """The one raw ``.clim`` file of *site* and *source_index* under *root*."""
    directory = root / f"ERA5_{site}_{source_index}"
    paths = sorted(directory.glob("ERA5.*.clim"))
    check_directory_holds_one_driver_file(directory, paths)
    return paths[0]


def prepare_driver_file(raw_path: Path, prepared_path: Path) -> None:
    """Write *raw_path*'s text with regular hour labels to *prepared_path*, checked."""
    raw_text = raw_path.read_text()
    prepared_text = with_regular_hour_column(raw_text)

    def write(partial: Path) -> None:
        partial.write_text(prepared_text)

    def check(partial: Path) -> None:
        check_only_the_hour_column_changed(raw_text, partial.read_text())
        # Reading validates the labels against the step lengths.
        drivers.read_driver_file(partial, time_zone=config.DRIVER_TIME_ZONE).xarray

    write_checked(prepared_path, write, check)


# ── helpers ──


def with_regular_hour_column(text: str) -> str:
    """*text*, a driver file's, with each row's hour set to ``3 * (k % 8)``."""
    rows = [line.split("\t") for line in text.splitlines()]
    check_rows_have_the_field_count(rows)
    check_rows_are_in_their_slots(rows)
    check_file_holds_whole_days(rows)
    for position, fields in enumerate(rows):
        fields[HOUR_FIELD] = f"{STEP_HOURS * (position % ROWS_PER_DAY):9.6f}"
    return "\n".join("\t".join(fields) for fields in rows) + "\n"


# ── checks ──


def check_directory_holds_one_driver_file(directory: Path, paths: list[Path]) -> None:
    """A member's driver directory holds exactly one ``.clim`` file."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no driver directory {directory}; copy it from the SCC's "
            "ERA5_2012_2024 directory (data/README.md, Drivers)"
        )
    if len(paths) != 1:
        raise ValueError(
            f"{directory} holds {len(paths)} .clim files, not one; "
            "leave only the file the runs should read"
        )


def check_rows_have_the_field_count(rows: list[list[str]]) -> None:
    """Every row has :data:`FIELDS_PER_ROW` tab-separated fields."""
    for position, fields in enumerate(rows):
        if len(fields) != FIELDS_PER_ROW:
            raise ValueError(
                f"row {position} has {len(fields)} tab-separated fields, not "
                f"{FIELDS_PER_ROW}; this is not an ERA5 driver file"
            )


def check_rows_are_in_their_slots(rows: list[list[str]]) -> None:
    """Every row's drifting hour still falls in its three-hour slot of the day."""
    for position, fields in enumerate(rows):
        if int(float(fields[HOUR_FIELD]) // STEP_HOURS) != position % ROWS_PER_DAY:
            raise ValueError(
                f"row {position}, hour {fields[HOUR_FIELD].strip()}, is not in "
                f"slot {position % ROWS_PER_DAY} of its day; the drift is not "
                "the one Note 15 describes, so relabeling by position is unsafe"
            )


def check_file_holds_whole_days(rows: list[list[str]]) -> None:
    """The file holds whole days of :data:`ROWS_PER_DAY` rows."""
    if len(rows) % ROWS_PER_DAY:
        raise ValueError(
            f"the file holds {len(rows)} rows, not whole days of "
            f"{ROWS_PER_DAY}; relabeling by position is unsafe"
        )


def check_only_the_hour_column_changed(raw_text: str, prepared_text: str) -> None:
    """The prepared file differs from the raw one only in the hour column."""
    raw_rows = [line.split("\t") for line in raw_text.splitlines()]
    prepared_rows = [line.split("\t") for line in prepared_text.splitlines()]
    if len(raw_rows) != len(prepared_rows):
        raise ValueError(
            f"the prepared file has {len(prepared_rows)} rows, the raw file "
            f"{len(raw_rows)}; the rewrite must keep every row"
        )
    for position, (raw, prepared) in enumerate(
        zip(raw_rows, prepared_rows, strict=True)
    ):
        if raw[:HOUR_FIELD] + raw[HOUR_FIELD + 1 :] != (
            prepared[:HOUR_FIELD] + prepared[HOUR_FIELD + 1 :]
        ):
            raise ValueError(
                f"row {position} changed outside the hour column; the rewrite "
                "must keep every other field as written"
            )


if __name__ == "__main__":
    sys.exit(main())
