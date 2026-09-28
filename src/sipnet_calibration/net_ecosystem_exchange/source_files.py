"""AmeriFlux's FLUXNET files: what one looks like, and how to parse it.

The observations arrive as AmeriFlux FLUXNET (ONEFlux) FULLSET files, one CSV
per tower at half-hourly or hourly resolution, present only on the SCC under
``data/raw/net_ecosystem_exchange/fluxnet/``. This module holds the contract
those files satisfy and the parser that enforces it.
:mod:`sipnet_calibration.net_ecosystem_exchange.raw` lays the parsed records on
one array per resolution, which everything else reads.

Contents
--------
:data:`SOURCE` is the format, one :class:`SourceFormat`: the file-name pattern,
a :class:`SourceColumn` per column the conversion keeps, which of those a file
may lack, the fill value, and the flag vocabularies.

:func:`read_source_file` parses one file onto the raw axis, holding it to that
format; :func:`discover_source_files` lists a directory's FULLSET files, and
:func:`parse_file_name` decodes a file name. Each parsed file is a
:class:`SourceFile`.

Notes
-----
**Timestamps are never parsed through a time zone.** The files' stamps are
local standard time with no daylight saving, as the AmeriFlux data-variables
page states (https://ameriflux.lbl.gov/data/aboutdata/data-variables/).
Parsing them with a zone -- including the session's default zone, which is what
an R ``as.POSIXct(..., tz = "")`` does -- applies that zone's daylight saving
and drops or duplicates the half-hours around each transition. Here a stamp is
read as a ``YYYYMMDDHHMM`` integer and turned into a naive timestamp, and the
shift to UTC is a fixed offset applied later, per tower.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from frozendict import frozendict

from sipnet_calibration.net_ecosystem_exchange.names import (
    RAW_START,
    RESOLUTIONS,
    Resolution,
)
from sipnet_calibration.validation import truncated

__all__ = [
    "FLAG",
    "SOURCE",
    "SourceColumn",
    "SourceFile",
    "SourceFormat",
    "VALUE",
    "discover_source_files",
    "parse_file_name",
    "read_source_file",
]

#: A column holding a physical value, ``float64`` with ``NaN`` for missing.
VALUE = "value"

#: A column holding a flag, ``int8`` with ``-1`` for missing.
FLAG = "flag"


@dataclass(frozen=True)
class SourceColumn:
    """One column the conversion keeps, as the FLUXNET2015 FULLSET table defines it."""

    name: str
    """The column's name in the source file and in the raw file."""

    kind: str
    """:data:`VALUE` or :data:`FLAG`."""

    units: str
    """The unit the FULLSET table gives for the half-hourly or hourly value, or ``""``."""

    long_name: str
    """The FULLSET table's description, shortened."""

    flag_values: tuple[int, ...] = ()
    """The values a flag may take; empty for a value column."""

    flag_meanings: str = ""
    """The CF ``flag_meanings`` string for a flag column."""


@dataclass(frozen=True)
class SourceFormat:
    """The contract every FULLSET file satisfies."""

    file_pattern: re.Pattern[str]
    """A FULLSET file name, with the tower, resolution code, years and version as groups."""

    columns: Mapping[str, SourceColumn]
    """The kept columns, by name, in the raw file's order."""

    optional_columns: frozenset[str]
    """Kept columns a file may lack, all together or not at all: ONEFlux's
    constant-u*-threshold estimates, which some towers' files do not carry."""

    fill_value: float
    """The source's missing-value marker."""

    documentation: str
    """Where the variables are defined."""

    @property
    def names(self) -> tuple[str, ...]:
        """The kept columns' names, in order."""
        return tuple(self.columns)


@dataclass(frozen=True)
class SourceFile:
    """One parsed FULLSET file, its kept columns laid on the raw axis."""

    tower: str
    resolution: Resolution
    first_year: int
    last_year: int
    version: str
    file_name: str
    absent_columns: tuple[str, ...]
    """Kept columns the file does not carry; stored as all-missing."""
    values: Mapping[str, np.ndarray] = field(repr=False)
    """Column name to its values on the raw axis of the resolution: ``float64``
    with ``NaN``, or ``int8`` with ``-1``, for every step the file does not
    cover or reports as missing."""


def parse_file_name(name: str) -> tuple[str, Resolution, int, int, str]:
    """``(tower, resolution, first_year, last_year, version)`` from a FULLSET file name.

    Raises
    ------
    ValueError
        If the name does not match the FULLSET pattern.
    """
    check_file_name_is_a_fullset_name(name)
    match = SOURCE.file_pattern.match(name)
    resolution = next(r for r in RESOLUTIONS.values() if r.source_code == match["code"])
    return (
        match["tower"],
        resolution,
        int(match["first_year"]),
        int(match["last_year"]),
        match["version"],
    )


def discover_source_files(root: Path | str) -> list[Path]:
    """Every FULLSET half-hourly or hourly CSV under *root*, sorted by name.

    The download holds other files beside them (the zips, the daily to yearly
    aggregates, the ERA5 and auxiliary files, the site listing), which are
    passed over.

    Raises
    ------
    FileNotFoundError
        If *root* is not a directory.
    ValueError
        If *root* holds no FULLSET file, or two for one tower.
    """
    root = Path(root)
    check_source_root_is_a_directory(root)
    paths = sorted(
        path for path in root.iterdir() if SOURCE.file_pattern.match(path.name) is not None
    )
    check_source_root_holds_files(paths, root)
    check_source_files_are_one_per_tower(paths)
    return paths


def read_source_file(path: Path | str) -> SourceFile:
    """Parse one FULLSET file onto the raw axis of its resolution.

    Parameters
    ----------
    path:
        A half-hourly or hourly FULLSET CSV.

    Returns
    -------
    SourceFile
        The kept columns on the raw axis, values unchanged: each ``float64`` is
        the nearest double to the source's decimal text, the fill value becomes
        ``NaN`` (``-1`` for a flag), and a step the file does not cover is
        missing.

    Raises
    ------
    ValueError
        If the name is not a FULLSET name; a column other than the optional
        ones is absent, or the optional ones are only partly present; a field is
        empty or not a number; the stamps are not contiguous whole years at the
        file's resolution; a value is non-finite or a fill-like number other
        than the fill value; a flag holds a value outside its vocabulary; or an
        NEE estimate's value is missing where its flag is present.
    """
    path = Path(path)
    tower, resolution, first_year, last_year, version = parse_file_name(path.name)
    header = _read_header(path)
    check_source_header_is_the_format(header, path)
    absent = _absent_optional_columns(header)
    frame = _parsed_source_file(path, [name for name in SOURCE.names if name not in absent])
    start = _stamps_of(frame, "TIMESTAMP_START", path)
    end = _stamps_of(frame, "TIMESTAMP_END", path)
    check_source_stamps_tile_the_declared_years(start, end, resolution, first_year, last_year, path)

    values = _values_on_the_raw_axis(frame, start, resolution, absent, path)
    check_source_values_are_present_where_flagged(values, absent, path)
    return SourceFile(
        tower=tower,
        resolution=resolution,
        first_year=first_year,
        last_year=last_year,
        version=version,
        file_name=path.name,
        absent_columns=tuple(sorted(absent)),
        values=values,
    )


# ── private helpers ───────────────────────────────────────────────────────────

#: The two stamp columns every file starts with.
_TIMESTAMP_COLUMNS = ("TIMESTAMP_START", "TIMESTAMP_END")


def _read_header(path: Path) -> list[str]:
    """The file's column names."""
    with path.open() as handle:
        return handle.readline().strip().split(",")


def _absent_optional_columns(header: list[str]) -> frozenset[str]:
    """The optional columns the header lacks."""
    return frozenset(name for name in SOURCE.optional_columns if name not in header)


def _parsed_source_file(path: Path, present: list[str]) -> pd.DataFrame:
    """The stamps and the present kept columns, read exactly."""
    try:
        return pd.read_csv(
            path,
            usecols=[*_TIMESTAMP_COLUMNS, *present],
            dtype={
                **{name: np.int64 for name in _TIMESTAMP_COLUMNS},
                **{name: np.float64 for name in present},
            },
            na_filter=False,
            float_precision="round_trip",
        )
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"{path.name}: a field is empty or not a number ({error}); the file is not a FULLSET "
            "file as downloaded."
        ) from error


def _stamps_of(frame: pd.DataFrame, column: str, path: Path) -> pd.DatetimeIndex:
    """A stamp column as naive timestamps, never through a time zone."""
    try:
        return pd.DatetimeIndex(pd.to_datetime(frame[column].to_numpy().astype(str), format="%Y%m%d%H%M"))
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"{path.name}: {column} holds a value that is not YYYYMMDDHHMM ({error}); the file is "
            "not a FULLSET file as downloaded."
        ) from error


def _values_on_the_raw_axis(
    frame: pd.DataFrame,
    start: pd.DatetimeIndex,
    resolution: Resolution,
    absent: frozenset[str],
    path: Path,
) -> dict[str, np.ndarray]:
    """Every kept column on the resolution's raw axis, missing outside the file."""
    n_raw = len(resolution.raw_step_starts())
    index = ((start - RAW_START) // resolution.step).to_numpy(np.int64)
    inside = (index >= 0) & (index < n_raw)
    values: dict[str, np.ndarray] = {}
    for name, column in SOURCE.columns.items():
        if column.kind == VALUE:
            array = np.full(n_raw, np.nan)
            if name not in absent:
                array[index[inside]] = _physical_values(frame[name].to_numpy(), name, path)[inside]
        else:
            array = np.full(n_raw, -1, dtype=np.int8)
            if name not in absent:
                array[index[inside]] = _flag_values(frame[name].to_numpy(), column, path)[inside]
        values[name] = array
    return values


def _physical_values(array: np.ndarray, name: str, path: Path) -> np.ndarray:
    """A value column with the fill value as ``NaN``."""
    check_source_values_are_finite(array, name, path)
    check_source_fill_is_the_fill_value(array, name, path)
    return np.where(array == SOURCE.fill_value, np.nan, array)


def _flag_values(array: np.ndarray, column: SourceColumn, path: Path) -> np.ndarray:
    """A flag column as ``int8``, with the fill value as ``-1``."""
    real = array[array != SOURCE.fill_value]
    check_source_flags_are_whole(real, column, path)
    check_source_flags_are_in_the_vocabulary(real, column, path)
    return np.where(array == SOURCE.fill_value, -1, array).astype(np.int8)


def _nee_columns(estimate: str) -> list[SourceColumn]:
    """The four columns of one NEE estimate: value, flag and two uncertainties."""
    words = {
        "VUT_REF": "variable u* threshold for each year, reference selected by model efficiency",
        "VUT_USTAR50": "variable u* threshold for each year, 50th percentile of the u* threshold",
        "CUT_REF": "constant u* threshold across years, reference selected by model efficiency",
        "CUT_USTAR50": "constant u* threshold across years, 50th percentile of the u* threshold",
    }[estimate]
    return [
        SourceColumn(f"NEE_{estimate}", VALUE, "umolCO2 m-2 s-1", f"Net ecosystem exchange, {words}"),
        SourceColumn(
            f"NEE_{estimate}_QC",
            FLAG,
            "",
            f"Quality flag for NEE_{estimate}",
            flag_values=(0, 1, 2, 3),
            flag_meanings="measured good_quality_gap_fill medium_quality_gap_fill poor_quality_gap_fill",
        ),
        SourceColumn(
            f"NEE_{estimate}_RANDUNC",
            VALUE,
            "umolCO2 m-2 s-1",
            f"Random uncertainty of NEE_{estimate}, from measured data only",
        ),
        SourceColumn(
            f"NEE_{estimate}_JOINTUNC",
            VALUE,
            "umolCO2 m-2 s-1",
            f"Joint uncertainty of NEE_{estimate}: random uncertainty and u* filtering uncertainty",
        ),
    ]


# ── checks ────────────────────────────────────────────────────────────────────


def check_file_name_is_a_fullset_name(name: str) -> None:
    """A file name is a FULLSET half-hourly or hourly name."""
    if SOURCE.file_pattern.match(name) is None:
        raise ValueError(
            f"{name!r} is not a FULLSET half-hourly or hourly file name; pass an "
            "AMF_<tower>_FLUXNET_FULLSET_{HH,HR}_<first>-<last>_<version>.csv file."
        )


def check_source_root_is_a_directory(root: Path) -> None:
    """The source root is a directory."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"source root {root} is not a directory; link data/raw/net_ecosystem_exchange/fluxnet "
            "to the AmeriFlux download, or pass --source-root."
        )


def check_source_root_holds_files(paths: list[Path], root: Path) -> None:
    """The source root holds at least one FULLSET file."""
    if not paths:
        raise ValueError(f"{root} holds no FULLSET half-hourly or hourly files; point at the download.")


def check_source_files_are_one_per_tower(paths: list[Path]) -> None:
    """The source root holds one FULLSET file per tower."""
    towers: dict[str, str] = {}
    for path in paths:
        tower = parse_file_name(path.name)[0]
        if tower in towers:
            raise ValueError(
                f"two files for tower {tower}: {towers[tower]} and {path.name}; one file per "
                "tower is the contract, so decide which release to use first."
            )
        towers[tower] = path.name


def check_source_header_is_the_format(header: list[str], path: Path) -> None:
    """A file's header starts with the stamps and carries every kept column the format requires."""
    if header[:2] != list(_TIMESTAMP_COLUMNS):
        raise ValueError(
            f"{path.name}: the first two columns are {header[:2]}, not the stamps; the file is not "
            "a FULLSET file as downloaded."
        )
    missing = [name for name in SOURCE.names if name not in header]
    required_missing = [name for name in missing if name not in SOURCE.optional_columns]
    if required_missing:
        raise ValueError(
            f"{path.name}: missing required columns {truncated(required_missing)}; the file is not "
            "a FULLSET file as downloaded."
        )
    optional_missing = frozenset(missing)
    if optional_missing and optional_missing != SOURCE.optional_columns:
        raise ValueError(
            f"{path.name}: carries some constant-u*-threshold columns and not others, missing "
            f"{truncated(sorted(optional_missing))}; the group is present whole or not at all, "
            "so the file is damaged."
        )


def check_source_stamps_tile_the_declared_years(
    start: pd.DatetimeIndex,
    end: pd.DatetimeIndex,
    resolution: Resolution,
    first_year: int,
    last_year: int,
    path: Path,
) -> None:
    """A file's steps are one step long, contiguous, and tile the whole years its name declares."""
    if len(start) == 0:
        raise ValueError(f"{path.name}: no rows; the file is damaged.")
    if not ((end - start) == resolution.step).all():
        raise ValueError(
            f"{path.name}: TIMESTAMP_END - TIMESTAMP_START is not one {resolution.name} step in "
            "every row; the file is not at the resolution its name declares."
        )
    if len(start) > 1 and not (np.diff(start.values) == resolution.step.to_timedelta64()).all():
        raise ValueError(
            f"{path.name}: the steps are not contiguous; a stamp is repeated, skipped or out of "
            "order, which a zone-aware writer would cause, so the file is not as downloaded."
        )
    expected_first = pd.Timestamp(year=first_year, month=1, day=1)
    expected_end = pd.Timestamp(year=last_year + 1, month=1, day=1)
    if start[0] != expected_first or end[-1] != expected_end:
        raise ValueError(
            f"{path.name}: the record runs {start[0]} to {end[-1]}, not the whole years "
            f"{first_year}-{last_year} its name declares; the file is damaged."
        )


def check_source_values_are_finite(array: np.ndarray, name: str, path: Path) -> None:
    """A value column holds no non-finite value; missing is the fill value."""
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}: {name} holds a non-finite value; the source marks missing with {SOURCE.fill_value}.")


def check_source_fill_is_the_fill_value(array: np.ndarray, name: str, path: Path) -> None:
    """A value column holds no fill-like number other than the fill value."""
    fill_like = (array <= -9990) & (array != SOURCE.fill_value)
    if fill_like.any():
        raise ValueError(
            f"{path.name}: {name} holds {array[fill_like][0]!r}, a fill-like value other than "
            f"{SOURCE.fill_value}; a second fill convention would be read as data."
        )


def check_source_flags_are_whole(real: np.ndarray, column: SourceColumn, path: Path) -> None:
    """A flag column's reported values are whole numbers."""
    if real.size and (not np.isfinite(real).all() or (real != np.floor(real)).any()):
        raise ValueError(f"{path.name}: {column.name} holds a value that is not a whole number; a flag is an integer.")


def check_source_flags_are_in_the_vocabulary(real: np.ndarray, column: SourceColumn, path: Path) -> None:
    """A flag column's reported values are in its vocabulary."""
    outside = ~np.isin(real, column.flag_values)
    if outside.any():
        raise ValueError(
            f"{path.name}: {column.name} holds {truncated(sorted(set(real[outside].tolist())))}, "
            f"outside its vocabulary {list(column.flag_values)}; extend the format only on the "
            "producer's word."
        )


def check_source_values_are_present_where_flagged(
    values: Mapping[str, np.ndarray], absent: frozenset[str], path: Path
) -> None:
    """An NEE estimate has a value wherever its quality flag is set."""
    for name, column in SOURCE.columns.items():
        if not name.startswith("NEE_") or column.kind != VALUE or name in absent:
            continue
        flag = values.get(f"{name}_QC")
        if flag is None:
            continue
        orphaned = (flag >= 0) & np.isnan(values[name])
        if orphaned.any():
            raise ValueError(
                f"{path.name}: {name} is missing at {int(orphaned.sum())} steps where "
                f"{name}_QC is set; a flag without a value cannot be told from a fill."
            )


# ── the format ────────────────────────────────────────────────────────────────

# Last in the module, since building the columns runs _nee_columns above.

SOURCE = SourceFormat(
    file_pattern=re.compile(
        r"^AMF_(?P<tower>[A-Z]{2}-[A-Za-z0-9]{3})_FLUXNET_FULLSET_(?P<code>HH|HR)_"
        r"(?P<first_year>\d{4})-(?P<last_year>\d{4})_(?P<version>\d+-\d+)\.csv$"
    ),
    columns=frozendict(
        {
            column.name: column
            for column in (
                *_nee_columns("VUT_REF"),
                *_nee_columns("VUT_USTAR50"),
                *_nee_columns("CUT_REF"),
                *_nee_columns("CUT_USTAR50"),
                SourceColumn(
                    "NIGHT",
                    FLAG,
                    "",
                    "Nighttime flag, from SW_IN_POT",
                    flag_values=(0, 1),
                    flag_meanings="daytime nighttime",
                ),
                SourceColumn(
                    "SW_IN_POT",
                    VALUE,
                    "W m-2",
                    "Shortwave radiation, incoming, potential (top of atmosphere)",
                ),
                SourceColumn("SW_IN_F", VALUE, "W m-2", "Shortwave radiation, incoming, consolidated"),
                SourceColumn(
                    "SW_IN_F_QC",
                    FLAG,
                    "",
                    "Quality flag for SW_IN_F",
                    flag_values=(0, 1, 2),
                    flag_meanings="measured good_quality_gap_fill downscaled_from_era",
                ),
            )
        }
    ),
    optional_columns=frozenset(
        f"NEE_CUT_{estimate}{suffix}"
        for estimate in ("REF", "USTAR50")
        for suffix in ("", "_QC", "_RANDUNC", "_JOINTUNC")
    ),
    fill_value=-9999.0,
    documentation="https://fluxnet.org/data/fluxnet2015-dataset/fullset-data-product/",
)
