"""The converted raw files: every tower's FULLSET columns as one array per resolution.

``data/raw/net_ecosystem_exchange/ameriflux_nee_half_hourly.nc`` and
``ameriflux_nee_hourly.nc`` hold the kept columns of every tower's FULLSET file,
laid on one shared local-standard-time axis, in the source's own column names
and values. They are written once, on the SCC, by
``scripts/raw_sources/convert_ameriflux_nee.py``, and are not tracked: they are
copied to a checkout rather than committed.

:func:`build_raw` assembles one, :func:`raw_encoding` says how it is stored, and
:func:`read_raw` reads it back and checks it.

Data model
----------
**Dimensions**: ``tower``, ``time_index``.

**Data variables**: one per :data:`SOURCE` column, under the source name, on
``(tower, time_index)``: a value column as ``float64`` with ``NaN`` for missing,
a flag column as ``int8`` with ``-1`` for missing. Missing means the source
reported its fill value, the step lies outside the tower's file, or the file
does not carry the column at all (``absent_columns``). Each carries the source
column's ``units``, ``long_name``, ``source_fill_value`` and a ``comment`` and,
for a flag, CF ``flag_values`` and ``flag_meanings``.

**Coordinates**

=========================== ============== ==============================================
Name                        Dims           Meaning
=========================== ============== ==============================================
``tower``                   ``tower``      the AmeriFlux site identifier, ascending
``time_index``              ``time_index`` ``int32``, 0..n-1
``TIMESTAMP_START``         ``time_index`` ``int64`` ``YYYYMMDDHHMM``, local standard time
``TIMESTAMP_END``           ``time_index`` the same, for the step end
``source_file``             ``tower``      the FULLSET file the row came from
``site_version``            ``tower``      its version, from the file name
``source_first_year``,      ``tower``      ``int16``, the years the file covers
``source_last_year``
``absent_columns``          ``tower``      kept columns the file lacks, comma separated
``source_md5``              ``tower``      the file's md5
=========================== ============== ==============================================

The stamps are the same for every tower: the axis runs from
:data:`~sipnet_calibration.net_ecosystem_exchange.names.RAW_START` to
:data:`~sipnet_calibration.net_ecosystem_exchange.names.RAW_END` at the
resolution's step, in local standard time, with no conversion to UTC.

**Attributes**: ``title``, ``resolution``, ``resolution_minutes``,
``time_axis``, ``source_root``, ``source_layout``, ``source_documentation``,
``source_fill_value``, ``n_towers``, ``conversion_script``, ``history`` and
``converted``.

**Values are the source's.** Nothing is renamed, converted, shifted or
masked beyond the fill value becoming ``NaN`` or ``-1``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.io import utc_timestamp
from sipnet_calibration.net_ecosystem_exchange.names import (
    RAW_END,
    RAW_START,
    RESOLUTIONS,
    TIME_INDEX,
    TOWER,
    Resolution,
    raw_path,
    resolve_resolution,
)
from sipnet_calibration.net_ecosystem_exchange.source_files import (
    FLAG,
    SOURCE,
    SourceFile,
)
from sipnet_calibration.validation import truncated

__all__ = [
    "TOWER_COORDINATES",
    "build_raw",
    "raw_encoding",
    "read_raw",
]

#: The per-tower coordinates of a raw file.
TOWER_COORDINATES = (
    "source_file",
    "site_version",
    "source_first_year",
    "source_last_year",
    "absent_columns",
    "source_md5",
)


def build_raw(
    files: Iterable[SourceFile],
    resolution: Resolution,
    *,
    md5_by_file: Mapping[str, str],
    source_root: str,
    conversion_script: str,
) -> xr.Dataset:
    """Assemble parsed files of one resolution into the raw Dataset.

    Parameters
    ----------
    files:
        Every parsed file of the resolution, in any order.
    resolution:
        The resolution they share.
    md5_by_file:
        Each file's md5, by file name.
    source_root:
        The directory they were read from, for the attributes.
    conversion_script:
        The script doing the conversion, for the attributes.

    Returns
    -------
    xarray.Dataset
        As the data model above describes.

    Raises
    ------
    ValueError
        If no file is given, a file is at another resolution, two files are for
        one tower, or a file's md5 is not given.

    Notes
    -----
    Pure: it neither reads nor writes files, so the conversion and the tests
    call it on the same records.
    """
    records = sorted(files, key=lambda record: record.tower)
    check_source_files_to_assemble_are_given(records, resolution)
    check_source_files_are_at_the_resolution(records, resolution)
    check_source_files_have_an_md5(records, md5_by_file)
    check_source_files_are_one_per_tower_to_assemble(records)
    towers = [record.tower for record in records]

    starts = resolution.raw_step_starts()
    data_vars = {
        name: (
            (TOWER, TIME_INDEX),
            np.stack([record.values[name] for record in records]),
            _column_attributes(name),
        )
        for name in SOURCE.names
    }
    coords = {
        TOWER: (TOWER, np.array(towers, dtype=object), {"long_name": "AmeriFlux site identifier"}),
        TIME_INDEX: (
            TIME_INDEX,
            np.arange(len(starts), dtype=np.int32),
            {"long_name": "Position on the local-standard-time axis"},
        ),
        "TIMESTAMP_START": (
            TIME_INDEX,
            _stamps(starts),
            {"long_name": "Step start, local standard time, YYYYMMDDHHMM"},
        ),
        "TIMESTAMP_END": (
            TIME_INDEX,
            _stamps(starts + resolution.step),
            {"long_name": "Step end, local standard time, YYYYMMDDHHMM"},
        ),
        "source_file": (TOWER, np.array([r.file_name for r in records], dtype=object)),
        "site_version": (TOWER, np.array([r.version for r in records], dtype=object)),
        "source_first_year": (TOWER, np.array([r.first_year for r in records], dtype=np.int16)),
        "source_last_year": (TOWER, np.array([r.last_year for r in records], dtype=np.int16)),
        "absent_columns": (
            TOWER,
            np.array([",".join(r.absent_columns) for r in records], dtype=object),
        ),
        "source_md5": (TOWER, np.array([md5_by_file[r.file_name] for r in records], dtype=object)),
    }
    attrs = {
        "title": f"AmeriFlux FLUXNET FULLSET NEE columns, {resolution.name}, as one array",
        "resolution": resolution.name,
        "resolution_minutes": int(resolution.step / pd.Timedelta(minutes=1)),
        "time_axis": (
            "local standard time, no daylight saving (AmeriFlux FLUXNET convention); the same "
            f"steps for every tower, {RAW_START.isoformat()} to {RAW_END.isoformat()}"
        ),
        "source_root": source_root,
        "source_layout": SOURCE.file_pattern.pattern,
        "source_documentation": SOURCE.documentation,
        "source_fill_value": SOURCE.fill_value,
        "n_towers": len(records),
        "conversion_script": conversion_script,
        "history": (
            f"{conversion_script}: read every {resolution.source_code} FULLSET file under the "
            "source root, checked each against the source format, and laid the kept columns on "
            "one local-standard-time axis unchanged, in the source's column names"
        ),
        "converted": utc_timestamp(),
    }
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def raw_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """The on-disk encoding: compressed, one chunk per tower, no coordinate fill."""
    n_time = dataset.sizes[TIME_INDEX]
    encoding: dict[str, dict[str, Any]] = {}
    for name in SOURCE.names:
        fill = None if SOURCE.columns[name].kind == FLAG else np.nan
        encoding[name] = {"zlib": True, "complevel": 4, "_FillValue": fill, "chunksizes": (1, n_time)}
    for name in dataset.coords:
        encoding[str(name)] = {"_FillValue": None}
    return encoding


def read_raw(source: Resolution | str | Path) -> xr.Dataset:
    """Read a converted raw file and check it against the source format.

    Parameters
    ----------
    source:
        A resolution, or its name, to read the file at the default path; or the
        path of a raw file, as a ``Path`` or a string.

    Returns
    -------
    xarray.Dataset
        As :func:`build_raw` returns it, opened lazily.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with where it comes from.
    ValueError
        If the file does not match the data model.
    """
    if isinstance(source, Resolution):
        path = raw_path(source)
    elif isinstance(source, str) and source in RESOLUTIONS:
        path = raw_path(resolve_resolution(source))
    else:
        path = Path(source)
    check_raw_file_exists(path)
    dataset = _opened_netcdf(path)
    try:
        check_raw_file_is_valid(dataset, message_name=str(path))
    except Exception:
        dataset.close()
        raise
    return dataset


# ── private helpers ───────────────────────────────────────────────────────────


def _column_attributes(name: str) -> dict[str, Any]:
    """The attributes of one kept column, from the source format."""
    column = SOURCE.columns[name]
    attrs: dict[str, Any] = {"long_name": column.long_name, "source_fill_value": SOURCE.fill_value}
    if column.units:
        attrs["units"] = column.units
    if column.kind == FLAG:
        attrs["flag_values"] = np.array(column.flag_values, dtype=np.int8)
        attrs["flag_meanings"] = column.flag_meanings
        attrs["comment"] = "The source's flag unchanged; -1 where the source reports none."
    else:
        attrs["comment"] = "The source's value unchanged; NaN where the source reports none."
    return attrs


def _stamps(times: pd.DatetimeIndex) -> np.ndarray:
    """Timestamps as the source's ``YYYYMMDDHHMM`` integers."""
    return np.array(times.strftime("%Y%m%d%H%M").astype(np.int64), dtype=np.int64)


def _opened_netcdf(path: Path) -> xr.Dataset:
    """*path* opened lazily; a file that is not netCDF-4 is a ``ValueError``."""
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except OSError as error:
        raise ValueError(
            f"{path}: not readable as netCDF-4/HDF5 ({error}); copy the raw file again."
        ) from error


# ── checks ────────────────────────────────────────────────────────────────────

#: The advice a refusal of a raw file ends with.
_REMAKE = (
    "copy it from the SCC again, or remake it there with "
    "scripts/raw_sources/convert_ameriflux_nee.py"
)


def check_source_files_to_assemble_are_given(records: list[SourceFile], resolution: Resolution) -> None:
    """At least one parsed file is given to assemble."""
    if not records:
        raise ValueError(f"no {resolution.name} files to assemble; parse at least one first.")


def check_source_files_are_at_the_resolution(records: list[SourceFile], resolution: Resolution) -> None:
    """Every parsed file to assemble is at the resolution asked for."""
    for record in records:
        if record.resolution != resolution:
            raise ValueError(
                f"{record.file_name} is {record.resolution.name}, not {resolution.name}; assemble "
                "each resolution on its own."
            )


def check_source_files_have_an_md5(records: list[SourceFile], md5_by_file: Mapping[str, str]) -> None:
    """Every parsed file to assemble has its md5 given."""
    missing = [record.file_name for record in records if record.file_name not in md5_by_file]
    if missing:
        raise ValueError(f"no md5 given for {truncated(missing)}; hash every file read.")


def check_source_files_are_one_per_tower_to_assemble(records: list[SourceFile]) -> None:
    """No two parsed files to assemble are for one tower."""
    towers = [record.tower for record in records]
    repeated = sorted({tower for tower in towers if towers.count(tower) > 1})
    if repeated:
        raise ValueError(f"two files for tower(s) {truncated(repeated)}; pass one file per tower.")


def check_raw_file_exists(path: Path) -> None:
    """A raw file exists."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file; the raw files are not tracked, so copy them from the SCC, "
            "or make them there with scripts/raw_sources/convert_ameriflux_nee.py (see "
            "data/raw/net_ecosystem_exchange/provenance.md)."
        )


def check_raw_file_is_valid(dataset: xr.Dataset, *, message_name: str) -> None:
    """A raw file follows the data model :func:`build_raw` describes."""
    check_raw_file_has_the_source_columns(dataset, message_name=message_name)
    check_raw_file_names_its_resolution(dataset, message_name=message_name)
    check_raw_file_stamps_are_the_raw_axis(dataset, message_name=message_name)
    check_raw_file_towers_ascend(dataset, message_name=message_name)
    check_raw_file_has_the_tower_coordinates(dataset, message_name=message_name)
    check_raw_file_columns_have_their_dims_and_dtypes(dataset, message_name=message_name)
    check_raw_file_flags_are_in_their_vocabularies(dataset, message_name=message_name)


def check_raw_file_has_the_source_columns(dataset: xr.Dataset, *, message_name: str) -> None:
    """A raw file's variables are exactly the source format's kept columns."""
    if set(dataset.data_vars) != set(SOURCE.names):
        raise ValueError(
            f"{message_name}: variables are {truncated(sorted(dataset.data_vars))}, expected "
            f"{truncated(sorted(SOURCE.names))}; {_REMAKE}."
        )


def check_raw_file_names_its_resolution(dataset: xr.Dataset, *, message_name: str) -> None:
    """A raw file names a known resolution, and its step agrees."""
    name = dataset.attrs.get("resolution", "")
    if name not in RESOLUTIONS:
        raise ValueError(f"{message_name}: resolution {name!r} is not one of {truncated(RESOLUTIONS)}; {_REMAKE}.")
    minutes = int(RESOLUTIONS[name].step / pd.Timedelta(minutes=1))
    if dataset.attrs.get("resolution_minutes") != minutes:
        raise ValueError(f"{message_name}: resolution_minutes does not match the resolution; {_REMAKE}.")


def check_raw_file_stamps_are_the_raw_axis(dataset: xr.Dataset, *, message_name: str) -> None:
    """A raw file's stamps and ``time_index`` are its resolution's raw axis."""
    resolution = resolve_resolution(dataset.attrs["resolution"])
    expected = _stamps(resolution.raw_step_starts())
    if not np.array_equal(dataset["TIMESTAMP_START"].values, expected):
        raise ValueError(f"{message_name}: TIMESTAMP_START is not the {resolution.name} raw axis; {_REMAKE}.")
    if not np.array_equal(dataset[TIME_INDEX].values, np.arange(expected.size)):
        raise ValueError(f"{message_name}: time_index is not 0..n-1; {_REMAKE}.")


def check_raw_file_towers_ascend(dataset: xr.Dataset, *, message_name: str) -> None:
    """A raw file holds towers, each once, ascending."""
    towers = [str(tower) for tower in dataset[TOWER].values]
    if not towers or towers != sorted(set(towers)):
        raise ValueError(f"{message_name}: tower is empty, repeated or not ascending; {_REMAKE}.")


def check_raw_file_has_the_tower_coordinates(dataset: xr.Dataset, *, message_name: str) -> None:
    """A raw file carries every per-tower coordinate, on ``tower``."""
    for name in TOWER_COORDINATES:
        if name not in dataset.coords or dataset[name].dims != (TOWER,):
            raise ValueError(f"{message_name}: missing the per-tower coordinate {name!r}; {_REMAKE}.")


def check_raw_file_columns_have_their_dims_and_dtypes(dataset: xr.Dataset, *, message_name: str) -> None:
    """Every column of a raw file is on ``(tower, time_index)``, a flag ``int8`` and a value ``float64``."""
    for name, column in SOURCE.columns.items():
        array = dataset[name]
        if array.dims != (TOWER, TIME_INDEX):
            raise ValueError(f"{message_name}: {name} has dims {array.dims}, expected (tower, time_index); {_REMAKE}.")
        expected = np.int8 if column.kind == FLAG else np.float64
        if array.dtype != expected:
            raise ValueError(f"{message_name}: {name} is {array.dtype}, expected {np.dtype(expected)}; {_REMAKE}.")


def check_raw_file_flags_are_in_their_vocabularies(dataset: xr.Dataset, *, message_name: str) -> None:
    """Every flag of a raw file is ``-1`` or in its vocabulary."""
    for name, column in SOURCE.columns.items():
        if column.kind != FLAG:
            continue
        outside = sorted(set(np.unique(dataset[name].values).tolist()) - {-1, *column.flag_values})
        if outside:
            raise ValueError(f"{message_name}: {name} holds {truncated(outside)}, outside its vocabulary; {_REMAKE}.")
