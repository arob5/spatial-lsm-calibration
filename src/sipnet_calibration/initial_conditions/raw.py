"""The tracked raw file: the 800,000 source files as one array.

``data/raw/initial_conditions/pecan_pool_initial_conditions.nc`` holds every
source file's values on ``(site, member)``, in the source files' own variable
names, units strings and 1-based member index. It is written once, on the SCC,
by ``scripts/raw_sources/convert_initial_conditions.py``, and is tracked.

:func:`build_raw` assembles it, :func:`raw_encoding` says how it is stored,
and :func:`read_raw` reads it back and checks it against the same rules,
grouped as :func:`check_raw_initial_conditions_are_valid`.
``scripts/ingest_initial_conditions.py`` takes it from here to
:mod:`sipnet_calibration.initial_conditions.processed`.

Data model
----------
**Dimensions**: ``site``, ``member`` -- in that order, the transpose of the
processed file's.

**Data variables**: one per :data:`SOURCE` variable, under the *source* names,
all ``float64`` on ``(site, member)``, ``NaN`` where none of a site's files
carries the variable. Each carries the source file's own ``units`` and
``long_name`` strings verbatim, plus ``source_fill_value`` and a ``comment``.

**Coordinates**: ``site`` ``int32`` ascending, the identifiers the source
directories are named for; ``member`` ``int16`` ascending, the source files'
**1-based** index, which the processed file renumbers.

**Attributes**: ``title``, ``source_root``, ``source_layout``,
``source_format``, ``source_fill_value``, the ``source_time_*`` triple,
``n_source_files``, ``n_sites``, ``n_members``, ``conversion_script``,
``history`` and ``converted``.

**Values are the source files', bit for bit.** Nothing is renamed, converted
or masked here, which is what keeps the file checkable against the originals.

Notes
-----
The writer and the reader are both here so that the schema is stated once and
the two cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from sipnet_calibration.conventions import SITE, SITE_ATTRIBUTES, SITE_DTYPE
from sipnet_calibration.initial_conditions.names import (
    RAW_MEMBER,
    raw_path,
)
from sipnet_calibration.initial_conditions.source_files import (
    SOURCE,
    SourceFile,
    check_source_variable_is_known,
)
from sipnet_calibration.io import utc_timestamp
from sipnet_calibration.validation import (
    check_integers_are_in_range,
    check_site_ids_are_in_range,
    truncated,
)

__all__ = [
    "build_raw",
    "check_presence_is_uniform_over_members",
    "raw_encoding",
    "read_raw",
]


def build_raw(
    files: Iterable[SourceFile],
    *,
    source_root: str,
    conversion_script: str,
) -> xr.Dataset:
    """Assemble parsed source files into the raw Dataset the conversion writes.

    Parameters
    ----------
    files:
        Every parsed file of the tree, in any order.
    source_root:
        The directory they were read from, for the attributes.
    conversion_script:
        The script doing the conversion, for the attributes.

    Returns
    -------
    xarray.Dataset
        One ``float64`` variable per ``SOURCE.names`` entry on
        ``(site, member)``, in source names with the source ``units`` and
        ``long_name`` strings, ``NaN`` where a file lacks the variable;
        ``site`` ``int32`` and ``member`` ``int16`` (the 1-based source index),
        both ascending; the source's time metadata and the counts as
        attributes.

    Raises
    ------
    ValueError
        If the files do not form the complete, consistent ensemble the data
        model describes, by any of this module's checks.

    Notes
    -----
    Pure: it neither reads nor writes files, so the conversion script and the
    tests call it on the same records. The guards here are the ones that a
    fancy-indexed assignment would otherwise turn into a silent overwrite.
    """
    records = list(files)
    check_source_files_are_given(records)
    # Checked before they are made int64, which would truncate 1.5 to 1 and
    # refuse NaN with NumPy's message rather than the module's.
    check_site_ids_are_in_range(
        np.array([record.site for record in records], dtype=object),
        message_name="the source files' sites",
    )
    check_integers_are_in_range(
        np.array([record.member for record in records], dtype=object),
        minimum=1,
        maximum=int(np.iinfo(np.int16).max),
        message_name="the source files' members",
    )
    sites = np.array(sorted({record.site for record in records}), dtype=np.int64)
    members = np.array(sorted({record.member for record in records}), dtype=np.int64)
    site_index = {int(site): i for i, site in enumerate(sites)}
    member_index = {int(member): j for j, member in enumerate(members)}

    shape = (sites.size, members.size)
    seen = np.zeros(shape, dtype=bool)
    arrays = {name: np.full(shape, np.nan) for name in SOURCE.names}
    for record in records:
        i, j = site_index[record.site], member_index[record.member]
        check_source_file_is_the_first_for_its_pair(seen[i, j], record)
        seen[i, j] = True
        for name, value in record.values.items():
            check_source_variable_is_known(
                name, message_name=f"site {record.site} member {record.member}"
            )
            arrays[name][i, j] = value
    check_every_site_has_every_member(seen, sites, members)
    check_presence_is_uniform_over_members(arrays, sites, message_name="the source files")

    dataset = xr.Dataset(
        {
            name: (
                (SITE, RAW_MEMBER),
                arrays[name],
                {
                    "units": SOURCE.variables[name].units,
                    "long_name": SOURCE.variables[name].long_name,
                    "source_fill_value": SOURCE.fill_value,
                    "comment": (
                        "The source file's value, bit for bit; NaN where none of the site's "
                        "files carries the variable."
                    ),
                },
            )
            for name in SOURCE.names
        },
        coords={
            SITE: (SITE, sites.astype(SITE_DTYPE), SITE_ATTRIBUTES),
            RAW_MEMBER: (
                RAW_MEMBER,
                members.astype(np.int16),
                {
                    "long_name": "Ensemble member index in the source file name",
                    "comment": "The 1-based <member> of the source file name.",
                },
            ),
        },
        attrs={
            "title": "PEcAn pool initial conditions for the 8000-site pool, as one array",
            "source_root": source_root,
            "source_layout": SOURCE.file_template,
            "source_format": "NETCDF3_CLASSIC, one file per (site, member)",
            "source_fill_value": SOURCE.fill_value,
            "source_time_units": SOURCE.time_units,
            "source_time_long_name": SOURCE.time_long_name,
            "source_time_value": SOURCE.time_value,
            "n_source_files": len(records),
            "n_sites": int(sites.size),
            "n_members": int(members.size),
            "conversion_script": conversion_script,
            "history": (
                f"{conversion_script}: read every {SOURCE.file_template} under the source "
                "root, checked each against the source template, and laid the values on "
                "(site, member) unchanged, in the source files' names and units strings"
            ),
            "converted": utc_timestamp(),
        },
    )
    return dataset


def raw_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """The on-disk encoding of the raw file: compressed, ``NaN`` as the fill."""
    encoding: dict[str, dict[str, Any]] = {
        name: {"zlib": True, "complevel": 4, "_FillValue": np.nan} for name in SOURCE.names
    }
    for name in dataset.coords:
        encoding[str(name)] = {"_FillValue": None}
    return encoding


def read_raw(path: Path | str | None = None) -> xr.Dataset:
    """Read the converted raw file and check it against the specs.

    Parameters
    ----------
    path:
        The netCDF to read. Defaults to
        :func:`sipnet_calibration.initial_conditions.names.raw_path`.

    Returns
    -------
    xarray.Dataset
        As :func:`build_raw` returns it.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with where it comes from.
    ValueError
        If the file is not netCDF-4, or does not follow the data model
        (:func:`check_raw_initial_conditions_are_valid`).
    """
    path = Path(path) if path is not None else raw_path()
    check_raw_initial_conditions_exist(path)
    return _open_checked_netcdf4(
        path, check_raw_initial_conditions_are_valid, remedy="restore it from version control"
    )


# ── private helpers ───────────────────────────────────────────────────────────


def _open_checked_netcdf4(
    path: Path, check: Callable[..., None], *, remedy: str
) -> xr.Dataset:
    """*path* opened lazily by ``h5netcdf`` and held to *check*, closed if it fails.

    A file ``h5netcdf`` cannot read is a ``ValueError`` ending in *remedy*.
    Shared with :mod:`sipnet_calibration.initial_conditions.processed`, which
    reads its file the same way.
    """
    try:
        dataset = xr.open_dataset(path, engine="h5netcdf")
    except OSError as error:
        raise ValueError(f"{path}: not readable as netCDF-4/HDF5 ({error}); {remedy}.") from error
    try:
        check(dataset, message_name=str(path))
    except Exception:
        dataset.close()
        raise
    return dataset


# ── checks ────────────────────────────────────────────────────────────────────


def check_raw_initial_conditions_are_valid(dataset: xr.Dataset, *, message_name: str) -> None:
    """A raw initial conditions file is the one :func:`build_raw` describes."""
    check_raw_variables_are_the_source_variables(dataset, message_name=message_name)
    for name in SOURCE.names:
        subject = f"{message_name}: {name}"
        check_raw_variable_is_on_site_and_member(dataset[name], message_name=subject)
        check_raw_variable_is_float64(dataset[name], message_name=subject)
        check_raw_variable_carries_the_source_strings(name, dataset[name], message_name=subject)
        check_raw_variable_has_no_infinite_value(dataset[name], message_name=subject)
    for coordinate, dtype in ((SITE, SITE_DTYPE), (RAW_MEMBER, np.int16)):
        values = dataset[coordinate].values
        subject = f"{message_name}: {coordinate}"
        check_raw_coordinate_ascends(values, message_name=subject)
        check_raw_coordinate_is_integer(values, message_name=subject)
        # The processed file narrows these with astype, which wraps silently.
        check_integers_are_in_range(
            values.astype(np.int64),
            minimum=1,
            maximum=int(np.iinfo(dtype).max),
            message_name=subject,
        )
    check_presence_is_uniform_over_members(
        {name: dataset[name].values for name in SOURCE.names},
        dataset[SITE].values,
        message_name=message_name,
    )
    check_raw_initial_conditions_carry_the_source_attributes(dataset, message_name=message_name)


def check_raw_initial_conditions_exist(path: Path) -> None:
    """The raw initial conditions file exists."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file; the raw file is tracked in version control, and if it is "
            "missing from a checkout, regenerate it on the SCC with "
            "scripts/raw_sources/convert_initial_conditions.py (see "
            "data/raw/initial_conditions/provenance.md)."
        )


def check_source_files_are_given(records: list[SourceFile]) -> None:
    """At least one parsed source file is given to assemble."""
    if not records:
        raise ValueError("no source files to assemble; pass the parsed files of the tree.")


def check_source_file_is_the_first_for_its_pair(seen: bool, record: SourceFile) -> None:
    """No two source files claim the same ``(site, member)``."""
    if seen:
        raise ValueError(
            f"two files for site {record.site} member {record.member}; each pair has one file, "
            "so drop the copy."
        )


def check_every_site_has_every_member(
    seen: np.ndarray, sites: np.ndarray, members: np.ndarray
) -> None:
    """Every site has a source file for every member."""
    if seen.all():
        return
    gaps = [f"site {sites[i]} member {members[j]}" for i, j in np.argwhere(~seen)]
    raise ValueError(
        f"{len(gaps)} (site, member) pairs have no file, {truncated(gaps)}; the ensemble is "
        "a complete rectangle, so a gap means the tree is incomplete."
    )


def check_presence_is_uniform_over_members(
    arrays: Mapping[str, np.ndarray], sites: np.ndarray, *, message_name: str
) -> None:
    """Each variable is present for every member of a site, or for none."""
    # Shared with the processed file's checks: it is what gives NaN its one
    # meaning there, so it has to be the same rule.
    for name, array in arrays.items():
        present = np.isfinite(array)
        mixed = present.any(axis=1) & ~present.all(axis=1)
        if mixed.any():
            raise ValueError(
                f"{message_name}: {name}: present for some members and absent for others at "
                f"{int(mixed.sum())} sites, {truncated(sites[mixed].tolist())}; presence is a "
                "property of the site in this ensemble, so the source tree is inconsistent."
            )


def check_raw_variables_are_the_source_variables(
    dataset: xr.Dataset, *, message_name: str
) -> None:
    """A raw file holds exactly the source format's variables."""
    if set(dataset.data_vars) != set(SOURCE.names):
        raise ValueError(
            f"{message_name}: variables are {truncated(sorted(dataset.data_vars))}, expected "
            f"{truncated(sorted(SOURCE.names))}; restore the file from version control."
        )


def check_raw_variable_is_on_site_and_member(array: xr.DataArray, *, message_name: str) -> None:
    """A raw variable is on ``(site, member)``."""
    if array.dims != (SITE, RAW_MEMBER):
        raise ValueError(
            f"{message_name} has dims {array.dims}, expected {(SITE, RAW_MEMBER)}; restore the "
            "file from version control."
        )


def check_raw_variable_is_float64(array: xr.DataArray, *, message_name: str) -> None:
    """A raw variable is ``float64``."""
    if array.dtype != np.float64:
        raise ValueError(
            f"{message_name} is {array.dtype}, expected float64; restore the file from "
            "version control."
        )


def check_raw_variable_carries_the_source_strings(
    name: str, array: xr.DataArray, *, message_name: str
) -> None:
    """A raw variable's ``units`` and ``long_name`` are the source files' strings."""
    variable = SOURCE.variables[name]
    for key, expected in (("units", variable.units), ("long_name", variable.long_name)):
        if array.attrs.get(key) != expected:
            raise ValueError(
                f"{message_name} has {key} {array.attrs.get(key)!r}, the source string is "
                f"{expected!r}; the raw file keeps the source's strings, so restore it."
            )


def check_raw_variable_has_no_infinite_value(array: xr.DataArray, *, message_name: str) -> None:
    """A raw variable holds no infinite value."""
    if np.isinf(array.values).any():
        raise ValueError(
            f"{message_name} holds an infinite value; no source file does, so restore the "
            "file from version control."
        )


def check_raw_coordinate_ascends(values: np.ndarray, *, message_name: str) -> None:
    """A raw coordinate is non-empty and strictly ascending."""
    if values.size == 0 or np.any(np.diff(values) <= 0):
        raise ValueError(
            f"{message_name} is empty or not strictly ascending; restore the file from "
            "version control."
        )


def check_raw_coordinate_is_integer(values: np.ndarray, *, message_name: str) -> None:
    """A raw coordinate has an integer dtype."""
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(
            f"{message_name} is {values.dtype}, expected an integer type; restore the file "
            "from version control."
        )


def check_raw_initial_conditions_carry_the_source_attributes(
    dataset: xr.Dataset, *, message_name: str
) -> None:
    """A raw file carries the source time metadata and its file count."""
    for key in ("source_time_units", "source_time_long_name", "source_time_value", "n_source_files"):
        if key not in dataset.attrs:
            raise ValueError(
                f"{message_name}: missing the {key!r} attribute; restore the file from "
                "version control."
            )
