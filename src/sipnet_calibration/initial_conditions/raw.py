"""The tracked raw file: the 800,000 source files as one array.

``data/raw/initial_conditions/pecan_pool_initial_conditions.nc`` holds every
source file's values on ``(site, member)``, in the source files' own variable
names, units strings and 1-based member index. It is written once, on the SCC,
by ``scripts/raw_sources/convert_initial_conditions.py``, and is small enough
to live in version control -- which is the only form in which the ensemble
exists off the SCC.

The writer and the reader are both here so that the file's schema is stated
once: :func:`build_raw` assembles it, :func:`raw_encoding` says how it is
stored, and :func:`read_raw` reads it back and checks it against the same
rules. ``scripts/ingest_initial_conditions.py`` takes it from here to
:mod:`sipnet_calibration.initial_conditions.processed`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from sipnet_calibration.initial_conditions.names import (
    MEMBER,
    SITE,
    _SITE_ATTRS,
    _utc_timestamp,
    raw_path,
)
from sipnet_calibration.initial_conditions.source_files import (
    SOURCE,
    SourceFile,
)

__all__ = [
    "build_raw",
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
        If two files claim the same ``(site, member)``, if the member set
        differs between sites, if a variable's presence differs between the
        members of one site, or if no file was given.

    Notes
    -----
    Pure: it neither reads nor writes files, so the conversion script and the
    tests call it on the same records. The guards here are the ones that a
    fancy-indexed assignment would otherwise turn into a silent overwrite.
    """
    records = list(files)
    if not records:
        raise ValueError("no source files to assemble")
    sites = np.array(sorted({record.site for record in records}), dtype=np.int64)
    members = np.array(sorted({record.member for record in records}), dtype=np.int64)
    site_index = {int(site): i for i, site in enumerate(sites)}
    member_index = {int(member): j for j, member in enumerate(members)}

    shape = (sites.size, members.size)
    seen = np.zeros(shape, dtype=bool)
    arrays = {name: np.full(shape, np.nan) for name in SOURCE.names}
    for record in records:
        i, j = site_index[record.site], member_index[record.member]
        if seen[i, j]:
            raise ValueError(f"two files for site {record.site} member {record.member}")
        seen[i, j] = True
        for name, value in record.values.items():
            if name not in arrays:
                raise ValueError(
                    f"site {record.site} member {record.member}: variable {name!r} is not "
                    f"one of {list(SOURCE.names)}"
                )
            arrays[name][i, j] = value
    _check_every_site_has_every_member(seen, sites, members)
    check_presence_is_uniform_over_members(arrays, sites)

    _check_site_ids_fit_dtype(sites, np.int32, "site")
    _check_site_ids_fit_dtype(members, np.int16, "member")
    dataset = xr.Dataset(
        {
            name: (
                (SITE, MEMBER),
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
            SITE: (SITE, sites.astype(np.int32), _SITE_ATTRS),
            MEMBER: (
                MEMBER,
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
            "converted": _utc_timestamp(),
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
        The netCDF to read. Defaults to :func:`raw_path`.

    Returns
    -------
    xarray.Dataset
        As :func:`build_raw` returns it.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with where it comes from.
    ValueError
        If the variables are not exactly ``SOURCE.names`` on
        ``(site, member)`` with the source attribute strings, the coordinates
        are not ascending, a value is non-finite without being ``NaN``, or a
        variable is present for some members of a site and not others.
    """
    path = Path(path) if path is not None else raw_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file. The raw file is tracked in version control; if it "
            "is missing from a checkout, regenerate it on the SCC with "
            "scripts/raw_sources/convert_initial_conditions.py (see "
            "data/raw/initial_conditions/provenance.md)."
        )
    dataset = xr.open_dataset(path, engine="h5netcdf")
    try:
        _check_raw(dataset, path)
    except Exception:
        dataset.close()
        raise
    return dataset


def _check_every_site_has_every_member(seen: np.ndarray, sites: np.ndarray, members: np.ndarray) -> None:
    if seen.all():
        return
    gaps = np.argwhere(~seen)
    examples = [f"site {sites[i]} member {members[j]}" for i, j in gaps[:5]]
    raise ValueError(
        f"{gaps.shape[0]} (site, member) pairs have no file, for example {examples}. The "
        "ensemble is a complete rectangle; a gap means the tree is incomplete."
    )


def check_presence_is_uniform_over_members(arrays: Mapping[str, np.ndarray], sites: np.ndarray) -> None:
    """Raise unless each variable is present for every member of a site or none.

    Not private: :mod:`sipnet_calibration.initial_conditions.processed` asserts
    the same invariant on the product, and it has to be the same rule, since it
    is what gives ``NaN`` its one meaning.
    """
    for name, array in arrays.items():
        present = np.isfinite(array)
        mixed = present.any(axis=1) & ~present.all(axis=1)
        if mixed.any():
            raise ValueError(
                f"{name}: present for some members and absent for others at "
                f"{int(mixed.sum())} sites, for example {sites[mixed][:5].tolist()}. "
                "Presence is a property of the site in this ensemble."
            )


def _check_site_ids_fit_dtype(values: np.ndarray, dtype: type, what: str) -> None:
    info = np.iinfo(dtype)
    if values.min() < max(info.min, 1) or values.max() > info.max:
        raise ValueError(f"{what} values {values.min()}-{values.max()} do not fit {dtype.__name__}")


def _check_raw(dataset: xr.Dataset, path: Path) -> None:
    """Raise unless *dataset* is the raw file :func:`build_raw` describes."""
    if set(dataset.data_vars) != set(SOURCE.names):
        raise ValueError(
            f"{path}: variables are {sorted(dataset.data_vars)}, expected {sorted(SOURCE.names)}"
        )
    for name in SOURCE.names:
        array = dataset[name]
        if array.dims != (SITE, MEMBER):
            raise ValueError(f"{path}: {name} has dims {array.dims}, expected ('site', 'member')")
        if array.dtype != np.float64:
            raise ValueError(f"{path}: {name} is {array.dtype}, expected float64")
        variable = SOURCE.variables[name]
        for key, expected in (("units", variable.units), ("long_name", variable.long_name)):
            if array.attrs.get(key) != expected:
                raise ValueError(
                    f"{path}: {name} has {key} {array.attrs.get(key)!r}, the source string is "
                    f"{expected!r}"
                )
        values = array.values
        if np.isinf(values).any():
            raise ValueError(f"{path}: {name} holds an infinite value")
    for coordinate, dtype in ((SITE, np.int32), (MEMBER, np.int16)):
        values = dataset[coordinate].values
        if values.size == 0 or np.any(np.diff(values) <= 0):
            raise ValueError(f"{path}: {coordinate} is empty or not strictly ascending")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{path}: {coordinate} is {values.dtype}, expected an integer type")
        # The product narrows these with astype, which wraps silently.
        _check_site_ids_fit_dtype(values.astype(np.int64), dtype, f"{path}: {coordinate}")
    check_presence_is_uniform_over_members(
        {name: dataset[name].values for name in SOURCE.names}, dataset[SITE].values
    )
    for key in ("source_time_units", "source_time_long_name", "source_time_value", "n_source_files"):
        if key not in dataset.attrs:
            raise ValueError(f"{path}: missing the {key!r} attribute")
