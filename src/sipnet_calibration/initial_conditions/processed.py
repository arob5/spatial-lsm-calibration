"""The processed product: the ensemble in the project's own names and shape.

``data/processed/initial_conditions.nc`` is what the rest of the project
reads. It is the raw file transposed onto ``(initial_condition_member, site)``,
its member dim and variables renamed to the processed names, placed on the
site table's pool with ``lon``/``lat``, and given the specs' fields as
attributes. The values are the source files', unchanged.

The package docstring gives the data model in full -- dimensions, variables,
coordinates, attributes and what ``NaN`` means.

Contents
--------
:func:`build_initial_conditions`
    Raw Dataset to product. Pure; ``scripts/ingest_initial_conditions.py``
    adds the checks and the write.
:func:`netcdf_encoding`
    How it is stored.
:func:`load_initial_conditions`
    Read it and check it against the specs.
:func:`initial_condition_fields`
    The product as one field per variable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.conventions import (
    BATCH_LABEL_DTYPE,
    CF_CONVENTIONS,
    LAT,
    LON,
    SITE,
    SITE_ID,
    SOURCE_INDEX,
    SOURCE_INDEX_ATTRIBUTES,
    SOURCE_MEMBER_ATTRIBUTES,
)
from sipnet_calibration.initial_conditions.names import (
    INITIAL_CONDITION_MEMBER,
    RAW_FILE,
    RAW_MEMBER,
    default_product_path,
)
from sipnet_calibration.initial_conditions.raw import (  # a shared package internal
    _check_presence_is_uniform_over_members,
)
from sipnet_calibration.initial_conditions.source_files import (
    NOMINAL_DATE,
    SOURCE,
    SOURCE_SCRIPT,
    SOURCE_SCRIPT_NOTE,
)
from sipnet_calibration.initial_conditions.specs import (
    INITIAL_CONDITION_NAMES,
    INITIAL_CONDITIONS,
    resolve_initial_condition,
)
from sipnet_calibration.io import utc_timestamp
from sipnet_calibration.sites import site_coordinates
from sipnet_calibration.validation import as_names, as_site_ids, truncated

__all__ = [
    "build_initial_conditions",
    "initial_condition_fields",
    "load_initial_conditions",
    "netcdf_encoding",
]


def build_initial_conditions(raw: xr.Dataset, sites: pd.DataFrame) -> xr.Dataset:
    """Turn the raw Dataset into the processed product the data model describes.

    Parameters
    ----------
    raw:
        As :func:`sipnet_calibration.initial_conditions.raw.read_raw` returns it.
    sites:
        The site table from :func:`sipnet_calibration.sites.load_sites`; its
        ``site_id`` is the pool and its ``lon``/``lat`` the coordinates.

    Returns
    -------
    xarray.Dataset
        The five specs' variables on ``(initial_condition_member, site)`` with
        their attributes, ``initial_condition_member`` 0-based with
        ``source_index`` beside it, ``site`` the pool with ``lon``/``lat``, and
        the dataset attributes of the data model.

    Raises
    ------
    ValueError
        If the raw file's sites are not exactly the site table's pool.

    Notes
    -----
    Pure, and structural only: the values are the raw file's, transposed. The
    raw file's ``member`` axis, the source files' 1-based index, becomes the
    0-based ``initial_condition_member``, and the source index is kept as
    ``source_index`` so a source file name can always be recovered.
    """
    pool = np.sort(sites[SITE_ID].to_numpy(np.int64))
    raw_sites = raw[SITE].values.astype(np.int64)
    if not np.array_equal(raw_sites, pool):
        extra = sorted(set(raw_sites.tolist()) - set(pool.tolist()))[:10]
        missing = sorted(set(pool.tolist()) - set(raw_sites.tolist()))[:10]
        raise ValueError(
            f"raw file sites are not the site table's pool: not in the table {extra}, "
            f"not in the file {missing}"
        )
    source_index = raw[RAW_MEMBER].values.astype(BATCH_LABEL_DTYPE)

    data_vars = {
        spec.name: (
            (INITIAL_CONDITION_MEMBER, SITE),
            np.ascontiguousarray(raw[spec.source_name].values.T),
            spec.xarray_attributes(),
        )
        for spec in INITIAL_CONDITIONS
    }
    coords = {
        INITIAL_CONDITION_MEMBER: (
            INITIAL_CONDITION_MEMBER,
            np.arange(source_index.size, dtype=BATCH_LABEL_DTYPE),
            dict(SOURCE_MEMBER_ATTRIBUTES),
        ),
        SOURCE_INDEX: (INITIAL_CONDITION_MEMBER, source_index, dict(SOURCE_INDEX_ATTRIBUTES)),
        **site_coordinates(pool.tolist(), sites),
    }
    return xr.Dataset(data_vars, coords=coords, attrs=_product_attributes(raw))


def netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """The on-disk encoding of the product: compressed, ``NaN`` as the fill,
    no ``_FillValue`` on any coordinate, as CF requires."""
    encoding: dict[str, dict[str, Any]] = {
        name: {"zlib": True, "complevel": 4, "_FillValue": np.nan}
        for name in INITIAL_CONDITION_NAMES
    }
    for name in dataset.coords:
        encoding[str(name)] = {"_FillValue": None}
    return encoding


def load_initial_conditions(path: Path | str | None = None) -> xr.Dataset:
    """Read the processed product and check it against the specs.

    Parameters
    ----------
    path:
        The netCDF to read. Defaults to
        :func:`sipnet_calibration.initial_conditions.names.default_product_path`.

    Returns
    -------
    xarray.Dataset
        The data model above.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with the command that produces it.
    ValueError
        If the file does not match the data model.
    """
    path = Path(path) if path is not None else default_product_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file. Produce it with:\n  python scripts/ingest_initial_conditions.py"
        )
    try:
        dataset = xr.open_dataset(path, engine="h5netcdf")
    except OSError as error:
        raise ValueError(f"{path}: not readable as netCDF-4/HDF5 ({error})") from error
    try:
        _check_product(dataset, path)
    except Exception:
        dataset.close()
        raise
    return dataset


def initial_condition_fields(
    names: Sequence[str] | None = None,
    *,
    sites: Iterable[int] | None = None,
    path: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The initial conditions as fields, one per variable.

    Parameters
    ----------
    names:
        Processed names, a sequence, in the order the result should carry
        them. Defaults to every one of :data:`INITIAL_CONDITION_NAMES`.
    sites:
        Site ids to keep, a sequence, in the order given, each once.
        Defaults to the whole pool.
    path:
        The product to read. Defaults to
        :func:`sipnet_calibration.initial_conditions.names.default_product_path`.

    Returns
    -------
    dict
        Name to its ``(initial_condition_member, site)`` array, with the
        variable's attributes and ``lon``/``lat`` on ``site``.

    Raises
    ------
    TypeError
        If *names* or *sites* is one value, a string or a set; if a name is
        not a string; or if a site id is a boolean, a float or not a number.
    ValueError
        If a site id is not from 1 to the largest ``int32``, is asked for
        twice, or *sites* is a two-dimensional array.
    KeyError
        If a name is not an initial condition, or a requested site is not in
        the product.
    """
    wanted_names = (
        INITIAL_CONDITION_NAMES if names is None else as_names(names, message_name="names")
    )
    for name in wanted_names:
        resolve_initial_condition(name)
    # A repeated site would make the site coordinate non-unique, and a table
    # built from it could not be addressed one cell at a time.
    wanted_sites = None if sites is None else list(as_site_ids(sites, message_name="sites"))

    dataset = load_initial_conditions(path)
    if wanted_sites is not None:
        check_product_holds_the_sites(dataset, wanted_sites)
        dataset = dataset.sel({SITE: wanted_sites})
    return {name: dataset[name] for name in wanted_names}



def _product_attributes(raw: xr.Dataset) -> dict[str, Any]:
    return {
        "Conventions": CF_CONVENTIONS,
        "title": "Initial condition ensemble for the 8000-site pool",
        "product": (
            "PEcAn pool initial conditions drawn for the North American reanalysis; one "
            "spec per variable in sipnet_calibration.initial_conditions"
        ),
        "source_file": RAW_FILE,
        "source_root": str(raw.attrs.get("source_root", "")),
        "source_script": SOURCE_SCRIPT,
        "source_script_note": SOURCE_SCRIPT_NOTE,
        "nominal_date": NOMINAL_DATE,
        "nominal_date_provenance": (
            "The sampling date in the PEcAn script; the source files themselves carry no "
            "date. Biomass is a 2010 annual map and soil carbon is undated."
        ),
        "source_time_units": SOURCE.time_units,
        "source_time_long_name": SOURCE.time_long_name,
        "source_time_value": SOURCE.time_value,
        "n_sites": int(raw.sizes[SITE]),
        "n_members": int(raw.sizes[RAW_MEMBER]),
        "history": (
            f"scripts/ingest_initial_conditions.py: read {RAW_FILE}, renamed the source "
            "variables to the spec names, renamed member to initial_condition_member and "
            "renumbered it from 1-based to 0-based, keeping the source index as "
            "source_index, placed the sites on the site table's pool with lon/lat, and "
            "wrote the spec fields as attributes; values unchanged"
        ),
        "created": utc_timestamp(),
    }


def _check_product(dataset: xr.Dataset, path: Path) -> None:
    """Raise unless *dataset* is the product the data model describes."""
    if set(dataset.data_vars) != set(INITIAL_CONDITION_NAMES):
        raise ValueError(
            f"{path}: variables are {sorted(dataset.data_vars)}, expected "
            f"{sorted(INITIAL_CONDITION_NAMES)}"
        )
    for spec in INITIAL_CONDITIONS:
        array = dataset[spec.name]
        if array.dims != (INITIAL_CONDITION_MEMBER, SITE):
            raise ValueError(
                f"{path}: {spec.name} has dims {array.dims}, expected "
                f"{(INITIAL_CONDITION_MEMBER, SITE)}; a product written before the "
                "member dim was renamed is re-made by scripts/ingest_initial_conditions.py"
            )
        if array.attrs.get("units") != spec.units:
            raise ValueError(
                f"{path}: {spec.name} has units {array.attrs.get('units')!r}, the spec says "
                f"{spec.units!r}"
            )
        if array.attrs.get("source_name") != spec.source_name:
            raise ValueError(
                f"{path}: {spec.name} was written from {array.attrs.get('source_name')!r}, "
                f"the spec says {spec.source_name!r}"
            )
        if array.attrs.get("long_name") != spec.long_label:
            raise ValueError(f"{path}: {spec.name} lacks the spec's long_name")
        if np.isinf(array.values).any():
            raise ValueError(f"{path}: {spec.name} holds an infinite value")
    for coordinate in (INITIAL_CONDITION_MEMBER, SOURCE_INDEX, SITE, LON, LAT):
        if coordinate not in dataset.coords:
            raise ValueError(f"{path}: missing the {coordinate!r} coordinate")
    for coordinate in (LON, LAT):
        if dataset[coordinate].dims != (SITE,):
            raise ValueError(f"{path}: {coordinate} must be on site, has dims {dataset[coordinate].dims}")
    if dataset[SOURCE_INDEX].dims != (INITIAL_CONDITION_MEMBER,):
        raise ValueError(f"{path}: {SOURCE_INDEX} must be on {INITIAL_CONDITION_MEMBER}")
    member = dataset[INITIAL_CONDITION_MEMBER].values
    if member.size == 0 or not np.array_equal(member, np.arange(member.size)):
        raise ValueError(f"{path}: {INITIAL_CONDITION_MEMBER} is not 0..n-1")
    source_index = dataset[SOURCE_INDEX].values
    if source_index.min() < 1 or np.any(np.diff(source_index) <= 0):
        raise ValueError(
            f"{path}: {SOURCE_INDEX} is not strictly ascending from 1 or more; a source "
            "file name could not be recovered from it"
        )
    site = dataset[SITE].values
    if site.size == 0 or np.any(np.diff(site) <= 0):
        raise ValueError(f"{path}: site is empty or not strictly ascending")
    lon, lat = dataset[LON].values, dataset[LAT].values
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError(f"{path}: lon or lat holds a non-finite value")
    if np.abs(lon).max() > 180 or np.abs(lat).max() > 90:
        raise ValueError(f"{path}: lon or lat is outside the geographic range; are they swapped?")
    _check_presence_is_uniform_over_members(
        {name: dataset[name].values.T for name in INITIAL_CONDITION_NAMES}, site
    )
    if dataset.attrs.get("Conventions") != CF_CONVENTIONS:
        raise ValueError(f"{path}: Conventions is {dataset.attrs.get('Conventions')!r}, expected {CF_CONVENTIONS!r}")


# ── checks ────────────────────────────────────────────────────────────────────


def check_product_holds_the_sites(dataset: xr.Dataset, site_ids: Sequence[int]) -> None:
    """The initial condition product holds every site asked of it."""
    held = set(dataset[SITE].values.tolist())
    missing = [site for site in site_ids if site not in held]
    if missing:
        raise KeyError(
            f"site(s) {truncated(missing)} are not in the initial condition product; ask "
            "only for sites of the site table it was built on."
        )
