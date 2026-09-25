"""The processed product: the ensemble in the project's own names and shape.

``data/processed/initial_conditions.nc`` is what the rest of the project
reads. It is the raw file transposed onto ``(member, site)``, renamed to the
specs' processed names, placed on the site table's pool with ``lon``/``lat``,
and given the specs' fields as attributes. The values are the source files',
unchanged.

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

from sipnet_calibration.conventions import CF_CONVENTIONS
from sipnet_calibration.initial_conditions.names import (
    MEMBER,
    RAW_FILE,
    SITE,
    SOURCE_MEMBER,
    _SITE_ATTRS,
    _utc_timestamp,
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
        The five specs' variables on ``(member, site)`` with their attributes,
        ``member`` 0-based with ``source_member`` beside it, ``site`` the pool
        with ``lon``/``lat``, and the dataset attributes of the data model.

    Raises
    ------
    ValueError
        If the raw file's sites are not exactly the site table's pool.

    Notes
    -----
    Pure, and structural only: the values are the raw file's, transposed. The
    member axis is renumbered from the source files' 1-based index to the
    project's 0-based one, and the source index is kept as a coordinate so a
    source file name can always be recovered.
    """
    pool = np.sort(sites["site_id"].to_numpy(np.int64))
    raw_sites = raw[SITE].values.astype(np.int64)
    if not np.array_equal(raw_sites, pool):
        extra = sorted(set(raw_sites.tolist()) - set(pool.tolist()))[:10]
        missing = sorted(set(pool.tolist()) - set(raw_sites.tolist()))[:10]
        raise ValueError(
            f"raw file sites are not the site table's pool: not in the table {extra}, "
            f"not in the file {missing}"
        )
    coordinates = sites.set_index("site_id").loc[pool, ["lon", "lat"]]
    source_member = raw[MEMBER].values.astype(np.int16)

    data_vars = {
        spec.name: (
            (MEMBER, SITE),
            np.ascontiguousarray(raw[spec.source_name].values.T),
            spec.xarray_attributes(),
        )
        for spec in INITIAL_CONDITIONS
    }
    coords = {
        MEMBER: (
            MEMBER,
            np.arange(source_member.size, dtype=np.int16),
            {
                "long_name": "Ensemble member",
                "comment": (
                    "0-based, meaningful only within this product; whether member i "
                    "corresponds to member i of another source is not established "
                    "(member_correspondence)."
                ),
            },
        ),
        SOURCE_MEMBER: (
            MEMBER,
            source_member,
            {
                "long_name": "Ensemble member index in the source file name",
                "comment": "The 1-based <member> of the source file name.",
            },
        ),
        SITE: (SITE, pool.astype(np.int32), _SITE_ATTRS),
        "lon": (SITE, coordinates["lon"].to_numpy(np.float64), _LON_ATTRS),
        "lat": (SITE, coordinates["lat"].to_numpy(np.float64), _LAT_ATTRS),
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
        Processed names, in the order the result should carry them, or one
        name on its own. Defaults to every one of
        :data:`INITIAL_CONDITION_NAMES`.
    sites:
        Site ids to keep, in the order given, or one id on its own. Each must
        be a whole number and appear once. Defaults to the whole pool.
    path:
        The product to read. Defaults to
        :func:`sipnet_calibration.initial_conditions.names.default_product_path`.

    Returns
    -------
    dict
        Name to its ``(member, site)`` array, with the variable's attributes
        and ``lon``/``lat`` on ``site``.

    Raises
    ------
    TypeError
        If *sites* is a string, or holds a value that is not a whole number.
    ValueError
        If a requested site is not in the pool, or is asked for twice.
    """
    if isinstance(names, str):
        names = [names]
    wanted_names = list(names) if names is not None else list(INITIAL_CONDITION_NAMES)
    for name in wanted_names:
        resolve_initial_condition(name)
    if isinstance(sites, (int, np.integer)):
        sites = [sites]
    elif isinstance(sites, str):
        raise TypeError(
            f"sites={sites!r} is a string, which would be read one character per site. "
            "Pass an integer or a sequence of integers."
        )
    wanted_sites = None if sites is None else [_site_id(site) for site in sites]
    if wanted_sites is not None and len(set(wanted_sites)) != len(wanted_sites):
        duplicates = sorted({site for site in wanted_sites if wanted_sites.count(site) > 1})
        raise ValueError(
            f"sites repeats {duplicates}. A repeated site makes the site coordinate "
            "non-unique, and a table built from it cannot be addressed one cell at a "
            "time."
        )

    dataset = load_initial_conditions(path)
    if wanted_sites is not None:
        missing = sorted(set(wanted_sites) - set(dataset[SITE].values.tolist()))
        if missing:
            raise ValueError(f"sites not in the pool: {missing[:10]}")
        dataset = dataset.sel({SITE: wanted_sites})
    return {name: dataset[name] for name in wanted_names}


_LON_ATTRS = {"standard_name": "longitude", "long_name": "Longitude", "units": "degrees_east"}

_LAT_ATTRS = {"standard_name": "latitude", "long_name": "Latitude", "units": "degrees_north"}


def _site_id(value: Any) -> int:
    """A site identifier as an int, refusing anything that is not already whole."""
    number = int(value)
    if number != value:
        raise TypeError(
            f"site {value!r} is not a whole number; int() would silently truncate it to "
            f"{number}, which is a different site."
        )
    return number


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
        "member_source": "ic",
        "member_correspondence": (
            "Not established: whether member i here corresponds to member i of the "
            "drivers or of any other ensemble is unknown, and xarray aligns integer "
            "member labels silently."
        ),
        "n_sites": int(raw.sizes[SITE]),
        "n_members": int(raw.sizes[MEMBER]),
        "history": (
            f"scripts/ingest_initial_conditions.py: read {RAW_FILE}, renamed the source "
            "variables to the spec names, renumbered member from 1-based to 0-based "
            "keeping the source index as source_member, placed the sites on the site "
            "table's pool with lon/lat, and wrote the spec fields as attributes; values "
            "unchanged"
        ),
        "created": _utc_timestamp(),
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
        if array.dims != (MEMBER, SITE):
            raise ValueError(f"{path}: {spec.name} has dims {array.dims}, expected ('member', 'site')")
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
    for coordinate in (MEMBER, SOURCE_MEMBER, SITE, "lon", "lat"):
        if coordinate not in dataset.coords:
            raise ValueError(f"{path}: missing the {coordinate!r} coordinate")
    for coordinate in ("lon", "lat"):
        if dataset[coordinate].dims != (SITE,):
            raise ValueError(f"{path}: {coordinate} must be on site, has dims {dataset[coordinate].dims}")
    if dataset[SOURCE_MEMBER].dims != (MEMBER,):
        raise ValueError(f"{path}: source_member must be on member")
    member = dataset[MEMBER].values
    if member.size == 0 or not np.array_equal(member, np.arange(member.size)):
        raise ValueError(f"{path}: member is not 0..n-1")
    source_member = dataset[SOURCE_MEMBER].values
    if source_member.min() < 1 or np.any(np.diff(source_member) <= 0):
        raise ValueError(
            f"{path}: source_member is not strictly ascending from 1 or more; a source "
            "file name could not be recovered from it"
        )
    site = dataset[SITE].values
    if site.size == 0 or np.any(np.diff(site) <= 0):
        raise ValueError(f"{path}: site is empty or not strictly ascending")
    lon, lat = dataset["lon"].values, dataset["lat"].values
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError(f"{path}: lon or lat holds a non-finite value")
    if np.abs(lon).max() > 180 or np.abs(lat).max() > 90:
        raise ValueError(f"{path}: lon or lat is outside the geographic range; are they swapped?")
    _check_presence_is_uniform_over_members(
        {name: dataset[name].values.T for name in INITIAL_CONDITION_NAMES}, site
    )
    if dataset.attrs.get("Conventions") != CF_CONVENTIONS:
        raise ValueError(f"{path}: Conventions is {dataset.attrs.get('Conventions')!r}, expected {CF_CONVENTIONS!r}")
