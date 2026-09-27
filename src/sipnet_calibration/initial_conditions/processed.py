"""The processed file: the ensemble in the project's own names and shape.

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
    Raw Dataset to processed Dataset. Pure; ``scripts/ingest_initial_conditions.py``
    adds the checks and the write.
:func:`netcdf_encoding`
    How it is stored.
:func:`load_initial_conditions`
    Read it and check it against the specs.
:func:`initial_condition_fields`
    The processed file as one field per variable.
The checks
    One invariant each, grouped as
    :func:`check_processed_initial_conditions_are_valid`, which
    :func:`load_initial_conditions` holds the file to.
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
    INITIAL_CONDITION_MEMBER,
    LAT,
    LON,
    SITE,
    SITE_ID,
    SOURCE_INDEX,
    SOURCE_INDEX_ATTRIBUTES,
)
from sipnet_calibration.fields import batch_coordinate
from sipnet_calibration.initial_conditions.names import (
    RAW_FILE,
    RAW_MEMBER,
    default_processed_path,
)
from sipnet_calibration.initial_conditions.raw import check_presence_is_uniform_over_members
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
from sipnet_calibration.sites import check_sites_are_the_site_table, site_coordinates
from sipnet_calibration.validation import as_names, as_site_ids, truncated

__all__ = [
    "build_initial_conditions",
    "check_processed_initial_conditions_are_valid",
    "check_processed_initial_conditions_declare_the_conventions",
    "check_processed_initial_conditions_exist",
    "check_processed_initial_conditions_have_the_coordinates",
    "check_processed_initial_conditions_hold_the_sites",
    "check_processed_locations_are_geographic",
    "check_processed_locations_are_on_site",
    "check_processed_member_count_is_recorded",
    "check_processed_member_is_its_source_index",
    "check_processed_members_count_from_zero",
    "check_processed_sites_ascend",
    "check_processed_source_index_ascends_from_one",
    "check_processed_source_index_is_on_the_member",
    "check_processed_variable_has_no_infinite_value",
    "check_processed_variable_has_the_spec_long_name",
    "check_processed_variable_has_the_spec_units",
    "check_processed_variable_is_on_member_and_site",
    "check_processed_variable_was_written_from_the_source",
    "check_processed_variables_are_the_specs",
    "initial_condition_fields",
    "load_initial_conditions",
    "netcdf_encoding",
]


def build_initial_conditions(raw: xr.Dataset, site_table: pd.DataFrame) -> xr.Dataset:
    """Turn the raw Dataset into the processed Dataset the data model describes.

    Parameters
    ----------
    raw:
        As :func:`sipnet_calibration.initial_conditions.raw.read_raw` returns it.
    site_table:
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
    KeyError
        If the raw file has a site the site table lacks.
    ValueError
        If the raw file lacks a site of the site table.

    Notes
    -----
    Pure, and structural only: the values are the raw file's, transposed. The
    raw file's ``member`` axis, the source files' 1-based index, becomes the
    0-based ``initial_condition_member``, and the source index is kept as
    ``source_index`` so a source file name can always be recovered.
    """
    pool = np.sort(site_table[SITE_ID].to_numpy(np.int64))
    check_sites_are_the_site_table(
        site_table, raw[SITE].values.tolist(), message_name="the raw file's site(s)"
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
        # A member's label is its identity, source_index - 1, as the drivers'
        # is; the ingest checks the source indices run 1..n, so it is 0..n-1.
        INITIAL_CONDITION_MEMBER: batch_coordinate(INITIAL_CONDITION_MEMBER, source_index - 1),
        SOURCE_INDEX: (INITIAL_CONDITION_MEMBER, source_index, dict(SOURCE_INDEX_ATTRIBUTES)),
        **site_coordinates(pool.tolist(), site_table),
    }
    return xr.Dataset(data_vars, coords=coords, attrs=_dataset_attributes(raw))


def netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """The on-disk encoding of the processed file: compressed, ``NaN`` as the fill,
    no ``_FillValue`` on any coordinate, as CF requires."""
    encoding: dict[str, dict[str, Any]] = {
        name: {"zlib": True, "complevel": 4, "_FillValue": np.nan}
        for name in INITIAL_CONDITION_NAMES
    }
    for name in dataset.coords:
        encoding[str(name)] = {"_FillValue": None}
    return encoding


def load_initial_conditions(path: Path | str | None = None) -> xr.Dataset:
    """Read the processed file and check it against the specs.

    Parameters
    ----------
    path:
        The netCDF to read. Defaults to
        :func:`sipnet_calibration.initial_conditions.names.default_processed_path`.

    Returns
    -------
    xarray.Dataset
        The data model above.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with the command that produces it.
    ValueError
        If the file is not netCDF-4, or does not follow the data model
        (:func:`check_processed_initial_conditions_are_valid`).
    """
    path = Path(path) if path is not None else default_processed_path()
    check_processed_initial_conditions_exist(path)
    dataset = _opened_netcdf4(path)
    try:
        check_processed_initial_conditions_are_valid(dataset, message_name=str(path))
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
        The processed file to read. Defaults to
        :func:`sipnet_calibration.initial_conditions.names.default_processed_path`.

    Returns
    -------
    dict
        Name to its ``(initial_condition_member, site)`` array, with the
        variable's attributes and ``lon``/``lat`` on ``site``.

    Raises
    ------
    TypeError, ValueError
        If *names* or *sites* is refused by
        :func:`~sipnet_calibration.validation.as_names` or
        :func:`~sipnet_calibration.validation.as_site_ids`.
    KeyError
        If a name is not an initial condition, or a requested site is not in
        the processed file.
    """
    wanted_names = (
        INITIAL_CONDITION_NAMES if names is None else as_names(names, message_name="names")
    )
    for name in wanted_names:
        resolve_initial_condition(name)
    # A repeated site would make the site coordinate non-unique, and a table
    # built from it could not be addressed one row at a time.
    wanted_sites = None if sites is None else list(as_site_ids(sites, message_name="sites"))

    dataset = load_initial_conditions(path)
    if wanted_sites is not None:
        check_processed_initial_conditions_hold_the_sites(dataset, wanted_sites)
        dataset = dataset.sel({SITE: wanted_sites})
    return {name: dataset[name] for name in wanted_names}

# ── private helpers ───────────────────────────────────────────────────────────


def _opened_netcdf4(path: Path) -> xr.Dataset:
    """*path* opened lazily through ``h5netcdf``; one it cannot read is a ``ValueError``."""
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except OSError as error:
        raise ValueError(
            f"{path}: not readable as netCDF-4/HDF5 ({error}); re-make it with "
            "scripts/ingest_initial_conditions.py."
        ) from error


def _dataset_attributes(raw: xr.Dataset) -> dict[str, Any]:
    """The processed file's dataset attributes, from the raw file's."""
    return {
        "Conventions": CF_CONVENTIONS,
        "title": "Initial condition ensemble for the 8000-site pool",
        "upstream_product": (
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
        "n_initial_condition_members": int(raw.sizes[RAW_MEMBER]),
        "history": (
            f"scripts/ingest_initial_conditions.py: read {RAW_FILE}, renamed the source "
            "variables to the spec names, renamed member to initial_condition_member and "
            "renumbered it from 1-based to 0-based, keeping the source index as "
            "source_index, placed the sites on the site table's pool with lon/lat, and "
            "wrote the spec fields as attributes; values unchanged"
        ),
        "created": utc_timestamp(),
    }


# ── checks ────────────────────────────────────────────────────────────────────

#: The advice every refusal of a processed file ends with.
_REMAKE = "re-make it with scripts/ingest_initial_conditions.py"


def check_processed_initial_conditions_are_valid(dataset: xr.Dataset, *, message_name: str) -> None:
    """A processed initial conditions file follows the data model."""
    check_processed_variables_are_the_specs(dataset, message_name=message_name)
    for spec in INITIAL_CONDITIONS:
        subject = f"{message_name}: {spec.name}"
        array = dataset[spec.name]
        check_processed_variable_is_on_member_and_site(array, message_name=subject)
        check_processed_variable_has_the_spec_units(array, spec.units, message_name=subject)
        check_processed_variable_was_written_from_the_source(
            array, spec.source_name, message_name=subject
        )
        check_processed_variable_has_the_spec_long_name(array, spec.long_label, message_name=subject)
        check_processed_variable_has_no_infinite_value(array, message_name=subject)
    check_processed_initial_conditions_have_the_coordinates(dataset, message_name=message_name)
    check_processed_locations_are_on_site(dataset, message_name=message_name)
    check_processed_source_index_is_on_the_member(dataset, message_name=message_name)
    check_processed_members_count_from_zero(dataset, message_name=message_name)
    check_processed_source_index_ascends_from_one(dataset, message_name=message_name)
    check_processed_member_is_its_source_index(dataset, message_name=message_name)
    check_processed_member_count_is_recorded(dataset, message_name=message_name)
    check_processed_sites_ascend(dataset, message_name=message_name)
    check_processed_locations_are_geographic(dataset, message_name=message_name)
    check_presence_is_uniform_over_members(
        {name: dataset[name].values.T for name in INITIAL_CONDITION_NAMES}, dataset[SITE].values
    )
    check_processed_initial_conditions_declare_the_conventions(dataset, message_name=message_name)


def check_processed_initial_conditions_exist(path: Path) -> None:
    """The processed initial conditions file exists."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file; produce it with "
            "`python scripts/ingest_initial_conditions.py`."
        )


def check_processed_variables_are_the_specs(dataset: xr.Dataset, *, message_name: str) -> None:
    """A processed file holds exactly the specs' variables."""
    if set(dataset.data_vars) != set(INITIAL_CONDITION_NAMES):
        raise ValueError(
            f"{message_name}: variables are {sorted(dataset.data_vars)}, expected "
            f"{sorted(INITIAL_CONDITION_NAMES)}; {_REMAKE}."
        )


def check_processed_variable_is_on_member_and_site(
    array: xr.DataArray, *, message_name: str
) -> None:
    """A processed variable is on ``(initial_condition_member, site)``."""
    if array.dims != (INITIAL_CONDITION_MEMBER, SITE):
        raise ValueError(
            f"{message_name} has dims {array.dims}, expected "
            f"{(INITIAL_CONDITION_MEMBER, SITE)}; a processed file written before the member "
            f"dim was renamed is out of date, so {_REMAKE}."
        )


def check_processed_variable_has_the_spec_units(
    array: xr.DataArray, units: str, *, message_name: str
) -> None:
    """A processed variable is in its spec's units."""
    if array.attrs.get("units") != units:
        raise ValueError(
            f"{message_name} has units {array.attrs.get('units')!r}, the spec says {units!r}; "
            f"{_REMAKE}."
        )


def check_processed_variable_was_written_from_the_source(
    array: xr.DataArray, source_name: str, *, message_name: str
) -> None:
    """A processed variable was written from its spec's source variable."""
    if array.attrs.get("source_name") != source_name:
        raise ValueError(
            f"{message_name} was written from {array.attrs.get('source_name')!r}, the spec says "
            f"{source_name!r}; {_REMAKE}."
        )


def check_processed_variable_has_the_spec_long_name(
    array: xr.DataArray, long_label: str, *, message_name: str
) -> None:
    """A processed variable's ``long_name`` is its spec's long label."""
    if array.attrs.get("long_name") != long_label:
        raise ValueError(f"{message_name} lacks the spec's long_name; {_REMAKE}.")


def check_processed_variable_has_no_infinite_value(
    array: xr.DataArray, *, message_name: str
) -> None:
    """A processed variable holds no infinite value."""
    if np.isinf(array.values).any():
        raise ValueError(f"{message_name} holds an infinite value; {_REMAKE}.")


def check_processed_initial_conditions_have_the_coordinates(
    dataset: xr.Dataset, *, message_name: str
) -> None:
    """A processed file carries the member, source index, site and location coordinates."""
    for coordinate in (INITIAL_CONDITION_MEMBER, SOURCE_INDEX, SITE, LON, LAT):
        if coordinate not in dataset.coords:
            raise ValueError(
                f"{message_name}: missing the {coordinate!r} coordinate; {_REMAKE}."
            )


def check_processed_locations_are_on_site(dataset: xr.Dataset, *, message_name: str) -> None:
    """A processed file's ``lon`` and ``lat`` are on ``site``."""
    for coordinate in (LON, LAT):
        if dataset[coordinate].dims != (SITE,):
            raise ValueError(
                f"{message_name}: {coordinate} must be on site, has dims "
                f"{dataset[coordinate].dims}; {_REMAKE}."
            )


def check_processed_source_index_is_on_the_member(
    dataset: xr.Dataset, *, message_name: str
) -> None:
    """A processed file's ``source_index`` is on ``initial_condition_member``."""
    if dataset[SOURCE_INDEX].dims != (INITIAL_CONDITION_MEMBER,):
        raise ValueError(
            f"{message_name}: {SOURCE_INDEX} must be on {INITIAL_CONDITION_MEMBER}; {_REMAKE}."
        )


def check_processed_members_count_from_zero(dataset: xr.Dataset, *, message_name: str) -> None:
    """A processed file's members are ``0`` to ``n - 1``."""
    member = dataset[INITIAL_CONDITION_MEMBER].values
    if member.size == 0 or not np.array_equal(member, np.arange(member.size)):
        raise ValueError(
            f"{message_name}: {INITIAL_CONDITION_MEMBER} is not 0..n-1; {_REMAKE}."
        )


def check_processed_source_index_ascends_from_one(
    dataset: xr.Dataset, *, message_name: str
) -> None:
    """A processed file's source indices are strictly ascending from 1 or more."""
    source_index = dataset[SOURCE_INDEX].values
    if source_index.min() < 1 or np.any(np.diff(source_index) <= 0):
        raise ValueError(
            f"{message_name}: {SOURCE_INDEX} is not strictly ascending from 1 or more, so a "
            f"source file name could not be recovered from it; {_REMAKE}."
        )


def check_processed_member_is_its_source_index(dataset: xr.Dataset, *, message_name: str) -> None:
    """A processed file's member label is its source index less one, the member's identity."""
    member = dataset[INITIAL_CONDITION_MEMBER].values
    if not np.array_equal(member, dataset[SOURCE_INDEX].values - 1):
        raise ValueError(
            f"{message_name}: {INITIAL_CONDITION_MEMBER} is not {SOURCE_INDEX} - 1, the "
            f"member's identity; {_REMAKE}."
        )


def check_processed_member_count_is_recorded(dataset: xr.Dataset, *, message_name: str) -> None:
    """A processed file's ``n_initial_condition_members`` attribute counts its members."""
    n_members = dataset.sizes[INITIAL_CONDITION_MEMBER]
    recorded = dataset.attrs.get("n_initial_condition_members")
    if recorded != n_members:
        raise ValueError(
            f"{message_name}: attribute n_initial_condition_members is {recorded!r}, not the "
            f"{n_members} members; a processed file written before the attribute was renamed "
            f"from n_members is out of date, so {_REMAKE}."
        )


def check_processed_sites_ascend(dataset: xr.Dataset, *, message_name: str) -> None:
    """A processed file's ``site`` is non-empty and strictly ascending."""
    site = dataset[SITE].values
    if site.size == 0 or np.any(np.diff(site) <= 0):
        raise ValueError(f"{message_name}: site is empty or not strictly ascending; {_REMAKE}.")


def check_processed_locations_are_geographic(dataset: xr.Dataset, *, message_name: str) -> None:
    """A processed file's ``lon`` and ``lat`` are finite and in the geographic range."""
    lon, lat = dataset[LON].values, dataset[LAT].values
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError(f"{message_name}: lon or lat holds a non-finite value; {_REMAKE}.")
    if np.abs(lon).max() > 180 or np.abs(lat).max() > 90:
        raise ValueError(
            f"{message_name}: lon or lat is outside the geographic range, as if swapped; "
            f"{_REMAKE}."
        )


def check_processed_initial_conditions_declare_the_conventions(
    dataset: xr.Dataset, *, message_name: str
) -> None:
    """A processed file declares the package's CF conventions."""
    declared = dataset.attrs.get("Conventions")
    if declared != CF_CONVENTIONS:
        raise ValueError(
            f"{message_name}: Conventions is {declared!r}, expected {CF_CONVENTIONS!r}; "
            f"{_REMAKE}."
        )


def check_processed_initial_conditions_hold_the_sites(
    dataset: xr.Dataset, site_ids: Sequence[int]
) -> None:
    """The initial conditions' processed file holds every site asked of it."""
    held = set(dataset[SITE].values.tolist())
    missing = [site for site in site_ids if site not in held]
    if missing:
        raise KeyError(
            f"site(s) {truncated(missing)} are not in the initial conditions' processed file; ask "
            "only for sites of the site table it was built on."
        )
