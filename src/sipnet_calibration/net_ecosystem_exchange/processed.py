"""The processed files: one series per file, on the site pool and a UTC axis.

``data/processed/net_ecosystem_exchange/<name>.nc`` holds one series: a tower
series (:mod:`sipnet_calibration.net_ecosystem_exchange.sources`) placed on the
sites of the primary towers, with the site table's coordinates, CF ``time``
and ``time_bounds``, and the spec's fields as attributes. The values are the
source's, unchanged.

The package docstring gives the data model in full.

Contents
--------
:func:`default_net_ecosystem_exchange_directory`, :func:`net_ecosystem_exchange_path`
    Where the processed files are.
:func:`build_net_ecosystem_exchange`
    Tower series to processed Dataset. Pure;
    ``scripts/ingest_net_ecosystem_exchange.py`` adds the checks and the write.
:func:`netcdf_encoding`
    How it is stored.
:func:`load_net_ecosystem_exchange`
    Read one processed file and check it against its spec.
:func:`net_ecosystem_exchange_fields`, :func:`net_ecosystem_exchange_quality_flags`,
:func:`net_ecosystem_exchange_random_uncertainties`, :func:`net_ecosystem_exchange_joint_uncertainties`
    One variable of several series, one field each.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.conventions import (
    BOUNDS,
    CF_CONVENTIONS,
    LAT,
    LON,
    SITE,
    SITE_ID,
    TIME,
    TIME_BOUNDS,
    data_root,
)
from sipnet_calibration.fields import window_coordinates
from sipnet_calibration.io import utc_timestamp
from sipnet_calibration.net_ecosystem_exchange.names import TOWER, resolve_resolution
from sipnet_calibration.net_ecosystem_exchange.source_files import SOURCE
from sipnet_calibration.net_ecosystem_exchange.specs import (
    NET_ECOSYSTEM_EXCHANGE_NAMES,
    NetEcosystemExchangeSpec,
    resolve_net_ecosystem_exchange,
)
from sipnet_calibration.sites import (
    check_processed_file_has_the_coordinates,
    check_processed_file_holds_sites,
    check_processed_file_holds_the_sites,
    check_processed_file_locations_are_on_site,
    check_processed_file_sites_ascend,
    check_site_table_lists_the_sites,
    site_coordinates,
)
from sipnet_calibration.validation import as_names, as_site_ids, truncated

__all__ = [
    "JOINT_UNCERTAINTY",
    "NIGHT",
    "QUALITY_FLAG",
    "RANDOM_UNCERTAINTY",
    "TOWER_SITE_COORDINATES",
    "VALUE",
    "build_net_ecosystem_exchange",
    "default_net_ecosystem_exchange_directory",
    "load_net_ecosystem_exchange",
    "net_ecosystem_exchange_fields",
    "net_ecosystem_exchange_joint_uncertainties",
    "net_ecosystem_exchange_path",
    "net_ecosystem_exchange_quality_flags",
    "net_ecosystem_exchange_random_uncertainties",
    "netcdf_encoding",
]

#: Name of the observed-values array in the processed file.
VALUE = "value"

#: Name of its quality flag.
QUALITY_FLAG = "quality_flag"

#: Name of its random uncertainty.
RANDOM_UNCERTAINTY = "random_uncertainty"

#: Name of its joint uncertainty.
JOINT_UNCERTAINTY = "joint_uncertainty"

#: Name of ONEFlux's nighttime flag.
NIGHT = "night"

#: The coordinates on ``site`` that describe its tower, besides ``site``,
#: ``lon`` and ``lat``.
TOWER_SITE_COORDINATES = (
    "ameriflux_site_id",
    "tower_lon",
    "tower_lat",
    "match_basis",
    "utc_offset",
    "doi",
    "site_version",
)

#: On-disk time encoding. Written explicitly so nothing is inherited from a default.
TIME_UNITS = "minutes since 2012-01-01 00:00:00"
CALENDAR = "proleptic_gregorian"


def default_net_ecosystem_exchange_directory() -> Path:
    """Where the processed files are: ``data/processed/net_ecosystem_exchange/``.

    ``$SIPNET_CALIBRATION_DATA`` replaces ``data/`` when set.
    """
    return data_root() / "processed" / "net_ecosystem_exchange"


def net_ecosystem_exchange_path(
    net_ecosystem_exchange: str | NetEcosystemExchangeSpec, directory: Path | str | None = None
) -> Path:
    """The processed file of a series: ``<directory>/<name>.nc``."""
    name = (
        net_ecosystem_exchange
        if isinstance(net_ecosystem_exchange, str)
        else net_ecosystem_exchange.name
    )
    base = Path(directory) if directory is not None else default_net_ecosystem_exchange_directory()
    return base / f"{name}.nc"


def build_net_ecosystem_exchange(
    spec: NetEcosystemExchangeSpec,
    tower_series: xr.Dataset,
    tower_table: pd.DataFrame,
    site_table: pd.DataFrame,
) -> xr.Dataset:
    """Turn a tower series into the processed Dataset the data model describes.

    Parameters
    ----------
    spec:
        The series.
    tower_series:
        As a reader of :data:`~sipnet_calibration.net_ecosystem_exchange.sources.SOURCE_READERS`
        returns it.
    tower_table:
        As :func:`~sipnet_calibration.net_ecosystem_exchange.towers.read_tower_table`
        returns it; each tower's site and metadata.
    site_table:
        As :func:`sipnet_calibration.sites.load_sites` returns it; the ``lon``
        and ``lat`` of each site.

    Returns
    -------
    xarray.Dataset
        The processed Dataset, sites ascending.

    Raises
    ------
    KeyError
        If a tower of the series is not in the tower table, or its site is not
        in the site table.
    ValueError
        If a tower of the series is not primary, or two share a site.

    Notes
    -----
    Pure and source-agnostic: it reads nothing but its arguments.
    """
    towers = [str(tower) for tower in tower_series[TOWER].values]
    rows = tower_table.set_index("tower")
    check_towers_are_in_the_tower_table(towers, rows)
    check_towers_are_primary(towers, rows)
    site_ids = rows.loc[towers, SITE_ID].astype(np.int64).to_numpy()
    check_towers_have_distinct_sites(site_ids)
    check_site_table_lists_the_sites(site_table, site_ids.tolist(), message_name=f"{spec.name}: site(s)")

    order = np.argsort(site_ids, kind="stable")
    series = tower_series.isel({TOWER: order}).drop_vars("utc_offset")
    sites = site_ids[order].tolist()
    ordered_towers = [towers[i] for i in order]
    series = series.rename({TOWER: SITE}).drop_vars(SITE, errors="ignore")

    data_vars = {
        str(variable): ((SITE, TIME), series[variable].transpose(SITE, TIME).values, _variable_attributes(spec, str(variable)))
        for variable in series.data_vars
    }
    coords: dict[str, Any] = {
        **site_coordinates(sites, site_table),
        **_tower_coordinates(rows.loc[ordered_towers]),
        **_time_coordinates(pd.DatetimeIndex(series[TIME].values), resolve_resolution(spec.resolution).step),
    }
    return xr.Dataset(data_vars, coords=coords, attrs=_dataset_attributes(spec, tower_table, ordered_towers))


def netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """The on-disk encoding: compressed, one chunk per site, integer time, no
    coordinate fill, as CF requires."""
    encoding: dict[str, dict[str, Any]] = {}
    for name, array in dataset.data_vars.items():
        fill = np.nan if array.dtype.kind == "f" else None
        encoding[str(name)] = {
            "zlib": True,
            "complevel": 4,
            "_FillValue": fill,
            "chunksizes": (1, array.sizes[TIME]),
        }
    for name in dataset.coords:
        encoding[str(name)] = {"_FillValue": None}
    for name in (TIME, TIME_BOUNDS):
        encoding[name].update({"units": TIME_UNITS, "calendar": CALENDAR, "dtype": "int32"})
    return encoding


def load_net_ecosystem_exchange(
    net_ecosystem_exchange: str | NetEcosystemExchangeSpec, path: Path | str | None = None
) -> xr.Dataset:
    """Read one processed file and check it against its spec.

    Parameters
    ----------
    net_ecosystem_exchange:
        A name in
        :data:`~sipnet_calibration.net_ecosystem_exchange.specs.NET_ECOSYSTEM_EXCHANGE_NAMES`,
        or a spec.
    path:
        The netCDF to read. Defaults to
        :func:`net_ecosystem_exchange_path`.

    Returns
    -------
    xarray.Dataset
        The data model, opened lazily.

    Raises
    ------
    KeyError
        If no series has that name.
    FileNotFoundError
        If the file is absent, with the command that produces it.
    ValueError
        If the file does not match the data model or the spec.
    """
    spec = (
        net_ecosystem_exchange
        if isinstance(net_ecosystem_exchange, NetEcosystemExchangeSpec)
        else resolve_net_ecosystem_exchange(net_ecosystem_exchange)
    )
    path = Path(path) if path is not None else net_ecosystem_exchange_path(spec)
    check_processed_net_ecosystem_exchange_exists(path, spec)
    dataset = _opened_netcdf(path)
    try:
        check_processed_net_ecosystem_exchange_is_valid(dataset, spec, message_name=str(path))
    except Exception:
        dataset.close()
        raise
    return dataset


def net_ecosystem_exchange_fields(
    names: Sequence[str] | None = None,
    *,
    sites: Iterable[int] | None = None,
    directory: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The observed values of several series, one field each.

    Parameters
    ----------
    names:
        Series names, a sequence, in the order the result should carry them.
        Defaults to every series in
        :data:`~sipnet_calibration.net_ecosystem_exchange.specs.NET_ECOSYSTEM_EXCHANGE_NAMES`.
    sites:
        Site ids to keep, a sequence, in the order given, each once; each must
        be in every series asked for. Defaults to all of each series' sites.
    directory:
        Where the processed files are. Defaults to
        :func:`default_net_ecosystem_exchange_directory`.

    Returns
    -------
    dict
        Series name to its ``value`` array on ``(site, time)``, renamed to the
        series, with the spec's attributes, ``lon``/``lat`` and the tower's
        coordinates on ``site``, and each step's window, read from the CF
        ``time_bounds``, as ``window_start`` and ``window_end`` on ``time``
        (:data:`~sipnet_calibration.conventions.WINDOW_START`,
        :data:`~sipnet_calibration.conventions.WINDOW_END`). A dict because
        the half-hourly and hourly series do not share a time axis.

    Raises
    ------
    TypeError
        If *names* or *sites* is not a sequence of names or of site ids.
    ValueError
        If a site id is not one, or is asked for twice.
    KeyError
        If a name is not a series, or a requested site is not in its
        processed file.
    """
    return _fields_of_variable(VALUE, names, sites, directory)


def net_ecosystem_exchange_quality_flags(
    names: Sequence[str] | None = None,
    *,
    sites: Iterable[int] | None = None,
    directory: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The quality flags, as :func:`net_ecosystem_exchange_fields` gives values."""
    return _fields_of_variable(QUALITY_FLAG, names, sites, directory)


def net_ecosystem_exchange_random_uncertainties(
    names: Sequence[str] | None = None,
    *,
    sites: Iterable[int] | None = None,
    directory: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The random uncertainties, as :func:`net_ecosystem_exchange_fields` gives values."""
    return _fields_of_variable(RANDOM_UNCERTAINTY, names, sites, directory)


def net_ecosystem_exchange_joint_uncertainties(
    names: Sequence[str] | None = None,
    *,
    sites: Iterable[int] | None = None,
    directory: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The joint uncertainties, as :func:`net_ecosystem_exchange_fields` gives values."""
    return _fields_of_variable(JOINT_UNCERTAINTY, names, sites, directory)


# ── private helpers ───────────────────────────────────────────────────────────

#: The advice a refusal of a processed file ends with.
_REMAKE = "re-make it with scripts/ingest_net_ecosystem_exchange.py"

#: Every variable a processed file may hold.
_VARIABLE_NAMES = (VALUE, QUALITY_FLAG, RANDOM_UNCERTAINTY, JOINT_UNCERTAINTY, NIGHT)


def _fields_of_variable(
    variable_name: str,
    names: Sequence[str] | None,
    sites: Iterable[int] | None,
    directory: Path | str | None,
) -> dict[str, xr.DataArray]:
    """One series' *variable_name* array per name, renamed to the series."""
    names = NET_ECOSYSTEM_EXCHANGE_NAMES if names is None else as_names(names, message_name="names")
    wanted = None if sites is None else list(as_site_ids(sites, message_name="sites"))
    fields: dict[str, xr.DataArray] = {}
    for name in names:
        spec = resolve_net_ecosystem_exchange(name)
        dataset = load_net_ecosystem_exchange(spec, net_ecosystem_exchange_path(spec, directory))
        check_processed_net_ecosystem_exchange_holds_the_variable(
            dataset, variable_name, message_name=f"the processed file of {name}"
        )
        field = dataset[variable_name].rename(name).assign_coords(window_coordinates(dataset[TIME_BOUNDS]))
        if wanted is not None:
            check_processed_file_holds_the_sites(dataset, wanted, message_name=f"the processed file of {name}")
            field = field.sel({SITE: wanted})
        fields[name] = field
    return fields


def _opened_netcdf(path: Path) -> xr.Dataset:
    """*path* opened lazily; a file that is not netCDF-4 is a ``ValueError``."""
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except OSError as error:
        raise ValueError(f"{path}: not readable as netCDF-4/HDF5 ({error}); {_REMAKE}.") from error


def _tower_coordinates(rows: pd.DataFrame) -> dict[str, tuple]:
    """The coordinates on ``site`` that describe each site's tower, in *rows*' order."""
    return {
        "ameriflux_site_id": (
            SITE,
            rows.index.to_numpy(object),
            {"long_name": "AmeriFlux site identifier of the tower"},
        ),
        "tower_lon": (
            SITE,
            rows["tower_lon"].to_numpy(np.float64),
            {"long_name": "Tower longitude, from AmeriFlux", "units": "degrees_east"},
        ),
        "tower_lat": (
            SITE,
            rows["tower_lat"].to_numpy(np.float64),
            {"long_name": "Tower latitude, from AmeriFlux", "units": "degrees_north"},
        ),
        "match_basis": (
            SITE,
            rows["match_basis"].to_numpy(object),
            {"long_name": "How the tower was matched to the site"},
        ),
        "utc_offset": (
            SITE,
            rows["utc_offset_hours"].to_numpy(np.float64),
            {
                "long_name": "Tower's local standard time minus UTC",
                "units": "hours",
                "comment": "Recovered from the tower's SW_IN_POT; local standard time is time + utc_offset.",
            },
        ),
        "doi": (SITE, rows["doi"].to_numpy(object), {"long_name": "AmeriFlux FLUXNET DOI of the site's dataset"}),
        "site_version": (SITE, rows["site_version"].to_numpy(object), {"long_name": "FULLSET version"}),
    }


def _time_coordinates(ends: pd.DatetimeIndex, step: pd.Timedelta) -> dict[str, tuple]:
    """CF ``time`` at each step's end and ``time_bounds`` around it."""
    ends = ends.as_unit("ns")
    starts = ends - step
    return {
        TIME: (
            TIME,
            ends.to_numpy(),
            {
                "standard_name": "time",
                "axis": "T",
                "long_name": "End of the averaging interval",
                "bounds": TIME_BOUNDS,
                "time_zone": "UTC",
                "comment": (
                    "The end of each averaging interval, UTC; the source stamps are local "
                    "standard time, shifted per site by utc_offset."
                ),
            },
        ),
        TIME_BOUNDS: (
            (TIME, BOUNDS),
            np.stack([starts.to_numpy(), ends.to_numpy()], axis=1),
            {
                "long_name": "Averaging interval",
                "comment": "The interval (start, time] each value is the mean rate over, in the CF bounds form.",
            },
        ),
    }


def _variable_attributes(spec: NetEcosystemExchangeSpec, variable_name: str) -> dict[str, Any]:
    """The attributes of one processed variable, from the spec."""
    if variable_name == VALUE:
        return spec.xarray_attributes()
    if variable_name == QUALITY_FLAG:
        return spec.quality_flag_attributes()
    if variable_name == RANDOM_UNCERTAINTY:
        return spec.random_uncertainty_attributes()
    if variable_name == JOINT_UNCERTAINTY:
        return spec.joint_uncertainty_attributes()
    check_tower_series_variable_is_known(variable_name)
    column = SOURCE.columns["NIGHT"]
    return {
        "long_name": "Nighttime flag",
        "flag_values": np.array(column.flag_values, dtype=np.int8),
        "flag_meanings": column.flag_meanings,
        "source_file": spec.raw_file,
        "source_column": "NIGHT",
        "comment": "ONEFlux's flag, from SW_IN_POT; -1 where the source reports none.",
    }


def _dataset_attributes(
    spec: NetEcosystemExchangeSpec, tower_table: pd.DataFrame, towers: list[str]
) -> dict[str, Any]:
    """The processed file's dataset attributes."""
    resolution = resolve_resolution(spec.resolution)
    minutes = int(resolution.step / pd.Timedelta(minutes=1))
    of_resolution = tower_table[tower_table["resolution_minutes"] == minutes]
    matched_not_primary = of_resolution[of_resolution[SITE_ID].notna() & ~of_resolution["primary"]]
    excluded = of_resolution[of_resolution["excluded_reason"] != ""]
    without_series = sorted(set(of_resolution.loc[of_resolution["primary"], "tower"]) - set(towers))
    return {
        "Conventions": CF_CONVENTIONS,
        "title": f"{spec.long_label}, {resolution.name}, on the site pool",
        "net_ecosystem_exchange": spec.name,
        "upstream_product": spec.upstream_product,
        "source_file": spec.raw_file,
        "resolution": resolution.name,
        "towers_not_primary": ", ".join(matched_not_primary["tower"]),
        "towers_excluded": ", ".join(excluded["tower"]),
        "towers_without_the_series": ", ".join(without_series),
        "tower_table": "data/raw/net_ecosystem_exchange/ameriflux_towers.csv, which records every choice and exclusion",
        "acknowledgement": (
            "AmeriFlux FLUXNET data, CC-BY-4.0: cite each site's dataset by its DOI (the doi "
            "coordinate) and follow the AmeriFlux data use policy, "
            "https://ameriflux.lbl.gov/data/data-policy/"
        ),
        "history": (
            f"scripts/ingest_net_ecosystem_exchange.py: read {spec.raw_file} and the tower table, "
            "kept each primary tower's series, shifted its local-standard-time stamps to UTC by "
            "its offset, placed it on its pool site with the site table's lon/lat, and wrote "
            "the spec's fields as attributes; values unchanged"
        ),
        "created": utc_timestamp(),
    }


# ── checks ────────────────────────────────────────────────────────────────────


def check_towers_are_in_the_tower_table(towers: list[str], rows: pd.DataFrame) -> None:
    """Every tower of a series is in the tower table; *rows* is indexed on tower."""
    unknown = [tower for tower in towers if tower not in rows.index]
    if unknown:
        raise KeyError(
            f"tower(s) {truncated(unknown)} are not in the tower table; rebuild it with "
            "scripts/raw_sources/build_ameriflux_towers.py."
        )


def check_towers_are_primary(towers: list[str], rows: pd.DataFrame) -> None:
    """Every tower of a series is a primary tower of the tower table; *rows* is indexed on tower."""
    unknown = [tower for tower in towers if not rows.at[tower, "primary"]]
    if unknown:
        raise ValueError(
            f"tower(s) {truncated(unknown)} are not primary in the tower table; read the series "
            "with its source's reader, which keeps only primary towers."
        )


def check_towers_have_distinct_sites(site_ids: np.ndarray) -> None:
    """No two towers of a series share a site."""
    repeated = sorted({site for site in site_ids.tolist() if (site_ids == site).sum() > 1})
    if repeated:
        raise ValueError(
            f"two towers of the series share site(s) {truncated(repeated)}; the tower table "
            "must name one primary tower per site and resolution."
        )


def check_tower_series_variable_is_known(variable_name: str) -> None:
    """A tower series' variable is one a processed file may hold."""
    if variable_name not in _VARIABLE_NAMES:
        raise ValueError(
            f"no attributes for a tower series variable {variable_name!r}; a reader returns only "
            f"{truncated(_VARIABLE_NAMES)}."
        )


def check_processed_net_ecosystem_exchange_exists(path: Path, spec: NetEcosystemExchangeSpec) -> None:
    """A series' processed file exists."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file; make it with "
            f"python scripts/ingest_net_ecosystem_exchange.py --series {spec.name}"
        )


def check_processed_net_ecosystem_exchange_is_valid(
    dataset: xr.Dataset, spec: NetEcosystemExchangeSpec, *, message_name: str
) -> None:
    """A processed NEE file follows the data model for its spec."""
    check_processed_net_ecosystem_exchange_is_for_the_spec(dataset, spec, message_name=message_name)
    check_processed_net_ecosystem_exchange_has_a_value(dataset, message_name=message_name)
    check_processed_net_ecosystem_exchange_holds_only_its_variables(dataset, message_name=message_name)
    check_processed_net_ecosystem_exchange_variables_are_on_site_and_time(dataset, message_name=message_name)
    check_processed_net_ecosystem_exchange_value_has_the_spec_attributes(dataset, spec, message_name=message_name)
    check_processed_file_has_the_coordinates(
        dataset,
        (SITE, LON, LAT, *TOWER_SITE_COORDINATES, TIME, TIME_BOUNDS),
        remedy=_REMAKE,
        message_name=message_name,
    )
    check_processed_file_locations_are_on_site(dataset, remedy=_REMAKE, message_name=message_name)
    check_processed_file_holds_sites(dataset, remedy=_REMAKE, message_name=message_name)
    check_processed_file_sites_ascend(dataset, remedy=_REMAKE, message_name=message_name)
    check_processed_net_ecosystem_exchange_time_is_the_processed_axis(dataset, spec, message_name=message_name)
    check_processed_net_ecosystem_exchange_bounds_end_at_time(dataset, spec, message_name=message_name)


def check_processed_net_ecosystem_exchange_is_for_the_spec(
    dataset: xr.Dataset, spec: NetEcosystemExchangeSpec, *, message_name: str
) -> None:
    """A processed NEE file is CF-1.11 and names its spec's series."""
    if dataset.attrs.get("Conventions") != CF_CONVENTIONS:
        raise ValueError(f"{message_name}: Conventions is not {CF_CONVENTIONS!r}; {_REMAKE}.")
    if dataset.attrs.get("net_ecosystem_exchange") != spec.name:
        raise ValueError(
            f"{message_name}: holds {dataset.attrs.get('net_ecosystem_exchange')!r}, not "
            f"{spec.name!r}; {_REMAKE}."
        )


def check_processed_net_ecosystem_exchange_has_a_value(dataset: xr.Dataset, *, message_name: str) -> None:
    """A processed NEE file holds ``value``."""
    if VALUE not in dataset.data_vars:
        raise ValueError(f"{message_name}: no {VALUE!r} variable; {_REMAKE}.")


def check_processed_net_ecosystem_exchange_holds_only_its_variables(
    dataset: xr.Dataset, *, message_name: str
) -> None:
    """A processed NEE file holds no variable the data model does not name."""
    unexpected = sorted(set(dataset.data_vars) - set(_VARIABLE_NAMES))
    if unexpected:
        raise ValueError(f"{message_name}: unexpected variables {truncated(unexpected)}; {_REMAKE}.")


def check_processed_net_ecosystem_exchange_variables_are_on_site_and_time(
    dataset: xr.Dataset, *, message_name: str
) -> None:
    """Every variable of a processed NEE file is on ``(site, time)``."""
    for name in dataset.data_vars:
        if dataset[name].dims != (SITE, TIME):
            raise ValueError(f"{message_name}: {name} has dims {dataset[name].dims}, expected (site, time); {_REMAKE}.")


def check_processed_net_ecosystem_exchange_value_has_the_spec_attributes(
    dataset: xr.Dataset, spec: NetEcosystemExchangeSpec, *, message_name: str
) -> None:
    """A processed NEE file's ``value`` carries its spec's units, column, kind and constituent."""
    expected = spec.xarray_attributes()
    for key in ("units", "source_column", "kind", "constituent"):
        if dataset[VALUE].attrs.get(key) != expected[key]:
            raise ValueError(
                f"{message_name}: value's {key} is {dataset[VALUE].attrs.get(key)!r}, the spec "
                f"says {expected[key]!r}; {_REMAKE}."
            )


def check_processed_net_ecosystem_exchange_holds_the_variable(
    dataset: xr.Dataset, variable_name: str, *, message_name: str
) -> None:
    """A processed NEE file holds the variable asked of it."""
    if variable_name not in dataset.data_vars:
        raise KeyError(f"{message_name} has no {variable_name!r}; its source does not report one.")


def check_processed_net_ecosystem_exchange_time_is_the_processed_axis(
    dataset: xr.Dataset, spec: NetEcosystemExchangeSpec, *, message_name: str
) -> None:
    """A processed NEE file's ``time`` is the step ends of its resolution's processed axis."""
    resolution = resolve_resolution(spec.resolution)
    expected = (resolution.processed_step_starts() + resolution.step).as_unit("ns").to_numpy()
    if not np.array_equal(dataset[TIME].values, expected):
        raise ValueError(f"{message_name}: time is not the {resolution.name} UTC axis; {_REMAKE}.")


def check_processed_net_ecosystem_exchange_bounds_end_at_time(
    dataset: xr.Dataset, spec: NetEcosystemExchangeSpec, *, message_name: str
) -> None:
    """A processed NEE file's ``time_bounds`` is one step ending at ``time``."""
    step = resolve_resolution(spec.resolution).step.to_timedelta64()
    bounds = dataset[TIME_BOUNDS].values
    if not (np.array_equal(bounds[:, 1], dataset[TIME].values) and (bounds[:, 1] - bounds[:, 0] == step).all()):
        raise ValueError(f"{message_name}: time_bounds is not one step ending at time; {_REMAKE}.")
