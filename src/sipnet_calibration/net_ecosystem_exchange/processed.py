"""The processed products: one series per file, on the site pool and a UTC axis.

``data/processed/net_ecosystem_exchange/<name>.nc`` holds one product: a tower
series (:mod:`sipnet_calibration.net_ecosystem_exchange.sources`) placed on the
sites of the primary towers, with the site table's coordinates, pySIPNET's
time coordinates, and the spec's fields as attributes. The values are the
source's, unchanged.

The package docstring gives the data model in full.

Contents
--------
:func:`build_net_ecosystem_exchange`
    Tower series to product. Pure; ``scripts/ingest_net_ecosystem_exchange.py``
    adds the checks and the write.
:func:`netcdf_encoding`
    How it is stored.
:func:`load_net_ecosystem_exchange`
    Read one product and check it against its spec.
:func:`net_ecosystem_exchange_values`, :func:`net_ecosystem_exchange_quality_flags`,
:func:`net_ecosystem_exchange_random_uncertainties`, :func:`net_ecosystem_exchange_joint_uncertainties`
    One variable of several products, one array per product.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.conventions import CF_CONVENTIONS
from sipnet_calibration.net_ecosystem_exchange.names import (
    BOUNDS,
    MEMBER,
    SITE,
    TIME,
    TIME_BOUNDS,
    TIME_STEP_LENGTH,
    TIME_STEP_START,
    TOWER,
    _LAT_ATTRS,
    _LON_ATTRS,
    _SITE_ATTRS,
    _utc_timestamp,
    product_path,
    resolve_resolution,
)
from sipnet_calibration.net_ecosystem_exchange.source_files import SOURCE
from sipnet_calibration.net_ecosystem_exchange.specs import (
    NET_ECOSYSTEM_EXCHANGE_NAMES,
    NetEcosystemExchangeSpec,
    resolve_net_ecosystem_exchange,
)

__all__ = [
    "SITE_COORDINATES",
    "build_net_ecosystem_exchange",
    "load_net_ecosystem_exchange",
    "net_ecosystem_exchange_joint_uncertainties",
    "net_ecosystem_exchange_quality_flags",
    "net_ecosystem_exchange_random_uncertainties",
    "net_ecosystem_exchange_values",
    "netcdf_encoding",
]

#: The coordinates on ``site``, besides ``site`` itself.
SITE_COORDINATES = (
    "lon",
    "lat",
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


def build_net_ecosystem_exchange(
    spec: NetEcosystemExchangeSpec,
    tower_series: xr.Dataset,
    tower_table: pd.DataFrame,
    site_table: pd.DataFrame,
) -> xr.Dataset:
    """Turn a tower series into the product the data model describes.

    Parameters
    ----------
    spec:
        The product.
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
        The product, sites ascending.

    Raises
    ------
    ValueError
        If a tower of the series is not a primary tower of the table, two
        towers share a site, or a site is not in the site table.

    Notes
    -----
    Pure and source-agnostic: it reads nothing but its arguments, and a
    ``member`` dimension on the series passes through to the product.
    """
    towers = [str(tower) for tower in tower_series[TOWER].values]
    rows = tower_table.set_index("tower")
    unknown = [tower for tower in towers if tower not in rows.index or not rows.at[tower, "primary"]]
    if unknown:
        raise ValueError(f"towers not primary in the tower table: {unknown[:10]}")
    site_ids = rows.loc[towers, "site_id"].astype(np.int64).to_numpy()
    if len(set(site_ids.tolist())) != len(site_ids):
        raise ValueError("two towers of the series share a site")
    pool = site_table.set_index("site_id")
    missing = sorted(set(site_ids.tolist()) - set(pool.index.tolist()))
    if missing:
        raise ValueError(f"sites not in the site table: {missing[:10]}")

    order = np.argsort(site_ids, kind="stable")
    series = tower_series.isel({TOWER: order})
    sites = site_ids[order]
    ordered_towers = [towers[i] for i in order]
    series = series.drop_vars("utc_offset").rename({TOWER: SITE}).assign_coords({SITE: sites.astype(np.int32)})

    data_vars = {}
    for variable in series.data_vars:
        array = series[variable]
        dims = tuple(d for d in (MEMBER, SITE, TIME) if d in array.dims)
        data_vars[variable] = (dims, array.transpose(*dims).values, _variable_attributes(spec, variable))
    resolution = resolve_resolution(spec.resolution)
    ends = pd.DatetimeIndex(series[TIME].values)
    starts = ends - resolution.step
    table = rows.loc[ordered_towers]
    coords: dict[str, Any] = {
        SITE: (SITE, sites.astype(np.int32), _SITE_ATTRS),
        "lon": (SITE, pool.loc[sites, "lon"].to_numpy(np.float64), _LON_ATTRS),
        "lat": (SITE, pool.loc[sites, "lat"].to_numpy(np.float64), _LAT_ATTRS),
        "ameriflux_site_id": (SITE, np.array(ordered_towers, dtype=object), {"long_name": "AmeriFlux site identifier of the tower"}),
        "tower_lon": (SITE, table["tower_lon"].to_numpy(np.float64), {"long_name": "Tower longitude, from AmeriFlux", "units": "degrees_east"}),
        "tower_lat": (SITE, table["tower_lat"].to_numpy(np.float64), {"long_name": "Tower latitude, from AmeriFlux", "units": "degrees_north"}),
        "match_basis": (SITE, table["match_basis"].to_numpy(object), {"long_name": "How the tower was matched to the site"}),
        "utc_offset": (
            SITE,
            table["utc_offset_hours"].to_numpy(np.float64),
            {
                "long_name": "Tower's local standard time minus UTC",
                "units": "hours",
                "comment": "Recovered from the tower's SW_IN_POT; local standard time is time + utc_offset.",
            },
        ),
        "doi": (SITE, table["doi"].to_numpy(object), {"long_name": "AmeriFlux FLUXNET DOI of the site's dataset"}),
        "site_version": (SITE, table["site_version"].to_numpy(object), {"long_name": "FULLSET version"}),
        TIME: (TIME, ends.as_unit("ns").to_numpy(), _time_attributes()),
        TIME_STEP_START: (TIME, starts.as_unit("ns").to_numpy(), {"long_name": "Start of timestep"}),
        TIME_STEP_LENGTH: (
            TIME,
            np.full(len(ends), resolution.step.to_timedelta64().astype("timedelta64[ns]")),
            {"long_name": "Timestep length"},
        ),
        TIME_BOUNDS: (
            (TIME, BOUNDS),
            np.stack([starts.as_unit("ns").to_numpy(), ends.as_unit("ns").to_numpy()], axis=1),
            {"long_name": "Timestep bounds", "comment": "The interval [time_step_start, time] each value covers."},
        ),
    }
    if MEMBER in series.dims:
        coords[MEMBER] = (MEMBER, series[MEMBER].values)
    return xr.Dataset(data_vars, coords=coords, attrs=_product_attributes(spec, tower_table, ordered_towers))


def netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """The on-disk encoding: compressed, one chunk per site, integer time, no
    coordinate fill, as CF requires."""
    encoding: dict[str, dict[str, Any]] = {}
    for name, array in dataset.data_vars.items():
        fill = np.nan if array.dtype.kind == "f" else None
        chunks = tuple(1 if d != TIME else array.sizes[TIME] for d in array.dims)
        encoding[str(name)] = {"zlib": True, "complevel": 4, "_FillValue": fill, "chunksizes": chunks}
    for name in dataset.coords:
        encoding[str(name)] = {"_FillValue": None}
    for name in (TIME, TIME_STEP_START, TIME_BOUNDS):
        encoding[name].update({"units": TIME_UNITS, "calendar": CALENDAR, "dtype": "int32"})
    encoding[TIME_STEP_LENGTH].update({"units": "minutes", "dtype": "int32"})
    return encoding


def load_net_ecosystem_exchange(product_name: str, path: Path | str | None = None) -> xr.Dataset:
    """Read one product and check it against its spec.

    Parameters
    ----------
    product_name:
        A name in :data:`~sipnet_calibration.net_ecosystem_exchange.specs.NET_ECOSYSTEM_EXCHANGE_NAMES`.
    path:
        The netCDF to read. Defaults to
        :func:`~sipnet_calibration.net_ecosystem_exchange.names.product_path`.

    Returns
    -------
    xarray.Dataset
        The data model, opened lazily.

    Raises
    ------
    KeyError
        If no product has that name.
    FileNotFoundError
        If the file is absent, with the command that produces it.
    ValueError
        If the file does not match the data model or the spec.
    """
    spec = resolve_net_ecosystem_exchange(product_name)
    path = Path(path) if path is not None else product_path(product_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file. Produce it with:\n"
            f"  python scripts/ingest_net_ecosystem_exchange.py --product {product_name}"
        )
    try:
        dataset = xr.open_dataset(path, engine="h5netcdf")
    except OSError as error:
        raise ValueError(f"{path}: not readable as netCDF-4/HDF5 ({error})") from error
    try:
        _check_product(dataset, spec, path)
    except Exception:
        dataset.close()
        raise
    return dataset


def net_ecosystem_exchange_values(
    product_names: Sequence[str] | str | None = None,
    *,
    sites: Iterable[int] | int | None = None,
    directory: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The ``value`` of several products, one array per product.

    Parameters
    ----------
    product_names:
        Product names, in the order the result should carry them, or one name
        on its own. Defaults to every product.
    sites:
        Site ids to keep, in the order given, or one on its own. Each must be in
        every product asked for. Defaults to all of each product's sites.
    directory:
        Where the products are. Defaults to
        :func:`~sipnet_calibration.net_ecosystem_exchange.names.default_product_dir`.

    Returns
    -------
    dict
        Product name to its ``([member,] site, time)`` array, named for the
        product, with the spec's attributes, ``lon``/``lat`` and the other site
        coordinates, and ``time_step_start``/``time_step_length`` on ``time``.
        A dict because the half-hourly and hourly products do not share a time
        axis.

    Raises
    ------
    TypeError
        If *sites* is a string, or holds a value that is not a whole number.
    ValueError
        If a requested site is not in a product, or is asked for twice.
    """
    return _variable_by_product("value", product_names, sites, directory)


def net_ecosystem_exchange_quality_flags(
    product_names: Sequence[str] | str | None = None,
    *,
    sites: Iterable[int] | int | None = None,
    directory: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The ``quality_flag`` of several products; as :func:`net_ecosystem_exchange_values`."""
    return _variable_by_product("quality_flag", product_names, sites, directory)


def net_ecosystem_exchange_random_uncertainties(
    product_names: Sequence[str] | str | None = None,
    *,
    sites: Iterable[int] | int | None = None,
    directory: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The ``random_uncertainty`` of several products; as :func:`net_ecosystem_exchange_values`."""
    return _variable_by_product("random_uncertainty", product_names, sites, directory)


def net_ecosystem_exchange_joint_uncertainties(
    product_names: Sequence[str] | str | None = None,
    *,
    sites: Iterable[int] | int | None = None,
    directory: Path | str | None = None,
) -> dict[str, xr.DataArray]:
    """The ``joint_uncertainty`` of several products; as :func:`net_ecosystem_exchange_values`."""
    return _variable_by_product("joint_uncertainty", product_names, sites, directory)


def _variable_by_product(
    variable: str,
    product_names: Sequence[str] | str | None,
    sites: Iterable[int] | int | None,
    directory: Path | str | None,
) -> dict[str, xr.DataArray]:
    if isinstance(product_names, str):
        product_names = [product_names]
    names = list(product_names) if product_names is not None else list(NET_ECOSYSTEM_EXCHANGE_NAMES)
    for name in names:
        resolve_net_ecosystem_exchange(name)
    wanted = _site_ids(sites)
    arrays = {}
    for name in names:
        dataset = load_net_ecosystem_exchange(name, product_path(name, directory))
        if variable not in dataset:
            raise ValueError(f"{name} has no {variable!r}")
        if wanted is not None:
            missing = sorted(set(wanted) - set(dataset[SITE].values.tolist()))
            if missing:
                raise ValueError(f"sites not in {name}: {missing[:10]}")
            dataset = dataset.sel({SITE: wanted})
        arrays[name] = dataset[variable].rename(name)
    return arrays


def _site_ids(sites: Iterable[int] | int | None) -> list[int] | None:
    if sites is None:
        return None
    if isinstance(sites, str):
        raise TypeError(
            f"sites={sites!r} is a string, which would be read one character per site. "
            "Pass an integer or a sequence of integers."
        )
    if isinstance(sites, (int, np.integer)):
        sites = [sites]
    wanted = []
    for site in sites:
        number = int(site)
        if number != site:
            raise TypeError(f"site {site!r} is not a whole number")
        wanted.append(number)
    if len(set(wanted)) != len(wanted):
        raise ValueError(f"sites repeats {sorted({s for s in wanted if wanted.count(s) > 1})}")
    return wanted


def _variable_attributes(spec: NetEcosystemExchangeSpec, variable: str) -> dict[str, Any]:
    if variable == "value":
        return spec.xarray_attributes()
    if variable == "quality_flag":
        return spec.quality_flag_attributes()
    if variable == "random_uncertainty":
        return spec.random_uncertainty_attributes()
    if variable == "joint_uncertainty":
        return spec.joint_uncertainty_attributes()
    if variable == "night":
        column = SOURCE.columns["NIGHT"]
        return {
            "long_name": "Nighttime flag",
            "flag_values": np.array(column.flag_values, dtype=np.int8),
            "flag_meanings": column.flag_meanings,
            "source_file": spec.raw_file,
            "source_column": "NIGHT",
            "comment": "ONEFlux's flag, from SW_IN_POT; -1 where the source reports none.",
        }
    raise ValueError(f"no attributes for a tower series variable {variable!r}")


def _time_attributes() -> dict[str, Any]:
    return {
        "standard_name": "time",
        "axis": "T",
        "long_name": "End of timestep",
        "bounds": TIME_BOUNDS,
        "time_zone": "UTC",
        "comment": (
            "The end of each averaging interval, UTC; the source stamps are local standard "
            "time, shifted per site by utc_offset."
        ),
    }


def _product_attributes(
    spec: NetEcosystemExchangeSpec, tower_table: pd.DataFrame, towers: list[str]
) -> dict[str, Any]:
    resolution = resolve_resolution(spec.resolution)
    of_resolution = tower_table[tower_table["resolution_minutes"] == resolution.minutes]
    matched_not_primary = of_resolution[of_resolution["site_id"].notna() & ~of_resolution["primary"]]
    excluded = of_resolution[of_resolution["excluded_reason"] != ""]
    without_series = sorted(set(of_resolution.loc[of_resolution["primary"], "tower"]) - set(towers))
    return {
        "Conventions": CF_CONVENTIONS,
        "title": f"{spec.long_label}, {resolution.name}, on the site pool",
        "product": spec.product,
        "product_name": spec.name,
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
        "created": _utc_timestamp(),
    }


def _check_product(dataset: xr.Dataset, spec: NetEcosystemExchangeSpec, path: Path) -> None:
    """Raise unless *dataset* is the product the data model and *spec* describe."""
    if dataset.attrs.get("Conventions") != CF_CONVENTIONS:
        raise ValueError(f"{path}: Conventions is not {CF_CONVENTIONS!r}")
    if dataset.attrs.get("product_name") != spec.name:
        raise ValueError(f"{path}: product_name is {dataset.attrs.get('product_name')!r}, not {spec.name!r}")
    if "value" not in dataset.data_vars:
        raise ValueError(f"{path}: no 'value' variable")
    unexpected = set(dataset.data_vars) - {"value", "quality_flag", "random_uncertainty", "joint_uncertainty", "night"}
    if unexpected:
        raise ValueError(f"{path}: unexpected variables {sorted(unexpected)}")
    value = dataset["value"]
    if value.dims[-2:] != (SITE, TIME):
        raise ValueError(f"{path}: value has dims {value.dims}, expected ([member,] site, time)")
    for key in ("units", "source_column", "kind", "constituent"):
        if value.attrs.get(key) != spec.xarray_attributes()[key]:
            raise ValueError(f"{path}: value's {key} is {value.attrs.get(key)!r}, the spec says {spec.xarray_attributes()[key]!r}")
    for name in (SITE, TIME, TIME_STEP_START, TIME_STEP_LENGTH, TIME_BOUNDS, *SITE_COORDINATES):
        if name not in dataset.coords:
            raise ValueError(f"{path}: missing the {name!r} coordinate")
    site = dataset[SITE].values
    if site.size and np.any(np.diff(site) <= 0):
        raise ValueError(f"{path}: site is not strictly ascending")
    resolution = resolve_resolution(spec.resolution)
    expected = (resolution.product_step_starts() + resolution.step).as_unit("ns").to_numpy()
    if not np.array_equal(dataset[TIME].values, expected):
        raise ValueError(f"{path}: time is not the {resolution.name} UTC axis")
    starts = dataset[TIME_STEP_START].values
    if not np.array_equal(dataset[TIME_BOUNDS].values[:, 0], starts) or not np.array_equal(
        dataset[TIME_BOUNDS].values[:, 1], dataset[TIME].values
    ):
        raise ValueError(f"{path}: time_bounds is not [time_step_start, time]")
