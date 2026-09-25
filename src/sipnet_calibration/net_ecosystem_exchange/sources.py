"""Source readers: a raw file to the standard intermediate every product is built from.

Each source of net ecosystem exchange has one reader, registered in
:data:`SOURCE_READERS`, and every reader returns the same shape, the
**tower series**: an ``xarray.Dataset`` on ``(tower, time)`` -- and ``member``,
for an ensemble source -- with ``time`` the UTC end of each step, and the
variables of :data:`TOWER_SERIES_VARIABLES` that the source has. Everything
after the reader, in :mod:`sipnet_calibration.net_ecosystem_exchange.processed`,
is shared, which is what lets a second source enter as one reader and one spec.

Data model
----------
**Dimensions**: ``tower``, ``time`` and, for an ensemble source, ``member``.

**Coordinates**: ``tower``, the AmeriFlux identifier; ``time``,
``datetime64[ns]``, the UTC end of each step on the product axis
(:meth:`~sipnet_calibration.net_ecosystem_exchange.names.Resolution.product_step_starts`
plus one step); ``utc_offset`` on ``tower``, hours, the shift applied.

**Data variables**: ``value`` (``float64``, ``NaN`` missing) is always
present; ``quality_flag`` and ``night`` (``int8``, ``-1`` missing) and
``random_uncertainty`` and ``joint_uncertainty`` (``float64``) where the source
has them. Values are the source's.

Contents
--------
:data:`SOURCE_READERS`
    Source name to reader.
:func:`read_ameriflux_tower_series`
    The AmeriFlux reader.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.net_ecosystem_exchange.names import (
    TIME,
    TIME_INDEX,
    TOWER,
    resolve_resolution,
)
from sipnet_calibration.net_ecosystem_exchange.specs import NetEcosystemExchangeSpec

__all__ = [
    "SOURCE_READERS",
    "TOWER_SERIES_VARIABLES",
    "read_ameriflux_tower_series",
]

#: The variables a tower series may carry, in order.
TOWER_SERIES_VARIABLES = (
    "value",
    "quality_flag",
    "random_uncertainty",
    "joint_uncertainty",
    "night",
)


def read_ameriflux_tower_series(
    spec: NetEcosystemExchangeSpec, raw: xr.Dataset, tower_table: pd.DataFrame
) -> xr.Dataset:
    """The tower series of one AmeriFlux product.

    Takes the primary towers of the tower table that the raw file holds and
    whose source file carries the spec's value column, and moves each onto the
    UTC axis by its fixed offset.

    Parameters
    ----------
    spec:
        The product.
    raw:
        The raw file of the spec's resolution, as
        :func:`sipnet_calibration.net_ecosystem_exchange.raw.read_raw` returns it.
    tower_table:
        As :func:`sipnet_calibration.net_ecosystem_exchange.towers.read_tower_table`
        returns it.

    Returns
    -------
    xarray.Dataset
        The tower series; see the data model.

    Raises
    ------
    ValueError
        If the raw file is at another resolution, or a primary tower's offset
        cannot move it onto the UTC axis.
    """
    resolution = resolve_resolution(spec.resolution)
    if raw.attrs.get("resolution") != resolution.name:
        raise ValueError(f"{spec.name} needs the {resolution.name} raw file, not {raw.attrs.get('resolution')}")
    towers = _towers_carrying_the_series(spec, raw, tower_table)
    offsets = tower_table.set_index("tower").loc[towers, "utc_offset_hours"].to_numpy(np.float64)
    n_time = len(resolution.product_step_starts())
    columns = {
        "value": spec.value_column,
        "quality_flag": spec.quality_column,
        "random_uncertainty": spec.random_uncertainty_column,
        "joint_uncertainty": spec.joint_uncertainty_column,
        "night": "NIGHT",
    }
    arrays = {}
    for variable, column in columns.items():
        if not column:
            continue
        source = raw[column]
        rows = []
        for tower, offset in zip(towers, offsets):
            first = resolution.raw_offset_of_product_start(offset)
            rows.append(source.sel({TOWER: tower}).isel({TIME_INDEX: slice(first, first + n_time)}).values)
        arrays[variable] = ((TOWER, TIME), np.stack(rows) if rows else np.empty((0, n_time), source.dtype))
    ends = resolution.product_step_starts() + resolution.step
    return xr.Dataset(
        arrays,
        coords={
            TOWER: (TOWER, np.array(towers, dtype=object)),
            TIME: (TIME, ends.as_unit("ns").to_numpy()),
            "utc_offset": (TOWER, offsets),
        },
        attrs={"resolution": resolution.name},
    )


#: Source name to its reader.
SOURCE_READERS: dict[
    str, Callable[[NetEcosystemExchangeSpec, xr.Dataset, pd.DataFrame], xr.Dataset]
] = {"ameriflux": read_ameriflux_tower_series}


def _towers_carrying_the_series(
    spec: NetEcosystemExchangeSpec, raw: xr.Dataset, tower_table: pd.DataFrame
) -> list[str]:
    """Primary towers of the raw file whose source file has the spec's value column."""
    in_raw = set(raw[TOWER].values.tolist())
    absent = dict(zip(raw[TOWER].values.tolist(), raw["absent_columns"].values.tolist()))
    primary = tower_table[tower_table["primary"]]
    return sorted(
        tower
        for tower in primary["tower"]
        if tower in in_raw and spec.value_column not in str(absent[tower]).split(",")
    )
