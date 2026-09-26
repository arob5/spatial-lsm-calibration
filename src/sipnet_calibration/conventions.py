"""The names, attributes and settings every module of the package shares.

Contents
--------
Dimensions
    :data:`SITE`, :data:`TIME`, :data:`SAMPLE`; the data sources' ensemble
    dims :data:`INITIAL_CONDITION_MEMBER` and :data:`DRIVER_MEMBER`,
    collected in :data:`DATA_SOURCE_MEMBER_NAMES`; and the reserved spatial
    names :data:`POINT`, :data:`LAT`, :data:`LON`, :data:`Y`, :data:`X`,
    collected in :data:`SPATIAL_DIM_NAMES`; :data:`NON_BATCH_DIM_NAMES`, the
    names no batch dim takes; :data:`BOUNDS`, the second dim of
    :data:`TIME_BOUNDS`.
Coordinates
    :data:`SOURCE_INDEX`, the 1-based file index beside a data source's
    own ensemble dim;
    :data:`LON` and :data:`LAT` on a site or a point; pySIPNET's timestep
    coordinates :data:`TIMESTEP_START` and :data:`TIMESTEP_LENGTH`, with
    ``time`` collected in :data:`TIME_COORD_NAMES`; an observation's window
    edges :data:`WINDOW_START` and :data:`WINDOW_END`; SIPNET's row labels,
    :data:`SIPNET_ROW_LABEL_NAMES`.
Columns
    :data:`SITE_ID`, the site table's key.
Variables
    :data:`TIME_BOUNDS`, the CF bounds variable of ``time`` a processed file
    or pySIPNET's output stores.
Attributes
    :data:`SITE_ATTRIBUTES`, :data:`LON_ATTRIBUTES`, :data:`LAT_ATTRIBUTES`,
    :data:`SAMPLE_ATTRIBUTES`, :data:`DATA_SOURCE_MEMBER_ATTRIBUTES` and
    :data:`SOURCE_INDEX_ATTRIBUTES`, the attributes of those coordinates,
    one wording each;
    :data:`STALE_TIME_ATTRIBUTE_NAMES`, the ``time`` attributes a field drops.
Dtypes and patterns
    :data:`SITE_DTYPE`, the dtype of a site id; :data:`BATCH_LABEL_DTYPE`, the
    dtype of a batch dim's labels; :data:`NAME_PATTERN`, what a processed name
    looks like.
Settings
    :data:`CF_CONVENTIONS`, the ``Conventions`` attribute the netCDF files
    declare; :data:`DATA_ROOT_ENV_VAR` and :func:`data_root`, where the
    storage-backed part of ``data/`` is.
Read-only containers
    :func:`read_only_copy`, a copy of a ``DataArray`` or ``Dataset`` whose
    arrays cannot be written; and :class:`ReadOnlyCopies`, a dataclass
    attribute that keeps and hands out such copies.

Notes
-----
A name or an attribute lives here once two modules have to agree on it, so
that no two of them can spell it differently. Modules import what they need
from here and define none of these values themselves, and none re-exports
one. This module imports nothing from the package, so everything else can
depend on it.

:data:`CF_CONVENTIONS` is this package's own, not pySIPNET's, although the two
agree today: it is what the processed files declare, and a pin bump of
pySIPNET must not change it without a re-ingest.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict
from pysipnet.dataset import BOUNDS_DIMENSION, TIME_DIMENSION

__all__ = [
    "BATCH_LABEL_DTYPE",
    "BOUNDS",
    "CF_CONVENTIONS",
    "DATA_ROOT_ENV_VAR",
    "DATA_SOURCE_MEMBER_ATTRIBUTES",
    "DATA_SOURCE_MEMBER_NAMES",
    "DRIVER_MEMBER",
    "INITIAL_CONDITION_MEMBER",
    "LAT",
    "LAT_ATTRIBUTES",
    "LON",
    "LON_ATTRIBUTES",
    "NAME_PATTERN",
    "NON_BATCH_DIM_NAMES",
    "POINT",
    "SAMPLE",
    "SAMPLE_ATTRIBUTES",
    "SIPNET_ROW_LABEL_NAMES",
    "SITE",
    "SITE_ATTRIBUTES",
    "SITE_DTYPE",
    "SITE_ID",
    "SOURCE_INDEX",
    "SOURCE_INDEX_ATTRIBUTES",
    "SPATIAL_DIM_NAMES",
    "STALE_TIME_ATTRIBUTE_NAMES",
    "TIME",
    "TIMESTEP_LENGTH",
    "TIMESTEP_START",
    "TIME_BOUNDS",
    "TIME_COORD_NAMES",
    "WINDOW_END",
    "WINDOW_START",
    "ReadOnlyCopies",
    "X",
    "Y",
    "data_root",
    "read_only_copy",
]


# ── dimensions ────────────────────────────────────────────────────────────────

#: The site dimension and coordinate: the site ids of the site table in use.
SITE = "site"

#: The time dimension, pySIPNET's own name for it.
TIME = TIME_DIMENSION

#: The default name of the batch dimension created from the rows of batched
#: Flat: one row of ``theta`` is one sample. Every function that creates one
#: takes ``batch_dim=`` to name it otherwise, and all default to this, so the
#: Fields of the two vectors align on one dim.
SAMPLE = "sample"

#: The batch dimension of PEcAn's initial condition ensemble, in the
#: processed file and every field made from it.
INITIAL_CONDITION_MEMBER = "initial_condition_member"

#: The batch dimension of the ERA5 driver ensemble.
DRIVER_MEMBER = "driver_member"

#: The batch dimensions of the data sources' own ensembles, each named for its
#: source so that two of them never pair by accident.
DATA_SOURCE_MEMBER_NAMES: tuple[str, ...] = (INITIAL_CONDITION_MEMBER, DRIVER_MEMBER)

#: The spatial dimension of locations that are not sites, such as spatial
#: prediction targets: integer labels with no meaning beyond the field, and
#: ``lon``/``lat`` coordinates.
POINT = "point"

#: Longitude, in degrees east: a coordinate on ``site`` or ``point``, or a
#: dimension of a ``(lat, lon)`` raster.
LON = "lon"

#: Latitude, in degrees north: a coordinate on ``site`` or ``point``, or a
#: dimension of a ``(lat, lon)`` raster.
LAT = "lat"

#: The projected easting of a ``(y, x)`` raster, in meters.
X = "x"

#: The projected northing of a ``(y, x)`` raster, in meters.
Y = "y"

#: The names reserved for a field's spatial dimension. None of them is ever a
#: batch dimension, whatever its labels.
SPATIAL_DIM_NAMES: tuple[str, ...] = (SITE, POINT, LAT, LON, Y, X)


# ── coordinates ───────────────────────────────────────────────────────────────

# pySIPNET exports no constants for these two coordinates; once it does,
# beside TIME_DIMENSION, these definitions become imports from
# pysipnet.dataset. Until then they carry pySIPNET's names as literals.

#: pySIPNET's coordinate on ``time`` for the start of the timestep a row
#: covers; ``time`` is its end.
TIMESTEP_START = "timestep_start"

#: pySIPNET's coordinate on ``time`` for the declared duration of the
#: timestep a row covers, as ``timedelta64``.
TIMESTEP_LENGTH = "timestep_length"

#: pySIPNET's time coordinates, which a model field and a driver field both
#: keep. ``time`` is the end of the timestep and ``timestep_start`` its
#: start, so the two are the CF bounds pair.
TIME_COORD_NAMES: tuple[str, ...] = (TIME, TIMESTEP_START, TIMESTEP_LENGTH)

#: The coordinate on an observation source's ``time`` for the start of the
#: window each value covers: the first edge of the CF ``time_bounds`` variable the
#: processed file stores, one-dimensional since a ``DataArray`` cannot carry
#: the two-dimensional ``(time, bounds)`` variable.
#: :func:`sipnet_calibration.constraints.constraint_fields` adds it and
#: :func:`sipnet_calibration.observation.time_alignment.windows_from_observed_values`
#: reads it.
WINDOW_START = "window_start"

#: The coordinate on an observation source's ``time`` for the end of the
#: window each value covers; see :data:`WINDOW_START`.
WINDOW_END = "window_end"


#: The coordinate beside a data source's own ensemble dim (such as
#: ``initial_condition_member`` or ``driver_member``) holding each member's
#: 1-based index in the source's file names, so a file can always be found
#: from a member. One name for every data source.
SOURCE_INDEX = "source_index"

#: The names that are never a batch dimension, whatever their labels, and
#: that no batch dimension may take: the spatial names, ``time``, and
#: :data:`SOURCE_INDEX`, which sits beside a data source's member dim as a
#: coordinate of it.
NON_BATCH_DIM_NAMES: tuple[str, ...] = (*SPATIAL_DIM_NAMES, TIME, SOURCE_INDEX)

#: SIPNET's own row labels, the start of each step, which pySIPNET's output
#: carries as integer or float coordinates on ``time``. A field drops them,
#: since ``timestep_start`` is the same instant; they are neither batch
#: labels nor names a batch dimension may take.
SIPNET_ROW_LABEL_NAMES: tuple[str, ...] = ("year", "day_of_year", "hour_of_day")


# ── variables ─────────────────────────────────────────────────────────────────

#: The CF bounds variable of ``time``, ``(time, bounds)``, that a processed
#: file stores where a value's support is documented and pySIPNET's output
#: stores for its timesteps. Its second dimension is pySIPNET's
#: ``pysipnet.dataset.BOUNDS_DIMENSION``.
TIME_BOUNDS = "time_bounds"

#: The second dimension of :data:`TIME_BOUNDS`, pySIPNET's
#: ``pysipnet.dataset.BOUNDS_DIMENSION``: the two edges of each interval. It
#: has no coordinate, so it is never a dimension of a field.
BOUNDS = BOUNDS_DIMENSION


# ── columns ───────────────────────────────────────────────────────────────────

#: The site table's key column, and the column every file keyed on sites
#: addresses its records by.
SITE_ID = "site_id"


# ── attributes ────────────────────────────────────────────────────────────────


#: The attributes of a ``site`` coordinate: read-only, so a coordinate can be
#: given them as they are (xarray copies what it is given).
SITE_ATTRIBUTES = frozendict(
    {
        "long_name": "Model site identifier",
        "comment": "Site identifier of the site table; never renumbered.",
    }
)

#: The CF attributes of a ``lon`` coordinate. Read-only, as
#: :data:`SITE_ATTRIBUTES`.
LON_ATTRIBUTES = frozendict(
    {"standard_name": "longitude", "long_name": "Longitude", "units": "degrees_east"}
)

#: The CF attributes of a ``lat`` coordinate. Read-only, as
#: :data:`SITE_ATTRIBUTES`.
LAT_ATTRIBUTES = frozendict(
    {"standard_name": "latitude", "long_name": "Latitude", "units": "degrees_north"}
)

#: The attributes of a ``sample`` coordinate. Read-only, as
#: :data:`SITE_ATTRIBUTES`.
SAMPLE_ATTRIBUTES = frozendict(
    {
        "long_name": "Sample",
        "comment": "Label of the row of batched Flat the value was computed from.",
    }
)

#: The attributes of a data source's own ensemble dim, such as
#: ``initial_condition_member``, whose label is its member's identity: its
#: :data:`SOURCE_INDEX` less one, whatever subset of the members is loaded.
#: Read-only, as :data:`SITE_ATTRIBUTES`.
DATA_SOURCE_MEMBER_ATTRIBUTES = frozendict(
    {
        "long_name": "Ensemble member of the data source",
        "comment": (
            "0-based: source_index - 1, the member's 1-based index in the source's file "
            "names less one, whatever members are loaded; meaningful only within this "
            "data source."
        ),
    }
)

#: The attributes of a :data:`SOURCE_INDEX` coordinate. Read-only, as
#: :data:`SITE_ATTRIBUTES`.
SOURCE_INDEX_ATTRIBUTES = frozendict(
    {
        "long_name": "Member index in the data source's file names",
        "comment": "1-based, as the data source numbers its files.",
    }
)

#: ``time`` attributes a field does not keep. ``bounds`` names pySIPNET's
#: two-dimensional ``time_bounds`` variable, which a field cannot carry and
#: which describes the source's timesteps, not a coarser one's.
STALE_TIME_ATTRIBUTE_NAMES: tuple[str, ...] = ("bounds",)


# ── read-only data ────────────────────────────────────────────────────────────


def read_only_copy(data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """A shallow copy of *data* whose NumPy arrays cannot be written.

    The copy has its own attributes and coordinate bindings; its arrays are
    *data*'s, made read-only.

    Parameters
    ----------
    data:
        A ``DataArray`` or ``Dataset``.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The copy.
    """
    copied = data.copy(deep=False)
    variables = (
        copied.variables.values()
        if isinstance(copied, xr.Dataset)
        else (copied.variable, *(c.variable for c in copied.coords.values()))
    )
    for variable in variables:
        if isinstance(variable.data, np.ndarray):
            variable.data.flags.writeable = False
    return copied


class ReadOnlyCopies:
    """A dataclass attribute that keeps a copy of what it is given and hands out copies.

    Assigned a ``DataArray`` or ``Dataset``, it keeps a deep, in-memory,
    read-only copy, and every read returns :func:`read_only_copy` of that;
    assigned a ``DataFrame``, it keeps a copy, and every read returns a copy;
    anything else is kept and returned as it is. It has no default.

    Notes
    -----
    The copy is made on assignment, so nothing the caller holds is made
    read-only, and a frozen dataclass whose attributes these are hands out
    nothing that changes the instance. A read freezes the kept copy again,
    so a ``copy.deepcopy`` of the instance, whose arrays NumPy makes
    writeable, still hands out read-only ones.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._stored = f"_{name}"

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            # What dataclasses reads for a default: none.
            raise AttributeError(self._stored)
        value = instance.__dict__[self._stored]
        if isinstance(value, (xr.DataArray, xr.Dataset)):
            return read_only_copy(value)
        if isinstance(value, pd.DataFrame):
            return value.copy()
        return value

    def __set__(self, instance: Any, value: Any) -> None:
        if isinstance(value, (xr.DataArray, xr.Dataset)):
            # load() computes a dask or lazily indexed copy in place, so the
            # arrays frozen are the ones kept rather than fresh ones per read.
            value = read_only_copy(value.copy(deep=True).load())
        elif isinstance(value, pd.DataFrame):
            value = value.copy()
        instance.__dict__[self._stored] = value


# ── dtypes and patterns ───────────────────────────────────────────────────────

#: The dtype of a site id, on a ``site`` coordinate and in the site table.
SITE_DTYPE = np.int32

#: The dtype of a batch dim's labels, which may be any distinct integers. The
#: labels a batched Flat is given are ``0`` to ``n_samples - 1``.
BATCH_LABEL_DTYPE = np.int64

#: What a processed name looks like: lower-case words of letters and digits,
#: joined by single underscores.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


# ── settings ──────────────────────────────────────────────────────────────────

#: The Climate and Forecast conventions the processed netCDFs declare, written
#: to the ``Conventions`` attribute and checked on load. CF governs the
#: coordinate and attribute vocabulary the processed files use: ``units``,
#: ``long_name``, ``standard_name`` on ``lon``/``lat``, and no ``_FillValue``
#: on a coordinate.
CF_CONVENTIONS = "CF-1.11"

#: The environment variable that moves ``data/`` elsewhere, for a run on the
#: SCC or against a copy of the tree.
DATA_ROOT_ENV_VAR = "SIPNET_CALIBRATION_DATA"

#: This package's directory, ``src/sipnet_calibration``. The data root is found
#: from here rather than by counting parents from each caller's ``__file__``,
#: which is what silently retargeted every path in ``initial_conditions`` when
#: that module became a package one directory deeper.
_PACKAGE_DIRECTORY = Path(__file__).resolve().parent


def data_root() -> Path:
    """``data/`` beside the package's ``src``, or ``$SIPNET_CALIBRATION_DATA``.

    Every module that reads or writes under ``data/`` goes through this, so a
    run pointed at a different tree moves all of them together.
    """
    root = os.environ.get(DATA_ROOT_ENV_VAR)
    return Path(root) if root else _PACKAGE_DIRECTORY.parents[1] / "data"
