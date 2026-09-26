"""The names, attributes and settings every module of the package shares.

Contents
--------
Dimensions
    :data:`SITE`, :data:`TIME`, :data:`SAMPLE`, and the reserved spatial
    names :data:`POINT`, :data:`LAT`, :data:`LON`, :data:`Y`, :data:`X`,
    collected in :data:`SPATIAL_DIM_NAMES`.
Coordinates
    :data:`LON` and :data:`LAT` on a site or a point; pySIPNET's timestep
    coordinates :data:`TIMESTEP_START` and :data:`TIMESTEP_LENGTH`, with
    ``time`` collected in :data:`TIME_COORD_NAMES`; an observation's window
    edges :data:`WINDOW_START` and :data:`WINDOW_END`.
Columns
    :data:`SITE_ID`, the site table's key.
Attributes
    :data:`SITE_ATTRIBUTES`, :data:`LON_ATTRIBUTES`, :data:`LAT_ATTRIBUTES`,
    the CF attributes of those coordinates, one wording each;
    :data:`STALE_TIME_ATTRIBUTE_NAMES`, the ``time`` attributes a field drops.
Dtypes and patterns
    :data:`SITE_DTYPE`, the dtype of a site id; :data:`NAME_PATTERN`, what a
    processed name looks like.
Settings
    :data:`CF_CONVENTIONS`, the ``Conventions`` attribute the netCDF files
    declare; :data:`DATA_ROOT_ENV_VAR` and :func:`data_root`, where ``data/``
    is.

Notes
-----
A name or an attribute lives here once two modules have to agree on it, so
that no two of them can spell it differently. Modules import what they need
from here and define none of these values themselves. This module imports
nothing from the package, so everything else can depend on it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import MappingProxyType

import numpy as np
from pysipnet.dataset import TIME_DIMENSION

__all__ = [
    "CF_CONVENTIONS",
    "DATA_ROOT_ENV_VAR",
    "LAT",
    "LAT_ATTRIBUTES",
    "LON",
    "LON_ATTRIBUTES",
    "NAME_PATTERN",
    "POINT",
    "SAMPLE",
    "SITE",
    "SITE_ATTRIBUTES",
    "SITE_DTYPE",
    "SITE_ID",
    "SPATIAL_DIM_NAMES",
    "STALE_TIME_ATTRIBUTE_NAMES",
    "TIME",
    "TIMESTEP_LENGTH",
    "TIMESTEP_START",
    "TIME_COORD_NAMES",
    "WINDOW_END",
    "WINDOW_START",
    "X",
    "Y",
    "data_root",
]


# ── dimensions ────────────────────────────────────────────────────────────────

#: The site dimension and coordinate: the site ids of the site table in use.
SITE = "site"

#: The time dimension, pySIPNET's own name for it.
TIME = TIME_DIMENSION

#: The batch dimension created from the rows of batched Flat: one row of
#: ``theta`` is one sample.
SAMPLE = "sample"

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

# pySIPNET is renaming these two coordinates and will export constants for
# them beside TIME_DIMENSION; these definitions then become imports from
# pysipnet.dataset. Until then they carry pySIPNET's current values.

#: pySIPNET's coordinate on ``time`` for the start of the timestep a row
#: covers; ``time`` is its end.
TIMESTEP_START = "time_step_start"

#: pySIPNET's coordinate on ``time`` for the declared duration of the
#: timestep a row covers, as ``timedelta64``.
TIMESTEP_LENGTH = "time_step_length"

#: pySIPNET's time coordinates, which a model field and a driver field both
#: keep. ``time`` is the end of the timestep and ``time_step_start`` its
#: start, so the two are the CF bounds pair.
TIME_COORD_NAMES: tuple[str, ...] = (TIME, TIMESTEP_START, TIMESTEP_LENGTH)

# These are to be renamed "window_start" and "window_end"; they keep the
# values the processed files' readers write today until that rename.

#: The coordinate on an observation's ``time`` for the start of the window its
#: value covers: the first edge of the CF ``time_bounds`` variable the
#: processed file stores, one-dimensional since a ``DataArray`` cannot carry
#: the two-dimensional ``(time, bounds)`` variable.
#: :func:`sipnet_calibration.constraints.constraint_fields` adds it and
#: :func:`sipnet_calibration.observation.time_alignment.windows_from_time_bounds`
#: reads it.
WINDOW_START = "time_bounds_start"

#: The coordinate on an observation's ``time`` for the end of the window its
#: value covers; see :data:`WINDOW_START`.
WINDOW_END = "time_bounds_end"


# ── columns ───────────────────────────────────────────────────────────────────

#: The site table's key column, and the column every file keyed on sites
#: addresses its records by.
SITE_ID = "site_id"


# ── attributes ────────────────────────────────────────────────────────────────

#: The attributes of a ``site`` coordinate. Read-only; a caller writing them
#: onto a coordinate passes a copy, ``dict(SITE_ATTRIBUTES)``.
SITE_ATTRIBUTES = MappingProxyType(
    {
        "long_name": "Model site identifier",
        "comment": "The handed-down 1-8000 identifier of the site table; never renumbered.",
    }
)

#: The CF attributes of a ``lon`` coordinate. Read-only, as
#: :data:`SITE_ATTRIBUTES`.
LON_ATTRIBUTES = MappingProxyType(
    {"standard_name": "longitude", "long_name": "Longitude", "units": "degrees_east"}
)

#: The CF attributes of a ``lat`` coordinate. Read-only, as
#: :data:`SITE_ATTRIBUTES`.
LAT_ATTRIBUTES = MappingProxyType(
    {"standard_name": "latitude", "long_name": "Latitude", "units": "degrees_north"}
)

#: ``time`` attributes a field does not keep. ``bounds`` names pySIPNET's
#: two-dimensional ``time_bounds`` variable, which a field cannot carry and
#: which describes the source's timesteps, not a coarser one's.
STALE_TIME_ATTRIBUTE_NAMES: tuple[str, ...] = ("bounds",)


# ── dtypes and patterns ───────────────────────────────────────────────────────

#: The dtype of a site id, on a ``site`` coordinate and in the site table.
SITE_DTYPE = np.int32

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
