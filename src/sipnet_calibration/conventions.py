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
Variables
    :data:`TIME_BOUNDS`, the CF bounds variable of ``time`` a processed file
    or pySIPNET's output stores.
Attributes
    :data:`SITE_ATTRIBUTES`, :data:`LON_ATTRIBUTES`, :data:`LAT_ATTRIBUTES`,
    the CF attributes of those coordinates, one wording each;
    :data:`STALE_TIME_ATTRIBUTE_NAMES`, the ``time`` attributes a field drops.
Dtypes and patterns
    :data:`SITE_DTYPE`, the dtype of a site id; :data:`NAME_PATTERN`, what a
    processed name looks like.
Settings
    :data:`CF_CONVENTIONS`, the ``Conventions`` attribute the netCDF files
    declare; :data:`DATA_ROOT_ENV_VAR` and :func:`data_root`, where the
    storage-backed part of ``data/`` is.
Read-only mappings
    :class:`FrozenMapping`, the one read-only mapping type of the package,
    which the attribute dicts above are.

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
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NoReturn

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
    "TIME_BOUNDS",
    "TIME_COORD_NAMES",
    "WINDOW_END",
    "WINDOW_START",
    "FrozenMapping",
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


# ── variables ─────────────────────────────────────────────────────────────────

#: The CF bounds variable of ``time``, ``(time, bounds)``, that a processed
#: file stores where a value's support is documented and pySIPNET's output
#: stores for its timesteps. Its second dimension is pySIPNET's
#: ``pysipnet.dataset.BOUNDS_DIMENSION``.
TIME_BOUNDS = "time_bounds"


# ── columns ───────────────────────────────────────────────────────────────────

#: The site table's key column, and the column every file keyed on sites
#: addresses its records by.
SITE_ID = "site_id"


# ── attributes ────────────────────────────────────────────────────────────────


class FrozenMapping(dict):
    """A dict that cannot be changed after it is built.

    Parameters
    ----------
    items:
        A mapping, or an iterable of key-value pairs, as ``dict`` takes; it is
        copied.

    Raises
    ------
    TypeError
        From every method that would change it (``m[key] = value``, ``del``,
        ``update``, ``pop``, ``popitem``, ``setdefault``, ``clear``, ``|=``),
        and from ``hash`` when a value is unhashable.

    Notes
    -----
    It is a ``dict`` subclass, so pandas builds one column per key from it
    and ``json`` writes it, as they would a dict; it compares equal to a dict
    with the same items. It pickles, copies, and hashes by its items, which
    ``types.MappingProxyType``, the standard read-only view, does not: a
    frozen dataclass holding one could be neither sent to a worker nor used
    as a key. ``dict(m)`` and ``m.copy()`` give an ordinary, mutable dict.
    """

    __slots__ = ()

    def __init__(self, items: Mapping[Any, Any] | Iterable[tuple[Any, Any]] = ()) -> None:
        super().__init__(items)

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __repr__(self) -> str:
        return f"FrozenMapping({dict.__repr__(self)})"

    def __reduce__(self) -> tuple[type[FrozenMapping], tuple[dict[Any, Any]]]:
        return (FrozenMapping, (dict(self),))

    def copy(self) -> dict[Any, Any]:
        """An ordinary, mutable ``dict`` of the same items."""
        return dict(self)

    def _refuse(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(
            "a FrozenMapping cannot be changed; copy it with dict(...) and change the copy."
        )

    __setitem__ = __delitem__ = __ior__ = _refuse
    update = pop = popitem = setdefault = clear = _refuse


#: The attributes of a ``site`` coordinate: read-only, so a coordinate can be
#: given them as they are (xarray copies what it is given).
SITE_ATTRIBUTES = FrozenMapping(
    {
        "long_name": "Model site identifier",
        "comment": "Site identifier of the site table; never renumbered.",
    }
)

#: The CF attributes of a ``lon`` coordinate. Read-only, as
#: :data:`SITE_ATTRIBUTES`.
LON_ATTRIBUTES = FrozenMapping(
    {"standard_name": "longitude", "long_name": "Longitude", "units": "degrees_east"}
)

#: The CF attributes of a ``lat`` coordinate. Read-only, as
#: :data:`SITE_ATTRIBUTES`.
LAT_ATTRIBUTES = FrozenMapping(
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
