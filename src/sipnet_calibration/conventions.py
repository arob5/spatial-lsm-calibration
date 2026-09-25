"""Metadata conventions shared by the project's processed products.

Contents
--------
:data:`CF_CONVENTIONS`
    The ``Conventions`` attribute the netCDF products declare.
:data:`DATA_ROOT_ENV_VAR`, :func:`data_root`
    Where ``data/`` is, and the variable that relocates it.
:data:`TIME_BOUNDS_START`, :data:`TIME_BOUNDS_END`
    The names of the one-dimensional coordinates a field carries for CF
    ``time_bounds``, written by the constraints and read by the observation
    operators.

Notes
-----
A constant lives here once it has to agree across products, so that two
ingests cannot declare different values of it. A product's own module imports
what it needs and re-exports it, so a caller reading about the constraints or
the initial conditions still finds the constant beside that product. The
netCDF products -- the constraints and the initial conditions -- are the ones
this currently covers; the site table is a CSV and declares nothing. The
time-bounds coordinate names are here because a product writes them and the
observation operators read them.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "CF_CONVENTIONS",
    "DATA_ROOT_ENV_VAR",
    "TIME_BOUNDS_END",
    "TIME_BOUNDS_START",
    "data_root",
]

#: The Climate and Forecast conventions the processed netCDFs declare, written
#: to the ``Conventions`` attribute and checked on load. CF governs the
#: coordinate and attribute vocabulary the products use: ``units``,
#: ``long_name``, ``standard_name`` on ``lon``/``lat``, and no ``_FillValue``
#: on a coordinate.
CF_CONVENTIONS = "CF-1.11"

#: The coordinates a field carries for the interval each of its values is
#: attributed to, one-dimensional on ``time``: CF ``time_bounds`` split into
#: its two edges, since a ``DataArray`` cannot carry the two-dimensional
#: ``(time, bounds)`` variable.
#: :func:`sipnet_calibration.constraints.constraint_fields` writes them and
#: :func:`sipnet_calibration.observation.time_alignment.windows_from_time_bounds`
#: reads them.
TIME_BOUNDS_START = "time_bounds_start"
TIME_BOUNDS_END = "time_bounds_end"

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
