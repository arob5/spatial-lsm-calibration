"""Metadata conventions shared by the project's processed products.

Contents
--------
:data:`CF_CONVENTIONS`
    The ``Conventions`` attribute the netCDF products declare.
:data:`DATA_ROOT_ENV_VAR`, :func:`data_root`
    Where ``data/`` is, and the variable that relocates it.
:data:`TIME_LABEL_ATTR`, :data:`INTERVAL_END`
    What a ``time`` label marks, and the one value the project writes so far.

Notes
-----
A constant lives here once it has to agree across products, so that two
ingests cannot declare different values of it. A product's own module imports
what it needs and re-exports it, so a caller reading about the constraints or
the initial conditions still finds the constant beside that product. The
netCDF products -- the constraints and the initial conditions -- are the ones
this currently covers; the site table is a CSV and declares nothing.

:data:`TIME_LABEL_ATTR` is deliberately a pair of strings rather than an
enumeration of every marking a label could have. The constraints already
describe how their records sit in time with
:class:`~sipnet_calibration.constraints.TimeStructure` and a ``time_reference``
sentence, which overlaps such an enumeration without matching it, and settling
one vocabulary against the other is worth doing on evidence from a second
producer rather than in advance of one.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "CF_CONVENTIONS",
    "DATA_ROOT_ENV_VAR",
    "INTERVAL_END",
    "TIME_LABEL_ATTR",
    "data_root",
]

#: The Climate and Forecast conventions the processed netCDFs declare, written
#: to the ``Conventions`` attribute and checked on load. CF governs the
#: coordinate and attribute vocabulary the products use: ``units``,
#: ``long_name``, ``standard_name`` on ``lon``/``lat``, and no ``_FillValue``
#: on a coordinate.
CF_CONVENTIONS = "CF-1.11"

#: The attribute on a ``time`` coordinate saying what its labels mark. A
#: consumer reads it rather than assuming a convention.
TIME_LABEL_ATTR = "time_label"

#: The only value anything here writes: the label is the **end** of the
#: interval the value covers, so a row labeled ``h`` with step ``d`` covers
#: ``(h - d, h]``. The drivers are labeled this way and SIPNET's output
#: inherits it. Other markings exist -- an interval start, an instant, a
#: bookkeeping key that is not a time -- but nothing writes one yet, and the
#: constraints describe their placement with ``TimeStructure`` and
#: ``time_reference`` instead; see the Notes.
INTERVAL_END = "interval_end"

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
