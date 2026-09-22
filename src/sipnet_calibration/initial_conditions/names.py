"""Dimension names, file locations, and the few helpers every module shares.

The names of the dimensions and coordinates the two netCDFs use, where each
file is expected on disk, and the small pieces -- the data root, a timestamp,
the site coordinate's attributes -- that more than one of the package's
modules needs. Nothing here reads or writes anything.

Contents
--------
:data:`SITE`, :data:`MEMBER`, :data:`SOURCE_MEMBER`
    The dimension and coordinate names, spelled once.
:data:`RAW_FILE`, :data:`PRODUCT_FILE`
    The two file names, without their directories.
:func:`default_source_root`, :func:`default_raw_dir`, :func:`raw_path`,
:func:`default_product_path`
    Where each is expected, all honoring ``$SIPNET_CALIBRATION_DATA``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from sipnet_calibration.sites import DATA_ROOT_ENV_VAR

__all__ = [
    "MEMBER",
    "PRODUCT_FILE",
    "RAW_FILE",
    "SITE",
    "SOURCE_MEMBER",
    "default_product_path",
    "default_raw_dir",
    "default_source_root",
    "raw_path",
]


#: Dimension and coordinate names.
SITE = "site"

MEMBER = "member"

SOURCE_MEMBER = "source_member"

#: The converted raw file and the processed product, under ``data/``.
RAW_FILE = "pecan_pool_initial_conditions.nc"

PRODUCT_FILE = "initial_conditions.nc"


def default_source_root() -> Path:
    """Where the source file tree is expected: ``data/raw/initial_conditions/files``.

    Present only on the SCC, as a symlink. ``$SIPNET_CALIBRATION_DATA``
    replaces ``data/`` when set.
    """
    return default_raw_dir() / "files"


def default_raw_dir() -> Path:
    """Where the converted raw file lives: ``data/raw/initial_conditions/``."""
    return _data_root() / "raw" / "initial_conditions"


def raw_path(directory: Path | str | None = None) -> Path:
    """The converted raw file: ``<directory>/pecan_pool_initial_conditions.nc``."""
    base = Path(directory) if directory is not None else default_raw_dir()
    return base / RAW_FILE


def default_product_path() -> Path:
    """Where the processed product is expected: ``data/processed/initial_conditions.nc``."""
    return _data_root() / "processed" / PRODUCT_FILE


#: CF attributes for the coordinates, written by both build_raw and
#: build_initial_conditions.
_SITE_ATTRS = {
    "long_name": "Model site identifier",
    "comment": "The handed-down 1-8000 identifier of the site table; never renumbered.",
}


#: The package directory, ``src/sipnet_calibration``. The data root is found
#: from here rather than by counting parents from this file, so that moving a
#: module deeper into the package cannot silently retarget it -- which is what
#: happened when ``initial_conditions.py`` became ``initial_conditions/``.
_PACKAGE_DIRECTORY = Path(__file__).resolve().parents[1]


def _data_root() -> Path:
    """``data/`` beside the package's ``src``, or ``$SIPNET_CALIBRATION_DATA``."""
    root = os.environ.get(DATA_ROOT_ENV_VAR)
    return Path(root) if root else _PACKAGE_DIRECTORY.parents[1] / "data"


def _utc_timestamp() -> str:
    """Now, as the ISO 8601 string the file attributes carry."""
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
