"""Dimension names and file locations the package's modules share.

The names of the dimensions and coordinates the two netCDFs use beyond the
shared ones of :mod:`sipnet_calibration.conventions`, where the source tree
and each netCDF are expected on disk. Nothing here reads or writes anything.

Contents
--------
:data:`MEMBER`, :data:`SOURCE_MEMBER`
    The member dimension and coordinate names, spelled once. The site
    dimension is :data:`sipnet_calibration.conventions.SITE`.
:data:`RAW_FILE`, :data:`PRODUCT_FILE`
    The two file names, without their directories.
The four path functions
    :func:`default_source_root`, :func:`default_raw_dir`, :func:`raw_path` and
    :func:`default_product_path` say where each is expected, all honoring
    ``$SIPNET_CALIBRATION_DATA``.
"""

from __future__ import annotations

from pathlib import Path

from sipnet_calibration import conventions

__all__ = [
    "MEMBER",
    "PRODUCT_FILE",
    "RAW_FILE",
    "SOURCE_MEMBER",
    "default_product_path",
    "default_raw_dir",
    "default_source_root",
    "raw_path",
]


#: The ensemble member dimension, 0-based in the product.
MEMBER = "member"

#: The coordinate on ``member`` holding the source files' 1-based index.
SOURCE_MEMBER = "source_member"

#: The converted raw file, under ``data/raw/initial_conditions/``.
RAW_FILE = "pecan_pool_initial_conditions.nc"

#: The processed product, under ``data/processed/``.
PRODUCT_FILE = "initial_conditions.nc"


def default_source_root() -> Path:
    """Where the source file tree is expected: ``data/raw/initial_conditions/files``.

    Present only on the SCC, as a symlink. ``$SIPNET_CALIBRATION_DATA``
    replaces ``data/`` when set.
    """
    return default_raw_dir() / "files"


def default_raw_dir() -> Path:
    """Where the converted raw file lives: ``data/raw/initial_conditions/``."""
    return conventions.data_root() / "raw" / "initial_conditions"


def raw_path(directory: Path | str | None = None) -> Path:
    """The converted raw file: ``<directory>/pecan_pool_initial_conditions.nc``."""
    base = Path(directory) if directory else default_raw_dir()
    return base / RAW_FILE


def default_product_path() -> Path:
    """Where the processed product is expected: ``data/processed/initial_conditions.nc``."""
    return conventions.data_root() / "processed" / PRODUCT_FILE
