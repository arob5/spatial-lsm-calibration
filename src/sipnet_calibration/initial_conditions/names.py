"""Dimension names and file locations the package's modules share.

The names of the dimensions the two netCDFs use beyond the shared ones of
:mod:`sipnet_calibration.conventions`, where the source tree and each netCDF
are expected on disk. Nothing here reads or writes anything.

Contents
--------
:data:`RAW_MEMBER`
    The member dimension of the raw file, spelled once. The processed
    file's is
    :data:`sipnet_calibration.conventions.INITIAL_CONDITION_MEMBER`, the site
    dimension :data:`sipnet_calibration.conventions.SITE`, and the 1-based
    source index beside the processed member
    :data:`sipnet_calibration.conventions.SOURCE_INDEX`.
:data:`RAW_FILE`, :data:`PROCESSED_FILE`
    The two file names, without their directories.
The four path functions
    :func:`default_source_root`, :func:`default_raw_dir`, :func:`raw_path` and
    :func:`default_processed_path` say where each is expected, all honoring
    ``$SIPNET_CALIBRATION_DATA``.
"""

from __future__ import annotations

from pathlib import Path

from sipnet_calibration import conventions

__all__ = [
    "PROCESSED_FILE",
    "RAW_FILE",
    "RAW_MEMBER",
    "default_processed_path",
    "default_raw_dir",
    "default_source_root",
    "raw_path",
]


#: The raw file's ensemble dim, holding the source files' 1-based index. The
#: raw file is never edited, so it keeps this name;
#: :func:`~sipnet_calibration.initial_conditions.processed.build_initial_conditions`
#: renames it on the way in.
RAW_MEMBER = "member"

#: The converted raw file, under ``data/raw/initial_conditions/``.
RAW_FILE = "pecan_pool_initial_conditions.nc"

#: The processed file, under ``data/processed/``.
PROCESSED_FILE = "initial_conditions.nc"


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


def default_processed_path() -> Path:
    """Where the processed file is expected: ``data/processed/initial_conditions.nc``."""
    return conventions.data_root() / "processed" / PROCESSED_FILE
