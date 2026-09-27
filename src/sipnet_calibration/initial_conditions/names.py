"""The names and file locations the package's modules share.

The name of the raw file's member dimension, which the shared ones of
:mod:`sipnet_calibration.conventions` do not cover, and where the source tree
and each netCDF are expected on disk. Nothing here reads or writes anything.

Contents
--------
:data:`RAW_MEMBER`
    The member dimension of the raw file. The processed file's dims and
    coordinates are :mod:`sipnet_calibration.conventions`' own.
:data:`RAW_FILE`, :data:`PROCESSED_FILE`
    The two file names, without their directories.
The path functions
    :func:`default_source_root` and :func:`default_processed_path`, which
    honor ``$SIPNET_CALIBRATION_DATA``; and :func:`default_raw_directory` and
    :func:`raw_path`, the tracked raw file's, found as
    :func:`sipnet_calibration.conventions.tracked_data_root` finds it.
"""

from __future__ import annotations

from pathlib import Path

from sipnet_calibration.conventions import data_root, tracked_data_root

__all__ = [
    "PROCESSED_FILE",
    "RAW_FILE",
    "RAW_MEMBER",
    "default_processed_path",
    "default_raw_directory",
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
    return data_root() / "raw" / "initial_conditions" / "files"


def default_raw_directory() -> Path:
    """Where the converted raw file is: ``data/raw/initial_conditions/``.

    It is tracked, so this is found from
    :func:`~sipnet_calibration.conventions.tracked_data_root`: in the checkout
    whatever ``$SIPNET_CALIBRATION_DATA`` says, and under that variable only
    for a non-editable install, which has no checkout.
    """
    return tracked_data_root() / "raw" / "initial_conditions"


def raw_path(directory: Path | str | None = None) -> Path:
    """The converted raw file: ``<directory>/pecan_pool_initial_conditions.nc``.

    *directory* defaults to :func:`default_raw_directory`.
    """
    base = Path(directory) if directory is not None else default_raw_directory()
    return base / RAW_FILE


def default_processed_path() -> Path:
    """The processed file's expected path: ``data/processed/initial_conditions.nc``.

    ``$SIPNET_CALIBRATION_DATA`` replaces ``data/`` when set.
    """
    return data_root() / "processed" / PROCESSED_FILE
