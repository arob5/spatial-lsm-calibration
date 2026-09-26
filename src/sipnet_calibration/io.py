"""Writing a file safely, and the provenance its writer records.

Contents
--------
:func:`write_checked`
    Write a file to a ``.partial`` path beside its destination, check it, and
    only then move it into place.
:func:`partial_path`
    The ``.partial`` path :func:`write_checked` writes to.
:func:`file_md5`
    The md5 digest of a file, for a provenance record.
:func:`utc_timestamp`
    Now, as the ISO 8601 UTC string a file's attributes record.

Every script that writes a processed file, a tracked raw input or a generated
definition writes it through :func:`write_checked`, so they all follow one
protocol and one failure policy.

Notes
-----
**The failure policy.** When the write or the check fails, the ``.partial``
file is kept and its path printed to standard error, and the error is raised.
It cannot be mistaken for the processed file, since its name differs, a rerun
overwrites it, and it is what a failed check was looking at, so it is kept for
inspection rather than deleted. The destination is never touched: it holds the
previous good file, or nothing.

The final rename is :meth:`pathlib.Path.replace`, which is atomic on a POSIX
filesystem, so the destination is at every moment either the previous file or
a fully checked new one.

Usage
-----
::

    from sipnet_calibration.io import write_checked

    write_checked(
        out,
        write=lambda partial: dataset.to_netcdf(partial),
        check=lambda partial: check_round_trip(dataset, partial),
    )
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

__all__ = [
    "PARTIAL_SUFFIX",
    "file_md5",
    "partial_path",
    "utc_timestamp",
    "write_checked",
]

#: The suffix appended to a destination's name for the file written first.
PARTIAL_SUFFIX = ".partial"

#: How much of a file :func:`file_md5` reads at once, in bytes.
_MD5_CHUNK_SIZE = 1 << 20


def write_checked(
    path: Path | str,
    write: Callable[[Path], object],
    check: Callable[[Path], object],
) -> Path:
    """Write *path* through a ``.partial`` file, checked before it is moved in.

    Parameters
    ----------
    path:
        The destination. Its directory is created if absent.
    write:
        Called with the ``.partial`` path; writes the whole file there.
    check:
        Called with the ``.partial`` path once it is written; raises if the
        file is not what it should be, typically by reading it back through
        the library's loader.

    Returns
    -------
    pathlib.Path
        *path*, now holding the checked file.

    Raises
    ------
    Exception
        Whatever *write*, *check* or the final rename raises. The ``.partial``
        file is then kept, its path printed to standard error, and *path* is
        left as it was.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(destination)
    try:
        write(partial)
        check(partial)
        partial.replace(destination)
    except BaseException:
        if partial.exists():
            print(
                f"kept the partial file for inspection: {partial}; {destination} is "
                "unchanged, and a rerun overwrites the partial file.",
                file=sys.stderr,
            )
        raise
    return destination


def partial_path(path: Path | str) -> Path:
    """The ``.partial`` file :func:`write_checked` writes before *path*."""
    path = Path(path)
    return path.with_name(path.name + PARTIAL_SUFFIX)


def file_md5(path: Path | str) -> str:
    """The md5 digest of a file, as hexadecimal, read a chunk at a time."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_MD5_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp() -> str:
    """Now, as the ISO 8601 UTC string a file's attributes carry."""
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
