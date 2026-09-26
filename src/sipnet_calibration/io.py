"""Writing a file safely, and the provenance its writer records.

Contents
--------
:func:`write_checked`
    Write a file to a ``.partial`` path beside its destination, check it, and
    only then move it into place.
:func:`write_checked_together`
    The same for several files that belong together, so that none is moved
    into place unless every one was written and checked.
:func:`partial_path`, :data:`PARTIAL_SUFFIX`, :data:`FileStep`
    The ``.partial`` path the two write to, its suffix, and the type of a
    write or a check.
:func:`file_md5`
    The md5 digest of a file, for a provenance record.
:func:`utc_timestamp`
    Now, as the ISO 8601 UTC string a file's attributes record.

Every script that writes a processed file, a tracked raw input or a generated
definition writes it through :func:`write_checked` or
:func:`write_checked_together`, so they all follow one protocol and one
failure policy.

Notes
-----
**The failure policy.** A stale ``.partial`` left by an earlier run is removed
before anything is written, so a ``.partial`` that exists after a run is this
run's. When a write or a check fails, the ``.partial`` files this run wrote are
kept and their paths printed to standard error, and the error is raised. A
kept file cannot be mistaken for the processed file, since its name differs,
and it is what the failed check was looking at, so it is kept for inspection
rather than deleted; a rerun replaces it. A directory the run made for a
destination is removed again if the failure leaves it empty. The destinations
are not touched: each holds its previous good file, or nothing -- unless a
write wrote a destination itself rather than the ``.partial`` path it was
given, which is caught before any check runs and reported as such.

The final rename is :meth:`pathlib.Path.replace`, which is atomic on a POSIX
filesystem, so each destination is at every moment either its previous file
or a fully checked new one.

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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "PARTIAL_SUFFIX",
    "FileStep",
    "file_md5",
    "partial_path",
    "utc_timestamp",
    "write_checked",
    "write_checked_together",
]

#: The suffix appended to a destination's name for the file written first.
PARTIAL_SUFFIX = ".partial"

#: How much of a file :func:`file_md5` reads at once, in bytes.
_MD5_CHUNK_SIZE = 1 << 20

#: What writes a file, and what checks it: each is called with the
#: ``.partial`` path.
FileStep = Callable[[Path], object]


def write_checked(path: Path | str, write: FileStep, check: FileStep) -> Path:
    """Write *path* through a ``.partial`` file, checked before it is moved in.

    Parameters
    ----------
    path:
        The destination. Its directory is created if absent, and removed
        again if a failure leaves it empty.
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
    FileNotFoundError
        If *write* did not write the ``.partial`` file.
    ValueError
        If *write* changed *path* itself.
    BaseException
        Whatever *write*, *check* or the final rename raises, an interrupt
        included. The ``.partial`` file, if this run wrote one, is then kept
        and its path printed to standard error, and *path* is left as it was
        unless *write* wrote it.
    """
    return write_checked_together([(path, write, check)])[0]


def write_checked_together(files: Iterable[tuple[Path | str, FileStep, FileStep]]) -> list[Path]:
    """Write several files that belong together through ``.partial`` files.

    Every file is written to its ``.partial`` path, then every one is
    checked, and only then are they moved into place, in the order given.

    Parameters
    ----------
    files:
        ``(path, write, check)`` for each file, as :func:`write_checked`
        takes them. A check only reads its file.

    Returns
    -------
    list of pathlib.Path
        The destinations, in the order given, each now holding its checked
        file.

    Raises
    ------
    ValueError
        If two files have one destination, before anything is written; or if
        a write changed a destination rather than only its ``.partial`` file.
    FileNotFoundError
        If a write did not write its ``.partial`` file.
    BaseException
        Whatever a write, a check or a rename raises, an interrupt included.

    Notes
    -----
    **All or nothing for a failed write or check.** If any write or check
    fails, no destination is touched: each keeps its previous file, or
    nothing, and the ``.partial`` files this run wrote are kept and their
    paths printed. Every write is checked to have written its ``.partial``
    file and left its destination alone before any check runs; a write that
    wrote its destination has already replaced it, and the report says so
    rather than that it is unchanged.

    **A failed rename is reported.** The renames are one per file, so they
    cannot be made atomic together. If the rename of one file fails, the
    files before it in *files* hold the new content, it and the files after
    it hold their previous content, and their ``.partial`` files are kept;
    standard error says which is which, and a rerun writes them together
    again.
    """
    staged = [_StagedFile(Path(path), write, check) for path, write, check in files]
    check_destinations_are_distinct([file.destination for file in staged])
    made = _make_directories([file.destination.parent for file in staged])
    for file in staged:
        # A stale partial from an earlier run would otherwise be reported as
        # this run's, or be what a check validates if a write wrote nothing.
        file.partial.unlink(missing_ok=True)
    before = {file.destination: _file_state(file.destination) for file in staged}
    try:
        for file in staged:
            file.write(file.partial)
        for file in staged:
            check_write_left_the_destination_alone(
                file.destination, before[file.destination], partial=file.partial
            )
            check_partial_was_written(file.partial)
        for file in staged:
            file.check(file.partial)
    except BaseException:
        _report_kept_partials(staged, before)
        _remove_empty_directories(made)
        raise
    moved: list[_StagedFile] = []
    try:
        for file in staged:
            file.partial.replace(file.destination)
            moved.append(file)
    except BaseException:
        _report_failed_move(staged, moved, before)
        raise
    return [file.destination for file in staged]


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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── supporting helpers ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _StagedFile:
    """One file of a :func:`write_checked_together` call."""

    destination: Path
    write: FileStep
    check: FileStep

    @property
    def partial(self) -> Path:
        return partial_path(self.destination)


#: What a destination was before the writes: its inode, size and modification
#: time, or ``None`` when it did not exist.
_FileState = tuple[int, int, int] | None


def _file_state(path: Path) -> _FileState:
    """*path*'s inode, size and modification time, or ``None`` if it does not exist."""
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_ino, status.st_size, status.st_mtime_ns


def _make_directories(directories: list[Path]) -> list[Path]:
    """Create each of *directories* that is absent; the ones made, deepest first."""
    made: list[Path] = []
    for directory in directories:
        missing = [d for d in (directory, *directory.parents) if not d.exists()]
        directory.mkdir(parents=True, exist_ok=True)
        made.extend(d for d in missing if d not in made)
    return sorted(made, key=lambda d: len(d.parts), reverse=True)


def _remove_empty_directories(directories: list[Path]) -> None:
    """Remove each of *directories*, deepest first, that a failure left empty."""
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue


def _report_kept_partials(staged: list[_StagedFile], before: dict[Path, _FileState]) -> None:
    """Print where each ``.partial`` this run wrote is, and what became of its destination."""
    for file in staged:
        changed = _file_state(file.destination) != before.get(file.destination)
        if file.partial.exists():
            status = (
                f"{file.destination} was changed by the write too and no longer holds its "
                "previous content"
                if changed
                else f"{file.destination} is unchanged"
            )
            print(
                f"kept the partial file for inspection: {file.partial}; {status}, and a "
                "rerun overwrites the partial file.",
                file=sys.stderr,
            )
        elif changed:
            print(
                f"the write wrote {file.destination} rather than {file.partial}, so "
                f"{file.destination} no longer holds its previous content; rerun once the "
                "write writes the path it is given.",
                file=sys.stderr,
            )


def _report_failed_move(
    staged: list[_StagedFile], moved: list[_StagedFile], before: dict[Path, _FileState]
) -> None:
    """Print which destinations were replaced before a rename failed, and which were not."""
    _report_kept_partials([file for file in staged if file not in moved], before)
    if moved:
        print(
            "the files were not all moved into place: "
            f"{[str(file.destination) for file in moved]} hold the new content and the "
            "rest their previous content; rerun to write them together.",
            file=sys.stderr,
        )


# ── checks ────────────────────────────────────────────────────────────────────


def check_destinations_are_distinct(destinations: list[Path]) -> None:
    """No destination is given twice, which would move one file over the other."""
    seen: set[Path] = set()
    for destination in destinations:
        resolved = destination.resolve()
        if resolved in seen:
            raise ValueError(
                f"write_checked_together was given {destination} more than once; give "
                "each file once."
            )
        seen.add(resolved)


def check_write_left_the_destination_alone(
    destination: Path, before: _FileState, *, partial: Path
) -> None:
    """A write changed only its ``.partial`` file, not its destination."""
    if _file_state(destination) != before:
        raise ValueError(
            f"the write changed {destination}; a write writes only the path it is "
            f"given, {partial}, which is moved into place once it is checked."
        )


def check_partial_was_written(partial: Path) -> None:
    """A write wrote its ``.partial`` file."""
    if not partial.is_file():
        raise FileNotFoundError(
            f"the write did not write {partial}; a write writes the whole file to the "
            "path it is given."
        )
