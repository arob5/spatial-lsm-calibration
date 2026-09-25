"""Where the runs execute: the one place the group's SCC queue settings live.

:func:`scc_backend` builds a PyEns ``GridEngineBackend`` for Boston
University's SCC with the project's queue directives and the environment
variables a SIPNET run needs, so an experiment picks a backend by name rather
than restating them::

    COMPUTE = {
        "laptop": LocalBackend(n_workers=8),
        "scc": scc_backend(walltime="00:30:00", n_jobs=50, work_dir=BATCH_DIR),
    }

``TMPDIR`` is left to Grid Engine, which gives each task node-local scratch,
where pySIPNET puts a run's working directory. Every ``map`` on this backend
pays one queue wait, which suits an EKI step (one map per iteration) and not
an MCMC step (one map per proposal); run a sampler locally.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from itertools import takewhile
from pathlib import Path
from typing import Any

from pyens import GridEngineBackend

__all__ = ["SCC_DIRECTIVES", "SCC_EXPORTED_VARIABLE_NAMES", "scc_backend"]

#: The queue every SCC job of this project goes to.
SCC_DIRECTIVES: tuple[str, ...] = ("-P dietzelab", "-l buyin")

#: Environment variables a compute node needs to find the SIPNET binary and the
#: data; exported into every task from the submitting shell.
SCC_EXPORTED_VARIABLE_NAMES: tuple[str, ...] = (
    "PYSIPNET_CACHE_DIR",
    "PYSIPNET_BINARY",
    "SIPNET_CALIBRATION_DATA",
)


def scc_backend(
    *,
    walltime: str | int,
    work_dir: str | Path,
    n_jobs: int | None = None,
    runs_per_job: int | None = None,
    **options: Any,
) -> GridEngineBackend:
    """A ``GridEngineBackend`` for the SCC with the project's directives.

    Parameters
    ----------
    walltime, work_dir, n_jobs, runs_per_job:
        As PyEns's ``GridEngineBackend``; exactly one of *n_jobs* and
        *runs_per_job* is given.
    **options:
        Any other ``GridEngineBackend`` field (``slots``, ``setup``,
        ``max_concurrent``, ...). ``directives`` given here, a sequence of
        strings, are appended to the project's. PyEns writes each string as
        one ``#$`` line, so one string may hold several qsub options.

    Returns
    -------
    pyens.GridEngineBackend
        With the directives :data:`SCC_DIRECTIVES`, then a ``-v`` directive
        exporting :data:`SCC_EXPORTED_VARIABLE_NAMES` from the submitting
        shell, then any given in *options*.

    Raises
    ------
    TypeError
        If ``directives`` is not a sequence of strings (one string, a set,
        ``None``, or an entry that is not a string).
    ValueError
        If an option of any directive names a project (``-P``, attached or
        separate) or a ``-l`` resource list names ``buyin`` (in any case),
        either of which would override the project's queue; if a directive
        does not split into shell words (an unbalanced quote); or, from
        ``GridEngineBackend``, if both or neither of *n_jobs* and
        *runs_per_job* are given.
    """
    extra = options.pop("directives", ())
    check_directives_are_strings(extra)
    for directive in extra:
        check_directive_splits_into_words(directive)
        qsub_options = shlex.split(directive)
        check_directive_keeps_the_project(directive, qsub_options)
        check_directive_keeps_buyin(directive, qsub_options)
    directives = (*SCC_DIRECTIVES, f"-v {','.join(SCC_EXPORTED_VARIABLE_NAMES)}", *extra)
    return GridEngineBackend(
        walltime=walltime,
        work_dir=Path(work_dir),
        n_jobs=n_jobs,
        runs_per_job=runs_per_job,
        directives=directives,
        **options,
    )


# ── supporting helpers ────────────────────────────────────────────────────────


def _resource_names(qsub_options: Sequence[str]) -> list[str]:
    """The lower-cased resource names every ``-l`` among *qsub_options* requests.

    A ``-l`` takes its list attached (``-lbuyin``) or as the next word, and
    every word up to the next option is read as part of it, so a list split
    at a comma and a space is still read whole.
    """
    names: list[str] = []
    for position, word in enumerate(qsub_options):
        if not word.startswith("-l"):
            continue
        following = takewhile(lambda w: not w.startswith("-"), qsub_options[position + 1 :])
        for resource_list in (word[2:], *following):
            names.extend(r.split("=", 1)[0].strip().lower() for r in resource_list.split(","))
    return names


# ── checks ────────────────────────────────────────────────────────────────────


def check_directives_are_strings(directives: Any) -> None:
    if isinstance(directives, str):
        raise TypeError("directives must be a sequence of strings, not one string.")
    if not isinstance(directives, Sequence):
        raise TypeError(
            "directives must be a sequence of strings such as ('-l mem_per_core=4G',), got "
            f"{type(directives).__name__}."
        )
    wrong = [d for d in directives if not isinstance(d, str)]
    if wrong:
        raise TypeError(f"every entry of directives must be a string, got {wrong[:5]!r}.")


def check_directive_splits_into_words(directive: str) -> None:
    try:
        shlex.split(directive)
    except ValueError as error:
        raise ValueError(
            f"{directive!r} does not split into shell words ({error}), so its qsub options "
            "cannot be read; balance its quotes."
        ) from None


def check_directive_keeps_the_project(directive: str, qsub_options: Sequence[str]) -> None:
    # Only -P names a project; qsub's lowercase -p is the job's priority.
    if any(word.startswith("-P") for word in qsub_options):
        raise ValueError(
            f"{directive!r} names a project, which would override the project's "
            f"{SCC_DIRECTIVES[0]!r}: qsub takes the last -P it sees. Drop it; every job of "
            "this project goes to dietzelab."
        )


def check_directive_keeps_buyin(directive: str, qsub_options: Sequence[str]) -> None:
    if "buyin" in _resource_names(qsub_options):
        raise ValueError(
            f"{directive!r} sets the buyin resource, which would override the project's "
            f"{SCC_DIRECTIVES[1]!r}. Drop it; every job of this project runs on the buy-in "
            "nodes."
        )
