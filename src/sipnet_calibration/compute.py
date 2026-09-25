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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pyens import GridEngineBackend

__all__ = ["SCC_DIRECTIVES", "SCC_EXPORTED_VARIABLES", "scc_backend"]

#: The queue every SCC job of this project goes to.
SCC_DIRECTIVES: tuple[str, ...] = ("-P dietzelab", "-l buyin")

#: Environment variables a compute node needs to find the SIPNET binary and the
#: data; exported into every task from the submitting shell.
SCC_EXPORTED_VARIABLES: tuple[str, ...] = (
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
        strings, are appended to the project's.

    Returns
    -------
    pyens.GridEngineBackend
        With the directives :data:`SCC_DIRECTIVES`, then a ``-v`` directive
        exporting :data:`SCC_EXPORTED_VARIABLES` from the submitting shell,
        then any given in *options*.

    Raises
    ------
    TypeError
        If ``directives`` is not a sequence of strings (one string, ``None``,
        or an entry that is not a string).
    ValueError
        If a directive names a project (``-P``) or sets the ``buyin``
        resource, either of which would override the project's queue.
    """
    extra = options.pop("directives", ())
    check_directives_are_strings(extra)
    for directive in extra:
        check_directive_keeps_the_project(directive)
        check_directive_keeps_buyin(directive)
    directives = (*SCC_DIRECTIVES, f"-v {','.join(SCC_EXPORTED_VARIABLES)}", *extra)
    return GridEngineBackend(
        walltime=walltime,
        work_dir=Path(work_dir),
        n_jobs=n_jobs,
        runs_per_job=runs_per_job,
        directives=directives,
        **options,
    )


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


def check_directive_keeps_the_project(directive: str) -> None:
    option = directive.split()[:1]
    if option and option[0].startswith("-P"):
        raise ValueError(
            f"{directive!r} names a project, which would override the project's "
            f"{SCC_DIRECTIVES[0]!r}: qsub takes the last -P it sees. Drop it; every job of "
            "this project goes to dietzelab."
        )


def check_directive_keeps_buyin(directive: str) -> None:
    tokens = directive.split()
    if not tokens or not tokens[0].startswith("-l"):
        return
    resources = ",".join([tokens[0][2:], *tokens[1:]]).split(",")
    if any(r.split("=", 1)[0].strip() == "buyin" for r in resources):
        raise ValueError(
            f"{directive!r} sets the buyin resource, which would override the project's "
            f"{SCC_DIRECTIVES[1]!r}. Drop it; every job of this project runs on the buy-in "
            "nodes."
        )
