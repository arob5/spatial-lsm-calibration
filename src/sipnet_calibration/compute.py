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
        ``max_concurrent``, ...). ``directives`` given here are appended to
        :data:`SCC_DIRECTIVES`.
    """
    extra = options.pop("directives", ())
    if isinstance(extra, str):
        raise TypeError("directives must be a sequence of strings, not one string.")
    extra = tuple(extra)
    for directive in extra:
        if directive.split()[:1] == ["-P"] or directive.startswith("-l buyin"):
            raise ValueError(
                f"{directive!r} would override the project's queue directives "
                f"{SCC_DIRECTIVES}; qsub takes the last -P it sees."
            )
    directives = (*SCC_DIRECTIVES, f"-v {','.join(SCC_EXPORTED_VARIABLES)}", *extra)
    return GridEngineBackend(
        walltime=walltime,
        work_dir=Path(work_dir),
        n_jobs=n_jobs,
        runs_per_job=runs_per_job,
        directives=directives,
        **options,
    )
