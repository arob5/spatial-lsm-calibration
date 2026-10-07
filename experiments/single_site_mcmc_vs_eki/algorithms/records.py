"""What every run writes, in one format, and what it cost.

A run's directory (``models.Model.directory(algorithm)``) holds:

- ``samples.nc``: the run's weighted samples on a ``sample`` dim:
  ``theta`` ``(sample, theta_entry)``, unconstrained; ``log_weight``, the
  normalized log weight (``-log n`` for an unweighted run); ``valid``;
  ``log_prior`` and ``log_likelihood`` (the model's: the scales' marginal
  where they are inferred); optionally ``predictions`` ``(sample, y_entry)``
  in the posterior's y order, each scaled source's quadratic form
  ``quadratic_form_<source>`` and drawn scale ``<source>_noise_scale``. Its
  attributes name the model, the algorithm and, for a seeded run, the run it
  was seeded from.
- ``natural_values.csv``: each sample's natural values, one row per sample,
  with its ``log_weight``.
- ``cost.json``: the run's phases, each with its SIPNET runs, forward-map
  calls, wall time, workers and node, and their total; a seeded run's
  seed phases are copied from the seed's ``cost.json``.
- ``history.csv`` where the algorithm has stages or steps, and
  ``provenance.json`` (``run/_provenance.py``).
"""

import json
import platform
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import config
from ..model import prior

__all__ = ["Cost", "append_history", "load_samples", "save_samples"]


class Cost:
    """The SIPNET runs, forward-map calls and wall time of a run, by phase.

    Usage: ``forward = cost.counted(forward)``, then ``with cost.phase("eki"):``
    around the work; :meth:`write` writes ``cost.json``.
    """

    def __init__(self) -> None:
        self.phases: list[dict] = []
        self._current: dict | None = None

    def counted(self, forward: Callable) -> Callable:
        """*forward*, counting its rows as SIPNET runs and each call as one call."""

        def counting(theta, *arguments, **keywords):
            if self._current is not None:
                self._current["sipnet_runs"] += int(np.shape(theta)[0])
                self._current["forward_calls"] += 1
            return forward(theta, *arguments, **keywords)

        return counting

    @contextmanager
    def phase(self, name: str):
        """Count the work inside the block as phase *name*."""
        self._current = {"phase": name, "sipnet_runs": 0, "forward_calls": 0}
        start = time.perf_counter()
        try:
            yield self._current
        finally:
            self._current["wall_seconds"] = time.perf_counter() - start
            self._current.update(_machine())
            self.phases.append(self._current)
            self._current = None

    def include(self, path: Path, *, prefix: str) -> None:
        """Copy the phases of another run's ``cost.json`` (a seed), named ``<prefix>:<phase>``."""
        for phase in json.loads(Path(path).read_text())["phases"]:
            self.phases.append({**phase, "phase": f"{prefix}:{phase['phase']}", "source": str(path)})

    def write(self, directory: Path) -> None:
        """``cost.json``: the phases and their total."""
        total = {
            name: sum(phase[name] for phase in self.phases)
            for name in ("sipnet_runs", "forward_calls", "wall_seconds")
        }
        (directory / "cost.json").write_text(
            json.dumps({"phases": self.phases, "total": total}, indent=2)
        )


def save_samples(
    directory: Path,
    posterior,
    theta,
    *,
    model_name: str,
    algorithm: str,
    log_weights=None,
    log_prior=None,
    log_likelihood=None,
    valid=None,
    predictions=None,
    extra: Mapping[str, np.ndarray] | None = None,
    attributes: Mapping[str, str] | None = None,
) -> None:
    """Write ``samples.nc`` and ``natural_values.csv`` for *theta* ``(n, D)``.

    *log_weights* are normalized here; ``None`` is equal weights. *extra*
    holds further per-sample variables, ``(n,)`` each.
    """
    theta = np.asarray(theta, dtype=float)
    n = theta.shape[0]
    log_weights = np.zeros(n) if log_weights is None else np.asarray(log_weights, float)
    log_weights = log_weights - np.logaddexp.reduce(log_weights)
    variables = {
        "theta": (("sample", "theta_entry"), theta),
        "log_weight": ("sample", log_weights),
    }
    for name, values in {
        "log_prior": log_prior,
        "log_likelihood": log_likelihood,
        "valid": valid,
        **(extra or {}),
    }.items():
        if values is not None:
            variables[name] = ("sample", np.asarray(values))
    if predictions is not None:
        variables["predictions"] = (("sample", "y_entry"), np.asarray(predictions, float))
    dataset = xr.Dataset(
        variables,
        coords={"sample": np.arange(n), "theta_entry": np.arange(theta.shape[1])},
        attrs={"model": model_name, "algorithm": algorithm, **(attributes or {})},
    )
    directory.mkdir(parents=True, exist_ok=True)
    # Written beside and then renamed, so a reader never sees half a file
    # (a running MCMC chain rewrites them).
    dataset.to_netcdf(directory / "samples.partial.nc")
    (directory / "samples.partial.nc").replace(directory / "samples.nc")
    natural = prior.natural_table(posterior, theta)
    natural.index = [f"sample_{i}" for i in range(n)]
    natural["log_weight"] = log_weights
    natural.to_csv(directory / "natural_values.partial.csv")
    (directory / "natural_values.partial.csv").replace(directory / "natural_values.csv")


def load_samples(directory: Path) -> xr.Dataset:
    """A run's ``samples.nc``, loaded."""
    with xr.open_dataset(Path(directory) / "samples.nc") as dataset:
        return dataset.load()


def append_history(directory: Path, row: Mapping) -> None:
    """One row of ``history.csv``."""
    path = directory / "history.csv"
    pd.DataFrame([dict(row)]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _machine() -> dict:
    """The workers and the node a phase ran on."""
    return {"workers": config.N_WORKERS, "node": platform.node(), "processor": _processor()}


def _processor() -> str:
    """The CPU model, from ``/proc/cpuinfo`` where there is one."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()
