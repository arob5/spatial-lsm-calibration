"""The forward model: unconstrained calibration parameters to predictions, over
a site set, with SIPNET run once per member and site.

Where this sits
---------------
::

    theta (J, D)  --ParameterVector.sipnet_table-->  SIPNET table (member, site)
                  --PyEns, one SIPNETModel run per (member, site)-->  model output
                  --ObservationVector.predict on the worker-->  predictions per site
                  --ObservationVector.flat on the driver-->  (J, N)

:class:`ForwardModel` is the one object this module adds: the callable
``(J, D) -> (J, N)`` that pyEKI's ``run`` takes as ``forward``, and that an
MCMC target calls. Everything it composes exists elsewhere: the parameter
vector (:mod:`sipnet_calibration.parameter_vector`), pySIPNET's
``SIPNETModel``, PyEns's ``PartialSpec``/``EnsembleRunner``/``Backend`` and
its xarray bridge, and the observation vector
(:mod:`sipnet_calibration.observation`). It adds no ensemble or run class of
its own.

What it reads
-------------
A :class:`pysipnet.model.SIPNETModel`, a
:class:`~sipnet_calibration.parameter_vector.ParameterVector`, one
``ClimateDrivers`` per site, a PyEns backend, and either an
:class:`~sipnet_calibration.observation.ObservationVector` (the calibration
path: each run reduced to its site's predictions on the worker) or the
output variable names each run should return (the prior-predictive path).
:class:`ForwardModel`'s ``Parameters`` say what each is for.

Data model
----------
:class:`ForwardEvaluation`, what one call of :meth:`ForwardModel.evaluate`
produced:

============== ==========================================================
``theta``      ``(B, D)`` float64, coerced from what was received (a ``(D,)``
               input is one row)
``sipnet_table`` the SIPNET table that was run, ``(member, site)``
``model_output`` ``xr.Dataset`` on ``(member, site, time)`` of the output
               variables, ``NaN`` where a run failed; ``None`` when an
               observation vector was given
``predictions`` ``(B, N)`` float64 in the observation vector's order, ``NaN``
               at every cell of a member with a failed run; ``None``
               without an observation vector
``run_succeeded`` bool ``(member, site)``
``failures``   a frame with ``member``, ``site``, ``error`` and ``message``,
               one row per failed run
``valid``      bool ``(B,)``: every run of the member succeeded and, where
               there are predictions, they are finite
============== ==========================================================

A run **fails at its parameters** when pySIPNET refuses them
(``pydantic.ValidationError``), SIPNET exits non-zero or writes nothing
(``SIPNETRunError``), the run times out (``subprocess.TimeoutExpired``), or
the output it wrote holds a non-finite value in a variable that was read
(:class:`ModelOutputNotFinite`, a blow-up SIPNET exits 0 on): the member's
row is ``NaN``, which pyEKI repairs and a sampler rejects. Anything else that
comes back from a worker (a PyEns ``TaskFailedError``, a ``RemoteError`` of
another type, a missing binary, an import error) is the **machinery**
failing, says nothing about the parameters, and is raised after the batch is
collected as a ``RuntimeError`` whose ``evaluation`` attribute holds what was
collected (``run_succeeded`` and ``failures`` filled, no predictions or
output), for diagnosis. On the prior-predictive path a batch in which every
run failed at its parameters is also raised, since there is no time axis to
stack NaN onto.

Functions
---------
:class:`ForwardModel`
    ``evaluate(theta) -> ForwardEvaluation``; ``__call__(theta)`` returns
    ``evaluate(theta).predictions``, ``(N,)`` for a ``(D,)`` input.
:class:`ForwardEvaluation`
    The record above.
:data:`MODEL_FAILURES`, :class:`ModelOutputNotFinite`
    The exceptions that mean a run failed at its parameters.

Notes
-----
**The ``PartialSpec`` is built once.** Its fixed inputs, the climate, the
site id and the site's slice of the observation vector along one site axis,
and its free fields, the SIPNET parameter names the vector sets, hold for the
lifetime of the model, and the site axis is what every call's parameter grids
zip with. The free names are learned by mapping one prior draw through
``sipnet_table`` in ``__init__``, which fails fast, before anything is
queued, on a hook that does not return a ``(member, site)`` table over the
vector's sites in pySIPNET's parameter names.

**The operators run on the worker** because reading a run's output back on
the driver costs more than the run (pySIPNET parses the whole file), and an
observation operator is pointwise in site, so each worker can reduce its own
run to the observed cells. Each run receives only its site's slice of the
observation vector, as an input field along the site axis, so a run at a
site no product observes carries nothing and returns nothing, and a
process backend pickles a few kilobytes per run rather than the whole
vector. The driver only places small arrays.

**Runs on different time axes** (sites with driver records of different
lengths) are stacked by an outer join on the prior-predictive path, so the
shorter records are ``NaN``-padded and the interval coordinates gain a
``site`` or ``member`` dimension; select one site before aggregating such
a stack, as :func:`~sipnet_calibration.observation.aggregate_time` asks.

**Parameters an operator reads but the vector does not set** (a fixed
SIPNET parameter, say ``leaf_carbon_per_area`` for the LAI operator) are
taken from the model's base parameter set once, in ``__init__``, and handed
to every run's operators beside the run's own overrides.

Usage
-----
::

    import pyeki.eki
    from pyens import LocalBackend
    from sipnet_calibration.forward import ForwardModel

    forward = ForwardModel(model, vector, climate=climate, backend=LocalBackend(8),
                           observation_vector=observations)
    result = pyeki.eki.run(state, forward, observations.y, noise_cov, schedule=...)

    prior = ForwardModel(model, vector, climate=climate, backend=LocalBackend(8),
                         output_variable_names=("nee", "leaf_carbon"), freq="1D")
    model_output = prior.evaluate(vector.sample(key, 100)).model_output  # (member, site, time)
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import ValidationError
from pyens import Axis, EnsembleRunner, Grid, PartialSpec, RemoteError, TaskFailedError
from pyens import EnsembleSpec, SequentialBackend
from pyens.backends import Backend
from pyens.xarray import fields_from_dataset
from pysipnet.climate import ClimateDrivers
from pysipnet.model import SIPNETModel
from pysipnet.parameters.model import SIPNET_PARAMS_BY_GROUP, resolve_parameter_name
from pysipnet.runner import SIPNETRunError
from pysipnet.variables import resolve_output_variable

from sipnet_calibration.fields import label_run, site_lookup
from sipnet_calibration.observation import ObservationVector, aggregate_time
from sipnet_calibration.parameter_vector import ParameterVector
from sipnet_calibration.sites import load_sites

__all__ = ["MODEL_FAILURES", "ForwardEvaluation", "ForwardModel", "ModelOutputNotFinite"]

SITE = "site"
MEMBER = "member"
TIME = "time"

class ModelOutputNotFinite(RuntimeError):
    """A run completed but wrote a non-finite value in a variable that was read.

    SIPNET exits 0 on a blow-up, so the failure shows only in the output; it
    is a failure at the run's parameters, and pickles as a plain message.
    """


#: The exceptions that mean a run failed **at its parameters**: the member's
#: row becomes NaN. Everything else a worker returns is the machinery failing
#: and is raised.
MODEL_FAILURES: tuple[type[BaseException], ...] = (
    SIPNETRunError,
    ValidationError,
    subprocess.TimeoutExpired,
    ModelOutputNotFinite,
)


@dataclass(frozen=True)
class ForwardEvaluation:
    """What one evaluation of a :class:`ForwardModel` produced; see the module docstring."""

    theta: np.ndarray
    sipnet_table: xr.Dataset
    model_output: xr.Dataset | None
    predictions: np.ndarray | None
    run_succeeded: xr.DataArray
    failures: pd.DataFrame
    valid: np.ndarray


class ForwardModel:
    """The forward map: unconstrained calibration parameters to predictions.

    Parameters
    ----------
    model:
        The :class:`pysipnet.model.SIPNETModel` every run goes through.
    parameter_vector:
        The calibration vector; its sites are the sites run.
    climate:
        ``{site id: ClimateDrivers}``, a superset of the vector's sites.
    backend:
        The PyEns backend the runs execute on.
    observation_vector:
        The observation vector whose operators reduce each run on the worker
        and whose order the predictions take. Its sites must be among the
        vector's; ``select`` it first otherwise.
    output_variable_names:
        Without an observation vector, which pySIPNET output variables each
        run returns. With one, defaults to the names its operators read.
    freq:
        Without an observation vector, aggregate each run's output on the
        worker to this pandas frequency with
        :func:`~sipnet_calibration.observation.aggregate_time`. Refused with
        an observation vector, whose operators decide their own alignment.
    to_sipnet_table:
        ``theta (B, D) -> SIPNET table (member, site)``; defaults to
        ``parameter_vector.sipnet_table``. The hook for an experiment that
        maps initial-condition state into SIPNET parameters.
    site_table:
        The site table the runs are labeled from (``lon``/``lat``);
        defaults to the vector's when it carries them, else
        :func:`sipnet_calibration.sites.load_sites`.
    """

    def __init__(
        self,
        model: SIPNETModel,
        parameter_vector: ParameterVector,
        *,
        climate: Mapping[int, ClimateDrivers],
        backend: Backend,
        observation_vector: ObservationVector | None = None,
        output_variable_names: Sequence[str] | None = None,
        freq: str | None = None,
        to_sipnet_table: Callable[[Any], xr.Dataset] | None = None,
        site_table: pd.DataFrame | None = None,
    ) -> None:
        check_model_is_a_sipnet_model(model)
        check_backend(backend)
        self.model = model
        self.parameter_vector = parameter_vector
        self.sites: tuple[int, ...] = tuple(int(s) for s in parameter_vector.sites)
        self.backend = backend
        self.observation_vector = observation_vector
        self.freq = freq
        check_freq_is_for_the_prior_predictive(freq, observation_vector)
        self._to_sipnet_table = to_sipnet_table or parameter_vector.sipnet_table
        self.climate = {int(s): climate[s] for s in self.sites if s in climate}
        check_climate_covers_the_sites(climate, self.sites)
        check_climate_is_file_backed(self.climate, backend)
        self.output_variable_names = _output_names(output_variable_names, observation_vector)
        check_output_variables_survive_the_flags(self.output_variable_names, model)
        if observation_vector is not None:
            check_observation_sites_are_run(observation_vector, self.sites)
        self.site_table = _site_table_with_locations(site_table, parameter_vector, self.sites)

        probe = self._to_sipnet_table(parameter_vector.sample(jax.random.key(0), 1))
        check_table_is_a_sipnet_table(probe, self.sites)
        self.sipnet_parameter_names: tuple[str, ...] = tuple(str(n) for n in probe.data_vars)
        self._base_values = _base_values_for(
            model,
            () if observation_vector is None else observation_vector.sipnet_parameter_names,
            self.sipnet_parameter_names,
        )
        self._site_axis = Axis(SITE, labels=list(self.sites))
        self._partial = self._build_partial()
        self._run = _Run(
            model=model,
            output_variable_names=self.output_variable_names,
            freq=freq,
            base_values=self._base_values,
            site_table=self.site_table,
        )

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """``D``, the length of ``theta``."""
        return self.parameter_vector.dimension

    @property
    def n_predictions(self) -> int:
        """``N``, the length of a predictions row; raises without an observation vector."""
        check_has_observation_vector(self.observation_vector, "n_predictions")
        assert self.observation_vector is not None
        return self.observation_vector.dimension

    def __repr__(self) -> str:
        what = (
            f"observations={self.observation_vector!r}"
            if self.observation_vector is not None
            else f"output_variable_names={self.output_variable_names!r}, freq={self.freq!r}"
        )
        return f"ForwardModel(D={self.dimension}, sites={len(self.sites)}, {what})"

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, theta: Any) -> ForwardEvaluation:
        """Run SIPNET once per member and site, and collect what came back."""
        batch = _as_batch(theta, self.dimension)
        table = self._to_sipnet_table(batch)
        check_table_is_a_sipnet_table(table, self.sites, expected=self.sipnet_parameter_names)
        members = np.asarray(table[MEMBER].values)
        spec = self._partial(**fields_from_dataset(table, axes={SITE: self._site_axis}))
        result = EnsembleRunner(self._run, self.backend).run(spec)

        outputs, succeeded, failures, broken = _sort_records(result, members, self.sites)
        if broken:
            partial = ForwardEvaluation(
                theta=batch, sipnet_table=table, model_output=None, predictions=None,
                run_succeeded=succeeded, failures=failures,
                valid=np.zeros(len(members), dtype=bool),
            )
            raise _machinery_error(broken, partial)
        if self.observation_vector is not None:
            predictions = _place_predictions(outputs, self.observation_vector, members)
            # A member with any failed run is invalid as a whole: pyEKI updates
            # per member, so a row that is partly a prediction cannot be used.
            predictions[~succeeded.all(SITE).values] = np.nan
            model_output = None
            finite = np.isfinite(predictions).all(axis=1)
        else:
            predictions = None
            model_output = _stack_model_output(outputs, members, self.sites, self.site_table)
            finite = np.ones(len(members), dtype=bool)
        valid = succeeded.all(SITE).values & finite
        return ForwardEvaluation(
            theta=batch,
            sipnet_table=table,
            model_output=model_output,
            predictions=predictions,
            run_succeeded=succeeded,
            failures=failures,
            valid=valid,
        )

    def __call__(self, theta: Any) -> np.ndarray:
        """``evaluate(theta).predictions``: ``(B, N)``, or ``(N,)`` for a ``(D,)`` theta."""
        check_has_observation_vector(self.observation_vector, "__call__")
        one = np.ndim(theta) == 1
        evaluation = self.evaluate(theta)
        assert evaluation.predictions is not None
        return evaluation.predictions[0] if one else evaluation.predictions

    # ── supporting methods ────────────────────────────────────────────────────

    def _build_partial(self) -> PartialSpec:
        climate = Grid({s: self.climate[s] for s in self.sites}, along=self._site_axis)
        site_ids = Grid(list(self.sites), along=self._site_axis)
        observations = Grid(
            {s: _site_slice(self.observation_vector, s) for s in self.sites}, along=self._site_axis
        )
        placeholders = {
            name: Grid([0.0] * len(self.sites), along=self._site_axis)
            for name in self.sipnet_parameter_names
        }
        spec = EnsembleSpec(
            inputs={"climate": climate, "site": site_ids, "observations": observations, **placeholders}
        )
        return spec.freeze(free=list(self.sipnet_parameter_names))


# ── the per-run callable ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _RunOutput:
    """What one run sends back: its output variables, or its predictions."""

    model_output: xr.Dataset | None
    predictions: dict[str, xr.DataArray] | None


@dataclass(frozen=True)
class _Run:
    """One SIPNET run, reduced on the worker; picklable, built once per model.

    ``observations`` is the site's slice of the observation vector, or
    ``None`` on the prior-predictive path and at a site no product observes.
    """

    model: SIPNETModel
    output_variable_names: tuple[str, ...]
    freq: str | None
    base_values: dict[str, float]
    site_table: pd.DataFrame

    def __call__(
        self,
        *,
        climate: ClimateDrivers,
        site: int,
        observations: ObservationVector | None = None,
        **sipnet_parameters: Any,
    ) -> _RunOutput:
        result = self.model(climate=climate, **sipnet_parameters)
        dataset = label_run(
            result.outputs.select(list(self.output_variable_names)),
            site=int(site),
            site_table=self.site_table,
        )
        _check_output_is_finite(dataset, int(site))
        if self.freq is not None:
            dataset = _aggregated(dataset, self.freq)
        if observations is None:
            return _RunOutput(model_output=dataset, predictions=None)
        values = {**self.base_values, **{k: float(v) for k, v in sipnet_parameters.items()}}
        return _RunOutput(
            model_output=None, predictions=observations.predict(dataset, sipnet_parameters=values)
        )


# ── supporting helpers ────────────────────────────────────────────────────────


def _site_slice(vector: ObservationVector | None, site: int) -> ObservationVector | None:
    """The observation vector restricted to one site, or ``None`` with no cell there."""
    if vector is None or vector.positions(site=site).size == 0:
        return None
    return vector.select(sites=[site])


def _check_output_is_finite(dataset: xr.Dataset, site: int) -> None:
    """Raise :class:`ModelOutputNotFinite` if a read variable holds a non-finite value."""
    for name, variable in dataset.data_vars.items():
        values = np.asarray(variable.values, dtype=np.float64)
        if not np.isfinite(values).all():
            where = int(np.flatnonzero(~np.isfinite(values.ravel()))[0])
            raise ModelOutputNotFinite(
                f"the run at site {site} wrote a non-finite {name!r} (first at flat position "
                f"{where} of {values.size}); SIPNET exits 0 on a blow-up, so this counts as a "
                "failure at the run's parameters."
            )


def _as_batch(theta: Any, dimension: int) -> np.ndarray:
    array = np.asarray(theta, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != dimension or array.shape[0] == 0:
        raise ValueError(
            f"theta must be (D,) or (B, D) with B >= 1 and D = {dimension}, got shape "
            f"{array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("theta holds a non-finite value; the parameter vector never produces one.")
    return array


def _output_names(names: Sequence[str] | None, observation_vector: ObservationVector | None) -> tuple[str, ...]:
    if names is None:
        if observation_vector is None:
            raise ValueError(
                "name the output variables each run returns (output_variable_names=), "
                "or give an observation_vector whose operators say what they read."
            )
        return tuple(observation_vector.output_variable_names)
    if isinstance(names, str):
        names = [names]
    resolved: list[str] = []
    for name in names:
        registry_name = resolve_output_variable(name).name
        if registry_name not in resolved:
            resolved.append(registry_name)
    if observation_vector is not None:
        missing = [n for n in observation_vector.output_variable_names if n not in resolved]
        if missing:
            raise ValueError(
                f"the observation vector's operators read {missing}, which "
                "output_variable_names does not include."
            )
    return tuple(resolved)


def _site_table_with_locations(
    site_table: pd.DataFrame | None, vector: ParameterVector, sites: Sequence[int]
) -> pd.DataFrame:
    table = site_table
    if table is None:
        own = vector.site_table
        table = own if {"lon", "lat"} <= set(own.columns) else load_sites()
    table = site_lookup(table)
    if not {"lon", "lat"} <= set(table.columns):
        raise ValueError(
            "the site table must carry 'lon' and 'lat' columns, as load_sites() returns them; "
            f"got {list(table.columns)}."
        )
    missing = [s for s in sites if s not in table.index]
    if missing:
        raise ValueError(f"the site table has no row for site(s) {missing[:10]}.")
    return table.loc[list(sites)]


def _base_values_for(
    model: SIPNETModel, read: Sequence[str], set_by_vector: Sequence[str]
) -> dict[str, float]:
    """The base value of each SIPNET parameter an operator reads and the vector does not set."""
    values: dict[str, float] = {}
    for name in read:
        flat = resolve_parameter_name(name)
        if flat in set_by_vector:
            continue
        group = next((g for g, names in SIPNET_PARAMS_BY_GROUP.items() if flat in names), None)
        if group is None:
            raise ValueError(f"{name!r} is in no SIPNET parameter group.")
        values[flat] = float(getattr(getattr(model.base_params, group), flat))
    return values


def _aggregated(dataset: xr.Dataset, freq: str) -> xr.Dataset:
    """Every variable of *dataset* aggregated to *freq* by its kind's default."""
    arrays = {str(name): aggregate_time(dataset[name], freq) for name in dataset.data_vars}
    first = next(iter(arrays.values()))
    merged = xr.Dataset(
        {name: array.drop_vars([c for c in array.coords if c != TIME]) for name, array in arrays.items()},
        coords={c: first[c] for c in first.coords},
    )
    merged.attrs = dict(dataset.attrs)
    return merged


def _sort_records(
    result: Any, members: np.ndarray, sites: Sequence[int]
) -> tuple[dict[tuple[int, int], _RunOutput], xr.DataArray, pd.DataFrame, list[tuple[dict, BaseException]]]:
    outputs: dict[tuple[int, int], _RunOutput] = {}
    succeeded = xr.DataArray(
        np.zeros((len(members), len(sites)), dtype=bool),
        dims=(MEMBER, SITE),
        coords={MEMBER: members, SITE: list(sites)},
        name="run_succeeded",
    )
    rows: list[dict[str, Any]] = []
    broken: list[tuple[dict, BaseException]] = []
    for record in result:
        member, site = int(record.coordinate[MEMBER]), int(record.coordinate[SITE])
        if not record.failed:
            outputs[(member, site)] = record.output
            succeeded.loc[{MEMBER: member, SITE: site}] = True
        elif _is_model_failure(record.output):
            error = record.output
            rows.append(
                {MEMBER: member, SITE: site, "error": type(error).__name__, "message": str(error)[:500]}
            )
        else:
            broken.append((record.coordinate, record.output))
    failures = pd.DataFrame(rows, columns=[MEMBER, SITE, "error", "message"])
    return outputs, succeeded, failures, broken


def _is_model_failure(error: BaseException) -> bool:
    if isinstance(error, MODEL_FAILURES):
        return True
    if isinstance(error, RemoteError):
        # PyEns qualifies the name ("pysipnet.runner.SIPNETRunError"); compare the class name.
        name = str(getattr(error, "type_name", "")).rsplit(".", 1)[-1]
        return name in {t.__name__ for t in MODEL_FAILURES}
    return False


def _machinery_error(
    broken: list[tuple[dict, BaseException]], evaluation: ForwardEvaluation
) -> RuntimeError:
    coordinate, error = broken[0]
    kinds = sorted({type(e).__name__ for _, e in broken})
    raised = RuntimeError(
        f"{len(broken)} run(s) failed in the machinery rather than at their parameters "
        f"({', '.join(kinds)}); the first, at {coordinate}, says: {error}. A "
        f"{TaskFailedError.__name__}, a missing binary or an import error on a worker says "
        "nothing about the parameters, so it is raised rather than turned into a NaN row; "
        "the runs that were collected are on this error's `evaluation` attribute."
    )
    raised.evaluation = evaluation  # type: ignore[attr-defined]
    return raised


def _place_predictions(
    outputs: Mapping[tuple[int, int], _RunOutput], vector: ObservationVector, members: np.ndarray
) -> np.ndarray:
    fields: dict[str, xr.DataArray] = {}
    for observation in vector.observations:
        values = observation.values
        full = xr.DataArray(
            np.full((len(members), *values.shape), np.nan),
            dims=(MEMBER, *values.dims),
            coords={MEMBER: members, **values.coords},
            attrs=dict(values.attrs),
            name=observation.product_name,
        )
        for (member, site), output in outputs.items():
            if output.predictions is None or observation.product_name not in output.predictions:
                continue
            predicted = output.predictions[observation.product_name].sel({SITE: site})
            full.loc[{MEMBER: member, SITE: site}] = predicted.values
        fields[observation.product_name] = full
    return vector.flat(fields)


def _stack_model_output(
    outputs: Mapping[tuple[int, int], _RunOutput],
    members: np.ndarray,
    sites: Sequence[int],
    site_table: pd.DataFrame,
) -> xr.Dataset:
    per_member: list[xr.Dataset] = []
    for member in members:
        per_site = [
            outputs[(int(member), site)].model_output.drop_vars(["lon", "lat"], errors="ignore")
            for site in sites
            if (int(member), site) in outputs and outputs[(int(member), site)].model_output is not None
        ]
        if not per_site:
            continue
        stacked = xr.concat(per_site, dim=SITE, join="outer", coords="different", compat="equals", combine_attrs="override")
        per_member.append(stacked.assign_coords({MEMBER: int(member)}))
    if not per_member:
        raise RuntimeError(
            "every run failed at its parameters; there is no model output to stack NaN onto. "
            "Draw parameters the model can run, or check failures on the evaluation."
        )
    dataset = xr.concat(per_member, dim=MEMBER, join="outer", coords="different", compat="equals", combine_attrs="override")
    dataset = dataset.reindex({MEMBER: members, SITE: list(sites)}).transpose(MEMBER, SITE, TIME, ...)
    lon = site_table.loc[list(sites), "lon"].to_numpy(dtype=np.float64)
    lat = site_table.loc[list(sites), "lat"].to_numpy(dtype=np.float64)
    return dataset.assign_coords(lon=(SITE, lon), lat=(SITE, lat))


# ── checks ────────────────────────────────────────────────────────────────────


def check_model_is_a_sipnet_model(model: Any) -> None:
    if not isinstance(model, SIPNETModel):
        raise TypeError(f"model must be a pysipnet SIPNETModel, got {type(model).__name__}.")


def check_backend(backend: Any) -> None:
    if not isinstance(backend, Backend):
        raise TypeError(f"backend must be a pyens Backend, got {type(backend).__name__}.")


def check_climate_covers_the_sites(climate: Mapping[int, Any], sites: Sequence[int]) -> None:
    missing = [s for s in sites if s not in climate]
    if missing:
        raise ValueError(
            f"climate has no drivers for site(s) {missing[:10]} of the parameter vector; "
            "every site run needs its drivers."
        )
    for site in sites:
        if not isinstance(climate[site], ClimateDrivers):
            raise TypeError(
                f"climate[{site}] must be a pysipnet ClimateDrivers, got "
                f"{type(climate[site]).__name__}."
            )


def check_climate_is_file_backed(climate: Mapping[int, ClimateDrivers], backend: Backend) -> None:
    if isinstance(backend, SequentialBackend):
        return
    in_memory = [s for s, c in climate.items() if c.source_path is None]
    if in_memory:
        raise ValueError(
            f"the drivers of site(s) {in_memory[:10]} are held in memory, and under "
            f"{type(backend).__name__} every run would carry a copy of them. Write them "
            "with ClimateDrivers.to_file and open them with ClimateDrivers.from_path."
        )


def check_output_variables_survive_the_flags(names: Sequence[str], model: SIPNETModel) -> None:
    flags = getattr(model.runner, "flags", None)
    if flags is None:
        return
    for name in names:
        flag = resolve_output_variable(name).requires_flag
        if flag and not getattr(flags, flag, True):
            raise ValueError(
                f"{name!r} is written as constant zero unless the model flag {flag!r} is on, "
                "and this model's runner has it off."
            )


def check_observation_sites_are_run(vector: ObservationVector, sites: Sequence[int]) -> None:
    extra = sorted(set(vector.sites) - set(sites))
    if extra:
        raise ValueError(
            f"the observation vector observes site(s) {extra[:10]} that the parameter vector "
            "does not run; select the observations to the vector's sites first "
            "(observation_vector.select(sites=...))."
        )


def check_table_is_a_sipnet_table(
    table: Any, sites: Sequence[int], expected: Sequence[str] | None = None
) -> None:
    if not isinstance(table, xr.Dataset):
        raise TypeError(f"to_sipnet_table must return an xr.Dataset, got {type(table).__name__}.")
    if MEMBER not in table.dims or SITE not in table.dims:
        raise ValueError(
            f"a SIPNET table for a batch has dims (member, site), got {tuple(table.dims)}."
        )
    if table[SITE].values.tolist() != list(sites):
        raise ValueError("the SIPNET table's sites are not the parameter vector's sites, in order.")
    for name in table.data_vars:
        try:
            resolve_parameter_name(str(name))
        except KeyError as error:
            raise ValueError(
                f"the SIPNET table sets {name!r}, which is not a pySIPNET parameter: {error}"
            ) from None
    if expected is not None and set(str(n) for n in table.data_vars) != set(expected):
        raise ValueError(
            f"the SIPNET table sets {sorted(table.data_vars)}, but the model was built for "
            f"{sorted(expected)}; a ForwardModel's free fields are fixed when it is built."
        )


def check_freq_is_for_the_prior_predictive(freq: Any, vector: ObservationVector | None) -> None:
    if freq is not None and vector is not None:
        raise ValueError(
            "freq= aggregates model output for the prior predictive; with an observation "
            "vector the operators decide their own alignment, so freq would be ignored. "
            "Drop one of the two."
        )


def check_has_observation_vector(vector: ObservationVector | None, what: str) -> None:
    if vector is None:
        raise ValueError(
            f"{what} needs an observation vector; this ForwardModel was built without one, "
            "for model output only. Use evaluate(theta).model_output."
        )
