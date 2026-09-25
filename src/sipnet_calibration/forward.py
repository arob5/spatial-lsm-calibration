"""The forward model: unconstrained calibration parameters to predictions, over
a site set, with SIPNET run once per member and site.

Where this sits
---------------
::

    theta (J, D)  --ParameterVector.sipnet_table-->  SIPNET table (member, site)
                  --PyEns, one SIPNETModel run per (member, site)-->  model output
                  --ObservationVector.predict and .flat on the worker-->
                    one site's block of Flat per run
                  --placed at positions(site=) by the calling process-->  (J, N)

:class:`ForwardModel` is the one object this module adds: the callable
``(J, D) -> (J, N)`` that pyEKI's ``run`` takes as ``forward``, and that an
MCMC target calls. Everything it composes exists elsewhere: the parameter
vector (:mod:`sipnet_calibration.parameter_vector`), pySIPNET's
``SIPNETModel``, PyEns's ``PartialSpec``/``EnsembleRunner``/``Backend`` and
its xarray bridge, the observation vector
(:mod:`sipnet_calibration.observation`), and the stacking of runs
(:func:`sipnet_calibration.fields.stack_model_outputs`). It adds no ensemble
or run class of its own.

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
produced for a batch of ``J`` rows of ``theta``:

``theta``
    ``(J, D)`` float64, coerced from what was received; a ``(D,)`` input is
    one row.
``sipnet_table``
    The SIPNET table that was run, an ``xr.Dataset`` on ``(member, site)``
    with ``member`` labeled ``0`` to ``J - 1`` in the order of ``theta``'s
    rows and ``site`` in the parameter vector's order.
``model_output``
    ``xr.Dataset`` on ``(member, site, time)`` of the output variables, as
    :func:`~sipnet_calibration.fields.stack_model_outputs` builds it: every
    variable a field, so without ``time_bounds`` or SIPNET's
    ``year``/``day_of_year``/``hour_of_day`` row labels, and ``NaN`` where a
    run failed. ``None`` when an observation vector was given.
``predictions``
    ``(J, N)`` float64 in the observation vector's order, ``NaN`` in every
    cell of a member with a failed run; ``None`` without an observation
    vector.
``run_succeeded``
    bool ``xr.DataArray`` on ``(member, site)``.
``failures``
    ``pd.DataFrame`` with columns ``member``, ``site``, ``error`` (the class
    name of the exception the run raised) and ``message``, one row per run
    that failed at its parameters.
``valid``
    bool ``(J,)``: every run of the member succeeded and, where there are
    predictions, they are finite.

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
run failed at its parameters is also raised, the same way, since there is no
time axis to stack ``NaN`` onto.

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
**The ``PartialSpec`` is built once.** Its fixed inputs (the climate, the
site id and the site's slice of the observation vector, all along one site
axis) and its free fields (the SIPNET parameter names the vector sets) hold
for the model's lifetime, and every call's parameter grids zip with that
site axis. The free names are learned by mapping one prior draw through
``sipnet_table`` in ``__init__``, which fails fast, before anything is
queued, on a hook that does not return a ``(member, site)`` table over the
vector's sites in pySIPNET's parameter names.

**The operators run on the worker** because reading a run's output back in
the calling process costs more than the run (pySIPNET parses the whole
file), and an observation operator is pointwise in site, so each worker can
reduce its own run to the observed cells. Each run receives only its site's
slice of the observation vector, as an input field along the site axis, so
a run at a site no product observes carries nothing and returns nothing,
and a process backend pickles only that site's cells rather than the whole
vector. The calling process only places small arrays.

**Predictions are placed by position.** A run returns its slice's Flat,
which the calling process writes at ``positions(site=)`` of the whole
vector. That is right because the observation vector is site-major: a
one-site selection's index is the whole vector's index at that site's
positions. ``__init__`` checks this for every observed site.

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

    forward = ForwardModel(model, parameter_vector, climate=climate,
                           backend=LocalBackend(8),
                           observation_vector=observation_vector)
    result = pyeki.eki.run(state, forward, observation_vector.y, noise_cov,
                           schedule=...)

    prior = ForwardModel(model, parameter_vector, climate=climate,
                         backend=LocalBackend(8),
                         output_variable_names=("nee", "leaf_carbon"),
                         freq="1D")
    theta = parameter_vector.sample(key, 100)
    model_output = prior.evaluate(theta).model_output  # (member, site, time)
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
from pysipnet.parameters.model import resolve_parameter_name
from pysipnet.runner import SIPNETRunError
from pysipnet.variables import resolve_output_variable

from sipnet_calibration.fields import (
    label_run,
    resolve_output_variable_names,
    site_lookup,
    stack_model_outputs,
)
from sipnet_calibration.observation import (
    DEFAULT_METHOD_FOR_KIND,
    ObservationVector,
    aggregate_time,
)
from sipnet_calibration.observation.time_alignment import check_frequency
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
    """What one evaluation of a :class:`ForwardModel` produced.

    The fields are described in the module docstring's Data model.
    """

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
        Under any backend but ``SequentialBackend`` each must be file-backed,
        opened with ``ClimateDrivers.from_path``.
    backend:
        The PyEns backend the runs execute on.
    observation_vector:
        The observation vector whose operators reduce each run on the worker
        and whose order the predictions take. Its sites must be among the
        parameter vector's; ``select`` it first otherwise.
    output_variable_names:
        Without an observation vector, which pySIPNET output variables each
        run returns. With one, defaults to the names its operators read.
    freq:
        Without an observation vector, aggregate each run's output on the
        worker to this pandas frequency with
        :func:`~sipnet_calibration.observation.aggregate_time`, each variable
        by the method that leaves its kind unchanged
        (:data:`~sipnet_calibration.observation.DEFAULT_METHOD_FOR_KIND`).
        Refused with an observation vector, whose operators decide their own
        alignment.
    to_sipnet_table:
        ``theta (J, D) -> SIPNET table (member, site)``; defaults to
        ``parameter_vector.sipnet_table``. The hook for an experiment that
        maps initial-condition state into SIPNET parameters. The table must
        be on exactly ``(member, site)``, with ``member`` labeled ``0`` to
        ``J - 1`` and the parameter vector's sites in its order.
    site_table:
        The site table the runs are labeled from (``lon``/``lat``), one row
        per site; defaults to the vector's when it carries them, else
        :func:`sipnet_calibration.sites.load_sites`.

    Raises
    ------
    TypeError
        If *model* is not a ``SIPNETModel``, *backend* not a PyEns
        ``Backend``, a site's drivers not ``ClimateDrivers``, or
        *to_sipnet_table* returns something other than an ``xr.Dataset``.
    ValueError
        If neither *output_variable_names* nor *observation_vector* is
        given; *freq* is given with an observation vector or is not a pandas
        frequency; a site has no drivers, or in-memory drivers under a
        process backend; the observation vector observes a site the
        parameter vector does not run; the output variables do not cover the
        operators, or one is switched off by the model's flags, or (with
        *freq*) has a kind no method keeps; the site table lacks
        ``lon``/``lat``, lists a site twice or misses one; or
        *to_sipnet_table* does not return a SIPNET table for the batch.
    KeyError
        If an output variable name is not a pySIPNET output variable.
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
        check_forward_model_arguments(
            model, parameter_vector, climate=climate, backend=backend,
            observation_vector=observation_vector,
            output_variable_names=output_variable_names, freq=freq, site_table=site_table,
        )
        self.model = model
        self.parameter_vector = parameter_vector
        self.sites: tuple[int, ...] = parameter_vector.sites
        self.backend = backend
        self.observation_vector = observation_vector
        self.freq = freq
        self.climate = {site: climate[site] for site in self.sites}
        self.output_variable_names = _output_variable_names(
            output_variable_names, observation_vector
        )
        check_output_variables_survive_the_flags(self.output_variable_names, model)
        self.site_table = _site_table_for(site_table, parameter_vector, self.sites)
        self._to_sipnet_table = to_sipnet_table or parameter_vector.sipnet_table
        self.sipnet_parameter_names = self._probe_sipnet_parameter_names()
        self._base_values = _base_values_for(
            model,
            () if observation_vector is None else observation_vector.sipnet_parameter_names,
            self.sipnet_parameter_names,
        )
        self._site_axis = Axis(SITE, labels=list(self.sites))
        self._site_slices, self._site_positions = _site_blocks(observation_vector)
        self._partial = self._build_partial()
        if freq is not None:
            check_every_kind_has_a_default_method(self.output_variable_names)
        self._run = _Run(
            model=model,
            output_variable_names=self.output_variable_names,
            returns_model_output=observation_vector is None,
            freq=freq,
            base_values=self._base_values,
            site_table=self.site_table,
        )

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def input_dimension(self) -> int:
        """``D``, the length of ``theta``: the parameter vector's dimension."""
        return self.parameter_vector.dimension

    @property
    def output_dimension(self) -> int:
        """``N``, the length of a predictions row: the observation vector's dimension.

        Raises
        ------
        ValueError
            Without an observation vector, since the model output then has no
            flat form.
        """
        return self._observation_vector_for("output_dimension").dimension

    def __repr__(self) -> str:
        what = (
            f"observations={self.observation_vector!r}"
            if self.observation_vector is not None
            else f"output_variable_names={self.output_variable_names!r}, freq={self.freq!r}"
        )
        return f"ForwardModel(D={self.input_dimension}, sites={len(self.sites)}, {what})"

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, theta: Any) -> ForwardEvaluation:
        """Run SIPNET once per member and site, and collect what came back.

        Parameters
        ----------
        theta:
            ``(J, D)`` unconstrained calibration parameters, one row per
            member, or ``(D,)`` for one member.

        Returns
        -------
        ForwardEvaluation
            With ``predictions`` on the calibration path and
            ``model_output`` on the prior-predictive path; see the module
            docstring's Data model.

        Raises
        ------
        ValueError
            If *theta* is not ``(D,)`` or ``(J, D)`` with ``J >= 1``, or holds
            a non-finite value; or if the SIPNET table the hook returns is not
            one for this batch (dims other than ``(member, site)``, members
            other than ``0`` to ``J - 1`` in order, sites other than the
            vector's in order, or other SIPNET parameters than the model was
            built for).
        RuntimeError
            If a run failed in the machinery rather than at its parameters,
            or, on the prior-predictive path, if every run failed at its
            parameters. The error's ``evaluation`` attribute is a
            :class:`ForwardEvaluation` of what was collected, with no
            predictions or model output.
        """
        batch = _as_batch(theta, self.input_dimension)
        table = self._to_sipnet_table(batch)
        check_table_is_a_sipnet_table(
            table, self.sites, n_members=len(batch), expected=self.sipnet_parameter_names
        )
        spec = self._partial(**fields_from_dataset(table, axes={SITE: self._site_axis}))
        result = EnsembleRunner(self._run, self.backend).run(spec)

        outputs, succeeded, failures, broken = _sort_records(result, len(batch), self.sites)
        members = table[MEMBER].values
        collected = ForwardEvaluation(
            theta=batch, sipnet_table=table, model_output=None, predictions=None,
            run_succeeded=_run_succeeded(succeeded, members, self.sites), failures=failures,
            valid=np.zeros(len(batch), dtype=bool),
        )
        check_no_run_failed_in_the_machinery(broken, collected)
        member_succeeded = succeeded.all(axis=1)
        if self.observation_vector is None:
            check_some_run_succeeded(collected)
            model_output = _stacked_model_output(outputs, len(batch), self.sites, self.site_table)
            return replace(collected, model_output=model_output, valid=member_succeeded)
        predictions = self._placed_predictions(outputs, len(batch))
        # A member with any failed run is invalid as a whole: pyEKI updates
        # per member, so a row that is partly a prediction cannot be used.
        predictions[~member_succeeded] = np.nan
        valid = member_succeeded & np.isfinite(predictions).all(axis=1)
        return replace(collected, predictions=predictions, valid=valid)

    def __call__(self, theta: Any) -> np.ndarray:
        """``evaluate(theta).predictions``: ``(J, N)``, or ``(N,)`` for a ``(D,)`` theta.

        Raises
        ------
        ValueError
            Without an observation vector, and as :meth:`evaluate` does.
        RuntimeError
            As :meth:`evaluate` does.
        """
        self._observation_vector_for("__call__")
        predictions = self.evaluate(theta).predictions
        return predictions[0] if np.ndim(theta) == 1 else predictions

    # ── supporting methods ────────────────────────────────────────────────────

    def _observation_vector_for(self, what: str) -> ObservationVector:
        """The observation vector, or a ``ValueError`` saying *what* needs one."""
        if self.observation_vector is None:
            raise ValueError(
                f"{what} needs an observation vector; this ForwardModel was built without one, "
                "for model output only. Use evaluate(theta).model_output."
            )
        return self.observation_vector

    def _probe_sipnet_parameter_names(self) -> tuple[str, ...]:
        """The SIPNET parameter names the table hook sets, from one prior draw."""
        probe = self._to_sipnet_table(self.parameter_vector.sample(jax.random.key(0), 1))
        check_table_is_a_sipnet_table(probe, self.sites, n_members=1)
        return tuple(str(name) for name in probe.data_vars)

    def _build_partial(self) -> PartialSpec:
        climate = Grid({s: self.climate[s] for s in self.sites}, along=self._site_axis)
        site_ids = Grid(list(self.sites), along=self._site_axis)
        observations = Grid(
            {s: self._site_slices.get(s) for s in self.sites}, along=self._site_axis
        )
        placeholders = {
            name: Grid([0.0] * len(self.sites), along=self._site_axis)
            for name in self.sipnet_parameter_names
        }
        spec = EnsembleSpec(
            inputs={
                "climate": climate, "site": site_ids, "observations": observations,
                **placeholders,
            }
        )
        return spec.freeze(free=list(self.sipnet_parameter_names))

    def _placed_predictions(
        self, outputs: Mapping[tuple[int, int], _RunOutput], n_members: int
    ) -> np.ndarray:
        """``(J, N)``: each run's block of Flat written at its site's positions."""
        vector = self._observation_vector_for("predictions")
        predictions = np.full((n_members, vector.dimension), np.nan, dtype=np.float64)
        for (member, site), output in outputs.items():
            if output.predictions is not None:
                predictions[member, self._site_positions[site]] = output.predictions
        return predictions


# ── the per-run callable ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _RunOutput:
    """What one run sends back: its model output, its site's predictions, or neither.

    ``predictions`` is the site's block of the observation vector's Flat,
    ``(N_site,)`` in the order of the site's slice.
    """

    model_output: xr.Dataset | None
    predictions: np.ndarray | None


@dataclass(frozen=True)
class _Run:
    """One SIPNET run, reduced on the worker; picklable, built once per model.

    ``returns_model_output`` is true on the prior-predictive path, where the
    run sends back its (aggregated, with ``freq``) output. Otherwise it sends
    back its site's predictions when it receives the site's slice of the
    observation vector as ``observations``, and nothing when it receives
    ``None``, at a site no product observes.
    """

    model: SIPNETModel
    output_variable_names: tuple[str, ...]
    returns_model_output: bool
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
        site = int(site)
        sipnet_result = self.model(climate=climate, **sipnet_parameters)
        dataset = sipnet_result.outputs.select(list(self.output_variable_names))
        check_output_is_finite(dataset, site)
        if self.returns_model_output:
            model_output = label_run(dataset, site=site, site_table=self.site_table)
            if self.freq is not None:
                model_output = _aggregated(model_output, self.freq)
            return _RunOutput(model_output=model_output, predictions=None)
        if observations is None:
            return _RunOutput(model_output=None, predictions=None)
        model_output = label_run(dataset, site=site, site_table=self.site_table)
        values = {**self.base_values, **{k: float(v) for k, v in sipnet_parameters.items()}}
        predicted = observations.predict(model_output, sipnet_parameters=values)
        return _RunOutput(model_output=None, predictions=observations.flat(predicted))


# ── supporting helpers ────────────────────────────────────────────────────────


def _as_batch(theta: Any, input_dimension: int) -> np.ndarray:
    """*theta* as a ``(J, D)`` float64 batch, a ``(D,)`` input as one row."""
    array = np.asarray(theta, dtype=np.float64)
    check_theta_has_the_batch_shape(array, input_dimension)
    check_theta_is_finite(array)
    return np.atleast_2d(array)


def _output_variable_names(
    output_variable_names: Sequence[str] | None, observation_vector: ObservationVector | None
) -> tuple[str, ...]:
    """The registry names each run returns: as given, else the operators'."""
    if output_variable_names is None:
        # check_output_variables_are_named has refused both being None.
        return tuple(observation_vector.output_variable_names)  # type: ignore[union-attr]
    names = tuple(resolve_output_variable_names(output_variable_names))
    if observation_vector is not None:
        check_output_variable_names_cover_the_operators(names, observation_vector)
    return names


def _site_table_for(
    site_table: pd.DataFrame | None, parameter_vector: ParameterVector, sites: Sequence[int]
) -> pd.DataFrame:
    """The site table's rows for *sites*, keyed on ``site_id``."""
    table = site_table
    if table is None:
        own = parameter_vector.site_table
        table = own if {"lon", "lat"} <= set(own.columns) else load_sites()
    return site_lookup(table).loc[list(sites)]


def _base_values_for(
    model: SIPNETModel, read: Sequence[str], set_by_vector: Sequence[str]
) -> dict[str, float]:
    """The base value of each SIPNET parameter an operator reads and the vector does not set."""
    values: dict[str, float] = {}
    for name in read:
        sipnet_name = resolve_parameter_name(name)
        if sipnet_name not in set_by_vector:
            values[sipnet_name] = float(model.base_params.dataarray(sipnet_name))
    return values


def _aggregated(model_output: xr.Dataset, freq: str) -> xr.Dataset:
    """Every variable of one run's output aggregated to *freq* by its kind's default.

    Through :func:`~sipnet_calibration.observation.aggregate_time`, the
    aggregation the plotting layer and the observation operators use, and
    merged with an exact join, so a variable can never be reindexed onto
    another's time axis.
    """
    arrays = [aggregate_time(model_output[name], freq) for name in model_output.data_vars]
    merged = xr.merge(arrays, join="exact", combine_attrs="drop_conflicts")
    return merged.assign_attrs(model_output.attrs)

def _site_blocks(
    vector: ObservationVector | None,
) -> tuple[dict[int, ObservationVector], dict[int, np.ndarray]]:
    """Per observed site, the vector restricted to it and where its block sits in Flat."""
    slices: dict[int, ObservationVector] = {}
    positions: dict[int, np.ndarray] = {}
    for site in () if vector is None else vector.sites:
        slices[site] = vector.select(sites=[site])
        positions[site] = vector.positions(site=site)
        check_site_slice_is_the_site_block(slices[site], vector.index[positions[site]], site)
    return slices, positions


def _sort_records(
    result: Any, n_members: int, sites: Sequence[int]
) -> tuple[
    dict[tuple[int, int], _RunOutput], np.ndarray, pd.DataFrame, list[tuple[dict, BaseException]]
]:
    """Split PyEns's records into outputs, successes, model failures and machinery failures.

    Member labels are row positions, which :func:`check_table_is_a_sipnet_table`
    guarantees.
    """
    column = {site: j for j, site in enumerate(sites)}
    outputs: dict[tuple[int, int], _RunOutput] = {}
    rows: list[dict[str, Any]] = []
    broken: list[tuple[dict, BaseException]] = []
    for record in result:
        member, site = int(record.coordinate[MEMBER]), int(record.coordinate[SITE])
        if not record.failed:
            outputs[(member, site)] = record.output
        elif _is_model_failure(record.output):
            rows.append({
                MEMBER: member, SITE: site, "error": _error_name(record.output),
                "message": str(record.output)[:500],
            })
        else:
            broken.append((record.coordinate, record.output))
    succeeded = np.zeros((n_members, len(sites)), dtype=bool)
    keys = np.asarray([(member, column[site]) for member, site in outputs], dtype=np.intp)
    if keys.size:
        succeeded[keys[:, 0], keys[:, 1]] = True
    failures = pd.DataFrame(rows, columns=[MEMBER, SITE, "error", "message"])
    return outputs, succeeded, failures, broken


def _run_succeeded(
    succeeded: np.ndarray, members: np.ndarray, sites: Sequence[int]
) -> xr.DataArray:
    """*succeeded*, ``(J, S)``, labeled on ``(member, site)``."""
    return xr.DataArray(
        succeeded, dims=(MEMBER, SITE), coords={MEMBER: members, SITE: list(sites)},
        name="run_succeeded",
    )


def _qualified_name(error_type: type) -> str:
    """The name PyEns's ``RemoteError.type_name`` gives an exception class."""
    return f"{error_type.__module__}.{error_type.__qualname__}"


def _is_model_failure(error: BaseException) -> bool:
    if isinstance(error, MODEL_FAILURES):
        return True
    if isinstance(error, RemoteError):
        # An exception that does not survive pickling crosses the process
        # boundary as a RemoteError naming its class in full; compare the full
        # name, so another library's ValidationError is not taken for pydantic's.
        return error.type_name in {_qualified_name(t) for t in MODEL_FAILURES}
    return False


def _error_name(error: BaseException) -> str:
    """The class name of the exception a run raised, through a ``RemoteError`` too."""
    if isinstance(error, RemoteError):
        return error.type_name.rsplit(".", 1)[-1]
    return type(error).__name__


def _stacked_model_output(
    outputs: Mapping[tuple[int, int], _RunOutput],
    n_members: int,
    sites: Sequence[int],
    site_table: pd.DataFrame,
) -> xr.Dataset:
    """The runs' output on the full ``(member, site, time)`` grid, ``NaN`` where a run failed."""
    runs = {(site, member): output.model_output for (member, site), output in outputs.items()}
    stacked = stack_model_outputs(runs, site_table=site_table)
    full = stacked.reindex({
        MEMBER: np.arange(n_members, dtype=stacked[MEMBER].dtype),
        SITE: np.asarray(sites, dtype=stacked[SITE].dtype),
    })
    # A site at which every run failed is absent from the stack, so the reindex
    # leaves its lon/lat NaN; they are the site table's whatever the runs did.
    located = {
        name: full[name].copy(data=site_table.loc[list(sites), name].to_numpy(np.float64))
        for name in ("lon", "lat")
    }
    return full.assign_coords(located)


def _with_evaluation(error: RuntimeError, evaluation: ForwardEvaluation) -> RuntimeError:
    error.evaluation = evaluation  # type: ignore[attr-defined]
    return error


# ── checks ────────────────────────────────────────────────────────────────────


def check_forward_model_arguments(
    model: Any,
    parameter_vector: ParameterVector,
    *,
    climate: Mapping[int, Any],
    backend: Any,
    observation_vector: ObservationVector | None,
    output_variable_names: Sequence[str] | None,
    freq: Any,
    site_table: pd.DataFrame | None,
) -> None:
    """Every check on :class:`ForwardModel`'s arguments that needs nothing computed."""
    sites = parameter_vector.sites
    check_model_is_a_sipnet_model(model)
    check_backend(backend)
    check_output_variables_are_named(output_variable_names, observation_vector)
    check_freq_is_for_the_prior_predictive(freq, observation_vector)
    if freq is not None:
        check_frequency(freq)
    check_climate_covers_the_sites(climate, sites)
    check_climate_is_file_backed(climate, sites, backend)
    if observation_vector is not None:
        check_observation_sites_are_run(observation_vector, sites)
    if site_table is not None:
        check_site_table_locates_the_sites(site_table, sites)


def check_model_is_a_sipnet_model(model: Any) -> None:
    if not isinstance(model, SIPNETModel):
        raise TypeError(f"model must be a pysipnet SIPNETModel, got {type(model).__name__}.")


def check_backend(backend: Any) -> None:
    if not isinstance(backend, Backend):
        raise TypeError(f"backend must be a pyens Backend, got {type(backend).__name__}.")


def check_output_variables_are_named(
    output_variable_names: Sequence[str] | None, vector: ObservationVector | None
) -> None:
    if output_variable_names is None and vector is None:
        raise ValueError(
            "name the output variables each run returns (output_variable_names=), "
            "or give an observation_vector whose operators say what they read."
        )


def check_freq_is_for_the_prior_predictive(freq: Any, vector: ObservationVector | None) -> None:
    if freq is not None and vector is not None:
        raise ValueError(
            "freq= aggregates model output for the prior predictive; with an observation "
            "vector the operators decide their own alignment, so freq would be ignored. "
            "Drop one of the two."
        )


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


def check_climate_is_file_backed(
    climate: Mapping[int, ClimateDrivers], sites: Sequence[int], backend: Backend
) -> None:
    if isinstance(backend, SequentialBackend):
        return
    in_memory = [s for s in sites if climate[s].source_path is None]
    if in_memory:
        raise ValueError(
            f"the drivers of site(s) {in_memory[:10]} are held in memory, and under "
            f"{type(backend).__name__} every run would carry a copy of them. Write them "
            "with ClimateDrivers.to_file and open them with ClimateDrivers.from_path."
        )


def check_observation_sites_are_run(vector: ObservationVector, sites: Sequence[int]) -> None:
    extra = sorted(set(vector.sites) - set(sites))
    if extra:
        raise ValueError(
            f"the observation vector observes site(s) {extra[:10]} that the parameter vector "
            "does not run; select the observations to the vector's sites first "
            "(observation_vector.select(sites=...))."
        )


def check_site_table_locates_the_sites(site_table: pd.DataFrame, sites: Sequence[int]) -> None:
    table = site_lookup(site_table)
    if not {"lon", "lat"} <= set(table.columns):
        raise ValueError(
            "the site table must carry 'lon' and 'lat' columns, as load_sites() returns them; "
            f"got {list(table.columns)}."
        )
    if table.index.has_duplicates:
        repeated = sorted(set(table.index[table.index.duplicated()].tolist()))
        raise ValueError(
            f"the site table lists site(s) {repeated[:10]} more than once; a site has one "
            "row, as load_sites() gives it. Drop the repeated rows."
        )
    missing = [s for s in sites if s not in table.index]
    if missing:
        raise ValueError(f"the site table has no row for site(s) {missing[:10]}.")


def check_output_variable_names_cover_the_operators(
    output_variable_names: Sequence[str], vector: ObservationVector
) -> None:
    missing = [n for n in vector.output_variable_names if n not in output_variable_names]
    if missing:
        raise ValueError(
            f"the observation vector's operators read {missing}, which "
            "output_variable_names does not include; add them, or omit "
            "output_variable_names to take the operators' own."
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


def check_every_kind_has_a_default_method(output_variable_names: Sequence[str]) -> None:
    for name in output_variable_names:
        check_kind_has_a_default_method(name, resolve_output_variable(name).kind)


def check_kind_has_a_default_method(name: str, kind: Any) -> None:
    if kind not in DEFAULT_METHOD_FOR_KIND:
        raise ValueError(
            f"{name!r} is of kind {getattr(kind, 'value', kind)!r}, which no resampling "
            "method leaves unchanged, so freq= has no rule to aggregate it by. Leave it "
            "out of output_variable_names, or drop freq= and aggregate it yourself."
        )


def check_site_slice_is_the_site_block(
    site_slice: ObservationVector, site_block: pd.MultiIndex, site: int
) -> None:
    if not site_slice.index.equals(site_block):
        raise ValueError(
            f"the observation vector's selection to site {site} is not that site's block of "
            "the whole vector, so its predictions cannot be placed by position. The vector "
            "must be site-major; this is a defect in ObservationVector, not in the inputs."
        )


def check_table_is_a_sipnet_table(
    table: Any, sites: Sequence[int], *, n_members: int, expected: Sequence[str] | None = None
) -> None:
    if not isinstance(table, xr.Dataset):
        raise TypeError(f"to_sipnet_table must return an xr.Dataset, got {type(table).__name__}.")
    if set(table.dims) != {MEMBER, SITE}:
        raise ValueError(
            f"a SIPNET table for a batch has dims exactly (member, site), got "
            f"{tuple(table.dims)}; reduce or select any other dimension in the hook."
        )
    if table[MEMBER].values.tolist() != list(range(n_members)):
        raise ValueError(
            f"the SIPNET table's members must be 0 to {n_members - 1}, in the order of theta's "
            f"{n_members} rows, got {table[MEMBER].values.tolist()[:10]}; a member is the row "
            "of theta it was made from."
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
    if expected is not None and {str(n) for n in table.data_vars} != set(expected):
        raise ValueError(
            f"the SIPNET table sets {sorted(table.data_vars)}, but the model was built for "
            f"{sorted(expected)}; a ForwardModel's free fields are fixed when it is built."
        )


def check_theta_has_the_batch_shape(theta: np.ndarray, input_dimension: int) -> None:
    if theta.ndim not in (1, 2) or theta.shape[-1] != input_dimension or theta.shape[0] == 0:
        raise ValueError(
            f"theta must be (D,) or (J, D) with J >= 1 and D = {input_dimension}, got shape "
            f"{theta.shape}."
        )


def check_theta_is_finite(theta: np.ndarray) -> None:
    if not np.isfinite(theta).all():
        raise ValueError("theta holds a non-finite value; the parameter vector never produces one.")


def check_output_is_finite(dataset: xr.Dataset, site: int) -> None:
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


def check_no_run_failed_in_the_machinery(
    broken: Sequence[tuple[dict, BaseException]], evaluation: ForwardEvaluation
) -> None:
    if not broken:
        return
    coordinate, error = broken[0]
    kinds = sorted({getattr(e, "type_name", type(e).__name__) for _, e in broken})
    raise _with_evaluation(
        RuntimeError(
            f"{len(broken)} run(s) failed in the machinery rather than at their parameters "
            f"({', '.join(kinds)}); the first, at {coordinate}, says: {error}. A "
            f"{TaskFailedError.__name__}, a missing binary or an import error on a worker says "
            "nothing about the parameters, so it is raised rather than turned into a NaN row; "
            "the runs that were collected are on this error's `evaluation` attribute."
        ),
        evaluation,
    )


def check_some_run_succeeded(evaluation: ForwardEvaluation) -> None:
    if not bool(evaluation.run_succeeded.any()):
        raise _with_evaluation(
            RuntimeError(
                "every run failed at its parameters; there is no model output to stack NaN "
                "onto. Draw parameters the model can run, or read the failures on this "
                "error's `evaluation` attribute."
            ),
            evaluation,
        )
