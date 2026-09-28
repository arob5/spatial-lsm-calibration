"""The forward model: unconstrained parameters to predictions, over a site
set, with SIPNET run once per run of the ensemble.

Where this sits
---------------
::

    theta (J, D), external inputs
      --SIPNETParameterMap.sipnet_parameter_fields-->  SIPNET parameter fields
      --PyEns, one SIPNETModel run per (sample, crossed labels, site)-->  model output
      --ObservationVector.predict and .flat on the worker-->  one site's segment of Flat
      --placed at its run and at positions(site=) by the calling process-->  (R, N)

:class:`ForwardModel` is the one object this module adds: the callable
``(J, D) -> (J, N)`` that pyEKI's ``run`` takes as ``forward``, and that an
MCMC target calls. Everything it composes exists elsewhere: the parameter
vector, the SIPNET parameter map, pySIPNET's ``SIPNETModel``, PyEns's
``PartialSpec``/``EnsembleRunner``/``Backend`` and its xarray bridge, the
observation vector, and the stacking of runs
(:func:`sipnet_calibration.fields.stack_model_outputs`).

What it reads
-------------
A :class:`pysipnet.model.SIPNETModel`, a
:class:`~sipnet_calibration.parameter_vector.ParameterVector` and a
:class:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap`, optional
:data:`~sipnet_calibration.sipnet_parameter_map.ExternalInputs`, one
``ClimateDrivers`` per site, a PyEns backend, and either an
:class:`~sipnet_calibration.observation.ObservationVector` (the calibration
path: each run reduced to its site's predictions on the worker) or the
output variable names each run returns (the prior-predictive path).

Data model
----------
**The runs.** External inputs pair with theta by dim name: an input on the
model's ``batch_dim`` zips with theta's rows, and one on any other batch dim
(a **crossed dim**) runs with every row. PyEns enumerates the runs, one per
combination of a row, the crossed dims' labels and a site: ``J * prod(M_k) *
S`` of them, :meth:`ForwardModel.runs_per_sample` ``* J``.

**The run index**, :attr:`ForwardEvaluation.run_index`, names the rows of an
evaluation: ``batch_dim``, then the crossed dims in the order they first
appear in the external inputs. It has ``R = J * prod(M_k)`` rows, in C
order over its levels; without crossed dims, ``R = J`` and it is the batch
labels.

:class:`ForwardEvaluation`, what one call of :meth:`ForwardModel.evaluate`
produced:

``theta``
    ``(J, D)`` float64 ``jax.Array``.
``sipnet_parameter_fields``
    The :data:`~sipnet_calibration.fields.SIPNETParameterFields` that were
    run, each variable on the dims it was computed from.
``run_index``
    ``pd.Index`` or ``pd.MultiIndex``, as above.
``model_output``
    The :data:`~sipnet_calibration.fields.ModelOutput` of the runs on
    ``(batch_dim, *crossed dims, site, time)``, ``NaN`` where a run failed;
    with ``freq`` its attributes gain pySIPNET's ``resampling_frequency``
    and ``timestep_length_source``. ``None`` when an observation vector was
    given.
``predictions``
    ``(R, N)`` float64 ``jax.Array`` in the run index's and the observation
    vector's order, ``NaN`` in every entry of a row with a failed run;
    ``None`` without an observation vector. :meth:`ForwardEvaluation.predicted_fields`
    is its labeled form.
``run_succeeded``
    bool field on ``(batch_dim, *crossed dims, site)``.
``failures``
    ``pd.DataFrame``: a column per level of the run index, ``site``,
    ``error`` (the exception's class name) and ``message``, one row per run
    that failed at its parameters.
``valid``
    bool ``(R,)`` ``jax.Array``: every run of the row succeeded, its SIPNET
    parameters are in pySIPNET's domains, and its predictions are finite.
``out_of_domain_fraction``
    The fraction of rows with a SIPNET parameter outside pySIPNET's domains;
    ``0.0`` unless ``out_of_domain="fail_row"``.

A run **fails at its parameters** when pySIPNET refuses them
(``pydantic.ValidationError``), SIPNET exits non-zero or writes nothing
(``SIPNETRunError``), the run times out (``subprocess.TimeoutExpired``), or
its output holds a non-finite value in a variable that was read
(:class:`ModelOutputNotFiniteError`): its row is ``NaN``, which pyEKI repairs
and a sampler rejects. Anything else a worker returns is the **machinery**
failing, and is raised after the batch is collected as a ``RuntimeError``
whose ``evaluation`` attribute holds what was collected. On the
prior-predictive path a batch in which every run failed at its parameters is
raised the same way.

**SIPNET parameters outside pySIPNET's domains** mean the prior and the map
put mass where SIPNET is undefined. By default the model refuses them before
anything runs (:class:`~sipnet_calibration.sipnet_parameter_map.SIPNETParametersOutOfDomainError`);
``out_of_domain="fail_row"`` marks their rows invalid instead, which
truncates the prior to the region the map sends into the domains: a prior
that neither ``Prior.log_prob`` nor ``Prior.sample`` knows.

Functions
---------
:class:`ForwardModel`
    ``evaluate(theta) -> ForwardEvaluation``; ``__call__(theta)``, the
    ``(J, D) -> (J, N)`` map, for evaluations without crossed dims;
    ``describe()``.
:class:`ForwardEvaluation`
    The record above.
:data:`MODEL_FAILURES`, :class:`ModelOutputNotFiniteError`
    The exceptions that mean a run failed at its parameters.

Notes
-----
**The operators run on the worker**, because reading a run's output back in
the calling process costs more than the run. Each run receives only its
site's slice of the observation vector and returns that slice's Flat, which
the calling process writes at ``positions(site=)``: right because the
observation vector is site-major, which ``__init__`` checks per site.

**The ``PartialSpec`` is built once**: the climate, the site ids and the
sites' observation slices along one site axis. On every call PyEns zips the
SIPNET parameter fields with that axis and crosses them along their other
dims.

**The SIPNET parameters an operator reads** are the run's own
``SIPNETResult.parameters``, whichever set them.

Usage
-----
::

    forward = ForwardModel(model, prior.parameter_vector, sipnet_map, climate=climate,
                           backend=LocalBackend(8), observation_vector=observation_vector)
    result = pyeki.eki.run(state, forward, observation_vector.y, noise_cov, schedule=...)

    predictive = ForwardModel(model, vector, sipnet_map, climate=climate,
                              backend=LocalBackend(8), external_inputs=initial_states,
                              output_variable_names=("nee",), freq="1D")
    evaluation = predictive.evaluate(prior.sample(key, 100))
    evaluation.model_output      # (sample, initial_condition_member, site, time)
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict
from pydantic import ValidationError
from pyens import (
    Axis,
    EnsembleRunner,
    EnsembleSpec,
    Grid,
    PartialSpec,
    RemoteError,
    SequentialBackend,
    TaskFailedError,
)
from pyens.backends import Backend
from pyens.xarray import fields_from_dataset
from pysipnet.climate import ClimateDrivers
from pysipnet.model import SIPNETModel
from pysipnet.parameters.model import SIPNETParameters
from pysipnet.resample import STEP_LENGTH_RESAMPLED
from pysipnet.runner import SIPNETRunError
from pysipnet.variables import resolve_output_variable

from sipnet_calibration.conventions import BATCH_LABEL_DTYPE, SAMPLE, SITE, SITE_DTYPE, ReadOnlyCopies
from sipnet_calibration.fields import (
    Field,
    ModelOutput,
    SIPNETParameterFields,
    batch_coordinate,
    check_batch_dim_name_is_not_a_model_output_name,
    check_batch_dim_name_is_not_reserved,
    resolve_output_variable_names,
    stack_model_outputs,
    to_model_output,
)
from sipnet_calibration.observation import (
    DEFAULT_METHOD_FOR_KIND,
    ObservationVector,
    aggregate_time,
    check_batch_dim_is_not_an_observation_source_name,
)
from sipnet_calibration.observation.time_alignment import check_frequency_is_an_offset_alias
from sipnet_calibration.parameter_vector import ParameterVector, check_batch_dim_name_is_not_taken
from sipnet_calibration.sipnet_parameter_map import (
    ExternalInputs,
    SIPNETParameterMap,
    SIPNETParametersOutOfDomainError,
    check_sipnet_parameter_map_fits,
    check_sipnet_parameter_map_is_in_domain_at_the_corners,
    validate_external_inputs,
)
from sipnet_calibration.sites import site_locations, site_lookup
from sipnet_calibration.validation import as_batched_flat, is_one_vector, truncated

__all__ = [
    "MODEL_FAILURES",
    "ForwardEvaluation",
    "ForwardModel",
    "ModelOutputNotFiniteError",
]


class ModelOutputNotFiniteError(RuntimeError):
    """A run completed but wrote a non-finite value in a variable that was read.

    SIPNET exits 0 on a blow-up, so the failure shows only in the output; it
    is a failure at the run's parameters, and pickles as a plain message.
    """


#: The exceptions that mean a run failed **at its parameters**: its row
#: becomes NaN. Everything else a worker returns is the machinery failing and
#: is raised.
MODEL_FAILURES: tuple[type[BaseException], ...] = (
    SIPNETRunError,
    ValidationError,
    subprocess.TimeoutExpired,
    ModelOutputNotFiniteError,
)


@dataclass(frozen=True, eq=False, kw_only=True)
class ForwardEvaluation:
    """What one evaluation of a :class:`ForwardModel` produced.

    The fields are described in the module docstring's Data model. Nothing
    read from it changes it: the arrays are ``jax.Array``\\ s, and the
    xarray and pandas members read-only copies, on every read
    (:class:`~sipnet_calibration.conventions.ReadOnlyCopies`). Compared and
    hashed by identity (``eq=False``).
    """

    theta: jax.Array
    sipnet_parameter_fields: SIPNETParameterFields = ReadOnlyCopies()
    run_index: pd.Index
    model_output: ModelOutput | None = ReadOnlyCopies()
    predictions: jax.Array | None
    run_succeeded: Field = ReadOnlyCopies()
    failures: pd.DataFrame = ReadOnlyCopies()
    valid: jax.Array
    out_of_domain_fraction: float = 0.0
    observation_vector: ObservationVector | None = None

    def predicted_fields(self) -> dict[str, Field]:
        """The predictions as the observation vector's Fields, each on
        ``(batch_dim, *crossed dims, site[, time])``.

        Raises
        ------
        ValueError
            Without predictions.
        """
        check_evaluation_has_predictions(self)
        run_dims = list(self.run_index.names)
        if len(run_dims) == 1:
            return self.observation_vector.fields(self.predictions, batch_dim=run_dims[0])
        flat_fields = self.observation_vector.fields(self.predictions, batch_dim=_ROW)
        out = {}
        for name, field in flat_fields.items():
            row_coordinates = xr.Coordinates.from_pandas_multiindex(self.run_index, _ROW)
            unstacked = field.assign_coords(row_coordinates).unstack(_ROW)
            unstacked = unstacked.transpose(*run_dims, *[d for d in field.dims if d != _ROW])
            out[name] = unstacked.assign_coords(
                {dim: batch_coordinate(dim, unstacked[dim].values) for dim in run_dims}
            )
        return out


class ForwardModel:
    """The forward map :math:`G`: unconstrained parameters to predictions.

    .. math::

        G(\\theta) = \\big(H_s(\\mathrm{SIPNET}(\\psi_s))\\big)_s, \\qquad
        \\psi_s = M\\big(T(\\theta)^{(s)}, u_s, c_s\\big),

    with :math:`T` the vector's transform, :math:`M` the SIPNET parameter
    map, :math:`u_s` a run's external inputs, and :math:`H_s` the
    observation operators at the run's site.

    Parameters
    ----------
    model:
        The :class:`pysipnet.model.SIPNETModel` every run goes through.
    parameter_vector:
        The vector theta is of; its sites are the sites run, located by its
        site table. Pass ``prior.parameter_vector``, so the prior and the
        forward model share one.
    sipnet_parameter_map:
        How theta, and the external inputs, reach SIPNET. It is checked to
        fit the vector and the external inputs.
    climate:
        ``{site id: ClimateDrivers}``, a superset of the vector's sites.
        Under any backend but ``SequentialBackend`` each must be
        file-backed, opened with ``ClimateDrivers.from_path``.
    backend:
        The PyEns backend the runs execute on.
    external_inputs:
        :data:`~sipnet_calibration.sipnet_parameter_map.ExternalInputs` the
        map reads, paired with theta by dim name.
    out_of_domain:
        ``"raise"`` (the default) refuses SIPNET parameters outside
        pySIPNET's domains, at construction for the map at the corners of
        theta and on every evaluation; ``"fail_row"`` marks their rows
        invalid, which truncates the prior.
    observation_vector:
        The observation vector whose operators reduce each run on the worker
        and whose order the predictions take. Its sites must be among the
        vector's.
    output_variable_names:
        Without an observation vector, the pySIPNET output variables each run
        returns. With one, defaults to those its operators read.
    freq:
        Without an observation vector, aggregate each run's output on the
        worker to this pandas frequency, each variable by the method that
        leaves its kind unchanged (:data:`~sipnet_calibration.observation.DEFAULT_METHOD_FOR_KIND`).
    batch_dim:
        The name of the batch dim of theta's rows, on every output.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If the arguments cannot make one forward model, checked before
        anything runs; the message names the rule.
    KeyError
        If an output variable is unknown, or the map reads a value neither
        the vector nor the external inputs hold.
    """

    def __init__(
        self,
        model: SIPNETModel,
        parameter_vector: ParameterVector,
        sipnet_parameter_map: SIPNETParameterMap,
        *,
        climate: Mapping[int, ClimateDrivers],
        backend: Backend,
        external_inputs: ExternalInputs | None = None,
        out_of_domain: Literal["raise", "fail_row"] = "raise",
        observation_vector: ObservationVector | None = None,
        output_variable_names: Sequence[str] | None = None,
        freq: str | None = None,
        batch_dim: str = SAMPLE,
    ) -> None:
        check_batch_dim_name_is_not_reserved(batch_dim, message_name="batch_dim")
        check_batch_dim_name_is_not_taken(parameter_vector, batch_dim)
        check_forward_model_arguments(
            model,
            parameter_vector,
            climate=climate,
            backend=backend,
            observation_vector=observation_vector,
            output_variable_names=output_variable_names,
            freq=freq,
            out_of_domain=out_of_domain,
        )
        if external_inputs is not None:
            validate_external_inputs(external_inputs, batch_dim=batch_dim)
        check_sipnet_parameter_map_fits(sipnet_parameter_map, parameter_vector, external_inputs)
        if out_of_domain == "raise":
            check_sipnet_parameter_map_is_in_domain_at_the_corners(
                sipnet_parameter_map, parameter_vector, external_inputs, batch_dim=batch_dim
            )
        self._model = model
        self._parameter_vector = parameter_vector
        self._sipnet_parameter_map = sipnet_parameter_map
        self._external_inputs = external_inputs
        self._out_of_domain = out_of_domain
        self._sites: tuple[int, ...] = parameter_vector.sites
        self._backend = backend
        self._observation_vector = observation_vector
        self._freq = freq
        self._batch_dim = batch_dim
        self._crossed_dims = _crossed_dims(external_inputs, batch_dim)
        self._climate = frozendict({site: climate[site] for site in self._sites})
        self._output_variable_names = _output_variable_names(output_variable_names, observation_vector)
        check_output_variables_can_be_returned(self._output_variable_names, model, freq)
        for dim in (batch_dim, *self._crossed_dims):
            check_batch_dim_name_is_not_reserved(dim, message_name="a batch dim")
            check_batch_dim_name_is_not_a_model_output_name(
                dim, self._output_variable_names, message_name="a batch dim"
            )
            if observation_vector is not None:
                check_batch_dim_is_not_an_observation_source_name(
                    observation_vector.observation_sources, dim
                )
        site_table = parameter_vector.site_table
        self._site_locations = site_locations(self._sites, site_table)
        self._site_table = site_lookup(site_table)
        if observation_vector is not None:
            check_base_parameters_set_what_the_operators_read(
                model,
                observation_vector.sipnet_parameter_names_read,
                sipnet_parameter_map.sipnet_parameter_names_written,
            )
        self._site_axis = Axis(SITE, labels=list(self._sites))
        self._site_slices, self._site_positions = _site_segments(observation_vector)
        self._partial = self._build_partial()
        self._run = _Run(
            model=model,
            output_variable_names=self._output_variable_names,
            returns_model_output=observation_vector is None,
            freq=freq,
            site_table=self._site_table,
        )

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def model(self) -> SIPNETModel:
        return self._model

    @property
    def parameter_vector(self) -> ParameterVector:
        return self._parameter_vector

    @property
    def sipnet_parameter_map(self) -> SIPNETParameterMap:
        return self._sipnet_parameter_map

    @property
    def external_inputs(self) -> ExternalInputs | None:
        """The external inputs, a copy."""
        return None if self._external_inputs is None else self._external_inputs.copy(deep=True)

    @property
    def out_of_domain(self) -> str:
        return self._out_of_domain

    @property
    def observation_vector(self) -> ObservationVector | None:
        return self._observation_vector

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def freq(self) -> str | None:
        return self._freq

    @property
    def climate(self) -> Mapping[int, ClimateDrivers]:
        """``{site id: ClimateDrivers}`` for the sites run, read-only."""
        return self._climate

    @property
    def sites(self) -> tuple[int, ...]:
        """The sites run: the vector's, in its order."""
        return self._sites

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return self._output_variable_names

    @property
    def batch_dim(self) -> str:
        return self._batch_dim

    @property
    def crossed_dims(self) -> tuple[str, ...]:
        """The external inputs' batch dims other than ``batch_dim``, in the
        order they first appear."""
        return self._crossed_dims

    @property
    def runs_per_sample(self) -> int:
        """The runs per row of theta: ``prod(M_k) * S``."""
        return int(np.prod([self._external_inputs.sizes[d] for d in self._crossed_dims], dtype=int)) * len(
            self._sites
        )

    @property
    def input_dimension(self) -> int:
        """``D``."""
        return self.parameter_vector.dimension

    @property
    def output_dimension(self) -> int:
        """``N``, the observation vector's dimension.

        Raises
        ------
        ValueError
            Without an observation vector.
        """
        return self._observation_vector_for("output_dimension").dimension

    def describe(self) -> pd.Series:
        """What the model runs, before it runs: ``sites``, ``batch_dim``,
        ``crossed_dims`` (name and size), ``runs_per_sample``, ``samples``
        (fixed by external inputs on ``batch_dim``, else ``None``),
        ``out_of_domain`` and the ``output``."""
        samples = (
            self._external_inputs.sizes[self._batch_dim]
            if self._external_inputs is not None and self._batch_dim in self._external_inputs.dims
            else None
        )
        return pd.Series(
            {
                "sites": len(self._sites),
                "batch_dim": self._batch_dim,
                "crossed_dims": {d: self._external_inputs.sizes[d] for d in self._crossed_dims},
                "runs_per_sample": self.runs_per_sample,
                "samples": samples,
                "out_of_domain": self._out_of_domain,
                "output": (
                    f"predictions, N = {self._observation_vector.dimension}"
                    if self._observation_vector is not None
                    else f"model output {list(self._output_variable_names)}, freq {self._freq}"
                ),
            },
            name="ForwardModel",
        )

    def __repr__(self) -> str:
        what = (
            f"observation_vector={self.observation_vector!r}"
            if self.observation_vector is not None
            else f"output_variable_names={self.output_variable_names!r}, freq={self.freq!r}"
        )
        return f"ForwardModel(D={self.input_dimension}, sites={len(self.sites)}, {what})"

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, theta: Any) -> ForwardEvaluation:
        """Run SIPNET once per run, and collect what came back.

        Parameters
        ----------
        theta:
            ``(J, D)``, one row per sample, or ``(D,)`` for one.

        Returns
        -------
        ForwardEvaluation
            With ``predictions`` on the calibration path and
            ``model_output`` on the prior-predictive path.

        Raises
        ------
        TypeError
            If *theta* is not a rectangular array of real numbers.
        ValueError
            If *theta* is not ``(D,)`` or ``(J, D)`` with ``J >= 1``, holds a
            non-finite value, or has other rows than external inputs on
            ``batch_dim`` label.
        SIPNETParametersOutOfDomainError
            With ``out_of_domain="raise"``, if a SIPNET parameter lies outside
            pySIPNET's domain; nothing runs.
        RuntimeError
            If a run failed in the machinery rather than at its parameters,
            or, on the prior-predictive path, every run failed at its
            parameters; the error's ``evaluation`` attribute holds what was
            collected.
        """
        theta = np.asarray(as_batched_flat(theta, self.input_dimension, message_name="theta"))
        check_theta_has_a_row(theta)
        check_theta_is_finite(theta)
        sipnet_parameter_fields = self._sipnet_parameter_map.sipnet_parameter_fields(
            self._parameter_vector, theta, external_inputs=self._external_inputs, batch_dim=self._batch_dim
        )
        outside = self._sipnet_parameter_map.out_of_domain(sipnet_parameter_fields)
        if self._out_of_domain == "raise":
            check_sipnet_parameters_are_in_the_domain(outside)
        run_index = self._run_index(len(theta), sipnet_parameter_fields)
        grids = fields_from_dataset(sipnet_parameter_fields, axes={SITE: self._site_axis})
        ensemble_result = EnsembleRunner(self._run, self.backend).run(self._partial(**grids))
        run_outputs, succeeded, failures, machinery_failures = _sort_records(
            ensemble_result, run_index, self.sites
        )
        row_in_domain = _rows_in_domain(outside, run_index)
        collected = ForwardEvaluation(
            theta=jnp.asarray(theta),
            sipnet_parameter_fields=sipnet_parameter_fields,
            run_index=run_index,
            model_output=None,
            predictions=None,
            run_succeeded=_run_succeeded(succeeded, run_index, self.sites, self._site_locations),
            failures=failures,
            valid=jnp.zeros(len(run_index), dtype=bool),
            out_of_domain_fraction=float(1.0 - row_in_domain.mean()),
            observation_vector=self.observation_vector,
        )
        check_no_run_failed_in_the_machinery(machinery_failures, collected)
        row_succeeded = succeeded.all(axis=1) & row_in_domain
        if self.observation_vector is None:
            check_some_run_succeeded(collected)
            model_output = _stacked_model_output(
                run_outputs, run_index, self.sites, site_table=self._site_table,
                site_locations=self._site_locations,
            )
            return replace(collected, model_output=model_output, valid=jnp.asarray(row_succeeded))
        predictions = self._placed_predictions(run_outputs, len(run_index))
        # A row with any failed run is invalid as a whole: pyEKI updates per
        # row, so a row that is partly a prediction cannot be used.
        predictions[~row_succeeded] = np.nan
        valid = row_succeeded & np.isfinite(predictions).all(axis=1)
        return replace(collected, predictions=jnp.asarray(predictions), valid=jnp.asarray(valid))

    def __call__(self, theta: Any) -> jax.Array:
        """``evaluate(theta).predictions``, pyEKI's ``(J, D) -> (J, N)``: ``(N,)``
        for a ``(D,)`` theta.

        Raises
        ------
        ValueError
            Without an observation vector, or with crossed dims, whose ``R``
            rows are not theta's ``J``: call ``evaluate`` and reduce over them.
        TypeError, RuntimeError, SIPNETParametersOutOfDomainError
            As :meth:`evaluate`.
        """
        self._observation_vector_for("__call__")
        check_no_crossed_dims(self._crossed_dims)
        predictions = self.evaluate(theta).predictions
        return predictions[0] if is_one_vector(theta) else predictions

    # ── supporting methods ────────────────────────────────────────────────────

    def _observation_vector_for(self, what: str) -> ObservationVector:
        """The observation vector, or a ``ValueError`` saying *what* needs one."""
        if self.observation_vector is None:
            raise ValueError(
                f"{what} needs an observation vector; this ForwardModel was built without one, "
                "for model output only. Use evaluate(theta).model_output."
            )
        return self.observation_vector

    def _run_index(self, n_samples: int, sipnet_parameter_fields: xr.Dataset) -> pd.Index:
        if not self._crossed_dims:
            return pd.Index(np.arange(n_samples, dtype=BATCH_LABEL_DTYPE), name=self._batch_dim)
        levels = [np.arange(n_samples, dtype=BATCH_LABEL_DTYPE)] + [
            sipnet_parameter_fields[d].values.astype(BATCH_LABEL_DTYPE) for d in self._crossed_dims
        ]
        return pd.MultiIndex.from_product(levels, names=[self._batch_dim, *self._crossed_dims])

    def _build_partial(self) -> PartialSpec:
        climate = Grid({s: self.climate[s] for s in self.sites}, along=self._site_axis)
        site_ids = Grid(list(self.sites), along=self._site_axis)
        site_observation_vectors = Grid(
            {s: self._site_slices.get(s) for s in self.sites}, along=self._site_axis
        )
        written = self._sipnet_parameter_map.sipnet_parameter_names_written
        placeholders = {name: Grid([0.0] * len(self.sites), along=self._site_axis) for name in written}
        spec = EnsembleSpec(
            inputs={
                "climate": climate,
                "site": site_ids,
                "site_observation_vector": site_observation_vectors,
                **placeholders,
            }
        )
        return spec.freeze(free=list(written))

    def _placed_predictions(self, run_outputs: Mapping[tuple[int, int], _RunOutput], n_rows: int) -> np.ndarray:
        """``(R, N)``: each run's segment of Flat at its row and its site's positions."""
        observation_vector = self._observation_vector_for("predictions")
        predictions = np.full((n_rows, observation_vector.dimension), np.nan)
        for (row, site), run_output in run_outputs.items():
            if run_output.predictions is not None:
                predictions[row, self._site_positions[site]] = run_output.predictions
        return predictions


# ── the per-run callable ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _RunOutput:
    """What one run sends back: its model output, its site's predictions, or neither."""

    model_output: xr.Dataset | None
    predictions: np.ndarray | None


@dataclass(frozen=True)
class _Run:
    """One SIPNET run, reduced on the worker; picklable, built once per model.

    On the prior-predictive path the run sends back its output (aggregated,
    with ``freq``); otherwise its site's predictions, or nothing at a site
    no observation source observes. The SIPNET parameters the operators read
    are the run's own ``SIPNETResult.parameters``.
    """

    model: SIPNETModel
    output_variable_names: tuple[str, ...]
    returns_model_output: bool
    freq: str | None
    site_table: pd.DataFrame

    def __call__(
        self,
        *,
        climate: ClimateDrivers,
        site: int,
        site_observation_vector: ObservationVector | None = None,
        **sipnet_overrides: Any,
    ) -> _RunOutput:
        site = int(site)
        sipnet_result = self.model(climate=climate, **sipnet_overrides)
        dataset = sipnet_result.outputs.select(list(self.output_variable_names))
        check_output_is_finite(dataset, site)
        if not self.returns_model_output and site_observation_vector is None:
            return _RunOutput(model_output=None, predictions=None)
        model_output = to_model_output(dataset, site=site, site_table=self.site_table)
        if self.returns_model_output:
            if self.freq is not None:
                model_output = _aggregated(model_output, self.freq)
            return _RunOutput(model_output=model_output, predictions=None)
        sipnet_parameter_fields = _run_sipnet_parameter_fields(
            getattr(sipnet_result, "parameters", None),
            site_observation_vector.sipnet_parameter_names_read,
            model_output,
        )
        predicted = site_observation_vector.predict(model_output, sipnet_parameter_fields=sipnet_parameter_fields)
        predictions = np.asarray(site_observation_vector.flat(predicted), dtype=np.float64)
        return _RunOutput(model_output=None, predictions=predictions)


# ── private helpers ───────────────────────────────────────────────────────────

#: The temporary name of the run index's rows while predictions are unstacked.
_ROW = "row"


def _crossed_dims(external_inputs: xr.Dataset | None, batch_dim: str) -> tuple[str, ...]:
    if external_inputs is None:
        return ()
    dims: dict[str, None] = {}
    for variable in external_inputs.data_vars.values():
        dims.update(dict.fromkeys(str(d) for d in variable.dims if d not in (SITE, batch_dim)))
    return tuple(dims)


def _output_variable_names(
    output_variable_names: Sequence[str] | None, observation_vector: ObservationVector | None
) -> tuple[str, ...]:
    """The registry names each run returns: as given, else the operators'."""
    if output_variable_names is None:
        return tuple(observation_vector.output_variable_names)  # type: ignore[union-attr]
    names = tuple(resolve_output_variable_names(output_variable_names))
    if observation_vector is not None:
        check_output_variable_names_cover_the_operators(names, observation_vector)
    return names


def _run_sipnet_parameter_fields(
    sipnet_parameters: SIPNETParameters, names: Sequence[str], model_output: xr.Dataset
) -> xr.Dataset | None:
    """One run's SIPNET parameter fields: *names* from its own parameters, 0-d,
    labeled with the run's scalar ``site`` and ``lon``/``lat``; ``None`` when no
    parameter is read."""
    if not names:
        return None
    check_run_result_carries_its_parameters(sipnet_parameters)
    location = {name: model_output[name] for name in (SITE, "lon", "lat") if name in model_output.coords}
    return xr.Dataset({name: sipnet_parameters.dataarray(name) for name in names}, coords=location)


def _aggregated(model_output: xr.Dataset, freq: str) -> xr.Dataset:
    """Every variable of one run's output aggregated to *freq* by its kind's
    default, through :func:`~sipnet_calibration.observation.aggregate_time`,
    with pySIPNET's ``resampling_frequency`` and ``timestep_length_source``."""
    aggregated = [aggregate_time(model_output[name], freq).to_dataset() for name in model_output.data_vars]
    # Every variable of one run shares one time axis, so a disagreement in the
    # interval coordinates is a defect to raise, never one to settle.
    merged = xr.merge(aggregated, join="exact", compat="identical", combine_attrs="drop_conflicts")
    merged.attrs = {
        **model_output.attrs,
        "timestep_length_source": STEP_LENGTH_RESAMPLED,
        "resampling_frequency": freq,
    }
    return merged


def _site_segments(
    observation_vector: ObservationVector | None,
) -> tuple[dict[int, ObservationVector], dict[int, np.ndarray]]:
    """Per observed site: the vector restricted to it, and its segment in Flat."""
    slices: dict[int, ObservationVector] = {}
    positions: dict[int, np.ndarray] = {}
    for site in () if observation_vector is None else observation_vector.sites:
        slices[site] = observation_vector.select(sites=[site])
        positions[site] = observation_vector.positions(site=site)
        check_site_slice_is_the_site_segment(slices[site], observation_vector.index[positions[site]], site)
    return slices, positions


def _sort_records(
    ensemble_result: Any, run_index: pd.Index, sites: Sequence[int]
) -> tuple[dict[tuple[int, int], _RunOutput], np.ndarray, pd.DataFrame, list[tuple[dict, BaseException]]]:
    """Split PyEns's records into outputs keyed ``(row, site)``, successes
    ``(R, S)``, model failures and machinery failures.

    Each record is placed by its coordinate on the run index's levels and
    ``site``, whatever order PyEns ran it in.
    """
    level_names = list(run_index.names)
    row_of = {key if isinstance(key, tuple) else (key,): i for i, key in enumerate(run_index)}
    column = {site: j for j, site in enumerate(sites)}
    run_outputs: dict[tuple[int, int], _RunOutput] = {}
    rows: list[dict[str, Any]] = []
    machinery_failures: list[tuple[dict, BaseException]] = []
    for record in ensemble_result:
        labels = tuple(int(record.coordinate[name]) for name in level_names)
        site = int(record.coordinate[SITE])
        if not record.failed:
            run_outputs[(row_of[labels], site)] = record.output
        elif _is_model_failure(record.output):
            rows.append(
                {
                    **dict(zip(level_names, labels)),
                    SITE: site,
                    "error": _error_name(record.output),
                    "message": str(record.output)[:500],
                }
            )
        else:
            machinery_failures.append((record.coordinate, record.output))
    succeeded = np.zeros((len(run_index), len(sites)), dtype=bool)
    for row, site in run_outputs:
        succeeded[row, column[site]] = True
    failures = pd.DataFrame(rows, columns=[*level_names, SITE, "error", "message"])
    return run_outputs, succeeded, failures, machinery_failures


def _rows_in_domain(outside: pd.DataFrame, run_index: pd.Index) -> np.ndarray:
    """``(R,)``: whether every SIPNET parameter of each row is in its domain.

    A value on fewer dims than the run index (a crossed input's, say) puts
    every row it reaches out of the domain.
    """
    in_domain = np.ones(len(run_index), dtype=bool)
    if outside.empty:
        return in_domain
    names = list(run_index.names)
    keys = run_index.to_frame(index=False)
    for _, row in outside.iterrows():
        match = np.ones(len(run_index), dtype=bool)
        for name in names:
            if name in row and not pd.isna(row[name]):
                match &= keys[name].to_numpy() == int(row[name])
        in_domain &= ~match
    return in_domain


def _run_succeeded(
    succeeded: np.ndarray, run_index: pd.Index, sites: Sequence[int], site_locations: Mapping[str, xr.DataArray]
) -> xr.DataArray:
    """*succeeded*, ``(R, S)``, as a field on ``(batch_dim, *crossed dims, site)``."""
    names = list(run_index.names)
    shape = [len(level) for level in getattr(run_index, "levels", [run_index])]
    coords = {
        name: batch_coordinate(name, np.asarray(level))
        for name, level in zip(names, getattr(run_index, "levels", [run_index.values]))
    }
    return xr.DataArray(
        succeeded.reshape(*shape, len(sites)),
        dims=(*names, SITE),
        coords={**coords, SITE: np.asarray(sites, dtype=SITE_DTYPE), **site_locations},
        name="run_succeeded",
        attrs={"long_name": "Whether the run succeeded"},
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
    run_outputs: Mapping[tuple[int, int], _RunOutput],
    run_index: pd.Index,
    sites: Sequence[int],
    *,
    site_table: pd.DataFrame,
    site_locations: Mapping[str, xr.DataArray],
) -> xr.Dataset:
    """The runs' output on the full ``(batch_dim, *crossed dims, site, time)``
    grid, ``NaN`` where a run failed."""
    names = list(run_index.names)
    keys = [key if isinstance(key, tuple) else (key,) for key in run_index]
    by_key = {(*keys[row], site): output.model_output for (row, site), output in run_outputs.items()}
    stacked = stack_model_outputs(by_key, key_dims=(*names, SITE), site_table=site_table)
    levels = getattr(run_index, "levels", [run_index.values])
    full = stacked.reindex(
        {
            **{name: np.asarray(level, dtype=BATCH_LABEL_DTYPE) for name, level in zip(names, levels)},
            SITE: np.asarray(sites, dtype=stacked[SITE].dtype),
        }
    )
    # A site at which every run failed is absent from the stack, so the
    # reindex leaves its lon/lat NaN; they are the site table's whatever the
    # runs did.
    return full.assign_coords(site_locations)


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
    out_of_domain: Any,
) -> None:
    """Every check on :class:`ForwardModel`'s arguments that needs nothing computed."""
    sites = parameter_vector.sites
    check_model_is_a_sipnet_model(model)
    check_backend_is_a_pyens_backend(backend)
    check_out_of_domain_is_known(out_of_domain)
    check_output_variables_are_named(output_variable_names, observation_vector)
    check_freq_is_for_the_prior_predictive(freq, observation_vector)
    if freq is not None:
        check_frequency_is_an_offset_alias(freq)
    check_climate_covers_the_sites(climate, sites)
    check_climate_is_file_backed(climate, sites, backend)
    if observation_vector is not None:
        check_observation_sites_are_run(observation_vector, sites)


def check_out_of_domain_is_known(out_of_domain: Any) -> None:
    """``out_of_domain`` is ``"raise"`` or ``"fail_row"``."""
    if out_of_domain not in ("raise", "fail_row"):
        raise ValueError(f"out_of_domain is 'raise' or 'fail_row', got {out_of_domain!r}.")


def check_sipnet_parameters_are_in_the_domain(outside: pd.DataFrame) -> None:
    """Every SIPNET parameter lies in pySIPNET's domain, since the prior and
    the map otherwise put mass where SIPNET is undefined."""
    if outside.empty:
        return
    first = outside.iloc[0].dropna().to_dict()
    raise SIPNETParametersOutOfDomainError(
        f"{len(outside)} SIPNET parameter value(s) lie outside pySIPNET's domains, the first "
        f"at {first}; nothing ran. Give the parameters supports or priors the map sends into "
        "the domains, or pass out_of_domain='fail_row' to mark such rows invalid, which "
        "truncates the prior."
    )


def check_no_crossed_dims(crossed_dims: Sequence[str]) -> None:
    """``forward(theta)`` returns one row per row of theta, which crossed
    dims multiply."""
    if crossed_dims:
        raise ValueError(
            f"the external inputs cross theta's rows with {list(crossed_dims)}, so an evaluation "
            "has more rows than theta; call evaluate(theta) and reduce its predictions over "
            "them, which is a modeling choice."
        )


def check_evaluation_has_predictions(evaluation: ForwardEvaluation) -> None:
    if evaluation.predictions is None:
        raise ValueError("the evaluation has no predictions: its model was built without an observation vector.")


def check_base_parameters_set_what_the_operators_read(
    model: SIPNETModel, read: Sequence[str], written: Sequence[str]
) -> None:
    """The base parameter set holds a value for every SIPNET parameter the
    operators read and the map does not write."""
    for name in read:
        if name in written:
            continue
        try:
            model.base_params.dataarray(name)
        except ValueError as error:
            raise ValueError(
                f"the observation operators read {name!r}, which the SIPNET parameter map does "
                f"not write and the model's base parameter set leaves unset ({error}); every "
                "run would fail after SIPNET ran. Set it in the base parameter set, or have the "
                "map fix or write it."
            ) from None


def check_run_result_carries_its_parameters(sipnet_parameters: Any) -> None:
    """A run's result carries the parameters it ran with, as ``.parameters``."""
    if not isinstance(sipnet_parameters, SIPNETParameters):
        raise TypeError(
            "the model's result carries no SIPNETParameters as .parameters, got "
            f"{type(sipnet_parameters).__name__}; the operators read SIPNET parameters "
            "from the run's own result, so use a SIPNETModel whose runner returns a "
            "SIPNETResult."
        )


def check_theta_has_a_row(theta: np.ndarray) -> None:
    """``theta`` holds at least one row, since an evaluation of none runs nothing."""
    if theta.shape[0] == 0:
        raise ValueError(
            f"theta must be at least one row of {theta.shape[1]} entries, got shape "
            f"{theta.shape}; pass (D,) or (J, D) with J >= 1."
        )


def check_theta_is_finite(theta: np.ndarray) -> None:
    """Every entry of ``theta`` is finite, since SIPNET cannot run at a NaN."""
    rows = np.flatnonzero(~np.isfinite(theta).all(axis=1)).tolist()
    if rows:
        raise ValueError(
            f"theta holds a non-finite value in row(s) {truncated(rows)}; every entry must "
            "be a finite number."
        )


def check_model_is_a_sipnet_model(model: Any) -> None:
    if not isinstance(model, SIPNETModel):
        raise TypeError(f"model must be a pysipnet SIPNETModel, got {type(model).__name__}.")


def check_backend_is_a_pyens_backend(backend: Any) -> None:
    if not isinstance(backend, Backend):
        raise TypeError(f"backend must be a pyens Backend, got {type(backend).__name__}.")


def check_output_variables_are_named(
    output_variable_names: Sequence[str] | None, observation_vector: ObservationVector | None
) -> None:
    if output_variable_names is None and observation_vector is None:
        raise ValueError(
            "name the output variables each run returns (output_variable_names=), "
            "or give an observation_vector whose operators say what they read."
        )


def check_freq_is_for_the_prior_predictive(freq: Any, observation_vector: ObservationVector | None) -> None:
    if freq is not None and observation_vector is not None:
        raise ValueError(
            "freq= aggregates model output for the prior predictive; with an observation "
            "vector the operators decide their own alignment, so freq would be ignored. "
            "Drop one of the two."
        )


def check_climate_covers_the_sites(climate: Mapping[int, Any], sites: Sequence[int]) -> None:
    missing = [s for s in sites if s not in climate]
    if missing:
        raise ValueError(
            f"climate has no drivers for site(s) {truncated(missing)} of the parameter vector; "
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
            f"the drivers of site(s) {truncated(in_memory)} are held in memory, and under "
            f"{type(backend).__name__} every run would carry a copy of them. Write them "
            "with ClimateDrivers.to_file and open them with ClimateDrivers.from_path."
        )


def check_observation_sites_are_run(observation_vector: ObservationVector, sites: Sequence[int]) -> None:
    extra = sorted(set(observation_vector.sites) - set(sites))
    if extra:
        raise ValueError(
            f"the observation vector observes site(s) {truncated(extra)} that the parameter vector "
            "does not run; restrict the observation vector to the parameter vector's sites "
            "first (observation_vector.restrict_to_sites(parameter_vector.sites))."
        )


def check_output_variable_names_cover_the_operators(
    output_variable_names: Sequence[str], observation_vector: ObservationVector
) -> None:
    missing = [n for n in observation_vector.output_variable_names if n not in output_variable_names]
    if missing:
        raise ValueError(
            f"the observation vector's operators read {missing}, which "
            "output_variable_names does not include; add them, or omit "
            "output_variable_names to take the operators' own."
        )


def check_output_variables_can_be_returned(
    output_variable_names: Sequence[str], model: SIPNETModel, freq: str | None
) -> None:
    """Each variable is written by this model and, with *freq*, has a default method."""
    flags = getattr(model.runner, "flags", None)
    for name in output_variable_names:
        variable = resolve_output_variable(name)
        flag = variable.requires_flag
        if flags is not None and flag and not getattr(flags, flag, True):
            raise ValueError(
                f"{name!r} is written as constant zero unless the model flag {flag!r} is on, "
                "and this model's runner has it off; turn the flag on in the runner's "
                "ModelFlags, or drop the variable from what is read."
            )
        if freq is not None and variable.kind not in DEFAULT_METHOD_FOR_KIND:
            raise ValueError(
                f"{name!r} is of kind {getattr(variable.kind, 'value', variable.kind)!r}, which "
                "no resampling method leaves unchanged, so freq= has no rule to aggregate it "
                "by. Leave it out of output_variable_names, or drop freq= and aggregate it "
                "yourself."
            )


def check_site_slice_is_the_site_segment(
    site_slice: ObservationVector, site_segment: pd.MultiIndex, site: int
) -> None:
    if not site_slice.index.equals(site_segment):
        raise ValueError(
            f"the observation vector's selection to site {site} is not that site's segment of "
            "the whole vector, so its predictions cannot be placed by position. The vector "
            "must be site-major; this is a defect in ObservationVector, not in the inputs."
        )


def check_output_is_finite(dataset: xr.Dataset, site: int) -> None:
    """Raise :class:`ModelOutputNotFiniteError` if a read variable holds a non-finite value."""
    for name, variable in dataset.data_vars.items():
        values = np.asarray(variable.values, dtype=np.float64)
        if not np.isfinite(values).all():
            where = int(np.flatnonzero(~np.isfinite(values.ravel()))[0])
            raise ModelOutputNotFiniteError(
                f"the run at site {site} wrote a non-finite {name!r} (first at flat position "
                f"{where} of {values.size}); SIPNET exits 0 on a blow-up, so this counts as a "
                "failure at the run's parameters."
            )


def check_no_run_failed_in_the_machinery(
    machinery_failures: Sequence[tuple[dict, BaseException]], evaluation: ForwardEvaluation
) -> None:
    if not machinery_failures:
        return
    coordinate, error = machinery_failures[0]
    error_type_names = sorted({getattr(e, "type_name", type(e).__name__) for _, e in machinery_failures})
    raise _with_evaluation(
        RuntimeError(
            f"{len(machinery_failures)} run(s) failed in the machinery rather than at their "
            f"parameters ({', '.join(error_type_names)}); the first, at {coordinate}, says: "
            f"{error}. A {TaskFailedError.__name__}, a missing binary or an import error on a "
            "worker says nothing about the parameters, so it is raised rather than turned into "
            "a NaN row; the runs that were collected are on this error's `evaluation` attribute."
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
