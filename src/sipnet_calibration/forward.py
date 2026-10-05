"""The forward model: values to predictions, over a site set, with SIPNET
run once per run of the ensemble.

Where this sits
---------------
::

    theta (J, D)                                                       ForwardModel only
      --ParameterVector.to_natural, DerivedParameters.values-->  values by parameter
      --values_to_dataset, merged with external inputs-->  labeled natural values
    labeled values, from a posterior (SIPNETSimulator) or any caller (SIPNETRuns)
      --SIPNETParameterMap.sipnet_parameter_fields, at the SiteDims' sites-->  SIPNET parameter fields
      --PyEns, one SIPNETModel run per (sample, crossed labels, site)-->  model output
      --ObservationVector.predict and .flat on the worker-->  one site's segment of Flat
      --placed at its run and at positions(site=) by the calling process-->  (R, N)

Three objects compose pieces that exist elsewhere: the site dims, the
SIPNET parameter map, pySIPNET's ``SIPNETModel``, PyEns's
``PartialSpec``/``EnsembleRunner``/``Backend`` and its xarray bridge, the
observation vectors, and the stacking of runs
(:func:`sipnet_calibration.fields.stack_model_outputs`); ``ForwardModel``
adds the parameter layer's vector and derived parameters.

- :class:`SIPNETRuns` runs SIPNET once per sample and site for a batch of
  labeled values and returns each requested reduction: predictions per
  observation vector and model output, from one pass.
- :class:`SIPNETSimulator` is the forward map as the probability layer's
  :class:`~sipnet_calibration.probability.Simulator`: it never sees theta, a
  posterior hands it the values it reads.
- :class:`ForwardModel` is today's callable ``(J, D) -> (J, N)``, which
  EnsKit's ``eki.run`` takes as ``forward``: theta through the parameter layer,
  then :class:`SIPNETRuns`. It stays until the parameter layer is removed.

What it reads
-------------
A :class:`pysipnet.model.SIPNETModel`; a
:class:`~sipnet_calibration.parameters.ParameterVector`, optionally its
:class:`~sipnet_calibration.parameters.DerivedParameters`, a
:class:`~sipnet_calibration.site_dims.SiteDims` (the sites run) and a
:class:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap`; optional
:data:`~sipnet_calibration.sipnet_parameter_map.ExternalInputs`; one
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
S`` of them, with ``M_k`` the size of crossed dim ``k``, which is
:attr:`ForwardModel.runs_per_sample` ``* J``.

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
    bool ``(R,)`` ``jax.Array``: every run of the row succeeded, its values
    are in their domains, and its predictions are finite.
``out_of_domain_fraction``
    The fraction of rows with a value outside its domain; ``0.0`` unless
    ``out_of_domain="fail_row"``.

A run **fails at its parameters** when pySIPNET refuses them
(``pydantic.ValidationError``), SIPNET exits non-zero or writes nothing
(``SIPNETRunError``), the run times out (``subprocess.TimeoutExpired``), or
its output holds a non-finite value in a variable that was read
(:class:`ModelOutputNotFiniteError`): its row is ``NaN``, which EKI repairs
and a sampler rejects. Anything else a worker returns is the **machinery**
failing, and is raised after the batch is collected as a ``RuntimeError``
whose ``evaluation`` attribute holds what was collected. On the
prior-predictive path a batch in which every run failed at its parameters is
raised the same way.

**Values outside their domains**, a rule input outside its requirement's or
a SIPNET parameter outside pySIPNET's
(:meth:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap.out_of_domain`),
mean the prior and the map put mass where a rule or SIPNET is undefined. By
default the model refuses them before anything runs
(:class:`~sipnet_calibration.sipnet_parameter_map.SIPNETParametersOutOfDomainError`),
and checks the map at the corners of theta when it is built;
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
:class:`SIPNETRuns`, :class:`SIPNETRunsEvaluation`
    ``evaluate(values, *, observation_vectors=, output_variable_names=,
    freq=, external_inputs=)``, ``select(sites=)``; the record, which is
    :class:`ForwardEvaluation`'s without ``theta``, with predictions per
    observation vector.
:class:`SIPNETSimulator`
    The forward map as a simulator: one prediction per observation source,
    each valid at a sample whose runs at the source's sites succeeded; its
    ``at`` restricts it to fewer sources and sites, and ``check_given``
    checks the map against the model's declarations and, under
    ``out_of_domain="raise"``, at the corners of the target.
:data:`MODEL_FAILURES`, :class:`ModelOutputNotFiniteError`
    The exceptions that mean a run failed at its parameters.

Notes
-----
**The operators run on the worker**, because reading a run's output back in
the calling process costs more than the run. Each run receives only its
site's slice of the observation vector and returns that slice's Flat, which
the calling process writes at ``positions(site=)``: right because the
observation vector is site-major, which ``__init__`` checks per site.

**The ``PartialSpec`` is built once per plan**: the climate, the site ids
and the sites' observation slices along one site axis. ``ForwardModel`` and
``SIPNETSimulator`` build theirs once; ``SIPNETRuns.evaluate`` builds one
per call. On every call PyEns zips the SIPNET parameter fields with that
axis and crosses them along their other dims.

**Validity.** ``ForwardModel`` marks a whole row invalid when any of its
runs failed, since EKI updates per row. ``SIPNETSimulator`` marks each
source's prediction invalid only where a run at one of the source's sites
failed, so a posterior that does not read a source is not truncated by its
sites. One pass for several reductions reads the union of their output
variables, each checked finite, so a non-finite value in any of them fails
the run for all.

**The SIPNET parameters an operator reads** are the run's own
``SIPNETResult.parameters``, whichever set them.

Usage
-----
::

    forward = ForwardModel(model, prior.parameter_vector, sipnet_map, site_dims=site_dims,
                           derived_parameters=prior.derived_parameters, climate=climate,
                           backend=LocalBackend(8), observation_vector=observation_vector)
    result = enskit.algorithms.eki.run(state, forward, observation_vector.y, noise_cov,
                                       update_rule=..., schedule=...)

    predictive = ForwardModel(model, vector, sipnet_map, site_dims=site_dims, climate=climate,
                              backend=LocalBackend(8), external_inputs=initial_states,
                              output_variable_names=("nee",), freq="1D")
    evaluation = predictive.evaluate(prior.sample(key, 100))
    evaluation.model_output      # (sample, initial_condition_member, site, time)

    runs = SIPNETRuns(model, sipnet_parameter_map=sipnet_map, site_dims=site_dims, climate=climate,
                      backend=LocalBackend(8))
    model_spec = joint(*noise_factors, SIPNETSimulator(runs, observation_vector=observation_vector),
                       *prior_factors)
    posterior = condition_on(model_spec.bind(coords={**site_dims.coords, **observation_vector.coords}),
                             observation_vector.observed_values_by_component())
    evaluation = posterior.evaluate(theta)                  # one batch of runs
    daily = runs.evaluate(posterior.simulator_inputs(theta, "sipnet"),
                          output_variable_names=("nee",), freq="1D").model_output
"""

from __future__ import annotations

import math
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

from sipnet_calibration.conventions import (
    BATCH_LABEL_DTYPE,
    RESERVED_NAMES,
    SAMPLE,
    SITE,
    SITE_DTYPE,
    ReadOnlyCopies,
)
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
from sipnet_calibration.observation.model import prediction_components
from sipnet_calibration.observation.time_alignment import (
    check_frequency_is_an_offset_alias,
)
from sipnet_calibration.parameters import (
    DerivedParameters,
    ParameterVector,
    check_parameter_vectors_share_a_layout,
)
from sipnet_calibration.sipnet_parameter_map import (
    ExternalInputs,
    SIPNETParameterMap,
    SIPNETParametersOutOfDomainError,
    check_sipnet_parameter_map_fits,
    validate_external_inputs,
)
from sipnet_calibration.probability import ArraySpec, LabeledValues, Simulator, SimulatorOutput
from sipnet_calibration.probability._probes import CORNERS, corner_points  # ForwardModel's, until R1
from sipnet_calibration.site_dims import SiteDims
from sipnet_calibration.sites import site_locations, site_lookup
from sipnet_calibration.validation import as_batched_flat, as_sequence, is_one_vector, truncated

__all__ = [
    "DOMAIN_CHECK_CORNERS",
    "MODEL_FAILURES",
    "ForwardEvaluation",
    "ForwardModel",
    "ModelOutputNotFiniteError",
    "SIPNETRuns",
    "SIPNETRunsEvaluation",
    "SIPNETSimulator",
    "check_map_is_in_domain_at_the_corners",
    "check_sipnet_parameter_map_is_in_domain_at_the_corners",
]


class ForwardModel:
    """The forward map :math:`G`: unconstrained parameters to predictions.

    .. math::

        G(\\theta) = \\big(H_s(\\mathrm{SIPNET}(\\psi_s))\\big)_s, \\qquad
        \\psi_s = M\\big((x, f(x))^{(s)}, u_s, c_s\\big), \\qquad x = T(\\theta),

    with :math:`T` the vector's transform, :math:`f` the derived
    parameters, :math:`(\\cdot)^{(s)}` a value read at site :math:`s`,
    :math:`M` the SIPNET parameter map, :math:`u_s` a run's external inputs,
    :math:`c_s` its constants and fixed SIPNET parameters, and :math:`H_s`
    the observation operators at the run's site.

    Parameters
    ----------
    model:
        The :class:`pysipnet.model.SIPNETModel` every run goes through.
    parameter_vector:
        The vector theta is of. Pass ``prior.parameter_vector``, so the
        prior and the forward model share one.
    sipnet_parameter_map:
        How the values, and the external inputs, reach SIPNET. It is checked
        to fit them, and, with ``out_of_domain="raise"``, at the corners of
        theta (:func:`check_sipnet_parameter_map_is_in_domain_at_the_corners`).
    site_dims:
        The sites run, located by their site table, and the dims the vector
        and derived parameters are on: each of their dims is a dim of
        ``site_dims.coords``, with every label some site carries.
    derived_parameters:
        The derived parameters the map reads, over this vector: pass
        ``prior.derived_parameters``.
    climate:
        ``{site id: ClimateDrivers}``, a superset of the sites run.
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
        sites run.
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
        anything runs: a dim the site dims lack; a parameter, derived
        parameter, element axis or external input named with a reserved
        name or the batch dim's; derived parameters over another vector;
        the map not fitting; values outside their domains at the corners;
        the message names the rule.
    KeyError
        If an output variable is unknown, the map reads a value nothing
        holds, or the vector or a derived parameter lacks a label some site
        carries.
    """

    def __init__(
        self,
        model: SIPNETModel,
        parameter_vector: ParameterVector,
        sipnet_parameter_map: SIPNETParameterMap,
        *,
        site_dims: SiteDims,
        derived_parameters: DerivedParameters | None = None,
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
        check_forward_model_composes(parameter_vector, derived_parameters, site_dims, external_inputs, batch_dim)
        check_forward_model_arguments(
            model,
            site_dims.sites,
            climate=climate,
            backend=backend,
            observation_vector=observation_vector,
            output_variable_names=output_variable_names,
            freq=freq,
            out_of_domain=out_of_domain,
        )
        if external_inputs is not None:
            validate_external_inputs(external_inputs, batch_dim=batch_dim)
        check_sipnet_parameter_map_fits(
            sipnet_parameter_map, _descriptions(parameter_vector, derived_parameters), external_inputs, site_dims
        )
        self._model = model
        self._parameter_vector = parameter_vector
        self._derived_parameters = derived_parameters
        self._site_dims = site_dims
        self._sipnet_parameter_map = sipnet_parameter_map
        self._external_inputs = external_inputs
        self._out_of_domain = out_of_domain
        self._sites: tuple[int, ...] = site_dims.sites
        self._backend = backend
        self._observation_vector = observation_vector
        self._freq = freq
        self._batch_dim = batch_dim
        self._crossed_dims = _crossed_dims(sipnet_parameter_map, external_inputs, batch_dim)
        if out_of_domain == "raise":
            check_sipnet_parameter_map_is_in_domain_at_the_corners(self)
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
        if observation_vector is not None:
            check_base_parameters_set_what_the_operators_read(
                model,
                observation_vector.sipnet_parameter_names_read,
                sipnet_parameter_map.sipnet_parameter_names_written,
            )
        self._runs = SIPNETRuns(
            model, sipnet_parameter_map=sipnet_parameter_map, site_dims=site_dims, climate=climate,
            backend=backend, out_of_domain=out_of_domain,
        )
        self._plan = self._runs._plan(
            observation_vectors=() if observation_vector is None else (observation_vector,),
            read_variable_names=self._output_variable_names,
            model_output_variable_names=self._output_variable_names if observation_vector is None else (),
            freq=freq,
        )

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def model(self) -> SIPNETModel:
        return self._model

    @property
    def parameter_vector(self) -> ParameterVector:
        return self._parameter_vector

    @property
    def derived_parameters(self) -> DerivedParameters | None:
        return self._derived_parameters

    @property
    def site_dims(self) -> SiteDims:
        return self._site_dims

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
        """The sites run: the site dims', ascending."""
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
        """``D``, ``parameter_vector.unconstrained.size``."""
        return self.parameter_vector.unconstrained.size

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
            With ``out_of_domain="raise"``, if a value lies outside its
            domain; nothing runs.
        RuntimeError
            If a run failed in the machinery rather than at its parameters,
            or, on the prior-predictive path, every run failed at its
            parameters; the error's ``evaluation`` attribute holds what was
            collected.
        """
        theta = np.asarray(as_batched_flat(theta, self.input_dimension, message_name="theta"))
        check_theta_has_a_row(theta)
        check_theta_is_finite(theta)
        values = self._labeled_values(theta, self._external_inputs)
        try:
            evaluation = self._runs._evaluate(
                self._plan, values, n_samples=len(theta), crossed_dims=self._crossed_dims, batch_dim=self._batch_dim
            )
        except RuntimeError as error:
            collected = getattr(error, "evaluation", None)
            if isinstance(collected, SIPNETRunsEvaluation):
                error.evaluation = self._forward_evaluation(theta, collected, collected=True)  # type: ignore[attr-defined]
            raise
        return self._forward_evaluation(theta, evaluation)

    def __call__(self, theta: Any) -> jax.Array:
        """``evaluate(theta).predictions``, EKI's ``(J, D) -> (J, N)``: ``(N,)``
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
        check_external_inputs_cross_no_dims(self._crossed_dims)
        predictions = self.evaluate(theta).predictions
        return predictions[0] if is_one_vector(theta) else predictions

    # ── supporting methods ────────────────────────────────────────────────────

    def _labeled_values(self, theta: np.ndarray, external_inputs: xr.Dataset | None) -> xr.Dataset:
        """The labeled natural values at *theta*, ``(J, D)``, rows on
        ``batch_dim``: the parameters', the derived parameters' and the
        external inputs', merged."""
        vector = self._parameter_vector
        values_by_parameter = vector.flat_to_values(vector.to_natural(theta))
        batch_dims = (self._batch_dim,)
        pieces = [vector.values_to_dataset(values_by_parameter, batch_dims=batch_dims)]
        if self._derived_parameters is not None:
            derived = self._derived_parameters.values(values_by_parameter)
            pieces.append(self._derived_parameters.values_to_dataset(derived, batch_dims=batch_dims))
        if external_inputs is not None:
            check_external_inputs_are_for_theta(external_inputs, len(theta), self._batch_dim)
            for name, variable in external_inputs.data_vars.items():
                if SITE in variable.dims:
                    check_external_input_covers_the_sites(str(name), variable, self._sites)
            pieces.append(external_inputs)
        # Each piece at the sites run alone, so one index per dim holds in the merge.
        at_sites = [piece.sel({SITE: list(self._sites)}) if SITE in piece.dims else piece for piece in pieces]
        return xr.merge(at_sites, join="exact", combine_attrs="drop_conflicts")

    def _observation_vector_for(self, what: str) -> ObservationVector:
        """The observation vector, or a ``ValueError`` saying *what* needs one."""
        if self.observation_vector is None:
            raise ValueError(
                f"{what} needs an observation vector; this ForwardModel was built without one, "
                "for model output only. Use evaluate(theta).model_output."
            )
        return self.observation_vector

    def _forward_evaluation(
        self, theta: np.ndarray, evaluation: SIPNETRunsEvaluation, *, collected: bool = False
    ) -> ForwardEvaluation:
        """The runs' evaluation as this model's: with *theta*, its one
        observation vector's predictions ``NaN`` in every row with a failed
        run, and ``valid`` false there; *collected* is what a failed batch
        collected, with neither predictions nor model output."""
        base = ForwardEvaluation(
            theta=jnp.asarray(theta),
            sipnet_parameter_fields=evaluation.sipnet_parameter_fields,
            run_index=evaluation.run_index,
            model_output=None,
            predictions=None,
            run_succeeded=evaluation.run_succeeded,
            failures=evaluation.failures,
            valid=jnp.zeros(len(evaluation.run_index), dtype=bool),
            out_of_domain_fraction=evaluation.out_of_domain_fraction,
            observation_vector=self.observation_vector,
        )
        if collected:
            return base
        row_succeeded = np.asarray(evaluation.valid)
        if self.observation_vector is None:
            return replace(base, model_output=evaluation.model_output, valid=jnp.asarray(row_succeeded))
        predictions = np.array(evaluation.predictions[0])
        # A row with any failed run is invalid as a whole: EKI updates per
        # row, so a row that is partly a prediction cannot be used.
        predictions[~row_succeeded] = np.nan
        valid = row_succeeded & np.isfinite(predictions).all(axis=1)
        return replace(base, predictions=jnp.asarray(predictions), valid=jnp.asarray(valid))


class SIPNETRuns:
    """SIPNET run once per sample and site for a batch of labeled values: the
    SIPNET parameter map applied at the sites, the runs executed through
    PyEns, each run reduced on the worker.

    Parameters
    ----------
    sipnet_model:
        Positional-only. The :class:`pysipnet.model.SIPNETModel` every run
        goes through.
    sipnet_parameter_map:
        Keyword-only. How the values reach SIPNET.
    site_dims:
        Keyword-only. The sites run, located by their site table.
    climate:
        Keyword-only. ``{site id: ClimateDrivers}``, a superset of the sites
        run; file-backed under any backend but ``SequentialBackend``.
    backend:
        Keyword-only. The PyEns backend the runs execute on.
    out_of_domain:
        Keyword-only. ``"raise"`` (the default) refuses values outside their
        domains before anything runs; ``"fail_row"`` runs them and marks
        their rows out of the domain.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If the climate lacks a site, is held in memory under a process
        backend, or *out_of_domain* is unknown.
    """

    def __init__(
        self,
        sipnet_model: SIPNETModel,
        /,
        *,
        sipnet_parameter_map: SIPNETParameterMap,
        site_dims: SiteDims,
        climate: Mapping[int, ClimateDrivers],
        backend: Backend,
        out_of_domain: Literal["raise", "fail_row"] = "raise",
    ) -> None:
        check_sipnet_runs_arguments(sipnet_model, sipnet_parameter_map, site_dims, climate, backend, out_of_domain)
        self._model = sipnet_model
        self._sipnet_parameter_map = sipnet_parameter_map
        self._site_dims = site_dims
        self._sites: tuple[int, ...] = site_dims.sites
        self._climate = frozendict({site: climate[site] for site in self._sites})
        self._backend = backend
        self._out_of_domain = out_of_domain
        self._site_locations = site_locations(self._sites, site_dims.site_table)
        self._site_table = site_lookup(site_dims.site_table)
        self._site_axis = Axis(SITE, labels=list(self._sites))

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def sipnet_model(self) -> SIPNETModel:
        return self._model

    @property
    def sipnet_parameter_map(self) -> SIPNETParameterMap:
        return self._sipnet_parameter_map

    @property
    def site_dims(self) -> SiteDims:
        return self._site_dims

    @property
    def sites(self) -> tuple[int, ...]:
        """The sites run: the site dims', ascending."""
        return self._sites

    @property
    def climate(self) -> Mapping[int, ClimateDrivers]:
        """``{site id: ClimateDrivers}`` for the sites run, read-only."""
        return self._climate

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def out_of_domain(self) -> str:
        return self._out_of_domain

    def __repr__(self) -> str:
        return f"SIPNETRuns(sites={len(self._sites)}, out_of_domain={self._out_of_domain!r})"

    # ── selection ─────────────────────────────────────────────────────────────

    def select(self, *, sites: Sequence[int]) -> SIPNETRuns:
        """The runs at *sites* alone, each one of :attr:`sites`.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`SiteDims.select <sipnet_calibration.site_dims.SiteDims.select>`.
        """
        return SIPNETRuns(
            self._model, sipnet_parameter_map=self._sipnet_parameter_map, site_dims=self._site_dims.select(sites),
            climate=self._climate, backend=self._backend, out_of_domain=self._out_of_domain,
        )

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        values: LabeledValues | xr.Dataset,
        *,
        observation_vectors: Sequence[ObservationVector] = (),
        output_variable_names: Sequence[str] = (),
        freq: str | None = None,
        external_inputs: ExternalInputs | None = None,
        batch_dim: str = SAMPLE,
    ) -> SIPNETRunsEvaluation:
        """Run SIPNET once per sample of *values* and site, and return each
        requested reduction: predictions per observation vector, and model
        output for *output_variable_names*, aggregated to *freq*. One pass
        serves a calibration vector, a validation vector and a daily figure.

        Parameters
        ----------
        values:
            Every value the map reads, on ``batch_dim`` labeled ``0`` to
            ``J - 1``, and on the dims of the site dims' coords and element
            axes: labeled values (a dict of DataArrays) or an ``xr.Dataset``.
            Values no rule reads are ignored.
        observation_vectors:
            Each vector whose operators reduce every run, its sites among
            those run; a sequence.
        output_variable_names:
            The pySIPNET output variables whose model output is returned.
        freq:
            With *output_variable_names*, aggregate that output on the worker
            to this pandas frequency, each variable by the method that leaves
            its kind unchanged.
        external_inputs:
            :data:`~sipnet_calibration.sipnet_parameter_map.ExternalInputs`
            the map reads, paired with the samples by dim name: on
            ``batch_dim`` they zip, on any other batch dim they cross.
        batch_dim:
            The values' batch dim, on every output.

        Returns
        -------
        SIPNETRunsEvaluation

        Raises
        ------
        TypeError
            If an argument has the wrong type.
        KeyError
            If a value or an external input lacks a site run, or an output
            variable is unknown.
        ValueError
            If nothing is asked for, *freq* is given without output
            variables, the values carry no ``batch_dim`` or label it other
            than ``0`` to ``J - 1``, or an observation vector observes a site
            not run.
        SIPNETParametersOutOfDomainError
            With ``out_of_domain="raise"``, if a value lies outside its
            domain; nothing runs.
        RuntimeError
            If a run failed in the machinery rather than at its parameters,
            or model output is asked for and every run failed at its
            parameters; the error's ``evaluation`` attribute holds what was
            collected.
        """
        vectors = tuple(as_sequence(observation_vectors, message_name="observation_vectors"))
        for vector in vectors:
            check_observation_vector_is_an_observation_vector(vector)
        names = tuple(resolve_output_variable_names(output_variable_names)) if len(output_variable_names) else ()
        check_something_is_asked_for(vectors, names)
        check_freq_is_for_model_output(freq, names)
        if freq is not None:
            check_frequency_is_an_offset_alias(freq)
        check_batch_dim_name_is_not_reserved(batch_dim, message_name="batch_dim")
        if external_inputs is not None:
            validate_external_inputs(external_inputs, batch_dim=batch_dim)
        read = tuple(dict.fromkeys([*names, *(n for v in vectors for n in v.output_variable_names)]))
        plan = self._plan(
            observation_vectors=vectors, read_variable_names=read, model_output_variable_names=names, freq=freq
        )
        merged, n_samples = self._values_at_the_sites(values, external_inputs, batch_dim)
        crossed_dims = _crossed_dims(self._sipnet_parameter_map, external_inputs, batch_dim)
        return self._evaluate(plan, merged, n_samples=n_samples, crossed_dims=crossed_dims, batch_dim=batch_dim)

    # ── supporting methods ────────────────────────────────────────────────────

    def _plan(
        self,
        *,
        observation_vectors: Sequence[ObservationVector],
        read_variable_names: Sequence[str],
        model_output_variable_names: Sequence[str],
        freq: str | None,
    ) -> _RunPlan:
        """What every run of an evaluation is sent and does: built once, and
        reused by a caller that evaluates the same reductions again."""
        written = self._sipnet_parameter_map.sipnet_parameter_names_written
        for vector in observation_vectors:
            check_observation_sites_are_run(vector, self._sites)
            check_base_parameters_set_what_the_operators_read(self._model, vector.sipnet_parameter_names_read, written)
        check_output_variables_can_be_returned(read_variable_names, self._model, None)
        check_output_variables_can_be_returned(model_output_variable_names, self._model, freq)
        segments = [_site_segments(vector) for vector in observation_vectors]
        site_observation_vectors = Grid(
            {s: tuple(slices.get(s) for slices, _ in segments) for s in self._sites}, along=self._site_axis
        )
        placeholders = {name: Grid([0.0] * len(self._sites), along=self._site_axis) for name in written}
        spec = EnsembleSpec(
            inputs={
                "climate": Grid({s: self._climate[s] for s in self._sites}, along=self._site_axis),
                "site": Grid(list(self._sites), along=self._site_axis),
                "site_observation_vectors": site_observation_vectors,
                **placeholders,
            }
        )
        run = _Run(
            model=self._model,
            read_variable_names=tuple(read_variable_names),
            model_output_variable_names=tuple(model_output_variable_names),
            freq=freq,
            site_table=self._site_table,
        )
        return _RunPlan(
            observation_vectors=tuple(observation_vectors),
            model_output_variable_names=tuple(model_output_variable_names),
            partial=spec.freeze(free=list(written)),
            run=run,
            site_positions=tuple(positions for _, positions in segments),
        )

    def _evaluate(
        self, plan: _RunPlan, values: Any, *, n_samples: int, crossed_dims: Sequence[str], batch_dim: str
    ) -> SIPNETRunsEvaluation:
        """Run *plan* at *values*, merged and at the sites, with *n_samples*
        rows on *batch_dim*, crossed with *crossed_dims*."""
        sipnet_map = self._sipnet_parameter_map
        sipnet_parameter_fields = sipnet_map.sipnet_parameter_fields(values, site_dims=self._site_dims)
        outside = sipnet_map.out_of_domain(sipnet_parameter_fields, values, site_dims=self._site_dims)
        if self._out_of_domain == "raise":
            check_sipnet_parameters_are_in_the_domain(outside)
        run_index = _run_index(n_samples, sipnet_parameter_fields, crossed_dims, batch_dim)
        grids = fields_from_dataset(sipnet_parameter_fields, axes={SITE: self._site_axis})
        ensemble_result = EnsembleRunner(plan.run, self._backend).run(plan.partial(**grids))
        run_outputs, succeeded, failures, machinery_failures = _sort_records(ensemble_result, run_index, self._sites)
        in_domain = _rows_in_domain(outside, run_index)
        collected = SIPNETRunsEvaluation(
            sipnet_parameter_fields=sipnet_parameter_fields,
            run_index=run_index,
            model_output=None,
            predictions=(),
            observation_vectors=plan.observation_vectors,
            run_succeeded=_run_succeeded(succeeded, run_index, self._sites, self._site_locations),
            failures=failures,
            in_domain=jnp.asarray(in_domain),
            valid=jnp.zeros(len(run_index), dtype=bool),
            out_of_domain_fraction=float(1.0 - in_domain.mean()),
        )
        check_no_run_failed_in_the_machinery(machinery_failures, collected)
        model_output = None
        if plan.model_output_variable_names:
            check_some_run_succeeded(collected)
            model_output = _stacked_model_output(
                run_outputs, run_index, self._sites, site_table=self._site_table, site_locations=self._site_locations,
            )
        predictions = tuple(
            jnp.asarray(_placed_predictions(run_outputs, k, vector, positions, in_domain))
            for k, (vector, positions) in enumerate(zip(plan.observation_vectors, plan.site_positions))
        )
        valid = succeeded.all(axis=1) & in_domain
        return replace(collected, model_output=model_output, predictions=predictions, valid=jnp.asarray(valid))

    def _values_at_the_sites(
        self, values: Any, external_inputs: xr.Dataset | None, batch_dim: str
    ) -> tuple[dict[str, xr.DataArray], int]:
        """The values the map reads and the external inputs, as one mapping,
        each at the sites run; and the number of samples on *batch_dim*."""
        check_values_are_labeled(values)
        read = self._sipnet_parameter_map.values_read
        merged = {str(name): array for name, array in values.items() if name in read}
        if external_inputs is not None:
            check_external_inputs_are_not_values(list(external_inputs.data_vars), merged)
            merged |= {str(name): external_inputs[name] for name in external_inputs.data_vars}
        for name, array in merged.items():
            if SITE in array.dims:
                check_value_covers_the_sites(name, array, self._sites)
        at_sites = {
            name: array.sel({SITE: list(self._sites)}) if SITE in array.dims else array for name, array in merged.items()
        }
        n_samples = _number_of_samples(at_sites, batch_dim)
        return at_sites, n_samples


class SIPNETSimulator(Simulator):
    """The forward map :math:`G` as a simulator: SIPNET runs reduced to an
    observation vector's predictions.

    It reads every value the SIPNET parameter map reads, and outputs one
    prediction per observation source (:func:`~sipnet_calibration.observation.model.prediction_components`),
    on the source's observation dim. It runs SIPNET only at the sites its
    observation vector observes.

    Parameters
    ----------
    runs:
        Positional-only. The runs, whose sites include the vector's.
    observation_vector:
        Keyword-only. The observations predicted.
    name:
        Keyword-only. Default ``"sipnet"``.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If the vector observes a site not run, or as
        :meth:`SIPNETRuns.evaluate` for its operators.
    """

    def __init__(self, runs: SIPNETRuns, /, *, observation_vector: ObservationVector, name: str = "sipnet") -> None:
        check_runs_are_sipnet_runs(runs)
        check_observation_vector_is_an_observation_vector(observation_vector)
        check_observation_sites_are_run(observation_vector, runs.sites)
        self._all_runs = runs
        self._runs = runs if runs.sites == observation_vector.sites else runs.select(sites=observation_vector.sites)
        self._observation_vector = observation_vector
        self._name = name
        self._outputs = prediction_components(observation_vector)
        self._plan = self._runs._plan(
            observation_vectors=(observation_vector,),
            read_variable_names=observation_vector.output_variable_names,
            model_output_variable_names=(),
            freq=None,
        )
        self._orders = {
            source: _block_order(observation_vector, source) for source in observation_vector.observation_source_names
        }

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def given(self) -> tuple[str, ...]:
        """Every name the SIPNET parameter map reads."""
        return tuple(self._runs.sipnet_parameter_map.values_read)

    @property
    def outputs(self) -> tuple[ArraySpec, ...]:
        """One prediction per observation source, in the vector's order."""
        return self._outputs

    @property
    def runs(self) -> SIPNETRuns:
        """The runs, at the sites the vector observes."""
        return self._runs

    @property
    def observation_vector(self) -> ObservationVector:
        return self._observation_vector

    def __repr__(self) -> str:
        return f"SIPNETSimulator({self._name!r}, sites={len(self._runs.sites)}, outputs={[o.name for o in self._outputs]})"

    # ── evaluation ────────────────────────────────────────────────────────────

    def __call__(self, given_values: LabeledValues) -> SimulatorOutput:
        """``runs.evaluate(given_values, observation_vectors=[observation_vector])``
        as a :class:`~sipnet_calibration.probability.SimulatorOutput`: one
        prediction per source, each valid at a sample whose values are in
        their domains, whose every run at the source's sites succeeded, and
        whose prediction is finite; the :class:`SIPNETRunsEvaluation` as the
        record.

        Raises
        ------
        SIPNETParametersOutOfDomainError, RuntimeError
            As :meth:`SIPNETRuns.evaluate`.
        """
        runs = self._runs
        merged, n_samples = runs._values_at_the_sites(given_values, None, SAMPLE)
        evaluation = runs._evaluate(self._plan, merged, n_samples=n_samples, crossed_dims=(), batch_dim=SAMPLE)
        vector = self._observation_vector
        predictions = np.asarray(evaluation.predictions[0])
        succeeded = evaluation.run_succeeded
        values, valid = {}, {}
        for source in vector.observation_source_names:
            name = vector.prediction_name(source)
            values[name] = predictions[:, vector.positions(observation_source_name=source)][:, self._orders[source]]
            sites = np.unique(vector[source].observation_labels.get_level_values(SITE)).tolist()
            ran = np.asarray(succeeded.sel({SITE: sites}).all(SITE).values, dtype=bool)
            valid[name] = ran & np.asarray(evaluation.in_domain) & np.isfinite(values[name]).all(axis=1)
        return SimulatorOutput(values=values, valid=valid, record=evaluation)

    def at(self, coords: Mapping[str, pd.Index], outputs: Sequence[str]) -> SIPNETSimulator:
        """Itself at its own sites and observations computing every output;
        at fewer, or fewer outputs, its observation vector restricted to the
        sources of *outputs* at the sites their labels in *coords* hold, and
        its runs to those sites.

        Raises
        ------
        ValueError
            If a kept source's labels in *coords* are not its observations at
            some sites, or a site the vector observes lacks a label in
            ``coords["site"]``.
        """
        vector = self._observation_vector
        by_output = {vector.prediction_name(s): s for s in vector.observation_source_names}
        sources = [by_output[name] for name in outputs if name in by_output]
        check_outputs_are_predictions(list(outputs), by_output)
        labels = {s: coords[vector.observation_dim_name(s)] for s in sources}
        sites = sorted({int(site) for s in sources for site in labels[s].get_level_values(SITE)})
        if sources != list(vector.observation_source_names) or sites != list(vector.sites):
            check_sites_are_observed(sites, vector)
            vector = vector.select(observation_source_names=sources, sites=sites)
        for source in sources:
            check_labels_are_the_observations(source, labels[source], vector)
        if SITE in coords:
            check_sites_have_values(vector.sites, coords[SITE])
        if vector is self._observation_vector:
            return self
        return SIPNETSimulator(self._all_runs, observation_vector=vector, name=self._name)

    def check_given(self, given_specs: Mapping[str, ArraySpec], corner_values: LabeledValues) -> None:
        """The map's fit to the given components' specs, so a rule's
        ``ValueRequirement`` is checked against what the model declares;
        under ``out_of_domain="raise"``, its domains at *corner_values*.

        Raises
        ------
        ValueError, KeyError
            As :func:`~sipnet_calibration.sipnet_parameter_map.check_sipnet_parameter_map_fits`,
            or if a value lies outside its domain at a corner.
        """
        runs = self._runs
        check_sipnet_parameter_map_fits(runs.sipnet_parameter_map, given_specs, None, runs.site_dims)
        if runs.out_of_domain != "raise":
            return
        values, _ = runs._values_at_the_sites(corner_values, None, SAMPLE)
        fields = runs.sipnet_parameter_map.sipnet_parameter_fields(values, site_dims=runs.site_dims)
        check_map_is_in_domain_at_the_corners(runs.sipnet_parameter_map.out_of_domain(fields, values, site_dims=runs.site_dims))


# ── what an evaluation returns, and the failures ──────────────────────────────


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


#: The values of each unconstrained number at which
#: :func:`check_sipnet_parameter_map_is_in_domain_at_the_corners` evaluates
#: the map: the probability layer's corner points, ``+-12`` spanning ten
#: orders of magnitude on a log scale and reaching ``1 - 6e-6`` on a logit
#: scale, while staying inside float64.
DOMAIN_CHECK_CORNERS: tuple[float, ...] = CORNERS


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


@dataclass(frozen=True, eq=False, kw_only=True)
class SIPNETRunsEvaluation:
    """What one evaluation of :class:`SIPNETRuns` produced.

    The fields are :class:`ForwardEvaluation`'s (the module docstring's Data
    model), but for these:

    ``predictions``
        One ``(R, N_k)`` float64 ``jax.Array`` per observation vector, in
        ``observation_vectors``' order, ``NaN`` in the entries of a failed
        run and in every entry of a row out of the domain; a failed run
        leaves the rest of its row.
    ``observation_vectors``
        The vectors predicted.
    ``in_domain``
        bool ``(R,)`` ``jax.Array``: every value of the row lies in its
        domain.
    ``valid``
        bool ``(R,)`` ``jax.Array``: every run of the row succeeded and it is
        in the domain. Whether its predictions are finite is a vector's own.

    There is no ``theta``. Nothing read from it changes it, as for
    :class:`ForwardEvaluation`.
    """

    sipnet_parameter_fields: SIPNETParameterFields = ReadOnlyCopies()
    run_index: pd.Index
    model_output: ModelOutput | None = ReadOnlyCopies()
    predictions: tuple[jax.Array, ...]
    observation_vectors: tuple[ObservationVector, ...]
    run_succeeded: Field = ReadOnlyCopies()
    failures: pd.DataFrame = ReadOnlyCopies()
    in_domain: jax.Array
    valid: jax.Array
    out_of_domain_fraction: float = 0.0


# ── the per-run callable ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _RunOutput:
    """What one run sends back: its model output, if asked for, and its
    site's predictions per observation vector, ``None`` where the vector
    does not observe the site."""

    model_output: xr.Dataset | None
    predictions: tuple[np.ndarray | None, ...]


@dataclass(frozen=True)
class _Run:
    """One SIPNET run, reduced on the worker; picklable, built once per plan.

    It reads *read_variable_names* from the run's output, each checked
    finite, and sends back the model output of
    *model_output_variable_names* (aggregated, with ``freq``) and its site's
    predictions per observation vector. The SIPNET parameters the operators
    read are the run's own ``SIPNETResult.parameters``.
    """

    model: SIPNETModel
    read_variable_names: tuple[str, ...]
    model_output_variable_names: tuple[str, ...]
    freq: str | None
    site_table: pd.DataFrame

    def __call__(
        self,
        *,
        climate: ClimateDrivers,
        site: int,
        site_observation_vectors: tuple[ObservationVector | None, ...] = (),
        **sipnet_overrides: Any,
    ) -> _RunOutput:
        site = int(site)
        sipnet_result = self.model(climate=climate, **sipnet_overrides)
        dataset = sipnet_result.outputs.select(list(self.read_variable_names))
        check_output_is_finite(dataset, site)
        nothing = tuple(None for _ in site_observation_vectors)
        if not self.model_output_variable_names and all(v is None for v in site_observation_vectors):
            return _RunOutput(model_output=None, predictions=nothing)
        model_output = to_model_output(dataset, site=site, site_table=self.site_table)
        returned = None
        if self.model_output_variable_names:
            returned = (
                model_output
                if self.model_output_variable_names == self.read_variable_names
                else model_output[list(self.model_output_variable_names)]
            )
            if self.freq is not None:
                returned = _aggregated(returned, self.freq)
        predictions = []
        for vector in site_observation_vectors:
            if vector is None:
                predictions.append(None)
                continue
            sipnet_parameter_fields = _run_sipnet_parameter_fields(
                getattr(sipnet_result, "parameters", None), vector.sipnet_parameter_names_read, model_output
            )
            predicted = vector.predict(model_output, sipnet_parameter_fields=sipnet_parameter_fields)
            predictions.append(np.asarray(vector.flat(predicted), dtype=np.float64))
        return _RunOutput(model_output=returned, predictions=tuple(predictions))


@dataclass(frozen=True, eq=False)
class _RunPlan:
    """What every run of an evaluation is sent and does: the observation
    vectors whose predictions come back, the model output asked for, the
    PyEns partial spec, the per-run callable, and each vector's site
    segments of its Flat."""

    observation_vectors: tuple[ObservationVector, ...]
    model_output_variable_names: tuple[str, ...]
    partial: PartialSpec
    run: _Run
    site_positions: tuple[dict[int, np.ndarray], ...]


# ── private helpers ───────────────────────────────────────────────────────────

#: The temporary name of the run index's rows while predictions are unstacked.
_ROW = "__row__"


def _descriptions(
    parameter_vector: ParameterVector, derived_parameters: DerivedParameters | None
) -> dict[str, Any]:
    """``{name: Parameter | DerivedParameter}``: what the map's fit check reads."""
    out: dict[str, Any] = {p.name: p for p in parameter_vector.parameters}
    if derived_parameters is not None:
        out |= {d.name: d for d in derived_parameters.derived_parameters}
    return out


def _crossed_dims(
    sipnet_parameter_map: SIPNETParameterMap, external_inputs: xr.Dataset | None, batch_dim: str
) -> tuple[str, ...]:
    """The batch dims of the external inputs the rules read, other than
    *batch_dim*, in the order they first appear: the dims crossed with
    theta's rows."""
    if external_inputs is None:
        return ()
    dims: dict[str, None] = {}
    for name, variable in external_inputs.data_vars.items():
        if name in sipnet_parameter_map.values_read:
            dims.update(dict.fromkeys(str(d) for d in variable.dims if d not in (SITE, batch_dim)))
    return tuple(dims)


def _corner_theta(parameter_vector: ParameterVector) -> np.ndarray:
    """The corner points of theta (the probability layer's): each parameter
    at every corner of :data:`DOMAIN_CHECK_CORNERS` over the ``e``
    unconstrained numbers of one value, the same at each of its blocks, the
    others at 0; for more than six numbers, at the ``2e + 3`` points 0,
    ``+-c * 1`` and ``+-c`` along each number, ``c`` the outer corner."""
    unconstrained = parameter_vector.unconstrained
    positions = unconstrained.flat_to_values(jnp.arange(unconstrained.size, dtype=jnp.float64))
    places = [
        np.asarray(positions[p.name]).astype(np.int64).reshape(-1, math.prod(p.shape)) for p in unconstrained.parameters
    ]
    return corner_points(places, unconstrained.size)


def _run_index(n_samples: int, sipnet_parameter_fields: xr.Dataset, crossed_dims: Sequence[str], batch_dim: str) -> pd.Index:
    """The rows of an evaluation: the samples, crossed with *crossed_dims*'
    labels in C order."""
    if not crossed_dims:
        return pd.Index(np.arange(n_samples, dtype=BATCH_LABEL_DTYPE), name=batch_dim)
    levels = [np.arange(n_samples, dtype=BATCH_LABEL_DTYPE)] + [
        sipnet_parameter_fields[d].values.astype(BATCH_LABEL_DTYPE) for d in crossed_dims
    ]
    return pd.MultiIndex.from_product(levels, names=[batch_dim, *crossed_dims])


def _placed_predictions(
    run_outputs: Mapping[tuple[int, int], _RunOutput],
    k: int,
    observation_vector: ObservationVector,
    site_positions: Mapping[int, np.ndarray],
    in_domain: np.ndarray,
) -> np.ndarray:
    """``(R, N)``: each run's segment of the *k*-th vector's Flat at its row
    and its site's positions; ``NaN`` where the run failed, and in a row out
    of the domain."""
    predictions = np.full((len(in_domain), observation_vector.dimension), np.nan)
    for (row, site), run_output in run_outputs.items():
        if run_output.predictions[k] is not None:
            predictions[row, site_positions[site]] = run_output.predictions[k]
    predictions[~in_domain] = np.nan
    return predictions


def _number_of_samples(values: Mapping[str, xr.DataArray], batch_dim: str) -> int:
    """The number of samples on *batch_dim*, which the values agree on and
    label ``0`` to ``J - 1``."""
    on_batch = {name: array for name, array in values.items() if batch_dim in array.dims}
    check_values_carry_the_batch_dim(on_batch, batch_dim)
    for name, array in on_batch.items():
        check_batch_labels_count_from_zero(name, array, batch_dim)
    sizes = {array.sizes[batch_dim] for array in on_batch.values()}
    check_values_agree_on_the_samples(sizes, batch_dim)
    return sizes.pop()


def _block_order(observation_vector: ObservationVector, observation_source_name: str) -> np.ndarray:
    """Where each label of a source's observation dim sits among the
    source's entries of the vector's Flat: the order that turns the one into
    the prediction's block, by label."""
    labels = observation_vector.coords[observation_vector.observation_dim_name(observation_source_name)]
    entries = observation_vector.index[observation_vector.positions(observation_source_name=observation_source_name)]
    keys = pd.MultiIndex.from_arrays([entries.get_level_values(name) for name in labels.names], names=labels.names)
    order = keys.get_indexer(labels)
    check_every_observation_has_an_entry(observation_source_name, order)
    return order


def _run_labels(run_index: pd.Index) -> list[np.ndarray]:
    """Each level's labels in the run index's own order, which a
    ``MultiIndex``'s sorted ``levels`` are not."""
    if not isinstance(run_index, pd.MultiIndex):
        return [np.asarray(run_index.values, dtype=BATCH_LABEL_DTYPE)]
    return [
        np.asarray(run_index.get_level_values(i).unique(), dtype=BATCH_LABEL_DTYPE)
        for i in range(run_index.nlevels)
    ]


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
    labels = _run_labels(run_index)
    shape = [len(level) for level in labels]
    coords = {name: batch_coordinate(name, level) for name, level in zip(names, labels)}
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
    full = stacked.reindex(
        {
            **dict(zip(names, _run_labels(run_index))),
            SITE: np.asarray(sites, dtype=stacked[SITE].dtype),
        }
    )
    # A site at which every run failed is absent from the stack, so the
    # reindex leaves its lon/lat NaN; they are the site table's whatever the
    # runs did.
    return full.assign_coords(site_locations)


def _with_evaluation(error: RuntimeError, evaluation: ForwardEvaluation | SIPNETRunsEvaluation) -> RuntimeError:
    error.evaluation = evaluation  # type: ignore[attr-defined]
    return error


# ── checks ────────────────────────────────────────────────────────────────────


def check_forward_model_arguments(
    model: Any,
    sites: Sequence[int],
    *,
    climate: Mapping[int, Any],
    backend: Any,
    observation_vector: ObservationVector | None,
    output_variable_names: Sequence[str] | None,
    freq: Any,
    out_of_domain: Any,
) -> None:
    """Every check on :class:`ForwardModel`'s arguments that needs nothing computed."""
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


def check_forward_model_composes(
    parameter_vector: ParameterVector,
    derived_parameters: DerivedParameters | None,
    site_dims: SiteDims,
    external_inputs: xr.Dataset | None,
    batch_dim: str,
) -> None:
    """The parameter layer's objects and the site dims make one model: the
    derived parameters over the vector, every dim one the sites carry, and
    names apart from the reserved ones and the batch dim's."""
    check_parameter_vector_is_a_parameter_vector(parameter_vector)
    check_site_dims_are_site_dims(site_dims)
    if derived_parameters is not None:
        check_derived_parameters_are_derived_parameters(derived_parameters)
        check_parameter_vectors_share_a_layout(parameter_vector, derived_parameters.parameter_vector)
    coords = parameter_vector.coords if derived_parameters is None else derived_parameters.coords
    for dim, labels in coords.items():
        check_dim_covers_the_sites_labels(dim, labels, site_dims)
    check_names_are_free(parameter_vector, derived_parameters, external_inputs, batch_dim)


def check_sipnet_parameter_map_is_in_domain_at_the_corners(forward_model: ForwardModel) -> None:
    """The map reads and writes values in their domains with each parameter
    at the corners of :data:`DOMAIN_CHECK_CORNERS`, the others at 0: an
    early warning, complete only for maps monotone in each number of theta."""
    inputs = forward_model._external_inputs
    batch_dim = forward_model.batch_dim
    if inputs is not None and batch_dim in inputs.dims:
        # Rows of theta here are corners, not samples: take the inputs of one.
        inputs = inputs.isel({batch_dim: 0}, drop=True)
    values = forward_model._labeled_values(_corner_theta(forward_model.parameter_vector), inputs)
    sipnet_map = forward_model.sipnet_parameter_map
    fields = sipnet_map.sipnet_parameter_fields(values, site_dims=forward_model.site_dims)
    check_map_is_in_domain_at_the_corners(sipnet_map.out_of_domain(fields, values, site_dims=forward_model.site_dims))


def check_map_is_in_domain_at_the_corners(outside: pd.DataFrame) -> None:
    """No value the map read or wrote at the corners of theta lies outside
    its domain (:meth:`SIPNETParameterMap.out_of_domain
    <sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap.out_of_domain>`'s
    report)."""
    if not outside.empty:
        names = sorted({str(n) for n in (*outside["sipnet_parameter"].dropna(), *outside["value_name"].dropna())})
        raise ValueError(
            f"the map can read or write {truncated(names)} outside their domains, at a corner of theta "
            f"in {DOMAIN_CHECK_CORNERS}; give the parameters supports or priors whose values the rules "
            "map into the domains, or pass out_of_domain='fail_row'."
        )


def check_parameter_vector_is_a_parameter_vector(parameter_vector: Any) -> None:
    """The vector is a :class:`~sipnet_calibration.parameters.ParameterVector`."""
    if not isinstance(parameter_vector, ParameterVector):
        raise TypeError(f"parameter_vector must be a ParameterVector, got {type(parameter_vector).__name__}.")


def check_site_dims_are_site_dims(site_dims: Any) -> None:
    """The sites are a :class:`~sipnet_calibration.site_dims.SiteDims`."""
    if not isinstance(site_dims, SiteDims):
        raise TypeError(f"site_dims must be a SiteDims, got {type(site_dims).__name__}.")


def check_derived_parameters_are_derived_parameters(derived_parameters: Any) -> None:
    """The derived parameters are a :class:`~sipnet_calibration.parameters.DerivedParameters`."""
    if not isinstance(derived_parameters, DerivedParameters):
        raise TypeError(f"derived_parameters must be a DerivedParameters, got {type(derived_parameters).__name__}.")


def check_dim_covers_the_sites_labels(dim: str, labels: pd.Index, site_dims: SiteDims) -> None:
    """A dim of the vector or derived parameters is a dim of the site dims,
    holding every label some site carries, so every site finds its value;
    extra labels are read at no site."""
    if dim not in site_dims.coords:
        raise ValueError(
            f"the parameters are indexed by {dim!r}, which is neither 'site' nor a site-labels name of the "
            f"site dims ({list(site_dims.coords)}); give its site labels to SiteDims."
        )
    missing = [label for label in site_dims.coords[dim] if label not in set(labels.tolist())]
    if missing:
        raise KeyError(
            f"the parameters' {dim!r} labels lack {truncated(missing)}, which some site carries; build "
            "the vector on site_dims.coords."
        )


def check_names_are_free(
    parameter_vector: ParameterVector,
    derived_parameters: DerivedParameters | None,
    external_inputs: xr.Dataset | None,
    batch_dim: str,
) -> None:
    """No parameter, derived parameter, element axis or external input is a
    reserved name or the batch dim's, nor is a dim the batch dim's, since the
    merged labeled values take them as dims and coordinates: a parameter
    named ``lon`` would overwrite one."""
    pieces = [*parameter_vector.parameters, *(() if derived_parameters is None else derived_parameters.derived_parameters)]
    names = {p.name for p in pieces} | {axis for p in pieces for axis in p.element_labels}
    if external_inputs is not None:
        names |= {str(n) for n in external_inputs.data_vars}
    dims = parameter_vector.coords if derived_parameters is None else derived_parameters.coords
    taken = sorted(n for n in names if n in RESERVED_NAMES or n == batch_dim)
    taken += [d for d in dims if d == batch_dim]
    if taken:
        raise ValueError(
            f"{truncated(taken)} name parameters, derived parameters, element axes, external inputs or dims, "
            f"but are reserved names ({sorted(RESERVED_NAMES)}) or the batch dim's {batch_dim!r}; rename them."
        )


def check_external_inputs_are_for_theta(external_inputs: xr.Dataset, n_rows: int, batch_dim: str) -> None:
    """External inputs on theta's batch dim are labeled ``0`` to ``J - 1``,
    one per row, with which they zip."""
    if batch_dim not in external_inputs.dims:
        return
    labels = external_inputs[batch_dim].values.tolist()
    if labels != list(range(n_rows)):
        raise ValueError(
            f"external inputs on {batch_dim!r} are labeled {truncated(labels)}, but theta has {n_rows} rows; "
            "label them 0 to J - 1, one per row of theta, or name their dim otherwise to cross them with "
            "theta."
        )


def check_external_input_covers_the_sites(name: str, variable: xr.DataArray, sites: Sequence[int]) -> None:
    """An external input on ``site`` holds every site run."""
    held = set(variable.indexes[SITE].tolist())
    missing = [site for site in sites if site not in held]
    if missing:
        raise KeyError(f"external input {name!r} has no value for site(s) {truncated(missing)}.")


def check_out_of_domain_is_known(out_of_domain: Any) -> None:
    """``out_of_domain`` is ``"raise"`` or ``"fail_row"``."""
    if out_of_domain not in ("raise", "fail_row"):
        raise ValueError(f"out_of_domain is 'raise' or 'fail_row', got {out_of_domain!r}.")


def check_sipnet_parameters_are_in_the_domain(outside: pd.DataFrame) -> None:
    """Every value lies in its domain, a rule input in its requirement's and
    a SIPNET parameter in pySIPNET's, since the prior and the map otherwise
    put mass where a rule or SIPNET is undefined."""
    if outside.empty:
        return
    first = outside.iloc[0].dropna().to_dict()
    raise SIPNETParametersOutOfDomainError(
        f"{len(outside)} value(s) lie outside their domains, the first at {first}; nothing ran. Give "
        "the parameters supports or priors the map sends into the domains, or pass "
        "out_of_domain='fail_row' to mark such rows invalid, which truncates the prior."
    )


def check_external_inputs_cross_no_dims(crossed_dims: Sequence[str]) -> None:
    """``forward(theta)`` returns one row per row of theta, which crossed
    dims multiply."""
    if crossed_dims:
        raise ValueError(
            f"the external inputs cross theta's rows with {list(crossed_dims)}, so an evaluation "
            "has more rows than theta; call evaluate(theta) and reduce its predictions over "
            "them, which is a modeling choice."
        )


def check_evaluation_has_predictions(evaluation: ForwardEvaluation) -> None:
    """An evaluation has predictions, which only an observation vector gives."""
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
    """The model is a pySIPNET ``SIPNETModel``."""
    if not isinstance(model, SIPNETModel):
        raise TypeError(f"model must be a pysipnet SIPNETModel, got {type(model).__name__}.")


def check_backend_is_a_pyens_backend(backend: Any) -> None:
    """The backend is a PyEns ``Backend``."""
    if not isinstance(backend, Backend):
        raise TypeError(f"backend must be a pyens Backend, got {type(backend).__name__}.")


def check_output_variables_are_named(
    output_variable_names: Sequence[str] | None, observation_vector: ObservationVector | None
) -> None:
    """The runs' output variables are named, or an observation vector names them."""
    if output_variable_names is None and observation_vector is None:
        raise ValueError(
            "name the output variables each run returns (output_variable_names=), "
            "or give an observation_vector whose operators say what they read."
        )


def check_freq_is_for_the_prior_predictive(freq: Any, observation_vector: ObservationVector | None) -> None:
    """``freq`` is given only without an observation vector, whose operators align on their own."""
    if freq is not None and observation_vector is not None:
        raise ValueError(
            "freq= aggregates model output for the prior predictive; with an observation "
            "vector the operators decide their own alignment, so freq would be ignored. "
            "Drop one of the two."
        )


def check_climate_covers_the_sites(climate: Mapping[int, Any], sites: Sequence[int]) -> None:
    """The climate has ``ClimateDrivers`` for every site run."""
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
    """Under a process backend every site's drivers are file-backed, so no run carries a copy."""
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
    """Every site the observation vector observes is run."""
    extra = sorted(set(observation_vector.sites) - set(sites))
    if extra:
        raise ValueError(
            f"the observation vector observes site(s) {truncated(extra)} that are not run; restrict "
            "the observation vector to the sites run first "
            "(observation_vector.restrict_to_sites(site_dims.sites))."
        )


def check_output_variable_names_cover_the_operators(
    output_variable_names: Sequence[str], observation_vector: ObservationVector
) -> None:
    """The output variables include every one the operators read."""
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
    """A site's slice of the observation vector is its segment of Flat, where its predictions are placed."""
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
    machinery_failures: Sequence[tuple[dict, BaseException]], evaluation: ForwardEvaluation | SIPNETRunsEvaluation
) -> None:
    """No run failed in the machinery, which says nothing of the parameters."""
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


def check_sipnet_runs_arguments(
    model: Any, sipnet_parameter_map: Any, site_dims: Any, climate: Any, backend: Any, out_of_domain: Any
) -> None:
    """Every check on :class:`SIPNETRuns`' arguments."""
    check_model_is_a_sipnet_model(model)
    check_sipnet_parameter_map_is_a_map(sipnet_parameter_map)
    check_site_dims_are_site_dims(site_dims)
    check_backend_is_a_pyens_backend(backend)
    check_out_of_domain_is_known(out_of_domain)
    check_climate_covers_the_sites(climate, site_dims.sites)
    check_climate_is_file_backed(climate, site_dims.sites, backend)


def check_sipnet_parameter_map_is_a_map(sipnet_parameter_map: Any) -> None:
    """The map is a :class:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap`."""
    if not isinstance(sipnet_parameter_map, SIPNETParameterMap):
        raise TypeError(f"sipnet_parameter_map must be a SIPNETParameterMap, got {type(sipnet_parameter_map).__name__}.")


def check_runs_are_sipnet_runs(runs: Any) -> None:
    """A simulator's runs are a :class:`SIPNETRuns`."""
    if not isinstance(runs, SIPNETRuns):
        raise TypeError(f"runs must be a SIPNETRuns, got {type(runs).__name__}; build one with SIPNETRuns(...).")


def check_observation_vector_is_an_observation_vector(observation_vector: Any) -> None:
    """An observation vector is an :class:`~sipnet_calibration.observation.ObservationVector`."""
    if not isinstance(observation_vector, ObservationVector):
        raise TypeError(
            f"an observation vector must be an ObservationVector, got {type(observation_vector).__name__}; "
            "build one from ObservationSources."
        )


def check_something_is_asked_for(observation_vectors: Sequence[Any], output_variable_names: Sequence[str]) -> None:
    """An evaluation returns predictions, model output or both."""
    if not observation_vectors and not output_variable_names:
        raise ValueError(
            "nothing is asked for: give observation_vectors whose predictions to return, "
            "output_variable_names whose model output to return, or both."
        )


def check_freq_is_for_model_output(freq: Any, output_variable_names: Sequence[str]) -> None:
    """``freq`` aggregates model output, which output variables name."""
    if freq is not None and not output_variable_names:
        raise ValueError(
            "freq= aggregates model output, but no output_variable_names are given; the operators "
            "decide their own alignment. Name the variables, or drop freq."
        )


def check_values_are_labeled(values: Any) -> None:
    """Values are labeled values (a mapping of DataArrays) or a Dataset."""
    if isinstance(values, xr.Dataset):
        return
    if not isinstance(values, Mapping) or not all(isinstance(v, xr.DataArray) for v in values.values()):
        raise TypeError(
            f"values must be labeled values, a mapping of DataArrays by name, or an xr.Dataset; got "
            f"{type(values).__name__}."
        )


def check_external_inputs_are_not_values(external_names: Sequence[Any], values: Mapping[str, Any]) -> None:
    """No external input is named like a value given, which it would replace."""
    shared = [str(n) for n in external_names if str(n) in values]
    if shared:
        raise ValueError(
            f"the external inputs {truncated(shared)} are named like values given; a value is given once, "
            "so rename one."
        )


def check_value_covers_the_sites(name: str, variable: xr.DataArray, sites: Sequence[int]) -> None:
    """A value or external input on ``site`` holds every site run."""
    held = set(variable.indexes[SITE].tolist())
    missing = [site for site in sites if site not in held]
    if missing:
        raise KeyError(
            f"the value {name!r} has no value for site(s) {truncated(missing)}, which are run; give it at every "
            "site of the site dims."
        )


def check_values_carry_the_batch_dim(on_batch: Mapping[str, Any], batch_dim: str) -> None:
    """Some value is on the batch dim, which says how many samples there are."""
    if not on_batch:
        raise ValueError(
            f"no value the map reads is on the batch dim {batch_dim!r}, so the number of samples is "
            "unknown; give the values on it, labeled 0 to J - 1."
        )


def check_batch_labels_count_from_zero(name: str, array: xr.DataArray, batch_dim: str) -> None:
    """A value's batch labels are ``0`` to ``J - 1``, the rows they zip with."""
    labels = array[batch_dim].values.tolist() if batch_dim in array.coords else list(range(array.sizes[batch_dim]))
    if labels != list(range(array.sizes[batch_dim])):
        raise ValueError(
            f"the value {name!r} labels {batch_dim!r} {truncated(labels)}; label the samples 0 to J - 1."
        )


def check_values_agree_on_the_samples(sizes: set[int], batch_dim: str) -> None:
    """Every value on the batch dim has the same number of samples."""
    if len(sizes) > 1:
        raise ValueError(f"the values hold {sorted(sizes)} samples on {batch_dim!r}; give each the same number.")


def check_outputs_are_predictions(outputs: Sequence[str], by_output: Mapping[str, str]) -> None:
    """A simulator is restricted to its own outputs."""
    unknown = [name for name in outputs if name not in by_output]
    if unknown:
        raise ValueError(
            f"the simulator outputs {list(by_output)}, not {truncated(unknown)}; restrict it to its own outputs."
        )


def check_sites_are_observed(sites: Sequence[int], observation_vector: ObservationVector) -> None:
    """The sites a restriction keeps are observed by the vector."""
    extra = sorted(set(sites) - set(observation_vector.sites))
    if extra:
        raise ValueError(
            f"the labels in use observe site(s) {truncated(extra)}, which the observation vector does not; "
            "the simulator cannot predict them."
        )


def check_labels_are_the_observations(
    observation_source_name: str, labels: pd.Index, observation_vector: ObservationVector
) -> None:
    """A source's labels in use are its observations at the sites kept, in
    the vector's order: a simulator predicts what its vector observes."""
    dim = observation_vector.observation_dim_name(observation_source_name)
    held = observation_vector.coords.get(dim)
    if held is None or not held.equals(labels):
        raise ValueError(
            f"the labels in use of {dim!r} are not the observation vector's observations of "
            f"{observation_source_name!r} at their sites; bind at observation_vector.coords, or select by site."
        )


def check_sites_have_values(sites: Sequence[int], site_labels: Any) -> None:
    """Every site predicted has values: the given values are read there."""
    held = set(np.asarray(site_labels).tolist())
    missing = [site for site in sites if site not in held]
    if missing:
        raise ValueError(
            f"the observation vector observes site(s) {truncated(missing)}, which the labels in use of 'site' "
            "lack, so no value would be read there; bind at every site observed."
        )


def check_every_observation_has_an_entry(observation_source_name: str, order: np.ndarray) -> None:
    """Every observation of a source has an entry in the vector's Flat; this
    is a defect in ObservationVector, not in the inputs."""
    if (order < 0).any():
        raise ValueError(
            f"the observation vector's Flat lacks an observation of {observation_source_name!r}; this is a "
            "defect in ObservationVector, not in the inputs."
        )


def check_some_run_succeeded(evaluation: ForwardEvaluation | SIPNETRunsEvaluation) -> None:
    """Some run succeeded, so there is model output to stack."""
    if not bool(evaluation.run_succeeded.any()):
        raise _with_evaluation(
            RuntimeError(
                "every run failed at its parameters; there is no model output to stack NaN "
                "onto. Draw parameters the model can run, or read the failures on this "
                "error's `evaluation` attribute."
            ),
            evaluation,
        )
