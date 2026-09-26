"""The forward model: unconstrained calibration parameters to predictions, over
a site set, with SIPNET run once per sample and site.

Where this sits
---------------
::

    theta (J, D)  --ParameterVector.sipnet_parameter_fields-->
                    SIPNET parameter fields (sample, site)
                  --PyEns, one SIPNETModel run per (sample, site)-->  model output
                  --ObservationVector.predict and .flat on the worker-->
                    one site's segment of Flat per run
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
    ``(J, D)`` float64 ``jax.Array``, coerced from what was received (any
    array-like); a ``(D,)`` input is one row.
``sipnet_parameter_fields``
    The :data:`~sipnet_calibration.fields.SIPNETParameterFields` that were
    run, every variable on ``(sample, site)``, ``sample`` labeled ``0`` to
    ``J - 1`` in ``theta``'s row order and ``site`` in the parameter vector's
    order, located from the model's site table. ``sample`` is the model's
    ``batch_dim``, the name every output below carries too.
``model_output``
    The :data:`~sipnet_calibration.fields.ModelOutput` of the runs on
    ``(sample, site, time)``, from
    :func:`~sipnet_calibration.fields.stack_model_outputs`, ``NaN`` where a
    run failed. Its attributes are the first run's; with ``freq`` they gain
    pySIPNET's ``resampling_frequency`` and ``time_step_length_source``.
    ``None`` when an observation vector was given.
``predictions``
    ``(J, N)`` float64 ``jax.Array`` in the observation vector's order,
    ``NaN`` in every entry of a sample with a failed run; ``None`` without an
    observation vector.
``run_succeeded``
    bool ``xr.DataArray`` on ``(sample, site)``, a field: ``int64``
    ``sample`` with its attributes, ``int32`` ``site`` with its ``lon``/``lat``,
    so it can be mapped.
``failures``
    ``pd.DataFrame`` with columns ``sample``, ``site``, ``error`` (the class
    name of the exception the run raised) and ``message``, one row per run
    that failed at its parameters.
``valid``
    bool ``(J,)`` ``jax.Array``: every run of the sample succeeded and, where
    there are predictions, they are finite.

A run **fails at its parameters** when pySIPNET refuses them
(``pydantic.ValidationError``), SIPNET exits non-zero or writes nothing
(``SIPNETRunError``), the run times out (``subprocess.TimeoutExpired``), or
the output it wrote holds a non-finite value in a variable that was read
(:class:`ModelOutputNotFiniteError`, a blow-up SIPNET exits 0 on): the
sample's row is ``NaN``, which pyEKI repairs and a sampler rejects. Anything
else that comes back from a worker (a PyEns ``TaskFailedError``, a
``RemoteError`` of another type, a missing binary, an import error) is the
**machinery** failing, says nothing about the parameters, and is raised after
the batch is collected as a ``RuntimeError`` whose ``evaluation`` attribute
holds what was collected (``run_succeeded`` and ``failures`` filled, no
predictions or output), for diagnosis. On the prior-predictive path a batch
in which every run failed at its parameters is also raised, the same way,
since there is no time axis to stack ``NaN`` onto.

Functions
---------
:class:`ForwardModel`
    ``evaluate(theta) -> ForwardEvaluation``; ``__call__(theta)`` returns
    ``evaluate(theta).predictions``, ``(N,)`` for a ``(D,)`` input. What it was
    built from and derived is read-only: ``model``, ``parameter_vector``,
    ``observation_vector``, ``backend``, ``freq``, ``climate``, ``sites``,
    ``site_table``, ``output_variable_names``,
    ``sipnet_parameter_names_written`` and ``batch_dim`` are properties, so
    the run machinery built from them cannot go stale.
:class:`ForwardEvaluation`
    The record above.
:data:`MODEL_FAILURES`, :class:`ModelOutputNotFiniteError`
    The exceptions that mean a run failed at its parameters.

Notes
-----
**Flat is JAX.** ``theta``, the predictions and ``valid`` are ``jax.Array``\\ s,
as the parameter and observation vectors' Flat is; every method accepts any
array-like. The runs themselves are placed in NumPy and converted on return.

**One batch dim.** A run is one row of ``theta``: SIPNET parameter fields on
another batch dim beside the model's (an ``initial_condition_member`` crossed
with ``sample``) are refused, with the advice to give ``theta`` one row per
combination and select each row's member in the hook.

**The ``PartialSpec`` is built once.** Its fixed inputs (the climate, the
site id and the site's slice of the observation vector, all along one site
axis) and its free inputs (the SIPNET parameter names the vector sets) hold
for the model's lifetime, and on every call the PyEns grid of SIPNET
parameter values is zipped with that site axis. The free names are learned by mapping one prior draw through
``sipnet_parameter_fields`` in ``__init__``, which fails fast, before
anything is queued, on a hook that does not return SIPNET parameter fields on
``(sample, site)`` over the vector's sites in pySIPNET's flat parameter
names.

**Samples are labeled by row.** The SIPNET-parameter-fields hook must label
its batch dim ``0`` to ``J - 1`` in the order of ``theta``'s rows, although a
batch dim may in general carry any distinct integers: the model maps each run
back to its row of the predictions and of ``valid`` by that label, and SIPNET
parameter fields whose labels were anything else could place a run in another
sample's row.

**The operators run on the worker** because reading a run's output back in
the calling process costs more than the run (pySIPNET parses the whole
file), and an observation operator is pointwise in site, so each worker can
reduce its own run to the observations. Each run receives only its site's
slice of the observation vector, as an input along the site axis, so a run
at a site no observation source observes carries nothing and returns
nothing, and a process backend pickles only that site's observations rather
than the whole vector. The calling process only places small arrays.

**Predictions are placed by position.** A run returns its slice's Flat,
which the calling process writes at ``positions(site=)`` of the whole
vector. That is right because the observation vector is site-major: a
one-site selection's index is the whole vector's index at that site's
positions. ``__init__`` checks this for every observed site.

**Runs on different time axes** (sites with driver records of different
lengths) are stacked by an outer join on the prior-predictive path, so the
shorter records are ``NaN``-padded and the interval coordinates gain a
``site`` or ``sample`` dimension; select one site before aggregating such
a stack, as :func:`~sipnet_calibration.observation.aggregate_time` asks.

**The SIPNET parameters an operator reads** (``leaf_carbon_per_area`` for
the LAI operator) are read off the run's own ``SIPNETResult.parameters``, the
complete set the run used, whether the parameter vector set them or the base
parameter set supplied them: the worker builds one run's zero-dimensional
SIPNET parameter fields from ``SIPNETParameters.dataarray(name)`` for every
name the operators read, and hands them to its site's slice of the
observation vector.

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
    model_output = prior.evaluate(theta).model_output  # (sample, site, time)
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
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
    LAT,
    LON,
    SAMPLE,
    SITE,
    SITE_DTYPE,
    FrozenMapping,
)
from sipnet_calibration.fields import (
    Field,
    ModelOutput,
    ReadOnlyCopies,
    SIPNETParameterFields,
    batch_coordinate,
    check_batch_dim_name_is_not_a_model_output_name,
    check_batch_dim_name_is_not_reserved,
    check_sipnet_parameter_fields_are_a_dataset,
    to_model_output,
    resolve_output_variable_names,
    stack_model_outputs,
    validate_sipnet_parameter_fields,
)
from sipnet_calibration.observation import (
    DEFAULT_METHOD_FOR_KIND,
    ObservationVector,
    aggregate_time,
    check_batch_dim_is_not_an_observation_source_name,
)
from sipnet_calibration.observation.time_alignment import (
    check_frequency_is_an_offset_alias,
)
from sipnet_calibration.parameter_vector import (
    ParameterVector,
    check_batch_dim_name_is_not_taken,
)
from sipnet_calibration.sites import (
    load_sites,
    site_locations,
    site_lookup,
)
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


#: The exceptions that mean a run failed **at its parameters**: the sample's
#: row becomes NaN. Everything else a worker returns is the machinery failing
#: and is raised.
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
    read from it changes it: ``theta``, ``predictions`` and ``valid`` are
    ``jax.Array``\\ s, which cannot be written to; ``sipnet_parameter_fields``,
    ``model_output`` and ``run_succeeded`` are read-only copies
    (:func:`~sipnet_calibration.fields.read_only_copy`) and ``failures`` a copy,
    on every read.

    Notes
    -----
    Compared and hashed by identity (``eq=False``), as the package's other
    records of arrays are: a generated ``==`` would compare arrays, whose
    truth value is ambiguous, and a generated hash would fail on them.
    """

    theta: jax.Array
    sipnet_parameter_fields: SIPNETParameterFields = ReadOnlyCopies()
    model_output: ModelOutput | None = ReadOnlyCopies()
    predictions: jax.Array | None
    run_succeeded: Field = ReadOnlyCopies()
    failures: pd.DataFrame = ReadOnlyCopies()
    valid: jax.Array


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
        parameter vector's; ``restrict_to_sites(parameter_vector.sites)`` it
        first otherwise.
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
    to_sipnet_parameter_fields:
        ``theta (J, D) -> SIPNET parameter fields (batch_dim, site)``;
        defaults to ``parameter_vector.sipnet_parameter_fields`` with
        *batch_dim*. The hook for an experiment that maps initial-condition
        state into SIPNET parameters. The SIPNET parameter fields must be on
        exactly ``(batch_dim, site)``, every variable on
        both, with *batch_dim* labeled ``0`` to ``J - 1`` in ``theta``'s row
        order, the parameter vector's sites in its order, and each variable
        named by pySIPNET's flat parameter name (``max_photosynthesis_rate``,
        not the alias ``aMax``).
    site_table:
        The site table the runs are labeled from (``lon``/``lat``), one row
        per site; defaults to the vector's when it carries them, else
        :func:`sipnet_calibration.sites.load_sites`.
    batch_dim:
        The name of the batch dim of ``theta``'s rows, which the SIPNET
        parameter fields, ``model_output``, ``run_succeeded`` and the
        ``failures`` column all carry; ``sample`` by default. Whatever
        *to_sipnet_parameter_fields* is, it may take no name the parameter
        vector refuses
        (:func:`~sipnet_calibration.parameter_vector.check_batch_dim_name_is_not_taken`),
        the model output uses
        (:func:`~sipnet_calibration.fields.check_batch_dim_name_is_not_a_model_output_name`)
        or the observation vector's sources take
        (:func:`~sipnet_calibration.observation.check_batch_dim_is_not_an_observation_source_name`);
        each is refused here, before anything runs.

    Raises
    ------
    TypeError
        If *model* is not a ``SIPNETModel``, *backend* not a PyEns
        ``Backend``, a site's drivers not ``ClimateDrivers``,
        *to_sipnet_parameter_fields* returns something other than an
        ``xr.Dataset``, or
        *batch_dim* is not a string.
    ValueError
        If *batch_dim* is a name it may not be (above); if neither
        *output_variable_names* nor *observation_vector* is given; *freq* is
        given with an observation vector or is not a pandas offset alias; a
        site has no drivers, or in-memory drivers under a process backend; the
        observation vector observes a site the parameter vector does not run;
        the output variables do not cover the operators, or one is switched off
        by the model's flags, or (with *freq*) has a kind no method keeps; the
        site table lacks ``lon``/``lat`` or lists a site twice;
        *to_sipnet_parameter_fields* does not return SIPNET parameter fields
        for the batch; or an operator reads a SIPNET parameter that neither the
        SIPNET parameter fields write nor the model's base parameter set holds.
    KeyError
        If an output variable name is not a pySIPNET output variable, a site
        of the parameter vector is not in the given site table, or
        *to_sipnet_parameter_fields* sets a SIPNET parameter pySIPNET does not
        have.
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
        to_sipnet_parameter_fields: Callable[[Any], xr.Dataset] | None = None,
        site_table: pd.DataFrame | None = None,
        batch_dim: str = SAMPLE,
    ) -> None:
        check_batch_dim_name_is_not_reserved(batch_dim, message_name="batch_dim")
        check_forward_model_arguments(
            model,
            parameter_vector,
            climate=climate,
            backend=backend,
            observation_vector=observation_vector,
            output_variable_names=output_variable_names,
            freq=freq,
            site_table=site_table,
        )
        self._model = model
        self._parameter_vector = parameter_vector
        self._sites: tuple[int, ...] = parameter_vector.sites
        self._backend = backend
        self._observation_vector = observation_vector
        self._freq = freq
        check_batch_dim_name_is_not_taken(parameter_vector, batch_dim)
        self._batch_dim = batch_dim
        self._climate = FrozenMapping({site: climate[site] for site in self._sites})
        self._output_variable_names = _output_variable_names(
            output_variable_names, observation_vector
        )
        check_output_variables_can_be_returned(self._output_variable_names, model, freq)
        check_batch_dim_name_is_not_a_model_output_name(
            batch_dim, self._output_variable_names, message_name="batch_dim"
        )
        if observation_vector is not None:
            check_batch_dim_is_not_an_observation_source_name(
                observation_vector.observation_sources, batch_dim
            )
        chosen_site_table = _chosen_site_table(site_table, parameter_vector)
        # site_locations checks the table locates the sites, once, before
        # the lookup below relies on it.
        self._site_locations = site_locations(self._sites, chosen_site_table)
        self._site_table = site_lookup(chosen_site_table).loc[list(self._sites)]
        self._to_sipnet_parameter_fields = to_sipnet_parameter_fields or partial(
            parameter_vector.sipnet_parameter_fields, batch_dim=batch_dim
        )
        self._sipnet_parameter_names_written = self._probe_sipnet_parameter_names()
        if observation_vector is not None:
            check_base_parameters_set_what_the_operators_read(
                model,
                observation_vector.sipnet_parameter_names_read,
                self._sipnet_parameter_names_written,
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
        """The ``SIPNETModel`` every run goes through."""
        return self._model

    @property
    def parameter_vector(self) -> ParameterVector:
        """The calibration vector; its sites are the sites run."""
        return self._parameter_vector

    @property
    def observation_vector(self) -> ObservationVector | None:
        """The observation vector the predictions follow, or ``None``."""
        return self._observation_vector

    @property
    def backend(self) -> Backend:
        """The PyEns backend the runs execute on."""
        return self._backend

    @property
    def freq(self) -> str | None:
        """The prior-predictive aggregation frequency, or ``None``."""
        return self._freq

    @property
    def climate(self) -> Mapping[int, ClimateDrivers]:
        """``{site id: ClimateDrivers}`` for the sites run, read-only."""
        return self._climate

    @property
    def sites(self) -> tuple[int, ...]:
        """The sites run: the parameter vector's, in its order."""
        return self._sites

    @property
    def site_table(self) -> pd.DataFrame:
        """The site table the runs are labeled from, one row per site run; a copy."""
        return self._site_table.copy()

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        """The pySIPNET output variables each run returns."""
        return self._output_variable_names

    @property
    def sipnet_parameter_names_written(self) -> tuple[str, ...]:
        """The SIPNET parameters the hook writes, fixed at construction."""
        return self._sipnet_parameter_names_written

    @property
    def batch_dim(self) -> str:
        """The name of the batch dim of ``theta``'s rows, on every output."""
        return self._batch_dim

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
            f"observation_vector={self.observation_vector!r}"
            if self.observation_vector is not None
            else f"output_variable_names={self.output_variable_names!r}, freq={self.freq!r}"
        )
        return f"ForwardModel(D={self.input_dimension}, sites={len(self.sites)}, {what})"

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, theta: Any) -> ForwardEvaluation:
        """Run SIPNET once per sample and site, and collect what came back.

        Parameters
        ----------
        theta:
            ``(J, D)`` unconstrained calibration parameters, one row per
            sample, or ``(D,)`` for one sample.

        Returns
        -------
        ForwardEvaluation
            With ``predictions`` on the calibration path and
            ``model_output`` on the prior-predictive path; see the module
            docstring's Data model.

        Raises
        ------
        TypeError
            If *theta* is not a rectangular array of real numbers, or the
            SIPNET-parameter-fields hook returns something other than an
            ``xr.Dataset``.
        ValueError
            If *theta* is not ``(D,)`` or ``(J, D)`` with ``J >= 1``, or holds
            a non-finite value; or if the SIPNET parameter fields the hook
            returns are not for this batch (dims other than ``(batch_dim, site)``, a
            variable not on both, labels other than ``0`` to ``J - 1`` in
            order, sites other than the vector's in order, a name that is not
            pySIPNET's flat parameter name, or other SIPNET parameters than the
            model was built for).
        RuntimeError
            If a run failed in the machinery rather than at its parameters,
            or, on the prior-predictive path, if every run failed at its
            parameters. The error's ``evaluation`` attribute is a
            :class:`ForwardEvaluation` of what was collected, with no
            predictions or model output.
        """
        theta = np.asarray(as_batched_flat(theta, self.input_dimension, message_name="theta"))
        check_theta_has_a_row(theta)
        check_theta_is_finite(theta)
        n_samples = len(theta)
        sipnet_parameter_fields = self._sipnet_parameter_fields_for(theta)
        check_sipnet_parameter_fields_set_the_parameters_built_for(
            sipnet_parameter_fields, self.sipnet_parameter_names_written
        )
        grids = fields_from_dataset(sipnet_parameter_fields, axes={SITE: self._site_axis})
        spec = self._partial(**grids)
        ensemble_result = EnsembleRunner(self._run, self.backend).run(spec)

        run_outputs_by_sample_site, succeeded, failures, machinery_failures = _sort_records(
            ensemble_result, n_samples, self.sites, self.batch_dim
        )
        collected = ForwardEvaluation(
            theta=jnp.asarray(theta),
            sipnet_parameter_fields=sipnet_parameter_fields,
            model_output=None,
            predictions=None,
            run_succeeded=_run_succeeded(
                succeeded,
                sipnet_parameter_fields[self.batch_dim].values,
                self.sites,
                batch_dim=self.batch_dim,
                site_locations=self._site_locations,
            ),
            failures=failures,
            valid=jnp.zeros(n_samples, dtype=bool),
        )
        check_no_run_failed_in_the_machinery(machinery_failures, collected)
        sample_succeeded = succeeded.all(axis=1)
        if self.observation_vector is None:
            check_some_run_succeeded(collected)
            model_output = _stacked_model_output(
                run_outputs_by_sample_site,
                n_samples,
                self.sites,
                batch_dim=self.batch_dim,
                site_table=self._site_table,
                site_locations=self._site_locations,
            )
            return replace(
                collected, model_output=model_output, valid=jnp.asarray(sample_succeeded)
            )
        predictions = self._placed_predictions(run_outputs_by_sample_site, n_samples)
        # A sample with any failed run is invalid as a whole: pyEKI updates
        # per row, so a row that is partly a prediction cannot be used.
        predictions[~sample_succeeded] = np.nan
        valid = sample_succeeded & np.isfinite(predictions).all(axis=1)
        return replace(
            collected, predictions=jnp.asarray(predictions), valid=jnp.asarray(valid)
        )

    def __call__(self, theta: Any) -> jax.Array:
        """``evaluate(theta).predictions``: ``(J, N)``, or ``(N,)`` for a ``(D,)`` theta.

        Raises
        ------
        ValueError
            Without an observation vector, and as :meth:`evaluate` does.
        TypeError, RuntimeError
            As :meth:`evaluate` does.
        """
        self._observation_vector_for("__call__")
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

    def _probe_sipnet_parameter_names(self) -> tuple[str, ...]:
        """The SIPNET parameter names the hook sets, learned from one prior draw."""
        draw = self.parameter_vector.sample(jax.random.key(0), 1)
        sipnet_parameter_fields = self._sipnet_parameter_fields_for(np.asarray(draw))
        return tuple(str(name) for name in sipnet_parameter_fields.data_vars)

    def _sipnet_parameter_fields_for(self, theta: np.ndarray) -> xr.Dataset:
        """The hook's SIPNET parameter fields for *theta*, checked for the batch.

        SIPNET parameter fields that carry neither ``lon`` nor ``lat``, as a
        parameter vector built from bare site ids gives them, are located from
        this model's site table before they are validated.
        """
        sipnet_parameter_fields = self._to_sipnet_parameter_fields(theta)
        check_sipnet_parameter_fields_are_for_the_batch(
            sipnet_parameter_fields, self.sites, n_samples=len(theta), batch_dim=self.batch_dim
        )
        if LON not in sipnet_parameter_fields.coords and LAT not in sipnet_parameter_fields.coords:
            sipnet_parameter_fields = sipnet_parameter_fields.assign_coords(self._site_locations)
        validate_sipnet_parameter_fields(
            sipnet_parameter_fields, message_name="what to_sipnet_parameter_fields returns"
        )
        return sipnet_parameter_fields

    def _build_partial(self) -> PartialSpec:
        climate = Grid({s: self.climate[s] for s in self.sites}, along=self._site_axis)
        site_ids = Grid(list(self.sites), along=self._site_axis)
        site_observation_vectors = Grid(
            {s: self._site_slices.get(s) for s in self.sites}, along=self._site_axis
        )
        placeholders = {
            name: Grid([0.0] * len(self.sites), along=self._site_axis)
            for name in self.sipnet_parameter_names_written
        }
        spec = EnsembleSpec(
            inputs={
                "climate": climate,
                "site": site_ids,
                "site_observation_vector": site_observation_vectors,
                **placeholders,
            }
        )
        return spec.freeze(free=list(self.sipnet_parameter_names_written))

    def _placed_predictions(
        self, run_outputs_by_sample_site: Mapping[tuple[int, int], _RunOutput], n_samples: int
    ) -> np.ndarray:
        """``(J, N)``: each run's segment of Flat written at its site's positions."""
        observation_vector = self._observation_vector_for("predictions")
        predictions = np.full((n_samples, observation_vector.dimension), np.nan)
        for (sample, site), run_output in run_outputs_by_sample_site.items():
            if run_output.predictions is not None:
                predictions[sample, self._site_positions[site]] = run_output.predictions
        return predictions


# ── the per-run callable ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _RunOutput:
    """What one run sends back: its model output, its site's predictions, or neither.

    ``predictions`` is the site's segment of the observation vector's Flat,
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
    observation vector as ``site_observation_vector``, and nothing when it
    receives ``None``, at a site no observation source observes. The SIPNET
    parameters the operators read are taken from the run's own
    ``SIPNETResult.parameters``, the complete set it ran with.
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
        predicted = site_observation_vector.predict(
            model_output, sipnet_parameter_fields=sipnet_parameter_fields
        )
        predictions = np.asarray(site_observation_vector.flat(predicted), dtype=np.float64)
        return _RunOutput(model_output=None, predictions=predictions)


# ── supporting helpers ────────────────────────────────────────────────────────


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


def _chosen_site_table(
    site_table: pd.DataFrame | None, parameter_vector: ParameterVector
) -> pd.DataFrame:
    """*site_table*, else the vector's own when it has ``lon``/``lat``, else the default one."""
    if site_table is not None:
        return site_table
    own = parameter_vector.site_table
    return own if {LON, LAT} <= set(own.columns) else load_sites()


def _run_sipnet_parameter_fields(
    sipnet_parameters: SIPNETParameters, names: Sequence[str], model_output: xr.Dataset
) -> xr.Dataset | None:
    """One run's SIPNET parameter fields: *names* from its own parameters, 0-d.

    Each variable is pySIPNET's ``SIPNETParameters.dataarray(name)``, labeled
    with the run's scalar ``site`` and its ``lon``/``lat`` from
    *model_output*; ``None`` when no parameter is read.
    """
    if not names:
        return None
    check_run_result_carries_its_parameters(sipnet_parameters)
    location = {name: model_output[name] for name in (SITE, LON, LAT) if name in model_output.coords}
    return xr.Dataset(
        {name: sipnet_parameters.dataarray(name) for name in names}, coords=location
    )


def _aggregated(model_output: xr.Dataset, freq: str) -> xr.Dataset:
    """Every variable of one run's output aggregated to *freq* by its kind's default.

    Through :func:`~sipnet_calibration.observation.aggregate_time`, the
    aggregation the plotting layer and the observation operators use, one
    variable at a time. The result's attributes are the run's, with
    pySIPNET's ``resampling_frequency`` and ``time_step_length_source`` as
    ``pysipnet.resample.resample`` sets them on a Dataset.
    """
    aggregated = [
        aggregate_time(model_output[name], freq).to_dataset() for name in model_output.data_vars
    ]
    # An exact join and identical coordinates: every variable of one run
    # shares one time axis, so any disagreement in the interval coordinates is
    # a defect to raise, never one to settle by taking the first. to_dataset()
    # leaves each variable's attributes on the variable, not the Dataset.
    merged = xr.merge(aggregated, join="exact", compat="identical", combine_attrs="drop_conflicts")
    merged.attrs = {
        **model_output.attrs,
        "time_step_length_source": STEP_LENGTH_RESAMPLED,
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
        check_site_slice_is_the_site_segment(
            slices[site], observation_vector.index[positions[site]], site
        )
    return slices, positions


def _sort_records(
    ensemble_result: Any, n_samples: int, sites: Sequence[int], batch_dim: str
) -> tuple[
    dict[tuple[int, int], _RunOutput], np.ndarray, pd.DataFrame, list[tuple[dict, BaseException]]
]:
    """Split PyEns's records into outputs, successes, model failures and machinery failures.

    Each record's coordinate is read for the *batch_dim* and ``site`` axes
    PyEns made from the SIPNET parameter fields' dims. The outputs are keyed
    ``(sample, site)``. Batch labels are row positions, which
    :func:`check_sipnet_parameter_fields_are_for_the_batch` guarantees.
    """
    column = {site: j for j, site in enumerate(sites)}
    run_outputs_by_sample_site: dict[tuple[int, int], _RunOutput] = {}
    rows: list[dict[str, Any]] = []
    machinery_failures: list[tuple[dict, BaseException]] = []
    for record in ensemble_result:
        sample, site = int(record.coordinate[batch_dim]), int(record.coordinate[SITE])
        if not record.failed:
            run_outputs_by_sample_site[(sample, site)] = record.output
        elif _is_model_failure(record.output):
            rows.append(
                {
                    batch_dim: sample,
                    SITE: site,
                    "error": _error_name(record.output),
                    "message": str(record.output)[:500],
                }
            )
        else:
            machinery_failures.append((record.coordinate, record.output))
    succeeded = np.zeros((n_samples, len(sites)), dtype=bool)
    for sample, site in run_outputs_by_sample_site:
        succeeded[sample, column[site]] = True
    failures = pd.DataFrame(rows, columns=[batch_dim, SITE, "error", "message"])
    return run_outputs_by_sample_site, succeeded, failures, machinery_failures


def _run_succeeded(
    succeeded: np.ndarray,
    labels: np.ndarray,
    sites: Sequence[int],
    *,
    batch_dim: str,
    site_locations: Mapping[str, xr.DataArray],
) -> xr.DataArray:
    """*succeeded*, ``(J, S)``, as a field on ``(batch_dim, site)``.

    *site_locations* is the ``lon``/``lat`` of *sites*, as
    :func:`sipnet_calibration.sites.site_locations` gives them.
    """
    return xr.DataArray(
        succeeded,
        dims=(batch_dim, SITE),
        coords={
            batch_dim: batch_coordinate(batch_dim, labels),
            SITE: np.asarray(sites, dtype=SITE_DTYPE),
            **site_locations,
        },
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
    run_outputs_by_sample_site: Mapping[tuple[int, int], _RunOutput],
    n_samples: int,
    sites: Sequence[int],
    *,
    batch_dim: str,
    site_table: pd.DataFrame,
    site_locations: dict[str, xr.DataArray],
) -> xr.Dataset:
    """The runs' output on the full ``(batch_dim, site, time)`` grid, ``NaN`` where a run failed."""
    model_outputs_by_sample_site = {
        key: run_output.model_output for key, run_output in run_outputs_by_sample_site.items()
    }
    stacked = stack_model_outputs(
        model_outputs_by_sample_site, key_dims=(batch_dim, SITE), site_table=site_table
    )
    full = stacked.reindex(
        {
            batch_dim: np.arange(n_samples, dtype=BATCH_LABEL_DTYPE),
            SITE: np.asarray(sites, dtype=stacked[SITE].dtype),
        }
    )
    # A site at which every run failed is absent from the stack, so the reindex
    # leaves its lon/lat NaN; they are the site table's whatever the runs did.
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
    site_table: pd.DataFrame | None,
) -> None:
    """Every check on :class:`ForwardModel`'s arguments that needs nothing computed."""
    sites = parameter_vector.sites
    check_model_is_a_sipnet_model(model)
    check_backend_is_a_pyens_backend(backend)
    check_output_variables_are_named(output_variable_names, observation_vector)
    check_freq_is_for_the_prior_predictive(freq, observation_vector)
    if freq is not None:
        check_frequency_is_an_offset_alias(freq)
    check_climate_covers_the_sites(climate, sites)
    check_climate_is_file_backed(climate, sites, backend)
    if observation_vector is not None:
        check_observation_sites_are_run(observation_vector, sites)


def check_base_parameters_set_what_the_operators_read(
    model: SIPNETModel, read: Sequence[str], written: Sequence[str]
) -> None:
    """The base parameter set holds a value for every SIPNET parameter the
    operators read and the SIPNET parameter fields do not write."""
    for name in read:
        if name in written:
            continue
        try:
            model.base_params.dataarray(name)
        except ValueError as error:
            raise ValueError(
                f"the observation operators read {name!r}, which the parameter vector does "
                f"not write and the model's base parameter set leaves unset ({error}); every "
                "run would fail after SIPNET ran. Set it in the base parameter set, or have "
                "the parameter vector fix or calibrate it."
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


def check_freq_is_for_the_prior_predictive(
    freq: Any, observation_vector: ObservationVector | None
) -> None:
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


def check_observation_sites_are_run(
    observation_vector: ObservationVector, sites: Sequence[int]
) -> None:
    extra = sorted(set(observation_vector.sites) - set(sites))
    if extra:
        raise ValueError(
            f"the observation vector observes site(s) {extra[:10]} that the parameter vector "
            "does not run; restrict the observation vector to the parameter vector's sites "
            "first (observation_vector.restrict_to_sites(parameter_vector.sites))."
        )


def check_output_variable_names_cover_the_operators(
    output_variable_names: Sequence[str], observation_vector: ObservationVector
) -> None:
    missing = [
        n for n in observation_vector.output_variable_names if n not in output_variable_names
    ]
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


def check_sipnet_parameter_fields_are_for_the_batch(
    sipnet_parameter_fields: Any,
    sites: Sequence[int],
    *,
    n_samples: int,
    batch_dim: str = SAMPLE,
) -> None:
    """*sipnet_parameter_fields* are a Dataset for a batch of *n_samples* over *sites*.

    Runs :func:`~sipnet_calibration.fields.check_sipnet_parameter_fields_are_a_dataset`,
    :func:`check_sipnet_parameter_fields_are_on_the_batch_dim_and_site`,
    :func:`check_sipnet_parameter_fields_batch_labels_are_the_rows_of_theta` and
    :func:`check_sipnet_parameter_fields_are_on_the_vectors_sites`, in that order.
    """
    check_sipnet_parameter_fields_are_a_dataset(
        sipnet_parameter_fields, "what to_sipnet_parameter_fields returns"
    )
    check_sipnet_parameter_fields_are_on_the_batch_dim_and_site(sipnet_parameter_fields, batch_dim)
    check_sipnet_parameter_fields_batch_labels_are_the_rows_of_theta(
        sipnet_parameter_fields, n_samples, batch_dim
    )
    check_sipnet_parameter_fields_are_on_the_vectors_sites(sipnet_parameter_fields, sites)


def check_sipnet_parameter_fields_are_on_the_vectors_sites(
    sipnet_parameter_fields: xr.Dataset, sites: Sequence[int]
) -> None:
    """The SIPNET parameter fields' sites are the parameter vector's, in its order."""
    if sipnet_parameter_fields[SITE].values.tolist() != list(sites):
        raise ValueError(
            "the SIPNET parameter fields' sites are not the parameter vector's sites, in "
            "order; keep their site dimension as parameter_vector.sipnet_parameter_fields "
            "gives it."
        )


def check_sipnet_parameter_fields_are_on_the_batch_dim_and_site(
    sipnet_parameter_fields: xr.Dataset, batch_dim: str
) -> None:
    """The SIPNET parameter fields are on exactly ``(batch_dim, site)``, every
    variable on both."""
    if set(sipnet_parameter_fields.dims) != {batch_dim, SITE}:
        raise ValueError(
            f"SIPNET parameter fields for a batch have dims exactly ({batch_dim}, site), got "
            f"{tuple(sipnet_parameter_fields.dims)}; reduce or select any other dimension in "
            f"the hook, and name the batch dim {batch_dim!r}, the model's batch_dim. One run "
            f"is one row of theta, so a batch crossed with a data source's ensemble (a "
            "sample and an initial_condition_member, say) is run by giving theta one row "
            "per combination, each sample's row repeated once per member, and selecting "
            "each row's member in the hook. (fields.stack_batch_dims(field, new_batch_dim='run') is "
            "how such crossed fields reach a vector's Flat; a run still needs one row of "
            "theta, which the hook maps to its own SIPNET parameters.)"
        )
    for name, variable in sipnet_parameter_fields.data_vars.items():
        if set(variable.dims) != {batch_dim, SITE}:
            raise ValueError(
                f"the SIPNET parameter fields' {name!r} is on {variable.dims}, but every "
                f"variable of SIPNET parameter fields is on both {batch_dim} and site, one "
                "value per run; broadcast it in the hook "
                "(sipnet_parameter_fields[name].broadcast_like(sipnet_parameter_fields))."
            )


def check_sipnet_parameter_fields_batch_labels_are_the_rows_of_theta(
    sipnet_parameter_fields: xr.Dataset, n_samples: int, batch_dim: str
) -> None:
    """The batch labels are ``0`` to ``J - 1``, in the order of theta's rows."""
    labels = sipnet_parameter_fields[batch_dim].values.tolist()
    dtype = sipnet_parameter_fields[batch_dim].dtype
    if dtype.kind not in "iu" or labels != list(range(n_samples)):
        raise ValueError(
            f"the SIPNET parameter fields' {batch_dim} labels must be the integers 0 to "
            f"{n_samples - 1}, in the order of theta's {n_samples} rows, got {labels[:10]} "
            f"({dtype}); the forward model places each run in the row of theta its label "
            "names."
        )


def check_sipnet_parameter_fields_set_the_parameters_built_for(
    sipnet_parameter_fields: xr.Dataset, expected_sipnet_parameter_names: Sequence[str]
) -> None:
    sipnet_parameter_names = {str(name) for name in sipnet_parameter_fields.data_vars}
    if sipnet_parameter_names != set(expected_sipnet_parameter_names):
        raise ValueError(
            f"the SIPNET parameter fields set {sorted(sipnet_parameter_names)}, but the model "
            f"was built for {sorted(expected_sipnet_parameter_names)}; a ForwardModel's free "
            "inputs are "
            "fixed when it is built, so the hook must set the same parameters every call."
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
    error_type_names = sorted(
        {getattr(e, "type_name", type(e).__name__) for _, e in machinery_failures}
    )
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
