"""The experiment's models, and the posteriors every algorithm reads.

A model is an NEE error model crossed with a noise treatment (``MODEL.md``,
"The models"):

.. math::

    \\theta \\sim \\pi_0, \\qquad
    y_k \\mid \\theta, s \\sim \\mathcal N\\big(\\mathcal G_k(\\theta),\\ s_k C^L_k\\big),
    \\qquad s_k \\sim \\mathrm{IG}(a, b),\\ \\operatorname{median} s_k = 1,

- the **NEE error model** ``L`` (``model/nee_error.py``) sets the memory of
  NEE's discrepancy in :math:`C^L_k`;
- the **noise treatment** is ``fixed``, every scale held at its prior median
  1, or ``inferred``, the scales unknown.

Every model shares the prior on the 13 parameters (``model/prior.py``), the
forward map (SIPNET at the site, reduced by each source's operator) and the
noise of the unscaled sources.

Functions
---------
:class:`Model`, :data:`MODEL_NAMES`
    A model, named ``"<nee error model>/<noise>"``.
:func:`joint_model`
    The model's graph over an observation vector, bound at the site.
:func:`fixed_posterior`
    Conditioned on the observations with every scale held at 1: what EKI
    reads, and the posterior of a model with fixed noise.
:func:`heldout_posterior`
    The same on the held-out tower's NEE, for scoring a run's samples.
:func:`marginal_posterior`
    With the scales integrated out (Student-t noise), for checking
    ``model/likelihood.py``'s closed form.
:func:`observation_vector`, :func:`predicted_fields`
    The vector a posterior's forward map predicts, and an evaluation's
    predictions as its fields.

Notes
-----
Every posterior here declares the same parameters in the same order, so a
theta is a point of each. Binding and conditioning the model takes about
30 s, so each posterior is built once per process.

Usage
-----
::

    from experiments.single_site_mcmc_vs_eki import models
    model = models.Model.parse("long_memory/inferred")
    posterior = models.fixed_posterior(model)
    posterior.describe()
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sipnet_calibration.conventions import SAMPLE, SITE
from sipnet_calibration.fields import Field
from sipnet_calibration.forward import SIPNETSimulator
from sipnet_calibration.observation import ObservationVector
from sipnet_calibration.observation.model import prediction_components
from sipnet_calibration.probability import (
    FactoredDistribution,
    Layout,
    Posterior,
    PosteriorEvaluation,
    condition_on,
    joint,
)

from . import config
from .model import noise, observations, prior, sipnet
from .model.nee_error import NEE_ERROR_MODELS

__all__ = [
    "MODEL_NAMES",
    "NEE_ERROR_MODELS",
    "NOISE_TREATMENTS",
    "SIMULATOR_NAME",
    "Model",
    "fixed_posterior",
    "heldout_posterior",
    "joint_model",
    "marginal_posterior",
    "observation_vector",
    "predicted_fields",
]

#: The noise treatments: the scales held at their prior median, or unknown.
NOISE_TREATMENTS = ("fixed", "inferred")

#: The name of the forward map, the model's one simulator.
SIMULATOR_NAME = "sipnet"


@dataclass(frozen=True)
class Model:
    """An NEE error model crossed with a noise treatment."""

    nee_error: str
    noise: str

    def __post_init__(self) -> None:
        if self.nee_error not in NEE_ERROR_MODELS or self.noise not in NOISE_TREATMENTS:
            raise KeyError(
                f"no model {self.name!r}; the models are {', '.join(MODEL_NAMES)}"
            )

    @property
    def name(self) -> str:
        """``"<nee error model>/<noise>"``."""
        return f"{self.nee_error}/{self.noise}"

    @classmethod
    def parse(cls, name: str) -> "Model":
        """The model called *name*, ``"<nee error model>/<noise>"``."""
        nee_error, _, noise_treatment = name.partition("/")
        return cls(nee_error, noise_treatment)

    def directory(self, algorithm: str) -> Path:
        """Where this model's run of *algorithm* writes."""
        return config.RUNS_DIRECTORY / self.nee_error / self.noise / algorithm


#: Every model's name.
MODEL_NAMES = tuple(
    f"{nee_error}/{noise_treatment}"
    for nee_error in NEE_ERROR_MODELS
    for noise_treatment in NOISE_TREATMENTS
)


def joint_model(
    nee_error_model_name: str, observation_vector: ObservationVector
) -> FactoredDistribution:
    """The prior, the forward map, the noise factors and the scales' priors
    over *observation_vector*, bound at the site."""
    runs = sipnet.sipnet_runs(prior.sipnet_parameter_map(), prior.site_dims())
    model_spec = joint(
        *prior.prior_factors(),
        SIPNETSimulator(runs, observation_vector=observation_vector, name=SIMULATOR_NAME),
        *noise.noise_factors(observation_vector, nee_error_model_name),
        *noise.noise_scale_factors(observation_vector),
        inputs=prior.input_specs(),
    )
    return model_spec.bind(
        coords={SITE: [config.SITE], **observation_vector.coords},
        inputs=prior.input_values(),
    )


def fixed_posterior(
    model: Model, observed_values: Mapping | None = None
) -> Posterior:
    """*model* conditioned on the calibration vector's observations, or on
    *observed_values*, with every noise scale held at 1.

    The posterior of a model with fixed noise, and the one whose forward map
    and :math:`C_k` a model with inferred noise reads: it depends on the NEE
    error model alone.
    """
    if observed_values is None:
        return _fixed_posterior(model.nee_error, "calibration")
    vector = observations.calibration_observation_vector()
    return condition_on(
        _joint_model(model.nee_error, "calibration"),
        {**observed_values, **_scales_at_one(vector)},
    )


def heldout_posterior(model: Model) -> Posterior:
    """:func:`fixed_posterior` on the held-out tower's NEE (US-xHA, 2021-2024)."""
    return _fixed_posterior(model.nee_error, "heldout")


def marginal_posterior(model: Model) -> Posterior:
    """*model* with its noise scales integrated out, conditioned on the
    calibration vector's observations: each scaled source's noise a
    Student-t, :math:`t_{2a}(\\mathcal G_k, (b/a) C_k)`."""
    vector = observations.calibration_observation_vector()
    marginal = _joint_model(model.nee_error, "calibration").marginalize(
        list(_scales_at_one(vector))
    )
    return condition_on(marginal, vector.observed_values_by_component())


def observation_vector(posterior: Posterior) -> ObservationVector:
    """The observation vector *posterior*'s forward map predicts."""
    return posterior.simulators[SIMULATOR_NAME].observation_vector


def predicted_fields(
    posterior: Posterior, evaluation: PosteriorEvaluation
) -> dict[str, Field]:
    """*evaluation*'s predictions as fields of :func:`observation_vector`, by source.

    Each on ``(sample, site[, time])``, the source's grid, ``NaN`` at an
    invalid sample.
    """
    vector = observation_vector(posterior)
    source_of = {
        vector.prediction_name(name): name for name in vector.observation_source_names
    }
    layout = Layout(prediction_components(vector), coords=vector.coords)
    labeled = layout.values_to_labeled(
        {name: evaluation.values[name] for name in source_of}, batch_dims=(SAMPLE,)
    )
    return {
        source_of[name]: field for name, field in vector.to_fields(labeled).items()
    }


# ── helpers ──

_VECTORS = {
    "calibration": observations.calibration_observation_vector,
    "heldout": observations.validation_observation_vector,
}


@functools.cache
def _joint_model(nee_error_model_name: str, vector_name: str) -> FactoredDistribution:
    """:func:`joint_model` over the named vector, built once."""
    return joint_model(nee_error_model_name, _VECTORS[vector_name]())


@functools.cache
def _fixed_posterior(nee_error_model_name: str, vector_name: str) -> Posterior:
    """The named vector's posterior with every scale held at 1, built once."""
    vector = _VECTORS[vector_name]()
    return condition_on(
        _joint_model(nee_error_model_name, vector_name),
        {**vector.observed_values_by_component(), **_scales_at_one(vector)},
    )


def _scales_at_one(vector: ObservationVector) -> dict[str, float]:
    """Every noise scale *vector*'s sources read, at 1."""
    return {
        noise.noise_scale_name(name): 1.0
        for name in vector.observation_source_names
        if name in config.NOISE_SCALED_SOURCES
    }
