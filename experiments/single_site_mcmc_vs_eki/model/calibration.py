"""The calibration's model, and the posterior every algorithm reads.

The model of ``MODEL.md``,

.. math::

    \\theta \\sim \\pi, \\qquad m = \\mathcal H\\big(\\mathcal M(\\theta)\\big), \\qquad
    y_k \\mid m_k \\sim \\mathcal N(m_k, R_k) \\quad (k = 1, \\dots, K),

declared once, here, from three parts:

- the prior factors of the 13 parameters (``model/prior.py``);
- the forward map, a :class:`~sipnet_calibration.forward.SIPNETSimulator`
  named :data:`SIMULATOR_NAME`: SIPNET at the site, reduced by each
  source's operator to its prediction :math:`m_k`, invalid at a sample
  whose run fails or whose SIPNET parameters leave pySIPNET's domain;
- one noise factor per observation source (``model/noise.py``);

with the initial states nothing calibrates as inputs. ``joint`` declares
the graph, ``bind`` labels it at the site and the vector's observations,
and ``condition_on`` conditions it on the observed values, so that theta is
the 15 unconstrained entries of the parameters, in declaration order.

Functions
---------
:func:`model`
    The model over an observation vector, bound.
:func:`posterior`
    It conditioned on the vector's observed values, or on others.
:func:`calibration_posterior`, :func:`validation_posterior`
    The posterior on the calibration vector, and on the held-out tower's.
:func:`observation_vector`, :func:`predicted_fields`
    The vector a posterior's forward map predicts, and an evaluation's
    predictions as that vector's fields.

Notes
-----
Every posterior here declares the same parameters in the same order, so a
theta is a point of each: the validation posterior scores a calibration's
samples against the held-out tower, under that vector's own noise model.

Usage
-----
::

    from experiments.single_site_mcmc_vs_eki.model import calibration
    posterior = calibration.calibration_posterior()
    posterior.describe()                      # each component's role
    theta = posterior.sample_prior(key, 8)    # (8, 15)
    evaluation = posterior.evaluate(theta)    # one batch of SIPNET runs
    evaluation.log_likelihood, evaluation.valid
"""

from collections.abc import Mapping

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

from .. import config
from . import noise, observations, prior, sipnet

__all__ = [
    "SIMULATOR_NAME",
    "calibration_posterior",
    "model",
    "observation_vector",
    "posterior",
    "predicted_fields",
    "validation_posterior",
]

#: The name of the forward map, the model's one simulator.
SIMULATOR_NAME = "sipnet"


def model(observation_vector: ObservationVector) -> FactoredDistribution:
    """The joint model of the parameters and *observation_vector*, bound at the site."""
    runs = sipnet.sipnet_runs(prior.sipnet_parameter_map(), prior.site_dims())
    model_spec = joint(
        *prior.prior_factors(),
        SIPNETSimulator(runs, observation_vector=observation_vector, name=SIMULATOR_NAME),
        *noise.noise_factors(observation_vector),
        inputs=prior.input_specs(),
    )
    return model_spec.bind(
        coords={SITE: [config.SITE], **observation_vector.coords},
        inputs=prior.input_values(),
    )


def posterior(
    observation_vector: ObservationVector, observed_values: Mapping | None = None
) -> Posterior:
    """:func:`model` conditioned on *observed_values*.

    *observed_values*, ``{observation source name: values}`` on each source's
    observation dim, default to the vector's own
    (``observation_vector.observed_values_by_component()``); synthetic ones
    leave ``R`` as the vector's.
    """
    if observed_values is None:
        observed_values = observation_vector.observed_values_by_component()
    return condition_on(model(observation_vector), observed_values)


def calibration_posterior() -> Posterior:
    """The posterior on the calibration vector's observations."""
    return posterior(observations.calibration_observation_vector())


def validation_posterior() -> Posterior:
    """The posterior on the held-out tower's NEE, for scoring a calibration's samples."""
    return posterior(observations.validation_observation_vector())


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
