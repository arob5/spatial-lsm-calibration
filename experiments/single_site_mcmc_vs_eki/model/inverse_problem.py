"""The calibration's inverse problem: what every inference algorithm conditions on.

One object holds the whole problem, so EKI and MCMC condition on the same
thing:

.. math::

    y = G(\\theta) + \\varepsilon, \\qquad \\varepsilon \\sim \\mathcal N(0, R),
    \\qquad \\theta \\sim \\mathcal N(m_0, C_0),

with :math:`\\theta` the 15 unconstrained entries of the calibration's
parameter vector, :math:`G` the forward model over the calibration
observation vector, and :math:`R` its noise covariance (``MODEL.md``). The
prior is exactly Gaussian in :math:`\\theta`: every prior term of
``model/prior.py`` is a Gaussian pushed through its parameter's bijector.

Functions
---------
:func:`calibration_problem`
    The problem on the observed calibration vector.
:class:`InverseProblem`
    Its pieces, the forward model over them, and
    :meth:`~InverseProblem.with_observations` for synthetic data.

Usage
-----
::

    from experiments.single_site_mcmc_vs_eki.model import inverse_problem
    problem = inverse_problem.calibration_problem()
    forward = problem.forward_model()     # theta (J, D) -> predictions (J, N)
    problem.likelihood.log_density(forward(theta))
"""

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal

import jax
import jax.numpy as jnp
import xarray as xr
from pyeki.gauss import Gaussian
from pyeki.linalg import PSDLinOp

from sipnet_calibration.forward import ForwardModel
from sipnet_calibration.observation import ObservationVector
from sipnet_calibration.parameter_vector import ParameterVector
from sipnet_calibration.prior import Prior
from sipnet_calibration.sipnet_parameter_map import SIPNETParameterMap

from .. import config
from . import noise, observations, prior, sipnet

__all__ = ["InverseProblem", "calibration_problem"]


@dataclass(frozen=True, eq=False, kw_only=True)
class InverseProblem:
    """The calibration's prior, forward model and data.

    Parameters
    ----------
    parameter_vector, prior, sipnet_parameter_map
        The three calibration objects of ``model/prior.py``.
    external_inputs
        The initial states the map reads and nothing calibrates.
    observation_vector
        What is predicted: its sources and operators.
    y
        The observations conditioned on, Flat ``(N,)``: the observation
        vector's own, or synthetic ones (:meth:`with_observations`).
    noise_covariance
        ``R``, ``(N, N)``, as a pyEKI operator.
    """

    parameter_vector: ParameterVector
    prior: Prior
    sipnet_parameter_map: SIPNETParameterMap
    external_inputs: xr.Dataset
    observation_vector: ObservationVector
    y: jax.Array
    noise_covariance: PSDLinOp

    @cached_property
    def prior_gaussian(self) -> Gaussian:
        """The prior over theta, ``N(m_0, C_0)``, exact."""
        return self.prior.gaussian()

    @property
    def likelihood(self) -> Gaussian:
        """``N(y, R)``, whose ``log_density(predictions)`` is the log likelihood."""
        return Gaussian(self.y, self.noise_covariance)

    def forward_model(
        self, *, out_of_domain: Literal["raise", "fail_row"] = "raise"
    ) -> ForwardModel:
        """``G``: theta ``(J, D)`` to predictions ``(J, N)``, NaN rows for failed runs."""
        return sipnet.forward_model(
            self.parameter_vector,
            self.sipnet_parameter_map,
            observation_vector=self.observation_vector,
            external_inputs=self.external_inputs,
            out_of_domain=out_of_domain,
        )

    def with_observations(self, y) -> "InverseProblem":
        """The same problem conditioned on other observations *y*, Flat ``(N,)``."""
        return replace(self, y=jnp.asarray(y, dtype=jnp.float64))


def calibration_problem() -> InverseProblem:
    """The problem on the observed calibration vector and its noise model."""
    vector, calibration_prior, sipnet_map = prior.calibration()
    calibration = observations.calibration_observation_vector()
    return InverseProblem(
        parameter_vector=vector,
        prior=calibration_prior,
        sipnet_parameter_map=sipnet_map,
        external_inputs=prior.external_inputs(),
        observation_vector=calibration,
        y=jnp.asarray(calibration.y),
        noise_covariance=noise.noise_covariance(
            calibration, config.CALIBRATION_NEE_SERIES
        ),
    )
