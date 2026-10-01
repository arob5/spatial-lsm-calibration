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
:func:`prior_gaussian`
    The prior as a pyEKI Gaussian over theta, which EKI starts from.

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
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from pyeki.gauss import Gaussian
from pyeki.linalg import PSDBlockDiag, PSDDiagonal, PSDLinOp

from sipnet_calibration.forward import ForwardModel
from sipnet_calibration.observation import ObservationVector
from sipnet_calibration.parameters import ParameterVector, Prior
from sipnet_calibration.sipnet_parameter_map import SIPNETParameterMap
from sipnet_calibration.site_dims import SiteDims

from .. import config
from . import noise, observations, prior, sipnet

__all__ = [
    "InverseProblem",
    "calibration_problem",
    "check_prior_gaussian_has_the_prior_density",
    "check_prior_terms_are_contiguous",
    "prior_gaussian",
]

#: How far the Gaussian's log density may be from the prior's at a draw.
LOG_DENSITY_TOLERANCE = 1e-9

#: The number of prior draws :func:`prior_gaussian` is checked at, and their seed.
N_CHECK_DRAWS = 64
CHECK_SEED = 0


@dataclass(frozen=True, eq=False, kw_only=True)
class InverseProblem:
    """The calibration's prior, forward model and data.

    Parameters
    ----------
    parameter_vector, prior, sipnet_parameter_map
        The three calibration objects of ``model/prior.py``.
    site_dims
        The site they are read at (``prior.site_dims``).
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
    site_dims: SiteDims
    external_inputs: xr.Dataset
    observation_vector: ObservationVector
    y: jax.Array
    noise_covariance: PSDLinOp

    @cached_property
    def prior_gaussian(self) -> Gaussian:
        """The prior over theta, ``N(m_0, C_0)``, exact."""
        return prior_gaussian(self.prior)

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
            site_dims=self.site_dims,
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
        site_dims=prior.site_dims(),
        external_inputs=prior.external_inputs(),
        observation_vector=calibration,
        y=jnp.asarray(calibration.y),
        noise_covariance=noise.noise_covariance(
            calibration, config.CALIBRATION_NEE_SERIES
        ),
    )


def prior_gaussian(calibration_prior: Prior) -> Gaussian:
    """The prior over theta as a pyEKI Gaussian, ``N(m_0, C_0)``.

    Each term's distribution in theta is its base: the Gaussian a log-normal,
    logit-normal or softmax-normal pushes through its bijector, or the term's
    own Normal where the bijector is the identity. ``C_0`` is block-diagonal,
    one diagonal block per term, in theta's order.

    Raises
    ------
    ValueError
        If the terms' entries do not tile theta in term order, or the
        Gaussian's log density is not the prior's at its draws, as when a
        term is not Gaussian in theta.

    Notes
    -----
    The library's adapter went with ``Prior.gaussian()``. This one builds
    what that method returned for this prior, block for block, so the
    initial ensemble ``EKIState.from_prior`` draws is unchanged.
    """
    unconstrained = calibration_prior.parameter_vector.unconstrained
    positions = [
        unconstrained.positions(parameter=list(term.parameter_names))
        for term in calibration_prior.terms
    ]
    check_prior_terms_are_contiguous(positions, unconstrained.size)
    bases = [_base_distribution(term.distribution) for term in calibration_prior.terms]
    mean = jnp.concatenate([jnp.reshape(base.mean(), (-1,)) for base in bases])
    covariance = PSDBlockDiag(
        tuple(PSDDiagonal(jnp.reshape(base.variance(), (-1,))) for base in bases)
    )
    gaussian = Gaussian(mean, covariance)
    check_prior_gaussian_has_the_prior_density(gaussian, calibration_prior)
    return gaussian


# ── helpers ──


def _base_distribution(distribution):
    """The distribution over theta that a term's distribution pushes forward."""
    if isinstance(
        distribution,
        tfp.distributions.LogNormal
        | tfp.distributions.LogitNormal
        | tfp.distributions.TransformedDistribution,
    ):
        return distribution.distribution
    return distribution


# ── checks ──


def check_prior_terms_are_contiguous(positions: list[np.ndarray], size: int) -> None:
    """Check the terms' entries tile theta in term order, one run each."""
    if not np.array_equal(np.concatenate(positions), np.arange(size)):
        raise ValueError(
            "the prior's terms do not tile theta in term order; declare the "
            "parameters in the order of their terms"
        )


def check_prior_gaussian_has_the_prior_density(
    gaussian: Gaussian, calibration_prior: Prior
) -> None:
    """Check the Gaussian's log density is the prior's at the prior's draws."""
    theta = calibration_prior.sample(jax.random.key(CHECK_SEED), N_CHECK_DRAWS)
    difference = float(
        jnp.max(
            jnp.abs(gaussian.log_density(theta) - calibration_prior.log_prob(theta))
        )
    )
    if not difference <= LOG_DENSITY_TOLERANCE:
        raise ValueError(
            f"the Gaussian's log density differs from the prior's by up to "
            f"{difference:.3g} at its draws; a term is not Gaussian in theta, so "
            "EKI cannot start from this prior's Gaussian"
        )
