"""ProbPipe-compatible likelihood base class for SIPNET experiments.

Note on the params argument in log_likelihood:
  RWMH in ProbPipe operates in the prior's flat (unconstrained) space.
  The `params` argument received by log_likelihood is a flat 1-D JAX array,
  NOT a ProbPipe Record. We unflatten it using prior.unflatten_value() so
  that parameter values can be accessed by name.

Design:
  SIPNETLikelihood is a thin base class that handles the ProbPipe protocol
  (unflatten + error handling). Subclasses implement _evaluate() with the
  actual physics/statistics. The prior is the sole source of truth for what
  parameters are being calibrated — subclasses derive everything from it.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from typing import Any

from probpipe import NumericRecord


class SIPNETLikelihood:
    """Minimal ProbPipe Likelihood base class for SIPNET experiments.

    The prior passed at construction is the authoritative definition of the
    calibrated parameter space. Fixed / structural parameters live in whatever
    SIPNETModel object is passed to a concrete subclass.

    Subclasses must override _evaluate().
    """

    def __init__(self, prior: Any) -> None:
        self.prior = prior

    def log_likelihood(self, params: Any, data: Any) -> float:
        """ProbPipe protocol entry point.

        Parameters
        ----------
        params:
            Flat 1-D JAX array (RWMH state in prior's unconstrained space).
        data:
            Observed timeseries passed through from condition_on().
        """
        try:
            named_params: NumericRecord = self.prior.unflatten_value(
                jnp.asarray(params)
            )
        except Exception:
            return -1e30
        try:
            return self._evaluate(named_params, data)
        except Exception:
            return -1e30

    def _evaluate(self, named_params: NumericRecord, data: Any) -> float:
        """Compute log-likelihood given unflattened parameter record and data.

        Parameters
        ----------
        named_params:
            NumericRecord with fields matching the prior. Access values via
            named_params["field_name"].
        data:
            Observed data as passed to log_likelihood.
        """
        raise NotImplementedError


class SingleSiteGaussianLikelihood(SIPNETLikelihood):
    """Gaussian observation likelihood for a single-site SIPNET run.

    Assumes iid Gaussian errors between modelled and observed output.

    Parameters
    ----------
    prior:
        ProbPipe ProductDistribution prior — defines calibrated parameters.
    model:
        Configured SIPNETModel instance (base params + climate already set).
        Called as model(**overrides) where overrides come from named_params.
    sigma_obs:
        Observation noise standard deviation.
    output:
        Which SIPNETResult method to call for modelled output.
        Must be a zero-argument method returning a pandas Series.
        Default "nee".
    """

    def __init__(
        self,
        prior: Any,
        model: Any,
        sigma_obs: float,
        output: str = "nee",
    ) -> None:
        super().__init__(prior)
        self.model = model
        self.sigma_obs = sigma_obs
        self.output = output

    def _evaluate(self, named_params: NumericRecord, data: Any) -> float:
        overrides = {
            name: float(named_params[name])
            for name in self.prior.record_template.fields
        }
        result = self.model(**overrides)
        predicted = getattr(result, self.output)().values
        obs = np.asarray(data)
        if len(predicted) != len(obs):
            return -1e30
        residuals = obs - predicted
        return float(-0.5 * np.sum(residuals**2) / self.sigma_obs**2)
