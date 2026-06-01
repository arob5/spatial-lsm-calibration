"""ProbPipe-compatible likelihood wrapper for SIPNET.

Note on the params argument in log_likelihood:
  RWMH in ProbPipe operates in the prior's flat (unconstrained) space.
  The `params` argument received by log_likelihood is a flat 1-D JAX array,
  NOT a ProbPipe Record. We unflatten it using the prior's record_template
  so that parameter values can be accessed by name.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SIPNETLikelihood:
    """Gaussian observation likelihood wrapping a SIPNETModel.

    Implements the ProbPipe Likelihood protocol:
        log_likelihood(params, data) -> float

    Parameters
    ----------
    model:
        A configured SIPNETModel instance. Called as model(**param_overrides).
    prior:
        The ProbPipe ProductDistribution prior. Used to unflatten the flat
        parameter array that RWMH passes to log_likelihood.
    param_names:
        List of SIPNET parameter leaf names to calibrate. Must match the
        field names in the prior (and must be valid SIPNETModel kwargs).
    sigma_obs:
        Observation noise standard deviation (gC m-2 day-1 for NEE).
    output_var:
        Which SIPNETResult method to call for the modelled output.
        Default "nee". Must be a zero-argument method returning pd.Series.
    """

    model: Any           # SIPNETModel
    prior: Any           # ProbPipe ProductDistribution — for unflatten
    param_names: list[str]
    sigma_obs: float
    output_var: str = "nee"

    def log_likelihood(self, params: Any, data: Any) -> float:
        """Evaluate Gaussian log-likelihood.

        Parameters
        ----------
        params:
            Flat 1-D JAX array (RWMH state in prior's unconstrained space).
            We unflatten it via prior.unflatten_value() to get a NumericRecord
            with fields accessible via bracket notation: record["a_max"].
        data:
            Observed output timeseries as a JAX or numpy 1-D array.
        """
        # Unflatten flat array -> NumericRecord with named fields
        try:
            param_record = self.prior.unflatten_value(jnp.asarray(params))
        except Exception:
            return -1e30

        # Build override dict: flat param-name -> Python float
        overrides = {}
        for name in self.param_names:
            try:
                overrides[name] = float(param_record[name])
            except Exception:
                return -1e30

        try:
            result = self.model(**overrides)
        except Exception:
            return -1e30

        predicted = getattr(result, self.output_var)().values
        obs = np.asarray(data)
        if len(predicted) != len(obs):
            return -1e30

        residuals = obs - predicted
        return float(-0.5 * np.sum(residuals ** 2) / self.sigma_obs ** 2)
