"""ProbPipe-compatible likelihood base class for SIPNET experiments.

Calling convention (RWMH)
-------------------------
ProbPipe's RWMH sampler (``probpipe.inference._rwmh``) is a **pure Python
for-loop** — it does not use ``jax.jit``, ``jax.vmap``, or any other JAX
transformation. At each MCMC step it calls::

    float(target_log_prob(params))

where ``target_log_prob(params) = prior._log_prob(params)
                                   + likelihood.log_likelihood(params, data)``.

Consequences for subclass authors:
  * ``params`` is always a **1-D flat** ``jnp.ndarray`` of shape ``(n_params,)``
    — one draw, never a batched array.
  * ``log_likelihood`` must return a Python ``float`` (or something that
    ``float()`` accepts). Returning a JAX scalar is fine.
  * Python side-effects (file I/O, subprocess calls like SIPNET) are safe.
  * No autodiff is required; RWMH is gradient-free.

Parameter unflattening
----------------------
RWMH operates in the prior's **flat unconstrained space** — the parameter
vector it proposes has no names. ``SIPNETLikelihood`` reconstructs a named
``NumericRecord`` from the flat vector via ``prior.unflatten_value(params)``
so that subclasses can access parameters by name.

The ``prior`` argument must be a ``RecordDistribution`` (e.g.
``ProductDistribution``) — specifically it must expose:

  * ``unflatten_value(flat: Array) -> NumericRecord``
  * ``record_template.fields`` — ordered collection of parameter names

``ProductDistribution`` satisfies both.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from probpipe import NumericRecord
from probpipe.core._record_distribution import RecordDistribution
from probpipe.custom_types import Array, ArrayLike

if TYPE_CHECKING:
    pass  # future imports go here


class SIPNETLikelihood:
    """Minimal ProbPipe Likelihood base class for SIPNET experiments.

    The ``prior`` passed at construction is the **sole source of truth** for
    the calibrated parameter space — it defines both which parameters vary and
    their names. Fixed / structural parameters live in whatever
    ``SIPNETModel`` is passed to a concrete subclass.

    Parameters
    ----------
    prior:
        A ``RecordDistribution`` (e.g. ``ProductDistribution``) with named
        fields. Must implement ``unflatten_value`` and expose
        ``record_template.fields``.

    Subclassing
    -----------
    Override ``_evaluate(named_params, data) -> float``. The base class
    handles unflattening and catches all exceptions (returning ``-1e30`` so
    MCMC continues). Subclass ``_evaluate`` should raise on invalid inputs
    rather than returning ``-1e30`` — the base class converts the exception.
    """

    def __init__(self, prior: RecordDistribution) -> None:
        self.prior = prior

    def log_likelihood(self, params: ArrayLike, data: ArrayLike) -> float:
        """ProbPipe Likelihood protocol entry point.

        Called once per MCMC step with a single flat parameter vector.
        Never called under ``jax.jit`` or ``jax.vmap`` by RWMH.

        Parameters
        ----------
        params:
            1-D flat ``jnp.ndarray`` of shape ``(n_params,)`` in the prior's
            unconstrained space. Unflattened to a ``NumericRecord`` before
            being passed to ``_evaluate``.
        data:
            Observed data as passed to ``condition_on``. May be a numpy
            array, JAX array, or any type that ``_evaluate`` understands.
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

    def _evaluate(self, named_params: NumericRecord, data: ArrayLike) -> float:
        """Compute log-likelihood given unflattened parameter record and data.

        Called by ``log_likelihood`` after unflattening. Subclasses should
        raise on invalid inputs; exceptions are caught by the base class and
        mapped to ``-1e30``.

        Parameters
        ----------
        named_params:
            ``NumericRecord`` with one field per calibrated parameter.
            Access values via ``named_params["field_name"]``.
        data:
            Observed data as passed to ``log_likelihood``.
        """
        raise NotImplementedError


class SingleSiteGaussianLikelihood(SIPNETLikelihood):
    """iid Gaussian observation likelihood for a single-site SIPNET run.

    Assumes the residuals ``obs - model(params)`` are iid
    ``N(0, sigma_obs²)`` at every timestep. The resulting log-likelihood is::

        -0.5 * sum((obs - predicted)² / sigma_obs²)

    In practice ``sigma_obs`` should be set to reflect both measurement noise
    *and* model structural error (temporal autocorrelation, missing processes).
    For the 8760-timestep sub-daily setting a value of ``sigma_obs = 2.0``
    gC m⁻² per 3-hr step gives ~23% RWMH acceptance with ``step_size = 0.01``.

    Parameters
    ----------
    prior:
        ``RecordDistribution`` defining calibrated parameters.
    model:
        Configured ``SIPNETModel`` instance (base params + climate already
        set). Called as ``model(**overrides)`` where overrides are derived
        from ``named_params``.
    sigma_obs:
        Observation noise standard deviation (same units as model output).
    output:
        ``SIPNETResult`` method name to call for modelled output. Must be a
        zero-argument method returning a ``pandas.Series``. Default ``"nee"``.
    """

    def __init__(
        self,
        prior: RecordDistribution,
        model: object,
        sigma_obs: float,
        output: str = "nee",
    ) -> None:
        super().__init__(prior)
        self.model = model
        self.sigma_obs = sigma_obs
        self.output = output

    def _evaluate(self, named_params: NumericRecord, data: ArrayLike) -> float:
        overrides = {
            name: float(named_params[name])
            for name in self.prior.record_template.fields
        }
        result = self.model(**overrides)
        predicted = getattr(result, self.output)().values
        obs = np.asarray(data)
        if predicted.shape != obs.shape:
            raise ValueError(
                f"Shape mismatch between model output {predicted.shape} "
                f"and observations {obs.shape}. Check that the climate file "
                f"and obs array were generated from the same setup_data() call."
            )
        residuals = obs - predicted
        return float(-0.5 * np.sum(residuals**2) / self.sigma_obs**2)
