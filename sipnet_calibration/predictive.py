"""Prior and posterior predictive checks via PyEns parallel runs.

Note on ProbPipe sample access:
  prior._sample(key, (n,)) returns a NumericRecordArray.
  Fields are accessed via bracket notation: arr["field_name"] -> shape (n,) array.
  This differs from PLAN.md's getattr pattern; bracket notation is the correct API.

Note on posterior samples:
  ApproximateDistribution.draws() returns the raw flat chain arrays (shape: n_draws x n_params).
  We unflatten using prior.unflatten_value() to get named fields.
"""
from __future__ import annotations

import numpy as np
import jax
from typing import Any


def prior_predictive(
    prior: Any,
    model: Any,
    param_names: list[str],
    n_samples: int = 200,
    seed: int = 0,
    n_workers: int = 4,
    output_var: str = "nee",
) -> np.ndarray:
    """Sample from prior and run SIPNET in parallel via PyEns.

    Parameters
    ----------
    prior:
        ProbPipe ProductDistribution prior.
    model:
        SIPNETModel instance (used as callable by PyEns).
    param_names:
        List of parameter names matching prior fields.
    n_samples:
        Number of prior samples to draw.
    seed:
        JAX PRNG seed.
    n_workers:
        Number of parallel workers for LocalBackend.
    output_var:
        Which SIPNETResult method to collect (default "nee").

    Returns
    -------
    np.ndarray of shape (n_succeeded, n_timesteps)
    """
    from pyens import EnsembleRunner, EnsembleSpec, Axis
    from pyens.backends import LocalBackend
    from pysipnet.ensemble import sipnet_member_fields

    # Sample parameter vectors from ProbPipe prior
    # Returns NumericRecordArray; fields accessed via ["field_name"]
    key = jax.random.PRNGKey(seed)
    samples = prior._sample(key, (n_samples,))

    members = Axis("member", size=n_samples)
    param_arrays = {
        name: [float(samples[name][i]) for i in range(n_samples)]
        for name in param_names
    }
    spec = EnsembleSpec(inputs=sipnet_member_fields(members, **param_arrays))

    runner = EnsembleRunner(model, LocalBackend(n_workers=n_workers))
    result = runner.run(spec)

    if result.n_failed > 0:
        print(f"Warning: {result.n_failed}/{n_samples} prior predictive runs failed.")

    timeseries = [
        getattr(rec.output, output_var)().values
        for rec in result.succeeded
    ]
    return np.stack(timeseries)  # shape: (n_succeeded, n_timesteps)


def posterior_predictive(
    posterior: Any,
    prior: Any,
    model: Any,
    param_names: list[str],
    n_samples: int = 200,
    seed: int = 1,
    n_workers: int = 4,
    output_var: str = "nee",
) -> np.ndarray:
    """Draw parameter samples from posterior and run SIPNET in parallel.

    Parameters
    ----------
    posterior:
        ProbPipe ApproximateDistribution returned by condition_on().
    prior:
        ProbPipe ProductDistribution prior (used to unflatten posterior draws).
    model:
        SIPNETModel instance.
    param_names:
        List of parameter names matching prior/posterior fields.
    n_samples:
        Number of posterior draws to use (thinned from all draws if needed).
    seed:
        Random seed for draw selection.
    n_workers:
        Number of parallel workers.
    output_var:
        Which SIPNETResult method to collect.

    Returns
    -------
    np.ndarray of shape (n_succeeded, n_timesteps)
    """
    from pyens import EnsembleRunner, EnsembleSpec, Axis
    from pyens.backends import LocalBackend
    from pysipnet.ensemble import sipnet_member_fields

    # posterior.chains is a list of flat arrays, each shape (n_draws, n_params).
    # (posterior.draws() returns NumericRecordArray — chains gives the raw flat arrays
    # that prior.unflatten_value() expects.)
    all_draws = np.concatenate([np.array(c) for c in posterior.chains], axis=0)
    n_total = all_draws.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    selected = all_draws[indices]  # shape (n_samples, n_params)

    # Unflatten each draw using the prior's template
    members = Axis("member", size=len(indices))
    param_arrays: dict[str, list[float]] = {name: [] for name in param_names}
    for i in range(len(indices)):
        rec = prior.unflatten_value(selected[i])
        for name in param_names:
            param_arrays[name].append(float(rec[name]))

    spec = EnsembleSpec(inputs=sipnet_member_fields(members, **param_arrays))
    runner = EnsembleRunner(model, LocalBackend(n_workers=n_workers))
    result = runner.run(spec)

    if result.n_failed > 0:
        print(f"Warning: {result.n_failed}/{len(indices)} posterior predictive runs failed.")

    timeseries = [
        getattr(rec.output, output_var)().values
        for rec in result.succeeded
    ]
    return np.stack(timeseries)
