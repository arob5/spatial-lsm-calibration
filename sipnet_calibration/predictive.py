"""Prior and posterior predictive checks via PyEns parallel runs.

Note on ProbPipe sample access:
  prior._sample(key, (n,)) returns a NumericRecordArray.
  Fields are accessed via bracket notation: arr["field_name"] -> shape (n,) array.

Note on posterior samples:
  posterior_predictive() accepts a raw chains array rather than an
  ApproximateDistribution object, so it can be used after loading from NetCDF.
  Pass the concatenated draws as (n_chains, n_draws, n_params) or (n_total, n_params).
"""
from __future__ import annotations

import numpy as np
import jax
from typing import Any

from probpipe.custom_types import Array


def prior_predictive(
    prior: Any,
    model: Any,
    n_samples: int = 200,
    seed: int = 0,
    n_workers: int = 4,
    output_var: str = "nee",
) -> Array:
    """Sample from prior and run SIPNET in parallel via PyEns.

    Parameters
    ----------
    prior:
        ProbPipe ProductDistribution prior. Parameter names are derived from
        prior.record_template.fields — no separate list needed.
    model:
        SIPNETModel instance (used as callable by PyEns).
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

    param_names: list[str] = list(prior.record_template.fields)

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
    posterior_chains: Array,
    prior: Any,
    model: Any,
    n_samples: int = 200,
    seed: int = 1,
    n_workers: int = 4,
    output_var: str = "nee",
) -> Array:
    """Draw parameter samples from posterior chains and run SIPNET in parallel.

    Parameters
    ----------
    posterior_chains:
        Raw posterior draws as a numpy array. Shape may be either
        (n_chains, n_draws, n_params) or (n_total, n_params).
        Typically loaded from ArviZ NetCDF via az.from_netcdf().
    prior:
        ProbPipe ProductDistribution prior (used to unflatten draws).
        Parameter names are derived from prior.record_template.fields.
    model:
        SIPNETModel instance.
    n_samples:
        Number of posterior draws to use (randomly thinned if needed).
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

    param_names: list[str] = list(prior.record_template.fields)

    chains = np.asarray(posterior_chains)
    if chains.ndim == 3:
        # (n_chains, n_draws, n_params) -> (n_total, n_params)
        chains = chains.reshape(-1, chains.shape[-1])

    n_total = chains.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    selected = chains[indices]  # shape (n_samples, n_params)

    members = Axis("member", size=len(indices))
    param_arrays: dict[str, list[float]] = {name: [] for name in param_names}
    for i in range(len(indices)):
        named = prior.unflatten_value(selected[i])
        for name in param_names:
            param_arrays[name].append(float(named[name]))

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
