# Implementation Plan: SIPNET Bayesian Calibration Demo

**Repo:** `spatial-lsm-calibration`  
**Date drafted:** 2026-05-31  
**Status:** Ready for implementation

---

## Scope Constraint

**All code changes go in this repo (`spatial-lsm-calibration`) only.**  
Do NOT modify `pySIPNET`, `ProbPipe`, or `PyEns`. If a limitation is discovered in one of those packages, note it in a comment and work around it here.

---

## Goal

A clean, well-documented demonstration of Bayesian parameter estimation for the SIPNET land-surface model using:
- **ProbPipe** (`condition_on`) for prior specification, MCMC (RWMH), and posterior inference
- **PyEns** for parallel batch execution of SIPNET (prior predictive and posterior predictive checks)
- **pySIPNET** as the physical model interface

The demo proceeds in three notebooks:
1. Timing benchmark → choose simulation time horizon
2. Prior predictive check → validate prior choices
3. MCMC calibration → synthetic data, RWMH posterior, diagnostics

---

## Package Management

Use **`uv`** throughout. Initialize the project with `uv init`, add dependencies with `uv add`, and run notebooks/scripts with `uv run`.

---

## Confirmed Facts About the Codebases

These have been verified by reading source code — do not re-derive them.

### pySIPNET

- **Public run interface:** `SIPNETModel(**param_overrides) -> SIPNETResult`
  - `param_overrides` keys are **flat leaf names** (e.g., `a_max`, `psn_t_opt`) — NOT dot-paths, NOT camelCase
  - Validated from `SIPNET_PARAMS_BY_GROUP` — pass an unrecognized name and it raises `ValueError` immediately
  - `climate=` and `events=` are reserved kwargs; everything else is treated as a parameter override
- **`_apply_overrides` is a private module-level function** — never call it directly from this repo; always go through `model(**kwargs)`
- **Climate IO optimization:** Use `ClimateDrivers.from_path(path)` (file-backed, lazy) combined with `SIPNETRunner(climate_staging=ClimateStaging.SYMLINK)` — the climate file is symlinked into each SIPNET temp dir rather than copied, eliminating per-run IO cost
- **Output access:** `result.nee()` returns a `pd.Series` of daily NEE (gC m⁻² day⁻¹)
- **Parameters with `ParameterDomain` metadata:** Import `get_parameter_specs` from `pysipnet.parameters.base` to read unit/domain info if needed

### ProbPipe

- **`condition_on(model, data, method="rwmh", **kwargs) -> ApproximateDistribution`**
- **RWMH is a Python-level for-loop** (`for t in range(chain_total):`), NOT JAX-traced. The log_prob is called as `float(target_log_prob(mu_curr))`. This means a subprocess call inside `log_likelihood` works without any `jax.pure_callback` workaround.
- **`SimpleModel(prior, likelihood)`** computes joint log-prob as `prior._log_prob(params) + likelihood.log_likelihood(params, data)`
- **`SimpleModel._log_prob((params, data))`** calls `likelihood.log_likelihood(params=params, data=data)` — `params` arrives with whatever type the prior produces
- **If prior is `Record(a_max=LogNormal(...), ...)`**, then `params` inside `log_likelihood` is a **ProbPipe `Record` instance** with named fields accessible via attribute access (`params.a_max`) or iteration
- **ProbPipe handles constrained sampling automatically** via bijectors — `LogNormal` priors cause RWMH to propose in log-space; the likelihood receives already-valid (positive) parameter values. No manual Jacobian correction needed.
- **`ApproximateDistribution` output:** `.draws()`, `.mean()`, `.cov()`, `.inference_data` (ArviZ-compatible DataTree)
- **Useful kwargs for `condition_on`:** `num_results`, `num_warmup`, `step_size`, `random_seed`, `num_chains`

### PyEns

- **`EnsembleRunner(model_callable, backend).run(spec) -> EnsembleResult`**
- `model_callable` is any `(**inputs) -> output` callable — `SIPNETModel` works directly, no adapter needed
- **`EnsembleSpec(inputs={"param_name": FieldSpec, ...})`**
- **`Axis("member", size=N)`** and **`Grid(values, along=axis)`** for parameter arrays
- **`sipnet_member_fields(members_axis, **{param_name: list_of_values})`** from `pysipnet.ensemble` — helper that creates Grid fields aligned on the member axis
- **`LocalBackend(n_workers=N)`** uses `ProcessPoolExecutor` — `SIPNETModel` must be defined at module level (not a lambda) for pickling to work
- **`EnsembleResult.succeeded`** — list of `RunRecord` objects that didn't fail; access output via `record.output`
- **Exception handling:** Failed runs are stored as data, not raised — check `result.failed` after each batch

---

## File Structure

```
spatial-lsm-calibration/
├── README.md                          (exists — update with project description)
├── PLAN.md                            (this file)
├── pyproject.toml                     (uv-managed, declare all deps)
├── uv.lock
├── sipnet_calibration/                (reusable integration package)
│   ├── __init__.py
│   ├── likelihood.py                  SIPNETLikelihood — ProbPipe Likelihood protocol
│   ├── priors.py                      build_prior() helper
│   └── predictive.py                  prior_predictive() and posterior_predictive() via PyEns
├── notebooks/
│   ├── 01_timing_benchmark.ipynb      How fast is SIPNET? Choose time horizon.
│   ├── 02_prior_predictive.ipynb      Prior → parallel runs → NEE fan chart
│   └── 03_mcmc_calibration.ipynb      Full MCMC: synthetic data → posterior → checks
└── data/
    └── era5_site1.clim                (copy from pySIPNET/data/ — read-only input)
```

---

## Dependencies (`pyproject.toml`)

```toml
[project]
name = "spatial-lsm-calibration"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pysipnet",          # local editable install: uv add --editable ../pySIPNET
    "probpipe",          # from TARPS-group — check what install method is needed
    "pyens",             # from arob5/PyEns — local editable: uv add --editable ../PyEns
    "numpy>=1.24",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "arviz>=0.13",
    "jax[cpu]>=0.4",
    "jupyter",
]
```

Use `uv add --editable <path>` for pySIPNET and PyEns (local repos). Check ProbPipe's installation method (pip install from GitHub or local editable).

---

## `sipnet_calibration/likelihood.py`

```python
"""ProbPipe-compatible likelihood wrapper for SIPNET."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
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
    param_names:
        List of SIPNET parameter leaf names to calibrate, in the same order
        as the fields of the ProbPipe Record prior. E.g. ["a_max", "psn_t_opt"].
        These must be valid kwargs for SIPNETModel.__call__.
    sigma_obs:
        Observation noise standard deviation (gC m-2 day-1 for NEE).
    output_var:
        Which SIPNETResult method to call for the modelled output.
        Default "nee". Must be a zero-argument method returning pd.Series.
    """

    model: Any                     # SIPNETModel — avoid circular import
    param_names: list[str]
    sigma_obs: float
    output_var: str = "nee"

    def log_likelihood(self, params: Any, data: np.ndarray) -> float:
        """Evaluate Gaussian log-likelihood.

        Parameters
        ----------
        params:
            ProbPipe Record with attributes matching self.param_names.
            Values are JAX scalars; convert to float before passing to pySIPNET.
        data:
            Observed output timeseries (1-D array, same length as SIPNET output).
        """
        # Build override dict: flat param-name -> Python float
        overrides = {name: float(getattr(params, name)) for name in self.param_names}

        try:
            result = self.model(**overrides)
        except Exception:
            # Any SIPNET failure (crash, validation error) -> reject this proposal
            return -1e30

        predicted = getattr(result, self.output_var)().values
        if len(predicted) != len(data):
            return -1e30

        residuals = data - predicted
        return float(-0.5 * np.sum(residuals ** 2) / self.sigma_obs ** 2)
```

---

## `sipnet_calibration/priors.py`

```python
"""Helpers for constructing ProbPipe priors over SIPNET parameters."""
from __future__ import annotations
import jax.numpy as jnp


def build_prior(param_specs: dict) -> "Record":
    """Build a ProbPipe Record prior from a dict of (name, spec) pairs.

    param_specs: dict mapping param_name -> dict with keys:
        "prior_type": "lognormal" | "normal" | "beta"
        "mu" or "loc": centre
        "sigma" or "scale": spread
        (for beta: "alpha", "beta")
    """
    from probpipe import Record, LogNormal, Normal, Beta

    dist_map = {}
    for name, spec in param_specs.items():
        kind = spec["prior_type"]
        if kind == "lognormal":
            dist_map[name] = LogNormal(mu=spec["mu"], sigma=spec["sigma"])
        elif kind == "normal":
            dist_map[name] = Normal(loc=spec["loc"], scale=spec["scale"])
        elif kind == "beta":
            dist_map[name] = Beta(alpha=spec["alpha"], beta=spec["beta"])
        else:
            raise ValueError(f"Unknown prior type: {kind!r}")
    return Record(**dist_map)
```

---

## `sipnet_calibration/predictive.py`

```python
"""Prior and posterior predictive checks via PyEns parallel runs."""
from __future__ import annotations
import numpy as np
import jax
from typing import Any


def prior_predictive(
    prior,
    model,
    param_names: list[str],
    n_samples: int = 200,
    seed: int = 0,
    n_workers: int = 4,
    output_var: str = "nee",
) -> np.ndarray:
    """Sample from prior and run SIPNET in parallel.

    Parameters
    ----------
    prior:
        ProbPipe Record distribution.
    model:
        SIPNETModel instance (used as callable by PyEns).
    param_names:
        Ordered list of parameter names matching prior Record fields.
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

    # 1. Sample parameter vectors from ProbPipe prior
    key = jax.random.PRNGKey(seed)
    samples = prior._sample(key, sample_shape=(n_samples,))
    # samples is a Record; each field has shape (n_samples,)

    # 2. Build PyEns EnsembleSpec
    members = Axis("member", size=n_samples)
    param_arrays = {
        name: [float(getattr(samples, name)[i]) for i in range(n_samples)]
        for name in param_names
    }
    spec = EnsembleSpec(inputs=sipnet_member_fields(members, **param_arrays))

    # 3. Run in parallel via PyEns
    runner = EnsembleRunner(model, LocalBackend(n_workers=n_workers))
    result = runner.run(spec)

    if result.n_failed > 0:
        print(f"Warning: {result.n_failed}/{n_samples} prior predictive runs failed.")

    # 4. Collect output timeseries
    timeseries = [
        getattr(rec.output, output_var)().values
        for rec in result.succeeded
    ]
    return np.stack(timeseries)  # shape: (n_succeeded, n_timesteps)


def posterior_predictive(
    posterior,
    model,
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
    model:
        SIPNETModel instance.
    param_names:
        Ordered list of parameter names matching posterior fields.
    n_samples:
        Number of posterior draws to use.
    seed:
        JAX PRNG seed.
    n_workers:
        Number of parallel workers.
    output_var:
        Which SIPNETResult method to collect.

    Returns
    -------
    np.ndarray of shape (n_succeeded, n_timesteps)
    """
    # Reuse prior_predictive logic with posterior samples instead
    key = jax.random.PRNGKey(seed)
    samples = posterior._sample(key, sample_shape=(n_samples,))

    from pyens import EnsembleRunner, EnsembleSpec, Axis
    from pyens.backends import LocalBackend
    from pysipnet.ensemble import sipnet_member_fields

    members = Axis("member", size=n_samples)
    param_arrays = {
        name: [float(getattr(samples, name)[i]) for i in range(n_samples)]
        for name in param_names
    }
    spec = EnsembleSpec(inputs=sipnet_member_fields(members, **param_arrays))
    runner = EnsembleRunner(model, LocalBackend(n_workers=n_workers))
    result = runner.run(spec)

    if result.n_failed > 0:
        print(f"Warning: {result.n_failed}/{n_samples} posterior predictive runs failed.")

    timeseries = [
        getattr(rec.output, output_var)().values
        for rec in result.succeeded
    ]
    return np.stack(timeseries)
```

---

## Ground Truth Parameters for Synthetic Data

These are reasonable literature-informed values for a temperate forest. The calibration
should recover them from synthetic observations.

```python
GROUND_TRUTH = {
    "a_max":          120.0,   # nmol CO2 g-1 leaf s-1
    "psn_t_opt":       20.0,   # deg C
    "half_sat_par":   200.0,   # Einstein m-2 day-1
    "base_veg_resp":    0.5,   # yr-1
    "veg_resp_q10":     2.0,   # dimensionless
    "base_soil_resp":   0.3,   # yr-1
    "soil_resp_q10":    2.2,   # dimensionless
    "wue_const":       10.0,   # dimensionless
}

SIGMA_OBS = 0.5   # gC m-2 day-1  (observation noise std)
```

All 8 parameters are in `ParameterDomain.POSITIVE` → use `LogNormal` priors.
Exception: `psn_t_opt` is a temperature (°C, possibly > 0 but not constrained positive by domain)
— use `Normal(loc=20, scale=8)` truncated at `psnTMin` if needed, or simply `Normal` and let
prior mass handle it (RWMH will rarely propose below 0°C for this parameter).

## Prior Specification

```python
PARAM_SPECS = {
    "a_max":          {"prior_type": "lognormal", "mu": jnp.log(120.0), "sigma": 0.5},
    "psn_t_opt":      {"prior_type": "normal",    "loc": 20.0,          "scale": 8.0},
    "half_sat_par":   {"prior_type": "lognormal", "mu": jnp.log(200.0), "sigma": 0.5},
    "base_veg_resp":  {"prior_type": "lognormal", "mu": jnp.log(0.5),   "sigma": 0.5},
    "veg_resp_q10":   {"prior_type": "lognormal", "mu": jnp.log(2.0),   "sigma": 0.3},
    "base_soil_resp": {"prior_type": "lognormal", "mu": jnp.log(0.3),   "sigma": 0.5},
    "soil_resp_q10":  {"prior_type": "lognormal", "mu": jnp.log(2.2),   "sigma": 0.3},
    "wue_const":      {"prior_type": "lognormal", "mu": jnp.log(10.0),  "sigma": 0.5},
}

PARAM_NAMES = list(PARAM_SPECS.keys())
```

---

## Notebook Outlines

### `01_timing_benchmark.ipynb`

1. Load `data/era5_site1.clim`
2. Slice to 1 month / 3 months / 6 months / 12 months
3. Write each slice to a temp file; use `ClimateDrivers.from_path()` + `ClimateStaging.SYMLINK`
4. For each horizon: run `SIPNETRunner.run()` 10× with the same base params, record wall time
5. Display mean ± std ms per run vs. horizon
6. **Decision rule:** Choose the longest horizon where mean < 100ms/run
7. Save the chosen climate slice to `data/benchmark_climate.clim`

### `02_prior_predictive.ipynb`

1. Build `SIPNETModel` with base params and the chosen climate (file-backed, symlinked)
2. Build `prior` using `build_prior(PARAM_SPECS)`
3. Call `prior_predictive(prior, model, PARAM_NAMES, n_samples=300, n_workers=8)`
4. Plot:
   - Fan chart of NEE timeseries (5th/25th/50th/75th/95th percentiles across samples)
   - Marginal histograms of each calibrated parameter across samples
5. Verify the prior is appropriately wide: ground truth values should be in ~50th–90th percentile range of prior predictive NEE variance
6. Adjust prior widths if prior predictive is unrealistically wide or tight

### `03_mcmc_calibration.ipynb`

**Section 1: Generate synthetic observations**
```python
true_result = model(**GROUND_TRUTH)
rng = np.random.default_rng(42)
obs_nee = true_result.nee().values + rng.normal(0, SIGMA_OBS, size=len(true_result.nee()))
```

**Section 2: Build ProbPipe model**
```python
from sipnet_calibration.likelihood import SIPNETLikelihood
from sipnet_calibration.priors import build_prior
from probpipe import SimpleModel, condition_on

likelihood = SIPNETLikelihood(model, PARAM_NAMES, sigma_obs=SIGMA_OBS)
prior = build_prior(PARAM_SPECS)
prob_model = SimpleModel(prior, likelihood)
```

**Section 3: MCMC**
```python
posterior = condition_on(
    prob_model,
    obs_nee,
    method="rwmh",
    num_results=5_000,
    num_warmup=1_000,
    step_size=0.05,
    random_seed=42,
)
```

**Section 4: Diagnostics**
- `arviz.plot_trace(posterior.inference_data)` — check mixing
- `arviz.summary(posterior.inference_data)` — R-hat, ESS per parameter
- Marginal posterior plots with ground truth overlaid as vertical line

**Section 5: Posterior predictive check**
```python
post_pred_nee = posterior_predictive(
    posterior, model, PARAM_NAMES, n_samples=200, n_workers=8
)
```
- Fan chart of posterior predictive NEE vs. synthetic observations
- Should visually bracket the observations

---

## Key Design Decisions

### PyEns + ProbPipe pairing for prior/posterior predictive

ProbPipe handles statistical operations (prior specification, sampling, posterior inference).
PyEns handles parallel SIPNET execution. They are NOT merged — they communicate at a simple
Python interface:

```
ProbPipe prior._sample() -> Record with shape (N,)
                            ↓ (bridge: convert Record fields to lists of floats)
PyEns EnsembleSpec         ↓
       EnsembleRunner.run() -> EnsembleResult
                            ↓ (collect NEE timeseries from each RunRecord)
np.ndarray (N, T)          ↓ → plot or feed back into ProbPipe as empirical distribution
```

This is the natural pairing: ProbPipe as the statistical layer, PyEns as the execution layer.
ProbPipe's `@workflow_function` broadcasting is designed for JAX-vectorizable models (vmap).
SIPNET is a subprocess — not JAX-traceable. PyEns's `ProcessPoolExecutor`-based backend is
the right tool for subprocess parallelism.

### Climate data IO optimization

The climate file should be written to disk once and never re-read per MCMC iteration.
Use `ClimateDrivers.from_path(path)` (file-backed, lazy) and `ClimateStaging.SYMLINK`
in the runner. This symlinks the climate file into each SIPNET temp dir instead of
copying it — eliminates per-run IO overhead across all iterations.

### No private API use

Only call `SIPNETModel(**param_overrides)` — the public `__call__` interface.
Parameter override keys are flat leaf names (e.g., `a_max`, `base_veg_resp`),
not dot-paths (`photosynthesis.a_max`) or camelCase (`aMax`).
The Record prior field names must match exactly.

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| ProbPipe `Record` prior sample access pattern differs from `getattr(samples, name)` | Inspect a small `prior._sample(key, (3,))` output in notebook and adjust accessor |
| `sipnet_member_fields()` expects specific value types (list vs. array) | Check PyEns source; wrap in `[float(v) for v in ...]` to be safe |
| `LocalBackend` pickling fails for `SIPNETModel` | Switch to `SequentialBackend` to debug; ensure model defined at module level |
| RWMH step_size needs tuning — acceptance rate target ~23% for high-dim | Monitor `posterior.inference_data` acceptance rate; adjust step_size if < 10% or > 50% |
| Per-run latency > 100ms → MCMC takes hours overnight | Shorten time horizon based on timing benchmark results |

---

## Do Not Do

- Do not modify `pySIPNET`, `ProbPipe`, or `PyEns` source code
- Do not call `SIPNETModel._apply_overrides` or any other private method from those packages
- Do not use `camelCase` parameter names (SIPNET's internal names) — use pySIPNET's `snake_case` leaf names
- Do not use dot-path parameter names (`photosynthesis.a_max`) — they are not valid kwargs for `SIPNETModel.__call__`
- Do not copy climate data into notebooks — always read from `data/` via `from_path()`
