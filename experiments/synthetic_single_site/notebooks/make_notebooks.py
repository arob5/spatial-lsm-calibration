"""Generate the notebooks for synthetic_single_site experiment."""
import json
from pathlib import Path

NB_DIR = Path(__file__).parent


def nb(cells, title=""):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }


def md(source, id="md"):
    return {
        "cell_type": "markdown",
        "id": id,
        "metadata": {},
        "source": source,
    }


def code(source, id="c"):
    return {
        "cell_type": "code",
        "id": id,
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


# ---------------------------------------------------------------------------
# 01_prior_predictive.ipynb
# ---------------------------------------------------------------------------

NB01_CELLS = [
md("""# 01 — Prior Predictive Check

Sample parameters from the prior defined in `config.py` and run SIPNET in
parallel via PyEns. Verify that the prior is appropriately wide and that the
ground truth lies well within the prior predictive envelope.

**Pre-requisite:** run `python config.py` in the experiment directory to
generate `data/climate.clim` and `data/obs_nee.npy`.
"""),

code("""\
import notebook_env  # noqa: F401 — sets sys.path for config import
import numpy as np
import matplotlib.pyplot as plt

from config import build_prior, load_model, GROUND_TRUTH, DATA_DIR
from sipnet_calibration import prior_predictive
from sipnet_calibration.plotting import fan_chart
""", "imports"),

code("""\
# Load the model (reads climate + obs from data/)
_prob_model, obs_nee = load_model()

# Build the prior and extract the base SIPNETModel from the likelihood
prior = build_prior()
param_names = list(prior.record_template.fields)
print("Calibrated parameters:", param_names)
print("Prior flat size (MCMC dimensionality):", prior.record_template.flat_size)
""", "setup"),

code("""\
# We need just the SIPNETModel, not the full ProbPipe model
# Re-create it so we can pass it to prior_predictive
from pysipnet import SIPNETRunner, SIPNETModel
from pysipnet.climate import ClimateDrivers
from pysipnet.runner import ClimateStaging
from sipnet_calibration import default_base_params

climate = ClimateDrivers.from_path(str(DATA_DIR / "climate.clim"))
sipnet_model = SIPNETModel(
    SIPNETRunner(climate_staging=ClimateStaging.SYMLINK),
    base_params=default_base_params(),
    base_climate=climate,
)

""", "build_model"),

code("""\
prior_nee = prior_predictive(prior, sipnet_model, n_samples=200, n_workers=4, seed=0)
print(f"Prior predictive shape: {prior_nee.shape}")
""", "run_prior_pred"),

code("""\
# Ground truth NEE for visual reference
truth_nee = sipnet_model(**GROUND_TRUTH).nee().values
t = np.arange(prior_nee.shape[1])

ax = fan_chart(
    prior_nee,
    t=t,
    obs=obs_nee,
    truth=truth_nee,
    ylabel="NEE (gC m$^{-2}$ per 3-hr step)",
    title="Prior predictive fan chart",
)
plt.tight_layout()
plt.show()

print(f"Ground truth NEE range: [{truth_nee.min():.2f}, {truth_nee.max():.2f}]")
print(f"Prior 5–95% range: [{np.percentile(prior_nee, 5):.2f}, {np.percentile(prior_nee, 95):.2f}]")
""", "fan_chart"),

code("""\
# Marginal prior distributions
import jax
key = jax.random.PRNGKey(99)
prior_samples = prior._sample(key, (2000,))

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.ravel()

for i, name in enumerate(param_names):
    ax = axes[i]
    vals = np.array(prior_samples[name])
    ax.hist(vals, bins=40, density=True, alpha=0.7, color="steelblue")
    ax.axvline(GROUND_TRUTH[name], color="red", lw=2, label="truth")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Value", fontsize=8)

axes[0].legend()
plt.suptitle("Prior marginal distributions  (red = ground truth)")
plt.tight_layout()
plt.show()
""", "marginals"),
]

# ---------------------------------------------------------------------------
# 02_mcmc_diagnostics.ipynb
# ---------------------------------------------------------------------------

NB02_CELLS = [
md("""# 02 — MCMC Diagnostics & Posterior Predictive

Load MCMC output from `outputs/mcmc/<run_id>.nc` and inspect convergence,
parameter recovery, and posterior predictive skill.

**Pre-requisites:**
1. `python config.py` — generate data
2. `python run_mcmc.py --run_id rwmh_v1` — run MCMC and save NetCDF

Change `RUN_ID` below to switch between saved runs.
"""),

code("""\
import notebook_env  # noqa: F401 — sets sys.path for config import
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from config import build_prior, OUTPUT_DIR, GROUND_TRUTH, DATA_DIR, SIGMA_OBS
from sipnet_calibration import default_base_params, posterior_predictive
from sipnet_calibration.plotting import fan_chart, plot_diagnostics
from pysipnet import SIPNETRunner, SIPNETModel
from pysipnet.climate import ClimateDrivers
from pysipnet.runner import ClimateStaging
""", "imports"),

code("""\
RUN_ID = "rwmh_v1"   # ← change this to load a different run

nc_path = OUTPUT_DIR / "mcmc" / f"{RUN_ID}.nc"
idata = az.from_netcdf(str(nc_path))
print(f"Loaded: {nc_path}")
print(idata)
""", "load"),

code("""\
prior = build_prior()
param_names = list(prior.record_template.fields)

plot_diagnostics(idata, param_names=param_names, ground_truth=GROUND_TRUTH)
plt.show()
""", "diagnostics"),

code("""\
# Posterior predictive check
# Reconstruct posterior chains from ArviZ InferenceData
# Shape: (n_chains, n_draws, n_params) from the raw RWMH flat array
# ArviZ stores flattened chains in idata.posterior as individual param DataArrays.
# We need the raw flat chain (pre-unflatten) — store it separately via run_mcmc.py
# OR re-derive from named params.

# Derive flat chains from named posterior params in idata
n_chains = idata.posterior.dims["chain"]
n_draws  = idata.posterior.dims["draw"]
n_params = prior.record_template.flat_size

# Build flat chains from the unflattened stored values
# (we'll let posterior_predictive handle this by passing the named arrays)

# Simpler: extract per-param arrays and pass directly to model
# Here we use a convenience path: build a (n_chains, n_draws, n_params) array
# by stacking named params in prior field order.
flat_chains = np.stack(
    [np.array(idata.posterior[name]) for name in param_names],
    axis=-1,
)  # shape: (n_chains, n_draws, n_params)

print(f"Flat chains shape: {flat_chains.shape}")

# Rebuild SIPNETModel
climate = ClimateDrivers.from_path(str(DATA_DIR / "climate.clim"))
sipnet_model = SIPNETModel(
    SIPNETRunner(climate_staging=ClimateStaging.SYMLINK),
    base_params=default_base_params(),
    base_climate=climate,
)

obs_nee = np.load(str(DATA_DIR / "obs_nee.npy"))
truth_nee = sipnet_model(**GROUND_TRUTH).nee().values
""", "setup_predictive"),

code("""\
# NOTE: flat_chains above is in NAMED (constrained) space, but posterior_predictive
# expects flat chains in the UNCONSTRAINED (prior) space for unflatten_value().
# Since ArviZ stores the constrained values, we skip unflatten and instead
# pass the model overrides directly.

# Custom posterior predictive (bypassing unflatten):
from pyens import EnsembleRunner, EnsembleSpec, Axis
from pyens.backends import LocalBackend
from pysipnet.ensemble import sipnet_member_fields

draws_flat = flat_chains.reshape(-1, flat_chains.shape[-1])  # (n_total, n_params)
rng = np.random.default_rng(1)
idx = rng.choice(len(draws_flat), size=min(200, len(draws_flat)), replace=False)
selected = draws_flat[idx]

param_arrays = {
    name: [float(selected[i, j]) for i in range(len(idx))]
    for j, name in enumerate(param_names)
}

members = Axis("member", size=len(idx))
spec = EnsembleSpec(inputs=sipnet_member_fields(members, **param_arrays))
runner = EnsembleRunner(sipnet_model, LocalBackend(n_workers=4))
result = runner.run(spec)

post_pred_nee = np.stack([r.output.nee().values for r in result.succeeded])
print(f"Posterior predictive shape: {post_pred_nee.shape}")
""", "run_post_pred"),

code("""\
t = np.arange(post_pred_nee.shape[1])

ax = fan_chart(
    post_pred_nee,
    t=t,
    obs=obs_nee,
    truth=truth_nee,
    ylabel="NEE (gC m$^{-2}$ per 3-hr step)",
    title=f"Posterior predictive check  [{RUN_ID}]",
    color="steelblue",
)
plt.tight_layout()
plt.show()
""", "post_pred_plot"),

code("""\
# Marginal posterior histograms vs ground truth
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.ravel()

for i, name in enumerate(param_names):
    ax = axes[i]
    vals = np.array(idata.posterior[name]).ravel()
    ax.hist(vals, bins=50, density=True, alpha=0.7, color="steelblue")
    ax.axvline(GROUND_TRUTH[name], color="red", lw=2, label="truth")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Value", fontsize=8)
    ax.legend(fontsize=7)

plt.suptitle(f"Posterior marginals  [{RUN_ID}]  (red = ground truth)")
plt.tight_layout()
plt.show()
""", "marginals"),
]

# ---------------------------------------------------------------------------
# Write notebooks
# ---------------------------------------------------------------------------

for fname, cells in [
    ("01_prior_predictive.ipynb", NB01_CELLS),
    ("02_mcmc_diagnostics.ipynb", NB02_CELLS),
]:
    path = NB_DIR / fname
    with open(path, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"Written {fname}")
