"""Parallel random-walk Metropolis: the MCMC baseline.

The target is the model's posterior of theta, :math:`\\pi(\\theta) \\propto
\\pi_0(\\theta)\\, p(y \\mid \\theta)` (the scales integrated out where they
are inferred). :math:`C` chains advance in lockstep, so each step is one
batch of SIPNET runs:

.. math::

    \\theta'_c = \\theta_c + \\tfrac{2.38}{\\sqrt D} L z_c,\\quad z_c \\sim \\mathcal N(0, I),
    \\qquad \\text{accepted with probability } \\min\\big(1, \\pi(\\theta'_c)/\\pi(\\theta_c)\\big).

with :math:`L` scaled by a factor :math:`\\lambda`. The chains start from an
importance-sampling run's draws, resampled by weight among those whose runs
succeeded; :math:`LL^\\top` starts as the covariance of the EKI ensemble that
run was seeded from (the run's own weights are too concentrated to estimate
one), and :math:`\\lambda = 1`. During the first
``config.MCMC_ADAPTATION_STEPS`` steps, every 25 steps
:math:`\\log\\lambda \\mathrel{+}= 2(\\bar a - 0.234)`, :math:`\\bar a` the
acceptance rate of those steps, and from step 100 on, every 50 steps,
:math:`LL^\\top` is re-estimated from the second half of all chains'
history; both are then frozen. The first ``config.MCMC_DISCARDED_STEPS``
steps are discarded, so the kept draws are a Metropolis chain with a fixed
kernel.

The run checkpoints every 25 steps (``checkpoint.npz``) and ``resume=True``
continues from the last checkpoint, for longer than first asked if
*n_steps* has grown (the kernel stays frozen). Every 200 steps past the
discarded ones, and at the end, it writes every draw of theta
(``chains.npz``: ``theta`` ``(T, C, D)``, ``log_density``, ``accepted``, and
each scaled source's quadratic form at the chain's state, from which the
scales' conditional is drawn), in ``samples.nc``, the kept draws, and in
``convergence.csv``, each parameter's split :math:`\\hat R` and bulk and tail
effective sample sizes over the kept draws (ArviZ), so a run's results can
be read while it continues.
"""

import json
from pathlib import Path

import arviz
import numpy as np
import pandas as pd

from sipnet_calibration.inference import eki_problem

from .. import config
from ..model import prior
from ..models import Model
from .records import Cost, append_history, load_samples, save_samples

__all__ = ["run_mcmc"]

#: Steps between checkpoints.
CHECKPOINT_EVERY = 25

#: Steps between the snapshots of the kept draws (``samples.nc``,
#: ``convergence.csv``) a running chain writes.
SNAPSHOT_EVERY = 200


def run_mcmc(
    model: Model,
    posterior,
    noise_model,
    directory,
    *,
    start_directory,
    n_chains: int,
    n_steps: int,
    resume: bool,
) -> None:
    """Run the chains for *model*, started from the importance-sampling run
    under *start_directory*, and write the run under *directory*."""
    forward_map = eki_problem(posterior).forward
    cost = Cost()
    cost.include(start_directory / "cost.json", prefix="initialization")
    forward = cost.counted(forward_map)
    scaled = noise_model.scaled_names

    def log_density(theta):
        predictions = np.asarray(forward(theta))
        forms = noise_model.quadratic_forms(predictions)
        values = np.asarray(posterior.log_prior(theta)) + noise_model.log_likelihood(predictions)
        return np.where(np.isnan(values), -np.inf, values), {k: forms[k] for k in scaled}

    checkpoint = directory / "checkpoint.npz"
    if resume and checkpoint.exists():
        state = _load_checkpoint(checkpoint)
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(state.pop("rng_state")))
        cost.phases.extend(json.loads(str(state.pop("cost_phases"))))
        _extend(state, n_steps)
        print(f"resuming at step {state['step']} of {n_steps}", flush=True)
    else:
        rng = np.random.default_rng(config.MCMC_SEED)
        start, covariance = _starting_points(start_directory, n_chains, rng)
        with cost.phase("initial evaluation"):
            log_p, forms = log_density(start)
        state = {
            "step": 0,
            "theta": start,
            "log_p": log_p,
            "forms": forms,
            "covariance": covariance,
            "log_scale": 0.0,
            "draws": np.empty((n_steps, n_chains, start.shape[1])),
            "log_densities": np.empty((n_steps, n_chains)),
            "accepted": np.zeros((n_steps, n_chains), bool),
            "draw_forms": {k: np.empty((n_steps, n_chains)) for k in scaled},
        }
    D = state["theta"].shape[1]
    while state["step"] < n_steps:
        phase = "adaptation" if state["step"] < _discarded_steps(n_steps) else "sampling"
        with cost.phase(phase):
            for _ in range(min(CHECKPOINT_EVERY, n_steps - state["step"])):
                _step(state, log_density, rng, D)
        _save_checkpoint(checkpoint, state, rng, cost)
        acceptance = state["accepted"][: state["step"]].mean()
        append_history(
            directory,
            {"step": state["step"], "acceptance": acceptance, "mean_log_density": float(np.mean(state["log_p"]))},
        )
        print(f"step {state['step']}: acceptance {acceptance:.3f}", flush=True)
        if state["step"] > _discarded_steps(n_steps) and state["step"] % SNAPSHOT_EVERY == 0:
            _write(model, posterior, noise_model, directory, state, start_directory, n_steps)
            cost.write(directory)
    _write(model, posterior, noise_model, directory, state, start_directory, n_steps)
    cost.write(directory)


def _step(state, log_density, rng, D: int) -> None:
    """One Metropolis step of every chain, adapting the proposal while allowed."""
    step = state["step"]
    cholesky = _scaled_cholesky(state["covariance"]) * np.exp(float(state["log_scale"]))
    proposal = state["theta"] + rng.normal(size=state["theta"].shape) @ cholesky.T
    log_p_new, forms_new = log_density(proposal)
    accept = np.log(rng.uniform(size=len(proposal))) < log_p_new - state["log_p"]
    state["theta"][accept] = proposal[accept]
    state["log_p"][accept] = log_p_new[accept]
    for name in state["forms"]:
        state["forms"][name][accept] = forms_new[name][accept]
        state["draw_forms"][name][step] = state["forms"][name]
    state["draws"][step] = state["theta"]
    state["log_densities"][step] = state["log_p"]
    state["accepted"][step] = accept
    if step < config.MCMC_ADAPTATION_STEPS and (step + 1) % 25 == 0:
        recent = state["accepted"][step - 24 : step + 1].mean()
        state["log_scale"] = float(state["log_scale"]) + 2.0 * (recent - 0.234)
    if 100 <= step < config.MCMC_ADAPTATION_STEPS and (step + 1) % 50 == 0:
        history = state["draws"][step // 2 : step + 1].reshape(-1, D)
        state["covariance"] = np.cov(history.T) + 1e-9 * np.eye(D)
    state["step"] = step + 1


def _starting_points(start_directory, n_chains: int, rng):
    """*n_chains* draws of the importance-sampling run, by weight among its
    valid samples, and the covariance of the EKI ensemble it was seeded from."""
    samples = load_samples(start_directory)
    theta = samples["theta"].values
    log_weights = np.where(samples["valid"].values.astype(bool), samples["log_weight"].values, -np.inf)
    weights = np.exp(log_weights - np.logaddexp.reduce(log_weights))
    chosen = rng.choice(len(theta), size=n_chains, p=weights)
    seed = load_samples(Path(samples.attrs["seed_run"]))
    ensemble = seed["theta"].values[seed["valid"].values.astype(bool)]
    return theta[chosen].copy(), np.cov(ensemble.T)


def _scaled_cholesky(covariance) -> np.ndarray:
    """:math:`(2.38/\\sqrt D) L`, :math:`LL^\\top` = *covariance*."""
    D = covariance.shape[0]
    return np.linalg.cholesky(covariance) * 2.38 / np.sqrt(D)


def _save_checkpoint(path, state, rng, cost) -> None:
    """Everything a resumed run needs."""
    arrays = {k: v for k, v in state.items() if not isinstance(v, dict)}
    arrays.update({f"forms__{k}": v for k, v in state["forms"].items()})
    arrays.update({f"draw_forms__{k}": v for k, v in state["draw_forms"].items()})
    own_phases = [phase for phase in cost.phases if not phase["phase"].startswith("initialization:")]
    np.savez(
        path.with_suffix(".partial.npz"),
        rng_state=json.dumps(rng.bit_generator.state),
        cost_phases=json.dumps(own_phases),
        **arrays,
    )
    path.with_suffix(".partial.npz").replace(path)


def _load_checkpoint(path) -> dict:
    """The state :func:`_save_checkpoint` wrote."""
    stored = dict(np.load(path, allow_pickle=False))
    state = {"forms": {}, "draw_forms": {}}
    for key, value in stored.items():
        if key.startswith("forms__"):
            state["forms"][key[len("forms__") :]] = value
        elif key.startswith("draw_forms__"):
            state["draw_forms"][key[len("draw_forms__") :]] = value
        else:
            state[key] = value
    state["step"] = int(state["step"])
    state["log_scale"] = float(state["log_scale"])
    return state


def _write(model, posterior, noise_model, directory, state, start_directory, n_steps) -> None:
    """``chains.npz`` (every draw so far), ``samples.nc`` (the kept draws,
    with each scale drawn from its conditional at the draw) and
    ``convergence.csv``."""
    done = slice(0, state["step"])
    chains = directory / "chains.partial.npz"
    np.savez(
        chains,
        theta=state["draws"][done],
        log_density=state["log_densities"][done],
        accepted=state["accepted"][done],
        **{f"quadratic_form_{k}": v[done] for k, v in state["draw_forms"].items()},
    )
    chains.replace(directory / "chains.npz")
    discarded = _discarded_steps(n_steps)
    kept = slice(discarded, state["step"])
    theta = state["draws"][kept].reshape(-1, state["draws"].shape[2])
    forms = {k: v[kept].reshape(-1) for k, v in state["draw_forms"].items()}
    rng = np.random.default_rng(config.MCMC_SEED + 1)
    scales = noise_model.draw_scales(rng, forms) if noise_model.inferred else {}
    log_prior = np.asarray(posterior.log_prior(theta))
    save_samples(
        directory,
        posterior,
        theta,
        model_name=model.name,
        algorithm="mcmc",
        log_prior=log_prior,
        log_likelihood=state["log_densities"][kept].reshape(-1) - log_prior,
        valid=np.isfinite(state["log_densities"][kept].reshape(-1)),
        extra={
            **{f"quadratic_form_{k}": v for k, v in forms.items()},
            **{f"{k}_noise_scale": v for k, v in scales.items()},
        },
        attributes={
            "start_run": str(start_directory),
            "chains": state["draws"].shape[1],
            "steps": state["step"],
            "discarded_steps": discarded,
        },
    )
    _convergence_table(posterior, state["draws"][kept]).to_csv(directory / "convergence.csv")


def _convergence_table(posterior, draws: np.ndarray) -> pd.DataFrame:
    """Each parameter's split R-hat and bulk and tail ESS over *draws*
    ``(steps, chains, D)``, in natural units."""
    steps, chains, _ = draws.shape
    natural = prior.natural_table(posterior, draws.reshape(steps * chains, -1))
    rows = {}
    for name in natural.columns:
        values = natural[name].to_numpy().reshape(steps, chains).T
        rows[name] = {
            "r_hat": float(arviz.rhat(values)),
            "bulk_ess": float(arviz.ess(values, method="bulk")),
            "tail_ess": float(arviz.ess(values, method="tail")),
        }
    return pd.DataFrame(rows).T.rename_axis("parameter")


def _extend(state, n_steps: int) -> None:
    """Grow *state*'s per-step arrays to *n_steps* rows, for a resumed run asked to go further."""
    extra = n_steps - state["draws"].shape[0]
    if extra <= 0:
        return
    state["draws"] = np.concatenate([state["draws"], np.empty((extra, *state["draws"].shape[1:]))])
    state["log_densities"] = np.concatenate(
        [state["log_densities"], np.empty((extra, state["log_densities"].shape[1]))]
    )
    state["accepted"] = np.concatenate(
        [state["accepted"], np.zeros((extra, state["accepted"].shape[1]), bool)]
    )
    state["draw_forms"] = {
        k: np.concatenate([v, np.empty((extra, v.shape[1]))]) for k, v in state["draw_forms"].items()
    }


def _discarded_steps(n_steps: int) -> int:
    """The steps discarded: ``config.MCMC_DISCARDED_STEPS``, or half of a shorter run."""
    return min(config.MCMC_DISCARDED_STEPS, n_steps // 2)
