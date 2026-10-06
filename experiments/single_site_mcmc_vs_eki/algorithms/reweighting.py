"""Importance sampling and tempered SMC from a density fitted to an EKI run.

The target is the model's posterior of theta, :math:`\\pi(\\theta) \\propto
\\pi_0(\\theta)\\, p(y \\mid \\theta)`, the likelihood ``NoiseModel.log_likelihood``:
Gaussian at scale 1 for fixed noise, the scales' Student-t marginal for
inferred noise. The base density is

.. math::

    q = (1 - \\alpha)\\, t_\\nu(\\hat m, \\kappa \\hat C) + \\alpha\\, \\pi_0,

the Student-t fitted to the seed EKI run's valid final ensemble
(``config.REWEIGHTING_*``). The library's tempered SMC (``smc.run_smc``)
then runs from :math:`q`: in one step (``one_step=True``), that is importance
sampling, :math:`\\log \\tilde w_m = \\log\\pi_0 + \\log p(y\\mid\\theta_m) -
\\log q`; otherwise through :math:`\\pi_\\beta \\propto q^{1-\\beta}(\\pi_0
L)^\\beta` with resampling and Metropolis moves. Each sample's predictions
travel with it, so the scales' conditional and the predictive need no new run.
"""

import numpy as np

from sipnet_calibration import smc
from sipnet_calibration.inference import PriorBaseDensity, eki_problem

from .. import config
from ..models import Model
from .records import Cost, append_history, load_samples, save_samples

__all__ = ["base_density", "run_reweighting"]


def run_reweighting(
    model: Model,
    posterior,
    noise_model,
    directory,
    *,
    seed_directory,
    method: str,
    n_samples: int,
) -> None:
    """Importance sampling (``method="is"``) or tempered SMC (``"smc"``) from
    the run under *seed_directory*; write the run under *directory*."""
    seed = load_samples(seed_directory)
    valid = seed["valid"].values.astype(bool) if "valid" in seed else np.ones(seed.sizes["sample"], bool)
    base = base_density(posterior, seed["theta"].values[valid])
    problem_forward = eki_problem(posterior).forward
    cost = Cost()
    cost.include(seed_directory / "cost.json", prefix="seed")
    forward = cost.counted(problem_forward)

    def log_likelihood(theta):
        predictions = np.asarray(forward(theta))
        values = noise_model.log_likelihood(predictions)
        failed = np.isnan(predictions).any(axis=1)
        return smc.LikelihoodEvaluation(
            log_likelihood=np.where(failed, np.nan, values), auxiliary=predictions
        )

    problem = smc.TemperingProblem(
        log_prior=posterior.log_prior, log_likelihood=log_likelihood, base=base
    )
    settings = smc.SMCSettings(
        n_samples=n_samples, seed=config.REWEIGHTING_SEED, one_step=method == "is"
    )
    with cost.phase(method):
        state = smc.initial_state(problem, settings)
        recorded_stage = -1
        for state in smc.run_smc(problem, state):
            if state.stage != recorded_stage:
                _record(directory, state)
                recorded_stage = state.stage
    smc.save_state(directory / "smc_state", state)
    predictions = np.asarray(state.auxiliary)
    log_weights = np.asarray(state.log_weights)
    forms = noise_model.quadratic_forms(predictions)
    rng = np.random.default_rng(config.REWEIGHTING_SEED)
    scales = noise_model.draw_scales(rng, forms) if noise_model.inferred else {}
    save_samples(
        directory,
        posterior,
        np.asarray(state.theta),
        model_name=model.name,
        algorithm=directory.name,
        log_weights=log_weights,
        log_prior=np.asarray(state.log_prior),
        log_likelihood=np.asarray(state.log_likelihood),
        valid=np.isfinite(np.asarray(state.log_likelihood)),
        predictions=predictions,
        extra={
            **{f"quadratic_form_{name}": forms[name] for name in noise_model.scaled_names},
            **{f"{name}_noise_scale": values for name, values in scales.items()},
        },
        attributes={
            "seed_run": str(seed_directory),
            "log_evidence": float(state.log_evidence),
            "effective_sample_size": float(smc.effective_sample_size(log_weights)),
            "pareto_k": float(smc.pareto_k(log_weights)) if method == "is" else np.nan,
            "stages": int(state.stage),
        },
    )
    cost.write(directory)


def base_density(posterior, theta) -> smc.DefensiveMixture:
    """:math:`q`: the Student-t fitted to *theta* ``(J, D)``, mixed with the prior."""
    return smc.DefensiveMixture(
        component=smc.fit_student_t(
            theta,
            covariance_inflation=config.REWEIGHTING_COVARIANCE_INFLATION,
            degrees_of_freedom=config.REWEIGHTING_DEGREES_OF_FREEDOM,
        ),
        defensive=PriorBaseDensity(posterior),
        defensive_fraction=config.REWEIGHTING_DEFENSIVE_FRACTION,
    )


def _record(directory, state) -> None:
    """One stage's row of ``history.csv``, printed."""
    log_weights = np.asarray(state.log_weights)
    row = {
        "stage": state.stage,
        "beta": state.beta,
        "effective_sample_size": float(smc.effective_sample_size(log_weights)),
        "log_evidence": float(state.log_evidence),
        "sipnet_runs": state.n_evaluations,
        "forward_calls": state.n_calls,
    }
    append_history(directory, row)
    print(
        f"stage {row['stage']}: beta {row['beta']:.4g}, ESS {row['effective_sample_size']:.1f}, "
        f"log Z {row['log_evidence']:.2f}, {row['sipnet_runs']} runs",
        flush=True,
    )
