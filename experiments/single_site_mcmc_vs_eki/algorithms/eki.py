"""EKI for a model with fixed noise, through EnsKit's driver.

An ensemble of ``config.EKI_ENSEMBLE_SIZE`` prior draws is moved through the
tempered posteriors :math:`\\pi_\\varphi \\propto \\pi_0\\, \\mathcal N(y;
\\mathcal G(\\theta), R)^\\varphi` by the perturbed-observation update,

.. math::

    \\theta_j \\leftarrow \\theta_j + \\hat C_{\\theta g}\\big(\\hat C_{gg} +
    R/\\Delta\\varphi\\big)^{-1}\\big(y - \\mathcal G(\\theta_j) - \\eta_j\\big),
    \\qquad \\eta_j \\sim \\mathcal N(0, R/\\Delta\\varphi),

each increment the largest with :math:`\\mathrm{ESS}(e^{-\\Delta\\varphi
\\Phi_j}) \\ge` ``config.EKI_ESS_FRACTION`` :math:`J`, until :math:`\\varphi
= 1`; the ensemble it ends with is then evaluated once more, for its
predictions. A member whose run fails is moved to the valid members' center
for that step (EnsKit's ``on_failure="repair"``).
"""

import jax
import numpy as np
from enskit import kalman
from enskit.algorithms import eki

from sipnet_calibration.inference import eki_problem
from sipnet_calibration.probability.names import THETA

from .. import config
from ..models import Model
from .records import Cost, append_history, save_samples

__all__ = ["run_eki"]


def run_eki(model: Model, posterior, noise_model, directory, *, ensemble_size: int) -> None:
    """Run EKI on *posterior* (the model's fixed posterior) and write the run."""
    problem = eki_problem(posterior)
    cost = Cost()
    forward = cost.counted(problem.forward)
    ensemble_key, run_key = jax.random.split(jax.random.key(config.EKI_SEED))
    state = eki.EKIState(problem.initial_ensemble(ensemble_key, ensemble_size), key=run_key)
    np.save(directory / "initial_ensemble.npy", np.asarray(state.ensemble[THETA]))
    arguments = (forward, problem.y, problem.noise_covariance)
    with cost.phase("eki"):
        steps = eki.iterate(
            state,
            *arguments,
            update_rule=kalman.Matheron(),
            schedule=eki.AdaptiveESSSchedule(ess_fraction=config.EKI_ESS_FRACTION),
            on_failure="repair",
        )
        for state, record, _ in steps:
            _record_step(directory, record)
        evaluation = eki.evaluate(state, *arguments)
    _record_step(directory, eki.HistoryRecord.from_evaluation(evaluation))
    theta = np.asarray(state.ensemble[THETA])
    predictions = np.asarray(evaluation.ensemble[eki.PREDICTION])
    save_samples(
        directory,
        posterior,
        theta,
        model_name=model.name,
        algorithm="eki",
        log_prior=np.asarray(posterior.log_prior(theta)),
        log_likelihood=noise_model.log_likelihood(predictions),
        valid=np.asarray(problem.last_evaluation.valid),
        predictions=predictions,
    )
    cost.write(directory)


def _record_step(directory, record) -> None:
    """One step's row of ``history.csv``, printed."""
    row = {name: np.asarray(value).item() for name, value in vars(record).items()}
    append_history(directory, row)
    print(
        f"step {row['step']}: beta {row['beta']:.4g} -> {row['beta_next']:.4g}, "
        f"misfit mean {row['misfit_mean']:.4g}, {row['n_valid']} valid",
        flush=True,
    )
