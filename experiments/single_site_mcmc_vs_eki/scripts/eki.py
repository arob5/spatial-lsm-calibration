"""EKI: an ensemble from the prior, moved up the tempering ladder to the posterior.

Overview
--------
Runs ensemble Kalman inversion in its sampling form on the calibration's
inverse problem (``model/inverse_problem.py``):

- the initial ensemble is ``config.EKI_ENSEMBLE_SIZE`` draws of the prior,
  which is exactly Gaussian in theta;
- each step conditions on ``y`` with the noise covariance ``R / delta``, by
  pyEKI's perturbed-observation update (``PathwiseUpdate``, the stochastic
  Matheron update), the increment ``delta`` chosen by
  ``AdaptiveESSSchedule`` so that the ensemble's effective sample size under
  it is ``config.EKI_ESS_FRACTION`` of J, until the increments sum to 1;
- a member whose run fails is moved to the valid members' center for that
  step (pyEKI's ``on_failure="repair"``), and so is one whose SIPNET
  parameters leave pySIPNET's domain (the forward model's
  ``out_of_domain="fail_row"``); both are recorded.

``--data observed`` conditions on the calibration vector's observations.
``--data synthetic`` conditions on synthetic ones, ``y* = G(theta*) + e``,
with ``theta*`` one prior draw and ``e`` one draw of ``N(0, R)``
(``config.EKI_SYNTHETIC_TRUTH_SEED``), so the run can be checked against a
known truth.

Input data
----------
Everything the forward model reads (``scripts/prior_predictive.py`` lists
it), and ``config``.

Output data
-----------
Under ``config.EKI_DIRECTORY / <data>``:

- ``history.csv``: one row per step, pyEKI's ``HistoryRecord`` (the level
  and increment, the misfits' mean, minimum and maximum, the mean
  prediction's misfit, the parameter spread, the effective sample size, the
  valid members), with the fraction of members out of pySIPNET's domain;
- ``initial_ensemble.npy``: the initial ensemble, theta ``(J, D)``, as drawn;
- ``steps/step_<k>.npz``: step ``k``'s evaluation, ``ensemble`` ``(J, D)``
  (failed members moved to the valid center), ``predictions`` ``(J, N)``,
  per-member ``misfits`` ``(J,)`` and ``valid`` ``(J,)``, whether the
  member's run succeeded, and the state after it, ``next_ensemble``,
  ``next_beta``, ``next_step`` and ``next_key`` (the key's data), from which
  ``--resume`` continues;
  ``steps/step_<k>_failures.csv``, the failed runs, when there are any;
- ``prior_ensemble.csv``, ``posterior_ensemble.csv``: the initial and the
  final ensembles' natural values, one row per member;
- ``synthetic`` only: ``truth.csv``, theta*'s natural values, and
  ``synthetic.npz``, ``theta_true`` ``(D,)``, ``predictions_true`` and
  ``y`` ``(N,)``;
- ``calibration_parameters.csv``, ``calibration_sipnet_parameters.csv`` and
  ``provenance.json``, as the prior predictive's.

A run started without ``--resume`` first removes what an earlier run of the
same setup and data left: its steps, history, posterior ensemble, posterior
predictive, diagnostics and discrepancy fit.

The last step is an evaluation of the final ensemble, at beta = 1, with no
update (its increment is 0), so its ``predictions`` are the posterior
ensemble's predictions of the calibration vector.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.scripts.eki --data synthetic
    uv run python -m experiments.single_site_mcmc_vs_eki.scripts.eki --data observed
    uv run python -m experiments.single_site_mcmc_vs_eki.scripts.eki --data observed --resume
"""

import argparse
import dataclasses
import shutil
import sys
import warnings
from pathlib import Path

import jax
import numpy as np
import pandas as pd
from pyeki.eki import (
    AdaptiveESSSchedule,
    EKIState,
    HistoryRecord,
    PathwiseUpdate,
    evaluate,
    iterate,
    misfits,
)
from pyeki.gauss import Gaussian

from .. import config
from ..model import inverse_problem, prior
from ..model.inverse_problem import InverseProblem
from ..model.outputs import load_eki_run
from . import provenance

__all__ = ["main"]


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Run, or resume, EKI on the chosen data and write every step."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    directory = config.EKI_DIRECTORY / arguments.data
    (directory / "steps").mkdir(parents=True, exist_ok=True)
    problem = inverse_problem.calibration_problem()
    try:
        if arguments.data == "synthetic":
            problem = synthetic_problem(problem, directory, resume=arguments.resume)
        state = (
            resumed_state(directory)
            if arguments.resume
            else initial_state(problem, arguments.ensemble_size)
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if arguments.resume and load_eki_run(directory)["finished"]:
        print(f"the run under {directory} is finished; nothing to resume")
        return 0
    if not arguments.resume:
        _write_start(directory, problem, state)
    state = run_ladder(problem, state, directory)
    write_ensemble(directory / "posterior_ensemble.csv", problem, state.ensemble)
    provenance.write_provenance(
        directory / "provenance.json", input_files=provenance.model_input_files()
    )
    print(f"beta {float(state.beta):g} after {state.step} steps; wrote {directory}")
    return 0


# ── the steps ──


def synthetic_problem(
    problem: InverseProblem, directory: Path, *, resume: bool
) -> InverseProblem:
    """The problem conditioned on ``y* = G(theta*) + e``, made once and then read back."""
    path = directory / "synthetic.npz"
    if resume:
        check_file_exists(path)
        return problem.with_observations(np.load(path)["y"])
    truth_key, noise_key = jax.random.split(
        jax.random.key(config.EKI_SYNTHETIC_TRUTH_SEED)
    )
    theta_true = np.asarray(problem.prior.sample(truth_key, 1))[0]
    predictions_true = np.asarray(problem.forward_model()(theta_true))
    check_truth_run_succeeded(predictions_true)
    noise = Gaussian(jax.numpy.zeros(problem.y.shape), problem.noise_covariance)
    y = predictions_true + np.asarray(noise.sample(noise_key, 1))[0]
    np.savez(path, theta_true=theta_true, predictions_true=predictions_true, y=y)
    write_ensemble(directory / "truth.csv", problem, theta_true[None, :], ["truth"])
    return problem.with_observations(y)


def initial_state(problem: InverseProblem, ensemble_size: int) -> EKIState:
    """*ensemble_size* draws of the prior, and the run's own key."""
    return EKIState.from_prior(
        jax.random.key(config.EKI_SEED), problem.prior_gaussian, ensemble_size
    )


def resumed_state(directory: Path) -> EKIState:
    """The state after the last step written."""
    paths = sorted((directory / "steps").glob("step_*.npz"))
    check_some_step_was_written(paths, directory)
    last = np.load(paths[-1])
    return EKIState(
        ensemble=jax.numpy.asarray(last["next_ensemble"]),
        beta=float(last["next_beta"]),
        step=int(last["next_step"]),
        key=jax.random.wrap_key_data(last["next_key"]),
    )


def run_ladder(problem: InverseProblem, state: EKIState, directory: Path) -> EKIState:
    """Move *state* up the ladder to beta = 1, writing each step as it is taken."""
    forward = _RecordingForward(problem.forward_model(out_of_domain="fail_row"))
    steps = iterate(
        state,
        forward,
        problem.y,
        problem.noise_covariance,
        schedule=AdaptiveESSSchedule(
            beta_target=1.0, ess_fraction=config.EKI_ESS_FRACTION
        ),
        update=PathwiseUpdate(),
        on_failure="repair",
    )
    for state, record, evaluation in steps:
        _write_step(directory, problem, state, record, evaluation, forward.last)
    # The ladder ends when the increments reach beta = 1, without evaluating
    # the ensemble it ends with; evaluate it once more, as the record of the
    # posterior ensemble's predictions.
    evaluation = evaluate(state, forward, problem.y, problem.noise_covariance)
    record = _terminal_record(problem, evaluation)
    _write_step(directory, problem, state, record, evaluation, forward.last)
    return state


def write_ensemble(
    path: Path,
    problem: InverseProblem,
    theta,
    labels: list[str] | None = None,
) -> None:
    """An ensemble's natural values, one row per member."""
    theta = np.asarray(theta)
    natural = prior.natural_table(problem.parameter_vector, theta)
    natural.index = labels or [f"member_{i}" for i in range(theta.shape[0])]
    natural.to_csv(path)


# ── helpers ──


class _RecordingForward:
    """The forward model as pyEKI calls it, keeping each call's evaluation."""

    def __init__(self, forward_model):
        self.forward_model = forward_model
        self.last = None

    def __call__(self, theta):
        self.last = self.forward_model.evaluate(theta)
        return self.last.predictions


def _write_start(directory: Path, problem: InverseProblem, state: EKIState) -> None:
    """The record of what runs, and the initial ensemble; an earlier run's outputs go."""
    for path in (directory / "steps").glob("step_*"):
        path.unlink()
    for name in (
        "history.csv",
        "posterior_ensemble.csv",
        "nee_discrepancy_fit.csv",
        "calibration.csv",
    ):
        (directory / name).unlink(missing_ok=True)
    for name in ("posterior_predictive", "diagnostics"):
        shutil.rmtree(directory / name, ignore_errors=True)
    np.save(directory / "initial_ensemble.npy", np.asarray(state.ensemble))
    provenance.write_calibration(
        directory, problem.parameter_vector, problem.prior, problem.sipnet_parameter_map
    )
    write_ensemble(directory / "prior_ensemble.csv", problem, state.ensemble)


def _write_step(directory, problem, state, record, evaluation, forward_evaluation):
    """One step's evaluation and the state after it, and its row of the history."""
    step = int(record.step)
    np.savez(
        directory / "steps" / f"step_{step:03d}.npz",
        ensemble=np.asarray(evaluation.ensemble),
        predictions=np.asarray(evaluation.predictions),
        misfits=np.asarray(
            misfits(problem.y, evaluation.predictions, problem.noise_covariance)
        ),
        next_ensemble=np.asarray(state.ensemble),
        next_beta=float(state.beta),
        next_step=state.step,
        next_key=np.asarray(jax.random.key_data(state.key)),
        valid=np.asarray(forward_evaluation.valid),
    )
    failures = forward_evaluation.failures
    if len(failures):
        failures.to_csv(directory / "steps" / f"step_{step:03d}_failures.csv")
    row = {
        field.name: np.asarray(getattr(record, field.name)).item()
        for field in dataclasses.fields(record)
    }
    row["out_of_domain_fraction"] = forward_evaluation.out_of_domain_fraction
    path = directory / "history.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    print(
        f"step {step}: beta {row['beta']:.4g} -> {row['beta_next']:.4g}, "
        f"misfit mean {row['misfit_mean']:.4g}, {row['n_valid']} valid",
        flush=True,
    )


def _terminal_record(problem: InverseProblem, evaluation) -> HistoryRecord:
    """The history row of an evaluation no update follows: increment 0, ESS J."""
    member_misfits = misfits(
        problem.y, evaluation.predictions, problem.noise_covariance
    )
    center = misfits(
        problem.y,
        evaluation.predictions.mean(axis=0),
        problem.noise_covariance,
    )
    zero = jax.numpy.zeros_like(evaluation.beta)
    return HistoryRecord(
        step=jax.numpy.asarray(evaluation.step),
        n_valid=evaluation.n_valid,
        beta=evaluation.beta,
        increment=zero,
        beta_next=evaluation.beta,
        misfit_mean=member_misfits.mean(),
        misfit_min=member_misfits.min(),
        misfit_max=member_misfits.max(),
        centre_misfit=center,
        spread=evaluation.rms_parameter_spread,
        ess=jax.numpy.asarray(float(evaluation.ensemble.shape[0])),
    )


def _parser() -> argparse.ArgumentParser:
    """The command line: which data, the ensemble's size, and whether to resume."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", choices=("synthetic", "observed"), required=True)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=config.EKI_ENSEMBLE_SIZE,
        help="J; a resumed run keeps its own",
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue from the last step written"
    )
    return parser


# ── checks ──


def check_file_exists(path: Path) -> None:
    """A file a resumed run reads exists."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run without --resume")


def check_some_step_was_written(paths: list[Path], directory: Path) -> None:
    """A resumed run has a step to resume from."""
    if not paths:
        raise FileNotFoundError(
            f"no step was written under {directory / 'steps'}; run without --resume"
        )


def check_truth_run_succeeded(predictions) -> None:
    """The synthetic truth's run succeeded, so its predictions are finite."""
    if not np.isfinite(predictions).all():
        raise ValueError(
            "the synthetic truth's run failed; change config.EKI_SYNTHETIC_TRUTH_SEED"
        )


if __name__ == "__main__":
    sys.exit(main())
