"""EKI: an ensemble from the prior, moved up the tempering ladder to the posterior.

Overview
--------
Runs ensemble Kalman inversion in its sampling form on the calibration's
posterior (``model/calibration.py``), through EnsKit's driver
(``enskit.algorithms.eki``) and the problem
:func:`sipnet_calibration.inference.eki_problem` reads from the posterior:
its forward map, ``y`` and ``R``.

- The initial ensemble is ``config.EKI_ENSEMBLE_SIZE`` draws of the prior.
- Each step conditions on ``y`` with the noise covariance ``R / delta`` by
  the stochastic Matheron update (``enskit.kalman.Matheron``), the increment
  ``delta`` chosen by ``AdaptiveESSSchedule`` so that the ensemble's
  effective sample size under it is ``config.EKI_ESS_FRACTION`` of J, until
  the increments sum to 1.
- A member whose sample is invalid, its run failed or its SIPNET parameters
  outside pySIPNET's domain, is moved to the valid members' center for that
  step (``on_failure="repair"``), and recorded.

``--data observed`` conditions on the calibration vector's observations.
``--data synthetic`` conditions on synthetic ones, ``y* = G(theta*) + e``,
``theta*`` one prior draw and ``e`` one draw of ``N(0, R)``
(``config.EKI_SYNTHETIC_TRUTH_SEED``), so the run can be checked against a
known truth.

Input data
----------
Everything the posterior reads (``run/prior_predictive.py`` lists it), and
``config``.

Output data
-----------
Under ``config.EKI_DIRECTORY / <data>``:

- ``history.csv``: one row per step, EnsKit's ``HistoryRecord`` (the level
  and increment, the misfits' mean, minimum and maximum, the mean
  prediction's misfit, the parameter spread, the effective sample size, the
  valid members), with the fraction of members out of pySIPNET's domain;
- ``initial_ensemble.npy``: the initial ensemble, theta ``(J, D)``, as drawn;
- ``steps/step_<k>.npz``: step ``k``'s evaluation, ``ensemble`` ``(J, D)``
  (failed members moved to the valid center), ``predictions`` ``(J, N)``
  in the posterior's y order, per-member ``misfits`` ``(J,)`` and
  ``valid`` ``(J,)``, whether the member's sample was valid, and the state
  after it, ``next_ensemble``, ``next_beta``, ``next_step`` and
  ``next_key`` (the key's data), from which ``--resume`` continues;
  ``steps/step_<k>_failures.csv``, the failed runs, when there are any;
- ``prior_ensemble.csv``, ``posterior_ensemble.csv``: the initial and the
  final ensembles' natural values, one row per member;
- ``synthetic`` only: ``truth.csv``, theta*'s natural values, and
  ``synthetic.npz``, ``theta_true`` ``(D,)`` and ``y`` ``(N,)``;
- ``calibration_parts.csv``, ``calibration_components.csv``,
  ``calibration_sipnet_parameters.csv`` and ``provenance.json``, as the prior
  predictive's.

The ladder ends when the increments reach beta = 1, without evaluating the
ensemble it ends with; the last step is then that ensemble's evaluation, at
beta = 1 with no update (its increment is 0), so its ``predictions`` are the
posterior ensemble's predictions of the calibration vector. A member
invalid there stops the run with an error, rather than being repaired, so
that it cannot enter the posterior ensemble.

Once the ladder reaches beta = 1, the run draws its ladder, its marginals
and, on synthetic data, its recovery of the truth (``figures/eki.py``) into
``config.FIGURE_DIRECTORY / config.EKI_RUN_NAME``. It is diagnosed by its
posterior predictive (``run/posterior_predictive.py``), which the held-out
tower's check reads.

A run started without ``--resume`` first removes what an earlier run of the
same setup and data left: its steps, history, posterior ensemble, posterior
predictive, diagnostics and discrepancy fit.

Notes
-----
The initial ensemble is the prior's own draws (``EKIProblem.initial_ensemble``),
so it differs from that of a run made before the probability layer, which
drew from a Gaussian built by hand: the same distribution, since every prior
factor is Gaussian in theta, but another random stream.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.eki --data synthetic
    uv run python -m experiments.single_site_mcmc_vs_eki.run.eki --data observed
    uv run python -m experiments.single_site_mcmc_vs_eki.run.eki --data observed --resume
"""

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import jax
import numpy as np
import pandas as pd
from enskit import kalman
from enskit.algorithms import eki
from enskit.distribution import Ensemble

from sipnet_calibration.inference import EKIProblem, eki_problem
from sipnet_calibration.probability import Posterior, condition_on
from sipnet_calibration.probability.names import THETA

from .. import config
from ..figures.eki import draw_eki_figures
from ..model import calibration, prior
from ..model.outputs import load_eki_run
from . import _provenance

__all__ = ["main"]


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Run, or resume, EKI on the chosen data and write every step."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    directory = config.EKI_DIRECTORY / arguments.data
    (directory / "steps").mkdir(parents=True, exist_ok=True)
    try:
        posterior = calibration.calibration_posterior()
        if arguments.data == "synthetic":
            posterior = synthetic_posterior(
                posterior, directory, resume=arguments.resume
            )
        problem = eki_problem(posterior)
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
        _write_start(directory, posterior, state)
    try:
        state = run_ladder(problem, state, directory)
    except eki.EKIError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    write_ensemble(
        directory / "posterior_ensemble.csv", posterior, state.ensemble[THETA]
    )
    _provenance.write_provenance(
        directory / "provenance.json", input_files=_provenance.model_input_files()
    )
    print(f"beta {float(state.beta):g} after {state.step} steps; wrote {directory}")
    draw_eki_figures(arguments.data)
    return 0


# ── the steps ──


def synthetic_posterior(
    posterior: Posterior, directory: Path, *, resume: bool
) -> Posterior:
    """*posterior*'s model conditioned on ``y* = G(theta*) + e``, made once and then read back.

    ``y*`` is one replicate of the observations at ``theta*``, a prior
    draw, so ``R`` is the calibration vector's.
    """
    path = directory / "synthetic.npz"
    if resume:
        check_file_exists(path)
        observed = posterior.observations.flat_to_values(np.load(path)["y"])
        return condition_on(posterior.model, observed)
    truth_key, noise_key = jax.random.split(
        jax.random.key(config.EKI_SYNTHETIC_TRUTH_SEED)
    )
    theta_true = posterior.sample_prior(truth_key, 1)
    replicated, computed = posterior.replicate(noise_key, theta_true)
    check_truth_run_succeeded(computed)
    synthetic = condition_on(
        posterior.model, {name: values[0] for name, values in replicated.items()}
    )
    np.savez(path, theta_true=np.asarray(theta_true[0]), y=np.asarray(synthetic.y))
    write_ensemble(directory / "truth.csv", posterior, theta_true, ["truth"])
    return synthetic


def initial_state(problem: EKIProblem, ensemble_size: int) -> eki.EKIState:
    """*ensemble_size* draws of the prior, and the run's own key."""
    ensemble_key, run_key = jax.random.split(jax.random.key(config.EKI_SEED))
    return eki.EKIState(
        problem.initial_ensemble(ensemble_key, ensemble_size), key=run_key
    )


def resumed_state(directory: Path) -> eki.EKIState:
    """The state after the last step written."""
    paths = sorted((directory / "steps").glob("step_*.npz"))
    check_some_step_was_written(paths, directory)
    last = np.load(paths[-1])
    return eki.EKIState(
        Ensemble({THETA: jax.numpy.asarray(last["next_ensemble"])}),
        key=jax.random.wrap_key_data(last["next_key"]),
        beta=float(last["next_beta"]),
        step=int(last["next_step"]),
    )


def run_ladder(
    problem: EKIProblem, state: eki.EKIState, directory: Path
) -> eki.EKIState:
    """Move *state* up the ladder to beta = 1, writing each step as it is taken,
    then evaluate the ensemble it ends with.

    Raises
    ------
    enskit.algorithms.eki.EKIError
        If fewer than two members of a step are valid, or any member of the
        final ensemble is not: the posterior ensemble is never written with
        an invalid sample in it.
    """
    arguments = (problem.forward, problem.y, problem.noise_covariance)
    steps = eki.iterate(
        state,
        *arguments,
        update_rule=kalman.Matheron(),
        schedule=eki.AdaptiveESSSchedule(ess_fraction=config.EKI_ESS_FRACTION),
        on_failure="repair",
    )
    for state, record, evaluation in steps:
        _write_step(directory, problem, state, record, evaluation)
    evaluation = eki.evaluate(state, *arguments)
    record = eki.HistoryRecord.from_evaluation(evaluation)
    _write_step(directory, problem, state, record, evaluation)
    return state


def write_ensemble(
    path: Path, posterior: Posterior, theta, labels: list[str] | None = None
) -> None:
    """An ensemble's natural values, one row per member."""
    natural = prior.natural_table(posterior, theta)
    natural.index = labels or [f"member_{i}" for i in range(len(natural))]
    natural.to_csv(path)


# ── helpers ──


def _write_start(directory: Path, posterior: Posterior, state: eki.EKIState) -> None:
    """The record of what runs, and the initial ensemble; an earlier run's outputs go."""
    for path in (directory / "steps").glob("step_*"):
        path.unlink()
    for name in (
        "history.csv",
        "posterior_ensemble.csv",
        "nee_discrepancy_fit.csv",
        *_provenance.CALIBRATION_FILE_NAMES,
    ):
        (directory / name).unlink(missing_ok=True)
    for name in ("posterior_predictive", "diagnostics"):
        shutil.rmtree(directory / name, ignore_errors=True)
    theta = state.ensemble[THETA]
    np.save(directory / "initial_ensemble.npy", np.asarray(theta))
    _provenance.write_calibration(directory, posterior)
    write_ensemble(directory / "prior_ensemble.csv", posterior, theta)


def _write_step(
    directory: Path,
    problem: EKIProblem,
    state: eki.EKIState,
    record: eki.HistoryRecord,
    evaluation: eki.Evaluation,
) -> None:
    """One step's evaluation and the state after it, and its row of the history."""
    step = int(record.step)
    runs = problem.last_evaluation.simulator_records[calibration.SIMULATOR_NAME]
    np.savez(
        directory / "steps" / f"step_{step:03d}.npz",
        ensemble=np.asarray(evaluation.ensemble[THETA]),
        predictions=np.asarray(evaluation.ensemble[eki.PREDICTION]),
        misfits=np.asarray(evaluation.misfits),
        next_ensemble=np.asarray(state.ensemble[THETA]),
        next_beta=float(state.beta),
        next_step=state.step,
        next_key=np.asarray(jax.random.key_data(state.key)),
        valid=np.asarray(problem.last_evaluation.valid),
    )
    if len(runs.failures):
        runs.failures.to_csv(directory / "steps" / f"step_{step:03d}_failures.csv")
    row = {name: np.asarray(value).item() for name, value in vars(record).items()}
    row["out_of_domain_fraction"] = runs.out_of_domain_fraction
    path = directory / "history.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    print(
        f"step {step}: beta {row['beta']:.4g} -> {row['beta_next']:.4g}, "
        f"misfit mean {row['misfit_mean']:.4g}, {row['n_valid']} valid",
        flush=True,
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


def check_truth_run_succeeded(computed: dict) -> None:
    """The synthetic truth's run succeeded, so every observation was replicated."""
    if not all(bool(np.all(valid)) for valid in computed.values()):
        raise ValueError(
            "the synthetic truth's run failed; change config.EKI_SYNTHETIC_TRUTH_SEED"
        )


if __name__ == "__main__":
    sys.exit(main())
