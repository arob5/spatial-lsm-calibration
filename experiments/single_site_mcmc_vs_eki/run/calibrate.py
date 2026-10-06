"""Calibrate one model with one algorithm, and write the run.

Overview
--------
Runs an algorithm of ``algorithms/`` on a model of ``models.py`` and writes
the run under ``config.RUNS_DIRECTORY / <nee error model> / <noise> /
<run name>`` in the format of ``algorithms/records.py``. The algorithms:

- ``eki``: EKI, for fixed noise;
- ``eki_gibbs_common``, ``eki_gibbs_per_particle``: EKI with Gibbs draws of
  the noise scales, for inferred noise, the two versions of the Kalman gain;
- ``is``, ``smc``: importance sampling or tempered SMC from a density fitted
  to the run ``--from`` names (an EKI run of the same model), written as
  ``<from>_is`` or ``<from>_smc``;
- ``mcmc``: parallel random-walk Metropolis, started from the importance
  sampling run ``--from`` names.

Input data
----------
Everything the model reads (``run/check_inputs.py`` lists it); for ``is``,
``smc`` and ``mcmc``, the run they start from.

Output data
-----------
The run's directory: ``samples.nc``, ``natural_values.csv``, ``cost.json``,
``history.csv``, ``provenance.json`` and the calibration's record
(``run/_provenance.py``); EKI also ``initial_ensemble.npy``, SMC
``smc_state``, MCMC ``chains.npz`` and ``checkpoint.npz``. A run started
without ``--resume`` first empties its directory.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.calibrate --model long_memory/fixed --algorithm eki
    uv run python -m experiments.single_site_mcmc_vs_eki.run.calibrate --model long_memory/inferred --algorithm eki_gibbs_common
    uv run python -m experiments.single_site_mcmc_vs_eki.run.calibrate --model long_memory/inferred --algorithm is --from eki_gibbs_common
    uv run python -m experiments.single_site_mcmc_vs_eki.run.calibrate --model long_memory/inferred --algorithm mcmc --from eki_gibbs_common_is --resume
"""

import argparse
import shutil
import sys
import warnings

from .. import config
from ..algorithms.eki import run_eki
from ..algorithms.eki_gibbs import run_eki_gibbs
from ..algorithms.mcmc import run_mcmc
from ..algorithms.reweighting import run_reweighting
from ..model.likelihood import NoiseModel
from ..models import MODEL_NAMES, Model, fixed_posterior
from . import _provenance

__all__ = ["ALGORITHMS", "main", "run_name"]

#: The algorithms, and the noise treatment each applies to (None: either).
ALGORITHMS = {
    "eki": "fixed",
    "eki_gibbs_common": "inferred",
    "eki_gibbs_per_particle": "inferred",
    "is": None,
    "smc": None,
    "mcmc": None,
}


def main(argv: list[str] | None = None) -> int:
    """Run the algorithm on the model and write the run."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    model = Model.parse(arguments.model)
    algorithm = arguments.algorithm
    if ALGORITHMS[algorithm] not in (None, model.noise):
        sys.exit(f"error: {algorithm} is for models with {ALGORITHMS[algorithm]} noise")
    if algorithm in ("is", "smc", "mcmc") and arguments.source is None:
        sys.exit(f"error: {algorithm} needs --from, the run it starts from")
    directory = model.directory(run_name(algorithm, arguments.source))
    if not arguments.resume:
        shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"{model.name}, {directory.name}: writing {directory}", flush=True)
    posterior = fixed_posterior(model)
    noise_model = NoiseModel(posterior, inferred=model.noise == "inferred")
    _provenance.write_calibration(directory, posterior)
    source = model.directory(arguments.source) if arguments.source else None
    if algorithm == "eki":
        run_eki(model, posterior, noise_model, directory, ensemble_size=arguments.ensemble_size)
    elif algorithm.startswith("eki_gibbs_"):
        run_eki_gibbs(
            model,
            posterior,
            noise_model,
            directory,
            gain=algorithm.removeprefix("eki_gibbs_"),
            ensemble_size=arguments.ensemble_size,
        )
    elif algorithm in ("is", "smc"):
        default = config.IMPORTANCE_SAMPLE_SIZE if algorithm == "is" else config.SMC_SAMPLE_SIZE
        run_reweighting(
            model,
            posterior,
            noise_model,
            directory,
            seed_directory=source,
            method=algorithm,
            n_samples=arguments.samples or default,
        )
    else:
        run_mcmc(
            model,
            posterior,
            noise_model,
            directory,
            start_directory=source,
            n_chains=arguments.chains or config.MCMC_CHAINS_PER_WORKER * config.N_WORKERS,
            n_steps=arguments.steps or config.MCMC_STEPS,
            resume=arguments.resume,
        )
    _provenance.write_provenance(
        directory / "provenance.json", input_files=_provenance.model_input_files()
    )
    print(f"wrote {directory}", flush=True)
    return 0


def run_name(algorithm: str, source: str | None) -> str:
    """The run's directory name: the algorithm's, or ``<from>_is`` / ``<from>_smc``."""
    return f"{source}_{algorithm}" if algorithm in ("is", "smc") else algorithm


def _parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), required=True)
    parser.add_argument("--from", dest="source", help="the run is, smc or mcmc start from")
    parser.add_argument("--ensemble-size", type=int, default=config.EKI_ENSEMBLE_SIZE)
    parser.add_argument("--samples", type=int, help="is/smc sample size")
    parser.add_argument("--chains", type=int, help="mcmc chains")
    parser.add_argument("--steps", type=int, help="mcmc steps per chain")
    parser.add_argument("--resume", action="store_true", help="mcmc: continue from the checkpoint")
    return parser


if __name__ == "__main__":
    sys.exit(main())
