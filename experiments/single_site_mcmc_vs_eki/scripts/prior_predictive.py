"""The prior predictive: SIPNET run once, and as an ensemble, against the observations.

Overview
--------
Runs the calibration's prior (``model/prior.py``) two ways and writes what
came back, for ``figures/prior_predictive.py`` to draw:

1. **One run, by hand**, at the prior mean, through pySIPNET directly. It
   shows the layers the forward model composes: theta to SIPNET parameter
   fields (the SIPNET parameter map), fields to a run's keywords (SIPNET
   overrides), one ``SIPNETModel`` call, the run's output as model output,
   and the observation vector's operators on it.
2. **An ensemble**, ``config.PRIOR_PREDICTIVE_ENSEMBLE_SIZE`` draws of the
   prior, through :class:`~sipnet_calibration.forward.ForwardModel` and
   PyEns on local workers: once for the predictions of the calibration and
   validation observations, once for daily model output.

Every run is scored under the calibration's likelihood (``model/noise.py``).

Input data
----------
The prepared driver file (``scripts/prepare_drivers.py``), the processed
files ``model/inputs.py`` reads, and ``config``.

Output data
-----------
Under ``config.PRIOR_PREDICTIVE_DIRECTORY``: the daily output, predictions,
observed values and ``parameters.csv`` in ``scripts/predictive.py``'s
layout, the one run by hand as ``single_run``, and

- ``calibration_parameters.csv``, ``calibration_sipnet_parameters.csv``:
  the calibration's record (``scripts/provenance.py``'s
  ``write_calibration``);
- ``provenance.json``: the code, packages, command and inputs of the run
  (``scripts/provenance.py``).

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.scripts.prior_predictive
    uv run python -m experiments.single_site_mcmc_vs_eki.scripts.prior_predictive --ensemble-size 32
"""

import argparse
import sys
import warnings

import jax
import numpy as np

from .. import config
from ..model import inverse_problem, prior
from . import predictive, provenance

__all__ = ["main"]


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Run the prior predictive and write its outputs."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    vector, calibration_prior, sipnet_map = prior.calibration()
    directory = config.PRIOR_PREDICTIVE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    provenance.write_calibration(directory, vector, calibration_prior, sipnet_map)
    samples = calibration_prior.sample(
        jax.random.key(config.PRIOR_PREDICTIVE_SEED), arguments.ensemble_size
    )
    predictive.run_predictive(
        directory,
        vector,
        sipnet_map,
        prior.site_dims(),
        prior.external_inputs(),
        samples,
        center=np.asarray(inverse_problem.prior_gaussian(calibration_prior).mean),
    )
    provenance.write_provenance(
        directory / "provenance.json", input_files=provenance.model_input_files()
    )
    print(f"wrote {directory}")
    return 0


# ── helpers ──


def _parser() -> argparse.ArgumentParser:
    """The command line: the ensemble's size."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--ensemble-size", type=int, default=config.PRIOR_PREDICTIVE_ENSEMBLE_SIZE
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
