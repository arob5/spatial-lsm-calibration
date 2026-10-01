"""The prior predictive: SIPNET run once, and as an ensemble, against the observations.

Overview
--------
Runs the calibration's prior (``model/prior.py``) two ways and writes what
came back:

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
The script then draws the prior predictive's figures
(``figures/prior_predictive.py``) and diagnoses it (``run/diagnose.py``):
its diagnostics tables and their figures. ``--no-diagnose`` skips the
diagnosis.

Input data
----------
The prepared driver file (``run/prepare_drivers.py``), the processed
files ``model/inputs.py`` reads, and ``config``.

Output data
-----------
Under ``config.PRIOR_PREDICTIVE_DIRECTORY``: the daily output, predictions,
observed values and ``parameters.csv`` in ``run/_predictive.py``'s
layout, the one run by hand as ``single_run``, ``diagnostics/`` as
``run/diagnose.py`` writes it, and

- ``calibration_parameters.csv``, ``calibration_sipnet_parameters.csv``:
  the calibration's record (``run/_provenance.py``'s
  ``write_calibration``);
- ``provenance.json``: the code, packages, command and inputs of the run
  (``run/_provenance.py``).

The figures go into ``config.FIGURE_DIRECTORY``.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.prior_predictive
    uv run python -m experiments.single_site_mcmc_vs_eki.run.prior_predictive --ensemble-size 32
"""

import argparse
import sys
import warnings

import jax
import numpy as np

from .. import config
from ..figures.prior_predictive import draw_prior_predictive_figures
from ..model import inverse_problem, prior
from . import _predictive, _provenance
from .diagnose import diagnose_run

__all__ = ["main"]


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Run the prior predictive, write its outputs, draw them and diagnose it."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    vector, calibration_prior, sipnet_map = prior.calibration()
    directory = config.PRIOR_PREDICTIVE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    _provenance.write_calibration(directory, vector, calibration_prior, sipnet_map)
    samples = calibration_prior.sample(
        jax.random.key(config.PRIOR_PREDICTIVE_SEED), arguments.ensemble_size
    )
    _predictive.run_predictive(
        directory,
        vector,
        sipnet_map,
        prior.site_dims(),
        prior.external_inputs(),
        samples,
        center=np.asarray(inverse_problem.prior_gaussian(calibration_prior).mean),
    )
    _provenance.write_provenance(
        directory / "provenance.json", input_files=_provenance.model_input_files()
    )
    print(f"wrote {directory}")
    draw_prior_predictive_figures()
    if not arguments.no_diagnose:
        diagnose_run("prior")
    return 0


# ── helpers ──


def _parser() -> argparse.ArgumentParser:
    """The command line: the ensemble's size, and whether to diagnose."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--ensemble-size", type=int, default=config.PRIOR_PREDICTIVE_ENSEMBLE_SIZE
    )
    parser.add_argument(
        "--no-diagnose",
        action="store_true",
        help="skip the run's diagnosis and its figures",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
