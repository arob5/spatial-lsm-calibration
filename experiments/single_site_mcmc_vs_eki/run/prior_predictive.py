"""The prior predictive: SIPNET run once, and as an ensemble, against the observations.

Overview
--------
Runs the calibration's prior (``model/prior.py``) two ways and writes what
came back:

1. **One run, by hand**, at the prior's center in theta, through pySIPNET
   directly. It shows the layers the forward map composes: theta to what
   the simulator reads, those values to SIPNET parameter fields (the SIPNET
   parameter map), fields to a run's keywords (SIPNET overrides), one
   ``SIPNETModel`` call, the run's output as model output, and the
   observation vectors' operators on it.
2. **An ensemble**, ``config.PRIOR_PREDICTIVE_ENSEMBLE_SIZE`` draws of the
   prior, run once through PyEns on local workers, each run reduced by the
   calibration and the held-out vectors' operators and to daily output.

The model, ``--model`` (``long_memory/fixed`` by default), sets the noise
model: every run is scored under its fixed posterior's likelihood, every
scale at 1 (``model/likelihood.py``). The script then draws the prior
predictive's figures under every model's noise (``figures/prior_predictive.py``) and diagnoses it
(``run/diagnose.py``); ``--no-diagnose`` skips the diagnosis.

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

- ``calibration_parts.csv``, ``calibration_components.csv``,
  ``calibration_sipnet_parameters.csv``: the calibration's record
  (``run/_provenance.py``'s ``write_calibration``);
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

from .. import config
from ..figures.prior_predictive import draw_prior_predictive_figures
from ..model import prior
from ..models import MODEL_NAMES, Model, fixed_posterior, heldout_posterior
from . import _predictive, _provenance
from .diagnose import diagnose_run

__all__ = ["main"]


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Run the prior predictive, write its outputs, draw them and diagnose it."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    model = Model.parse(arguments.model)
    posterior = fixed_posterior(model)
    directory = config.PRIOR_PREDICTIVE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    _provenance.write_calibration(directory, posterior)
    theta = posterior.sample_prior(
        jax.random.key(config.PRIOR_PREDICTIVE_SEED), arguments.ensemble_size
    )
    _predictive.run_predictive(
        directory,
        posterior,
        heldout_posterior(model),
        theta,
        center=prior.prior_center(posterior),
    )
    _provenance.write_provenance(
        directory / "provenance.json", input_files=_provenance.model_input_files()
    )
    print(f"wrote {directory}")
    for name in MODEL_NAMES:
        draw_prior_predictive_figures(Model.parse(name))
    if not arguments.no_diagnose:
        diagnose_run(model.name, "prior")
    return 0


# ── helpers ──


def _parser() -> argparse.ArgumentParser:
    """The command line: the model, the ensemble's size, and whether to diagnose."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=MODEL_NAMES, default="long_memory/fixed")
    parser.add_argument(
        "--ensemble-size", type=int, default=config.PRIOR_PREDICTIVE_ENSEMBLE_SIZE
    )
    parser.add_argument(
        "--no-diagnose",
        action="store_true",
        help="skip the run's diagnosis",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
