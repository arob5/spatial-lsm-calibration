"""The posterior predictive: EKI's final ensemble run against both observation vectors.

Overview
--------
Reads the ensemble an EKI run ended with (``run/eki.py``, beta = 1) and
runs every member through the forward model for the calibration and
validation predictions and for daily model output, as the prior predictive
runs the prior's draws, so the two predictives are written, and drawn, alike.
There is no one run by hand: a posterior's center is not a run it predicts.
The script then draws the posterior predictive's figures
(``figures/eki.py``) and diagnoses the run (``run/diagnose.py``):
its diagnostics tables and their figures. ``--no-diagnose`` skips the
diagnosis.

Input data
----------
``steps/`` of ``config.EKI_DIRECTORY / <data>``, of a finished run, and
everything the forward model reads.

Output data
-----------
Under ``config.EKI_DIRECTORY / <data> / "posterior_predictive"``: the daily
output, predictions, observed values and ``parameters.csv`` in
``run/_predictive.py``'s layout, and ``provenance.json``; the run's
``diagnostics/`` as ``run/diagnose.py`` writes it; and the figures, into
``config.FIGURE_DIRECTORY / config.EKI_RUN_NAME``. The observed
values are the calibration and validation vectors' own, also for
``--data synthetic``.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.posterior_predictive --data observed
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from .. import config
from ..figures.eki import draw_posterior_predictive_figures
from ..model import prior
from ..model.outputs import check_eki_run_finished, load_eki_run
from . import _predictive, _provenance
from .diagnose import diagnose_run

__all__ = ["main"]


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Run one EKI run's final ensemble, write its predictive, draw it and diagnose the run."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    data = arguments.data
    run_directory = config.EKI_DIRECTORY / data
    try:
        samples = final_ensemble(run_directory)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    vector, _, sipnet_map = prior.calibration()
    directory = run_directory / "posterior_predictive"
    _predictive.run_predictive(
        directory,
        vector,
        sipnet_map,
        prior.site_dims(),
        prior.external_inputs(),
        samples,
    )
    _provenance.write_provenance(
        directory / "provenance.json", input_files=_provenance.model_input_files()
    )
    print(f"wrote {directory}")
    draw_posterior_predictive_figures(data)
    if not arguments.no_diagnose:
        diagnose_run(data)
    return 0


# ── the steps ──


def final_ensemble(run_directory: Path) -> np.ndarray:
    """The ensemble a finished EKI run ended with, ``(J, D)``."""
    run = load_eki_run(run_directory)
    check_eki_run_finished(run, run_directory)
    return run["theta_posterior"]


# ── helpers ──


def _parser() -> argparse.ArgumentParser:
    """The command line: which EKI run, and whether to diagnose it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", choices=("synthetic", "observed"), required=True)
    parser.add_argument(
        "--no-diagnose",
        action="store_true",
        help="skip the run's diagnosis and its figures",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
