"""Redraw a stored run's figures from what it wrote.

Overview
--------
Draws every figure of one run without running anything: what the run's own
script draws at its end, and the figures of its diagnosis where it has
been diagnosed. The runs draw their figures themselves; this is for
redrawing them after a change to ``figures/``.

- ``--run prior``: the prior predictive's figures
  (``figures/prior_predictive.py``);
- ``--run synthetic``, ``--run observed``: the EKI run's
  (``figures/eki.py``), and its posterior predictive's where it has run;

and, for any run that has been diagnosed, its diagnostics' figures
(``figures/diagnostics.py``).

Input data
----------
The run's directory, as its script wrote it (``model/outputs.py`` reads
it).

Output data
-----------
The figures, under ``config.FIGURE_DIRECTORY`` for the prior predictive and
``config.FIGURE_DIRECTORY / config.EKI_RUN_NAME`` for an EKI run.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.draw_figures --run observed
    uv run python -m experiments.single_site_mcmc_vs_eki.run.draw_figures --run prior
"""

import argparse
import sys

from ..figures.diagnostics import draw_diagnostic_figures
from ..figures.eki import draw_eki_figures, draw_posterior_predictive_figures
from ..figures.prior_predictive import draw_prior_predictive_figures
from ..model.outputs import run_directory

__all__ = ["main"]


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Draw one stored run's figures."""
    run_name = _parser().parse_args(argv).run
    directory = run_directory(run_name)
    try:
        if run_name == "prior":
            draw_prior_predictive_figures()
        else:
            draw_eki_figures(run_name)
            if (directory / "posterior_predictive" / "ensemble_daily.nc").exists():
                draw_posterior_predictive_figures(run_name)
        if (directory / "diagnostics").is_dir():
            draw_diagnostic_figures(run_name)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


# ── helpers ──


def _parser() -> argparse.ArgumentParser:
    """The command line: which run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--run", choices=("prior", "synthetic", "observed"), required=True
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
