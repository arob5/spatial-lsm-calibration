"""The runs compared: one table per comparison, across every model and run.

Overview
--------
Builds ``analysis/compare.py``'s tables from every run under
``config.RUNS_DIRECTORY`` (runs not yet written are left out), writes them
and prints them. Runs no model.

Input data
----------
Each run's ``samples.nc``, ``natural_values.csv`` and ``cost.json``
(``algorithms/records.py``), and its ``predictive/heldout_scores.csv``
where ``run/predict.py`` has run.

Output data
-----------
Under ``comparison/`` beside ``config.RUNS_DIRECTORY`` (``output/comparison``
by default): ``cost.csv``, ``parameters.csv``, ``against_reference.csv``,
``noise_scales.csv``, ``reweighting.csv`` and ``heldout_scores.csv``, as
``analysis/compare.py``'s tables of those names describe them; and, per
model, ``figures/algorithms.py``'s figures under
``config.FIGURE_DIRECTORY / "algorithms"``, and ``figures/error_models.py``'s
under ``config.FIGURE_DIRECTORY / "error_models"`` (``--no-figures`` skips
them).

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.compare
"""

import argparse
import sys

import pandas as pd

from .. import config
from ..analysis.compare import comparison_tables
from ..figures.algorithms import draw_algorithm_figures
from ..figures.error_models import draw_error_model_figures
from ..models import MODEL_NAMES, Model

__all__ = ["COMPARISON_DIRECTORY", "main"]

#: Where the tables go: beside the runs, so a smoke test's stay with its runs.
COMPARISON_DIRECTORY = config.RUNS_DIRECTORY.parent / "comparison"


def main(argv: list[str] | None = None) -> int:
    """Build every comparison table, write it and print it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-figures", action="store_true", help="write the tables only")
    arguments = parser.parse_args(argv)
    tables = comparison_tables(config.RUNS_DIRECTORY)
    COMPARISON_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with pd.option_context(
        "display.width", 200, "display.max_columns", 20, "display.precision", 3
    ):
        for name, table in tables.items():
            table.to_csv(COMPARISON_DIRECTORY / f"{name}.csv")
            print(f"\n{name}\n{table.to_string()}")
    print(f"\nwrote {COMPARISON_DIRECTORY}")
    if not arguments.no_figures:
        for name in MODEL_NAMES:
            draw_algorithm_figures(Model.parse(name))
        draw_error_model_figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
