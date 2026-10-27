"""Draw every figure the slides show, from the saved results.

Overview
--------
Each entry of ``FIGURES`` names a figure file, the results it reads and the
``plots`` function that draws it. A figure whose results have not been
computed yet is skipped, with the script that computes them named.

Input data
----------
``config.TOY_RESULTS_FILE``, written by ``run/toy_experiment.py``;
``config.SITE_POSTERIORS_FILE``, written by ``run/site_posteriors.py``; and
``config.NORTH_AMERICA_FILE``,
``config.UNCERTAINTY_FILE`` and ``config.CALIBRATION_OPTIONS_FILE``, written
by ``run/illustrations.py``.

Output data
-----------
One PNG per figure under ``config.FIGURE_DIRECTORY``, transparent, so it
sits on the slide's background.

Usage
-----
::

    uv run python -m experiments.presentation_aimes_talk.run.draw_figures
"""

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import config, plots, results  # noqa: E402
from ..style import apply_style  # noqa: E402


@dataclass(frozen=True)
class FigureSpec:
    """One figure: the file it is saved as, what it reads and what draws it."""

    file_name: str
    results_file: Path
    draw: Callable
    keywords: dict = field(default_factory=dict)


#: Which results file each results type lives in, and the script that writes it.
RESULT_TYPES = {
    config.TOY_RESULTS_FILE: (results.ToyResults, "run/toy_experiment.py"),
    config.SITE_POSTERIORS_FILE: (results.SitePosteriors, "run/site_posteriors.py"),
    config.NORTH_AMERICA_FILE: (results.NorthAmericaPrior, "run/illustrations.py"),
    config.CALIBRATION_OPTIONS_FILE: (results.CalibrationOptions, "run/illustrations.py"),
    config.UNCERTAINTY_FILE: (results.UncertaintyMaps, "run/illustrations.py"),
}

PRODUCTIVITY, TURNOVER = 0, 1

FIGURES = (
    FigureSpec("site_pool_map.png", config.NORTH_AMERICA_FILE, plots.site_pool_map),
    FigureSpec("option_independent.png", config.CALIBRATION_OPTIONS_FILE, plots.option_independent_map),
    FigureSpec("option_pft.png", config.CALIBRATION_OPTIONS_FILE, plots.option_pft_map),
    FigureSpec("option_spatial.png", config.NORTH_AMERICA_FILE, plots.option_spatial_map),
    FigureSpec("north_america_prior_draws.png", config.NORTH_AMERICA_FILE, plots.north_america_prior_draws),
    FigureSpec("uncertainty_maps.png", config.UNCERTAINTY_FILE, plots.uncertainty_maps),
    FigureSpec("toy_decomposition.png", config.TOY_RESULTS_FILE, plots.toy_decomposition, {"parameter": TURNOVER}),
    FigureSpec("toy_world.png", config.TOY_RESULTS_FILE, plots.toy_world, {"parameter": TURNOVER}),
    FigureSpec("posterior_tower.png", config.SITE_POSTERIORS_FILE, plots.site_posterior_comparison,
               {"which": "tower"}),
    FigureSpec("posterior_sparse.png", config.SITE_POSTERIORS_FILE, plots.site_posterior_comparison,
               {"which": "sparse"}),
    FigureSpec("maps_estimate_turnover.png", config.TOY_RESULTS_FILE, plots.maps_estimate, {"parameter": TURNOVER}),
    FigureSpec("maps_estimate_productivity.png", config.TOY_RESULTS_FILE, plots.maps_estimate,
               {"parameter": PRODUCTIVITY}),
    FigureSpec("held_out_errors.png", config.TOY_RESULTS_FILE, plots.held_out_errors),
    FigureSpec("ipl_agreement_turnover.png", config.TOY_RESULTS_FILE, plots.ipl_agreement, {"parameter": TURNOVER}),
    FigureSpec("cost_bars.png", config.TOY_RESULTS_FILE, plots.cost_bars),
    FigureSpec("ipl_convergence.png", config.TOY_RESULTS_FILE, plots.ipl_convergence),
)


def main() -> None:
    apply_style(font_scale=1.15)
    config.FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for spec in FIGURES:
        if not spec.results_file.exists():
            print(f"skipped {spec.file_name}: run {RESULT_TYPES[spec.results_file][1]} first")
            continue
        figure = spec.draw(_load(spec.results_file), **spec.keywords)
        path = config.FIGURE_DIRECTORY / spec.file_name
        figure.savefig(path)
        plt.close(figure)
        print(f"wrote {path.name}")


@cache
def _load(results_file: Path):
    result_type, _ = RESULT_TYPES[results_file]
    return results.load(result_type, results_file)


if __name__ == "__main__":
    main()
