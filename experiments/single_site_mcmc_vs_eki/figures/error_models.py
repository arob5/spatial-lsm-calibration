"""Figures comparing the NEE error models under one algorithm.

- :func:`plot_error_model_nee` (``error_model_nee_<noise>``): NEE's seasonal
  cycle, each error model's posterior predictive (the 90% band and median
  over samples of the replicated data's weekly means) against the observed
  weekly means, at the calibration tower and the held-out one.

The runs are the reference algorithm's (:data:`REFERENCE_ALGORITHM`, MCMC) of
the three error models with one noise treatment; the replicated data are
``figures/algorithms.py``'s, noise included. :func:`draw_error_model_figures`
draws the figure for each noise treatment into
``config.FIGURE_DIRECTORY / "error_models"``.
"""

from collections.abc import Mapping

import matplotlib.pyplot as plt

from sipnet_calibration.conventions import TIME
from sipnet_calibration.plotting.style import use_project_style

from .. import config
from ..models import NEE_ERROR_MODELS, NOISE_TREATMENTS, Model
from .algorithms import (
    COMPARED_ALGORITHMS,
    PREDICTIVE_VECTORS,
    AlgorithmRun,
    draw_band,
    load_algorithm_runs,
    weekly_band,
)
from .common import NEE_TITLES, SLIDE_STYLE, one_legend, save_figure, week_of_year

__all__ = [
    "ERROR_MODEL_COLORS",
    "ERROR_MODEL_LABELS",
    "REFERENCE_ALGORITHM",
    "draw_error_model_figures",
    "load_error_model_runs",
    "plot_error_model_nee",
]

#: The algorithm whose runs the figures compare.
REFERENCE_ALGORITHM = next(a for a in COMPARED_ALGORITHMS if a.label == "MCMC")

#: Each error model's legend label.
ERROR_MODEL_LABELS = {
    "short_memory": "short memory",
    "long_memory": "long memory",
    "recurring_bias": "recurring bias",
}

#: Each error model's color.
ERROR_MODEL_COLORS = {
    "short_memory": "#D55E00",
    "long_memory": "#0072B2",
    "recurring_bias": "#009E73",
}


def load_error_model_runs(noise_treatment: str) -> dict[str, AlgorithmRun]:
    """The reference algorithm's run of each error model with
    *noise_treatment*, with its replicated data; those written so far."""
    runs = {}
    for name in NEE_ERROR_MODELS:
        loaded = load_algorithm_runs(Model(name, noise_treatment), (REFERENCE_ALGORITHM,))
        if loaded and loaded[0].replicated is not None:
            runs[name] = loaded[0]
    return runs


def plot_error_model_nee(runs: Mapping[str, AlgorithmRun]) -> plt.Figure:
    """NEE by week of year: each error model's posterior predictive, the 90%
    band and median over samples of the replicated data's weekly means,
    against the observed weekly means; the calibration tower above, the
    held-out tower below."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, squeeze=False)
    any_run = next(iter(runs.values()))
    for row, (vector, vector_label) in enumerate(PREDICTIVE_VECTORS.items()):
        for column, name in enumerate(config.NEE_WINDOWS):
            ax = axes[row, column]
            observed = any_run.predictive["observed"][vector][name]["value"]
            for error_model, run in runs.items():
                band = weekly_band(run.replicated[vector][name], observed[TIME])
                draw_band(
                    ax,
                    band.index,
                    band,
                    ERROR_MODEL_COLORS[error_model],
                    ERROR_MODEL_LABELS[error_model],
                )
            weekly = observed.to_series().groupby(week_of_year(observed[TIME])).mean()
            ax.plot(weekly.index, weekly.to_numpy(), "o", color="black", markersize=3.5, label="observed")
            ax.axhline(0.0, color="#999999", linewidth=0.6, zorder=0)
            ax.set_title(f"{NEE_TITLES[name]}\n{vector_label}")
        axes[row, 0].set_ylabel("NEE (µmol CO₂ m⁻² s⁻¹), weekly mean")
    for ax in axes[-1]:
        ax.set_xlabel("week of year")
    one_legend(figure, axes)
    return figure


def draw_error_model_figures() -> None:
    """Draw :func:`plot_error_model_nee` for each noise treatment."""
    directory = config.FIGURE_DIRECTORY / "error_models"
    use_project_style()
    with plt.rc_context(SLIDE_STYLE):
        for noise_treatment in NOISE_TREATMENTS:
            runs = load_error_model_runs(noise_treatment)
            if not runs:
                print(f"{noise_treatment} noise: no {REFERENCE_ALGORITHM.label} run yet")
                continue
            figure = plot_error_model_nee(runs)
            figure.suptitle(f"The NEE error models, {noise_treatment} noise: {REFERENCE_ALGORITHM.label}")
            save_figure(figure, f"error_model_nee_{noise_treatment}", directory)
