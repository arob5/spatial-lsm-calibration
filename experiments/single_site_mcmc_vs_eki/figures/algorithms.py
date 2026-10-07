"""Figures comparing the inference algorithms on one model: EKI, SMC and MCMC.

Each figure overlays the algorithms of :data:`COMPARED_ALGORITHMS`, read from
the runs of one model (``algorithms/records.py``); an algorithm whose run is
not written yet is left out. Nothing here runs a model.

- :func:`plot_marginal_histograms` (``algorithm_marginals``): each
  parameter's weighted marginal histogram, in natural units, over the
  prior's density;
- :func:`plot_pairwise_contours` (``algorithm_pairs``): for a set of
  parameters, each pair's joint density as the contours enclosing 50% and
  90% of its mass (a weighted kernel density estimate), with the marginal
  densities, the prior's among them, on the diagonal; the prior's contours
  are left out, since at the posterior's scale they lie outside the panels;
- :func:`plot_predictive_nee` (``algorithm_predictive_nee``): NEE's seasonal
  cycle, each algorithm's posterior predictive (90% band and median of the
  weekly means of replicated data) against the observed weekly means, at
  the calibration tower and the held-out one;
- :func:`plot_predictive_pools` (``algorithm_predictive_pools``): the pool
  constraints, each algorithm's posterior predictive 90% interval and median
  against the observations.

The predictives are of the data, noise included: at each sample a predictive
ran (``run/predict.py``, equally weighted), replicated data
:math:`\\mathcal G(\\theta_m) + \\sqrt{s_m} L z_m`
(``analysis/predictive.py``), with the sample's drawn noise scales where the
noise is inferred.

:func:`draw_algorithm_figures` draws them all for one model into
``config.FIGURE_DIRECTORY / "algorithms" / <error model>_<noise>``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from sipnet_calibration.conventions import TIME
from sipnet_calibration.plotting.style import role_style, use_project_style

from .. import config
from ..analysis.compare import weighted_quantiles
from ..analysis.predictive import replicated_observations
from ..model import prior as prior_model
from ..model.likelihood import NoiseModel
from ..model.outputs import load_predictive, load_run
from ..models import Model, fixed_posterior, heldout_posterior
from .common import (
    NEE_TITLES,
    PARAMETER_TITLES,
    SLIDE_STYLE,
    SOURCE_LABELS,
    one_legend,
    save_figure,
    week_of_year,
)

__all__ = [
    "COMPARED_ALGORITHMS",
    "PAIR_PARAMETER_NAMES",
    "AlgorithmRun",
    "ComparedAlgorithm",
    "draw_algorithm_figures",
    "load_algorithm_runs",
    "plot_marginal_histograms",
    "plot_pairwise_contours",
    "plot_predictive_nee",
    "plot_predictive_pools",
    "prior_natural_values",
    "draw_band",
    "weekly_band",
]


@dataclass(frozen=True)
class ComparedAlgorithm:
    """An algorithm the figures compare: its label, the run it reads under
    each noise treatment, and its color."""

    label: str
    runs: Mapping[str, str]
    color: str


#: The algorithms compared, in drawing order: EKI (for inferred noise, with
#: Gibbs scale updates and the common gain), SMC seeded by it, and MCMC.
COMPARED_ALGORITHMS = (
    ComparedAlgorithm("EKI", {"fixed": "eki", "inferred": "eki_gibbs_common"}, color="#E69F00"),
    ComparedAlgorithm(
        "SMC", {"fixed": "eki_smc", "inferred": "eki_gibbs_common_smc"}, color="#0072B2"
    ),
    ComparedAlgorithm("MCMC", {"fixed": "mcmc", "inferred": "mcmc"}, color="#009E73"),
)

#: The parameters of the pairwise figure: those the NEE error model moves
#: (MODEL.md, "How we got here").
PAIR_PARAMETER_NAMES = (
    "photosynthetic_capacity",
    "half_saturation_light",
    "respiration_share",
    "optimum_photosynthesis_temperature",
    "soil_respiration_q10",
    "wood_respiration_rate_at_10c",
)

#: The pool constraints of the predictive pools figure, in drawing order.
POOL_SOURCE_NAMES = (
    "modis_leaf_area_index",
    "landtrendr_aboveground_biomass",
    "soilgrids_soil_organic_carbon",
)

#: The vectors a predictive predicts, by its directory name (``run/predict.py``),
#: with their panel labels: the calibration tower, and the held-out one.
PREDICTIVE_VECTORS = {
    "calibration": "calibration, US-Ha1 2012-2020",
    "validation": "held out, US-xHA 2021-2024",
}

#: The mass the pairwise contours enclose, innermost first.
CONTOUR_MASSES = (0.5, 0.9)

#: The most samples a kernel density estimate reads; a longer run is thinned evenly.
DENSITY_SAMPLE_LIMIT = 4000

#: The number of prior draws the figures show.
PRIOR_SAMPLE_SIZE = 4000

#: The seed of the prior draws and of the replicated data's noise.
FIGURE_SEED = 20261007

#: The quantiles of a predictive band: its lower edge, median and upper edge.
BAND_QUANTILES = (0.05, 0.5, 0.95)


@dataclass(frozen=True)
class AlgorithmRun:
    """One algorithm's run of a model: its samples' natural values and
    normalized weights, and, once its predictive has run, the predictive's
    files and its replicated data, ``{vector: {source: (K, n)}}``."""

    algorithm: ComparedAlgorithm
    natural_values: pd.DataFrame
    weights: np.ndarray
    predictive: dict | None
    replicated: dict[str, dict[str, np.ndarray]] | None


def load_algorithm_runs(
    model: Model,
    algorithms: Sequence[ComparedAlgorithm] = COMPARED_ALGORITHMS,
    *,
    replicate: bool = True,
) -> list[AlgorithmRun]:
    """The runs of *model* that *algorithms* read, those written so far,
    each with its replicated data where its predictive has run and
    *replicate* is true (which builds the model's posteriors, a minute or
    two; the parameter figures need none).

    A sample whose run failed weighs 0.
    """
    noise_models = None
    runs = []
    for algorithm in algorithms:
        directory = model.directory(algorithm.runs[model.noise])
        if not (directory / "samples.nc").exists():
            continue
        run = load_run(directory)
        samples = run["samples"]
        log_weights = np.where(
            samples["valid"].values.astype(bool), samples["log_weight"].values, -np.inf
        )
        predictive, replicated = None, None
        if replicate and (directory / "predictive" / "ensemble_daily.nc").exists():
            predictive = load_predictive(directory / "predictive")
            noise_models = noise_models or _noise_models(model)
            scales = pd.read_csv(directory / "predictive" / "samples.csv")
            replicated = _replicated(predictive, noise_models, scales)
        runs.append(
            AlgorithmRun(
                algorithm=algorithm,
                natural_values=run["natural_values"].drop(columns="log_weight"),
                weights=np.exp(log_weights - np.logaddexp.reduce(log_weights)),
                predictive=predictive,
                replicated=replicated,
            )
        )
    return runs


def prior_natural_values() -> pd.DataFrame:
    """:data:`PRIOR_SAMPLE_SIZE` draws of the prior, in natural units, one row
    per draw; no model is run."""
    posterior = prior_model.prior_alone()
    theta = posterior.sample_prior(jax.random.key(FIGURE_SEED), PRIOR_SAMPLE_SIZE)
    return prior_model.natural_table(posterior, theta)


def plot_marginal_histograms(
    runs: Sequence[AlgorithmRun],
    prior: pd.DataFrame,
    parameter_names: Sequence[str] | None = None,
    *,
    n_columns: int = 4,
) -> plt.Figure:
    """Each parameter's weighted marginal histogram, one panel per parameter,
    the algorithms overlaid on shared bins, over the prior's density (*prior*,
    its draws in natural units). A panel spans the runs' central 99% and the
    prior's central 90%."""
    parameter_names = list(parameter_names or runs[0].natural_values.columns)
    n_rows = -(-len(parameter_names) // n_columns)
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(4 * n_columns, 2.8 * n_rows), squeeze=False
    )
    for ax, name in zip(axes.flat, parameter_names):
        value_range = _union(_shared_range(runs, name), _prior_range(prior, name))
        _draw_prior_density(ax, prior[name], value_range)
        bins = np.linspace(*value_range, 41)
        for run in runs:
            density, _ = np.histogram(
                run.natural_values[name], bins=bins, weights=run.weights, density=True
            )
            color = run.algorithm.color
            ax.stairs(density, bins, color=color, alpha=0.2, fill=True)
            ax.stairs(density, bins, color=color, linewidth=1.4, label=run.algorithm.label)
        ax.set_title(PARAMETER_TITLES.get(name, name))
        ax.set_yticks([])
    for ax in axes.flat[len(parameter_names) :]:
        ax.set_visible(False)
    one_legend(figure, axes)
    return figure


def plot_pairwise_contours(
    runs: Sequence[AlgorithmRun],
    prior: pd.DataFrame,
    parameter_names: Sequence[str] = PAIR_PARAMETER_NAMES,
) -> plt.Figure:
    """For each pair of *parameter_names*, the contours enclosing
    :data:`CONTOUR_MASSES` of each algorithm's weighted kernel density
    estimate (below the diagonal), and each marginal density, the prior's
    among them (on it)."""
    n = len(parameter_names)
    figure, axes = plt.subplots(n, n, figsize=(2.4 * n, 2.4 * n), squeeze=False)
    ranges = {name: _shared_range(runs, name) for name in parameter_names}
    for row, row_name in enumerate(parameter_names):
        for column, column_name in enumerate(parameter_names):
            ax = axes[row, column]
            if column > row:
                ax.set_visible(False)
                continue
            if row == column:
                _draw_prior_density(ax, prior[row_name], ranges[row_name])
            for run in runs:
                values, weights = _thinned(run, [column_name, row_name])
                if row == column:
                    _draw_marginal_density(ax, values[:, 0], weights, ranges[row_name], run)
                else:
                    _draw_mass_contours(
                        ax, values, weights, ranges[column_name], ranges[row_name], run
                    )
            ax.set_xlim(ranges[column_name])
            if row != column:
                ax.set_ylim(ranges[row_name])
            _label_corner_axes(ax, row, column, n, column_name, row_name)
    # Proxy lines say which contour is which in the shared legend.
    axes[1, 0].plot([], [], color="#555555", linewidth=1.6, label="50% of the mass")
    axes[1, 0].plot([], [], color="#555555", linewidth=1.0, label="90% of the mass")
    one_legend(figure, axes)
    return figure


def plot_predictive_nee(runs: Sequence[AlgorithmRun]) -> plt.Figure:
    """NEE by week of year: each algorithm's posterior predictive, the 90%
    band and median over samples of the replicated data's weekly means,
    against the observed weekly means; the calibration tower above, the
    held-out tower below."""
    runs = [run for run in runs if run.replicated is not None]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, squeeze=False)
    for row, (vector, vector_label) in enumerate(PREDICTIVE_VECTORS.items()):
        for column, name in enumerate(config.NEE_WINDOWS):
            ax = axes[row, column]
            observed = runs[0].predictive["observed"][vector][name]["value"]
            weeks = week_of_year(observed[TIME])
            for run in runs:
                band = _band(
                    pd.DataFrame(run.replicated[vector][name].T).groupby(weeks).mean().T
                )
                draw_band(ax, band.index, band, run.algorithm.color, run.algorithm.label)
            observed_weekly = pd.Series(observed.to_numpy()).groupby(weeks).mean()
            ax.plot(
                observed_weekly.index,
                observed_weekly.to_numpy(),
                "o",
                color="black",
                markersize=3.5,
                label="observed",
            )
            ax.axhline(0.0, color="#999999", linewidth=0.6, zorder=0)
            ax.set_title(f"{NEE_TITLES[name]}\n{vector_label}")
        axes[row, 0].set_ylabel("NEE (µmol CO₂ m⁻² s⁻¹), weekly mean")
    for ax in axes[-1]:
        ax.set_xlabel("week of year")
    one_legend(figure, axes)
    return figure


def plot_predictive_pools(runs: Sequence[AlgorithmRun]) -> plt.Figure:
    """The pool constraints: each algorithm's posterior predictive, the 90%
    interval and median over samples of the replicated data, side by side,
    against the observations. MODIS LAI is averaged over each summer's
    composites, observed and replicated alike."""
    runs = [run for run in runs if run.replicated is not None]
    figure, axes = plt.subplots(
        1, len(POOL_SOURCE_NAMES), figsize=(15, 4.5), width_ratios=(3, 2, 1), squeeze=False
    )
    # Each year's slots, centered on its tick: one per algorithm, then the observation.
    offsets = np.linspace(-0.3, 0.3, len(runs) + 1)
    for ax, name in zip(axes[0], POOL_SOURCE_NAMES, strict=True):
        observed = runs[0].predictive["observed"]["calibration"][name]["value"]
        years = _years(observed)
        ticks = np.arange(len(set(years)))
        for offset, run in zip(offsets, runs):
            band = _band(pd.DataFrame(run.replicated["calibration"][name].T).groupby(years).mean().T)
            ax.errorbar(
                ticks + offset,
                band[0.5],
                yerr=[band[0.5] - band[0.05], band[0.95] - band[0.5]],
                fmt="s",
                markersize=4,
                capsize=2,
                color=run.algorithm.color,
                label=f"{run.algorithm.label}, median and 90%",
            )
        observed_by_year = pd.Series(np.ravel(observed.to_numpy())).groupby(years).mean()
        ax.plot(
            ticks + offsets[-1],
            observed_by_year.to_numpy(),
            "o",
            markersize=5,
            color="black",
            label="observed",
        )
        ax.set_xticks(ticks, [str(year) for year in observed_by_year.index])
        ax.set_title(SOURCE_LABELS[name])
        ax.set_ylabel(observed.attrs.get("units", ""))
    one_legend(figure, axes)
    return figure


def draw_algorithm_figures(model: Model) -> None:
    """Draw every figure of this module for *model*."""
    runs = load_algorithm_runs(model)
    if not runs:
        print(f"{model.name}: no run to compare yet")
        return
    directory = config.FIGURE_DIRECTORY / "algorithms" / model.name.replace("/", "_")
    labels = ", ".join(run.algorithm.label for run in runs)
    prior = prior_natural_values()
    use_project_style()
    with plt.rc_context(SLIDE_STYLE):
        for name, figure in (
            ("algorithm_marginals", plot_marginal_histograms(runs, prior)),
            ("algorithm_pairs", plot_pairwise_contours(runs, prior)),
            ("algorithm_predictive_nee", plot_predictive_nee(runs)),
            ("algorithm_predictive_pools", plot_predictive_pools(runs)),
        ):
            figure.suptitle(f"{model.name}: {labels}")
            save_figure(figure, name, directory)


def weekly_band(replicated: np.ndarray, times) -> pd.DataFrame:
    """The :data:`BAND_QUANTILES` over samples of each week's mean of
    *replicated* ``(K, n)``, the windows placed by *times*; one row per week."""
    return _band(pd.DataFrame(replicated.T).groupby(week_of_year(times)).mean().T)


def draw_band(ax, positions, band: pd.DataFrame, color: str, label: str) -> None:
    """A 90% band and its median, colored *color* and labeled *label*."""
    ax.fill_between(
        positions, band[0.05], band[0.95], color=color, alpha=0.2, linewidth=0, label=f"{label}, 90%"
    )
    ax.plot(positions, band[0.5], color=color, linewidth=1.5, label=f"{label}, median")


# ── helpers ──


def _band(values_by_sample: pd.DataFrame) -> pd.DataFrame:
    """The :data:`BAND_QUANTILES` over samples (rows) of each column."""
    return values_by_sample.quantile(list(BAND_QUANTILES)).T


def _noise_models(model: Model) -> dict[str, NoiseModel]:
    """Each predicted vector's noise model under *model*'s error model, by
    its directory name; it supplies :math:`C_k`, the scales come from the run."""
    return {
        "calibration": NoiseModel(fixed_posterior(model), inferred=False),
        "validation": NoiseModel(heldout_posterior(model), inferred=False),
    }


def _replicated(
    predictive: dict, noise_models: Mapping[str, NoiseModel], scales: pd.DataFrame
) -> dict[str, dict[str, np.ndarray]]:
    """Replicated data of every source of each predicted vector, at the
    predictive's samples and their scales (the columns of its ``samples.csv``)."""
    rng = np.random.default_rng(FIGURE_SEED)
    return {
        vector: replicated_observations(
            predictive["predicted"]["ensemble"][vector],
            predictive["observed"][vector],
            noise_model,
            {name: scales[name].to_numpy() for name in scales if name != "run_sample"},
            rng,
        )
        for vector, noise_model in noise_models.items()
    }


def _years(observed) -> np.ndarray:
    """Each observation's calendar year, or one label for a static source."""
    if TIME not in observed.dims:
        return np.array(["static"])
    return pd.DatetimeIndex(observed[TIME].to_numpy()).year.to_numpy()


def _shared_range(runs: Sequence[AlgorithmRun], name: str) -> tuple[float, float]:
    """A range holding the central 99% of every run's weighted marginal of *name*."""
    low, high = zip(
        *(
            weighted_quantiles(run.natural_values[name], run.weights, (0.005, 0.995))
            for run in runs
        ),
        strict=True,
    )
    margin = 0.05 * (max(high) - min(low))
    return min(low) - margin, max(high) + margin


def _prior_range(prior: pd.DataFrame, name: str) -> tuple[float, float]:
    """The prior's central 90% of *name*."""
    low, high = np.quantile(prior[name], (0.05, 0.95))
    return float(low), float(high)


def _union(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float]:
    """The smallest range holding both."""
    return min(first[0], second[0]), max(first[1], second[1])


def _draw_prior_density(ax, values, value_range) -> None:
    """The prior's kernel density estimate over *value_range*, as the prior's dashed line."""
    grid = np.linspace(*value_range, 200)
    density = scipy.stats.gaussian_kde(np.asarray(values))(grid)
    ax.plot(grid, density, **role_style("prior", "line"), label="prior")


def _thinned(run: AlgorithmRun, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """*run*'s values of *names* ``(n, len(names))`` and weights, at most
    :data:`DENSITY_SAMPLE_LIMIT` samples of positive weight, evenly spaced."""
    keep = np.flatnonzero(run.weights > 0)
    if len(keep) > DENSITY_SAMPLE_LIMIT:
        keep = keep[np.linspace(0, len(keep) - 1, DENSITY_SAMPLE_LIMIT).astype(int)]
    weights = run.weights[keep]
    return run.natural_values[names].to_numpy()[keep], weights / weights.sum()


def _draw_marginal_density(ax, values, weights, value_range, run: AlgorithmRun) -> None:
    """A weighted kernel density estimate of *values* over *value_range*."""
    grid = np.linspace(*value_range, 200)
    density = scipy.stats.gaussian_kde(values, weights=weights)(grid)
    ax.plot(grid, density, color=run.algorithm.color, linewidth=1.4, label=run.algorithm.label)
    ax.set_yticks([])


def _draw_mass_contours(ax, values, weights, x_range, y_range, run: AlgorithmRun) -> None:
    """Contours of a weighted 2-D kernel density estimate enclosing
    :data:`CONTOUR_MASSES` of its mass: on a grid, the levels :math:`t` with
    :math:`\\sum_{f \\ge t} f = m \\sum f`."""
    x, y = np.meshgrid(np.linspace(*x_range, 60), np.linspace(*y_range, 60))
    density = scipy.stats.gaussian_kde(values.T, weights=weights)(
        np.vstack([x.ravel(), y.ravel()])
    ).reshape(x.shape)
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    levels = sorted(ordered[np.searchsorted(cumulative, mass)] for mass in CONTOUR_MASSES)
    ax.contour(x, y, density, levels=levels, colors=run.algorithm.color, linewidths=(1.0, 1.6))
    # contour draws no legend entry, so a line carries it.
    ax.plot([], [], color=run.algorithm.color, label=run.algorithm.label)


def _label_corner_axes(ax, row, column, n, column_name, row_name) -> None:
    """Axis labels on the outer panels of the corner plot only."""
    if row == n - 1:
        ax.set_xlabel(PARAMETER_TITLES.get(column_name, column_name), fontsize=9)
    else:
        ax.set_xticklabels([])
    if column == 0 and row > 0:
        ax.set_ylabel(PARAMETER_TITLES.get(row_name, row_name), fontsize=9)
    elif column != row:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=8)
