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
  cycle, each algorithm's 90% band and median of the weekly means of
  :math:`\mathcal G(\theta)`, and one run's 90% band of the weekly means
  of replicated data, :math:`\mathcal G(\theta) + \varepsilon`,
  :math:`\varepsilon \sim \mathcal N(0, s C)`, against the observed, for
  the calibration tower and the held-out one;
- :func:`plot_predictive_pools` (``algorithm_predictive_pools``): the pool
  constraints, the posterior predictive's 90% interval against the
  observations.

The predictive figures read the samples ``run/predict.py`` ran (equally
weighted). Their bands of :math:`\\mathcal G(\\theta)` carry the parameters'
uncertainty alone; the replicated data's band adds the noise model's, the
error the model itself expects.

:func:`draw_algorithm_figures` draws them all for one model into
``config.FIGURE_DIRECTORY / "algorithms" / <error model>_<noise>``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from sipnet_calibration.conventions import SAMPLE, TIME
from sipnet_calibration.plotting.style import role_style, use_project_style

from .. import config
from ..analysis.compare import weighted_quantiles
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
    weekly_nee_quantiles,
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
    "replicated_nee",
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
    ComparedAlgorithm(
        "EKI", {"fixed": "eki", "inferred": "eki_gibbs_common"}, color="#E69F00"
    ),
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

#: The mass the pairwise contours enclose, innermost first.
CONTOUR_MASSES = (0.5, 0.9)

#: The most samples a kernel density estimate reads; a longer run is thinned evenly.
DENSITY_SAMPLE_LIMIT = 4000

#: The prior draws the figures show, and their seed.
PRIOR_SAMPLE_SIZE = 4000
PRIOR_SEED = 20261007

#: The vectors of the predictive NEE figure: the calibration tower, and the
#: held-out one (``run/predict.py``'s directory names).
PREDICTIVE_VECTORS = {
    "calibration": "calibration, US-Ha1 2012-2020",
    "validation": "held out, US-xHA 2021-2024",
}


@dataclass(frozen=True)
class AlgorithmRun:
    """One algorithm's run of a model: its samples' natural values and
    normalized weights, and its posterior predictive if it has run."""

    algorithm: ComparedAlgorithm
    natural_values: pd.DataFrame
    weights: np.ndarray
    predictive: dict | None
    predictive_directory: Path


def load_algorithm_runs(
    model: Model, algorithms: Sequence[ComparedAlgorithm] = COMPARED_ALGORITHMS
) -> list[AlgorithmRun]:
    """The runs of *model* that *algorithms* read, those written so far.

    A sample whose run failed weighs 0.
    """
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
        weights = np.exp(log_weights - np.logaddexp.reduce(log_weights))
        predictive = directory / "predictive"
        runs.append(
            AlgorithmRun(
                algorithm=algorithm,
                natural_values=run["natural_values"].drop(columns="log_weight"),
                weights=weights,
                predictive=(
                    load_predictive(predictive)
                    if (predictive / "ensemble_daily.nc").exists()
                    else None
                ),
                predictive_directory=predictive,
            )
        )
    return runs


def prior_natural_values() -> pd.DataFrame:
    """:data:`PRIOR_SAMPLE_SIZE` draws of the prior, in natural units, one row
    per draw; no model is run."""
    posterior = prior_model.prior_alone()
    theta = posterior.sample_prior(jax.random.key(PRIOR_SEED), PRIOR_SAMPLE_SIZE)
    return prior_model.natural_table(posterior, theta)


def replicated_nee(model: Model, run: AlgorithmRun) -> dict[str, dict[str, np.ndarray]]:
    """Replicated NEE data at *run*'s predictive samples, ``{vector: {source:
    (K, n)}}``: :math:`\\mathcal G(\\theta_m) + \\varepsilon_m`,
    :math:`\\varepsilon_m \\sim \\mathcal N(0, s_m C)`, with :math:`C` the
    source's reference covariance under *model*'s error model and
    :math:`s_m` the sample's drawn scale (1 for fixed noise).
    """
    rng = np.random.default_rng(PRIOR_SEED)
    scales = pd.read_csv(run.predictive_directory / "samples.csv")
    posteriors = {"calibration": fixed_posterior(model), "validation": heldout_posterior(model)}
    replicated = {}
    for vector, posterior in posteriors.items():
        sources = {source.name: source for source in NoiseModel(posterior, inferred=False).sources}
        replicated[vector] = {}
        for name in config.NEE_WINDOWS:
            predicted = run.predictive["predicted"]["ensemble"][vector][name]
            values = predicted.transpose(SAMPLE, TIME).to_numpy()
            scale = scales[name].to_numpy() if name in scales else np.ones(len(values))
            noise = rng.standard_normal(values.shape) @ sources[name].cholesky.T
            replicated[vector][name] = values + np.sqrt(scale)[:, None] * noise
    return replicated


def plot_marginal_histograms(
    runs: Sequence[AlgorithmRun],
    prior: pd.DataFrame,
    parameter_names: Sequence[str] | None = None,
) -> plt.Figure:
    """Each parameter's weighted marginal histogram, one panel per parameter,
    the algorithms overlaid on shared bins, over the prior's density (*prior*,
    its draws in natural units). A panel spans the runs' central 99% and the
    prior's central 90%."""
    parameter_names = list(parameter_names or runs[0].natural_values.columns)
    n_columns = 4
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
    estimate (below the diagonal), and each marginal density (on it)."""
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


def plot_predictive_nee(
    runs: Sequence[AlgorithmRun],
    replicated: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    replicated_label: str = "",
) -> plt.Figure:
    """NEE by week of year: each algorithm's 90% band and median of the
    weekly means of :math:`\\mathcal G(\\theta)`, against the observed weekly
    means; the calibration tower above, the held-out tower below. With
    *replicated* (``replicated_nee``), the 90% band of the replicated data's
    weekly means is drawn behind, labeled *replicated_label*."""
    runs = [run for run in runs if run.predictive is not None]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, squeeze=False)
    for row, (vector, vector_label) in enumerate(PREDICTIVE_VECTORS.items()):
        for column, name in enumerate(config.NEE_WINDOWS):
            ax = axes[row, column]
            if replicated is not None:
                observed = runs[0].predictive["observed"][vector][name]["value"]
                band = _weekly_quantiles(replicated[vector][name], observed[TIME])
                ax.fill_between(
                    band.index,
                    band[0.05],
                    band[0.95],
                    color="#bbbbbb",
                    alpha=0.5,
                    linewidth=0,
                    label=f"{replicated_label}, replicated data, 90%",
                )
            for run in runs:
                observed, quantiles = weekly_nee_quantiles(run.predictive, name, vector)
                color = run.algorithm.color
                ax.fill_between(
                    quantiles.index,
                    quantiles[0.05],
                    quantiles[0.95],
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                    label=f"{run.algorithm.label}, 90%",
                )
                ax.plot(
                    quantiles.index,
                    quantiles[0.5],
                    color=color,
                    linewidth=1.5,
                    label=f"{run.algorithm.label}, median",
                )
            ax.plot(
                observed.index,
                observed.to_numpy(),
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
    """The pool constraints: each algorithm's posterior predictive 90%
    interval and median, side by side, against the observations with their
    noise standard deviations (at scale 1). MODIS LAI is averaged over each
    summer's composites, its error bar a composite's mean noise standard
    deviation."""
    runs = [run for run in runs if run.predictive is not None]
    names = (
        "modis_leaf_area_index",
        "landtrendr_aboveground_biomass",
        "soilgrids_soil_organic_carbon",
    )
    figure, axes = plt.subplots(
        1, len(names), figsize=(15, 4.5), width_ratios=(3, 2, 1), squeeze=False
    )
    # Each year's slots, centered on its tick: one per algorithm, then the observation.
    offsets = np.linspace(-0.3, 0.3, len(runs) + 1)
    for ax, name in zip(axes[0], names, strict=True):
        observed = runs[0].predictive["observed"]["calibration"][name]
        years, observed_values, observed_errors = _by_year(observed)
        ticks = np.arange(len(years))
        for offset, run in zip(offsets, runs):
            predicted = run.predictive["predicted"]["ensemble"]["calibration"][name]
            quantiles = _predicted_quantiles_by_year(predicted)
            ax.errorbar(
                ticks + offset,
                quantiles[0.5],
                yerr=[quantiles[0.5] - quantiles[0.05], quantiles[0.95] - quantiles[0.5]],
                fmt="s",
                markersize=4,
                capsize=2,
                color=run.algorithm.color,
                label=f"{run.algorithm.label}, median and 90%",
            )
        ax.errorbar(
            ticks + offsets[-1],
            observed_values,
            yerr=observed_errors,
            fmt="o",
            markersize=4,
            capsize=2,
            color="black",
            label="observed ± noise sd",
        )
        ax.set_xticks(ticks, years)
        ax.set_title(SOURCE_LABELS[name])
        ax.set_ylabel(observed["value"].attrs.get("units", ""))
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
    # The replicated data are drawn at the last algorithm's samples, the
    # reference: MCMC once it has run, else SMC.
    reference = [run for run in runs if run.predictive is not None][-1]
    use_project_style()
    with plt.rc_context(SLIDE_STYLE):
        for name, figure in (
            ("algorithm_marginals", plot_marginal_histograms(runs, prior)),
            ("algorithm_pairs", plot_pairwise_contours(runs, prior)),
            (
                "algorithm_predictive_nee",
                plot_predictive_nee(
                    runs, replicated_nee(model, reference), reference.algorithm.label
                ),
            ),
            ("algorithm_predictive_pools", plot_predictive_pools(runs)),
        ):
            figure.suptitle(f"{model.name}: {labels}")
            save_figure(figure, name, directory)


# ── helpers ──


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


def _weekly_quantiles(values: np.ndarray, times) -> pd.DataFrame:
    """The 5%, 50% and 95% quantiles over samples of each week's mean of
    *values* ``(K, n)``, the windows placed by *times*."""
    weekly = pd.DataFrame(values.T).groupby(week_of_year(times)).mean()
    return weekly.quantile([0.05, 0.5, 0.95], axis=1).T


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
    ax.contour(
        x, y, density, levels=levels, colors=run.algorithm.color, linewidths=(1.0, 1.6)
    )
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


def _by_year(observed) -> tuple[list[str], np.ndarray, np.ndarray]:
    """An observed source's values and noise standard deviations by year
    (each year's means), or its one value for a static source."""
    if TIME not in observed.dims:
        return (
            ["static"],
            np.atleast_1d(float(observed["value"])),
            np.atleast_1d(float(observed["noise_standard_deviation"])),
        )
    frame = observed.to_dataframe()[["value", "noise_standard_deviation"]]
    grouped = frame.groupby(frame.index.get_level_values(TIME).year)
    return (
        [str(year) for year in grouped.groups],
        grouped["value"].mean().to_numpy(),
        grouped["noise_standard_deviation"].mean().to_numpy(),
    )


def _predicted_quantiles_by_year(predicted) -> dict[float, np.ndarray]:
    """The 5%, 50% and 95% quantiles over samples of each year's mean prediction."""
    if TIME not in predicted.dims:
        values = predicted.to_numpy().reshape(-1, 1)
    else:
        yearly = predicted.groupby(f"{TIME}.year").mean(TIME)
        values = yearly.transpose(SAMPLE, "year").to_numpy()
    return {q: np.nanquantile(values, q, axis=0) for q in (0.05, 0.5, 0.95)}
