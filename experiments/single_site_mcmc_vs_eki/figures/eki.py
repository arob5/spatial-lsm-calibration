"""EKI's figures: the ladder, the marginals, and, on synthetic data, recovery of the truth.

Reads what ``run/eki.py`` wrote under ``config.EKI_DIRECTORY / <data>``
and draws, into ``config.FIGURE_DIRECTORY / config.EKI_RUN_NAME``:

- :func:`plot_ladder` (``eki_ladder_<data>``): per step, the level beta and
  the increment, the members' misfits against ``N / 2``, the value expected
  at the truth, and the effective sample size and valid members as fractions
  of J;
- :func:`plot_marginals` (``eki_marginals_<data>``): each parameter's prior
  and posterior ensembles, in natural units, with the truth on synthetic
  data;
- :func:`plot_recovery` (``eki_recovery_synthetic``, synthetic data only):
  each entry of theta standardized by the prior, ``(theta - m_0) / sd_0``,
  with the prior's and the posterior ensemble's 90% intervals and the
  truth, and the posterior's standard deviation as a fraction of the
  prior's;
- once ``run/posterior_predictive.py`` has run, the prior predictive's
  figures drawn for the posterior predictive (``eki_posterior_predictive_*``),
  ``figures/prior_predictive.py``'s functions with ``kind="posterior"``, and
  its coverage of the held-out tower as well.

:func:`draw_eki_figures` draws the first three, which ``run/eki.py`` calls
at the end of a run, and :func:`draw_posterior_predictive_figures` the
rest, which ``run/posterior_predictive.py`` calls at the end of its run;
``run/draw_figures.py --run <data>`` redraws them all.
"""

import matplotlib.pyplot as plt
import numpy as np

from sipnet_calibration.plotting.style import role_style, use_project_style

from .. import config
from ..model import prior
from . import prior_predictive
from ..model.outputs import load_eki_run, load_predictive
from .common import save_figure
from .prior_predictive import PARAMETER_TITLES, SLIDE_STYLE

__all__ = [
    "draw_eki_figures",
    "draw_posterior_predictive_figures",
    "plot_ladder",
    "plot_marginals",
    "plot_recovery",
]


def plot_ladder(run: dict) -> plt.Figure:
    """The ladder, step by step: the level, the misfits, the ensemble's health."""
    history = run["history"]
    steps = history["step"].to_numpy()
    n_members = run["theta_prior"].shape[0]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    level, misfit, health = axes
    level.plot(steps, history["beta_next"], "o-", label="beta after the step")
    updates = history["increment"] > 0
    level.plot(steps[updates], history["increment"][updates], "s--", label="increment")
    level.set_yscale("log")
    level.set_title("The ladder")
    for column, style in (("misfit_mean", "o-"), ("misfit_min", "v--")):
        misfit.plot(steps, history[column], style, label=column.replace("_", " "))
    misfit.plot(steps, history["centre_misfit"], "s:", label="misfit of the mean")
    misfit.axhline(
        run["predictions"].shape[1] / 2,
        color="black",
        linewidth=0.8,
        label="N / 2, expected at the truth",
    )
    misfit.set_yscale("log")
    misfit.set_title("Misfit, 0.5 |R^-1/2 (y - G(theta))|^2")
    health.plot(steps, history["ess"] / n_members, "o-", label="ESS / J")
    health.plot(steps, history["n_valid"] / n_members, "s--", label="valid / J")
    health.plot(
        steps, history["out_of_domain_fraction"], "v:", label="out of domain / J"
    )
    health.set_ylim(-0.05, 1.3)
    health.set_title("The ensemble")
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(fontsize=9, loc="best")
    return figure


def plot_marginals(run: dict) -> plt.Figure:
    """Each parameter's prior and posterior ensembles, with the truth if known."""
    names = [name for name in PARAMETER_TITLES if name in run["posterior"]]
    figure, axes = plt.subplots(4, 4, figsize=(14, 10))
    for ax, name in zip(axes.flat, names, strict=False):
        prior_values = run["prior"][name].to_numpy()
        posterior_values = run["posterior"][name].to_numpy()
        both = np.concatenate([prior_values, posterior_values])
        log_scale = both.min() > 0 and both.max() / both.min() > 20
        bins = (
            np.geomspace(both.min(), both.max(), 30)
            if log_scale
            else np.linspace(both.min(), both.max(), 30)
        )
        for values, role, label in (
            (prior_values, "prior", "prior ensemble"),
            (posterior_values, "posterior", "posterior ensemble"),
        ):
            ax.hist(
                values,
                bins=bins,
                color=role_style(role)["color"],
                alpha=0.55,
                label=label,
            )
        if run["truth"] is not None:
            ax.axvline(
                float(run["truth"][name].iloc[0]),
                **role_style("truth", "line"),
                label="truth",
            )
        if log_scale:
            ax.set_xscale("log")
        ax.set_title(PARAMETER_TITLES[name], fontsize=10)
        ax.set_yticks([])
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    return figure


def plot_recovery(run: dict, entry_names: list[str]) -> plt.Figure:
    """Each entry of theta standardized by the prior: intervals and the truth."""
    prior_theta, posterior_theta = run["theta_prior"], run["theta_posterior"]
    mean, sd = prior_theta.mean(axis=0), prior_theta.std(axis=0, ddof=1)
    positions = np.arange(len(entry_names))
    figure, (intervals, ratios) = plt.subplots(
        1, 2, figsize=(14, 6), width_ratios=(2.2, 1), sharey=True
    )
    for theta, role, offset, label in (
        (prior_theta, "prior", 0.15, "prior ensemble, 90%"),
        (posterior_theta, "posterior", -0.15, "posterior ensemble, 90%"),
    ):
        standardized = (theta - mean) / sd
        low, middle, high = np.quantile(standardized, [0.05, 0.5, 0.95], axis=0)
        intervals.hlines(
            positions + offset,
            low,
            high,
            color=role_style(role)["color"],
            linewidth=3,
            label=label,
        )
        intervals.plot(
            middle,
            positions + offset,
            "|",
            color=role_style(role)["color"],
            markersize=10,
        )
    intervals.plot(
        (run["theta_true"] - mean) / sd,
        positions,
        "D",
        color=role_style("truth")["color"],
        label="truth",
    )
    intervals.axvline(0.0, color="#999999", linewidth=0.6, zorder=0)
    intervals.set_yticks(positions, entry_names)
    intervals.invert_yaxis()
    intervals.set_xlabel("(theta - prior mean) / prior sd")
    intervals.legend(loc="lower right", fontsize=9)
    intervals.set_title("Recovery of the synthetic truth")
    ratio = posterior_theta.std(axis=0, ddof=1) / sd
    ratios.barh(positions, ratio, color=role_style("posterior")["color"], alpha=0.7)
    ratios.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ratios.set_xlabel("posterior sd / prior sd")
    ratios.set_title("How much each entry was learned")
    return figure


# ── drawing a run ──


def draw_eki_figures(data: str) -> None:
    """Draw one EKI run's ladder, marginals and, on synthetic data, its recovery.

    The figures go into ``config.FIGURE_DIRECTORY / config.EKI_RUN_NAME``.

    Raises
    ------
    FileNotFoundError
        If the run under ``config.EKI_DIRECTORY / data`` has not started.
    """
    use_project_style()
    run = load_eki_run(config.EKI_DIRECTORY / data)
    figures = {
        f"eki_ladder_{data}": plot_ladder(run),
        f"eki_marginals_{data}": plot_marginals(run),
    }
    if run["theta_true"] is not None:
        entry_names = list(prior.calibration()[0].unconstrained.entry_names)
        figures["eki_recovery_synthetic"] = plot_recovery(run, entry_names)
    for name, figure in figures.items():
        save_figure(figure, name, eki_run=True)


def draw_posterior_predictive_figures(data: str) -> None:
    """Draw an EKI run's posterior predictive with the prior predictive's figures.

    Raises
    ------
    FileNotFoundError
        If the run's posterior predictive has not run.
    """
    use_project_style()
    outputs = load_predictive(config.EKI_DIRECTORY / data / "posterior_predictive")
    prefix = f"eki_posterior_predictive_{data}"
    for name, draw in (
        ("nee", prior_predictive.plot_nee_windows),
        ("pools", prior_predictive.plot_pool_observations),
        ("trajectories", prior_predictive.plot_daily_trajectories),
    ):
        save_figure(draw(outputs, kind="posterior"), f"{prefix}_{name}", eki_run=True)
    with plt.rc_context(SLIDE_STYLE):
        for name, figure in (
            (
                "nee_seasonal",
                prior_predictive.plot_nee_seasonal_cycle(outputs, kind="posterior"),
            ),
            ("nee_annual", prior_predictive.plot_nee_annual(outputs, kind="posterior")),
            ("coverage", prior_predictive.plot_coverage(outputs, kind="posterior")),
            (
                "coverage_validation",
                prior_predictive.plot_coverage(
                    outputs, kind="posterior", vector="validation"
                ),
            ),
        ):
            save_figure(figure, f"{prefix}_{name}", eki_run=True)
