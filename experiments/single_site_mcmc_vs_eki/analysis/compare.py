"""The runs compared: cost, parameters, agreement with a reference, noise scales.

Reads every run under a runs directory (``config.RUNS_DIRECTORY``, laid out
``<nee error model>/<noise>/<run>/`` as ``algorithms/records.py`` writes
it) and builds one table per comparison; a run that is missing is left
out. Nothing here runs a model or writes a file (``run/compare.py`` writes
the tables).

A run's samples carry normalized log weights :math:`\\log W_i`; every
summary weighs sample :math:`i` by :math:`W_i`, a sample whose likelihood
is not finite (``valid`` false) by 0, renormalized.

Functions
---------
:func:`find_runs`
    Every run under a runs directory, as ``(model name, run name, directory)``.
:func:`summarize_run`
    What the tables read of one run, its samples loaded once.
:func:`comparison_tables`
    Every table below, from the runs' summaries.
:func:`cost_table`, :func:`parameter_table`, :func:`against_reference_table`,
:func:`noise_scale_table`, :func:`reweighting_table`, :func:`heldout_score_table`
    One table each: see their docstrings.
:func:`weighted_quantiles`, :func:`gaussian_kl`
    The two computations the tables share.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg

from .. import config
from ..model.outputs import load_run
from ..models import MODEL_NAMES

__all__ = [
    "QUANTILES",
    "SEED_RUN_NAMES",
    "RunSummary",
    "against_reference_table",
    "comparison_tables",
    "cost_table",
    "find_runs",
    "gaussian_kl",
    "heldout_score_table",
    "noise_scale_table",
    "parameter_table",
    "reference_run",
    "reweighting_table",
    "summarize_run",
    "weighted_quantiles",
]

#: The quantiles the parameter and scale tables report.
QUANTILES = (0.05, 0.5, 0.95)

#: The EKI runs that seed importance sampling and SMC, in the order the
#: reference run is looked for among their SMC runs.
SEED_RUN_NAMES = ("eki", "eki_gibbs_common", "eki_gibbs_per_particle")

#: The prefixes of the phases a run copied from the run it was seeded or
#: started from (``algorithms/records.py``'s ``Cost.include``).
_SEED_PHASE_PREFIXES = ("seed:", "initialization:")


@dataclass(frozen=True, eq=False)
class RunSummary:
    """What the comparison reads of one run.

    ``weights`` are :math:`W_i`, 0 at an invalid sample; ``natural`` the
    natural values, one column per parameter element; ``theta`` the
    unconstrained samples ``(n, D)``; ``scales`` each drawn noise scale
    ``(n,)`` by source; ``attributes`` those of ``samples.nc``.
    """

    model_name: str
    run: str
    directory: Path
    weights: np.ndarray
    natural: pd.DataFrame
    theta: np.ndarray
    scales: dict[str, np.ndarray]
    attributes: dict
    cost: dict


def find_runs(
    runs_directory: Path = config.RUNS_DIRECTORY,
) -> list[tuple[str, str, Path]]:
    """Every run under *runs_directory* that wrote ``samples.nc``, by model
    in ``MODEL_NAMES`` order, then run name."""
    return [
        (model_name, path.parent.name, path.parent)
        for model_name in MODEL_NAMES
        for path in sorted((runs_directory / model_name).glob("*/samples.nc"))
    ]


def summarize_run(model_name: str, run: str, directory: Path) -> RunSummary:
    """One run's :class:`RunSummary`."""
    loaded = load_run(directory)
    samples = loaded["samples"]
    weights = np.exp(samples["log_weight"].values)
    if "valid" in samples:
        weights = np.where(samples["valid"].values.astype(bool), weights, 0.0)
    return RunSummary(
        model_name=model_name,
        run=run,
        directory=directory,
        weights=weights / weights.sum(),
        natural=loaded["natural_values"].drop(columns="log_weight"),
        theta=samples["theta"].values,
        scales={
            name: samples[f"{name}_noise_scale"].values
            for name in config.NOISE_SCALED_SOURCES
            if f"{name}_noise_scale" in samples
        },
        attributes=dict(samples.attrs),
        cost=loaded["cost"],
    )


def comparison_tables(
    runs_directory: Path = config.RUNS_DIRECTORY,
) -> dict[str, pd.DataFrame]:
    """Every comparison table of the runs under *runs_directory*, by file name."""
    summaries = [summarize_run(*found) for found in find_runs(runs_directory)]
    return {
        "cost": cost_table(summaries),
        "parameters": parameter_table(summaries),
        "against_reference": against_reference_table(summaries),
        "noise_scales": noise_scale_table(summaries),
        "reweighting": reweighting_table(summaries),
        "heldout_scores": heldout_score_table(summaries),
    }


# ── the tables ──


def cost_table(summaries: list[RunSummary]) -> pd.DataFrame:
    """Per run: SIPNET runs, forward-map calls and wall seconds, in total and
    split into the run's own phases (``own_*``) and those copied from the run
    it was seeded or started from (``seed_*``); the workers and nodes of its
    own phases."""
    rows = []
    for summary in summaries:
        phases = summary.cost["phases"]
        seed = [p for p in phases if p["phase"].startswith(_SEED_PHASE_PREFIXES)]
        own = [p for p in phases if not p["phase"].startswith(_SEED_PHASE_PREFIXES)]
        row = {"model": summary.model_name, "run": summary.run}
        for quantity in ("sipnet_runs", "forward_calls", "wall_seconds"):
            row[quantity] = sum(p[quantity] for p in phases)
            row[f"own_{quantity}"] = sum(p[quantity] for p in own)
            row[f"seed_{quantity}"] = sum(p[quantity] for p in seed)
        row["workers"] = " ".join(sorted({str(p.get("workers")) for p in own}))
        row["node"] = " ".join(sorted({str(p.get("node")) for p in own}))
        rows.append(row)
    return pd.DataFrame(rows).set_index(["model", "run"])


def parameter_table(summaries: list[RunSummary]) -> pd.DataFrame:
    """Per run and parameter element: the weighted mean, standard deviation
    and :data:`QUANTILES` (``q05``, ``q50``, ``q95``) of its natural values."""
    rows = []
    for summary in summaries:
        for column in summary.natural.columns:
            values = summary.natural[column].to_numpy()
            mean = summary.weights @ values
            rows.append(
                {
                    "model": summary.model_name,
                    "run": summary.run,
                    "parameter": column,
                    "mean": mean,
                    "standard_deviation": np.sqrt(
                        summary.weights @ (values - mean) ** 2
                    ),
                    **_quantile_columns(values, summary.weights),
                }
            )
    return pd.DataFrame(rows).set_index(["model", "run", "parameter"])


def against_reference_table(summaries: list[RunSummary]) -> pd.DataFrame:
    """Per model, each run against the model's :func:`reference_run`.

    One row per run and parameter element: ``mean_difference``,
    :math:`|\\bar x - \\bar x_{\\rm ref}| / s_{\\rm ref}`, and ``sd_ratio``,
    :math:`s / s_{\\rm ref}`, in natural units; and one row per run with
    parameter ``theta``: ``gaussian_kl``, :func:`gaussian_kl` from the
    reference's Gaussian fit in theta to the run's. A model with no
    reference run is left out.
    """
    by_model: dict[str, list[RunSummary]] = {}
    for summary in summaries:
        by_model.setdefault(summary.model_name, []).append(summary)
    rows = []
    for model_name, runs in by_model.items():
        reference = reference_run(runs)
        if reference is None:
            continue
        reference_mean, reference_sd = _mean_and_sd(reference)
        reference_moments = _theta_moments(reference)
        for summary in runs:
            if summary is reference:
                continue
            mean, sd = _mean_and_sd(summary)
            labels = {
                "model": model_name,
                "run": summary.run,
                "reference": reference.run,
            }
            for column in reference_mean.index:
                rows.append(
                    {
                        **labels,
                        "parameter": column,
                        "mean_difference": abs(mean[column] - reference_mean[column])
                        / reference_sd[column],
                        "sd_ratio": sd[column] / reference_sd[column],
                    }
                )
            rows.append(
                {
                    **labels,
                    "parameter": "theta",
                    "gaussian_kl": gaussian_kl(
                        *reference_moments, *_theta_moments(summary)
                    ),
                }
            )
    columns = [
        "model",
        "run",
        "reference",
        "parameter",
        "mean_difference",
        "sd_ratio",
        "gaussian_kl",
    ]
    return pd.DataFrame(rows, columns=columns).set_index(["model", "run", "parameter"])


def noise_scale_table(summaries: list[RunSummary]) -> pd.DataFrame:
    """Per run and scaled source: the :data:`QUANTILES` of its drawn noise scale."""
    rows = [
        {
            "model": summary.model_name,
            "run": summary.run,
            "observation_source": name,
            **_quantile_columns(values, summary.weights),
        }
        for summary in summaries
        for name, values in summary.scales.items()
    ]
    index = ["model", "run", "observation_source"]
    return pd.DataFrame(rows, columns=[*index, *_quantile_names()]).set_index(index)


def reweighting_table(summaries: list[RunSummary]) -> pd.DataFrame:
    """Per importance-sampling or SMC run: its seed run, the effective sample
    size, as a count and as a fraction of the samples, Pareto :math:`\\hat k`
    (importance sampling only), the log evidence and the number of stages."""
    rows = []
    for summary in summaries:
        attributes = summary.attributes
        if "effective_sample_size" not in attributes:
            continue
        n_samples = len(summary.weights)
        rows.append(
            {
                "model": summary.model_name,
                "run": summary.run,
                "seed_run": Path(attributes.get("seed_run", "")).name,
                "samples": n_samples,
                "effective_sample_size": attributes["effective_sample_size"],
                "effective_sample_fraction": attributes["effective_sample_size"]
                / n_samples,
                "pareto_k": attributes.get("pareto_k", np.nan),
                "log_evidence": attributes.get("log_evidence", np.nan),
                "stages": attributes.get("stages"),
            }
        )
    columns = [
        "model",
        "run",
        "seed_run",
        "samples",
        "effective_sample_size",
        "effective_sample_fraction",
        "pareto_k",
        "log_evidence",
        "stages",
    ]
    return pd.DataFrame(rows, columns=columns).set_index(["model", "run"])


def heldout_score_table(summaries: list[RunSummary]) -> pd.DataFrame:
    """Every run's ``predictive/heldout_scores.csv`` (``run/predict.py``), one
    row per run and held-out source; a run with no predictive is left out."""
    frames = [
        pd.read_csv(path).assign(model=summary.model_name, run=summary.run)
        for summary in summaries
        if (path := summary.directory / "predictive" / "heldout_scores.csv").exists()
    ]
    index = ["model", "run", "observation_source"]
    if not frames:
        return pd.DataFrame(columns=index).set_index(index)
    return pd.concat(frames, ignore_index=True).set_index(index)


def reference_run(runs: list[RunSummary]) -> RunSummary | None:
    """A model's reference among *runs*: MCMC if it ran, else the SMC run of
    the first of :data:`SEED_RUN_NAMES` that has one, else ``None``."""
    by_name = {summary.run: summary for summary in runs}
    for name in ("mcmc", *(f"{seed}_smc" for seed in SEED_RUN_NAMES)):
        if name in by_name:
            return by_name[name]
    return None


# ── the shared computations ──


def weighted_quantiles(values, weights, quantiles) -> np.ndarray:
    """The *quantiles* of *values* under normalized *weights*.

    The sorted values are placed at the midpoints of their cumulative
    weights, :math:`c_i - W_i / 2`, and the quantile function is their
    linear interpolation, constant beyond the first and last.
    """
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    positions = np.cumsum(weights) - weights / 2
    return np.interp(quantiles, positions / weights.sum(), values)


def gaussian_kl(mean_from, covariance_from, mean_to, covariance_to) -> float:
    """:math:`\\mathrm{KL}(\\mathcal N(m_0, \\Sigma_0) \\,\\|\\, \\mathcal N(m_1, \\Sigma_1))`,

    .. math::

        \\tfrac12\\big[\\operatorname{tr}(\\Sigma_1^{-1}\\Sigma_0)
        + (m_1 - m_0)^\\top \\Sigma_1^{-1} (m_1 - m_0) - D
        + \\log(|\\Sigma_1| / |\\Sigma_0|)\\big];

    :math:`+\\infty` when :math:`\\Sigma_1` is singular (a run with no more
    samples than dimensions), ``NaN`` when :math:`\\Sigma_0` is.
    """
    dimension = len(mean_from)
    if np.linalg.matrix_rank(covariance_from) < dimension:
        return np.nan
    if np.linalg.matrix_rank(covariance_to) < dimension:
        return np.inf
    factor = scipy.linalg.cho_factor(covariance_to, lower=True)
    difference = mean_to - mean_from
    log_determinant_to = 2.0 * np.log(np.diag(factor[0])).sum()
    log_determinant_from = np.linalg.slogdet(covariance_from)[1]
    return 0.5 * float(
        np.trace(scipy.linalg.cho_solve(factor, covariance_from))
        + difference @ scipy.linalg.cho_solve(factor, difference)
        - dimension
        + log_determinant_to
        - log_determinant_from
    )


# ── helpers ──


def _quantile_names() -> list[str]:
    """The column of each of :data:`QUANTILES`, ``q05``, ``q50``, ``q95``."""
    return [f"q{round(100 * q):02d}" for q in QUANTILES]


def _quantile_columns(values, weights) -> dict[str, float]:
    """:data:`QUANTILES` of *values*, by column name."""
    quantiles = weighted_quantiles(values, weights, QUANTILES)
    return dict(zip(_quantile_names(), quantiles, strict=True))


def _mean_and_sd(summary: RunSummary) -> tuple[pd.Series, pd.Series]:
    """The weighted mean and standard deviation of each natural column."""
    mean = summary.weights @ summary.natural
    variance = summary.weights @ (summary.natural - mean) ** 2
    return mean, np.sqrt(variance)


def _theta_moments(summary: RunSummary) -> tuple[np.ndarray, np.ndarray]:
    """The weighted mean and covariance of theta,
    :math:`\\sum_i W_i (\\theta_i - m)(\\theta_i - m)^\\top`."""
    mean = summary.weights @ summary.theta
    centered = summary.theta - mean
    return mean, (summary.weights[:, None] * centered).T @ centered
