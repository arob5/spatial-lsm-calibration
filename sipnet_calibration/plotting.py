"""Composable plotting utilities for SIPNET calibration experiments.

Three-tier structure:
  Tier 1 — primitives: operate on a single Axes object.
  Tier 2 — composed: combine primitives into a standard figure panel.
  Tier 3 — diagnostics: wrap ArviZ for MCMC output.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Tier 1 — primitives
# ---------------------------------------------------------------------------

def plot_fan(
    ax: Axes,
    t: np.ndarray,
    samples: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.75, 0.9),
    color: str = "steelblue",
    alpha_inner: float = 0.35,
    alpha_outer: float = 0.15,
    label: Optional[str] = None,
) -> None:
    """Draw a shaded fan (credible interval envelope) from an ensemble.

    Parameters
    ----------
    ax:
        Target Axes.
    t:
        1-D array of x-coordinates (e.g. time index or dates).
    samples:
        2-D array of shape (n_samples, n_t).
    quantiles:
        Four quantile levels (low_outer, low_inner, high_inner, high_outer).
    color, alpha_inner, alpha_outer:
        Fill colour and transparency for the two bands.
    label:
        If provided, attaches a legend entry to the inner band.
    """
    if samples.ndim != 2 or samples.shape[1] != len(t):
        raise ValueError(
            f"samples must have shape (n_samples, {len(t)}), got {samples.shape}"
        )
    q = np.quantile(samples, quantiles, axis=0)
    ax.fill_between(t, q[0], q[3], color=color, alpha=alpha_outer)
    ax.fill_between(t, q[1], q[2], color=color, alpha=alpha_inner, label=label)


def plot_timeseries(
    ax: Axes,
    t: np.ndarray,
    values: np.ndarray,
    *,
    color: str = "black",
    linewidth: float = 1.0,
    linestyle: str = "-",
    label: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Draw a single timeseries line."""
    ax.plot(t, values, color=color, linewidth=linewidth,
            linestyle=linestyle, label=label, **kwargs)


def plot_scatter_obs(
    ax: Axes,
    obs: np.ndarray,
    t_obs: Optional[np.ndarray] = None,
    *,
    color: str = "black",
    s: float = 6.0,
    alpha: float = 0.6,
    label: Optional[str] = "Observations",
    **kwargs: Any,
) -> None:
    """Scatter-plot observations.

    Parameters
    ----------
    ax:
        Target Axes.
    obs:
        1-D array of observed values.
    t_obs:
        x-coordinates. If None, uses np.arange(len(obs)).
    """
    if t_obs is None:
        t_obs = np.arange(len(obs))
    ax.scatter(t_obs, obs, color=color, s=s, alpha=alpha, label=label, **kwargs)


# ---------------------------------------------------------------------------
# Tier 2 — composed panels
# ---------------------------------------------------------------------------

def fan_chart(
    samples: np.ndarray,
    t: Optional[np.ndarray] = None,
    obs: Optional[np.ndarray] = None,
    obs_t: Optional[np.ndarray] = None,
    truth: Optional[np.ndarray] = None,
    *,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    samples_label: str = "Ensemble",
    truth_label: str = "Truth",
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.75, 0.9),
    color: str = "steelblue",
) -> Axes:
    """Fan chart combining ensemble envelope, observations, and optional truth.

    Parameters
    ----------
    samples:
        2-D array (n_samples, n_t) of model output trajectories.
    t:
        x-axis coordinates for the model output (length n_t). Defaults to
        integer indices.
    obs:
        Optional observed values. Length may differ from n_t; use obs_t to
        align them on the x-axis.
    obs_t:
        x-coordinates for observations. If obs is provided and obs_t is None,
        obs is assumed to share the same x-axis as samples.
    truth:
        Optional "ground-truth" timeseries (same length as t).
    ylabel, title:
        Axis labels.
    ax:
        Axes to plot into; creates a new figure if None.
    quantiles:
        Passed through to plot_fan().

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    n_t = samples.shape[1]
    if t is None:
        t = np.arange(n_t)

    plot_fan(ax, t, samples, quantiles=quantiles, color=color, label=samples_label)

    if truth is not None:
        plot_timeseries(ax, t, truth, color="red", linewidth=1.2, label=truth_label)

    if obs is not None:
        plot_scatter_obs(ax, obs, t_obs=obs_t)

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8)

    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Tier 3 — MCMC diagnostics
# ---------------------------------------------------------------------------

def plot_diagnostics(
    idata: Any,
    param_names: Optional[list[str]] = None,
    ground_truth: Optional[dict[str, float]] = None,
) -> None:
    """Trace plots and summary statistics for an ArviZ InferenceData object.

    Parameters
    ----------
    idata:
        ArviZ InferenceData (e.g. from az.from_netcdf()).
    param_names:
        Parameter names to plot. Defaults to all variables in the posterior.
    ground_truth:
        Optional dict mapping param name -> true value. Plotted as vertical
        lines on the marginal posteriors.
    """
    import arviz as az

    if param_names is None:
        param_names = list(idata.posterior.data_vars)

    ax_array = az.plot_trace(idata, var_names=param_names, compact=False)

    if ground_truth:
        # ax_array has shape (n_params, 2); column 1 is the marginal histogram
        for row_idx, name in enumerate(param_names):
            if name in ground_truth:
                true_val = ground_truth[name]
                for ax_row in ax_array:
                    for ax in ax_row:
                        pass  # will target via returned structure
                # az.plot_trace returns ndarray of axes; KDE panel is col 0
                try:
                    kde_ax = ax_array[row_idx, 0]
                    kde_ax.axvline(true_val, color="red", linestyle="--",
                                   linewidth=1.0, label="truth")
                except (IndexError, TypeError):
                    pass

    plt.tight_layout()

    print("\nMCMC Summary:")
    print(az.summary(idata, var_names=param_names))
