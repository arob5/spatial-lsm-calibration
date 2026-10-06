"""Fit NEE's model discrepancy to a calibration's residuals, by maximum marginal likelihood.

Overview
--------
``MODEL.md``, "NEE error", "Estimating the parameters", states the method;
this script carries it out on one EKI run. For each NEE observation source
and each of four variants of the discrepancy (``model/discrepancy.py``),

- ``single``: the short term alone, refitted (the first calibration's form);
- ``two_term``: the short and long terms, no recurring one;
- ``three_term``: the short, long and recurring terms;
- ``drifting``: the three terms, the recurring one drifting from year to year,

it maximizes the Gaussian log likelihood of the residuals of the run's
median prediction under ``R_k(phi) = diag(sigma_obs^2) + Sigma_delta(phi)``,
over the logarithms of the parameters, by L-BFGS-B with JAX's gradient, from
several starts. It then checks each fit against the towers' floor and scores
it on the held-out tower's windows under the run's posterior predictive.

The fitted values are not written into ``config``: ``config`` stays the one
source of truth, and adopting a fit is a decision made by copying it there.

Input data
----------
``output/eki/<setup>/<data>``, ``<setup>`` ``config.EKI_RUN_NAME`` unless
``--setup`` names another: the run's final step, its posterior
predictive and the diagnosis it ends with (``run/posterior_predictive.py``),
for the held-out score and the towers' floor.

Output data
-----------
``nee_discrepancy_fit.csv`` in the run's directory, one row per source and
variant: the fitted parameters (timescales in days), the log likelihood
``log_likelihood``, the number of parameters ``k`` and ``aic``,
``floor_holds`` (the fitted total discrepancy variance at least the towers'
representativeness variance), and, with a posterior predictive,
``heldout_log_density`` (the held-out tower's log predictive density) and
``heldout_ratio`` (twice its median member's misfit over its windows).

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.fit_nee_discrepancy --data observed
    uv run python -m experiments.single_site_mcmc_vs_eki.run.fit_nee_discrepancy --data observed --setup single_term_discrepancy
"""

import argparse
import sys
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from frozendict import frozendict
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import logsumexp

from .. import config
from ..model import calibration, diagnostics
from ..model.discrepancy import NEEDiscrepancy
from ..model.outputs import load_diagnostics, load_eki_run, load_predictive

__all__ = [
    "BOUNDS",
    "VARIANTS",
    "Variant",
    "calibration_sources",
    "check_residuals_are_finite",
    "fit_variant",
    "heldout_scores",
    "heldout_sources",
    "main",
]


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Fit every variant to every NEE source and write the table."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    data = arguments.data
    directory = config.OUTPUT_DIRECTORY / "eki" / arguments.setup / data
    try:
        run = load_eki_run(directory)
        floors = load_diagnostics(directory)["nee_towers_summary"][
            "representativeness_standard_deviation"
        ]
    except (FileNotFoundError, KeyError) as error:
        print(
            f"error: {error}; run run/eki.py and run/posterior_predictive.py first",
            file=sys.stderr,
        )
        return 1
    calibration = calibration_sources(run)
    heldout = heldout_sources(directory, data)
    rows = []
    for name in config.NEE_WINDOWS:
        for variant in VARIANTS:
            fitted, log_likelihood, convergence = fit_variant(
                calibration[name], variant
            )
            rows.append(
                {
                    "source": name,
                    "variant": variant.name,
                    **fitted.describe(),
                    "log_likelihood": log_likelihood,
                    "k": len(variant.parameter_names),
                    "aic": 2 * len(variant.parameter_names) - 2 * log_likelihood,
                    "floor_holds": fitted.total_variance() >= floors[name] ** 2,
                    **convergence,
                    **heldout_scores(heldout, name, fitted),
                }
            )
            print(f"{name} {variant.name}: log likelihood {log_likelihood:.1f}")
    table = pd.DataFrame(rows).set_index(["source", "variant"])
    table.to_csv(directory / "nee_discrepancy_fit.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.drop(columns="provenance").round(3).to_string())
    print(f"wrote {directory / 'nee_discrepancy_fit.csv'}")
    return 0


# ── the steps ──


def calibration_sources(run: dict) -> dict:
    """The run's final predictions of the calibration vector, per source."""
    return diagnostics.sources_from_eki_run(run)


def heldout_sources(directory, data: str) -> dict | None:
    """The posterior predictive's predictions of the held-out tower, if there are any."""
    predictive = directory / "posterior_predictive"
    if data != "observed" or not (predictive / "ensemble_daily.nc").exists():
        return None
    return diagnostics.sources_from_predictive(
        load_predictive(predictive), "validation", calibration.validation_posterior()
    )


def fit_variant(source, variant: "Variant") -> tuple[NEEDiscrepancy, float, dict]:
    """The maximum-likelihood discrepancy of *variant* for *source*'s residuals.

    Returns the fit, its log likelihood, and ``converged`` (whether the best
    start's optimizer reported success) with ``at_bound`` (the parameters
    within 1% of a bound, which a wider bound might move).

    Raises
    ------
    ValueError
        If a residual is not finite.
    """
    residual = source.y - np.median(source.predictions, axis=0)
    check_residuals_are_finite(residual)
    times = _days_since_first(source.times)
    measurement_variance = source.measurement_standard_deviation**2
    scale = np.sqrt(max(np.var(residual) - measurement_variance.mean(), 1e-6))
    negative_log_likelihood = _negative_log_likelihood(
        variant,
        jnp.asarray(residual),
        jnp.asarray(times),
        jnp.asarray(measurement_variance),
    )
    bounds = [tuple(np.log(BOUNDS[name])) for name in variant.parameter_names]
    best = None
    for start in variant.starts:
        values = [
            start[name] * (scale if name in _RELATIVE_START_NAMES else 1.0)
            for name in variant.parameter_names
        ]
        result = minimize(
            negative_log_likelihood,
            np.log(values),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or result.fun < best.fun:
            best = result
    parameters = dict(
        zip(variant.parameter_names, np.exp(best.x).tolist(), strict=True)
    )
    at_bound = [
        name
        for name, value, (low, high) in zip(
            variant.parameter_names, best.x, bounds, strict=True
        )
        if min(value - low, high - value) < np.log(1.01)
    ]
    return (
        _discrepancy_of(parameters, provenance=f"fitted, variant {variant.name}"),
        -float(best.fun),
        {"converged": bool(best.success), "at_bound": " ".join(at_bound)},
    )


def heldout_scores(
    heldout: dict | None, name: str, discrepancy: NEEDiscrepancy
) -> dict:
    """The held-out tower's log predictive density and misfit ratio under *discrepancy*."""
    if heldout is None:
        return {}
    source = heldout[name]
    covariance = np.diag(
        source.measurement_standard_deviation**2
    ) + discrepancy.covariance(_days_since_first(source.times))
    factor = np.linalg.cholesky(covariance)
    residuals = source.y[None, :] - source.predictions
    whitened = solve_triangular(factor, residuals.T, lower=True).T
    member_misfits = 0.5 * np.sum(whitened**2, axis=1)
    log_normalizer = np.sum(np.log(np.diag(factor))) + 0.5 * source.y.size * np.log(
        2 * np.pi
    )
    log_densities = -member_misfits - log_normalizer
    return {
        "heldout_log_density": float(
            logsumexp(log_densities) - np.log(len(log_densities))
        ),
        "heldout_ratio": float(2 * np.median(member_misfits) / source.y.size),
    }


# ── helpers ──


def _negative_log_likelihood(variant, residual, times, measurement_variance):
    """``-l_k(phi)`` and its gradient in the log parameters, for SciPy."""
    n = residual.size

    def value(log_parameters):
        parameters = dict(
            zip(variant.parameter_names, jnp.exp(log_parameters), strict=True)
        )
        covariance = jnp.diag(measurement_variance) + _discrepancy_of(
            parameters
        ).covariance(times, xp=jnp)
        factor = jnp.linalg.cholesky(covariance)
        whitened = jax.scipy.linalg.solve_triangular(factor, residual, lower=True)
        return (
            0.5 * whitened @ whitened
            + jnp.sum(jnp.log(jnp.diag(factor)))
            + 0.5 * n * jnp.log(2 * jnp.pi)
        )

    value_and_gradient = jax.jit(jax.value_and_grad(value))

    def scipy_objective(log_parameters):
        result, gradient = value_and_gradient(jnp.asarray(log_parameters))
        if not np.isfinite(result):
            return np.inf, np.zeros_like(log_parameters)
        return float(result), np.asarray(gradient, dtype=float)

    return scipy_objective


def _discrepancy_of(parameters: dict, provenance: str = "") -> NEEDiscrepancy:
    """A discrepancy from fitted values, terms left out at 0, timescales in days."""
    return NEEDiscrepancy(
        short_standard_deviation=parameters["short_standard_deviation"],
        short_timescale=parameters["short_timescale"],
        long_standard_deviation=parameters.get("long_standard_deviation", 0.0),
        long_timescale=parameters.get("long_timescale", 30.0),
        recurring_standard_deviation=parameters.get(
            "recurring_standard_deviation", 0.0
        ),
        recurring_width=parameters.get("recurring_width", 0.5),
        recurring_timescale=parameters.get("recurring_timescale"),
        provenance=provenance,
    )


def _days_since_first(times: pd.DatetimeIndex) -> np.ndarray:
    """Window ends in days since the first: ``model/noise.py``'s times, shifted."""
    return ((times - times[0]) / pd.Timedelta(days=1)).to_numpy()


def _parser() -> argparse.ArgumentParser:
    """The command line: which EKI run, by its data and its setup."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", choices=("synthetic", "observed"), required=True)
    parser.add_argument(
        "--setup",
        default=config.EKI_RUN_NAME,
        help="the EKI setup whose run is fitted; config.EKI_RUN_NAME by default",
    )
    return parser


# ── supporting types, and the variants ──
#
# The variants are built last: their construction needs Variant.


@dataclass(frozen=True)
class Variant:
    """One form of the discrepancy: its free parameters, their bounds and starts."""

    name: str
    parameter_names: tuple[str, ...]
    starts: tuple[dict[str, float], ...]


#: The bounds of each parameter, in its natural units (umol m-2 s-1, days, or
#: dimensionless for the recurring width); the fit works in their logarithms.
BOUNDS = frozendict(
    {
        "short_standard_deviation": (1e-3, 20.0),
        "short_timescale": (0.05, 60.0),
        "long_standard_deviation": (1e-3, 20.0),
        "long_timescale": (5.0, 730.0),
        "recurring_standard_deviation": (1e-3, 20.0),
        "recurring_width": (0.05, 3.0),
        "recurring_timescale": (180.0, 36500.0),
    }
)

#: The variants fitted, each from its starts; a start's standard deviations
#: are fractions of the source's non-measurement residual sd, its timescales
#: in days.
VARIANTS = (
    Variant(
        "single",
        ("short_standard_deviation", "short_timescale"),
        ({"short_standard_deviation": 1.0, "short_timescale": 2.0},),
    ),
    Variant(
        "two_term",
        (
            "short_standard_deviation",
            "short_timescale",
            "long_standard_deviation",
            "long_timescale",
        ),
        (
            {
                "short_standard_deviation": 0.7,
                "short_timescale": 1.0,
                "long_standard_deviation": 0.6,
                "long_timescale": 30.0,
            },
            {
                "short_standard_deviation": 0.5,
                "short_timescale": 0.5,
                "long_standard_deviation": 0.8,
                "long_timescale": 60.0,
            },
        ),
    ),
    Variant(
        "three_term",
        (
            "short_standard_deviation",
            "short_timescale",
            "long_standard_deviation",
            "long_timescale",
            "recurring_standard_deviation",
            "recurring_width",
        ),
        (
            {
                "short_standard_deviation": 0.7,
                "short_timescale": 1.0,
                "long_standard_deviation": 0.5,
                "long_timescale": 30.0,
                "recurring_standard_deviation": 0.4,
                "recurring_width": 0.5,
            },
            {
                "short_standard_deviation": 0.5,
                "short_timescale": 0.5,
                "long_standard_deviation": 0.7,
                "long_timescale": 60.0,
                "recurring_standard_deviation": 0.4,
                "recurring_width": 0.3,
            },
            {
                "short_standard_deviation": 0.8,
                "short_timescale": 2.0,
                "long_standard_deviation": 0.4,
                "long_timescale": 15.0,
                "recurring_standard_deviation": 0.5,
                "recurring_width": 1.0,
            },
        ),
    ),
    Variant(
        "drifting",
        (
            "short_standard_deviation",
            "short_timescale",
            "long_standard_deviation",
            "long_timescale",
            "recurring_standard_deviation",
            "recurring_width",
            "recurring_timescale",
        ),
        (
            {
                "short_standard_deviation": 0.7,
                "short_timescale": 1.0,
                "long_standard_deviation": 0.5,
                "long_timescale": 30.0,
                "recurring_standard_deviation": 0.4,
                "recurring_width": 0.5,
                "recurring_timescale": 1500.0,
            },
        ),
    ),
)

#: The parameters given in units of the non-measurement residual sd at the
#: starts.
_RELATIVE_START_NAMES = (
    "short_standard_deviation",
    "long_standard_deviation",
    "recurring_standard_deviation",
)


# ── checks ──


def check_residuals_are_finite(residual: np.ndarray) -> None:
    """Every residual the fit reads is finite."""
    if not np.isfinite(residual).all():
        raise ValueError(
            "a residual is not finite; diagnose the run and check its failed "
            "members before fitting"
        )


if __name__ == "__main__":
    sys.exit(main())
