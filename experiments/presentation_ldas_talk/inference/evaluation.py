"""Predictions from any method's posterior, and the summaries that compare methods.

Two kinds of prediction, both from a method's ``PosteriorMoments`` at the
training sites through ``SpatialPrior.predict``:

- **maps**: the smooth field on the map grid;
- **held-out sites**: the parameters at sites with no data, which carry
  their own site noise too.

Site-by-site calibration makes neither: a site with no data gets only its
prior (``prior_predictions``), and nothing connects one site's estimate to
the next.

Two kinds of score:

- **against the reference** (training sites): how far an approximation's
  mean is from the exact posterior mean, in units of the exact posterior SD,
  and the ratio of its SD to the exact SD;
- **against the truth** (held-out sites): root-mean-square error of the
  predicted mean, and how often the central 90% interval holds the truth.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..toy.spatial_prior import SpatialPrior
from .moments import PosteriorMoments

__all__ = [
    "HeldOutScores",
    "Predictions",
    "held_out_scores",
    "predict_from_posterior",
    "prior_predictions",
    "sd_ratio_to_reference",
    "standardized_error_to_reference",
]

#: The central interval whose coverage is scored.
INTERVAL_PROBABILITY = 0.90


@dataclass(frozen=True)
class Predictions:
    """Predicted mean and SD, each ``(n_locations, n_parameters)``."""

    mean: np.ndarray
    sd: np.ndarray


@dataclass(frozen=True)
class HeldOutScores:
    """Per-parameter scores at the held-out sites, each ``(n_parameters,)``."""

    root_mean_square_error: np.ndarray
    interval_coverage: np.ndarray


# ── predictions ─────────────────────────────────────────────────────────────


def predict_from_posterior(
    prior: SpatialPrior,
    site_locations: np.ndarray,
    posterior: PosteriorMoments,
    target_locations: np.ndarray,
    *,
    include_site_noise: bool,
) -> Predictions:
    """Predict every parameter at ``target_locations`` from a posterior at the sites."""
    means, sds = zip(
        *[
            prior.predict(
                p,
                site_locations,
                posterior.mean[:, p],
                posterior.parameter_covariance(p),
                target_locations,
                include_site_noise=include_site_noise,
            )
            for p in range(prior.n_parameters)
        ]
    )
    return Predictions(mean=np.column_stack(means), sd=np.column_stack(sds))


def prior_predictions(prior: SpatialPrior, target_locations: np.ndarray, *, include_site_noise: bool) -> Predictions:
    """The prior alone at ``target_locations``: the trend, and the prior SD."""
    sd = prior.site_marginal_sd() if include_site_noise else prior.field_sd
    mean = prior.trend(target_locations)
    return Predictions(mean=mean, sd=np.broadcast_to(sd, mean.shape).copy())


# ── scores ──────────────────────────────────────────────────────────────────


def standardized_error_to_reference(posterior: PosteriorMoments, reference: PosteriorMoments) -> np.ndarray:
    """``|mean - reference mean| / reference SD``, ``(n_sites, n_parameters)``."""
    return np.abs(posterior.mean - reference.mean) / reference.sd


def sd_ratio_to_reference(posterior: PosteriorMoments, reference: PosteriorMoments) -> np.ndarray:
    """``SD / reference SD``, ``(n_sites, n_parameters)``."""
    return posterior.sd / reference.sd


def held_out_scores(predictions: Predictions, truth: np.ndarray) -> HeldOutScores:
    """Score predictions against the true values ``(n_sites, n_parameters)``."""
    error = predictions.mean - truth
    half_width = norm.ppf(0.5 + INTERVAL_PROBABILITY / 2) * predictions.sd
    return HeldOutScores(
        root_mean_square_error=np.sqrt(np.mean(error**2, axis=0)),
        interval_coverage=np.mean(np.abs(error) <= half_width, axis=0),
    )
