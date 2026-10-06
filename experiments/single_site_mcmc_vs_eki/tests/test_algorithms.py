"""Tests of the experiment's numerical pieces on small synthetic problems, no SIPNET.

- ``NoiseModel``: the Gaussian and scale-marginal likelihoods against SciPy
  and against quadrature over the scale.
- ``kalman_step``: both gains against the dense per-particle formula.
- ``scale_marginal_increment``: the ESS it keeps.

Run with ``uv run pytest experiments/single_site_mcmc_vs_eki/tests``.
"""

import numpy as np
import pytest
import scipy.integrate
import scipy.stats

from experiments.single_site_mcmc_vs_eki import config
from experiments.single_site_mcmc_vs_eki.algorithms.eki_gibbs import (
    kalman_step,
    scale_marginal_increment,
)
from experiments.single_site_mcmc_vs_eki.model.likelihood import NoiseModel, SourceNoise

RNG = np.random.default_rng(20261006)


def _covariance(n: int, timescale: float) -> np.ndarray:
    times = np.sort(RNG.uniform(0, 50, n))
    return 0.3 * np.eye(n) + np.exp(-np.abs(times[:, None] - times[None, :]) / timescale)


def _noise_model(sizes=(12, 7, 3), scaled=(True, True, False), inferred=True) -> NoiseModel:
    """A NoiseModel over three sources, built without a posterior."""
    noise_model = NoiseModel.__new__(NoiseModel)
    sources, start = [], 0
    for index, (size, is_scaled) in enumerate(zip(sizes, scaled, strict=True)):
        cholesky = np.linalg.cholesky(_covariance(size, 2.0 + index))
        sources.append(
            SourceNoise(
                name=f"source_{index}",
                positions=slice(start, start + size),
                cholesky=cholesky,
                log_determinant=2 * np.log(np.diag(cholesky)).sum(),
                scaled=is_scaled,
            )
        )
        start += size
    noise_model.sources = tuple(sources)
    noise_model.y = RNG.normal(size=start)
    noise_model.inferred = inferred
    noise_model.shape = config.NOISE_SCALE_SHAPE
    noise_model.prior_scale = float(scipy.stats.gamma(config.NOISE_SCALE_SHAPE).median())
    return noise_model


def _block_diagonal(noise_model, scales_of_one_particle) -> np.ndarray:
    n = noise_model.y.size
    R = np.zeros((n, n))
    for source in noise_model.sources:
        C = source.cholesky @ source.cholesky.T
        R[source.positions, source.positions] = scales_of_one_particle.get(source.name, 1.0) * C
    return R


def test_the_gaussian_likelihood_at_scales_is_scipys():
    noise_model = _noise_model()
    predictions = RNG.normal(size=(4, noise_model.y.size))
    scales = {"source_0": np.full(4, 2.0), "source_1": np.full(4, 0.5)}
    expected = [
        scipy.stats.multivariate_normal(predictions[j], _block_diagonal(noise_model, {k: v[j] for k, v in scales.items()})).logpdf(noise_model.y)
        for j in range(4)
    ]
    np.testing.assert_allclose(noise_model.log_likelihood_at(predictions, scales), expected, rtol=1e-10)


def test_the_marginal_likelihood_integrates_the_scales_out():
    noise_model = _noise_model(sizes=(5, 4), scaled=(True, False))
    prediction = RNG.normal(size=noise_model.y.size)
    prior = scipy.stats.invgamma(noise_model.shape, scale=noise_model.prior_scale)

    def integrand(scale):
        return np.exp(noise_model.log_likelihood_at(prediction[None], {"source_0": np.array([scale])})[0]) * prior.pdf(scale)

    expected, _ = scipy.integrate.quad(integrand, 0, np.inf, limit=200)
    np.testing.assert_allclose(noise_model.log_marginal_likelihood(prediction[None])[0], np.log(expected), rtol=1e-7)


def test_a_failed_run_has_likelihood_minus_infinity():
    noise_model = _noise_model()
    predictions = RNG.normal(size=(2, noise_model.y.size))
    predictions[1, 3] = np.nan
    assert np.isneginf(noise_model.log_likelihood(predictions)[1])
    assert np.isfinite(noise_model.log_likelihood(predictions)[0])


@pytest.mark.parametrize("gain", ["common", "per_particle"])
def test_the_kalman_step_is_the_dense_formula(gain):
    noise_model = _noise_model()
    J, D, increment = 9, 4, 0.37
    A = RNG.normal(size=(noise_model.y.size, D))
    theta = RNG.normal(size=(J, D))
    predictions = theta @ A.T
    scales = {name: RNG.uniform(0.5, 3.0, J) for name in noise_model.scaled_names}
    valid = np.ones(J, bool)
    new = kalman_step(noise_model, theta, predictions, scales, valid, increment, np.random.default_rng(1), gain=gain)
    # The same perturbations, drawn in the same order.
    rng = np.random.default_rng(1)
    if gain == "common":
        used = {k: np.full(J, 1 / np.mean(1 / v)) for k, v in scales.items()}
    else:
        used = scales
    T = (theta - theta.mean(0)) / np.sqrt(J - 1)
    U = (predictions - predictions.mean(0)) / np.sqrt(J - 1)
    if gain == "per_particle":
        Z = np.column_stack([np.log(scales[k]) for k in noise_model.scaled_names])
        Z -= Z.mean(0)
        P = Z @ np.linalg.pinv(Z)
        T, U = T - P @ T, U - P @ U
    eta = np.zeros((J, noise_model.y.size))
    for source in noise_model.sources:
        draw = (source.cholesky @ rng.normal(size=(source.size, J))).T
        scale = used.get(source.name, np.ones(J))
        eta[:, source.positions] = draw * np.sqrt(scale / increment)[:, None]
    expected = np.array([
        theta[j] + T.T @ U @ np.linalg.solve(
            U.T @ U + _block_diagonal(noise_model, {k: v[j] for k, v in used.items()}) / increment,
            noise_model.y - predictions[j] - eta[j],
        )
        for j in range(J)
    ])
    np.testing.assert_allclose(new, expected, rtol=1e-9, atol=1e-12)


def test_the_increment_keeps_the_target_ess():
    noise_model = _noise_model()
    predictions = RNG.normal(scale=3.0, size=(50, noise_model.y.size))
    forms = noise_model.quadratic_forms(predictions)
    increment, ess = scale_marginal_increment(noise_model, forms, np.ones(50, bool), 0.0)
    assert 0 < increment < 1
    assert ess == pytest.approx(config.EKI_ESS_FRACTION * 50, rel=1e-6)
