"""Tests of tempered SMC and importance sampling, all against analytic targets.

Three targets with exact answers, none of them SIPNET:

- a **linear-Gaussian** inverse problem, ``y = A theta + e``, whose posterior,
  evidence and every tempered target are Gaussian in closed form; with a
  failed half-space ``theta_1 > c`` its truncated posterior mean and evidence
  are too;
- a strongly contracted 15-dimensional **Gaussian** posterior, and
- a 15-dimensional **banana**, a curved ridge in ``(theta_1, theta_2)``, whose
  exact posterior is sampled on a grid and whose evidence is a
  one-dimensional quadrature.

The last two, with prior ``N(0, I)`` and a base density at the exact
posterior moments with the covariance doubled (an optimistic stand-in for an
EKI ensemble's), are the targets a survey of gradient-free samplers for this
calibration was benchmarked on.

Every run has a fixed seed, so the tests are deterministic; each tolerance is
about twice the largest error seen over eight seeds at the same settings,
and is stated beside it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from enskit.algorithms.eki import effective_sample_size as enskit_effective_sample_size
from scipy import integrate, stats

from sipnet_calibration import smc

# ── the targets ───────────────────────────────────────────────────────────────


class Counter:
    """A batched log likelihood that records every point it is called at."""

    def __init__(self, log_likelihood):
        self.log_likelihood = log_likelihood
        self.batches: list[np.ndarray] = []

    def __call__(self, theta):
        self.batches.append(np.array(theta))
        return self.log_likelihood(theta)

    @property
    def points(self) -> np.ndarray:
        return np.concatenate(self.batches)


class LinearGaussian:
    """``theta ~ N(m0, C0)``, ``y = A theta + e``, ``e ~ N(0, R)``, all exact."""

    def __init__(self, dimension=4, n_observations=6, noise=0.05, seed=0):
        rng = np.random.default_rng(seed)
        factor = rng.standard_normal((dimension, dimension))
        self.prior_mean = rng.standard_normal(dimension)
        self.prior_covariance = factor @ factor.T / dimension + np.eye(dimension)
        self.forward = rng.standard_normal((n_observations, dimension))
        self.noise_covariance = noise**2 * np.eye(n_observations)
        truth = self.prior_mean + np.linalg.cholesky(self.prior_covariance) @ rng.standard_normal(dimension)
        self.y = self.forward @ truth + noise * rng.standard_normal(n_observations)
        precision = np.linalg.inv(self.prior_covariance) + self.forward.T @ np.linalg.solve(
            self.noise_covariance, self.forward
        )
        self.posterior_covariance = np.linalg.inv(precision)
        self.posterior_mean = self.posterior_covariance @ (
            np.linalg.solve(self.prior_covariance, self.prior_mean)
            + self.forward.T @ np.linalg.solve(self.noise_covariance, self.y)
        )
        self.log_evidence = stats.multivariate_normal(
            self.forward @ self.prior_mean,
            self.forward @ self.prior_covariance @ self.forward.T + self.noise_covariance,
        ).logpdf(self.y)
        self.prior = smc.MultivariateStudentT(mean=self.prior_mean, covariance=self.prior_covariance)
        self.posterior_sd = np.sqrt(np.diag(self.posterior_covariance))

    def log_likelihood(self, theta):
        residuals = self.y - np.asarray(theta) @ self.forward.T
        return np.atleast_1d(stats.multivariate_normal(cov=self.noise_covariance).logpdf(residuals))

    def base(self, *, shift=2.0, inflation=3.0, degrees_of_freedom=5.0):
        """A base density off the posterior by *shift* posterior sds and wider."""
        return smc.MultivariateStudentT(
            mean=self.posterior_mean + shift * self.posterior_sd,
            covariance=inflation * self.posterior_covariance,
            degrees_of_freedom=degrees_of_freedom,
        )

    def tempered_moments(self, base, beta):
        """The mean of ``pi_beta``, Gaussian when the base is."""
        base_precision = np.linalg.inv(np.asarray(base.covariance))
        posterior_precision = np.linalg.inv(self.posterior_covariance)
        precision = (1 - beta) * base_precision + beta * posterior_precision
        return np.linalg.solve(
            precision,
            (1 - beta) * base_precision @ np.asarray(base.mean)
            + beta * posterior_precision @ self.posterior_mean,
        )


def _survey_gaussian_part(dimension, seed=0):
    rng = np.random.default_rng(seed)
    scales = np.logspace(np.log10(0.01), np.log10(0.5), dimension)
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    rotation = q * np.sign(np.diag(r))
    return rotation @ (rng.standard_normal(dimension) * 0.3), (rotation * scales**2) @ rotation.T


class SurveyGaussian:
    """Posterior ``N(m, C)`` in 15 dimensions, sds 0.01 to 0.5, prior ``N(0, I)``."""

    dimension = 15

    def __init__(self):
        self.mean, self.covariance = _survey_gaussian_part(self.dimension)
        self.precision = np.linalg.inv(self.covariance)
        self.log_det = np.linalg.slogdet(self.covariance)[1]

    def log_likelihood(self, theta):
        z = theta - self.mean
        log_posterior = -0.5 * np.einsum("...i,ij,...j->...", z, self.precision, z) - 0.5 * self.log_det
        return log_posterior + 0.5 * np.sum(theta**2, axis=-1)

    def reference(self, n, rng):
        return rng.multivariate_normal(self.mean, self.covariance, size=n)


class SurveyBanana:
    """``(theta_1, theta_2 + 2 theta_1^2)`` observed with sds ``(1, 0.05)``; the rest Gaussian."""

    dimension = 15
    curvature, sd1, sd2 = 2.0, 1.0, 0.05

    def __init__(self):
        self.mean, self.covariance = _survey_gaussian_part(self.dimension - 2)
        self.precision = np.linalg.inv(self.covariance)
        self.log_det = np.linalg.slogdet(self.covariance)[1]

    def log_likelihood(self, theta):
        t1, t2, rest = theta[..., 0], theta[..., 1], theta[..., 2:]
        ridge = -0.5 * (t1 / self.sd1) ** 2 - 0.5 * ((t2 + self.curvature * t1**2) / self.sd2) ** 2
        z = rest - self.mean
        gaussian = -0.5 * np.einsum("...i,ij,...j->...", z, self.precision, z) - 0.5 * self.log_det
        return ridge + gaussian + 0.5 * np.sum(rest**2, axis=-1)

    def reference(self, n, rng):
        grid = np.linspace(-6, 6, 200001)
        variance = 1 + self.sd2**2
        log_density = -0.5 * grid**2 - 0.5 * (grid / self.sd1) ** 2 - 0.5 * (self.curvature * grid**2) ** 2 / variance
        cumulative = np.cumsum(np.exp(log_density - log_density.max()))
        t1 = np.interp(rng.uniform(size=n), cumulative / cumulative[-1], grid)
        precision2 = 1 + 1 / self.sd2**2
        t2 = (-self.curvature * t1**2 / self.sd2**2) / precision2 + rng.standard_normal(n) / np.sqrt(precision2)
        rest = rng.multivariate_normal(self.mean, self.covariance, size=n)
        return np.column_stack([t1, t2, rest])

    def log_evidence(self):
        """``log int N(theta; 0, I) L(theta)``: the rest integrates to 1, theta_2 in closed form."""
        s = self.sd2

        def integrand(t1):
            # int N(t2; 0, 1) exp(-(t2 + a)^2 / (2 s^2)) dt2 = s sqrt(2 pi) N(a; 0, 1 + s^2)
            ridge = s * math.sqrt(2 * math.pi) * stats.norm.pdf(self.curvature * t1**2, scale=math.sqrt(1 + s**2))
            return stats.norm.pdf(t1) * math.exp(-0.5 * (t1 / self.sd1) ** 2) * ridge

        return math.log(integrate.quad(integrand, -8, 8, points=[0.0], limit=200)[0])


def run(problem, settings):
    """A run to its end, and every state it passed through."""
    states = [smc.initial_state(problem, settings)]
    states.extend(smc.run_smc(problem, states[0]))
    return states


def weighted_moments(state):
    weights = np.asarray(state.weights)
    theta = np.asarray(state.theta)
    mean = weights @ theta
    centered = theta - mean
    return mean, (weights[:, None] * centered).T @ centered


def survey_problem(target, *, counter=None):
    reference = target.reference(100_000, np.random.default_rng(99))
    base = smc.MultivariateStudentT(mean=reference.mean(0), covariance=2 * np.cov(reference.T))
    prior = smc.MultivariateStudentT(mean=np.zeros(target.dimension), covariance=np.eye(target.dimension))
    return smc.TemperingProblem(
        log_prior=prior.log_prob, log_likelihood=counter or target.log_likelihood, base=base
    )


# ── densities ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("degrees_of_freedom", [None, 5.0])
def test_student_t_log_density_is_scipys(degrees_of_freedom):
    rng = np.random.default_rng(0)
    factor = rng.standard_normal((3, 3))
    mean, covariance = rng.standard_normal(3), factor @ factor.T + np.eye(3)
    density = smc.MultivariateStudentT(mean=mean, covariance=covariance, degrees_of_freedom=degrees_of_freedom)
    theta = rng.standard_normal((50, 3)) * 3
    if degrees_of_freedom is None:
        expected = stats.multivariate_normal(mean, covariance).logpdf(theta)
    else:
        nu = degrees_of_freedom
        expected = stats.multivariate_t(mean, covariance * (nu - 2) / nu, df=nu).logpdf(theta)
    np.testing.assert_allclose(np.asarray(density.log_prob(theta)), expected, rtol=1e-12)


def test_student_t_draws_have_its_mean_and_covariance():
    covariance = np.array([[2.0, 0.6], [0.6, 1.0]])
    density = smc.MultivariateStudentT(mean=[1.0, -1.0], covariance=covariance, degrees_of_freedom=6.0)
    draws = np.asarray(density.sample(np.random.default_rng(1), 400_000))
    np.testing.assert_allclose(draws.mean(0), [1.0, -1.0], atol=0.01)
    # The t's covariance converges slowly (its fourth moment is barely finite at nu = 6).
    np.testing.assert_allclose(np.cov(draws.T), covariance, atol=0.05)


def test_defensive_mixture_is_the_mixture_of_its_densities():
    component = smc.MultivariateStudentT(mean=[0.0, 0.0], covariance=0.1 * np.eye(2))
    defensive = smc.MultivariateStudentT(mean=[1.0, 1.0], covariance=np.eye(2))
    mixture = smc.DefensiveMixture(component=component, defensive=defensive, defensive_fraction=0.2)
    theta = np.random.default_rng(2).standard_normal((20, 2))
    expected = np.log(
        0.8 * np.exp(np.asarray(component.log_prob(theta))) + 0.2 * np.exp(np.asarray(defensive.log_prob(theta)))
    )
    np.testing.assert_allclose(np.asarray(mixture.log_prob(theta)), expected, rtol=1e-12)
    draws = np.asarray(mixture.sample(np.random.default_rng(3), 100_000))
    np.testing.assert_allclose(draws.mean(0), [0.2, 0.2], atol=0.01)


def test_fit_student_t_is_the_weighted_mean_and_inflated_unbiased_covariance():
    rng = np.random.default_rng(4)
    samples, log_weights = rng.standard_normal((200, 3)), rng.standard_normal(200)
    fitted = smc.fit_student_t(samples, log_weights=log_weights, covariance_inflation=2.0)
    weights = np.exp(log_weights)
    np.testing.assert_allclose(np.asarray(fitted.mean), np.average(samples, axis=0, weights=weights))
    np.testing.assert_allclose(np.asarray(fitted.covariance), 2.0 * np.cov(samples.T, aweights=weights), rtol=1e-12)


def test_student_t_refuses_a_covariance_that_is_not_positive_definite():
    with pytest.raises(ValueError, match="positive definite"):
        smc.MultivariateStudentT(mean=[0.0, 0.0], covariance=[[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="degrees of freedom exceed 2"):
        smc.MultivariateStudentT(mean=[0.0], covariance=[[1.0]], degrees_of_freedom=2.0)


# ── the pieces ────────────────────────────────────────────────────────────────


def test_cess_with_equal_weights_is_enskits_effective_sample_size_over_n():
    log_ratios = np.random.default_rng(5).standard_normal(300) * 40
    log_weights = np.full(300, -math.log(300))
    for increment in (0.0, 0.01, 0.1, 1.0):
        expected = float(enskit_effective_sample_size(-log_ratios, increment)) / 300
        assert smc.conditional_effective_sample_size(log_weights, log_ratios, increment) == pytest.approx(expected)


def test_next_increment_is_the_largest_meeting_the_target_relative_to_the_valid_mass():
    rng = np.random.default_rng(6)
    log_ratios = rng.standard_normal(500) * 30
    log_ratios[:300] = -np.inf  # 60% failed: a target relative to 1 would have no root
    log_weights = np.full(500, -math.log(500))
    increment = smc.next_increment(log_weights, log_ratios, cess_fraction=0.5, max_increment=1.0)
    target = 0.5 * smc.conditional_effective_sample_size(log_weights, log_ratios, 0.0)
    assert target == pytest.approx(0.5 * 0.4)
    assert 0.0 < increment < 1.0
    assert smc.conditional_effective_sample_size(log_weights, log_ratios, increment) >= target
    assert smc.conditional_effective_sample_size(log_weights, log_ratios, increment * (1 + 1e-9)) < target


def test_systematic_resampling_takes_each_sample_about_n_times_its_weight_and_never_a_zero():
    rng = np.random.default_rng(7)
    log_weights = rng.standard_normal(100)
    log_weights[::3] = -np.inf
    log_weights[-5:] = -np.inf  # trailing zero weights, which a cumulative sum rounding to below 1 would take
    weights = np.exp(log_weights - np.logaddexp.reduce(log_weights))
    for seed in range(50):
        counts = np.bincount(smc.systematic_resample(log_weights, 100, np.random.default_rng(seed)), minlength=100)
        assert np.all(np.abs(counts - 100 * weights) < 1 + 1e-9)
        assert np.all(counts[~np.isfinite(log_weights)] == 0)


# ── exactness ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("from_prior", "mean_tolerance", "evidence_tolerance"),
    # Largest over eight seeds, from the base and from the prior: mean or sd
    # error 0.054 and 0.047 posterior sds, evidence error 0.059 and 0.113.
    [(False, 0.11, 0.12), (True, 0.1, 0.23)],
)
def test_tempered_smc_recovers_the_linear_gaussian_posterior_and_evidence(
    from_prior, mean_tolerance, evidence_tolerance
):
    problem_ = LinearGaussian()
    base = problem_.prior if from_prior else problem_.base()
    problem = smc.TemperingProblem(log_prior=problem_.prior.log_prob, log_likelihood=problem_.log_likelihood, base=base)
    final = run(problem, smc.SMCSettings(n_samples=2000, seed=0))[-1]
    mean, covariance = weighted_moments(final)
    assert final.finished and final.beta == 1.0
    assert np.max(np.abs(mean - problem_.posterior_mean) / problem_.posterior_sd) < mean_tolerance
    assert np.max(np.abs(np.sqrt(np.diag(covariance)) / problem_.posterior_sd - 1)) < mean_tolerance
    assert abs(final.log_evidence - problem_.log_evidence) < evidence_tolerance


@pytest.mark.parametrize("bounds", [(0.0, 0.0), (1.0, 1.0)], ids=["random_walk", "independent"])
def test_each_kernel_alone_recovers_the_linear_gaussian_posterior(bounds):
    problem_ = LinearGaussian()
    problem = smc.TemperingProblem(
        log_prior=problem_.prior.log_prob, log_likelihood=problem_.log_likelihood, base=problem_.base()
    )
    final = run(problem, smc.SMCSettings(n_samples=2000, seed=0, independent_fraction_bounds=bounds))[-1]
    mean, covariance = weighted_moments(final)
    # Largest over eight seeds, either kernel: mean or sd error 0.049
    # posterior sds, evidence error 0.117.
    assert np.max(np.abs(mean - problem_.posterior_mean) / problem_.posterior_sd) < 0.1
    assert np.max(np.abs(np.sqrt(np.diag(covariance)) / problem_.posterior_sd - 1)) < 0.1
    assert abs(final.log_evidence - problem_.log_evidence) < 0.24


@pytest.mark.parametrize("bounds", [(0.0, 0.0), (1.0, 1.0)], ids=["random_walk", "independent"])
def test_each_kernel_proposes_from_its_stated_distribution(bounds):
    target = SurveyGaussian()
    counter = Counter(target.log_likelihood)
    problem = survey_problem(target, counter=counter)
    settings = smc.SMCSettings(n_samples=20_000, seed=9, independent_fraction_bounds=bounds)
    resampled = next(smc.run_smc(problem, smc.initial_state(problem, settings)))
    next(smc.run_smc(problem, resampled))
    proposals, theta = counter.batches[1], np.asarray(resampled.theta)
    covariance = np.asarray(resampled.proposal_covariance)
    whitening = np.linalg.inv(np.linalg.cholesky(covariance))
    if bounds == (0.0, 0.0):
        # theta' - theta ~ N(0, lambda^2 C-hat), lambda = 2.38 / sqrt(D) at stage 1
        increments = (proposals - theta) @ whitening.T / resampled.random_walk_scale
        assert resampled.random_walk_scale == pytest.approx(2.38 / math.sqrt(15))
        np.testing.assert_allclose(increments.mean(0), 0.0, atol=0.03)
    else:
        # theta' ~ t_5(m-hat, C-hat): whitened, mean 0 and covariance I
        increments = (proposals - np.asarray(resampled.proposal_mean)) @ whitening.T
    # 20000 draws: a covariance entry's standard error is about 0.01 (0.02 for the t).
    np.testing.assert_allclose(np.cov(increments.T), np.eye(15), atol=0.1)


def test_every_stage_samples_its_tempered_target():
    problem_ = LinearGaussian()
    base = problem_.base(degrees_of_freedom=None)
    problem = smc.TemperingProblem(log_prior=problem_.prior.log_prob, log_likelihood=problem_.log_likelihood, base=base)
    stage_ends = [
        state for state in run(problem, smc.SMCSettings(n_samples=2000, seed=1)) if state.phase in ("reweight", "done")
    ][1:]
    assert len(stage_ends) >= 2
    for state in stage_ends:
        # A Gaussian base makes every pi_beta Gaussian, with this mean. The
        # largest error over eight seeds is 3.2 standard errors of 2000
        # independent draws.
        expected = problem_.tempered_moments(base, state.beta)
        spread = np.asarray(state.theta).std(0)
        assert np.all(np.abs(np.asarray(state.theta).mean(0) - expected) < 6.5 * spread / math.sqrt(2000))


def test_one_step_is_importance_sampling_from_the_base():
    problem_ = LinearGaussian()
    counter = Counter(problem_.log_likelihood)
    base = problem_.base(shift=0.3, inflation=1.5)
    problem = smc.TemperingProblem(log_prior=problem_.prior.log_prob, log_likelihood=counter, base=base)
    states = run(problem, smc.SMCSettings(n_samples=20_000, seed=2, one_step=True))
    initial, final = states[0], states[-1]
    log_ratios = (
        np.asarray(initial.log_prior) + np.asarray(initial.log_likelihood) - np.asarray(initial.log_base)
    )
    assert len(states) == 2 and final.finished and len(counter.batches) == 1
    np.testing.assert_array_equal(np.asarray(final.theta), np.asarray(initial.theta))
    np.testing.assert_allclose(np.asarray(final.log_weights), log_ratios - np.logaddexp.reduce(log_ratios), atol=1e-12)
    assert final.log_evidence == pytest.approx(np.logaddexp.reduce(log_ratios) - math.log(20_000), abs=1e-12)
    assert final.records[-1]["pareto_k"] == pytest.approx(smc.pareto_k(log_ratios))
    # With a good base the estimates are the posterior's; the largest errors
    # over eight seeds are 0.025 posterior sds and 0.010 of the log evidence,
    # and k-hat is at most -0.25.
    mean, _ = weighted_moments(final)
    assert np.max(np.abs(mean - problem_.posterior_mean) / problem_.posterior_sd) < 0.05
    assert abs(final.log_evidence - problem_.log_evidence) < 0.02
    assert final.records[-1]["pareto_k"] < final.records[-1]["pareto_k_threshold"]


def test_one_step_from_a_poor_base_is_flagged_by_its_pareto_k():
    problem_ = LinearGaussian()
    problem = smc.TemperingProblem(
        log_prior=problem_.prior.log_prob, log_likelihood=problem_.log_likelihood, base=problem_.base()
    )
    record = run(problem, smc.SMCSettings(n_samples=2000, seed=3, one_step=True))[-1].records[-1]
    # Over eight seeds k-hat exceeds its threshold by 0.54 to 0.99, and the
    # ESS is at most 16 of 2000.
    assert record["pareto_k"] > record["pareto_k_threshold"] + 0.3
    assert record["ess"] < 0.02 * 2000


@pytest.mark.parametrize(
    ("from_prior", "mean_tolerance", "evidence_tolerance"),
    # Largest over eight seeds, from the base and from the prior: mean error
    # 0.050 and 0.057 posterior sds, evidence error 0.047 and 0.28.
    [(False, 0.1, 0.1), (True, 0.12, 0.55)],
)
def test_failed_runs_truncate_the_posterior_and_its_evidence(from_prior, mean_tolerance, evidence_tolerance):
    problem_ = LinearGaussian()
    mean, sd = problem_.posterior_mean[0], problem_.posterior_sd[0]
    cut = mean + 0.5 * sd  # fails on about 31% of the posterior's mass
    counter = Counter(
        lambda theta: np.where(theta[:, 0] > cut, np.nan, problem_.log_likelihood(theta))
    )
    base = problem_.prior if from_prior else problem_.base(shift=0.0, inflation=2.0)
    problem = smc.TemperingProblem(log_prior=problem_.prior.log_prob, log_likelihood=counter, base=base)
    final = run(problem, smc.SMCSettings(n_samples=2000, seed=4))[-1]
    # The posterior truncated to theta_1 <= cut: theta_1 is a truncated normal,
    # and the rest regress on it.
    a = (cut - mean) / sd
    truncated_mean_1 = mean - sd * stats.norm.pdf(a) / stats.norm.cdf(a)
    covariance = problem_.posterior_covariance
    expected = problem_.posterior_mean + covariance[:, 0] / covariance[0, 0] * (truncated_mean_1 - mean)
    theta = np.asarray(final.theta)
    assert np.all(theta[:, 0] <= cut)
    assert np.all(np.isfinite(np.asarray(final.log_likelihood)))
    assert sum(record["failed_runs"] for record in final.records) > 0
    assert np.isnan(counter.log_likelihood(counter.points)).any()
    assert np.max(np.abs(theta.mean(0) - expected) / problem_.posterior_sd) < mean_tolerance
    assert abs(final.log_evidence - (problem_.log_evidence + stats.norm.logcdf(a))) < evidence_tolerance


def test_tempered_smc_recovers_the_survey_gaussian():
    target = SurveyGaussian()
    final = run(survey_problem(target), smc.SMCSettings(n_samples=1000, seed=0))[-1]
    reference = target.reference(200_000, np.random.default_rng(123))
    theta = np.asarray(final.theta)
    whitening = np.linalg.inv(np.linalg.cholesky(target.covariance))
    # Largest over eight seeds: mean error 0.089 posterior sds, whitened
    # covariance error 0.27 (1000 independent draws give about 0.24), in 23
    # to 38 calls.
    assert np.max(np.abs(theta.mean(0) - reference.mean(0)) / reference.std(0)) < 0.18
    assert np.linalg.norm(whitening @ np.cov(theta.T) @ whitening.T - np.eye(15), 2) < 0.5


def test_tempered_smc_recovers_the_survey_banana_and_its_evidence():
    target = SurveyBanana()
    final = run(survey_problem(target), smc.SMCSettings(n_samples=1000, seed=0))[-1]
    reference = target.reference(200_000, np.random.default_rng(123))
    theta = np.asarray(final.theta)
    scale = reference.std(0)
    # Largest over eight seeds: mean error 0.22 posterior sds, W1 distances
    # 0.11 and 0.22 of theta_1's and theta_2's, evidence error 0.13, in 79 to
    # 81 calls.
    assert np.max(np.abs(theta.mean(0) - reference.mean(0)) / scale) < 0.45
    for i, tolerance in ((0, 0.22), (1, 0.45)):
        assert stats.wasserstein_distance(theta[:, i], reference[:, i]) / scale[i] < tolerance
    assert abs(final.log_evidence - target.log_evidence()) < 0.3


def test_one_step_on_the_survey_banana_is_flagged_by_its_effective_sample_size():
    target = SurveyBanana()
    record = run(survey_problem(target), smc.SMCSettings(n_samples=5000, seed=0, one_step=True))[-1].records[-1]
    # Over eight seeds the ESS is 1.0% to 1.6% of the draws. k-hat is the
    # weaker flag here: it exceeds its threshold, 0.70, in three of the eight,
    # ranging from 0.52 to 0.90.
    assert record["ess"] < 0.03 * 5000


# ── the likelihood's calls ────────────────────────────────────────────────────


def test_the_likelihood_is_called_once_per_new_batch_and_never_at_a_point_twice():
    problem_ = LinearGaussian()
    counter = Counter(problem_.log_likelihood)
    problem = smc.TemperingProblem(log_prior=problem_.prior.log_prob, log_likelihood=counter, base=problem_.base())
    final = run(problem, smc.SMCSettings(n_samples=500, seed=5))[-1]
    assert all(batch.shape == (500, 4) for batch in counter.batches)
    assert final.n_calls == len(counter.batches) == 1 + sum(r["n_move_steps"] for r in final.records[1:])
    assert final.n_evaluations == 500 * final.n_calls
    assert np.unique(counter.points, axis=0).shape[0] == counter.points.shape[0]


def test_auxiliary_values_travel_with_their_samples():
    problem_ = LinearGaussian()

    def log_likelihood(theta):
        return smc.LikelihoodEvaluation(
            log_likelihood=problem_.log_likelihood(theta), auxiliary=theta @ problem_.forward.T
        )

    problem = smc.TemperingProblem(log_prior=problem_.prior.log_prob, log_likelihood=log_likelihood, base=problem_.base())
    final = run(problem, smc.SMCSettings(n_samples=500, seed=6))[-1]
    np.testing.assert_array_equal(np.asarray(final.auxiliary), np.asarray(final.theta) @ problem_.forward.T)


# ── resuming ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("phase", "occurrence"), [("move", 0), ("move", 5), ("reweight", 0), ("reweight", 1)]
)
def test_a_resumed_run_equals_an_uninterrupted_one(tmp_path, phase, occurrence):
    problem_ = LinearGaussian()

    def problem(counter):
        return smc.TemperingProblem(log_prior=problem_.prior.log_prob, log_likelihood=counter, base=problem_.base())

    settings = smc.SMCSettings(n_samples=300, seed=7)
    states = run(problem(Counter(problem_.log_likelihood)), settings)
    uninterrupted = states[-1]
    # Interrupt after the yield that left the run in *phase* for the
    # *occurrence*-th time: mid-stage, or between stages.
    interrupt_after = [k for k, state in enumerate(states) if k and state.phase == phase][occurrence]

    counter = Counter(problem_.log_likelihood)
    state = smc.initial_state(problem(counter), settings)
    for k, state in enumerate(smc.run_smc(problem(counter), state), start=1):
        if k == interrupt_after:
            break
    path = smc.save_state(tmp_path / "state.npz", state)
    for state in smc.run_smc(problem(counter), smc.load_state(path)):
        pass
    smc.check_saved_state_equals(state, uninterrupted)
    assert np.unique(counter.points, axis=0).shape[0] == counter.points.shape[0]


def test_a_run_stopped_at_its_stage_limit_continues_under_a_larger_one():
    problem_ = LinearGaussian()
    problem = smc.TemperingProblem(
        log_prior=problem_.prior.log_prob, log_likelihood=problem_.log_likelihood, base=problem_.prior
    )
    stopped = run(problem, smc.SMCSettings(n_samples=300, seed=8, max_stages=2))[-1]
    assert not stopped.finished and stopped.stage == 2 and stopped.beta < 1.0
    continued = list(smc.run_smc(problem, replace(stopped, settings=replace(stopped.settings, max_stages=500))))
    assert continued[-1].finished


# ── edge cases ────────────────────────────────────────────────────────────────


class ZeroBeyond:
    """A base density that is 0 where ``theta_1 > cut``, and never draws there."""

    def __init__(self, inner, cut):
        self.inner, self.cut = inner, cut
        self.dimension = inner.dimension

    def log_prob(self, theta):
        theta = np.atleast_2d(np.asarray(theta))
        return np.where(theta[:, 0] > self.cut, -np.inf, np.asarray(self.inner.log_prob(theta)))

    def sample(self, rng, n_samples):
        draws = np.array(self.inner.sample(rng, n_samples))
        while np.any(outside := draws[:, 0] > self.cut):
            draws[outside] = np.asarray(self.inner.sample(rng, int(outside.sum())))
        return draws


def test_a_base_that_is_zero_somewhere_leaves_the_move_records_finite():
    problem_ = LinearGaussian()
    cut = problem_.posterior_mean[0] + problem_.posterior_sd[0]
    base = ZeroBeyond(problem_.base(shift=0.0, inflation=2.0), cut)
    problem = smc.TemperingProblem(log_prior=problem_.prior.log_prob, log_likelihood=problem_.log_likelihood, base=base)
    final = run(problem, smc.SMCSettings(n_samples=500, seed=10))[-1]
    # At beta = 1 the target is p_0 L, so (1 - beta) log q, 0 times -inf,
    # must not enter it as NaN.
    for record in final.records[1:]:
        assert math.isfinite(record["acceptance"]) and math.isfinite(record["jump_random_walk"])
    assert np.any(np.asarray(final.theta)[:, 0] > cut)


class LastUniform:
    """A generator whose one uniform is the largest float below 1."""

    def random(self):
        return np.nextafter(1.0, 0.0)


@pytest.mark.parametrize("n_samples", [2, 3, 1000, 12345])
def test_systematic_resampling_never_returns_an_index_past_the_last(n_samples):
    log_weights = np.zeros(n_samples)
    log_weights[-1] = -np.inf
    indices = smc.systematic_resample(log_weights, n_samples, LastUniform())
    assert indices.max() == n_samples - 2


def test_pareto_k_is_nan_without_a_tail_to_fit():
    assert math.isnan(smc.pareto_k([0.0] + [-np.inf] * 20))
    assert math.isnan(smc.pareto_k(np.zeros(100)))
    assert math.isfinite(smc.pareto_k(np.random.default_rng(0).standard_normal(100)))


def test_settings_hold_plain_python_values_a_state_file_can_save(tmp_path):
    problem_ = LinearGaussian()
    problem = smc.TemperingProblem(
        log_prior=problem_.prior.log_prob, log_likelihood=problem_.log_likelihood, base=problem_.prior
    )
    settings = smc.SMCSettings(n_samples=np.int64(20), seed=np.int64(1), cess_fraction=np.float64(0.5))
    assert type(settings.n_samples) is int and type(settings.seed) is int
    smc.save_state(tmp_path / "state.npz", smc.initial_state(problem, settings))


# ── refusals ──────────────────────────────────────────────────────────────────


def linear_problem(log_likelihood):
    problem_ = LinearGaussian()
    return smc.TemperingProblem(log_prior=problem_.prior.log_prob, log_likelihood=log_likelihood, base=problem_.prior)


def test_a_log_likelihood_of_positive_infinity_is_refused():
    with pytest.raises(ValueError, match=r"\+inf"):
        smc.initial_state(linear_problem(lambda theta: np.full(len(theta), np.inf)), smc.SMCSettings(n_samples=10, seed=0))


def test_a_log_likelihood_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="one value per sample"):
        smc.initial_state(linear_problem(lambda theta: np.zeros((len(theta), 1))), smc.SMCSettings(n_samples=10, seed=0))


def test_a_run_whose_every_run_failed_is_refused():
    state = smc.initial_state(linear_problem(lambda theta: np.full(len(theta), np.nan)), smc.SMCSettings(n_samples=10, seed=0))
    with pytest.raises(ValueError, match="every sample's run failed"):
        next(smc.run_smc(linear_problem(lambda theta: np.full(len(theta), np.nan)), state))


@pytest.mark.parametrize(
    "overrides",
    [
        {"cess_fraction": 1.0},
        {"min_move_steps": 5, "max_move_steps": 4},
        {"independent_fraction_bounds": (0.6, 0.4)},
        {"n_samples": 1},
        {"seed": -1},
        {"independent_covariance_inflation": 0.0},
        {"independent_degrees_of_freedom": 2.0},
    ],
)
def test_settings_out_of_range_are_refused(overrides):
    with pytest.raises(ValueError):
        smc.SMCSettings(**{"n_samples": 10, "seed": 0, **overrides})


def test_a_one_step_setting_that_is_not_a_boolean_is_refused():
    with pytest.raises(TypeError, match="boolean"):
        smc.SMCSettings(n_samples=10, seed=0, one_step="yes")


def test_a_nan_log_prior_is_refused():
    problem_ = LinearGaussian()
    problem = smc.TemperingProblem(
        log_prior=lambda theta: np.full(len(theta), np.nan), log_likelihood=problem_.log_likelihood, base=problem_.prior
    )
    with pytest.raises(ValueError, match="the log prior is NaN"):
        smc.initial_state(problem, smc.SMCSettings(n_samples=10, seed=0))
