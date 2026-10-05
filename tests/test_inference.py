"""Tests for the inference adapters, against a linear-Gaussian toy in closed form.

The toy is declared in the probability layer as an experiment would declare
SIPNET: a Gaussian prior on three coefficients ``u`` (theta itself, on
``REAL``), a simulator computing ``m = A u`` on five observations, and
``y ~ N(m, R)`` with ``R`` dense. Its posterior, evidence, and the
posterior truncated to the half-space where the simulator runs
(``u_0 <= c``) are all closed-form. Each adapter is checked against them:

- EKI from an ensemble with the prior's exact moments is the posterior to
  floating point, and from prior draws within Monte Carlo error;
- tempered SMC from the prior, and importance sampling from a Student-t,
  give the posterior's moments and the evidence, and tempered SMC from the
  Student-t the truncated posterior's;
- the MCMC log density is the closed-form log posterior, ``-inf`` where the
  simulator fails, and the starting points are the first finite draws.

Every run has a fixed seed, so the tests are deterministic; each Monte Carlo
tolerance is about twice the largest error seen over eight seeds at the
same settings, and is stated beside it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pyeki.eki
from scipy import stats
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration import smc
from sipnet_calibration.inference import (
    EKIProblem,
    PriorBaseDensity,
    batched_log_density,
    eki_problem,
    initial_points,
    log_density,
    tempering_problem,
)
from sipnet_calibration.probability import (
    ArraySpec,
    DenseSpec,
    DeterministicSpec,
    FactorSpec,
    GaussianSpec,
    PosteriorEvaluation,
    Simulator,
    SimulatorOutput,
    condition_on,
    joint,
)

tfd = tfp.distributions

RNG = np.random.default_rng(20261005)
D, N = 3, 5
COEFFICIENTS = ("a", "b", "c")
PRIOR_MEAN = RNG.normal(size=D)
_FACTOR = RNG.normal(size=(D, D))
PRIOR_COVARIANCE = _FACTOR @ _FACTOR.T / D + np.eye(D)
A = RNG.normal(size=(N, D))
_NOISE = RNG.normal(size=(N, N))
R = 0.05 * (_NOISE @ _NOISE.T / N + np.eye(N))
Y = A @ (PRIOR_MEAN + np.linalg.cholesky(PRIOR_COVARIANCE) @ RNG.normal(size=D)) + RNG.multivariate_normal(
    np.zeros(N), R
)

POSTERIOR_COVARIANCE = np.linalg.inv(np.linalg.inv(PRIOR_COVARIANCE) + A.T @ np.linalg.solve(R, A))
POSTERIOR_MEAN = POSTERIOR_COVARIANCE @ (
    np.linalg.solve(PRIOR_COVARIANCE, PRIOR_MEAN) + A.T @ np.linalg.solve(R, Y)
)
POSTERIOR_SD = np.sqrt(np.diag(POSTERIOR_COVARIANCE))
LOG_EVIDENCE = stats.multivariate_normal(A @ PRIOR_MEAN, A @ PRIOR_COVARIANCE @ A.T + R).logpdf(Y)
#: The simulator fails where ``u_0`` exceeds this: the posterior mean of
#: ``u_0``, so the truncation keeps half the posterior's mass.
FAILS_ABOVE = float(POSTERIOR_MEAN[0])


def _truncated_mean_and_log_evidence():
    """The posterior given ``u_0 <= c``: its mean and the log evidence of the truncation."""
    alpha = (FAILS_ABOVE - POSTERIOR_MEAN[0]) / POSTERIOR_SD[0]
    mean_0 = POSTERIOR_MEAN[0] - POSTERIOR_SD[0] * stats.norm.pdf(alpha) / stats.norm.cdf(alpha)
    mean = POSTERIOR_MEAN + POSTERIOR_COVARIANCE[:, 0] / POSTERIOR_COVARIANCE[0, 0] * (mean_0 - POSTERIOR_MEAN[0])
    return mean, LOG_EVIDENCE + stats.norm.logcdf(alpha)


def _log_posterior(theta):
    """The unnormalized log posterior in theta = u, by SciPy."""
    theta = np.atleast_2d(theta)
    prior = stats.multivariate_normal(PRIOR_MEAN, PRIOR_COVARIANCE).logpdf(theta)
    likelihood = stats.multivariate_normal(cov=R).logpdf(Y - theta @ A.T)
    return np.atleast_1d(prior + likelihood)


class Linear(Simulator):
    """``m = A u`` on the observations, failing where ``u_0`` exceeds
    *fails_above*; each call's batch size is recorded."""

    name = "linear"
    given = ("u",)

    def __init__(self, fails_above=np.inf):
        self.fails_above = fails_above
        self.batch_sizes = []

    @property
    def outputs(self):
        return (ArraySpec("m", units="1", indexed_by=("obs",)),)

    def __call__(self, given_values):
        u = given_values["u"].transpose("sample", "coefficient").values
        self.batch_sizes.append(u.shape[0])
        return SimulatorOutput(
            values={"m": u @ A.T}, valid={"m": u[:, 0] <= self.fails_above}, record=("linear", u.shape[0])
        )

    def at(self, coords, outputs):
        return self


def _prior():
    return FactorSpec(
        ArraySpec("u", units="1", element_axes={"coefficient": COEFFICIENTS}),
        law=tfd.MultivariateNormalTriL(jnp.asarray(PRIOR_MEAN), jnp.asarray(np.linalg.cholesky(PRIOR_COVARIANCE))),
    )


def _observed(mean="m"):
    return FactorSpec(
        ArraySpec("y", units="1", indexed_by=("obs",)),
        law=GaussianSpec(mean=mean, covariance=DenseSpec(lambda: jnp.asarray(R))),
    )


def _posterior(fails_above=np.inf, *parts, mean="m"):
    simulator = Linear(fails_above)
    model = joint(_observed(mean), simulator, _prior(), *parts).bind(coords={"obs": np.arange(N)})
    return condition_on(model, {"y": Y}), simulator


def _ensemble_with_the_priors_moments(n):
    """*n* members whose mean and covariance (J - 1 normalized) are the prior's exactly."""
    z = RNG.normal(size=(n, D))
    z -= z.mean(axis=0)
    z = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z.T))).T
    return PRIOR_MEAN + z @ np.linalg.cholesky(PRIOR_COVARIANCE).T


def _weighted_moments(state):
    weights = np.asarray(state.weights)
    theta = np.asarray(state.theta)
    mean = weights @ theta
    centered = theta - mean
    return mean, (weights[:, None] * centered).T @ centered


def _student_t():
    """A base density at the posterior's moments with its covariance doubled,
    an optimistic stand-in for one fitted to an EKI ensemble."""
    return smc.MultivariateStudentT(mean=POSTERIOR_MEAN, covariance=2.0 * POSTERIOR_COVARIANCE, degrees_of_freedom=5.0)


def _run_smc(problem, settings):
    state = smc.initial_state(problem, settings)
    for state in smc.run_smc(problem, state):
        pass
    return state


# ── EKI ───────────────────────────────────────────────────────────────────────


def test_the_eki_problem_reads_the_gaussian_likelihood():
    posterior, _ = _posterior()
    problem = eki_problem(posterior)
    assert isinstance(problem, EKIProblem) and problem.posterior is posterior
    assert problem.last_evaluation is None
    np.testing.assert_array_equal(np.asarray(problem.y), Y)
    np.testing.assert_allclose(np.asarray(problem.noise_covariance.to_dense()), R, rtol=1e-12)
    theta = RNG.normal(size=(4, D))
    np.testing.assert_allclose(np.asarray(problem.forward(theta)), theta @ A.T, rtol=1e-12)
    assert isinstance(problem.last_evaluation, PosteriorEvaluation)
    np.testing.assert_array_equal(np.asarray(problem.last_evaluation.theta), theta)


def test_eki_from_an_ensemble_with_the_priors_moments_is_the_posterior():
    """pyEKI's exactness claim for the affine-Gaussian case, through the adapter."""
    problem = eki_problem(_posterior()[0])
    state = pyeki.eki.EKIState(jnp.asarray(_ensemble_with_the_priors_moments(50)), 0.0, 0, jax.random.key(1))
    result = pyeki.eki.run(
        state, problem.forward, problem.y, problem.noise_covariance,
        schedule=pyeki.eki.FixedSchedule.constant(1.0, n_steps=1),
    )
    ensemble = np.asarray(result.state.ensemble)
    np.testing.assert_allclose(ensemble.mean(axis=0), POSTERIOR_MEAN, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.cov(ensemble.T), POSTERIOR_COVARIANCE, rtol=1e-8, atol=1e-12)


def test_eki_from_prior_draws_is_the_posterior_within_monte_carlo_error():
    problem = eki_problem(_posterior()[0])
    ensemble_key, run_key = jax.random.split(jax.random.key(3))
    state = pyeki.eki.EKIState(problem.initial_ensemble(ensemble_key, 2000), 0.0, 0, run_key)
    result = pyeki.eki.run(
        state, problem.forward, problem.y, problem.noise_covariance,
        schedule=pyeki.eki.AdaptiveESSSchedule(ess_fraction=0.5),
    )
    assert float(result.state.beta) == 1.0
    ensemble = np.asarray(result.state.ensemble)
    # Largest errors over eight seeds: 0.028 posterior sds in the mean, 0.7% in an sd.
    np.testing.assert_array_less(np.abs(ensemble.mean(axis=0) - POSTERIOR_MEAN) / POSTERIOR_SD, 0.06)
    np.testing.assert_allclose(ensemble.std(axis=0, ddof=1), POSTERIOR_SD, rtol=0.015)
    assert problem.last_evaluation.theta.shape == (2000, D)


def test_the_initial_ensemble_is_the_priors_draws():
    posterior, simulator = _posterior()
    problem = eki_problem(posterior)
    key = jax.random.key(5)
    np.testing.assert_array_equal(np.asarray(problem.initial_ensemble(key, 7)), np.asarray(posterior.sample_prior(key, 7)))
    assert simulator.batch_sizes == []


def test_a_failed_sample_is_a_nan_row_of_predictions():
    problem = eki_problem(_posterior(FAILS_ABOVE)[0])
    theta = np.array([[FAILS_ABOVE - 1.0, 0.0, 0.0], [FAILS_ABOVE + 1.0, 0.0, 0.0]])
    predictions = np.asarray(problem.forward(theta))
    np.testing.assert_allclose(predictions[0], theta[0] @ A.T, rtol=1e-12)
    assert np.isnan(predictions[1]).all()
    assert np.asarray(problem.last_evaluation.valid).tolist() == [True, False]


def test_eki_repairs_failed_members_and_runs_to_the_end():
    problem = eki_problem(_posterior(FAILS_ABOVE)[0])
    ensemble_key, run_key = jax.random.split(jax.random.key(4))
    state = pyeki.eki.EKIState(problem.initial_ensemble(ensemble_key, 200), 0.0, 0, run_key)
    with pytest.warns(UserWarning, match="evaluations failed"):
        result = pyeki.eki.run(
            state, problem.forward, problem.y, problem.noise_covariance,
            schedule=pyeki.eki.AdaptiveESSSchedule(ess_fraction=0.5), on_failure="repair",
        )
    assert float(result.state.beta) == 1.0
    assert not bool(np.asarray(problem.last_evaluation.valid).all())


def test_eki_needs_a_gaussian_likelihood():
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("obs",)), law=lambda m: tfd.Independent(tfd.Normal(m, 1.0), 1)),
        Linear(), _prior(),
    )
    posterior = condition_on(spec.bind(coords={"obs": np.arange(N)}), {"y": Y})
    with pytest.raises(ValueError, match="not a GaussianSpec"):
        eki_problem(posterior)


def test_eki_needs_a_posterior():
    with pytest.raises(TypeError, match="must be a Posterior, not dict"):
        eki_problem({})


# ── SMC and importance sampling ───────────────────────────────────────────────


def test_the_prior_base_density_is_the_prior_in_theta():
    posterior, simulator = _posterior()
    base = PriorBaseDensity(posterior)
    assert base.dimension == D
    theta = RNG.normal(size=(4, D))
    np.testing.assert_allclose(
        np.asarray(base.log_prob(theta)), stats.multivariate_normal(PRIOR_MEAN, PRIOR_COVARIANCE).logpdf(theta),
        rtol=1e-12,
    )
    rng = np.random.default_rng(0)
    draws = np.asarray(base.sample(rng, 6))
    assert draws.shape == (6, D)
    np.testing.assert_array_equal(draws, np.asarray(base.sample(np.random.default_rng(0), 6)))
    assert not np.array_equal(draws, np.asarray(base.sample(rng, 6)))
    assert not np.array_equal(draws, np.asarray(base.sample(np.random.default_rng(1), 6)))
    assert simulator.batch_sizes == []


def test_smc_from_the_prior_gives_the_posterior_and_its_evidence():
    posterior, _ = _posterior()
    state = _run_smc(tempering_problem(posterior), smc.SMCSettings(n_samples=1000, seed=2))
    mean, covariance = _weighted_moments(state)
    # Largest errors over eight seeds: 0.096 posterior sds, 4% in an sd, 0.117 in
    # log Z; test_smc's largest evidence error from the prior is 0.113.
    np.testing.assert_array_less(np.abs(mean - POSTERIOR_MEAN) / POSTERIOR_SD, 0.2)
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), POSTERIOR_SD, rtol=0.09)
    assert abs(float(state.log_evidence) - LOG_EVIDENCE) < 0.24


def test_the_predictions_travel_with_each_sample():
    posterior, _ = _posterior()
    state = _run_smc(tempering_problem(posterior), smc.SMCSettings(n_samples=200, seed=2))
    np.testing.assert_allclose(np.asarray(state.auxiliary), np.asarray(state.theta) @ A.T, rtol=1e-10)


def test_importance_sampling_from_a_student_t_gives_the_evidence():
    posterior, simulator = _posterior()
    state = _run_smc(
        tempering_problem(posterior, base=_student_t()), smc.SMCSettings(n_samples=2000, seed=3, one_step=True)
    )
    mean, _ = _weighted_moments(state)
    # Largest errors over eight seeds: 0.038 posterior sds, 0.013 in log Z.
    np.testing.assert_array_less(np.abs(mean - POSTERIOR_MEAN) / POSTERIOR_SD, 0.08)
    assert abs(float(state.log_evidence) - LOG_EVIDENCE) < 0.025
    assert simulator.batch_sizes == [2000]


def test_smc_gives_the_posterior_truncated_to_where_the_simulator_runs():
    posterior, _ = _posterior(FAILS_ABOVE)
    state = _run_smc(tempering_problem(posterior, base=_student_t()), smc.SMCSettings(n_samples=1000, seed=4))
    mean, _ = _weighted_moments(state)
    truncated_mean, truncated_log_evidence = _truncated_mean_and_log_evidence()
    assert (np.asarray(state.theta)[:, 0] <= FAILS_ABOVE).all()
    # Largest errors over eight seeds: 0.037 posterior sds, 0.063 in log Z.
    np.testing.assert_array_less(np.abs(mean - truncated_mean) / POSTERIOR_SD, 0.08)
    assert abs(float(state.log_evidence) - truncated_log_evidence) < 0.13
    assert state.records[0]["failed_runs"] > 0


def test_a_failed_run_is_nan_and_a_traced_failure_minus_infinity():
    """``g = m + sqrt(u_1)`` is NaN where ``u_1 < 0``: the simulator ran, the density did not."""
    shifted = DeterministicSpec(
        ArraySpec("g", units="1", indexed_by=("obs",)), function=lambda m, u: m + jnp.sqrt(u[..., 1:2])
    )
    posterior, _ = _posterior(FAILS_ABOVE, shifted, mean="g")
    theta = np.array([[FAILS_ABOVE - 1.0, 0.5, 0.0], [FAILS_ABOVE - 1.0, -0.5, 0.0], [FAILS_ABOVE + 1.0, 0.5, 0.0]])
    evaluation = tempering_problem(posterior).log_likelihood(theta)
    values = np.asarray(evaluation.log_likelihood)
    expected = stats.multivariate_normal(theta[0] @ A.T + np.sqrt(0.5), R).logpdf(Y)
    np.testing.assert_allclose(values[0], expected, rtol=1e-12)
    assert np.isneginf(values[1])
    assert np.isnan(values[2])
    assert np.isnan(np.asarray(evaluation.auxiliary)[1:]).all()


def test_without_a_gaussian_likelihood_smc_reads_the_log_likelihood_alone():
    """A Normal law about ``m + sqrt(u_1)``: NaN where the run failed, ``-inf`` where the root did."""
    spec = joint(
        FactorSpec(
            ArraySpec("y", units="1", indexed_by=("obs",)),
            law=lambda m, u: tfd.Independent(tfd.Normal(m + jnp.sqrt(u[..., 1:2]), 1.0), 1),
        ),
        Linear(FAILS_ABOVE), _prior(),
    )
    posterior = condition_on(spec.bind(coords={"obs": np.arange(N)}), {"y": Y})
    theta = np.array([[FAILS_ABOVE - 1.0, 0.25, 0.0], [FAILS_ABOVE + 1.0, 0.25, 0.0], [FAILS_ABOVE - 1.0, -0.25, 0.0]])
    values = np.asarray(tempering_problem(posterior).log_likelihood(theta))
    expected = stats.norm(theta[0] @ A.T + 0.5, 1.0).logpdf(Y).sum()
    np.testing.assert_allclose(values[0], expected, rtol=1e-12)
    assert np.isnan(values[1])
    assert np.isneginf(values[2])


@pytest.mark.parametrize("dimension", [D - 1, D + 1])
def test_a_base_of_another_dimension_is_refused(dimension):
    base = smc.MultivariateStudentT(mean=np.zeros(dimension), covariance=np.eye(dimension))
    with pytest.raises(ValueError, match=f"dimension {dimension}, but theta has D = 3"):
        tempering_problem(_posterior()[0], base=base)


# ── gradient-free MCMC ────────────────────────────────────────────────────────


def test_the_batched_log_density_is_the_log_posterior_and_minus_infinity_where_it_fails():
    posterior, simulator = _posterior(FAILS_ABOVE)
    density = batched_log_density(posterior)
    theta = np.array([[FAILS_ABOVE - 1.0, 0.2, -0.3], [FAILS_ABOVE + 1.0, 0.0, 0.0], [FAILS_ABOVE - 0.5, -1.0, 1.0]])
    values = density(theta)
    assert isinstance(values, np.ndarray) and values.dtype == np.float64 and values.shape == (3,)
    np.testing.assert_allclose(values[[0, 2]], _log_posterior(theta[[0, 2]]), rtol=1e-12)
    assert np.isneginf(values[1])
    assert simulator.batch_sizes == [3]


def test_the_log_density_is_one_samples():
    posterior, _ = _posterior(FAILS_ABOVE)
    density = log_density(posterior)
    theta = np.array([FAILS_ABOVE - 1.0, 0.2, -0.3])
    value = density(theta)
    assert isinstance(value, float)
    np.testing.assert_allclose(value, _log_posterior(theta)[0], rtol=1e-12)
    assert density(np.array([FAILS_ABOVE + 1.0, 0.0, 0.0])) == -np.inf


@pytest.mark.parametrize(("make", "theta", "match"), [
    (batched_log_density, np.zeros(D), r"takes theta of shape \(J, D\), not \(3,\); use log_density"),
    (batched_log_density, np.zeros((2, D + 1)), "takes theta with D = 3 entries, not 4"),
    (log_density, np.zeros((2, D)), r"takes theta of shape \(D,\), not \(2, 3\); use batched_log_density"),
    (log_density, np.zeros(D + 1), "takes theta with D = 3 entries, not 4"),
])
def test_a_theta_of_the_wrong_shape_is_refused(make, theta, match):
    with pytest.raises(ValueError, match=match):
        make(_posterior()[0])(theta)


def test_initial_points_are_the_first_finite_prior_draws():
    posterior, simulator = _posterior(FAILS_ABOVE)
    # Key 3's tenth finite draw is the 23rd, inside a batch of ten.
    key = jax.random.key(3)
    theta, evaluation = initial_points(posterior, key, 10, max_draws=100)
    draws = np.asarray(posterior.sample_prior(key, 100))
    expected = draws[draws[:, 0] <= FAILS_ABOVE][:10]
    np.testing.assert_array_equal(theta, expected)
    assert isinstance(evaluation, PosteriorEvaluation)
    direct = posterior.evaluate(expected)
    # BLAS rounds a product by its batch's size, so the stitched values are
    # the direct evaluation's to rounding.
    for field in ("theta", "valid", "simulator_valid"):
        np.testing.assert_array_equal(np.asarray(getattr(evaluation, field)), np.asarray(getattr(direct, field)))
    for field in ("log_prior", "log_likelihood", "log_density"):
        np.testing.assert_allclose(np.asarray(getattr(evaluation, field)), np.asarray(getattr(direct, field)), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(evaluation.values["m"]), np.asarray(direct.values["m"]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(evaluation.log_density), _log_posterior(expected), rtol=1e-12)
    simulator.batch_sizes.clear()
    theta, evaluation = initial_points(posterior, key, 10, max_draws=100)
    # Each batch is the number still needed, so no draw past the tenth finite one,
    # well inside the hundred, is evaluated.
    tenth = int(np.flatnonzero(draws[:, 0] <= FAILS_ABOVE)[9])
    assert tenth < 50 and sum(simulator.batch_sizes) == tenth + 1
    assert simulator.batch_sizes[0] == 10 and len(simulator.batch_sizes) > 1
    assert evaluation.simulator_records["linear"] == tuple(("linear", size) for size in simulator.batch_sizes)


def test_too_few_finite_draws_is_an_error():
    posterior, _ = _posterior(FAILS_ABOVE - 10.0 * POSTERIOR_SD[0] - 10.0)
    with pytest.raises(RuntimeError, match="only 0 of 8 prior draws have a finite posterior density"):
        initial_points(posterior, jax.random.key(0), 2)


def test_one_finite_draw_short_is_an_error():
    posterior, _ = _posterior(FAILS_ABOVE)
    key = jax.random.key(6)
    found = int((np.asarray(posterior.sample_prior(key, 12))[:, 0] <= FAILS_ABOVE).sum())
    assert 0 < found < 12
    with pytest.raises(RuntimeError, match=f"only {found} of 12 prior draws"):
        initial_points(posterior, key, found + 1, max_draws=12)


def test_max_draws_fewer_than_n_is_refused():
    with pytest.raises(ValueError, match="max_draws"):
        initial_points(_posterior()[0], jax.random.key(0), 5, max_draws=4)


class _NoDimension:
    def log_prob(self, theta):
        return theta

    def sample(self, rng, n_samples):
        return None


class _BooleanDimension(_NoDimension):
    dimension = True


@pytest.mark.parametrize(("base", "match"), [
    (_NoDimension(), "a _NoDimension, has no dimension; pass an smc.BaseDensity"),
    (_BooleanDimension(), "dimension must be an integer, not bool"),
])
def test_a_base_without_an_integer_dimension_is_refused(base, match):
    with pytest.raises(TypeError, match=match):
        tempering_problem(_posterior()[0], base=base)


def test_an_initial_ensemble_of_one_member_is_refused():
    with pytest.raises(ValueError, match="n"):
        eki_problem(_posterior()[0]).initial_ensemble(jax.random.key(0), 1)
