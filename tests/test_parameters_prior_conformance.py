"""Conformance of the prior to its mathematics.

- The change of variables matches a finite-difference Jacobian.
- Densities in theta integrate to one, by importance sampling, on the
  simplex under ``SoftmaxCentered`` and ``IteratedSigmoidCentered``.
- Joint terms, by base density and by change of variables, and a
  centered hierarchy given its hyperparameters, integrate to one.
- Each builder's declared Gaussian, the copula's included, agrees with
  ``log_prob`` on draws and with the moments of 20 000 draws.
- ``select`` gives the marginal, and a term's draws depend on its name
  and what it is given alone, whatever the declaration order.
- The support checks accept and refuse what they should.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from scipy import stats
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.parameters.parameter import Parameter
from sipnet_calibration.parameters.prior import (
    Prior,
    PriorTerm,
    gaussian_copula,
    iid_over_dim,
    independent_over_dim,
    log_normal,
    logit_normal,
    logit_normal_from_interval,
    softmax_normal,
)
from sipnet_calibration.parameters.support import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    Interval,
)
from sipnet_calibration.parameters.vector import ParameterVector

tfd, tfb = tfp.distributions, tfp.bijectors

SITES = np.array([1, 27, 4711], dtype=np.int32)
PFT_OF_SITE = ("deciduous", "conifer", "deciduous")
PFT = ("conifer", "deciduous")
NAMES = ("a", "b", "c", "d")


def vector_of(*parameters: Parameter) -> ParameterVector:
    coords = {"pft": PFT, "site": SITES}
    used = {d for p in parameters for d in p.indexed_by}
    return ParameterVector(parameters=parameters, coords={d: v for d, v in coords.items() if d in used})


def prior_of(parameter: Parameter, distribution, constants=None) -> Prior:
    term = PriorTerm(distribution, constants=constants or {}, provenance="test")
    return Prior(vector_of(parameter), {parameter.name: term})


def simplex(bijector=None) -> Parameter:
    return Parameter(name="shares", support=SIMPLEX, units="1", shape=(4,), element_labels={"part": NAMES},
                     bijector=bijector)


def by_site(values, sites=SITES) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=np.float64), dims="site", coords={"site": np.asarray(sites)})


def log_density(gaussian, theta) -> np.ndarray:
    """The log density of a Gaussian's moments at *theta*."""
    return stats.multivariate_normal(np.asarray(gaussian.mean), np.asarray(gaussian.covariance)).logpdf(np.asarray(theta))


def integral_by_importance_sampling(prior: Prior, *, n: int = 400_000, scale: float = 3.0) -> float:
    """:math:`\\int \\pi_\\theta = E_q[\\pi_\\theta / q]` with
    :math:`q = \\mathcal N(0, \\mathrm{scale}^2 I)`."""
    dimension = prior.parameter_vector.unconstrained.size
    proposal = tfd.MultivariateNormalDiag(jnp.zeros(dimension), jnp.full(dimension, scale))
    theta = proposal.sample(n, seed=jax.random.key(7))
    return float(jnp.mean(jnp.exp(prior.log_prob(theta) - proposal.log_prob(theta))))


# ── the change of variables ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "parameter, distribution",
    [
        (Parameter(name="rate", support=POSITIVE, units="1"), tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0))),
        (Parameter(name="rate", support=POSITIVE, units="1", bijector=tfb.Softplus()),
         tfd.HalfNormal(jnp.float64(2.0))),
        (Parameter(name="rate", support=POSITIVE, units="1", bijector=tfb.Softplus()),
         log_normal(median=1.0, geometric_sd=2.0)),
        (Parameter(name="share", support=Interval(1.0, 5.0), units="1"),
         tfd.Uniform(jnp.float64(1.0), jnp.float64(5.0))),
    ],
)
def test_the_change_of_variables_matches_finite_differences(parameter, distribution):
    prior = prior_of(parameter, distribution)
    assert prior.describe().iloc[0]["evaluated_by"] == "change of variables"
    theta = jnp.linspace(-3.0, 3.0, 13)
    step = 1e-6
    derivative = (
        parameter.bijector.forward(theta + step) - parameter.bijector.forward(theta - step)
    ) / (2 * step)
    expected = distribution.log_prob(parameter.bijector.forward(theta)) + jnp.log(jnp.abs(derivative))
    np.testing.assert_allclose(prior.log_prob(theta[:, None]), expected, atol=1e-6)


def test_the_dirichlet_jacobian_matches_finite_differences_on_the_first_coordinates():
    parameter = simplex(tfb.IteratedSigmoidCentered())
    dirichlet = tfd.Dirichlet(jnp.asarray([2.0, 3.0, 1.5, 4.0]))
    prior = prior_of(parameter, dirichlet)
    theta = jnp.asarray([0.3, -0.7, 1.1])
    step = 1e-6
    jacobian = np.stack(
        [
            (parameter.bijector.forward(theta + step * e)[:-1] - parameter.bijector.forward(theta - step * e)[:-1])
            / (2 * step)
            for e in np.eye(3)
        ],
        axis=1,
    )
    expected = dirichlet.log_prob(parameter.bijector.forward(theta)) + np.linalg.slogdet(jacobian)[1]
    assert float(prior.log_prob(theta)) == pytest.approx(float(expected), abs=1e-6)


# ── densities integrate to one ────────────────────────────────────────────────


@pytest.mark.parametrize("bijector", [None, tfb.IteratedSigmoidCentered()], ids=["softmax", "iterated"])
def test_a_dirichlet_integrates_to_one_in_theta(bijector):
    prior = prior_of(simplex(bijector), tfd.Dirichlet(jnp.full(4, 2.0)))
    assert integral_by_importance_sampling(prior) == pytest.approx(1.0, abs=0.03)


def test_a_correlated_softmax_normal_integrates_to_one_in_theta():
    covariance = jnp.asarray([[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]])
    distribution = tfd.TransformedDistribution(
        tfd.MultivariateNormalTriL(jnp.asarray([0.2, -0.3, 0.1]), jnp.linalg.cholesky(covariance)),
        tfb.SoftmaxCentered(),
    )
    prior = prior_of(simplex(), distribution)
    assert prior.describe().iloc[0]["evaluated_by"] == "base density"
    assert integral_by_importance_sampling(prior, scale=2.0) == pytest.approx(1.0, abs=0.03)


def test_a_gamma_integrates_to_one_in_theta():
    prior = prior_of(Parameter(name="rate", support=POSITIVE, units="1"), tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)))
    assert integral_by_importance_sampling(prior) == pytest.approx(1.0, abs=0.02)


# ── declarations ──────────────────────────────────────────────────────────────

RATE = Parameter(name="rate", support=POSITIVE, units="1")
SHARE = Parameter(name="share", support=OPEN_UNIT_INTERVAL, units="1")
Q10 = Parameter(name="q10", support=Interval(1.0, 5.0), units="1")
OFFSET = Parameter(name="offset", support=REAL, units=None)
ALLOCATION_BY_PFT = Parameter(name="allocation", support=SIMPLEX, units="1", shape=(4,), indexed_by=("pft",),
                              element_labels={"part": NAMES})
RATE_BY_SITE = Parameter(name="rate", support=POSITIVE, units="1", indexed_by=("site",))
Q10_BY_PFT = Parameter(name="q10", support=Interval(1.0, 5.0), units="1", indexed_by=("pft",))
OFFSET_BY_SITE = Parameter(name="offset", support=REAL, units=None, indexed_by=("site",))
CENTERS = xr.DataArray([[0.3, 0.3, 0.2, 0.2], [0.2, 0.4, 0.1, 0.3]], dims=("pft", "part"),
                       coords={"pft": list(PFT), "part": list(NAMES)})

DECLARING = [
    (RATE, log_normal(median=2.0, geometric_sd=1.7), None),
    (SHARE, logit_normal(median=0.3, logit_sd=0.8), None),
    (Q10, logit_normal_from_interval(lower=1.5, upper=3.0, support=Interval(1.0, 5.0)), None),
    (OFFSET, tfd.Normal(jnp.float64(1.0), jnp.float64(2.0)), None),
    (simplex(), softmax_normal(center=(0.1, 0.2, 0.3, 0.4), logit_sd=(0.5, 0.7, 0.9)), None),
    (ALLOCATION_BY_PFT, iid_over_dim(softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5)), None),
    (RATE_BY_SITE, iid_over_dim(log_normal(median=2.0, geometric_sd=1.7)), None),
    (RATE_BY_SITE, independent_over_dim(log_normal, geometric_sd=1.5), {"median": by_site([1.0, 2.0, 3.0])}),
    (Q10_BY_PFT, iid_over_dim(logit_normal(median=2.0, logit_sd=0.5, support=Interval(1.0, 5.0))), None),
    (OFFSET_BY_SITE, iid_over_dim(tfd.Normal(jnp.float64(0.0), jnp.float64(1.0))), None),
    (ALLOCATION_BY_PFT, independent_over_dim(softmax_normal, logit_sd=0.4), {"center": CENTERS}),
]


@pytest.mark.parametrize("parameter, distribution, constants", DECLARING)
def test_each_declaration_agrees_with_log_prob_on_draws(parameter, distribution, constants):
    prior = prior_of(parameter, distribution, constants)
    assert prior.describe().iloc[0]["declared_gaussian"]
    gaussian = prior.gaussian()
    theta = prior.sample(jax.random.key(3), 200)
    log_prob = prior.log_prob(theta)
    tolerance = 1e-10 * jnp.abs(log_prob) + 1e-12 * theta.shape[-1]
    assert bool(jnp.all(jnp.abs(log_density(gaussian, theta) - log_prob) <= tolerance))


@pytest.mark.parametrize("parameter, distribution, constants", DECLARING)
def test_each_declaration_matches_the_moments_of_draws(parameter, distribution, constants):
    prior = prior_of(parameter, distribution, constants)
    gaussian = prior.gaussian()
    theta = np.asarray(prior.sample(jax.random.key(4), 20_000))
    covariance = np.asarray(gaussian.covariance)
    standard_error = np.sqrt(np.diag(covariance) / len(theta))
    np.testing.assert_array_less(np.abs(theta.mean(axis=0) - gaussian.mean), 5 * standard_error)
    np.testing.assert_allclose(np.cov(theta.T).reshape(covariance.shape), covariance, atol=0.05 * covariance.max())


def test_a_logit_normal_on_an_open_interval_is_declared_exactly():
    gaussian = prior_of(Q10, logit_normal(median=2.0, logit_sd=0.5, support=Interval(1.0, 5.0))).gaussian()
    assert float(gaussian.mean[0]) == pytest.approx(np.log(0.25 / 0.75))
    assert float(gaussian.covariance[0, 0]) == pytest.approx(0.25)


# ── select, and keys ──────────────────────────────────────────────────────────


def test_select_gives_the_marginal():
    prior = Prior(
        vector_of(RATE_BY_SITE, ALLOCATION_BY_PFT),
        {
            "rate": PriorTerm(independent_over_dim(log_normal, geometric_sd=1.5),
                              constants={"median": by_site([1.0, 2.0, 3.0])}, provenance="t"),
            "allocation": PriorTerm(
                iid_over_dim(softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5)), provenance="t"
            ),
        },
    )
    selectors = {"site": [1, 4711], "pft": ["deciduous"]}
    smaller = prior.select(**selectors)
    kept = prior.parameter_vector.unconstrained.positions(**selectors)
    full, marginal = prior.gaussian(), smaller.gaussian()
    np.testing.assert_allclose(marginal.mean, full.mean[kept])
    np.testing.assert_allclose(marginal.covariance, full.covariance[np.ix_(kept, kept)])


def test_a_terms_draws_depend_on_its_name_alone():
    rate = PriorTerm(log_normal(median=2.0, geometric_sd=1.7), provenance="t")
    share = PriorTerm(logit_normal(median=0.3, logit_sd=0.8), provenance="t")
    offset = PriorTerm(tfd.Normal(jnp.float64(0.0), jnp.float64(1.0)), provenance="t")
    key = jax.random.key(11)
    forward = Prior(vector_of(RATE, SHARE), {"rate": rate, "share": share}).sample(key, 5)
    reversed_ = Prior(vector_of(SHARE, RATE), {"share": share, "rate": rate}).sample(key, 5)
    appended = Prior(vector_of(RATE, SHARE, OFFSET), {"rate": rate, "share": share, "offset": offset}).sample(key, 5)
    np.testing.assert_array_equal(forward[:, 0], reversed_[:, 1])
    np.testing.assert_array_equal(forward[:, 1], reversed_[:, 0])
    np.testing.assert_array_equal(forward, appended[:, :2])


def test_independent_over_dim_is_aligned_by_label_whatever_the_order():
    shuffled = by_site([3.0, 1.0, 2.0, 0.5], sites=[4711, 1, 27, 9])
    ordered = by_site([1.0, 2.0, 3.0])
    family = independent_over_dim(log_normal, geometric_sd=1.5)
    first = prior_of(RATE_BY_SITE, family, {"median": shuffled})
    second = prior_of(RATE_BY_SITE, family, {"median": ordered})
    np.testing.assert_allclose(first.gaussian().mean, second.gaussian().mean)
    np.testing.assert_allclose(first.gaussian().mean, np.log([1.0, 2.0, 3.0]))


# ── support checks ────────────────────────────────────────────────────────────


def mixture() -> tfd.Distribution:
    return tfd.MixtureSameFamily(
        tfd.Categorical(probs=jnp.asarray([0.3, 0.7])),
        tfd.LogNormal(jnp.asarray([-1.0, 1.0]), jnp.asarray([0.5, 0.5])),
    )


def test_a_mixture_without_a_default_bijector_is_checked_by_draws():
    assert mixture().experimental_default_event_space_bijector() is None
    assert prior_of(RATE, mixture()).describe().iloc[0]["evaluated_by"] == "change of variables"
    with pytest.raises(ValueError, match="mass outside its declared support"):
        prior_of(SHARE, mixture())


def test_a_log_normal_on_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="mass outside its declared support"):
        prior_of(SHARE, log_normal(median=0.3, geometric_sd=1.5))


def test_a_truncated_normal_on_the_positive_line_is_refused():
    truncated = tfd.TruncatedNormal(jnp.float64(1.0), jnp.float64(2.0), low=jnp.float64(0.02), high=jnp.float64(50.0))
    with pytest.raises(ValueError, match="no density at some values"):
        prior_of(RATE, truncated)


def test_a_non_dirichlet_density_on_the_simplex_is_refused():
    with pytest.raises(ValueError, match="other than a Dirichlet"):
        prior_of(simplex(tfb.IteratedSigmoidCentered()), softmax_normal(center=(0.1, 0.2, 0.3, 0.4), logit_sd=1.0))


def test_a_gamma_whose_draws_reach_the_boundary_is_refused():
    with pytest.raises(ValueError, match="or on its boundary"):
        prior_of(RATE, tfd.Gamma(jnp.float64(0.02), jnp.float64(1.0)))


def test_an_oversized_monte_carlo_block_is_refused():
    prior = prior_of(RATE_BY_SITE, iid_over_dim(tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0))))
    with pytest.raises(ValueError, match="is singular"):
        prior.gaussian(key=jax.random.key(0), n_moment_samples=3)
    assert prior.gaussian(key=jax.random.key(0), n_moment_samples=4).mean.shape == (3,)


# ── joint terms and terms given others ────────────────────────────────────────

MEAN_BY_PFT = Parameter(name="mean", support=REAL, units=None, indexed_by=("pft",))
SPREAD = Parameter(name="spread", support=POSITIVE, units=None)
CARBON_BY_SITE = Parameter(name="carbon", support=POSITIVE, units="1", indexed_by=("site",))
PFT_OF_SITE_MEMBERSHIP = xr.DataArray(list(PFT_OF_SITE), dims="site", coords={"site": SITES}, name="pft")


def carbon_given_pft(index_shape, mean, spread, pft_of_site):
    loc = mean[pft_of_site]
    return tfd.TransformedDistribution(tfd.Independent(tfd.Normal(loc, spread), 1), tfb.Exp())


def centered_terms() -> dict:
    return {
        "mean": PriorTerm(iid_over_dim(tfd.Normal(jnp.float64(0.0), jnp.float64(1.0))), provenance="t"),
        "spread": PriorTerm(log_normal(median=1.0, geometric_sd=1.5), provenance="t"),
        "carbon": PriorTerm(carbon_given_pft, given=("mean", "spread"),
                            memberships={"pft_of_site": PFT_OF_SITE_MEMBERSHIP}, provenance="t"),
    }


def copula_term(order=("rate", "share")) -> PriorTerm:
    marginals = {"rate": log_normal(median=2.0, geometric_sd=1.7), "share": logit_normal(median=0.3, logit_sd=0.8)}
    return PriorTerm(
        gaussian_copula({n: marginals[n] for n in order}, correlation=[[1.0, -0.5], [-0.5, 1.0]]),
        provenance="t",
    )


def test_a_copula_integrates_to_one_in_theta():
    prior = Prior(vector_of(RATE, SHARE), {("rate", "share"): copula_term()})
    assert integral_by_importance_sampling(prior) == pytest.approx(1.0, abs=0.02)


def test_a_joint_term_by_change_of_variables_integrates_to_one_in_theta():
    joint = tfd.JointDistributionNamed({
        "rate": tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)),
        "share": tfd.Beta(jnp.float64(2.0), jnp.float64(5.0)),
    })
    prior = Prior(vector_of(RATE, SHARE), {("rate", "share"): PriorTerm(joint, provenance="t")})
    assert prior.describe().iloc[0]["evaluated_by"] == "change of variables"
    assert integral_by_importance_sampling(prior) == pytest.approx(1.0, abs=0.02)


def test_a_centered_hierarchy_integrates_to_one_in_theta():
    prior = Prior(vector_of(MEAN_BY_PFT, SPREAD, CARBON_BY_SITE), centered_terms())
    assert prior.parameter_vector.unconstrained.size == 6
    assert integral_by_importance_sampling(prior, n=1_000_000, scale=2.0) == pytest.approx(1.0, abs=0.03)


def test_a_given_terms_draws_depend_on_its_name_and_parents_alone():
    key = jax.random.key(12)
    first = Prior(vector_of(MEAN_BY_PFT, SPREAD, CARBON_BY_SITE), centered_terms())
    reordered = Prior(vector_of(CARBON_BY_SITE, MEAN_BY_PFT, SPREAD), centered_terms())
    appended = Prior(
        vector_of(MEAN_BY_PFT, SPREAD, CARBON_BY_SITE, OFFSET),
        {**centered_terms(), "offset": PriorTerm(tfd.Normal(jnp.float64(0.0), jnp.float64(1.0)), provenance="t")},
    )
    draws = [prior.sample(key, 6) for prior in (first, reordered, appended)]
    for name in ("mean", "spread", "carbon"):
        wanted = [
            prior.parameter_vector.unconstrained.flat_to_values(theta)[name]
            for theta, prior in zip(draws, (first, reordered, appended))
        ]
        np.testing.assert_array_equal(wanted[0], wanted[1])
        np.testing.assert_array_equal(wanted[0], wanted[2])


def test_a_given_term_draws_from_its_conditional():
    prior = Prior(vector_of(MEAN_BY_PFT, SPREAD, CARBON_BY_SITE), centered_terms())
    vector = prior.parameter_vector
    theta = prior.sample(jax.random.key(13), 40_000)
    natural_values = vector.flat_to_values(vector.to_natural(theta))
    pft = [PFT.index(p) for p in PFT_OF_SITE]
    standardized = (
        np.asarray(vector.unconstrained.flat_to_values(theta)["carbon"]) - np.asarray(natural_values["mean"])[:, pft]
    ) / np.asarray(natural_values["spread"])[:, None]
    np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=0.02)
    np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=0.02)


@pytest.mark.parametrize("order", [("rate", "share"), ("share", "rate")], ids=["key order", "other order"])
def test_the_copulas_declaration_agrees_with_log_prob_and_the_moments_of_draws(order):
    prior = Prior(vector_of(RATE, SHARE), {("rate", "share"): copula_term(order)})
    assert prior.describe().iloc[0]["declared_gaussian"]
    gaussian = prior.gaussian()
    theta = prior.sample(jax.random.key(14), 20_000)
    log_prob = prior.log_prob(theta[:200])
    tolerance = 1e-10 * jnp.abs(log_prob) + 1e-12 * theta.shape[-1]
    assert bool(jnp.all(jnp.abs(log_density(gaussian, theta[:200]) - log_prob) <= tolerance))
    theta = np.asarray(theta)
    covariance = np.asarray(gaussian.covariance)
    standard_error = np.sqrt(np.diag(covariance) / len(theta))
    np.testing.assert_array_less(np.abs(theta.mean(axis=0) - gaussian.mean), 5 * standard_error)
    np.testing.assert_allclose(np.cov(theta.T), covariance, atol=0.05 * covariance.max())


def test_the_change_of_variables_sums_the_jacobian_over_labels():
    gamma = tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0))
    prior = prior_of(RATE_BY_SITE, iid_over_dim(gamma))
    assert prior.describe().iloc[0]["evaluated_by"] == "change of variables"
    theta = jax.random.normal(jax.random.key(15), (4, 3))
    expected = (gamma.log_prob(jnp.exp(theta)) + theta).sum(axis=-1)
    np.testing.assert_allclose(prior.log_prob(theta), expected, rtol=1e-12)


@pytest.mark.parametrize("concentration", [0.15, 0.5])
def test_the_simplex_jacobian_holds_where_coordinates_are_tiny(concentration):
    """Under SoftmaxCentered the Jacobian is the closed form; autodiff lost
    whole nats, or gave -inf, at a sparse Dirichlet's own draws."""
    dirichlet = tfd.Dirichlet(jnp.full(4, concentration))
    prior = prior_of(simplex(), dirichlet)
    theta = prior.sample(jax.random.key(16), 20_000)
    x = prior.parameter_vector.flat_to_values(prior.parameter_vector.to_natural(theta))["shares"]
    padded = jnp.concatenate([theta, jnp.zeros((len(theta), 1))], axis=-1)
    exact = dirichlet.log_prob(x) + jax.nn.log_softmax(padded, axis=-1).sum(axis=-1)
    assert bool(jnp.isfinite(prior.log_prob(theta)).all())
    np.testing.assert_allclose(prior.log_prob(theta), exact, rtol=1e-12, atol=1e-10)
