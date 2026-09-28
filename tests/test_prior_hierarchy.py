"""Tests for hierarchy and dependence in the prior: terms given others,
joint terms and the Gaussian copula, derived parameters in the prior, the
``given`` graph, dependent sets and their Gaussian blocks.

The worked examples are partial pooling of a site-level quantity within PFTs
in both forms, non-centered through a derived parameter and centered through
``given``; a covariate regression, and a centered term given its derived
mean; and a correlated pair. The mathematics common to every term (densities
integrating to one, declarations against draws, key stability) is in
``test_prior_conformance.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyeki.linalg import DensePSD, PSDDiagonal
from tensorflow_probability.substrates import jax as tfp

from conftest import site_table_of
from sipnet_calibration.parameter_vector import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    DerivedParameter,
    Parameter,
    ParameterVector,
    site_positions,
)
from sipnet_calibration.prior import (
    Prior,
    PriorTerm,
    gaussian_copula,
    iid_over_dim,
    log_normal,
    logit_normal,
    softmax_normal,
)

tfd, tfb = tfp.distributions, tfp.bijectors

SITES = (1, 27, 4711)
PFT = ("deciduous", "conifer", "deciduous")
LOG_MEDIAN = float(np.log(1e4))


def term(distribution, *, given=()) -> PriorTerm:
    return PriorTerm(distribution, given=given, provenance="test")


def normal(loc: float, scale: float) -> tfd.Normal:
    return tfd.Normal(jnp.float64(loc), jnp.float64(scale))


MEAN = Parameter(name="mean", support=REAL, units=None, dim="pft")
SPREAD = Parameter(name="spread", support=POSITIVE, units=None)
STANDARDIZED = Parameter(name="standardized", support=REAL, units=None, dim="site")
SOIL_CARBON = Parameter(name="soil_carbon", support=POSITIVE, units="g m-2", dim="site")


def soil_carbon_non_centered(dim_index, site_table, mean, spread, standardized):
    return jnp.exp(mean[site_positions(site_table, "pft")] + spread * standardized)


def soil_carbon_given_pft(dim_index, site_table, mean, spread):
    loc = mean[site_positions(site_table, "pft")]
    return tfd.TransformedDistribution(tfd.Independent(tfd.Normal(loc, spread), 1), tfb.Exp())


HYPERPRIORS = {
    "mean": term(iid_over_dim(normal(LOG_MEDIAN, 1.0))),
    "spread": term(log_normal(median=0.5, geometric_sd=2.0)),
}


def vector_of(*parameters: Parameter, derived_parameters=(), **arguments) -> ParameterVector:
    return ParameterVector(
        parameters=parameters,
        derived_parameters=derived_parameters,
        site_table=site_table_of(*SITES, **arguments),
        site_labels={"pft": PFT},
    )


@pytest.fixture(scope="module")
def non_centered() -> Prior:
    derived = DerivedParameter(
        name="soil_carbon", units="g m-2", dim="site", support=POSITIVE,
        derived_from=("mean", "spread", "standardized"), compute=soil_carbon_non_centered,
    )
    vector = vector_of(MEAN, SPREAD, STANDARDIZED, derived_parameters=[derived])
    return Prior(vector, {**HYPERPRIORS, "standardized": term(iid_over_dim(normal(0.0, 1.0)))})


@pytest.fixture(scope="module")
def centered() -> Prior:
    return Prior(
        vector_of(MEAN, SPREAD, SOIL_CARBON),
        {**HYPERPRIORS, "soil_carbon": term(soil_carbon_given_pft, given=("mean", "spread"))},
    )


def log_soil_carbon_draws(prior: Prior, n: int) -> np.ndarray:
    theta = prior.sample(jax.random.key(5), n)
    return np.log(np.asarray(prior.parameter_vector.to_natural(theta)["soil_carbon"]))


# ── partial pooling ───────────────────────────────────────────────────────────


def test_the_non_centered_form_is_declared_exactly(non_centered):
    assert non_centered.describe()["declared_gaussian"].all()
    gaussian = non_centered.gaussian()
    assert [type(b) for b in gaussian.cov.blocks] == [PSDDiagonal] * 3
    np.testing.assert_allclose(gaussian.mean[:2], LOG_MEDIAN)
    np.testing.assert_allclose(gaussian.cov.to_dense().diagonal()[3:], 1.0)


def test_the_centered_form_is_evaluated_by_its_base_density(centered):
    row = centered.describe().loc["soil_carbon"]
    assert row["given"] == "mean, spread"
    assert row["evaluated_by"] == "base density"
    assert not row["declared_gaussian"]


def test_the_centered_density_is_the_hierarchy(centered):
    vector = centered.parameter_vector
    theta = centered.sample(jax.random.key(0), 4)
    natural_values = vector.to_natural(theta)
    mean, spread = np.asarray(natural_values["mean"]), np.asarray(natural_values["spread"])
    log_carbon = np.asarray(theta[:, vector.positions(parameter_name="soil_carbon")])
    pft = site_positions(vector.site_table, "pft")
    spread_prior = log_normal(median=0.5, geometric_sd=2.0).distribution
    expected = (
        normal(LOG_MEDIAN, 1.0).log_prob(mean).sum(axis=-1)
        + spread_prior.log_prob(np.log(spread))
        + normal(0.0, 1.0).log_prob((log_carbon - mean[:, pft]) / spread[:, None]).sum(axis=-1)
        - 3 * np.log(spread)
    )
    np.testing.assert_allclose(centered.log_prob(theta), expected, rtol=1e-12)
    assert centered.log_prob(theta.reshape((2, 2, -1))).shape == (2, 2)
    gradient = jax.jit(jax.grad(centered.log_prob))(theta[0])
    assert bool(jnp.isfinite(gradient).all())


def test_both_forms_are_one_model(non_centered, centered):
    non_centered_draws = log_soil_carbon_draws(non_centered, 40_000)
    centered_draws = log_soil_carbon_draws(centered, 40_000)
    np.testing.assert_allclose(non_centered_draws.mean(axis=0), centered_draws.mean(axis=0), atol=0.03)
    np.testing.assert_allclose(non_centered_draws.std(axis=0), centered_draws.std(axis=0), rtol=0.03)
    # Sites 1 and 4711 share a PFT, so their values are correlated through its mean.
    shared = np.corrcoef(centered_draws[:, 0], centered_draws[:, 2])[0, 1]
    assert shared == pytest.approx(np.corrcoef(non_centered_draws[:, 0], non_centered_draws[:, 2])[0, 1], abs=0.02)
    assert shared > 0.5 and abs(np.corrcoef(centered_draws[:, 0], centered_draws[:, 1])[0, 1]) < 0.02


def test_the_centered_form_gets_one_monte_carlo_block(centered):
    with pytest.raises(NotImplementedError, match="pass key="):
        centered.gaussian()
    gaussian = centered.gaussian(key=jax.random.key(1), n_moment_samples=20_000)
    (block,) = gaussian.cov.blocks
    assert isinstance(block, DensePSD) and block.to_dense().shape == (6, 6)
    np.testing.assert_allclose(gaussian.mean[:2], LOG_MEDIAN, atol=0.03)
    with pytest.raises(ValueError, match=r"the dependent set \['mean', 'spread', 'soil_carbon'\] has 6"):
        centered.gaussian(key=jax.random.key(1), n_moment_samples=6)


def test_a_given_term_draws_do_not_depend_on_where_it_is_declared(centered):
    reordered = Prior(vector_of(SOIL_CARBON, SPREAD, MEAN), dict(centered.terms))
    key = jax.random.key(9)
    first, second = centered.sample(key, 5), reordered.sample(key, 5)
    for name in ("mean", "spread", "soil_carbon"):
        np.testing.assert_array_equal(
            first[:, centered.parameter_vector.positions(parameter_name=name)],
            second[:, reordered.parameter_vector.positions(parameter_name=name)],
        )


def test_select_rebuilds_a_given_term_on_the_kept_sites(centered):
    smaller = centered.select(sites=[27])
    theta = smaller.sample(jax.random.key(2), 3)
    assert theta.shape == (3, smaller.parameter_vector.dimension)
    assert bool(jnp.isfinite(smaller.log_prob(theta)).all())


# ── covariates and derived means ──────────────────────────────────────────────

ANOMALY = [-1.0, 0.0, 2.0]
INTERCEPT = Parameter(name="intercept", support=REAL, units=None)
SLOPE = Parameter(name="slope", support=REAL, units="K-1")


def covariate_vector(*parameters: Parameter, derived: DerivedParameter) -> ParameterVector:
    return ParameterVector(
        parameters=parameters,
        derived_parameters=[derived],
        site_table=site_table_of(*SITES).assign(temperature_anomaly=ANOMALY),
        site_covariate_names=["temperature_anomaly"],
    )


def regression(dim_index, site_table, intercept, slope):
    return intercept + slope * site_table["temperature_anomaly"].to_numpy()


def test_a_covariate_regression_is_a_prior_on_its_coefficients():
    rate = DerivedParameter(
        name="respiration", units="yr-1", dim="site", support=POSITIVE, derived_from=("intercept", "slope"),
        compute=lambda dim_index, site_table, intercept, slope: jnp.exp(
            regression(dim_index, site_table, intercept, slope)
        ),
    )
    prior = Prior(
        covariate_vector(INTERCEPT, SLOPE, derived=rate),
        {
            ("intercept", "slope"): term(gaussian_copula(
                {"intercept": normal(np.log(0.01), 0.5), "slope": normal(0.0, 0.1)},
                correlation=[[1.0, -0.4], [-0.4, 1.0]],
            )),
        },
    )
    assert prior.parameter_vector.dimension == 2
    draws = np.log(np.asarray(prior.parameter_vector.to_natural(prior.sample(jax.random.key(3), 50_000))["respiration"]))
    # log rate at a site is intercept + slope * w: mean log(0.01), variance 0.25 + 0.01 w^2 - 0.04 w.
    w = np.asarray(ANOMALY)
    np.testing.assert_allclose(draws.mean(axis=0), np.log(0.01), atol=0.01)
    np.testing.assert_allclose(draws.var(axis=0), 0.25 + 0.01 * w**2 - 2 * 0.4 * 0.05 * w, rtol=0.03)


def test_a_centered_term_may_be_given_a_derived_mean():
    log_mean = DerivedParameter(
        name="log_mean", units=None, dim="site", derived_from=("intercept", "slope"), compute=regression
    )
    respiration = Parameter(name="respiration", support=POSITIVE, units="yr-1", dim="site")

    def respiration_given_mean(dim_index, site_table, log_mean, spread):
        return tfd.TransformedDistribution(tfd.Independent(tfd.Normal(log_mean, spread), 1), tfb.Exp())

    prior = Prior(
        covariate_vector(INTERCEPT, SLOPE, SPREAD, respiration, derived=log_mean),
        {
            "intercept": term(normal(np.log(0.01), 0.5)),
            "slope": term(normal(0.0, 0.1)),
            "spread": term(log_normal(median=0.2, geometric_sd=1.5)),
            "respiration": term(respiration_given_mean, given=("log_mean", "spread")),
        },
    )
    vector = prior.parameter_vector
    theta = prior.sample(jax.random.key(4), 3)
    natural_values = vector.to_natural(theta)
    log_rate = np.asarray(theta[:, vector.positions(parameter_name="respiration")])
    spread = np.asarray(natural_values["spread"])[:, None]
    conditional = normal(0.0, 1.0).log_prob((log_rate - np.asarray(natural_values["log_mean"])) / spread) - np.log(spread)
    marginals = sum(prior._built[n].log_prob(theta[:, vector.positions(parameter_name=n)], None)
                    for n in ("intercept", "slope", "spread"))
    np.testing.assert_allclose(prior.log_prob(theta), marginals + conditional.sum(axis=-1), rtol=1e-12)
    # The derived mean ties the respiration to the coefficients it is computed from.
    (block,) = prior.gaussian(key=jax.random.key(5), n_moment_samples=5_000).cov.blocks
    assert block.to_dense().shape == (6, 6)


# ── a correlated pair ─────────────────────────────────────────────────────────

RATE = Parameter(name="rate", support=POSITIVE, units="yr-1")
SHARE = Parameter(name="share", support=OPEN_UNIT_INTERVAL, units="1")
CORRELATION = [[1.0, 0.6], [0.6, 1.0]]


def copula(**marginals) -> PriorTerm:
    return term(gaussian_copula(marginals, correlation=CORRELATION))


RATE_MARGINAL = log_normal(median=2.0, geometric_sd=1.5)
SHARE_MARGINAL = logit_normal(median=0.3, logit_sd=0.8)


def test_a_correlated_pair_is_a_multivariate_normal_in_theta():
    prior = Prior(vector_of(RATE, SHARE), {("rate", "share"): copula(rate=RATE_MARGINAL, share=SHARE_MARGINAL)})
    row = prior.describe().loc["rate+share"]
    assert row["prior"] == "gaussian copula" and row["evaluated_by"] == "base density"
    assert row["declared_gaussian"]
    sigma = np.asarray([np.log(1.5), 0.8])
    mu = np.asarray([np.log(2.0), np.log(0.3 / 0.7)])
    covariance = sigma[:, None] * np.asarray(CORRELATION) * sigma[None, :]
    gaussian = prior.gaussian()
    np.testing.assert_allclose(gaussian.mean, mu, rtol=1e-12)
    np.testing.assert_allclose(gaussian.cov.to_dense(), covariance, rtol=1e-12)
    theta = prior.sample(jax.random.key(6), 50_000)
    reference = tfd.MultivariateNormalFullCovariance(mu, covariance)
    np.testing.assert_allclose(prior.log_prob(theta[:5]), reference.log_prob(theta[:5]), rtol=1e-12)
    assert float(np.corrcoef(np.asarray(theta).T)[0, 1]) == pytest.approx(0.6, abs=0.01)
    rates = np.asarray(prior.parameter_vector.to_natural(theta)["rate"])
    assert float(np.median(rates)) == pytest.approx(2.0, rel=0.02)


def test_a_copula_listed_in_another_order_than_its_key_is_still_declared():
    prior = Prior(vector_of(RATE, SHARE), {("share", "rate"): copula(rate=RATE_MARGINAL, share=SHARE_MARGINAL)})
    row = prior.describe().loc["share+rate"]
    assert row["evaluated_by"] == "change of variables" and row["declared_gaussian"]
    np.testing.assert_allclose(prior.gaussian().mean, [np.log(2.0), np.log(0.3 / 0.7)], rtol=1e-12)


def test_a_joint_term_may_be_any_dict_valued_distribution():
    joint = tfd.JointDistributionNamed({
        "rate": tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)),
        "share": tfd.Beta(jnp.float64(2.0), jnp.float64(5.0)),
    })
    prior = Prior(vector_of(RATE, SHARE), {("rate", "share"): term(joint)})
    assert prior.describe().loc["rate+share", "evaluated_by"] == "change of variables"
    theta = prior.sample(jax.random.key(7), 4)
    natural_values = prior.parameter_vector.to_natural(theta)
    expected = (
        tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)).log_prob(natural_values["rate"]) + theta[:, 0]
        + tfd.Beta(jnp.float64(2.0), jnp.float64(5.0)).log_prob(natural_values["share"])
        + np.log(natural_values["share"] * (1 - natural_values["share"]))
    )
    np.testing.assert_allclose(prior.log_prob(theta), expected, rtol=1e-10)


# ── construction checks ───────────────────────────────────────────────────────


def test_a_joint_term_covers_parameters_on_one_dim():
    with pytest.raises(ValueError, match="share one dim or none"):
        Prior(vector_of(RATE, SOIL_CARBON), {("rate", "soil_carbon"): term(RATE_MARGINAL)})


def test_a_joint_term_draws_a_dict_of_its_names():
    with pytest.raises(ValueError, match="a joint term's draws are a dict"):
        Prior(vector_of(RATE, SHARE), {("rate", "share"): term(RATE_MARGINAL)})


def test_a_term_given_others_is_a_function():
    with pytest.raises(TypeError, match="is given \\['rate'\\] but is a distribution"):
        Prior(vector_of(RATE, SHARE), {"rate": term(RATE_MARGINAL), "share": term(SHARE_MARGINAL, given=("rate",))})


def test_given_names_something_of_the_vector():
    with pytest.raises(KeyError, match="is given 'nothing'"):
        Prior(vector_of(RATE, SHARE), {"rate": term(RATE_MARGINAL),
                                       "share": term(lambda d, t, nothing: SHARE_MARGINAL, given=("nothing",))})
    with pytest.raises(TypeError, match="sequence"):
        PriorTerm(RATE_MARGINAL, given="rate", provenance="test")


def share_given(dim_index, site_table, rate):
    return logit_normal(median=0.3, logit_sd=0.5 + 0.0 * rate)


def rate_given(dim_index, site_table, share):
    return log_normal(median=2.0, geometric_sd=1.5 + 0.0 * share)


def test_a_cycle_of_given_links_is_refused_by_name():
    with pytest.raises(ValueError, match="cycle, rate -> share -> rate"):
        Prior(vector_of(RATE, SHARE), {"rate": term(rate_given, given=("share",)),
                                       "share": term(share_given, given=("rate",))})


def test_a_cycle_through_a_derived_parameter_is_refused():
    doubled = DerivedParameter(name="doubled", units="yr-1", derived_from=("rate",),
                               compute=lambda dim_index, site_table, rate: 2.0 * rate)
    vector = vector_of(RATE, derived_parameters=[doubled])
    with pytest.raises(ValueError, match="cycle, rate -> doubled -> rate"):
        Prior(vector, {"rate": term(lambda d, t, doubled: RATE_MARGINAL, given=("doubled",))})


def test_a_term_that_changes_structure_with_its_given_values_is_refused():
    calls = []

    def switching(dim_index, site_table, rate):
        # A different class at each ancestral draw, as a branch on the given
        # value would give.
        calls.append(rate)
        return logit_normal(median=0.3, logit_sd=0.5) if len(calls) == 1 else tfd.Beta(
            jnp.float64(2.0), jnp.float64(2.0))

    with pytest.raises(ValueError, match="changes its structure"):
        Prior(vector_of(RATE, SHARE), {"rate": term(log_normal(median=2.0, geometric_sd=3.0)),
                                       "share": term(switching, given=("rate",))})


def test_select_refuses_to_drop_what_a_kept_term_needs(centered, non_centered):
    with pytest.raises(ValueError, match="drops \\['spread'\\], which the prior term 'soil_carbon'"):
        centered.select(parameter_names=["mean", "soil_carbon"])
    pair = Prior(vector_of(RATE, SHARE), {("rate", "share"): copula(rate=RATE_MARGINAL, share=SHARE_MARGINAL)})
    with pytest.raises(ValueError, match="which the prior term 'rate\\+share' covers"):
        pair.select(parameter_names=["rate"])
    # A derived parameter is dropped with its inputs; the prior never sees it.
    assert non_centered.select(parameter_names=["mean", "spread"]).parameter_vector.derived_parameter_names == ()


def test_select_refuses_to_drop_what_a_derived_parameter_given_needs():
    log_mean = DerivedParameter(name="log_mean", units=None, dim="site", derived_from=("intercept", "slope"),
                                compute=regression)
    respiration = Parameter(name="respiration", support=POSITIVE, units="yr-1", dim="site")
    prior = Prior(
        covariate_vector(INTERCEPT, SLOPE, respiration, derived=log_mean),
        {
            "intercept": term(normal(0.0, 1.0)),
            "slope": term(normal(0.0, 0.1)),
            "respiration": term(
                lambda d, t, log_mean: tfd.TransformedDistribution(
                    tfd.Independent(tfd.Normal(log_mean, jnp.float64(0.3)), 1), tfb.Exp()),
                given=("log_mean",),
            ),
        },
    )
    with pytest.raises(ValueError, match="drops \\['log_mean'\\]"):
        prior.select(parameter_names=["intercept", "respiration"])


def test_a_dependent_set_must_be_contiguous_for_the_gaussian():
    other = Parameter(name="other", support=POSITIVE, units=None)
    prior = Prior(
        vector_of(MEAN, other, SPREAD, SOIL_CARBON),
        {**HYPERPRIORS, "other": term(RATE_MARGINAL),
         "soil_carbon": term(soil_carbon_given_pft, given=("mean", "spread"))},
    )
    with pytest.raises(ValueError, match="is not contiguous"):
        prior.gaussian(key=jax.random.key(0), n_moment_samples=100)
    assert bool(jnp.isfinite(prior.log_prob(prior.sample(jax.random.key(0), 2))).all())


def test_a_joint_density_on_the_simplex_is_refused():
    shares = Parameter(name="shares", support=SIMPLEX, units="1", natural_names=("a", "b", "c"))
    joint = tfd.JointDistributionNamed({
        "shares": tfd.Dirichlet(jnp.full(3, 2.0)), "rate": tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)),
    })
    with pytest.raises(ValueError, match="other than a Dirichlet"):
        Prior(vector_of(shares, RATE), {("shares", "rate"): term(joint)})


@pytest.mark.parametrize(
    "marginals, correlation, message",
    [
        ({"rate": tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)), "share": SHARE_MARGINAL}, CORRELATION,
         "not a scalar pushforward of a Gaussian"),
        ({"rate": softmax_normal(center=(0.5, 0.5), logit_sd=1.0), "share": SHARE_MARGINAL}, CORRELATION,
         "not a scalar pushforward"),
        ({"rate": RATE_MARGINAL, "share": SHARE_MARGINAL}, [[1.0, 0.6], [0.5, 1.0]], "symmetric"),
        ({"rate": RATE_MARGINAL, "share": SHARE_MARGINAL}, [[2.0, 0.6], [0.6, 1.0]], "unit diagonal"),
        ({"rate": RATE_MARGINAL, "share": SHARE_MARGINAL}, [[1.0, 1.5], [1.5, 1.0]], "positive definite"),
        ({"rate": RATE_MARGINAL, "share": SHARE_MARGINAL}, [[1.0]], "shape \\(1, 1\\)"),
    ],
)
def test_the_copula_refuses_bad_arguments(marginals, correlation, message):
    with pytest.raises(ValueError, match=message):
        gaussian_copula(marginals, correlation=correlation)


def test_a_copula_is_for_parameters_without_a_dim():
    by_site = Parameter(name="rate", support=POSITIVE, units="yr-1", dim="site")
    with pytest.raises(TypeError, match="prior of parameters without a dim"):
        Prior(vector_of(by_site, SOIL_CARBON), {("rate", "soil_carbon"): copula(rate=RATE_MARGINAL,
                                                                                soil_carbon=RATE_MARGINAL)})


def test_zero_draws_of_a_prior_evaluated_by_change_of_variables(non_centered, centered):
    assert non_centered.sample(jax.random.key(0), 0).shape == (0, non_centered.parameter_vector.dimension)
    assert centered.sample(jax.random.key(0), 0).shape == (0, centered.parameter_vector.dimension)


def test_a_joint_term_given_others_may_be_a_joint_distribution():
    def pair_given_spread(dim_index, site_table, spread):
        return tfd.JointDistributionNamedAutoBatched({
            "rate": tfd.Gamma(jnp.float64(3.0), spread), "share": tfd.Beta(jnp.float64(2.0), spread),
        })

    prior = Prior(vector_of(RATE, SHARE, SPREAD), {
        ("rate", "share"): term(pair_given_spread, given=("spread",)),
        "spread": HYPERPRIORS["spread"],
    })
    theta = prior.sample(jax.random.key(8), 4)
    natural_values = prior.parameter_vector.to_natural(theta)
    spread = natural_values["spread"]
    expected = (
        tfd.Gamma(jnp.float64(3.0), spread).log_prob(natural_values["rate"]) + theta[:, 0]
        + tfd.Beta(jnp.float64(2.0), spread).log_prob(natural_values["share"])
        + jnp.log(natural_values["share"] * (1 - natural_values["share"]))
        + prior._built["spread"].log_prob(theta[:, 2:], None)
    )
    np.testing.assert_allclose(jax.jit(prior.log_prob)(theta), expected, rtol=1e-10)


@pytest.mark.parametrize("key", [("rate",), ()])
def test_a_joint_term_is_keyed_by_two_or_more_names(key):
    with pytest.raises(ValueError, match="a joint term covers two or more"):
        Prior(vector_of(RATE), {key: term(RATE_MARGINAL)})


def test_a_term_is_not_given_what_it_covers_or_a_name_twice():
    with pytest.raises(ValueError, match="is given \\['rate'\\], which it covers"):
        Prior(vector_of(RATE), {"rate": term(rate_given, given=("rate",))})
    with pytest.raises(ValueError, match="given"):
        PriorTerm(share_given, given=("rate", "rate"), provenance="test")
