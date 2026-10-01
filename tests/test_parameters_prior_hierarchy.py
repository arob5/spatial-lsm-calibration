"""Tests for hierarchy and dependence in the prior: terms given others,
joint terms and the Gaussian copula, derived parameters in the prior, and
the ``given`` graph.

The worked examples are partial pooling of a site-level quantity within PFTs
in both forms, non-centered through a derived parameter and centered through
``given`` a derived location at each site; a covariate regression, and a
centered term given its derived mean; and a correlated pair. The mathematics
common to every term (densities integrating to one, each builder's Gaussian
in theta, key stability) is in ``test_parameters_prior_conformance.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from conftest import theta_gaussian

from sipnet_calibration.parameters.derived import DerivedParameter, DerivedParameters
from sipnet_calibration.parameters.parameter import Parameter
from sipnet_calibration.parameters.families import (
    log_normal,
    logit_normal,
    softmax_normal,
)
from sipnet_calibration.parameters.prior import (
    Prior,
    PriorTerm,
)
from sipnet_calibration.parameters.prior_functions import (
    gaussian_copula,
    iid_over_dim,
)
from sipnet_calibration.parameters.support import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
)
from sipnet_calibration.parameters.vector import ParameterVector

tfd, tfb = tfp.distributions, tfp.bijectors

SITES = np.array([1, 27, 4711], dtype=np.int32)
PFT_OF_SITE = ("deciduous", "conifer", "deciduous")
PFT = ("conifer", "deciduous")
PFT_POSITIONS = [PFT.index(p) for p in PFT_OF_SITE]
LOG_MEDIAN = float(np.log(1e4))


def term(parameter_names, distribution, **arguments) -> PriorTerm:
    """A test term; one parameter's name may be given alone."""
    names = (parameter_names,) if isinstance(parameter_names, str) else parameter_names
    return PriorTerm(parameter_names=names, distribution=distribution, provenance="test", **arguments)


def normal(loc: float, scale: float) -> tfd.Normal:
    return tfd.Normal(jnp.float64(loc), jnp.float64(scale))


def vector_of(*parameters: Parameter) -> ParameterVector:
    coords = {"pft": PFT, "site": SITES}
    used = {d for p in parameters for d in p.indexed_by}
    return ParameterVector(parameters=parameters, coords={d: v for d, v in coords.items() if d in used})


def pft_of_site() -> xr.DataArray:
    return xr.DataArray(list(PFT_OF_SITE), dims="site", coords={"site": SITES}, name="pft")


def natural(prior: Prior, theta):
    """Every natural value at theta, the derived parameters' included."""
    values = prior.parameter_vector.flat_to_values(prior.parameter_vector.to_natural(theta))
    if prior.derived_parameters is not None:
        values |= prior.derived_parameters.values(values)
    return values


def unconstrained(prior: Prior, theta, name: str) -> np.ndarray:
    return np.asarray(prior.parameter_vector.unconstrained.flat_to_values(theta)[name])


MEAN = Parameter(name="mean", support=REAL, units=None, indexed_by=("pft",))
SPREAD = Parameter(name="spread", support=POSITIVE, units=None)
STANDARDIZED = Parameter(name="standardized", support=REAL, units=None, indexed_by=("site",))
SOIL_CARBON = Parameter(name="soil_carbon", support=POSITIVE, units="g m-2", indexed_by=("site",))


def soil_carbon_non_centered(mean, spread, standardized, pft_of_site):
    return jnp.exp(mean[pft_of_site] + spread * standardized)


def site_log_mean(vector: ParameterVector) -> DerivedParameters:
    """Each site's location, its PFT's mean: what a centered term is given."""
    return DerivedParameters(parameter_vector=vector, derived_parameters=[DerivedParameter(
        name="site_log_mean", units=None, indexed_by=("site",), given=("mean",),
        memberships={"pft_of_site": pft_of_site()}, function=lambda mean, pft_of_site: mean[pft_of_site],
    )])


def soil_carbon_given_its_mean(site_log_mean, spread):
    return tfd.TransformedDistribution(tfd.Independent(tfd.Normal(site_log_mean, spread), 1), tfb.Exp())


def centered_prior(*parameters: Parameter, terms=()) -> Prior:
    """The centered hierarchy over *parameters*' vector, with *terms* for any
    parameters beside mean, spread and soil_carbon."""
    vector = vector_of(*parameters)
    soil_carbon = term("soil_carbon", soil_carbon_given_its_mean, given=("site_log_mean", "spread"))
    return Prior(vector, [*HYPERPRIORS, *terms, soil_carbon], derived_parameters=site_log_mean(vector))


HYPERPRIORS = [
    term("mean", iid_over_dim(normal(LOG_MEDIAN, 1.0))),
    term("spread", log_normal(median=0.5, geometric_sd=2.0)),
]


@pytest.fixture(scope="module")
def non_centered() -> Prior:
    vector = vector_of(MEAN, SPREAD, STANDARDIZED)
    derived = DerivedParameters(parameter_vector=vector, derived_parameters=[DerivedParameter(
        name="soil_carbon", units="g m-2", indexed_by=("site",), support=POSITIVE,
        given=("mean", "spread", "standardized"), memberships={"pft_of_site": pft_of_site()},
        function=soil_carbon_non_centered,
    )])
    return Prior(vector, [*HYPERPRIORS, term("standardized", iid_over_dim(normal(0.0, 1.0)))],
                 derived_parameters=derived)


@pytest.fixture(scope="module")
def centered() -> Prior:
    return centered_prior(MEAN, SPREAD, SOIL_CARBON)


def log_soil_carbon_draws(prior: Prior, n: int) -> np.ndarray:
    theta = prior.sample(jax.random.key(5), n)
    return np.log(np.asarray(natural(prior, theta)["soil_carbon"]))


# ── partial pooling ───────────────────────────────────────────────────────────


def test_the_non_centered_form_is_gaussian_in_theta(non_centered):
    np.testing.assert_allclose(theta_gaussian(non_centered, "mean")[0], LOG_MEDIAN)
    np.testing.assert_allclose(theta_gaussian(non_centered, "standardized")[1], 1.0)


def test_iid_over_dim_repeats_a_values_distribution_given_others():
    offsets = Parameter(name="offsets", support=REAL, units=None, indexed_by=("site",))
    prior = Prior(vector_of(SPREAD, offsets), [
        term("spread", log_normal(median=0.5, geometric_sd=2.0)),
        term("offsets", iid_over_dim(lambda spread: tfd.Normal(jnp.float64(0.0), spread)), given=("spread",)),
    ])
    theta = prior.sample(jax.random.key(6), 5)
    values = natural(prior, theta)
    spread = np.asarray(values["spread"])
    expected = (
        log_normal(median=0.5, geometric_sd=2.0).distribution.log_prob(np.log(spread))
        + normal(0.0, 1.0).log_prob(np.asarray(values["offsets"]) / spread[:, None]).sum(axis=-1)
        - len(SITES) * np.log(spread)
    )
    np.testing.assert_allclose(prior.log_prob(theta), expected, rtol=1e-12)


def test_a_function_whose_event_lacks_the_index_dims_points_to_iid_over_dim():
    with pytest.raises(ValueError, match="identically distributed are iid_over_dim"):
        Prior(vector_of(SPREAD, SOIL_CARBON), [
            term("spread", log_normal(median=0.5, geometric_sd=2.0)),
            term("soil_carbon", lambda spread: tfd.LogNormal(jnp.float64(0.0), spread), given=("spread",)),
        ])


def test_the_centered_form_is_evaluated_by_its_base_density(centered):
    row = centered.describe().loc["soil_carbon"]
    assert row["given"] == "site_log_mean, spread"
    assert row["evaluated_by"] == "base density"


def test_the_centered_density_is_the_hierarchy(centered):
    theta = centered.sample(jax.random.key(0), 4)
    values = natural(centered, theta)
    mean, spread = np.asarray(values["mean"]), np.asarray(values["spread"])
    log_carbon = unconstrained(centered, theta, "soil_carbon")
    spread_prior = log_normal(median=0.5, geometric_sd=2.0).distribution
    expected = (
        normal(LOG_MEDIAN, 1.0).log_prob(mean).sum(axis=-1)
        + spread_prior.log_prob(np.log(spread))
        + normal(0.0, 1.0).log_prob((log_carbon - mean[:, PFT_POSITIONS]) / spread[:, None]).sum(axis=-1)
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


def test_a_given_term_draws_do_not_depend_on_where_it_is_declared(centered):
    reordered = centered_prior(SOIL_CARBON, SPREAD, MEAN)
    key = jax.random.key(9)
    first, second = centered.sample(key, 5), reordered.sample(key, 5)
    for name in ("mean", "spread", "soil_carbon"):
        np.testing.assert_array_equal(unconstrained(centered, first, name), unconstrained(reordered, second, name))


def test_select_rebuilds_a_given_term_on_the_kept_sites(centered):
    smaller = centered.select(site=[27])
    theta = smaller.sample(jax.random.key(2), 3)
    assert theta.shape == (3, smaller.parameter_vector.unconstrained.size)
    assert bool(jnp.isfinite(smaller.log_prob(theta)).all())


# ── covariates and derived means ──────────────────────────────────────────────

ANOMALY = xr.DataArray([-1.0, 0.0, 2.0], dims="site", coords={"site": SITES})
INTERCEPT = Parameter(name="intercept", support=REAL, units=None)
SLOPE = Parameter(name="slope", support=REAL, units="K-1")


def regression(intercept, slope, anomaly):
    return intercept + slope * anomaly


def test_a_covariate_regression_is_a_prior_on_its_coefficients():
    vector = vector_of(INTERCEPT, SLOPE)
    rate = DerivedParameter(
        name="respiration", units="yr-1", indexed_by=("site",), support=POSITIVE,
        given=("intercept", "slope"), constants={"anomaly": ANOMALY},
        function=lambda intercept, slope, anomaly: jnp.exp(regression(intercept, slope, anomaly)),
    )
    derived = DerivedParameters(parameter_vector=vector, derived_parameters=[rate], coords={"site": SITES})
    prior = Prior(
        vector,
        [
            term(("intercept", "slope"), gaussian_copula(
                {"intercept": normal(np.log(0.01), 0.5), "slope": normal(0.0, 0.1)},
                correlation=[[1.0, -0.4], [-0.4, 1.0]],
            )),
        ],
        derived_parameters=derived,
    )
    assert prior.parameter_vector.unconstrained.size == 2
    draws = np.log(np.asarray(natural(prior, prior.sample(jax.random.key(3), 50_000))["respiration"]))
    # log rate at a site is intercept + slope * w: mean log(0.01), variance 0.25 + 0.01 w^2 - 0.04 w.
    w = ANOMALY.values
    np.testing.assert_allclose(draws.mean(axis=0), np.log(0.01), atol=0.01)
    np.testing.assert_allclose(draws.var(axis=0), 0.25 + 0.01 * w**2 - 2 * 0.4 * 0.05 * w, rtol=0.03)


def centered_given_a_derived_mean() -> Prior:
    respiration = Parameter(name="respiration", support=POSITIVE, units="yr-1", indexed_by=("site",))
    vector = vector_of(INTERCEPT, SLOPE, SPREAD, respiration)
    derived = DerivedParameters(parameter_vector=vector, derived_parameters=[DerivedParameter(
        name="log_mean", units=None, indexed_by=("site",), given=("intercept", "slope"),
        constants={"anomaly": ANOMALY}, function=regression,
    )])

    def respiration_given_mean(log_mean, spread):
        return tfd.TransformedDistribution(tfd.Independent(tfd.Normal(log_mean, spread), 1), tfb.Exp())

    return Prior(
        vector,
        [
            term("intercept", normal(np.log(0.01), 0.5)),
            term("slope", normal(0.0, 0.1)),
            term("spread", log_normal(median=0.2, geometric_sd=1.5)),
            term("respiration", respiration_given_mean, given=("log_mean", "spread")),
        ],
        derived_parameters=derived,
    )


def test_a_centered_term_may_be_given_a_derived_mean():
    prior = centered_given_a_derived_mean()
    theta = prior.sample(jax.random.key(4), 3)
    values = natural(prior, theta)
    log_rate = unconstrained(prior, theta, "respiration")
    spread = np.asarray(values["spread"])[:, None]
    conditional = normal(0.0, 1.0).log_prob((log_rate - np.asarray(values["log_mean"])) / spread) - np.log(spread)
    marginals = sum(prior._built[n].log_prob(theta[:, prior._built[n].positions], None)
                    for n in ("intercept", "slope", "spread"))
    np.testing.assert_allclose(prior.log_prob(theta), marginals + conditional.sum(axis=-1), rtol=1e-12)


# ── a correlated pair ─────────────────────────────────────────────────────────

RATE = Parameter(name="rate", support=POSITIVE, units="yr-1")
SHARE = Parameter(name="share", support=OPEN_UNIT_INTERVAL, units="1")
CORRELATION = [[1.0, 0.6], [0.6, 1.0]]


def copula_in_theta() -> tfd.Distribution:
    """The copula of RATE_MARGINAL and SHARE_MARGINAL in theta, (rate, share):
    a Gaussian of the marginals' means, standard deviations and CORRELATION."""
    sigma = np.asarray([np.log(1.5), 0.8])
    mu = np.asarray([np.log(2.0), np.log(0.3 / 0.7)])
    return tfd.MultivariateNormalFullCovariance(mu, sigma[:, None] * np.asarray(CORRELATION) * sigma[None, :])


def copula(parameter_names, **marginals) -> PriorTerm:
    return term(parameter_names, gaussian_copula(marginals, correlation=CORRELATION))


RATE_MARGINAL = log_normal(median=2.0, geometric_sd=1.5)
SHARE_MARGINAL = logit_normal(median=0.3, logit_sd=0.8)


def test_a_correlated_pair_is_a_multivariate_normal_in_theta():
    prior = Prior(vector_of(RATE, SHARE), [copula(("rate", "share"), rate=RATE_MARGINAL, share=SHARE_MARGINAL)])
    row = prior.describe().loc["rate+share"]
    assert row["prior"] == "gaussian copula" and row["evaluated_by"] == "base density"
    theta = prior.sample(jax.random.key(6), 50_000)
    reference = copula_in_theta()
    np.testing.assert_allclose(prior.log_prob(theta[:5]), reference.log_prob(theta[:5]), rtol=1e-12)
    assert float(np.corrcoef(np.asarray(theta).T)[0, 1]) == pytest.approx(0.6, abs=0.01)
    rates = np.asarray(natural(prior, theta)["rate"])
    assert float(np.median(rates)) == pytest.approx(2.0, rel=0.02)


def test_a_copula_listed_in_another_order_than_its_parameters_is_the_same_density():
    prior = Prior(vector_of(RATE, SHARE), [copula(("share", "rate"), rate=RATE_MARGINAL, share=SHARE_MARGINAL)])
    assert prior.describe().loc["share+rate", "evaluated_by"] == "change of variables"
    theta = prior.sample(jax.random.key(7), 5)
    np.testing.assert_allclose(prior.log_prob(theta), copula_in_theta().log_prob(theta), rtol=1e-10)


def test_a_joint_term_may_be_any_dict_valued_distribution():
    joint = tfd.JointDistributionNamed({
        "rate": tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)),
        "share": tfd.Beta(jnp.float64(2.0), jnp.float64(5.0)),
    })
    prior = Prior(vector_of(RATE, SHARE), [term(("rate", "share"), joint)])
    assert prior.describe().loc["rate+share", "evaluated_by"] == "change of variables"
    theta = prior.sample(jax.random.key(7), 4)
    values = natural(prior, theta)
    expected = (
        tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)).log_prob(values["rate"]) + theta[:, 0]
        + tfd.Beta(jnp.float64(2.0), jnp.float64(5.0)).log_prob(values["share"])
        + np.log(values["share"] * (1 - values["share"]))
    )
    np.testing.assert_allclose(prior.log_prob(theta), expected, rtol=1e-10)


# ── construction checks ───────────────────────────────────────────────────────


def test_a_joint_term_covers_parameters_indexed_alike():
    with pytest.raises(ValueError, match="a joint term's parameters are indexed alike"):
        Prior(vector_of(RATE, SOIL_CARBON), [term(("rate", "soil_carbon"), RATE_MARGINAL)])


def test_a_joint_term_draws_a_dict_of_its_names():
    with pytest.raises(ValueError, match="a joint term's draws are a dict"):
        Prior(vector_of(RATE, SHARE), [term(("rate", "share"), RATE_MARGINAL)])


def test_a_term_given_others_is_a_function():
    with pytest.raises(TypeError, match="is given \\['rate'\\] but is a distribution"):
        Prior(vector_of(RATE, SHARE), [term("rate", RATE_MARGINAL), term("share", SHARE_MARGINAL, given=("rate",))])


def test_given_names_something_of_the_vector():
    with pytest.raises(KeyError, match="is given 'nothing'"):
        Prior(vector_of(RATE, SHARE), [term("rate", RATE_MARGINAL),
                                       term("share", lambda nothing: SHARE_MARGINAL, given=("nothing",))])
    with pytest.raises(TypeError, match="sequence"):
        PriorTerm(parameter_names=("share",), distribution=RATE_MARGINAL, given="rate", provenance="test")


def share_given(rate):
    return logit_normal(median=0.3, logit_sd=0.5 + 0.0 * rate)


def rate_given(share):
    return log_normal(median=2.0, geometric_sd=1.5 + 0.0 * share)


def test_a_cycle_of_given_links_is_refused_by_name():
    with pytest.raises(ValueError, match="cycle, rate -> share -> rate"):
        Prior(vector_of(RATE, SHARE), [term("rate", rate_given, given=("share",)),
                                       term("share", share_given, given=("rate",))])


def test_a_cycle_through_a_derived_parameter_is_refused():
    vector = vector_of(RATE)
    derived = DerivedParameters(parameter_vector=vector, derived_parameters=[DerivedParameter(
        name="doubled", units="yr-1", given=("rate",), function=lambda rate: 2.0 * rate,
    )])
    with pytest.raises(ValueError, match="cycle, rate -> doubled -> rate"):
        Prior(vector, [term("rate", lambda doubled: RATE_MARGINAL, given=("doubled",))],
              derived_parameters=derived)


def test_a_term_that_changes_structure_with_its_given_values_is_refused():
    calls = []

    def switching(rate):
        # A different class at each ancestral draw, as a branch on the given
        # value would give.
        calls.append(rate)
        return logit_normal(median=0.3, logit_sd=0.5) if len(calls) == 1 else tfd.Beta(
            jnp.float64(2.0), jnp.float64(2.0))

    with pytest.raises(ValueError, match="changes its structure"):
        Prior(vector_of(RATE, SHARE), [term("rate", log_normal(median=2.0, geometric_sd=3.0)),
                                       term("share", switching, given=("rate",))])


def test_select_refuses_to_drop_what_a_kept_term_needs(centered, non_centered):
    with pytest.raises(ValueError, match="drops \\['spread'\\], which the prior term 'soil_carbon'"):
        centered.select(parameter=["mean", "soil_carbon"])
    pair = Prior(vector_of(RATE, SHARE), [copula(("rate", "share"), rate=RATE_MARGINAL, share=SHARE_MARGINAL)])
    with pytest.raises(ValueError, match="which the prior term 'rate\\+share' covers"):
        pair.select(parameter=["rate"])
    # A derived parameter is dropped with its inputs; the prior never sees it.
    assert non_centered.select(parameter=["mean", "spread"]).derived_parameters.derived_parameter_names == ()


def test_select_refuses_to_drop_what_a_derived_parameter_given_needs():
    with pytest.raises(ValueError, match="drops \\['log_mean'\\]"):
        centered_given_a_derived_mean().select(parameter=["intercept", "spread", "respiration"])


def test_a_joint_density_on_the_simplex_is_refused():
    shares = Parameter(name="shares", support=SIMPLEX, units="1", shape=(3,))
    joint = tfd.JointDistributionNamed({
        "shares": tfd.Dirichlet(jnp.full(3, 2.0)), "rate": tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)),
    })
    with pytest.raises(ValueError, match="other than a Dirichlet"):
        Prior(vector_of(shares, RATE), [term(("shares", "rate"), joint)])


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


def test_a_copula_is_for_parameters_indexed_by_nothing():
    by_site = Parameter(name="rate", support=POSITIVE, units="yr-1", indexed_by=("site",))
    with pytest.raises(TypeError, match="prior of parameters indexed by nothing"):
        Prior(vector_of(by_site, SOIL_CARBON), [copula(("rate", "soil_carbon"), rate=RATE_MARGINAL,
                                                       soil_carbon=RATE_MARGINAL)])


def test_zero_draws_of_a_prior_evaluated_by_change_of_variables(non_centered, centered):
    assert non_centered.sample(jax.random.key(0), 0).shape == (0, non_centered.parameter_vector.unconstrained.size)
    assert centered.sample(jax.random.key(0), 0).shape == (0, centered.parameter_vector.unconstrained.size)


def test_a_joint_term_given_others_may_be_a_joint_distribution():
    def pair_given_spread(spread):
        return tfd.JointDistributionNamedAutoBatched({
            "rate": tfd.Gamma(jnp.float64(3.0), spread), "share": tfd.Beta(jnp.float64(2.0), spread),
        })

    prior = Prior(vector_of(RATE, SHARE, SPREAD), [
        term(("rate", "share"), pair_given_spread, given=("spread",)),
        term("spread", log_normal(median=0.5, geometric_sd=2.0)),
    ])
    theta = prior.sample(jax.random.key(8), 4)
    values = natural(prior, theta)
    spread = values["spread"]
    expected = (
        tfd.Gamma(jnp.float64(3.0), spread).log_prob(values["rate"]) + theta[:, 0]
        + tfd.Beta(jnp.float64(2.0), spread).log_prob(values["share"])
        + jnp.log(values["share"] * (1 - values["share"]))
        + prior._built["spread"].log_prob(theta[:, prior._built["spread"].positions], None)
    )
    np.testing.assert_allclose(jax.jit(prior.log_prob)(theta), expected, rtol=1e-10)


def test_a_term_over_one_parameter_has_its_value_as_its_event():
    prior = Prior(vector_of(RATE), [term(("rate",), RATE_MARGINAL)])
    assert prior.describe().index.tolist() == ["rate"]
    assert not prior._built["rate"].joint


def test_a_term_is_not_given_what_it_covers_or_a_name_twice():
    with pytest.raises(ValueError, match="is given \\['rate'\\], which it covers"):
        Prior(vector_of(RATE), [term("rate", rate_given, given=("rate",))])
    with pytest.raises(ValueError, match="given"):
        PriorTerm(parameter_names=("share",), distribution=share_given, given=("rate", "rate"), provenance="test")


def test_a_term_whose_evaluation_changes_with_its_given_values_is_refused():
    calls = []

    def shifted(rate):
        # One structure at both ancestral draws, but a pushforward through the
        # parameter's own bijector at the first only.
        calls.append(rate)
        return tfd.TransformedDistribution(normal(0.0, 1.0), tfb.Shift(jnp.float64(0.0 if len(calls) == 1 else 1.0)))

    offset = Parameter(name="offset", support=REAL, units=None)
    with pytest.raises(ValueError, match="changes its structure"):
        Prior(vector_of(RATE, offset), [term("rate", RATE_MARGINAL), term("offset", shifted, given=("rate",))])
