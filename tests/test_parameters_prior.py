"""Tests for the prior: the family builders, the priors over the index
dims, the Prior's members, and every check a term passes at construction.

The mathematics (densities integrating to one, the change of variables, the
declarations against draws) is in ``test_parameters_prior_conformance.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from conftest import theta_gaussian

from sipnet_calibration.parameters import prior as module
from sipnet_calibration.parameters.parameter import Parameter
from sipnet_calibration.parameters.families import (
    log_normal,
    log_normal_from_interval,
    log_normal_from_samples,
    logit_normal,
    logit_normal_from_interval,
    logit_normal_from_samples,
    softmax_normal,
)
from sipnet_calibration.parameters.prior import (
    Prior,
    PriorTerm,
)
from sipnet_calibration.parameters.prior_functions import (
    iid_over_dim,
    independent_over_dim,
)
from sipnet_calibration.parameters.support import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    UNIT_INTERVAL,
    Interval,
)
from sipnet_calibration.parameters.vector import ParameterVector

tfd, tfb = tfp.distributions, tfp.bijectors

SITES = np.array([1, 27, 4711], dtype=np.int32)
PFT = ("conifer", "deciduous")
CENTER = (0.18, 0.40, 0.07, 0.35)
PARTS = ("leaf", "wood", "fine_root", "coarse_root")


def vector_of(*parameters: Parameter) -> ParameterVector:
    coords = {"pft": PFT, "site": SITES}
    used = {d for p in parameters for d in p.indexed_by}
    return ParameterVector(parameters=parameters, coords={d: v for d, v in coords.items() if d in used})


def term(parameter_names, distribution, **arguments) -> PriorTerm:
    """A test term; one parameter's name may be given alone."""
    names = (parameter_names,) if isinstance(parameter_names, str) else parameter_names
    return PriorTerm(parameter_names=names, distribution=distribution, provenance="test", **arguments)


def by_site(values) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=np.float64), dims="site", coords={"site": SITES})


RATE = Parameter(name="rate", support=POSITIVE, units="yr-1")
SHARE = Parameter(name="share", support=OPEN_UNIT_INTERVAL, units="1")
ALLOCATION = Parameter(name="allocation", support=SIMPLEX, units="1", shape=(4,), indexed_by=("pft",),
                       element_labels={"allocation_part": PARTS})
SOIL = Parameter(name="soil_carbon", support=POSITIVE, units="g m-2", indexed_by=("site",))


@pytest.fixture(scope="module")
def prior() -> Prior:
    return Prior(
        vector_of(RATE, SHARE, ALLOCATION, SOIL),
        [
            term("rate", log_normal_from_interval(lower=0.004, upper=0.02)),
            term("share", logit_normal(median=0.2, logit_sd=0.4)),
            term("allocation", iid_over_dim(softmax_normal(center=CENTER, logit_sd=0.5))),
            term("soil_carbon", independent_over_dim(log_normal, geometric_sd=2.0),
                                constants={"median": by_site([1e4, 2e4, 3e4])}),
        ],
    )


# ── the family builders ───────────────────────────────────────────────────────


def test_log_normal_has_its_median_and_interval():
    assert float(log_normal(median=0.01, geometric_sd=2.0).quantile(0.5)) == pytest.approx(0.01)
    interval = log_normal_from_interval(lower=0.004, upper=0.020)
    assert float(interval.quantile(0.025)) == pytest.approx(0.004)
    assert float(interval.quantile(0.975)) == pytest.approx(0.020)


def test_logit_normal_has_its_median_and_interval():
    assert float(logit_normal(median=0.3, logit_sd=1.0).quantile(0.5)) == pytest.approx(0.3)
    interval = logit_normal_from_interval(lower=0.1, upper=0.4)
    assert float(interval.quantile(0.975)) == pytest.approx(0.4)


def test_logit_normal_on_an_interval_takes_the_logit_of_the_fraction():
    q10 = logit_normal(median=2.0, logit_sd=0.5, support=Interval(1.0, 5.0))
    assert type(q10) is tfd.TransformedDistribution
    assert float(q10.bijector.forward(q10.distribution.loc)) == pytest.approx(2.0)
    np.testing.assert_allclose(q10.distribution.loc, np.log(0.25 / 0.75))
    interval = logit_normal_from_interval(lower=1.5, upper=3.0, support=Interval(1.0, 5.0))
    draws = interval.sample(20_000, seed=jax.random.key(0))
    assert float(jnp.mean((draws > 1.5) & (draws < 3.0))) == pytest.approx(0.95, abs=0.01)


def test_a_closed_interval_puts_the_logit_normal_on_its_interior():
    closed = logit_normal(median=0.3, logit_sd=1.0, support=UNIT_INTERVAL)
    assert type(closed) is tfd.LogitNormal


def test_fitting_to_samples_recovers_the_parameters():
    draws = log_normal(median=2.0, geometric_sd=1.5).sample(20_000, seed=jax.random.key(1))
    fitted = log_normal_from_samples(draws)
    assert float(fitted.quantile(0.5)) == pytest.approx(2.0, rel=0.02)
    shares = 1.0 + 4.0 * logit_normal(median=0.25, logit_sd=0.3).sample(20_000, seed=jax.random.key(2))
    fitted = logit_normal_from_samples(shares, support=Interval(1.0, 5.0))
    assert float(fitted.bijector.forward(fitted.distribution.loc)) == pytest.approx(2.0, rel=0.02)


def test_softmax_normal_is_centered_on_its_center():
    distribution = softmax_normal(center=CENTER, logit_sd=0.5)
    np.testing.assert_allclose(distribution.bijector.forward(distribution.distribution.mean()), CENTER)


@pytest.mark.parametrize(
    "build, error, message",
    [
        (lambda: log_normal(median=-1.0, geometric_sd=2.0), ValueError, "not finite and positive"),
        (lambda: log_normal(median=1.0, geometric_sd=0.5), ValueError, "give a value above 1"),
        (lambda: log_normal_from_interval(lower=2.0, upper=1.0), ValueError, "upper does not exceed lower"),
        (lambda: log_normal_from_interval(lower=1.0, upper=2.0, mass=1.0), ValueError, "outside \\(0, 1\\)"),
        (lambda: logit_normal(median=1.2, logit_sd=1.0), ValueError, "give a value inside it"),
        (lambda: logit_normal(median=0.5, logit_sd=1.0, support=POSITIVE), ValueError, "on a finite interval"),
        (lambda: logit_normal(median=0.5, logit_sd=1.0, support=SIMPLEX), TypeError, "must be an Interval"),
        (lambda: log_normal_from_samples([1.0]), ValueError, "at least two"),
        (lambda: log_normal_from_samples([1.0, -1.0]), ValueError, "outside the support"),
        (lambda: log_normal_from_samples([1.0, 1.0]), ValueError, "all equal"),
        (lambda: softmax_normal(center=(0.5, 0.6), logit_sd=1.0), ValueError, "summing to 1"),
        (lambda: softmax_normal(center=(1.0,), logit_sd=1.0), ValueError, "k >= 2"),
        (lambda: softmax_normal(center=CENTER, logit_sd=(1.0, 1.0)), ValueError, "one value per unconstrained"),
        (lambda: softmax_normal(center=[CENTER[1:] + (0.18,)] * 3, logit_sd=(1.0, 1.0, 1.0)), ValueError,
         "could be one value per label"),
    ],
)
def test_the_builders_refuse_bad_arguments(build, error, message):
    with pytest.raises(error, match=message):
        build()


# ── priors over the index dims ────────────────────────────────────────────────


def test_iid_over_dim_puts_the_bijector_outside(prior):
    distribution = prior._built["allocation"].distribution
    assert type(distribution) is tfd.TransformedDistribution
    assert type(distribution.distribution) is tfd.Sample
    assert tuple(distribution.event_shape) == (2, 4)


def test_iid_over_dim_covers_the_product_of_the_index_dims():
    rate = Parameter(name="rate", support=POSITIVE, units="yr-1", indexed_by=("site", "pft"))
    prior = Prior(vector_of(rate), [term("rate", iid_over_dim(log_normal(median=1.0, geometric_sd=2.0)))])
    assert tuple(prior._built["rate"].distribution.event_shape) == (3, 2)
    _, variances = theta_gaussian(prior, "rate")
    np.testing.assert_allclose(variances, np.full(6, np.log(2.0) ** 2))


def test_independent_over_dim_reads_its_constants_by_label():
    medians = xr.DataArray([3e4, 1.0, 1e4, 2e4], dims="site", coords={"site": [4711, 99, 1, 27]})
    prior = Prior(vector_of(SOIL), [term("soil_carbon", independent_over_dim(log_normal, geometric_sd=2.0),
                                                        constants={"median": medians})])
    base = prior._built["soil_carbon"].distribution.distribution.distribution
    np.testing.assert_allclose(np.exp(base.loc), [1e4, 2e4, 3e4])


def test_a_constant_on_an_element_axis_is_read_at_the_element_labels():
    centers = xr.DataArray(
        [[0.25, 0.25, 0.25, 0.25], [0.35, 0.07, 0.40, 0.18]],
        dims=("pft", "allocation_part"), coords={"pft": ["deciduous", "conifer"], "allocation_part": list(PARTS[::-1])},
    )
    logit_sd = xr.DataArray([5.0, 0.1], dims="pft", coords={"pft": ["deciduous", "conifer"]})
    prior = Prior(vector_of(ALLOCATION), [term("allocation", independent_over_dim(softmax_normal),
                                                              constants={"center": centers, "logit_sd": logit_sd})])
    mean, variances = theta_gaussian(prior, "allocation")
    np.testing.assert_allclose(variances, [0.01] * 3 + [25.0] * 3)
    np.testing.assert_allclose(mean[:3], np.log(np.asarray(CENTER[:3]) / CENTER[3]))


def test_sample_takes_zero_draws(prior):
    assert prior.sample(jax.random.key(0), 0).shape == (0, prior.parameter_vector.unconstrained.size)


def test_a_constant_missing_a_label_is_refused():
    with pytest.raises(KeyError, match="no value at 'site' label"):
        Prior(vector_of(SOIL), [term("soil_carbon", independent_over_dim(log_normal, geometric_sd=2.0),
                                                    constants={"median": by_site([1.0, 2.0, 3.0]).isel(site=[0])})])


def test_independent_over_dim_needs_one_distribution_per_label():
    with pytest.raises(ValueError, match="give at least one argument per label"):
        Prior(vector_of(SOIL), [term("soil_carbon", independent_over_dim(log_normal, median=1.0, geometric_sd=2.0))])


def test_a_prior_over_the_index_dims_needs_an_indexed_parameter():
    with pytest.raises(TypeError, match="law over a component's index dims"):
        Prior(vector_of(RATE), [term("rate", iid_over_dim(log_normal(median=1.0, geometric_sd=2.0)))])


def test_an_indexed_parameter_needs_a_prior_function():
    with pytest.raises(TypeError, match="iid_over_dim"):
        Prior(vector_of(SOIL), [term("soil_carbon", log_normal(median=1.0, geometric_sd=2.0))])


def test_a_term_reading_constants_needs_a_prior_function():
    with pytest.raises(TypeError, match="reads \\['scale'\\] but is a distribution"):
        Prior(vector_of(RATE), [term("rate", log_normal(median=1.0, geometric_sd=2.0),
                                             constants={"scale": xr.DataArray(1.0)})])


# ── the prior ─────────────────────────────────────────────────────────────────


def test_sample_draws_theta_in_the_vectors_order(prior):
    theta = prior.sample(jax.random.key(0), 7)
    assert theta.shape == (7, prior.parameter_vector.unconstrained.size)
    assert bool(jnp.isfinite(prior.log_prob(theta)).all())


def test_log_prob_takes_any_leading_shape_and_is_traceable(prior):
    theta = prior.sample(jax.random.key(0), 6).reshape((2, 3, -1))
    assert prior.log_prob(theta).shape == (2, 3)
    gradient = jax.jit(jax.grad(lambda t: prior.log_prob(t)))(theta[0, 0])
    assert bool(jnp.isfinite(gradient).all())
    with pytest.raises(ValueError, match="ends in 11 entries"):
        prior.log_prob(jnp.zeros(3))


def test_the_prior_follows_any_order():
    vector = vector_of(RATE, ALLOCATION, SOIL)
    site_major = ParameterVector(parameters=vector.parameters, coords=vector.coords, order=("site", "pft", "parameter"))
    terms = [
        term("rate", log_normal_from_interval(lower=0.004, upper=0.02)),
        term("allocation", iid_over_dim(softmax_normal(center=CENTER, logit_sd=0.5))),
        term("soil_carbon", independent_over_dim(log_normal, geometric_sd=2.0), constants={"median": by_site([1e4, 2e4, 3e4])}),
    ]
    first, second = Prior(vector, terms), Prior(site_major, terms)
    theta = first.sample(jax.random.key(3), 5)
    values = vector.unconstrained.flat_to_values(theta)
    moved = site_major.unconstrained.values_to_flat(values)
    np.testing.assert_allclose(second.log_prob(moved), first.log_prob(theta), rtol=1e-13)
    np.testing.assert_allclose(site_major.unconstrained.flat_to_values(second.sample(jax.random.key(3), 5))["soil_carbon"],
                               values["soil_carbon"])


def test_describe_says_how_each_term_is_evaluated(prior):
    table = prior.describe()
    assert list(table.index) == ["rate", "share", "allocation", "soil_carbon"]
    assert set(table["evaluated_by"]) == {"base density"}
    assert table.loc["allocation", "prior"] == "iid softmax-normal"


def test_select_rebuilds_the_terms_on_the_kept_labels(prior):
    smaller = prior.select(site=[27], parameter=["allocation", "soil_carbon"])
    assert smaller.parameter_vector.unconstrained.size == 7
    base = smaller._built["soil_carbon"].distribution.distribution.distribution
    np.testing.assert_allclose(np.exp(base.loc), [2e4])


def test_getitem_returns_the_term_covering_a_parameter(prior):
    assert prior["rate"].provenance == "test" and prior["rate"].parameter_names == ("rate",)
    assert repr(prior) == "Prior(D=11, terms=['rate', 'share', 'allocation', 'soil_carbon'])"
    with pytest.raises(KeyError, match="no term covering 'nothing'"):
        prior["nothing"]


def test_the_terms_are_held_in_the_vectors_order_whatever_theirs(prior):
    reversed_ = Prior(prior.parameter_vector, prior.terms[::-1])
    assert [t.name for t in reversed_.terms] == [t.name for t in prior.terms]


# ── construction checks ───────────────────────────────────────────────────────


def test_every_parameter_is_covered_by_one_term():
    vector = vector_of(RATE, SHARE)
    with pytest.raises(ValueError, match="no prior term"):
        Prior(vector, [term("rate", log_normal(median=1.0, geometric_sd=2.0))])
    with pytest.raises(KeyError, match="no parameter of the vector"):
        Prior(vector_of(RATE), [term("rate", log_normal(median=1.0, geometric_sd=2.0)),
                                term("other", log_normal(median=1.0, geometric_sd=2.0))])
    with pytest.raises(TypeError, match="give a sequence of PriorTerms"):
        Prior(vector, {"rate": term("rate", log_normal(median=1.0, geometric_sd=2.0))})
    with pytest.raises(ValueError, match="covered by the terms"):
        Prior(vector, [term("rate", log_normal(median=1.0, geometric_sd=2.0)),
                       term(("rate", "share"), log_normal(median=1.0, geometric_sd=2.0))])


def test_a_term_is_a_prior_term_with_a_provenance():
    with pytest.raises(TypeError, match="wrap it as PriorTerm"):
        Prior(vector_of(RATE), [log_normal(median=1.0, geometric_sd=2.0)])
    with pytest.raises(ValueError, match="needs a provenance"):
        PriorTerm(parameter_names=("rate",), distribution=log_normal(median=1.0, geometric_sd=2.0), provenance=" ")


def test_a_term_names_the_parameters_it_is_the_prior_of():
    with pytest.raises(ValueError, match="has no parameter_names"):
        PriorTerm(parameter_names=(), distribution=log_normal(median=1.0, geometric_sd=2.0), provenance="test")
    with pytest.raises(TypeError, match="parameter_names"):
        PriorTerm(parameter_names="rate", distribution=log_normal(median=1.0, geometric_sd=2.0), provenance="test")
    assert term(("rate", "share"), log_normal(median=1.0, geometric_sd=2.0)).name == "rate+share"


def test_a_terms_distribution_is_a_distribution_or_a_function():
    with pytest.raises(TypeError, match="give a TFP distribution or a prior function"):
        term("rate", 1.0)


def test_a_terms_constant_may_be_on_an_element_axis_of_what_it_is_given():
    center = Parameter(name="center", support=SIMPLEX, units="1", shape=(2,),
                       element_labels={"part": ("leaf", "wood")})
    shares = Parameter(name="shares", support=SIMPLEX, units="1", shape=(2,))
    scale = xr.DataArray([0.5, 1.0], dims="part", coords={"part": ["leaf", "wood"]})

    def shares_given_center(center, scale):
        loc = jnp.log(center[:-1] / center[-1:])
        return tfd.TransformedDistribution(tfd.MultivariateNormalDiag(loc, scale[:-1]), tfb.SoftmaxCentered())

    prior = Prior(vector_of(center, shares), [
        term("center", softmax_normal(center=(0.5, 0.5), logit_sd=0.5)),
        term("shares", shares_given_center, given=("center",), constants={"scale": scale}),
    ])
    assert bool(jnp.isfinite(prior.log_prob(prior.sample(jax.random.key(3), 2))).all())


def test_a_terms_keywords_name_one_thing_each():
    with pytest.raises(ValueError, match="more than once"):
        term("rate", lambda mean: None, given=("mean",), constants={"mean": xr.DataArray(1.0)})
    with pytest.raises(ValueError, match="more than once"):
        term("rate", lambda rate: None, constants={"rate": xr.DataArray(1.0)})
    with pytest.raises(TypeError, match="give a DataArray"):
        term("rate", lambda scale: None, constants={"scale": 1.0})


def test_a_term_covers_the_whole_value_in_float64():
    with pytest.raises(ValueError, match="TFP batch shape"):
        Prior(vector_of(RATE), [term("rate", log_normal(median=[1.0, 2.0], geometric_sd=2.0))])
    with pytest.raises(TypeError, match="not a TFP distribution"):
        Prior(vector_of(SOIL), [term("soil_carbon", lambda: 1.0)])


def test_a_draw_on_the_boundary_is_refused():
    # A draw of exactly 0 on the positive line has theta = -inf; whether a
    # given sampler reaches it depends on TFP's batch shapes, so the check is
    # provoked directly.
    with pytest.raises(ValueError, match="whose theta is not finite"):
        module.check_draws_map_to_finite_theta("rate", jnp.asarray([0.0, -jnp.inf]))


def test_a_dirichlet_on_the_simplex_is_accepted():
    shares = Parameter(name="shares", support=SIMPLEX, units="1", shape=(3,))
    prior = Prior(vector_of(shares), [term("shares", tfd.Dirichlet(jnp.full(3, 2.0)))])
    assert prior.describe().loc["shares", "evaluated_by"] == "change of variables"


def test_a_prior_that_can_neither_be_sampled_nor_mapped_is_refused():
    class Opaque(tfd.Normal):
        def _sample_n(self, n, seed=None):
            raise NotImplementedError

        def experimental_default_event_space_bijector(self, *args, **kwargs):
            return None

    with pytest.raises(ValueError, match="neither a default event-space bijector nor a sampler"):
        Prior(vector_of(Parameter(name="offset", support=REAL, units=None)),
              [term("offset", Opaque(jnp.float64(0.0), jnp.float64(1.0)))])


def test_a_family_under_another_bijector_is_evaluated_by_change_of_variables():
    softplus_rate = Parameter(name="rate", support=POSITIVE, units="1", bijector=tfb.Softplus())
    prior = Prior(vector_of(softplus_rate), [term("rate", log_normal(median=1.0, geometric_sd=2.0))])
    assert prior.describe().loc["rate", "evaluated_by"] == "change of variables"


def test_a_change_of_variables_needs_a_known_measure():
    from dataclasses import dataclass

    from sipnet_calibration.parameters.support import Support

    @dataclass(frozen=True)
    class Halfline(Support):
        event_ndims = 0

        @property
        def name(self):
            return "halfline"

        def contains(self, values):
            values = jnp.asarray(values, dtype=jnp.float64)
            return jnp.isfinite(values) & (values > 0)

        def closure(self):
            return self

    rate = Parameter(name="rate", support=Halfline(), units="1", bijector=tfb.Exp())
    with pytest.raises(ValueError, match="no known reference measure"):
        Prior(vector_of(rate), [term("rate", tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)))])


def test_derived_parameters_over_another_vector_are_refused():
    from sipnet_calibration.parameters.derived import (
        DerivedParameter,
        DerivedParameters,
    )

    other = vector_of(SHARE)
    derived = DerivedParameters(parameter_vector=other, derived_parameters=[
        DerivedParameter(name="double", units="1", given=("share",), function=lambda share: 2 * share),
    ])
    with pytest.raises(ValueError, match="differ in their"):
        Prior(vector_of(RATE), [term("rate", log_normal(median=1.0, geometric_sd=2.0))], derived_parameters=derived)
    with pytest.raises(TypeError, match="must be a DerivedParameters"):
        Prior(vector_of(RATE), [term("rate", log_normal(median=1.0, geometric_sd=2.0))], derived_parameters=[])


def test_a_constant_is_read_in_the_parameters_dim_order_whatever_its_own():
    rate = Parameter(name="rate", support=POSITIVE, units="yr-1", indexed_by=("pft", "site"))
    vector = ParameterVector(parameters=[rate], coords={"pft": ["a", "b"], "site": [3, 5]})
    # Built on (site, pft), with as many sites as PFTs, so a transposed read would pass unnoticed.
    median = xr.DataArray([[1.0, 10.0], [100.0, 1000.0]], dims=("site", "pft"), coords={"site": [3, 5], "pft": ["a", "b"]})
    prior = Prior(vector, [term("rate", independent_over_dim(log_normal, geometric_sd=1.5), constants={"median": median})])
    mean, _ = theta_gaussian(prior, "rate")
    np.testing.assert_allclose(np.exp(mean).reshape(2, 2), median.transpose("pft", "site").values)
