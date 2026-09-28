"""Tests for the prior: the family builders, the priors over a dim, the
Prior's members, and every check a term passes at construction.

The mathematics (densities integrating to one, the change of variables, the
declarations against draws) is in ``test_prior_conformance.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pyeki.linalg import DensePSD, PSDDiagonal
from tensorflow_probability.substrates import jax as tfp

from conftest import site_table_of
from sipnet_calibration import prior as module
from sipnet_calibration.parameter_vector import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    OpenInterval,
    Parameter,
    ParameterVector,
)
from sipnet_calibration.prior import (
    Prior,
    PriorTerm,
    independent_over_dim,
    iid_over_dim,
    log_normal,
    log_normal_from_interval,
    log_normal_from_samples,
    logit_normal,
    logit_normal_from_interval,
    logit_normal_from_samples,
    softmax_normal,
)

tfd, tfb = tfp.distributions, tfp.bijectors

SITES = (1, 27, 4711)
PFT = ("deciduous", "conifer", "deciduous")
CENTER = (0.18, 0.40, 0.07, 0.35)


def vector_of(*parameters: Parameter) -> ParameterVector:
    return ParameterVector(parameters=parameters, site_table=site_table_of(*SITES), site_labels={"pft": PFT})


def term(distribution) -> PriorTerm:
    return PriorTerm(distribution, provenance="test")


RATE = Parameter(name="rate", support=POSITIVE, units="yr-1")
SHARE = Parameter(name="share", support=OPEN_UNIT_INTERVAL, units="1")
ALLOCATION = Parameter(
    name="allocation", support=SIMPLEX, units="1", dim="pft",
    natural_names=("leaf", "wood", "fine_root", "coarse_root"),
)
SOIL = Parameter(name="soil_carbon", support=POSITIVE, units="g m-2", dim="site")


@pytest.fixture(scope="module")
def prior() -> Prior:
    return Prior(
        vector_of(RATE, SHARE, ALLOCATION, SOIL),
        {
            "rate": term(log_normal_from_interval(lower=0.004, upper=0.02)),
            "share": term(logit_normal(median=0.2, logit_sd=0.4)),
            "allocation": term(iid_over_dim(softmax_normal(center=CENTER, logit_sd=0.5))),
            "soil_carbon": term(independent_over_dim(log_normal, median={1: 1e4, 27: 2e4, 4711: 3e4},
                                                     geometric_sd=2.0)),
        },
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


def test_logit_normal_on_an_open_interval_takes_the_logit_of_the_fraction():
    q10 = logit_normal(median=2.0, logit_sd=0.5, support=OpenInterval(1.0, 5.0))
    assert type(q10) is tfd.TransformedDistribution
    assert float(q10.bijector.forward(q10.distribution.loc)) == pytest.approx(2.0)
    np.testing.assert_allclose(q10.distribution.loc, np.log(0.25 / 0.75))
    interval = logit_normal_from_interval(lower=1.5, upper=3.0, support=OpenInterval(1.0, 5.0))
    draws = interval.sample(20_000, seed=jax.random.key(0))
    assert float(jnp.mean((draws > 1.5) & (draws < 3.0))) == pytest.approx(0.95, abs=0.01)


def test_fitting_to_samples_recovers_the_parameters():
    draws = log_normal(median=2.0, geometric_sd=1.5).sample(20_000, seed=jax.random.key(1))
    fitted = log_normal_from_samples(draws)
    assert float(fitted.quantile(0.5)) == pytest.approx(2.0, rel=0.02)
    shares = 1.0 + 4.0 * logit_normal(median=0.25, logit_sd=0.3).sample(20_000, seed=jax.random.key(2))
    fitted = logit_normal_from_samples(shares, support=OpenInterval(1.0, 5.0))
    assert float(fitted.bijector.forward(fitted.distribution.loc)) == pytest.approx(2.0, rel=0.02)


def test_softmax_normal_is_centered_on_its_center():
    distribution = softmax_normal(center=CENTER, logit_sd=0.5)
    np.testing.assert_allclose(distribution.bijector.forward(distribution.distribution.mean()), CENTER)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: log_normal(median=-1.0, geometric_sd=2.0), "finite and positive"),
        (lambda: log_normal(median=1.0, geometric_sd=0.5), "must exceed 1"),
        (lambda: log_normal_from_interval(lower=2.0, upper=1.0), "upper must exceed lower"),
        (lambda: log_normal_from_interval(lower=1.0, upper=2.0, mass=1.0), "mass must lie"),
        (lambda: logit_normal(median=1.2, logit_sd=1.0), "must lie inside"),
        (lambda: logit_normal(median=0.5, logit_sd=1.0, support=POSITIVE), "open interval"),
        (lambda: log_normal_from_samples([1.0]), "at least two"),
        (lambda: log_normal_from_samples([1.0, -1.0]), "outside the support"),
        (lambda: log_normal_from_samples([1.0, 1.0]), "all equal"),
        (lambda: softmax_normal(center=(0.5, 0.6), logit_sd=1.0), "summing to 1"),
        (lambda: softmax_normal(center=(1.0,), logit_sd=1.0), "k >= 2"),
        (lambda: softmax_normal(center=CENTER, logit_sd=(1.0, 1.0)), "one value per unconstrained"),
    ],
)
def test_the_builders_refuse_bad_arguments(build, message):
    with pytest.raises(ValueError, match=message):
        build()


# ── priors over a dim ─────────────────────────────────────────────────────────


def test_iid_over_dim_puts_the_bijector_outside(prior):
    distribution = prior._built["allocation"].distribution
    assert type(distribution) is tfd.TransformedDistribution
    assert type(distribution.distribution) is tfd.Sample
    assert tuple(distribution.event_shape) == (2, 4)


def test_independent_over_dim_aligns_by_dim_label():
    medians = pd.Series({4711: 3e4, 99: 1.0, 1: 1e4, 27: 2e4})
    prior = Prior(vector_of(SOIL), {"soil_carbon": term(independent_over_dim(log_normal, median=medians,
                                                                              geometric_sd=2.0))})
    base = prior._built["soil_carbon"].distribution.distribution.distribution
    np.testing.assert_allclose(np.exp(base.loc), [1e4, 2e4, 3e4])


def test_independent_over_dim_refuses_a_missing_dim_label():
    with pytest.raises(KeyError, match="no value for dim label"):
        Prior(vector_of(SOIL), {"soil_carbon": term(independent_over_dim(log_normal, median={1: 1.0},
                                                                          geometric_sd=2.0))})


def test_independent_over_dim_needs_one_distribution_per_dim_label():
    with pytest.raises(ValueError, match="key at least one argument"):
        Prior(vector_of(SOIL), {"soil_carbon": term(independent_over_dim(log_normal, median=1.0,
                                                                          geometric_sd=2.0))})


def test_a_prior_over_a_dim_needs_a_parameter_with_one():
    with pytest.raises(TypeError, match="prior over a dim"):
        Prior(vector_of(RATE), {"rate": term(iid_over_dim(log_normal(median=1.0, geometric_sd=2.0)))})


def test_a_parameter_with_a_dim_needs_a_prior_function():
    with pytest.raises(TypeError, match="iid_over_dim"):
        Prior(vector_of(SOIL), {"soil_carbon": term(log_normal(median=1.0, geometric_sd=2.0))})


# ── the prior ─────────────────────────────────────────────────────────────────


def test_sample_draws_theta_in_the_vectors_order(prior):
    theta = prior.sample(jax.random.key(0), 7)
    assert theta.shape == (7, prior.parameter_vector.dimension)
    assert bool(jnp.isfinite(prior.log_prob(theta)).all())


def test_log_prob_takes_any_leading_shape_and_is_traceable(prior):
    theta = prior.sample(jax.random.key(0), 6).reshape((2, 3, -1))
    assert prior.log_prob(theta).shape == (2, 3)
    gradient = jax.jit(jax.grad(lambda t: prior.log_prob(t)))(theta[0, 0])
    assert bool(jnp.isfinite(gradient).all())
    with pytest.raises(ValueError, match="must end in the vector's dimension"):
        prior.log_prob(jnp.zeros(3))


def test_describe_says_how_each_term_is_evaluated(prior):
    table = prior.describe()
    assert list(table.index) == ["rate", "share", "allocation", "soil_carbon"]
    assert set(table["evaluated_by"]) == {"base density"}
    assert table["declared_gaussian"].all()
    assert table.loc["allocation", "prior"] == "iid softmax-normal"


def test_gaussian_is_exact_for_declared_terms(prior):
    gaussian = prior.gaussian()
    blocks = gaussian.cov.blocks
    assert len(blocks) == 4 and all(isinstance(b, PSDDiagonal) for b in blocks)
    rate = log_normal_from_interval(lower=0.004, upper=0.02)
    assert float(gaussian.mean[0]) == pytest.approx(float(rate.distribution.loc))
    assert float(blocks[0].diagonal[0]) == pytest.approx(float(rate.distribution.scale) ** 2)


def test_gaussian_moment_matches_an_undeclared_term():
    gamma = Prior(vector_of(RATE), {"rate": term(tfd.Gamma(jnp.float64(3.0), jnp.float64(2.0)))})
    with pytest.raises(NotImplementedError, match="pass key="):
        gamma.gaussian()
    gaussian = gamma.gaussian(key=jax.random.key(0), n_moment_samples=20_000)
    assert isinstance(gaussian.cov.blocks[0], DensePSD)
    draws = jnp.log(tfd.Gamma(3.0, 2.0).sample(200_000, seed=jax.random.key(1)))
    assert float(gaussian.mean[0]) == pytest.approx(float(draws.mean()), abs=0.02)


def test_select_rebuilds_the_terms_on_the_kept_dim_labels(prior):
    smaller = prior.select(sites=[27], parameter_names=["allocation", "soil_carbon"])
    assert smaller.parameter_vector.dimension == 4
    base = smaller._built["soil_carbon"].distribution.distribution.distribution
    np.testing.assert_allclose(np.exp(base.loc), [2e4])


def test_getitem_returns_the_term(prior):
    assert prior["rate"].provenance == "test"
    with pytest.raises(KeyError, match="no term"):
        prior["nothing"]


# ── construction checks ───────────────────────────────────────────────────────


def test_every_parameter_needs_one_term_keyed_by_its_name():
    vector = vector_of(RATE, SHARE)
    with pytest.raises(ValueError, match="no prior term"):
        Prior(vector, {"rate": term(log_normal(median=1.0, geometric_sd=2.0))})
    with pytest.raises(KeyError, match="no parameter of the vector"):
        Prior(vector_of(RATE), {"rate": term(log_normal(median=1.0, geometric_sd=2.0)),
                                "other": term(log_normal(median=1.0, geometric_sd=2.0))})
    with pytest.raises(TypeError, match="joint terms"):
        Prior(vector, {("rate", "share"): term(log_normal(median=1.0, geometric_sd=2.0))})


def test_a_term_is_a_prior_term_with_a_provenance():
    with pytest.raises(TypeError, match="wrap it as PriorTerm"):
        Prior(vector_of(RATE), {"rate": log_normal(median=1.0, geometric_sd=2.0)})
    with pytest.raises(ValueError, match="needs a provenance"):
        PriorTerm(log_normal(median=1.0, geometric_sd=2.0), provenance=" ")


def test_a_term_covers_the_whole_value_in_float64():
    with pytest.raises(ValueError, match="batch shape"):
        Prior(vector_of(RATE), {"rate": term(log_normal(median=[1.0, 2.0], geometric_sd=2.0))})
    with pytest.raises(TypeError, match="not a TFP distribution"):
        Prior(vector_of(SOIL), {"soil_carbon": term(lambda dim_index, site_table: 1.0)})


def test_a_draw_on_the_boundary_is_refused():
    # A draw of exactly 0 on the positive line has theta = -inf; whether a
    # given sampler reaches it depends on TFP's batch shapes, so the check is
    # provoked directly.
    with pytest.raises(ValueError, match="whose theta is not finite"):
        module.check_draws_map_to_finite_theta("rate", jnp.asarray([0.0, -jnp.inf]))


def test_a_dirichlet_on_the_simplex_is_accepted():
    shares = Parameter(name="shares", support=SIMPLEX, units="1", natural_names=("a", "b", "c"))
    prior = Prior(vector_of(shares), {"shares": term(tfd.Dirichlet(jnp.full(3, 2.0)))})
    assert prior.describe().loc["shares", "evaluated_by"] == "change of variables"


def test_a_prior_that_can_neither_be_sampled_nor_mapped_is_refused():
    class Opaque(tfd.Normal):
        def _sample_n(self, n, seed=None):
            raise NotImplementedError

        def experimental_default_event_space_bijector(self, *args, **kwargs):
            return None

    with pytest.raises(ValueError, match="neither a default event-space bijector nor a sampler"):
        Prior(vector_of(Parameter(name="offset", support=REAL, units=None)),
              {"offset": term(Opaque(jnp.float64(0.0), jnp.float64(1.0)))})


def test_a_declaration_that_disagrees_with_log_prob_is_refused(monkeypatch):
    real = module._family_gaussian

    def widened(distribution):
        found = real(distribution)
        return None if found is None else (found[0], 2.0 * found[1], found[2])

    monkeypatch.setattr(module, "_family_gaussian", widened)
    with pytest.raises(ValueError, match="disagrees with its log density"):
        Prior(vector_of(RATE), {"rate": term(log_normal(median=1.0, geometric_sd=2.0))})


def test_a_declaration_under_another_bijector_is_not_honored():
    softplus_rate = Parameter(name="rate", support=POSITIVE, units="1", bijector=tfb.Softplus())
    prior = Prior(vector_of(softplus_rate), {"rate": term(log_normal(median=1.0, geometric_sd=2.0))})
    row = prior.describe().loc["rate"]
    assert row["evaluated_by"] == "change of variables" and not row["declared_gaussian"]
