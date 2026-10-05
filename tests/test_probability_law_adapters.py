"""Tests for laws from other packages: numpyro's distributions and EnsKit's
Gaussian, adapted by ``as_law``, and GPJax's Gaussian through numpyro's.

Each adapted law is checked against its TFP equivalent or SciPy's density:
the same density in theta and at natural values, evaluated by the base
density where it is a pushforward through its component's own bijector,
and draws of the same law. A numpyro law with no density, or with a batch
shape, is refused as TFP's is; the builders, which repeat TFP laws, refuse
numpyro's; ``as_law`` and ``pushforward`` refuse what they cannot adapt.
The GPJax test runs only with the optional ``gpjax`` dependency group
installed (``uv sync --group gpjax``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as nd
import pytest
import scipy.stats as st
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability import (
    POSITIVE,
    POSITIVE_DEFINITE,
    SIMPLEX,
    ArraySpec,
    FactorSpec,
    GaussianLaw,
    NumpyroLaw,
    PushforwardLaw,
    _linalg,
    _numpyro,
    as_law,
    bijector_for,
    condition_on,
    iid_over_dim,
    joint,
    pushforward,
)

tfd, tfb = tfp.distributions, tfp.bijectors

SITES = [10, 20, 30]
KEY = jax.random.key(11)


def _posterior(*specs, coords=None):
    """The posterior of a model of *specs* with nothing observed: its prior."""
    return condition_on(joint(*specs).bind(coords=coords or {}), {})


def _field(name, law, **arguments):
    """A factor over a positive value per site."""
    return FactorSpec(ArraySpec(name, units="1", support=POSITIVE, indexed_by=("site",)), law=law, **arguments)


# ── as_law ────────────────────────────────────────────────────────────────────


def test_as_law_returns_a_tfp_distribution_itself():
    normal = tfd.Normal(0.0, 1.0)
    assert as_law(normal) is normal


def test_as_law_adapts_a_numpyro_distribution():
    law = as_law(nd.Normal(jnp.zeros(3), 2.0).to_event(1))
    assert isinstance(law, NumpyroLaw)
    assert law.event_shape == (3,) and law.batch_shape == () and law.dtype == jnp.float64
    x = jnp.arange(12.0).reshape((2, 2, 3))
    np.testing.assert_allclose(law.log_prob(x), st.norm(0.0, 2.0).logpdf(np.asarray(x)).sum(-1), rtol=1e-12)
    np.testing.assert_array_equal(law.sample((4,), seed=KEY), law.distribution.sample(KEY, (4,)))
    assert law.sample(5, seed=KEY).shape == (5, 3)


def test_a_numpyro_log_prob_that_does_not_broadcast_is_vmapped():
    class Unbroadcast(nd.Normal):
        """numpyro's Normal, refusing values with leading axes, as GPJax's
        Gaussian does."""

        def log_prob(self, value):
            assert value.shape == self.batch_shape + self.event_shape
            return super().log_prob(value)

    law = as_law(Unbroadcast(0.0, 1.0))
    np.testing.assert_allclose(law.log_prob(jnp.ones((2, 3))), st.norm.logpdf(np.ones((2, 3))), rtol=1e-12)


def test_as_law_adapts_an_enskit_gaussian():
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    law = as_law(_linalg.Gaussian.independent(u=(jnp.array([1.0, -1.0]), _linalg.DensePSD(jnp.asarray(covariance)))))
    assert isinstance(law, GaussianLaw)
    x = np.array([[0.0, 0.0], [1.5, -2.0]])
    np.testing.assert_allclose(law.log_prob(x), st.multivariate_normal([1.0, -1.0], covariance).logpdf(x), rtol=1e-12)


def test_as_law_adapts_an_enskit_gaussian_with_a_factor_row_by_its_blocks_covariance():
    factor = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]])
    gaussian = _linalg.Gaussian(
        {"u": jnp.zeros(3)}, factors={"u": jnp.asarray(factor)}, block_covs={"u": _linalg.PSDDiagonal(jnp.ones(3))}
    )
    law = as_law(gaussian)
    x = np.array([[0.0, 1.0, -1.0], [2.0, 0.5, 0.0]])
    expected = st.multivariate_normal(np.zeros(3), factor @ factor.T + np.eye(3)).logpdf(x)
    np.testing.assert_allclose(law.log_prob(x), expected, rtol=1e-12)
    draws = np.asarray(law.sample((20_000,), seed=KEY))
    np.testing.assert_allclose(np.cov(draws.T), factor @ factor.T + np.eye(3), atol=0.15)


def test_as_law_refuses_an_enskit_gaussian_it_cannot_score_as_one_block():
    covariance = _linalg.PSDDiagonal(jnp.ones(2))
    with pytest.raises(ValueError, match=r"over one block, not 2 \('u', 'v'\).*marginal"):
        as_law(_linalg.Gaussian.independent(u=(jnp.zeros(2), covariance), v=(jnp.zeros(2), covariance)))
    with pytest.raises(ValueError, match="block 'u' has no independent term.*add_noise"):
        as_law(_linalg.Gaussian({"u": jnp.zeros(3)}, factors={"u": jnp.ones((3, 2))}))


def test_as_law_refuses_what_it_does_not_know():
    with pytest.raises(TypeError, match="not a law"):
        as_law(object())
    with pytest.raises(TypeError, match="not a law"):
        as_law(nd.Normal)
    with pytest.raises(TypeError, match="adapts a numpyro distribution"):
        NumpyroLaw(tfd.Normal(0.0, 1.0))
    with pytest.raises(TypeError, match="seed"):
        as_law(nd.Normal(0.0, 1.0)).sample(2)


# ── pushforward ───────────────────────────────────────────────────────────────


def test_pushforward_of_a_numpyro_base_matches_tfps():
    ours = pushforward(nd.Normal(jnp.array([0.5, -1.0]), 0.7).to_event(1), support=POSITIVE)
    theirs = pushforward(tfd.Independent(tfd.Normal(jnp.array([0.5, -1.0]), 0.7), 1), support=POSITIVE)
    assert isinstance(ours, PushforwardLaw) and type(theirs) is tfd.TransformedDistribution
    assert ours.event_shape == (2,) and ours.batch_shape == ()
    x = jnp.array([[1.0, 2.0], [0.3, 0.1]])
    np.testing.assert_allclose(ours.log_prob(x), theirs.log_prob(x), rtol=1e-12)
    draws = ours.sample((3,), seed=KEY)
    np.testing.assert_allclose(draws, jnp.exp(ours.distribution.sample((3,), seed=KEY)), rtol=1e-12)


def test_a_numpyro_pushforward_onto_the_positive_definite_matrices_matches_tfps():
    """The bijector's log-Jacobian is over the value's rank, two here, not the
    base's, one."""
    spec = ArraySpec("covariance", units="1", support=POSITIVE_DEFINITE, element_axes={"row": 2, "column": 2})
    ours = _posterior(FactorSpec(spec, law=pushforward(nd.Normal(jnp.zeros(3), 1.0).to_event(1), support=POSITIVE_DEFINITE)))
    theirs = _posterior(FactorSpec(spec, law=pushforward(tfd.Independent(tfd.Normal(jnp.zeros(3), 1.0), 1),
                                                        support=POSITIVE_DEFINITE)))
    theta = jax.random.normal(KEY, (4, 3))
    np.testing.assert_allclose(ours.log_prior(theta), theirs.log_prior(theta), rtol=1e-12)
    law = pushforward(nd.Normal(jnp.zeros(3), 1.0).to_event(1), bijector=tfb.Chain([tfb.Scale(jnp.float64(2.0)), bijector_for(POSITIVE_DEFINITE)]))
    reference = tfd.TransformedDistribution(tfd.Independent(tfd.Normal(jnp.zeros(3), 1.0), 1), law.bijector)
    value = reference.sample(3, seed=KEY)
    np.testing.assert_allclose(law.log_prob(value), reference.log_prob(value), rtol=1e-12)


def test_a_numpyro_law_takes_a_value_without_its_batch_axes():
    law = as_law(nd.Normal(jnp.zeros((4, 3, 2)), 1.0).to_event(1))
    value = jnp.ones((3, 2))
    np.testing.assert_array_equal(law.log_prob(value), law.distribution.log_prob(value))


def test_a_numpyro_law_with_float32_parameters_is_refused():
    law = nd.Normal(np.float32(0.1), np.float32(1.3))
    assert as_law(law).dtype == jnp.float32
    with pytest.raises(ValueError, match="float32"):
        joint(FactorSpec(ArraySpec("q10", units="1"), law=law)).bind(coords={})


def test_a_pushforward_of_a_law_without_a_dtype_takes_its_draws():
    class Draws:
        event_shape = (3,)

        def log_prob(self, value):
            return jnp.sum(-0.5 * value**2, axis=-1)

        def sample(self, sample_shape=(), seed=None):
            return jax.random.normal(seed, (*sample_shape, 3))

    assert pushforward(Draws(), support=POSITIVE).dtype == jnp.float64


def test_pushforward_refuses_a_base_that_is_not_a_law():
    with pytest.raises(TypeError, match="give a law"):
        pushforward(object(), support=POSITIVE)

    class NoEventShape:
        def log_prob(self, value):
            return jnp.zeros(())

        def sample(self, sample_shape=(), seed=None):
            return jnp.zeros(sample_shape)

    with pytest.raises(TypeError, match="no event_shape"):
        pushforward(NoEventShape(), support=POSITIVE)


# ── factors ───────────────────────────────────────────────────────────────────


def test_a_numpyro_field_through_its_bijector_is_evaluated_by_its_base_density():
    def numpyro_field(location):
        return pushforward(nd.Normal(jnp.broadcast_to(location, (3,)), 0.4).to_event(1), support=POSITIVE)

    def tfp_field(location):
        return pushforward(tfd.Independent(tfd.Normal(jnp.broadcast_to(location, (3,)), 0.4), 1), support=POSITIVE)

    location = FactorSpec(ArraySpec("location", units="1"), law=tfd.Normal(jnp.float64(0.0), jnp.float64(1.0)))
    ours = joint(_field("capacity", numpyro_field), location).bind(coords={"site": SITES})
    theirs = joint(_field("capacity", tfp_field), location).bind(coords={"site": SITES})
    assert ours.describe().loc["capacity", "evaluated_by"] == "base density"
    assert ours.describe().loc["capacity", "law"] == "numpyro_field"

    theta = jax.random.normal(KEY, (6, 4))
    ours_posterior, theirs_posterior = condition_on(ours, {}), condition_on(theirs, {})
    np.testing.assert_allclose(ours_posterior.log_prior(theta), theirs_posterior.log_prior(theta), rtol=1e-12)
    values = theirs.sample(KEY, 5)
    np.testing.assert_allclose(ours.log_prob(values), theirs.log_prob(values), rtol=1e-12)


def test_a_bare_numpyro_law_matches_its_tfp_equivalent_by_change_of_variables():
    spec = ArraySpec("q10", units="1", support=POSITIVE)
    ours = condition_on(joint(FactorSpec(spec, law=nd.LogNormal(0.7, 0.3))).bind(coords={}), {})
    theirs = condition_on(joint(FactorSpec(spec, law=tfd.LogNormal(jnp.float64(0.7), jnp.float64(0.3)))).bind(coords={}), {})
    assert ours.model.describe().loc["q10", "evaluated_by"] == "change of variables"
    assert ours.model.describe().loc["q10", "law"] == "numpyro LogNormal"
    theta = jnp.linspace(-3.0, 3.0, 7)[:, None]
    np.testing.assert_allclose(ours.log_prior(theta), theirs.log_prior(theta), rtol=1e-12)

    draws = ours.sample_prior(KEY, 20_000)[:, 0]
    assert abs(float(draws.mean()) - 0.7) < 4 * 0.3 / np.sqrt(20_000)
    assert abs(float(draws.std()) - 0.3) < 0.01


def test_a_numpyro_law_reading_another_component_is_vmapped_over_draws():
    location = FactorSpec(ArraySpec("location", units="1"), law=tfd.Normal(jnp.float64(0.0), jnp.float64(1.0)))

    def ours(location):
        return nd.Normal(location, 2.0).expand([len(SITES)]).to_event(1)

    def theirs(location):
        return tfd.Independent(tfd.Normal(jnp.broadcast_to(location, (len(SITES),)), 2.0), 1)

    value = ArraySpec("value", units="1", indexed_by=("site",))
    ours_posterior = _posterior(FactorSpec(value, law=ours), location, coords={"site": SITES})
    theirs_posterior = _posterior(FactorSpec(value, law=theirs), location, coords={"site": SITES})
    theta = jax.random.normal(KEY, (5, 4))
    np.testing.assert_allclose(ours_posterior.log_prior(theta), theirs_posterior.log_prior(theta), rtol=1e-12)

    draws = ours_posterior.sample_prior(KEY, 4000)
    spread = np.asarray(draws[:, :3] - draws[:, 3:])
    assert abs(spread.std() - 2.0) < 0.1


def test_an_enskit_gaussian_is_a_factors_law():
    covariance = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, -0.4], [0.0, -0.4, 0.5]])
    mean = jnp.array([1.0, 0.0, -1.0])
    spec = ArraySpec("offsets", units="1", element_axes={"pool": ["leaf", "wood", "soil"]})
    gaussian = _linalg.Gaussian.independent(offsets=(mean, _linalg.DensePSD(jnp.asarray(covariance))))
    bare = FactorSpec(spec, law=gaussian)
    assert isinstance(bare.law, GaussianLaw)

    shifted = FactorSpec(
        spec,
        law=lambda shift: _linalg.Gaussian.independent(offsets=(mean + shift, _linalg.DensePSD(jnp.asarray(covariance)))),
    )
    shift = FactorSpec(ArraySpec("shift", units="1"), law=tfd.Normal(jnp.float64(0.0), jnp.float64(1.0)))

    theta = jax.random.normal(KEY, (4, 3))
    expected = st.multivariate_normal(np.asarray(mean), covariance).logpdf(np.asarray(theta))
    np.testing.assert_allclose(_posterior(bare).log_prior(theta), expected, rtol=1e-12)

    with_shift = jnp.concatenate([theta, jnp.full((4, 1), 0.5)], axis=1)
    expected = st.multivariate_normal(np.asarray(mean) + 0.5, covariance).logpdf(np.asarray(theta)) + st.norm.logpdf(0.5)
    np.testing.assert_allclose(_posterior(shifted, shift).log_prior(with_shift), expected, rtol=1e-12)

    draws = _posterior(bare).sample_prior(KEY, 20_000)
    np.testing.assert_allclose(np.cov(np.asarray(draws).T), covariance, atol=0.05)


def test_a_numpyro_dirichlet_matches_tfps_on_the_simplex():
    spec = ArraySpec("share", units="1", support=SIMPLEX, element_axes={"part": ["a", "b", "c"]})
    concentration = jnp.array([2.0, 3.0, 4.0])
    ours = _posterior(FactorSpec(spec, law=nd.Dirichlet(concentration)))
    theirs = _posterior(FactorSpec(spec, law=tfd.Dirichlet(concentration)))
    theta = jax.random.normal(KEY, (5, 2))
    np.testing.assert_allclose(ours.log_prior(theta), theirs.log_prior(theta), rtol=1e-12)

    per_site = ArraySpec("share", units="1", support=SIMPLEX, indexed_by=("site",), element_axes={"part": ["a", "b", "c"]})
    ours = _posterior(FactorSpec(per_site, law=lambda: nd.Dirichlet(concentration).expand([3]).to_event(1)),
                      coords={"site": SITES})
    theirs = _posterior(FactorSpec(per_site, law=iid_over_dim(tfd.Dirichlet(concentration))), coords={"site": SITES})
    theta = jax.random.normal(KEY, (5, 6))
    np.testing.assert_allclose(ours.log_prior(theta), theirs.log_prior(theta), rtol=1e-12)


@pytest.mark.parametrize(
    "law",
    [
        nd.Delta(1.0),
        nd.Delta(jnp.ones(2)).to_event(1),
        nd.MixtureSameFamily(nd.Categorical(jnp.array([0.5, 0.5])), nd.Delta(jnp.array([1.0, 2.0]))),
        nd.Poisson(3.0),
    ],
    ids=["point mass", "wrapped", "mixture", "discrete"],
)
def test_a_numpyro_law_with_no_density_is_refused(law):
    """A point mass is refused by name, inside a wrapper or a mixture too; a
    discrete law, which draws integers, by its dtype first."""
    axes = {"element_axes": {"n": law.event_shape[0]}} if law.event_shape else {}
    spec = ArraySpec("amount", units="1", support=POSITIVE, **axes)
    with pytest.raises(ValueError, match="Delta|int64"):
        joint(FactorSpec(spec, law=law)).bind(coords={})


def test_a_numpyro_mixture_of_listed_components_holding_a_point_mass_is_refused():
    law = nd.MixtureGeneral(nd.Categorical(jnp.array([0.5, 0.5])), [nd.Delta(0.0), nd.Delta(1.0)])
    with pytest.raises(ValueError, match="Delta"):
        joint(FactorSpec(ArraySpec("amount", units="1"), law=law)).bind(coords={})


def _switching(first, second):
    """A law function of a location returning *first()* at its first call and
    *second()* after, as a branch on the location would."""
    calls = []

    def law(location):
        calls.append(location)
        return first() if len(calls) == 1 else second()

    return law


@pytest.mark.parametrize(
    "law",
    [
        _switching(lambda: nd.LogNormal(0.0, 1.0), lambda: nd.Gamma(2.0, 1.0)),
        _switching(lambda: pushforward(nd.Normal(0.0, 1.0), bijector=tfb.Softplus()),
                   lambda: pushforward(nd.Normal(0.0, 1.0), bijector=tfb.Chain([tfb.Exp(), tfb.Shift(jnp.float64(0.1))]))),
    ],
    ids=["numpyro family", "pushforward bijector"],
)
def test_an_adapted_law_that_changes_its_structure_with_what_it_reads_is_refused(law):
    location = FactorSpec(ArraySpec("location", units="1"), law=tfd.Normal(jnp.float64(0.0), jnp.float64(1.0)))
    spec = ArraySpec("value", units="1", support=POSITIVE)
    with pytest.raises(ValueError, match="changes its structure"):
        joint(FactorSpec(spec, law=law), location).bind(coords={})


def test_numpyro_laws_are_told_apart_by_their_density():
    assert _numpyro.has_no_density(nd.Poisson(3.0)) and _numpyro.has_no_density(nd.LKJ(3, 2.0))
    assert not _numpyro.has_no_density(nd.Normal(0.0, 1.0)) and not _numpyro.has_no_density(nd.Dirichlet(jnp.ones(3)))
    assert _numpyro.is_dirichlet(nd.Dirichlet(jnp.ones(3))) and not _numpyro.is_dirichlet(nd.Normal(0.0, 1.0))
    assert not _numpyro.is_numpyro_distribution(nd.Normal) and not _numpyro.is_numpyro_distribution(tfd.Normal(0.0, 1.0))


def test_an_lkj_law_from_numpyro_is_refused():
    spec = ArraySpec("correlation", units="1", support=POSITIVE_DEFINITE, element_axes={"row": 3, "column": 3})
    with pytest.raises(ValueError, match="LKJ"):
        joint(FactorSpec(spec, law=nd.LKJ(3, 2.0))).bind(coords={})


def test_a_numpyro_law_with_a_batch_shape_is_refused():
    with pytest.raises(ValueError, match="batch shape"):
        joint(_field("capacity", lambda: nd.LogNormal(jnp.zeros(3), 1.0))).bind(coords={"site": SITES})


def test_the_builders_refuse_a_numpyro_law():
    with pytest.raises(TypeError, match="repeats TFP laws"):
        iid_over_dim(nd.Normal(0.0, 1.0))
    with pytest.raises(TypeError, match="repeats TFP laws"):
        joint(_field("capacity", iid_over_dim(lambda: nd.LogNormal(0.0, 1.0)))).bind(coords={"site": SITES})


# ── GPJax ─────────────────────────────────────────────────────────────────────


def test_a_gpjax_field_matches_tfps_gaussian_process():
    gpx = pytest.importorskip("gpjax")
    lx = pytest.importorskip("lineax")
    from gpjax.distributions import GaussianDistribution

    rng = np.random.default_rng(3)
    locations = xr.DataArray(rng.normal(size=(len(SITES), 2)) * 2e5, dims=("site", "coordinate"),
                             coords={"site": SITES})
    jitter = 1e-6 * jnp.eye(len(SITES))

    def gpjax_field(log_median, site_locations):
        kernel = gpx.kernels.Matern32(lengthscale=200e3, variance=0.4**2)
        gram = kernel.gram(site_locations).as_matrix() + jitter
        mean = jnp.broadcast_to(log_median, (gram.shape[0],))
        field = GaussianDistribution(loc=mean, scale=lx.MatrixLinearOperator(gram, lx.positive_semidefinite_tag))
        return pushforward(field, support=POSITIVE)

    def tfp_field(log_median, site_locations):
        kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=jnp.float64(0.4), length_scale=jnp.float64(200e3))
        gram = kernel.matrix(site_locations, site_locations) + jitter
        mean = jnp.broadcast_to(log_median, (gram.shape[0],))
        return pushforward(tfd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(gram)), support=POSITIVE)

    median = FactorSpec(ArraySpec("log_median", units="1"), law=tfd.Normal(jnp.float64(5.5), jnp.float64(0.5)))
    arguments = {"constants": {"site_locations": locations}, "own_dims": ("coordinate",)}
    ours = joint(_field("capacity", gpjax_field, **arguments), median).bind(coords={"site": SITES})
    theirs = joint(_field("capacity", tfp_field, **arguments), median).bind(coords={"site": SITES})
    assert ours.describe().loc["capacity", "evaluated_by"] == "base density"

    theta = jax.random.normal(KEY, (6, 4)) + jnp.array([5.5, 5.5, 5.5, 5.5])
    np.testing.assert_allclose(condition_on(ours, {}).log_prior(theta), condition_on(theirs, {}).log_prior(theta),
                               rtol=1e-9)
    draws = condition_on(ours, {}).sample_prior(KEY, 4000)
    field = np.asarray(draws[:, :3] - draws[:, 3:])
    np.testing.assert_allclose(np.cov(field.T), np.asarray(tfp_field(0.0, jnp.asarray(locations.values)).distribution
                                                          .covariance()), atol=0.02)
