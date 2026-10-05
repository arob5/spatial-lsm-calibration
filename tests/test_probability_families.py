"""Tests for the families and laws: each family's density against SciPy's,
and the pushforward.

The families are one value's law from a few numbers; their densities are
compared with SciPy's at draws, the simplex's against the density of its
additive log-ratios, and the three conjugate families' arguments are
checked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability import (
    POSITIVE,
    SIMPLEX,
    Interval,
    InverseWishart,
    as_law,
    inverse_gamma,
    inverse_wishart,
    log_normal,
    log_normal_from_interval,
    logit_normal,
    logit_normal_from_interval,
    normal,
    pushforward,
    softmax_normal,
)

tfd, tfb = tfp.distributions, tfp.bijectors

KEY = jax.random.key(20261004)


def _draws(law, n=6):
    return np.asarray(law.sample(n, seed=KEY))


def test_normal_is_scipys():
    law = normal(mean=1.5, standard_deviation=0.4)
    x = _draws(law)
    np.testing.assert_allclose(law.log_prob(x), st.norm(1.5, 0.4).logpdf(x), rtol=1e-12)
    assert type(law) is tfd.Normal


def test_inverse_gamma_is_scipys():
    law = inverse_gamma(shape=3.0, scale=2.0)
    x = _draws(law)
    np.testing.assert_allclose(law.log_prob(x), st.invgamma(3.0, scale=2.0).logpdf(x), rtol=1e-12)
    np.testing.assert_allclose(law.mean(), 2.0 / (3.0 - 1.0))
    assert type(law) is tfd.InverseGamma


def test_inverse_wishart_is_scipys():
    scale = 64.0 * (0.4 * np.eye(3) + 0.6 * np.ones((3, 3)))
    law = inverse_wishart(degrees_of_freedom=7.0, scale=scale)
    x = _draws(law)
    np.testing.assert_allclose(law.log_prob(x), st.invwishart(df=7.0, scale=scale).logpdf(np.moveaxis(x, 0, -1)),
                               rtol=1e-10)
    np.testing.assert_allclose(np.mean(_draws(law, 20_000), axis=0), scale / (7.0 - 3.0 - 1.0), rtol=0.1)
    assert isinstance(law, InverseWishart) and law.degrees_of_freedom == 7.0
    np.testing.assert_array_equal(law.scale, scale)


def test_inverse_wishart_is_traceable():
    scale = np.eye(2)
    x = _draws(inverse_wishart(degrees_of_freedom=4.0, scale=scale))
    traced = jax.jit(lambda s, v: InverseWishart(4.0, s).log_prob(v))(scale, x)
    np.testing.assert_allclose(traced, st.invwishart(df=4.0, scale=scale).logpdf(np.moveaxis(x, 0, -1)), rtol=1e-10)


def test_log_normal_is_scipys():
    law = log_normal_from_interval(lower=1.3, upper=3.2)
    x = _draws(law)
    sigma = (np.log(3.2) - np.log(1.3)) / (2 * st.norm.ppf(0.975))
    np.testing.assert_allclose(law.log_prob(x), st.lognorm(sigma, scale=np.sqrt(1.3 * 3.2)).logpdf(x), rtol=1e-10)
    np.testing.assert_allclose(log_normal(median=2.0, geometric_sd=1.5).log_prob(x),
                               st.lognorm(np.log(1.5), scale=2.0).logpdf(x), rtol=1e-12)


@pytest.mark.parametrize("support", [Interval(0.0, 1.0), Interval(-2.0, 5.0)])
def test_logit_normal_is_the_normal_of_its_logit(support):
    law = logit_normal(median=0.6, logit_sd=0.7, support=support) if support.low == 0 else \
        logit_normal_from_interval(lower=-1.0, upper=4.0, support=support)
    x = _draws(law)
    fraction = (x - support.low) / (support.high - support.low)
    t = np.log(fraction) - np.log1p(-fraction)
    base = law.distribution if hasattr(law, "distribution") else None
    loc, scale = (float(base.loc), float(base.scale))
    jacobian = np.log(support.high - support.low) + np.log(fraction) + np.log1p(-fraction)
    np.testing.assert_allclose(law.log_prob(x), st.norm(loc, scale).logpdf(t) - jacobian, rtol=1e-9)


def test_softmax_normal_draws_follow_its_log_ratios():
    center, logit_sd = np.array([0.2, 0.3, 0.5]), np.array([0.4, 0.6])
    x = _draws(softmax_normal(center=center, logit_sd=logit_sd), 20_000)
    ratios = np.log(x[:, :-1] / x[:, -1:])
    np.testing.assert_allclose(ratios.mean(axis=0), np.log(center[:-1] / center[-1]), atol=0.02)
    np.testing.assert_allclose(ratios.std(axis=0), logit_sd, rtol=0.03)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: normal(mean=0.0, standard_deviation=0.0), "standard_deviation"),
        (lambda: inverse_gamma(shape=-1.0, scale=1.0), "shape"),
        (lambda: inverse_gamma(shape=1.0, scale=0.0), "scale"),
        (lambda: inverse_wishart(degrees_of_freedom=1.5, scale=np.eye(3)), "p - 1"),
        (lambda: inverse_wishart(degrees_of_freedom=5.0, scale=np.array([[1.0, 2.0], [2.0, 1.0]])), "positive definite"),
        (lambda: inverse_wishart(degrees_of_freedom=5.0, scale=np.ones(3)), r"\(p, p\)"),
    ],
)
def test_the_conjugate_families_refuse_bad_arguments(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_pushforward_is_an_exact_transformed_distribution():
    base = tfd.Normal(jnp.float64(0.0), jnp.float64(1.0))
    law = pushforward(base, support=POSITIVE)
    assert type(law) is tfd.TransformedDistribution and isinstance(law.bijector, tfb.Exp)
    assert isinstance(pushforward(base, bijector=tfb.Softplus()).bijector, tfb.Softplus)
    simplex = pushforward(tfd.MultivariateNormalDiag(jnp.zeros(2), jnp.ones(2)), support=SIMPLEX)
    assert isinstance(simplex.bijector, tfb.SoftmaxCentered)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "exactly one"),
        ({"support": POSITIVE, "bijector": tfb.Exp()}, "exactly one"),
        ({"bijector": "exp"}, "TFP bijector"),
    ],
)
def test_pushforward_names_its_map_once(arguments, message):
    with pytest.raises(TypeError, match=message):
        pushforward(tfd.Normal(0.0, 1.0), **arguments)


def test_pushforward_takes_a_law():
    with pytest.raises(TypeError, match="give a law of the unconstrained values"):
        pushforward(object(), support=POSITIVE)


def test_as_law_takes_tfp_laws_and_objects_implementing_the_protocol():
    normal_law = tfd.Normal(0.0, 1.0)
    assert as_law(normal_law) is normal_law

    class Uniform:
        def log_prob(self, value):
            return jnp.zeros_like(value)

        def sample(self, sample_shape=(), seed=None):
            return jax.random.uniform(seed, sample_shape, dtype=jnp.float64)

    uniform = Uniform()
    assert as_law(uniform) is uniform
    with pytest.raises(TypeError, match="not a law"):
        as_law(lambda x: x)
