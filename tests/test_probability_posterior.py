"""Tests for conditioning: the roles the graph gives each factor, and the
posterior's densities and forms.

Observing some factors makes the unobserved ones with no observed
descendant barren, splits the observed ones by whether a target factor is
among their ancestors, and leaves the rest as the target; the posterior's
prior, likelihood and constant are those of the design's Definition 3,
checked against SciPy and against the same model with a held value
declared an input.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability import (
    POSITIVE,
    ArraySpec,
    DeterministicSpec,
    FactorSpec,
    Posterior,
    condition_on,
    iid_over_dim,
    joint,
    log_normal,
    normal,
)

tfd = tfp.distributions

SITES = [10, 20, 30]
Y = np.array([0.5, -0.2, 1.0])
KEY = jax.random.key(11)


def _location():
    return FactorSpec(ArraySpec("location", units="1", indexed_by=("site",)),
                      law=iid_over_dim(normal(mean=0.0, standard_deviation=1.0)))


def _spread():
    return FactorSpec(ArraySpec("spread", units="1", support=POSITIVE), law=log_normal(median=1.0, geometric_sd=2.0))


def _source(name):
    return FactorSpec(ArraySpec(name, units="1", indexed_by=("site",)),
                      law=lambda location, spread: tfd.Independent(tfd.Normal(location, spread), 1))


def _model():
    shifted = DeterministicSpec(ArraySpec("shifted_validation", units="1", indexed_by=("site",)),
                                function=lambda validation: validation + 1.0)
    return joint(_source("y"), _source("validation"), shifted, _location(), _spread()).bind(coords={"site": SITES})


# ── the roles ─────────────────────────────────────────────────────────────────


def test_nothing_observed_is_the_prior_over_every_factor():
    posterior = condition_on(_model(), {})
    assert posterior.parameter_names == ("y", "validation", "location", "spread")
    assert posterior.barren_names == () and posterior.constant_names == () and posterior.observations is None
    assert posterior.y.shape == (0,) and posterior.dimension == 10
    theta = posterior.sample_prior(KEY, 3)
    np.testing.assert_array_equal(posterior.log_likelihood(theta), np.zeros(3))
    np.testing.assert_array_equal(posterior.log_density(theta), posterior.log_prior(theta))


def test_an_unobserved_source_is_barren_and_leaves_theta():
    posterior = condition_on(_model(), {"y": Y})
    assert posterior.parameter_names == ("location", "spread") and posterior.barren_names == ("validation",)
    assert posterior.observations.component_names == ("y",) and posterior.dimension == 4
    np.testing.assert_array_equal(posterior.y, Y)
    roles = posterior.describe()["role"]
    assert dict(roles) == {
        "y": "observed", "validation": "barren", "shifted_validation": "computed",
        "location": "parameter", "spread": "parameter",
    }


def test_the_likelihood_is_the_observed_factors_density():
    posterior = condition_on(_model(), {"y": Y})
    theta = posterior.sample_prior(KEY, 5)
    values = posterior.natural_values(theta)
    expected = st.norm(np.asarray(values["location"]), np.asarray(values["spread"])[:, None]).logpdf(Y).sum(-1)
    np.testing.assert_allclose(posterior.log_likelihood(theta), expected, rtol=1e-10)
    prior = (st.norm(0.0, 1.0).logpdf(np.asarray(values["location"])).sum(-1)
             + st.norm(0.0, np.log(2.0)).logpdf(np.asarray(theta[:, -1])))
    np.testing.assert_allclose(posterior.log_prior(theta), prior, rtol=1e-10)
    np.testing.assert_allclose(posterior.log_density(theta), prior + expected, rtol=1e-10)


def test_the_posterior_is_traceable():
    posterior = condition_on(_model(), {"y": Y})
    theta = posterior.sample_prior(KEY, 4)
    np.testing.assert_allclose(jax.jit(posterior.log_density)(theta), posterior.log_density(theta), rtol=1e-12)
    gradient = jax.grad(lambda t: posterior.log_density(t).sum())(theta)
    assert gradient.shape == theta.shape and bool(jnp.all(jnp.isfinite(gradient)))
    assert posterior.log_density(theta.reshape((2, 2, 4))).shape == (2, 2)


def test_an_observed_hyperparameter_with_no_target_ancestor_is_held():
    held = condition_on(_model(), {"y": Y, "spread": 1.5})
    assert held.parameter_names == ("location",) and held.constant_names == ("spread",)
    assert held.observations.component_names == ("y",)
    np.testing.assert_allclose(held.log_constant, st.lognorm(np.log(2.0), scale=1.0).logpdf(1.5), rtol=1e-12)
    theta = held.sample_prior(KEY, 4)
    expected = st.norm(np.asarray(theta), 1.5).logpdf(Y).sum(-1)
    np.testing.assert_allclose(held.log_likelihood(theta), expected, rtol=1e-10)


def test_holding_a_value_is_declaring_it_an_input():
    def hierarchy(*extra, inputs=()):
        return joint(
            FactorSpec(ArraySpec("location", units="1", indexed_by=("site",)),
                       law=iid_over_dim(lambda spread: tfd.Normal(jnp.float64(0.0), spread))),
            _source("y"), *extra, inputs=inputs,
        )

    observed = condition_on(hierarchy(_spread()).bind(coords={"site": SITES}), {"y": Y, "spread": 1.5})
    as_input = condition_on(
        hierarchy(inputs=[ArraySpec("spread", units="1", support=POSITIVE)]).bind(coords={"site": SITES},
                                                                                  inputs={"spread": 1.5}),
        {"y": Y},
    )
    theta = observed.sample_prior(KEY, 6)
    np.testing.assert_array_equal(theta, as_input.sample_prior(KEY, 6))
    np.testing.assert_array_equal(observed.log_prior(theta), as_input.log_prior(theta))
    np.testing.assert_array_equal(observed.log_likelihood(theta), as_input.log_likelihood(theta))
    assert as_input.log_constant == 0.0 and observed.log_constant != 0.0


def test_an_observed_factor_with_a_target_ancestor_is_in_the_likelihood():
    shared = FactorSpec(ArraySpec("rate", units="1", support=POSITIVE), law=log_normal(median=1.0, geometric_sd=2.0))
    spread = FactorSpec(ArraySpec("spread", units="1", support=POSITIVE),
                        law=lambda rate: tfd.LogNormal(jnp.log(rate), 0.3))
    model = joint(_source("y"), _location(), spread, shared).bind(coords={"site": SITES})
    posterior = condition_on(model, {"y": Y, "spread": 1.5})
    assert posterior.parameter_names == ("location", "rate") and posterior.constant_names == ()
    assert posterior.observations.component_names == ("y", "spread") and posterior.y.shape == (4,)
    theta = posterior.sample_prior(KEY, 3)
    rate = np.exp(np.asarray(theta[:, -1]))
    spread_density = st.lognorm(0.3, scale=rate).logpdf(1.5)
    y_density = st.norm(np.asarray(theta[:, :3]), 1.5).logpdf(Y).sum(-1)
    np.testing.assert_allclose(posterior.log_likelihood(theta), spread_density + y_density, rtol=1e-10)


# ── refusals ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("observed", "error", "message"),
    [
        ({"nothing": 1.0}, KeyError, "not a component a factor declares"),
        ({"shifted_validation": Y}, KeyError, "not a component a factor declares"),
        ({"y": np.array([0.5, np.nan, 1.0])}, ValueError, "not finite"),
        ({"spread": -1.0}, ValueError, "outside"),
        ({"y": Y[:2]}, ValueError, "block shape"),
        ({"y": Y, "validation": Y, "location": Y, "spread": 1.0}, ValueError, "nothing is left to infer"),
    ],
)
def test_condition_on_refuses_what_it_cannot_condition_on(observed, error, message):
    with pytest.raises(error, match=message):
        condition_on(_model(), observed)


def test_part_of_a_joint_factor_cannot_be_observed():
    pair = FactorSpec([ArraySpec("a", units="1"), ArraySpec("b", units="1")],
                      law=lambda: tfd.JointDistributionNamed({"a": tfd.Normal(jnp.float64(0.0), 1.0),
                                                              "b": tfd.Normal(jnp.float64(0.0), 1.0)}))
    model = joint(pair, _spread()).bind(coords={})
    with pytest.raises(ValueError, match="part of the factor"):
        condition_on(model, {"a": 0.0})


def test_an_input_cannot_be_observed():
    offset = ArraySpec("offset", units="1")
    model = joint(FactorSpec(ArraySpec("x", units="1"), law=lambda offset: tfd.Normal(offset, 1.0)),
                  inputs=[offset]).bind(coords={}, inputs={"offset": 0.0})
    with pytest.raises(KeyError, match="not a component a factor declares"):
        condition_on(model, {"offset": 1.0})


def test_a_target_with_no_density_at_a_held_value_is_refused():
    bound = FactorSpec(ArraySpec("bound", units="1", support=POSITIVE), law=log_normal(median=5.0, geometric_sd=1.2))
    shift = FactorSpec(ArraySpec("shift", units="1"), law=lambda bound: tfd.Normal(jnp.float64(0.0), bound - 1.0))
    reading = FactorSpec(ArraySpec("reading", units="1"), law=lambda shift: tfd.Normal(shift, jnp.float64(1.0)))
    model = joint(reading, shift, bound).bind(coords={})
    with pytest.raises(ValueError, match="no finite density at the observed values"):
        condition_on(model, {"bound": 0.3, "reading": 0.0})


def test_an_observed_value_is_read_by_label():
    labeled = xr.DataArray(Y[::-1], dims="site", coords={"site": SITES[::-1]})
    posterior = condition_on(_model(), {"y": labeled})
    np.testing.assert_array_equal(posterior.y, Y)
    assert list(posterior.observed["y"].indexes["site"]) == SITES


def test_condition_on_takes_a_bound_model():
    with pytest.raises(TypeError, match="FactoredDistribution"):
        condition_on(joint(_spread()), {})
    with pytest.raises(TypeError, match="mapping"):
        condition_on(_model(), [("y", Y)])


# ── forms ─────────────────────────────────────────────────────────────────────


def test_natural_values_include_what_is_computable():
    posterior = condition_on(_model(), {"y": Y, "validation": Y + 1.0})
    values = posterior.natural_values(posterior.sample_prior(KEY, 2))
    assert sorted(values) == ["location", "shifted_validation", "spread"]
    np.testing.assert_array_equal(values["shifted_validation"], np.broadcast_to(Y + 2.0, (2, 3)))
    barren = condition_on(_model(), {"y": Y})
    assert sorted(barren.natural_values(barren.sample_prior(KEY, 2))) == ["location", "spread"]


def test_to_labeled_holds_the_natural_values_and_theta():
    posterior = condition_on(_model(), {"y": Y})
    theta = posterior.sample_prior(KEY, 3)
    labeled = posterior.to_labeled(theta)
    assert sorted(labeled) == ["location", "spread", "theta"]
    assert labeled["location"].dims == ("sample", "site") and list(labeled["location"].indexes["site"]) == SITES
    assert labeled["theta"].dims == ("sample", "theta_entry")
    assert list(labeled["theta"].indexes["theta_entry"]) == list(posterior.parameters.unconstrained.entry_names)
    np.testing.assert_array_equal(posterior.parameters.labeled_to_flat(labeled),
                                  posterior.parameters.to_natural(theta))
    with pytest.raises(ValueError, match="batch"):
        posterior.to_labeled(theta[0])


def test_theta_must_end_in_the_dimension():
    posterior = condition_on(_model(), {"y": Y})
    with pytest.raises(ValueError):
        posterior.log_prior(np.zeros((2, 3)))


def test_a_posterior_is_frozen():
    posterior = condition_on(_model(), {})
    assert isinstance(posterior, Posterior)
    with pytest.raises(AttributeError, match="frozen"):
        posterior.y = None
