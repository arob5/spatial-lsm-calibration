"""Tests for models: joint, binding, and the bound model's draws and density.

A model's graph is read off what its parts read, and refused when a name is
undeclared or the links cycle; binding reads constants, label maps and
inputs at the labels in use and checks each part at the probe points; and
the bound model draws ancestrally and evaluates its density against each
support's reference measure.
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
    POSITIVE_DEFINITE,
    SIMPLEX,
    ArraySpec,
    DeterministicSpec,
    FactorSpec,
    ModelSpec,
    iid_over_dim,
    independent_over_dim,
    joint,
    log_normal,
    normal,
    softmax_normal,
)

tfd, tfb = tfp.distributions, tfp.bijectors

SITES = [10, 20, 30]
KEY = jax.random.key(7)


def _location():
    return FactorSpec(ArraySpec("location", units="1", indexed_by=("site",)),
                      law=iid_over_dim(normal(mean=0.0, standard_deviation=1.0)))


def _spread():
    return FactorSpec(ArraySpec("spread", units="1", support=POSITIVE), law=log_normal(median=1.0, geometric_sd=2.0))


def _value():
    return FactorSpec(ArraySpec("value", units="1", indexed_by=("site",)),
                      law=lambda location, spread: tfd.Independent(tfd.Normal(location, spread), 1))


def _model(**coords):
    return joint(_value(), _location(), _spread()).bind(coords={"site": SITES, **coords})


# ── joint ─────────────────────────────────────────────────────────────────────


def test_joint_flattens_models_and_keeps_declaration_order():
    offset = ArraySpec("offset", units="1", indexed_by=("site",))
    inner = joint(_location(), _spread(), inputs=[offset])
    spec = joint(_value(), inner)
    assert isinstance(spec, ModelSpec)
    assert spec.component_names == ("value", "location", "spread") and spec.input_names == ("offset",)
    assert spec.component_spec("offset") is offset and spec.component_spec("spread").support is POSITIVE
    with pytest.raises(KeyError, match="no component or input"):
        spec.component_spec("nothing")


def test_describe_has_a_row_per_part():
    table = joint(_value(), _location(), _spread()).describe()
    assert list(table.index) == ["value", "location", "spread"]
    assert table.loc["value", "given"] == "location, spread" and table.loc["location", "indexed_by"] == "site"
    assert table.loc["location", "law"] == "iid Normal"


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        ((), "no part"),
        ((DeterministicSpec(ArraySpec("v", units="1"), function=lambda spread: spread),), "deterministics alone"),
        ((_spread(), _spread()), "more than once"),
        ((_value(), _location()), "no part declares"),
        ((FactorSpec(ArraySpec("spread", units="1", support=POSITIVE),
                     law=lambda location: log_normal(median=1.0, geometric_sd=2.0)),
          FactorSpec(ArraySpec("location", units="1"), law=lambda spread: tfd.Normal(0.0, spread))), "cycle"),
    ],
    ids=["empty", "no factor", "twice", "undeclared", "cycle"],
)
def test_joint_refuses_a_malformed_graph(parts, message):
    with pytest.raises(ValueError, match=message):
        joint(*parts)


def test_a_constant_named_like_a_component_is_refused():
    location = xr.DataArray([0.0, 1.0, 2.0], dims="site", coords={"site": SITES})
    value = FactorSpec(ArraySpec("value", units="1", indexed_by=("site",)),
                       law=lambda location: tfd.Independent(tfd.Normal(location, 1.0), 1), constants={"location": location})
    with pytest.raises(ValueError, match="named like components"):
        joint(value, _location())


def test_an_element_axis_means_one_set_of_labels():
    first = FactorSpec(ArraySpec("a", units="1", support=SIMPLEX, element_axes={"part": ("x", "y")}),
                       law=softmax_normal(center=[0.5, 0.5], logit_sd=1.0))
    second = FactorSpec(ArraySpec("b", units="1", support=SIMPLEX, element_axes={"part": ("x", "z")}),
                        law=softmax_normal(center=[0.5, 0.5], logit_sd=1.0))
    with pytest.raises(ValueError, match="one set of labels"):
        joint(first, second)


def test_joint_takes_parts_only():
    with pytest.raises(TypeError, match="part"):
        joint(tfd.Normal(0.0, 1.0))
    with pytest.raises(TypeError, match="ArraySpec"):
        joint(_spread(), inputs=["offset"])


# ── binding ───────────────────────────────────────────────────────────────────


def test_bind_keeps_the_dims_in_use_and_needs_each():
    model = _model(pft=["a", "b"])
    assert list(model.coords) == ["site"] and model.block_shape("value") == (3,)
    with pytest.raises(KeyError, match="no labels for"):
        joint(_value(), _location(), _spread()).bind(coords={})


def _input_model():
    offset = ArraySpec("offset", units="1", support=POSITIVE, indexed_by=("site",))
    shifted = DeterministicSpec(ArraySpec("shifted", units="1", indexed_by=("site",)), function=lambda offset: offset + 1.0)
    value = FactorSpec(ArraySpec("value", units="1", indexed_by=("site",)),
                       law=lambda shifted: tfd.Independent(tfd.Normal(shifted, 1.0), 1))
    return joint(value, shifted, inputs=[offset])


def test_an_input_is_read_by_label_or_as_a_block():
    labeled = xr.DataArray([3.0, 1.0, 2.0, 9.0], dims="site", coords={"site": [30, 10, 20, 99]})
    model = _input_model().bind(coords={"site": SITES}, inputs={"offset": labeled})
    np.testing.assert_array_equal(model.inputs["offset"].values, [1.0, 2.0, 3.0])
    assert list(model.inputs["offset"].indexes["site"]) == SITES
    same = _input_model().bind(coords={"site": SITES}, inputs={"offset": np.array([1.0, 2.0, 3.0])})
    draws = model.sample(KEY, 5)
    np.testing.assert_array_equal(draws["shifted"], np.broadcast_to([2.0, 3.0, 4.0], (5, 3)))
    np.testing.assert_array_equal(draws["value"], same.sample(KEY, 5)["value"])


@pytest.mark.parametrize(
    ("inputs", "error", "message"),
    [
        ({}, ValueError, "have no value"),
        ({"offset": np.ones(3), "other": 1.0}, KeyError, "not inputs"),
        ({"offset": np.array([1.0, np.nan, 1.0])}, ValueError, "not finite"),
        ({"offset": np.array([1.0, -1.0, 1.0])}, ValueError, "outside"),
        ({"offset": np.ones(2)}, ValueError, "block shape"),
        ({"offset": np.array([True, True, True])}, TypeError, "not a number"),
        ({"offset": xr.DataArray(["1", "2", "3"], dims="site", coords={"site": SITES})}, TypeError, "not a number"),
        ({"offset": xr.DataArray([1.0, 2.0], dims="site", coords={"site": [10, 20]})}, KeyError, "no value at"),
        ({"offset": xr.DataArray(np.ones((3, 2)), dims=("site", "x"), coords={"site": SITES})}, ValueError, "on those dims"),
    ],
)
def test_bind_refuses_a_bad_input(inputs, error, message):
    with pytest.raises(error, match=message):
        _input_model().bind(coords={"site": SITES}, inputs=inputs)


def test_a_constant_is_read_at_the_labels_in_use():
    median = xr.DataArray([5.0, 1.0, 2.0, 3.0], dims="site", coords={"site": [99, 10, 20, 30]})
    spec = FactorSpec(ArraySpec("carbon", units="1", support=POSITIVE, indexed_by=("site",)),
                      law=independent_over_dim(log_normal, geometric_sd=2.0), constants={"median": median})
    model = joint(spec).bind(coords={"site": SITES})
    law = model.law("carbon", given={})
    np.testing.assert_allclose(np.exp(law.distribution.distribution.loc), [1.0, 2.0, 3.0])
    with pytest.raises(KeyError, match="no value at"):
        joint(spec).bind(coords={"site": [10, 40]})


@pytest.mark.parametrize(
    ("part", "message"),
    [
        (FactorSpec(ArraySpec("carbon", units="1", indexed_by=("site",)),
                    law=lambda: tfd.Normal(jnp.zeros(2), 1.0)), "event shape"),
        (FactorSpec(ArraySpec("carbon", units="1", support=POSITIVE), law=tfd.Normal(jnp.float64(0.0), 1.0)),
         "outside its declared support"),
        (FactorSpec(ArraySpec("carbon", units="1"), law=tfd.HalfNormal(jnp.float64(1.0))),
         "no density at some values"),
        (FactorSpec(ArraySpec("carbon", units="1"), law=tfd.Normal(0.0, 1.0)), "not float64"),
    ],
    ids=["shape", "mass outside", "smaller support", "float32"],
)
def test_bind_checks_each_factor_at_the_probe_points(part, message):
    with pytest.raises(ValueError, match=message):
        joint(part).bind(coords={"site": SITES})


@pytest.mark.parametrize(
    ("function", "support", "message"),
    [
        (lambda location: location[:2], None, "of shape"),
        (lambda location: location, POSITIVE, "outside its declared support"),
    ],
)
def test_bind_checks_each_deterministic_at_the_probe_points(function, support, message):
    arguments = {} if support is None else {"support": support}
    computed = DeterministicSpec(ArraySpec("computed", units="1", indexed_by=("site",), **arguments), function=function)
    with pytest.raises(ValueError, match=message):
        joint(_location(), computed).bind(coords={"site": SITES})


# ── evaluation ────────────────────────────────────────────────────────────────


def test_sample_draws_every_component_ancestrally():
    draws = _model().sample(KEY, 4)
    assert {name: value.shape for name, value in draws.items()} == {"value": (4, 3), "location": (4, 3), "spread": (4,)}
    assert bool(jnp.all(draws["spread"] > 0))


def test_a_factors_draws_depend_on_its_name_not_its_place():
    reordered = joint(_spread(), _location(), _value()).bind(coords={"site": SITES})
    first, second = _model().sample(KEY, 4), reordered.sample(KEY, 4)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_sample_draws_only_what_the_names_asked_for_need():
    draws = _model().sample(KEY, 4, component_names=["location"])
    assert list(draws) == ["location"]
    np.testing.assert_array_equal(draws["location"], _model().sample(KEY, 4)["location"])
    with pytest.raises(KeyError, match="no component"):
        _model().sample(KEY, 4, component_names=["nothing"])


def test_log_prob_is_the_sum_of_the_factors_densities():
    model = _model()
    draws = model.sample(KEY, 5)
    location, spread, value = (np.asarray(draws[n]) for n in ("location", "spread", "value"))
    expected = (
        st.norm(0.0, 1.0).logpdf(location).sum(-1)
        + st.lognorm(np.log(2.0), scale=1.0).logpdf(spread)
        + st.norm(location, spread[:, None]).logpdf(value).sum(-1)
    )
    np.testing.assert_allclose(model.log_prob(draws), expected, rtol=1e-10)
    np.testing.assert_allclose(jax.jit(model.log_prob)(draws), expected, rtol=1e-10)


def test_log_prob_on_the_simplex_is_against_the_first_coordinates():
    law = softmax_normal(center=[0.2, 0.3, 0.5], logit_sd=[0.4, 0.6])
    model = joint(FactorSpec(ArraySpec("share", units="1", support=SIMPLEX, element_axes={"part": 3}), law=law)).bind(coords={})
    x = model.sample(KEY, 4)
    # TFP's density is against the simplex's surface measure, 0.5 log k below.
    np.testing.assert_allclose(model.log_prob(x), law.log_prob(x["share"]) + 0.5 * np.log(3.0), rtol=1e-10)


def test_log_prob_takes_every_event_and_nothing_computed():
    model = _input_model().bind(coords={"site": SITES}, inputs={"offset": np.ones(3)})
    with pytest.raises(KeyError, match="lacks"):
        model.log_prob({})
    with pytest.raises(ValueError, match="computed or bound"):
        model.log_prob({"value": np.zeros(3), "shifted": np.zeros(3)})
    with pytest.raises(ValueError, match="block shape"):
        model.log_prob({"value": np.zeros(2)})


def test_law_is_a_factors_law_at_one_draw():
    model = _model()
    law = model.law("value", given={"location": np.zeros(3), "spread": 2.0})
    np.testing.assert_allclose(law.log_prob(np.zeros(3)), 3 * st.norm(0.0, 2.0).logpdf(0.0))
    with pytest.raises(KeyError, match="reads"):
        model.law("value", given={"location": np.zeros(3)})
    with pytest.raises(KeyError, match="no component"):
        model.law("nothing", given={})
    computed = _input_model().bind(coords={"site": SITES}, inputs={"offset": np.ones(3)})
    with pytest.raises(ValueError, match="has no law"):
        computed.law("shifted", given={})


def test_select_binds_again_at_fewer_labels():
    model = _input_model().bind(coords={"site": SITES}, inputs={"offset": np.array([1.0, 2.0, 3.0])})
    selected = model.select(site=[30, 10])
    assert list(selected.coords["site"]) == [10, 30]
    np.testing.assert_array_equal(selected.inputs["offset"].values, [1.0, 3.0])
    whole = model.log_prob({"value": np.array([0.0, 0.0, 0.0])})
    part = selected.log_prob({"value": np.array([0.0, 0.0])})
    np.testing.assert_allclose(whole - part, st.norm(3.0, 1.0).logpdf(0.0))
    with pytest.raises(KeyError, match="dims and levels in use"):
        model.select(component=["value"])
    with pytest.raises(KeyError, match="dims and levels in use"):
        model.select(pft=["a"])


def test_describe_says_how_each_factor_is_evaluated():
    table = _model().describe()
    assert table.loc["spread", "evaluated_by"] == "base density"
    assert table.loc["value", "evaluated_by"] == "change of variables"
    assert table.loc["location", "block_shapes"] == "location: (3,)"


def test_a_bound_model_is_frozen():
    model = _model()
    with pytest.raises(AttributeError, match="frozen"):
        model.coords = {}


def test_log_prob_is_minus_infinity_outside_a_support():
    model = joint(_spread()).bind(coords={})
    np.testing.assert_array_equal(model.log_prob({"spread": jnp.array([-1.0, 0.0])}), [-np.inf, -np.inf])


def test_a_chain_of_deterministics_varies_by_draw():
    twice = DeterministicSpec(ArraySpec("twice", units="1"), function=lambda spread: 2.0 * spread)
    again = DeterministicSpec(ArraySpec("again", units="1"), function=lambda twice: twice + 1.0)
    reading = FactorSpec(ArraySpec("reading", units="1"), law=lambda again: tfd.Normal(jnp.float64(0.0), again))
    model = joint(reading, again, twice, _spread()).bind(coords={})
    draws = model.sample(KEY, 2000)
    np.testing.assert_allclose(draws["again"], 2.0 * draws["spread"] + 1.0)
    standardized = np.asarray(draws["reading"] / draws["again"])
    assert abs(standardized.std() - 1.0) < 0.05


def test_a_transformed_law_whose_base_is_not_thetas_block_is_a_change_of_variables():
    base = tfd.Independent(tfd.Normal(jnp.zeros((1, 2)), jnp.float64(1.0)), 2)
    law = tfd.TransformedDistribution(base, tfb.Chain([tfb.Exp(), tfb.Reshape([2], [1, 2])]))
    model = joint(FactorSpec(ArraySpec("pair", units="1", support=POSITIVE, element_axes={"k": 2}), law=law)).bind(coords={})
    assert model.describe().loc["pair", "evaluated_by"] == "change of variables"
    x = jnp.array([[0.5, 2.0]])
    np.testing.assert_allclose(model.log_prob({"pair": x}), st.lognorm(1.0).logpdf(np.asarray(x)).sum(-1), rtol=1e-10)


_RATE = jnp.float64(3.0)


@pytest.mark.parametrize(
    "part",
    [
        FactorSpec(ArraySpec("correlation", units="1", support=POSITIVE_DEFINITE,
                             element_axes={"row": ("a", "b"), "column": ("a", "b")}),
                   law=tfd.LKJ(2, jnp.float64(2.0))),
        FactorSpec(ArraySpec("count", units="1", support=POSITIVE), law=tfd.Poisson(_RATE)),
        FactorSpec(ArraySpec("point", units="1"), law=tfd.Deterministic(jnp.float64(1.0))),
        FactorSpec(ArraySpec("counts", units="1", support=POSITIVE, indexed_by=("site",)),
                   law=iid_over_dim(tfd.Poisson(_RATE))),
        FactorSpec(ArraySpec("shifted", units="1"),
                   law=tfd.TransformedDistribution(tfd.Poisson(_RATE), tfb.Shift(jnp.float64(0.5)))),
        FactorSpec([ArraySpec("a", units="1"), ArraySpec("b", units="1", support=POSITIVE)],
                   law=lambda: tfd.JointDistributionNamed({"a": tfd.Normal(jnp.float64(0.0), 1.0),
                                                           "b": tfd.Poisson(_RATE)})),
    ],
    ids=["LKJ", "Poisson", "point mass", "iid Poisson", "shifted Poisson", "joint with a Poisson"],
)
def test_a_law_with_no_density_on_its_support_is_refused(part):
    with pytest.raises(ValueError, match="has no density against the reference measure"):
        joint(part).bind(coords={"site": SITES})


def test_a_mixture_of_densities_is_a_density():
    mixture = tfd.MixtureSameFamily(tfd.Categorical(probs=jnp.array([0.3, 0.7])),
                                    tfd.Normal(jnp.array([-1.0, 2.0]), jnp.float64(1.0)))
    model = joint(FactorSpec(ArraySpec("mixed", units="1"), law=mixture)).bind(coords={})
    assert model.describe().loc["mixed", "evaluated_by"] == "change of variables"
