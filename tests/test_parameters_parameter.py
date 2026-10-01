"""Tests for a parameter: what it describes, and its unconstrained
counterpart.

A value of any shape has element labels, the defaults filled in; its
unconstrained counterpart has the bijector's inverse shape, labels that
match it and the transform's name. A custom bijector is held to the
support. Every check is provoked once.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.parameters.parameter import Parameter
from sipnet_calibration.parameters.support import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    UNIT_INTERVAL,
)

tfb = tfp.bijectors

ALLOCATION = Parameter(
    name="allocation", support=SIMPLEX, units="1", shape=(4,),
    element_labels={"allocation_part": ("leaf", "wood", "fine_root", "coarse_root")},
)


# ── what it describes ─────────────────────────────────────────────────────────


def test_a_scalar_is_the_default():
    parameter = Parameter(name="rate", units="yr-1")
    assert (parameter.shape, parameter.indexed_by, dict(parameter.element_labels)) == ((), (), {})
    assert parameter.support == REAL and isinstance(parameter.bijector, tfb.Identity)


def test_default_element_labels_are_positional_strings():
    loading = Parameter(name="loading", units="1", shape=(2, 3))
    assert list(loading.element_labels) == ["loading_axis_0", "loading_axis_1"]
    assert list(loading.element_labels["loading_axis_1"]) == ["0", "1", "2"]


def test_the_bijector_is_stored_resolved():
    assert isinstance(Parameter(name="rate", support=POSITIVE, units="yr-1").bijector, tfb.Exp)
    assert isinstance(ALLOCATION.bijector, tfb.SoftmaxCentered)


def test_a_simplex_of_higher_rank_is_several_simplices():
    parameter = Parameter(name="shares", support=SIMPLEX, units="1", shape=(2, 3))
    assert parameter.unconstrained_shape == (2, 2)


# ── the unconstrained counterpart ─────────────────────────────────────────────


def test_the_unconstrained_counterpart_describes_theta():
    unconstrained = Parameter(name="rate", support=POSITIVE, units="yr-1", indexed_by=("site",)).unconstrained()
    assert (unconstrained.support, unconstrained.units, unconstrained.indexed_by) == (REAL, None, ("site",))
    assert isinstance(unconstrained.bijector, tfb.Identity)
    assert unconstrained.long_name == "log(rate)"


def test_the_simplex_drops_its_last_label():
    unconstrained = ALLOCATION.unconstrained()
    assert unconstrained.shape == (3,)
    assert list(unconstrained.element_labels["allocation_part"]) == ["leaf", "wood", "fine_root"]
    assert unconstrained.long_name == "alr(allocation)"


def test_the_long_name_names_the_transform():
    share = Parameter(name="share", support=OPEN_UNIT_INTERVAL, units="1", long_name="respiration share")
    assert share.unconstrained().long_name == "logit(respiration share)"
    assert Parameter(name="x", units=None, long_name="x value").unconstrained().long_name == "x value"


def test_a_bijector_changing_the_shape_takes_default_labels():
    parameter = Parameter(name="shares", support=SIMPLEX, units="1", shape=(3,),
                          bijector=tfb.IteratedSigmoidCentered())
    unconstrained = parameter.unconstrained()
    assert unconstrained.shape == (2,)
    assert list(unconstrained.element_labels["shares_axis_0"]) == ["0", "1"]
    assert unconstrained.long_name == "iterated_sigmoid(shares)"


# ── custom bijectors ──────────────────────────────────────────────────────────


def test_a_custom_bijector_onto_the_support_is_accepted():
    parameter = Parameter(name="rate", support=POSITIVE, units="yr-1", bijector=tfb.Softplus())
    assert isinstance(parameter.bijector, tfb.Softplus)
    np.testing.assert_allclose(parameter.bijector.forward(jnp.asarray(0.0)), np.log(2.0), rtol=1e-12)


def test_a_custom_bijector_not_onto_the_support_is_refused():
    with pytest.raises(ValueError, match="does not map theta onto the support"):
        Parameter(name="share", support=OPEN_UNIT_INTERVAL, units="1", bijector=tfb.Exp())


def test_a_custom_bijector_must_act_on_the_supports_events():
    with pytest.raises(ValueError, match="acts on 0-dimensional events"):
        Parameter(name="shares", support=SIMPLEX, units="1", shape=(3,), bijector=tfb.Exp())


def test_a_custom_bijector_is_a_bijector():
    with pytest.raises(TypeError, match="give a TFP bijector"):
        Parameter(name="rate", support=POSITIVE, units="yr-1", bijector=np.exp)


def test_a_closed_support_takes_the_bijector_onto_its_interior():
    parameter = Parameter(name="fraction", support=UNIT_INTERVAL, units="1")
    assert isinstance(parameter.bijector, tfb.Sigmoid)


# ── refusals ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"name": "base rate"}, ValueError, "not a Python identifier"),
        ({"name": "lambda"}, ValueError, "is a keyword"),
        ({"name": 3}, TypeError, "name is a string"),
        ({"units": 1.0}, TypeError, "give a string, or None"),
        ({"long_name": 2}, TypeError, "long_name"),
        ({"support": "positive"}, TypeError, "give a Support"),
        ({"shape": (0,)}, ValueError, "at least one number"),
        ({"shape": (2.0,)}, TypeError, "sequence of integers"),
        ({"shape": 4}, TypeError, "must be a sequence"),
        ({"indexed_by": "site"}, TypeError, "must be a sequence"),
        ({"indexed_by": ("site", "site")}, ValueError, "more than once"),
    ],
)
def test_a_malformed_argument_is_refused(arguments, error, match):
    with pytest.raises(error, match=match):
        Parameter(**{"name": "rate", "units": "yr-1", **arguments})


def test_element_labels_fit_the_shape():
    with pytest.raises(ValueError, match="axes of lengths"):
        Parameter(name="x", units="1", shape=(3,), element_labels={"part": ("a", "b")})
    with pytest.raises(ValueError, match="axes of lengths"):
        Parameter(name="x", units="1", shape=(2, 2), element_labels={"part": ("a", "b")})


def test_element_labels_are_unique_strings_on_named_axes():
    with pytest.raises(TypeError, match="must be strings"):
        Parameter(name="x", units="1", shape=(2,), element_labels={"part": (0, 1)})
    with pytest.raises(ValueError, match="more than once"):
        Parameter(name="x", units="1", shape=(2,), element_labels={"part": ("a", "a")})
    with pytest.raises(TypeError, match="axis names are strings"):
        Parameter(name="x", units="1", shape=(2,), element_labels={0: ("a", "b")})
    with pytest.raises(TypeError, match="give a mapping"):
        Parameter(name="x", units="1", shape=(2,), element_labels=[("a", "b")])


def test_an_axis_is_named_neither_for_the_value_nor_for_a_dim():
    with pytest.raises(ValueError, match="like itself or a dim"):
        Parameter(name="x", units="1", shape=(2,), element_labels={"x": ("a", "b")})
    with pytest.raises(ValueError, match="like itself or a dim"):
        Parameter(name="x", units="1", shape=(2,), indexed_by=("pft",), element_labels={"pft": ("a", "b")})


def test_the_shape_has_the_supports_event_axes():
    with pytest.raises(ValueError, match="at least two numbers"):
        Parameter(name="shares", support=SIMPLEX, units="1")
    with pytest.raises(ValueError, match="at least two numbers"):
        Parameter(name="shares", support=SIMPLEX, units="1", shape=(1,))
