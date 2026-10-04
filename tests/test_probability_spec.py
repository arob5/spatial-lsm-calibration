"""Tests for ArraySpec, the declaration of one component.

A spec states each element axis once, by labels or by length; its
unconstrained spec follows the module's table, the positive-definite case
labeling theta's entries in FillTriangular's packing; and every check is
provoked once.
"""

from __future__ import annotations

import pickle

import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.parameters.parameter import Parameter
from sipnet_calibration.probability.spec import ArraySpec
from sipnet_calibration.probability.support import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    POSITIVE_DEFINITE,
    REAL,
    SIMPLEX,
    Interval,
    bijector_for,
)

tfb = tfp.bijectors

PARTS = ("leaf", "wood", "fine_root", "coarse_root")
YEARS = ("2012", "2013", "2014")


def test_a_scalar_is_the_default():
    spec = ArraySpec("respiration_share", units="1", support=OPEN_UNIT_INTERVAL)
    assert spec.shape == () and spec.indexed_by == () and dict(spec.element_axes) == {}
    assert isinstance(spec.bijector, tfb.Sigmoid) and spec.unconstrained_shape == ()


def test_an_element_axis_is_given_by_labels_or_by_length():
    by_labels = ArraySpec("allocation", units="1", support=SIMPLEX, indexed_by=["pft"],
                          element_axes={"allocation_part": PARTS})
    by_length = ArraySpec("loading", units=None, element_axes={"row": 2, "column": 3})
    assert by_labels.shape == (4,) and by_labels.indexed_by == ("pft",)
    assert by_labels.element_axes["allocation_part"].tolist() == list(PARTS)
    assert by_labels.element_axes["allocation_part"].name == "allocation_part"
    assert by_length.shape == (2, 3) and by_length.element_axes["column"].tolist() == ["0", "1", "2"]


def test_the_name_is_positional_only_and_the_rest_keyword_only():
    with pytest.raises(TypeError):
        ArraySpec(name="x", units=None)
    with pytest.raises(TypeError):
        ArraySpec("x", None)
    with pytest.raises(TypeError):
        ArraySpec("x")  # units are required


def test_a_spec_is_frozen_and_pickles():
    spec = ArraySpec("cov", units="Mg2 ha-2", support=POSITIVE_DEFINITE, element_axes={"year": YEARS, "other_year": YEARS})
    with pytest.raises(AttributeError):
        spec.units = "1"
    copy = pickle.loads(pickle.dumps(spec))
    assert copy.name == "cov" and copy.shape == (3, 3) and copy.support == POSITIVE_DEFINITE


def test_a_custom_bijector_survives_pickling():
    custom = tfb.Softplus()
    copy = pickle.loads(pickle.dumps(ArraySpec("rate", units="d-1", support=POSITIVE, bijector=custom)))
    assert isinstance(copy.bijector, tfb.Softplus)


# ── the unconstrained spec ────────────────────────────────────────────────────


def test_the_unconstrained_spec_is_real_unitless_and_keeps_the_dims():
    spec = ArraySpec("soil", units="kg m-2", support=POSITIVE, indexed_by=("site",))
    unconstrained = spec.unconstrained()
    assert (unconstrained.support, unconstrained.units, unconstrained.indexed_by) == (REAL, None, ("site",))
    assert isinstance(unconstrained.bijector, tfb.Identity)


def test_a_simplex_drops_its_last_label():
    spec = ArraySpec("allocation", units="1", support=SIMPLEX, element_axes={"allocation_part": PARTS})
    assert spec.unconstrained().element_axes["allocation_part"].tolist() == list(PARTS[:-1])


def test_a_positive_definite_value_labels_the_cholesky_entries_in_their_packing():
    spec = ArraySpec("cov", units="Mg2 ha-2", support=POSITIVE_DEFINITE, element_axes={"year": YEARS, "other_year": YEARS})
    axes = spec.unconstrained().element_axes
    assert list(axes) == ["year_other_year_cholesky"] and spec.unconstrained_shape == (6,)
    labels = axes["year_other_year_cholesky"].tolist()
    # Each label names the entry of L that theta's entry fills.
    filled = np.asarray(tfb.FillTriangular().forward(jnp.arange(6.0)))
    for k, label in enumerate(labels):
        i, j = (YEARS.index(y) for y in label[2:-1].split(","))
        assert i >= j and filled[i, j] == k
    assert sorted(labels) == sorted(f"L[{YEARS[i]},{YEARS[j]}]" for i in range(3) for j in range(i + 1))


def test_leading_axes_of_a_positive_definite_value_are_kept():
    spec = ArraySpec("covs", units="1", support=POSITIVE_DEFINITE,
                     element_axes={"part": ("a", "b"), "row": 2, "column": 2})
    assert list(spec.unconstrained().element_axes) == ["part", "row_column_cholesky"]
    assert spec.unconstrained_shape == (2, 3)


def test_another_bijector_keeps_the_axes_when_it_keeps_the_shape():
    spec = ArraySpec("rate", units="d-1", support=POSITIVE, element_axes={"pool": ("a", "b")}, bijector=tfb.Softplus())
    assert spec.unconstrained().element_axes["pool"].tolist() == ["a", "b"]


def test_another_bijector_that_changes_the_shape_takes_default_axes():
    spec = ArraySpec("allocation", units="1", support=SIMPLEX, element_axes={"part": 3},
                     bijector=tfb.IteratedSigmoidCentered())
    assert list(spec.unconstrained().element_axes) == ["allocation_axis_0"]


def test_the_unconstrained_spec_agrees_with_the_parameter_layers():
    """ArraySpec is Parameter with element_axes for shape and labels."""
    for support, shape, labels in [(POSITIVE, (), None), (SIMPLEX, (4,), {"part": PARTS}), (OPEN_UNIT_INTERVAL, (2,), {"k": ("a", "b")})]:
        parameter = Parameter(name="x", support=support, units="1", shape=shape, element_labels=labels)
        spec = ArraySpec("x", units="1", support=support, element_axes=labels or {})
        assert spec.shape == parameter.shape and spec.unconstrained_shape == parameter.unconstrained_shape
        theirs = parameter.unconstrained().element_labels
        assert [list(v) for v in spec.unconstrained().element_axes.values()] == [list(v) for v in theirs.values()]


# ── the checks ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["2x", "not an identifier", "lambda", "theta", "sample", "component", "element"])
def test_a_name_must_be_an_unreserved_identifier(name):
    with pytest.raises(ValueError, match="identifier|reserves"):
        ArraySpec(name, units=None)


def test_a_name_must_be_a_string():
    with pytest.raises(TypeError, match="string"):
        ArraySpec(3, units=None)


def test_units_must_be_a_string_or_none():
    with pytest.raises(TypeError, match="units"):
        ArraySpec("x", units=1)


def test_a_support_must_be_a_support():
    with pytest.raises(TypeError, match="Support"):
        ArraySpec("x", units=None, support=(0, 1))


@pytest.mark.parametrize(
    ("element_axes", "error", "match"),
    [
        ([("a", 2)], TypeError, "mapping"),
        ({1: 2}, TypeError, "strings"),
        ({"a": 0}, ValueError, "at least one"),
        ({"a": []}, ValueError, "no label"),
        ({"a": ("x", "x")}, ValueError, "more than once"),
        ({"a": (1, 2)}, TypeError, "strings"),
        ({"a": "xy"}, TypeError, "one string"),
        ({"x": 2}, ValueError, "like itself"),
        ({"site": 2}, ValueError, "like itself or a dim"),
        ({"sample": 2}, ValueError, "reserves"),
    ],
)
def test_element_axes_are_checked(element_axes, error, match):
    with pytest.raises(error, match=match):
        ArraySpec("x", units=None, indexed_by=("site",), element_axes=element_axes)


def test_dims_are_unique_and_unreserved():
    with pytest.raises(ValueError, match="more than once"):
        ArraySpec("x", units=None, indexed_by=("site", "site"))
    with pytest.raises(ValueError, match="reserves"):
        ArraySpec("x", units=None, indexed_by=("theta_entry",))
    with pytest.raises(TypeError, match="one string"):
        ArraySpec("x", units=None, indexed_by="site")


@pytest.mark.parametrize(
    ("support", "element_axes"),
    [
        (SIMPLEX, {}),
        (SIMPLEX, {"part": 1}),
        (POSITIVE_DEFINITE, {"year": 3}),
        (POSITIVE_DEFINITE, {"year": 3, "other_year": 2}),
    ],
)
def test_the_shape_has_the_supports_event_axes(support, element_axes):
    with pytest.raises(ValueError, match="decides membership"):
        ArraySpec("x", units=None, support=support, element_axes=element_axes)


def test_a_positive_definite_values_rows_and_columns_share_their_labels():
    with pytest.raises(ValueError, match="same labels"):
        ArraySpec("x", units=None, support=POSITIVE_DEFINITE, element_axes={"year": YEARS, "other_year": ("a", "b", "c")})


def test_a_custom_bijector_must_be_a_bijector():
    with pytest.raises(TypeError, match="TFP bijector"):
        ArraySpec("x", units=None, support=POSITIVE, bijector=np.exp)


def test_a_custom_bijector_must_map_onto_values_of_the_supports_rank():
    with pytest.raises(ValueError, match="maps onto"):
        ArraySpec("x", units=None, support=SIMPLEX, element_axes={"part": 3}, bijector=tfb.Exp())


def test_a_custom_bijector_must_map_onto_the_support():
    with pytest.raises(ValueError, match="does not map theta onto"):
        ArraySpec("x", units=None, support=Interval(0.0, 2.0), bijector=tfb.Sigmoid())


def test_the_default_bijector_given_explicitly_passes_the_check():
    """The ill-conditioned probes of a positive-definite value are left out
    of the round trip, so its own default passes."""
    for support, axes in [(POSITIVE, {}), (SIMPLEX, {"part": 3}), (POSITIVE_DEFINITE, {"year": YEARS, "other_year": YEARS})]:
        spec = ArraySpec("x", units=None, support=support, element_axes=axes, bijector=bijector_for(support))
        assert spec.unconstrained_shape == ArraySpec("x", units=None, support=support, element_axes=axes).unconstrained_shape
