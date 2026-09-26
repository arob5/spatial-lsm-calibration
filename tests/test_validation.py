"""Tests for the shared argument coercion in ``sipnet_calibration.validation``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sipnet_calibration.validation import (
    FrozenMapping,
    as_batched_flat,
    as_bbox,
    as_frozen_mapping,
    as_integer,
    as_names,
    as_positive_integer,
    as_site_id,
    as_site_ids,
    truncated,
)


class TestAsSiteIds:
    def test_keeps_the_order_given_as_plain_integers(self):
        site_ids = as_site_ids([27, np.int32(1), 4711.0, np.array(3)])
        assert site_ids == (27, 1, 4711, 3)
        assert all(type(site_id) is int for site_id in site_ids)

    def test_reads_a_generator_once(self):
        assert as_site_ids(site for site in (3, 1)) == (3, 1)

    @pytest.mark.parametrize("sites", ["27", b"27", 27, np.array(27), None])
    def test_one_string_or_one_value_is_a_type_error(self, sites):
        with pytest.raises(TypeError, match="sequence of site ids"):
            as_site_ids(sites)

    def test_one_id_is_accepted_when_asked_for(self):
        assert as_site_ids(27, allow_one_id=True) == (27,)
        assert as_site_ids(np.int64(27), allow_one_id=True) == (27,)
        with pytest.raises(TypeError, match="one character per site"):
            as_site_ids("27", allow_one_id=True)

    @pytest.mark.parametrize("site", [True, np.bool_(False), "1", None, 1 + 0j])
    def test_a_boolean_or_a_non_number_is_a_type_error(self, site):
        with pytest.raises(TypeError, match="must be an integer"):
            as_site_ids([site])

    @pytest.mark.parametrize("site", [1.5, np.nan, np.inf, 0, -3, 2**31])
    def test_a_fraction_or_an_id_out_of_range_is_a_value_error(self, site):
        with pytest.raises(ValueError, match="each of sites"):
            as_site_ids([site])

    def test_a_repeated_id_is_refused(self):
        with pytest.raises(ValueError, match=r"site\(s\) \[1\] more than once"):
            as_site_ids([1, 27, 1.0])

    def test_the_message_names_the_argument(self):
        with pytest.raises(TypeError, match="^ids must be"):
            as_site_ids("1", message_name="ids")

    def test_a_two_dimensional_array_is_refused(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            as_site_ids(np.array([[1, 2]]))


class TestAsSiteId:
    def test_accepts_integers_and_whole_floats(self):
        assert as_site_id(np.int16(27)) == 27
        assert as_site_id(27.0) == 27

    def test_refuses_a_boolean_and_a_fraction_by_type_and_value(self):
        with pytest.raises(TypeError):
            as_site_id(True)
        with pytest.raises(ValueError, match="whole number"):
            as_site_id(27.5)


class TestIntegers:
    def test_as_integer_accepts_python_and_numpy_integers(self):
        assert as_integer(np.int64(-2), message_name="n") == -2

    @pytest.mark.parametrize("value", [True, 2.0, "2", None])
    def test_as_integer_refuses_a_boolean_a_float_and_a_string(self, value):
        with pytest.raises(TypeError, match="^n must be an integer"):
            as_integer(value, message_name="n")

    def test_as_positive_integer_refuses_zero_as_a_value_error(self):
        assert as_positive_integer(3, message_name="ncol") == 3
        with pytest.raises(ValueError, match="ncol must be a positive integer"):
            as_positive_integer(0, message_name="ncol")
        with pytest.raises(TypeError):
            as_positive_integer(True, message_name="ncol")


class TestAsBatchedFlat:
    def test_one_vector_becomes_one_row(self):
        batched, was_one_vector = as_batched_flat([1, 2, 3], 3, message_name="theta")
        assert batched.shape == (1, 3) and batched.dtype == np.float64
        assert was_one_vector

    def test_a_batch_is_returned_as_given(self):
        batched, was_one_vector = as_batched_flat(np.zeros((4, 3)), 3, message_name="theta")
        assert batched.shape == (4, 3) and not was_one_vector

    @pytest.mark.parametrize("shape", [(2,), (1, 2, 3), (), (4, 2)])
    def test_the_wrong_shape_is_a_value_error(self, shape):
        with pytest.raises(ValueError, match="theta must be one vector of 3 entries"):
            as_batched_flat(np.zeros(shape), 3, message_name="theta")

    def test_no_rows_and_non_finite_entries_are_refused_only_when_asked(self):
        assert as_batched_flat(np.zeros((0, 3)), 3, message_name="y")[0].shape == (0, 3)
        with pytest.raises(ValueError, match="at least one vector"):
            as_batched_flat(np.zeros((0, 3)), 3, message_name="y", allow_no_rows=False)
        assert np.isnan(as_batched_flat([np.nan, 0, 0], 3, message_name="y")[0]).any()
        with pytest.raises(ValueError, match="non-finite"):
            as_batched_flat([np.nan, 0, 0], 3, message_name="y", allow_non_finite=False)

    def test_values_that_are_not_numbers_are_a_type_error(self):
        with pytest.raises(TypeError, match="array of numbers"):
            as_batched_flat(["a", "b"], 2, message_name="theta")

    def test_a_jax_array_stays_a_jax_array_and_can_be_traced(self):
        batched, _ = as_batched_flat(jnp.ones(3), 3, message_name="theta")
        assert isinstance(batched, jax.Array)

        def total(theta):
            batched, _ = as_batched_flat(theta, 3, message_name="theta", allow_non_finite=False)
            return batched.sum()

        assert float(jax.jit(total)(jnp.ones(3))) == 3.0
        np.testing.assert_allclose(jax.grad(total)(jnp.ones(3)), np.ones(3))


class TestAsBbox:
    def test_four_numbers_become_four_floats(self):
        assert as_bbox((-125, 24, np.float32(-66), 50)) == (-125.0, 24.0, -66.0, 50.0)

    @pytest.mark.parametrize(
        "bbox", [None, "abcd", {"w": 1, "s": 2, "e": 3, "n": 4}, ("-125", "24", "-66", "50")]
    )
    def test_the_wrong_type_is_a_type_error(self, bbox):
        with pytest.raises(TypeError):
            as_bbox(bbox)

    @pytest.mark.parametrize(
        ("bbox", "match"),
        [
            ((1, 2, 3), "3 value"),
            ((-66, 24, -125, 50), "antimeridian"),
            ((-125, 50, -66, 24), "north of north"),
            ((-125, 24, -66, np.nan), "finite"),
        ],
    )
    def test_the_wrong_value_is_a_value_error(self, bbox, match):
        with pytest.raises(ValueError, match=match):
            as_bbox(bbox)


class TestAsNames:
    def test_keeps_the_order_given(self):
        assert as_names(["b", "a"], message_name="names") == ("b", "a")

    @pytest.mark.parametrize("names", ["nee", {"nee"}, frozenset({"nee"}), 3, ["nee", 3]])
    def test_one_string_a_set_or_a_non_string_is_a_type_error(self, names):
        with pytest.raises(TypeError, match="names"):
            as_names(names, message_name="names")


class TestFrozenMapping:
    def test_is_a_mapping_that_cannot_change(self):
        frozen = as_frozen_mapping({"b": 1, "a": 2}, message_name="value")
        assert list(frozen) == ["b", "a"] and frozen["a"] == 2 and len(frozen) == 2
        assert frozen == {"b": 1, "a": 2}
        with pytest.raises(TypeError):
            frozen["a"] = 3
        with pytest.raises(AttributeError):
            frozen._items = {}

    def test_pickles_and_hashes(self):
        import pickle

        frozen = FrozenMapping({1: "conifer", 2: "grass"})
        assert pickle.loads(pickle.dumps(frozen)) == frozen
        assert hash(frozen) == hash(FrozenMapping({2: "grass", 1: "conifer"}))

    def test_copies_what_it_is_given(self):
        source = {"a": 1}
        frozen = as_frozen_mapping(source, message_name="value")
        source["a"] = 2
        assert frozen["a"] == 1
        assert as_frozen_mapping(frozen, message_name="value") is frozen

    def test_refuses_what_is_not_a_mapping(self):
        with pytest.raises(TypeError, match="value must be a mapping"):
            as_frozen_mapping([("a", 1)], message_name="value")


def test_truncated_shows_at_most_the_limit_and_counts_the_rest():
    assert truncated([1, 2, 3]) == "[1, 2, 3]"
    assert truncated(range(12)) == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and 2 more"
    assert truncated("abc", limit=2) == "['a', 'b'] and 1 more"
