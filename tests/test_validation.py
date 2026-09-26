"""Tests for the shared argument coercion in ``sipnet_calibration.validation``."""

from __future__ import annotations

import copy
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sipnet_calibration.conventions import (
    LAT_ATTRIBUTES,
    LON_ATTRIBUTES,
    SITE_ATTRIBUTES,
    FrozenMapping,
)
from sipnet_calibration.validation import (
    as_batched_flat,
    as_bbox,
    as_bounded_integer,
    as_frozen_mapping,
    as_integer,
    as_names,
    as_positive_integer,
    as_positive_integers,
    as_site_id,
    as_site_ids,
    check_integers_are_in_range,
    check_site_ids_are_in_range,
    is_one_vector,
    range_summary,
    truncated,
)


class TestAsSiteIds:
    def test_keeps_the_order_given_as_plain_integers(self):
        site_ids = as_site_ids([27, np.int32(1), 4711, np.int64(3)], message_name="sites")
        assert site_ids == (27, 1, 4711, 3)
        assert all(type(site_id) is int for site_id in site_ids)

    def test_reads_a_generator_once(self):
        assert as_site_ids((site for site in (3, 1)), message_name="sites") == (3, 1)

    @pytest.mark.parametrize(
        "sites",
        [
            np.array([27, 1]),
            jnp.array([27, 1]),
            xr.DataArray([27, 1], dims="site"),
            pd.Series([27, 1]),
            pd.Index([27, 1]),
            pd.array([27, 1], dtype="Int64"),
            {27: "a", 1: "b"}.keys(),
        ],
        ids=["numpy", "jax", "xarray", "series", "index", "nullable", "keys"],
    )
    def test_array_likes_and_ordered_views_are_read_in_order(self, sites):
        assert as_site_ids(sites, message_name="sites") == (27, 1)

    @pytest.mark.parametrize(
        "sites", ["27", b"27", 27, np.int64(27), np.array(27), jnp.array(27), None]
    )
    def test_one_string_or_one_value_is_a_type_error(self, sites):
        with pytest.raises(TypeError, match="sequence of site ids.*such as"):
            as_site_ids(sites, message_name="sites")

    @pytest.mark.parametrize("sites", [{27, 1}, frozenset({27})])
    def test_a_set_is_a_type_error(self, sites):
        with pytest.raises(TypeError, match="no order to keep"):
            as_site_ids(sites, message_name="sites")

    @pytest.mark.parametrize("site", [True, np.bool_(False), "1", None, 1 + 0j])
    def test_a_boolean_or_a_non_number_is_a_type_error(self, site):
        with pytest.raises(TypeError, match=r"^sites\[0\] must be an integer"):
            as_site_ids([site], message_name="sites")

    @pytest.mark.parametrize("site", [27.0, 1.5, np.float32(3), np.nan, np.inf])
    def test_a_float_is_a_type_error_even_a_whole_one(self, site):
        with pytest.raises(TypeError, match="cast it with int"):
            as_site_ids([site], message_name="sites")

    def test_a_float_array_is_a_type_error(self):
        with pytest.raises(TypeError, match="float"):
            as_site_ids(np.array([1.0, 27.0]), message_name="sites")

    @pytest.mark.parametrize("site", [0, -3, 2**31])
    def test_an_id_out_of_range_is_a_value_error(self, site):
        with pytest.raises(ValueError, match=r"^sites\[0\] must be a site id from 1 to"):
            as_site_ids([site], message_name="sites")

    def test_a_repeated_id_is_refused(self):
        with pytest.raises(ValueError, match=r"site\(s\) \[1\] more than once"):
            as_site_ids([1, 27, 1], message_name="sites")

    def test_the_message_names_the_argument(self):
        with pytest.raises(TypeError, match="^ids must be"):
            as_site_ids("1", message_name="ids")

    def test_a_two_dimensional_array_is_refused(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            as_site_ids(np.array([[1, 2]]), message_name="sites")

    def test_every_message_names_a_fix(self):
        for bad in ("27", [1.5], [0], [1, 1], {1}):
            with pytest.raises((TypeError, ValueError)) as caught:
                as_site_ids(bad, message_name="sites")
            message = str(caught.value)
            assert "; " in message and ".." not in message and "each of" not in message


class TestAsSiteId:
    @pytest.mark.parametrize(
        "site",
        [27, np.int16(27), np.array(27), jnp.array(27), xr.DataArray(27)],
        ids=["int", "numpy", "0-d numpy", "0-d jax", "0-d xarray"],
    )
    def test_accepts_integers_and_zero_dimensional_arrays_of_one(self, site):
        assert as_site_id(site, message_name="site") == 27

    def test_refuses_a_boolean_and_a_float_by_type(self):
        with pytest.raises(TypeError, match="boolean"):
            as_site_id(True, message_name="site")
        with pytest.raises(TypeError, match="float"):
            as_site_id(27.0, message_name="site")


class TestIntegers:
    def test_as_integer_accepts_python_and_numpy_integers(self):
        assert as_integer(np.int64(-2), message_name="n") == -2

    @pytest.mark.parametrize("value", [True, 2.0, "2", None])
    def test_as_integer_refuses_a_boolean_a_float_and_a_string(self, value):
        with pytest.raises(TypeError, match="^n must be an integer.*; "):
            as_integer(value, message_name="n")

    def test_as_positive_integer_refuses_zero_as_a_value_error(self):
        assert as_positive_integer(3, message_name="ncol") == 3
        with pytest.raises(ValueError, match="ncol must be at least 1, got 0; "):
            as_positive_integer(0, message_name="ncol")
        with pytest.raises(TypeError):
            as_positive_integer(True, message_name="ncol")

    def test_as_bounded_integer_holds_both_ends(self):
        assert as_bounded_integer(0, minimum=0, maximum=2, message_name="k") == 0
        assert as_bounded_integer(2, minimum=0, maximum=2, message_name="k") == 2
        with pytest.raises(ValueError, match="from 0 to 2, got 3"):
            as_bounded_integer(3, minimum=0, maximum=2, message_name="k")

    def test_as_positive_integers_keeps_order_and_repeats(self):
        assert as_positive_integers(np.array([2, 1, 2]), message_name="members") == (2, 1, 2)
        with pytest.raises(TypeError, match="sequence of integers"):
            as_positive_integers(2, message_name="members")
        with pytest.raises(TypeError, match=r"members\[0\]"):
            as_positive_integers([2.0], message_name="members")


class TestAsBatchedFlat:
    def test_one_vector_becomes_one_row(self):
        batched = as_batched_flat([1, 2, 3], 3, message_name="theta")
        assert batched.shape == (1, 3) and batched.dtype == np.float64

    def test_a_batch_is_returned_as_given(self):
        assert as_batched_flat(np.zeros((4, 3)), 3, message_name="theta").shape == (4, 3)

    def test_no_rows_and_non_finite_entries_are_the_callers_to_refuse(self):
        assert as_batched_flat(np.zeros((0, 3)), 3, message_name="y").shape == (0, 3)
        assert np.isnan(as_batched_flat([np.nan, 0, 0], 3, message_name="y")).any()

    @pytest.mark.parametrize("shape", [(2,), (1, 2, 3), (), (4, 2)])
    def test_the_wrong_shape_is_a_value_error(self, shape):
        with pytest.raises(ValueError, match="theta must be one vector of 3 entries"):
            as_batched_flat(np.zeros(shape), 3, message_name="theta")

    @pytest.mark.parametrize(
        "values",
        [["a", "b"], [None, None], [True, False], [1j, 2], [[1, 2], [3]]],
        ids=["strings", "none", "booleans", "complex", "ragged"],
    )
    def test_values_that_are_not_real_numbers_are_a_type_error(self, values):
        with pytest.raises(TypeError, match="^theta must"):
            as_batched_flat(values, 2, message_name="theta")

    def test_a_complex_jax_array_is_a_type_error(self):
        with pytest.raises(TypeError, match="real numbers"):
            as_batched_flat(jnp.array([1j, 2]), 2, message_name="theta")

    @pytest.mark.parametrize("values", [np.ones(3, np.float32), jnp.ones(3, jnp.float32)])
    def test_float32_comes_back_float64_for_numpy_and_jax_alike(self, values):
        assert as_batched_flat(values, 3, message_name="x").dtype == np.float64

    def test_a_jax_array_stays_a_jax_array_and_can_be_traced(self):
        assert isinstance(as_batched_flat(jnp.ones(3), 3, message_name="theta"), jax.Array)

        def total(theta):
            return as_batched_flat(theta, 3, message_name="theta").sum()

        assert float(jax.jit(total)(jnp.ones(3))) == 3.0
        np.testing.assert_allclose(jax.grad(total)(jnp.ones(3)), np.ones(3))

    def test_a_list_of_tracers_is_accepted_under_jit(self):
        def total(theta):
            return as_batched_flat([theta[i] for i in range(3)], 3, message_name="theta").sum()

        assert float(jax.jit(total)(jnp.arange(3.0))) == 3.0


class TestIsOneVector:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([1.0, 2.0], True),
            ([[1.0, 2.0]], False),
            (np.zeros(3), True),
            (np.zeros((2, 3)), False),
            (jnp.zeros(3), True),
            ([jnp.asarray(1.0), jnp.asarray(2.0)], True),
            ([jnp.zeros(2), jnp.zeros(2)], False),
        ],
        ids=["list", "nested list", "numpy", "numpy batch", "jax", "jax scalars", "jax rows"],
    )
    def test_reads_the_rank(self, values, expected):
        assert is_one_vector(values) is expected

    def test_reads_a_list_of_tracers_under_jit(self):
        def rank_one(theta):
            return jnp.where(is_one_vector([theta[i] for i in range(3)]), 1.0, 0.0)

        assert float(jax.jit(rank_one)(jnp.arange(3.0))) == 1.0


class TestAsBbox:
    @pytest.mark.parametrize(
        "bbox",
        [
            (-125, 24, np.float32(-66), 50),
            np.array([-125, 24, -66, 50]),
            jnp.array([-125.0, 24, -66, 50]),
        ],
        ids=["tuple", "numpy", "jax"],
    )
    def test_four_numbers_become_four_floats(self, bbox):
        assert as_bbox(bbox, message_name="bbox") == (-125.0, 24.0, -66.0, 50.0)

    @pytest.mark.parametrize(
        "bbox",
        [
            None,
            "abcd",
            {"w": 1, "s": 2, "e": 3, "n": 4},
            {1, 2, 3, 4},
            (value for value in (1, 2, 3, 4)),
            ("-125", "24", "-66", "50"),
        ],
        ids=["none", "string", "mapping", "set", "generator", "strings"],
    )
    def test_the_wrong_type_is_a_type_error(self, bbox):
        with pytest.raises(TypeError):
            as_bbox(bbox, message_name="bbox")

    @pytest.mark.parametrize(
        "bbox", [(True, 0, 1, 1), (0, np.bool_(False), 1, 1), np.array([True, False, True, True])]
    )
    def test_a_boolean_is_a_type_error(self, bbox):
        with pytest.raises(TypeError, match="numbers"):
            as_bbox(bbox, message_name="bbox")

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
            as_bbox(bbox, message_name="bbox")

    def test_the_message_knows_nothing_of_the_site_pool(self):
        with pytest.raises(ValueError) as caught:
            as_bbox((-66, 24, -125, 50), message_name="bbox")
        assert "pool" not in str(caught.value)


class TestAsNames:
    def test_keeps_the_order_given(self):
        assert as_names(["b", "a"], message_name="names") == ("b", "a")

    @pytest.mark.parametrize(
        "names", [{"b": 1, "a": 2}.keys(), np.array(["b", "a"]), pd.Index(["b", "a"])]
    )
    def test_an_ordered_view_or_an_array_is_read_in_order(self, names):
        assert as_names(names, message_name="names") == ("b", "a")

    @pytest.mark.parametrize("names", ["nee", {"nee"}, frozenset({"nee"}), 3, ["nee", 3]])
    def test_one_string_a_set_or_a_non_string_is_a_type_error(self, names):
        with pytest.raises(TypeError, match="names"):
            as_names(names, message_name="names")


class TestFrozenMapping:
    def test_is_a_dict_that_cannot_change(self):
        frozen = as_frozen_mapping({"b": 1, "a": 2}, message_name="value")
        assert isinstance(frozen, dict)
        assert list(frozen) == ["b", "a"] and frozen["a"] == 2 and len(frozen) == 2
        assert frozen == {"b": 1, "a": 2}
        for change in (
            lambda: frozen.__setitem__("a", 3),
            lambda: frozen.__delitem__("a"),
            lambda: frozen.update(a=3),
            lambda: frozen.pop("a"),
            lambda: frozen.popitem(),
            lambda: frozen.setdefault("c", 3),
            lambda: frozen.clear(),
        ):
            with pytest.raises(TypeError, match="cannot be changed"):
                change()
        with pytest.raises(AttributeError):
            frozen.extra = {}
        assert frozen == {"b": 1, "a": 2}

    def test_pandas_and_json_read_it_as_a_dict(self):
        frozen = FrozenMapping({"a": (1, 2), "b": (3, 4)})
        assert pd.DataFrame(frozen).shape == (2, 2)
        assert json.loads(json.dumps(FrozenMapping({"a": 1}))) == {"a": 1}

    def test_pickles_copies_and_hashes(self):
        frozen = FrozenMapping({1: "conifer", 2: "grass"})
        for copied in (
            pickle.loads(pickle.dumps(frozen)),
            copy.copy(frozen),
            copy.deepcopy(frozen),
        ):
            assert copied == frozen and isinstance(copied, FrozenMapping)
        assert hash(frozen) == hash(FrozenMapping({2: "grass", 1: "conifer"}))
        assert type(frozen.copy()) is dict

    def test_copies_what_it_is_given(self):
        source = {"a": 1}
        frozen = as_frozen_mapping(source, message_name="value")
        source["a"] = 2
        assert frozen["a"] == 1
        assert as_frozen_mapping(frozen, message_name="value") is frozen

    def test_refuses_what_is_not_a_mapping(self):
        with pytest.raises(TypeError, match="value must be a mapping"):
            as_frozen_mapping([("a", 1)], message_name="value")

    @pytest.mark.parametrize("attributes", [SITE_ATTRIBUTES, LON_ATTRIBUTES, LAT_ATTRIBUTES])
    def test_the_conventions_attribute_dicts_are_read_only(self, attributes):
        with pytest.raises(TypeError):
            attributes["units"] = "m"

    def test_xarray_takes_an_attribute_dict_as_it_is_and_copies_it(self):
        array = xr.DataArray([1], dims="site", attrs=SITE_ATTRIBUTES)
        array.attrs["extra"] = 1
        assert "extra" not in SITE_ATTRIBUTES


class TestChecks:
    def test_site_ids_in_range(self):
        check_site_ids_are_in_range(np.array([1, 2**31 - 1]), message_name="ids")
        with pytest.raises(ValueError, match=r"got \[0\]"):
            check_site_ids_are_in_range(np.array([0, 5]), message_name="ids")

    def test_integers_in_range(self):
        check_integers_are_in_range(np.array([1, 3]), minimum=1, maximum=3, message_name="k")
        with pytest.raises(ValueError, match="from 1 to 3"):
            check_integers_are_in_range(np.array([4]), minimum=1, maximum=3, message_name="k")


def test_truncated_shows_at_most_the_limit_and_counts_the_rest():
    assert truncated([1, 2, 3]) == "[1, 2, 3]"
    assert truncated(range(10)) == repr(list(range(10)))
    assert truncated(range(11)).endswith("and 1 more")
    assert truncated(range(12)) == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and 2 more"
    assert truncated("abc", limit=2) == "['a', 'b'] and 1 more"


def test_range_summary_gives_four_columns_or_dashes():
    assert range_summary(np.array([])).split() == ["-", "-", "-", "-"]
    assert range_summary(np.array([-1.0, 2.0, 3.0])).split() == ["-1", "2", "3", "1"]
