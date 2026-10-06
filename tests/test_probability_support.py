"""Tests for the probability layer's supports and their default bijectors.

Each support contains what it declares and nothing else, its closure adds
its finite ends (for the positive-definite matrices, the semi-definite
ones), and its default bijector is the module's table, mapping the probe
points onto its interior. Every check is provoked once.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

import sipnet_calibration.probability.support as supports
from sipnet_calibration.probability._probes import (
    bijectors_agree,
    joint_probe_points,
    probe_points,
)
from sipnet_calibration.probability.support import (
    DEFAULT_BIJECTORS,
    NON_NEGATIVE,
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    POSITIVE_DEFINITE,
    REAL,
    SIMPLEX,
    UNIT_INTERVAL,
    Interval,
    PositiveDefinite,
    Simplex,
    Support,
    bijector_for,
)

tfb = tfp.bijectors


# ── intervals ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("support", "inside", "outside"),
    [
        (REAL, [-1e300, 0.0, 1e300], [np.inf, -np.inf, np.nan]),
        (POSITIVE, [1e-300, 1.0], [0.0, -1.0, np.inf]),
        (NON_NEGATIVE, [0.0, 1.0], [-1e-300, np.inf]),
        (OPEN_UNIT_INTERVAL, [1e-12, 0.5], [0.0, 1.0]),
        (UNIT_INTERVAL, [0.0, 1.0], [-1e-12, 1.0 + 1e-12]),
        (Interval(-2.0, math.inf, low_closed=True), [-2.0], [-2.0 - 1e-9]),
        (Interval(-math.inf, 3.0), [2.999], [3.0]),
    ],
)
def test_an_interval_contains_what_it_declares(support, inside, outside):
    assert bool(jnp.all(support.contains(inside)))
    assert not bool(jnp.any(support.contains(outside)))


def test_contains_keeps_the_shape_of_numbers():
    assert POSITIVE.contains(np.ones((3, 2))).shape == (3, 2)


def test_the_closure_adds_the_finite_ends_only():
    assert POSITIVE.closure() == NON_NEGATIVE
    assert OPEN_UNIT_INTERVAL.closure() == UNIT_INTERVAL
    assert REAL.closure() == REAL
    assert not bool(NON_NEGATIVE.closure().contains(np.inf))


def test_names_say_which_ends_are_closed():
    assert [s.name for s in (REAL, POSITIVE, NON_NEGATIVE, UNIT_INTERVAL)] == [
        "real", "(0, inf)", "[0, inf)", "[0, 1]",
    ]
    assert Interval(-math.inf, 2.5).name == "(-inf, 2.5)"


def test_supports_compare_by_value():
    assert Interval(0, math.inf) == POSITIVE
    assert isinstance(Interval(0.0, 1.0).low, float)


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ((1.0, 1.0), ValueError, "low < high"),
        ((2.0, 1.0), ValueError, "low < high"),
        ((math.nan, 1.0), ValueError, "low < high"),
        (("0", 1.0), TypeError, "real numbers"),
        ((True, 2.0), TypeError, "real numbers"),
    ],
)
def test_an_interval_needs_ordered_real_ends(arguments, error, match):
    with pytest.raises(error, match=match):
        Interval(*arguments)


def test_an_infinite_end_is_never_closed():
    with pytest.raises(ValueError, match="closes an infinite end"):
        Interval(0.0, math.inf, high_closed=True)


def test_the_closed_flags_are_booleans():
    with pytest.raises(TypeError, match="booleans"):
        Interval(0.0, 1.0, low_closed=1)


# ── the simplex ───────────────────────────────────────────────────────────────


def test_the_simplex_contains_positive_values_summing_to_one():
    values = np.array([[0.2, 0.8], [0.0, 1.0], [0.5, 0.6], [np.nan, 1.0]])
    assert SIMPLEX.contains(values).tolist() == [True, False, False, False]
    assert Simplex(closed=True).contains(values).tolist() == [True, True, False, False]
    assert SIMPLEX.closure() == Simplex(closed=True)
    assert SIMPLEX.contains(np.full((2, 3, 4), 0.25)).shape == (2, 3)


def test_event_ndims_say_what_membership_is_decided_over():
    assert (REAL.event_ndims, SIMPLEX.event_ndims) == (0, 1)
    assert isinstance(SIMPLEX, Support) and isinstance(REAL, Support)


# ── the default bijectors ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("support", "forward"),
    [
        (REAL, lambda t: t),
        (POSITIVE, np.exp),
        (NON_NEGATIVE, np.exp),
        (Interval(2.0, math.inf), lambda t: 2.0 + np.exp(t)),
        (Interval(-math.inf, 3.0), lambda t: 3.0 - np.exp(t)),
        (OPEN_UNIT_INTERVAL, lambda t: 1.0 / (1.0 + np.exp(-t))),
        (UNIT_INTERVAL, lambda t: 1.0 / (1.0 + np.exp(-t))),
        (Interval(-1.0, 4.0), lambda t: -1.0 + 5.0 / (1.0 + np.exp(-t))),
    ],
)
def test_the_default_bijector_is_the_tables(support, forward):
    theta = np.linspace(-5.0, 5.0, 11)
    np.testing.assert_allclose(bijector_for(support).forward(jnp.asarray(theta)), forward(theta), rtol=1e-12)


def test_the_simplex_bijector_inverts_the_additive_log_ratio():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    theta = bijector_for(SIMPLEX).inverse(jnp.asarray(x))
    np.testing.assert_allclose(theta, np.log(x[:-1] / x[-1]), rtol=1e-12)


@pytest.mark.parametrize("support", [REAL, POSITIVE, Interval(2.0, math.inf), Interval(-math.inf, 3.0),
                                     OPEN_UNIT_INTERVAL, Interval(-1.0, 4.0)])
def test_the_default_bijector_maps_the_probes_into_the_closure(support):
    images = bijector_for(support).forward(jnp.asarray(probe_points((3,))))
    assert bool(jnp.all(support.closure().contains(images)))


def test_a_support_without_an_entry_is_a_key_error():
    class Unknown(Support):
        event_ndims = 0
        name = "unknown"

        def contains(self, values):
            return jnp.ones(jnp.shape(values), dtype=bool)

        def closure(self):
            return self

    with pytest.raises(KeyError, match="no bijector is registered for a Unknown support"):
        bijector_for(Unknown())


def test_a_registry_is_read_by_type_then_by_base_class():
    class Shifted(Interval):
        pass

    assert isinstance(bijector_for(Shifted(0.0, math.inf)), tfb.Exp)
    registry = {**DEFAULT_BIJECTORS, Shifted: lambda support: tfb.Softplus()}
    assert isinstance(bijector_for(Shifted(0.0, math.inf), registry), tfb.Softplus)


def test_bijectors_are_compared_by_their_images():
    probes = probe_points(())
    assert bijectors_agree(tfb.Sigmoid(), tfb.Sigmoid(low=jnp.float64(0.0), high=jnp.float64(1.0)), probes)
    assert not bijectors_agree(tfb.Exp(), tfb.Softplus(), probes)


# ── probe points (private to the parameter layer) ──────────────────────────────────────────────────────────────


def test_probe_points_run_along_the_first_values_numbers():
    points = probe_points((2, 3), value_size=3)
    assert points.shape == (1 + 6 + 4 * 3 + 4, 2, 3)
    axis = points[7:19].reshape(12, 6)
    assert np.count_nonzero(axis, axis=1).tolist() == [1] * 12
    assert set(np.flatnonzero(axis.any(axis=0))) == {0, 1, 2}
    np.testing.assert_allclose(np.linalg.norm(points[-4:].reshape(4, -1), axis=1), 10.0)


def test_joint_probe_points_concatenate_the_parts():
    first, second = joint_probe_points([((2,), None), ((), None)])
    assert first.shape == (1 + 6 + 4 * 3 + 4, 2) and second.shape == (first.shape[0],)
    np.testing.assert_array_equal(probe_points((2,)), joint_probe_points([((2,), None)])[0])


# ── positive-definite matrices ────────────────────────────────────────────────

COVARIANCE = np.array([[4.0, 1.2, 0.0], [1.2, 2.0, 0.3], [0.0, 0.3, 1.0]])


@pytest.mark.parametrize(
    ("matrix", "inside"),
    [
        (COVARIANCE, True),
        (np.eye(1), True),
        (-COVARIANCE, False),
        (np.zeros((3, 3)), False),
        (np.array([[1.0, 2.0], [0.0, 1.0]]), False),
        (np.array([[1.0, 1.0], [1.0, 1.0]]), False),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), False),
        (np.array([[np.inf, 0.0], [0.0, 1.0]]), False),
    ],
)
def test_a_matrix_is_positive_definite_when_symmetric_and_factored(matrix, inside):
    assert bool(POSITIVE_DEFINITE.contains(matrix)) is inside


def test_symmetry_is_relative_to_the_largest_entry():
    """Rounding of L L^T is within tolerance at any scale; an asymmetry
    beyond it is refused at any scale."""
    for scale in (1e-8, 1.0, 1e8):
        rounded = scale * COVARIANCE + np.triu(np.full((3, 3), scale * 1e-13), 1)
        skewed = scale * COVARIANCE + np.triu(np.full((3, 3), scale * 1e-6), 1)
        assert bool(POSITIVE_DEFINITE.contains(rounded)) and not bool(POSITIVE_DEFINITE.contains(skewed))


def test_contains_is_decided_per_matrix_over_the_last_two_axes():
    stack = np.stack([COVARIANCE, -COVARIANCE, 2.0 * COVARIANCE]).reshape(3, 1, 3, 3)
    assert POSITIVE_DEFINITE.contains(stack).tolist() == [[True], [False], [True]]


def test_the_closure_is_the_positive_semi_definite_matrices():
    closure = POSITIVE_DEFINITE.closure()
    assert closure == PositiveDefinite(closed=True) and closure.name == "positive semi-definite"
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert bool(closure.contains(singular)) and bool(closure.contains(np.zeros((2, 2))))
    assert not bool(closure.contains(-np.eye(2)))
    assert closure.closure() == closure


def test_the_default_bijector_maps_the_probes_into_the_set():
    """The Cholesky factor with a log diagonal: theta of p(p+1)/2 numbers to
    a positive-definite matrix, and back."""
    bijector = bijector_for(POSITIVE_DEFINITE)
    assert tuple(bijector.inverse_event_shape((3, 3))) == (6,)
    probes = jnp.asarray(probe_points((6,)))
    matrices = bijector.forward(probes)
    assert matrices.shape == (len(probes), 3, 3)
    assert bool(jnp.all(POSITIVE_DEFINITE.closure().contains(matrices)))
    # -10 * 1 and -20 * 1 put e^-10 on the diagonal under off-diagonal -10:
    # positive definite in exact arithmetic, too ill-conditioned to invert.
    conditioned = (np.abs(probes).max(axis=-1) <= 3.0) | (np.count_nonzero(probes, axis=-1) == 1)
    assert bool(jnp.all(POSITIVE_DEFINITE.contains(matrices[conditioned])))
    back = bijector.inverse(jnp.array(np.asarray(matrices[conditioned])))
    assert np.allclose(back, probes[conditioned], atol=1e-8)


def test_theta_zero_is_the_identity_matrix():
    assert np.allclose(bijector_for(POSITIVE_DEFINITE).forward(jnp.zeros(6)), np.eye(3))


def test_the_support_is_registered_with_rank_two():
    assert POSITIVE_DEFINITE.event_ndims == 2 and POSITIVE_DEFINITE.name == "positive definite"
    assert PositiveDefinite in supports.DEFAULT_BIJECTORS


def test_the_closure_refuses_an_indefinite_matrix():
    closure = POSITIVE_DEFINITE.closure()
    assert not bool(closure.contains(np.diag([2.0, -1.0]))) and not bool(closure.contains(-COVARIANCE))


def test_the_symmetry_tolerance_is_one_in_ten_billion():
    for asymmetry, inside in [(1e-11, True), (1e-9, False)]:
        matrix = COVARIANCE + np.triu(np.full((3, 3), 4.0 * asymmetry), 1)
        assert bool(POSITIVE_DEFINITE.contains(matrix)) is inside
