"""Tests for the probability layer's supports: the positive-definite matrices
it adds, and the parameter layer's re-export of the rest.

The interval and simplex supports are tested through the parameter layer's
re-export, in ``test_parameters_support.py``, unchanged by their move.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

import sipnet_calibration.parameters.support as parameter_supports
import sipnet_calibration.probability.support as supports
from sipnet_calibration.probability._probes import probe_points
from sipnet_calibration.probability.support import (
    POSITIVE_DEFINITE,
    PositiveDefinite,
    bijector_for,
)

tfb = tfp.bijectors

COVARIANCE = np.array([[4.0, 1.2, 0.0], [1.2, 2.0, 0.3], [0.0, 0.3, 1.0]])


def test_the_parameter_layer_re_exports_the_same_objects():
    """A support built by either layer is one class, so the parameter
    layer's isinstance checks accept the probability layer's supports."""
    for name in parameter_supports.__all__:
        assert getattr(parameter_supports, name) is getattr(supports, name)


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
