"""The probe points: fixed points of unconstrained space at which bijectors,
deterministic values and laws are checked, and the comparison of two
bijectors there. Private to the probability layer, and read by the
parameter layer through its re-export.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

__all__ = ["CORNERS", "bijectors_agree", "corner_points", "joint_probe_points", "probe_points"]

tfb = tfp.bijectors

#: The values of each unconstrained number at the corner points: ``+-12``
#: spans ten orders of magnitude on a log scale and reaches ``1 - 6e-6`` on
#: a logit scale, while staying inside float64.
CORNERS: tuple[float, ...] = (-12.0, 0.0, 12.0)

#: The seed of the probe points' random directions.
_PROBE_SEED = 20260926

#: The most unconstrained numbers per value whose every corner is a corner
#: point: ``3^6 = 729`` points per value; a value with more takes the axis
#: points instead.
_MOST_CORNER_NUMBERS = 6


def bijectors_agree(first: tfb.Bijector, second: tfb.Bijector, probes: Any) -> bool:
    """Whether two bijectors map *probes* alike, to a relative tolerance of
    ``1e-10``.

    Images are compared, never bijectors: ``tfb.Sigmoid()`` and
    ``tfb.Sigmoid(low=0., high=1.)`` compare unequal. Each bijector is given
    its own copy of *probes*, since TFP caches a bijector's pairs.
    """
    probes = jnp.asarray(probes)
    return bool(np.allclose(first.forward(probes), second.forward(jnp.array(probes)), rtol=1e-10, atol=0.0))


def probe_points(shape: tuple[int, ...], *, value_size: int | None = None) -> np.ndarray:
    """The fixed points of unconstrained space at which bijectors, derived
    values and laws are probed.

    For an array of shape *shape* (one value, or a component's block), whose
    first *value_size* numbers in C order are its first value: :math:`\\theta = 0`; :math:`\\pm c \\mathbf 1` for
    :math:`c \\in \\{3, 10, 20\\}`; :math:`\\pm c\\, e_i` along each number
    :math:`i` of the first value, for :math:`c \\in \\{10, 20\\}`; and four
    fixed-seed random directions of norm 10.

    Parameters
    ----------
    shape:
        The shape of one point.
    value_size:
        The numbers of one value, along which the axis probes run; every
        number of the point when ``None``.

    Returns
    -------
    numpy.ndarray
        ``float64``, ``(n_probes, *shape)``.

    Notes
    -----
    They stop at 20: :math:`e^{20} \\approx 4.9 \\times 10^8` is beyond any
    plausible magnitude, and the logistic at 20 is within
    :math:`2 \\times 10^{-9}` of its bound, while it rounds to exactly 1 only
    near 37. A probe is not a proof: a support that differs only beyond the
    probes passes.
    """
    return joint_probe_points([(shape, value_size)])[0]


def corner_points(places: Sequence[np.ndarray], size: int) -> np.ndarray:
    """The corner points of a flat vector of *size* entries, ``(n, size)``.

    Each of *places* is one parameter's entries, ``int64`` of shape
    ``(n_values, e)``: row ``i`` the positions of the ``e`` unconstrained
    numbers of its value at label tuple ``i``. Each parameter in turn takes
    every combination of :data:`CORNERS` over the ``e`` numbers of one
    value, the same at every label, the others 0; a value of more than
    :data:`_MOST_CORNER_NUMBERS` numbers takes instead the ``2e + 3`` points
    0, ``+-c * 1`` and ``+-c e_i``, ``c`` the outer corner.
    """
    rows = []
    for place in places:
        place = np.asarray(place, dtype=np.int64)
        e = place.shape[1]
        if e <= _MOST_CORNER_NUMBERS:
            corners = itertools.product(CORNERS, repeat=e)
        else:
            outer = max(abs(c) for c in CORNERS)
            corners = [np.zeros(e), np.full(e, outer), np.full(e, -outer)]
            corners += [sign * outer * np.eye(e)[i] for i in range(e) for sign in (1.0, -1.0)]
        for corner in corners:
            row = np.zeros(size)
            row[place] = np.asarray(corner, dtype=np.float64)
            rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape((len(rows), size))


def joint_probe_points(parts: Sequence[tuple[tuple[int, ...], int | None]]) -> list[np.ndarray]:
    """:func:`probe_points` over several arrays at once.

    Each part is ``(shape, value_size)``, as :func:`probe_points` takes. The
    points are :func:`probe_points`' in the space of every part
    concatenated: :math:`\\theta = 0`, :math:`\\pm c \\mathbf 1`,
    :math:`\\pm c\\, e_i` along the first value's numbers of each part in
    turn, and four fixed-seed random directions of norm 10.

    Returns
    -------
    list of numpy.ndarray
        One per part, ``(n_probes, *shape)``, ``float64``, with one
        ``n_probes`` for all.
    """
    shapes = [tuple(shape) for shape, _ in parts]
    sizes = [math.prod(shape) for shape in shapes]
    total = sum(sizes)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    points = [np.zeros(total)]
    points += [sign * c * np.ones(total) for c in (3.0, 10.0, 20.0) for sign in (1.0, -1.0)]
    for (_, value_size), size, offset in zip(parts, sizes, offsets):
        for i in range(size if value_size is None else min(value_size, size)):
            for c in (10.0, 20.0):
                for sign in (1.0, -1.0):
                    point = np.zeros(total)
                    point[offset + i] = sign * c
                    points.append(point)
    directions = np.random.default_rng(_PROBE_SEED).standard_normal((4, total))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    points += list(10.0 * directions / np.where(norms > 0, norms, 1.0))
    stacked = np.asarray(points, dtype=np.float64)
    return [
        stacked[:, offset : offset + size].reshape((len(points), *shape))
        for shape, size, offset in zip(shapes, sizes, offsets)
    ]
