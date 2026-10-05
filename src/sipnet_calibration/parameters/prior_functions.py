"""The builders of prior functions, moved to
:mod:`sipnet_calibration.probability.builders`; this module re-exports them
for the parameter layer until it is removed."""

from collections.abc import Callable

from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability.builders import (
    gaussian_copula,
    iid_over_dim,
    independent_over_dim,
)

__all__ = [
    "PriorFunction",
    "gaussian_copula",
    "iid_over_dim",
    "independent_over_dim",
]

#: A function building a term's distribution for the labels in use: one a
#: user writes is called as ``f(**given, **constants)``, as
#: :class:`~sipnet_calibration.parameters.prior.PriorTerm` says; one a builder
#: returns is also passed the index shape, first.
type PriorFunction = Callable[..., tfp.distributions.Distribution]
