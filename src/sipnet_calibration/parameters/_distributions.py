"""What the prior and the builders of prior functions both know of TFP's
distributions. Private to the parameter layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tensorflow_probability.substrates import jax as tfp

__all__ = [
    "CARRIES_ITS_BIJECTOR",
    "IndexShapedFunction",
    "distribution_name",
]

tfd = tfp.distributions
tfb = tfp.bijectors

#: TFP's classes whose ``.distribution`` and ``.bijector`` are a base in theta
#: and a map from it. Subclasses such as ``MultivariateNormalTriL`` carry an
#: internal reparameterization instead, so only the exact classes count.
CARRIES_ITS_BIJECTOR = (tfd.TransformedDistribution, tfd.LogNormal, tfd.LogitNormal)


class IndexShapedFunction(ABC):
    """A builder's prior function, which the prior calls with the index
    shape first, ``f(index_shape, **given, **constants)``: the number of
    labels in use of each dim the parameters are indexed by, which nothing
    it reads may carry."""

    name: str

    @abstractmethod
    def __call__(self, index_shape: tuple[int, ...], **reads: Any) -> tfd.Distribution: ...


def distribution_name(distribution: tfd.Distribution | None) -> str:
    """A short name for a distribution, for a prior's description: the
    family builders' names, or the class and its bijector."""
    kind = type(distribution)
    if kind is tfd.LogNormal:
        return "log-normal"
    if kind is tfd.LogitNormal:
        return "logit-normal"
    if kind is tfd.TransformedDistribution:
        base = distribution.distribution
        inner = base.distribution if type(base) in (tfd.Sample, tfd.Independent) else base
        if isinstance(distribution.bijector, tfb.SoftmaxCentered):
            return "softmax-normal" if type(inner) is tfd.MultivariateNormalDiag else "softmax-transformed"
        return f"{type(inner).__name__} through {distribution.bijector.name}"
    return kind.__name__
