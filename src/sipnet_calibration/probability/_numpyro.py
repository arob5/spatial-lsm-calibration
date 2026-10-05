"""numpyro's distributions, recognized by class without importing numpyro.
Private to the probability layer.

numpyro is not a dependency of the package: a model may hold a numpyro
distribution, GPJax's ``GaussianDistribution`` among them, only if the
caller has numpyro installed. So a distribution is recognized by the
qualified names of the classes in its MRO, never by ``isinstance``, and
nothing here imports numpyro.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "has_no_density",
    "independent_base",
    "is_dirichlet",
    "is_numpyro_distribution",
    "laws_wrapped_by",
]

#: The qualified name of numpyro's base distribution class.
_DISTRIBUTION = ("numpyro.distributions.distribution", "Distribution")

#: numpyro's continuous laws with no density against the reference measure
#: of any support a component may declare: a point mass, a log factor, and
#: the LKJ laws, whose draws (correlation matrices, or their Cholesky
#: factors) are a null set of the positive-definite matrices, as TFP's
#: are. Discrete laws are recognized by their support instead.
_WITHOUT_A_DENSITY = frozenset(
    {
        ("numpyro.distributions.distribution", "Delta"),
        ("numpyro.distributions.distribution", "Unit"),
        ("numpyro.distributions.continuous", "LKJ"),
        ("numpyro.distributions.continuous", "LKJCholesky"),
    }
)

#: numpyro's Dirichlet, whose density is against Lebesgue measure on the
#: first ``k - 1`` coordinates, as TFP's is.
_DIRICHLET = ("numpyro.distributions.continuous", "Dirichlet")

#: numpyro's wrappers that repeat one law over batch axes: ``Independent``,
#: which reinterprets them as event axes, and ``ExpandedDistribution``.
_REPEATING = frozenset(
    {
        ("numpyro.distributions.distribution", "Independent"),
        ("numpyro.distributions.distribution", "ExpandedDistribution"),
    }
)


def is_numpyro_distribution(value: Any) -> bool:
    """Whether *value* is an instance of a numpyro distribution class."""
    return not isinstance(value, type) and _DISTRIBUTION in _qualified_names(type(value))


def has_no_density(distribution: Any) -> bool:
    """Whether a numpyro distribution is discrete, a point mass, a log
    factor or an LKJ law: one with no density a model can evaluate."""
    if getattr(getattr(distribution, "support", None), "is_discrete", False):
        return True
    return _qualified_name(type(distribution)) in _WITHOUT_A_DENSITY


def is_dirichlet(distribution: Any) -> bool:
    """Whether *distribution* is exactly numpyro's ``Dirichlet``."""
    return is_numpyro_distribution(distribution) and _qualified_name(type(distribution)) == _DIRICHLET


def independent_base(distribution: Any) -> Any:
    """The law a numpyro distribution repeats over its batch: the base under
    any ``Independent`` and ``ExpandedDistribution``, or *distribution*
    itself."""
    while _qualified_name(type(distribution)) in _REPEATING:
        distribution = distribution.base_dist
    return distribution


def laws_wrapped_by(distribution: Any) -> list[Any]:
    """The numpyro distributions *distribution* wraps whose values are its
    values, or parts of them: the base of an ``Independent``, an
    ``ExpandedDistribution``, a ``MaskedDistribution`` or a
    ``TransformedDistribution``, and a mixture's components."""
    wrapped = [getattr(distribution, "base_dist", None), getattr(distribution, "component_distribution", None)]
    wrapped.extend(getattr(distribution, "component_distributions", None) or ())
    return [law for law in wrapped if is_numpyro_distribution(law)]


def _qualified_names(kind: type) -> set[tuple[str, str]]:
    """The qualified names of the classes in *kind*'s MRO."""
    return {_qualified_name(base) for base in kind.__mro__}


def _qualified_name(kind: type) -> tuple[str, str]:
    """A class's module and qualified name."""
    return (kind.__module__, kind.__qualname__)
