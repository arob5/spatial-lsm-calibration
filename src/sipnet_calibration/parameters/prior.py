"""The prior over a parameter vector: what is believed before the data.

Where this sits
---------------
::

    parameters.vector.ParameterVector        (the unknowns x, and theta = T^-1(x))
    parameters.derived.DerivedParameters     (y = f(x), which a term may be given)
      -> parameters.prior.Prior              (pi(x), and so the density of theta)
      -> a sampler (sample, log_prob), an ensemble Kalman method (gaussian)

It imports TFP's distributions, and nothing of the package outside
``parameters``.

What it reads
-------------
A :class:`~sipnet_calibration.parameters.vector.ParameterVector`, optionally
its :class:`~sipnet_calibration.parameters.derived.DerivedParameters`, and
one :class:`PriorTerm` per parameter or group of parameters. A term covers
one parameter, keyed by its name, or several indexed by the same dims,
keyed by a tuple of their names (a **joint term**); it may be **given**
other parameters or derived parameters, and read constants and
memberships. Its distribution is the prior of the whole natural value of
what it covers, its event shape the covered parameters' block shape
``(*index shape, *shape)`` (a dict of them for a joint term), its TFP batch
shape ``()``, in ``float64``:

- **a parameter indexed by nothing, given nothing, reading nothing**: a TFP
  distribution, or a :data:`PriorFunction`;
- **any other term**: a :data:`PriorFunction`, called as
  ``f(index_shape, **given, **constants, **memberships)`` with the covered
  parameters' index shape, one draw's natural value of each name given (of
  its block shape), and the term's constants and memberships read at the
  coords' labels (``int64`` positions for a membership). :func:`iid_over_dim`
  and :func:`independent_over_dim` make one over the index dims;
  :func:`gaussian_copula` makes a joint term.

The distributions of the family builders (:func:`log_normal`,
:func:`logit_normal`, :func:`softmax_normal` and their ``_from_*`` forms)
are pushforwards of a Gaussian through their support's default bijector.

The density
-----------
With parameters :math:`x`, derived parameters :math:`y`, terms
:math:`B_1, \\dots, B_m`, and :math:`x_{g(b)}` and :math:`y_{g(b)}` what term
:math:`b` is given,

.. math::

    \\pi(x) = \\prod_{b=1}^{m} \\pi_b\\big(x_{B_b} \\mid x_{g(b)}, y_{g(b)}\\big),

and with coordinates :math:`\\theta = T^{-1}(x)`,

.. math::

    \\log \\pi_\\theta(\\theta) = \\sum_b \\Big[ \\log \\pi_b\\big(T(\\theta)_{B_b} \\mid
        T(\\theta)_{g(b)}, y_{g(b)}\\big) + \\sum_{p \\in B_b} \\log J_p(\\theta_p) \\Big],

with :math:`\\log J_p` the log-Jacobian of parameter :math:`p`'s transform,
summed over its numbers: :math:`\\log |T'(\\theta)|` per number on an
interval, and on the simplex
:math:`\\log |\\det \\partial(x_1, \\dots, x_{k-1}) / \\partial \\theta|`, against
Lebesgue measure on the first :math:`k - 1` coordinates, the measure a
``Dirichlet``'s density is written against; TFP's
``SoftmaxCentered.forward_log_det_jacobian`` is against the simplex's
surface measure instead, and exceeds this by :math:`\\tfrac12 \\log k`.
:meth:`Prior.log_prob` evaluates each term by its base density where it is a pushforward through
its parameters' own bijectors, which needs no Jacobian, and by change of
variables otherwise. The ``given`` links, with each derived parameter
linked to its parameter names, must form a directed acyclic graph; terms
are drawn in its topological order.

A term's :math:`\\theta_{B_b}` is its parameters' unconstrained values in
the order of its key, each parameter's in C order of its block. A joint
term's base, when it is evaluated by its base density, has that vector as
its event.

Functions and classes
---------------------
:class:`PriorTerm`, :data:`PriorFunction`, :data:`TermKey`
    One term's prior, what it is given and reads, and its provenance.
:class:`Prior`
    ``sample``, ``log_prob``, ``gaussian``, ``select``, ``describe``.
:class:`GaussianMoments`
    What ``gaussian`` returns: a mean and a dense covariance over theta.
:func:`iid_over_dim`, :func:`independent_over_dim`
    Priors over a parameter's index dims, independent across their labels.
:func:`gaussian_copula`
    A joint prior of scalar parameters whose unconstrained values are
    correlated Gaussians.
The family builders
    :func:`log_normal`, :func:`log_normal_from_interval`,
    :func:`log_normal_from_samples`, :func:`logit_normal`,
    :func:`logit_normal_from_interval`, :func:`logit_normal_from_samples`,
    :func:`softmax_normal`.
:func:`term_name`
    A term's name, which keys its randomness.
:class:`DeclaresGaussian`
    The protocol a builder implements to declare its term's Gaussian in
    theta.
:func:`check_prior_term_is_valid`
    The checks a term passes at construction.

Notes
-----
**Gaussians are declared, not detected.** The builders know the Gaussian in
theta they built: :func:`iid_over_dim` and :func:`independent_over_dim` over
a family builder's distribution, :func:`gaussian_copula`, and a family
builder's distribution given directly, declare it (:class:`DeclaresGaussian`).
A declaration is honored when every covered parameter's bijector agrees with
the one the builder assumes at the probe points, and is checked against
``log_prob`` at construction. It feeds only :meth:`Prior.gaussian`, so a
wrong declaration could distort the Gaussian approximation but never
``log_prob``, and the check catches it first. A term given others declares
nothing.

**A prior function must marginalize consistently.** ``prior.select(...)``
rebuilds each term on fewer labels, which is the exact marginal only if the
distribution at a set of labels is the marginal of the one at any larger
set. The builders here satisfy this, and so does a term whose labels are
independent given what it is given. A user-written function that normalizes
over the labels present, or standardizes a covariate inside, does not, and
changes meaning under ``select``.

Usage
-----
::

    prior = Prior(vector, {
        ("respiration_share", "leaf_fall_fraction"): PriorTerm(
            gaussian_copula({"respiration_share": logit_normal(median=0.18, logit_sd=0.35),
                             "leaf_fall_fraction": logit_normal(median=0.5, logit_sd=1.0)},
                            correlation=[[1.0, 0.3], [0.3, 1.0]]),
            provenance="..."),
        "allocation": PriorTerm(
            iid_over_dim(softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5)),
            provenance="..."),
        "initial_soil_carbon": PriorTerm(
            independent_over_dim(log_normal, geometric_sd=2.0),
            constants={"median": median_by_site}, provenance="..."),
    })
    theta = prior.sample(jax.random.key(0), 50)   # (50, D)
    prior.log_prob(theta)                         # (50,)
    prior.gaussian()                              # GaussianMoments(mean, covariance), exact here

A centered hierarchy, a site-level value given its PFT's mean and a shared
spread::

    def soil_carbon_given_pft(index_shape, mean, spread, pft_of_site):
        loc = mean[pft_of_site]
        return tfd.TransformedDistribution(tfd.Independent(tfd.Normal(loc, spread), 1), tfb.Exp())

    PriorTerm(soil_carbon_given_pft, given=("mean", "spread"),
              memberships={"pft_of_site": site_dims.labels("pft")}, provenance="...")
"""

from __future__ import annotations

import dataclasses
import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from frozendict import frozendict
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.parameters._labels import (
    aligned_constants,
    aligned_memberships,
    as_constants,
    as_memberships,
)
from sipnet_calibration.parameters._probes import bijectors_agree, joint_probe_points
from sipnet_calibration.parameters._validation import (
    as_count,
    as_names,
    check_names_are_unique,
    truncated,
)
from sipnet_calibration.parameters.derived import DerivedParameters
from sipnet_calibration.parameters.parameter import Parameter
from sipnet_calibration.parameters.support import (
    OPEN_UNIT_INTERVAL,
    Interval,
    Simplex,
    Support,
    bijector_for,
)
from sipnet_calibration.parameters.vector import (
    ParameterVector,
    check_flat_ends_in_the_size,
    check_parameter_vectors_share_a_layout,
)

__all__ = [
    "DeclaresGaussian",
    "GaussianMoments",
    "Prior",
    "PriorFunction",
    "PriorTerm",
    "TermKey",
    "check_prior_term_is_valid",
    "gaussian_copula",
    "iid_over_dim",
    "independent_over_dim",
    "log_normal",
    "log_normal_from_interval",
    "log_normal_from_samples",
    "logit_normal",
    "logit_normal_from_interval",
    "logit_normal_from_samples",
    "softmax_normal",
    "term_name",
]

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jax.Array

#: A prior over a whole natural value, built for the labels at hand: called
#: as ``f(index_shape, **given, **constants, **memberships)`` and returning a
#: TFP distribution.
type PriorFunction = Callable[..., tfd.Distribution]

#: What a term covers: one parameter's name, or a tuple of names for a joint
#: term.
type TermKey = str | tuple[str, ...]


class GaussianMoments(NamedTuple):
    """A Gaussian over theta, as arrays: ``mean`` ``(D,)`` and
    ``covariance`` ``(D, D)``, both ``float64`` and in theta's order."""

    mean: Array
    covariance: Array


# ── the prior ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class PriorTerm:
    """The prior of what a term covers, what it is given and reads, and
    where it came from.

    A term covering parameters :math:`B` and given :math:`g` is the
    conditional density :math:`\\pi_B(x_B \\mid x_g, y_g)`: at each draw, the
    distribution its function returns for that draw's given values.

    Parameters
    ----------
    distribution:
        A TFP distribution over the whole natural value of what the term
        covers, or a :data:`PriorFunction` that builds one; the module
        docstring says which a term takes.
    given:
        The parameters and derived parameters the distribution is
        conditioned on, passed to its function by name, one draw's natural
        value each.
    constants:
        ``{name: xr.DataArray}``, labeled values the function reads, as a
        derived parameter's are: on dims of the coords or element axes, read
        at their labels, their dims in the covered parameters' ``indexed_by``
        order then their element axes, and passed by name.
    memberships:
        ``{name: xr.DataArray}``, for each label of one dim the label of
        another, passed by name as ``int64`` positions, as a derived
        parameter's are.
    provenance:
        Where the prior came from, with its citation; a placeholder says it
        is one.

    Raises
    ------
    TypeError
        If *given* is one string rather than a sequence of names, or a
        constant or membership is not a DataArray of the right kind.
    ValueError
        If a keyword is given twice, a membership is not on one dim, or
        *provenance* is empty.
    """

    distribution: tfd.Distribution | PriorFunction
    _: KW_ONLY
    given: tuple[str, ...] = ()
    constants: Mapping[str, Any] = field(default_factory=frozendict)
    memberships: Mapping[str, Any] = field(default_factory=frozendict)
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "given", as_names(self.given, message_name="given"))
        object.__setattr__(self, "constants", as_constants(self.constants, message_name="the term's constants"))
        object.__setattr__(
            self, "memberships", as_memberships(self.memberships, message_name="the term's memberships")
        )
        check_names_are_unique(
            [*self.given, *self.constants, *self.memberships], message_name="the term's given, constants and memberships"
        )
        check_provenance_is_given(self.provenance)


@dataclass(frozen=True, eq=False, repr=False)
class Prior:
    """A prior over a parameter vector: one term per parameter, or per group
    of parameters prior'd jointly.

    Parameters
    ----------
    parameter_vector:
        The vector the prior is over.
    terms:
        ``{key: PriorTerm}``, a key being one parameter's name or a tuple of
        the names a joint term covers; every parameter is covered once.
    derived_parameters:
        The derived parameters over the vector, needed when a term is given
        one; checked to share the vector's layout.

    Raises
    ------
    TypeError
        If a key is not a name or a tuple of names, a term is not a
        :class:`PriorTerm`, or a term that needs a function (indexed, given
        others, or reading constants or memberships) is given a bare
        distribution.
    KeyError
        If a key or a ``given`` names nothing of the vector or its derived
        parameters.
    ValueError
        If a parameter is covered by no term or by two, a joint term's
        parameters are indexed differently, the ``given`` links form a
        cycle, the derived parameters are over another vector, or a term
        fails a check of :func:`check_prior_term_is_valid`; the message
        names the term.
    """

    parameter_vector: ParameterVector
    terms: Mapping[TermKey, PriorTerm]
    _: KW_ONLY
    derived_parameters: DerivedParameters | None = None
    _built: Mapping[TermKey, _BuiltTerm] = field(init=False, repr=False)
    _draw_order: tuple[TermKey, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", frozendict(self.terms))
        if self.derived_parameters is not None:
            check_derived_parameters_are_derived_parameters(self.derived_parameters)
            check_parameter_vectors_share_a_layout(self.parameter_vector, self.derived_parameters.parameter_vector)
        check_terms_cover_the_parameters(self.terms, self.parameter_vector)
        for key, term in self.terms.items():
            check_term_is_a_prior_term(term_name(key), term)
            check_given_names_are_held(term_name(key), term.given, self)
            check_term_is_not_given_what_it_covers(key, term.given)
        order = _terms_in_draw_order(self)
        object.__setattr__(self, "_draw_order", order)
        object.__setattr__(self, "_built", frozendict(self._build_terms(order)))

    # ── identity ──────────────────────────────────────────────────────────────

    def __getitem__(self, key: TermKey) -> PriorTerm:
        """The term keyed by *key*: a parameter's name, or a joint term's tuple."""
        check_term_is_held(key, self)
        return self.terms[key]

    def __repr__(self) -> str:
        return f"Prior(D={self.parameter_vector.unconstrained.size}, terms={list(self.terms)})"

    @property
    def coords(self) -> Mapping[str, pd.Index]:
        """The labels the terms' constants and memberships are read at: the
        derived parameters' coords, or the vector's."""
        if self.derived_parameters is not None:
            return self.derived_parameters.coords
        return self.parameter_vector.coords

    def describe(self) -> pd.DataFrame:
        """One row per term, in the vector's declaration order, indexed by
        ``parameters`` (a joint term's names joined with ``"+"``): ``prior``,
        ``given`` (comma-separated), ``evaluated_by`` (``"base density"`` or
        ``"change of variables"``), ``declared_gaussian`` and ``provenance``."""
        rows = [
            {
                "parameters": built.name,
                "prior": _prior_name(built.source, built.distribution),
                "given": ", ".join(built.given),
                "evaluated_by": built.evaluated_by,
                "declared_gaussian": built.declared is not None,
                "provenance": self.terms[built.key].provenance,
            }
            for built in self._built.values()
        ]
        return pd.DataFrame(rows).set_index("parameters")

    # ── selection ─────────────────────────────────────────────────────────────

    def select(self, **selectors: Any) -> Prior:
        """The prior of ``parameter_vector.select(**selectors)``: the kept
        terms, each rebuilt on its kept labels, with the derived parameters
        selected alike.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`ParameterVector.select` and :class:`Prior`;
            ``ValueError`` too if the selection drops a parameter a kept term
            covers or is given, directly or through a derived parameter.
        """
        derived = None if self.derived_parameters is None else self.derived_parameters.select(**selectors)
        vector = self.parameter_vector.select(**selectors) if derived is None else derived.parameter_vector
        kept = {*vector.parameter_names, *(() if derived is None else derived.names)}
        terms = {}
        for key, term in self.terms.items():
            names = _names_of(key)
            if any(name in kept for name in names):
                check_selection_keeps_what_a_term_needs(term_name(key), (*names, *term.given), kept)
                terms[key] = term
        return Prior(vector, terms, derived_parameters=derived)

    # ── evaluation ────────────────────────────────────────────────────────────

    def sample(self, key: Array, n: int) -> Array:
        """``n`` draws of theta from the prior, ``(n, D)``.

        Terms are drawn in topological order of the ``given`` links. A term
        draws with ``jax.random.fold_in(key, crc32(name))``, its name being
        its key, a joint term's joined with ``"+"``; a term given others
        draws one draw at a time, with that key split ``n`` ways, from its
        distribution at that draw's given values. So a term's draws depend on
        its name and on what it is given, not on where it is declared, and
        adding or reordering other terms leaves them unchanged.

        Raises
        ------
        TypeError
            If *n* is not an integer.
        ValueError
            If *n* is negative, or a draw maps to a non-finite theta, which a
            prior with mass on its support's boundary gives in ``float64``;
            the message names the term.
        """
        n = as_count(n, message_name="n")
        theta = jnp.zeros((n, self.parameter_vector.unconstrained.size), dtype=jnp.float64)
        values_by_parameter: dict[str, Array] = {}
        for term_key in self._draw_order:
            built = self._built[term_key]
            given_values = self._given_values(built.given, values_by_parameter)
            theta_b = built.sample_theta(_random_key_for_term(key, built.name), n, given_values)
            check_draws_map_to_finite_theta(built.name, theta_b)
            theta = theta.at[:, built.positions].set(theta_b)
            values_by_parameter |= built.natural_values(theta_b)
        return theta

    def log_prob(self, theta: Any) -> Array:
        """:math:`\\log \\pi_\\theta(\\theta)`, ``(..., D) -> (...)``.

        Each term contributes, at its :math:`\\theta_B` and given the natural
        values :math:`v = (x_{g(b)}, y_{g(b)})` at theta,

        - by **base density**, when its distribution is
          ``TransformedDistribution(base, b)`` with ``b`` equal, at the probe
          points, to the map :math:`\\theta_B \\mapsto \\{p: T_p(\\theta_p)\\}`
          of its parameters' own bijectors:
          :math:`\\log \\mathrm{base}(\\theta_B \\mid v)`, exactly;
        - by **change of variables** otherwise:
          :math:`\\log \\pi_b(T(\\theta_B) \\mid v) + \\sum_{p \\in B}
          \\log J_p(\\theta_p)`, with :math:`\\log J_p` the module's
          log-Jacobian, summed over the parameter's numbers.

        A term given others is evaluated one draw at a time under
        ``jax.vmap``. Traceable under ``jax.jit``, ``jax.grad`` and
        ``jax.vmap``.

        Raises
        ------
        ValueError
            If the last axis of *theta* is not ``D`` long.
        """
        theta = jnp.asarray(theta, dtype=jnp.float64)
        check_flat_ends_in_the_size(theta.shape, self.parameter_vector.unconstrained.size)
        values_by_parameter = self._natural_values_given(theta)
        total = jnp.zeros(theta.shape[:-1], dtype=jnp.float64)
        for built in self._built.values():
            given_values = self._given_values(built.given, values_by_parameter)
            total = total + built.log_prob(theta[..., built.positions], given_values)
        return total

    def gaussian(self, *, key: Array | None = None, n_moment_samples: int = 0) -> GaussianMoments:
        """The prior as a Gaussian over theta: its mean and a dense
        covariance, in theta's order.

        The covariance is block-diagonal over the **dependent sets**, its
        blocks placed at each set's positions in theta. Two parameters are
        in one dependent set when one term covers both, or a term covering
        one is given the other, directly or through a derived parameter it
        is computed from; the prior makes different dependent sets
        independent. A dependent set whose one term declares its Gaussian,
        honored, gets that exact :math:`(m, C)`. Any other is moment-matched
        from :math:`M` = *n_moment_samples* joint draws :math:`\\theta^{(i)}`
        of :meth:`sample` with *key*, restricted to the set's entries:

        .. math::

            \\hat m = \\frac{1}{M} \\sum_i \\theta^{(i)}, \\qquad
            \\hat C = \\frac{1}{M - 1} \\sum_i (\\theta^{(i)} - \\hat m)
                     (\\theta^{(i)} - \\hat m)^\\top.

        Raises
        ------
        NotImplementedError
            If a dependent set needs moment matching and no *key* was given.
        ValueError
            If a moment-matched block has at least *n_moment_samples*
            entries: :math:`\\hat C` then has rank at most :math:`M - 1` and
            is singular.
        """
        n_moment_samples = as_count(n_moment_samples, message_name="n_moment_samples")
        size = self.parameter_vector.unconstrained.size
        mean = np.zeros(size)
        covariance = np.zeros((size, size))
        draws = None
        for names in self._dependent_sets():
            terms = [b for b in self._built.values() if b.names[0] in names]
            if len(terms) == 1 and terms[0].declared is not None:
                positions = terms[0].positions
                block_mean, block = terms[0].declared
            else:
                positions = np.concatenate([self._theta_positions[n] for n in names])
                check_moment_matching_is_possible(names, len(positions), key, n_moment_samples)
                if draws is None:
                    draws = self.sample(key, n_moment_samples)
                block_mean, block = _moment_matched(draws[:, positions])
            mean[positions] = np.asarray(block_mean)
            covariance[np.ix_(positions, positions)] = np.asarray(block)
        return GaussianMoments(mean=jnp.asarray(mean), covariance=jnp.asarray(covariance))

    # ── supporting methods ────────────────────────────────────────────────────

    def _build_terms(self, order: tuple[TermKey, ...]) -> dict[TermKey, _BuiltTerm]:
        """Every term built in draw order. When some term is given others,
        two ancestral draws are made as the terms are built, and each term
        given others is built and checked at both."""
        ancestral = any(term.given for term in self.terms.values())
        ancestral_key = jax.random.key(_ANCESTRAL_SEED)
        positions = self._theta_positions
        values_by_parameter: dict[str, Array] = {}
        built: dict[TermKey, _BuiltTerm] = {}
        for term_key in order:
            term = self.terms[term_key]
            given_values = self._given_values(term.given, values_by_parameter)
            built[term_key] = _BuiltTerm.build(term_key, term, self, positions, given_values)
            if ancestral:
                name = built[term_key].name
                key = _random_key_for_term(ancestral_key, name)
                theta_b = built[term_key].sample_theta(key, _N_ANCESTRAL_DRAWS, given_values)
                check_draws_map_to_finite_theta(name, theta_b)
                values_by_parameter |= built[term_key].natural_values(theta_b)
        # Held in the vector's declaration order, which describe() follows.
        declared = {n: i for i, n in enumerate(self.parameter_vector.parameter_names)}
        return {key: built[key] for key in sorted(built, key=lambda k: declared[_names_of(k)[0]])}

    @property
    def _labels_by_dim(self) -> Mapping[str, pd.Index]:
        """The coords and every element axis of the parameters and derived
        parameters: the labels a term's constants may be read at."""
        pieces = [*self.parameter_vector.parameters]
        if self.derived_parameters is not None:
            pieces += self.derived_parameters.derived_parameters
        return {**{axis: labels for p in pieces for axis, labels in p.element_labels.items()}, **self.coords}

    @property
    def _theta_positions(self) -> Mapping[str, np.ndarray]:
        """Each parameter's positions in theta, in C order of its block."""
        unconstrained = self.parameter_vector.unconstrained
        values = unconstrained.flat_to_values(jnp.arange(unconstrained.size, dtype=jnp.float64))
        return {name: np.asarray(value).ravel().astype(np.int64) for name, value in values.items()}

    def _natural_values_given(self, theta: Array) -> dict[str, Array]:
        """The natural values at *theta* that some term is given, and the
        parameters' values the derived ones among them are computed from."""
        given = list(dict.fromkeys(name for built in self._built.values() for name in built.given))
        if not given:
            return {}
        vector = self.parameter_vector
        values_by_parameter = vector.flat_to_values(vector.to_natural(theta))
        return values_by_parameter | self._derived_needed(given, values_by_parameter)

    def _given_values(self, given: Sequence[str], values_by_parameter: dict[str, Array]) -> dict[str, Array] | None:
        """The values of the names *given*, from *values_by_parameter*,
        computing the derived parameters among them there; ``None`` when
        nothing is given."""
        if not given:
            return None
        values_by_parameter |= self._derived_needed(given, values_by_parameter)
        return {name: values_by_parameter[name] for name in given}

    def _derived_needed(self, names: Sequence[str], values_by_parameter: Mapping[str, Array]) -> dict[str, Array]:
        """The derived parameters among *names* that *values_by_parameter* lacks,
        computed from it."""
        if self.derived_parameters is None:
            return {}
        missing = [n for n in names if n in self.derived_parameters.names and n not in values_by_parameter]
        if not missing:
            return {}
        return self.derived_parameters.values(values_by_parameter, names=missing)

    def _dependent_sets(self) -> list[tuple[str, ...]]:
        """The dependent sets, each in the vector's order, ordered by their
        first parameter."""
        vector = self.parameter_vector
        root = {name: name for name in vector.parameter_names}

        def find(name: str) -> str:
            while root[name] != name:
                name = root[name]
            return name

        for built in self._built.values():
            linked = [*built.names, *sorted(self._parameters_behind(built.given))]
            for name in linked[1:]:
                root[find(name)] = find(linked[0])
        sets: dict[str, list[str]] = {}
        for name in vector.parameter_names:
            sets.setdefault(find(name), []).append(name)
        return [tuple(names) for names in sets.values()]

    def _parameters_behind(self, names: Sequence[str]) -> set[str]:
        if self.derived_parameters is None:
            return set(names)
        return set(self.derived_parameters.parameters_behind(names))


# ── priors over the index dims, and joint priors ──────────────────────────────


def iid_over_dim(distribution: tfd.Distribution) -> PriorFunction:
    """Every label of the index dims independently distributed as
    *distribution*:

    .. math::

        \\pi(x) = \\prod_{\\ell} \\pi_0(x_\\ell),

    over the labels :math:`\\ell` of the product of the parameter's index
    dims. A ``TransformedDistribution(base, b)`` (a family builder's, say)
    is built as ``TransformedDistribution(Sample(base, index_shape), b)``,
    with the bijector outside, so that :meth:`Prior.log_prob` evaluates it by
    its base density; any other distribution as ``Sample(distribution,
    index_shape)``. Over a family builder's distribution with Gaussian base
    :math:`(m_0, C_0)` it declares
    :math:`(\\mathbf 1_n \\otimes m_0,\\ I_n \\otimes C_0)`, with :math:`n`
    the number of labels.

    Parameters
    ----------
    distribution:
        The prior of one label's value, TFP batch shape ``()``.

    Returns
    -------
    PriorFunction
    """
    return _OverDim(
        per_label=lambda index_shape, **constants: distribution,
        repeated=True,
        name=f"iid {_distribution_name(distribution)}",
    )


def independent_over_dim(distribution_family: Callable[..., tfd.Distribution], /, **arguments: Any) -> PriorFunction:
    """Independent across the labels of the index dims, each label's
    distribution made by *distribution_family* from its own arguments:

    .. math::

        \\pi(x) = \\prod_{\\ell} \\pi_{\\mathrm{family}}\\big(x_\\ell;\\ a_\\ell\\big).

    Parameters
    ----------
    distribution_family:
        Builds a distribution from keyword arguments, such as
        :func:`log_normal`, giving it the TFP batch shape of the index when
        an argument has one entry per label.
    **arguments:
        Arguments shared by every label, passed as they are. An argument per
        label is a constant of the term (``PriorTerm(constants=...)``),
        passed to the family by its name at the coords' labels, its dims in
        the parameter's ``indexed_by`` order then its element axes.

    Returns
    -------
    PriorFunction
        A transformed family is built with its bijector outside, as for
        :func:`iid_over_dim`. Over a family builder, with label
        :math:`\\ell`'s base Gaussian :math:`(m_\\ell, C_\\ell)`, it declares
        :math:`\\big((m_1, \\dots, m_n),\\ \\mathrm{blockdiag}(C_1, \\dots, C_n)\\big)`.

    Raises
    ------
    ValueError
        When called, if the family's distribution is not a batch of one per
        label, as when no argument is per label (:func:`iid_over_dim` is that
        prior).
    """

    def per_label(index_shape: tuple[int, ...], **constants: Any) -> tfd.Distribution:
        distribution = distribution_family(**arguments, **constants)
        check_family_is_one_per_label(distribution, index_shape)
        return distribution

    return _OverDim(
        per_label=per_label,
        repeated=False,
        name=f"independent {getattr(distribution_family, '__name__', 'family')}",
    )


def gaussian_copula(marginals: Mapping[str, tfd.Distribution], *, correlation: Any) -> PriorFunction:
    """The joint prior of scalar parameters indexed by nothing whose
    marginals are *marginals* and whose unconstrained values are correlated
    Gaussians.

    Each marginal :math:`i` is a family builder's scalar distribution, a
    pushforward :math:`x_i = T_i(t_i)` of :math:`t_i \\sim \\mathcal
    N(\\mu_i, \\sigma_i^2)`. The copula keeps the marginals and correlates
    the :math:`t_i`:

    .. math::

        t \\sim \\mathcal N\\big(\\mu,\\ \\mathrm{diag}(\\sigma)\\, R\\,
            \\mathrm{diag}(\\sigma)\\big), \\qquad x_i = T_i(t_i),

    with :math:`R` the *correlation*, in the order of *marginals*. It
    declares that Gaussian, which is exact in theta when every parameter's
    bijector is its marginal's :math:`T_i`.

    Parameters
    ----------
    marginals:
        ``{parameter name: distribution}``, each :func:`log_normal`,
        :func:`logit_normal` (on any finite interval), their ``_from_*``
        forms, or a ``tfd.Normal``, with TFP batch shape ``()``. List them in
        the order of the term's key, which is the order its base is laid out
        in.
    correlation:
        ``(m, m)``: symmetric, unit diagonal, positive definite.

    Returns
    -------
    PriorFunction
        Called with ``index_shape=()``; its draws are dicts keyed like
        *marginals*.

    Raises
    ------
    ValueError
        If a marginal is not a scalar pushforward of a Gaussian, or
        *correlation* is not a correlation matrix of that size.
    """
    names = tuple(marginals)
    families = []
    for name in names:
        family = _family_gaussian(marginals[name])
        check_copula_marginal_is_a_scalar_gaussian_pushforward(name, marginals[name], family)
        families.append(family)
    correlation = jnp.asarray(correlation, dtype=jnp.float64)
    check_correlation_is_a_correlation_matrix(correlation, len(names))
    return _GaussianCopula(
        names=names,
        loc=jnp.stack([jnp.asarray(f[0], dtype=jnp.float64) for f in families]),
        scale=jnp.stack([jnp.asarray(f[1], dtype=jnp.float64) for f in families]),
        bijectors=tuple(f[2] for f in families),
        correlation=correlation,
    )


def term_name(key: TermKey) -> str:
    """A term's name: its key, or a joint term's names joined with ``"+"``.
    It keys the term's randomness in :meth:`Prior.sample` and names its row
    in :meth:`Prior.describe`."""
    return key if isinstance(key, str) else "+".join(key)


# ── the family builders ───────────────────────────────────────────────────────


def log_normal(*, median: Any, geometric_sd: Any) -> tfd.LogNormal:
    """The log-normal of the given median and geometric standard deviation:

    .. math::

        \\log x \\sim \\mathcal N\\big(\\log \\mathrm{median},\\ (\\log \\mathrm{geometric\\_sd})^2\\big),

    so its central 95% interval is
    :math:`\\mathrm{median} \\cdot \\mathrm{geometric\\_sd}^{\\pm 1.96}`.

    Parameters
    ----------
    median:
        Positive; one value per label gives a batch.
    geometric_sd:
        Above 1.

    Raises
    ------
    ValueError
        If *median* is not positive or *geometric_sd* not above 1.
    """
    median = _positive_array("log_normal median", median)
    geometric_sd = _positive_array("log_normal geometric_sd", geometric_sd)
    check_geometric_sd_exceeds_one(geometric_sd)
    return tfd.LogNormal(loc=jnp.log(median), scale=jnp.log(geometric_sd))


def log_normal_from_interval(*, lower: Any, upper: Any, mass: float = 0.95) -> tfd.LogNormal:
    """The log-normal whose central *mass* interval is ``[lower, upper]``:

    .. math::

        \\mu = \\tfrac12 (\\log l + \\log u), \\qquad
        \\sigma = \\frac{\\log u - \\log l}{2 z}, \\qquad
        z = \\Phi^{-1}\\big(\\tfrac12 + \\tfrac{\\mathrm{mass}}{2}\\big).

    Raises
    ------
    ValueError
        If an end is not positive, ``upper <= lower``, or *mass* is not in
        ``(0, 1)``.
    """
    lower = _positive_array("log_normal_from_interval lower", lower)
    upper = _positive_array("log_normal_from_interval upper", upper)
    loc, scale = _normal_from_interval(jnp.log(lower), jnp.log(upper), mass)
    return tfd.LogNormal(loc=loc, scale=scale)


def log_normal_from_samples(samples: Any) -> tfd.LogNormal:
    """The maximum-likelihood log-normal of positive samples:
    :math:`\\mu = \\overline{\\log x}`, :math:`\\sigma` the standard deviation
    of :math:`\\log x`.

    Raises
    ------
    ValueError
        If fewer than two samples are given, one is missing or not positive,
        or all are equal.
    """
    logs = jnp.log(_samples_in_support("log_normal_from_samples", samples, lambda v: v > 0))
    return tfd.LogNormal(loc=jnp.mean(logs), scale=_positive_std(logs, "log_normal_from_samples"))


def logit_normal(*, median: Any, logit_sd: Any, support: Interval = OPEN_UNIT_INTERVAL) -> tfd.Distribution:
    """The logit-normal on the finite interval *support*, :math:`(a, b)`:

    .. math::

        \\operatorname{logit}\\frac{x - a}{b - a} \\sim
            \\mathcal N\\Big(\\operatorname{logit}\\frac{\\mathrm{median} - a}{b - a},\\
            \\mathrm{logit\\_sd}^2\\Big).

    On :math:`(0, 1)` this is ``tfd.LogitNormal``; on :math:`(a, b)`,
    ``TransformedDistribution(Normal, Sigmoid(low=a, high=b))``. A
    ``logit_sd`` near 1.7 is close to flat. Its mass is on the interior
    whichever ends *support* closes.

    Raises
    ------
    TypeError
        If *support* is not an :class:`Interval`.
    ValueError
        If *median* is outside the interior of *support*, *logit_sd* is not
        positive, or an end of *support* is infinite.
    """
    check_support_is_a_finite_interval(support)
    fraction = _interval_fraction("logit_normal median", median, support)
    logit_sd = _positive_array("logit_normal logit_sd", logit_sd)
    return _logit_normal_on(support, _logit(fraction), logit_sd)


def logit_normal_from_interval(
    *, lower: Any, upper: Any, mass: float = 0.95, support: Interval = OPEN_UNIT_INTERVAL
) -> tfd.Distribution:
    """The logit-normal on *support* whose central *mass* interval is
    ``[lower, upper]``: :func:`log_normal_from_interval`'s formulas on the
    logit scale of :math:`(x - a)/(b - a)`.

    Raises
    ------
    TypeError, ValueError
        As :func:`logit_normal`, for an end outside *support*, ``upper <=
        lower``, or *mass* not in ``(0, 1)``.
    """
    check_support_is_a_finite_interval(support)
    lower = _interval_fraction("logit_normal_from_interval lower", lower, support)
    upper = _interval_fraction("logit_normal_from_interval upper", upper, support)
    loc, scale = _normal_from_interval(_logit(lower), _logit(upper), mass)
    return _logit_normal_on(support, loc, scale)


def logit_normal_from_samples(samples: Any, *, support: Interval = OPEN_UNIT_INTERVAL) -> tfd.Distribution:
    """The maximum-likelihood logit-normal on *support*, :math:`(a, b)`, of
    samples inside it: with :math:`t = \\operatorname{logit}((x - a)/(b - a))`,
    :math:`\\mu = \\bar t` and :math:`\\sigma` the standard deviation of
    :math:`t`.

    Raises
    ------
    TypeError, ValueError
        As :func:`log_normal_from_samples`, for samples outside the interior
        of *support*, and as :func:`logit_normal` for *support*.
    """
    check_support_is_a_finite_interval(support)
    fractions = (
        _samples_in_support(
            "logit_normal_from_samples", samples, lambda v: (v > support.low) & (v < support.high)
        )
        - support.low
    ) / (support.high - support.low)
    logits = _logit(fractions)
    return _logit_normal_on(support, jnp.mean(logits), _positive_std(logits, "logit_normal_from_samples"))


def softmax_normal(*, center: Any, logit_sd: Any) -> tfd.TransformedDistribution:
    """A Gaussian on :math:`\\mathbb{R}^{k-1}` pushed through
    ``SoftmaxCentered`` onto the open simplex:

    .. math::

        t_i = \\log\\frac{x_i}{x_k} \\sim \\mathcal N\\Big(\\log\\frac{c_i}{c_k},\\
            \\mathrm{logit\\_sd}_i^2\\Big), \\quad i < k,

    independently, with :math:`c` the *center*, so the base's mean maps to
    *center*.

    Parameters
    ----------
    center:
        ``(k,)`` positive fractions summing to 1, ``k >= 2``, or ``(n, k)``
        for one per label.
    logit_sd:
        A scalar; one value per unconstrained number, ``(k - 1,)``; or, with
        an ``(n, k)`` center, one value per label, ``(n,)``, or one per
        label and unconstrained number, ``(n, k - 1)``.

    Raises
    ------
    ValueError
        For a center that is not a point of the simplex, or a *logit_sd* of
        another shape, of an ``(n,)`` shape that could be read either way
        (``n = k - 1``), or not positive.

    Notes
    -----
    Aitchison's name for the family is the logistic-normal; the logit-normal
    is its ``k = 2`` case, hence softmax-normal here.
    """
    center = jnp.asarray(center, dtype=jnp.float64)
    check_center_is_on_the_simplex(center)
    logit_sd = _positive_array("softmax_normal logit_sd", logit_sd)
    loc = jnp.log(center[..., :-1] / center[..., -1:])
    check_logit_sd_fits_the_center(logit_sd, loc)
    if loc.ndim == 2 and logit_sd.shape == loc.shape[:1]:
        logit_sd = logit_sd[:, None]  # one per label, across its numbers
    scale = jnp.broadcast_to(logit_sd, loc.shape)
    return tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale), tfb.SoftmaxCentered()
    )


# ── the declared Gaussian ─────────────────────────────────────────────────────


@runtime_checkable
class DeclaresGaussian(Protocol):
    """A prior builder that knows its term's Gaussian in theta.

    ``unconstrained_gaussian(index_shape, parameter_names, **constants)``
    returns ``(mean, covariance, assumed_bijectors)`` for the term covering
    *parameter_names* at this index shape, with these constants: the mean
    ``(D_b,)`` and the dense covariance ``(D_b, D_b)`` of :math:`\\theta_B`,
    laid out as the module docstring says, and the bijector each parameter's
    value is assumed to be the image of, keyed by those names. It returns
    ``None`` when it declares nothing. The declaration is honored only where
    every assumed bijector agrees with its parameter's at the probe points.
    """

    def unconstrained_gaussian(
        self, index_shape: tuple[int, ...], parameter_names: tuple[str, ...], **constants: Any
    ) -> tuple[Array, Array, Mapping[str, tfb.Bijector]] | None: ...


# ── private constants ─────────────────────────────────────────────────────────

#: TFP's classes whose ``.distribution`` and ``.bijector`` are a base in theta
#: and a map from it. Subclasses such as ``MultivariateNormalTriL`` carry an
#: internal reparameterization instead, so only the exact classes count.
_CARRIES_ITS_BIJECTOR = (tfd.TransformedDistribution, tfd.LogNormal, tfd.LogitNormal)

#: The number of draws of the draw-based support check, and the size of each
#: batch of them, which bounds its memory for a term over many labels.
_SUPPORT_DRAWS, _SUPPORT_DRAW_BATCH = 10_000, 1_000

#: The seed of the draw-based support check.
_SUPPORT_SEED = 20260927

#: The seed and number of the ancestral draws a term given others is built
#: and checked at.
_ANCESTRAL_SEED, _N_ANCESTRAL_DRAWS = 20260928, 2

#: The declaration check's relative and absolute tolerances; see
#: :func:`check_declaration_agrees_with_log_prob`.
_DECLARATION_RELATIVE_TOLERANCE, _DECLARATION_ABSOLUTE_TOLERANCE = 1e-10, 1e-12


# ── private: what the builders return ─────────────────────────────────────────



@dataclass(frozen=True, eq=False)
class _OverDim:
    """A :data:`PriorFunction` over the index dims that declares its
    Gaussian when each label's distribution is a family builder's.

    ``per_label(index_shape, **constants)`` gives one label's distribution
    when *repeated*, else every label's as a batch of the index shape.
    """

    per_label: Callable[..., tfd.Distribution]
    repeated: bool
    name: str

    def __call__(self, index_shape: tuple[int, ...], **constants: Any) -> tfd.Distribution:
        check_prior_over_a_dim_has_a_dim(index_shape, self.name)
        distribution = self.per_label(index_shape, **constants)
        if type(distribution) in _CARRIES_ITS_BIJECTOR:
            base = distribution.distribution
            base = tfd.Sample(base, index_shape) if self.repeated else tfd.Independent(base, len(index_shape))
            return tfd.TransformedDistribution(base, distribution.bijector)
        if self.repeated:
            return tfd.Sample(distribution, index_shape)
        return tfd.Independent(distribution, len(index_shape))

    def unconstrained_gaussian(
        self, index_shape: tuple[int, ...], parameter_names: tuple[str, ...], **constants: Any
    ) -> tuple[Array, Array, Mapping[str, tfb.Bijector]] | None:
        family = _family_gaussian(self.per_label(index_shape, **constants))
        if family is None:
            return None
        loc, scale, bijector = family
        if self.repeated:
            loc = jnp.broadcast_to(loc, (*index_shape, *loc.shape))
            scale = jnp.broadcast_to(scale, (*index_shape, *scale.shape))
        (name,) = parameter_names
        return jnp.ravel(loc), jnp.diag(jnp.ravel(scale) ** 2), {name: bijector}


@dataclass(frozen=True, eq=False)
class _GaussianCopula:
    """:func:`gaussian_copula`'s prior function, which declares its Gaussian."""

    names: tuple[str, ...]
    loc: Array
    scale: Array
    bijectors: tuple[tfb.Bijector, ...]
    correlation: Array
    name: str = "gaussian copula"

    def __call__(self, index_shape: tuple[int, ...]) -> tfd.Distribution:
        check_prior_without_a_dim_has_none(index_shape, self.name)
        to_values = tfb.Chain([
            tfb.JointMap({n: tfb.Chain([b, tfb.Reshape([], [1])]) for n, b in zip(self.names, self.bijectors)}),
            tfb.Restructure({n: i for i, n in enumerate(self.names)}),
            tfb.Split(len(self.names)),
        ])
        base = tfd.MultivariateNormalTriL(self.loc, jnp.linalg.cholesky(self._covariance))
        return tfd.TransformedDistribution(base, to_values)

    def unconstrained_gaussian(
        self, index_shape: tuple[int, ...], parameter_names: tuple[str, ...], **constants: Any
    ) -> tuple[Array, Array, Mapping[str, tfb.Bijector]]:
        # Laid out in the term's order, which a key listed in another order
        # than the marginals permutes.
        order = [self.names.index(n) for n in parameter_names]
        covariance = self._covariance[np.ix_(order, order)]
        return self.loc[np.asarray(order)], covariance, dict(zip(self.names, self.bijectors))

    @property
    def _covariance(self) -> Array:
        return self.scale[:, None] * self.correlation * self.scale[None, :]


@dataclass(frozen=True, eq=False)
class _GivenDirectly:
    """A family builder's distribution given directly for a parameter
    indexed by nothing, which declares its own Gaussian."""

    distribution: tfd.Distribution

    def unconstrained_gaussian(
        self, index_shape: tuple[int, ...], parameter_names: tuple[str, ...], **constants: Any
    ) -> tuple[Array, Array, Mapping[str, tfb.Bijector]] | None:
        family = _family_gaussian(self.distribution)
        if family is None:
            return None
        (name,) = parameter_names
        return jnp.ravel(family[0]), jnp.diag(jnp.ravel(family[1]) ** 2), {name: family[2]}


# ── private: a term built for the vector at hand ──────────────────────────────


@dataclass(frozen=True, eq=False)
class _BuiltTerm:
    """A term built for the vector at hand.

    Its theta is flat, ``(..., D_b)``, in key order, each parameter's
    unconstrained block in C order; ``positions`` are those entries'
    positions in theta. ``labeled_arguments`` are its constants and
    memberships, read at the coords' labels. A term given others keeps the
    distribution at the first ancestral draw, for its checks and its
    description, and rebuilds it per draw to evaluate and sample.
    """

    key: TermKey
    parameters: tuple[Parameter, ...]
    given: tuple[str, ...]
    source: tfd.Distribution | PriorFunction
    index_shape: tuple[int, ...]
    labeled_arguments: Mapping[str, Any]
    distribution: tfd.Distribution
    positions: np.ndarray
    shapes: tuple[tuple[int, ...], ...]
    by_base_density: bool
    declared: tuple[Array, Array] | None

    @classmethod
    def build(
        cls,
        key: TermKey,
        term: PriorTerm,
        prior: Prior,
        theta_positions: Mapping[str, np.ndarray],
        given_values: Mapping[str, Array] | None,
    ) -> _BuiltTerm:
        """Build and check a term; one given others at each ancestral draw
        in *given_values* (``(draws, *block shape)`` each)."""
        vector = prior.parameter_vector
        names = _names_of(key)
        unbuilt = cls(
            key=key,
            parameters=tuple(vector[n] for n in names),
            given=term.given,
            source=term.distribution,
            index_shape=vector.index_shape(names[0]),
            labeled_arguments=frozendict(_labeled_arguments(key, term, prior)),
            distribution=None,
            positions=np.concatenate([theta_positions[n] for n in names]),
            shapes=tuple(vector.unconstrained.block_shape(n) for n in names),
            by_base_density=False,
            declared=None,
        )
        check_term_needing_a_function_is_one(unbuilt)
        probes = unbuilt.probes()
        variants = [unbuilt._with_distribution(d, probes) for d in unbuilt._distributions(given_values)]
        check_term_keeps_its_structure(variants)
        for variant in variants:
            check_prior_term_is_valid(variant)
        built = variants[0]
        declared = None if term.given else _declaration(built, probes)
        if declared is None:
            return built
        check_declaration_agrees_with_log_prob(built, declared, probes)
        return dataclasses.replace(built, declared=declared)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def name(self) -> str:
        return term_name(self.key)

    @property
    def joint(self) -> bool:
        return not isinstance(self.key, str)

    @property
    def size(self) -> int:
        return len(self.positions)

    @property
    def evaluated_by(self) -> str:
        return "base density" if self.by_base_density else "change of variables"

    def probes(self) -> Array:
        """:func:`joint_probe_points` over the covered parameters' blocks,
        the axis probes along each one's first value, flat, ``(n_probes,
        D_b)``."""
        index_ndim = len(self.index_shape)
        parts = joint_probe_points([(shape, math.prod(shape[index_ndim:])) for shape in self.shapes])
        return jnp.asarray(np.concatenate([part.reshape((len(part), -1)) for part in parts], axis=-1))

    def distribution_at(self, given_values: Mapping[str, Array]) -> tfd.Distribution:
        """The distribution at one draw's given values."""
        return self.source(self.index_shape, **given_values, **self.labeled_arguments)

    def split(self, theta: Array) -> dict[str, Array]:
        """Flat theta, ``(..., D_b)``, as each parameter's unconstrained block."""
        lead, out, start = theta.shape[:-1], {}, 0
        for parameter, shape in zip(self.parameters, self.shapes):
            size = math.prod(shape)
            out[parameter.name] = theta[..., start : start + size].reshape(lead + shape)
            start += size
        return out

    def natural_values(self, theta: Array) -> dict[str, Array]:
        """``{name: T_p(theta_p)}`` for the covered parameters."""
        split = self.split(theta)
        return {p.name: p.bijector.forward(split[p.name]) for p in self.parameters}

    def pushes_through(self, distribution: tfd.Distribution, probes: Array) -> bool:
        """Whether *distribution* is ``TransformedDistribution(base, b)``
        with ``b`` the covered parameters' own map at the probe points: an
        exact ``TransformedDistribution``, ``LogNormal`` or ``LogitNormal``."""
        if type(distribution) not in _CARRIES_ITS_BIJECTOR:
            return False
        if not self.joint:
            (shape,) = self.shapes
            return bijectors_agree(
                distribution.bijector, self.parameters[0].bijector, probes.reshape((len(probes), *shape))
            )
        try:
            images = distribution.bijector.forward(jnp.array(probes))
        except (TypeError, ValueError):
            return False
        expected = self.natural_values(probes)
        return isinstance(images, Mapping) and set(images) == set(expected) and all(
            np.allclose(images[n], expected[n], rtol=1e-10, atol=0.0) for n in expected
        )

    def log_prob(self, theta: Array, given_values: Mapping[str, Array] | None) -> Array:
        """This term's contribution at its theta, ``(..., D_b) -> (...)``."""
        if not self.given:
            return self._log_prob_under(self.distribution, theta)
        lead = theta.shape[:-1]
        flat_given = {n: v.reshape((-1, *v.shape[len(lead):])) for n, v in given_values.items()}
        out = jax.vmap(lambda t, v: self._log_prob_under(self.distribution_at(v), t))(
            theta.reshape((-1, theta.shape[-1])), flat_given
        )
        return out.reshape(lead)

    def sample_theta(self, key: Array, n: int, given_values: Mapping[str, Array] | None) -> Array:
        """``n`` draws of this term's theta, ``(n, D_b)``."""
        if not self.given:
            return self._draw(self.distribution, key, n)
        keys = jax.random.split(key, n)
        return jax.vmap(lambda k, v: self._draw(self.distribution_at(v), k, None))(keys, dict(given_values))

    # ── supporting methods ────────────────────────────────────────────────────

    def _distributions(self, given_values: Mapping[str, Array] | None) -> list[tfd.Distribution]:
        """The distributions the term is built and checked at: one per
        ancestral draw for a term given others, else its one distribution
        (its function's, when it is one)."""
        if self.given:
            draws = [{n: v[i] for n, v in given_values.items()} for i in range(_N_ANCESTRAL_DRAWS)]
            return [self.distribution_at(values) for values in draws]
        if isinstance(self.source, tfd.Distribution):
            return [self.source]
        return [self.source(self.index_shape, **self.labeled_arguments)]

    def _with_distribution(self, distribution: tfd.Distribution, probes: Array) -> _BuiltTerm:
        """This term at *distribution*, checked to be over the whole value,
        and evaluated by base density where it pushes through."""
        natural_shapes = {p.name: (*self.index_shape, *p.shape) for p in self.parameters}
        check_term_is_over_the_whole_value(
            self.name, distribution, natural_shapes if self.joint else natural_shapes[self.names[0]]
        )
        return dataclasses.replace(
            self, distribution=distribution, by_base_density=self.pushes_through(distribution, probes)
        )

    def _log_prob_under(self, distribution: tfd.Distribution, theta: Array) -> Array:
        if self.by_base_density:
            return distribution.distribution.log_prob(self._base_event(theta))
        split = self.split(theta)
        values = {p.name: p.bijector.forward(split[p.name]) for p in self.parameters}
        density = distribution.log_prob(values if self.joint else values[self.names[0]])
        lead_ndim = theta.ndim - 1
        for parameter in self.parameters:
            log_jacobian = _log_jacobian(parameter.support, parameter.bijector, split[parameter.name])
            density = density + log_jacobian.sum(axis=tuple(range(lead_ndim, log_jacobian.ndim)))
        return density

    def _draw(self, distribution: tfd.Distribution, key: Array, n: int | None) -> Array:
        """Draws of theta, ``(n, D_b)``, or one, ``(D_b,)``, for ``n=None``."""
        sample_shape = () if n is None else (n,)
        if self.by_base_density:
            draws = distribution.distribution.sample(sample_shape, seed=key)
            return draws.reshape(sample_shape + (self.size,))
        values = distribution.sample(sample_shape, seed=key)
        values = values if self.joint else {self.names[0]: values}
        pieces = [
            p.bijector.inverse(values[p.name]).reshape(sample_shape + (math.prod(shape),))
            for p, shape in zip(self.parameters, self.shapes)
        ]
        return jnp.concatenate(pieces, axis=-1)

    def _base_event(self, theta: Array) -> Array:
        """Flat theta as the base's event: a joint term's is flat, one
        parameter's is its unconstrained block shape."""
        if self.joint:
            return theta
        return theta.reshape(theta.shape[:-1] + self.shapes[0])


def _labeled_arguments(key: TermKey, term: PriorTerm, prior: Prior) -> dict[str, Any]:
    """The term's constants and memberships, read at the prior's labels: a
    constant's dims in the covered parameters' ``indexed_by`` order, then
    their element axes."""
    vector, names = prior.parameter_vector, _names_of(key)
    dim_order = (*vector[names[0]].indexed_by, *(axis for n in names for axis in vector[n].element_labels))
    constants = aligned_constants(
        term.constants,
        prior._labels_by_dim,
        dim_order=dim_order,
        message_name=f"the prior of {term_name(key)!r} constants",
    )
    memberships = aligned_memberships(
        term.memberships, prior.coords, message_name=f"the prior of {term_name(key)!r} memberships"
    )
    return {**constants, **memberships}


def _declaration(built: _BuiltTerm, probes: Array) -> tuple[Array, Array] | None:
    """The term's declared Gaussian, when its builder declares one and every
    covered parameter's bijector agrees with the one it assumes."""
    source = built.source
    if isinstance(source, DeclaresGaussian):
        declarer = source
    elif isinstance(source, tfd.Distribution) and not built.joint:
        declarer = _GivenDirectly(source)
    else:
        return None
    declared = declarer.unconstrained_gaussian(built.index_shape, built.names, **built.labeled_arguments)
    if declared is None:
        return None
    mean, covariance, assumed = declared
    split = built.split(probes)
    if not all(bijectors_agree(assumed[p.name], p.bijector, split[p.name]) for p in built.parameters):
        return None
    return jnp.asarray(mean, dtype=jnp.float64), jnp.asarray(covariance, dtype=jnp.float64)


def _family_gaussian(distribution: tfd.Distribution) -> tuple[Array, Array, tfb.Bijector] | None:
    """A family builder's distribution as its base's mean and standard
    deviations, broadcast to ``(*batch, e?)``, and its bijector; ``None`` for
    any other distribution. Recognized by exact class."""
    if type(distribution) is tfd.Normal:
        loc, scale, bijector = distribution.loc, distribution.scale, tfb.Identity()
    elif type(distribution) in (tfd.LogNormal, tfd.LogitNormal) or _is_pushforward(
        distribution, tfd.Normal, tfb.Sigmoid
    ):
        base = distribution.distribution
        loc, scale, bijector = base.loc, base.scale, distribution.bijector
    elif _is_pushforward(distribution, tfd.MultivariateNormalDiag, tfb.SoftmaxCentered):
        base = distribution.distribution
        loc, scale, bijector = base.mean(), base.stddev(), distribution.bijector
    else:
        return None
    loc, scale = jnp.broadcast_arrays(jnp.asarray(loc), jnp.asarray(scale))
    return loc, scale, bijector


def _is_pushforward(distribution: tfd.Distribution, base_class: type, bijector_class: type) -> bool:
    """Whether *distribution* is exactly a ``TransformedDistribution`` of an
    exact *base_class* through an exact *bijector_class*."""
    return (
        type(distribution) is tfd.TransformedDistribution
        and type(distribution.distribution) is base_class
        and type(distribution.bijector) is bijector_class
    )


def _log_jacobian(support: Support, bijector: tfb.Bijector, theta: Array) -> Array:
    """:math:`\\log J(\\theta)`, the log absolute Jacobian determinant of
    *bijector* against the reference measure of *support*'s densities.

    On an :class:`Interval` the measure is Lebesgue measure and
    :math:`\\log J = \\log |T'(\\theta)|`, number by number, the shape of
    *theta*. On the :class:`Simplex` it is Lebesgue measure on the first
    :math:`k - 1` coordinates, the measure a ``Dirichlet``'s density is
    written against, which makes :math:`\\pi_\\theta` a normalized density on
    :math:`\\mathbb{R}^{k-1}`:

    .. math::

        \\log J(\\theta) = \\log \\left| \\det
            \\frac{\\partial (x_1, \\dots, x_{k-1})}{\\partial \\theta} \\right|,
        \\qquad x = T(\\theta),

    over the last axis, ``(..., k - 1) -> (...)``. For ``SoftmaxCentered``
    it is the closed form

    .. math::

        \\log J(\\theta) = \\sum_{i=1}^{k} \\log x_i,
        \\qquad \\log x = \\operatorname{log\\_softmax}([\\theta, 0]),

    computed from :math:`\\theta` so that no coordinate rounds to 0; for any
    other bijector, automatic differentiation of the determinant, which
    loses precision when coordinates are tiny. TFP's
    ``SoftmaxCentered.forward_log_det_jacobian`` is
    :math:`\\tfrac12 \\log \\det(J^\\top J)` of the full :math:`k \\times (k-1)`
    Jacobian instead, against the simplex's surface measure: it is this plus
    :math:`\\tfrac12 \\log k`.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    if isinstance(support, Interval):
        return bijector.forward_log_det_jacobian(theta, event_ndims=0)
    if type(bijector) is tfb.SoftmaxCentered:
        padded = jnp.concatenate([theta, jnp.zeros((*theta.shape[:-1], 1))], axis=-1)
        return jax.nn.log_softmax(padded, axis=-1).sum(axis=-1)

    def log_determinant(t: Array) -> Array:
        jacobian = jax.jacfwd(lambda u: bijector.forward(u)[:-1])(t)
        return jnp.linalg.slogdet(jacobian)[1]

    flat = theta.reshape((-1, theta.shape[-1]))
    return jax.vmap(log_determinant)(flat).reshape(theta.shape[:-1])


def _gaussian_log_density(mean: Array, covariance: Array, points: Array) -> Array:
    """:math:`\\log \\mathcal N(x; m, C) = -\\tfrac12 \\big(d \\log 2\\pi +
    \\log\\det C + (x - m)^\\top C^{-1} (x - m)\\big)`, through the Cholesky
    factor of :math:`C`, ``(..., d) -> (...)``."""
    factor = jnp.linalg.cholesky(covariance)
    residual = jnp.asarray(points) - mean
    whitened = jax.scipy.linalg.solve_triangular(factor, residual.reshape((-1, len(mean))).T, lower=True).T
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
    quadratic = jnp.sum(whitened**2, axis=-1).reshape(residual.shape[:-1])
    return -0.5 * (len(mean) * jnp.log(2.0 * jnp.pi) + log_determinant + quadratic)


# ── private helpers ───────────────────────────────────────────────────────────


def _names_of(key: TermKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


def _random_key_for_term(key: Array, name: str) -> Array:
    return jax.random.fold_in(key, zlib.crc32(name.encode()))


def _terms_in_draw_order(prior: Prior) -> tuple[TermKey, ...]:
    """The terms in a topological order of the ``given`` links, each after
    everything it is given: the covering term of a parameter, and, through a
    derived parameter, the covering terms of what it is computed from."""
    terms, vector = prior.terms, prior.parameter_vector
    owner = {name: key for key in terms for name in _names_of(key)}
    derived = prior.derived_parameters
    derived_names = () if derived is None else derived.names

    def links(node: tuple[str, Any]) -> list[tuple[str, Any]]:
        kind, value = node
        names = terms[value].given if kind == "term" else derived[value].parameter_names
        return [("derived", n) if n in derived_names else ("term", owner[n]) for n in names]

    order: list[TermKey] = []
    state: dict[tuple[str, Any], str] = {}
    first_position = {key: min(vector.parameter_names.index(n) for n in _names_of(key)) for key in terms}
    for start in sorted(terms, key=first_position.__getitem__):
        if ("term", start) in state:
            continue
        path, stack = [("term", start)], [iter(links(("term", start)))]
        state[("term", start)] = "open"
        while stack:
            child = next(stack[-1], None)
            if child is None:
                node = path.pop()
                stack.pop()
                state[node] = "done"
                if node[0] == "term":
                    order.append(node[1])
                continue
            cycle = path[path.index(child):] + [child] if state.get(child) == "open" else None
            check_given_links_are_acyclic(cycle)
            if child not in state:
                state[child] = "open"
                path.append(child)
                stack.append(iter(links(child)))
    return tuple(order)


def _moment_matched(draws: Array) -> tuple[Array, Array]:
    mean = draws.mean(axis=0)
    centered = draws - mean
    return mean, centered.T @ centered / (len(draws) - 1)


def _logit_normal_on(support: Interval, loc: Array, scale: Array) -> tfd.Distribution:
    if (support.low, support.high) == (0.0, 1.0):
        return tfd.LogitNormal(loc=loc, scale=scale)
    return tfd.TransformedDistribution(tfd.Normal(loc, scale), bijector_for(support))


def _interval_fraction(what: str, value: Any, support: Interval) -> Array:
    """``(value - a) / (b - a)``, checked inside :math:`(0, 1)`."""
    array = jnp.asarray(value, dtype=jnp.float64)
    fraction = (array - support.low) / (support.high - support.low)
    check_values_are_inside(what, fraction, value, support)
    return fraction


def _positive_array(what: str, value: Any) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    check_values_are_positive(what, array, value)
    return array


def _logit(p: Array) -> Array:
    return jnp.log(p) - jnp.log1p(-p)


def _normal_from_interval(lower: Array, upper: Array, mass: float) -> tuple[Array, Array]:
    """The Normal whose central *mass* interval is ``[lower, upper]``."""
    check_interval_ends_and_mass_are_valid(lower, upper, mass)
    z = tfd.Normal(jnp.float64(0.0), jnp.float64(1.0)).quantile(jnp.float64(0.5 + mass / 2))
    return (lower + upper) / 2.0, (upper - lower) / (2.0 * z)


def _samples_in_support(what: str, samples: Any, in_support: Callable[[Array], Array]) -> Array:
    array = jnp.asarray(samples, dtype=jnp.float64).ravel()
    check_samples_are_usable(what, array, in_support)
    return array


def _positive_std(values: Array, what: str) -> Array:
    std = jnp.std(values)
    check_samples_vary(what, std)
    return std


def _prior_name(source: Any, distribution: tfd.Distribution) -> str:
    if isinstance(source, (_OverDim, _GaussianCopula)):
        return source.name
    return _distribution_name(distribution)


def _distribution_name(distribution: tfd.Distribution | None) -> str:
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


def _structure_of(distribution: tfd.Distribution) -> Any:
    """A distribution's pytree structure: its class, its parts' classes and
    its static parameters; its class alone where TFP gives it no pytree
    structure, as for a ``JointDistributionNamed``."""
    try:
        return jax.tree_util.tree_structure(distribution)
    except (AttributeError, TypeError):
        return type(distribution)


def _shape_of(shape: Any) -> Any:
    """A TFP shape, or a dict of them for a joint distribution, as tuples."""
    if isinstance(shape, Mapping):
        return {name: tuple(value) for name, value in shape.items()}
    return tuple(shape)


def _inside_at_each_probe(built: _BuiltTerm, values: Mapping[str, Array], *, closure: bool = False) -> Array:
    """``(n_probes,)``: whether every covered parameter's value lies in its
    support (its closure when *closure*)."""
    inside = []
    for p in built.parameters:
        support = p.support.closure() if closure else p.support
        inside.append(support.contains(values[p.name]).reshape((len(values[p.name]), -1)).all(axis=-1))
    return jnp.all(jnp.stack(inside), axis=0)


def _as_values(built: _BuiltTerm, draws: Any) -> Mapping[str, Array]:
    """A distribution's draws as ``{name: value}``."""
    return draws if built.joint else {built.names[0]: draws}


def _probe_log_prob_is_finite(built: _BuiltTerm, probes: Array) -> bool:
    """Whether the density is finite at the image of every probe point that
    lies inside the support; one the bijector rounds onto the boundary
    (``IteratedSigmoidCentered`` at 20) says nothing of the prior's support."""
    values = built.natural_values(probes)
    inside = _inside_at_each_probe(built, values)
    log_prob = built.distribution.log_prob(values if built.joint else values[built.names[0]])
    return bool(jnp.all(jnp.isfinite(log_prob) | ~inside))


def _default_bijector_images_lie_in(built: _BuiltTerm) -> bool | None:
    """Whether the distribution's own default event-space bijector maps the
    probe points into the closure of the supports; ``None`` when it has none."""
    distribution = built.distribution
    try:
        bijector = distribution.experimental_default_event_space_bijector()
    except NotImplementedError:
        return None
    if bijector is None:
        return None
    unconstrained = bijector.inverse_event_shape(distribution.event_shape)
    leaves, structure = jax.tree_util.tree_flatten(unconstrained, is_leaf=lambda s: hasattr(s, "as_list"))
    shapes = [tuple(leaf) for leaf in leaves]
    parts = joint_probe_points([(shape, shape[-1] if shape else 1) for shape in shapes])
    images = bijector.forward(jax.tree_util.tree_unflatten(structure, [jnp.asarray(p) for p in parts]))
    # The closure: a bijector may round onto the boundary at the outer probes,
    # which is float64, not a wrong support.
    return bool(jnp.all(_inside_at_each_probe(built, _as_values(built, images), closure=True)))


def _draws_lie_in_the_support(built: _BuiltTerm) -> bool | None:
    """Whether fixed-seed draws lie in the supports and map to finite theta;
    ``None`` when the distribution cannot be sampled."""
    key = jax.random.key(_SUPPORT_SEED)
    for batch in range(_SUPPORT_DRAWS // _SUPPORT_DRAW_BATCH):
        try:
            draws = built.distribution.sample(_SUPPORT_DRAW_BATCH, seed=jax.random.fold_in(key, batch))
        except NotImplementedError:
            return None
        values = _as_values(built, draws)
        theta = [p.bijector.inverse(values[p.name]) for p in built.parameters]
        if not (
            bool(jnp.all(_inside_at_each_probe(built, values)))
            and all(bool(jnp.all(jnp.isfinite(t))) for t in theta)
        ):
            return False
    return True


# ── checks ────────────────────────────────────────────────────────────────────


def check_prior_term_is_valid(built: _BuiltTerm) -> None:
    """A term's prior has the supports its parameters declare, and a density
    :meth:`Prior.log_prob` can evaluate."""
    check_change_of_variables_has_a_measure(built)
    check_simplex_density_is_a_dirichlet(built)
    check_declared_support_lies_in_the_priors(built)
    check_priors_support_lies_in_the_declared(built)


def check_derived_parameters_are_derived_parameters(derived_parameters: Any) -> None:
    """The derived parameters are a :class:`DerivedParameters`."""
    if not isinstance(derived_parameters, DerivedParameters):
        raise TypeError(
            f"derived_parameters must be a DerivedParameters, got {type(derived_parameters).__name__}."
        )


def check_terms_cover_the_parameters(terms: Mapping[Any, Any], vector: ParameterVector) -> None:
    """The terms are keyed by the vector's parameter names, or tuples of
    them indexed alike, and cover every parameter once, so no density is
    counted twice or left out."""
    owner: dict[str, str] = {}
    for key in terms:
        check_term_key_is_names(key)
        for name in _names_of(key):
            check_term_covers_a_parameter_once(name, term_name(key), vector, owner)
            owner[name] = term_name(key)
        check_joint_term_is_indexed_alike(key, vector)
    check_every_parameter_has_a_term(owner, vector)


def check_term_key_is_names(key: Any) -> None:
    """A term is keyed by a parameter's name, or a joint term by a tuple of
    two or more, which is how it is read; a shorter tuple would be taken for
    a joint term and refused for its draws' shape."""
    if not (isinstance(key, str) or (isinstance(key, tuple) and all(isinstance(n, str) for n in key))):
        raise TypeError(
            f"a prior term is keyed by a parameter's name, or a tuple of names for a joint term, "
            f"got {key!r}."
        )
    if isinstance(key, tuple) and len(key) < 2:
        raise ValueError(
            f"the prior term {key!r} is a tuple of {len(key)} name(s), but a joint term covers two "
            "or more; key one parameter's term by its name."
        )


def check_term_covers_a_parameter_once(
    name: str, term: str, vector: ParameterVector, owner: Mapping[str, str]
) -> None:
    """A term's name is a parameter of the vector that no other term covers,
    whose density would otherwise be counted twice."""
    if name not in vector:
        raise KeyError(
            f"a prior term names {name!r}, which is no parameter of the vector; name one of "
            f"{truncated(list(vector.parameter_names))}."
        )
    if name in owner:
        raise ValueError(
            f"parameter {name!r} is covered by the terms {owner[name]!r} and {term!r}; cover each "
            "parameter by one term."
        )


def check_every_parameter_has_a_term(owner: Mapping[str, str], vector: ParameterVector) -> None:
    """Every parameter is covered by a term, whose density would otherwise be
    left out."""
    missing = [name for name in vector.parameter_names if name not in owner]
    if missing:
        raise ValueError(f"the parameter(s) {truncated(missing)} have no prior term; give each one.")


def check_joint_term_is_indexed_alike(key: TermKey, vector: ParameterVector) -> None:
    """A joint term's parameters are indexed by the same dims: dependence
    across dims is what ``given`` expresses."""
    indexed = {vector[name].indexed_by for name in _names_of(key)}
    if len(indexed) > 1:
        raise ValueError(
            f"the joint term {term_name(key)!r} covers parameters indexed by {sorted(indexed)}; a "
            "joint term's parameters are indexed alike, so give each its own term and link them "
            "with given=."
        )


def check_given_names_are_held(name: str, given: Sequence[str], prior: Prior) -> None:
    """What a term is given is a parameter or derived parameter, which it
    would otherwise be passed no value for."""
    derived = () if prior.derived_parameters is None else prior.derived_parameters.names
    for given_name in given:
        if given_name not in prior.parameter_vector and given_name not in derived:
            raise KeyError(
                f"the prior of {name!r} is given {given_name!r}, which is no parameter of the vector "
                "nor a derived parameter of derived_parameters=."
            )


def check_term_is_not_given_what_it_covers(key: TermKey, given: Sequence[str]) -> None:
    """A term is not given a parameter it covers, whose density it is."""
    covered = [name for name in given if name in _names_of(key)]
    if covered:
        raise ValueError(
            f"the prior of {term_name(key)!r} is given {covered}, which it covers; a term is the "
            "density of what it covers, so give it only other parameters."
        )


def check_given_links_are_acyclic(cycle: Sequence[tuple[str, Any]] | None) -> None:
    """The ``given`` links, through derived parameters, form no cycle, which
    no order of draws could satisfy; *cycle* is the one found, if any."""
    if cycle is None:
        return
    names = [term_name(value) if kind == "term" else value for kind, value in cycle]
    raise ValueError(
        f"the given links form a cycle, {' -> '.join(names)}; a term cannot depend on itself, so "
        "break the cycle."
    )


def check_term_needing_a_function_is_one(built: _BuiltTerm) -> None:
    """A term over indexed parameters, given others, or reading constants or
    memberships is a function of them; a bare distribution would ignore
    them."""
    if not isinstance(built.source, tfd.Distribution):
        return
    if built.given:
        raise TypeError(
            f"the prior of {built.name!r} is given {list(built.given)} but is a distribution; give "
            "a function f(index_shape, **given_values) that returns one."
        )
    if built.index_shape:
        raise TypeError(
            f"the prior of {built.name!r} is over parameters indexed by "
            f"{built.parameters[0].indexed_by}, so it is built for the labels present; wrap the "
            "distribution as iid_over_dim(distribution) or independent_over_dim(family, ...)."
        )
    if built.labeled_arguments:
        raise TypeError(
            f"the prior of {built.name!r} reads {sorted(built.labeled_arguments)} but is a distribution; "
            "give a function that reads them."
        )


def check_term_keeps_its_structure(variants: Sequence[_BuiltTerm]) -> None:
    """A term given others has one structure at every draw, depending on the
    given values only through its parameters, which evaluating it one draw
    at a time needs."""
    first, *rest = variants
    for variant in rest:
        if (
            _structure_of(variant.distribution) != _structure_of(first.distribution)
            or variant.by_base_density != first.by_base_density
        ):
            raise ValueError(
                f"the prior of {first.name!r} changes its structure with the values it is given: its "
                "class, its parts, or whether it is a pushforward through the parameters' own "
                "bijectors; let the given values enter only through the distribution's parameters."
            )


def check_selection_keeps_what_a_term_needs(name: str, needed: Sequence[str], kept: set[str]) -> None:
    """A kept term keeps every parameter it covers and everything it is
    given, which it could not otherwise be built on."""
    dropped = [n for n in needed if n not in kept]
    if dropped:
        raise ValueError(
            f"the selection drops {truncated(dropped)}, which the prior term {name!r} covers or is "
            "given; keep them, or drop the term's parameters too."
        )


def check_term_is_held(key: Any, prior: Prior) -> None:
    """The prior has a term for the key asked for."""
    if key not in prior.terms:
        raise KeyError(f"the prior has no term {key!r}; name one of {truncated(list(prior.terms))}.")


def check_term_is_a_prior_term(name: str, term: Any) -> None:
    """A term is a :class:`PriorTerm`, which carries its provenance."""
    if not isinstance(term, PriorTerm):
        raise TypeError(
            f"the prior of {name!r} is a {type(term).__name__}; wrap it as "
            "PriorTerm(distribution, provenance=...)."
        )


def check_provenance_is_given(provenance: Any) -> None:
    """A prior says where it came from."""
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(
            "a prior term needs a provenance: where the prior came from, or that it is a "
            "placeholder."
        )


def check_prior_over_a_dim_has_a_dim(index_shape: tuple[int, ...], name: str) -> None:
    """A prior over the index dims is given to a parameter indexed by some."""
    if not index_shape:
        raise TypeError(
            f"{name} is a prior over a parameter's index dims, given to a parameter indexed by "
            "nothing; give that parameter the distribution itself."
        )


def check_prior_without_a_dim_has_none(index_shape: tuple[int, ...], name: str) -> None:
    """A prior of parameters indexed by nothing is not given indexed ones,
    whose draws would otherwise be refused for their shape, far from the
    reason."""
    if index_shape:
        raise TypeError(
            f"{name} is a prior of parameters indexed by nothing, given parameters of index shape "
            f"{index_shape}; give those parameters a prior over their index dims."
        )


def check_term_is_over_the_whole_value(name: str, distribution: Any, expected: Any) -> None:
    """A term's distribution is ``float64`` over the whole value it covers:
    event shape the block shape (a dict of them for a joint term), TFP batch
    shape ``()``."""
    if not isinstance(distribution, tfd.Distribution):
        raise TypeError(
            f"the prior of {name!r} built a {type(distribution).__name__}, not a TFP distribution."
        )
    event, batch = _shape_of(distribution.event_shape), _shape_of(distribution.batch_shape)
    dtypes = distribution.dtype.values() if isinstance(distribution.dtype, Mapping) else [distribution.dtype]
    batches = batch.values() if isinstance(batch, Mapping) else [batch]
    if event != expected or any(b != () for b in batches) or any(d != jnp.float64 for d in dtypes):
        raise ValueError(
            f"the prior of {name!r} has event shape {event}, TFP batch shape {batch} and dtype "
            f"{distribution.dtype}, but must be a float64 distribution over the whole value it "
            f"covers: event shape {expected}, TFP batch shape (). A batch of priors, one per label, "
            "is iid_over_dim or independent_over_dim; a joint term's draws are a dict keyed by the "
            "names it covers."
        )


def check_change_of_variables_has_a_measure(built: _BuiltTerm) -> None:
    """A term evaluated by change of variables is over intervals and
    simplices, the supports whose Jacobian's reference measure is known."""
    if built.by_base_density:
        return
    unknown = [p.name for p in built.parameters if not isinstance(p.support, (Interval, Simplex))]
    if unknown:
        raise ValueError(
            f"the prior of {built.name!r} is evaluated by change of variables over {unknown}, whose "
            "supports have no known reference measure; write it as a pushforward through the "
            "parameters' bijectors, TransformedDistribution(base, bijector)."
        )


def check_simplex_density_is_a_dirichlet(built: _BuiltTerm) -> None:
    """On the simplex, a term evaluated by change of variables is one
    parameter's ``Dirichlet``, whose density is against the first ``k - 1``
    coordinates, as the Jacobian is."""
    if built.by_base_density or not any(isinstance(p.support, Simplex) for p in built.parameters):
        return
    inner = built.distribution
    if type(inner) in (tfd.Sample, tfd.Independent):
        inner = inner.distribution
    if built.joint or type(inner) is not tfd.Dirichlet:
        raise ValueError(
            f"the prior of {built.name!r} is a density on the simplex other than a Dirichlet, "
            "whose reference measure is unknown; write it as a pushforward through the "
            "parameters' bijectors, TransformedDistribution(base, bijector)."
        )


def check_declared_support_lies_in_the_priors(built: _BuiltTerm) -> None:
    """The prior's density is finite at the image of every probe point, so
    the declared supports lie in the prior's."""
    if not _probe_log_prob_is_finite(built, built.probes()):
        raise ValueError(
            f"the prior of {built.name!r} has no density at some values of its declared "
            f"support; its own support is smaller. Declare the support the prior has, or choose "
            "a prior over the whole support."
        )


def check_priors_support_lies_in_the_declared(built: _BuiltTerm) -> None:
    """The prior puts no mass outside the declared supports, by its own
    default bijector's images and by draws that map to finite theta."""
    by_bijector = _default_bijector_images_lie_in(built)
    by_draws = _draws_lie_in_the_support(built)
    if by_bijector is None and by_draws is None:
        raise ValueError(
            f"the prior of {built.name!r} has neither a default event-space bijector nor a "
            "sampler, so its support cannot be checked; give a distribution TFP can sample."
        )
    if by_bijector is False or by_draws is False:
        supports = ", ".join(p.support.name for p in built.parameters)
        raise ValueError(
            f"the prior of {built.name!r} puts mass outside its declared support ({supports}), "
            "or on its boundary, where theta is not finite; declare the support the prior has, "
            "or choose a prior inside it."
        )


def check_declaration_agrees_with_log_prob(built: _BuiltTerm, declared: tuple[Array, Array], probes: Array) -> None:
    """A declared Gaussian's log density :math:`\\log q` equals the term's,
    :math:`\\log p`, at every probe point, to
    :math:`|\\log q - \\log p| \\le 10^{-10} |\\log p| + 10^{-12} D_b`, with
    :math:`D_b` the term's number of entries of theta."""
    mean, covariance = declared
    declared_log_density = _gaussian_log_density(mean, covariance, probes)
    log_density = built.log_prob(probes, None)
    tolerance = (
        _DECLARATION_RELATIVE_TOLERANCE * jnp.abs(log_density)
        + _DECLARATION_ABSOLUTE_TOLERANCE * probes.shape[-1]
    )
    if not bool(jnp.all(jnp.abs(declared_log_density - log_density) <= tolerance)):
        raise ValueError(
            f"the Gaussian declared for {built.name!r} disagrees with its log density at the probe "
            "points; this is a defect in the prior builder, not in the inputs."
        )


def check_draws_map_to_finite_theta(name: str, theta: Array) -> None:
    """Every draw maps to a finite theta."""
    if not bool(jnp.all(jnp.isfinite(theta))):
        raise ValueError(
            f"the prior of {name!r} drew a value on its support's boundary, whose theta is not "
            "finite; choose a prior with less mass at the boundary."
        )


def check_moment_matching_is_possible(
    names: Sequence[str], size: int, key: Array | None, n_moment_samples: int
) -> None:
    """A moment-matched block has a key, and more draws than entries, since
    its estimate is otherwise singular."""
    what = f"the prior of {names[0]!r}" if len(names) == 1 else f"the dependent set {list(names)}"
    if key is None:
        raise NotImplementedError(
            f"{what} declares no Gaussian, so it is moment-matched from draws; pass key= and "
            "n_moment_samples=."
        )
    if size >= n_moment_samples:
        raise ValueError(
            f"{what} has {size} entries and would be moment-matched from {n_moment_samples} "
            f"draws, whose covariance has rank at most {max(n_moment_samples - 1, 0)} and is "
            f"singular; draw more than {size}, or write the prior in a form that declares its "
            "Gaussian: a family builder's distribution, or a hierarchy written non-centered, "
            "with a derived parameter."
        )


def check_copula_marginal_is_a_scalar_gaussian_pushforward(
    name: str, marginal: Any, family: tuple[Array, Array, tfb.Bijector] | None
) -> None:
    """A copula's marginal is a family builder's scalar distribution, whose
    Gaussian in theta the copula correlates."""
    if family is None or tuple(marginal.batch_shape) != () or tuple(marginal.event_shape) != ():
        raise ValueError(
            f"gaussian_copula's marginal {name!r} is not a scalar pushforward of a Gaussian; give "
            "log_normal, logit_normal, their _from_* forms, or tfd.Normal, with TFP batch shape ()."
        )


def check_correlation_is_a_correlation_matrix(correlation: Array, n_marginals: int) -> None:
    """A copula's correlation is an ``(m, m)`` symmetric, unit-diagonal,
    positive definite matrix: TFP would otherwise read its lower triangle, or
    fail its Cholesky factor, without an error."""
    shape = (n_marginals, n_marginals)
    if (
        correlation.shape != shape
        or not bool(jnp.allclose(correlation, correlation.T, rtol=0.0, atol=1e-12))
        or not bool(jnp.allclose(jnp.diag(correlation), 1.0, rtol=0.0, atol=1e-12))
        or not bool(jnp.all(jnp.linalg.eigvalsh(correlation) > 0.0))
    ):
        raise ValueError(
            f"gaussian_copula's correlation must be a {shape} symmetric, positive definite matrix "
            f"with unit diagonal, got shape {correlation.shape}; give a correlation matrix in the "
            "order of the marginals."
        )


def check_family_is_one_per_label(distribution: Any, index_shape: tuple[int, ...]) -> None:
    """A family's distribution is a batch of one per label, which the labels
    would otherwise be misaligned with."""
    batch = tuple(getattr(distribution, "batch_shape", ()))
    if batch != tuple(index_shape):
        raise ValueError(
            f"independent_over_dim built a distribution of TFP batch shape {batch} for the index "
            f"shape {tuple(index_shape)}; give at least one argument per label, as a constant of "
            "the term, or use iid_over_dim for one distribution shared by every label."
        )


def check_geometric_sd_exceeds_one(geometric_sd: Array) -> None:
    """A geometric standard deviation exceeds 1, since its log is the scale."""
    if not bool(jnp.all(geometric_sd > 1.0)):
        raise ValueError("log_normal: geometric_sd is at most 1, and it multiplies; give a value above 1.")


def check_values_are_positive(what: str, array: Array, value: Any) -> None:
    """A family builder's positive argument is finite and positive."""
    if not bool(jnp.all(jnp.isfinite(array)) and jnp.all(array > 0)):
        raise ValueError(f"{what} is {value!r}, which is not finite and positive; give a positive value.")


def check_values_are_inside(what: str, fraction: Array, value: Any, support: Support) -> None:
    """A logit-normal's median or end lies inside the interior of its support."""
    if not bool(jnp.all((fraction > 0) & (fraction < 1))):
        raise ValueError(f"{what} is {value!r}, outside the interior of {support.name}; give a value inside it.")


def check_support_is_a_finite_interval(support: Any) -> None:
    """A logit-normal's support is an interval with finite ends."""
    if not isinstance(support, Interval):
        raise TypeError(f"support must be an Interval, got {type(support).__name__}; use Interval(low, high).")
    if math.isinf(support.low) or math.isinf(support.high):
        raise ValueError(
            f"a logit-normal is on a finite interval, got support {support.name}; use log_normal on "
            "a half-line."
        )


def check_interval_ends_and_mass_are_valid(lower: Array, upper: Array, mass: float) -> None:
    """An interval's mass is in (0, 1) and its upper end exceeds its lower."""
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass is {mass}, outside (0, 1); give the central mass as a fraction.")
    if not bool(jnp.all(upper > lower)):
        raise ValueError("upper does not exceed lower; give the interval's ends in order.")


def check_samples_are_usable(what: str, array: Array, in_support: Callable[[Array], Array]) -> None:
    """Samples to fit are at least two, finite and inside the support."""
    if array.size < 2:
        raise ValueError(f"{what}: {array.size} sample(s) cannot be fitted; give at least two.")
    bad = ~(jnp.isfinite(array) & in_support(array))
    if bool(jnp.any(bad)):
        raise ValueError(
            f"{what}: {int(bad.sum())} of {array.size} samples are missing or outside the "
            "support; exclude and count them before fitting."
        )


def check_samples_vary(what: str, std: Array) -> None:
    """Samples to fit are not all equal, or no scale can be fitted."""
    if not bool(std > 0):
        raise ValueError(f"{what}: the samples are all equal, so no scale can be fitted; give samples that vary.")


def check_center_is_on_the_simplex(center: Array) -> None:
    """A softmax-normal's center is a point of the simplex, per label or shared."""
    if center.ndim not in (1, 2) or center.shape[-1] < 2:
        raise ValueError(f"softmax_normal: center has shape {center.shape}; give (k,) or (n, k) with k >= 2.")
    if not bool(jnp.all(center > 0)) or not bool(jnp.allclose(center.sum(axis=-1), 1.0, atol=1e-8)):
        raise ValueError("softmax_normal: center is not a point of the simplex; give positive fractions summing to 1.")


def check_logit_sd_fits_the_center(logit_sd: Array, loc: Array) -> None:
    """A ``logit_sd`` has a shape that reads one way against the center: an
    ``(n,)`` one equal to ``(k - 1,)`` would be laid along the wrong axis."""
    per_number, per_label = loc.shape[-1:], loc.shape[:-1]
    allowed = {(), per_number} | ({per_label, loc.shape} if loc.ndim == 2 else set())
    if logit_sd.shape not in allowed:
        raise ValueError(
            f"softmax_normal: logit_sd has shape {logit_sd.shape}, which fits the center of shape "
            f"{(*loc.shape[:-1], loc.shape[-1] + 1)} as none of {sorted(allowed)}; give a scalar, one "
            "value per unconstrained number, or, for a center per label, one per label or one per "
            "label and number."
        )
    if loc.ndim == 2 and per_label == per_number and logit_sd.shape == per_number:
        raise ValueError(
            f"softmax_normal: logit_sd of shape {logit_sd.shape} could be one value per label "
            "or one per unconstrained number, since there are as many of each; give it as "
            f"{loc.shape}."
        )
