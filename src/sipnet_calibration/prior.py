"""The prior over a parameter vector: what is believed before the data.

Where this sits
---------------
::

    parameter_vector.ParameterVector      (the unknowns x, and theta = T^-1(x))
      -> prior.Prior                      (pi(x), and so the density of theta)
      -> pyEKI (sample, gaussian), an MCMC target (log_prob)

This module imports TFP's distributions and pyEKI's ``Gaussian`` and
operators, and not pySIPNET.

What it reads
-------------
A :class:`~sipnet_calibration.parameter_vector.ParameterVector` and its
:class:`PriorTerm`\\ s. A term covers one parameter, keyed by its name, or
several that share one dim or none, keyed by a tuple of their names (a
**joint term**), and it may be **given** other parameters or derived
parameters. Its distribution is the prior of the whole natural value of
what it covers:

- **one parameter without a dim, given nothing**: a TFP distribution whose
  event shape is the parameter's value shape, with batch shape ``()``, in
  ``float64``; or a :data:`PriorFunction` called with ``dim_index=None``;
- **one parameter with a dim**: a :data:`PriorFunction`, called as
  ``f(dim_index, site_table)`` with the vector's
  :meth:`~sipnet_calibration.parameter_vector.ParameterVector.dim_index` and
  site table, returning such a distribution over every dim label at once.
  :func:`iid_over_dim` and :func:`independent_over_dim` make one;
- **a joint term**: the same, with draws that are dicts keyed by the names
  covered, each of its parameter's value shape. :func:`gaussian_copula`
  makes one;
- **a term given others**: a :data:`PriorFunction` called as
  ``f(dim_index, site_table, **given_values)`` with one draw's natural
  value of each name given, for one draw at a time.

The distributions of the family builders (:func:`log_normal`,
:func:`logit_normal`, :func:`softmax_normal` and their ``_from_*`` forms)
are pushforwards of a Gaussian through their support's default bijector.

The density
-----------
With parameters :math:`x`, derived parameters :math:`y`, terms
:math:`B_1, \\dots, B_m` and :math:`g(b)` what term :math:`b` is given,

.. math::

    \\pi(x) = \\prod_{b=1}^{m} \\pi_b\\big(x_{B_b} \\mid x_{g(b)}, y_{g(b)}\\big),

and with coordinates :math:`\\theta = T^{-1}(x)`,

.. math::

    \\log \\pi_\\theta(\\theta) = \\sum_b \\Big[ \\log \\pi_b\\big(T(\\theta)_{B_b} \\mid
        T(\\theta)_{g(b)}, y_{g(b)}\\big) + \\sum_{p \\in B_b} \\log J_p(\\theta_p) \\Big],

each density and :math:`J_p` against the reference measure of
:meth:`~sipnet_calibration.parameter_vector.Support.log_jacobian`.
:meth:`Prior.log_prob` evaluates each term by its base density where it is a
pushforward through its parameters' own bijectors, and by change of
variables otherwise. The ``given`` links, with each derived parameter linked
to what it is computed from, must form a directed acyclic graph; terms are
drawn in its topological order.

A term's :math:`\\theta_{B_b}` is its parameters' entries of theta in the
order of its key, each parameter's in theta's order. A joint term's base,
when it is evaluated by its base density, has that vector as its event.

Functions and classes
---------------------
:class:`Prior`
    ``sample``, ``log_prob``, ``gaussian``, ``select``, ``describe``.
:class:`PriorTerm`, :data:`PriorFunction`
    One term's prior, what it is given, and its provenance.
:func:`iid_over_dim`, :func:`independent_over_dim`
    Priors over a dim, independent across dim labels.
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

**A prior function must marginalize consistently.** ``Prior(vector.select(...),
prior.terms)`` rebuilds each term on fewer dim labels, which is the exact
marginal only if the distribution at a set of dim labels is the marginal of
the one at any larger set. The builders here satisfy this, and so does a
term whose dim labels are independent given what it is given. A
user-written function that normalizes over the labels present, or
standardizes a covariate inside, does not, and changes meaning under
``select``.

Usage
-----
::

    import jax
    from sipnet_calibration.prior import (
        Prior, PriorTerm, gaussian_copula, iid_over_dim, log_normal_from_interval,
        logit_normal, softmax_normal,
    )

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
            iid_over_dim(log_normal_from_interval(lower=5e3, upper=8e4)), provenance="..."),
    })
    theta = prior.sample(jax.random.key(0), 50)   # (50, D)
    prior.log_prob(theta)                         # (50,)
    prior.gaussian()                              # pyeki.gauss.Gaussian, exact here
    prior.describe()                              # one row per term

A centered hierarchy, a site-level value given its PFT's mean and a shared
spread::

    def soil_carbon_given_pft(dim_index, site_table, mean, spread):
        loc = mean[site_positions(site_table, "pft")]
        return tfd.TransformedDistribution(
            tfd.Independent(tfd.Normal(loc, spread), 1), tfb.Exp())

    PriorTerm(soil_carbon_given_pft, given=("mean", "spread"), provenance="...")
"""

from __future__ import annotations

import dataclasses
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from frozendict import frozendict
from pyeki.gauss import Gaussian
from pyeki.linalg import DensePSD, PSDBlockDiag, PSDDiagonal, PSDLinOp
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.parameter_vector import (
    OPEN_UNIT_INTERVAL,
    Parameter,
    ParameterVector,
    Support,
    bijectors_agree,
    check_theta_ends_in_the_dimension,
    joint_probe_points,
)
from sipnet_calibration.validation import as_bounded_integer, as_names, truncated

__all__ = [
    "DeclaresGaussian",
    "Prior",
    "PriorFunction",
    "PriorTerm",
    "TermKey",
    "check_prior_term_is_valid",
    "gaussian_copula",
    "independent_over_dim",
    "iid_over_dim",
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

#: A prior over a whole natural value, built for the vector at hand: called as
#: ``f(dim_index, site_table, **given_values)`` and returning a TFP
#: distribution.
type PriorFunction = Callable[..., tfd.Distribution]

#: What a term covers: one parameter's name, or a tuple of names for a joint
#: term.
type TermKey = str | tuple[str, ...]


# ── the prior ─────────────────────────────────────────────────────────────────


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

    Raises
    ------
    TypeError
        If a key is not a name or a tuple of names, a term is not a
        :class:`PriorTerm`, a parameter with a dim is given a bare
        distribution, or a term given others is not a function.
    KeyError
        If a key or a ``given`` names nothing of the vector.
    ValueError
        If a parameter is covered by no term or by two, a joint term's
        parameters have different dims, the ``given`` links form a cycle, or
        a term fails a check of :func:`check_prior_term_is_valid`; the
        message names the term.
    """

    parameter_vector: ParameterVector
    terms: Mapping[TermKey, PriorTerm]
    _built: Mapping[TermKey, _BuiltTerm] = field(init=False, repr=False)
    _draw_order: tuple[TermKey, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", frozendict(self.terms))
        check_terms_cover_the_parameters(self.terms, self.parameter_vector)
        for key, term in self.terms.items():
            check_term_is_a_prior_term(term_name(key), term)
            check_given_names_are_held(term_name(key), term.given, self.parameter_vector)
        order = _draw_order(self.terms, self.parameter_vector)
        object.__setattr__(self, "_draw_order", order)
        object.__setattr__(self, "_built", frozendict(self._build_terms(order)))

    # ── identity ──────────────────────────────────────────────────────────────

    def __getitem__(self, key: TermKey) -> PriorTerm:
        """The term keyed by *key*: a parameter's name, or a joint term's tuple."""
        check_term_is_held(key, self)
        return self.terms[key]

    def __repr__(self) -> str:
        return f"Prior(D={self.parameter_vector.dimension}, terms={list(self.terms)})"

    def describe(self) -> pd.DataFrame:
        """One row per term, in theta's order, indexed by ``parameters`` (a
        joint term's names joined with ``"+"``): ``prior``, ``given``
        (comma-separated), ``evaluated_by`` (``"base density"`` or ``"change
        of variables"``), ``declared_gaussian`` and ``provenance``."""
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
        """``Prior(vector.select(**selectors), kept terms)``: the prior of a
        smaller vector, each term rebuilt on its dim labels.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`ParameterVector.select` and :class:`Prior`;
            ``ValueError`` too if the selection drops a parameter a kept term
            covers or is given, directly or through a derived parameter.
        """
        vector = self.parameter_vector.select(**selectors)
        kept = {*vector.parameter_names, *vector.derived_parameter_names}
        terms = {}
        for key, term in self.terms.items():
            names = _names_of(key)
            if any(name in kept for name in names):
                check_selection_keeps_what_a_term_needs(term_name(key), (*names, *term.given), kept)
                terms[key] = term
        return Prior(vector, terms)

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
        n = as_bounded_integer(n, minimum=0, message_name="n")
        theta = jnp.zeros((n, self.parameter_vector.dimension), dtype=jnp.float64)
        natural_values: dict[str, Array] = {}
        for term_key in self._draw_order:
            built = self._built[term_key]
            given_values = self._given_values(built, natural_values)
            theta_b = built.sample_theta(_term_key(key, built.name), n, given_values)
            check_draws_map_to_finite_theta(built.name, theta_b)
            theta = theta.at[:, built.positions].set(theta_b)
            natural_values |= built.natural_values(theta_b)
        return theta

    def log_prob(self, theta: Any) -> Array:
        """:math:`\\log \\pi_\\theta(\\theta)`, ``(..., D) -> (...)``.

        Each term contributes, at its :math:`\\theta_B` and given the
        natural values :math:`v = (x_{g(b)}, y_{g(b)})` at theta,

        - by **base density**, when its distribution is
          ``TransformedDistribution(base, b)`` with ``b`` equal, at the probe
          points, to the map :math:`\\theta_B \\mapsto \\{p: T_p(\\theta_p)\\}`
          of its parameters' own bijectors:
          :math:`\\log \\mathrm{base}(\\theta_B \\mid v)`, exactly;
        - by **change of variables** otherwise:
          :math:`\\log \\pi_b(T(\\theta_B) \\mid v) + \\sum_{p \\in B}
          \\sum_i \\log J_p(\\theta_p)_i`, summed over each parameter's
          numbers (over its dim labels on the simplex), with
          :math:`\\log J_p` from
          :meth:`~sipnet_calibration.parameter_vector.Support.log_jacobian`.

        A term given others is evaluated one draw at a time under
        ``jax.vmap``. Traceable under ``jax.jit``, ``jax.grad`` and
        ``jax.vmap``.

        Raises
        ------
        ValueError
            If the last axis of *theta* is not ``D`` long.
        """
        theta = jnp.asarray(theta, dtype=jnp.float64)
        check_theta_ends_in_the_dimension(theta.shape, self.parameter_vector.dimension)
        natural_values = self._natural_values_given(theta)
        total = jnp.zeros(theta.shape[:-1], dtype=jnp.float64)
        for built in self._built.values():
            given_values = self._given_values(built, natural_values)
            total = total + built.log_prob(theta[..., built.positions], given_values)
        return total

    def gaussian(self, *, key: Array | None = None, n_moment_samples: int = 0) -> Gaussian:
        """The prior as a Gaussian over theta, ``pyeki.gauss.Gaussian``.

        The covariance is a ``PSDBlockDiag`` with one block per **dependent
        set**, in theta's order. Two parameters are in one dependent set when
        one term covers both, or a term covering one is given the other,
        directly or through a derived parameter it is computed from; the
        prior makes different dependent sets independent. A dependent set
        whose one term declares its Gaussian, honored, gets that exact
        :math:`(m, C)`. Any other is moment-matched from
        :math:`M` = *n_moment_samples* joint draws :math:`\\theta^{(i)}` of
        :meth:`sample` with *key*, restricted to the set's entries:

        .. math::

            \\hat m = \\frac{1}{M} \\sum_i \\theta^{(i)}, \\qquad
            \\hat C = \\frac{1}{M - 1} \\sum_i (\\theta^{(i)} - \\hat m)
                     (\\theta^{(i)} - \\hat m)^\\top.

        Raises
        ------
        NotImplementedError
            If a dependent set needs moment matching and no *key* was given.
        ValueError
            If a dependent set is not one contiguous run of the vector's
            parameters, which a block-diagonal covariance needs, or a
            moment-matched block has at least *n_moment_samples* entries:
            :math:`\\hat C` then has rank at most :math:`M - 1` and is
            singular.
        """
        n_moment_samples = as_bounded_integer(n_moment_samples, minimum=0, message_name="n_moment_samples")
        vector = self.parameter_vector
        sets = self._dependent_sets()
        for names in sets:
            check_dependent_set_is_contiguous(names, vector)
        draws = None
        means, blocks = [], []
        for names in sets:
            terms = [b for b in self._built.values() if b.names[0] in names]
            positions = np.concatenate([vector.positions(parameter_name=n) for n in names])
            if len(terms) == 1 and terms[0].declared is not None:
                mean, block = _in_theta_order(terms[0].declared, terms[0].positions)
            else:
                check_moment_matching_is_possible(names, len(positions), key, n_moment_samples)
                if draws is None:
                    draws = self.sample(key, n_moment_samples)
                mean, block = _moment_matched(draws[:, positions])
            means.append(jnp.asarray(mean, dtype=jnp.float64))
            blocks.append(block)
        return Gaussian(mean=jnp.concatenate(means), cov=PSDBlockDiag(tuple(blocks)))

    # ── supporting methods ────────────────────────────────────────────────────

    def _build_terms(self, order: tuple[TermKey, ...]) -> dict[TermKey, _BuiltTerm]:
        """Every term built in draw order. When some term is given others,
        two ancestral draws are made as the terms are built, and each term
        given others is built and checked at both."""
        vector = self.parameter_vector
        ancestral = any(term.given for term in self.terms.values())
        natural_values: dict[str, Array] = {}
        built: dict[TermKey, _BuiltTerm] = {}
        for term_key in order:
            term = self.terms[term_key]
            given_values = None
            if term.given:
                natural_values |= _derived_needed(vector, term.given, natural_values)
                given_values = {name: natural_values[name] for name in term.given}
            built[term_key] = _BuiltTerm.build(term_key, term, vector, given_values)
            if ancestral:
                name = built[term_key].name
                theta_b = built[term_key].sample_theta(
                    _term_key(jax.random.key(_ANCESTRAL_SEED), name), _N_ANCESTRAL_DRAWS, given_values
                )
                check_draws_map_to_finite_theta(name, theta_b)
                natural_values |= built[term_key].natural_values(theta_b)
        return {key: built[key] for key in sorted(built, key=lambda k: int(built[k].positions.min()))}

    def _natural_values_given(self, theta: Array) -> dict[str, Array]:
        """The natural values at *theta* that some term is given, and those
        the derived ones among them are computed from."""
        vector = self.parameter_vector
        given = list(dict.fromkeys(name for built in self._built.values() for name in built.given))
        if not given:
            return {}
        needed = _parameters_behind(vector, given)
        derived = [name for name in given if name in vector.derived_parameter_names]
        natural_values = {
            p.name: p.bijector.forward(
                theta[..., vector.positions(parameter_name=p.name)].reshape(
                    theta.shape[:-1] + vector.unconstrained_shape(p.name)
                )
            )
            for p in vector.parameters
            if p.name in needed
        }
        return natural_values | vector.derived_values(natural_values, derived_parameter_names=derived)

    def _given_values(self, built: _BuiltTerm, natural_values: dict[str, Array]) -> dict[str, Array] | None:
        """What *built* is given, from *natural_values*, computing the
        derived parameters among it there; ``None`` when it is given nothing."""
        if not built.given:
            return None
        natural_values |= _derived_needed(self.parameter_vector, built.given, natural_values)
        return {name: natural_values[name] for name in built.given}

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
            linked = [*built.names, *sorted(_parameters_behind(vector, built.given))]
            for name in linked[1:]:
                root[find(name)] = find(linked[0])
        sets: dict[str, list[str]] = {}
        for name in vector.parameter_names:
            sets.setdefault(find(name), []).append(name)
        return [tuple(names) for names in sets.values()]


# ── its pieces ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class PriorTerm:
    """The prior of what a term covers, what it is given, and where it came
    from.

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
    provenance:
        Where the prior came from, with its citation; a placeholder says it
        is one.

    Raises
    ------
    TypeError
        If *given* is one string rather than a sequence of names.
    ValueError
        If *provenance* is empty.
    """

    distribution: tfd.Distribution | PriorFunction
    _: KW_ONLY
    given: tuple[str, ...] = ()
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "given", as_names(self.given, message_name="given"))
        check_provenance_is_given(self.provenance)


# ── priors over a dim, and joint priors ───────────────────────────────────────


def iid_over_dim(distribution: tfd.Distribution) -> PriorFunction:
    """Every dim label independently distributed as *distribution*:

    .. math::

        \\pi(x) = \\prod_{\\ell=1}^{n} \\pi_0(x_\\ell).

    A ``TransformedDistribution(base, b)`` (a family builder's, say) is
    built as ``TransformedDistribution(Sample(base, n), b)``, with the
    bijector outside, so that :meth:`Prior.log_prob` evaluates it by its base
    density; any other distribution as ``Sample(distribution, n)``. Over a
    family builder's distribution with Gaussian base :math:`(m_0, C_0)` it
    declares :math:`(\\mathbf 1_n \\otimes m_0,\\ I_n \\otimes C_0)`.

    Parameters
    ----------
    distribution:
        The prior of one dim label's value, batch shape ``()``.

    Returns
    -------
    PriorFunction
    """
    return _OverDim(
        per_dim_label=lambda dim_index: distribution,
        repeated=True,
        name=f"iid {_distribution_name(distribution)}",
    )


def independent_over_dim(
    distribution_family: Callable[..., tfd.Distribution], /, **arguments: Any
) -> PriorFunction:
    """Independent across dim labels, each dim label's distribution made by
    *distribution_family* from its own arguments:

    .. math::

        \\pi(x) = \\prod_{\\ell=1}^{n} \\pi_{\\mathrm{family}}\\big(x_\\ell;\\ a_\\ell\\big).

    Parameters
    ----------
    distribution_family:
        Builds a distribution from keyword arguments, such as
        :func:`log_normal`, and gives it batch shape ``(n,)`` when each
        argument has one entry per dim label.
    **arguments:
        Each either one value shared by every dim label, passed as it is, or
        a ``Mapping`` or ``pd.Series`` keyed by dim label, read at the
        vector's dim labels (a superset is allowed) and passed as an array
        with one leading entry per dim label. A transformed family is built
        with its bijector outside, as for :func:`iid_over_dim`. Over a family
        builder, with dim label :math:`\\ell`'s base Gaussian
        :math:`(m_\\ell, C_\\ell)`, it declares
        :math:`\\big((m_1, \\dots, m_n),\\ \\mathrm{blockdiag}(C_1, \\dots, C_n)\\big)`.

    Returns
    -------
    PriorFunction

    Raises
    ------
    KeyError
        When called, for a dim label a keyed argument lacks.
    ValueError
        When called, if the family's distribution is not a batch of one per
        dim label, as when no argument is keyed (:func:`iid_over_dim` is that
        prior).
    """

    def per_dim_label(dim_index: pd.Index) -> tfd.Distribution:
        aligned = {
            name: _aligned_argument(name, value, dim_index) for name, value in arguments.items()
        }
        distribution = distribution_family(**aligned)
        check_family_is_one_per_dim_label(distribution, len(dim_index))
        return distribution

    return _OverDim(
        per_dim_label=per_dim_label,
        repeated=False,
        name=f"independent {getattr(distribution_family, '__name__', 'family')}",
    )


def gaussian_copula(marginals: Mapping[str, tfd.Distribution], *, correlation: Any) -> PriorFunction:
    """The joint prior of scalar parameters without a dim whose marginals are
    *marginals* and whose unconstrained values are correlated Gaussians.

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
        :func:`logit_normal` (on any open interval), their ``_from_*`` forms,
        or a ``tfd.Normal``, with batch shape ``()``. List them in the order
        of the term's key, which is the order its base is laid out in.
    correlation:
        ``(m, m)``: symmetric, unit diagonal, positive definite.

    Returns
    -------
    PriorFunction
        Called with ``dim_index=None``; its draws are dicts keyed like
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
        Positive; one value per dim label gives a batch.
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


def logit_normal(
    *, median: Any, logit_sd: Any, support: Support = OPEN_UNIT_INTERVAL
) -> tfd.Distribution:
    """The logit-normal on the open interval *support*, :math:`(a, b)`:

    .. math::

        \\operatorname{logit}\\frac{x - a}{b - a} \\sim
            \\mathcal N\\Big(\\operatorname{logit}\\frac{\\mathrm{median} - a}{b - a},\\
            \\mathrm{logit\\_sd}^2\\Big).

    On :math:`(0, 1)` this is ``tfd.LogitNormal``; on :math:`(a, b)`,
    ``TransformedDistribution(Normal, Sigmoid(low=a, high=b))``. A
    ``logit_sd`` near 1.7 is close to flat.

    Raises
    ------
    ValueError
        If *median* is outside *support*, *logit_sd* is not positive, or
        *support* is not an interval.
    """
    check_support_is_an_interval(support)
    fraction = _interval_fraction("logit_normal median", median, support)
    logit_sd = _positive_array("logit_normal logit_sd", logit_sd)
    return _logit_normal_on(support, _logit(fraction), logit_sd)


def logit_normal_from_interval(
    *, lower: Any, upper: Any, mass: float = 0.95, support: Support = OPEN_UNIT_INTERVAL
) -> tfd.Distribution:
    """The logit-normal on *support* whose central *mass* interval is
    ``[lower, upper]``: :func:`log_normal_from_interval`'s formulas on the
    logit scale of :math:`(x - a)/(b - a)`.

    Raises
    ------
    ValueError
        If an end is outside *support*, ``upper <= lower``, or *mass* is not
        in ``(0, 1)``.
    """
    check_support_is_an_interval(support)
    lower = _interval_fraction("logit_normal_from_interval lower", lower, support)
    upper = _interval_fraction("logit_normal_from_interval upper", upper, support)
    loc, scale = _normal_from_interval(_logit(lower), _logit(upper), mass)
    return _logit_normal_on(support, loc, scale)


def logit_normal_from_samples(samples: Any, *, support: Support = OPEN_UNIT_INTERVAL) -> tfd.Distribution:
    """The maximum-likelihood logit-normal on *support*, :math:`(a, b)`, of
    samples inside it: with :math:`t = \\operatorname{logit}((x - a)/(b - a))`,
    :math:`\\mu = \\bar t` and :math:`\\sigma` the standard deviation of
    :math:`t`.

    Raises
    ------
    ValueError
        As :func:`log_normal_from_samples`, for samples outside *support*.
    """
    check_support_is_an_interval(support)
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
        for one per dim label.
    logit_sd:
        A scalar; one value per unconstrained number, ``(k - 1,)``; or, with
        an ``(n, k)`` center, one value per dim label, ``(n,)``, or one per
        dim label and unconstrained number, ``(n, k - 1)``.

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
        logit_sd = logit_sd[:, None]  # one per dim label, across its numbers
    scale = jnp.broadcast_to(logit_sd, loc.shape)
    return tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale), tfb.SoftmaxCentered()
    )


# ── the declared Gaussian and term evaluation ─────────────────────────────────


@runtime_checkable
class DeclaresGaussian(Protocol):
    """A prior builder that knows its term's Gaussian in theta.

    ``unconstrained_gaussian(dim_index, site_table, parameter_names)``
    returns ``(mean, covariance, assumed_bijectors)`` for the term covering
    *parameter_names* on these dim labels: the mean ``(D_b,)`` and a
    ``PSDLinOp`` covariance of :math:`\\theta_B`, laid out as the module
    docstring says, and the bijector each parameter's value is assumed to
    be the image of, keyed by those names. It returns ``None`` when it
    declares nothing for these dim labels. The declaration is honored only
    where every assumed bijector agrees with its parameter's at the probe
    points.
    """

    def unconstrained_gaussian(
        self, dim_index: pd.Index | None, site_table: pd.DataFrame, parameter_names: tuple[str, ...]
    ) -> tuple[Array, PSDLinOp, Mapping[str, tfb.Bijector]] | None: ...


@dataclass(frozen=True, eq=False)
class _OverDim:
    """A :data:`PriorFunction` over a dim that declares its Gaussian when
    each dim label's distribution is a family builder's.

    ``per_dim_label(dim_index)`` gives one dim label's distribution when
    *repeated*, else every dim label's as a batch of ``n``.
    """

    per_dim_label: Callable[[pd.Index], tfd.Distribution]
    repeated: bool
    name: str

    def __call__(self, dim_index: pd.Index | None, site_table: pd.DataFrame) -> tfd.Distribution:
        check_prior_over_a_dim_has_a_dim(dim_index, self.name)
        distribution = self.per_dim_label(dim_index)
        n = len(dim_index)
        if type(distribution) in _CARRIES_ITS_BIJECTOR:
            base = distribution.distribution
            base = tfd.Sample(base, n) if self.repeated else tfd.Independent(base, 1)
            return tfd.TransformedDistribution(base, distribution.bijector)
        return tfd.Sample(distribution, n) if self.repeated else tfd.Independent(distribution, 1)

    def unconstrained_gaussian(
        self, dim_index: pd.Index, site_table: pd.DataFrame, parameter_names: tuple[str, ...]
    ) -> tuple[Array, PSDLinOp, Mapping[str, tfb.Bijector]] | None:
        family = _family_gaussian(self.per_dim_label(dim_index))
        if family is None:
            return None
        loc, scale, bijector = family
        if self.repeated:
            n = len(dim_index)
            loc = jnp.broadcast_to(loc, (n, *loc.shape))
            scale = jnp.broadcast_to(scale, (n, *scale.shape))
        (name,) = parameter_names
        return jnp.ravel(loc), PSDDiagonal(jnp.ravel(scale) ** 2), {name: bijector}


@dataclass(frozen=True, eq=False)
class _GaussianCopula:
    """:func:`gaussian_copula`'s prior function, which declares its Gaussian."""

    names: tuple[str, ...]
    loc: Array
    scale: Array
    bijectors: tuple[tfb.Bijector, ...]
    correlation: Array
    name: str = "gaussian copula"

    def __call__(self, dim_index: pd.Index | None, site_table: pd.DataFrame) -> tfd.Distribution:
        check_prior_without_a_dim_has_none(dim_index, self.name)
        to_values = tfb.Chain([
            tfb.JointMap({n: tfb.Chain([b, tfb.Reshape([], [1])]) for n, b in zip(self.names, self.bijectors)}),
            tfb.Restructure({n: i for i, n in enumerate(self.names)}),
            tfb.Split(len(self.names)),
        ])
        base = tfd.MultivariateNormalTriL(self.loc, jnp.linalg.cholesky(self._covariance))
        return tfd.TransformedDistribution(base, to_values)

    def unconstrained_gaussian(
        self, dim_index: pd.Index | None, site_table: pd.DataFrame, parameter_names: tuple[str, ...]
    ) -> tuple[Array, PSDLinOp, Mapping[str, tfb.Bijector]]:
        # Laid out in the term's order, which a key listed in another order
        # than the marginals permutes.
        order = [self.names.index(n) for n in parameter_names]
        covariance = self._covariance[np.ix_(order, order)]
        return self.loc[np.asarray(order)], DensePSD(covariance), dict(zip(self.names, self.bijectors))

    @property
    def _covariance(self) -> Array:
        return self.scale[:, None] * self.correlation * self.scale[None, :]


@dataclass(frozen=True, eq=False)
class _GivenDirectly:
    """A family builder's distribution given directly for a parameter
    without a dim, which declares its own Gaussian."""

    distribution: tfd.Distribution

    def unconstrained_gaussian(
        self, dim_index: None, site_table: pd.DataFrame, parameter_names: tuple[str, ...]
    ) -> tuple[Array, PSDLinOp, Mapping[str, tfb.Bijector]] | None:
        family = _family_gaussian(self.distribution)
        if family is None:
            return None
        (name,) = parameter_names
        return jnp.ravel(family[0]), PSDDiagonal(jnp.ravel(family[1]) ** 2), {name: family[2]}


@dataclass(frozen=True, eq=False)
class _BuiltTerm:
    """A term built for the vector at hand.

    Its theta is flat, ``(..., D_b)``, in key order. A term given others
    keeps the distribution at the first ancestral draw, for its checks and
    its description, and rebuilds it per draw to evaluate and sample.
    """

    key: TermKey
    parameters: tuple[Parameter, ...]
    given: tuple[str, ...]
    source: tfd.Distribution | PriorFunction
    dim_index: pd.Index | None
    site_table: pd.DataFrame
    distribution: tfd.Distribution
    positions: np.ndarray
    shapes: tuple[tuple[int, ...], ...]
    by_base_density: bool
    declared: tuple[Array, PSDLinOp] | None

    @classmethod
    def build(
        cls,
        key: TermKey,
        term: PriorTerm,
        vector: ParameterVector,
        given_values: Mapping[str, Array] | None,
    ) -> _BuiltTerm:
        """Build and check a term; one given others at each ancestral draw
        in *given_values* (``(draws, *value_shape)`` each)."""
        names = _names_of(key)
        parameters = tuple(vector[n] for n in names)
        dim = parameters[0].dim
        built = cls(
            key=key,
            parameters=parameters,
            given=term.given,
            source=term.distribution,
            dim_index=vector.dim_index(dim) if dim else None,
            site_table=vector.site_table,
            distribution=None,
            positions=np.concatenate([vector.positions(parameter_name=n) for n in names]),
            shapes=tuple(vector.unconstrained_shape(n) for n in names),
            by_base_density=False,
            declared=None,
        )
        check_term_given_others_is_a_function(built)
        probes = built.probes()
        expected = {n: vector.value_shape(n) for n in names} if built.joint else vector.value_shape(names[0])
        if term.given:
            draws = [{n: v[i] for n, v in given_values.items()} for i in range(_N_ANCESTRAL_DRAWS)]
            distributions = [built.distribution_at(values) for values in draws]
        else:
            distributions = [_distribution_for(built)]
        variants = []
        for distribution in distributions:
            check_term_is_over_the_whole_value(built.name, distribution, expected)
            variants.append(
                dataclasses.replace(
                    built, distribution=distribution, by_base_density=built.pushes_through(distribution, probes)
                )
            )
        check_term_keeps_its_structure(variants)
        for variant in variants:
            check_prior_term_is_valid(variant)
        built = variants[0]
        declared = None if term.given else _declaration(built, probes)
        if declared is not None:
            check_declaration_agrees_with_log_prob(built, declared, probes)
            object.__setattr__(built, "declared", declared)
        return built

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
        """:func:`joint_probe_points` over the covered parameters, flat,
        ``(n_probes, D_b)``."""
        parts = joint_probe_points([(s, p.unconstrained_size) for s, p in zip(self.shapes, self.parameters)])
        return jnp.asarray(np.concatenate([part.reshape((len(part), -1)) for part in parts], axis=-1))

    def distribution_at(self, given_values: Mapping[str, Array]) -> tfd.Distribution:
        """The distribution at one draw's given values."""
        return self.source(self.dim_index, self.site_table, **given_values)

    def split(self, theta: Array) -> dict[str, Array]:
        """Flat theta, ``(..., D_b)``, as each parameter's unconstrained value."""
        lead, out, start = theta.shape[:-1], {}, 0
        for parameter, shape in zip(self.parameters, self.shapes):
            size = int(np.prod(shape, dtype=int))
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

    def _log_prob_under(self, distribution: tfd.Distribution, theta: Array) -> Array:
        if self.by_base_density:
            return distribution.distribution.log_prob(self._base_event(theta))
        split = self.split(theta)
        values = {p.name: p.bijector.forward(split[p.name]) for p in self.parameters}
        density = distribution.log_prob(values if self.joint else values[self.names[0]])
        lead_ndim = theta.ndim - 1
        for parameter in self.parameters:
            log_jacobian = parameter.support.log_jacobian(parameter.bijector, split[parameter.name])
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
        pieces = [p.bijector.inverse(values[p.name]).reshape(sample_shape + (-1,)) for p in self.parameters]
        return jnp.concatenate(pieces, axis=-1)

    def _base_event(self, theta: Array) -> Array:
        """Flat theta as the base's event: a joint term's is flat, one
        parameter's is its unconstrained shape."""
        if self.joint:
            return theta
        return theta.reshape(theta.shape[:-1] + self.shapes[0])


def _declaration(built: _BuiltTerm, probes: Array) -> tuple[Array, PSDLinOp] | None:
    """The term's declared Gaussian, when its builder declares one and every
    covered parameter's bijector agrees with the one it assumes."""
    source = built.source
    if isinstance(source, DeclaresGaussian):
        declarer = source
    elif isinstance(source, tfd.Distribution) and not built.joint:
        declarer = _GivenDirectly(source)
    else:
        return None
    declared = declarer.unconstrained_gaussian(built.dim_index, built.site_table, built.names)
    if declared is None:
        return None
    mean, covariance, assumed = declared
    split = built.split(probes)
    if not all(bijectors_agree(assumed[p.name], p.bijector, split[p.name]) for p in built.parameters):
        return None
    return mean, covariance


def _family_gaussian(distribution: tfd.Distribution) -> tuple[Array, Array, tfb.Bijector] | None:
    """A family builder's distribution as its base's mean and standard
    deviations, broadcast to ``(*batch, e?)``, and its bijector; ``None`` for
    any other distribution. Recognized by exact class."""
    kind = type(distribution)
    if kind is tfd.Normal:
        found = distribution.loc, distribution.scale, tfb.Identity()
    elif kind in (tfd.LogNormal, tfd.LogitNormal):
        base = distribution.distribution
        found = base.loc, base.scale, distribution.bijector
    elif kind is tfd.TransformedDistribution and type(distribution.distribution) is tfd.Normal and type(
        distribution.bijector
    ) is tfb.Sigmoid:
        base = distribution.distribution
        found = base.loc, base.scale, distribution.bijector
    elif kind is tfd.TransformedDistribution and type(
        distribution.distribution
    ) is tfd.MultivariateNormalDiag and type(distribution.bijector) is tfb.SoftmaxCentered:
        base = distribution.distribution
        found = base.mean(), base.stddev(), distribution.bijector
    else:
        return None
    loc, scale, bijector = found
    loc, scale = jnp.broadcast_arrays(jnp.asarray(loc), jnp.asarray(scale))
    return loc, scale, bijector


# ── private helpers ───────────────────────────────────────────────────────────

#: TFP's classes whose ``.distribution`` and ``.bijector`` are a base in theta
#: and a map from it. Subclasses such as ``MultivariateNormalTriL`` carry an
#: internal reparameterization instead, so only the exact classes count.
_CARRIES_ITS_BIJECTOR = (tfd.TransformedDistribution, tfd.LogNormal, tfd.LogitNormal)

#: The number of draws of the draw-based support check, and the size of each
#: batch of them, which bounds its memory for a term over many dim labels.
_SUPPORT_DRAWS, _SUPPORT_DRAW_BATCH = 10_000, 1_000

#: The seed of the draw-based support check.
_SUPPORT_SEED = 20260927

#: The seed and number of the ancestral draws a term given others is built
#: and checked at.
_ANCESTRAL_SEED, _N_ANCESTRAL_DRAWS = 20260928, 2

#: The declaration check's relative and absolute tolerances; see
#: :func:`check_declaration_agrees_with_log_prob`.
_DECLARATION_RELATIVE_TOLERANCE, _DECLARATION_ABSOLUTE_TOLERANCE = 1e-10, 1e-12


def _names_of(key: TermKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


def _term_key(key: Array, name: str) -> Array:
    return jax.random.fold_in(key, zlib.crc32(name.encode()))


def _parameters_behind(vector: ParameterVector, names: Sequence[str]) -> set[str]:
    """The parameters among *names*, and those the derived parameters among
    them are computed from, directly or through others."""
    derived = {d.name: d for d in vector.derived_parameters}
    out, pending = set(), list(names)
    while pending:
        name = pending.pop()
        if name in derived:
            pending.extend(derived[name].derived_from)
        else:
            out.add(name)
    return out


def _derived_needed(
    vector: ParameterVector, names: Sequence[str], natural_values: Mapping[str, Array]
) -> dict[str, Array]:
    """The derived parameters among *names* that *natural_values* lacks,
    computed from it."""
    missing = [n for n in names if n in vector.derived_parameter_names and n not in natural_values]
    if not missing:
        return {}
    return vector.derived_values(natural_values, derived_parameter_names=missing)


def _draw_order(terms: Mapping[TermKey, PriorTerm], vector: ParameterVector) -> tuple[TermKey, ...]:
    """The terms in a topological order of the ``given`` links, each after
    everything it is given: the covering term of a parameter, and, through a
    derived parameter, the covering terms of what it is computed from."""
    owner = {name: key for key in terms for name in _names_of(key)}
    derived = {d.name: d for d in vector.derived_parameters}

    def links(node: tuple[str, Any]) -> list[tuple[str, Any]]:
        kind, value = node
        names = terms[value].given if kind == "term" else derived[value].derived_from
        return [("derived", n) if n in derived else ("term", owner[n]) for n in names]

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


def _in_theta_order(declared: tuple[Array, PSDLinOp], positions: np.ndarray) -> tuple[Array, PSDLinOp]:
    """A declared Gaussian over a term's theta, in key order, reordered into
    theta's order."""
    mean, covariance = declared
    if np.all(np.diff(positions) > 0):
        return mean, covariance
    order = np.argsort(positions)
    return mean[order], DensePSD(covariance.to_dense()[np.ix_(order, order)])


def _moment_matched(draws: Array) -> tuple[Array, PSDLinOp]:
    mean = draws.mean(axis=0)
    centered = draws - mean
    return mean, DensePSD(centered.T @ centered / (len(draws) - 1))


def _distribution_for(built: _BuiltTerm) -> tfd.Distribution:
    """The distribution of a term given nothing: called when a function."""
    source = built.source
    if isinstance(source, tfd.Distribution):
        check_bare_distribution_has_no_dim(built)
        return source
    return source(built.dim_index, built.site_table)


def _aligned_argument(name: str, value: Any, dim_index: pd.Index) -> Any:
    """One argument of :func:`independent_over_dim`: a keyed one as one entry
    per dim label, a shared one as it is."""
    if not isinstance(value, (Mapping, pd.Series)):
        return value
    check_argument_covers_the_dim_labels(name, value, dim_index)
    return jnp.asarray(np.asarray([value[label] for label in dim_index], dtype=np.float64))


def _logit_normal_on(support: Support, loc: Array, scale: Array) -> tfd.Distribution:
    if (support.low, support.high) == (0.0, 1.0):
        return tfd.LogitNormal(loc=loc, scale=scale)
    return tfd.TransformedDistribution(tfd.Normal(loc, scale), support.bijector())


def _interval_fraction(what: str, value: Any, support: Support) -> Array:
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
    check_interval_is_valid(lower, upper, mass)
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


def _shape_of(shape: Any) -> Any:
    """A TFP shape, or a dict of them for a joint distribution, as tuples."""
    if isinstance(shape, Mapping):
        return {name: tuple(value) for name, value in shape.items()}
    return tuple(shape)


def _inside_at_each_probe(built: _BuiltTerm, values: Mapping[str, Array], *, closure: bool = False) -> Array:
    """``(n_probes,)``: whether every covered parameter's value lies in its
    support (its closure when *closure*)."""
    inside = [
        p.support.contains(values[p.name], closure=closure).reshape((len(values[p.name]), -1)).all(axis=-1)
        for p in built.parameters
    ]
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
    """A term's prior has the supports its parameters declare, and on the
    simplex a density :meth:`Prior.log_prob` can evaluate."""
    check_simplex_density_is_a_dirichlet(built)
    check_declared_support_lies_in_the_priors(built)
    check_priors_support_lies_in_the_declared(built)


def check_terms_cover_the_parameters(terms: Mapping[Any, Any], vector: ParameterVector) -> None:
    """The terms are keyed by the vector's parameter names, or tuples of
    them over one dim, and cover every parameter once, so no density is
    counted twice or left out."""
    owner: dict[str, str] = {}
    for key in terms:
        check_term_key_is_names(key)
        for name in _names_of(key):
            check_term_covers_a_parameter_once(name, term_name(key), vector, owner)
            owner[name] = term_name(key)
        check_joint_term_shares_a_dim(key, vector)
    check_every_parameter_has_a_term(owner, vector)


def check_term_key_is_names(key: Any) -> None:
    """A term is keyed by a parameter's name or a tuple of names, which is
    how it is read."""
    if not (isinstance(key, str) or (isinstance(key, tuple) and all(isinstance(n, str) for n in key))):
        raise TypeError(
            f"a prior term is keyed by a parameter's name, or a tuple of names for a joint term, "
            f"got {key!r}."
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


def check_joint_term_shares_a_dim(key: TermKey, vector: ParameterVector) -> None:
    """A joint term's parameters share one dim or none: dependence across
    dims is what ``given`` expresses."""
    dims = {vector[name].dim for name in _names_of(key)}
    if len(dims) > 1:
        raise ValueError(
            f"the joint term {term_name(key)!r} covers parameters on the dims "
            f"{sorted(map(str, dims))}; a joint term's parameters share one dim or none, so give "
            "each dim its own term and link them with given=."
        )


def check_given_names_are_held(name: str, given: Sequence[str], vector: ParameterVector) -> None:
    """What a term is given is a parameter or derived parameter of the
    vector, which it would otherwise be passed no value for."""
    for given_name in given:
        if given_name not in vector and given_name not in vector.derived_parameter_names:
            raise KeyError(
                f"the prior of {name!r} is given {given_name!r}, which is no parameter or derived "
                "parameter of the vector."
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


def check_term_given_others_is_a_function(built: _BuiltTerm) -> None:
    """A term given others is a function of their values; a distribution
    would ignore them."""
    if built.given and isinstance(built.source, tfd.Distribution):
        raise TypeError(
            f"the prior of {built.name!r} is given {list(built.given)} but is a distribution; give "
            "a function f(dim_index, site_table, **given_values) that returns one."
        )


def check_term_keeps_its_structure(variants: Sequence[_BuiltTerm]) -> None:
    """A term given others has one structure at every draw, depending on the
    given values only through its parameters, which evaluating it one draw
    at a time needs."""
    first, *rest = variants
    for variant in rest:
        if (
            jax.tree_util.tree_structure(variant.distribution)
            != jax.tree_util.tree_structure(first.distribution)
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


def check_dependent_set_is_contiguous(names: Sequence[str], vector: ParameterVector) -> None:
    """A dependent set is one contiguous run of the vector's parameters,
    which one block of a block-diagonal covariance needs."""
    positions = sorted(vector.parameter_names.index(n) for n in names)
    if positions[-1] - positions[0] + 1 != len(positions):
        raise ValueError(
            f"the dependent set {list(names)} is not contiguous in the vector's parameters, so its "
            "block of the Gaussian cannot be formed; declare those parameters next to each other."
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


def check_bare_distribution_has_no_dim(built: _BuiltTerm) -> None:
    """A term over a dim takes a prior function, since its prior depends on
    the dim labels present."""
    if built.dim_index is not None:
        raise TypeError(
            f"the prior of {built.name!r} varies over {built.dim_index.name!r}, so it is built for "
            "the dim labels present; wrap the distribution as iid_over_dim(distribution) or "
            "independent_over_dim(family, ...)."
        )


def check_prior_over_a_dim_has_a_dim(dim_index: Any, name: str) -> None:
    """A prior over a dim is given to a parameter with a dim."""
    if dim_index is None:
        raise TypeError(
            f"{name} is a prior over a dim, given to a parameter without one; give that "
            "parameter the distribution itself."
        )


def check_prior_without_a_dim_has_none(dim_index: Any, name: str) -> None:
    """A prior of parameters without a dim is not given parameters with one,
    whose draws would otherwise be refused for their shape, far from the
    reason."""
    if dim_index is not None:
        raise TypeError(
            f"{name} is a prior of parameters without a dim, given parameters on "
            f"{dim_index.name!r}; give those parameters a prior over their dim."
        )


def check_term_is_over_the_whole_value(name: str, distribution: Any, expected: Any) -> None:
    """A term's distribution is ``float64`` over the whole value it covers:
    event shape the value shape (a dict of them for a joint term), batch
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
            f"the prior of {name!r} has event shape {event}, batch shape {batch} and dtype "
            f"{distribution.dtype}, but must be a float64 distribution over the whole value it "
            f"covers: event shape {expected}, batch shape (). A batch of priors, one per dim "
            "label, is iid_over_dim or independent_over_dim; a joint term's draws are a dict "
            "keyed by the names it covers."
        )


def check_simplex_density_is_a_dirichlet(built: _BuiltTerm) -> None:
    """On the simplex, a term evaluated by change of variables is one
    parameter's ``Dirichlet``, whose density is against the first ``k - 1``
    coordinates, as the Jacobian is."""
    if built.by_base_density or not any(p.support.kind == "simplex" for p in built.parameters):
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


def check_declaration_agrees_with_log_prob(
    built: _BuiltTerm, declared: tuple[Array, PSDLinOp], probes: Array
) -> None:
    """A declared Gaussian's log density :math:`\\log q` equals the term's,
    :math:`\\log p`, at every probe point, to
    :math:`|\\log q - \\log p| \\le 10^{-10} |\\log p| + 10^{-12} D_b`, with
    :math:`D_b` the term's number of entries of theta."""
    mean, covariance = declared
    declared_log_density = Gaussian(mean=mean, cov=covariance).log_density(probes)
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
            "log_normal, logit_normal, their _from_* forms, or tfd.Normal, with batch shape ()."
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


def check_argument_covers_the_dim_labels(name: str, value: Mapping[Any, Any], dim_index: pd.Index) -> None:
    """A keyed argument has a value for every dim label."""
    missing = [label for label in dim_index if label not in value]
    if missing:
        raise KeyError(
            f"independent_over_dim argument {name!r} has no value for dim label(s) "
            f"{truncated(missing)}; key it by every dim label."
        )


def center_shape_of(loc: Array) -> tuple[int, ...]:
    """The center's shape, from the base's location: one more number."""
    return (*loc.shape[:-1], loc.shape[-1] + 1)


def check_family_is_one_per_dim_label(distribution: Any, n_dim_labels: int) -> None:
    """A family's distribution is a batch of one per dim label, which the dim
    labels would otherwise be misaligned with."""
    batch = tuple(getattr(distribution, "batch_shape", ()))
    if batch != (n_dim_labels,):
        raise ValueError(
            f"independent_over_dim built a distribution of batch shape {batch} for "
            f"{n_dim_labels} dim labels; key at least one argument by dim label, or use "
            "iid_over_dim for one distribution shared by every dim label."
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
    """A logit-normal's median or end lies inside its support."""
    if not bool(jnp.all((fraction > 0) & (fraction < 1))):
        raise ValueError(f"{what} is {value!r}, outside {support.name}; give a value inside it.")


def check_support_is_an_interval(support: Any) -> None:
    """A logit-normal's support is an open interval."""
    if not isinstance(support, Support):
        raise TypeError(f"support must be a Support, got {type(support).__name__}; use OpenInterval(low, high).")
    if support.kind != "interval":
        raise ValueError(
            f"a logit-normal is on an open interval, got support {support!r}; use "
            "OPEN_UNIT_INTERVAL or OpenInterval(low, high)."
        )


def check_interval_is_valid(lower: Array, upper: Array, mass: float) -> None:
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
    """A softmax-normal's center is a point of the simplex, per dim label or shared."""
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
            f"{center_shape_of(loc)} as none of {sorted(allowed)}; give a scalar, one value per "
            "unconstrained number, or, for a center per dim label, one per dim label or one per "
            "dim label and number."
        )
    if loc.ndim == 2 and per_label == per_number and logit_sd.shape == per_number:
        raise ValueError(
            f"softmax_normal: logit_sd of shape {logit_sd.shape} could be one value per dim label "
            "or one per unconstrained number, since there are as many of each; give it as "
            f"{loc.shape}."
        )
