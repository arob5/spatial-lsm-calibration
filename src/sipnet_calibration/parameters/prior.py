"""The prior over a parameter vector: what is believed before the data.

Where this sits
---------------
::

    parameters.vector.ParameterVector        (the unknowns x, and theta = T^-1(x))
    parameters.derived.DerivedParameters     (y = f(x), which a term may be given)
    parameters.prior_functions, families     (a term's distribution)
      -> parameters.prior.Prior              (pi(x), and so the density of theta)
      -> a sampler or an ensemble method      (sample, log_prob)

It imports TFP's distributions, and nothing of the package outside
``parameters``.

What it reads
-------------
A :class:`~sipnet_calibration.parameters.vector.ParameterVector`, optionally
its :class:`~sipnet_calibration.parameters.derived.DerivedParameters`, and
one :class:`PriorTerm` per parameter or group of parameters. A term is the
prior of the parameters it names, one or several indexed by the same dims
(a **joint term**); it may be **given** other parameters or derived
parameters, and read **constants**. Its distribution is over the term's
event, the parameters' **blocks**: every value of each parameter, at every
label of the dims it is indexed by, of block shape ``(*index shape,
*shape)``, where ``shape`` is one value's shape. :class:`PriorTerm` states
what a term's distribution or prior function receives and returns;
:mod:`~sipnet_calibration.parameters.prior_functions` builds prior
functions, and :mod:`~sipnet_calibration.parameters.families` one value's
distribution.

The density
-----------
With parameters :math:`x`, derived parameters :math:`y`, terms
:math:`b = 1, \\dots, m` over the parameters :math:`B_b`, and
:math:`x_{g(b)}` and :math:`y_{g(b)}` what term :math:`b` is given,

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
:meth:`Prior.log_prob` evaluates each term by its base density where it is
a pushforward through its parameters' own bijectors, which needs no
Jacobian, and by change of variables otherwise. The ``given`` links, with
each derived parameter linked to what it is given, must form a directed
acyclic graph; terms are drawn in its topological order.

A term's :math:`\\theta_{B_b}` is its parameters' unconstrained values in
the order of its parameter names, each parameter's in C order of its block.
A joint term's base, when it is evaluated by its base density, has that
vector as its event.

Functions and classes
---------------------
:class:`PriorTerm`
    One factor of the prior: its parameters, what it is given and reads,
    its distribution, and its provenance.
:class:`Prior`
    ``sample``, ``log_prob``, ``select``, ``describe``.

Usage
-----
::

    prior = Prior(vector, [
        PriorTerm(
            parameter_names=("respiration_share", "leaf_fall_fraction"),
            distribution=gaussian_copula(
                {"respiration_share": logit_normal(median=0.18, logit_sd=0.35),
                 "leaf_fall_fraction": logit_normal(median=0.5, logit_sd=1.0)},
                correlation=[[1.0, 0.3], [0.3, 1.0]]),
            provenance="..."),
        PriorTerm(
            parameter_names=("allocation",),
            distribution=iid_over_dim(softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5)),
            provenance="..."),
        PriorTerm(
            parameter_names=("initial_soil_carbon",),
            distribution=independent_over_dim(log_normal, geometric_sd=2.0),
            constants={"median": median_by_site}, provenance="..."),
    ])
    theta = prior.sample(jax.random.key(0), 50)   # (50, D)
    prior.log_prob(theta)                         # (50,)

A centered hierarchy, a site-level value given the location at its site,
a derived parameter ``pft_mean[pft_of_site]`` (see
:class:`~sipnet_calibration.parameters.derived.DerivedParameter`), and a
shared spread::

    def soil_carbon_given_its_mean(soil_carbon_log_mean, spread):
        normal = tfd.Independent(tfd.Normal(soil_carbon_log_mean, spread), 1)
        return tfd.TransformedDistribution(normal, tfb.Exp())

    PriorTerm(parameter_names=("soil_carbon",), given=("soil_carbon_log_mean", "spread"),
              distribution=soil_carbon_given_its_mean, provenance="...")
"""

from __future__ import annotations

import dataclasses
import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from frozendict import frozendict
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.parameters._distributions import (
    CARRIES_ITS_BIJECTOR,
    IndexShapedFunction,
    distribution_name,
)
from sipnet_calibration.parameters._probes import bijectors_agree, joint_probe_points
from sipnet_calibration.parameters._validation import (
    as_count,
    as_names,
    as_sequence,
    check_names_are_unique,
    truncated,
)
from sipnet_calibration.parameters.derived import DerivedParameters
from sipnet_calibration.parameters.labels import aligned_constants, as_constants
from sipnet_calibration.parameters.parameter import Parameter
from sipnet_calibration.parameters.prior_functions import PriorFunction
from sipnet_calibration.parameters.support import Interval, Simplex, Support
from sipnet_calibration.parameters.vector import (
    ParameterVector,
    check_flat_ends_in_the_size,
    check_parameter_vectors_share_a_layout,
)

__all__ = [
    "Prior",
    "PriorTerm",
]

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jax.Array


# ── the prior ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, kw_only=True)
class PriorTerm:
    """One factor of the prior: the distribution of some parameters'
    natural values, given the values of others.

    A term over parameters :math:`B`, given parameters and derived
    parameters :math:`g` and reading constants :math:`c`, is the
    conditional density

    .. math::

        \\pi_B\\big(x_B \\mid x_g;\\ c\\big),

    and a :class:`Prior` is the product of its terms. Its **event** is the
    parameters' blocks: every value of each, at every label of the dims it
    is indexed by, of block shape ``(*index shape, *shape)``. A term is a
    declaration: it holds no labels and does not know the vector. The prior
    it is given to reads its constants at the labels in use, calls its
    function there and evaluates the result.

    Parameters
    ----------
    parameter_names:
        The parameters the term is the prior of: one, or several indexed by
        the same dims (a **joint term**). Their order is the order of the
        term's unconstrained values in its part of theta.
    distribution:
        The distribution of the parameters' blocks, in one of two forms.

        - **A TFP distribution**, only for a term over one parameter that
          is indexed by nothing, is given nothing and reads no constant;
          its event shape is the parameter's ``shape``.
        - **A prior function**, in every case:
          ``distribution(**given, **constants)``. It is called for the
          labels in use and receives, by name, one draw's natural value of
          each name in *given*, an array of that parameter's or derived
          parameter's block shape, and each constant, read as *constants*
          says. It returns a ``float64`` distribution of TFP batch shape
          ``()`` whose event is the term's: an array of the block shape for
          one parameter, a dict of them keyed by *parameter_names* for a
          joint term. The event shape comes from what the function reads;
          where nothing it reads has the index dims, as for labels
          independent and identically distributed, it is
          :func:`~sipnet_calibration.parameters.prior_functions.iid_over_dim`.
          The function is traced by JAX, and vmapped over draws when the
          term is given something, so it is a pure function of its
          arguments.

        The builders of :mod:`~sipnet_calibration.parameters.prior_functions`
        return prior functions that the prior also passes the index shape,
        privately, since for labels independent and identically
        distributed nothing they read carries it;
        :mod:`~sipnet_calibration.parameters.families` builds one value's
        distribution.
    given:
        The parameters and derived parameters the distribution is
        conditioned on, none of them in *parameter_names*. Default none.
    constants:
        ``{name: xr.DataArray}``: values the function reads that are the
        same in every draw, such as a covariate per site, a prior median per
        site or a simplex center per PFT. Each is a constant as
        :mod:`~sipnet_calibration.parameters.labels` defines one, read at
        the labels of the vector's coords (with the derived parameters'
        dims) and of the element axes of the parameters and of what the
        term is given, with the parameters' ``indexed_by`` dims first, then
        their element axes. Default none.
    provenance:
        Where the prior came from, with its citation; a placeholder says it
        is one.

    Raises
    ------
    TypeError
        If *parameter_names* or *given* is one string rather than a sequence
        of names, *distribution* is neither a TFP distribution nor callable,
        or a constant is not a ``float64`` or ``bool`` DataArray.
    ValueError
        If *parameter_names* is empty, a name repeats across
        *parameter_names*, *given* and *constants*, or *provenance* is
        empty.

    The checks that need the vector are the :class:`Prior`'s: that the names
    exist, that a bare distribution's parameter is indexed by nothing, the
    event shape, the supports, and that the ``given`` links are acyclic.

    Notes
    -----
    **Indexing one dim by another is a derived parameter's job.** A centered
    hierarchy over sites within PFTs is given the location at each site, a
    derived parameter ``pft_mean[pft_of_site]``, rather than the PFT means
    and a membership. A fixed value per PFT is a constant per site, built
    before the term sees it, ``median_by_pft.sel(pft=pft_of_site)`` with
    ``pft_of_site`` each site's PFT on ``site``.
    A Gaussian process over sites reads its locations as constants, ``lon``
    and ``lat`` on ``site``.

    **A prior function must marginalize consistently.** ``Prior.select``
    calls it on fewer labels, which gives the marginal prior only if the
    distribution at a set of labels is the marginal of its distribution at
    any larger set. The builders satisfy this, and so does any term whose
    labels are independent given what it is given.
    """

    parameter_names: tuple[str, ...]
    distribution: tfd.Distribution | PriorFunction
    given: tuple[str, ...] = ()
    constants: Mapping[str, Any] = field(default_factory=frozendict)
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_names", as_names(self.parameter_names, message_name="parameter_names"))
        object.__setattr__(self, "given", as_names(self.given, message_name="given"))
        object.__setattr__(self, "constants", as_constants(self.constants, message_name="the term's constants"))
        check_term_covers_a_parameter(self.parameter_names)
        check_term_is_not_given_what_it_covers(self.name, self.parameter_names, self.given)
        check_names_are_unique(
            [*self.parameter_names, *self.given, *self.constants],
            message_name=f"the prior of {self.name!r}'s parameter names, given and constants",
        )
        check_distribution_is_a_distribution_or_a_function(self.name, self.distribution)
        check_provenance_is_given(self.provenance)

    @property
    def name(self) -> str:
        """``"+".join(parameter_names)``: the term's row in
        :meth:`Prior.describe`, and the key of its randomness in
        :meth:`Prior.sample`."""
        return "+".join(self.parameter_names)


@dataclass(frozen=True, eq=False, repr=False)
class Prior:
    """The prior over a parameter vector, the product of its terms:

    .. math::

        \\pi(x) = \\prod_b \\pi_b\\big(x_{B_b} \\mid x_{g(b)}, y_{g(b)};\\ c_b\\big).

    Parameters
    ----------
    parameter_vector:
        The vector the prior is over.
    terms:
        The :class:`PriorTerm`\\ s, in any order, covering every parameter
        exactly once.
    derived_parameters:
        The derived parameters over the vector, needed when a term is given
        one; checked to share the vector's layout.

    Attributes
    ----------
    terms : tuple of PriorTerm
        The terms, ordered by the vector's declaration order of their first
        parameter.

    Raises
    ------
    TypeError
        If *terms* is not a sequence, a term is not a :class:`PriorTerm`,
        or a term that needs a function (indexed, given others, or reading
        constants) is given a bare distribution.
    KeyError
        If a term or a ``given`` names nothing of the vector or its derived
        parameters.
    ValueError
        If a parameter is covered by no term or by two, a joint term's
        parameters are indexed differently, the ``given`` links form a
        cycle, the derived parameters are over another vector, or a term's
        distribution is not over its parameters' blocks, has another
        support, or has no density the prior can evaluate; the message names
        the term.
    """

    parameter_vector: ParameterVector
    terms: tuple[PriorTerm, ...]
    derived_parameters: DerivedParameters | None = field(default=None, kw_only=True)
    # Built and checked at construction.
    _built: Mapping[str, _BuiltTerm] = field(init=False, repr=False)
    _draw_order: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        check_terms_are_not_keyed(self.terms)
        terms = as_sequence(self.terms, message_name="terms")
        for term in terms:
            check_term_is_a_prior_term(term)
        if self.derived_parameters is not None:
            check_derived_parameters_are_derived_parameters(self.derived_parameters)
            check_parameter_vectors_share_a_layout(self.parameter_vector, self.derived_parameters.parameter_vector)
        check_terms_cover_the_parameters(terms, self.parameter_vector)
        for term in terms:
            check_given_names_are_held(term.name, term.given, self)
        declared = {name: i for i, name in enumerate(self.parameter_vector.parameter_names)}
        object.__setattr__(self, "terms", tuple(sorted(terms, key=lambda t: declared[t.parameter_names[0]])))
        order = _terms_in_draw_order(self)
        object.__setattr__(self, "_draw_order", order)
        object.__setattr__(self, "_built", frozendict(self._build_terms(order)))

    # ── identity ──────────────────────────────────────────────────────────────

    def __getitem__(self, parameter_name: str) -> PriorTerm:
        """The term covering *parameter_name*, joint or not."""
        check_parameter_is_in_the_vector(parameter_name, self)
        return next(t for t in self.terms if parameter_name in t.parameter_names)

    def __repr__(self) -> str:
        return f"Prior(D={self.parameter_vector.unconstrained.size}, terms={[t.name for t in self.terms]})"

    def describe(self) -> pd.DataFrame:
        """One row per term, in the vector's declaration order, indexed by
        ``parameters`` (the term's :attr:`~PriorTerm.name`): ``prior``,
        ``given`` (comma-separated), ``evaluated_by`` (``"base density"`` or
        ``"change of variables"``) and ``provenance``."""
        rows = [
            {
                "parameters": built.name,
                "prior": _prior_name(built.source, built.distribution),
                "given": ", ".join(built.given),
                "evaluated_by": built.evaluated_by,
                "provenance": term.provenance,
            }
            for term, built in zip(self.terms, self._built.values())
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
        kept = {*vector.parameter_names, *(() if derived is None else derived.derived_parameter_names)}
        terms = []
        for term in self.terms:
            if any(name in kept for name in term.parameter_names):
                check_selection_keeps_what_a_term_needs(term.name, (*term.parameter_names, *term.given), kept)
                terms.append(term)
        return Prior(vector, terms, derived_parameters=derived)

    # ── evaluation ────────────────────────────────────────────────────────────

    def sample(self, key: Array, n: int) -> Array:
        """``n`` draws of theta from the prior, ``(n, D)``.

        Terms are drawn in topological order of the ``given`` links. A term
        draws with ``jax.random.fold_in(key, crc32(name))``, its name being
        :attr:`PriorTerm.name`; a term given others
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

    # ── supporting methods ────────────────────────────────────────────────────

    def _build_terms(self, order: tuple[str, ...]) -> dict[str, _BuiltTerm]:
        """Every term built in draw order, held in :attr:`terms`' order. When
        some term is given others, two ancestral draws are made as the terms
        are built, and each term given others is built and checked at both."""
        terms_by_name = {term.name: term for term in self.terms}
        ancestral = any(term.given for term in self.terms)
        ancestral_key = jax.random.key(_ANCESTRAL_SEED)
        positions = self._theta_positions
        values_by_parameter: dict[str, Array] = {}
        built: dict[str, _BuiltTerm] = {}
        for name in order:
            term = terms_by_name[name]
            given_values = self._given_values(term.given, values_by_parameter)
            built[name] = _BuiltTerm.build(term, self, positions, given_values)
            if ancestral:
                key = _random_key_for_term(ancestral_key, name)
                theta_b = built[name].sample_theta(key, _N_ANCESTRAL_DRAWS, given_values)
                check_draws_map_to_finite_theta(name, theta_b)
                values_by_parameter |= built[name].natural_values(theta_b)
        return {term.name: built[term.name] for term in self.terms}

    @property
    def _coords(self) -> Mapping[str, pd.Index]:
        """The labels the terms' constants are read at: the derived
        parameters' coords, or the vector's."""
        if self.derived_parameters is not None:
            return self.derived_parameters.coords
        return self.parameter_vector.coords

    def _labels_for(self, term: PriorTerm) -> Mapping[str, pd.Index]:
        """The labels *term*'s constants are read at: the coords, and the
        element axes of the parameters it covers and of what it is given."""
        pieces = [
            self.parameter_vector[n] if n in self.parameter_vector else self.derived_parameters[n]
            for n in (*term.parameter_names, *term.given)
        ]
        element_axes = {axis: labels for p in pieces for axis, labels in p.element_labels.items()}
        return {**element_axes, **self._coords}

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
        missing = [n for n in names if n in self.derived_parameters.derived_parameter_names and n not in values_by_parameter]
        if not missing:
            return {}
        return self.derived_parameters.values(values_by_parameter, derived_parameter_names=missing)

# ── private constants ─────────────────────────────────────────────────────────


#: The number of draws of the draw-based support check, and the size of each
#: batch of them, which bounds its memory for a term over many labels.
_SUPPORT_DRAWS, _SUPPORT_DRAW_BATCH = 10_000, 1_000

#: The seed of the draw-based support check.
_SUPPORT_SEED = 20260927

#: The seed and number of the ancestral draws a term given others is built
#: and checked at.
_ANCESTRAL_SEED, _N_ANCESTRAL_DRAWS = 20260928, 2

# ── private: a term built for the vector at hand ──────────────────────────────


@dataclass(frozen=True, eq=False)
class _BuiltTerm:
    """A term built for the vector at hand.

    Its theta is flat, ``(..., D_b)``, in the order of its parameter names,
    each parameter's unconstrained block in C order; ``positions`` are
    those entries' positions in theta, and ``shapes`` the unconstrained
    block shapes. ``constants`` are its constants read at the labels in use.
    A term given others keeps the distribution at the first ancestral draw,
    for its checks and its description, and rebuilds it per draw to evaluate
    and sample.
    """

    name: str
    parameters: tuple[Parameter, ...]
    given: tuple[str, ...]
    source: tfd.Distribution | PriorFunction
    index_shape: tuple[int, ...]
    constants: Mapping[str, Array]
    distribution: tfd.Distribution
    positions: np.ndarray
    shapes: tuple[tuple[int, ...], ...]
    by_base_density: bool

    @classmethod
    def build(
        cls,
        term: PriorTerm,
        prior: Prior,
        theta_positions: Mapping[str, np.ndarray],
        given_values: Mapping[str, Array] | None,
    ) -> _BuiltTerm:
        """Build and check a term; one given others at each ancestral draw
        in *given_values* (``(draws, *block shape)`` each)."""
        vector = prior.parameter_vector
        names = term.parameter_names
        unbuilt = cls(
            name=term.name,
            parameters=tuple(vector[n] for n in names),
            given=term.given,
            source=term.distribution,
            index_shape=vector.index_shape(names[0]),
            constants=frozendict(_constants_read_for(term, prior)),
            distribution=None,
            positions=np.concatenate([theta_positions[n] for n in names]),
            shapes=tuple(vector.unconstrained.block_shape(n) for n in names),
            by_base_density=False,
        )
        check_term_needing_a_function_is_one(unbuilt)
        probes = unbuilt.probes()
        variants = [unbuilt._with_distribution(d, probes) for d in unbuilt._distributions(given_values)]
        check_term_keeps_its_structure(variants)
        for variant in variants:
            check_prior_term_is_valid(variant)
        return variants[0]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def joint(self) -> bool:
        return len(self.parameters) > 1

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
        """The distribution its function returns at one draw's given values;
        a builder's is also passed the index shape."""
        if isinstance(self.source, IndexShapedFunction):
            return self.source(self.index_shape, **given_values, **self.constants)
        return self.source(**given_values, **self.constants)

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
        if type(distribution) not in CARRIES_ITS_BIJECTOR:
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
        return [self.distribution_at({})]

    def _with_distribution(self, distribution: tfd.Distribution, probes: Array) -> _BuiltTerm:
        """This term at *distribution*, checked to be over its parameters' blocks,
        and evaluated by base density where it pushes through."""
        natural_shapes = {p.name: (*self.index_shape, *p.shape) for p in self.parameters}
        check_term_is_over_its_parameters_blocks(
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


def _constants_read_for(term: PriorTerm, prior: Prior) -> dict[str, Array]:
    """The term's constants read at the prior's labels, each with the
    parameters' ``indexed_by`` dims first, then their element axes."""
    vector = prior.parameter_vector
    # A joint term's parameters may share an element axis; it is named once.
    dim_order = tuple(dict.fromkeys((
        *vector[term.parameter_names[0]].indexed_by,
        *(axis for name in term.parameter_names for axis in vector[name].element_labels),
    )))
    return aligned_constants(
        term.constants, prior._labels_for(term), dim_order=dim_order,
        message_name=f"the prior of {term.name!r} constants",
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


# ── private helpers ───────────────────────────────────────────────────────────


def _random_key_for_term(key: Array, name: str) -> Array:
    return jax.random.fold_in(key, zlib.crc32(name.encode()))


def _terms_in_draw_order(prior: Prior) -> tuple[str, ...]:
    """The terms' names in a topological order of the ``given`` links, each
    after everything it is given: the covering term of a parameter, and,
    through a derived parameter, the covering terms of what it is computed
    from."""
    vector = prior.parameter_vector
    terms = {term.name: term for term in prior.terms}
    owner = {name: term.name for term in prior.terms for name in term.parameter_names}
    derived = prior.derived_parameters
    derived_names = () if derived is None else derived.derived_parameter_names

    def links(node: tuple[str, Any]) -> list[tuple[str, Any]]:
        kind, value = node
        names = terms[value].given if kind == "term" else derived[value].given
        return [("derived", n) if n in derived_names else ("term", owner[n]) for n in names]

    order: list[str] = []
    state: dict[tuple[str, Any], str] = {}
    first_position = {
        term.name: min(vector.parameter_names.index(n) for n in term.parameter_names) for term in prior.terms
    }
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
















def _prior_name(source: Any, distribution: tfd.Distribution) -> str:
    if isinstance(source, IndexShapedFunction):
        return source.name
    return distribution_name(distribution)




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


def check_terms_cover_the_parameters(terms: Sequence[PriorTerm], vector: ParameterVector) -> None:
    """The terms are over the vector's parameters, a joint term's indexed
    alike, and cover every parameter once, so no density is counted twice
    or left out."""
    owner: dict[str, str] = {}
    for term in terms:
        for name in term.parameter_names:
            check_term_covers_a_parameter_once(name, term.name, vector, owner)
            owner[name] = term.name
        check_joint_term_is_indexed_alike(term, vector)
    check_every_parameter_has_a_term(owner, vector)


def check_terms_are_not_keyed(terms: Any) -> None:
    """The terms are a sequence, not a mapping: each names the parameters it
    covers, so a key would say it twice."""
    if isinstance(terms, Mapping):
        raise TypeError(
            "terms is a mapping; give a sequence of PriorTerms, each naming its parameters, such as "
            "[PriorTerm(parameter_names=('rate',), distribution=..., provenance=...)]."
        )


def check_term_covers_a_parameter(parameter_names: Sequence[str]) -> None:
    """A term is the prior of at least one parameter."""
    if not parameter_names:
        raise ValueError("a prior term has no parameter_names; name the parameters it is the prior of.")


def check_distribution_is_a_distribution_or_a_function(name: str, distribution: Any) -> None:
    """A term's distribution is a TFP distribution or a prior function,
    which are the two forms the prior can build."""
    if not (isinstance(distribution, tfd.Distribution) or callable(distribution)):
        raise TypeError(
            f"the prior of {name!r} is a {type(distribution).__name__}; give a TFP distribution or a "
            "prior function f(**given, **constants) that returns one."
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


def check_joint_term_is_indexed_alike(term: PriorTerm, vector: ParameterVector) -> None:
    """A joint term's parameters are indexed by the same dims: dependence
    across dims is what ``given`` expresses."""
    indexed = {vector[name].indexed_by for name in term.parameter_names}
    if len(indexed) > 1:
        raise ValueError(
            f"the joint term {term.name!r} covers parameters indexed by {sorted(indexed)}; a "
            "joint term's parameters are indexed alike, so give each its own term and link them "
            "with given=."
        )


def check_given_names_are_held(name: str, given: Sequence[str], prior: Prior) -> None:
    """What a term is given is a parameter or derived parameter, which it
    would otherwise be passed no value for."""
    derived = () if prior.derived_parameters is None else prior.derived_parameters.derived_parameter_names
    for given_name in given:
        if given_name not in prior.parameter_vector and given_name not in derived:
            raise KeyError(
                f"the prior of {name!r} is given {given_name!r}, which is no parameter of the vector "
                "nor a derived parameter of derived_parameters=."
            )


def check_term_is_not_given_what_it_covers(name: str, parameter_names: Sequence[str], given: Sequence[str]) -> None:
    """A term is not given a parameter it covers, whose density it is."""
    covered = [n for n in given if n in parameter_names]
    if covered:
        raise ValueError(
            f"the prior of {name!r} is given {covered}, which it covers; a term is the density of "
            "what it covers, so give it only other parameters."
        )


def check_given_links_are_acyclic(cycle: Sequence[tuple[str, Any]] | None) -> None:
    """The ``given`` links, through derived parameters, form no cycle, which
    no order of draws could satisfy; *cycle* is the one found, if any."""
    if cycle is None:
        return
    names = [value for _, value in cycle]
    raise ValueError(
        f"the given links form a cycle, {' -> '.join(names)}; a term cannot depend on itself, so "
        "break the cycle."
    )


def check_term_needing_a_function_is_one(built: _BuiltTerm) -> None:
    """A term over indexed parameters, given others, or reading constants is
    a function of them; a bare distribution would ignore them."""
    if not isinstance(built.source, tfd.Distribution):
        return
    if built.given:
        raise TypeError(
            f"the prior of {built.name!r} is given {list(built.given)} but is a distribution; give "
            "a prior function f(**given) that returns one."
        )
    if built.index_shape:
        raise TypeError(
            f"the prior of {built.name!r} is over parameters indexed by "
            f"{built.parameters[0].indexed_by}, so it is built for the labels in use; wrap the "
            "distribution as iid_over_dim(distribution) or independent_over_dim(family, ...)."
        )
    if built.constants:
        raise TypeError(
            f"the prior of {built.name!r} reads {sorted(built.constants)} but is a distribution; "
            "give a prior function that reads them."
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


def check_parameter_is_in_the_vector(parameter_name: Any, prior: Prior) -> None:
    """The prior has a term for the parameter asked for, as it has for every
    parameter of its vector."""
    if not isinstance(parameter_name, str) or parameter_name not in prior.parameter_vector:
        raise KeyError(
            f"the prior has no term covering {parameter_name!r}; name a parameter of its vector, one "
            f"of {truncated(list(prior.parameter_vector.parameter_names))}."
        )


def check_term_is_a_prior_term(term: Any) -> None:
    """A term is a :class:`PriorTerm`, which names its parameters and
    carries its provenance."""
    if not isinstance(term, PriorTerm):
        raise TypeError(
            f"a prior term is a {type(term).__name__}; wrap it as "
            "PriorTerm(parameter_names=(...), distribution=..., provenance=...)."
        )


def check_provenance_is_given(provenance: Any) -> None:
    """A prior says where it came from."""
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(
            "a prior term needs a provenance: where the prior came from, or that it is a "
            "placeholder."
        )






def check_term_is_over_its_parameters_blocks(name: str, distribution: Any, expected: Any) -> None:
    """A term's distribution is ``float64`` over its parameters' blocks:
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
            f"{distribution.dtype}, but must be a float64 distribution over the blocks of the "
            f"parameters it covers: event shape {expected}, TFP batch shape (). Labels independent and "
            "identically distributed are iid_over_dim, one prior per label independent_over_dim; "
            "a joint term's draws are a dict keyed by its parameter names."
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


def check_draws_map_to_finite_theta(name: str, theta: Array) -> None:
    """Every draw maps to a finite theta."""
    if not bool(jnp.all(jnp.isfinite(theta))):
        raise ValueError(
            f"the prior of {name!r} drew a value on its support's boundary, whose theta is not "
            "finite; choose a prior with less mass at the boundary."
        )
























