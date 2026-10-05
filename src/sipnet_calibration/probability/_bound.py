"""Parts bound to the labels in use: a factor's law, density and draws, a
deterministic's values, and the checks each passes when a model is bound.
Private to the probability layer.

A bound factor works in two coordinates. In **theta**, its components'
unconstrained blocks flattened and concatenated in event order,
``(..., D_b)``, it draws and evaluates the density a target contributes; in
**natural** values, ``{name: (..., *block)}``, it evaluates the density an
observed factor contributes. Both are against each support's reference
measure (:mod:`~sipnet_calibration.probability.laws`).

What a law reads arrives in two parts: values that differ draw by draw
(``per_draw``, each ``(..., *block)``), over which the law is vmapped, and
values the same in every draw (``fixed``: inputs, observed values, and
what is computed from them alone), with which it is built once.
"""

from __future__ import annotations

import dataclasses
import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability._probes import bijectors_agree, joint_probe_points
from sipnet_calibration.probability.builders import Builder
from sipnet_calibration.probability.laws import CARRIES_ITS_BIJECTOR, is_law
from sipnet_calibration.probability.parts import DeterministicSpec, FactorSpec, Simulator
from sipnet_calibration.probability.spec import ArraySpec
from sipnet_calibration.probability.support import (
    REAL,
    Interval,
    PositiveDefinite,
    Simplex,
    Support,
    bijector_for,
)

__all__ = [
    "ANCESTRAL_DRAWS",
    "ANCESTRAL_SEED",
    "BoundDeterministic",
    "BoundFactor",
    "check_deterministic_has_its_block_shapes",
    "check_deterministic_lies_in_its_supports",
    "check_draws_map_to_finite_theta",
    "coords_of",
    "deterministics_behind",
    "finite_or_minus_infinity",
    "log_jacobian",
    "random_key_for",
    "simulator_at",
    "simulator_outputs_behind",
    "split_reads",
]

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jax.Array

#: The seed and number of the ancestral draws a factor reading other
#: components is built and checked at.
ANCESTRAL_SEED, ANCESTRAL_DRAWS = 20260928, 2

#: The number of draws of the draw-based support check, and the size of each
#: batch of them, which bounds its memory for a factor over many labels.
_SUPPORT_DRAWS, _SUPPORT_DRAW_BATCH = 10_000, 1_000

#: The seed of the draw-based support check.
_SUPPORT_SEED = 20260927

#: TFP's laws with no density against the reference measure of any support a
#: component may declare: discrete laws and point masses, whose mass sits on
#: a null set of an interval, and the LKJ laws, whose draws (correlation
#: matrices, or their Cholesky factors) are a null set of the
#: positive-definite matrices. By name, so a class a TFP version lacks is
#: skipped.
_LAWS_WITHOUT_A_DENSITY = tuple(
    getattr(tfd, name)
    for name in (
        "Bernoulli", "BetaBinomial", "Binomial", "Categorical", "CholeskyLKJ", "Deterministic",
        "DirichletMultinomial", "Empirical", "FiniteDiscrete", "Geometric", "LKJ", "Multinomial",
        "NegativeBinomial", "OneHotCategorical", "PlackettLuce", "Poisson",
        "PoissonLogNormalQuadratureCompound", "QuantizedDistribution", "Skellam", "VectorDeterministic",
        "ZeroInflatedNegativeBinomial", "Zipf",
    )
    if hasattr(tfd, name)
)

#: The magnitude of theta beyond which a probe is an outer one, where a
#: computed value may round onto its support's boundary or overflow: every
#: probe but theta = 0 and +-3 * 1.
_OUTER_PROBE = 3.0


def random_key_for(key: Array, name: str) -> Array:
    """A part's key: ``jax.random.fold_in(key, crc32(name))``."""
    return jax.random.fold_in(key, zlib.crc32(name.encode()))


def finite_or_minus_infinity(log_density: Array) -> Array:
    """*log_density* with every non-finite value, NaN included, as ``-inf``:
    a density that is not finite is no density."""
    return jnp.where(jnp.isfinite(log_density), log_density, -jnp.inf)


def coords_of(specs: Sequence[ArraySpec], coords: Mapping[str, pd.Index]) -> dict[str, pd.Index]:
    """The coords of the dims *specs* are indexed by."""
    used = {d for c in specs for d in c.indexed_by}
    return {d: labels for d, labels in coords.items() if d in used}


def split_reads(given: Sequence[str], per_draw: Mapping[str, Array], fixed: Mapping[str, Array]) -> tuple[dict, dict]:
    """What a part reads, split into the values that vary by draw and those
    held."""
    reads = {name: per_draw[name] for name in given if name in per_draw}
    held = {name: fixed[name] for name in given if name not in per_draw}
    return reads, held


def deterministics_behind(spec: Any, names: Sequence[str]) -> set[str]:
    """The deterministics and simulators (by name) computing *names*,
    directly or through other deterministics and simulators."""
    seen: set[str] = set()
    pending = list(names)
    while pending:
        part = spec._owner.get(pending.pop())
        if isinstance(part, (DeterministicSpec, Simulator)) and part.name not in seen:
            seen.add(part.name)
            pending.extend(part.given)
    return seen


def simulator_at(simulator: Simulator, coords: Mapping[str, pd.Index], outputs: Sequence[str] | None = None) -> Simulator:
    """*simulator* at the labels *coords*, computing *outputs* (every one by
    default), checked to read what it read and compute exactly those."""
    names = [o.name for o in simulator.outputs] if outputs is None else list(outputs)
    bound = simulator.at(coords, names)
    check_simulator_at_keeps_its_parts(simulator, bound, names)
    return bound


def simulator_outputs_behind(spec: Any, names: Sequence[str]) -> set[str]:
    """The simulator outputs *names* are or descend from, through any part:
    those whose failure leaves *names* undefined."""
    found: set[str] = set()
    seen: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        part = spec._owner.get(name)
        if isinstance(part, Simulator):
            found.add(name)
        if part is not None:
            pending.extend(part.given)
    return found


# ── a factor ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class BoundFactor:
    """A factor at the labels in use.

    ``fixed_reads`` are its constants and label maps read at the labels;
    ``shapes`` its components' unconstrained block shapes, in event order.
    ``law`` is its law at the first draw it was built at, kept for its
    checks and description; ``by_base_density`` whether it is a
    pushforward through its components' own bijectors.
    """

    spec: FactorSpec
    index_shape: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]
    fixed_reads: Mapping[str, Any]
    law: Any = None
    by_base_density: bool = False

    @classmethod
    def build(
        cls,
        spec: FactorSpec,
        *,
        index_shape: tuple[int, ...],
        fixed_reads: Mapping[str, Any],
        per_draw_values: Sequence[Mapping[str, Array]],
        fixed_values: Mapping[str, Array],
        downstream_of_a_simulator: bool = False,
    ) -> BoundFactor:
        """Build and check a factor at each draw of *per_draw_values* (one
        empty mapping when it reads nothing that varies by draw).

        *downstream_of_a_simulator* says those values hold placeholders for
        simulator outputs, so only the law's form is checked: its event,
        TFP batch shape and dtype, and that it has a density a model can
        evaluate; its support is not, since that would be checked at values
        no simulation gave."""
        unbuilt = cls(
            spec=spec,
            index_shape=index_shape,
            shapes=tuple((*index_shape, *c.unconstrained_shape) for c in spec.event),
            fixed_reads=fixed_reads,
        )
        probes = unbuilt.probes()
        variants = [unbuilt._at(unbuilt.law_at({**values, **fixed_values}), probes) for values in per_draw_values]
        check_factor_keeps_its_structure(variants)
        for variant in variants:
            if downstream_of_a_simulator:
                check_factor_law_has_a_measured_density(variant)
            else:
                check_factor_law_is_valid(variant)
        return variants[0]

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def components(self) -> tuple[ArraySpec, ...]:
        return self.spec.event

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.spec.event)

    @property
    def joint(self) -> bool:
        return len(self.spec.event) > 1

    @property
    def size(self) -> int:
        return sum(math.prod(shape) for shape in self.shapes)

    @property
    def natural_shapes(self) -> dict[str, tuple[int, ...]]:
        return {c.name: (*self.index_shape, *c.shape) for c in self.components}

    @property
    def evaluated_by(self) -> str:
        return "base density" if self.by_base_density else "change of variables"

    # ── the law ───────────────────────────────────────────────────────────────

    def law_at(self, given: Mapping[str, Array]) -> Any:
        """The law at one draw's values of what it reads."""
        law = self.spec.law
        if isinstance(law, Builder):
            return law(self.index_shape, **self._reads(given))
        if is_law(law):
            return law
        return law(**self._reads(given))

    def probes(self) -> Array:
        """:func:`joint_probe_points` over the components' unconstrained
        blocks, the axis probes along each one's first value, flat,
        ``(n_probes, D_b)``."""
        index_ndim = len(self.index_shape)
        parts = joint_probe_points([(shape, math.prod(shape[index_ndim:])) for shape in self.shapes])
        return jnp.asarray(np.concatenate([part.reshape((len(part), -1)) for part in parts], axis=-1))

    def split(self, theta: Array) -> dict[str, Array]:
        """Flat theta, ``(..., D_b)``, as each component's unconstrained block."""
        lead, out, start = theta.shape[:-1], {}, 0
        for component, shape in zip(self.components, self.shapes):
            size = math.prod(shape)
            out[component.name] = theta[..., start : start + size].reshape(lead + shape)
            start += size
        return out

    def natural_values(self, theta: Array) -> dict[str, Array]:
        """``{name: T(theta)}`` for its components."""
        split = self.split(theta)
        return {c.name: c.bijector.forward(split[c.name]) for c in self.components}

    def unconstrained_values(self, values: Mapping[str, Array]) -> Array:
        """Natural values as flat theta, ``(..., D_b)``."""
        lead = None
        pieces = []
        for component, shape in zip(self.components, self.shapes):
            theta = component.bijector.inverse(jnp.asarray(values[component.name], dtype=jnp.float64))
            lead = theta.shape[: theta.ndim - len(shape)] if lead is None else lead
            pieces.append(theta.reshape((*lead, math.prod(shape))))
        return jnp.concatenate(pieces, axis=-1)

    # ── evaluation ────────────────────────────────────────────────────────────

    def log_prob_theta(self, theta: Array, per_draw: Mapping[str, Array], fixed: Mapping[str, Array]) -> Array:
        """Its density at its theta, ``(..., D_b) -> (...)``: by base density
        where it pushes through its components' bijectors, else by change of
        variables, the log-Jacobian added."""
        if not per_draw:
            return self._log_prob_under(self.law_at(fixed), theta)
        lead = theta.shape[:-1]
        flat = _flattened(per_draw, len(lead))
        out = jax.vmap(lambda t, v: self._log_prob_under(self.law_at({**v, **fixed}), t))(
            theta.reshape((-1, theta.shape[-1])), flat
        )
        return out.reshape(lead)

    def log_prob_natural(
        self,
        values: Mapping[str, Array],
        per_draw: Mapping[str, Array],
        fixed: Mapping[str, Array],
        lead: tuple[int, ...],
    ) -> Array:
        """Its density at natural values, ``{name: (*lead, *block)} ->
        lead``, *per_draw* also over *lead*; values without the batch are
        held for every draw."""
        if not per_draw:
            return self._log_prob_natural_under(self.law_at(fixed), values)
        held = {n: jnp.broadcast_to(v, (*lead, *self.natural_shapes[n])) for n, v in values.items()}
        out = jax.vmap(lambda x, v: self._log_prob_natural_under(self.law_at({**v, **fixed}), x))(
            _flattened(held, len(lead)), _flattened(per_draw, len(lead))
        )
        return out.reshape(lead)

    def sample_theta(self, key: Array, n: int, per_draw: Mapping[str, Array], fixed: Mapping[str, Array]) -> Array:
        """``n`` draws of its theta, ``(n, D_b)``: at once when nothing it
        reads varies by draw, else one per draw with the key split ``n``
        ways."""
        if not per_draw:
            return self._draw(self.law_at(fixed), key, n)
        keys = jax.random.split(key, n)
        return jax.vmap(lambda k, v: self._draw(self.law_at({**v, **fixed}), k, None))(keys, dict(per_draw))

    # ── supporting methods ────────────────────────────────────────────────────

    def _reads(self, given: Mapping[str, Array]) -> dict[str, Any]:
        """What its law reads: the given values it names, and its fixed reads."""
        return {**{name: given[name] for name in self.spec.given}, **self.fixed_reads}

    def _at(self, law: Any, probes: Array) -> BoundFactor:
        """This factor at *law*, checked to be over its event, and evaluated
        by base density where it pushes through."""
        check_factor_law_is_over_its_event(self.name, law, self.natural_shapes if self.joint else self.natural_shapes[self.names[0]])
        return dataclasses.replace(self, law=law, by_base_density=self._pushes_through(law, probes))

    def _pushes_through(self, law: Any, probes: Array) -> bool:
        """Whether *law* is ``TransformedDistribution(base, b)``, the base's
        event theta's block, with ``b`` the components' own map at the probe
        points: an exact ``TransformedDistribution``, ``LogNormal`` or
        ``LogitNormal``."""
        if type(law) not in CARRIES_ITS_BIJECTOR:
            return False
        base_event = tuple(law.distribution.event_shape)
        if not self.joint:
            (shape,) = self.shapes
            if base_event != shape:
                return False
            usable = _invertible(self.components[0], probes.reshape((len(probes), *shape)))
            return bijectors_agree(law.bijector, self.components[0].bijector, probes[usable].reshape((-1, *shape)))
        if base_event != (self.size,):
            return False
        try:
            images = law.bijector.forward(jnp.array(probes))
        except (TypeError, ValueError):
            return False
        expected = self.natural_values(probes)
        return isinstance(images, Mapping) and set(images) == set(expected) and all(
            np.allclose(images[n], expected[n], rtol=1e-10, atol=0.0) for n in expected
        )

    def _log_prob_under(self, law: Any, theta: Array) -> Array:
        """Its density at theta under *law*, the log-Jacobian added where it
        is evaluated by change of variables."""
        if self.by_base_density:
            return law.distribution.log_prob(self._base_event(theta))
        split = self.split(theta)
        values = {c.name: c.bijector.forward(split[c.name]) for c in self.components}
        density = law.log_prob(values if self.joint else values[self.names[0]])
        return density + self._log_jacobian(split, theta.ndim - 1)

    def _log_prob_natural_under(self, law: Any, values: Mapping[str, Array]) -> Array:
        """Its density at natural values under *law*: the law's own, or, by
        base density, the base's at theta less the log-Jacobian, so that it
        is against the supports' reference measure whatever TFP's bijector
        measures against."""
        if not self.by_base_density:
            return law.log_prob(dict(values) if self.joint else values[self.names[0]])
        theta = self.unconstrained_values(values)
        return law.distribution.log_prob(self._base_event(theta)) - self._log_jacobian(self.split(theta), theta.ndim - 1)

    def _log_jacobian(self, split: Mapping[str, Array], lead_ndim: int) -> Array:
        """The log-Jacobian of its components' bijectors at *split*, summed
        over each block: ``(...)``."""
        total = 0.0
        for component in self.components:
            jacobian = log_jacobian(component.support, component.bijector, split[component.name])
            total = total + jacobian.sum(axis=tuple(range(lead_ndim, jacobian.ndim)))
        return total

    def _draw(self, law: Any, key: Array, n: int | None) -> Array:
        """Draws of theta, ``(n, D_b)``, or one, ``(D_b,)``, for ``n=None``."""
        sample_shape = () if n is None else (n,)
        if self.by_base_density:
            draws = law.distribution.sample(sample_shape, seed=key)
            return draws.reshape(sample_shape + (self.size,))
        values = law.sample(sample_shape, seed=key)
        values = values if self.joint else {self.names[0]: values}
        pieces = [
            c.bijector.inverse(values[c.name]).reshape(sample_shape + (math.prod(shape),))
            for c, shape in zip(self.components, self.shapes)
        ]
        return jnp.concatenate(pieces, axis=-1)

    def _base_event(self, theta: Array) -> Array:
        """Flat theta as the base's event: a joint factor's is flat, one
        component's is its unconstrained block shape."""
        if self.joint:
            return theta
        return theta.reshape(theta.shape[:-1] + self.shapes[0])


# ── a deterministic ───────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class BoundDeterministic:
    """A deterministic at the labels in use: its constants and label maps
    read, and its outputs' block shapes."""

    spec: DeterministicSpec
    fixed_reads: Mapping[str, Any]
    block_shapes: Mapping[str, tuple[int, ...]]

    @property
    def name(self) -> str:
        return self.spec.name

    def compute(
        self, per_draw: Mapping[str, Array], fixed: Mapping[str, Array], lead: tuple[int, ...]
    ) -> dict[str, Array]:
        """Its outputs, ``{name: (*lead, *block)}``: vmapped over the draws
        of *per_draw*, each ``(*lead, *block)``, or once, unbatched, when it
        is empty."""
        if not per_draw:
            return self._one_draw(fixed)
        out = jax.vmap(lambda v: self._one_draw({**v, **fixed}))(_flattened(per_draw, len(lead)))
        return {name: value.reshape((*lead, *value.shape[1:])) for name, value in out.items()}

    def _one_draw(self, given: Mapping[str, Array]) -> dict[str, Array]:
        reads = {**{name: given[name] for name in self.spec.given}, **self.fixed_reads}
        out = self.spec.function(**reads)
        names = tuple(s.name for s in self.spec.outputs)
        if len(names) == 1:
            return {names[0]: jnp.asarray(out, dtype=jnp.float64)}
        check_outputs_are_a_dict_of_the_outputs(self.name, out, names)
        return {name: jnp.asarray(out[name], dtype=jnp.float64) for name in names}


# ── the reference measures ────────────────────────────────────────────────────


def log_jacobian(support: Support, bijector: tfb.Bijector, theta: Array) -> Array:
    """:math:`\\log J(\\theta)`, the log absolute Jacobian determinant of
    *bijector* against the reference measure of *support*'s densities.

    On an :class:`Interval` the measure is Lebesgue measure and
    :math:`\\log J = \\log |T'(\\theta)|`, number by number, the shape of
    *theta*. On the :class:`Simplex` it is Lebesgue measure on the first
    :math:`k - 1` coordinates, the measure a ``Dirichlet``'s density is
    written against:

    .. math::

        \\log J(\\theta) = \\log \\left| \\det
            \\frac{\\partial (x_1, \\dots, x_{k-1})}{\\partial \\theta} \\right|,
        \\qquad x = T(\\theta),

    over the last axis, ``(..., k - 1) -> (...)``; for ``SoftmaxCentered``
    the closed form :math:`\\sum_{i=1}^{k} \\log x_i`, :math:`\\log x =
    \\operatorname{log\\_softmax}([\\theta, 0])`, computed from
    :math:`\\theta` so that no coordinate rounds to 0. On the
    :class:`PositiveDefinite` matrices it is Lebesgue measure on the lower
    triangle,

    .. math::

        \\log J(\\theta) = \\log \\left| \\det
            \\frac{\\partial \\operatorname{vech}(\\Sigma)}{\\partial \\theta} \\right|,
        \\qquad \\Sigma = T(\\theta),

    over the last axis, ``(..., p(p+1)/2) -> (...)``. Any bijector but
    ``SoftmaxCentered`` on the simplex is differentiated automatically.

    Notes
    -----
    TFP's ``SoftmaxCentered.forward_log_det_jacobian`` is
    :math:`\\tfrac12 \\log \\det(J^\\top J)` of the full :math:`k \\times (k-1)`
    Jacobian, against the simplex's surface measure: this plus
    :math:`\\tfrac12 \\log k`.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    if isinstance(support, Interval):
        return bijector.forward_log_det_jacobian(theta, event_ndims=0)
    if isinstance(support, Simplex) and type(bijector) is tfb.SoftmaxCentered:
        padded = jnp.concatenate([theta, jnp.zeros((*theta.shape[:-1], 1))], axis=-1)
        return jax.nn.log_softmax(padded, axis=-1).sum(axis=-1)
    if isinstance(support, Simplex):
        def coordinates(t: Array) -> Array:
            return bijector.forward(t)[:-1]
    else:
        def coordinates(t: Array) -> Array:
            matrix = bijector.forward(t)
            rows, columns = np.tril_indices(matrix.shape[-1])
            return matrix[rows, columns]

    def log_determinant(t: Array) -> Array:
        return jnp.linalg.slogdet(jax.jacfwd(coordinates)(t))[1]

    flat = theta.reshape((-1, theta.shape[-1]))
    return jax.vmap(log_determinant)(flat).reshape(theta.shape[:-1])


# ── helpers ───────────────────────────────────────────────────────────────────


def _flattened(values: Mapping[str, Array], lead_ndim: int) -> dict[str, Array]:
    """Each value with its leading *lead_ndim* axes flattened into one."""
    return {name: value.reshape((-1, *value.shape[lead_ndim:])) for name, value in values.items()}


def _invertible(component: ArraySpec, probes: Array) -> np.ndarray:
    """``(n_probes,)``: probes where the component's support's default
    bijector inverts its own image. All of them but on the positive-definite
    matrices, where :math:`-10 \\cdot 1` and :math:`-20 \\cdot 1` give values
    too ill-conditioned to invert."""
    if not isinstance(component.support, PositiveDefinite):
        return np.ones(len(probes), dtype=bool)
    default = bijector_for(component.support)
    images = default.forward(jnp.array(np.asarray(probes)))
    back = np.asarray(default.inverse(jnp.array(np.asarray(images))))
    close = np.isfinite(back) & np.isclose(back, np.asarray(probes), rtol=1e-6, atol=1e-6)
    return np.all(close, axis=tuple(range(1, close.ndim)))


def _laws_within(law: Any) -> list[Any]:
    """*law* and every law it wraps whose values are its values, or parts of
    them: the base of a ``Sample``, an ``Independent``, a
    ``TransformedDistribution``, a ``BatchBroadcast``, a ``BatchReshape`` or
    a ``Masked``; a mixture's components, not the law choosing among them;
    a joint law's parts where they are laws rather than functions."""
    found, pending = [], [law]
    while pending:
        current = pending.pop()
        found.append(current)
        if isinstance(current, tfd.MixtureSameFamily):
            pending.append(current.components_distribution)
        elif isinstance(current, tfd.Mixture):
            pending.extend(current.components)
        elif isinstance(current, (tfd.JointDistributionNamed, tfd.JointDistributionSequential)):
            parts = current.model.values() if isinstance(current.model, Mapping) else current.model
            pending.extend(part for part in parts if isinstance(part, tfd.Distribution))
        elif isinstance(getattr(current, "distribution", None), tfd.Distribution):
            pending.append(current.distribution)
    return found


def _structure_of(law: Any) -> Any:
    """A law's pytree structure: its class, its parts' classes and its
    static parameters; its class alone where it has no pytree structure, as
    for a ``JointDistributionNamed``."""
    try:
        return jax.tree_util.tree_structure(law)
    except (AttributeError, TypeError):
        return type(law)


def _event_shape_and_dtype(law: Any) -> tuple[Any, Any, Any]:
    """A law's event shape, TFP batch shape and dtype: TFP's own, or for any
    other law the shape and dtype of one draw, with batch shape ``()``."""
    if isinstance(law, tfd.Distribution):
        return _shape_of(law.event_shape), _shape_of(law.batch_shape), law.dtype
    draw = jax.eval_shape(lambda key: law.sample((), seed=key), jax.random.key(0))
    if isinstance(draw, Mapping):
        return {n: tuple(d.shape) for n, d in draw.items()}, (), {n: d.dtype for n, d in draw.items()}
    return tuple(draw.shape), (), draw.dtype


def _shape_of(shape: Any) -> Any:
    """A TFP shape, or a dict of them for a joint law, as tuples."""
    if isinstance(shape, Mapping):
        return {name: tuple(value) for name, value in shape.items()}
    return tuple(shape)


def _inside_at_each_probe(bound: BoundFactor, values: Mapping[str, Array], *, closure: bool = False) -> Array:
    """``(n_probes,)``: whether every component's value lies in its support
    (its closure when *closure*)."""
    inside = []
    for c in bound.components:
        support = c.support.closure() if closure else c.support
        inside.append(support.contains(values[c.name]).reshape((len(values[c.name]), -1)).all(axis=-1))
    return jnp.all(jnp.stack(inside), axis=0)


def _as_values(bound: BoundFactor, draws: Any) -> Mapping[str, Array]:
    """A law's draws as ``{name: value}``."""
    return draws if bound.joint else {bound.names[0]: draws}


def _probe_log_prob_is_finite(bound: BoundFactor, probes: Array) -> bool:
    """Whether the density is finite at the image of every probe point that
    lies inside the support and that each default bijector inverts; one the
    bijector rounds onto the boundary (``IteratedSigmoidCentered`` at 20)
    says nothing of the law's support."""
    values = bound.natural_values(probes)
    split = bound.split(probes)
    usable = np.all([_invertible(c, split[c.name]) for c in bound.components], axis=0)
    inside = _inside_at_each_probe(bound, values) & usable
    log_prob = bound.law.log_prob(values if bound.joint else values[bound.names[0]])
    return bool(jnp.all(jnp.isfinite(log_prob) | ~inside))


def _default_bijector_images_lie_in(bound: BoundFactor) -> bool | None:
    """Whether the law's own default event-space bijector maps the probe
    points into the closure of the supports; ``None`` when it has none, or
    one that cannot be applied in ``float64`` (an ``InverseWishart``'s)."""
    law = bound.law
    try:
        bijector = law.experimental_default_event_space_bijector()
    except (NotImplementedError, AttributeError):
        return None
    if bijector is None:
        return None
    unconstrained = bijector.inverse_event_shape(law.event_shape)
    leaves, structure = jax.tree_util.tree_flatten(unconstrained, is_leaf=lambda s: hasattr(s, "as_list"))
    shapes = [tuple(leaf) for leaf in leaves]
    parts = joint_probe_points([(shape, shape[-1] if shape else 1) for shape in shapes])
    try:
        images = bijector.forward(jax.tree_util.tree_unflatten(structure, [jnp.asarray(p) for p in parts]))
    except (TypeError, ValueError):
        return None
    images = _as_values(bound, images)
    # The closure: a bijector may round onto the boundary at the outer probes,
    # which is float64, not a wrong support; and a positive-definite image
    # too ill-conditioned to factor says nothing of the support either.
    inside = _inside_at_each_probe(bound, images, closure=True)
    usable = np.all([_factorable(c, images[c.name]) for c in bound.components], axis=0)
    return bool(jnp.all(inside | ~usable))


def _factorable(component: ArraySpec, values: Array) -> np.ndarray:
    """``(n,)``: values a default bijector can take back to theta: all of
    them but positive-definite matrices whose inverse is not finite."""
    if not isinstance(component.support, PositiveDefinite):
        return np.ones(len(values), dtype=bool)
    theta = np.asarray(bijector_for(component.support).inverse(jnp.array(np.asarray(values))))
    return np.all(np.isfinite(theta).reshape((len(values), -1)), axis=-1)


def _draws_lie_in_the_support(bound: BoundFactor) -> bool | None:
    """Whether fixed-seed draws lie in the supports and map to finite theta;
    ``None`` when the law cannot be sampled."""
    key = jax.random.key(_SUPPORT_SEED)
    for batch in range(_SUPPORT_DRAWS // _SUPPORT_DRAW_BATCH):
        try:
            draws = bound.law.sample(_SUPPORT_DRAW_BATCH, seed=jax.random.fold_in(key, batch))
        except NotImplementedError:
            return None
        values = _as_values(bound, draws)
        theta = [c.bijector.inverse(values[c.name]) for c in bound.components]
        if not (
            bool(jnp.all(_inside_at_each_probe(bound, values)))
            and all(bool(jnp.all(jnp.isfinite(t))) for t in theta)
        ):
            return False
    return True


def _lies_in_the_support_or_overflows(support: Support, values: Array) -> Array:
    """Whether each value lies in *support*'s closure, or is infinite at an
    end the support leaves unbounded, which is float64 overflow; NaN never
    does."""
    values = jnp.asarray(values, dtype=jnp.float64)
    inside = support.closure().contains(values)
    if isinstance(support, Interval):
        overflow = (jnp.isposinf(values) & np.isinf(support.high)) | (jnp.isneginf(values) & np.isinf(support.low))
        return inside | overflow
    return inside


# ── checks ────────────────────────────────────────────────────────────────────


def check_factor_law_is_valid(bound: BoundFactor) -> None:
    """A factor's law has the supports its components declare, and a
    density a model can evaluate."""
    check_factor_law_has_a_measured_density(bound)
    check_declared_support_lies_in_the_laws(bound)
    check_laws_support_lies_in_the_declared(bound)


def check_factor_law_has_a_measured_density(bound: BoundFactor) -> None:
    """A factor's law has a density against its supports' reference
    measure that a model can evaluate, judged by its form alone."""
    check_law_has_a_density(bound)
    check_change_of_variables_has_a_measure(bound)
    check_simplex_density_is_a_dirichlet(bound)


def check_factor_law_is_over_its_event(name: str, law: Any, expected: Any) -> None:
    """A factor's law is a ``float64`` law over its components' blocks:
    event shape the block shape (a dict of them for a joint factor), TFP
    batch shape ``()``."""
    if not is_law(law):
        raise TypeError(f"the law of {name!r} built a {type(law).__name__}, not a law.")
    event, batch, dtype = _event_shape_and_dtype(law)
    dtypes = dtype.values() if isinstance(dtype, Mapping) else [dtype]
    batches = batch.values() if isinstance(batch, Mapping) else [batch]
    if any(d != jnp.float64 for d in dtypes):
        raise ValueError(
            f"the law of {name!r} is of dtype {dtype}, not float64; build it from float64 arrays, such as "
            "tfd.Normal(jnp.float64(0.0), jnp.float64(1.0)), or with the family builders."
        )
    if event != expected or any(b != () for b in batches):
        raise ValueError(
            f"the law of {name!r} has event shape {event} and TFP batch shape {batch}, but must be a law "
            f"over the blocks of the components it declares: event shape {expected}, TFP batch shape (). "
            "Labels independent and identically distributed are iid_over_dim, one law per label "
            "independent_over_dim; a joint factor's draws are a dict keyed by its components' names."
        )


def check_factor_keeps_its_structure(variants: Sequence[BoundFactor]) -> None:
    """A factor reading other components has one structure at every draw,
    depending on them only through its law's parameters, which evaluating
    it one draw at a time needs."""
    first, *rest = variants
    for variant in rest:
        if _structure_of(variant.law) != _structure_of(first.law) or variant.by_base_density != first.by_base_density:
            raise ValueError(
                f"the law of {first.name!r} changes its structure with the values it reads: its class, "
                "its parts, or whether it is a pushforward through the components' own bijectors; let "
                "the values enter only through the law's parameters."
            )


def check_law_has_a_density(bound: BoundFactor) -> None:
    """A factor's law, and every law it wraps, has a density against its
    support's reference measure: no discrete law, point mass or LKJ law,
    which probe points and draws cannot tell from one that has."""
    singular = [type(law).__name__ for law in _laws_within(bound.law) if isinstance(law, _LAWS_WITHOUT_A_DENSITY)]
    if singular:
        raise ValueError(
            f"the law of {bound.name!r} is or wraps {sorted(set(singular))}, which has no density against "
            "the reference measure of its components' supports (a discrete law or point mass on an "
            "interval, an LKJ law on the positive-definite matrices); give a law with a density, such as "
            "an InverseWishart for a covariance."
        )


def check_change_of_variables_has_a_measure(bound: BoundFactor) -> None:
    """A factor evaluated by change of variables is over intervals,
    simplices and positive-definite matrices, the supports whose
    Jacobian's reference measure is known."""
    if bound.by_base_density:
        return
    unknown = [c.name for c in bound.components if not isinstance(c.support, (Interval, Simplex, PositiveDefinite))]
    if unknown:
        raise ValueError(
            f"the law of {bound.name!r} is evaluated by change of variables over {unknown}, whose "
            "supports have no known reference measure; write it as a pushforward through the "
            "components' bijectors, pushforward(base, support=...)."
        )


def check_simplex_density_is_a_dirichlet(bound: BoundFactor) -> None:
    """On the simplex, a factor evaluated by change of variables is one
    component's ``Dirichlet``, whose density is against the first ``k - 1``
    coordinates, as the Jacobian is."""
    if bound.by_base_density or not any(isinstance(c.support, Simplex) for c in bound.components):
        return
    inner = bound.law
    if type(inner) in (tfd.Sample, tfd.Independent):
        inner = inner.distribution
    if bound.joint or type(inner) is not tfd.Dirichlet:
        raise ValueError(
            f"the law of {bound.name!r} is a density on the simplex other than a Dirichlet, whose "
            "reference measure is unknown; write it as a pushforward through the components' "
            "bijectors, pushforward(base, support=SIMPLEX)."
        )


def check_declared_support_lies_in_the_laws(bound: BoundFactor) -> None:
    """The law's density is finite at the image of every probe point, so
    the declared supports lie in the law's."""
    if not _probe_log_prob_is_finite(bound, bound.probes()):
        raise ValueError(
            f"the law of {bound.name!r} has no density at some values of its declared support; its own "
            "support is smaller. Declare the support the law has, or choose a law over the whole support."
        )


def check_laws_support_lies_in_the_declared(bound: BoundFactor) -> None:
    """The law puts no mass outside the declared supports, by its own
    default bijector's images and by draws that map to finite theta."""
    by_bijector = _default_bijector_images_lie_in(bound)
    by_draws = _draws_lie_in_the_support(bound)
    if by_bijector is None and by_draws is None:
        raise ValueError(
            f"the law of {bound.name!r} has neither a default event-space bijector nor a sampler, so its "
            "support cannot be checked; give a law that can be sampled."
        )
    if by_bijector is False or by_draws is False:
        supports = ", ".join(c.support.name for c in bound.components)
        raise ValueError(
            f"the law of {bound.name!r} puts mass outside its declared support ({supports}), or on its "
            "boundary, where theta is not finite; declare the support the law has, or choose a law "
            "inside it."
        )


def check_simulator_at_keeps_its_parts(simulator: Simulator, bound: Any, outputs: Sequence[str]) -> None:
    """A simulator at some labels is a simulator of the same name, reading
    what it read and computing exactly the outputs asked for."""
    if not isinstance(bound, Simulator):
        raise TypeError(f"the simulator {simulator.name!r}'s at() returned a {type(bound).__name__}, not a Simulator.")
    kept = sorted(o.name for o in bound.outputs)
    if bound.name != simulator.name or tuple(bound.given) != tuple(simulator.given) or kept != sorted(outputs):
        raise ValueError(
            f"the simulator {simulator.name!r}'s at() returned {bound!r}, which does not read what it read or "
            f"compute exactly {sorted(outputs)}; at() restricts the labels and outputs only."
        )


def check_draws_map_to_finite_theta(name: str, theta: Array) -> None:
    """Every draw maps to a finite theta."""
    if not bool(jnp.all(jnp.isfinite(theta))):
        raise ValueError(
            f"the law of {name!r} drew a value on its support's boundary, whose theta is not finite; "
            "choose a law with less mass at the boundary."
        )


def check_outputs_are_a_dict_of_the_outputs(name: str, out: Any, names: Sequence[str]) -> None:
    """A deterministic of several outputs returns a dict keyed by their names."""
    if not isinstance(out, Mapping) or set(out) != set(names):
        found = sorted(out) if isinstance(out, Mapping) else type(out).__name__
        raise ValueError(
            f"the deterministic {name!r} returned {found}; one of several outputs returns a dict keyed by "
            f"their names, {list(names)}."
        )


def check_deterministic_has_its_block_shapes(bound: BoundDeterministic, values: Mapping[str, Array], lead_ndim: int) -> None:
    """A deterministic returns each output's block shape, which would
    otherwise broadcast or misalign against the labels."""
    for name, expected in bound.block_shapes.items():
        shape = tuple(jnp.shape(values[name]))[lead_ndim:]
        if shape != expected:
            raise ValueError(
                f"the deterministic {bound.name!r} computes {name!r} of shape {shape} for one draw, but its "
                f"indexed_by and element axes give {expected}; return (*index shape, *shape)."
            )


def check_deterministic_lies_in_its_supports(
    bound: BoundDeterministic, values: Mapping[str, Array], theta: np.ndarray
) -> None:
    """A deterministic's outputs at the probe points lie in the supports
    they declare; at the outer probes the closure passes, as does infinity
    at an unbounded end, which float64 underflow and overflow reach."""
    outer = np.abs(theta).max(axis=-1) > _OUTER_PROBE if theta.size else np.zeros(len(theta), dtype=bool)
    for spec in bound.spec.outputs:
        if spec.support is REAL:
            continue
        value = values[spec.name]
        n_probes = value.shape[0]
        inside = spec.support.contains(value).reshape((n_probes, -1)).all(axis=-1)
        lenient = _lies_in_the_support_or_overflows(spec.support, value).reshape((n_probes, -1)).all(axis=-1)
        if not bool(jnp.all(jnp.where(outer, lenient, inside))):
            raise ValueError(
                f"the deterministic {bound.name!r} computes {spec.name!r} outside its declared support "
                f"{spec.support.name!r} at the probe points; declare the support its values have, or REAL."
            )

