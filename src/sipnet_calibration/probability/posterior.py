"""Conditioning: a bound model given the values of some of its random
nodes, as the target inference reads.

Where this sits
---------------
::

    probability.model.FactoredDistribution   (the model at labels)
      -> probability.posterior.condition_on  (Bayes' rule)
      -> Posterior                           (theta's density, and its forms)
      -> an inference algorithm

The posterior
-------------
Let :math:`O` be the observed factors, with values :math:`y`. The
**barren** factors :math:`B` are the unobserved ones with no observed
descendant (through any part); the **target** is the rest,
:math:`T = R \\setminus (O \\cup B)`, and its components are the
**parameters**. With nothing observed, :math:`B = \\varnothing` and
:math:`T = R`. Splitting :math:`O` by whether a factor has a target
ancestor, :math:`O_\\theta` and :math:`O_c`,

.. math::

    \\pi(z_T) = \\prod_{v \\in T} p_v(z_v \\mid z_{\\mathrm{pa}(v)}), \\quad
    L(z_T) = \\prod_{v \\in O_\\theta} p_v(y_v \\mid z_{\\mathrm{pa}(v)}), \\quad
    C = \\prod_{v \\in O_c} p_v(y_v \\mid z_{\\mathrm{pa}(v)}),

every observed value held wherever it is read, and
:math:`p(z_T \\mid y) \\propto \\pi(z_T) L(z_T)`. The barren factors drop
out exactly. In theta, :math:`z_T = T(\\theta)`, the parameters' bijections
in declaration order, and

.. math::

    \\log p(\\theta \\mid y) = \\log \\pi(T(\\theta)) + \\sum_p \\log |J_p(\\theta_p)|
        + \\log L(T(\\theta)) + \\text{const},

a factor that pushes forward through its components' own bijectors being
evaluated by its base density, with no Jacobian.

Data model
----------
Theta is Flat, ``(..., D)``, laid out by ``posterior.parameters.unconstrained``;
y is Flat, ``(N,)``, laid out by ``posterior.observations``, the
:math:`O_\\theta` components in declaration order. Observed values are kept
as labeled values, a DataArray per observed component.

Functions and classes
---------------------
:func:`condition_on`
    Bayes' rule.
:class:`Posterior`
    ``sample_prior``, ``log_prior``, ``log_likelihood``, ``log_density``,
    ``natural_values``, ``to_labeled``, ``describe``.

Notes
-----
No part here runs a simulator: the simulator seam, with the batched
evaluation it brings, arrives in the simulator PR (P5). Until then the
likelihood is a traced function of theta.

Usage
-----
::

    posterior = condition_on(model, {"y": y_observed})
    theta = posterior.sample_prior(jax.random.key(0), 100)   # (100, D)
    posterior.log_density(theta)                              # (100,)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.probability._bound import (
    BoundFactor,
    coords_of,
    deterministics_behind,
    split_reads,
)
from sipnet_calibration.probability._validation import as_count, truncated
from sipnet_calibration.probability.layout import (
    LabeledValues,
    Layout,
    ValuesByName,
    check_flat_ends_in_the_size,
)
from sipnet_calibration.probability.model import FactoredDistribution, block_at_labels
from sipnet_calibration.probability.names import SAMPLE, THETA, THETA_ENTRY
from sipnet_calibration.probability.parts import DeterministicSpec, FactorSpec

__all__ = [
    "Posterior",
    "check_something_is_left_to_infer",
    "condition_on",
]

Array = jax.Array


def condition_on(model: FactoredDistribution, observed: Mapping[str, Any]) -> Posterior:
    """Bayes' rule: *model* given the values *observed* of some random nodes.

    With :math:`O` the observed factors, the barren factors :math:`B`
    (unobserved, no observed descendant) are dropped, and the target is the
    law of the rest, :math:`T`,

    .. math::

        p(z_T \\mid y) \\propto \\prod_{v \\in T} p_v(z_v \\mid z_{\\mathrm{pa}(v)})
            \\prod_{v \\in O} p_v(y_v \\mid z_{\\mathrm{pa}(v)}).

    Observed factors whose kernels depend on no target factor contribute
    only a constant, so observing a hyperparameter at a value holds it
    fixed.

    Parameters
    ----------
    model : FactoredDistribution
    observed : Mapping[str, Any]
        ``{component name: value}``, every component of each observed
        factor; a labeled ``xr.DataArray`` read by label, or an array of the
        block shape. Empty gives the prior over every factor.

    Raises
    ------
    TypeError
        If *model* is not a FactoredDistribution or *observed* not a mapping.
    KeyError
        If a name is not a component a factor declares, or a labeled value
        lacks a label in use.
    ValueError
        If part of a factor's event is observed; a value is not finite or
        lies outside its support; nothing is left to infer; or a target or
        constant observed factor has no finite density at the observed
        values.
    """
    return Posterior(model, observed)


class Posterior:
    """A model conditioned on values: the unnormalized density of its
    parameters in unconstrained coordinates,

    .. math::

        \\log p(\\theta \\mid y) = \\log \\pi_\\theta(\\theta) + \\log L(\\theta) + \\text{const},

    and what an inference algorithm reads from it. It holds no samples.
    Made by :func:`condition_on`.

    Parameters
    ----------
    model, observed:
        Positional-only. As :func:`condition_on`.

    Attributes
    ----------
    model : FactoredDistribution
    parameters : Layout
        The parameters' natural layout; ``parameters.unconstrained`` is
        theta's, of ``D`` entries.
    observations : Layout or None
        The layout of the observed components the likelihood depends on
        (:math:`O_\\theta`), of ``N`` entries; ``None`` when there are none.
    y : jax.Array
        ``(N,)``, their values.
    observed : frozendict of str to xr.DataArray
        Every observed value.
    parameter_names : tuple of str
        The target's components, :math:`T`.
    barren_names : tuple of str
        The barren factors' components, :math:`B`.
    constant_names : tuple of str
        Observed components with no target ancestor, :math:`O_c`: in the
        evidence, not in the likelihood.
    log_constant : float
        :math:`\\log C`, their kernels' log density at the observed values.
    dimension : int
        ``D``.

    Raises
    ------
    TypeError, KeyError, ValueError
        As :func:`condition_on`.
    """

    def __init__(self, model: FactoredDistribution, observed: Mapping[str, Any], /) -> None:
        check_model_is_a_factored_distribution(model)
        check_observed_is_a_mapping(observed)
        spec = model.spec
        for name in observed:
            check_name_is_a_factor_component(name, model)
        observed_parts = {spec._part_of(name).name for name in observed}
        for name in observed_parts:
            check_event_is_wholly_observed(model._factors[name].spec, observed)
        values = {
            name: block_at_labels(spec.component_spec(name), value, model.coords, message_name=f"the observed {name!r}")
            for name, value in observed.items()
        }
        roles = _roles(model, observed_parts)
        check_something_is_left_to_infer(roles["target"])
        factors = model._factors
        names = {role: tuple(c.name for f in parts for c in factors[f].components) for role, parts in roles.items()}
        _set(self, "model", model)
        _set(self, "parameter_names", names["target"])
        _set(self, "barren_names", names["barren"])
        _set(self, "constant_names", names["constant"])
        _set(self, "_target", tuple(factors[f] for f in roles["target"]))
        _set(self, "_likelihood", tuple(factors[f] for f in roles["likelihood"]))
        _set(self, "_constant", tuple(factors[f] for f in roles["constant"]))
        _set(self, "parameters", _layout_of(model, names["target"]))
        _set(self, "observations", _layout_of(model, names["likelihood"]) if names["likelihood"] else None)
        observed_order = [c for c in spec.component_names if c in values]
        _set(self, "observed", frozendict(
            _layout_of(model, observed_order).values_to_labeled({n: values[n] for n in observed_order})
            if observed_order else {}
        ))
        _set(self, "y", (
            self.observations.values_to_flat({n: values[n] for n in names["likelihood"]})
            if self.observations is not None else jnp.zeros((0,), dtype=jnp.float64)
        ))
        _set(self, "_fixed", frozendict({**model._fixed, **values}))
        _set(self, "_slices", tuple(_slice_of(self.parameters.unconstrained, bound) for bound in self._target))
        _set(self, "log_constant", self._log_constant())
        check_target_density_is_finite_at_the_held_values(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a Posterior is frozen; condition again rather than setting {name!r}.")

    def __repr__(self) -> str:
        n = 0 if self.observations is None else self.observations.size
        return f"Posterior(D={self.dimension}, N={n}, parameters={list(self.parameter_names)})"

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """``D``, the number of entries of theta."""
        return self.parameters.unconstrained.size

    def describe(self) -> pd.DataFrame:
        """One row per component and input, indexed by ``name``: ``role``
        (``"parameter"``, ``"observed"``, ``"observed, constant"``,
        ``"barren"``, ``"input"`` or ``"computed"``), and the declaring
        part's name and class (empty for an input)."""
        spec = self.model.spec
        observed = {c.name for b in self._likelihood for c in b.components}
        rows = []
        for name in (*spec.component_names, *spec.input_names):
            part = spec._owner.get(name)
            if part is None:
                role = "input"
            elif isinstance(part, DeterministicSpec):
                role = "computed"
            elif name in self.parameter_names:
                role = "parameter"
            elif name in observed:
                role = "observed"
            elif name in self.constant_names:
                role = "observed, constant"
            else:
                role = "barren"
            rows.append({
                "name": name, "role": role, "part": "" if part is None else part.name,
                "class": "" if part is None else type(part).__name__,
            })
        return pd.DataFrame(rows).set_index("name")

    # ── evaluation ────────────────────────────────────────────────────────────

    def sample_prior(self, key: Array, n: int) -> Array:
        """``n`` draws of theta from :math:`\\pi`, ``(n, D)``: the target
        factors sampled ancestrally with every observed value held, each
        keyed by ``jax.random.fold_in(key, crc32(name))``.

        Raises
        ------
        TypeError
            If *n* is not an integer.
        ValueError
            If *n* is negative, or a draw lands on a support's boundary,
            where theta is not finite; the message names the factor.
        """
        n = as_count(n, message_name="n")
        theta = jnp.zeros((n, self.dimension), dtype=jnp.float64)
        needed = {bound.name for bound in self._target} | deterministics_behind(
            self.model.spec, [g for bound in self._target for g in bound.spec.given]
        )
        drawn: dict[str, Array] = {}
        self.model._ancestral(key, n, self._fixed, needed=needed, theta_out=drawn)
        for bound, positions in zip(self._target, self._slices):
            theta = theta.at[:, positions].set(drawn[bound.name])
        return theta

    def log_prior(self, theta: Any) -> Array:
        """:math:`\\log \\pi_\\theta(\\theta)`, ``(..., D) -> (...)``,
        normalized, ``-inf`` where a target factor's density is not finite.
        Each factor contributes by base density where it pushes forward
        through its components' own bijectors, else by change of variables.
        Traceable under ``jax.jit``, ``jax.grad`` and ``jax.vmap``.

        Raises
        ------
        ValueError
            If the last axis of *theta* is not ``D`` long.
        """
        theta, lead = self._theta(theta)
        per_draw, fixed = self._values_read(theta, lead, self._target)
        total = jnp.zeros(lead, dtype=jnp.float64)
        for bound, positions in zip(self._target, self._slices):
            reads, held = split_reads(bound.spec.given, per_draw, fixed)
            total = total + _finite_or_minus_infinity(bound.log_prob_theta(theta[..., positions], reads, held))
        return total

    def log_likelihood(self, theta: Any) -> Array:
        """:math:`\\log L(T(\\theta))`, ``(..., D) -> (...)``: each
        :math:`O_\\theta` factor's density at its observed values, ``-inf``
        where one is not finite; zero when nothing the target reaches is
        observed. Traceable.

        Raises
        ------
        ValueError
            If the last axis of *theta* is not ``D`` long.
        """
        theta, lead = self._theta(theta)
        per_draw, fixed = self._values_read(theta, lead, self._likelihood)
        total = jnp.zeros(lead, dtype=jnp.float64)
        for bound in self._likelihood:
            own = {name: self._fixed[name] for name in bound.names}
            reads, held = split_reads(bound.spec.given, per_draw, fixed)
            total = total + _finite_or_minus_infinity(bound.log_prob_natural(own, reads, held, lead))
        return total

    def log_density(self, theta: Any) -> Array:
        """``log_prior(theta) + log_likelihood(theta)``, ``(..., D) ->
        (...)``: the unnormalized log posterior in theta. Traceable."""
        return self.log_prior(theta) + self.log_likelihood(theta)

    def natural_values(self, theta: Any) -> ValuesByName:
        """The parameters' values at theta, and those of every deterministic
        computable from the parameters, observed values and inputs,
        ``{name: (..., *block)}``. Traceable.

        Raises
        ------
        ValueError
            If the last axis of *theta* is not ``D`` long.
        """
        theta, lead = self._theta(theta)
        natural = self._parameter_values(theta)
        computable = self._computable_names()
        per_draw, fixed = self.model._computed(natural, self._fixed, lead, computable)
        out = dict(natural)
        for name in computable:
            value = per_draw[name] if name in per_draw else jnp.broadcast_to(fixed[name], (*lead, *fixed[name].shape))
            out[name] = value
        return out

    def to_labeled(self, theta: Any) -> LabeledValues:
        """A batch of theta as labeled values, batch dim ``sample``: the
        natural values of :meth:`natural_values`, and ``"theta"`` on
        ``(sample, theta_entry)``, its entries labeled by
        ``parameters.unconstrained.entry_names``.

        Raises
        ------
        ValueError
            If *theta* is not ``(J, D)``.
        """
        theta = jnp.asarray(theta, dtype=jnp.float64)
        check_theta_is_a_batch(theta.shape, self.dimension)
        values = self.natural_values(theta)
        computable = self._computable_names()
        labeled = self.parameters.values_to_labeled({n: values[n] for n in self.parameter_names}, batch_dims=(SAMPLE,))
        if computable:
            layout = _layout_of(self.model, computable)
            labeled |= layout.values_to_labeled({n: values[n] for n in computable}, batch_dims=(SAMPLE,))
        labeled[THETA] = xr.DataArray(
            np.asarray(theta), dims=(SAMPLE, THETA_ENTRY),
            coords={SAMPLE: np.arange(theta.shape[0], dtype=np.int64),
                    THETA_ENTRY: np.asarray(self.parameters.unconstrained.entry_names, dtype=object)},
            name=THETA,
        )
        return labeled

    # ── supporting methods ────────────────────────────────────────────────────

    def _theta(self, theta: Any) -> tuple[Array, tuple[int, ...]]:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        check_flat_ends_in_the_size(theta.shape, self.dimension)
        return theta, tuple(theta.shape[:-1])

    def _parameter_values(self, theta: Array) -> dict[str, Array]:
        """The parameters' natural values at theta, ``{name: (..., *block)}``."""
        unconstrained = self.parameters.unconstrained.flat_to_values(theta)
        return {c.name: c.bijector.forward(unconstrained[c.name]) for c in self.parameters.components}

    def _values_read(self, theta: Array, lead: tuple[int, ...], readers: Sequence[BoundFactor]) -> tuple[dict, dict]:
        """What *readers* read at theta: the parameters' values, observed
        values, inputs, and the deterministics behind them."""
        if not any(bound.spec.given for bound in readers):
            return {}, dict(self._fixed)
        natural = self._parameter_values(theta)
        return self.model._computed(natural, self._fixed, lead, [g for b in readers for g in b.spec.given])

    def _computable_names(self) -> tuple[str, ...]:
        """The deterministic components computable from the parameters,
        observed values and inputs, in declaration order."""
        spec = self.model.spec
        available = set(self.parameter_names) | set(self._fixed)
        out = []
        for part in spec._order:
            if isinstance(part, DeterministicSpec) and all(g in available for g in part.given):
                available.update(c.name for c in part.outputs)
                out.extend(c.name for c in part.outputs)
        return tuple(c for c in spec.component_names if c in out)

    def _log_constant(self) -> float:
        """:math:`\\log C`, the :math:`O_c` factors' density at the observed
        values, each checked finite."""
        if not self._constant:
            return 0.0
        _, fixed = self.model._computed({}, self._fixed, (), [g for b in self._constant for g in b.spec.given])
        total = 0.0
        for bound in self._constant:
            own = {name: self._fixed[name] for name in bound.names}
            value = float(bound.log_prob_natural(own, {}, {g: fixed[g] for g in bound.spec.given}, ()))
            check_density_is_finite_at_the_observed_values(bound.name, value)
            total += value
        return total


# ── helpers ───────────────────────────────────────────────────────────────────


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _roles(model: FactoredDistribution, observed_parts: set[str]) -> dict[str, tuple[str, ...]]:
    """Each factor's role, by name, in declaration order: ``target``,
    ``barren``, ``likelihood`` (:math:`O_\\theta`) and ``constant``
    (:math:`O_c`)."""
    spec = model.spec
    factors = [p.name for p in spec.parts if isinstance(p, FactorSpec)]
    if observed_parts:
        barren = {f for f in factors if f not in observed_parts and not (spec._descendants(f) & observed_parts)}
    else:
        barren = set()
    target = [f for f in factors if f not in observed_parts and f not in barren]
    likelihood = [f for f in factors if f in observed_parts and spec._ancestors([f]) & set(target)]
    return {
        "target": tuple(target),
        "barren": tuple(f for f in factors if f in barren),
        "likelihood": tuple(likelihood),
        "constant": tuple(f for f in factors if f in observed_parts and f not in likelihood),
    }


def _layout_of(model: FactoredDistribution, names: Sequence[str]) -> Layout:
    """The layout of the components *names*, in that order, at the model's
    labels."""
    specs = [model.spec.component_spec(name) for name in names]
    return Layout(specs, coords=coords_of(specs, model.coords))


def _slice_of(unconstrained: Layout, bound: BoundFactor) -> np.ndarray:
    """A target factor's entries of theta, its components' in event order."""
    return np.concatenate([np.arange(unconstrained.slice_of(n).start, unconstrained.slice_of(n).stop) for n in bound.names])


def _finite_or_minus_infinity(log_density: Array) -> Array:
    return jnp.where(jnp.isfinite(log_density), log_density, -jnp.inf)


# ── checks ────────────────────────────────────────────────────────────────────


def check_model_is_a_factored_distribution(model: Any) -> None:
    """What is conditioned is a bound model."""
    if not isinstance(model, FactoredDistribution):
        raise TypeError(
            f"condition_on takes a FactoredDistribution, got {type(model).__name__}; bind the model spec first."
        )


def check_observed_is_a_mapping(observed: Any) -> None:
    """Observed values are ``{component name: value}``."""
    if not isinstance(observed, Mapping):
        raise TypeError(f"observed must be a mapping {{component name: value}}, got {type(observed).__name__}.")


def check_name_is_a_factor_component(name: Any, model: FactoredDistribution) -> None:
    """What is observed is a component a factor declares: a deterministic's
    is computed and an input's is bound."""
    spec = model.spec
    if not isinstance(name, str) or not isinstance(spec._owner.get(name), FactorSpec):
        factor_components = [c for p in spec.parts if isinstance(p, FactorSpec) for c in (s.name for s in p.event)]
        raise KeyError(
            f"{name!r} is not a component a factor declares, so it cannot be observed; observe one of "
            f"{truncated(factor_components)}."
        )


def check_event_is_wholly_observed(spec: FactorSpec, observed: Mapping[str, Any]) -> None:
    """A factor's event is observed whole or not at all."""
    missing = [c.name for c in spec.event if c.name not in observed]
    if missing:
        raise ValueError(
            f"part of the factor {spec.name!r} is observed, but not {missing}; observe its whole event, or "
            "split the factor."
        )


def check_something_is_left_to_infer(target: Sequence[str]) -> None:
    """Conditioning leaves at least one factor to infer."""
    if not target:
        raise ValueError("every factor is observed or barren, so nothing is left to infer; observe less.")


def check_density_is_finite_at_the_observed_values(name: str, value: float) -> None:
    """An observed factor's density is finite at the observed values."""
    if not np.isfinite(value):
        raise ValueError(
            f"the factor {name!r} has no finite density at the observed values; check them against its law."
        )


def check_target_density_is_finite_at_the_held_values(posterior: Posterior) -> None:
    """Each target factor that reads an observed value has a finite density
    at theta = 0 with it held."""
    held = set(posterior.observed)
    theta = jnp.zeros((1, posterior.dimension), dtype=jnp.float64)
    per_draw, fixed = posterior._values_read(theta, (1,), posterior._target)
    for bound, positions in zip(posterior._target, posterior._slices):
        if not any(g in held for g in bound.spec.given):
            continue
        reads, kept = split_reads(bound.spec.given, per_draw, fixed)
        value = float(bound.log_prob_theta(theta[..., positions], reads, kept)[0])
        check_density_is_finite_at_the_observed_values(bound.name, value)


def check_theta_is_a_batch(shape: tuple[int, ...], dimension: int) -> None:
    """Theta is a batch, ``(J, D)``."""
    if len(shape) != 2 or shape[1] != dimension:
        raise ValueError(f"theta has shape {shape}; give a batch (J, {dimension}).")
