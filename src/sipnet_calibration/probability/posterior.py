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

**Simulators.** No target factor may have a simulator among its
ancestors, so :math:`\\pi` is evaluated and sampled without one. A
simulator computes only the outputs the likelihood depends on, once per
batch of theta (:meth:`Posterior.evaluate`). With :math:`\\mathcal V` the
set of :math:`z_T` at which each of them is computed, the posterior is the
truncation :math:`p(z_T \\mid y) \\propto \\pi(z_T) L(z_T) \\mathbf 1\\{z_T
\\in \\mathcal V\\}`: a sample is **valid** when its outputs were computed
and every :math:`O_\\theta` factor's density is finite there, and its log
likelihood is :math:`-\\infty` otherwise.

**Gaussian factors.** A factor whose law is a
:class:`~sipnet_calibration.probability.parts.GaussianSpec`, and whose
covariance is computable from the held values alone (observed values,
inputs, constants, and what is computed from them), has its covariance
built and factored once, here, and refused if it is not positive definite.
When every :math:`O_\\theta` factor is such a factor, the likelihood is
:math:`y \\sim \\mathcal N(G(\\theta), R)`, :math:`G` their means and :math:`R`
their covariances, block-diagonal in y's order
(:meth:`Posterior.gaussian_likelihood`), which an ensemble Kalman method
reads.

Data model
----------
Theta is Flat, ``(..., D)``, laid out by ``posterior.parameters.unconstrained``;
y is Flat, ``(N,)``, laid out by ``posterior.observations``, the
:math:`O_\\theta` components in declaration order. Observed values are kept
as labeled values, a DataArray per observed component.
:class:`PosteriorEvaluation`, what one evaluation computed, holds ``(J,)``
arrays and values by name, ``(J, *block)``.

Functions and classes
---------------------
:func:`condition_on`
    Bayes' rule.
:class:`Posterior`
    ``sample_prior``, ``log_prior``, ``evaluate``, ``log_likelihood``,
    ``log_density``, ``log_density_given``, ``predict``, ``replicate``,
    ``simulator_inputs``, ``natural_values``, ``to_labeled``,
    ``gaussian_likelihood``, ``describe``.
:class:`PosteriorEvaluation`
    One batch of theta, evaluated.
:class:`GaussianLikelihood`
    The likelihood as :math:`y \\sim \\mathcal N(G(\\theta), R)`.

Usage
-----
::

    posterior = condition_on(model, {"y": y_observed})
    theta = posterior.sample_prior(jax.random.key(0), 100)   # (100, D)
    evaluation = posterior.evaluate(theta)                    # one simulator call
    evaluation.log_density, evaluation.valid                  # (100,), (100,)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.probability import _linalg
from sipnet_calibration.probability._bound import (
    BoundFactor,
    check_gaussian_covariance_is_positive_definite,
    coords_of,
    deterministics_behind,
    finite_or_minus_infinity,
    simulator_at,
    simulator_outputs_behind,
    split_reads,
)
from sipnet_calibration.probability._probes import corner_points
from sipnet_calibration.probability._validation import as_count, truncated
from sipnet_calibration.probability.layout import (
    LabeledValues,
    Layout,
    ValuesByName,
    check_flat_ends_in_the_size,
)
from sipnet_calibration.probability.model import FactoredDistribution, block_at_labels
from sipnet_calibration.probability.names import SAMPLE, THETA, THETA_ENTRY
from sipnet_calibration.probability.parts import DeterministicSpec, FactorSpec, Simulator, SimulatorOutput

__all__ = [
    "GaussianLikelihood",
    "Posterior",
    "PosteriorEvaluation",
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
    fixed. Each simulator the likelihood depends on is restricted to the
    outputs it depends on, and its ``check_given`` is called at the corner
    points of the target.

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
        lies outside its support; nothing is left to infer; a target factor,
        or an observed one with no target ancestor, has a simulator among
        its ancestors; a target or constant observed factor has no finite
        density at the observed values; a Gaussian factor's covariance that
        the held values fix is not positive definite; or a simulator's
        ``check_given`` fails.
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
        The barren factors' components, :math:`B`, drawn by :meth:`predict`.
    constant_names : tuple of str
        Observed components with no target ancestor, :math:`O_c`: in the
        evidence, not in the likelihood.
    log_constant : float
        :math:`\\log C`, their kernels' log density at the observed values.
    dimension : int
        ``D``.
    simulators : frozendict of str to Simulator
        Each simulator the likelihood depends on, computing only the outputs
        it depends on.
    simulator_free_positions : numpy.ndarray
        ``int64``, ascending: the entries of theta with no simulator among
        their descendants, which :meth:`log_density_given` varies.

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
            name: block_at_labels(spec.component_spec(name), value, model.coords, message_name="observed")
            for name, value in observed.items()
        }
        roles = _roles(model, observed_parts)
        check_something_is_left_to_infer(roles["target"])
        check_no_factor_reads_a_simulator(model, roles["target"], role="target")
        check_no_factor_reads_a_simulator(model, roles["constant"], role="observed factor with no target ancestor")
        factors = model._factors
        names = {role: tuple(c.name for f in parts for c in factors[f].components) for role, parts in roles.items()}
        _set(self, "model", model)
        _set(self, "parameter_names", names["target"])
        _set(self, "barren_names", names["barren"])
        _set(self, "constant_names", names["constant"])
        _set(self, "_target", tuple(factors[f] for f in roles["target"]))
        _set(self, "_likelihood", tuple(factors[f] for f in roles["likelihood"]))
        _set(self, "_constant", tuple(factors[f] for f in roles["constant"]))
        _set(self, "_barren", tuple(factors[f] for f in roles["barren"]))
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
        _set(self, "_target", tuple(_with_held_covariance(self, bound) for bound in self._target))
        _set(self, "_likelihood", tuple(_with_held_covariance(self, bound) for bound in self._likelihood))
        _set(self, "_slices", tuple(_slice_of(self.parameters.unconstrained, bound) for bound in self._target))
        _set(self, "simulators", frozendict(_simulators_read(model, self._likelihood)))
        _set(self, "simulator_free_positions", _simulator_free_positions(self))
        _set(self, "log_constant", self._log_constant())
        check_target_density_is_finite_at_the_held_values(self)
        check_simulators_take_the_corner_points(self)

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
            elif isinstance(part, (DeterministicSpec, Simulator)):
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
        keyed by ``jax.random.fold_in(key, crc32(name))``. No simulator runs.

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
            total = total + finite_or_minus_infinity(bound.log_prob_theta(theta[..., positions], reads, held))
        return total

    def evaluate(self, theta: Any) -> PosteriorEvaluation:
        """Evaluate the target at a batch of theta, running each simulator
        once.

        Parameters
        ----------
        theta:
            ``(J, D)``, or ``(D,)`` for one sample; finite.

        Returns
        -------
        PosteriorEvaluation

        Raises
        ------
        ValueError
            If *theta* is not ``(J, D)`` or ``(D,)`` with ``J >= 1``, not
            finite, or traced.
        TypeError, ValueError
            If a simulator returns other than a whole ``SimulatorOutput``.
        Exception
            Whatever a simulator's machinery raises.
        """
        theta = self._theta_batch(theta)
        lead = (theta.shape[0],)
        runs: dict[str, SimulatorOutput] = {}
        natural = self._parameter_values(theta)
        computable = self._computable_names()
        read = [g for bound in self._likelihood for g in bound.spec.given]
        per_draw, fixed = self.model._computed(
            natural, self._fixed, lead, [*read, *computable], simulators=self.simulators, runs=runs
        )
        log_likelihood, finite = self._likelihood_at(per_draw, fixed, lead)
        simulator_valid = jnp.asarray(_all_computed(runs, lead[0]))
        valid = simulator_valid & finite
        log_prior = self.log_prior(theta)
        log_likelihood = jnp.where(valid, log_likelihood, -jnp.inf)
        spec = self.model.spec
        computed = [n for n in spec.component_names if isinstance(spec._owner[n], (DeterministicSpec, Simulator))]
        values = {
            n: per_draw[n] if n in per_draw else jnp.broadcast_to(fixed[n], (*lead, *fixed[n].shape))
            for n in computed if n in per_draw or n in fixed
        }
        return PosteriorEvaluation(
            theta=theta,
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            log_density=log_prior + log_likelihood,
            valid=valid,
            simulator_valid=simulator_valid,
            values=frozendict(values),
            simulator_records=frozendict({name: run.record for name, run in runs.items()}),
        )

    def log_likelihood(self, theta: Any) -> Array:
        """:math:`\\log L(T(\\theta))`: each :math:`O_\\theta` factor's density
        at its observed values, ``-inf`` where one is not finite or a
        simulator output it reads was not computed; zero when nothing the
        target reaches is observed. Without a simulator, a traced function,
        ``(..., D) -> (...)``; with one, :meth:`evaluate`'s, ``(J, D) ->
        (J,)`` and ``(D,) -> ()``.

        Raises
        ------
        ValueError
            If the last axis of *theta* is not ``D`` long; with a simulator,
            as :meth:`evaluate`.
        """
        if self.simulators:
            return self._of_one_or_a_batch(theta, lambda evaluation: evaluation.log_likelihood)
        theta, lead = self._theta(theta)
        per_draw, fixed = self._values_read(theta, lead, self._likelihood)
        return self._likelihood_at(per_draw, fixed, lead)[0]

    def log_density(self, theta: Any) -> Array:
        """``log_prior(theta) + log_likelihood(theta)``: the unnormalized log
        posterior in theta. Without a simulator, traced, ``(..., D) ->
        (...)``; with one, :meth:`evaluate`'s, ``(J, D) -> (J,)`` and ``(D,)
        -> ()``."""
        if self.simulators:
            return self._of_one_or_a_batch(theta, lambda evaluation: evaluation.log_density)
        return self.log_prior(theta) + self.log_likelihood(theta)

    def log_density_given(self, evaluation: PosteriorEvaluation) -> Callable[[Array], Array]:
        """The log posterior as a function of the entries of theta no
        simulator depends on, every other entry and every simulator output
        held at *evaluation*'s: ``(J, D_free) -> (J,)``, traceable, ``-inf``
        where a held simulator output was not computed.

        Its gradient is the posterior's in those entries, which sit at
        :attr:`simulator_free_positions` of theta.

        Raises
        ------
        TypeError
            If *evaluation* is not a :class:`PosteriorEvaluation`.
        ValueError
            If its theta is not ``(J, D)``.
        KeyError
            If it lacks a simulator output: it is another posterior's.
        """
        check_evaluation_is_an_evaluation(evaluation)
        check_theta_is_a_batch(tuple(evaluation.theta.shape), self.dimension)
        free = jnp.asarray(self.simulator_free_positions)
        held_theta = evaluation.theta
        lead = (held_theta.shape[0],)
        outputs = [o.name for simulator in self.simulators.values() for o in simulator.outputs]
        computed = evaluation.simulator_valid
        # A failed row's outputs are NaN; zero them, so the row's discarded
        # branch of the where below has a finite gradient, not NaN.
        held = {
            name: jnp.where(computed.reshape((-1,) + (1,) * (evaluation.values[name].ndim - 1)), evaluation.values[name], 0.0)
            for name in outputs
        }

        def log_density(theta_free: Array) -> Array:
            theta = held_theta.at[:, free].set(jnp.asarray(theta_free, dtype=jnp.float64))
            per_draw, fixed = self.model._computed(
                {**self._parameter_values(theta), **held}, self._fixed, lead,
                [g for bound in self._likelihood for g in bound.spec.given], simulators=self.simulators,
            )
            log_likelihood, _ = self._likelihood_at(per_draw, fixed, lead)
            return jnp.where(computed, self.log_prior(theta) + log_likelihood, -jnp.inf)

        return log_density

    def predict(self, key: Array, theta: Any) -> tuple[ValuesByName, dict[str, Array]]:
        """Draws of the barren components at a batch of theta, each factor
        sampled once per sample, keyed by ``jax.random.fold_in(key,
        crc32(name))``, with the observed values held: the posterior
        predictive of what was not observed, such as a validation source.
        The simulator outputs only barren factors read are computed here.

        Returns
        -------
        (ValuesByName, dict of str to jax.Array)
            ``{name: (J, *block)}``, ``NaN`` where a simulator output it
            depends on was not computed, and per component ``(J,)`` bool,
            whether each was.

        Raises
        ------
        ValueError, TypeError, Exception
            As :meth:`evaluate`.
        """
        return self._drawn(key, theta, self._barren, self._fixed, outputs_from="model")

    def replicate(self, key: Array, theta: Any) -> tuple[ValuesByName, dict[str, Array]]:
        """Replicated draws of the :math:`O_\\theta` components at a batch of
        theta, each factor sampled once per sample from its kernel given the
        sample, keyed as :meth:`predict`'s: a posterior predictive check of
        the observations. :math:`O_c` components are held, and not
        replicated. Returns as :meth:`predict`."""
        replicated = {c.name for bound in self._likelihood for c in bound.components}
        fixed = {name: value for name, value in self._fixed.items() if name not in replicated}
        return self._drawn(key, theta, self._likelihood, fixed, outputs_from="posterior")

    def simulator_inputs(self, theta: Any, simulator_name: str) -> LabeledValues:
        """What the simulator *simulator_name* reads at a batch of theta, as
        it would receive it, ``sample`` labeled ``0`` to ``J - 1``: for
        running it with other reductions, a daily output say.

        Raises
        ------
        KeyError
            If the model has no simulator *simulator_name*.
        ValueError, TypeError, Exception
            As :meth:`evaluate`, when a simulator upstream of it runs.
        """
        check_simulator_is_the_models(simulator_name, self.model)
        theta = self._theta_batch(theta)
        return self._simulator_inputs_at(theta, simulator_name)

    def natural_values(self, theta: Any) -> ValuesByName:
        """The parameters' values at theta, and those of every deterministic
        computable from the parameters, observed values and inputs with no
        simulator, ``{name: (..., *block)}``. Traceable.

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

    def gaussian_likelihood(self) -> GaussianLikelihood:
        """The likelihood as :math:`y \\sim \\mathcal N(G(\\theta), R)`: every
        :math:`O_\\theta` factor Gaussian, its covariance held.

        Raises
        ------
        ValueError
            If nothing observed depends on the parameters, or an
            :math:`O_\\theta` factor's law is not a ``GaussianSpec``, or its
            covariance reads what the parameters decide; the message names
            the factor and those parameters, and suggests observing them at
            values to hold them.
        """
        check_something_observed_depends_on_the_parameters(self)
        for bound in self._likelihood:
            check_likelihood_factor_is_gaussian_with_a_held_covariance(self, bound)
        operators = [bound.held_covariance for bound in self._likelihood]
        return GaussianLikelihood(
            posterior=self,
            y=self.y,
            noise_covariance=_linalg.block_diag(*operators),
            mean_names=tuple(bound.spec.law.mean[0] for bound in self._likelihood),
        )

    # ── supporting methods ────────────────────────────────────────────────────

    def _theta(self, theta: Any) -> tuple[Array, tuple[int, ...]]:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        check_flat_ends_in_the_size(theta.shape, self.dimension)
        return theta, tuple(theta.shape[:-1])

    def _theta_batch(self, theta: Any) -> Array:
        """*theta* as a finite batch, ``(J, D)``, ``J >= 1``."""
        check_theta_is_concrete(theta)
        theta = jnp.asarray(theta, dtype=jnp.float64)
        theta = theta[None] if theta.ndim == 1 else theta
        check_theta_is_a_batch(tuple(theta.shape), self.dimension)
        check_theta_has_a_row(tuple(theta.shape))
        check_theta_is_finite(np.asarray(theta))
        return theta

    def _of_one_or_a_batch(self, theta: Any, read: Callable[[PosteriorEvaluation], Array]) -> Array:
        """``read(evaluate(theta))``, ``()`` for a ``(D,)`` theta."""
        out = read(self.evaluate(theta))
        return out[0] if jnp.ndim(theta) == 1 else out

    def _parameter_values(self, theta: Array) -> dict[str, Array]:
        """The parameters' natural values at theta, ``{name: (..., *block)}``."""
        unconstrained = self.parameters.unconstrained.flat_to_values(theta)
        return {c.name: c.bijector.forward(unconstrained[c.name]) for c in self.parameters.components}

    def _values_read(self, theta: Array, lead: tuple[int, ...], readers: Sequence[BoundFactor]) -> tuple[dict, dict]:
        """What *readers* read at theta: the parameters' values, observed
        values, inputs, and the deterministics behind them, with no
        simulator."""
        if not any(bound.spec.given for bound in readers):
            return {}, dict(self._fixed)
        natural = self._parameter_values(theta)
        return self.model._computed(natural, self._fixed, lead, [g for b in readers for g in b.spec.given])

    def _likelihood_at(self, per_draw: Mapping[str, Array], fixed: Mapping[str, Array], lead: tuple[int, ...]) -> tuple[Array, Array]:
        """``(log L, every factor's density finite)``, each ``lead``, from
        what the :math:`O_\\theta` factors read; ``-inf`` where a factor's
        density is not finite."""
        total = jnp.zeros(lead, dtype=jnp.float64)
        finite = jnp.ones(lead, dtype=bool)
        for bound in self._likelihood:
            own = {name: self._fixed[name] for name in bound.names}
            reads, held = split_reads(bound.spec.given, per_draw, fixed)
            density = bound.log_prob_natural(own, reads, held, lead)
            finite = finite & jnp.isfinite(density)
            total = total + finite_or_minus_infinity(density)
        return total, finite

    def _simulator_inputs_at(self, theta: Array, simulator_name: str) -> LabeledValues:
        """What a simulator reads at a ``(J, D)`` theta, any simulator
        upstream of it run for the outputs it reads alone."""
        model = self.model
        simulator = model.simulators[simulator_name]
        lead = (theta.shape[0],)
        upstream = _simulators_restricted(model, simulator.given)
        per_draw, fixed = model._computed(
            self._parameter_values(theta), self._fixed, lead, simulator.given, simulators=upstream
        )
        return model._labeled_given(simulator, per_draw, fixed, lead)

    def _drawn(
        self, key: Array, theta: Any, bounds: Sequence[BoundFactor], fixed: Mapping[str, Array], *, outputs_from: str
    ) -> tuple[ValuesByName, dict[str, Array]]:
        """Draws of *bounds*' components at a batch of theta, *fixed* held,
        each with the draws at which the simulator outputs behind it were
        computed. Simulators are the posterior's, or the model's restricted
        to what *bounds* read (``outputs_from="model"``)."""
        theta = self._theta_batch(theta)
        if not bounds:
            return {}, {}
        spec, n = self.model.spec, theta.shape[0]
        read = [g for bound in bounds for g in bound.spec.given]
        simulators = self.simulators if outputs_from == "posterior" else _simulators_restricted(self.model, read)
        needed = {bound.name for bound in bounds} | deterministics_behind(spec, read)
        runs: dict[str, SimulatorOutput] = {}
        values, _ = self.model._ancestral(
            key, n, fixed, needed=needed, per_draw=self._parameter_values(theta), simulators=simulators, runs=runs,
        )
        drawn = {c.name: values[c.name] for bound in bounds for c in bound.components}
        computed = {}
        for name in drawn:
            behind = self.model._draws_computed([name], runs)
            computed[name] = jnp.ones((n,), dtype=bool) if behind is None else jnp.asarray(behind)
        return drawn, computed

    def _computable_names(self) -> tuple[str, ...]:
        """The deterministic components computable from the parameters,
        observed values and inputs with no simulator, in declaration order."""
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


@dataclass(frozen=True, eq=False, kw_only=True)
class PosteriorEvaluation:
    """What one :meth:`Posterior.evaluate` computed. Compared and hashed by
    identity.

    Attributes
    ----------
    theta : jax.Array
        ``(J, D)``.
    log_prior : jax.Array
        ``(J,)``, :math:`\\log \\pi` everywhere, as :meth:`Posterior.log_prior`.
    log_likelihood, log_density : jax.Array
        ``(J,)``; ``-inf`` where ``valid`` is false.
    valid : jax.Array
        ``(J,)`` bool: every simulator output the likelihood depends on was
        computed, and every :math:`O_\\theta` factor's log density is finite.
    simulator_valid : jax.Array
        ``(J,)`` bool: those simulator outputs were computed. Where it holds
        and ``valid`` does not, the traced part failed numerically.
    values : frozendict of str to jax.Array
        Every deterministic component computed, ``(J, *block)``: those the
        likelihood reads, simulator outputs among them, and those
        computable without a simulator; ``NaN`` where not computed.
    simulator_records : frozendict of str to Any
        Each simulator's ``SimulatorOutput.record``, by its name.
    """

    theta: Array
    log_prior: Array
    log_likelihood: Array
    log_density: Array
    valid: Array
    simulator_valid: Array
    values: Mapping[str, Array]
    simulator_records: Mapping[str, Any]


@dataclass(frozen=True, eq=False, kw_only=True)
class GaussianLikelihood:
    """The likelihood :math:`y \\sim \\mathcal N(G(\\theta), R)` over
    :math:`O_\\theta`: :math:`G(\\theta)` the Gaussian factors' means and
    :math:`R` their covariances, which depend on no parameter. Made by
    :meth:`Posterior.gaussian_likelihood`. Compared and hashed by identity.

    Attributes
    ----------
    posterior : Posterior
    y : jax.Array
        ``(N,)``, ``posterior.y``.
    noise_covariance : PSDLinOp
        :math:`R`, one of pyEKI's positive-definite operators: block-diagonal
        over the observed factors, in ``posterior.observations``' order, each
        block its factor's covariance.
    mean_names : tuple of str
        Each factor's mean component, in that order.
    """

    posterior: Posterior
    y: Array
    noise_covariance: Any
    mean_names: tuple[str, ...]

    def forward(self, theta: Any) -> tuple[Array, Array, PosteriorEvaluation]:
        """:math:`G(\\theta)`, each factor's mean at a batch of theta in
        ``posterior.observations``' order, from one
        :meth:`Posterior.evaluate`.

        Returns
        -------
        (jax.Array, jax.Array, PosteriorEvaluation)
            The predictions ``(J, N)``, ``NaN`` in an invalid sample; the
            evaluation's ``valid``, ``(J,)``; and the evaluation.

        Raises
        ------
        ValueError, TypeError, Exception
            As :meth:`Posterior.evaluate`.
        """
        evaluation = self.posterior.evaluate(theta)
        n = evaluation.theta.shape[0]
        natural = self.posterior._parameter_values(evaluation.theta)
        held = self.posterior._fixed
        blocks = []
        for name in self.mean_names:
            if name in evaluation.values:
                value = evaluation.values[name]
            elif name in natural:
                value = natural[name]
            else:
                value = jnp.broadcast_to(held[name], (n, *jnp.shape(held[name])))
            blocks.append(jnp.reshape(value, (n, -1)))
        predictions = jnp.concatenate(blocks, axis=1)
        predictions = jnp.where(evaluation.valid[:, None], predictions, jnp.nan)
        return predictions, evaluation.valid, evaluation


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


def _with_held_covariance(posterior: Posterior, bound: BoundFactor) -> BoundFactor:
    """A Gaussian factor with its covariance built and factored once, when
    the held values (observed values, inputs, constants, and what is
    computed from them with no simulator) fix it; any other factor as it
    is."""
    if not bound.gaussian:
        return bound
    names = [n for n in bound.spec.law.covariance.reads if n not in bound.fixed_reads]
    if simulator_outputs_behind(posterior.model.spec, names):
        return bound
    _, fixed = posterior.model._computed({}, posterior._fixed, (), names)
    if not all(name in fixed for name in names):
        return bound
    operator = bound.covariance.operator(bound.covariance_reads(fixed))
    check_gaussian_covariance_is_positive_definite(bound.name, operator)
    return dataclasses.replace(bound, held_covariance=operator)


def _parameters_read_by_the_covariance(posterior: Posterior, bound: BoundFactor) -> list[str]:
    """The parameters a Gaussian factor's covariance depends on, through any
    part."""
    spec = posterior.model.spec
    owners = [spec._owner[n].name for n in bound.spec.law.covariance.reads if n in spec._owner]
    ancestors = spec._ancestors(owners)
    return [name for name in posterior.parameter_names if spec._owner[name].name in ancestors]


def _layout_of(model: FactoredDistribution, names: Sequence[str]) -> Layout:
    """The layout of the components *names*, in that order, at the model's
    labels."""
    specs = [model.spec.component_spec(name) for name in names]
    return Layout(specs, coords=coords_of(specs, model.coords))


def _slice_of(unconstrained: Layout, bound: BoundFactor) -> np.ndarray:
    """A target factor's entries of theta, its components' in event order."""
    return np.concatenate([np.arange(unconstrained.slice_of(n).start, unconstrained.slice_of(n).stop) for n in bound.names])


def _simulators_restricted(model: FactoredDistribution, names: Sequence[str]) -> dict[str, Simulator]:
    """The model's simulators *names* are computed from, each at the
    model's labels computing only the outputs behind *names*, in
    declaration order."""
    behind = simulator_outputs_behind(model.spec, names)
    out = {}
    for name, simulator in model.simulators.items():
        kept = [o.name for o in simulator.outputs if o.name in behind]
        if kept:
            out[name] = simulator_at(simulator, model.coords, kept)
    return out


def _simulators_read(model: FactoredDistribution, likelihood: Sequence[BoundFactor]) -> dict[str, Simulator]:
    """The simulators the :math:`O_\\theta` factors depend on, each
    restricted to the outputs they depend on."""
    return _simulators_restricted(model, [g for bound in likelihood for g in bound.spec.given])


def _simulator_free_positions(posterior: Posterior) -> np.ndarray:
    """The entries of theta whose factor has none of the posterior's
    simulators among its descendants: a simulator only barren factors read
    is not run, so it holds nothing fixed."""
    spec = posterior.model.spec
    simulator_names = set(posterior.simulators)
    free = [
        positions for bound, positions in zip(posterior._target, posterior._slices)
        if not spec._descendants(bound.name) & simulator_names
    ]
    return np.sort(np.concatenate(free)).astype(np.int64) if free else np.zeros((0,), dtype=np.int64)


def _all_computed(runs: Mapping[str, SimulatorOutput], n: int) -> np.ndarray:
    """``(n,)``: every output of every run was computed."""
    masks = [valid for run in runs.values() for valid in run.valid.values()]
    return np.logical_and.reduce(masks) if masks else np.ones(n, dtype=bool)


def _corner_theta(posterior: Posterior) -> np.ndarray:
    """The corner points of the target, ``(n, D)``
    (:func:`~sipnet_calibration.probability._probes.corner_points`)."""
    unconstrained = posterior.parameters.unconstrained
    places = []
    for component in unconstrained.components:
        block = unconstrained.slice_of(component.name)
        size = int(np.prod(component.shape, dtype=int))
        places.append(np.arange(block.start, block.stop).reshape((-1, size)))
    return corner_points(places, unconstrained.size)


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


def check_no_factor_reads_a_simulator(model: FactoredDistribution, factor_names: Sequence[str], *, role: str) -> None:
    """No factor of *role* has a simulator among its ancestors: the prior
    and the evidence's constant are evaluated without a simulator run."""
    spec = model.spec
    for name in factor_names:
        simulators = sorted(spec._ancestors([name]) & set(model.simulators))
        if simulators:
            raise ValueError(
                f"the {role} {name!r} has the simulator(s) {simulators} among its ancestors, but the prior and "
                "the evidence's constant are evaluated without running a simulator; observe it, or have it "
                "read no simulator output."
            )


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


def check_simulators_take_the_corner_points(posterior: Posterior) -> None:
    """Each simulator the likelihood depends on accepts what the prior can
    produce: its ``check_given`` at the corner points of the target. One
    downstream of another simulator is not checked, which would run that
    one at every corner point."""
    if not posterior.simulators:
        return
    spec = posterior.model.spec
    theta = jnp.asarray(_corner_theta(posterior))
    for name, simulator in posterior.simulators.items():
        if simulator_outputs_behind(spec, simulator.given):
            continue
        given_specs = {g: spec.component_spec(g) for g in simulator.given}
        simulator.check_given(given_specs, posterior._simulator_inputs_at(theta, name))


def check_something_observed_depends_on_the_parameters(posterior: Posterior) -> None:
    """A likelihood has a factor: something observed depends on the
    parameters."""
    if posterior.observations is None:
        raise ValueError(
            "nothing observed depends on the parameters, so there is no likelihood to write as a Gaussian; "
            "observe a factor the parameters reach."
        )


def check_likelihood_factor_is_gaussian_with_a_held_covariance(posterior: Posterior, bound: BoundFactor) -> None:
    """An :math:`O_\\theta` factor of a Gaussian likelihood has a
    ``GaussianSpec`` law whose covariance the held values fix."""
    if not bound.gaussian:
        raise ValueError(
            f"the observed factor {bound.name!r} depends on the parameters, but its law is "
            f"{bound.spec.law_name!r}, not a GaussianSpec; a Gaussian likelihood needs every such factor Gaussian."
        )
    if bound.held_covariance is None:
        parameters = _parameters_read_by_the_covariance(posterior, bound)
        raise ValueError(
            f"the covariance of {bound.name!r} depends on the parameter(s) {truncated(parameters)}; a Gaussian "
            "likelihood's covariance is fixed, so observe them at values to hold them."
        )


def check_theta_is_a_batch(shape: tuple[int, ...], dimension: int) -> None:
    """Theta is a batch, ``(J, D)``."""
    if len(shape) != 2 or shape[1] != dimension:
        raise ValueError(f"theta has shape {shape}; give a batch (J, {dimension}).")


def check_theta_is_concrete(theta: Any) -> None:
    """Theta evaluated with a simulator is a value, not a JAX trace: a
    simulator runs outside JAX."""
    if isinstance(theta, jax.core.Tracer):
        raise ValueError(
            "theta is traced (jit, grad or vmap), but this evaluation runs a simulator, which runs outside "
            "JAX; call it outside the trace, or trace log_density_given(evaluation) instead."
        )


def check_theta_has_a_row(shape: tuple[int, ...]) -> None:
    """A batch of theta has at least one row, since evaluating none runs nothing."""
    if shape[0] == 0:
        raise ValueError(f"theta has shape {shape}; give at least one row.")


def check_theta_is_finite(theta: np.ndarray) -> None:
    """Every entry of theta is finite, since a simulator cannot run at a NaN."""
    rows = np.flatnonzero(~np.isfinite(theta).all(axis=1)).tolist()
    if rows:
        raise ValueError(f"theta holds a non-finite value in row(s) {truncated(rows)}; give finite values.")


def check_evaluation_is_an_evaluation(evaluation: Any) -> None:
    """What is held is a posterior's evaluation."""
    if not isinstance(evaluation, PosteriorEvaluation):
        raise TypeError(f"give a PosteriorEvaluation, from Posterior.evaluate, got {type(evaluation).__name__}.")


def check_simulator_is_the_models(name: Any, model: FactoredDistribution) -> None:
    """A simulator named is one of the model's."""
    if not isinstance(name, str) or name not in model.simulators:
        raise KeyError(f"the model has no simulator {name!r}; name one of {list(model.simulators)}.")
