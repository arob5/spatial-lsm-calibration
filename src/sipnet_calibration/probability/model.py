"""Models: parts declared together, then bound to labels as one
distribution.

Where this sits
---------------
::

    probability.parts                   (FactorSpec, DeterministicSpec, Simulator)
      -> probability.model.joint        (ModelSpec: the declared model)
      -> ModelSpec.bind                 (FactoredDistribution: the model at labels)
      -> probability.posterior          (condition_on: the model given values)

The model
---------
A model is a directed acyclic graph whose nodes are its parts and its
inputs. A **factor** (random node) :math:`v` carries a kernel
:math:`p_v(z_v \\mid z_{\\mathrm{pa}(v)})`; a **deterministic** carries a map
:math:`z_v = f_v(z_{\\mathrm{pa}(v)})`, traced, or computed outside JAX by a
**simulator**, which may fail at some samples; an **input** has no parents and its
value is supplied. Each part's parents are the components and inputs its
functions read (the keyword rule, :mod:`~sipnet_calibration.probability.parts`),
so the graph is derived, never stored. Given inputs :math:`u`, the random
values have the joint density

.. math::

    p(z_R \\mid u) = \\prod_{v\\ \\text{factor}} p_v\\big(z_v \\mid z_{\\mathrm{pa}(v)}\\big),

every deterministic value substituted, against the product of each
support's reference measure (:mod:`~sipnet_calibration.probability.laws`).
A deterministic adds nothing to the density.

A **Gaussian factor**, one whose law is a
:class:`~sipnet_calibration.probability.parts.GaussianSpec`, has its
covariance spec bound to the labels in use with the factor: which entries
form blocks is fixed then, and at each draw only the blocks' values are
computed (:mod:`~sipnet_calibration.probability.covariance`). A Student-t
factor (:mod:`~sipnet_calibration.probability.scale_mixtures`) is bound
alike.

What it reads
-------------
Parts and input declarations (:func:`joint`); then, to bind, the labels of
every dim (``coords``, as :mod:`~sipnet_calibration.probability.labels`
defines them) and each input's value.

Data model
----------
A bound model's values are values by name, ``{name: (*batch, *block)}``,
each component's block of shape ``(*index shape, *element shape)`` at the
labels in use (:mod:`~sipnet_calibration.probability.layout`). Its inputs
are kept as labeled values, a DataArray per input on ``(*indexed_by,
*element axes)``.

Functions and classes
---------------------
:func:`joint`
    Parts and inputs as one :class:`ModelSpec`.
:class:`ModelSpec`
    The declared model: ``component_spec``, ``bind``, ``describe``.
:class:`FactoredDistribution`
    The model at labels: ``block_shape``, ``law``, ``select``,
    ``marginalize``, ``sample``, ``log_prob``, ``describe``.
:func:`block_at_labels`
    A labeled value or an array as a component's block at the labels in use.

Notes
-----
Binding runs the probe-point checks on every factor, whatever its later
role: its law at the probe points of its own unconstrained coordinates,
at two ancestral draws of what it reads when that varies by draw. Each
deterministic's outputs are checked at the probe points of the random
components it is computed from, pushed through. A Gaussian factor, on
``REAL``, is checked instead for its event shape and, when its covariance
reads nothing that varies by draw, for a positive-definite covariance; one
that does is checked per sample, which it makes invalid where it fails. A
Student-t factor is checked as a Gaussian one is.

Binding runs no simulator: it binds each one to the labels in use
(:meth:`Simulator.at <sipnet_calibration.probability.parts.Simulator.at>`)
and stands a placeholder in for each of its outputs, its bijector's image
of 0. A factor downstream of a simulator is checked there for its form
alone (event shape, TFP batch shape, dtype, a density the model can
evaluate), and a deterministic downstream for its block shapes when a
factor reads a value that varies by draw, the case in which draws are made. Sampling
and the joint density run each simulator once for the whole batch; what is
drawn or computed from an output that failed is ``NaN``, and a density that
reads one is ``-inf``.

Usage
-----
::

    model_spec = joint(
        FactorSpec(ArraySpec("q10", units="1", support=POSITIVE),
                   law=log_normal_from_interval(lower=1.3, upper=3.2), provenance="..."),
        FactorSpec(ArraySpec("soil_carbon", units="kg m-2", support=POSITIVE, indexed_by=("site",)),
                   law=independent_over_dim(log_normal, geometric_sd=2.0),
                   constants={"median": median_by_site}, provenance="..."),
    )
    model = model_spec.bind(coords={"site": [620, 865]})
    draws = model.sample(jax.random.key(0), 100)     # {"q10": (100,), "soil_carbon": (100, 2)}
    model.log_prob(draws)                            # (100,)
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.probability._bound import (
    ANCESTRAL_DRAWS,
    ANCESTRAL_SEED,
    BoundDeterministic,
    BoundFactor,
    check_deterministic_has_its_block_shapes,
    check_deterministic_lies_in_its_supports,
    check_draws_map_to_finite_theta,
    coords_of,
    deterministics_behind,
    finite_or_minus_infinity,
    random_key_for,
    simulator_at,
    simulator_outputs_behind,
    split_reads,
)
from sipnet_calibration.probability._probes import joint_probe_points
from sipnet_calibration.probability._validation import (
    as_count,
    as_names,
    check_names_are_unique,
    truncated,
)
from sipnet_calibration.probability.covariance import _Scope
from sipnet_calibration.probability.labels import (
    aligned_constants,
    aligned_label_maps,
    as_coords,
    is_stacked,
)
from sipnet_calibration.probability.layout import (
    LabeledValues,
    Layout,
    ValuesByName,
    validate_values_by_name,
)
from sipnet_calibration.probability.names import SAMPLE
from sipnet_calibration.probability.parts import (
    CENTERED_LAW_SPECS,
    DeterministicSpec,
    FactorSpec,
    GaussianSpec,
    Simulator,
    SimulatorOutput,
    check_simulator_is_valid,
)
from sipnet_calibration.probability.spec import ArraySpec

__all__ = [
    "FactoredDistribution",
    "ModelSpec",
    "block_at_labels",
    "joint",
]

Array = jax.Array

#: A part of a model.
type Part = FactorSpec | DeterministicSpec | Simulator


def joint(*parts: FactorSpec | DeterministicSpec | Simulator | ModelSpec, inputs: Sequence[ArraySpec] = ()) -> ModelSpec:
    """The model whose parts are *parts*, flattened, conditional on *inputs*.

    Parts may come in any order; declaration order fixes theta's order.

    Parameters
    ----------
    *parts : FactorSpec, DeterministicSpec, Simulator or ModelSpec
        A model's parts and inputs are taken as its own.
    inputs : Sequence[ArraySpec]
        Keyword-only. Input nodes: values the model is conditional on, bound
        by :meth:`ModelSpec.bind`.

    Raises
    ------
    TypeError
        If a part is none of the four, or an input is not an ArraySpec.
    ValueError
        As :class:`ModelSpec`.
    """
    flat_parts: list[Part] = []
    flat_inputs: list[ArraySpec] = []
    for part in parts:
        check_part_is_a_part(part)
        if isinstance(part, ModelSpec):
            flat_parts.extend(part.parts)
            flat_inputs.extend(part.inputs)
        else:
            flat_parts.append(part)
    check_inputs_are_array_specs(inputs)
    return ModelSpec(flat_parts, inputs=[*flat_inputs, *inputs])


class ModelSpec:
    """A declaration of a model: its parts and inputs, and the density they
    define,

    .. math::

        p(z \\mid u) = \\prod_{b\\ \\text{factor}} p_b(z_{B_b} \\mid z_{g(b)}, u)
            \\prod_{d\\ \\text{deterministic}} \\delta\\big(z_{B_d} - f_d(z_{g(d)}, u)\\big).

    Made by :func:`joint`.

    Parameters
    ----------
    parts : Sequence[FactorSpec | DeterministicSpec]
        Positional-only. In declaration order.
    inputs : Sequence[ArraySpec]
        Keyword-only. Default none.

    Attributes
    ----------
    parts : tuple
        In declaration order.
    inputs : tuple of ArraySpec
        In declaration order.
    component_names, input_names : tuple of str
        In declaration order.

    Raises
    ------
    ValueError
        If there is no part; a component or input is declared twice; a part
        reads a name nothing declares, or holds a constant or label map named
        like a component or input; an element axis name has two sets of
        labels; a Gaussian factor's mean differs from its event in layout or
        units; or the links form a cycle.
    """

    __slots__ = ("parts", "inputs", "_owner", "_order")

    parts: tuple[Part, ...]
    inputs: tuple[ArraySpec, ...]

    def __init__(self, parts: Sequence[Part], /, *, inputs: Sequence[ArraySpec] = ()) -> None:
        object.__setattr__(self, "parts", tuple(parts))
        object.__setattr__(self, "inputs", tuple(inputs))
        check_model_spec_is_valid(self)
        object.__setattr__(self, "_owner", frozendict({c.name: p for p in self.parts for c in _declared(p)}))
        object.__setattr__(self, "_order", _parts_in_topological_order(self))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a ModelSpec is frozen; build another rather than setting {name!r}.")

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def component_names(self) -> tuple[str, ...]:
        """Every component a part declares, in declaration order."""
        return tuple(c.name for p in self.parts for c in _declared(p))

    @property
    def input_names(self) -> tuple[str, ...]:
        """The inputs, in declaration order."""
        return tuple(i.name for i in self.inputs)

    def component_spec(self, name: str) -> ArraySpec:
        """The spec of a component or input; ``KeyError`` for any other name."""
        check_name_is_declared(name, self)
        if name in self.input_names:
            return self.inputs[self.input_names.index(name)]
        return next(c for c in _declared(self._owner[name]) if c.name == name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.component_names)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (name in self._owner or name in self.input_names)

    def __repr__(self) -> str:
        inputs = f", inputs={list(self.input_names)}" if self.inputs else ""
        return f"ModelSpec(parts={[p.name for p in self.parts]}{inputs})"

    def describe(self) -> pd.DataFrame:
        """One row per part, indexed by ``part`` (its name): ``class``,
        ``law`` (family or function name), ``given`` (comma-separated),
        ``indexed_by`` and ``provenance`` (empty for a deterministic)."""
        rows = [
            {
                "part": part.name,
                "class": type(part).__name__,
                "law": part.law_name,
                "given": ", ".join(part.given),
                "indexed_by": ", ".join(dict.fromkeys(d for c in _declared(part) for d in c.indexed_by)),
                "provenance": (part.provenance or "") if isinstance(part, FactorSpec) else "",
            }
            for part in self.parts
        ]
        return pd.DataFrame(rows).set_index("part")

    # ── binding ───────────────────────────────────────────────────────────────

    def bind(self, *, coords: Mapping[str, Sequence[Any]], inputs: Mapping[str, Any] | None = None) -> FactoredDistribution:
        """The model at the labels *coords* and the input values *inputs*:
        one distribution over its random nodes.

        Parameters
        ----------
        coords:
            Keyword-only. The labels of every dim a component or input is
            indexed by; other dims are ignored, so a site table's coords may
            be passed whole.
        inputs:
            Keyword-only. One value per declared input, a labeled
            ``xr.DataArray`` read by label or an array of its block shape.

        Raises
        ------
        KeyError
            If a dim is missing, *inputs* names no declared input, or a
            constant, label map or input lacks a label in use.
        ValueError
            If an input is missing, not finite or outside its support; a
            check at the probe points fails, naming the part; or a Gaussian
            factor's covariance cannot be built over its labels
            (:mod:`~sipnet_calibration.probability.covariance`) or is not
            positive definite.
        """
        return FactoredDistribution(self, coords=coords, inputs=inputs)

    # ── the graph ─────────────────────────────────────────────────────────────

    def _part_of(self, name: str) -> Part:
        """The part declaring the component *name*."""
        return self._owner[name]

    def _parents(self, part: Part) -> tuple[str, ...]:
        """The parts (by name) and inputs a part reads."""
        return tuple(dict.fromkeys(name if name in self.input_names else self._owner[name].name for name in part.given))

    def _ancestors(self, part_names: Sequence[str]) -> set[str]:
        """The parts *part_names* descend from, through any part, and
        themselves."""
        parts = {p.name: p for p in self.parts}
        seen, pending = set(), list(part_names)
        while pending:
            name = pending.pop()
            if name in seen or name not in parts:
                continue
            seen.add(name)
            pending.extend(self._parents(parts[name]))
        return seen

    def _descendants(self, part_name: str) -> set[str]:
        """The parts that descend from *part_name*, through any part, not
        itself."""
        children: dict[str, list[str]] = {p.name: [] for p in self.parts}
        for part in self.parts:
            for parent in self._parents(part):
                if parent in children:
                    children[parent].append(part.name)
        seen, pending = set(), list(children[part_name])
        while pending:
            name = pending.pop()
            if name not in seen:
                seen.add(name)
                pending.extend(children[name])
        return seen


class FactoredDistribution:
    """A model at a set of labels: the joint distribution of its components,
    conditional on its inputs. Made by :meth:`ModelSpec.bind`.

    Parameters
    ----------
    spec : ModelSpec
        Positional-only.
    coords, inputs:
        Keyword-only. As :meth:`ModelSpec.bind`.

    Attributes
    ----------
    spec : ModelSpec
    coords : frozendict of str to pandas.Index
        The labels of every dim a component or input is indexed by.
    inputs : frozendict of str to xr.DataArray
        Each input's value at the labels in use.
    simulators : frozendict of str to Simulator
        Each simulator at the labels in use (:meth:`Simulator.at
        <sipnet_calibration.probability.parts.Simulator.at>`), computing
        every output, by name.

    Raises
    ------
    TypeError, KeyError, ValueError
        As :meth:`ModelSpec.bind`.
    """

    def __init__(self, spec: ModelSpec, /, *, coords: Mapping[str, Sequence[Any]], inputs: Mapping[str, Any] | None = None) -> None:
        check_spec_is_a_model_spec(spec)
        used = _coords_in_use(spec, as_coords(coords))
        layout = Layout([*(c for p in spec.parts for c in _declared(p)), *spec.inputs], coords=used)
        input_values = _input_values(spec, {} if inputs is None else inputs, used)
        _set(self, "spec", spec)
        _set(self, "coords", layout.coords)
        _set(self, "_layout", layout)
        _set(self, "inputs", frozendict(
            Layout(list(spec.inputs), coords=coords_of(spec.inputs, used)).values_to_labeled(input_values)
            if spec.inputs else {}
        ))
        _set(self, "_element_axes", frozendict({a: labels for c in layout.components for a, labels in c.element_axes.items()}))
        factors, deterministics, fixed, simulators = _bound_parts(self, input_values)
        _set(self, "_factors", frozendict(factors))
        _set(self, "_deterministics", frozendict(deterministics))
        _set(self, "_fixed", frozendict(fixed))
        _set(self, "_simulators", frozendict(simulators))
        _set(self, "_given_layouts", frozendict({
            name: _layout_of(self, simulator.given) for name, simulator in simulators.items()
        }))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a FactoredDistribution is frozen; bind again rather than setting {name!r}.")

    def __repr__(self) -> str:
        coords = {dim: len(labels) for dim, labels in self.coords.items()}
        return f"FactoredDistribution(parts={[p.name for p in self.spec.parts]}, coords={coords})"

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def simulators(self) -> frozendict:
        """Each simulator at the labels in use, by name."""
        return self._simulators

    def block_shape(self, name: str) -> tuple[int, ...]:
        """A component's or input's block shape at the labels in use."""
        check_name_is_declared(name, self.spec)
        return self._layout.block_shape(name)

    def describe(self) -> pd.DataFrame:
        """``spec.describe()``, with ``evaluated_by`` (``"base density"`` or
        ``"change of variables"`` for a factor, ``"-"`` for a deterministic)
        and ``block_shapes``, each declared component's."""
        table = self.spec.describe()
        table["evaluated_by"] = [
            self._factors[p.name].evaluated_by if isinstance(p, FactorSpec) else "-" for p in self.spec.parts
        ]
        table["block_shapes"] = [
            ", ".join(f"{c.name}: {self._layout.block_shape(c.name)}" for c in _declared(p)) for p in self.spec.parts
        ]
        return table

    def law(self, component_name: str, /, *, given: Mapping[str, Any]) -> Any:
        """The law of the factor declaring *component_name*, for one draw.

        Parameters
        ----------
        component_name:
            Positional-only.
        given:
            Keyword-only. One value of its block shape for each component and
            input the factor reads.

        Returns
        -------
        Law
            A TFP distribution, or the law object the factor's law returns.

        Raises
        ------
        TypeError
            If *given* is not a mapping.
        KeyError
            If no component is called *component_name*, or *given* lacks a
            name the factor reads.
        ValueError
            If a deterministic declares *component_name*, or a value given
            is not of its block shape.
        """
        check_name_is_a_component(component_name, self.spec)
        part = self.spec._part_of(component_name)
        check_part_is_a_factor(component_name, part)
        check_given_holds_what_the_factor_reads(part, given)
        values = {name: jnp.asarray(given[name], dtype=jnp.float64) for name in part.given}
        for name, value in values.items():
            check_value_has_its_block_shape(name, value.shape, self._layout.block_shape(name))
        return self._factors[part.name].law_at(values)

    # ── selection ─────────────────────────────────────────────────────────────

    def select(self, **selectors: Sequence[Any]) -> FactoredDistribution:
        """``spec.bind`` at fewer labels: each selector, a dim or a level of a
        stacked dim, keeps those labels everywhere; the inputs are read again
        at them.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`Layout.select <sipnet_calibration.probability.layout.Layout.select>`
            and :meth:`ModelSpec.bind`; ``KeyError`` for ``component=``,
            since the spec decides the components.
        """
        check_selectors_name_dims_or_levels(selectors, self.coords)
        selected = self._layout.select(**selectors)
        return FactoredDistribution(self.spec, coords=selected.coords, inputs=dict(self.inputs))

    def marginalize(self, component_names: Sequence[str]) -> FactoredDistribution:
        """The model with *component_names* integrated out by a conjugate
        rule (:mod:`~sipnet_calibration.probability.conjugacy`): each one's
        factor, and the Gaussian factor that reads it, become one factor, a
        Student-t or a matrix Student-t, bound at the same labels and
        inputs. The names are
        integrated out in order, each from the model the last one left.

        Raises
        ------
        TypeError
            If *component_names* is not a sequence of names.
        KeyError
            If a name is not a component.
        ValueError
            If *component_names* is empty or names a component twice, or no
            rule applies to a name; the message names the condition that
            failed.
        """
        from sipnet_calibration.probability.conjugacy import marginalize

        return marginalize(self, component_names)

    # ── evaluation ────────────────────────────────────────────────────────────

    def sample(self, key: Array, n: int, *, component_names: Sequence[str] | None = None) -> ValuesByName:
        """``n`` ancestral draws, ``{name: (n, *block)}``.

        Parts are evaluated in topological order. A factor that reads
        nothing that varies by draw draws ``n`` values at once with
        ``jax.random.fold_in(key, crc32(name))``; one that does splits that
        key ``n`` ways and draws one value per draw. A simulator runs once
        for the ``n`` draws; where it fails, its outputs and everything
        computed or drawn from them are ``NaN``, as are a Gaussian factor's
        draws where its covariance is not positive definite. With *component_names*,
        only those and their ancestors are drawn, and only those returned,
        so a simulator runs only when one of them needs it; otherwise every
        component is.

        Raises
        ------
        TypeError
            If *n* is not an integer, or *component_names* is not a sequence
            of names.
        KeyError
            If a name is no component.
        ValueError
            If *n* is negative, or a draw lands on a support's boundary,
            where theta is not finite; the message names the factor.
        TypeError, ValueError
            If a simulator returns other than a whole ``SimulatorOutput``.
        Exception
            Whatever a simulator's machinery raises.
        """
        n = as_count(n, message_name="n")
        if component_names is None:
            wanted = self.spec.component_names
        else:
            wanted = as_names(component_names, message_name="component_names")
            for name in wanted:
                check_name_is_a_component(name, self.spec)
        needed = self.spec._ancestors([self.spec._part_of(name).name for name in wanted])
        values, fixed = self._ancestral(key, n, self._fixed, needed=needed)
        return {name: values[name] if name in values else jnp.broadcast_to(fixed[name], (n, *self._layout.block_shape(name)))
                for name in wanted}

    def log_prob(self, values: Mapping[str, Any]) -> Array:
        """The joint log density of every factor's event, ``{name: (*batch,
        *block)} -> (*batch,)``, against each support's reference measure,
        ``-inf`` where a factor's density is not finite (a value outside or
        on the boundary of its support). Deterministic components are
        computed, running a simulator if one is upstream of a factor, once
        for the whole batch; a factor reading a simulator output that failed
        is ``-inf``. Traceable when no simulator runs.

        Raises
        ------
        TypeError
            If *values* is not a mapping.
        KeyError
            If *values* lacks a factor's event, or names no component.
        ValueError
            If *values* holds a deterministic component or an input, or a
            value does not end in its block shape, or the batch shapes
            differ; or a simulator would run under a JAX trace.
        TypeError, ValueError
            If a simulator returns other than a whole ``SimulatorOutput``.
        Exception
            Whatever a simulator's machinery raises.
        """
        check_values_hold_each_event(values, self)
        events = [c for p in self.spec.parts if isinstance(p, FactorSpec) for c in p.event]
        validate_values_by_name(values, Layout(events, coords=coords_of(events, self.coords)))
        first = jnp.shape(values[events[0].name])
        lead = tuple(first[: len(first) - len(self._layout.block_shape(events[0].name))])
        random = {c.name: jnp.asarray(values[c.name], dtype=jnp.float64) for c in events}
        read = [g for bound in self._factors.values() for g in bound.spec.given]
        runs: dict[str, SimulatorOutput] = {}
        per_draw, fixed = self._computed(random, self._fixed, lead, read, runs=runs)
        total = jnp.zeros(lead, dtype=jnp.float64)
        for bound in self._factors.values():
            own = {name: per_draw[name] for name in bound.names}
            reads, held = split_reads(bound.spec.given, per_draw, fixed)
            density = finite_or_minus_infinity(bound.log_prob_natural(own, reads, held, lead))
            computed = self._draws_computed(bound.spec.given, runs)
            total = total + (density if computed is None else jnp.where(jnp.asarray(computed).reshape(lead), density, -jnp.inf))
        return total

    # ── supporting methods ────────────────────────────────────────────────────

    def _ancestral(
        self,
        key: Array,
        n: int,
        fixed: Mapping[str, Array],
        *,
        needed: set[str],
        per_draw: Mapping[str, Array] | None = None,
        theta_out: dict[str, Array] | None = None,
        simulators: Mapping[str, Simulator] | None = None,
        runs: dict[str, SimulatorOutput] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Ancestral draws of the parts *needed*: ``(per-draw values (n,
        *block), fixed values)``, *fixed* held and *per_draw* taken as drawn.
        A part whose components are all at hand is not drawn. With
        *theta_out*, each factor's theta is recorded there, by factor name.
        A simulator is *simulators*' (the model's own by default), its
        checked output recorded in *runs*; what is drawn or computed from a
        failed output is ``NaN``."""
        values: dict[str, Array] = dict(per_draw or {})
        fixed = dict(fixed)
        simulators = self._simulators if simulators is None else simulators
        runs = {} if runs is None else runs
        for part in self.spec._order:
            declared = simulators[part.name].outputs if isinstance(part, Simulator) and part.name in needed else _declared(part)
            if part.name not in needed or all(c.name in fixed or c.name in values for c in declared):
                continue
            if isinstance(part, Simulator):
                values |= self._simulate(simulators[part.name], values, fixed, (n,), runs)
                continue
            reads, held = split_reads(part.given, values, fixed)
            computed = self._draws_computed(part.given, runs)
            if isinstance(part, DeterministicSpec):
                out = self._deterministics[part.name].compute(reads, held, (n,))
                (values if reads else fixed).update(_masked(out, computed, (n,)))
                continue
            bound = self._factors[part.name]
            theta = bound.sample_theta(random_key_for(key, part.name), n, reads, held)
            # A centered draw, on REAL, is not finite only where its covariance
            # is not positive definite: no density there, so it stays NaN.
            if not bound.centered:
                check_draws_map_to_finite_theta(part.name, theta if computed is None else theta[computed])
            if computed is not None:
                theta = jnp.where(jnp.asarray(computed)[:, None], theta, jnp.nan)
            if theta_out is not None:
                theta_out[part.name] = theta
            values |= bound.natural_values(theta)
        return values, fixed

    def _computed(
        self,
        per_draw: Mapping[str, Array],
        fixed: Mapping[str, Array],
        lead: tuple[int, ...],
        names: Sequence[str],
        *,
        simulators: Mapping[str, Simulator] | None = None,
        runs: dict[str, SimulatorOutput] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """*per_draw* and *fixed* with every deterministic and simulator
        behind *names* computed where what it reads is at hand: ``(per-draw
        values, fixed values)``. A deterministic reading only fixed values is
        computed once, unbatched; a simulator runs once for the batch, as in
        :meth:`_ancestral`."""
        needed = self._deterministics_behind(names)
        per_draw, fixed = dict(per_draw), dict(fixed)
        simulators = self._simulators if simulators is None else simulators
        runs = {} if runs is None else runs
        for part in self.spec._order:
            if part.name not in needed:
                continue
            declared = simulators[part.name].outputs if isinstance(part, Simulator) else part.outputs
            if all(c.name in fixed or c.name in per_draw for c in declared):
                continue
            if not all(g in per_draw or g in fixed for g in part.given):
                continue
            if isinstance(part, Simulator):
                per_draw |= self._simulate(simulators[part.name], per_draw, fixed, lead, runs)
                continue
            reads, held = split_reads(part.given, per_draw, fixed)
            out = self._deterministics[part.name].compute(reads, held, lead)
            (per_draw if reads else fixed).update(_masked(out, self._draws_computed(part.given, runs), lead))
        return per_draw, fixed

    def _deterministics_behind(self, names: Sequence[str]) -> set[str]:
        """The deterministics and simulators (by name) computing *names*,
        directly or through others."""
        return deterministics_behind(self.spec, names)

    def _simulate(
        self,
        simulator: Simulator,
        per_draw: Mapping[str, Array],
        fixed: Mapping[str, Array],
        lead: tuple[int, ...],
        runs: dict[str, SimulatorOutput],
    ) -> dict[str, Array]:
        """One call of *simulator* for the batch *lead*: its outputs,
        ``{name: (*lead, *block)}``, ``NaN`` where not computed, as recorded
        in *runs* by its name. An empty batch calls nothing."""
        n = math.prod(lead)
        shapes = {o.name: self._layout.block_shape(o.name) for o in simulator.outputs}
        if n == 0:
            runs[simulator.name] = SimulatorOutput(
                values=frozendict({name: np.zeros((0, *shape)) for name, shape in shapes.items()}),
                valid=frozendict({name: np.zeros(0, dtype=bool) for name in shapes}),
            )
            return {name: jnp.zeros((*lead, *shape)) for name, shape in shapes.items()}
        output = simulator(self._labeled_given(simulator, per_draw, fixed, lead))
        check_simulator_output_is_whole(simulator.name, output, shapes, n)
        returned = {name: np.asarray(output.values[name], dtype=np.float64) for name in shapes}
        valid = {
            name: np.asarray(output.valid[name], dtype=bool) & np.isfinite(returned[name]).reshape((n, -1)).all(axis=1)
            for name in shapes
        }
        computed = {
            name: np.where(valid[name].reshape((n,) + (1,) * len(shape)), returned[name], np.nan)
            for name, shape in shapes.items()
        }
        runs[simulator.name] = SimulatorOutput(values=frozendict(computed), valid=frozendict(valid), record=output.record)
        return {name: jnp.asarray(value.reshape((*lead, *shapes[name]))) for name, value in computed.items()}

    def _labeled_given(
        self, simulator: Simulator, per_draw: Mapping[str, Array], fixed: Mapping[str, Array], lead: tuple[int, ...]
    ) -> LabeledValues:
        """What *simulator* reads for the batch *lead*, as it receives it:
        labeled values on ``(sample, *indexed_by, *element axes)``, the
        batch flattened to ``sample``, fixed values repeated for each."""
        check_simulator_runs_outside_a_trace(simulator.name, [per_draw[g] for g in simulator.given if g in per_draw])
        n = math.prod(lead)
        given = {}
        for name in simulator.given:
            shape = self._layout.block_shape(name)
            value = per_draw[name] if name in per_draw else jnp.broadcast_to(fixed[name], (*lead, *shape))
            given[name] = np.asarray(value, dtype=np.float64).reshape((n, *shape))
        return self._given_layouts[simulator.name].values_to_labeled(given, batch_dims=(SAMPLE,))

    def _draws_computed(self, names: Sequence[str], runs: Mapping[str, SimulatorOutput]) -> np.ndarray | None:
        """``(n,)``: the draws at which every simulator output behind *names*
        was computed; ``None`` when none is behind them."""
        behind = simulator_outputs_behind(self.spec, names)
        masks = [run.valid[name] for run in runs.values() for name in run.valid if name in behind]
        return None if not masks else np.logical_and.reduce(masks)


def block_at_labels(spec: ArraySpec, value: Any, coords: Mapping[str, pd.Index], *, message_name: str) -> Array:
    """*value* as *spec*'s block at the labels *coords*, ``float64``.

    A DataArray is read by label: its dims are the spec's ``indexed_by`` and
    element axes, in any order, each holding at least the labels in use. Any
    other value is an array of the block shape. *message_name* names the
    mapping the value came from (``"inputs"``), so a message names it as
    ``inputs['offset']``.

    Raises
    ------
    TypeError
        If the value is not numeric: a boolean, a string or an object.
    KeyError
        If a DataArray lacks a label in use.
    ValueError
        If a DataArray is on other dims, an array has another shape, or a
        value is not finite or lies outside the spec's support.
    """
    expected = (*(len(coords[d]) for d in spec.indexed_by), *spec.shape)
    subject = f"{message_name}[{spec.name!r}]"
    if isinstance(value, xr.DataArray):
        check_value_is_numeric(value.dtype, subject)
        dims = (*spec.indexed_by, *spec.element_axes)
        check_labeled_value_is_on_its_dims(spec.name, tuple(map(str, value.dims)), dims, message_name=subject)
        labels = {**{d: coords[d] for d in spec.indexed_by}, **spec.element_axes}
        (block,) = aligned_constants(
            {spec.name: value.astype(np.float64)}, labels, dim_order=dims, message_name=message_name
        ).values()
    else:
        check_value_is_numeric(np.asarray(value).dtype, subject)
        block = jnp.asarray(value, dtype=jnp.float64)
        check_value_has_its_block_shape(spec.name, tuple(block.shape), expected)
    check_value_is_finite_and_in_its_support(spec, block, message_name=subject)
    return block


# ── private: binding ──────────────────────────────────────────────────────────


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _declared(part: Part) -> tuple[ArraySpec, ...]:
    """The components a part declares."""
    return part.event if isinstance(part, FactorSpec) else part.outputs


def _coords_in_use(spec: ModelSpec, coords: Mapping[str, pd.Index]) -> dict[str, pd.Index]:
    """The coords of the dims a component or input is indexed by, in the
    coords' order."""
    specs = [*(c for p in spec.parts for c in _declared(p)), *spec.inputs]
    used = {d for c in specs for d in c.indexed_by}
    check_coords_hold_every_dim(used, coords)
    return {d: labels for d, labels in coords.items() if d in used}


def _input_values(spec: ModelSpec, inputs: Any, coords: Mapping[str, pd.Index]) -> dict[str, Array]:
    """Each input's block at the labels in use."""
    check_inputs_are_a_mapping(inputs)
    check_inputs_name_the_declared(inputs, spec)
    return {
        i.name: block_at_labels(i, inputs[i.name], coords, message_name="inputs") for i in spec.inputs
    }


def _bound_parts(
    model: FactoredDistribution, input_values: Mapping[str, Array]
) -> tuple[dict[str, BoundFactor], dict[str, BoundDeterministic], dict[str, Array], dict[str, Simulator]]:
    """Every part bound and checked in topological order, held in
    declaration order; the values fixed for every draw: the inputs and
    what is computed from them alone; and each simulator at the labels.

    When some factor reads a value that varies by draw, two ancestral draws
    are made as the parts are bound, and each such factor is built and
    checked at both. No simulator runs: its outputs are placeholders there,
    each its bijector's image of 0, at which the factors downstream are
    checked for their form only, and the deterministics downstream for
    their block shapes, when the draws are made."""
    spec = model.spec
    varying = _names_varying_by_draw(spec)
    downstream = _parts_downstream_of_a_simulator(spec)
    ancestral = any(n in varying for p in spec.parts if isinstance(p, FactorSpec) for n in p.given)
    key = jax.random.key(ANCESTRAL_SEED)
    draws: dict[str, Array] = {}
    fixed: dict[str, Array] = dict(input_values)
    factors: dict[str, BoundFactor] = {}
    deterministics: dict[str, BoundDeterministic] = {}
    simulators: dict[str, Simulator] = {}
    for part in spec._order:
        if isinstance(part, Simulator):
            simulators[part.name] = simulator_at(part, model.coords)
            if ancestral:
                draws.update(_placeholders(model, part.outputs, ANCESTRAL_DRAWS))
            continue
        fixed_reads = _fixed_reads(model, part)
        if isinstance(part, DeterministicSpec):
            bound = BoundDeterministic(
                spec=part, fixed_reads=frozendict(fixed_reads),
                block_shapes=frozendict({c.name: model._layout.block_shape(c.name) for c in part.outputs}),
            )
            deterministics[part.name] = bound
            if not any(g in varying for g in part.given):
                out = bound.compute({}, {g: fixed[g] for g in part.given}, ())
                check_deterministic_has_its_block_shapes(bound, out, 0)
                check_deterministic_lies_in_its_supports(bound, {n: v[None] for n, v in out.items()}, np.zeros((1, 0)))
                fixed.update(out)
            elif ancestral:
                reads, held = split_reads(part.given, draws, fixed)
                out = bound.compute(reads, held, (ANCESTRAL_DRAWS,))
                if part.name in downstream:
                    check_deterministic_has_its_block_shapes(bound, out, 1)
                draws.update(out)
            continue
        reads, held = split_reads(part.given, draws, fixed)
        per_draw_values = [{n: v[i] for n, v in reads.items()} for i in range(ANCESTRAL_DRAWS)] if reads else [{}]
        centered = isinstance(part.law, CENTERED_LAW_SPECS)
        bound = BoundFactor.build(
            part, index_shape=model._layout.index_shape(part.event[0].name), fixed_reads=frozendict(fixed_reads),
            per_draw_values=per_draw_values, fixed_values=held, downstream_of_a_simulator=part.name in downstream,
            covariance=_law_structure_at(part.law, _covariance_scope(model, part, fixed_reads)) if centered else None,
            covariance_is_checked=not centered or not any(n in varying for n in _structure_reads(part.law)),
        )
        factors[part.name] = bound
        if ancestral:
            theta = bound.sample_theta(random_key_for(key, part.name), ANCESTRAL_DRAWS, reads, held)
            if part.name not in downstream and not bound.centered:
                check_draws_map_to_finite_theta(part.name, theta)
            draws |= bound.natural_values(theta)
    for bound in deterministics.values():
        if any(g in varying for g in bound.spec.given) and bound.name not in downstream:
            _evaluate_deterministic_at_the_probes(model, bound, deterministics, fixed)
    order = [p.name for p in spec.parts]
    return (
        {name: factors[name] for name in order if name in factors},
        {name: deterministics[name] for name in order if name in deterministics},
        fixed,
        {name: simulators[name] for name in order if name in simulators},
    )


def _covariance_scope(model: FactoredDistribution, part: FactorSpec, fixed_reads: Mapping[str, Any]) -> _Scope:
    """What a Gaussian factor's covariance spec is bound over: the event's
    entries, and the dims of everything the factor reads."""
    (component,) = part.event
    event_dim = component.indexed_by[0] if component.indexed_by else None
    spec = model.spec
    return _Scope(
        factor_name=part.name,
        size=math.prod(model._layout.index_shape(component.name)),
        event_dim=event_dim,
        labels=None if event_dim is None else model.coords[event_dim],
        read_dims={name: _dims_read(model, part, name) for name in part.reads},
        fixed=dict(fixed_reads),
        specs={name: spec.component_spec(name) for name in part.given},
        label_map_targets={name: str(label_map.name) for name, label_map in part.label_maps.items()},
        coords={**_own_dim_labels(part), **model.coords},
    )


def _law_structure_at(law: Any, scope: _Scope) -> Any:
    """A centered law's structure at its factor's labels: a Gaussian's
    covariance spec, or a Student-t spec itself, bound over *scope*."""
    return law.covariance._at(scope) if isinstance(law, GaussianSpec) else law._at(scope)


def _structure_reads(law: Any) -> tuple[str, ...]:
    """What a centered law's covariance reads, beyond its mean."""
    covariance = getattr(law, "covariance", None)
    return () if covariance is None else covariance.reads


def _own_dim_labels(part: FactorSpec) -> dict[str, pd.Index]:
    """The labels of each of a factor's own dims, which its constants on it
    share: a group reads them at its label, as it reads a dim of the
    coords."""
    labels: dict[str, pd.Index] = {}
    for name, constant in part.constants.items():
        for dim in map(str, constant.dims):
            if dim in part.own_dims and dim in constant.indexes:
                check_own_dim_is_labeled_alike(part, name, dim, labels.get(dim), constant.indexes[dim])
                labels.setdefault(dim, constant.indexes[dim])
    return labels


def _dims_read(model: FactoredDistribution, part: FactorSpec, name: str) -> tuple[str, ...]:
    """The dims of what a factor reads as *name*, in the order its array
    arrives: a constant's as :func:`_fixed_reads` orders them, a label
    map's one dim, a component's or input's ``indexed_by`` and element
    axes."""
    if name in part.constants:
        dims = tuple(map(str, part.constants[name].dims))
        order = [d for c in part.event for d in (*c.indexed_by, *c.element_axes)]
        first = [d for d in dict.fromkeys(order) if d in dims]
        return (*first, *(d for d in dims if d not in first))
    if name in part.label_maps:
        return (str(part.label_maps[name].dims[0]),)
    spec = model.spec.component_spec(name)
    return (*spec.indexed_by, *spec.element_axes)


def _names_varying_by_draw(spec: ModelSpec) -> set[str]:
    """The components that vary by draw: every factor's and simulator's,
    and every deterministic output computed from one."""
    varying: set[str] = set()
    for part in spec._order:
        if isinstance(part, (FactorSpec, Simulator)) or any(g in varying for g in part.given):
            varying.update(c.name for c in _declared(part))
    return varying


def _parts_downstream_of_a_simulator(spec: ModelSpec) -> set[str]:
    """The parts (by name) with a simulator among their ancestors."""
    downstream: set[str] = set()
    for part in spec._order:
        parents = [spec._owner[g] for g in part.given if g in spec._owner]
        if any(isinstance(p, Simulator) or p.name in downstream for p in parents):
            downstream.add(part.name)
    return downstream


def _placeholders(model: FactoredDistribution, specs: Sequence[ArraySpec], n: int) -> dict[str, Array]:
    """``n`` copies of a value inside each component's support, its
    bijector's image of 0, ``{name: (n, *block)}``."""
    return {
        c.name: c.bijector.forward(jnp.zeros((n, *model._layout.index_shape(c.name), *c.unconstrained_shape)))
        for c in specs
    }


def _layout_of(model: FactoredDistribution, names: Sequence[str]) -> Layout:
    """The layout of the components and inputs *names*, in that order, at
    the model's labels."""
    specs = [model.spec.component_spec(name) for name in names]
    return Layout(specs, coords=coords_of(specs, model.coords))


def _masked(values: Mapping[str, Array], computed: np.ndarray | None, lead: tuple[int, ...]) -> dict[str, Array]:
    """*values*, each ``(*lead, *block)``, ``NaN`` at the draws not
    *computed*, which is flat over *lead*."""
    if computed is None:
        return dict(values)
    mask = jnp.asarray(computed).reshape(lead)
    return {
        name: jnp.where(mask.reshape(lead + (1,) * (v.ndim - len(lead))), v, jnp.nan) for name, v in values.items()
    }


def _fixed_reads(model: FactoredDistribution, part: Part) -> dict[str, Any]:
    """A part's constants and label maps read at the labels in use, each
    constant with the part's ``indexed_by`` dims first, then its element
    axes."""
    declared = _declared(part)
    dim_order = tuple(dict.fromkeys((
        *(d for c in declared for d in c.indexed_by),
        *(axis for c in declared for axis in c.element_axes),
    )))
    labels_by_dim = {**model._element_axes, **model.coords}
    constants = aligned_constants(
        part.constants, labels_by_dim, dim_order=dim_order, own_dims=part.own_dims,
        message_name=f"the {_kind(part)} {part.name!r} constants",
    )
    label_maps = aligned_label_maps(
        part.label_maps, model.coords, element_axes=model._element_axes,
        message_name=f"the {_kind(part)} {part.name!r} label maps",
    )
    return {**constants, **label_maps}


def _evaluate_deterministic_at_the_probes(
    model: FactoredDistribution,
    bound: BoundDeterministic,
    deterministics: Mapping[str, BoundDeterministic],
    fixed: Mapping[str, Array],
) -> None:
    """A deterministic's outputs have their block shapes and lie in their
    supports at the joint probe points of the random components it is
    computed from."""
    spec = model.spec
    random = _random_behind(spec, bound.spec.given)
    shapes = [(*model._layout.index_shape(c.name), *c.unconstrained_shape) for c in random]
    parts = joint_probe_points([(shape, int(np.prod(c.unconstrained_shape))) for shape, c in zip(shapes, random)])
    n_probes = len(parts[0])
    per_draw = {c.name: c.bijector.forward(jnp.asarray(points)) for c, points in zip(random, parts)}
    held = dict(fixed)
    theta = np.concatenate([p.reshape((n_probes, -1)) for p in parts], axis=-1)
    behind = deterministics_behind(spec, bound.spec.given) | {bound.name}
    for part in spec._order:
        if part.name not in behind or all(c.name in held for c in part.outputs):
            continue
        reads, given_held = split_reads(part.given, per_draw, held)
        out = deterministics[part.name].compute(reads, given_held, (n_probes,))
        if part.name == bound.name:
            check_deterministic_has_its_block_shapes(bound, out, 1)
            check_deterministic_lies_in_its_supports(bound, out, theta)
        (per_draw if reads else held).update(out)


def _parts_in_topological_order(spec: ModelSpec) -> tuple[Part, ...]:
    """The parts, each after every part it reads, otherwise in declaration
    order; a cycle is refused by name."""
    parts = {p.name: p for p in spec.parts}
    order: list[Part] = []
    state: dict[str, str] = {}
    for start in parts:
        if start in state:
            continue
        path, stack = [start], [iter(spec._parents(parts[start]))]
        state[start] = "open"
        while stack:
            child = next(stack[-1], None)
            if child is None:
                node = path.pop()
                stack.pop()
                state[node] = "done"
                order.append(parts[node])
                continue
            if child not in parts:
                continue
            check_links_are_acyclic(path, child, state)
            if child not in state:
                state[child] = "open"
                path.append(child)
                stack.append(iter(spec._parents(parts[child])))
    return tuple(order)


def _random_behind(spec: ModelSpec, names: Sequence[str]) -> list[ArraySpec]:
    """The factors' components *names* are, or are computed from through
    deterministics, in declaration order."""
    found: set[str] = set()
    pending, seen = list(names), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        part = spec._owner.get(name)
        if isinstance(part, DeterministicSpec):
            pending.extend(part.given)
        elif isinstance(part, FactorSpec):
            found.add(name)
    return [c for p in spec.parts if isinstance(p, FactorSpec) for c in p.event if c.name in found]


def _kind(part: Part) -> str:
    if isinstance(part, Simulator):
        return "simulator"
    return "factor" if isinstance(part, FactorSpec) else "deterministic"


# ── checks ────────────────────────────────────────────────────────────────────


def check_model_spec_is_valid(spec: ModelSpec) -> None:
    """A model declares each component and input once, every name its parts
    read, and one set of labels per element axis name."""
    check_model_has_a_part(spec.parts)
    for part in spec.parts:
        check_part_is_a_factor_deterministic_or_simulator(part)
        if isinstance(part, Simulator):
            check_simulator_is_valid(part)
    check_model_has_a_factor(spec.parts)
    check_inputs_are_array_specs(spec.inputs)
    names = [c.name for p in spec.parts for c in _declared(p)] + [i.name for i in spec.inputs]
    check_names_are_unique(names, message_name="the model's components and inputs")
    for part in spec.parts:
        check_part_reads_declared_names(part, set(names))
        check_constants_are_named_apart(part, set(names))
    check_element_axes_agree([*(c for p in spec.parts for c in _declared(p)), *spec.inputs])
    for part in spec.parts:
        if isinstance(part, FactorSpec) and isinstance(part.law, CENTERED_LAW_SPECS):
            check_gaussian_mean_matches_its_event(part, spec)


def check_part_is_a_part(part: Any) -> None:
    """A model's part is a factor, a deterministic, a simulator or a model."""
    if not isinstance(part, (FactorSpec, DeterministicSpec, Simulator, ModelSpec)):
        raise TypeError(
            f"a model's part is a {type(part).__name__}; give FactorSpecs, DeterministicSpecs, Simulators or "
            "ModelSpecs."
        )


def check_part_is_a_factor_deterministic_or_simulator(part: Any) -> None:
    """A model spec's part is a factor, a deterministic or a simulator;
    models are flattened into one by :func:`joint`."""
    if not isinstance(part, (FactorSpec, DeterministicSpec, Simulator)):
        raise TypeError(
            f"a ModelSpec's part is a {type(part).__name__}; give FactorSpecs, DeterministicSpecs and "
            "Simulators, and combine models with joint()."
        )


def check_inputs_are_array_specs(inputs: Any) -> None:
    """Inputs are declared by ArraySpecs, so each has a shape, labels and units."""
    if isinstance(inputs, (str, ArraySpec, Mapping)) or not isinstance(inputs, Sequence):
        raise TypeError(f"inputs is a {type(inputs).__name__}; give a sequence of ArraySpecs.")
    wrong = [type(i).__name__ for i in inputs if not isinstance(i, ArraySpec)]
    if wrong:
        raise TypeError(f"inputs holds {truncated(wrong)}; declare each input with an ArraySpec.")


def check_model_has_a_part(parts: Sequence[Any]) -> None:
    """A model has at least one part."""
    if not parts:
        raise ValueError("a model has no part; give at least one FactorSpec.")


def check_model_has_a_factor(parts: Sequence[Part]) -> None:
    """A model has a factor: deterministics alone define no distribution."""
    if not any(isinstance(p, FactorSpec) for p in parts):
        raise ValueError("a model of deterministics alone defines no distribution; give at least one FactorSpec.")


def check_part_reads_declared_names(part: Part, declared: set[str]) -> None:
    """Every component or input a part reads is declared, by a part or as an
    input."""
    missing = [n for n in part.given if n not in declared]
    if missing:
        raise ValueError(
            f"the {_kind(part)} {part.name!r} reads {truncated(missing)}, which no part declares and no input "
            "is; declare them, give them as constants, or check the spelling of its keywords."
        )


def check_constants_are_named_apart(part: Part, declared: set[str]) -> None:
    """No constant or label map is named like a component or input, which
    the function's keyword would then name twice."""
    clashing = [n for n in (*getattr(part, "constants", ()), *getattr(part, "label_maps", ())) if n in declared]
    if clashing:
        raise ValueError(
            f"the {_kind(part)} {part.name!r} holds constants or label maps {truncated(clashing)} named like "
            "components or inputs; name them apart."
        )


def check_element_axes_agree(specs: Sequence[ArraySpec]) -> None:
    """An element axis name means one set of labels across a model."""
    seen: dict[str, tuple[str, pd.Index]] = {}
    for spec in specs:
        for axis, labels in spec.element_axes.items():
            if axis in seen and not seen[axis][1].equals(labels):
                raise ValueError(
                    f"{seen[axis][0]!r} and {spec.name!r} both have an element axis {axis!r}, with different "
                    "labels; an element axis name means one set of labels across a model, so name them apart."
                )
            seen.setdefault(axis, (spec.name, labels))


def check_gaussian_mean_matches_its_event(part: FactorSpec, spec: ModelSpec) -> None:
    """A Gaussian or Student-t factor's mean has its event's layout and
    units: indexed by the same dims, with the same element axes, in the same
    units."""
    (event,) = part.event
    (name,) = part.law.mean
    declared = {c.name: c for p in spec.parts for c in _declared(p)} | {i.name: i for i in spec.inputs}
    if name not in declared:
        raise ValueError(
            f"the {part.law_name} factor {part.name!r} is centered on {name!r}, which is not a component or input; "
            "center it on a component, such as a prediction."
        )
    mean = declared[name]
    if (mean.indexed_by, dict(mean.element_axes), mean.units) != (event.indexed_by, dict(event.element_axes), event.units):
        raise ValueError(
            f"the {part.law_name} factor {part.name!r} is centered on {name!r}, indexed by {mean.indexed_by} in "
            f"{mean.units!r}, but its event is indexed by {event.indexed_by} in {event.units!r}; a mean has its "
            "event's layout and units, so convert it before it is one."
        )


def check_own_dim_is_labeled_alike(part: FactorSpec, name: str, dim: str, held: pd.Index | None, labels: pd.Index) -> None:
    """A factor's constants on one of its own dims share its labels, in one
    order, by which a group of its covariance reads them."""
    if held is not None and not held.equals(labels):
        raise ValueError(
            f"the factor {part.name!r} holds constants on its own dim {dim!r} with different labels ({name!r} among "
            "them); give every constant on it the same labels, in the same order."
        )


def check_simulator_runs_outside_a_trace(name: str, values: Sequence[Any]) -> None:
    """A simulator runs on concrete values: it is code outside JAX, which a
    traced value cannot reach."""
    if any(isinstance(v, jax.core.Tracer) for v in values):
        raise ValueError(
            f"the simulator {name!r} would run under a JAX trace (jit, grad or vmap), which it cannot; "
            "evaluate it outside the trace, or hold its outputs (Posterior.log_density_given)."
        )


def check_simulator_output_is_whole(
    name: str, output: Any, shapes: Mapping[str, tuple[int, ...]], n: int
) -> None:
    """A simulator returns a ``SimulatorOutput`` holding each of its outputs
    at every sample, ``(J, *block)``, and its validity, ``(J,)``."""
    if not isinstance(output, SimulatorOutput):
        raise TypeError(f"the simulator {name!r} returned a {type(output).__name__}, not a SimulatorOutput.")
    for what, held in (("values", output.values), ("valid", output.valid)):
        if not isinstance(held, Mapping) or set(held) != set(shapes):
            found = sorted(held) if isinstance(held, Mapping) else type(held).__name__
            raise ValueError(
                f"the simulator {name!r} returned {what} for {found}, but computes {sorted(shapes)}; give one "
                "for each output."
            )
    for output_name, shape in shapes.items():
        if tuple(np.shape(output.values[output_name])) != (n, *shape):
            raise ValueError(
                f"the simulator {name!r} returned {output_name!r} of shape {tuple(np.shape(output.values[output_name]))}, "
                f"but {n} samples of its block are {(n, *shape)}."
            )
        if tuple(np.shape(output.valid[output_name])) != (n,) or np.asarray(output.valid[output_name]).dtype != bool:
            raise ValueError(
                f"the simulator {name!r} returned the validity of {output_name!r} as "
                f"{np.asarray(output.valid[output_name]).dtype} of shape {tuple(np.shape(output.valid[output_name]))}; "
                f"give a bool array of shape ({n},)."
            )


def check_links_are_acyclic(path: Sequence[str], child: str, state: Mapping[str, str]) -> None:
    """The parts' links form no cycle, which no order of draws could
    satisfy."""
    if state.get(child) == "open":
        cycle = [*path[path.index(child):], child]
        raise ValueError(f"the model's parts read one another in a cycle, {' -> '.join(cycle)}; break it.")


def check_spec_is_a_model_spec(spec: ModelSpec) -> None:
    """A bound model is made from a ModelSpec."""
    if not isinstance(spec, ModelSpec):
        raise TypeError(f"a FactoredDistribution is made from a ModelSpec, got {type(spec).__name__}; use joint().")


def check_coords_hold_every_dim(used: set[str], coords: Mapping[str, pd.Index]) -> None:
    """The coords give labels to every dim a component or input is indexed by."""
    missing = sorted(d for d in used if d not in coords)
    if missing:
        raise KeyError(f"the coords have no labels for {missing}, which components or inputs are indexed by; give them.")


def check_inputs_are_a_mapping(inputs: Any) -> None:
    """Input values are ``{name: value}``."""
    if not isinstance(inputs, Mapping):
        raise TypeError(f"inputs must be a mapping {{name: value}}, got {type(inputs).__name__}.")


def check_inputs_name_the_declared(inputs: Mapping[str, Any], spec: ModelSpec) -> None:
    """Values are given for exactly the declared inputs."""
    unknown = [n for n in inputs if n not in spec.input_names]
    if unknown:
        raise KeyError(f"inputs names {truncated(unknown)}, which are not inputs of the model {list(spec.input_names)}.")
    missing = [n for n in spec.input_names if n not in inputs]
    if missing:
        raise ValueError(f"the model's inputs {truncated(missing)} have no value; give one for each.")


def check_name_is_declared(name: Any, spec: ModelSpec) -> None:
    """A name is a component or input of the model."""
    if not isinstance(name, str) or name not in spec:
        raise KeyError(f"the model has no component or input {name!r}; name one of {truncated([*spec.component_names, *spec.input_names])}.")


def check_name_is_a_component(name: Any, spec: ModelSpec) -> None:
    """A name is a component of the model."""
    if not isinstance(name, str) or name not in spec.component_names:
        raise KeyError(f"the model has no component {name!r}; name one of {truncated(list(spec.component_names))}.")


def check_part_is_a_factor(name: str, part: Part) -> None:
    """A law is a factor's; a deterministic computes its components."""
    if not isinstance(part, FactorSpec):
        raise ValueError(f"{name!r} is computed by the deterministic {part.name!r}, which has no law.")


def check_given_holds_what_the_factor_reads(part: FactorSpec, given: Any) -> None:
    """Every component and input a factor reads has a value."""
    if not isinstance(given, Mapping):
        raise TypeError(f"given must be a mapping {{name: value}}, got {type(given).__name__}.")
    missing = [n for n in part.given if n not in given]
    if missing:
        raise KeyError(f"the law of {part.name!r} reads {truncated(missing)}; give a value of each in given=.")


def check_value_has_its_block_shape(name: str, shape: tuple[int, ...], expected: tuple[int, ...]) -> None:
    """A value is one block, of its block shape."""
    if tuple(shape) != tuple(expected):
        raise ValueError(f"{name!r} has shape {tuple(shape)}, but its block shape at the labels in use is {tuple(expected)}.")


def check_labeled_value_is_on_its_dims(name: str, dims: tuple[str, ...], expected: tuple[str, ...], *, message_name: str) -> None:
    """A labeled value is on its component's dims and element axes."""
    if set(dims) != set(expected) or len(dims) != len(expected):
        raise ValueError(f"{message_name} is on {dims}, but {name!r} is on {expected}; give it on those dims.")


def check_value_is_numeric(dtype: np.dtype, subject: str) -> None:
    """A value given for a component or input is a number: a boolean or a
    string is not one, though NumPy would convert it."""
    if np.dtype(dtype).kind not in "fiu":
        raise TypeError(f"{subject} is of dtype {dtype}, not a number; give floats.")


def check_value_is_finite_and_in_its_support(spec: ArraySpec, block: Array, *, message_name: str) -> None:
    """A value given for a component or input is finite and lies in its support."""
    if not bool(jnp.all(jnp.isfinite(block))):
        raise ValueError(f"{message_name} is not finite everywhere; give finite values.")
    if not bool(jnp.all(spec.support.contains(block))):
        raise ValueError(f"{message_name} lies outside its support {spec.support.name!r} somewhere; give values inside it.")


def check_selectors_name_dims_or_levels(selectors: Mapping[str, Any], coords: Mapping[str, pd.Index]) -> None:
    """A selection keeps labels of dims or levels in use; the spec decides
    the components."""
    levels = {name for labels in coords.values() if is_stacked(labels) for name in labels.names}
    unknown = [key for key in selectors if key not in coords and key not in levels]
    if unknown:
        raise KeyError(
            f"select takes dims and levels in use, {truncated(sorted({*coords, *levels}))}, not "
            f"{truncated(unknown)}; the model spec decides the components."
        )


def check_values_hold_each_event(values: Any, model: FactoredDistribution) -> None:
    """Values for a density are every factor's event and nothing computed or
    bound."""
    if not isinstance(values, Mapping):
        raise TypeError(f"values must be a mapping {{name: value}}, got {type(values).__name__}.")
    spec = model.spec
    computed = [n for n in values if n in model._layout and isinstance(spec._owner.get(n), DeterministicSpec)]
    bound = [n for n in values if n in spec.input_names]
    if computed or bound:
        raise ValueError(
            f"values holds {truncated([*computed, *bound])}, which are computed or bound inputs; a density is "
            "evaluated at the factors' events, and the rest follows from them."
        )
    unknown = [n for n in values if n not in spec]
    if unknown:
        raise KeyError(f"values names {truncated(unknown)}, which are no components of the model.")
    missing = [c.name for p in spec.parts if isinstance(p, FactorSpec) for c in p.event if c.name not in values]
    if missing:
        raise KeyError(f"values lacks {truncated(missing)}; give every factor's event.")
