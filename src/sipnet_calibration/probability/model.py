"""Models: parts declared together, then bound to labels as one
distribution.

Where this sits
---------------
::

    probability.parts                   (FactorSpec, DeterministicSpec)
      -> probability.model.joint        (ModelSpec: the declared model)
      -> ModelSpec.bind                 (FactoredDistribution: the model at labels)
      -> probability.posterior          (condition_on: the model given values)

The model
---------
A model is a directed acyclic graph whose nodes are its parts and its
inputs. A **factor** (random node) :math:`v` carries a kernel
:math:`p_v(z_v \\mid z_{\\mathrm{pa}(v)})`; a **deterministic** carries a map
:math:`z_v = f_v(z_{\\mathrm{pa}(v)})`; an **input** has no parents and its
value is supplied. Each part's parents are the components and inputs its
functions read (the keyword rule, :mod:`~sipnet_calibration.probability.parts`),
so the graph is derived, never stored. Given inputs :math:`u`, the random
values have the joint density

.. math::

    p(z_R \\mid u) = \\prod_{v\\ \\text{factor}} p_v\\big(z_v \\mid z_{\\mathrm{pa}(v)}\\big),

every deterministic value substituted, against the product of each
support's reference measure (:mod:`~sipnet_calibration.probability.laws`).
A deterministic adds nothing to the density.

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
    The model at labels: ``law``, ``select``, ``sample``, ``log_prob``,
    ``describe``.
:func:`block_at_labels`
    A labeled value or an array as a component's block at the labels in use.

Notes
-----
Binding runs the probe-point checks on every factor, whatever its later
role: its law at the probe points of its own unconstrained coordinates,
at two ancestral draws of what it reads when that varies by draw. Each
deterministic's outputs are checked at the probe points of the random
components it is computed from, pushed through.

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
    random_key_for,
    split_reads,
)
from sipnet_calibration.probability._probes import joint_probe_points
from sipnet_calibration.probability._validation import (
    as_count,
    as_names,
    check_names_are_unique,
    truncated,
)
from sipnet_calibration.probability.labels import (
    aligned_constants,
    aligned_label_maps,
    as_coords,
)
from sipnet_calibration.probability.layout import (
    Layout,
    ValuesByName,
    validate_values_by_name,
)
from sipnet_calibration.probability.names import COMPONENT_LEVEL
from sipnet_calibration.probability.parts import DeterministicSpec, FactorSpec
from sipnet_calibration.probability.spec import ArraySpec

__all__ = [
    "FactoredDistribution",
    "ModelSpec",
    "block_at_labels",
    "check_model_spec_is_valid",
    "joint",
]

Array = jax.Array

#: A part of a model.
type Part = FactorSpec | DeterministicSpec


def joint(*parts: FactorSpec | DeterministicSpec | ModelSpec, inputs: Sequence[ArraySpec] = ()) -> ModelSpec:
    """The model whose parts are *parts*, flattened, conditional on *inputs*.

    Parts may come in any order; declaration order fixes theta's order.

    Parameters
    ----------
    *parts : FactorSpec, DeterministicSpec or ModelSpec
        A model's parts and inputs are taken as its own.
    inputs : Sequence[ArraySpec]
        Keyword-only. Input nodes: values the model is conditional on, bound
        by :meth:`ModelSpec.bind`.

    Raises
    ------
    TypeError
        If a part is none of the three, or an input is not an ArraySpec.
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
        labels; or the links form a cycle.
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
            If an input is missing, not finite or outside its support, or a
            check at the probe points fails, naming the part.
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
        factors, deterministics, fixed = _bound_parts(self, input_values)
        _set(self, "_factors", frozendict(factors))
        _set(self, "_deterministics", frozendict(deterministics))
        _set(self, "_fixed", frozendict(fixed))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a FactoredDistribution is frozen; bind again rather than setting {name!r}.")

    def __repr__(self) -> str:
        coords = {dim: len(labels) for dim, labels in self.coords.items()}
        return f"FactoredDistribution(parts={[p.name for p in self.spec.parts]}, coords={coords})"

    # ── identity ──────────────────────────────────────────────────────────────

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
        check_selectors_name_no_component(selectors)
        selected = self._layout.select(**selectors)
        return FactoredDistribution(self.spec, coords=selected.coords, inputs=dict(self.inputs))

    # ── evaluation ────────────────────────────────────────────────────────────

    def sample(self, key: Array, n: int, *, component_names: Sequence[str] | None = None) -> ValuesByName:
        """``n`` ancestral draws, ``{name: (n, *block)}``.

        Parts are evaluated in topological order. A factor that reads
        nothing that varies by draw draws ``n`` values at once with
        ``jax.random.fold_in(key, crc32(name))``; one that does splits that
        key ``n`` ways and draws one value per draw. With *component_names*,
        only those and their ancestors are drawn, and only those returned;
        otherwise every component is.

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
        *block)} -> (*batch,)``, against each support's reference measure.
        Deterministic components are computed. Traceable.

        Raises
        ------
        TypeError
            If *values* is not a mapping.
        KeyError
            If *values* lacks a factor's event, or names no component.
        ValueError
            If *values* holds a deterministic component or an input, or a
            value does not end in its block shape, or the batch shapes
            differ.
        """
        check_values_hold_each_event(values, self)
        events = [c for p in self.spec.parts if isinstance(p, FactorSpec) for c in p.event]
        validate_values_by_name(values, Layout(events, coords=coords_of(events, self.coords)))
        first = jnp.shape(values[events[0].name])
        lead = tuple(first[: len(first) - len(self._layout.block_shape(events[0].name))])
        random = {c.name: jnp.asarray(values[c.name], dtype=jnp.float64) for c in events}
        read = [g for bound in self._factors.values() for g in bound.spec.given]
        per_draw, fixed = self._computed(random, self._fixed, lead, read)
        total = jnp.zeros(lead, dtype=jnp.float64)
        for bound in self._factors.values():
            own = {name: per_draw[name] for name in bound.names}
            reads, held = split_reads(bound.spec.given, per_draw, fixed)
            total = total + bound.log_prob_natural(own, reads, held, lead)
        return total

    # ── supporting methods ────────────────────────────────────────────────────

    def _ancestral(
        self, key: Array, n: int, fixed: Mapping[str, Array], *, needed: set[str], theta_out: dict[str, Array] | None = None
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Ancestral draws of the parts *needed*, *fixed* values held:
        ``(per-draw values (n, *block), fixed values)``. A factor whose
        event is all held is not drawn. With *theta_out*, each factor's
        theta is recorded there, by factor name."""
        values: dict[str, Array] = {}
        fixed = dict(fixed)
        for part in self.spec._order:
            if part.name not in needed:
                continue
            if isinstance(part, DeterministicSpec):
                if all(c.name in fixed for c in part.outputs):
                    continue
                reads, held = split_reads(part.given, values, fixed)
                out = self._deterministics[part.name].compute(reads, held, (n,))
                (values if reads else fixed).update(out)
                continue
            if all(c.name in fixed for c in part.event):
                continue
            bound = self._factors[part.name]
            reads, held = split_reads(part.given, values, fixed)
            theta = bound.sample_theta(random_key_for(key, part.name), n, reads, held)
            check_draws_map_to_finite_theta(part.name, theta)
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
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """*per_draw* and *fixed* with every deterministic behind *names*
        computed where what it reads is at hand: ``(per-draw values, fixed
        values)``. A deterministic reading only fixed values is computed
        once, unbatched."""
        needed = self._deterministics_behind(names)
        per_draw, fixed = dict(per_draw), dict(fixed)
        for part in self.spec._order:
            if part.name not in needed or all(c.name in fixed or c.name in per_draw for c in part.outputs):
                continue
            if not all(g in per_draw or g in fixed for g in part.given):
                continue
            reads, held = split_reads(part.given, per_draw, fixed)
            out = self._deterministics[part.name].compute(reads, held, lead)
            (per_draw if reads else fixed).update(out)
        return per_draw, fixed

    def _deterministics_behind(self, names: Sequence[str]) -> set[str]:
        """The deterministics (by name) computing *names*, directly or
        through other deterministics."""
        return deterministics_behind(self.spec, names)


def block_at_labels(spec: ArraySpec, value: Any, coords: Mapping[str, pd.Index], *, message_name: str) -> Array:
    """*value* as *spec*'s block at the labels *coords*, ``float64``.

    A DataArray is read by label: its dims are the spec's ``indexed_by`` and
    element axes, in any order, each holding at least the labels in use. Any
    other value is an array of the block shape.

    Raises
    ------
    KeyError
        If a DataArray lacks a label in use.
    ValueError
        If a DataArray is on other dims, an array has another shape, or a
        value is not finite or lies outside the spec's support.
    """
    expected = (*(len(coords[d]) for d in spec.indexed_by), *spec.shape)
    if isinstance(value, xr.DataArray):
        dims = (*spec.indexed_by, *spec.element_axes)
        check_labeled_value_is_on_its_dims(spec.name, tuple(map(str, value.dims)), dims, message_name=message_name)
        labels = {**{d: coords[d] for d in spec.indexed_by}, **spec.element_axes}
        (block,) = aligned_constants(
            {spec.name: value.astype(np.float64)}, labels, dim_order=dims, message_name=message_name
        ).values()
    else:
        block = jnp.asarray(value, dtype=jnp.float64)
        check_value_has_its_block_shape(spec.name, tuple(block.shape), expected)
    check_value_is_finite_and_in_its_support(spec, block, message_name=message_name)
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
        i.name: block_at_labels(i, inputs[i.name], coords, message_name=f"the input {i.name!r}") for i in spec.inputs
    }


def _bound_parts(
    model: FactoredDistribution, input_values: Mapping[str, Array]
) -> tuple[dict[str, BoundFactor], dict[str, BoundDeterministic], dict[str, Array]]:
    """Every part bound and checked in topological order, held in
    declaration order, and the values fixed for every draw: the inputs and
    what is computed from them alone.

    When some factor reads a value that varies by draw, two ancestral draws
    are made as the parts are bound, and each such factor is built and
    checked at both."""
    spec = model.spec
    varying = _names_varying_by_draw(spec)
    ancestral = any(n in varying for p in spec.parts if isinstance(p, FactorSpec) for n in p.given)
    key = jax.random.key(ANCESTRAL_SEED)
    draws: dict[str, Array] = {}
    fixed: dict[str, Array] = dict(input_values)
    factors: dict[str, BoundFactor] = {}
    deterministics: dict[str, BoundDeterministic] = {}
    for part in spec._order:
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
                draws.update(bound.compute(reads, held, (ANCESTRAL_DRAWS,)))
            continue
        reads, held = split_reads(part.given, draws, fixed)
        per_draw_values = [{n: v[i] for n, v in reads.items()} for i in range(ANCESTRAL_DRAWS)] if reads else [{}]
        bound = BoundFactor.build(
            part, index_shape=model._layout.index_shape(part.event[0].name), fixed_reads=frozendict(fixed_reads),
            per_draw_values=per_draw_values, fixed_values=held,
        )
        factors[part.name] = bound
        if ancestral:
            theta = bound.sample_theta(random_key_for(key, part.name), ANCESTRAL_DRAWS, reads, held)
            check_draws_map_to_finite_theta(part.name, theta)
            draws |= bound.natural_values(theta)
    for bound in deterministics.values():
        if any(g in varying for g in bound.spec.given):
            _check_deterministic_at_the_probes(model, bound, deterministics, fixed)
    order = [p.name for p in spec.parts]
    return (
        {name: factors[name] for name in order if name in factors},
        {name: deterministics[name] for name in order if name in deterministics},
        fixed,
    )


def _names_varying_by_draw(spec: ModelSpec) -> set[str]:
    """The components that vary by draw: every factor's, and every
    deterministic output computed from one."""
    varying: set[str] = set()
    for part in spec._order:
        if isinstance(part, FactorSpec) or any(g in varying for g in part.given):
            varying.update(c.name for c in _declared(part))
    return varying


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


def _check_deterministic_at_the_probes(
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
    return "factor" if isinstance(part, FactorSpec) else "deterministic"


# ── checks ────────────────────────────────────────────────────────────────────


def check_model_spec_is_valid(spec: ModelSpec) -> None:
    """A model declares each component and input once, every name its parts
    read, and one set of labels per element axis name."""
    check_model_has_a_part(spec.parts)
    for part in spec.parts:
        check_part_is_a_factor_or_a_deterministic(part)
    check_inputs_are_array_specs(spec.inputs)
    names = [c.name for p in spec.parts for c in _declared(p)] + [i.name for i in spec.inputs]
    check_names_are_unique(names, message_name="the model's components and inputs")
    for part in spec.parts:
        check_part_reads_declared_names(part, set(names))
        check_constants_are_named_apart(part, set(names))
    check_element_axes_agree([*(c for p in spec.parts for c in _declared(p)), *spec.inputs])


def check_part_is_a_part(part: Any) -> None:
    """A model's part is a factor, a deterministic or a model."""
    if not isinstance(part, (FactorSpec, DeterministicSpec, ModelSpec)):
        raise TypeError(
            f"a model's part is a {type(part).__name__}; give FactorSpecs, DeterministicSpecs or ModelSpecs."
        )


def check_part_is_a_factor_or_a_deterministic(part: Any) -> None:
    """A model spec's part is a factor or a deterministic; models are
    flattened into one by :func:`joint`."""
    if not isinstance(part, (FactorSpec, DeterministicSpec)):
        raise TypeError(
            f"a ModelSpec's part is a {type(part).__name__}; give FactorSpecs and DeterministicSpecs, and "
            "combine models with joint()."
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
    clashing = [n for n in (*part.constants, *part.label_maps) if n in declared]
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


def check_value_is_finite_and_in_its_support(spec: ArraySpec, block: Array, *, message_name: str) -> None:
    """A value given for a component or input is finite and lies in its support."""
    if not bool(jnp.all(jnp.isfinite(block))):
        raise ValueError(f"{message_name} is not finite everywhere; give finite values.")
    if not bool(jnp.all(spec.support.contains(block))):
        raise ValueError(f"{message_name} lies outside {spec.name!r}'s support {spec.support.name!r} somewhere.")


def check_selectors_name_no_component(selectors: Mapping[str, Any]) -> None:
    """A selection keeps labels; the spec decides the components."""
    if COMPONENT_LEVEL in selectors:
        raise KeyError("select takes dims and levels, not component=; the model spec decides the components.")


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
