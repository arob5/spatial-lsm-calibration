"""The parts of a model: factors, which declare conditional laws;
deterministics, which declare computed components; and simulators, which
compute components outside JAX.

Where this sits
---------------
::

    probability.spec.ArraySpec            (what a component is)
    probability.laws, families, builders  (a factor's law)
      -> probability.parts                (FactorSpec, DeterministicSpec)
      -> probability.model.joint          (the parts as one model)

A part is a declaration: it holds no labels and no numbers, and is
evaluated only inside a bound model
(:class:`~sipnet_calibration.probability.model.FactoredDistribution`).

What a part reads
-----------------
A part's functions read components, inputs, constants and label maps, each
by name. The names a function reads are its parameters without defaults
(the **keyword rule**); a function with ``*args``, ``**kwargs`` or a
positional-only parameter is refused, and a ``functools.partial``'s bound
arguments are defaults, so they are not read. Each name it reads is one of
the part's constants or label maps, or else a component or input, which
together are the part's ``given``; :func:`~sipnet_calibration.probability.model.joint`
refuses a given name nothing declares. Constants and label maps are as
:mod:`~sipnet_calibration.probability.labels` defines them, and a constant
arrives transposed: the part's ``indexed_by`` dims first, then its element
axes, then its other dims in its own order.

Classes and functions
---------------------
:class:`FactorSpec`, :func:`factor`
    A conditional law over one component or several indexed alike.
:class:`GaussianSpec`
    A factor's law as a Gaussian centered on a component, its covariance
    declared by a covariance spec (:mod:`~sipnet_calibration.probability.covariance`).
:class:`DeterministicSpec`, :func:`deterministic`
    Components computed by a pure JAX function.
:class:`Simulator`, :class:`SimulatorOutput`
    Components computed for a batch of samples by code outside JAX, such
    as a model run, which may fail at some samples; and what one call
    returns. A simulator is real code, not a declaration: it reads what it
    names in ``given``, not by the keyword rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability._keywords import function_reads
from sipnet_calibration.probability._validation import (
    as_names,
    check_names_are_unique,
    truncated,
)
from sipnet_calibration.probability.builders import Builder
from sipnet_calibration.probability.covariance import CovarianceSpec
from sipnet_calibration.probability.labels import as_constants, as_label_maps
from sipnet_calibration.probability.laws import Law, distribution_name, is_law
from sipnet_calibration.probability.layout import LabeledValues
from sipnet_calibration.probability.spec import ArraySpec
from sipnet_calibration.probability.support import REAL

__all__ = [
    "DeterministicSpec",
    "FactorSpec",
    "GaussianSpec",
    "Simulator",
    "SimulatorOutput",
    "deterministic",
    "factor",
]

tfd = tfp.distributions

Array = jax.Array


class FactorSpec:
    """A declaration of one factor: the conditional law of the components it
    declares, given the components its law reads,

    .. math::

        p_B(z_B \\mid z_g;\\ c).

    Evaluated at one draw of what it reads, it is a law over its event.

    Parameters
    ----------
    event : ArraySpec or Sequence[ArraySpec]
        Positional-only. One component, or several indexed by the same dims
        (a **joint factor**), in the order of their entries in theta.
    law : Law, callable, Builder or GaussianSpec
        Keyword-only. The law, or how to build it per draw:

        - a law (a TFP distribution, or an object implementing
          :class:`~sipnet_calibration.probability.laws.Law`), when the
          factor reads nothing and its one component is indexed by nothing;
        - a function ``(**reads) -> law`` for one draw: a law of TFP batch
          shape ``()`` over the block, or over a dict of blocks keyed by the
          event's names for a joint factor; traced, and vmapped over draws
          when it reads a component;
        - a builder's result (:func:`~sipnet_calibration.probability.builders.iid_over_dim`
          and the others), which the model also gives the index shape;
        - a :class:`GaussianSpec`, over one component on
          :data:`~sipnet_calibration.probability.support.REAL` indexed by one
          dim at most, with no element axes.
    constants, label_maps : Mapping[str, xr.DataArray], optional
        Keyword-only. Fixed data the law reads
        (:mod:`~sipnet_calibration.probability.labels`). Each must be read.
    own_dims : Sequence[str]
        Keyword-only. Dims of the constants that are neither dims of the
        coords nor element axes, passed whole. Default ``()``.
    provenance : str, optional
        Keyword-only. Where the law comes from, with its citation. This
        project requires it of every factor; the layer does not.

    Attributes
    ----------
    name : str
        ``"+".join`` of the event's names; the factor's randomness is keyed
        by it.
    given : tuple of str
        The components and inputs the law reads, inferred from its keywords.
    reads : tuple of str
        Every name the law reads: ``given``, constants and label maps.
    law_name : str
        A short name for the law, for a description.

    Raises
    ------
    TypeError
        If *event* is not one or more :class:`ArraySpec`; *law* is none of
        the four forms, or a function breaking the keyword rule; a bare
        law is given to a factor that is indexed, joint, or holds constants;
        a constant or label map is not a DataArray; or *provenance* is not a
        string.
    ValueError
        If the event is empty; a joint factor's components are indexed
        differently; a name repeats across the event, the constants and the
        label maps; the law reads a component of its own event; a constant
        or label map is never read; *provenance* is empty; or a Gaussian's
        event is not one component on ``REAL`` indexed by one dim at most
        with no element axes.

    Notes
    -----
    A law must marginalize consistently: bound at fewer labels, it must be
    the marginal of the law at more. The builders satisfy this, and so does a
    Gaussian process evaluated at the labels in use. A law that standardizes
    a covariate over the labels in use does not.
    """

    __slots__ = ("event", "law", "constants", "label_maps", "own_dims", "provenance", "reads")

    event: tuple[ArraySpec, ...]
    law: Any
    constants: frozendict
    label_maps: frozendict
    own_dims: tuple[str, ...]
    provenance: str | None
    reads: tuple[str, ...]

    def __init__(
        self,
        event: ArraySpec | Sequence[ArraySpec],
        /,
        *,
        law: Any,
        constants: Mapping[str, xr.DataArray] | None = None,
        label_maps: Mapping[str, xr.DataArray] | None = None,
        own_dims: Sequence[str] = (),
        provenance: str | None = None,
    ) -> None:
        event = _as_specs(event, what="a factor's event")
        name = "+".join(spec.name for spec in event)
        _set(self, "event", event)
        _set(self, "law", law)
        _set(self, "constants", as_constants({} if constants is None else constants, message_name=f"{name!r} constants"))
        _set(self, "label_maps", as_label_maps({} if label_maps is None else label_maps, message_name=f"{name!r} label_maps"))
        _set(self, "own_dims", as_names(own_dims, message_name=f"{name!r} own_dims"))
        _set(self, "provenance", provenance)
        _set(self, "reads", _law_reads(name, law))
        check_factor_spec_is_valid(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a FactorSpec is frozen; build another rather than setting {name!r}.")

    @property
    def name(self) -> str:
        """``"+".join`` of the event's names."""
        return "+".join(spec.name for spec in self.event)

    @property
    def given(self) -> tuple[str, ...]:
        """The components and inputs the law reads."""
        return tuple(n for n in self.reads if n not in self.constants and n not in self.label_maps)

    @property
    def law_name(self) -> str:
        """A short name for the law: ``"Gaussian"``, a builder's or family's
        name, a function's ``__name__``, or the law's class."""
        if isinstance(self.law, GaussianSpec):
            return "Gaussian"
        if isinstance(self.law, Builder):
            return self.law.name
        if is_law(self.law):
            return distribution_name(self.law)
        return getattr(self.law, "__name__", type(self.law).__name__)

    def __repr__(self) -> str:
        given = f", given={list(self.given)}" if self.given else ""
        return f"FactorSpec({self.name!r}, law={self.law_name!r}{given})"


def factor(
    event: ArraySpec | Sequence[ArraySpec],
    /,
    *,
    constants: Mapping[str, xr.DataArray] | None = None,
    label_maps: Mapping[str, xr.DataArray] | None = None,
    own_dims: Sequence[str] = (),
    provenance: str | None = None,
) -> Callable[[Callable[..., Law]], FactorSpec]:
    """Decorate a law function to make a :class:`FactorSpec` with it as
    ``law``.

    Usage
    -----
    ::

        @factor(ArraySpec("soil_carbon", units="kg m-2", support=POSITIVE, indexed_by=("site",)),
                provenance="...")
        def soil_carbon(soil_carbon_location, soil_carbon_spread):
            normal = tfd.Independent(tfd.Normal(soil_carbon_location, soil_carbon_spread), 1)
            return pushforward(normal, support=POSITIVE)
    """

    def decorate(law: Callable[..., Law]) -> FactorSpec:
        return FactorSpec(
            event, law=law, constants=constants, label_maps=label_maps, own_dims=own_dims, provenance=provenance
        )

    return decorate


class GaussianSpec:
    """A Gaussian law centered on a component, for a :class:`FactorSpec`'s
    ``law``:

    .. math::

        z \\mid m, \\phi \\sim \\mathcal N\\big(m,\\ \\Sigma(\\phi)\\big).

    Evaluated at one draw it is a
    :class:`~sipnet_calibration.probability.laws.GaussianLaw` over the
    event's block: mean :math:`m`, the component *mean* names, and
    covariance :math:`\\Sigma` built by *covariance*.

    Parameters
    ----------
    mean : str or Sequence[str]
        Keyword-only. The component or input the event is centered on, of
        the event's layout and units, which
        :func:`~sipnet_calibration.probability.model.joint` checks; one, for
        an event of one component.
    covariance : CovarianceSpec
        Keyword-only. :math:`\\Sigma` over the event's entries.

    Attributes
    ----------
    mean : tuple of str
    covariance : CovarianceSpec
    reads : tuple of str
        The mean, then what the covariance reads.

    Raises
    ------
    TypeError
        If *mean* is not one or more names, or *covariance* not a
        covariance spec.
    ValueError
        If *mean* names more than one component.

    Notes
    -----
    The model reads the declaration without evaluating it, to tell whether
    :math:`\\Sigma` depends on the parameters
    (:meth:`Posterior.gaussian_likelihood <sipnet_calibration.probability.posterior.Posterior.gaussian_likelihood>`).
    A covariance that does not is built and factored once, when the model is
    conditioned. A law function returning a
    :class:`~sipnet_calibration.probability.laws.GaussianLaw` evaluates to
    the same law but is opaque to both.
    """

    __slots__ = ("mean", "covariance")

    mean: tuple[str, ...]
    covariance: CovarianceSpec

    def __init__(self, *, mean: str | Sequence[str], covariance: CovarianceSpec) -> None:
        mean = (mean,) if isinstance(mean, str) else as_names(mean, message_name="GaussianSpec's mean")
        check_gaussian_has_one_mean(mean)
        check_covariance_is_a_covariance_spec(covariance)
        _set(self, "mean", mean)
        _set(self, "covariance", covariance)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a GaussianSpec is frozen; build another rather than setting {name!r}.")

    @property
    def reads(self) -> tuple[str, ...]:
        """The mean, then what the covariance reads."""
        return tuple(dict.fromkeys((*self.mean, *self.covariance.reads)))

    def __repr__(self) -> str:
        return f"GaussianSpec(mean={self.mean[0]!r}, covariance={self.covariance!r})"


class DeterministicSpec:
    """A declaration of components computed from others by a pure JAX
    function,

    .. math::

        z_B = f(z_g;\\ c).

    It adds nothing to the density and has no entries in theta. Calling it
    calls the function for one draw.

    Parameters
    ----------
    outputs : ArraySpec or Sequence[ArraySpec]
        Positional-only. An output's support other than ``REAL`` is checked
        at the probe points when the model is bound.
    function : callable
        Keyword-only. ``(**reads) -> block``, or a dict of blocks keyed by
        the outputs' names for several outputs, for one draw; traced, and
        vmapped over draws.
    constants, label_maps : Mapping[str, xr.DataArray], optional
        Keyword-only. As :class:`FactorSpec`'s.
    own_dims : Sequence[str]
        Keyword-only. As :class:`FactorSpec`'s.

    Attributes
    ----------
    name : str
        ``"+".join`` of the outputs' names.
    given : tuple of str
        The components and inputs it reads; at least one.
    reads : tuple of str
        Every name the function reads.

    Raises
    ------
    TypeError, ValueError
        As :class:`FactorSpec`'s, and ``ValueError`` if the function reads
        no component or input.
    """

    __slots__ = ("outputs", "function", "constants", "label_maps", "own_dims", "reads")

    outputs: tuple[ArraySpec, ...]
    function: Callable[..., Any]
    constants: frozendict
    label_maps: frozendict
    own_dims: tuple[str, ...]
    reads: tuple[str, ...]

    def __init__(
        self,
        outputs: ArraySpec | Sequence[ArraySpec],
        /,
        *,
        function: Callable[..., Any],
        constants: Mapping[str, xr.DataArray] | None = None,
        label_maps: Mapping[str, xr.DataArray] | None = None,
        own_dims: Sequence[str] = (),
    ) -> None:
        outputs = _as_specs(outputs, what="a deterministic's outputs")
        name = "+".join(spec.name for spec in outputs)
        _set(self, "outputs", outputs)
        _set(self, "function", function)
        _set(self, "constants", as_constants({} if constants is None else constants, message_name=f"{name!r} constants"))
        _set(self, "label_maps", as_label_maps({} if label_maps is None else label_maps, message_name=f"{name!r} label_maps"))
        _set(self, "own_dims", as_names(own_dims, message_name=f"{name!r} own_dims"))
        _set(self, "reads", function_reads(function, message_name=f"the function of {name!r}"))
        check_deterministic_spec_is_valid(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a DeterministicSpec is frozen; build another rather than setting {name!r}.")

    def __call__(self, **reads: Any) -> Array | dict[str, Array]:
        """``function(**reads)``."""
        return self.function(**reads)

    @property
    def name(self) -> str:
        """``"+".join`` of the outputs' names."""
        return "+".join(spec.name for spec in self.outputs)

    @property
    def given(self) -> tuple[str, ...]:
        """The components and inputs the function reads."""
        return tuple(n for n in self.reads if n not in self.constants and n not in self.label_maps)

    @property
    def law_name(self) -> str:
        """The function's name, for a description."""
        return getattr(self.function, "__name__", type(self.function).__name__)

    def __repr__(self) -> str:
        return f"DeterministicSpec({self.name!r}, given={list(self.given)})"


def deterministic(
    outputs: ArraySpec | Sequence[ArraySpec],
    /,
    *,
    constants: Mapping[str, xr.DataArray] | None = None,
    label_maps: Mapping[str, xr.DataArray] | None = None,
    own_dims: Sequence[str] = (),
) -> Callable[[Callable[..., Any]], DeterministicSpec]:
    """Decorate a function to make a :class:`DeterministicSpec` computing
    with it.

    Usage
    -----
    ::

        @deterministic(ArraySpec("soil_carbon_location", units="kg m-2", indexed_by=("site",)),
                       label_maps={"site_pft": site_dims.labels("pft")})
        def soil_carbon_location(soil_carbon_pft_mean, site_pft):
            return soil_carbon_pft_mean[site_pft]
    """

    def decorate(function: Callable[..., Any]) -> DeterministicSpec:
        return DeterministicSpec(outputs, function=function, constants=constants, label_maps=label_maps, own_dims=own_dims)

    return decorate


class Simulator(ABC):
    """A deterministic part computed outside JAX for a whole batch of
    samples: a simulation that may fail at some of them.

    The model calls it once per batch with the labeled values of what it
    reads, and reads at which samples each output was computed. Its outputs
    are constants to differentiation. It is real code, not a declaration:
    :meth:`at` is its own binding. A subclass defines :attr:`name`,
    :attr:`given`, :attr:`outputs`, :meth:`__call__` and :meth:`at`, and may
    define :meth:`check_given`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Its name in a description and in a posterior's evaluation."""

    @property
    @abstractmethod
    def given(self) -> tuple[str, ...]:
        """The components and inputs it reads."""

    @property
    @abstractmethod
    def outputs(self) -> tuple[ArraySpec, ...]:
        """The components it produces."""

    @abstractmethod
    def __call__(self, given_values: LabeledValues) -> SimulatorOutput:
        """Its outputs at every sample of a batch.

        Parameters
        ----------
        given_values : LabeledValues
            Each component and input it reads, on ``(sample, *indexed_by,
            *element axes)``, ``sample`` labeled ``0`` to ``J - 1``.

        Returns
        -------
        SimulatorOutput

        Raises
        ------
        Exception
            Whatever its machinery raises. A failure at a sample's values is
            reported in ``SimulatorOutput.valid``, never raised.
        """

    @abstractmethod
    def at(self, coords: Mapping[str, pd.Index], outputs: Sequence[str]) -> Simulator:
        """This simulator at the labels *coords*, computing only *outputs*:
        itself when both are what it was built for, a restricted copy
        otherwise. Binding calls it with every output, and a posterior with
        the outputs its likelihood depends on, so an output only barren
        factors read is never computed.

        Raises
        ------
        ValueError
            If it cannot produce its outputs at *coords*.
        """

    def check_given(self, given_specs: Mapping[str, ArraySpec], corner_values: LabeledValues) -> None:
        """Check, once, that what it reads fits it; ``condition_on`` calls it.
        Default: no check.

        Parameters
        ----------
        given_specs : Mapping[str, ArraySpec]
            The spec of each component and input it reads.
        corner_values : LabeledValues
            What :meth:`__call__` would receive at the corner points of the
            target.

        Raises
        ------
        ValueError
            If they do not fit, naming the component.
        """

    @property
    def law_name(self) -> str:
        """Its class's name, for the ``law`` column of a model's
        description, which names what each part evaluates."""
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, given={list(self.given)}, outputs={[o.name for o in self.outputs]})"


@dataclass(frozen=True, eq=False, kw_only=True)
class SimulatorOutput:
    """What one call of a :class:`Simulator` returns.

    Attributes
    ----------
    values : Mapping[str, numpy.ndarray]
        ``{output name: (J, *block)}``, ``float64``, ``NaN`` where not
        computed. Keyword-only, as are the others.
    valid : Mapping[str, numpy.ndarray]
        ``{output name: (J,)}`` bool: the output was computed at the sample
        and is finite.
    record : Any
        Kept on the evaluation for diagnostics; the model never reads it.
        Default ``None``.
    """

    values: Mapping[str, np.ndarray]
    valid: Mapping[str, np.ndarray]
    record: Any = None

    def __post_init__(self) -> None:
        for name in ("values", "valid"):
            held = getattr(self, name)
            if isinstance(held, Mapping):
                object.__setattr__(self, name, frozendict(held))


# ── helpers ───────────────────────────────────────────────────────────────────


def _set(part: Any, name: str, value: Any) -> None:
    object.__setattr__(part, name, value)


def _as_specs(specs: Any, *, what: str) -> tuple[ArraySpec, ...]:
    """One ArraySpec or a sequence of them, as a tuple."""
    if isinstance(specs, ArraySpec):
        return (specs,)
    check_specs_are_a_sequence(specs, what=what)
    specs = tuple(specs)
    check_specs_are_array_specs(specs, what=what)
    check_specs_are_not_empty(specs, what=what)
    return specs


def _law_reads(name: str, law: Any) -> tuple[str, ...]:
    """What a factor's law reads: nothing for a law, a builder's
    :attr:`~Builder.reads`, a Gaussian's mean and covariance's, a
    function's keywords."""
    if is_law(law):
        return ()
    if isinstance(law, (Builder, GaussianSpec)):
        return law.reads
    check_law_is_one_of_the_forms(name, law)
    return function_reads(law, message_name=f"the law of {name!r}")


# ── checks ────────────────────────────────────────────────────────────────────


def check_factor_spec_is_valid(spec: FactorSpec) -> None:
    """A factor declares distinct components indexed alike, its law reads
    what it holds and nothing it declares, and a bare law needs nothing the
    labels in use decide."""
    check_names_are_unique([s.name for s in spec.event], message_name=f"the factor {spec.name!r}'s event")
    check_joint_factor_is_indexed_alike(spec)
    check_bare_law_needs_nothing_bound(spec)
    check_part_reads_what_it_holds(spec, kind="factor")
    check_law_does_not_read_its_own_event(spec)
    check_provenance_is_a_string_or_none(spec)
    check_gaussian_event_is_one_vector_on_the_reals(spec)


def check_deterministic_spec_is_valid(spec: DeterministicSpec) -> None:
    """A deterministic computes distinct components from at least one
    component or input, reading what it holds and none of its outputs."""
    check_names_are_unique([s.name for s in spec.outputs], message_name=f"the deterministic {spec.name!r}'s outputs")
    check_part_reads_what_it_holds(spec, kind="deterministic")
    check_deterministic_reads_a_component(spec)
    check_deterministic_does_not_read_its_outputs(spec)


def check_simulator_is_valid(simulator: Simulator) -> None:
    """A simulator has a name, reads at least one component or input, each
    named once, and computes distinct components, none of which it reads."""
    check_simulator_name_is_a_string(simulator)
    what = f"the simulator {simulator.name!r}'s"
    check_simulator_given_is_names(simulator)
    check_names_are_unique(list(simulator.given), message_name=f"{what} given")
    check_specs_are_a_sequence(simulator.outputs, what=f"{what} outputs")
    check_specs_are_array_specs(simulator.outputs, what=f"{what} outputs")
    check_specs_are_not_empty(simulator.outputs, what=f"{what} outputs")
    check_names_are_unique([o.name for o in simulator.outputs], message_name=f"{what} outputs")
    check_simulator_does_not_read_its_outputs(simulator)


def check_simulator_name_is_a_string(simulator: Simulator) -> None:
    """A simulator is named by a non-empty string, which keys its outputs'
    validity and its record."""
    if not isinstance(simulator.name, str) or not simulator.name:
        raise TypeError(f"a {type(simulator).__name__}'s name is {simulator.name!r}; give it a non-empty string.")


def check_simulator_given_is_names(simulator: Simulator) -> None:
    """A simulator reads a tuple of at least one name."""
    given = simulator.given
    if not isinstance(given, tuple) or not all(isinstance(g, str) for g in given):
        raise TypeError(f"the simulator {simulator.name!r}'s given is {given!r}; give a tuple of names.")
    if not given:
        raise ValueError(f"the simulator {simulator.name!r} reads no component or input; a fixed value is a constant.")


def check_simulator_does_not_read_its_outputs(simulator: Simulator) -> None:
    """A simulator reads none of the components it computes."""
    own = [g for g in simulator.given if g in {o.name for o in simulator.outputs}]
    if own:
        raise ValueError(
            f"the simulator {simulator.name!r} reads {own}, which it computes; a component is computed from others."
        )


def check_specs_are_a_sequence(specs: Any, *, what: str) -> None:
    """Components are given as one ArraySpec or a sequence of them."""
    if isinstance(specs, (str, bytes, Mapping)) or not isinstance(specs, Sequence):
        raise TypeError(f"{what} is a {type(specs).__name__}; give an ArraySpec or a sequence of them.")


def check_specs_are_array_specs(specs: Sequence[Any], *, what: str) -> None:
    """Every component is declared by an ArraySpec."""
    wrong = [type(s).__name__ for s in specs if not isinstance(s, ArraySpec)]
    if wrong:
        raise TypeError(f"{what} holds {truncated(wrong)}; declare each component with an ArraySpec.")


def check_specs_are_not_empty(specs: Sequence[ArraySpec], *, what: str) -> None:
    """A part declares at least one component."""
    if not specs:
        raise ValueError(f"{what} is empty; declare at least one component.")


def check_joint_factor_is_indexed_alike(spec: FactorSpec) -> None:
    """A joint factor's components are indexed by the same dims: dependence
    across dims is what a law reading another component expresses."""
    indexed = {s.indexed_by for s in spec.event}
    if len(indexed) > 1:
        raise ValueError(
            f"the joint factor {spec.name!r} declares components indexed by {sorted(indexed)}; a joint "
            "factor's components are indexed alike, so give each its own factor, the one's law reading "
            "the other."
        )


def check_part_reads_what_it_holds(spec: FactorSpec | DeterministicSpec, *, kind: str) -> None:
    """A part names each of its components, constants and label maps once,
    and reads every constant and label map it holds, which it would
    otherwise ignore."""
    declared = [s.name for s in (spec.event if kind == "factor" else spec.outputs)]
    check_names_are_unique(
        [*declared, *spec.constants, *spec.label_maps],
        message_name=f"the {kind} {spec.name!r}'s components, constants and label maps",
    )
    unread = [n for n in (*spec.constants, *spec.label_maps) if n not in spec.reads]
    if unread:
        raise ValueError(
            f"the {kind} {spec.name!r} holds {truncated(unread)}, which its function never reads (it "
            f"reads {list(spec.reads)}); name each as a keyword parameter of the function, or drop it."
        )


def check_law_does_not_read_its_own_event(spec: FactorSpec) -> None:
    """A factor's law reads no component it declares, whose law it is."""
    own = [n for n in spec.given if n in {s.name for s in spec.event}]
    if own:
        raise ValueError(
            f"the law of {spec.name!r} reads {own}, which the factor declares; a law is the density "
            "of its event, so it reads only other components."
        )


def check_bare_law_needs_nothing_bound(spec: FactorSpec) -> None:
    """A bare law is over one component indexed by nothing, and reads no
    constant: a law over labels, or of constants read at them, is built for
    the labels in use."""
    if not is_law(spec.law):
        return
    if len(spec.event) > 1:
        raise TypeError(
            f"the joint factor {spec.name!r} is given a bare law; give a function returning a law over "
            "the dict of its blocks, or gaussian_copula(...)."
        )
    if spec.event[0].indexed_by:
        raise TypeError(
            f"the factor {spec.name!r} is over a component indexed by {spec.event[0].indexed_by}, so its "
            "law is built for the labels in use; wrap the law as iid_over_dim(law) or "
            "independent_over_dim(family, ...)."
        )
    if spec.constants or spec.label_maps:
        raise TypeError(
            f"the factor {spec.name!r} holds {sorted([*spec.constants, *spec.label_maps])} but is given a "
            "bare law; give a function of them that returns one."
        )


def check_gaussian_has_one_mean(mean: tuple[str, ...]) -> None:
    """A Gaussian is centered on one component: a Gaussian whose event spans
    several components, each on its own dim, is not built."""
    if len(mean) != 1:
        raise ValueError(
            f"a GaussianSpec is centered on {list(mean)}; center it on one component. A Gaussian whose event "
            "spans several components is not supported: give each its own factor."
        )


def check_covariance_is_a_covariance_spec(covariance: Any) -> None:
    """A Gaussian's covariance is declared by a covariance spec."""
    if not isinstance(covariance, CovarianceSpec):
        raise TypeError(
            f"a GaussianSpec's covariance is a {type(covariance).__name__}; give a covariance spec, such as "
            "DiagonalSpec(...) or BlockDiagonalSpec(DenseSpec(...), by='site')."
        )


def check_gaussian_event_is_one_vector_on_the_reals(spec: FactorSpec) -> None:
    """A Gaussian factor's event is one component on ``REAL``, indexed by one
    dim at most and with no element axes, so its entries are its labels."""
    if not isinstance(spec.law, GaussianSpec):
        return
    component, *rest = spec.event
    if rest or component.support != REAL or len(component.indexed_by) > 1 or component.element_axes:
        raise ValueError(
            f"the factor {spec.name!r} is Gaussian, so its event is one component on REAL, indexed by one dim "
            f"at most and with no element axes; it declares {[c.name for c in spec.event]}, the first on "
            f"{component.support.name!r}, indexed by {component.indexed_by}, with element axes "
            f"{list(component.element_axes)}."
        )


def check_deterministic_reads_a_component(spec: DeterministicSpec) -> None:
    """A deterministic reads a component or an input: a value computed from
    constants alone is a constant."""
    if not spec.given:
        raise ValueError(
            f"the deterministic {spec.name!r} reads no component or input; a value fixed across draws "
            "is a constant."
        )


def check_deterministic_does_not_read_its_outputs(spec: DeterministicSpec) -> None:
    """A deterministic reads none of the components it computes."""
    own = [n for n in spec.given if n in {s.name for s in spec.outputs}]
    if own:
        raise ValueError(
            f"the deterministic {spec.name!r} reads {own}, which it computes; a component is "
            "computed from others."
        )


def check_law_is_one_of_the_forms(name: str, law: Any) -> None:
    """A factor's law is a law, a builder or a function returning a law."""
    if not callable(law):
        raise TypeError(
            f"the law of {name!r} is a {type(law).__name__}; give a TFP distribution, a builder such as "
            "iid_over_dim(...), or a function of what it reads that returns a law."
        )


def check_provenance_is_a_string_or_none(spec: FactorSpec) -> None:
    """A provenance, when given, is a sentence."""
    if spec.provenance is None:
        return
    if not isinstance(spec.provenance, str):
        raise TypeError(f"the factor {spec.name!r} has provenance {spec.provenance!r}; give a string, or leave it out.")
    if not spec.provenance.strip():
        raise ValueError(
            f"the factor {spec.name!r} has an empty provenance; give a sentence saying where the law comes "
            "from, or leave it out."
        )
