"""A parameter: one array-valued unknown, and the bijection that gives it
unconstrained coordinates.

A :class:`Parameter` describes one representation of a value: its name, the
:class:`~sipnet_calibration.parameters.support.Support` its values lie in,
its units, the shape and labels of one value, and the dims it is replicated
over. Its **transform** is a bijection

.. math::

    T : \\mathbb{R}^{u} \\to A, \\qquad \\theta \\mapsto x,

from unconstrained space onto the interior of its support :math:`A`, acting
on one value, whose natural shape is ``shape`` and whose unconstrained shape
``u`` is ``bijector.inverse_event_shape(shape)``: the same shape for a
support decided per number, ``(..., k - 1)`` for ``(..., k)`` on the
simplex. :meth:`Parameter.unconstrained` is the parameter describing
:math:`\\theta = T^{-1}(x)`.

A value is one unit: its numbers are called its **elements**, laid out in C
(row-major) order, and named by its element labels, one axis name and one
tuple of string labels per axis of ``shape``, its **element axes**.

Values and blocks
-----------------
``shape`` is the shape of **one value**, never of the parameter as a whole.
A parameter indexed by dims has one value at each tuple of their labels;
all of them together are its **block**, of block shape ``(*index shape,
*shape)``, the index shape being the number of labels in use of each dim
in ``indexed_by``. For example, an allocation simplex with ``shape=(4,)``
indexed by ``pft`` has, over three PFTs, three values of shape ``(4,)`` and
a block of shape ``(3, 4)``; a soil carbon indexed by ``site``, with
``shape=()``, has a block of shape ``(S,)``. The labels, and so the index
shape, belong to the vector (``ParameterVector.coords``), and a selection
changes them; ``shape`` never changes.

A prior term's TFP event is its parameters' blocks. One value's axes, the
element axes, are the event of one value's distribution, which
:func:`~sipnet_calibration.parameters.prior_functions.iid_over_dim`
repeats over the block. ``support.event_ndims`` is a third thing: how many trailing element
axes the support constrains jointly, 0 on an interval and 1 on the simplex.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from frozendict import frozendict
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.parameters._description import (
    as_shape,
    check_description_is_valid,
    resolved_element_labels,
)
from sipnet_calibration.parameters._probes import probe_points
from sipnet_calibration.parameters._validation import as_names
from sipnet_calibration.parameters.support import REAL, Support, bijector_for

__all__ = [
    "Parameter",
    "check_parameter_is_valid",
    "check_shape_has_the_supports_event_axes",
]

tfb = tfp.bijectors


@dataclass(frozen=True, eq=False, kw_only=True)
class Parameter:
    """An array-valued quantity: its name, the set its values lie in, its
    units, the shape and labels of one value, the dims it is replicated
    over, and a bijection from unconstrained space onto its set.

    Parameters
    ----------
    name:
        The parameter's name.
    support:
        The set one value lies in. Default :data:`REAL`.
    units:
        The value's units, or ``None`` for none. Required.
    shape:
        The shape of one value, at one tuple of labels of *indexed_by*, not
        of the block. Default ``()``: a scalar.
    element_labels:
        ``{axis name: labels}``, one entry per axis of *shape*, in order,
        each as long as its axis, labels unique strings. They name the
        labeled form's element axes and coordinates and the index's
        ``element`` level; nothing numeric reads them. ``None`` names the
        axes ``<name>_axis_0``, ``<name>_axis_1``, ... labeled ``"0"`` to
        ``"n - 1"``.
    indexed_by:
        The dims the value is replicated over, in order: the parameter has
        one value per tuple of their labels, and its block is ``(*index
        shape, *shape)``. Their labels are the vector's
        (``ParameterVector.coords``). Default ``()``.
    bijector:
        :math:`T`, from unconstrained space onto *support*, acting on the
        last ``support.event_ndims`` axes. ``None`` takes
        ``bijector_for(support)``. Stored resolved.
    long_name:
        A human-readable name, for tables and figures; it carries the
        transform's name into :meth:`unconstrained`.

    Attributes
    ----------
    element_labels : frozendict of str to pandas.Index
        Each axis's labels, the defaults filled in.

    Raises
    ------
    TypeError
        For an argument of the wrong type.
    ValueError
        For a name that is not a Python identifier, or is a keyword (prior
        and derived functions receive values as keyword arguments named for
        it); repeated dims, labels or axis names, or axis names taken by the
        name or a dim; element labels whose lengths are not *shape*; a shape
        lacking the support's event axes (a simplex needs a last axis of 2
        or more); or a custom bijector that does not map onto the support at
        the probe points.
    """

    name: str
    support: Support = REAL
    units: str | None
    shape: tuple[int, ...] = ()
    element_labels: Mapping[str, Sequence[str]] | None = None
    indexed_by: tuple[str, ...] = ()
    bijector: tfb.Bijector | None = None
    long_name: str | None = None

    def __post_init__(self) -> None:
        check_support_is_a_support(self.name, self.support)
        object.__setattr__(self, "shape", as_shape(self.shape, message_name=f"{self.name!r} shape"))
        object.__setattr__(self, "indexed_by", as_names(self.indexed_by, message_name=f"{self.name!r} indexed_by"))
        object.__setattr__(
            self, "element_labels", resolved_element_labels(self.name, self.shape, self.element_labels)
        )
        custom = self.bijector is not None
        if not custom:
            object.__setattr__(self, "bijector", bijector_for(self.support))
        check_parameter_is_valid(self, custom_bijector=custom)

    @property
    def unconstrained_shape(self) -> tuple[int, ...]:
        """One unconstrained value's shape, ``bijector.inverse_event_shape(shape)``."""
        return tuple(int(n) for n in self.bijector.inverse_event_shape(self.shape))

    def unconstrained(self) -> Parameter:
        """The description of theta's part for this parameter, :math:`T^{-1}`
        of one value: the same name and ``indexed_by``; support
        :data:`REAL`; units ``None``; the identity; shape
        :attr:`unconstrained_shape`.

        Its long name is :math:`g(n)`, with :math:`n` this parameter's long
        name or name and :math:`g` the name of :math:`T^{-1}`: ``log`` for
        ``Exp``, ``logit`` for ``Sigmoid``, and ``alr``, the additive
        log-ratio :math:`\\theta_i = \\log(x_i / x_k)`, for
        ``SoftmaxCentered``; under the identity it is this parameter's long
        name. Its element labels are the natural ones, the last label of the
        last axis dropped under ``SoftmaxCentered``. Another bijector gives
        :math:`g` its own name, and keeps the natural labels when it keeps
        the shape, the defaults otherwise.
        """
        coordinates = _UNCONSTRAINED_COORDINATES.get(type(self.bijector))
        if coordinates is None:
            transform = self.bijector.name
            labels = dict(self.element_labels) if self.unconstrained_shape == self.shape else None
        else:
            transform, labels = coordinates.transform, coordinates.element_labels(self.element_labels)
        return Parameter(
            name=self.name,
            support=REAL,
            units=None,
            shape=self.unconstrained_shape,
            element_labels=None if labels is None else {axis: list(v) for axis, v in labels.items()},
            indexed_by=self.indexed_by,
            long_name=self.long_name if transform is None else f"{transform}({self.long_name or self.name})",
        )

    def __repr__(self) -> str:
        indexed = f", indexed_by={self.indexed_by}" if self.indexed_by else ""
        shape = f", shape={self.shape}" if self.shape else ""
        return f"Parameter(name={self.name!r}, support={self.support.name!r}, units={self.units!r}{shape}{indexed})"


# ── helpers ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Coordinates:
    """What theta's coordinates are under one type of bijector: the name of
    :math:`T^{-1}` (``None`` for the identity, which keeps the long name),
    and theta's element labels from the natural ones."""

    transform: str | None
    element_labels: Callable[[Mapping[str, pd.Index]], dict[str, pd.Index]]


def _same_labels(labels: Mapping[str, pd.Index]) -> dict[str, pd.Index]:
    return dict(labels)


def _without_the_last_label(labels: Mapping[str, pd.Index]) -> dict[str, pd.Index]:
    """The labels with the last label of the last axis dropped: theta's
    element :math:`i` is the additive log-ratio :math:`\\log(x_i / x_k)`."""
    out = dict(labels)
    last = list(out)[-1]
    out[last] = out[last][:-1]
    return out


#: theta's coordinates under each type of default bijector: ``log`` for
#: ``Exp``, ``logit`` for ``Sigmoid``, and ``alr``, the additive log-ratio
#: :math:`\theta_i = \log(x_i / x_k)`, for ``SoftmaxCentered``, which drops
#: the last label.
_UNCONSTRAINED_COORDINATES: Mapping[type, _Coordinates] = frozendict(
    {
        tfb.Identity: _Coordinates(None, _same_labels),
        tfb.Exp: _Coordinates("log", _same_labels),
        tfb.Sigmoid: _Coordinates("logit", _same_labels),
        tfb.SoftmaxCentered: _Coordinates("alr", _without_the_last_label),
    }
)


def _maps_onto_the_support(parameter: Parameter) -> bool:
    """Whether the parameter's bijector maps the probe points into the
    support's closure, and the default bijector's images of them back to
    finite points that it maps there again; the first half alone for a
    support whose type has no default bijector."""
    try:
        shape = parameter.unconstrained_shape
    except (TypeError, ValueError):
        return False
    probes = jnp.asarray(probe_points(shape))
    support = parameter.support
    # The closure: a bijector may round onto the boundary at the outer probes
    # (IteratedSigmoidCentered does at 20), which is float64, not a wrong map.
    into = bool(jnp.all(support.closure().contains(parameter.bijector.forward(probes))))
    try:
        default = bijector_for(support)
    except KeyError:
        return into
    targets = default.forward(jnp.array(probes))
    back = parameter.bijector.inverse(targets)
    # A fresh copy: TFP caches the pair, and would hand targets back unchanged.
    again = parameter.bijector.forward(jnp.array(np.asarray(back)))
    return (
        into
        and bool(jnp.all(jnp.isfinite(back)))
        and bool(np.allclose(again, targets, rtol=1e-6, atol=1e-9))
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_parameter_is_valid(parameter: Parameter, *, custom_bijector: bool = False) -> None:
    """A parameter describes a value the vector can lay out, and its
    bijector maps onto its support."""
    check_description_is_valid(
        name=parameter.name,
        units=parameter.units,
        element_labels=parameter.element_labels,
        indexed_by=parameter.indexed_by,
        long_name=parameter.long_name,
        what="a parameter",
    )
    check_shape_has_the_supports_event_axes(parameter.name, parameter.shape, parameter.support)
    if custom_bijector:
        check_bijector_is_a_bijector(parameter)
        check_bijector_acts_on_the_supports_events(parameter)
        check_bijector_maps_onto_the_support(parameter)


def check_support_is_a_support(name: str, support: Any) -> None:
    """A support is a :class:`Support`."""
    if not isinstance(support, Support):
        raise TypeError(f"{name!r} has support {support!r}; give a Support, such as POSITIVE or Interval(0, 1).")


def check_shape_has_the_supports_event_axes(name: str, shape: tuple[int, ...], support: Support) -> None:
    """A value has the axes its support decides membership over, and a
    simplex at least two numbers, since the support would otherwise be read
    across the value's other axes."""
    if len(shape) < support.event_ndims or (support.event_ndims == 1 and shape[-1] < 2):
        raise ValueError(
            f"{name!r} has shape {shape}, but its support {support.name!r} decides membership over "
            f"the last {support.event_ndims} axes, and a simplex needs at least two numbers; give a "
            "shape such as (4,)."
        )


def check_bijector_is_a_bijector(parameter: Parameter) -> None:
    """A custom bijector is a TFP bijector."""
    if not isinstance(parameter.bijector, tfb.Bijector):
        raise TypeError(
            f"{parameter.name!r} has bijector {parameter.bijector!r}; give a TFP bijector, or omit it "
            "to take the support's."
        )


def check_bijector_acts_on_the_supports_events(parameter: Parameter) -> None:
    """A custom bijector acts on the axes its support decides membership
    over, as the Jacobian is computed: each number alone on an interval, a
    whole vector on the simplex."""
    if parameter.bijector.forward_min_event_ndims != parameter.support.event_ndims:
        raise ValueError(
            f"{parameter.name!r}: its bijector acts on {parameter.bijector.forward_min_event_ndims}"
            f"-dimensional events, but its support {parameter.support.name!r} decides membership over "
            f"{parameter.support.event_ndims}; give a bijector acting on those."
        )


def check_bijector_maps_onto_the_support(parameter: Parameter) -> None:
    """A custom bijector maps the probe points into the support's closure,
    and the support's default image of them back to finite points that it
    maps there again, since a wrong bijector changes the support silently."""
    if not _maps_onto_the_support(parameter):
        raise ValueError(
            f"{parameter.name!r}: its bijector {parameter.bijector.name!r} does not map theta onto "
            f"the support {parameter.support.name!r} at the probe points; give a bijector onto the "
            "support, or omit bijector= to take the support's own."
        )
