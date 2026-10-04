"""ArraySpec: the declaration of one component, a named array a draw of a
model holds.

Where this sits
---------------
::

    probability.support, probability.names
      -> probability.spec.ArraySpec   (what a component is: no labels, no numbers)
      -> probability.layout.Layout    (components at the labels in use)

It reads nothing from disk. :class:`ArraySpec` is its one class;
:meth:`ArraySpec.unconstrained` gives theta's part for a component.

An :class:`ArraySpec` says what a component's values are, and which axes
they have. There are two kinds of axes, and their labels come from
different places:

============= ======================================================== ==================
axes          what they are                                            labels come from
============= ======================================================== ==================
element axes  the axes of one value: the parts of an allocation         the spec
              simplex, the rows and columns of a covariance matrix      (``element_axes``)
indexed_by    the dims the value is replicated over: one value per     the coords, given
              site, per PFT                                             when a model is
                                                                        bound
============= ======================================================== ==================

A component's **block** is all its values, of shape ``(*index shape,
*shape)``, the index shape being the number of labels of each
``indexed_by`` dim and ``shape`` one value's, the lengths of its element
axes.

Its **transform** is a bijection

.. math::

    T : \\mathbb{R}^{u} \\to A, \\qquad \\theta \\mapsto x,

from unconstrained space onto the interior of its support :math:`A`,
acting on one value; ``u``, the value's unconstrained shape, is
``bijector.inverse_event_shape(shape)``. :meth:`ArraySpec.unconstrained`
is the spec of :math:`\\theta = T^{-1}(x)`, whose element axes are:

=================================== ================================================
transform                           theta's element axes
=================================== ================================================
one that keeps the shape            the value's
``SoftmaxCentered`` (the simplex)   the value's, the last label of the last axis
                                    dropped: :math:`\\theta_i = \\log(x_i / x_k)`
the positive-definite default       the last two axes ``(row, column)`` replaced by
                                    one, ``"<row>_<column>_cholesky"``, labeled
                                    ``"L[i,j]"`` by the rows' and columns' labels, in
                                    ``tfb.FillTriangular``'s packing
any other                           ``"<name>_axis_<i>"``, labeled ``"0"`` to
                                    ``"n - 1"``
=================================== ================================================

The transform is read only when the component is a parameter, where it
fixes theta's coordinates.
"""

from __future__ import annotations

import keyword
from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from frozendict import frozendict
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability._probes import probe_points
from sipnet_calibration.probability._validation import as_names, check_names_are_unique, truncated
from sipnet_calibration.probability.names import RESERVED_NAMES
from sipnet_calibration.probability.support import REAL, PositiveDefinite, Support, bijector_for

__all__ = [
    "ArraySpec",
    "check_array_spec_is_valid",
    "check_shape_has_the_supports_event_axes",
]

tfb = tfp.bijectors


class ArraySpec:
    """A declaration of one component: a named array a draw of a model holds.

    A component has one **value** at each tuple of labels of its
    ``indexed_by`` dims; a value's own axes are its **element axes**. All its
    values are its **block**, of shape ``(n_1, ..., n_m, e_1, ..., e_r)``,
    with ``n_i`` the number of labels of the ``i``-th ``indexed_by`` dim,
    given when the model is bound, and ``e_j`` the length of the ``j``-th
    element axis, given here.

    Parameters
    ----------
    name : str
        Positional-only. A Python identifier and not a keyword, since
        functions receive the component's values by this name; not one of
        :data:`~sipnet_calibration.probability.names.RESERVED_NAMES`.
    units : str or None
        Keyword-only, required. A UDUNITS string, ``"1"`` for a
        dimensionless quantity, or ``None`` when units do not apply (an
        innovation, a coefficient of standardized covariates).
    support : Support
        Keyword-only. The set one value lies in. Default :data:`REAL`.
    indexed_by : Sequence[str]
        Keyword-only. The dims the value is replicated over, in order.
        Default ``()``.
    element_axes : Mapping[str, Sequence[str] | int]
        Keyword-only. One entry per axis of one value, in order: the axis's
        name, and its labels (unique strings), or its length ``n``, which
        labels it ``"0"`` to ``"n - 1"``. Default ``{}``: a scalar.
    bijector : tfb.Bijector, optional
        Keyword-only. :math:`T`, from unconstrained space onto the support,
        acting on the last ``support.event_ndims`` element axes; default
        ``bijector_for(support)``.

    Attributes
    ----------
    element_axes : frozendict of str to pandas.Index
        Each axis's labels, named for the axis.
    shape : tuple of int
        One value's shape, the lengths of the element axes.
    unconstrained_shape : tuple of int
        One unconstrained value's shape.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If the name is not an identifier, is a keyword or is reserved; a
        dim, axis or label repeats or is reserved; an axis is named like the
        component or a dim; the shape lacks the support's event axes (a
        simplex needs a last axis of at least 2, a positive-definite value
        two last axes with one set of labels); or a custom bijector does not
        map onto the support at the probe points.

    Usage
    -----
    ::

        ArraySpec("respiration_share", units="1", support=OPEN_UNIT_INTERVAL)
        ArraySpec("allocation", units="1", support=SIMPLEX, indexed_by=("pft",),
                  element_axes={"allocation_part": ("leaf", "wood", "fine_root", "coarse_root")})
        years = ("2012", "2013", "2014")
        ArraySpec("biomass_error_covariance", units="Mg2 ha-2", support=POSITIVE_DEFINITE,
                  element_axes={"year": years, "other_year": years})
    """

    __slots__ = ("name", "units", "support", "indexed_by", "element_axes", "bijector", "_custom_bijector")

    name: str
    units: str | None
    support: Support
    indexed_by: tuple[str, ...]
    element_axes: frozendict
    bijector: tfb.Bijector

    def __init__(
        self,
        name: str,
        /,
        *,
        units: str | None,
        support: Support = REAL,
        indexed_by: Sequence[str] = (),
        element_axes: Mapping[str, Sequence[str] | int] | None = None,
        bijector: tfb.Bijector | None = None,
    ) -> None:
        check_name_is_an_identifier(name, what="a component")
        check_support_is_a_support(name, support)
        _set(self, "name", name)
        _set(self, "units", units)
        _set(self, "support", support)
        _set(self, "indexed_by", as_names(indexed_by, message_name=f"{name!r} indexed_by"))
        _set(self, "element_axes", _as_element_axes(name, {} if element_axes is None else element_axes))
        _set(self, "_custom_bijector", bijector is not None)
        _set(self, "bijector", bijector_for(support) if bijector is None else bijector)
        check_array_spec_is_valid(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"an ArraySpec is frozen; build another rather than setting {name!r}.")

    def __reduce__(self) -> tuple[Any, ...]:
        return (_spec_from_arguments, (self.name, self._arguments()))

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, ...]:
        """One value's shape, the lengths of the element axes."""
        return tuple(len(labels) for labels in self.element_axes.values())

    @property
    def unconstrained_shape(self) -> tuple[int, ...]:
        """One unconstrained value's shape, ``bijector.inverse_event_shape(shape)``."""
        return tuple(int(n) for n in self.bijector.inverse_event_shape(self.shape))

    def unconstrained(self) -> ArraySpec:
        """The spec of theta's part for this component, :math:`T^{-1}` of one
        value: the same name and ``indexed_by``; support :data:`REAL`; units
        ``None``; the identity; element axes as the module's table gives
        them."""
        return ArraySpec(
            self.name,
            units=None,
            indexed_by=self.indexed_by,
            element_axes={axis: list(labels) for axis, labels in _unconstrained_element_axes(self).items()},
        )

    def __repr__(self) -> str:
        indexed = f", indexed_by={self.indexed_by}" if self.indexed_by else ""
        axes = f", element_axes={ {axis: len(labels) for axis, labels in self.element_axes.items()} }" if self.element_axes else ""
        return f"ArraySpec({self.name!r}, units={self.units!r}, support={self.support.name!r}{indexed}{axes})"

    # ── supporting methods ────────────────────────────────────────────────────

    def _arguments(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "support": self.support,
            "indexed_by": self.indexed_by,
            "element_axes": {axis: list(labels) for axis, labels in self.element_axes.items()},
            "bijector": self.bijector if self._custom_bijector else None,
        }


# ── helpers ───────────────────────────────────────────────────────────────────


def _set(spec: ArraySpec, name: str, value: Any) -> None:
    object.__setattr__(spec, name, value)


def _spec_from_arguments(name: str, arguments: Mapping[str, Any]) -> ArraySpec:
    """An ArraySpec from its arguments, for pickling."""
    return ArraySpec(name, **arguments)


def _as_element_axes(name: str, element_axes: Any) -> frozendict:
    """``{axis: pd.Index of string labels named for the axis}``: labels as
    given, or ``"0"`` to ``"n - 1"`` for a length ``n``."""
    check_element_axes_are_a_mapping(name, element_axes)
    out = {}
    for axis, labels in element_axes.items():
        check_axis_name_is_a_string(name, axis)
        if isinstance(labels, (int, np.integer)) and not isinstance(labels, (bool, np.bool_)):
            check_axis_length_is_positive(name, axis, int(labels))
            listed: tuple[str, ...] = tuple(str(j) for j in range(int(labels)))
        else:
            listed = as_names(labels, message_name=f"{name!r} element_axes[{axis!r}]")
            check_names_are_unique(listed, message_name=f"{name!r} element_axes[{axis!r}]")
            check_axis_has_a_label(name, axis, listed)
        out[axis] = pd.Index(listed, name=axis, dtype=object)
    return frozendict(out)


def _unconstrained_element_axes(spec: ArraySpec) -> dict[str, pd.Index]:
    """Theta's element axes for *spec*, as the module's table gives them."""
    axes = dict(spec.element_axes)
    if isinstance(spec.bijector, tfb.SoftmaxCentered):
        last = list(axes)[-1]
        axes[last] = axes[last][:-1]
        return axes
    if isinstance(spec.support, PositiveDefinite) and not spec._custom_bijector:
        *leading, row, column = list(axes)
        cholesky = f"{row}_{column}_cholesky"
        out = {axis: axes[axis] for axis in leading}
        out[cholesky] = pd.Index(_cholesky_labels(axes[row], axes[column]), name=cholesky, dtype=object)
        return out
    if spec.unconstrained_shape == spec.shape:
        return axes
    return {
        f"{spec.name}_axis_{i}": pd.Index([str(j) for j in range(n)], name=f"{spec.name}_axis_{i}", dtype=object)
        for i, n in enumerate(spec.unconstrained_shape)
    }


def _cholesky_labels(rows: pd.Index, columns: pd.Index) -> list[str]:
    """``"L[i,j]"`` for each entry of theta's packing of a lower-triangular
    factor, by the labels of its row and column."""
    p = len(rows)
    n = p * (p + 1) // 2
    filled = np.asarray(tfb.FillTriangular().forward(jnp.arange(n, dtype=jnp.float64)))
    i, j = np.tril_indices(p)
    entry = filled[i, j].astype(int)
    out = [""] * n
    for k, row, column in zip(entry, i, j):
        out[k] = f"L[{rows[row]},{columns[column]}]"
    return out


def _maps_onto_the_support(spec: ArraySpec) -> bool:
    """Whether the spec's bijector maps the probe points into the support's
    closure, and the default bijector's images of them back to finite points
    that it maps there again; the first half alone for a support whose type
    has no default bijector."""
    try:
        shape = spec.unconstrained_shape
    except (TypeError, ValueError):
        return False
    probes = jnp.asarray(probe_points(shape))
    support = spec.support
    # The closure: a bijector may round onto the boundary at the outer probes
    # (IteratedSigmoidCentered does at 20), which is float64, not a wrong map.
    into = bool(jnp.all(support.closure().contains(spec.bijector.forward(probes))))
    try:
        default = bijector_for(support)
    except KeyError:
        return into
    targets = default.forward(jnp.array(probes))
    # Only where the default bijector inverts its own image: at the probes
    # -10 * 1 and -20 * 1 a positive-definite image is too ill-conditioned
    # for any inverse. Fresh copies throughout: TFP caches each pair, and
    # would hand an input back unchanged.
    reference = np.asarray(default.inverse(jnp.array(np.asarray(targets))))
    invertible = np.all(np.isfinite(reference) & np.isclose(reference, probes, rtol=1e-6, atol=1e-6),
                        axis=tuple(range(1, reference.ndim)))
    targets = jnp.array(np.asarray(targets)[invertible])
    back = spec.bijector.inverse(targets)
    again = spec.bijector.forward(jnp.array(np.asarray(back)))
    return (
        into
        and bool(jnp.all(jnp.isfinite(back)))
        and bool(np.allclose(again, targets, rtol=1e-6, atol=1e-9))
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_array_spec_is_valid(spec: ArraySpec) -> None:
    """A spec describes one value that the layouts and labeled forms can
    hold, and its bijector maps onto its support."""
    check_units_are_a_string_or_none(spec.name, spec.units)
    check_names_are_unique(spec.indexed_by, message_name=f"{spec.name!r} indexed_by")
    check_names_are_not_reserved(spec.name, [*spec.indexed_by, *spec.element_axes])
    check_axis_names_are_free(spec.name, spec.element_axes, spec.indexed_by)
    check_shape_has_the_supports_event_axes(spec.name, spec.shape, spec.support)
    if isinstance(spec.support, PositiveDefinite):
        check_matrix_axes_share_their_labels(spec.name, spec.element_axes)
    if spec._custom_bijector:
        check_bijector_is_a_bijector(spec)
        check_bijector_acts_on_the_supports_events(spec)
        check_bijector_maps_onto_the_support(spec)


def check_name_is_an_identifier(name: Any, *, what: str) -> None:
    """A name is a Python identifier, not a keyword and not reserved, since
    functions receive values as keyword arguments named for it."""
    if not isinstance(name, str):
        raise TypeError(f"{what}'s name is a string, got {type(name).__name__} {name!r}; name it with a string.")
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            f"{what} is named {name!r}, which is not a Python identifier, or is a keyword; values "
            "are passed as keyword arguments named for it, so name it like 'base_soil_respiration'."
        )
    if name in RESERVED_NAMES:
        raise ValueError(f"{what} is named {name!r}, which the probability layer reserves; name it for what it is.")


def check_support_is_a_support(name: str, support: Any) -> None:
    """A support is a :class:`Support`."""
    if not isinstance(support, Support):
        raise TypeError(f"{name!r} has support {support!r}; give a Support, such as POSITIVE or Interval(0, 1).")


def check_units_are_a_string_or_none(name: str, units: Any) -> None:
    """Units are a string, or ``None`` where units do not apply."""
    if units is not None and not isinstance(units, str):
        raise TypeError(f"{name!r} has units {units!r}; give a string, or None where units do not apply.")


def check_element_axes_are_a_mapping(name: str, element_axes: Any) -> None:
    """Element axes are ``{axis name: labels or length}``."""
    if not isinstance(element_axes, Mapping):
        raise TypeError(
            f"{name!r} element_axes is a {type(element_axes).__name__}; give a mapping "
            "{axis name: labels or length}, one entry per axis of one value."
        )


def check_axis_name_is_a_string(name: str, axis: Any) -> None:
    """An element axis is named by a string, which names a dim of the labeled forms."""
    if not isinstance(axis, str):
        raise TypeError(f"{name!r} names an element axis {axis!r}, but axis names are strings; name it with one.")


def check_axis_length_is_positive(name: str, axis: str, length: int) -> None:
    """An element axis given by its length holds at least one number."""
    if length < 1:
        raise ValueError(f"{name!r} element_axes[{axis!r}] has length {length}; an axis holds at least one number.")


def check_axis_has_a_label(name: str, axis: str, labels: Sequence[str]) -> None:
    """An element axis holds at least one label."""
    if not labels:
        raise ValueError(f"{name!r} element_axes[{axis!r}] holds no label; give at least one.")


def check_names_are_not_reserved(name: str, names: Sequence[str]) -> None:
    """No dim or element axis takes a name the layer reserves."""
    reserved = [n for n in names if n in RESERVED_NAMES]
    if reserved:
        raise ValueError(
            f"{name!r} names a dim or element axis {truncated(reserved)}, which the probability layer "
            "reserves; name it for what it indexes."
        )


def check_axis_names_are_free(name: str, element_axes: Mapping[str, pd.Index], indexed_by: tuple[str, ...]) -> None:
    """An element axis is named neither for the component nor for a dim it
    is indexed by, which the labeled forms would take for one another."""
    taken = [axis for axis in element_axes if axis == name or axis in indexed_by]
    if taken:
        raise ValueError(
            f"{name!r} names element axes {truncated(taken)} like itself or a dim it is indexed "
            "by; name its axes for what they index."
        )


def check_shape_has_the_supports_event_axes(name: str, shape: tuple[int, ...], support: Support) -> None:
    """A value has the axes its support decides membership over: a simplex
    at least two numbers, a positive-definite value two last axes of one
    length."""
    n = support.event_ndims
    if len(shape) < n or (n == 1 and shape[-1] < 2) or (n == 2 and shape[-1] != shape[-2]):
        raise ValueError(
            f"{name!r} has shape {shape}, but its support {support.name!r} decides membership over "
            f"the last {n} axes; a simplex needs at least two numbers, and a matrix two axes of one "
            "length, so give element axes such as {'part': 4} or {'year': YEARS, 'other_year': YEARS}."
        )


def check_matrix_axes_share_their_labels(name: str, element_axes: Mapping[str, pd.Index]) -> None:
    """A positive-definite value's rows and columns have one set of labels,
    since it is symmetric."""
    *_, rows, columns = element_axes.values()
    if rows.tolist() != columns.tolist():
        raise ValueError(
            f"{name!r} is symmetric, but its rows are labeled {truncated(rows.tolist())} and its "
            f"columns {truncated(columns.tolist())}; give both axes the same labels."
        )


def check_bijector_is_a_bijector(spec: ArraySpec) -> None:
    """A custom bijector is a TFP bijector."""
    if not isinstance(spec.bijector, tfb.Bijector):
        raise TypeError(
            f"{spec.name!r} has bijector {spec.bijector!r}; give a TFP bijector, or omit it to take the "
            "support's."
        )


def check_bijector_acts_on_the_supports_events(spec: ArraySpec) -> None:
    """A custom bijector's image is decided over the axes its support decides
    membership over: each number alone on an interval, a vector on the
    simplex, a matrix on the positive-definite matrices."""
    if spec.bijector.inverse_min_event_ndims != spec.support.event_ndims:
        raise ValueError(
            f"{spec.name!r}: its bijector maps onto {spec.bijector.inverse_min_event_ndims}-dimensional "
            f"values, but its support {spec.support.name!r} decides membership over "
            f"{spec.support.event_ndims}; give a bijector onto those."
        )


def check_bijector_maps_onto_the_support(spec: ArraySpec) -> None:
    """A custom bijector maps the probe points into the support's closure,
    and the support's default image of them back to finite points that it
    maps there again, since a wrong bijector changes the support silently."""
    if not _maps_onto_the_support(spec):
        raise ValueError(
            f"{spec.name!r}: its bijector {spec.bijector.name!r} does not map theta onto the support "
            f"{spec.support.name!r} at the probe points; give a bijector onto the support, or omit "
            "bijector= to take the support's own."
        )
