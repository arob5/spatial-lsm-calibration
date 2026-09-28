"""The parameter vector: the unknowns of a calibration, the sites they are
defined over, and the coordinates a sampler moves in.

Where this sits
---------------
::

    sites.select_sites, site_labels.load_site_labels    (the site table, the dim labels)
      -> parameter_vector.ParameterVector               (what is calibrated)
      -> prior.Prior                                    (what is believed beforehand)
      -> sipnet_parameter_map, forward                  (how a value reaches SIPNET)

It uses NumPy, pandas, xarray, JAX and TFP's bijectors. A prior over the
vector lives in :mod:`sipnet_calibration.prior`, and the map to SIPNET
parameters in :mod:`sipnet_calibration.sipnet_parameter_map`.

What it reads
-------------
Nothing from disk. A :class:`ParameterVector` is built over a site table the
caller has loaded (:func:`sipnet_calibration.sites.select_sites`) and, for
each site-labels data source a parameter varies over, its site labels
(:func:`sipnet_calibration.site_labels.load_site_labels`).

Data model
----------
The vector holds parameters :math:`x_1, \\dots, x_P`. Parameter :math:`p`
varies over at most one **dim** :math:`d_p`, with :math:`n_p` **dim
labels**: ``"site"``, whose dim labels are the site ids, or a site-labels
name, whose dim labels are the classes some site carries. It has a
:class:`Support` :math:`A_p`, a natural size :math:`k_p` (numbers per dim
label), an unconstrained size :math:`e_p` (the dimension of :math:`A_p`) and
a bijection :math:`T_p : \\mathbb{R}^{e_p} \\to A_p`.

It may also hold **derived parameters** :math:`y_q = f_q(x)`
(:class:`DerivedParameter`): deterministic functions of the parameters and
earlier derived parameters, each with a dim, natural names and units like a
parameter's, but with no entries of theta and no prior.

A value takes three forms:

**theta** (Flat). ``float64``, ``(D,)`` or ``(J, D)``, with
:math:`D = \\sum_p n_p e_p` and :math:`\\theta_p = T_p^{-1}(x_p)`: parameters
in declaration order; within one, dim labels in :meth:`~ParameterVector.dim_index`
order; within a dim label, unconstrained names in order.
:attr:`ParameterVector.index` names every entry.

**Natural values** (:data:`NaturalValues`). ``{name: array}`` for every
parameter and, where computed, every derived parameter, each ``(...,
*value_shape)``, in its units and support:

=========================== ============ ============= =========== ============
                            no dim,      no dim,       dim,        dim,
                            scalar       vector        scalar      vector
=========================== ============ ============= =========== ============
natural value (value shape) ``()``       ``(k,)``      ``(n,)``    ``(n, k)``
unconstrained value         ``()``       ``(e,)``      ``(n,)``    ``(n, e)``
entries of theta            1            ``e``         ``n``       ``n e``
at sites                    ``(S,)``     ``(S, k)``    ``(S,)``    ``(S, k)``
=========================== ============ ============= =========== ============

**The labeled form** (:data:`ParameterDataset`, checked by
:func:`validate_parameter_dataset`), an ``xr.Dataset``:

============ ===============================================================
dims         a batch dim (``sample`` unless ``batch_dim=`` says otherwise)
             when built from ``(J, D)``; ``site`` when a parameter varies
             over it; one dim per site-labels name a parameter varies over
coordinates  the batch dim, ``int64`` ``0`` to ``J - 1``; ``site``, ``int32``
             site ids with ``lon``/``lat``, with the attributes of
             :mod:`sipnet_calibration.conventions`; a site-labels dim, its
             dim labels (strings), with a ``long_name``
variables    one ``float64`` variable per natural name of every parameter and
             derived parameter: ``<name>`` for a scalar,
             ``<name>.<natural name>`` for a vector, on ``(batch?, dim?)``
attributes   ``parameter`` (its name); ``natural_name`` (a vector's);
             ``units`` (omitted when ``None``); ``support`` (a parameter's,
             and a derived parameter's when it declares one);
             ``derived_from`` (a derived parameter's inputs, comma-separated)
missing      never
============ ===============================================================

Its index of a site-labels dim equals :meth:`ParameterVector.dim_index`. It is
not a collection of fields: a site-labels dim is not a field dim.
:meth:`ParameterVector.site_fields` is the per-site view, which is.

Functions and classes
---------------------
:class:`ParameterVector`
    Identity, selection, the coordinate maps (``to_natural``,
    ``derived_values``, ``to_unconstrained``, ``at_sites``) and the labeled
    form (``dataset``, ``flat``, ``site_fields``).
:class:`Parameter`, :class:`DerivedParameter`, :class:`Support`
    One unknown, a quantity computed from them, and the open set a value
    lives in: :data:`REAL`, :data:`POSITIVE`, :data:`OPEN_UNIT_INTERVAL`,
    :data:`SIMPLEX` and :func:`OpenInterval`.
:func:`site_positions`, :func:`dim_label_positions`
    How a value on one dim is read at the sites, or at the dim labels of
    another dim, inside a prior function, a derived parameter or a rule.
:func:`probe_points`, :func:`joint_probe_points`
    The fixed points in theta at which bijectors and priors are probed.
:data:`NaturalValues`, :data:`ParameterDataset`
    The aliases, with :func:`validate_natural_values` and
    :func:`validate_parameter_dataset`.
:func:`check_parameter_vectors_share_a_layout`
    Whether two vectors give theta one meaning.

Notes
-----
**Nothing ships a vector to a worker.** The forward model sends each run its
SIPNET parameter values, never the vector, so a vector holding a custom
bijector need not pickle.

The vector computes in ``float64``: importing the package turns on JAX's
64-bit mode.

Usage
-----
::

    import jax.numpy as jnp
    from sipnet_calibration.parameter_vector import (
        OPEN_UNIT_INTERVAL, POSITIVE, SIMPLEX, Parameter, ParameterVector,
    )

    vector = ParameterVector(
        parameters=[
            Parameter(name="respiration_share", support=OPEN_UNIT_INTERVAL, units="1"),
            Parameter(name="allocation", support=SIMPLEX, units="1", dim="pft",
                      natural_names=("leaf", "wood", "fine_root", "coarse_root")),
            Parameter(name="initial_soil_carbon", support=POSITIVE, units="g m-2",
                      dim="site"),
        ],
        site_table=select_sites(load_sites(), site_ids=[620, 865, 1037]),
        site_labels={"pft": load_site_labels("reanalysis_3pft")},
    )
    vector.dimension                         # 1 + 2 * 3 + 3 = 10 with two PFTs present
    vector.index[:2]                         # (parameter, dim, dim_label, unconstrained_name)
    vector.positions(parameter_name="allocation", dim="pft", dim_label="boreal.coniferous")

    theta = jnp.zeros((4, vector.dimension))
    natural_values = vector.to_natural(theta)       # {"allocation": (4, 2, 4), ...}
    vector.at_sites(natural_values)["allocation"]   # (4, 3, 4)
    parameter_dataset = vector.dataset(theta)       # Dataset on (sample, pft) and (sample, site)
    vector.flat(parameter_dataset)                  # theta again
    vector.site_fields(parameter_dataset)           # fields on (sample, site)
    vector.select(dim_labels={"pft": ["boreal.coniferous"]})
"""

from __future__ import annotations

import keyword
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.conventions import (
    DATA_SOURCE_MEMBER_NAMES,
    LAT,
    LON,
    NAME_PATTERN,
    NON_BATCH_DIM_NAMES,
    SAMPLE,
    SITE,
    SITE_DTYPE,
    SITE_ID,
)
from sipnet_calibration.fields import (
    batch_coordinate,
    batch_dims,
    check_at_most_one_batch_dim,
    check_batch_dim_name_is_not_reserved,
)
from sipnet_calibration.site_labels import LABEL_COLUMN
from sipnet_calibration.sites import (
    check_site_table_has_locations,
    check_site_table_is_keyed_on_site_ids,
    site_coordinates,
    site_lookup,
)
from sipnet_calibration.validation import (
    as_batched_flat,
    as_names,
    as_sequence,
    as_site_ids,
    check_names_are_unique,
    is_one_vector,
    truncated,
)

__all__ = [
    "INDEX_LEVEL_NAMES",
    "NO_DIM",
    "OPEN_UNIT_INTERVAL",
    "POSITIVE",
    "REAL",
    "RESERVED_NAMES",
    "SIMPLEX",
    "DerivedParameter",
    "NaturalValues",
    "OpenInterval",
    "Parameter",
    "ParameterDataset",
    "ParameterVector",
    "Support",
    "bijectors_agree",
    "check_batch_dim_name_is_not_taken",
    "check_parameter_vector_is_valid",
    "check_parameter_vectors_share_a_layout",
    "check_theta_ends_in_the_dimension",
    "dim_label_positions",
    "joint_probe_points",
    "probe_points",
    "site_positions",
    "validate_natural_values",
    "validate_parameter_dataset",
]

tfb = tfp.bijectors

Array = jax.Array

#: The levels of :attr:`ParameterVector.index`, in order.
INDEX_LEVEL_NAMES: tuple[str, ...] = ("parameter", "dim", "dim_label", "unconstrained_name")

#: The ``dim`` and ``dim_label`` of an entry of a parameter without a dim.
NO_DIM = ""

#: Names no parameter, site-labels source or site covariate may take, since a
#: labeled form's variable or dim of that name would collide with a
#: coordinate: ``sample``, the names that are never batch dims, the data
#: sources' member dims and the site table's ``site_id``.
RESERVED_NAMES = frozenset({SAMPLE, *NON_BATCH_DIM_NAMES, *DATA_SOURCE_MEMBER_NAMES, SITE_ID})


# ── the vector ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class ParameterVector:
    """The unknowns of a calibration, over a set of sites.

    Parameters
    ----------
    parameters:
        The :class:`Parameter`\\ s, in the order they occupy theta.
    derived_parameters:
        The :class:`DerivedParameter`\\ s, each computed from parameters and
        earlier derived parameters.
    site_table:
        The sites, one row each in ascending ``site_id``, with ``lon`` and
        ``lat``, such as :func:`sipnet_calibration.sites.select_sites`
        returns. The vector keeps its own copy of what it reads
        (:attr:`site_table`).
    site_labels:
        ``{site_labels_name: site labels}`` for every site-labels name a
        parameter varies over, and any other a prior function or SIPNET rule
        reads: a site-labels table (``site_id`` and ``label``) such as
        :func:`sipnet_calibration.site_labels.load_site_labels` returns,
        which labels every site; a pandas categorical; or one label per site,
        in site order. Labels are strings. The dim labels are the classes
        some site carries, in the categories' order, or sorted for plain
        labels.
    site_covariate_names:
        The columns of *site_table* a prior function, derived parameter or
        SIPNET rule may read, each ``float64`` and finite. No other column is
        kept.

    Raises
    ------
    TypeError
        If a site label, or a named site covariate, has the wrong type.
    KeyError
        If a named site covariate is not a column of *site_table*, or a
        derived parameter is computed from a name the vector lacks.
    ValueError
        If the parameters, derived parameters, sites and site labels do not
        make one vector; the message names the rule.
    """

    parameters: Sequence[Parameter]
    derived_parameters: Sequence[DerivedParameter] = ()
    # A keyword only: the attribute of the same name is the property attached
    # below the class, so the dataclass must not read it as a default.
    site_table: InitVar[pd.DataFrame]
    site_labels: Mapping[str, Any] = field(default_factory=frozendict)
    site_covariate_names: Sequence[str] = ()
    _table: pd.DataFrame = field(init=False, repr=False)
    _declared_dim_labels: Mapping[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self, site_table: pd.DataFrame) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "derived_parameters", tuple(self.derived_parameters))
        object.__setattr__(
            self,
            "site_covariate_names",
            as_names(self.site_covariate_names, message_name="site_covariate_names"),
        )
        table = _normalized_site_table(site_table, self.site_covariate_names)
        declared = {}
        for name, value in dict(self.site_labels).items():
            table[name], declared[name] = _site_labels_column(name, value, table[SITE_ID])
        object.__setattr__(
            self,
            "site_labels",
            frozendict({name: tuple(table[name].astype(str)) for name in declared}),
        )
        object.__setattr__(self, "_table", table)
        object.__setattr__(self, "_declared_dim_labels", frozendict(declared))
        check_parameter_vector_is_valid(self)

    # ── identity ──────────────────────────────────────────────────────────────

    def __getitem__(self, name: str) -> Parameter:
        """The parameter called *name*; ``KeyError`` for any other."""
        check_parameter_names_are_held([name], self)
        return self.parameters[self.parameter_names.index(name)]

    def __contains__(self, name: object) -> bool:
        return name in self.parameter_names

    def __iter__(self) -> Iterator[str]:
        return iter(self.parameter_names)

    def __reversed__(self) -> Iterator[str]:
        return reversed(self.parameter_names)

    def __len__(self) -> int:
        return len(self.parameters)

    def __repr__(self) -> str:
        derived = (
            f", derived_parameters={list(self.derived_parameter_names)}"
            if self.derived_parameters
            else ""
        )
        return (
            f"ParameterVector(D={self.dimension}, parameters={list(self.parameter_names)}"
            f"{derived}, sites={self.n_sites})"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The parameter names, in declaration order."""
        return tuple(p.name for p in self.parameters)

    @property
    def derived_parameter_names(self) -> tuple[str, ...]:
        """The derived parameter names, in declaration order."""
        return tuple(d.name for d in self.derived_parameters)

    @property
    def dimension(self) -> int:
        """``D``, the number of entries of theta."""
        return sum(self._sizes.values())

    @property
    def sites(self) -> tuple[int, ...]:
        """The site ids, ascending."""
        return tuple(int(s) for s in self._table[SITE_ID])

    @property
    def n_sites(self) -> int:
        """``S``, the number of sites."""
        return len(self._table)

    @property
    def dim_names(self) -> tuple[str, ...]:
        """The dims the parameters and derived parameters vary over, in the
        order they first appear."""
        pieces = (*self.parameters, *self.derived_parameters)
        return tuple(dict.fromkeys(p.dim for p in pieces if p.dim is not None))

    def dim_index(self, dim_name: str) -> pd.Index:
        """The dim labels of ``"site"`` (the site ids) or of a site-labels name.

        Raises
        ------
        KeyError
            For a name that is neither ``"site"`` nor one of the vector's
            site-labels names.
        """
        if dim_name == SITE:
            return pd.Index(self._table[SITE_ID].to_numpy(SITE_DTYPE), name=SITE)
        check_dim_is_the_vectors(dim_name, self)
        return pd.Index(list(self._table[dim_name].cat.categories), name=dim_name)

    def value_shape(self, name: str) -> tuple[int, ...]:
        """The shape of one natural value of a parameter or derived
        parameter: ``(n?, k?)``."""
        piece = self._piece(name)
        return self._dim_shape(piece) + piece.natural_shape

    def unconstrained_shape(self, parameter_name: str) -> tuple[int, ...]:
        """The shape of one unconstrained value of a parameter: ``(n?, e?)``."""
        parameter = self[parameter_name]
        return self._dim_shape(parameter) + parameter.unconstrained_shape

    @cached_property
    def index(self) -> pd.MultiIndex:
        """One row per entry of theta, levels :data:`INDEX_LEVEL_NAMES`.

        A parameter without a dim has ``dim`` and ``dim_label`` both
        :data:`NO_DIM`. A site's dim label is its integer site id.
        """
        rows = [
            (p.name, p.dim or NO_DIM, label, name)
            for p in self.parameters
            for label in (self.dim_index(p.dim) if p.dim else (NO_DIM,))
            for name in p.unconstrained_names
        ]
        return pd.MultiIndex.from_tuples(rows, names=INDEX_LEVEL_NAMES)

    @cached_property
    def entry_names(self) -> tuple[str, ...]:
        """``D`` strings: ``name``, ``name[dim label]``, ``name[unconstrained
        name]`` or ``name[dim label][unconstrained name]``, as the parameter
        needs."""
        out = []
        for name, dim, label, unconstrained_name in self.index:
            entry = name
            if dim != NO_DIM:
                entry += f"[{label}]"
            if self[name].natural_names is not None:
                entry += f"[{unconstrained_name}]"
            out.append(entry)
        return tuple(out)

    def positions(
        self,
        *,
        parameter_name: str | None = None,
        dim: str | None = None,
        dim_label: Any = None,
        unconstrained_name: str | None = None,
    ) -> np.ndarray:
        """Where in theta the entries matching every selector given sit.

        Parameters
        ----------
        parameter_name, dim, unconstrained_name:
            Values of the index levels of those names.
        dim_label:
            A dim label, compared with the dim's own labels, so a site is
            ``620``, never ``"620"``. Requires *dim* or *parameter_name*,
            since labels of two dims may coincide.

        Returns
        -------
        numpy.ndarray
            ``int64`` positions, ascending; possibly none.

        Raises
        ------
        ValueError
            If *dim_label* is given without *dim* or *parameter_name*.
        KeyError
            For a selector no entry has, :data:`NO_DIM` included.
        """
        check_dim_label_has_a_dim(dim_label, dim, parameter_name)
        check_selector_is_not_no_dim(dim, dim_label)
        mask = np.ones(self.dimension, dtype=bool)
        if parameter_name is not None:
            check_parameter_names_are_held([parameter_name], self)
            mask &= self.index.get_level_values("parameter") == parameter_name
        for level, value in (("dim", dim), ("unconstrained_name", unconstrained_name)):
            if value is not None:
                values = self.index.get_level_values(level)
                check_selector_is_in_the_index(level, value, values)
                mask &= values == value
        if dim_label is not None:
            label_dim = dim if dim is not None else self[parameter_name].dim
            labels = list(self.dim_index(label_dim)) if label_dim is not None else []
            check_dim_label_is_the_dims(dim_label, labels, label_dim)
            mask &= np.asarray(
                [_same_label(label, dim_label) for label in self.index.get_level_values("dim_label")]
            )
        return np.flatnonzero(mask).astype(np.int64)

    def describe(self) -> pd.DataFrame:
        """One row per parameter, then per derived parameter, indexed by
        ``parameter``: ``dim``, ``dim_labels`` (their number), ``support``,
        ``bijector``, ``units``, ``natural_size``, ``unconstrained_size``,
        ``entries`` (of theta) and ``derived_from``. A derived parameter has
        no bijector and no entries, and its support is the one it declares,
        if any."""
        rows = [
            {
                "parameter": p.name,
                "dim": p.dim or NO_DIM,
                "dim_labels": self._dim_shape(p)[0] if p.dim else 1,
                "support": p.support.name,
                "bijector": p.bijector.name,
                "units": p.units,
                "natural_size": p.natural_size,
                "unconstrained_size": p.unconstrained_size,
                "entries": self._sizes[p.name],
                "derived_from": "",
            }
            for p in self.parameters
        ]
        rows += [
            {
                "parameter": d.name,
                "dim": d.dim or NO_DIM,
                "dim_labels": self._dim_shape(d)[0] if d.dim else 1,
                "support": "" if d.support is None else d.support.name,
                "bijector": "",
                "units": d.units,
                "natural_size": d.natural_size,
                "unconstrained_size": 0,
                "entries": 0,
                "derived_from": ", ".join(d.derived_from),
            }
            for d in self.derived_parameters
        ]
        return pd.DataFrame(rows).set_index("parameter")

    # ── selection ─────────────────────────────────────────────────────────────

    def select(
        self,
        *,
        parameter_names: Sequence[str] | None = None,
        sites: Sequence[int] | None = None,
        dim_labels: Mapping[str, Sequence[str]] | None = None,
    ) -> ParameterVector:
        """A smaller vector: some parameters, over some sites.

        Parameters
        ----------
        parameter_names:
            The parameters to keep; they keep this vector's order.
        sites:
            The site ids to keep.
        dim_labels:
            ``{site_labels_name: [dim label, ...]}``: keep the sites carrying
            one of those dim labels, intersected with *sites*.

        Returns
        -------
        ParameterVector
            Over the kept sites, whose dim indexes are recomputed from them;
            the site covariates and site labels are carried, and so is every
            derived parameter whose inputs are all kept.

        Raises
        ------
        TypeError
            If an argument is one value rather than a sequence.
        KeyError
            For an unknown parameter, site, site-labels name or dim label.
        ValueError
            If a name or site is given twice, or no site remains.
        """
        names = self.parameter_names
        if parameter_names is not None:
            wanted = as_names(parameter_names, message_name="parameter_names")
            check_names_are_unique(wanted, message_name="parameter_names")
            check_parameter_names_are_held(wanted, self)
            names = tuple(n for n in names if n in wanted)
        kept = self._selected_site_mask(sites, dim_labels)
        table = self._table[kept]
        site_labels = {
            name: pd.Categorical(table[name].astype(str), categories=self._declared_dim_labels[name])
            for name in self._declared_dim_labels
        }
        available, derived = set(names), []
        for d in self.derived_parameters:
            if available.issuperset(d.derived_from):
                derived.append(d)
                available.add(d.name)
        return ParameterVector(
            parameters=[self[n] for n in names],
            derived_parameters=derived,
            site_table=table[[SITE_ID, LON, LAT, *self.site_covariate_names]],
            site_labels=site_labels,
            site_covariate_names=self.site_covariate_names,
        )

    # ── coordinates ───────────────────────────────────────────────────────────

    def to_natural(self, theta: Any) -> NaturalValues:
        """Theta to natural values: :math:`x_p = T_p(\\theta_p)` for every
        parameter, then :math:`y_q = f_q(x)` for every derived parameter.

        :math:`\\theta_p` is theta at ``positions(parameter_name=p)``, shaped
        ``(..., *unconstrained_shape(p))``, and :math:`T_p` is
        ``parameter.bijector``, applied per dim label; the derived
        parameters are :meth:`derived_values`. Traceable under ``jax.jit``,
        ``jax.grad`` and ``jax.vmap``.

        Parameters
        ----------
        theta:
            ``(..., D)``.

        Returns
        -------
        NaturalValues
            ``{name: (..., *value_shape(name))}``, ``float64``, parameters
            first.

        Raises
        ------
        ValueError
            If the last axis of *theta* is not ``D`` long.
        """
        theta = jnp.asarray(theta, dtype=jnp.float64)
        check_theta_ends_in_the_dimension(theta.shape, self.dimension)
        natural_values = self._parameters_at(theta)
        return natural_values | self.derived_values(natural_values)

    def derived_values(
        self, natural_values: NaturalValues, *, derived_parameter_names: Sequence[str] | None = None
    ) -> NaturalValues:
        """The derived parameters computed from natural values:
        :math:`y_q = f_q(x, y_{<q})`, one draw at a time.

        Each ``compute`` is called as ``compute(dim_index, site_table,
        **inputs)``, with the index of the derived parameter's dim (``None``
        without one), the vector's site table, and one draw's value of each
        name it is derived from, and it is ``jax.vmap``-ed over the leading
        shape. Traceable under ``jax.jit``, ``jax.grad`` and ``jax.vmap``.

        Parameters
        ----------
        natural_values:
            ``{name: (..., *value_shape(name))}`` holding at least what the
            requested derived parameters are computed from.
        derived_parameter_names:
            The derived parameters wanted; those they are computed from are
            computed too. ``None``: every one.

        Returns
        -------
        NaturalValues
            ``{name: (..., *value_shape(name))}``, ``float64``, for the
            wanted derived parameters and those they need, in declaration
            order.

        Raises
        ------
        KeyError
            If a name is not a derived parameter of the vector, or an input
            a derived parameter needs is missing from *natural_values*.
        """
        wanted = self._derived_closure(derived_parameter_names)
        values = dict(natural_values)
        out = {}
        for derived in self.derived_parameters:
            if derived.name in wanted:
                check_natural_values_hold_the_inputs(derived, values)
                out[derived.name] = values[derived.name] = self._computed(derived, values)
        return out

    def to_unconstrained(self, natural_values: NaturalValues) -> Array:
        """Natural values to theta: :math:`\\theta_p = T_p^{-1}(x_p)`, flattened
        in theta's order.

        Parameters
        ----------
        natural_values:
            :data:`NaturalValues` with one leading shape; keys that are not
            parameters are ignored.

        Returns
        -------
        jax.Array
            ``(..., D)``. An entry outside its support maps to a non-finite
            value, which is not an error.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :func:`validate_natural_values`.
        """
        validate_natural_values(natural_values, self)
        pieces = []
        for p in self.parameters:
            values = jnp.asarray(natural_values[p.name], dtype=jnp.float64)
            lead = values.shape[: values.ndim - len(self.value_shape(p.name))]
            pieces.append(p.bijector.inverse(values).reshape(lead + (self._sizes[p.name],)))
        return jnp.concatenate(pieces, axis=-1)

    def at_sites(self, natural_values: NaturalValues) -> NaturalValues:
        """Every value read at every site: :math:`x_p^{(s)} = x_p[\\ell_{d_p}(s)]`,
        with :math:`\\ell_{d_p}(s)` site :math:`s`'s dim label along the
        parameter's dim :math:`d_p`, or :math:`x_p` for a parameter without a
        dim.

        Parameters
        ----------
        natural_values:
            :data:`NaturalValues` with one leading shape; derived parameters
            are read at the sites too where present.

        Returns
        -------
        NaturalValues
            The same keys, each ``(..., S)`` or ``(..., S, k)``.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :func:`validate_natural_values`.
        """
        validate_natural_values(natural_values, self)
        pieces = (*self.parameters, *(d for d in self.derived_parameters if d.name in natural_values))
        return {p.name: self._at_sites(p, natural_values[p.name]) for p in pieces}

    # ── representations ───────────────────────────────────────────────────────

    def dataset(self, theta: Any, *, batch_dim: str = SAMPLE) -> ParameterDataset:
        """Theta to the labeled form.

        Parameters
        ----------
        theta:
            ``(D,)``, or ``(J, D)``, whose rows are labeled ``0`` to ``J - 1``
            on *batch_dim*.
        batch_dim:
            The batch dim's name.

        Returns
        -------
        ParameterDataset
            As the module's data model describes it.

        Raises
        ------
        TypeError
            If *theta* is not an array of real numbers, or *batch_dim* not a
            string.
        ValueError
            If *theta* is not ``(D,)`` or ``(J, D)``, or *batch_dim* is a
            reserved name or one of the vector's.
        """
        check_batch_dim_name_is_not_taken(self, batch_dim)
        batched = jnp.asarray(as_batched_flat(theta, self.dimension, message_name="theta"))
        one = is_one_vector(theta)
        natural_values = self.to_natural(batched[0] if one else batched)
        variables = {}
        for p in (*self.parameters, *self.derived_parameters):
            dims = ((batch_dim,) if not one else ()) + ((p.dim,) if p.dim else ())
            values = np.asarray(natural_values[p.name], dtype=np.float64)
            for i, (variable_name, attributes) in enumerate(_dataset_variables(p)):
                variable_values = values[..., i] if p.natural_names is not None else values
                variables[variable_name] = (dims, variable_values, attributes)
        return xr.Dataset(variables, coords=self._coordinates(None if one else len(batched), batch_dim))

    def flat(self, parameter_dataset: ParameterDataset) -> Array:
        """The labeled form to theta: the inverse of :meth:`dataset`.

        Parameters
        ----------
        parameter_dataset:
            A :data:`ParameterDataset` holding a variable for every natural
            name of every parameter, with every dim label of this vector:
            a larger vector's projects onto this one, unless this vector has
            a derived parameter that is not pointwise, whose dataset must
            then have exactly this vector's dim labels. Derived parameters'
            variables are not read. Its rows keep their order.

        Returns
        -------
        jax.Array
            ``(J, D)`` when it has a batch dim, ``(D,)`` otherwise.

        Raises
        ------
        TypeError, ValueError
            As :func:`validate_parameter_dataset`; ``ValueError`` too for a
            variable on other dims than its parameter, a value outside its
            support, or a projection a derived parameter that is not
            pointwise refuses.
        KeyError
            For a missing variable or dim label.
        """
        validate_parameter_dataset(parameter_dataset)
        check_dataset_can_project_onto_the_vector(parameter_dataset, self)
        batch = batch_dims(parameter_dataset)
        pieces = []
        for p in self.parameters:
            values = jnp.asarray(self._natural_values_of(p, parameter_dataset, batch))
            check_values_are_in_the_support(p, values)
            pieces.append(p.bijector.inverse(values).reshape(values.shape[: len(batch)] + (self._sizes[p.name],)))
        return jnp.concatenate(pieces, axis=-1)

    def site_fields(self, parameter_dataset: ParameterDataset) -> xr.Dataset:
        """The labeled form read at every site: one field per variable, on
        ``(batch?, site)``, carrying the variable's attributes, with ``units``
        ``"1"`` for a parameter without physical units. Not invertible.

        Derived parameters are read from their variables like parameters.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`flat`.
        """
        validate_parameter_dataset(parameter_dataset)
        check_dataset_can_project_onto_the_vector(parameter_dataset, self)
        batch = batch_dims(parameter_dataset)
        coordinates = site_coordinates(self.sites, self._table)
        if batch:
            coordinates[batch[0]] = parameter_dataset[batch[0]]
        variables = {}
        for p in (*self.parameters, *self.derived_parameters):
            values = self._natural_values_of(p, parameter_dataset, batch)
            if p.support is not None:
                check_values_are_in_the_support(p, jnp.asarray(values))
            on_sites = np.asarray(self._at_sites(p, values))
            for i, (variable_name, _) in enumerate(_dataset_variables(p)):
                variable_values = on_sites[..., i] if p.natural_names is not None else on_sites
                attributes = {"units": "1", **parameter_dataset[variable_name].attrs}
                variables[variable_name] = ((*batch, SITE), variable_values, attributes)
        return xr.Dataset(variables, coords=coordinates)

    # ── supporting methods ────────────────────────────────────────────────────

    @cached_property
    def _sizes(self) -> Mapping[str, int]:
        return frozendict(
            {p.name: int(np.prod(self.unconstrained_shape(p.name), dtype=int)) for p in self.parameters}
        )

    @cached_property
    def _slices(self) -> Mapping[str, slice]:
        out, start = {}, 0
        for p in self.parameters:
            out[p.name] = slice(start, start + self._sizes[p.name])
            start += self._sizes[p.name]
        return frozendict(out)

    def _dim_shape(self, parameter: Parameter | DerivedParameter) -> tuple[int, ...]:
        return (len(self.dim_index(parameter.dim)),) if parameter.dim else ()

    def _parameters_at(self, theta: Any) -> dict[str, Array]:
        """The parameters' natural values at theta, ``(..., D)``."""
        theta = jnp.asarray(theta, dtype=jnp.float64)
        lead = theta.shape[:-1]
        return {
            p.name: p.bijector.forward(
                theta[..., self._slices[p.name]].reshape(lead + self.unconstrained_shape(p.name))
            )
            for p in self.parameters
        }

    def _piece(self, name: str) -> Parameter | DerivedParameter:
        """The parameter or derived parameter called *name*."""
        if name in self.derived_parameter_names:
            return self.derived_parameters[self.derived_parameter_names.index(name)]
        return self[name]

    def _derived_closure(self, names: Sequence[str] | None) -> set[str]:
        """The derived parameters *names* need, themselves included; every
        one for ``None``."""
        if names is None:
            return set(self.derived_parameter_names)
        wanted = set(as_names(names, message_name="derived_parameter_names"))
        check_derived_parameter_names_are_held(wanted, self)
        for derived in reversed(self.derived_parameters):
            if derived.name in wanted:
                wanted.update(n for n in derived.derived_from if n in self.derived_parameter_names)
        return wanted

    def _computed(self, derived: DerivedParameter, values: Mapping[str, Any]) -> Array:
        """One derived parameter from its inputs' values, ``(..., *shape)``,
        its ``compute`` vmapped over the leading shape."""
        inputs = {name: jnp.asarray(values[name], dtype=jnp.float64) for name in derived.derived_from}
        first = derived.derived_from[0]
        lead = inputs[first].shape[: inputs[first].ndim - len(self.value_shape(first))]
        dim_index = self.dim_index(derived.dim) if derived.dim else None
        site_table = self._table.copy()

        def one_draw(draw: Mapping[str, Array]) -> Array:
            return jnp.asarray(derived.compute(dim_index, site_table, **draw), dtype=jnp.float64)

        if not lead:
            return one_draw(inputs)
        flat = {name: value.reshape((-1, *value.shape[len(lead):])) for name, value in inputs.items()}
        out = jax.vmap(one_draw)(flat)
        return out.reshape(lead + out.shape[1:])

    def _at_sites(self, parameter: Parameter | DerivedParameter, values: Any) -> Array:
        """One natural value, ``(..., *value_shape)``, at every site."""
        values = jnp.asarray(values)
        lead_ndim = values.ndim - len(self.value_shape(parameter.name))
        if parameter.dim is not None:
            return jnp.take(values, site_positions(self._table, parameter.dim), axis=lead_ndim)
        values = jnp.expand_dims(values, lead_ndim)
        shape = values.shape[:lead_ndim] + (self.n_sites,) + parameter.natural_shape
        return jnp.broadcast_to(values, shape)

    def _selected_site_mask(
        self, sites: Sequence[int] | None, dim_labels: Mapping[str, Sequence[str]] | None
    ) -> np.ndarray:
        kept = np.ones(self.n_sites, dtype=bool)
        if sites is not None:
            requested = as_site_ids(sites, message_name="sites")
            check_sites_are_held(requested, self)
            kept &= self._table[SITE_ID].isin(requested).to_numpy()
        for name, labels in (dim_labels or {}).items():
            check_dim_labels_select_a_site_labels_dim(name, self)
            wanted = as_sequence(labels, message_name=f"dim_labels[{name!r}]")
            check_names_are_unique(wanted, message_name=f"dim_labels[{name!r}]")
            for label in wanted:
                check_dim_label_is_the_dims(label, self._declared_dim_labels[name], name)
            kept &= self._table[name].astype(str).isin(wanted).to_numpy()
        check_selection_keeps_a_site(kept)
        return kept

    def _coordinates(self, n_samples: int | None, batch_dim: str) -> dict[str, Any]:
        """The coordinates of the labeled form: the batch dim when *n_samples*
        is given, and the dims the parameters vary over."""
        coordinates: dict[str, Any] = {}
        if n_samples is not None:
            coordinates[batch_dim] = batch_coordinate(batch_dim, np.arange(n_samples))
        for dim_name in self.dim_names:
            if dim_name == SITE:
                coordinates.update(site_coordinates(self.sites, self._table))
            else:
                labels = np.asarray(self.dim_index(dim_name), dtype=str)
                coordinates[dim_name] = xr.DataArray(
                    labels,
                    dims=dim_name,
                    attrs={"long_name": f"dim labels of the site labels {dim_name!r}"},
                )
        return coordinates

    def _natural_values_of(
        self, parameter: Parameter, parameter_dataset: xr.Dataset, batch: tuple[str, ...]
    ) -> np.ndarray:
        """A parameter's natural values from the labeled form:
        ``(batch?, n?, k?)``, at this vector's dim labels."""
        dims = (*batch, parameter.dim) if parameter.dim else batch
        columns = []
        for variable_name, _ in _dataset_variables(parameter):
            variable = parameter_dataset[variable_name]
            check_variable_is_on_the_parameters_dims(variable, variable_name, dims)
            if parameter.dim:
                check_dataset_dim_is_labeled(variable, variable_name, parameter.dim)
                variable = variable.sel({parameter.dim: list(self.dim_index(parameter.dim))})
            values = np.asarray(variable.transpose(*dims).values, dtype=np.float64)
            check_parameter_dataset_variable_is_finite(variable_name, values)
            columns.append(values)
        return np.stack(columns, axis=-1) if parameter.natural_names is not None else columns[0]


# ``site_table`` is both a keyword of the constructor (an ``InitVar``) and this
# read-only attribute, attached after the dataclass is made so that the
# dataclass does not take the property for the keyword's default.
def _site_table(self: ParameterVector) -> pd.DataFrame:
    """A copy of the vector's site table: ``site_id`` (``int32``),
    ``lon``, ``lat``, the site covariates (``float64``), and one categorical
    column per site-labels name, whose categories are its dim labels."""
    return self._table.copy()


ParameterVector.site_table = property(_site_table)


# ── its pieces ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, kw_only=True)
class Parameter:
    """One unknown of the calibration.

    Parameters
    ----------
    name:
        ``lower_case_with_underscores``, not a Python keyword, and none of
        :data:`RESERVED_NAMES`.
    support:
        The open set a natural value lies in, per dim label.
    units:
        UDUNITS units of the natural value, or ``None`` for a quantity
        without physical units. Required.
    dim:
        ``"site"``, a site-labels name of the vector, or ``None``.
    natural_names:
        The names of one dim label's ``k`` natural numbers, or ``None`` for a
        scalar. :data:`SIMPLEX` needs at least two.
    bijector:
        :math:`T_p : \\mathbb{R}^e \\to A_p`; ``None`` takes the support's
        default, :meth:`Support.bijector`. A custom bijector must map onto
        the support, which is checked at probe points in both directions.

    Attributes
    ----------
    bijector : tfb.Bijector
        :math:`T_p`, never ``None`` once built.
    natural_size, unconstrained_size : int
        ``k`` and ``e``.
    natural_shape, unconstrained_shape : tuple
        ``()`` for a scalar, ``(k,)`` and ``(e,)`` for a vector.
    unconstrained_names : tuple of str
        The names of one dim label's ``e`` unconstrained numbers:
        :meth:`Support.unconstrained_names`, or ``<bijector name>_inverse(x)``
        under a custom bijector.

    Raises
    ------
    ValueError
        For a malformed name or natural names, or a custom bijector that does
        not map onto the support.
    """

    name: str
    support: Support
    units: str | None
    dim: str | None = None
    natural_names: tuple[str, ...] | None = None
    bijector: tfb.Bijector | None = None
    _custom_bijector: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if self.natural_names is not None:
            object.__setattr__(
                self,
                "natural_names",
                as_names(self.natural_names, message_name=f"{self.name!r} natural_names"),
            )
        object.__setattr__(self, "_custom_bijector", self.bijector is not None)
        if self.bijector is None:
            object.__setattr__(self, "bijector", self.support.bijector())
        check_parameter_is_valid(self)

    @property
    def natural_size(self) -> int:
        return 1 if self.natural_names is None else len(self.natural_names)

    @property
    def unconstrained_size(self) -> int:
        return self.support.unconstrained_size(self.natural_size)

    @property
    def natural_shape(self) -> tuple[int, ...]:
        return () if self.natural_names is None else (self.natural_size,)

    @property
    def unconstrained_shape(self) -> tuple[int, ...]:
        return () if self.natural_names is None else (self.unconstrained_size,)

    @property
    def unconstrained_names(self) -> tuple[str, ...]:
        natural_names = (self.name,) if self.natural_names is None else self.natural_names
        if not self._custom_bijector:
            return self.support.unconstrained_names(natural_names)
        return tuple(
            f"{self.bijector.name}_inverse({x})" for x in natural_names[: self.unconstrained_size]
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class DerivedParameter:
    """A deterministic function of the parameters: :math:`y = f(x)`.

    It has no entries of theta and no prior; its distribution is the
    pushforward :math:`f_\\# \\pi` of the prior on :math:`x`. It is computed
    by :meth:`ParameterVector.to_natural` after the parameters, carried by
    the labeled form, and read by SIPNET rules by name.

    Parameters
    ----------
    name:
        As :class:`Parameter`'s, and distinct from every parameter's.
    units:
        UDUNITS units of the value, or ``None``. Required.
    dim:
        ``"site"``, a site-labels name of the vector, or ``None``.
    natural_names:
        The names of one dim label's ``k`` numbers, or ``None`` for a scalar.
    derived_from:
        The parameters and earlier derived parameters it is computed from.
    compute:
        ``compute(dim_index, site_table, **inputs)``: one draw's value, of
        shape ``(n?, k?)``, from ``dim_index`` (the index of *dim*, or
        ``None``), the vector's site table, and one draw's natural value of
        each name in *derived_from*, passed by name. It must be traceable
        by JAX. :func:`site_positions` reads an input on another dim at this
        one's sites.
    pointwise:
        Whether the value at each dim label depends only on the inputs at
        that dim label, at its dim label along another dim, and on inputs
        without a dim. A non-centered Gaussian process,
        :math:`x = \\exp(m + L(\\ell) z)`, is not: every :math:`x_s` depends
        on every :math:`z_{s'}`.
    support:
        An open set every value lies in, or ``None``. Declaring one lets a
        SIPNET rule's bounds be checked before anything runs.

    Raises
    ------
    ValueError
        For a malformed name or natural names, or no *derived_from*.

    Notes
    -----
    The vector checks, at construction, that ``compute`` gives the value
    shape at :math:`\\theta = 0`; that its values at the probe points lie in
    the declared support, an infinite value at an unbounded end being
    float64 overflow and not a violation; and, when *pointwise*, that
    changing each input on the same dim at one dim label changes the value
    there only. These are probes, not proofs.

    A derived parameter that is not pointwise is recomputed from the kept
    dim labels alone after :meth:`ParameterVector.select`, so the same
    theta gives other values at the same sites: the prior is still the
    right marginal, but draws cannot move between the two vectors, and
    :meth:`ParameterVector.flat` refuses a dataset over other dim labels.

    Standardize a site covariate once, before the vector is built, never
    inside ``compute``: the site table after ``select`` has fewer rows, and
    a standardization over them would change what the parameters mean.
    """

    name: str
    units: str | None
    dim: str | None = None
    natural_names: tuple[str, ...] | None = None
    derived_from: tuple[str, ...]
    compute: Callable[..., Array]
    pointwise: bool = True
    support: Support | None = None

    def __post_init__(self) -> None:
        if self.natural_names is not None:
            object.__setattr__(
                self,
                "natural_names",
                as_names(self.natural_names, message_name=f"{self.name!r} natural_names"),
            )
        object.__setattr__(
            self, "derived_from", as_names(self.derived_from, message_name=f"{self.name!r} derived_from")
        )
        check_derived_parameter_is_valid(self)

    @property
    def natural_size(self) -> int:
        return 1 if self.natural_names is None else len(self.natural_names)

    @property
    def natural_shape(self) -> tuple[int, ...]:
        return () if self.natural_names is None else (self.natural_size,)


@dataclass(frozen=True)
class Support:
    """An open set a natural value lies in, and its default bijection from
    :math:`\\mathbb{R}^e`.

    ========================= =========================================== ========= ==========================
    support                   default :math:`T_A`                         :math:`e` unconstrained names
    ========================= =========================================== ========= ==========================
    ``real``                  identity                                    :math:`k` ``x``
    ``positive``              :math:`\\exp`                                :math:`k` ``log(x)``
    ``interval`` (0, 1)       :math:`\\sigma(t) = 1 / (1 + e^{-t})`        :math:`k` ``logit(x)``
    ``interval`` (a, b)       :math:`a + (b - a)\\,\\sigma(t)`               :math:`k` ``logit(x in (a, b))``
    ``simplex``               :math:`x_i = e^{t_i} / (1 + \\sum_j e^{t_j})`, :math:`k-1` ``alr(x:last)``
                              :math:`x_k = 1 / (1 + \\sum_j e^{t_j})`
    ========================= =========================================== ========= ==========================

    The simplex's default is TFP's ``SoftmaxCentered``, the inverse of the
    additive log-ratio :math:`t_i = \\log(x_i / x_k)`. Every support is open:
    it is the image of a bijection from :math:`\\mathbb{R}^e`.

    Parameters
    ----------
    kind:
        ``"real"``, ``"positive"``, ``"interval"`` or ``"simplex"``.
    low, high:
        The ends of an interval, finite, ``low < high``; ``None`` otherwise.

    Raises
    ------
    ValueError
        For an unknown kind, or ends that do not fit it.
    """

    kind: Literal["real", "positive", "interval", "simplex"]
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        check_support_is_valid(self)

    @property
    def name(self) -> str:
        """``real``, ``positive``, ``(a, b)`` or ``simplex``."""
        if self.kind == "interval":
            return f"({self.low:g}, {self.high:g})"
        return self.kind

    def bijector(self) -> tfb.Bijector:
        """The default :math:`T_A` of the table above."""
        if self.kind == "real":
            return tfb.Identity()
        if self.kind == "positive":
            return tfb.Exp()
        if self.kind == "simplex":
            return tfb.SoftmaxCentered()
        if (self.low, self.high) == (0.0, 1.0):
            return tfb.Sigmoid()
        return tfb.Sigmoid(low=jnp.float64(self.low), high=jnp.float64(self.high))

    def unconstrained_size(self, natural_size: int) -> int:
        """``e``: ``k - 1`` for the simplex, ``k`` otherwise."""
        return natural_size - 1 if self.kind == "simplex" else natural_size

    def unconstrained_names(self, natural_names: tuple[str, ...]) -> tuple[str, ...]:
        """The names of the unconstrained numbers, as the table above has them."""
        if self.kind == "simplex":
            return tuple(f"alr({x}:{natural_names[-1]})" for x in natural_names[:-1])
        if self.kind == "positive":
            return tuple(f"log({x})" for x in natural_names)
        if self.kind == "interval" and (self.low, self.high) == (0.0, 1.0):
            return tuple(f"logit({x})" for x in natural_names)
        if self.kind == "interval":
            return tuple(f"logit({x} in {self.name})" for x in natural_names)
        return tuple(natural_names)

    def contains(self, values: Any, *, closure: bool = False) -> Array:
        """Whether each value lies in the support, finite values only; in its
        closure when *closure*.

        Elementwise for ``real``, ``positive`` and ``interval``; over the
        last axis for the simplex, ``(..., k) -> (...)``, whose values must
        be positive (non-negative in the closure) and sum to 1 within
        ``1e-10``.
        """
        values = jnp.asarray(values, dtype=jnp.float64)
        finite = jnp.isfinite(values)
        above = jnp.greater_equal if closure else jnp.greater
        below = jnp.less_equal if closure else jnp.less
        if self.kind == "real":
            return finite
        if self.kind == "positive":
            return finite & above(values, 0.0)
        if self.kind == "interval":
            return finite & above(values, self.low) & below(values, self.high)
        inside = jnp.all(finite & above(values, 0.0), axis=-1)
        return inside & (jnp.abs(values.sum(axis=-1) - 1.0) <= _SIMPLEX_SUM_TOLERANCE)

    def log_jacobian(self, transform: tfb.Bijector, theta: Any) -> Array:
        """:math:`\\log J(\\theta)`, the log absolute Jacobian determinant of
        *transform* against this support's reference measure.

        For an elementwise support the measure is Lebesgue measure and
        :math:`\\log J = \\log |T'(\\theta)|`, elementwise, the shape of
        *theta*. For the simplex it is Lebesgue measure on the first
        :math:`k - 1` coordinates, which is Dirichlet's convention, and

        .. math::

            \\log J(\\theta) = \\log \\left| \\det
                \\frac{\\partial (x_1, \\dots, x_{k-1})}{\\partial \\theta} \\right|,
            \\qquad x = T(\\theta),

        computed by automatic differentiation over the last axis,
        ``(..., k - 1) -> (...)``, whatever the bijector.
        """
        theta = jnp.asarray(theta, dtype=jnp.float64)
        if self.kind != "simplex":
            return transform.forward_log_det_jacobian(theta, event_ndims=0)

        def log_determinant(t: Array) -> Array:
            jacobian = jax.jacfwd(lambda u: transform.forward(u)[:-1])(t)
            return jnp.linalg.slogdet(jacobian)[1]

        flat = theta.reshape((-1, theta.shape[-1]))
        return jax.vmap(log_determinant)(flat).reshape(theta.shape[:-1])


def OpenInterval(low: float, high: float) -> Support:  # noqa: N802 - reads as a constant
    """The open interval :math:`(\\mathrm{low}, \\mathrm{high})`, whose bijector
    is ``tfb.Sigmoid(low=low, high=high)``.

    Raises
    ------
    ValueError
        If an end is not finite or ``low >= high``.
    """
    return Support("interval", float(low), float(high))


# ── functions ─────────────────────────────────────────────────────────────────


def site_positions(site_table: pd.DataFrame, dim_name: str) -> np.ndarray:
    """Each site's position along a dim: :math:`\\ell_d(s)` for every site
    :math:`s`, as an index into the dim's labels.

    With *site_table* the table a prior function, derived parameter or rule
    was given, a value ``v`` on dim *dim_name*, shape ``(n, ...)``, is read at
    the sites as ``v[site_positions(site_table, dim_name)]``, shape
    ``(S, ...)``.

    Parameters
    ----------
    site_table:
        The vector's site table, in its own row order, with a categorical
        column per site-labels name whose categories are that dim's labels.
    dim_name:
        ``"site"``, for which the positions are ``0`` to ``S - 1``, or a
        site-labels name.

    Returns
    -------
    numpy.ndarray
        ``(S,)`` ``int64``: the position of each site's dim label within
        ``site_table[dim_name].cat.categories``, which is
        :meth:`ParameterVector.dim_index` order.

    Raises
    ------
    KeyError, TypeError
        If *site_table* has no categorical column *dim_name*.
    """
    if dim_name == SITE:
        return np.arange(len(site_table), dtype=np.int64)
    check_site_table_has_a_categorical_dim(site_table, dim_name)
    return site_table[dim_name].cat.codes.to_numpy(np.int64)


def dim_label_positions(site_table: pd.DataFrame, from_dim: str, to_dim: str) -> np.ndarray:
    """Each dim label of one dim's position along another that it nests in.

    For dim label :math:`i` of *from_dim*, whose sites are
    :math:`\\{s : \\ell_{\\mathrm{from}}(s) = i\\}`, the result is the one
    :math:`j` with :math:`\\ell_{\\mathrm{to}}(s) = j` at all of them. A value
    ``v`` on *to_dim* is then read at *from_dim*'s labels as
    ``v[dim_label_positions(site_table, from_dim, to_dim)]``: PFT-level
    values around biome-level means, for example.

    Parameters
    ----------
    site_table:
        As :func:`site_positions`.
    from_dim, to_dim:
        ``"site"`` or site-labels names.

    Returns
    -------
    numpy.ndarray
        ``(n_from,)`` ``int64``, in *from_dim*'s dim-label order.

    Raises
    ------
    KeyError, TypeError
        As :func:`site_positions`.
    ValueError
        If the sites of some *from_dim* dim label carry more than one
        *to_dim* dim label, so the two dims do not nest; the message names
        that dim label.
    """
    from_positions = site_positions(site_table, from_dim)
    to_positions = site_positions(site_table, to_dim)
    labels = _dim_labels_of(site_table, from_dim)
    pairs = np.unique(np.stack([from_positions, to_positions], axis=1), axis=0)
    check_dims_nest(pairs, labels, from_dim, to_dim)
    return pairs[:, 1].astype(np.int64)


def bijectors_agree(first: tfb.Bijector, second: tfb.Bijector, probes: Any) -> bool:
    """Whether two bijectors map *probes* alike, to a relative tolerance of
    ``1e-10``. Images are compared, never bijectors: ``tfb.Sigmoid()`` and
    ``tfb.Sigmoid(low=0., high=1.)`` compare unequal."""
    probes = jnp.asarray(probes)
    return bool(
        np.allclose(first.forward(probes), second.forward(jnp.array(probes)), rtol=1e-10, atol=0.0)
    )


def probe_points(unconstrained_shape: tuple[int, ...], *, unconstrained_size: int) -> np.ndarray:
    """The fixed points of theta at which bijectors and priors are probed.

    For one unconstrained value of shape *unconstrained_shape*, whose first
    dim label holds its first *unconstrained_size* numbers (``e``):
    :math:`\\theta = 0`; :math:`\\pm c \\mathbf 1` for
    :math:`c \\in \\{3, 10, 20\\}`; :math:`\\pm c\\, e_i` along each
    unconstrained number :math:`i` of the first dim label, for
    :math:`c \\in \\{10, 20\\}`; and four fixed-seed random directions of
    norm 10.

    Returns
    -------
    numpy.ndarray
        ``float64``, ``(n_probes, *unconstrained_shape)``.

    Notes
    -----
    They stop at 20: :math:`e^{20} \\approx 4.9 \\times 10^8` is beyond any
    plausible magnitude, and the logistic at 20 is within
    :math:`2 \\times 10^{-9}` of its bound, while it rounds to exactly 1
    only near 37. A probe test is not a proof: a support that differs only
    beyond the probes passes it.
    """
    return joint_probe_points([(unconstrained_shape, unconstrained_size)])[0]


def joint_probe_points(parts: Sequence[tuple[tuple[int, ...], int]]) -> list[np.ndarray]:
    """:func:`probe_points` over several unconstrained values at once.

    Each part is ``(unconstrained_shape, unconstrained_size)``, as
    :func:`probe_points` takes. The points are those of :func:`probe_points`
    in the space of all parts concatenated: :math:`\\theta = 0`,
    :math:`\\pm c \\mathbf 1`, :math:`\\pm c\\, e_i` along the first dim
    label's unconstrained numbers of every part in turn, and four
    fixed-seed random directions of norm 10. For one part it is
    :func:`probe_points`.

    Returns
    -------
    list of numpy.ndarray
        One per part, ``(n_probes, *unconstrained_shape)``, ``float64``,
        with one ``n_probes`` for all.
    """
    shapes = [tuple(shape) for shape, _ in parts]
    sizes = [int(np.prod(shape, dtype=int)) for shape in shapes]
    total = sum(sizes)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    points = [np.zeros(total)]
    points += [sign * c * np.ones(total) for c in (3.0, 10.0, 20.0) for sign in (1.0, -1.0)]
    for (_, unconstrained_size), size, offset in zip(parts, sizes, offsets):
        for i in range(min(unconstrained_size, size)):
            for c in (10.0, 20.0):
                for sign in (1.0, -1.0):
                    point = np.zeros(total)
                    point[offset + i] = sign * c
                    points.append(point)
    directions = np.random.default_rng(_PROBE_SEED).standard_normal((4, total))
    points += list(10.0 * directions / np.linalg.norm(directions, axis=1, keepdims=True))
    stacked = np.asarray(points, dtype=np.float64)
    return [
        stacked[:, offset : offset + size].reshape((len(points), *shape))
        for shape, size, offset in zip(shapes, sizes, offsets)
    ]


# ── the aliases and their validators ──────────────────────────────────────────

#: Natural values, ``{parameter name: (..., *value_shape)}``, as the module's
#: data model has them; checked by :func:`validate_natural_values`.
type NaturalValues = Mapping[str, Array]

#: The labeled form of theta, as the module's data model has it; checked by
#: :func:`validate_parameter_dataset`.
type ParameterDataset = xr.Dataset


def validate_natural_values(
    natural_values: Any, parameter_vector: ParameterVector, *, at_sites: bool = False
) -> None:
    """Check that *natural_values* are :data:`NaturalValues` of the vector,
    or its values at sites when *at_sites*. A derived parameter's value is
    checked where present.

    Raises
    ------
    TypeError
        If *natural_values* is not a mapping.
    KeyError
        If a parameter has no value.
    ValueError
        If a value's trailing shape is not its parameter's, or the leading
        shapes differ.
    """
    check_natural_values_are_a_mapping(natural_values)
    leads = []
    present = [d for d in parameter_vector.derived_parameters if d.name in natural_values]
    for p in (*parameter_vector.parameters, *present):
        expected = (
            (parameter_vector.n_sites, *p.natural_shape)
            if at_sites
            else parameter_vector.value_shape(p.name)
        )
        shape = tuple(jnp.shape(natural_values[p.name]))
        check_natural_value_ends_in_the_shape(p.name, shape, expected)
        leads.append((p.name, shape[: len(shape) - len(expected)]))
    check_natural_values_share_a_leading_shape(leads)


def validate_parameter_dataset(parameter_dataset: Any) -> None:
    """Check that *parameter_dataset* is a :data:`ParameterDataset` a vector
    can read, as the module's data model has it. The values a vector reads
    are checked finite as :meth:`ParameterVector.flat` reads them.

    Raises
    ------
    TypeError
        If it is not an ``xr.Dataset``.
    ValueError
        If it has more than one batch dim.
    """
    check_parameter_dataset_is_a_dataset(parameter_dataset)
    check_at_most_one_batch_dim(batch_dims(parameter_dataset), message_name="the parameter dataset")


# ── private helpers ───────────────────────────────────────────────────────────

#: How far a simplex value's sum may be from 1: float64 rounding of a
#: ``SoftmaxCentered`` image is about ``1e-16`` per number.
_SIMPLEX_SUM_TOLERANCE = 1e-10

#: The seed of :func:`probe_points`' random directions.
_PROBE_SEED = 20260926


def _same_label(label: Any, wanted: Any) -> bool:
    """Whether dim label *label* is *wanted*: strings match strings, and
    integers (site ids) match integers, never a boolean."""
    if isinstance(label, str):
        return isinstance(wanted, str) and label == wanted
    return isinstance(wanted, (int, np.integer)) and not isinstance(wanted, bool) and label == wanted


def _dataset_variables(parameter: Parameter | DerivedParameter) -> list[tuple[str, dict[str, Any]]]:
    """``(variable name, attributes)`` for each natural name of a parameter
    or derived parameter."""
    names = (None,) if parameter.natural_names is None else parameter.natural_names
    out = []
    for natural_name in names:
        attributes: dict[str, Any] = {"parameter": parameter.name}
        if natural_name is not None:
            attributes["natural_name"] = natural_name
        if parameter.units is not None:
            attributes["units"] = parameter.units
        if parameter.support is not None:
            attributes["support"] = parameter.support.name
        if isinstance(parameter, DerivedParameter):
            attributes["derived_from"] = ", ".join(parameter.derived_from)
        variable_name = parameter.name if natural_name is None else f"{parameter.name}.{natural_name}"
        out.append((variable_name, attributes))
    return out


def _dim_labels_of(site_table: pd.DataFrame, dim_name: str) -> list[Any]:
    """A dim's labels in the table: the site ids for ``"site"``, else the
    categories of its column."""
    if dim_name == SITE:
        return site_table[SITE_ID].tolist()
    return list(site_table[dim_name].cat.categories)


def _derived_probe_theta(parameter_vector: ParameterVector, derived: DerivedParameter) -> np.ndarray:
    """Theta at :func:`joint_probe_points` over the parameters *derived* is
    computed from, directly or through earlier derived parameters; every
    other entry 0."""
    needed = set(derived.derived_from)
    for earlier in reversed(parameter_vector.derived_parameters):
        if earlier.name in needed:
            needed.update(earlier.derived_from)
    parameters = [p for p in parameter_vector.parameters if p.name in needed]
    parts = joint_probe_points(
        [(parameter_vector.unconstrained_shape(p.name), p.unconstrained_size) for p in parameters]
    )
    theta = np.zeros((len(parts[0]), parameter_vector.dimension))
    for parameter, points in zip(parameters, parts):
        theta[:, parameter_vector.positions(parameter_name=parameter.name)] = points.reshape((len(points), -1))
    return theta


def _lies_in_the_support_or_overflows(support: Support, values: Array) -> Array:
    """Whether each value lies in *support*'s closure, or is infinite at an
    end the support leaves unbounded (float64 overflow); NaN never does."""
    values = jnp.asarray(values, dtype=jnp.float64)
    inside = support.contains(values, closure=True)
    if support.kind == "real":
        return ~jnp.isnan(values)
    if support.kind == "positive":
        return inside | jnp.isposinf(values)
    return inside


def _normalized_site_table(site_table: Any, site_covariate_names: tuple[str, ...]) -> pd.DataFrame:
    """``site_id`` (``int32``), ``lon``, ``lat`` and the named covariates of
    *site_table*, in its row order, after the site-table checks."""
    check_site_table_is_keyed_on_site_ids(site_table)
    check_site_table_has_locations(site_table)
    table = site_table if SITE_ID in site_table.columns else site_table.reset_index()
    site_ids = as_site_ids(table[SITE_ID].to_numpy(), message_name="the site table's site_id")
    check_site_table_has_a_site(site_ids)
    check_site_ids_are_ascending(np.asarray(site_ids))
    for name in site_covariate_names:
        check_site_covariate_is_a_column(name, table)
        check_site_covariate_is_float64(name, table[name])
    out = pd.DataFrame(
        {
            SITE_ID: np.asarray(site_ids, dtype=SITE_DTYPE),
            LON: table[LON].to_numpy(np.float64),
            LAT: table[LAT].to_numpy(np.float64),
            **{name: table[name].to_numpy(np.float64, copy=True) for name in site_covariate_names},
        }
    )
    check_site_table_values_are_finite(out, (LON, LAT, *site_covariate_names))
    return out


def _site_labels_column(
    name: str, value: Any, site_ids: pd.Series
) -> tuple[pd.Categorical, tuple[str, ...]]:
    """One site-labels argument as (each site's label, a categorical whose
    categories are the labels present in declared order; the declared
    labels)."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        if isinstance(value, pd.DataFrame):
            check_site_labels_table_has_the_columns(name, value)
            indexed = site_lookup(value)[LABEL_COLUMN]
        else:
            indexed = value  # keyed by its index, which holds site ids
        check_site_labels_label_every_site(name, indexed.index, site_ids)
        declared = _declared_labels(indexed)
        labels = indexed.loc[site_ids.tolist()].tolist()
    elif isinstance(getattr(value, "dtype", None), pd.CategoricalDtype) or isinstance(
        value, pd.Categorical
    ):
        categorical = pd.Categorical(value)
        labels, declared = categorical.tolist(), tuple(categorical.categories.tolist())
    else:
        labels = list(as_sequence(value, message_name=f"site_labels[{name!r}]"))
        declared = None
    check_site_labels_are_one_per_site(name, labels, len(site_ids))
    check_site_labels_are_strings(name, labels)
    if declared is None:
        declared = tuple(sorted(set(labels)))
    present = set(labels)
    categories = [label for label in declared if label in present]
    return pd.Categorical(labels, categories=categories), tuple(declared)


def _declared_labels(labels: pd.Series) -> tuple[str, ...] | None:
    """A categorical's categories; ``None`` for plain labels."""
    if isinstance(labels.dtype, pd.CategoricalDtype):
        return tuple(labels.cat.categories.tolist())
    return None


# ── checks ────────────────────────────────────────────────────────────────────


def check_parameter_vector_is_valid(parameter_vector: ParameterVector) -> None:
    """The parameters, derived parameters, site table and site labels make
    one vector."""
    check_vector_has_a_parameter(parameter_vector.parameters)
    check_parameter_names_are_unique(parameter_vector.parameter_names)
    check_names_are_unique(parameter_vector.derived_parameter_names, message_name="the derived parameter names")
    names = {
        "derived parameter": parameter_vector.derived_parameter_names,
        "site covariate": parameter_vector.site_covariate_names,
        "site-labels name": tuple(parameter_vector.site_labels),
    }
    for what, taken in names.items():
        check_names_are_not_reserved(taken, what)
    check_names_are_distinct(parameter_vector.parameter_names, names)
    for parameter in (*parameter_vector.parameters, *parameter_vector.derived_parameters):
        check_parameter_dim_is_the_vectors(parameter, parameter_vector)
    for derived in parameter_vector.derived_parameters:
        check_derived_parameter_fits_the_vector(derived, parameter_vector)


def check_parameter_is_valid(parameter: Parameter) -> None:
    """A parameter's name, natural names and bijector are usable."""
    check_name_is_usable(parameter.name, "a parameter")
    if parameter.natural_names is not None:
        for natural_name in parameter.natural_names:
            check_name_is_usable(natural_name, f"a natural name of {parameter.name!r}", reserved=False)
        check_names_are_unique(parameter.natural_names, message_name=f"{parameter.name!r} natural_names")
    check_simplex_has_two_natural_names(parameter)
    if parameter._custom_bijector:
        check_bijector_acts_per_number(parameter)
        check_bijector_maps_onto_the_support(parameter)


def check_parameter_vectors_share_a_layout(first: ParameterVector, second: ParameterVector) -> None:
    """Two vectors give theta one meaning: the same index, sites, dim indexes,
    supports, and transforms that agree at the probe points; site covariates
    are not compared."""
    differences = [
        ("index", lambda: first.index.equals(second.index)),
        ("sites", lambda: first.sites == second.sites),
        ("dim indexes", lambda: all(
            first.dim_index(d).equals(second.dim_index(d)) for d in first.dim_names
        )),
        ("supports", lambda: all(
            p.support == q.support for p, q in zip(first.parameters, second.parameters)
        )),
        ("transforms", lambda: all(
            bijectors_agree(
                p.bijector,
                q.bijector,
                probe_points(p.unconstrained_shape, unconstrained_size=p.unconstrained_size),
            )
            for p, q in zip(first.parameters, second.parameters)
        )),
    ]
    for what, same in differences:
        if not same():
            raise ValueError(
                f"the two parameter vectors differ in their {what}, so theta means different "
                "things under them; pair a prior and a forward model built on the same "
                "vector, as prior.parameter_vector."
            )


def check_batch_dim_name_is_not_taken(parameter_vector: ParameterVector, batch_dim: Any) -> None:
    """*batch_dim* can name the labeled form's batch dim: no reserved name,
    and none of the vector's dims or variables, which it would collide with."""
    check_batch_dim_name_is_not_reserved(batch_dim, message_name="batch_dim")
    pieces = (*parameter_vector.parameters, *parameter_vector.derived_parameters)
    taken = {
        *parameter_vector.site_labels,
        *(name for p in pieces for name, _ in _dataset_variables(p)),
    }
    if batch_dim in taken:
        raise ValueError(
            f"batch_dim {batch_dim!r} is a dim or variable of the vector's labeled form; name "
            "the batch dim for what it indexes, such as 'sample'."
        )


def check_vector_has_a_parameter(parameters: tuple[Parameter, ...]) -> None:
    """A vector holds at least one parameter."""
    if not parameters:
        raise ValueError("a parameter vector needs at least one parameter; give parameters=[...].")


def check_parameter_names_are_unique(names: tuple[str, ...]) -> None:
    """Every parameter has its own name, which keys theta's segments."""
    check_names_are_unique(names, message_name="the parameter names")


def check_name_is_usable(name: Any, what: str, *, reserved: bool = True) -> None:
    """*name* is ``lower_case_with_underscores`` and not a Python keyword,
    since values are passed as keyword arguments named for it; and, when
    *reserved*, none of
    :data:`RESERVED_NAMES`, which the labeled form's coordinates take."""
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name) or keyword.iskeyword(name):
        raise ValueError(
            f"{what} is named {name!r}; a name is lower_case_with_underscores and not a Python "
            "keyword, such as 'base_soil_respiration'."
        )
    if reserved and name in RESERVED_NAMES:
        raise ValueError(
            f"{what} is named {name!r}, which a coordinate of the labeled form takes; choose "
            f"a name other than {sorted(RESERVED_NAMES)}."
        )


def check_names_are_not_reserved(names: Sequence[str], what: str) -> None:
    """No site covariate or site-labels name is one of :data:`RESERVED_NAMES`,
    which the labeled form's coordinates take."""
    for name in names:
        if name in RESERVED_NAMES:
            raise ValueError(
                f"the {what} {name!r} is a reserved name; rename it to something other than "
                f"{sorted(RESERVED_NAMES)}."
            )


def check_names_are_distinct(
    parameter_names: tuple[str, ...], other_names: Mapping[str, Sequence[str]]
) -> None:
    """No parameter shares a name with a site covariate or site-labels name,
    nor those with each other, since all are read by name."""
    seen = {name: "parameter" for name in parameter_names}
    for what, names in other_names.items():
        for name in names:
            if name in seen:
                raise ValueError(
                    f"{name!r} is both a {seen[name]} and a {what}; they are read by name, so "
                    "rename one."
                )
            seen[name] = what


def check_parameter_dim_is_the_vectors(
    parameter: Parameter | DerivedParameter, parameter_vector: ParameterVector
) -> None:
    """A parameter's or derived parameter's dim is ``"site"`` or a
    site-labels name of the vector."""
    if parameter.dim is not None and parameter.dim != SITE and parameter.dim not in parameter_vector.site_labels:
        raise ValueError(
            f"parameter {parameter.name!r} varies over {parameter.dim!r}, which is neither "
            f"'site' nor a site-labels name of the vector ({sorted(parameter_vector.site_labels)}); "
            "give its site labels as site_labels={...}."
        )


def check_derived_parameter_fits_the_vector(
    derived: DerivedParameter, parameter_vector: ParameterVector
) -> None:
    """A derived parameter reads earlier names, and its value has its shape,
    lies in its declared support, and is pointwise when it says so."""
    check_derived_parameter_reads_earlier_names(derived, parameter_vector)
    check_derived_parameter_has_its_value_shape(derived, parameter_vector)
    if derived.support is not None:
        check_derived_parameter_lies_in_its_support(derived, parameter_vector)
    if derived.pointwise and derived.dim is not None:
        check_derived_parameter_is_pointwise(derived, parameter_vector)


def check_derived_parameter_is_valid(derived: DerivedParameter) -> None:
    """A derived parameter's name and natural names are usable, and it is
    computed from something."""
    check_name_is_usable(derived.name, "a derived parameter")
    if derived.natural_names is not None:
        for natural_name in derived.natural_names:
            check_name_is_usable(natural_name, f"a natural name of {derived.name!r}", reserved=False)
        check_names_are_unique(derived.natural_names, message_name=f"{derived.name!r} natural_names")
    check_simplex_has_two_natural_names(derived)
    check_derived_parameter_is_computed_from_something(derived)


def check_derived_parameter_is_computed_from_something(derived: DerivedParameter) -> None:
    """A derived parameter has inputs, whose draws give its values their
    leading shape; a constant would not broadcast over the draws."""
    if not derived.derived_from:
        raise ValueError(
            f"derived parameter {derived.name!r} is computed from nothing; a value fixed across "
            "draws is a Fixed SIPNET parameter or an external input, not a derived parameter."
        )


def check_derived_parameter_reads_earlier_names(
    derived: DerivedParameter, parameter_vector: ParameterVector
) -> None:
    """A derived parameter is computed from parameters and derived parameters
    declared before it, which :meth:`ParameterVector.to_natural` has computed
    by then."""
    derived_names = parameter_vector.derived_parameter_names
    earlier = set(parameter_vector.parameter_names) | set(derived_names[: derived_names.index(derived.name)])
    for name in derived.derived_from:
        if name in earlier:
            continue
        if name in derived_names:
            raise ValueError(
                f"derived parameter {derived.name!r} is computed from {name!r}, which is declared "
                "after it; declare the derived parameters in the order they are computed."
            )
        raise KeyError(
            f"derived parameter {derived.name!r} is computed from {name!r}, which is no parameter "
            "or derived parameter of the vector; name one of them."
        )


def check_derived_parameter_has_its_value_shape(
    derived: DerivedParameter, parameter_vector: ParameterVector
) -> None:
    """A derived parameter's ``compute`` returns its value shape at theta = 0,
    which would otherwise broadcast or misalign against the sites."""
    # One draw's worth of theta with a leading axis, so compute runs vmapped
    # as it does at run time.
    natural_values = parameter_vector._parameters_at(np.zeros((1, parameter_vector.dimension)))
    value = parameter_vector.derived_values(natural_values, derived_parameter_names=[derived.name])
    shape, expected = tuple(jnp.shape(value[derived.name]))[1:], parameter_vector.value_shape(derived.name)
    if shape != expected:
        raise ValueError(
            f"derived parameter {derived.name!r} computes a value of shape {shape}, but its dim and "
            f"natural names give {expected}; return one draw's value, shape (n?, k?)."
        )


def check_derived_parameter_lies_in_its_support(
    derived: DerivedParameter, parameter_vector: ParameterVector
) -> None:
    """A derived parameter's values at the probe points lie in the support it
    declares, which the SIPNET map would otherwise trust in its bounds check."""
    natural_values = parameter_vector._parameters_at(_derived_probe_theta(parameter_vector, derived))
    values = parameter_vector.derived_values(natural_values, derived_parameter_names=[derived.name])
    if not bool(jnp.all(_lies_in_the_support_or_overflows(derived.support, values[derived.name]))):
        raise ValueError(
            f"derived parameter {derived.name!r} takes values outside its declared support "
            f"{derived.support.name!r} at the probe points; declare the support its values have, "
            "or none."
        )


def check_derived_parameter_is_pointwise(derived: DerivedParameter, parameter_vector: ParameterVector) -> None:
    """A derived parameter declared pointwise changes at one dim label only
    when its inputs on the same dim change there, as selection and
    localization assume."""
    natural_values = parameter_vector._parameters_at(np.zeros(parameter_vector.dimension))
    natural_values |= parameter_vector.derived_values(
        natural_values, derived_parameter_names=[derived.name]
    )
    before = np.asarray(natural_values[derived.name])
    for name in derived.derived_from:
        if parameter_vector._piece(name).dim != derived.dim:
            continue
        value = jnp.asarray(natural_values[name])
        nudged = {**natural_values, name: value.at[0].add(0.5 * (1.0 + jnp.abs(value[0])))}
        after = np.asarray(parameter_vector._computed(derived, nudged))
        if not np.allclose(after[1:], before[1:], rtol=1e-12, atol=0.0, equal_nan=True):
            raise ValueError(
                f"derived parameter {derived.name!r} is declared pointwise, but changing {name!r} "
                f"at one {derived.dim} dim label changes its value at others; declare it "
                "pointwise=False."
            )


def check_simplex_has_two_natural_names(parameter: Parameter | DerivedParameter) -> None:
    """A value on the simplex names at least two natural numbers; a scalar's
    support would otherwise be read across its dim labels."""
    if parameter.support is not None and parameter.support.kind == "simplex" and parameter.natural_size < 2:
        raise ValueError(
            f"parameter {parameter.name!r} is on the simplex, which needs at least two natural "
            "numbers; give natural_names=(...) with two or more names."
        )


def check_bijector_maps_onto_the_support(parameter: Parameter) -> None:
    """A custom bijector maps the probe points into the support's closure,
    and the support's default image of them back to finite theta that it maps
    there again."""
    probes = jnp.asarray(
        probe_points(parameter.unconstrained_shape, unconstrained_size=parameter.unconstrained_size)
    )
    support = parameter.support
    # The closure: a bijector may round onto the boundary at the outer probes
    # (IteratedSigmoidCentered does at 20), which is float64, not a wrong map.
    into = support.contains(parameter.bijector.forward(probes), closure=True)
    targets = support.bijector().forward(jnp.array(probes))
    back = parameter.bijector.inverse(targets)
    # A fresh copy: TFP caches the pair, and would hand targets back unchanged.
    again = parameter.bijector.forward(jnp.array(np.asarray(back)))
    if not (
        bool(jnp.all(into))
        and bool(jnp.all(jnp.isfinite(back)))
        and np.allclose(again, targets, rtol=1e-6, atol=1e-9)
    ):
        raise ValueError(
            f"parameter {parameter.name!r}: its bijector {parameter.bijector.name!r} does not map "
            f"theta onto the support {support.name!r} at the probe points; give a bijector from "
            "R^e onto the support, or omit bijector= to take the support's own."
        )


def check_bijector_acts_per_number(parameter: Parameter) -> None:
    """A custom bijector on an elementwise support acts on each number alone,
    as its Jacobian is computed, and one on the simplex on a whole value."""
    expected = 1 if parameter.support.kind == "simplex" else 0
    if parameter.bijector.forward_min_event_ndims != expected:
        raise ValueError(
            f"parameter {parameter.name!r}: its bijector acts on {parameter.bijector.forward_min_event_ndims}"
            f"-dimensional events, but on a {parameter.support.kind} support it must act on "
            f"{'a whole simplex value' if expected else 'each number alone'}; give such a bijector."
        )


def check_support_is_valid(support: Support) -> None:
    """A support is a known kind, with finite ``low < high`` exactly when it
    is an interval."""
    if support.kind not in ("real", "positive", "interval", "simplex"):
        raise ValueError(
            f"a support is 'real', 'positive', 'interval' or 'simplex', got {support.kind!r}; "
            "use REAL, POSITIVE, OPEN_UNIT_INTERVAL, OpenInterval(low, high) or SIMPLEX."
        )
    ends = (support.low, support.high)
    if support.kind != "interval":
        if ends != (None, None):
            raise ValueError(f"a {support.kind} support has no ends, got {ends}; drop them.")
        return
    if (
        None in ends
        or not np.isfinite(ends).all()
        or not support.low < support.high
    ):
        raise ValueError(
            f"an interval support needs finite ends with low < high, got {ends}; use "
            "OpenInterval(low, high)."
        )


def check_theta_ends_in_the_dimension(shape: tuple[int, ...], dimension: int) -> None:
    """Theta's last axis has ``D`` entries."""
    if not shape or shape[-1] != dimension:
        raise ValueError(
            f"theta must end in the vector's dimension {dimension}, got shape {shape}; pass "
            "(..., D)."
        )


def check_parameter_names_are_held(names: Sequence[str], parameter_vector: ParameterVector) -> None:
    """Every name is one of the vector's parameters."""
    for name in names:
        if name not in parameter_vector.parameter_names:
            raise KeyError(
                f"the vector has no parameter {name!r}; name one of "
                f"{truncated(list(parameter_vector.parameter_names))}."
            )


def check_dim_is_the_vectors(dim_name: Any, parameter_vector: ParameterVector) -> None:
    """*dim_name* is ``"site"`` or one of the vector's site-labels names."""
    if dim_name not in parameter_vector.site_labels:
        raise KeyError(
            f"the vector has no dim {dim_name!r}; name 'site' or one of its site-labels names "
            f"{sorted(parameter_vector.site_labels)}."
        )


def check_dim_label_has_a_dim(dim_label: Any, dim: Any, parameter_name: Any) -> None:
    """A dim label is selected together with its dim or its parameter, since
    labels of two dims may coincide."""
    if dim_label is not None and dim is None and parameter_name is None:
        raise ValueError(
            f"dim_label={dim_label!r} is given without its dim, and two dims may share a label; "
            "pass dim= or parameter_name= too."
        )


def check_selector_is_in_the_index(level: str, value: Any, values: pd.Index) -> None:
    """A selector names a value of its index level."""
    if value not in set(values):
        raise KeyError(
            f"no entry of theta has {level} {value!r}; name one of {truncated(sorted(set(values), key=str))}."
        )


def check_dim_label_is_the_dims(dim_label: Any, labels: Sequence[Any], dim: str | None) -> None:
    """A dim label is one of its dim's, compared with the dim's own type, so a
    site id is never matched by a string."""
    if not any(_same_label(label, dim_label) for label in labels):
        raise KeyError(
            f"{dim_label!r} is not a dim label of {dim!r}; name one of {truncated(list(labels))}."
        )


def check_selector_is_not_no_dim(dim: Any, dim_label: Any) -> None:
    """``NO_DIM`` marks entries without a dim and selects nothing."""
    if NO_DIM in (dim, dim_label):
        raise KeyError(
            "NO_DIM marks the entries of parameters without a dim and is not a selector; "
            "select them with parameter_name=."
        )


def check_dim_labels_select_a_site_labels_dim(name: Any, parameter_vector: ParameterVector) -> None:
    """``dim_labels=`` names a site-labels dim; sites are selected with
    ``sites=``."""
    if name not in parameter_vector.site_labels:
        raise KeyError(
            f"dim_labels names {name!r}, which is not a site-labels name of the vector "
            f"({sorted(parameter_vector.site_labels)}); select sites with sites=."
        )


def check_sites_are_held(sites: Sequence[int], parameter_vector: ParameterVector) -> None:
    """Every site asked of ``select`` is one of the vector's."""
    held = set(parameter_vector.sites)
    unknown = [site for site in sites if site not in held]
    if unknown:
        raise KeyError(
            f"the vector has no site(s) {truncated(unknown)}; select from its sites "
            f"{truncated(list(parameter_vector.sites))}."
        )


def check_site_table_has_a_site(site_ids: Sequence[int]) -> None:
    """The site table holds at least one site; an empty one makes an empty
    vector, which fails far from its cause."""
    if not site_ids:
        raise ValueError("the site table holds no site; give at least one.")


def check_dataset_dim_is_labeled(variable: xr.DataArray, name: str, dim: str) -> None:
    """A labeled form's dim has its labels, since xarray would otherwise select
    by position."""
    if dim not in variable.indexes:
        raise ValueError(
            f"the parameter dataset's {name!r} has no {dim!r} coordinate, so its values cannot be "
            f"matched to dim labels; give {dim!r} its labels."
        )


def check_selection_keeps_a_site(kept: np.ndarray) -> None:
    """A selection keeps at least one site."""
    if not kept.any():
        raise ValueError("the selection keeps no site; widen sites= or dim_labels=.")


def check_site_ids_are_ascending(site_ids: np.ndarray) -> None:
    """The site table lists its sites in ascending ``site_id``, which is the
    ``site`` dim's order."""
    if np.any(np.diff(site_ids) <= 0):
        raise ValueError(
            "the site table's site_id must be ascending; sort it with "
            "site_table.sort_values('site_id')."
        )


def check_site_covariate_is_a_column(name: str, table: pd.DataFrame) -> None:
    """A named site covariate is a column of the site table."""
    if name not in table.columns:
        raise KeyError(
            f"site covariate {name!r} is not a column of the site table; join it on first, or "
            "drop it from site_covariate_names."
        )


def check_site_covariate_is_float64(name: str, column: pd.Series) -> None:
    """A named site covariate is ``float64``: a code or a string is no
    covariate until it is deliberately converted."""
    if column.dtype != np.float64:
        raise TypeError(
            f"site covariate {name!r} is {column.dtype}, not float64; convert it deliberately "
            "(a class code is not a quantity), or drop it from site_covariate_names."
        )


def check_site_table_values_are_finite(table: pd.DataFrame, names: Sequence[str]) -> None:
    """``lon``, ``lat`` and every site covariate are finite at every site."""
    for name in names:
        values = table[name].to_numpy()
        if not np.isfinite(values).all():
            sites = table[SITE_ID].to_numpy()[~np.isfinite(values)].tolist()
            raise ValueError(
                f"the site table's {name!r} is not finite at site(s) {truncated(sites)}; drop "
                "those sites or fill the column first."
            )


def check_site_labels_table_has_the_columns(name: str, table: pd.DataFrame) -> None:
    """A site-labels table has ``site_id`` and ``label``."""
    missing = [c for c in (SITE_ID, LABEL_COLUMN) if c not in table.columns]
    if missing:
        raise KeyError(
            f"site labels {name!r} lack the column(s) {missing}; pass the table "
            "load_site_labels() returns."
        )


def check_site_labels_label_every_site(name: str, labeled: pd.Index, site_ids: pd.Series) -> None:
    """A site-labels table labels every site of the vector."""
    missing = sorted(set(site_ids.tolist()) - set(labeled.tolist()))
    if missing:
        raise KeyError(
            f"site labels {name!r} give no label for site(s) {truncated(missing)}; select sites "
            "the site labels cover."
        )


def check_site_labels_are_one_per_site(name: str, labels: Sequence[Any], n_sites: int) -> None:
    """Plain site labels are one per site."""
    if len(labels) != n_sites:
        raise ValueError(
            f"site labels {name!r} hold {len(labels)} labels for {n_sites} sites; give one per "
            "site, in site order."
        )


def check_site_labels_are_strings(name: str, labels: Sequence[Any]) -> None:
    """Every site's label is a string: a missing label leaves a site on no
    dim label, and a number would be read as a site id."""
    bad = [label for label in labels if not isinstance(label, str)]
    if bad:
        raise TypeError(
            f"site labels {name!r} hold labels that are not strings, such as {bad[0]!r}; every "
            "site needs a string label."
        )


def check_natural_values_hold_the_inputs(derived: DerivedParameter, natural_values: Mapping[str, Any]) -> None:
    """The natural values hold everything a derived parameter is computed
    from."""
    missing = [name for name in derived.derived_from if name not in natural_values]
    if missing:
        raise KeyError(
            f"derived parameter {derived.name!r} is computed from {missing}, which the natural "
            "values lack; give every parameter it is computed from."
        )


def check_derived_parameter_names_are_held(names: Sequence[str], parameter_vector: ParameterVector) -> None:
    """Every name is one of the vector's derived parameters; any other would
    be computed as nothing, and missed far from the request."""
    for name in names:
        if name not in parameter_vector.derived_parameter_names:
            raise KeyError(
                f"the vector has no derived parameter {name!r}; name one of "
                f"{truncated(list(parameter_vector.derived_parameter_names))}."
            )


def check_dataset_can_project_onto_the_vector(parameter_dataset: xr.Dataset, parameter_vector: ParameterVector) -> None:
    """A vector with a derived parameter that is not pointwise reads only a
    labeled form over its own dim labels: over others, the same theta would
    give that derived parameter other values at the same sites."""
    if all(d.pointwise for d in parameter_vector.derived_parameters):
        return
    for dim in parameter_vector.dim_names:
        held = parameter_dataset.indexes.get(dim)
        # flat and site_fields read by label, so the order does not matter.
        if held is not None and set(held) != set(parameter_vector.dim_index(dim)):
            raise ValueError(
                f"the parameter dataset's {dim!r} dim labels are not the vector's, and the vector has a "
                "derived parameter that is not pointwise, whose values depend on every dim label "
                "present; read the dataset with the vector it was made by, or give it this "
                "vector's dim labels."
            )


def check_site_table_has_a_categorical_dim(site_table: pd.DataFrame, dim_name: str) -> None:
    """A site table has a categorical column for the dim, whose categories
    are the dim labels positions are counted along."""
    if dim_name not in site_table.columns:
        raise KeyError(
            f"the site table has no column {dim_name!r}; pass the site_table a prior function, "
            "derived parameter or rule was given, which has one per site-labels name."
        )
    if not isinstance(site_table[dim_name].dtype, pd.CategoricalDtype):
        raise TypeError(
            f"the site table's {dim_name!r} is not categorical, so its dim labels have no order; pass "
            "the site_table a prior function, derived parameter or rule was given."
        )


def check_dims_nest(pairs: np.ndarray, from_labels: Sequence[Any], from_dim: str, to_dim: str) -> None:
    """The sites of each dim label of one dim carry exactly one dim label of
    the other, so a value on the second can be read at the first's labels."""
    counts = np.bincount(pairs[:, 0], minlength=len(from_labels))
    bad = np.flatnonzero(counts != 1)
    if bad.size:
        label = from_labels[bad[0]]
        raise ValueError(
            f"the sites of {from_dim} dim label {label!r} carry {int(counts[bad[0]])} {to_dim} dim "
            f"labels, so {from_dim!r} does not nest in {to_dim!r}; pool over dims that nest."
        )


def check_values_are_in_the_support(parameter: Parameter | DerivedParameter, values: Array) -> None:
    """Natural values read from the labeled form lie in their support."""
    inside = parameter.support.contains(values)
    if not bool(jnp.all(inside)):
        raise ValueError(
            f"parameter {parameter.name!r} holds values outside its support "
            f"{parameter.support.name!r}, which no theta maps to; give values inside it."
        )


def check_variable_is_on_the_parameters_dims(
    variable: xr.DataArray, name: str, dims: tuple[str, ...]
) -> None:
    """A variable is on exactly its parameter's batch and dim, which would
    otherwise be broadcast or dropped silently."""
    if set(variable.dims) != set(dims):
        raise ValueError(
            f"the parameter dataset's {name!r} is on {variable.dims}, but its parameter is on "
            f"{dims}; the two vectors do not define the parameter alike."
        )


def check_natural_values_are_a_mapping(natural_values: Any) -> None:
    """Natural values are a mapping by parameter name, which is how they are read."""
    if not isinstance(natural_values, Mapping):
        raise TypeError(
            f"natural values are a mapping of parameter names to arrays, got "
            f"{type(natural_values).__name__}."
        )


def check_natural_value_ends_in_the_shape(
    name: str, shape: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    """A natural value ends in its parameter's shape, which would otherwise
    broadcast silently."""
    if len(shape) < len(expected) or shape[len(shape) - len(expected):] != expected:
        raise ValueError(
            f"the natural value of {name!r} has shape {shape}, which does not end in the "
            f"parameter's {expected}."
        )


def check_natural_values_share_a_leading_shape(leads: Sequence[tuple[str, tuple[int, ...]]]) -> None:
    """Every natural value has one leading shape, one per draw."""
    shapes = {lead for _, lead in leads}
    if len(shapes) > 1:
        raise ValueError(
            f"the natural values have different leading shapes {dict(leads)}; give every "
            "parameter the same draws."
        )


def check_parameter_dataset_is_a_dataset(parameter_dataset: Any) -> None:
    """A labeled form is an ``xr.Dataset``, which is how it is read."""
    if not isinstance(parameter_dataset, xr.Dataset):
        raise TypeError(
            f"a parameter dataset is an xarray Dataset, got {type(parameter_dataset).__name__}; "
            "build it with ParameterVector.dataset."
        )


def check_parameter_dataset_variable_is_finite(name: str, values: np.ndarray) -> None:
    """The values read from a labeled form's variable are finite, as theta's
    always are."""
    if not np.isfinite(values).all():
        raise ValueError(
            f"the parameter dataset's {name!r} holds a missing or non-finite value, and no theta "
            "does; drop or fill those draws first."
        )


# ── the supports ──────────────────────────────────────────────────────────────
# Last, since building a Support runs its check, defined above.

#: The real line, per number; its bijector is the identity.
REAL = Support("real")

#: :math:`(0, \\infty)`; its bijector is :math:`\\exp`.
POSITIVE = Support("positive")

#: :math:`(0, 1)`, open, as pySIPNET's ``ParameterDomain.OPEN_UNIT_INTERVAL``
#: is; its bijector is ``tfb.Sigmoid()``.
OPEN_UNIT_INTERVAL = Support("interval", 0.0, 1.0)

#: The open simplex: ``k`` positive numbers summing to 1; its bijector is
#: ``tfb.SoftmaxCentered()``.
SIMPLEX = Support("simplex")
