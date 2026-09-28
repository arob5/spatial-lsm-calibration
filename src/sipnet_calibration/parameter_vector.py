"""The parameter vector: the unknowns of a calibration, the sites they are
defined over, and the coordinates a sampler moves in.

Where this sits
---------------
::

    sites.select_sites, site_labels.load_site_labels    (the site table, the dim labels)
      -> parameter_vector.ParameterVector               (what is calibrated)
      -> prior.Prior                                    (what is believed beforehand)
      -> sipnet_parameter_map, forward                  (how a value reaches SIPNET)

This module imports NumPy, pandas, xarray, JAX and TFP's bijectors, and
neither pySIPNET, pyEKI nor TFP's distributions: a prior over the vector
lives in :mod:`sipnet_calibration.prior`, and the map to SIPNET parameters
in :mod:`sipnet_calibration.sipnet_parameter_map`.

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

A value takes three forms:

**theta** (Flat). ``float64``, ``(D,)`` or ``(J, D)``, with
:math:`D = \\sum_p n_p e_p` and :math:`\\theta_p = T_p^{-1}(x_p)`: parameters
in declaration order; within one, dim labels in :meth:`~ParameterVector.dim_index`
order; within a dim label, unconstrained names in order.
:attr:`ParameterVector.index` names every entry.

**Natural values** (:data:`NaturalValues`). ``{parameter name: array}``,
each ``(..., *value_shape)``, in the parameter's units and support:

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
             site ids with ``lon``/``lat``; a site-labels dim, its dim labels
             (strings); each with the attributes of
             :mod:`sipnet_calibration.conventions`
variables    one ``float64`` variable per natural name of every parameter:
             ``<name>`` for a scalar, ``<name>.<natural name>`` for a vector,
             on ``(batch?, dim?)``
attributes   ``parameter``; ``natural_name`` (a vector's); ``units``
             (omitted when ``None``); ``support``
missing      never
============ ===============================================================

Its index of a site-labels dim equals :meth:`ParameterVector.dim_index`. It is
not a collection of fields: a site-labels dim is not a field dim.
:meth:`ParameterVector.site_fields` is the per-site view, which is.

Functions and classes
---------------------
:class:`ParameterVector`
    Identity, selection, the coordinate maps (``to_natural``,
    ``to_unconstrained``, ``at_sites``) and the labeled form (``dataset``,
    ``flat``, ``site_fields``).
:class:`Parameter`, :class:`Support`
    One unknown, and the open set it lives in: :data:`REAL`,
    :data:`POSITIVE`, :data:`OPEN_UNIT_INTERVAL`, :data:`SIMPLEX` and
    :func:`OpenInterval`.
:func:`probe_points`
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
from collections.abc import Iterator, Mapping, Sequence
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
    check_sites_are_the_vectors,
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
    "NaturalValues",
    "OpenInterval",
    "Parameter",
    "ParameterDataset",
    "ParameterVector",
    "Support",
    "check_batch_dim_name_is_not_taken",
    "check_parameter_vector_is_valid",
    "check_parameter_vectors_share_a_layout",
    "probe_points",
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
    site_table:
        The sites, one row each in ascending ``site_id``, with ``lon`` and
        ``lat``, such as :func:`sipnet_calibration.sites.select_sites`
        returns. Read, not kept: :attr:`site_table` is the vector's own.
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
        The columns of *site_table* a prior function or SIPNET rule may read,
        each ``float64`` and finite. No other column is kept.

    Raises
    ------
    TypeError
        If a site label, or a named site covariate, has the wrong type.
    KeyError
        If a named site covariate is not a column of *site_table*.
    ValueError
        If the parameters, sites and site labels do not make one vector; the
        message names the rule.
    """

    parameters: Sequence[Parameter]
    # A keyword only: the attribute of the same name is the property attached
    # below the class, so the dataclass must not read it as a default.
    site_table: InitVar[pd.DataFrame]
    site_labels: Mapping[str, Any] = field(default_factory=frozendict)
    site_covariate_names: Sequence[str] = ()
    _table: pd.DataFrame = field(init=False, repr=False)
    _declared_dim_labels: Mapping[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self, site_table: pd.DataFrame) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(
            self,
            "site_covariate_names",
            as_names(self.site_covariate_names, message_name="site_covariate_names"),
        )
        table = _normalized_site_table(site_table, self.site_covariate_names)
        declared = {}
        for name, value in dict(self.site_labels).items():
            table[name], declared[name] = _site_labels_column(name, value, table[SITE_ID])
        object.__setattr__(self, "site_labels", frozendict(self.site_labels))
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
        return (
            f"ParameterVector(D={self.dimension}, parameters={list(self.parameter_names)}, "
            f"sites={self.n_sites})"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The parameter names, in declaration order."""
        return tuple(p.name for p in self.parameters)

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
        """The dims the parameters vary over, in the order they first appear."""
        return tuple(dict.fromkeys(p.dim for p in self.parameters if p.dim is not None))

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

    def value_shape(self, parameter_name: str) -> tuple[int, ...]:
        """The shape of one natural value of a parameter: ``(n?, k?)``."""
        parameter = self[parameter_name]
        return self._dim_shape(parameter) + parameter.natural_shape

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
        """One row per parameter, indexed by ``parameter``: ``dim``,
        ``dim_labels`` (their number), ``support``, ``bijector``, ``units``,
        ``natural_size``, ``unconstrained_size`` and ``entries`` (of theta)."""
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
            }
            for p in self.parameters
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
            the site covariates and site labels are carried.

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
        return ParameterVector(
            parameters=[self[n] for n in names],
            site_table=table[[SITE_ID, LON, LAT, *self.site_covariate_names]],
            site_labels=site_labels,
            site_covariate_names=self.site_covariate_names,
        )

    # ── coordinates ───────────────────────────────────────────────────────────

    def to_natural(self, theta: Any) -> NaturalValues:
        """Theta to natural values: :math:`x_p = T_p(\\theta_p)` for every parameter.

        :math:`\\theta_p` is theta at ``positions(parameter_name=p)``, shaped
        ``(..., *unconstrained_shape(p))``, and :math:`T_p` is
        ``parameter.bijector``, applied per dim label. Traceable under
        ``jax.jit``, ``jax.grad`` and ``jax.vmap``.

        Parameters
        ----------
        theta:
            ``(..., D)``.

        Returns
        -------
        NaturalValues
            ``{name: (..., *value_shape(name))}``, ``float64``.

        Raises
        ------
        ValueError
            If the last axis of *theta* is not ``D`` long.
        """
        theta = jnp.asarray(theta, dtype=jnp.float64)
        check_theta_ends_in_the_dimension(theta.shape, self.dimension)
        lead = theta.shape[:-1]
        # Derived parameters, computed from these, will be added here.
        return {
            p.name: p.bijector.forward(
                theta[..., self._slices[p.name]].reshape(lead + self.unconstrained_shape(p.name))
            )
            for p in self.parameters
        }

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
            pieces.append(p.bijector.inverse(values).reshape(lead + (-1,)))
        return jnp.concatenate(pieces, axis=-1)

    def at_sites(self, natural_values: NaturalValues) -> NaturalValues:
        """Every value read at every site: :math:`x_p^{(s)} = x_p[\\ell_{d_p}(s)]`,
        or :math:`x_p` for a parameter without a dim.

        Parameters
        ----------
        natural_values:
            :data:`NaturalValues` with one leading shape.

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
        return {p.name: self._at_sites(p, natural_values[p.name]) for p in self.parameters}

    # ── the labeled form ──────────────────────────────────────────────────────

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
        for p in self.parameters:
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
            a larger vector's projects onto this one. Its rows keep their
            order.

        Returns
        -------
        jax.Array
            ``(J, D)`` when it has a batch dim, ``(D,)`` otherwise.

        Raises
        ------
        TypeError, ValueError
            As :func:`validate_parameter_dataset`; ``ValueError`` too for a
            variable on other dims than its parameter, or a value outside
            its support.
        KeyError
            For a missing variable or dim label.
        """
        validate_parameter_dataset(parameter_dataset)
        batch = batch_dims(parameter_dataset)
        pieces = []
        for p in self.parameters:
            values = jnp.asarray(self._natural_values_of(p, parameter_dataset, batch))
            check_values_are_in_the_support(p, values)
            pieces.append(p.bijector.inverse(values).reshape(values.shape[: len(batch)] + (-1,)))
        return jnp.concatenate(pieces, axis=-1)

    def site_fields(self, parameter_dataset: ParameterDataset) -> xr.Dataset:
        """The labeled form read at every site: one field per variable, on
        ``(batch?, site)``, carrying the variable's attributes, with ``units``
        ``"1"`` for a parameter without physical units. Not invertible.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`flat`.
        """
        validate_parameter_dataset(parameter_dataset)
        batch = batch_dims(parameter_dataset)
        coordinates = site_coordinates(self.sites, self._table)
        if batch:
            coordinates[batch[0]] = parameter_dataset[batch[0]]
        variables = {}
        for p in self.parameters:
            values = self._natural_values_of(p, parameter_dataset, batch)
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

    def _dim_shape(self, parameter: Parameter) -> tuple[int, ...]:
        return (len(self.dim_index(parameter.dim)),) if parameter.dim else ()

    def _at_sites(self, parameter: Parameter, values: Any) -> Array:
        """One parameter's natural values, ``(..., *value_shape)``, at every site."""
        values = jnp.asarray(values)
        lead_ndim = values.ndim - len(self.value_shape(parameter.name))
        if parameter.dim is not None:
            return jnp.take(values, self._site_positions(parameter.dim), axis=lead_ndim)
        values = jnp.expand_dims(values, lead_ndim)
        shape = values.shape[:lead_ndim] + (self.n_sites,) + parameter.natural_shape
        return jnp.broadcast_to(values, shape)

    def _site_positions(self, dim_name: str) -> np.ndarray:
        """``(S,)``: each site's position along ``dim_index(dim_name)``."""
        # A public, tested form of this is the planned site_positions helper.
        if dim_name == SITE:
            return np.arange(self.n_sites)
        return self._table[dim_name].cat.codes.to_numpy(np.int64)

    def _selected_site_mask(
        self, sites: Sequence[int] | None, dim_labels: Mapping[str, Sequence[str]] | None
    ) -> np.ndarray:
        kept = np.ones(self.n_sites, dtype=bool)
        if sites is not None:
            requested = as_site_ids(sites, message_name="sites")
            check_sites_are_the_vectors(requested, self.sites, message_name="the vector")
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
                labels = list(self.dim_index(parameter.dim))
                variable = variable.sel({parameter.dim: labels})
            columns.append(np.asarray(variable.transpose(*dims).values, dtype=np.float64))
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

    def contains(self, values: Any) -> Array:
        """Whether each value lies in the support, finite values only.

        Elementwise for ``real``, ``positive`` and ``interval``; over the
        last axis for the simplex, ``(..., k) -> (...)``, whose values must
        be positive and sum to 1 within ``1e-10``.
        """
        values = jnp.asarray(values, dtype=jnp.float64)
        finite = jnp.isfinite(values)
        if self.kind == "real":
            return finite
        if self.kind == "positive":
            return finite & (values > 0)
        if self.kind == "interval":
            return finite & (values > self.low) & (values < self.high)
        inside = jnp.all(finite & (values > 0), axis=-1)
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
    shape = tuple(unconstrained_shape)
    size = int(np.prod(shape, dtype=int))
    points = [np.zeros(size)]
    points += [sign * c * np.ones(size) for c in (3.0, 10.0, 20.0) for sign in (1.0, -1.0)]
    for i in range(min(unconstrained_size, size)):
        for c in (10.0, 20.0):
            for sign in (1.0, -1.0):
                point = np.zeros(size)
                point[i] = sign * c
                points.append(point)
    directions = np.random.default_rng(_PROBE_SEED).standard_normal((4, size))
    points += list(10.0 * directions / np.linalg.norm(directions, axis=1, keepdims=True))
    return np.asarray(points, dtype=np.float64).reshape((len(points), *shape))


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
    """Check that *natural_values* are :data:`NaturalValues` of the vector:
    a value for every parameter, of its value shape (its shape at sites
    when *at_sites*), with one leading shape.

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
    for p in parameter_vector.parameters:
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
    can read: at most one batch dim, and every variable finite.

    Raises
    ------
    TypeError
        If it is not an ``xr.Dataset``.
    ValueError
        If it has more than one batch dim, or a variable holds a missing or
        non-finite value.
    """
    check_parameter_dataset_is_a_dataset(parameter_dataset)
    check_at_most_one_batch_dim(batch_dims(parameter_dataset), message_name="the parameter dataset")
    for name, variable in parameter_dataset.data_vars.items():
        check_parameter_dataset_variable_is_finite(str(name), variable)


# ── private helpers ───────────────────────────────────────────────────────────

#: How far a simplex value's sum may be from 1: float64 rounding of a
#: ``SoftmaxCentered`` image is about ``1e-16`` per number.
_SIMPLEX_SUM_TOLERANCE = 1e-10

#: The seed of :func:`probe_points`' random directions.
_PROBE_SEED = 20260926


def _bijectors_agree(first: Parameter, second: Parameter) -> bool:
    probes = jnp.asarray(probe_points(first.unconstrained_shape, unconstrained_size=first.unconstrained_size))
    return bool(
        np.allclose(
            first.bijector.forward(probes), second.bijector.forward(jnp.array(probes)), rtol=1e-12
        )
    )


def _in_closure(support: Support, values: Array) -> Array:
    """Whether finite *values* lie in the closure of *support*, as
    :meth:`Support.contains` reads them."""
    finite = jnp.isfinite(values)
    if support.kind == "real":
        return finite
    if support.kind == "positive":
        return finite & (values >= 0)
    if support.kind == "interval":
        return finite & (values >= support.low) & (values <= support.high)
    inside = jnp.all(finite & (values >= 0), axis=-1)
    return inside & (jnp.abs(values.sum(axis=-1) - 1.0) <= _SIMPLEX_SUM_TOLERANCE)


def _same_label(label: Any, wanted: Any) -> bool:
    """Whether dim label *label* is *wanted*: strings match strings, and
    integers (site ids) match integers, never a boolean."""
    if isinstance(label, str):
        return isinstance(wanted, str) and label == wanted
    return isinstance(wanted, (int, np.integer)) and not isinstance(wanted, bool) and label == wanted


def _dataset_variables(parameter: Parameter) -> list[tuple[str, dict[str, Any]]]:
    """``(variable name, attributes)`` for each natural name of *parameter*."""
    names = (None,) if parameter.natural_names is None else parameter.natural_names
    out = []
    for natural_name in names:
        attributes: dict[str, Any] = {"parameter": parameter.name}
        if natural_name is not None:
            attributes["natural_name"] = natural_name
        if parameter.units is not None:
            attributes["units"] = parameter.units
        attributes["support"] = parameter.support.name
        variable_name = parameter.name if natural_name is None else f"{parameter.name}.{natural_name}"
        out.append((variable_name, attributes))
    return out


def _normalized_site_table(site_table: Any, site_covariate_names: tuple[str, ...]) -> pd.DataFrame:
    """``site_id`` (``int32``), ``lon``, ``lat`` and the named covariates of
    *site_table*, in its row order, after the site-table checks."""
    check_site_table_is_keyed_on_site_ids(site_table)
    check_site_table_has_locations(site_table)
    table = site_table if SITE_ID in site_table.columns else site_table.reset_index()
    check_site_ids_are_ascending(table[SITE_ID].to_numpy())
    for name in site_covariate_names:
        check_site_covariate_is_a_column(name, table)
        check_site_covariate_is_float64(name, table[name])
    out = pd.DataFrame(
        {
            SITE_ID: table[SITE_ID].to_numpy(SITE_DTYPE),
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
    if isinstance(value, pd.DataFrame):
        check_site_labels_table_has_the_columns(name, value)
        indexed = site_lookup(value)[LABEL_COLUMN]
        check_site_labels_label_every_site(name, indexed.index, site_ids)
        labels = indexed.loc[site_ids.tolist()]
        declared = _declared_labels(indexed)
        labels = labels.tolist()
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
    """The parameters, site table and site labels make one vector."""
    check_vector_has_a_parameter(parameter_vector.parameters)
    check_parameter_names_are_unique(parameter_vector.parameter_names)
    names = {
        "site covariate": parameter_vector.site_covariate_names,
        "site-labels name": tuple(parameter_vector.site_labels),
    }
    for what, taken in names.items():
        check_names_are_not_reserved(taken, what)
    check_names_are_distinct(parameter_vector.parameter_names, names)
    for parameter in parameter_vector.parameters:
        check_parameter_dim_is_the_vectors(parameter, parameter_vector)


def check_parameter_is_valid(parameter: Parameter) -> None:
    """A parameter's name, natural names and bijector are usable."""
    check_name_is_usable(parameter.name, "a parameter")
    if parameter.natural_names is not None:
        for natural_name in parameter.natural_names:
            check_name_is_usable(natural_name, f"a natural name of {parameter.name!r}", reserved=False)
        check_names_are_unique(parameter.natural_names, message_name=f"{parameter.name!r} natural_names")
    check_simplex_has_two_natural_names(parameter)
    if parameter._custom_bijector:
        check_bijector_maps_onto_the_support(parameter)


def check_parameter_vectors_share_a_layout(first: ParameterVector, second: ParameterVector) -> None:
    """Two vectors give theta one meaning: the same index, sites, dim indexes,
    supports, and transforms that agree at the probe points. Derived
    parameters and site covariates are not compared."""
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
            _bijectors_agree(p, q) for p, q in zip(first.parameters, second.parameters)
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
    taken = {
        *parameter_vector.site_labels,
        *(name for p in parameter_vector.parameters for name, _ in _dataset_variables(p)),
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
    which values are passed by; and, when *reserved*, none of
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


def check_parameter_dim_is_the_vectors(parameter: Parameter, parameter_vector: ParameterVector) -> None:
    """A parameter's dim is ``"site"`` or a site-labels name of the vector."""
    if parameter.dim is not None and parameter.dim != SITE and parameter.dim not in parameter_vector.site_labels:
        raise ValueError(
            f"parameter {parameter.name!r} varies over {parameter.dim!r}, which is neither "
            f"'site' nor a site-labels name of the vector ({sorted(parameter_vector.site_labels)}); "
            "give its site labels as site_labels={...}."
        )


def check_simplex_has_two_natural_names(parameter: Parameter) -> None:
    """A simplex parameter names at least two natural numbers."""
    if parameter.support.kind == "simplex" and parameter.natural_size < 2:
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
    into = _in_closure(support, parameter.bijector.forward(probes))
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
            f"dim_label={dim_label!r} needs dim= or parameter_name=, since two dims may share a "
            "label."
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
        raise ValueError(
            f"site labels {name!r} lack the column(s) {missing}; pass the table "
            "load_site_labels() returns."
        )


def check_site_labels_label_every_site(name: str, labeled: pd.Index, site_ids: pd.Series) -> None:
    """A site-labels table labels every site of the vector."""
    missing = sorted(set(site_ids.tolist()) - set(labeled.tolist()))
    if missing:
        raise ValueError(
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


def check_values_are_in_the_support(parameter: Parameter, values: Array) -> None:
    """Natural values read from the labeled form lie in their support."""
    inside = parameter.support.contains(values)
    if not bool(jnp.all(inside)):
        raise ValueError(
            f"parameter {parameter.name!r} holds values outside its support "
            f"{parameter.support.name!r}; no theta maps to them."
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
    if not isinstance(parameter_dataset, xr.Dataset):
        raise TypeError(
            f"a parameter dataset is an xarray Dataset, got {type(parameter_dataset).__name__}; "
            "build it with ParameterVector.dataset."
        )


def check_parameter_dataset_variable_is_finite(name: str, variable: xr.DataArray) -> None:
    """A labeled form's variable holds no missing or non-finite value."""
    values = np.asarray(variable.values, dtype=np.float64)
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
