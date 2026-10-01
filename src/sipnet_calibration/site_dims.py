"""The sites, and the dims they define: what each site is, for the adapter
layer.

Where this sits
---------------
::

    sites.select_sites, site_labels.load_site_labels   (the site table, the site labels)
      -> site_dims.SiteDims                           (each site's id, location, covariates, labels)
      -> parameters.ParameterVector(coords=site_dims.coords)
         DerivedParameter(constants=..., memberships=site_dims.labels(...)), PriorTerm(constants=...)
      -> sipnet_parameter_map, forward                (values read at the sites)

The parameter layer knows nothing of sites; :class:`SiteDims` is how a
calibration over sites talks to it. It gives the vector its coords, gives
derived and prior functions their constants, derived functions their
memberships, and reads labeled values at the sites.

What it reads
-------------
A site table, such as :func:`sipnet_calibration.sites.select_sites`
returns; for each site-labels data source in use, its site labels
(:func:`sipnet_calibration.site_labels.load_site_labels`); and the names of
the site covariates kept.

Data model
----------
**The dims.** ``site``, whose labels are the site ids (``int32``,
ascending), and one dim per site-labels name, whose labels are the classes
some site carries, in their declared order (a categorical's categories, or
sorted for plain labels). :attr:`SiteDims.coords` holds them as
``{dim: pd.Index}``, what a vector over these sites is built with.

**A membership** (:meth:`SiteDims.labels`) is an ``xr.DataArray`` of
strings, named for one dim, on another: each site's class along ``site``,
or each class's class under a coarser site-labels name (a PFT's biome).

**The values at the sites** (:meth:`SiteDims.at_sites`). Every variable of a
labeled Dataset is read at every site: a dim of the coords is replaced by
``site``, pointwise, so a value on ``(site, pft)`` is read at each site's own
PFT, and a variable on none of them is broadcast. The other dims are kept:
those with integer labels, or none, are batch dims and come first; those
with string labels are element axes and come last; so each variable is on
``(*batch dims, site, *element axes)``. The ``site`` coordinate carries
``lon`` and ``lat`` as :mod:`sipnet_calibration.conventions` defines them.

Functions and classes
---------------------
:class:`SiteDims`
    ``coords``, ``labels``, ``covariate``, ``at_sites``, ``site_fields``,
    ``select``.

Usage
-----
::

    site_dims = SiteDims(site_table=select_sites(load_sites(), site_ids=[620, 865, 1037]),
                         site_labels={"pft": load_site_labels("reanalysis_3pft")})
    vector = ParameterVector(parameters=[...], coords=site_dims.coords)
    site_dims.labels("pft")                     # each site's PFT, on site, named "pft"
    site_dims.at_sites(parameter_dataset)       # every value on (sample, site, ...)
    boreal = site_dims.select(pft=["boreal.coniferous"])
    vector.select(site=boreal.sites, pft=["boreal.coniferous"])
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.conventions import (
    LAT,
    LON,
    RESERVED_NAMES,
    SITE,
    SITE_DTYPE,
    SITE_ID,
    ReadOnlyCopies,
)
from sipnet_calibration.site_labels import LABEL_COLUMN
from sipnet_calibration.sites import (
    check_site_table_has_locations,
    check_site_table_is_keyed_on_site_ids,
    site_coordinates,
    site_lookup,
)
from sipnet_calibration.validation import (
    as_names,
    as_sequence,
    as_site_ids,
    check_names_are_unique,
    truncated,
)

__all__ = [
    "SiteDims",
    "check_names_are_not_reserved",
    "check_site_dims_are_valid",
]


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class SiteDims:
    """The sites, and the dims they define: each site's id, location,
    covariates and site labels.

    Parameters
    ----------
    site_table:
        The sites, one row each in ascending ``site_id``, with ``lon``,
        ``lat`` and the covariate columns, keyed on ``site_id`` or holding
        it as a column. Kept as a copy of ``site_id`` (``int32``), ``lon``,
        ``lat`` and the covariates, with a categorical column per
        site-labels name, whose categories are its declared classes.
    site_labels:
        ``{name: site labels}``: a site-labels table (``site_id`` and
        ``label``), such as :func:`~sipnet_calibration.site_labels.load_site_labels`
        returns, or a pandas Series keyed by site id, either labeling every
        site; a pandas categorical, one label per site, whose categories are
        the declared classes; or one string label per site, in site order.
    covariate_names:
        The columns of *site_table* kept as site covariates, each
        ``float64`` and finite.

    Raises
    ------
    TypeError
        If a covariate is not ``float64`` or a site label not a string.
    KeyError
        If a covariate is not a column of the site table, or a site-labels
        table does not label every site.
    ValueError
        If the site table holds no site or is not ascending, a value is not
        finite, a site-labels name or covariate is a reserved name
        (:data:`~sipnet_calibration.conventions.RESERVED_NAMES`) or names
        two things, or plain site labels are not one per site.
    """

    site_table: pd.DataFrame = ReadOnlyCopies()
    site_labels: Mapping[str, Any] = field(default_factory=frozendict)
    covariate_names: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "covariate_names", as_names(self.covariate_names, message_name="covariate_names"))
        table = _normalized_site_table(self.site_table, self.covariate_names)
        for name, value in dict(self.site_labels).items():
            table[name] = _site_labels_column(name, value, table[SITE_ID])
        object.__setattr__(self, "site_labels", frozendict({name: tuple(table[name].astype(str)) for name in self.site_labels}))
        object.__setattr__(self, "site_table", table)
        check_site_dims_are_valid(self)

    # ── identity ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SiteDims(sites={len(self.sites)}, site_labels={list(self.site_labels)}, "
            f"covariate_names={list(self.covariate_names)})"
        )

    @property
    def sites(self) -> tuple[int, ...]:
        """The site ids, ascending."""
        return tuple(int(s) for s in self._table[SITE_ID])

    @property
    def n_sites(self) -> int:
        """``S``, the number of sites."""
        return len(self._table)

    @property
    def coords(self) -> frozendict:
        """``{"site": the site ids (int32), <site-labels name>: the classes
        some site carries, in declared order}``: what a vector over these
        sites is built with."""
        coords = {SITE: pd.Index(self._table[SITE_ID].to_numpy(SITE_DTYPE), name=SITE)}
        for name in self.site_labels:
            column = self._table[name]
            present = set(column.astype(str))
            coords[name] = pd.Index([c for c in column.cat.categories if c in present], name=name)
        return frozendict(coords)

    # ── labels and constants ──────────────────────────────────────────────────

    def labels(self, dim: str, *, along: str = SITE) -> xr.DataArray:
        """A membership: a DataArray on *along*, named *dim*, holding each
        *along* label's *dim* label.

        Along ``"site"``, each site's class. Along another site-labels dim,
        the one *dim* label all of that label's sites carry, such as each
        PFT's biome.

        Raises
        ------
        KeyError
            If *dim* is not a site-labels name, or *along* is neither
            ``"site"`` nor one.
        ValueError
            If *dim* is ``"site"``, which labels nothing but itself, or is
            *along*; or if some *along* label's sites carry two *dim*
            labels, so the dims do not nest; the message names that label.
        """
        check_labeled_dim_is_a_site_labels_name(dim, self)
        check_along_is_a_dim(along, self)
        check_dims_differ(dim, along)
        if along == SITE:
            return xr.DataArray(
                np.asarray(self._table[dim].astype(str), dtype=object), dims=SITE,
                coords={SITE: self.coords[SITE].to_numpy()}, name=dim,
            )
        pairs = self._table[[along, dim]].astype(str).drop_duplicates()
        out = []
        for label in self.coords[along]:
            carried = pairs.loc[pairs[along] == label, dim].tolist()
            check_dims_nest(label, carried, along, dim)
            out.append(carried[0])
        return xr.DataArray(np.asarray(out, dtype=object), dims=along, coords={along: list(self.coords[along])}, name=dim)

    def covariate(self, name: str) -> xr.DataArray:
        """A site covariate on ``site``, ``float64``: a constant for a derived
        or prior function.

        Raises
        ------
        KeyError
            If *name* is not one of :attr:`covariate_names`.
        """
        check_covariate_is_held(name, self)
        return xr.DataArray(
            self._table[name].to_numpy(np.float64, copy=True), dims=SITE,
            coords={SITE: self.coords[SITE].to_numpy()}, name=name,
        )

    # ── the values at the sites ───────────────────────────────────────────────

    def at_sites(self, values: xr.Dataset) -> xr.Dataset:
        """Every variable read at every site, as the module's data model has
        it: a dim of :attr:`coords` replaced by ``site``, pointwise; a
        variable on none of them broadcast; batch dims first and element
        axes last; ``site``, ``lon`` and ``lat`` the site table's.

        Parameters
        ----------
        values:
            A labeled Dataset whose dims are dims of :attr:`coords`, batch
            dims (integer labels) and element axes (string labels); the
            dims of coords labeled.

        Returns
        -------
        xr.Dataset
            Each variable on ``(*batch dims, site, *element axes)``, its
            attributes kept.

        Raises
        ------
        TypeError
            If *values* is not an ``xr.Dataset``.
        KeyError
            If a variable lacks a label some site carries.
        ValueError
            If a dim of coords is not labeled.
        """
        check_values_are_a_dataset(values)
        coordinates = site_coordinates(self.sites, self._table)
        variables = {}
        for name, variable in values.data_vars.items():
            variables[str(name)] = self._variable_at_sites(str(name), variable)
        return xr.Dataset(variables).assign_coords(coordinates)

    def site_fields(self, values: xr.Dataset) -> xr.Dataset:
        """:meth:`at_sites`, with each value of rank 1 or more split into one
        field per element, ``<name>.<label>`` (``<name>.<label>.<label>``
        for rank 2, ...): the field contract's form, for plotting. Each
        carries its variable's attributes, ``units`` ``"1"`` where it has
        none, and ``element``, its element labels joined by ``"."``.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`at_sites`.
        """
        at_sites = self.at_sites(values)
        fields = {}
        for name, variable in at_sites.data_vars.items():
            elements = [d for d in variable.dims if _is_element_axis(variable, d)]
            attributes = {"units": "1", **variable.attrs}
            if not elements:
                fields[str(name)] = variable.assign_attrs(attributes)
                continue
            stacked = variable.stack(__element__=elements)
            for position, labels in enumerate(stacked["__element__"].values):
                labels = labels if isinstance(labels, tuple) else (labels,)
                field_name = ".".join([str(name), *map(str, labels)])
                fields[field_name] = (
                    stacked.isel(__element__=position, drop=True).assign_attrs({**attributes, "element": ".".join(labels)})
                )
        return xr.Dataset(fields).assign_coords(site_coordinates(self.sites, self._table))

    # ── selection ─────────────────────────────────────────────────────────────

    def select(self, sites: Sequence[int] | None = None, **labels: Sequence[str]) -> SiteDims:
        """The sites among *sites* (every one when ``None``) that carry one
        of the given labels on each named site-labels dim.

        ``select(pft=["boreal.coniferous"])`` is the sites of that PFT, and
        a forward model over them is built on it.

        Returns
        -------
        SiteDims
            Over the kept sites, in this one's order, with its covariates
            and site labels, the declared classes kept.

        Raises
        ------
        TypeError
            If a selector is one value rather than a sequence, a set or a
            mapping, or holds values of the wrong type.
        KeyError
            For an unknown site, site-labels name or label.
        ValueError
            For a value given twice, a selector keeping nothing, or no site
            remaining.
        """
        kept = np.ones(self.n_sites, dtype=bool)
        if sites is not None:
            requested = as_site_ids(sites, message_name="sites")
            check_sites_are_held(requested, self)
            kept &= self._table[SITE_ID].isin(requested).to_numpy()
        for name, wanted in labels.items():
            check_selector_is_not_site(name)
            check_labeled_dim_is_a_site_labels_name(name, self)
            wanted = as_sequence(wanted, message_name=f"select {name}=")
            check_names_are_unique(wanted, message_name=f"select {name}=")
            check_labels_are_the_dims(name, wanted, self.coords[name])
            kept &= self._table[name].astype(str).isin(wanted).to_numpy()
        check_selection_keeps_a_site(kept)
        table = self._table[kept]
        return SiteDims(
            site_table=table[[SITE_ID, LON, LAT, *self.covariate_names]],
            site_labels={name: pd.Categorical(table[name]) for name in self.site_labels},
            covariate_names=self.covariate_names,
        )

    # ── supporting methods ────────────────────────────────────────────────────

    @property
    def _table(self) -> pd.DataFrame:
        # ReadOnlyCopies keeps the table under this name and hands out
        # copies; the methods read it without one.
        return self.__dict__["_site_table"]

    def _variable_at_sites(self, name: str, variable: xr.DataArray) -> xr.DataArray:
        """One variable read at every site, on ``(*batch dims, site, *element axes)``."""
        on = [d for d in variable.dims if d in self.coords]
        indexers = {}
        for dim in on:
            check_dim_is_labeled(name, variable, str(dim))
            wanted = self.coords[SITE] if dim == SITE else self.labels(str(dim))
            held = set(variable.indexes[dim].tolist())
            missing = [label for label in dict.fromkeys(np.asarray(wanted).tolist()) if label not in held]
            check_variable_has_the_sites_labels(name, str(dim), missing)
            indexers[dim] = xr.DataArray(np.asarray(wanted), dims=SITE)
        variable = variable.drop_vars([c for c in variable.coords if c not in variable.dims])
        if indexers:
            variable = variable.sel(indexers)
            variable = variable.drop_vars([c for c in variable.coords if c not in variable.dims or c == SITE])
        else:
            variable = variable.expand_dims({SITE: self.n_sites})
        elements = [d for d in variable.dims if d != SITE and _is_element_axis(variable, d)]
        batch = [d for d in variable.dims if d != SITE and d not in elements]
        return variable.transpose(*batch, SITE, *elements).drop_vars(SITE, errors="ignore")


# ── helpers ───────────────────────────────────────────────────────────────────


def _is_element_axis(variable: xr.DataArray, dim: Any) -> bool:
    """Whether *dim* is an element axis: labeled, with string labels."""
    index = variable.indexes.get(dim)
    return index is not None and len(index) > 0 and all(isinstance(v, str) for v in index)


def _normalized_site_table(site_table: Any, covariate_names: tuple[str, ...]) -> pd.DataFrame:
    """``site_id`` (``int32``), ``lon``, ``lat`` and the named covariates of
    *site_table*, in its row order, after the site-table checks."""
    check_site_table_is_keyed_on_site_ids(site_table)
    check_site_table_has_locations(site_table)
    table = site_table if SITE_ID in site_table.columns else site_table.reset_index()
    site_ids = as_site_ids(table[SITE_ID].to_numpy(), message_name="the site table's site_id")
    check_site_table_has_a_site(site_ids)
    check_site_ids_are_ascending(np.asarray(site_ids))
    for name in covariate_names:
        check_covariate_is_a_column(name, table)
        check_covariate_is_float64(name, table[name])
    out = pd.DataFrame(
        {
            SITE_ID: np.asarray(site_ids, dtype=SITE_DTYPE),
            LON: table[LON].to_numpy(np.float64),
            LAT: table[LAT].to_numpy(np.float64),
            **{name: table[name].to_numpy(np.float64, copy=True) for name in covariate_names},
        }
    )
    check_site_table_values_are_finite(out, (LON, LAT, *covariate_names))
    return out


def _site_labels_column(name: str, value: Any, site_ids: pd.Series) -> pd.Categorical:
    """One site-labels argument as each site's label, a categorical whose
    categories are the declared classes."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        if isinstance(value, pd.DataFrame):
            check_site_labels_table_has_the_columns(name, value)
            indexed = site_lookup(value)[LABEL_COLUMN]
        else:
            indexed = value  # keyed by its index, which holds site ids
        check_site_labels_label_every_site(name, indexed.index, site_ids)
        labels = indexed.loc[site_ids.tolist()].astype(object).tolist()
        declared = tuple(indexed.cat.categories) if isinstance(indexed.dtype, pd.CategoricalDtype) else None
    elif isinstance(getattr(value, "dtype", None), pd.CategoricalDtype) or isinstance(value, pd.Categorical):
        categorical = pd.Categorical(value)
        labels, declared = categorical.tolist(), tuple(categorical.categories.tolist())
    else:
        labels = list(as_sequence(value, message_name=f"site_labels[{name!r}]"))
        declared = None
    check_site_labels_are_one_per_site(name, labels, len(site_ids))
    check_site_labels_are_strings(name, labels)
    if declared is None:
        declared = tuple(sorted(set(labels)))
    return pd.Categorical(labels, categories=list(declared))


# ── checks ────────────────────────────────────────────────────────────────────


def check_site_dims_are_valid(site_dims: SiteDims) -> None:
    """The site labels and covariates are named apart from each other and
    from the reserved names, which the labeled values' coordinates take."""
    names = {"site-labels name": tuple(site_dims.site_labels), "site covariate": site_dims.covariate_names}
    for what, taken in names.items():
        check_names_are_not_reserved(taken, what)
    check_names_are_distinct(names)


def check_names_are_not_reserved(names: Sequence[str], what: str) -> None:
    """No name is one of :data:`~sipnet_calibration.conventions.RESERVED_NAMES`,
    which the labeled values' dims and coordinates take."""
    for name in names:
        if name in RESERVED_NAMES:
            raise ValueError(
                f"the {what} {name!r} is a reserved name; rename it to something other than "
                f"{sorted(RESERVED_NAMES)}."
            )


def check_names_are_distinct(names_by_kind: Mapping[str, Sequence[str]]) -> None:
    """No site-labels name is also a site covariate, since both are read by name."""
    seen: dict[str, str] = {}
    for what, names in names_by_kind.items():
        for name in names:
            if name in seen:
                raise ValueError(f"{name!r} is both a {seen[name]} and a {what}; rename one.")
            seen[name] = what


def check_labeled_dim_is_a_site_labels_name(dim: Any, site_dims: SiteDims) -> None:
    """A dim whose labels are asked for is a site-labels name; ``site``
    labels nothing but itself."""
    if dim == SITE:
        raise ValueError("'site' labels nothing but itself; ask for a site-labels name's labels.")
    if dim not in site_dims.site_labels:
        raise KeyError(f"there are no site labels {dim!r}; name one of {list(site_dims.site_labels)}.")


def check_selector_is_not_site(name: str) -> None:
    """Sites are selected by ``sites=``, the selector of site ids, not by a
    site-labels name."""
    if name == SITE:
        raise ValueError("select sites by their ids with sites=[...], not site=.")


def check_along_is_a_dim(along: Any, site_dims: SiteDims) -> None:
    """A membership runs along ``site`` or a site-labels name."""
    if along != SITE and along not in site_dims.site_labels:
        raise KeyError(f"{along!r} is neither 'site' nor a site-labels name ({list(site_dims.site_labels)}).")


def check_dims_differ(dim: str, along: str) -> None:
    """A membership maps one dim to another."""
    if dim == along:
        raise ValueError(f"labels({dim!r}, along={along!r}) maps a dim to itself; name two dims.")


def check_dims_nest(label: str, carried: Sequence[str], along: str, dim: str) -> None:
    """The sites of each label of one dim carry exactly one label of the
    other, so a value on the second can be read at the first's labels."""
    if len(carried) != 1:
        raise ValueError(
            f"the sites of {along} {label!r} carry the {dim} labels {sorted(carried)}, so {along!r} does "
            f"not nest in {dim!r}; pool over dims that nest."
        )


def check_covariate_is_held(name: Any, site_dims: SiteDims) -> None:
    """A covariate asked for is one of the kept covariates."""
    if name not in site_dims.covariate_names:
        raise KeyError(f"there is no site covariate {name!r}; name one of {list(site_dims.covariate_names)}.")


def check_values_are_a_dataset(values: Any) -> None:
    """Values read at the sites are an ``xr.Dataset``."""
    if not isinstance(values, xr.Dataset):
        raise TypeError(f"values read at the sites are an xarray Dataset, got {type(values).__name__}.")


def check_dim_is_labeled(name: str, variable: xr.DataArray, dim: str) -> None:
    """A dim read at the sites is labeled, since xarray would otherwise read
    it by position."""
    if dim not in variable.indexes:
        raise ValueError(f"{name!r} has no {dim!r} coordinate, so it cannot be read at the sites; label it.")


def check_variable_has_the_sites_labels(name: str, dim: str, missing: Sequence[Any]) -> None:
    """A variable has a value at every label the sites carry."""
    if missing:
        raise KeyError(f"{name!r} has no value at the {dim} label(s) {truncated(missing)} some site carries.")


def check_sites_are_held(sites: Sequence[int], site_dims: SiteDims) -> None:
    """Every site asked of ``select`` is one of the site dims'."""
    held = set(site_dims.sites)
    unknown = [s for s in sites if s not in held]
    if unknown:
        raise KeyError(f"there is no site {truncated(unknown)}; select from {truncated(list(site_dims.sites))}.")


def check_labels_are_the_dims(name: str, wanted: Sequence[Any], labels: pd.Index) -> None:
    """A selector holds labels of its dim, strings, at least one."""
    wrong = [w for w in wanted if not isinstance(w, str)]
    if wrong:
        raise TypeError(f"the labels of {name!r} are strings, got {truncated(wrong)}.")
    if not wanted:
        raise ValueError(f"select {name}=[] keeps nothing; name at least one, or omit the selector.")
    unknown = [w for w in wanted if w not in set(labels)]
    if unknown:
        raise KeyError(f"{name!r} has no label {truncated(unknown)} at these sites; name some of {truncated(list(labels))}.")


def check_selection_keeps_a_site(kept: np.ndarray) -> None:
    """A selection keeps at least one site."""
    if not kept.any():
        raise ValueError("the selection keeps no site; widen sites= or the labels.")


def check_site_table_has_a_site(site_ids: Sequence[int]) -> None:
    """The site table holds at least one site; an empty one fails far from its cause."""
    if not site_ids:
        raise ValueError("the site table holds no site; give at least one.")


def check_site_ids_are_ascending(site_ids: np.ndarray) -> None:
    """The site table lists its sites in ascending ``site_id``, which is the
    ``site`` dim's order."""
    if np.any(np.diff(site_ids) <= 0):
        raise ValueError(
            "the site table's site_id must be ascending; sort it with site_table.sort_values('site_id')."
        )


def check_covariate_is_a_column(name: str, table: pd.DataFrame) -> None:
    """A named site covariate is a column of the site table."""
    if name not in table.columns:
        raise KeyError(
            f"site covariate {name!r} is not a column of the site table; join it on first, or drop it "
            "from covariate_names."
        )


def check_covariate_is_float64(name: str, column: pd.Series) -> None:
    """A named site covariate is ``float64``: a code or a string is no
    covariate until it is deliberately converted."""
    if column.dtype != np.float64:
        raise TypeError(
            f"site covariate {name!r} is {column.dtype}, not float64; convert it deliberately (a class "
            "code is not a quantity), or drop it from covariate_names."
        )


def check_site_table_values_are_finite(table: pd.DataFrame, names: Sequence[str]) -> None:
    """``lon``, ``lat`` and every site covariate are finite at every site."""
    for name in names:
        values = table[name].to_numpy()
        if not np.isfinite(values).all():
            sites = table[SITE_ID].to_numpy()[~np.isfinite(values)].tolist()
            raise ValueError(
                f"the site table's {name!r} is not finite at site(s) {truncated(sites)}; drop those sites "
                "or fill the column first."
            )


def check_site_labels_table_has_the_columns(name: str, table: pd.DataFrame) -> None:
    """A site-labels table has ``site_id`` and ``label``."""
    missing = [c for c in (SITE_ID, LABEL_COLUMN) if c not in table.columns]
    if missing:
        raise KeyError(f"site labels {name!r} lack the column(s) {missing}; pass the table load_site_labels() returns.")


def check_site_labels_label_every_site(name: str, labeled: pd.Index, site_ids: pd.Series) -> None:
    """A site-labels table labels every site."""
    missing = sorted(set(site_ids.tolist()) - set(labeled.tolist()))
    if missing:
        raise KeyError(f"site labels {name!r} give no label for site(s) {truncated(missing)}; select sites they cover.")


def check_site_labels_are_one_per_site(name: str, labels: Sequence[Any], n_sites: int) -> None:
    """Plain site labels are one per site."""
    if len(labels) != n_sites:
        raise ValueError(
            f"site labels {name!r} hold {len(labels)} labels for {n_sites} sites; give one per site, in "
            "site order."
        )


def check_site_labels_are_strings(name: str, labels: Sequence[Any]) -> None:
    """Every site's label is a string: a missing label leaves a site on no
    class, and a number would be read as a site id."""
    bad = [label for label in labels if not isinstance(label, str)]
    if bad:
        raise TypeError(
            f"site labels {name!r} hold labels that are not strings, such as {bad[0]!r}; every site needs "
            "a string label."
        )
