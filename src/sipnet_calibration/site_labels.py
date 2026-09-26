"""The data model for site labels: a class per site, one product per source.

Overview
--------
A *site-labels product* maps every site to a class. This module defines how one
is represented -- its columns, dtypes and the class names it may hold -- and
provides the functions for reading one and for parsing the raw file it is built
from. It is the single description of that layout: the ingest script that
writes a site-labels product gets its schema and its checks from here rather
than declaring its own.

It sits downstream of the one script that builds the products, and the
dependency runs one way::

    raw/site_labels/<raw_file>
      -> scripts/ingest_site_labels.py   processed/site_labels/<name>.csv
      -> this module                     load_site_labels(name) -> pandas.DataFrame

Site labels are **not** site metadata. Which site-labels product to use is an
experimental choice, so it is not a column of the site table:
:mod:`sipnet_calibration.sites` deliberately carries no plant functional type,
and a caller joins a site-labels product on before selecting. Several products
coexist, one file each, and a calibration names the one it used.

Input data
----------
``data/raw/site_labels/<spec.raw_file>``
    The producer's table, read by :func:`read_raw` in its own column names.
    ``data/raw/site_labels/provenance.md`` records where each came from.

``data/processed/site_labels/<name>.csv``
    The product, read by :func:`load_site_labels`, whose layout is the
    `Data model`_ below. :func:`site_labels_path` says where it is expected to
    be, honoring the ``$SIPNET_CALIBRATION_DATA`` override in
    :data:`~sipnet_calibration.conventions.DATA_ROOT_ENV_VAR`.

Data model
----------
:func:`load_site_labels` returns a ``pandas.DataFrame`` with one row per labeled
site, in ascending ``site_id`` order, holding the columns of
:data:`SITE_LABELS_COLUMNS`. ``site_id`` takes its dtype from
:data:`SITE_LABELS_COLUMN_DTYPES`, which both the read and the build impose;
``label`` is a categorical the spec builds, since its categories depend on
which product it is.

============= ================== ==============================================
Column        Dtype              Meaning
============= ================== ==============================================
``site_id``   ``int32``          site identifier, the ``site_id`` of the pool
``label``     ``category``       the class, categories being ``spec.labels``
============= ================== ==============================================

The categorical's categories are :attr:`SiteLabelsSpec.labels`, **in the spec's
order and always all of them**, whether or not every class is used. A class the
spec does not declare is an error on read, never a silently admitted new
category.

**Missing values.** None. A site that a product does not label is absent from
its file; there is no unlabeled class and no ``NaN``. Where
:attr:`SiteLabelsSpec.covers_pool` is set, every site in the pool is present, and
the ingest refuses a file where one is not.

**No other columns.** Not ``lon``/``lat``, not ``landcover``. Those are site
metadata, and duplicating them here is how a join key drifts from its table.
Join :func:`sipnet_calibration.sites.load_sites` on ``site_id``.

Functions
---------
:func:`load_site_labels`
    Read one processed site-labels product and check it against its spec.

:func:`site_labels_field`
    The same product as a ``(site,)`` array of CF flag codes with ``lon`` and
    ``lat``, the form the map plotting layer draws.

:func:`read_raw`
    Parse a product's raw file exactly, in its source column names.

:func:`build_site_labels`
    Turn a raw frame into the product the data model describes.

:func:`resolve_site_labels`
    The spec of a name, or a ``KeyError`` listing the names that exist.

:func:`site_labels_path`, :func:`default_raw_dir`, :func:`default_site_labels_dir`
    Where things are expected to be.

:func:`describe`
    A spec as readable prose, for a script's log.

:data:`SITE_LABELS`, :data:`SITE_LABELS_NAMES`
    The registry, and its keys in order.

Notes
-----
**The class column is ``label``, not ``pft``.** One schema across products, so
that :func:`load_site_labels` returns the same frame shape whatever it is asked
for and code that pools over classes -- a partial-pooling prior, a facet-by-class
figure -- indexes ``label`` without knowing which product it got. The product's
identity lives in its name, and :attr:`SiteLabelsSpec.label_kind` carries the
domain word for prose and axis labels. A product whose classes are not plant
functional types then costs no schema change.

**Class names are the producer's, verbatim.** ``boreal.coniferous`` and
``temperate.deciduous.HPDA`` are neither
``lower_case_with_underscores`` nor free of abbreviations, and are kept anyway:
they are the join key to the reanalysis's per-PFT trait sample tables, so
renaming them would be renaming a shared key. The project's naming convention
governs the names this project chooses -- the column, the product, the registry
key -- and the class values inside a product are data.

**The expected row count is a constant, not a measurement to look up.** Two
files named ``site_pft.csv`` sit one directory apart upstream, with the same
header and the same three class names, one covering the 8000-site pool and one
the older 6400-site pool. Nothing inside either announces which it is, so
:attr:`SiteLabelsSpec.expected_rows` is what tells them apart and the ingest
refuses a mismatch by name. See ``data/raw/site_labels/provenance.md``.

**A ``landcover_mapping`` is a discovered relation, not a definition.** Where
a product's classes turn out to be an exact function of the site table's
``landcover``, recording it means a regenerated raw file that broke the relation
is refused rather than ingested. It is optional: ``None`` for a product with no
such relation, and the check is then skipped.

Usage
-----
Read a site-labels product and join it to the site table::

    from sipnet_calibration.site_labels import load_site_labels
    from sipnet_calibration.sites import load_sites, select_sites

    labels = load_site_labels("reanalysis_3pft")
    sites = select_sites(load_sites(), bbox=EXTENTS["CONUS"]).merge(labels, on="site_id")
    for name, group in sites.groupby("label", observed=False):
        ...

List what exists, and what one is::

    from sipnet_calibration.site_labels import (
        SITE_LABELS_NAMES, describe, resolve_site_labels,
    )

    print(SITE_LABELS_NAMES)
    print(describe(resolve_site_labels("reanalysis_3pft")))
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration import conventions
from sipnet_calibration.conventions import (
    NAME_PATTERN,
    SITE,
    SITE_ATTRIBUTES,
    SITE_DTYPE,
    SITE_ID,
)
from sipnet_calibration.sites import load_sites, site_locations
from sipnet_calibration.validation import as_frozen_mapping

__all__ = [
    "LABEL_COLUMN",
    "SITE_LABELS",
    "SITE_LABELS_COLUMNS",
    "SITE_LABELS_COLUMN_DTYPES",
    "SITE_LABELS_NAMES",
    "SiteLabelsSpec",
    "build_site_labels",
    "default_raw_dir",
    "default_site_labels_dir",
    "describe",
    "label_dtype",
    "load_site_labels",
    "read_raw",
    "resolve_site_labels",
    "site_labels_field",
    "site_labels_path",
]

#: Column holding the class. Generic on purpose; see the module Notes.
LABEL_COLUMN = "label"

#: The product's columns, in order.
SITE_LABELS_COLUMNS = (SITE_ID, LABEL_COLUMN)

#: Dtype per column. ``label`` is not here because its categorical dtype
#: depends on the spec; :func:`label_dtype` builds it.
#:
#: Read-only: this is the schema, and a caller that mutated it would change what
#: every later read of a site-labels product produces.
SITE_LABELS_COLUMN_DTYPES = MappingProxyType({SITE_ID: SITE_DTYPE})


@dataclass(frozen=True)
class SiteLabelsSpec:
    """Everything a consumer needs to know about one site-labels product.

    One instance per raw file. The fields describe the classes, the raw file
    that carries them, and the invariants the ingest enforces.

    Notes
    -----
    The two mappings are stored as
    :class:`~sipnet_calibration.validation.FrozenMapping` copies, so a spec
    cannot change after its checks, and hashes and pickles like the other
    specs.
    """

    name: str
    """Processed name: the registry key and the output file's stem."""

    long_label: str
    """Plot-ready name, e.g. ``"Reanalysis three-PFT site labels"``."""

    label_kind: str
    """What the classes are, as a noun phrase: ``"plant functional type"``."""

    labels: tuple[str, ...]
    """The classes, verbatim from the producer, in the order they are indexed in.

    The order is the product's categorical order, so it is what a hierarchical
    prior's class axis is laid out in. Choose it for meaning rather than for
    frequency, and never reorder it once anything has been run against it.
    """

    description: str
    """What the product is and how the producer constructed it, with the citation."""

    product: str
    """The source the product came from, e.g. ``"NALCR 8000-site SDA"``."""

    raw_file: str
    """File name under ``data/raw/site_labels/``."""

    raw_columns: tuple[str, ...]
    """The raw file's header, in order; :func:`read_raw` refuses any other."""

    site_column: str
    """Raw column holding the site identifier."""

    label_column: str
    """Raw column holding the class."""

    expected_rows: int
    """Data rows the raw file must have. What tells two same-named files apart."""

    covers_pool: bool = True
    """Whether every site in the site table must be labeled."""

    landcover_mapping: Mapping[int, str] | None = None
    """Exact relation to the site table's ``landcover``, where one holds.

    ``None`` where the classes are not a function of ``landcover``, and the
    check is then skipped. Where set, it must be total over the ``landcover``
    values the pool actually uses and its values must all be in *labels*.
    """

    display_names: Mapping[str, str] | None = None
    """Plot-ready name per class, where the producer supplied one.

    The class values are the producer's join keys and are never renamed, so a
    figure or a table that wants readable labels looks them up here. ``None``
    where the class names are already readable.
    """

    comment: str = ""
    """Anything else a consumer must know before using the classes."""

    notes: tuple[str, ...] = field(default_factory=tuple)
    """Further caveats, one per string, for :func:`describe`."""

    def __post_init__(self) -> None:
        # frozen=True freezes the field, not the dict behind it.
        for name in ("landcover_mapping", "display_names"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, as_frozen_mapping(value, message_name=name))
        if not NAME_PATTERN.match(self.name):
            raise ValueError(f"Site-labels name {self.name!r} is not lower_case_with_underscores.")
        if not self.description or not self.long_label or not self.product:
            raise ValueError(
                f"Site labels {self.name!r} needs a description, long_label and product."
            )
        if not self.label_kind:
            raise ValueError(f"Site labels {self.name!r} needs a label_kind.")
        if len(self.labels) < 2:
            raise ValueError(f"Site labels {self.name!r}: a product needs at least two classes.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"Site labels {self.name!r}: labels repeats a class.")
        if any(not label for label in self.labels):
            raise ValueError(f"Site labels {self.name!r}: a class name is empty.")
        if len(set(self.raw_columns)) != len(self.raw_columns):
            raise ValueError(f"Site labels {self.name!r}: raw_columns repeats a column.")
        for role, column in (("site_column", self.site_column), ("label_column", self.label_column)):
            if column not in self.raw_columns:
                raise ValueError(
                    f"Site labels {self.name!r}: {role} {column!r} is not in raw_columns "
                    f"{self.raw_columns}."
                )
        if self.site_column == self.label_column:
            raise ValueError(
                f"Site labels {self.name!r}: site_column and label_column are the same."
            )
        if self.expected_rows <= 0:
            raise ValueError(f"Site labels {self.name!r}: expected_rows must be positive.")
        if self.display_names is not None:
            unknown = sorted(set(self.display_names) - set(self.labels))
            if unknown:
                raise ValueError(
                    f"Site labels {self.name!r}: display_names names {unknown}, which are "
                    f"not classes of this product."
                )
            absent = [label for label in self.labels if label not in self.display_names]
            if absent:
                raise ValueError(
                    f"Site labels {self.name!r}: display_names has no entry for {absent}. "
                    "Give every class a display name or none."
                )
        if self.landcover_mapping is not None:
            unknown = sorted(set(self.landcover_mapping.values()) - set(self.labels))
            if unknown:
                raise ValueError(
                    f"Site labels {self.name!r}: landcover_mapping sends cover classes to "
                    f"{unknown}, which are not in labels {list(self.labels)}."
                )


# ── the registry ──────────────────────────────────────────────────────────────

#: Every site-labels product, in registry order. One entry per raw file.
SITE_LABELS: tuple[SiteLabelsSpec, ...] = (
    SiteLabelsSpec(
        name="reanalysis_3pft",
        long_label="Reanalysis three-PFT site labels",
        label_kind="plant functional type",
        # Ordered by the landcover classes they aggregate, which is the order
        # the classes are generated in, rather than by how many sites each has.
        labels=(
            "boreal.coniferous",
            "temperate.deciduous.HPDA",
            "semiarid.grassland_HPDA",
        ),
        description=(
            "The three plant functional types the 8000-site North American Land Carbon "
            "Reanalysis assimilated under, one per site. Produced for that analysis by "
            "its authors; the file carries no record of how it was derived, but the "
            "classes are an exact aggregation of the site shapefile's eight landcover "
            "classes. Zhang et al. (2026), doi:10.3334/ORNLDAAC/2507."
        ),
        product="NALCR 8000-site SDA",
        raw_file="reanalysis_site_pft.csv",
        raw_columns=("site", "pft"),
        site_column="site",
        label_column="pft",
        expected_rows=8000,
        covers_pool=True,
        landcover_mapping=MappingProxyType(
            {
                1: "boreal.coniferous",
                2: "boreal.coniferous",
                3: "temperate.deciduous.HPDA",
                4: "temperate.deciduous.HPDA",
                5: "semiarid.grassland_HPDA",
                6: "semiarid.grassland_HPDA",
                7: "semiarid.grassland_HPDA",
                8: "semiarid.grassland_HPDA",
            }
        ),
        comment=(
            "Three classes over a pool spanning 7-82 degrees north, so the classes are "
            "coarse and two of the three names mislead: boreal.coniferous is neither "
            "boreal nor reliably coniferous (805 of its 2369 sites lie south of 40 N, "
            "and it holds a California annual grassland), and semiarid.grassland_HPDA "
            "is the catch-all for everything non-forest, 992 of its sites lying north "
            "of the Arctic Circle. A prior built on these labels inherits that "
            "coarseness."
        ),
        notes=(
            "The landcover relation is measured over all 8000 rows, not stated by the "
            "producer; see data/README.md open question 24(k).",
            "A finer 16-class product over the same pool is tracked as "
            "raw/site_labels/site_pft_16class.csv and is the one this project intends "
            "to calibrate under. The two do not nest: every one of its classes draws "
            "sites from at least two of these three. See data/README.md open "
            "question 11.",
        ),
    ),
    SiteLabelsSpec(
        name="pft_16class",
        long_label="Sixteen-class PFT site labels",
        label_kind="plant functional type",
        # Ordered by cover type -- needleleaf, broadleaf, mixed, shrub, open,
        # cropland, wetland -- rather than by site count, so that neighboring
        # classes in a prior's class axis are ecologically neighboring too.
        labels=(
            "Evergreen_Needleleaf_Forest__P1",
            "Evergreen_Needleleaf_Forest__P2",
            "Evergreen_Broadleaf_Forest",
            "Deciduous_Broadleaf_Forest__P1_P2_P3",
            "Deciduous_Broadleaf_Forest__P4_P5",
            "Mixed_Forest__P1",
            "Mixed_Forest__P2",
            "Open_Shrublands__P1",
            "Open_Shrublands__P2",
            "Open_Vegetation_Complex_P1",
            "Open_Vegetation_Complex_P2",
            "Open_Vegetation_Complex_P3",
            "Open_Vegetation_Complex_P4",
            "CroplandPool__Broad_Croplands",
            "CroplandPool__Cereal_Croplands",
            "Permanent_Wetlands",
        ),
        description=(
            "The sixteen plant functional types this project calibrates under, one per "
            "site. Assembled for the SIPNET calibration by a colleague in the Dietze "
            "lab and received on 2026-09-22, from MODIS land cover refined by clustering "
            "on climate, vegetation structure, soil and biogeography. 7637 sites were "
            "assigned directly and 363 by nearest median ecological profile."
        ),
        product="Dietze lab PFT assignment",
        raw_file="site_pft_16class.csv",
        raw_columns=(
            "index",
            "final_pft_direct",
            "final_source_type",
            "source_file",
            "pam_used_for_clustering",
            "pam_k",
            "pam_cluster",
            "subpft_name",
            "dbf_original_pam_cluster",
            "cropland_final_group",
            "final_pft",
            "final_pft_assignment_method",
            "final_pft_source_final",
            "nearest_ecological_distance",
            "second_nearest_final_pft",
            "second_nearest_ecological_distance",
            "distance_margin",
            "n_vars_used_in_distance",
        ),
        site_column="index",
        label_column="final_pft",
        expected_rows=8000,
        covers_pool=True,
        # Not a function of the shapefile's landcover: the classes come from a
        # different land cover product refined by clustering, so no exact
        # relation holds and none is asserted.
        landcover_mapping=None,
        display_names=MappingProxyType(
            {
                "Evergreen_Needleleaf_Forest__P1": "Open Cold-seasonal ENF",
                "Evergreen_Needleleaf_Forest__P2": "Closed Long-season ENF",
                "Evergreen_Broadleaf_Forest": "Evergreen Broadleaf Forest",
                "Deciduous_Broadleaf_Forest__P1_P2_P3": "Strongly Seasonal High C-N DBF",
                "Deciduous_Broadleaf_Forest__P4_P5": "Weakly Seasonal Low C-N DBF",
                "Mixed_Forest__P1": "Open Strongly Seasonal MF",
                "Mixed_Forest__P2": "Closed Weakly Seasonal MF",
                "Open_Shrublands__P1": "Cold Shrublands",
                "Open_Shrublands__P2": "Warm Shrublands",
                "Open_Vegetation_Complex_P1": "High latitude grassland",
                "Open_Vegetation_Complex_P2": "High seasonal open woodland",
                "Open_Vegetation_Complex_P3": "Greener open woodland",
                "Open_Vegetation_Complex_P4": "Arid grassland",
                "CroplandPool__Broad_Croplands": "Broad Croplands",
                "CroplandPool__Cereal_Croplands": "Cereal Croplands",
                "Permanent_Wetlands": "Permanent Wetlands",
            }
        ),
        comment=(
            "Does not nest inside reanalysis_3pft: every one of these classes draws "
            "sites from at least two of those three, and twelve from all three, so a "
            "prior cannot be transferred from the coarse product to this one by "
            "inheritance."
        ),
        notes=(
            "The raw file is one half of a 60-column table the producer assembled; the "
            "other half is the covariates at raw/covariates/, split by "
            "scripts/raw_sources/split_site_pft_16class.py. Those covariates are what "
            "the classes were derived from, so a model using both a class effect and "
            "them relates the two by construction.",
            "363 sites were assigned by nearest ecological profile rather than directly, "
            "and 292 of those have a runner-up class nearly as close as the one chosen; "
            "second_nearest_final_pft and distance_margin are kept in the raw file so "
            "that sensitivity can be measured.",
        ),
    ),
)

#: The site-labels names, in registry order.
SITE_LABELS_NAMES: tuple[str, ...] = tuple(spec.name for spec in SITE_LABELS)


def resolve_site_labels(name: str) -> SiteLabelsSpec:
    """The spec named *name*, or a ``KeyError`` listing the names that exist."""
    for spec in SITE_LABELS:
        if spec.name == name:
            return spec
    raise KeyError(f"No site labels named {name!r}. Known: {list(SITE_LABELS_NAMES)}")


# ── the product ───────────────────────────────────────────────────────────────


def default_raw_dir() -> Path:
    """Where the raw site-labels files are expected: ``data/raw/site_labels/``.

    ``$SIPNET_CALIBRATION_DATA`` replaces ``data/`` when set.
    """
    return _data_root() / "raw" / "site_labels"


def default_site_labels_dir() -> Path:
    """Where the processed products are expected: ``data/processed/site_labels/``.

    ``$SIPNET_CALIBRATION_DATA`` replaces ``data/`` when set.
    """
    return _data_root() / "processed" / "site_labels"


def site_labels_path(
    site_labels: str | SiteLabelsSpec, directory: Path | str | None = None
) -> Path:
    """The processed file of a site-labels product: ``<directory>/<name>.csv``."""
    name = site_labels if isinstance(site_labels, str) else site_labels.name
    base = Path(directory) if directory is not None else default_site_labels_dir()
    return base / f"{name}.csv"


def label_dtype(spec: SiteLabelsSpec) -> pd.CategoricalDtype:
    """The ``label`` column's dtype: *spec.labels* as unordered categories.

    Unordered because the classes have no ranking; the *order of the
    categories* is still the spec's, which is what fixes a class axis.
    """
    return pd.CategoricalDtype(categories=list(spec.labels), ordered=False)


def load_site_labels(
    site_labels: str | SiteLabelsSpec, path: Path | str | None = None
) -> pd.DataFrame:
    """Read one processed site-labels product and check it against its spec.

    Parameters
    ----------
    site_labels:
        A site-labels name from :data:`SITE_LABELS_NAMES`, or a spec.
    path:
        The CSV to read. Defaults to :func:`site_labels_path`.

    Returns
    -------
    pandas.DataFrame
        The columns of :data:`SITE_LABELS_COLUMNS`, in ascending ``site_id``
        order, with ``label`` a categorical over ``spec.labels``.

    Raises
    ------
    FileNotFoundError
        If the file is absent, with the command that produces it.
    ValueError
        If the file does not match the data model: a wrong header, a duplicate
        or non-ascending identifier, an identifier outside ``int32``, a missing
        value, or a class the spec does not declare.
    """
    spec = (
        site_labels
        if isinstance(site_labels, SiteLabelsSpec)
        else resolve_site_labels(site_labels)
    )
    path = Path(path) if path is not None else site_labels_path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Produce it with:\n"
            f"  python scripts/ingest_site_labels.py --site-labels {spec.name}"
        )

    try:
        with warnings.catch_warnings():
            # A row with more fields than the header only warns by default and
            # loses its trailing field; here that is a malformed file.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(
                path,
                # site_id is read wide and narrowed after checking, as
                # load_sites does: reading straight into int32 wraps silently.
                dtype={SITE_ID: np.int64, LABEL_COLUMN: str},
                keep_default_na=False,
                index_col=False,
            )
    except (ValueError, OverflowError, pd.errors.ParserWarning) as error:
        raise ValueError(f"{path}: could not be parsed as site labels: {error}") from error

    if tuple(frame.columns) != SITE_LABELS_COLUMNS:
        raise ValueError(
            f"{path}: header is {tuple(frame.columns)}, expected {SITE_LABELS_COLUMNS}."
        )
    if frame.empty:
        raise ValueError(f"{path}: holds no rows")
    _check_site_ids(frame[SITE_ID], path)
    _check_labels_are_declared(frame[LABEL_COLUMN], spec, path)

    return pd.DataFrame(
        {
            SITE_ID: frame[SITE_ID].astype(SITE_LABELS_COLUMN_DTYPES[SITE_ID]),
            LABEL_COLUMN: frame[LABEL_COLUMN].astype(label_dtype(spec)),
        }
    )


def site_labels_field(
    site_labels: str | SiteLabelsSpec,
    *,
    sites: pd.DataFrame | None = None,
    path: Path | str | None = None,
) -> xr.DataArray:
    """A site-labels product as a categorical field on ``site``.

    Parameters
    ----------
    site_labels:
        A site-labels name from :data:`SITE_LABELS_NAMES`, or a spec.
    sites:
        The site table, for ``lon``/``lat``. Defaults to
        :func:`sipnet_calibration.sites.load_sites`.
    path:
        The product to read. Defaults to :func:`site_labels_path`.

    Returns
    -------
    xarray.DataArray
        ``int8`` class codes on ``site`` (``int32``, ascending), one per labeled
        site, with ``lon`` and ``lat`` on ``site``. The attributes follow CF's
        convention for coded classes: ``flag_values`` is ``0 .. k-1`` and
        ``flag_meanings`` the spec's class names, space-separated, in the spec's
        order and all of them whether or not every class is used. There are no
        ``units``. ``long_name`` names the label kind and the product, and the
        array's name is the product's. Where the spec has
        :attr:`~SiteLabelsSpec.display_names`, ``flag_display_names`` holds
        them as a tuple aligned with ``flag_meanings``; it is this project's
        attribute, not CF's, and is absent otherwise.

    Raises
    ------
    KeyError
        If a labeled site is not in the site table.
    ValueError
        If a class name contains whitespace, which a CF flag meaning cannot.
    """
    spec = (
        site_labels
        if isinstance(site_labels, SiteLabelsSpec)
        else resolve_site_labels(site_labels)
    )
    labels = load_site_labels(spec, path)
    sites = load_sites() if sites is None else sites
    _check_labels_are_flag_meanings(spec)
    attrs = {
        "long_name": f"{spec.label_kind[:1].upper()}{spec.label_kind[1:]} ({spec.name})",
        "flag_values": np.arange(len(spec.labels), dtype=np.int8),
        "flag_meanings": " ".join(spec.labels),
    }
    if spec.display_names is not None:
        # A tuple, since a display name may contain spaces and flag_meanings
        # is space-separated.
        attrs["flag_display_names"] = tuple(spec.display_names[label] for label in spec.labels)
    return xr.DataArray(
        labels[LABEL_COLUMN].cat.codes.to_numpy(np.int8),
        dims=SITE,
        coords={
            SITE: (SITE, labels[SITE_ID].to_numpy(SITE_DTYPE), dict(SITE_ATTRIBUTES)),
            **site_locations(labels[SITE_ID].tolist(), sites),
        },
        attrs=attrs,
        name=spec.name,
    )


def read_raw(spec: SiteLabelsSpec, root: Path | str | None = None) -> pd.DataFrame:
    """Parse a product's raw file exactly, in its source column names.

    Parameters
    ----------
    spec:
        Which site-labels product.
    root:
        The directory holding the raw files. Defaults to :func:`default_raw_dir`.

    Returns
    -------
    pandas.DataFrame
        The columns of ``spec.raw_columns``, in order, the site column as
        ``int64`` and every other column as a string.

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    ValueError
        If the header is not ``spec.raw_columns``, or the file has no rows.

    Notes
    -----
    ``keep_default_na=False`` is what keeps a class literally named ``NA`` a
    string, as it does for the eight sites named ``NA`` in the site table. A
    site-labels product has no missing values, so nothing should become ``NaN``
    here and
    an empty field is caught downstream as an undeclared class rather than
    silently read as missing.

    The row count is **not** checked here. It is an invariant of the file
    rather than of parsing it, and the ingest script raises on it with a message
    naming the other file it might be; see ``data/raw/site_labels/provenance.md``.
    """
    path = (Path(root) if root is not None else default_raw_dir()) / spec.raw_file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see data/raw/site_labels/provenance.md")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(
                path,
                dtype=_raw_dtypes(spec),
                keep_default_na=False,
                index_col=False,
            )
    except (ValueError, OverflowError, pd.errors.ParserWarning) as error:
        raise ValueError(f"{path}: could not be parsed as its spec declares: {error}") from error
    if tuple(frame.columns) != spec.raw_columns:
        raise ValueError(
            f"{path}: header is {tuple(frame.columns)}, expected {spec.raw_columns}. "
            "A changed raw file is a spec change, not a new row."
        )
    if frame.empty:
        raise ValueError(f"{path}: holds no rows")
    return frame


def build_site_labels(spec: SiteLabelsSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw frame into the product the data model describes.

    Parameters
    ----------
    spec:
        Which site-labels product.
    frame:
        The raw frame, as :func:`read_raw` returns it.

    Returns
    -------
    pandas.DataFrame
        The columns of :data:`SITE_LABELS_COLUMNS`, in ascending ``site_id``
        order, with ``label`` a categorical over ``spec.labels``.

    Raises
    ------
    ValueError
        If an identifier is duplicated or outside ``int32``, or a class is not
        one the spec declares.

    Notes
    -----
    Renaming to the processed column names is safe here because the raw file
    names its columns: each record carries its own identity, so nothing is
    matched positionally. Checks that need the site table -- that every
    identifier is a real site, that the pool is covered, that the classes agree
    with ``landcover`` -- are the ingest script's, since this module does not
    read the site table.
    """
    site = frame[spec.site_column]
    label = frame[spec.label_column]
    _check_site_ids(site, spec.raw_file, sorted_required=False)
    _check_labels_are_declared(label, spec, spec.raw_file)
    built = pd.DataFrame(
        {
            SITE_ID: site.to_numpy(dtype=SITE_LABELS_COLUMN_DTYPES[SITE_ID]),
            LABEL_COLUMN: pd.Categorical(label, dtype=label_dtype(spec)),
        }
    )
    return built.sort_values(SITE_ID, ignore_index=True)


def describe(spec: SiteLabelsSpec) -> str:
    """A spec as readable prose, for a script's log."""
    lines = [
        f"{spec.name}: {spec.long_label}",
        f"  {spec.label_kind}, {len(spec.labels)} classes: {', '.join(spec.labels)}",
        f"  source: {spec.product}, raw/site_labels/{spec.raw_file}, {spec.expected_rows} rows",
        f"  {spec.description}",
    ]
    if spec.display_names is not None:
        width = max(len(label) for label in spec.labels)
        lines.append("  display names:")
        lines += [
            f"    {label:<{width}} : {spec.display_names[label]}" for label in spec.labels
        ]
    if spec.comment:
        lines.append(f"  comment: {spec.comment}")
    lines.extend(f"  note: {note}" for note in spec.notes)
    if spec.landcover_mapping is not None:
        groups: dict[str, list[int]] = {}
        for cover, label in sorted(spec.landcover_mapping.items()):
            groups.setdefault(label, []).append(cover)
        rule = "; ".join(
            f"landcover {_compact(covers)} -> {label}" for label, covers in groups.items()
        )
        lines.append(f"  landcover relation: {rule}")
    return "\n".join(lines)


# ── supporting helpers ────────────────────────────────────────────────────────


def _data_root() -> Path:
    return conventions.data_root()


def _raw_dtypes(spec: SiteLabelsSpec) -> dict[str, Any]:
    """What to hand pandas per column, so nothing is inferred."""
    # The site column is read wide and narrowed after checking; everything else
    # is a string, since a class name is text and must not be inferred numeric.
    dtypes: dict[str, Any] = {column: str for column in spec.raw_columns}
    dtypes[spec.site_column] = np.int64
    return dtypes


def _compact(values: list[int]) -> str:
    """``[1, 2]`` as ``"1-2"``, ``[5, 6, 7, 8]`` as ``"5-8"``, otherwise a list."""
    if len(values) > 1 and values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ", ".join(str(value) for value in values)


def _check_site_ids(site: pd.Series, source: object, *, sorted_required: bool = True) -> None:
    """Identifiers are positive, fit ``int32``, are unique, and are ascending."""
    if site.isna().any():
        raise ValueError(f"{source}: {SITE_ID} has a missing value.")
    high = np.iinfo(SITE_LABELS_COLUMN_DTYPES[SITE_ID]).max
    if site.min() < 1 or site.max() > high:
        raise ValueError(
            f"{source}: {SITE_ID} runs {site.min()}-{site.max()}, which is not a "
            f"positive int32. Site identifiers are the 1-8000 of the site table."
        )
    duplicated = site[site.duplicated()].unique()
    if duplicated.size:
        raise ValueError(
            f"{source}: {SITE_ID} repeats {duplicated[:5].tolist()}"
            f"{' and more' if duplicated.size > 5 else ''}. "
            "A site-labels product gives each site exactly one class."
        )
    if sorted_required and not site.is_monotonic_increasing:
        raise ValueError(f"{source}: {SITE_ID} is not in ascending order.")


def _check_labels_are_declared(label: pd.Series, spec: SiteLabelsSpec, source: object) -> None:
    """Every class is one the spec declares."""
    if label.isna().any():
        raise ValueError(
            f"{source}: {LABEL_COLUMN} has a missing value. A site-labels product has no "
            "unlabeled "
            "class; a site it does not label is absent from its file."
        )
    unknown = sorted(set(label.unique()) - set(spec.labels))
    if unknown:
        raise ValueError(
            f"{source}: holds classes {unknown} that {spec.name!r} does not declare. "
            f"Declared: {list(spec.labels)}. A new class is a spec change, not a new row."
        )


def _check_labels_are_flag_meanings(spec: SiteLabelsSpec) -> None:
    spaced = [label for label in spec.labels if re.search(r"\s", label)]
    if spaced:
        raise ValueError(
            f"class name(s) {spaced} of {spec.name!r} contain whitespace, which a CF "
            "flag_meanings entry cannot; the categorical field cannot represent them"
        )
