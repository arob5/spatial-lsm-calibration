"""Figures for the 2026-09-25 progress presentation.

Each public function draws one figure from the library's loaders and plotting
functions and returns it; ``slides.qmd`` calls them. Nothing here saves a file:
the deck renders what is returned. What a slide shows as code is the library
call itself, so these functions hold only the plumbing around it -- marking the
featured sites, grouping by class, laying out panels.
"""

import textwrap
from collections.abc import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, ScalarFormatter

from sipnet_calibration.plotting import member_summary, plot_map_grid
from sipnet_calibration.plotting.style import category_colors
from sipnet_calibration.projection import SITE_PROJECTION
from sipnet_calibration.site_labels import load_site_labels, resolve_site_labels
from sipnet_calibration.sites import load_sites

import config


def mark_sites(ax: Axes, sites: Mapping[str, int] | None = None) -> Axes:
    """Mark and name *sites* on a map drawn by ``plot_map``.

    Parameters
    ----------
    ax:
        A map's axes, in the site projection's meters.
    sites:
        Display name to site id. Defaults to ``config.FEATURED_SITES``.

    Returns
    -------
    matplotlib.axes.Axes
        *ax*.
    """
    sites = config.FEATURED_SITES if sites is None else sites
    table = load_sites().set_index("site_id")
    for i, (name, site) in enumerate(sites.items()):
        x, y = SITE_PROJECTION.forward(table.at[site, "lon"], table.at[site, "lat"])
        ax.plot(x, y, marker="*", markersize=14, color="black", markeredgecolor="white", zorder=5)
        # Stagger the labels: the featured sites can be a few km apart.
        ax.annotate(
            name, (x, y), xytext=(10, 6 - 16 * i), textcoords="offset points", fontsize=11,
            fontweight="bold", zorder=5,
        )
    return ax


def class_counts(site_labels: str = config.SITE_LABELS) -> Figure:
    """Sites per class of a site-labels product, most sites first, in the map's colors.

    Parameters
    ----------
    site_labels:
        A site-labels name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    spec = resolve_site_labels(site_labels)
    counts = load_site_labels(spec)["label"].value_counts().reindex(spec.labels, fill_value=0)
    # A class's color is set by its position in the spec, as on the maps, so
    # the colors are reordered with the classes.
    order = np.argsort(-counts.to_numpy(), kind="stable")
    names = np.asarray(_display_names(spec))[order]
    colors = np.asarray(category_colors(len(names)))[order]
    figure, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1), layout="constrained")
    ax.barh(names, counts.to_numpy()[order], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("sites")
    ax.bar_label(ax.containers[0], padding=3, fontsize=9)
    return figure


def label_crosstab(
    rows: str = config.SITE_LABELS, columns: str = config.REANALYSIS_SITE_LABELS
) -> pd.DataFrame:
    """Sites per pair of classes of two site-labels products.

    Parameters
    ----------
    rows, columns:
        Site-labels names.

    Returns
    -------
    pandas.DataFrame
        Counts, one row per class of *rows* and one column per class of
        *columns*, both in their specs' order and by display name.
    """
    row_spec, column_spec = resolve_site_labels(rows), resolve_site_labels(columns)
    joined = load_site_labels(row_spec).merge(
        load_site_labels(column_spec), on="site_id", suffixes=("_rows", "_columns")
    )
    table = pd.crosstab(joined["label_rows"], joined["label_columns"], dropna=False)
    table.index = _display_names(row_spec)
    table.columns = _display_names(column_spec)
    table.index.name = None
    table.columns.name = None
    return table


def label_crosstab_heatmap(
    rows: str = config.SITE_LABELS, columns: str = config.REANALYSIS_SITE_LABELS
) -> Figure:
    """``label_crosstab`` drawn as an annotated heatmap.

    Parameters
    ----------
    rows, columns:
        Site-labels names.

    Returns
    -------
    matplotlib.figure.Figure
    """
    table = label_crosstab(rows, columns)
    counts = table.to_numpy()
    figure, ax = plt.subplots(
        figsize=(1.6 * table.shape[1] + 3.5, 0.34 * table.shape[0] + 1.2), layout="constrained"
    )
    ax.imshow(counts, cmap="Blues", aspect="auto")
    for (i, j), count in np.ndenumerate(counts):
        color = "white" if count > 0.6 * counts.max() else "black"
        ax.text(j, i, count, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=20, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel(resolve_site_labels(columns).name)
    ax.set_ylabel(resolve_site_labels(rows).name)
    return figure


def ensemble_summary_maps(field: xr.DataArray, **map_kwargs) -> Figure:
    """The ensemble median beside the ensemble standard deviation.

    Parameters
    ----------
    field:
        A ``(member, site)`` field.
    **map_kwargs:
        Passed to ``plot_map_grid``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = {
        "ensemble median": member_summary(field, "median"),
        "ensemble standard deviation": member_summary(field, "standard_deviation"),
    }
    for panel in panels.values():
        # The panel title says which statistic; the colorbar needs only units.
        panel.attrs["long_name"] = field.attrs.get("long_name", field.name)
    map_kwargs.setdefault("extent", config.MAP_EXTENT)
    map_kwargs.setdefault("robust", True)
    figure, axes = plot_map_grid(panels, scale="each", ncol=2, **map_kwargs)
    for ax in axes.flat:
        if ax.get_visible():
            mark_sites(ax)
            ax.set_title(ax.get_title(), pad=18)
    return figure


def negative_member_fraction(field: xr.DataArray) -> xr.DataArray:
    """The fraction of members below zero at each site.

    Parameters
    ----------
    field:
        A ``(member, site)`` field.

    Returns
    -------
    xarray.DataArray
        On ``site``, with ``lon``/``lat`` kept; ``NaN`` where every member is
        missing.
    """
    present = field.notnull().sum("member")
    fraction = (field < 0).sum("member") / present.where(present > 0)
    fraction.attrs = {
        "long_name": f"{field.attrs.get('long_name', field.name)}: fraction of members below zero",
        "units": "1",
    }
    fraction.name = f"{field.name}_negative_fraction"
    return fraction


def by_class(
    field: xr.DataArray, site_labels: str = config.SITE_LABELS, *, stat: str = "median",
    log: bool = False,
) -> Figure:
    """One box per class of the per-site ensemble statistic.

    Parameters
    ----------
    field:
        A ``(member, site)`` field.
    site_labels:
        The site-labels name to group by.
    stat:
        The per-site statistic over members, as ``member_summary`` takes it.
    log:
        A logarithmic value axis, for skewed stocks; values at or below zero
        are left out of the boxes and counted in the title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    spec = resolve_site_labels(site_labels)
    per_site = member_summary(field, stat).to_series().rename("value").rename_axis("site_id")
    joined = load_site_labels(spec).merge(per_site.reset_index(), on="site_id").dropna()
    dropped = int((joined["value"] <= 0).sum()) if log else 0
    if log:
        joined = joined[joined["value"] > 0]
    groups = [joined.loc[joined["label"] == label, "value"].to_numpy() for label in spec.labels]
    names = _display_names(spec)

    figure, ax = plt.subplots(figsize=(8, 0.38 * len(names) + 1.2), layout="constrained")
    boxes = ax.boxplot(groups, orientation="horizontal", tick_labels=names, patch_artist=True,
                       showfliers=False, widths=0.6,
                       medianprops={"color": "black"})
    for patch, color in zip(boxes["boxes"], category_colors(len(names))):
        patch.set_facecolor(color)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
    units = field.attrs.get("units", "")
    ax.set_xlabel(f"{field.attrs.get('long_name', field.name)}, per-site {stat} ({units})")
    if dropped:
        ax.set_title(f"{dropped} sites at or below zero left out of the log axis", fontsize=10)
    return figure


def members_at_sites(
    fields: Mapping[str, xr.DataArray], sites: Mapping[str, int] | None = None
) -> Figure:
    """Histograms of the ensemble members, one row per variable, one column per site.

    Parameters
    ----------
    fields:
        Name to ``(member, site)`` field.
    sites:
        Display name to site id. Defaults to ``config.FEATURED_SITES``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sites = config.FEATURED_SITES if sites is None else sites
    figure, axes = plt.subplots(
        len(fields), len(sites), figsize=(3.6 * len(sites), 2.0 * len(fields)),
        layout="constrained", squeeze=False,
    )
    for row, (name, field) in enumerate(fields.items()):
        for column, (site_name, site) in enumerate(sites.items()):
            ax = axes[row, column]
            values = field.sel(site=site).to_numpy()
            values = values[np.isfinite(values)]
            if values.size:
                ax.hist(values, bins=20, color="0.45")
                if values.min() < 0:
                    ax.axvline(0, color="firebrick", linewidth=0.8)
            else:
                ax.text(0.5, 0.5, "absent at this site", ha="center", va="center",
                        transform=ax.transAxes)
            if row == 0:
                ax.set_title(site_name)
            if column == 0:
                label = field.attrs.get("long_name", name).removeprefix("Initial ")
                ax.set_ylabel(textwrap.fill(label, 22), fontsize=9)
            ax.set_xlabel(field.attrs.get("units", ""), fontsize=9)
    return figure


def spec_table(specs: Iterable, fields: Sequence[str]) -> pd.DataFrame:
    """Chosen fields of a registry of specs, one row per spec.

    Parameters
    ----------
    specs:
        Spec dataclasses, such as ``CONSTRAINTS`` or ``INITIAL_CONDITIONS``.
    fields:
        The spec fields to show, in column order. ``name`` is the index.

    Returns
    -------
    pandas.DataFrame
        Blank where a spec leaves a field empty.
    """
    rows = {spec.name: {field: str(getattr(spec, field)) for field in fields} for spec in specs}
    return pd.DataFrame.from_dict(rows, orient="index")


def _display_names(spec) -> list[str]:
    if spec.display_names is None:
        return list(spec.labels)
    return [spec.display_names[label] for label in spec.labels]
