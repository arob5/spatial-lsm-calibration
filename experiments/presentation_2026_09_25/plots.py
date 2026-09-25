"""Figures for the 2026-09-25 progress presentation.

Each public function draws one figure from the library's loaders and plotting
functions and returns it; ``slides.qmd`` calls them. Nothing here saves a file:
the deck renders what is returned. What a slide shows as code is the library
call itself, so these functions hold only the plumbing around it -- grouping
by class, laying out panels.
"""

import calendar
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator, YearLocator
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator

from sipnet_calibration.constraints import CONSTRAINTS, constraint_fields
from sipnet_calibration.obs_ops import aggregate_time
from sipnet_calibration.plotting import (
    member_summary, plot_by_variable, plot_map_grid, plot_time_series,
)
from sipnet_calibration.plotting.style import category_colors
from sipnet_calibration.site_labels import load_site_labels, resolve_site_labels

import config


def class_counts(site_labels: str = config.SITE_LABELS) -> Figure:
    """Sites per class of a site-labels product, most sites first, in the map's colors.

    Each bar is labeled with its count and its share of all labeled sites.

    Parameters
    ----------
    site_labels:
        A site-labels name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    spec = resolve_site_labels(site_labels)
    labels = load_site_labels(spec)
    order = _classes_by_size(spec, labels)
    names, colors = _names_and_colors(spec, order)
    counts = labels["label"].value_counts().reindex(spec.labels, fill_value=0)
    sorted_counts = counts.to_numpy()[order]
    figure, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1), layout="constrained")
    ax.barh(names, sorted_counts, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("sites")
    labels = [_count_and_share(count, sorted_counts.sum()) for count in sorted_counts]
    ax.bar_label(ax.containers[0], labels=labels, padding=3, fontsize=9)
    ax.margins(x=0.18)  # room for the longest bar's label
    return figure


def driver_members(
    fields: Mapping[str, xr.DataArray], site: int, year: int = config.DRIVER_SERIES_YEAR
) -> Figure:
    """Every driver at one site through one year, daily, one curve per member.

    Parameters
    ----------
    fields:
        Variable name to ``(member, site, time)`` field, as
        ``sipnet_calibration.drivers.driver_fields`` returns.
    site:
        The site to show.
    year:
        The calendar year, by the start of each day.

    Returns
    -------
    matplotlib.figure.Figure
    """
    daily = {
        name: _days_of_year(aggregate_time(field.sel(site=site), "1D"), year)
        for name, field in fields.items()
    }
    panel = partial(plot_time_series, show="spaghetti", label="_nolegend_")
    figure, axes = _driver_panels(daily, panel)
    for ax in axes:
        ax.xaxis.set_major_locator(MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
    return figure


def driver_members_window(
    fields: Mapping[str, xr.DataArray],
    site: int,
    window: tuple[str, str] = config.DRIVER_SERIES_WINDOW,
) -> Figure:
    """Every driver at one site over a short window, at its own time step.

    Each member is its own color, so members that lie on top of one another at
    a coarser scale can be told apart.

    Parameters
    ----------
    fields:
        Variable name to ``(member, site, time)`` field, as
        ``sipnet_calibration.drivers.driver_fields`` returns.
    site:
        The site to show.
    window:
        The first and last day, inclusive, by the start of each step.

    Returns
    -------
    matplotlib.figure.Figure
    """
    windowed = {
        name: _steps_in_window(field.sel(site=site), window) for name, field in fields.items()
    }
    figure, axes = _driver_panels(windowed, _colored_members)
    for ax in axes:
        ax.xaxis.set_major_locator(DayLocator(interval=4))
        ax.xaxis.set_major_formatter(DateFormatter("%-d %b"))
    return figure


def load_driver_summary(stem: str) -> xr.Dataset:
    """One of the summaries ``precompute_drivers.py`` writes, by file stem.

    Parameters
    ----------
    stem:
        ``"driver_annual"`` or ``"driver_monthly_climatology"``.

    Returns
    -------
    xarray.Dataset
    """
    return xr.load_dataset(config.DRIVER_SUMMARY_DIR / f"{stem}.nc", engine="h5netcdf")


def driver_climatology_frames(field: xr.DataArray) -> xr.DataArray:
    """A ``(member, site, month)`` climatology as ensemble-mean maps, one per month.

    The month dimension is renamed for the variable and labeled by month name,
    so that ``animate_map`` titles each frame, for example, "Air temperature,
    July".

    Parameters
    ----------
    field:
        One variable of ``driver_monthly_climatology``.

    Returns
    -------
    xarray.DataArray
        On ``(site, <variable name>,)``, with ``lon``/``lat`` kept.
    """
    mean = field.mean("member", keep_attrs=True)
    summed = field.attrs.get("resampling", "").startswith("sum")
    mean.attrs["long_name"] = "ensemble mean, monthly total" if summed else "ensemble mean"
    dim = f"{_short_name(field)},"
    names = [calendar.month_name[month] for month in field["month"].to_numpy()]
    return mean.rename(month=dim).assign_coords({dim: names})


def driver_annual_means(variables: Iterable[str] = config.DRIVER_PFT_VARIABLES) -> dict:
    """Each driver's per-site mean over the years, from ``driver_annual``.

    Parameters
    ----------
    variables:
        The driver variables to take.

    Returns
    -------
    dict
        Name to ``(member, site)`` field, its ``long_name`` saying whether it
        is a yearly total or a yearly mean.
    """
    annual = load_driver_summary("driver_annual")
    fields = {}
    for name in variables:
        field = annual[name].mean("year", keep_attrs=True)
        summed = field.attrs.get("resampling", "").startswith("sum")
        field.attrs["long_name"] += ", yearly total" if summed else ", yearly mean"
        fields[name] = field
    return fields


def animation_columns(variables: Iterable[str] = config.DRIVER_ANIMATION_VARIABLES) -> str:
    """Markdown placing each variable's GIF side by side, for an ``asis`` cell.

    Parameters
    ----------
    variables:
        The variables whose GIFs ``make_driver_animations.py`` wrote.

    Returns
    -------
    str
    """
    variables = list(variables)
    width = f"{100 / len(variables):.0f}%"
    directory = config.DRIVER_ANIMATION_DIR.relative_to(config.EXPERIMENT_DIR)
    columns = [
        f'::: {{.column width="{width}"}}\n![]({directory / f"{name}.gif"})\n:::'
        for name in variables
    ]
    return ":::: {.columns}\n" + "\n".join(columns) + "\n::::"


def constraint_table() -> pd.DataFrame:
    """Each constraint product's quantity, units, time structure and time support.

    Returns
    -------
    pandas.DataFrame
        One row per product in ``CONSTRAINTS``, by display name.
    """
    rows = {
        config.CONSTRAINT_DISPLAY_NAMES[spec.name]: {
            "quantity": spec.long_label,
            "units": spec.units,
            "values": str(spec.time_structure),
            "each value covers": config.CONSTRAINT_TIME_SUPPORT[spec.name],
        }
        for spec in CONSTRAINTS
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def constraint_site_fields() -> dict[str, xr.DataArray]:
    """Every constraint product as one field, titled by its display name.

    Returns
    -------
    dict
        Product name to its ``(site, time)`` field, or ``(site,)`` for a static
        product, with ``long_name`` set to the display name.
    """
    fields = {}
    for spec in CONSTRAINTS:
        field = constraint_fields(spec.name)[spec.name]
        field.attrs["long_name"] = config.CONSTRAINT_DISPLAY_NAMES[spec.name]
        fields[spec.name] = field
    return fields


def constraint_maps(fields: Mapping[str, xr.DataArray], what: str = "mean") -> Figure:
    """One map per constraint product: its mean over time, or its number of dates.

    Parameters
    ----------
    fields:
        As ``constraint_site_fields`` returns.
    what:
        ``"mean"`` for each site's mean over its dates, or ``"count"`` for how
        many dates it has; a site with none is left blank either way.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels, colorbar_labels = {}, []
    for field in fields.values():
        if what == "mean":
            panel = _constraint_site_mean(field)
            colorbar_labels.append(f"mean ({field.attrs.get('units', '')})")
        else:
            panel = _constraint_date_count(field)
            colorbar_labels.append("dates")
        panels[field.attrs["long_name"]] = panel
    figure, axes = plot_map_grid(
        panels, scale="each", ncol=3, extent=config.MAP_EXTENT, robust=True,
        panel_size=(5.0, 3.9),
    )
    for ax, label in zip(axes, colorbar_labels):
        ax.set_title(ax.get_title(), pad=20)
        # The colorbar is the map's inset axes; a short label fits its height.
        for colorbar in ax.child_axes:
            colorbar.set_ylabel(label)
    return figure


def constraint_site_means(fields: Mapping[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Each constraint product's mean over its dates at each site, titled by product.

    Parameters
    ----------
    fields:
        As ``constraint_site_fields`` returns.

    Returns
    -------
    dict
        Product name to ``(site,)`` field.
    """
    means = {}
    for name, field in fields.items():
        mean = _constraint_site_mean(field)
        mean.attrs["long_name"] = field.attrs["long_name"]
        means[name] = mean
    return means


def constraint_series_at_sites(
    fields: Mapping[str, xr.DataArray], sites: Mapping[str, int] | None = None
) -> Figure:
    """Each dated constraint product over time, one row per site.

    Parameters
    ----------
    fields:
        As ``constraint_site_fields`` returns. Static products are left out.
    sites:
        Display name to site id. Defaults to ``config.FEATURED_SITES``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sites = config.FEATURED_SITES if sites is None else sites
    dated = {name: field for name, field in fields.items() if "time" in field.dims}
    figure, axes = plt.subplots(
        len(sites), len(dated), figsize=(3.6 * len(dated), 3.0 * len(sites)),
        layout="constrained", squeeze=False, sharex=True,
    )
    for column, field in enumerate(dated.values()):
        for row, (site_name, site) in enumerate(sites.items()):
            ax = axes[row, column]
            series = field.sel(site=site).dropna("time")
            if series.sizes["time"]:
                # Points, not a line: a line would join the gaps between dates
                # (MODIS LAI has only June to August) as if they were data.
                plot_time_series(
                    series, ax=ax, show="line", label="_nolegend_", marker="o", markersize=3,
                    linestyle="none",
                )
            else:
                ax.text(0.5, 0.5, "no data at this site", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(field.attrs["long_name"] if row == 0 else "")
            ax.set_ylabel(f"{site_name}\n{field.attrs.get('units', '')}" if column == 0
                          else field.attrs.get("units", ""))
            ax.set_xlabel("")
            ax.xaxis.set_major_locator(YearLocator(4))
            ax.xaxis.set_major_formatter(DateFormatter("%Y"))
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
    """``label_crosstab`` drawn as an annotated heatmap, with totals.

    Rows and columns are sorted by their totals, largest first. A last column
    and a last row give each class's total over the other
    product, as a count and a share of all sites; the corner is the number of
    sites. The colors are the counts of the table itself, not the totals.

    Parameters
    ----------
    rows, columns:
        Site-labels names.

    Returns
    -------
    matplotlib.figure.Figure
    """
    table = label_crosstab(rows, columns)
    table = table.iloc[
        np.argsort(-table.sum(axis=1).to_numpy(), kind="stable"),
        np.argsort(-table.sum(axis=0).to_numpy(), kind="stable"),
    ]
    counts = table.to_numpy()
    n_rows, n_columns = counts.shape
    total = counts.sum()
    figure, ax = plt.subplots(
        figsize=(1.6 * (n_columns + 1) + 3.5, 0.34 * (n_rows + 1) + 1.2), layout="constrained"
    )
    ax.imshow(counts, cmap="Blues", aspect="auto")
    for (i, j), count in np.ndenumerate(counts):
        color = "white" if count > 0.6 * counts.max() else "black"
        ax.text(j, i, count, ha="center", va="center", fontsize=9, color=color)
    for i, count in enumerate(counts.sum(axis=1)):
        ax.text(n_columns, i, _count_and_share(count, total), ha="center", va="center", fontsize=9)
    for j, count in enumerate(counts.sum(axis=0)):
        ax.text(j, n_rows, _count_and_share(count, total), ha="center", va="center", fontsize=9)
    ax.text(n_columns, n_rows, total, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.axvline(n_columns - 0.5, color="black", linewidth=0.8)
    ax.axhline(n_rows - 0.5, color="black", linewidth=0.8)
    ax.set_xlim(-0.5, n_columns + 0.5)
    ax.set_ylim(n_rows + 0.5, -0.5)
    ax.set_xticks(range(n_columns + 1), [*table.columns, "total"], rotation=20, ha="right")
    ax.set_yticks(range(n_rows + 1), [*table.index, "total"])
    ax.set_xlabel(resolve_site_labels(columns).name)
    ax.set_ylabel(resolve_site_labels(rows).name)
    return figure


def initial_condition_maps(fields: Mapping[str, xr.DataArray], stat: str = "median") -> Figure:
    """One map per initial condition of one statistic over its members.

    Parameters
    ----------
    fields:
        Name to ``(member, site)`` field, as ``initial_condition_fields``
        returns.
    stat:
        The statistic over members, as ``member_summary`` takes it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = {}
    for field in fields.values():
        summary = member_summary(field, stat)
        # The panel title names the variable, so the colorbar names the statistic.
        summary.attrs["long_name"] = stat.replace("_", " ")
        panels[textwrap.fill(_short_name(field), 30)] = summary
    figure, axes = plot_map_grid(
        panels, scale="each", ncol=3, extent=config.MAP_EXTENT, robust=True,
        panel_size=(5.0, 3.9),
    )
    for ax in axes:
        # Lift the title clear of the longitude labels along the map's top edge.
        ax.set_title(ax.get_title(), pad=20)
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
        "long_name": f"{_short_name(field)}: members below zero",
        "units": "fraction",
    }
    fraction.name = f"{field.name}_negative_fraction"
    return fraction


def by_class_grid(
    fields: Mapping[str, xr.DataArray],
    site_labels: str = config.SITE_LABELS,
    *,
    stat: str = "median",
    log: Iterable[str] = (
        "initial_aboveground_biomass_carbon", "initial_leaf_carbon",
        "initial_soil_organic_carbon",
    ),
) -> Figure:
    """One panel per field of boxes by class of the per-site ensemble statistic.

    Classes run most sites first, as in ``class_counts``, in the map's colors.

    Parameters
    ----------
    fields:
        Name to ``(member, site)`` field.
    site_labels:
        The site-labels name to group by.
    stat:
        The per-site statistic over members, as ``member_summary`` takes it.
    log:
        The fields drawn on a logarithmic axis, for skewed stocks; their values
        at or below zero are left out and counted under the axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    spec = resolve_site_labels(site_labels)
    labels = load_site_labels(spec)
    order = _classes_by_size(spec, labels)
    names, colors = _names_and_colors(spec, order)
    classes = [spec.labels[i] for i in order]
    log = set(log)
    figure, axes = plt.subplots(
        1, len(fields), sharey=True, figsize=(2.6 * len(fields) + 3.0, 0.36 * len(names) + 1.6),
        layout="constrained", squeeze=False,
    )
    for ax, (name, field) in zip(axes[0], fields.items()):
        _class_boxes(ax, field, labels, classes, colors, stat=stat, log=name in log)
    axes[0, 0].set_yticks(range(len(names)), names)
    axes[0, 0].invert_yaxis()
    return figure


def members_at_sites(
    fields: Mapping[str, xr.DataArray], sites: Mapping[str, int] | None = None
) -> Figure:
    """Histograms of the ensemble members, one row per site, one column per variable.

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
        len(sites), len(fields), figsize=(3.0 * len(fields), 3.0 * len(sites)),
        layout="constrained", squeeze=False,
    )
    for column, (name, field) in enumerate(fields.items()):
        for row, (site_name, site) in enumerate(sites.items()):
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
                ax.set_title(textwrap.fill(_short_name(field), 18))
            if column == 0:
                ax.set_ylabel(f"{site_name}\nmembers")
            ax.set_xlabel(field.attrs.get("units", ""))
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


def _constraint_site_mean(field: xr.DataArray) -> xr.DataArray:
    """A constraint field's mean over its dates at each site, units kept."""
    mean = field.mean("time", keep_attrs=True) if "time" in field.dims else field.copy()
    mean.attrs["long_name"] = "mean over dates" if "time" in field.dims else "value"
    return mean


def _constraint_date_count(field: xr.DataArray) -> xr.DataArray:
    """How many dates a constraint field has at each site; blank where none."""
    if "time" in field.dims:
        count = field.notnull().sum("time").astype(float)
    else:
        count = field.notnull().astype(float)
    count = count.where(count > 0)
    count.attrs = {"long_name": "dates", "units": "1"}
    return count


def _classes_by_size(spec, labels: pd.DataFrame) -> np.ndarray:
    """Positions of *spec*'s classes, most sites first."""
    counts = labels["label"].value_counts().reindex(spec.labels, fill_value=0)
    return np.argsort(-counts.to_numpy(), kind="stable")


def _names_and_colors(spec, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Display names and colors of *spec*'s classes, in *order*."""
    # A class's color is set by its position in the spec, as on the maps, so
    # the colors are reordered with the classes.
    names = np.asarray(_display_names(spec))[order]
    colors = np.asarray(category_colors(len(spec.labels)))[order]
    return names, colors


def _class_boxes(
    ax: Axes, field: xr.DataArray, labels: pd.DataFrame, classes: Sequence[str],
    colors: Sequence[str], *, stat: str, log: bool,
) -> None:
    """Boxes of *field*'s per-site statistic, one per class, at positions 0, 1, ..."""
    summary = member_summary(field, stat) if "member" in field.dims else field
    per_site = summary.to_series().rename("value").rename_axis("site_id")
    joined = labels.merge(per_site.reset_index(), on="site_id").dropna()
    dropped = int((joined["value"] <= 0).sum()) if log else 0
    if log:
        joined = joined[joined["value"] > 0]
    groups = [joined.loc[joined["label"] == label, "value"].to_numpy() for label in classes]
    boxes = ax.boxplot(
        groups, positions=range(len(groups)), orientation="horizontal", patch_artist=True,
        showfliers=False, widths=0.6, medianprops={"color": "black"},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
    if log:
        ax.set_xscale("log")
        low, high = ax.get_xlim()
        # Ticks at 1 and 3 of each decade on a short axis; decades only on a long one.
        subs = (1.0,) if np.log10(high / low) > 2.5 else (1.0, 3.0)
        ax.xaxis.set_major_locator(LogLocator(subs=subs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_title(textwrap.fill(_short_name(field), 18))
    units = field.attrs.get("units", "")
    note = f"\n{dropped} sites \u2264 0 not\nshown on log axis" if dropped else ""
    ax.set_xlabel(units + note)


def _short_name(field: xr.DataArray) -> str:
    """A field's ``long_name`` without the ``Initial`` every initial condition starts with."""
    name = field.attrs.get("long_name", field.name)
    name = name.removeprefix("Initial ")
    return name[:1].upper() + name[1:]


def _driver_panels(data: Mapping[str, xr.DataArray], panel) -> tuple[Figure, np.ndarray]:
    """One panel per driver in a 2 x 4 grid, with no legend and units on y."""
    figure, axes = plot_by_variable(
        dict(data), panel_fn=panel, ncol=4, panel_size=(3.6, 2.9), legend="none"
    )
    for ax, field in zip(axes, data.values()):
        # The panel title names the variable, so the y label need only say units.
        ax.set_ylabel(field.attrs.get("units", ""))
        ax.set_xlabel("")
    return figure, axes


def _colored_members(field: xr.DataArray, ax: Axes) -> Axes:
    """One ``(member, time)`` field as one curve per member, each its own color."""
    colors = plt.get_cmap("tab10").colors
    for i in range(field.sizes["member"]):
        plot_time_series(
            field.isel(member=i), ax=ax, color=colors[i % len(colors)], linewidth=1,
            label="_nolegend_",
        )
    return ax


def _steps_in_window(field: xr.DataArray, window: tuple[str, str]) -> xr.DataArray:
    """The steps of *field* that start on or between the two days of *window*."""
    start = field["time_step_start"].to_index()
    first, last = pd.Timestamp(window[0]), pd.Timestamp(window[1]) + pd.Timedelta(days=1)
    return field.isel(time=(start >= first) & (start < last))


def _days_of_year(field: xr.DataArray, year: int) -> xr.DataArray:
    """The daily cells of *field* that start in *year*."""
    # A daily cell's time is its end, so select by its start instead.
    return field.isel(time=(field["time_step_start"].dt.year == year).to_numpy())


def _count_and_share(count: int, total: int) -> str:
    """``"1681 (21.0%)"``: a count and its percentage of *total*."""
    return f"{count} ({100 * count / total:.1f}%)"


def _display_names(spec) -> list[str]:
    if spec.display_names is None:
        return list(spec.labels)
    return [spec.display_names[label] for label in spec.labels]
