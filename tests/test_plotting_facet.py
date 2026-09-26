"""Tests for :mod:`sipnet_calibration.plotting.facet`."""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sipnet_calibration.plotting.facet import (
    SHARE_OPTIONS,
    build_plot_grid,
    plot_by_site,
    plot_by_variable,
)
from sipnet_calibration.plotting.series import plot_time_series


def draw_nothing(ax, item):
    """A panel callback that draws nothing."""


def draw_line(ax, item):
    """A panel callback that draws one labeled curve of height *item*."""
    ax.plot([0, 1], [0, item], label=f"line {item}")


# ── the grid ──────────────────────────────────────────────────────────────────


def test_the_grid_shape_follows_ncol():
    """Seven items at ``ncol=3`` give three rows."""
    figure, axes = build_plot_grid(range(7), draw_nothing, ncol=3)
    assert len(figure.axes) == 9
    assert len(axes) == 7


def test_ncol_is_capped_at_the_number_of_items():
    """Two items at ``ncol=3`` do not leave a third of the figure blank."""
    figure, _ = build_plot_grid([1, 2], draw_nothing, ncol=3)
    assert len(figure.axes) == 2


def test_unused_axes_are_hidden():
    """The padding axes of the last row are not visible."""
    figure, axes = build_plot_grid(range(4), draw_nothing, ncol=3)
    hidden = [a for a in figure.axes if not a.get_visible()]
    assert len(hidden) == 2
    assert all(a.get_visible() for a in axes)


def test_the_returned_axes_align_with_the_items():
    """One axes per item, in order, with the hidden ones left out."""
    items = [3, 1, 2]
    _, axes = build_plot_grid(items, draw_line, ncol=2)
    assert len(axes) == len(items)
    for ax, item in zip(axes, items):
        assert ax.lines[0].get_ydata()[1] == item


def test_the_figure_size_follows_panel_size_and_the_grid():
    """A six-panel grid is not the size of a two-panel one."""
    figure, _ = (
        build_plot_grid(range(6), draw_nothing, ncol=3, panel_size=(2.0, 1.0))
    )
    np.testing.assert_allclose(figure.get_size_inches(), (6.0, 2.0))


def test_the_callback_is_called_once_per_item_in_order():
    """``panel_fn(ax, item)`` sees the items in the order given."""
    seen = []
    build_plot_grid(["c", "a", "b"], lambda ax, i: seen.append(i))
    assert seen == ["c", "a", "b"]


def test_the_callback_return_is_ignored(field_time):
    """A callback returning an ``Axes`` -- as the plotting functions do -- is
    fine.
    """
    _, axes = (
        build_plot_grid([field_time], lambda ax, f: plot_time_series(f, ax=ax))
    )
    assert len(axes[0].lines) == 1


# ── sharing ───────────────────────────────────────────────────────────────────


def test_share_y_gives_one_common_y_limit():
    """With ``share="y"`` every panel ends with the same y limits."""
    _, axes = build_plot_grid([1, 1000], draw_line, ncol=2, share="y")
    assert axes[0].get_ylim() == axes[1].get_ylim()


def test_share_none_leaves_the_limits_independent():
    """Panels with different data ranges keep different limits."""
    _, axes = build_plot_grid([1, 1000], draw_line, ncol=2, share="none")
    assert axes[0].get_ylim() != axes[1].get_ylim()


def test_an_unknown_share_is_rejected():
    """The message lists :data:`SHARE_OPTIONS`."""
    with pytest.raises(ValueError, match="share must be one of") as raised:
        build_plot_grid([1], draw_nothing, share="diagonal")
    assert all(mode in str(raised.value) for mode in SHARE_OPTIONS)


# ── titles ────────────────────────────────────────────────────────────────────


def test_labels_as_a_sequence_title_the_panels():
    """Each panel's title is the corresponding element."""
    _, axes = (
        build_plot_grid([1, 2], draw_nothing, labels=["first", "second"])
    )
    assert [a.get_title() for a in axes] == ["first", "second"]


def test_labels_as_a_callable_title_the_panels():
    """The callable is applied to each item."""
    _, axes = build_plot_grid([1, 2], draw_nothing, labels=lambda i: f"p{i}")
    assert [a.get_title() for a in axes] == ["p1", "p2"]


def test_no_labels_means_no_titles():
    """``labels=None`` leaves every panel untitled."""
    _, axes = build_plot_grid([1, 2], draw_nothing)
    assert [a.get_title() for a in axes] == ["", ""]


def test_a_single_string_for_labels_is_rejected():
    """A bare string would title the panels one character each."""
    with pytest.raises(ValueError, match="one character each"):
        build_plot_grid([1, 2, 3, 4], draw_nothing, labels="Site")


def test_a_labels_sequence_of_the_wrong_length_is_rejected():
    """A mismatch raises rather than titling some panels."""
    with pytest.raises(ValueError, match="they must match"):
        build_plot_grid([1, 2], draw_nothing, labels=["only one"])


# ── legend ────────────────────────────────────────────────────────────────────


def two_roles(ax, item, field):
    """Draw the same two roles onto every panel."""
    plot_time_series(field, ax=ax, role="prior")
    plot_time_series(field, ax=ax, role="observation", show="points")


def test_dedup_keeps_one_entry_per_label(field_time):
    """Panels drawing the same two roles give a legend with two entries."""
    figure, _ = (
        build_plot_grid([1, 2, 3], partial(two_roles, field=field_time))
    )
    assert len(figure.legends) == 1
    assert [t.get_text() for t in figure.legends[0].get_texts()] == ["prior", "observation"]


def test_dedup_keeps_first_seen_order():
    """The legend order is the order the labels first appeared."""

    def panel(ax, item):
        for name in item:
            ax.plot([0, 1], [0, 1], label=name)

    figure, _ = build_plot_grid([["b", "a"], ["a", "c"]], panel)
    assert [t.get_text() for t in figure.legends[0].get_texts()] == ["b", "a", "c"]


def test_dedup_keeps_the_first_handle_for_a_label():
    """The handle kept is the first seen, not the last."""

    def panel(ax, item):
        ax.plot([0, 1], [0, 1], color=item, label="shared")

    figure, _ = build_plot_grid(["#111111", "#eeeeee"], panel)
    (handle,) = figure.legends[0].legend_handles
    assert handle.get_color() == "#111111"


def test_legend_each_adds_nothing_to_an_unlabeled_panel():
    """``legend="each"`` leaves a panel with no labeled artists alone."""
    figure, axes = build_plot_grid([1, 2], draw_nothing, legend="each")
    assert all(ax.get_legend() is None for ax in axes)


def test_the_dedup_legend_sits_outside_the_panels(field_time):
    """A legend drawn over the panels would hide the data it describes."""
    figure, axes = (
        build_plot_grid([1, 2], partial(two_roles, field=field_time))
    )
    figure.canvas.draw()
    legend_box = figure.legends[0].get_window_extent()
    for ax in axes:
        assert not legend_box.overlaps(ax.get_window_extent())


def test_legend_none_draws_no_legend(field_time):
    """No figure legend and no panel legends."""
    figure, axes = (
        build_plot_grid([1, 2], partial(two_roles, field=field_time), legend="none")
    )
    assert figure.legends == []
    assert all(a.get_legend() is None for a in axes)


def test_legend_each_gives_every_panel_its_own(field_time):
    """Each panel carries a legend and the figure carries none."""
    figure, axes = (
        build_plot_grid([1, 2], partial(two_roles, field=field_time), legend="each")
    )
    assert figure.legends == []
    assert all(a.get_legend() is not None for a in axes)


def test_nothing_labeled_means_no_legend():
    """A grid whose panels label nothing gets no empty legend box."""
    figure, _ = build_plot_grid([1, 2], draw_nothing)
    assert figure.legends == []


def test_an_unknown_legend_is_rejected():
    """An unrecognized legend mode raises."""
    with pytest.raises(ValueError, match="legend must be one of"):
        build_plot_grid([1], draw_nothing, legend="footnote")


# ── input validation ──────────────────────────────────────────────────────────


def test_empty_items_is_rejected():
    """An empty sequence raises rather than returning a blank figure."""
    with pytest.raises(ValueError, match="nothing to draw"):
        build_plot_grid([], draw_nothing)


def test_a_non_positive_ncol_is_rejected():
    """``ncol=0`` raises."""
    with pytest.raises(ValueError, match="at least 1"):
        build_plot_grid([1], draw_nothing, ncol=0)


# ── plot_by_site ──────────────────────────────────────────────────────────────


def test_plot_by_site_draws_one_panel_per_site(field_sample_site_time):
    """One panel per site, and each panel's data are that site's slice."""
    _, axes = plot_by_site(field_sample_site_time, ncol=2)
    assert len(axes) == field_sample_site_time.sizes["site"]
    for ax, site in zip(axes, field_sample_site_time["site"].values):
        expected = field_sample_site_time.sel(site=site).median("sample").values
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)


def test_plot_by_site_titles_the_panels_with_the_site_ids(
    field_sample_site_time
):
    """The default labels name the site."""
    _, axes = plot_by_site(field_sample_site_time, ncol=2)
    assert [a.get_title() for a in axes] == [
        f"site {site}" for site in field_sample_site_time["site"].values
    ]


def test_plot_by_site_selects_and_orders_by_the_sites_given(
    field_sample_site_time
):
    """``sites=`` picks a subset, in the order given."""
    sites = list(field_sample_site_time["site"].values)[::-1]
    _, axes = plot_by_site(field_sample_site_time, sites=sites, ncol=2)
    assert [a.get_title() for a in axes] == [f"site {site}" for site in sites]


def test_plot_by_site_accepts_a_bound_plotting_function(
    field_sample_site_time
):
    """A partial of ``plot_time_series`` reaches the panel with its keywords."""
    _, axes = (
        plot_by_site(
            field_sample_site_time,
            partial(plot_time_series, show="spaghetti"),
            ncol=2,
        )
    )
    assert len(axes[0].lines) == field_sample_site_time.sizes["sample"]


def test_a_failing_callback_does_not_leak_a_figure():
    """The figure is closed when a panel callback raises partway through."""

    def explode(ax, item):
        if item == 2:
            raise RuntimeError("boom")

    before = set(plt.get_fignums())
    with pytest.raises(RuntimeError, match="boom"):
        build_plot_grid([1, 2, 3], explode)
    assert set(plt.get_fignums()) == before


def test_plot_by_site_rejects_a_site_dim_without_a_site_coordinate():
    """A ``site`` dimension with no ids cannot name or select panels."""
    import xarray as xr

    data = xr.DataArray(
        np.zeros((2, 4)),
        dims=("site", "time"),
        coords={"time": np.arange(4)},
        attrs={"units": "u", "long_name": "L"},
    )
    with pytest.raises(ValueError, match="carry no coordinate"):
        plot_by_site(data)


def test_plot_by_site_rejects_a_bare_site_id(field_sample_site_time):
    """``sites=1``, a string or a set raises rather than failing on iteration."""
    for sites in (1, "1", {1}):
        with pytest.raises(TypeError, match="sites"):
            plot_by_site(field_sample_site_time, sites=sites)


def test_plot_by_site_refuses_a_repeated_site(field_sample_site_time):
    with pytest.raises(ValueError, match="more than once"):
        plot_by_site(field_sample_site_time, sites=[1, 1])


def test_plot_by_site_rejects_a_field_without_a_site_dim(field_sample_time):
    """A field with no ``site`` dimension raises."""
    with pytest.raises(ValueError, match="needs 'site'"):
        plot_by_site(field_sample_time)


def test_plot_by_site_rejects_a_site_that_is_not_on_the_field(
    field_sample_site_time,
):
    """An unknown site id raises rather than yielding an empty panel."""
    with pytest.raises(KeyError, match="no such site"):
        plot_by_site(field_sample_site_time, sites=[9999])


# ── plot_by_variable ──────────────────────────────────────────────────────────


def test_plot_by_variable_draws_one_panel_per_variable(field_time):
    """One panel per entry of the mapping, in its order."""
    fields = {"a": field_time, "b": field_time * 2}
    _, axes = plot_by_variable(fields, ncol=2)
    assert len(axes) == 2
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), field_time.values * 2)


def test_plot_by_variable_titles_the_panels_with_the_long_names(field_time):
    """The default labels come from each field's ``long_name``."""
    named = field_time.copy()
    named.attrs = dict(field_time.attrs, long_name="Photosynthetic radiation")
    _, axes = plot_by_variable({"par": named}, ncol=1)
    assert axes[0].get_title() == "Photosynthetic radiation"


def test_plot_by_variable_falls_back_to_the_variable_name(field_time):
    """A field with no ``long_name`` is titled by its key.

    Drawn with a plotting function that does not need the attributes, since
    :func:`~sipnet_calibration.plotting.style.axis_label` deliberately refuses
    a field whose ``long_name`` has been lost.
    """
    bare = field_time.copy()
    bare.attrs = {"units": "deg C"}
    _, axes = (
        plot_by_variable(
            {"tair": bare}, lambda data, ax: ax.plot(data.values), ncol=1
        )
    )
    assert axes[0].get_title() == "tair"


def test_plot_by_variable_accepts_fields_on_different_time_axes(field_time):
    """Two variables on different time axes in one call."""
    from conftest import make_field

    fields = {"a": field_time, "b": make_field(("time",), n_time=7)}
    _, axes = plot_by_variable(fields, ncol=2)
    assert len(axes[0].lines[0].get_ydata()) != len(axes[1].lines[0].get_ydata())


def test_plot_by_variable_rejects_an_empty_mapping():
    """An empty mapping raises."""
    with pytest.raises(ValueError, match="nothing to draw"):
        plot_by_variable({})
