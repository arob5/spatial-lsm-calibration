"""Tests for :mod:`sipnet_calibration.plotting.series`.

The sample-dimension rule -- ``time`` is the x axis and every other dimension
is a sample dimension -- is what most of this file exists to protect, since it
is the branch a future change is most likely to break silently.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.container import ErrorbarContainer

from sipnet_calibration.plotting.primitives import nanquantile
from sipnet_calibration.plotting.series import SHOW_KINDS, plot_time_series
from sipnet_calibration.plotting.style import CURVE_COLORS, ROLES, axis_label


def band_bounds(ax):
    """The lowest and highest y reached by any band on *ax*."""
    vertices = np.concatenate(
        [path.vertices for c in ax.collections for path in c.get_paths()]
    )
    return vertices[:, 1].min(), vertices[:, 1].max()


# ── the sample-dimension rule ─────────────────────────────────────────────────


def test_a_time_only_field_draws_one_line(ax, field_time):
    """``(time,)`` under ``show="auto"`` gives one curve and no bands."""
    plot_time_series(field_time, ax=ax)
    assert len(ax.lines) == 1
    assert len(ax.collections) == 0
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), field_time.values)


def test_a_member_field_draws_a_fan(ax, field_member_time):
    """``(member, time)`` under ``show="auto"`` gives bands over the members."""
    plot_time_series(field_member_time, ax=ax, levels=(0.5,))
    (drawn,) = ax.collections
    vertices = drawn.get_paths()[0].vertices
    assert vertices[:, 1].min() == pytest.approx(
        np.percentile(field_member_time.values, 25, axis=0).min()
    )
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), np.median(field_member_time.values, axis=0)
    )


def test_a_site_field_draws_a_fan_over_sites(ax, field_site_time):
    """``(site, time)`` fans over ``site``: the rule is not about ``member``."""
    plot_time_series(field_site_time, ax=ax)
    assert len(ax.collections) == 2
    expected = nanquantile(field_site_time.values, 0.5)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)


def test_two_sample_dims_are_stacked(ax, field_member_site_time):
    """``(member, site, time)`` takes quantiles over member and site at once.

    The bounds are the quantiles of the six stacked curves, not the quantiles
    of per-site quantiles: a quantile of quantiles is not a quantile.
    """
    plot_time_series(field_member_site_time, ax=ax, levels=(0.9,))
    stacked = field_member_site_time.transpose("member", "site", "time").values
    stacked = stacked.reshape(-1, field_member_site_time.sizes["time"])
    assert stacked.shape[0] == 6
    lower, upper = band_bounds(ax)
    assert lower == pytest.approx(nanquantile(stacked, 0.05).min())
    assert upper == pytest.approx(nanquantile(stacked, 0.95).max())


def test_stacking_is_not_a_quantile_of_quantiles(ax, field_member_site_time):
    """The stacked bounds differ from summarizing each site and then pooling."""
    plot_time_series(field_member_site_time, ax=ax, levels=(0.5,))
    stacked_low = nanquantile(
        field_member_site_time.transpose("member", "site", "time").values.reshape(
            -1, field_member_site_time.sizes["time"]
        ),
        0.25,
    )
    per_site = field_member_site_time.quantile(0.25, dim="member")
    pooled_low = per_site.quantile(0.25, dim="site").values
    assert not np.allclose(stacked_low, pooled_low)
    assert band_bounds(ax)[0] == pytest.approx(stacked_low.min())


def test_the_stored_dimension_order_does_not_matter(ax):
    """A field stored as ``(time, member)`` still gives one curve per member.

    Nothing upstream promises canonical dimension order, and reshaping without
    transposing first scrambles members across timesteps: it produces a
    plausible figure of the wrong data. Written because removing the transpose
    in ``_stacked_samples`` left every other test passing.
    """
    values = np.arange(12.0).reshape(4, 3)
    data = xr.DataArray(
        values,
        dims=("time", "member"),
        coords={"time": np.arange(4), "member": np.arange(3)},
        attrs={"units": "u", "long_name": "L"},
    )
    plot_time_series(data, ax=ax, show="spaghetti")
    assert len(ax.lines) == 3
    for member, drawn in enumerate(ax.lines):
        np.testing.assert_array_equal(drawn.get_ydata(), values[:, member])


def test_an_explicit_color_overrides_the_per_curve_palette(ax, field_site_time):
    """``color=`` with ``label_by`` wins, as the style precedence says."""
    plot_time_series(
        field_site_time, ax=ax, show="spaghetti", label_by="site", color="#123456"
    )
    assert {line.get_color() for line in ax.lines} == {"#123456"}
    assert len(ax.get_legend_handles_labels()[1]) == field_site_time.sizes["site"]


def test_a_field_without_time_is_rejected(ax, field_member_site):
    """``(member, site)`` raises, and the message names the dims it found."""
    with pytest.raises(ValueError, match="needs 'time'") as raised:
        plot_time_series(field_member_site, ax=ax)
    assert "member" in str(raised.value)


def test_a_non_canonical_dim_is_rejected(ax):
    """A ``variable`` dim raises rather than being fanned over."""
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("variable", "time"),
        attrs={"units": "u", "long_name": "L"},
    )
    with pytest.raises(ValueError, match="unexpected dimension"):
        plot_time_series(data, ax=ax)


def test_something_other_than_a_data_array_is_rejected(ax):
    """A path or a frame is an adapter's input, not a plotter's."""
    with pytest.raises(ValueError, match="DataArray"):
        plot_time_series("data/processed/constraints_annual.nc", ax=ax)


# ── show ──────────────────────────────────────────────────────────────────────


def test_show_line_on_an_ensemble_is_rejected(ax, field_member_time):
    """``show="line"`` with a sample dim raises, not reduces silently."""
    with pytest.raises(ValueError, match="one curve"):
        plot_time_series(field_member_time, ax=ax, show="line")


def test_show_points_on_an_ensemble_is_rejected(ax, field_member_time):
    """``show="points"`` draws one series, so a sample dim is an error too."""
    with pytest.raises(ValueError, match="one curve"):
        plot_time_series(field_member_time, ax=ax, show="points")


@pytest.mark.parametrize("show", ["fan", "spaghetti"])
def test_summarizing_without_a_sample_dim_is_rejected(ax, field_time, show):
    """``show="fan"`` or ``"spaghetti"`` on ``(time,)`` raises."""
    with pytest.raises(ValueError, match="only 'time'"):
        plot_time_series(field_time, ax=ax, show=show)


def test_show_spaghetti_draws_one_curve_per_sample(ax, field_member_time):
    """The curves' data equal the members, in order."""
    plot_time_series(field_member_time, ax=ax, show="spaghetti")
    assert len(ax.lines) == field_member_time.sizes["member"]
    for artist, expected in zip(ax.lines, field_member_time.values):
        np.testing.assert_array_equal(artist.get_ydata(), expected)


def test_show_spaghetti_honors_n_max(ax, field_member_time):
    """``n_max`` reaches the drawing function."""
    plot_time_series(field_member_time, ax=ax, show="spaghetti", n_max=2)
    assert len(ax.lines) == 2


def test_show_points_draws_scattered_observations(ax, field_time):
    """``show="points"`` gives an ``ErrorbarContainer`` and no curve."""
    plot_time_series(field_time, ax=ax, role="obs", show="points")
    assert len(ax.containers) == 1
    assert isinstance(ax.containers[0], ErrorbarContainer)


def test_points_are_drawn_with_a_marker(ax, field_time):
    """The observation role gives points a marker, so they are visible.

    Drawing them with the line keywords instead leaves ``linestyle="none"``
    and no marker, and the observations disappear from the figure while the
    error bars remain.
    """
    plot_time_series(field_time, ax=ax, role="obs", show="points")
    marker = ax.containers[0][0]
    assert marker.get_marker() == ROLES["obs"]["marker"]
    assert marker.get_markersize() == ROLES["obs"]["markersize"]


def test_an_unknown_show_is_rejected(ax, field_time):
    """The message lists :data:`SHOW_KINDS`."""
    with pytest.raises(ValueError, match="show must be one of") as raised:
        plot_time_series(field_time, ax=ax, show="violin")
    assert all(kind in str(raised.value) for kind in SHOW_KINDS)


# ── the fan's median ──────────────────────────────────────────────────────────


def test_a_fan_also_draws_the_median(ax, field_member_time):
    """``show="fan"`` draws a curve whose data are the median of the samples."""
    plot_time_series(field_member_time, ax=ax, show="fan")
    assert len(ax.lines) == 1
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), np.median(field_member_time.values, axis=0)
    )


def test_a_fan_takes_only_the_color_from_the_role(ax, field_member_time):
    """Bands get the role's color and not its line width.

    Passing the line keywords to ``fill_between`` instead draws a visible
    edge around every band.
    """
    plot_time_series(field_member_time, ax=ax, role="posterior")
    default = matplotlib.rcParams["patch.linewidth"]
    assert default != ROLES["posterior"]["linewidth"]
    for collection in ax.collections:
        assert collection.get_linewidth()[0] == pytest.approx(default)


def test_the_fan_legend_entry_is_on_the_median(ax, field_member_time):
    """Exactly one labeled artist, and it is the median curve."""
    plot_time_series(field_member_time, ax=ax, role="posterior")
    handles, labels = ax.get_legend_handles_labels()
    assert labels == ["posterior"]
    assert handles[0] is ax.lines[0]


# ── roles, labels and style ───────────────────────────────────────────────────


def test_the_role_decides_the_color(ax, field_time):
    """A ``prior`` curve and a ``posterior`` curve differ in color."""
    plot_time_series(field_time, ax=ax, role="prior")
    plot_time_series(field_time, ax=ax, role="posterior")
    assert ax.lines[0].get_color() == ROLES["prior"]["color"]
    assert ax.lines[1].get_color() == ROLES["posterior"]["color"]


def test_an_explicit_keyword_beats_the_role(ax, field_time):
    """``color=`` overrides the role's color."""
    plot_time_series(field_time, ax=ax, role="prior", color="#123456")
    assert ax.lines[0].get_color() == "#123456"


def test_the_label_defaults_to_the_role(ax, field_time):
    """With no ``label``, the legend entry is the role's name."""
    plot_time_series(field_time, ax=ax, role="truth")
    assert ax.get_legend_handles_labels()[1] == ["truth"]


def test_the_label_can_be_suppressed(ax, field_time):
    """``label="_nolegend_"`` leaves the panel with no legend entry."""
    plot_time_series(field_time, ax=ax, label="_nolegend_")
    assert ax.get_legend_handles_labels()[1] == []


def test_label_by_labels_and_colors_each_curve(ax, field_site_time):
    """``label_by="site"`` labels the curves with the site ids.

    Each curve takes a distinct color from :data:`CURVE_COLORS` rather than
    the role's single color, which is what makes one panel with a curve per
    site readable.
    """
    plot_time_series(field_site_time, ax=ax, show="spaghetti", label_by="site")
    labels = ax.get_legend_handles_labels()[1]
    assert labels == [f"site {site}" for site in field_site_time["site"].values]
    colors = [artist.get_color() for artist in ax.lines]
    assert colors == list(CURVE_COLORS[: len(colors)])


def test_label_by_pairs_labels_and_colors_with_the_curves_it_draws(ax):
    """With thinning and colour cycling, each curve keeps its own label.

    ``field_site_time`` has two curves, which is below both ``n_max`` and the
    length of the palette, so it cannot catch a label taken by drawing
    position rather than by sample index.
    """
    n_site = 12
    values = np.arange(n_site * 4, dtype=float).reshape(n_site, 4)
    data = xr.DataArray(
        values,
        dims=("site", "time"),
        coords={
            "site": np.arange(1, n_site + 1),
            "time": np.arange(4),
            "lon": ("site", np.zeros(n_site)),
            "lat": ("site", np.zeros(n_site)),
        },
        attrs={"units": "u", "long_name": "L"},
    )
    plot_time_series(data, ax=ax, show="spaghetti", label_by="site", n_max=5)

    drawn = ax.lines
    assert len(drawn) == 5
    for artist in drawn:
        site = int(artist.get_label().removeprefix("site "))
        np.testing.assert_array_equal(artist.get_ydata(), values[site - 1])
    assert [a.get_color() for a in drawn] == list(CURVE_COLORS[:5])


def test_label_by_cycles_the_palette_when_curves_outnumber_it(ax):
    """More curves than colors reuses the palette from the start."""
    n_site = len(CURVE_COLORS) + 2
    data = xr.DataArray(
        np.zeros((n_site, 3)),
        dims=("site", "time"),
        coords={
            "site": np.arange(1, n_site + 1),
            "time": np.arange(3),
            "lon": ("site", np.zeros(n_site)),
            "lat": ("site", np.zeros(n_site)),
        },
        attrs={"units": "u", "long_name": "L"},
    )
    plot_time_series(data, ax=ax, show="spaghetti", label_by="site", n_max=n_site)
    colors = [a.get_color() for a in ax.lines]
    assert colors[: len(CURVE_COLORS)] == list(CURVE_COLORS)
    assert colors[len(CURVE_COLORS) :] == list(CURVE_COLORS[:2])


def test_label_by_requires_spaghetti(ax, field_site_time):
    """``label_by`` with ``show="fan"`` raises."""
    with pytest.raises(ValueError, match="spaghetti"):
        plot_time_series(field_site_time, ax=ax, show="fan", label_by="site")


def test_label_by_rejects_an_unknown_coordinate(ax, field_site_time):
    """Naming a coordinate that is not there raises."""
    with pytest.raises(ValueError, match="not a coordinate"):
        plot_time_series(field_site_time, ax=ax, show="spaghetti", label_by="pft")


def test_label_by_rejects_a_coordinate_not_on_a_sample_dim(ax, field_site_time):
    """Naming ``time`` raises: it cannot tell one curve from another."""
    with pytest.raises(ValueError, match="sample dimension"):
        plot_time_series(field_site_time, ax=ax, show="spaghetti", label_by="time")


def test_label_by_accepts_a_non_dimension_coordinate(ax, field_site_time):
    """``lon`` is on ``site``, so it can name a curve."""
    plot_time_series(field_site_time, ax=ax, show="spaghetti", label_by="lon")
    assert all("lon" in label for label in ax.get_legend_handles_labels()[1])


# ── observation error ─────────────────────────────────────────────────────────


def error_bar_half_lengths(ax):
    """Half the length of every error bar drawn on *ax*."""
    segments = ax.containers[0].lines[2][0].get_segments()
    return np.array([(s[-1][1] - s[0][1]) / 2 for s in segments])


def test_variance_becomes_a_standard_deviation_bar(ax, field_time):
    """The bar half-length is ``n_sigma * sqrt(variance)``."""
    variance = field_time.copy(data=np.full(field_time.shape, 9.0))
    plot_time_series(field_time, ax=ax, show="points", variance=variance)
    np.testing.assert_allclose(error_bar_half_lengths(ax), 3.0)


def test_standard_deviation_is_used_as_given(ax, field_time):
    """The bar half-length is ``n_sigma * standard_deviation``."""
    deviation = field_time.copy(data=np.full(field_time.shape, 3.0))
    plot_time_series(field_time, ax=ax, show="points", standard_deviation=deviation)
    np.testing.assert_allclose(error_bar_half_lengths(ax), 3.0)


def test_n_sigma_scales_the_bars(ax, field_time):
    """``n_sigma=2`` doubles the half-lengths."""
    deviation = field_time.copy(data=np.full(field_time.shape, 1.5))
    plot_time_series(
        field_time, ax=ax, show="points", standard_deviation=deviation, n_sigma=2.0
    )
    np.testing.assert_allclose(error_bar_half_lengths(ax), 3.0)


def test_variance_and_standard_deviation_together_are_rejected(ax, field_time):
    """Giving both raises rather than picking one."""
    error = field_time.copy(data=np.ones(field_time.shape))
    with pytest.raises(ValueError, match="not both"):
        plot_time_series(
            field_time,
            ax=ax,
            show="points",
            variance=error,
            standard_deviation=error,
        )


def test_an_error_field_without_show_points_is_rejected(ax, field_time):
    """``variance=`` with ``show="line"`` raises."""
    variance = field_time.copy(data=np.ones(field_time.shape))
    with pytest.raises(ValueError, match="show='points'"):
        plot_time_series(field_time, ax=ax, show="line", variance=variance)


def test_a_misaligned_error_field_is_rejected(ax, field_time):
    """An error field on a different time axis raises rather than aligning."""
    from conftest import make_canonical_field

    other = make_canonical_field(("time",), n_time=field_time.sizes["time"] + 3)
    with pytest.raises(ValueError, match="not aligned"):
        plot_time_series(field_time, ax=ax, show="points", variance=other)


def test_an_error_field_with_other_dims_is_rejected(ax, field_time, field_member_time):
    """An error field of a different shape raises."""
    with pytest.raises(ValueError, match="dimensions"):
        plot_time_series(
            field_time, ax=ax, show="points", variance=field_member_time
        )


def test_a_negative_variance_is_rejected(ax, field_time):
    """A negative variance raises rather than producing a ``NaN`` bar."""
    variance = field_time.copy(data=np.full(field_time.shape, -1.0))
    with pytest.raises(ValueError, match="cannot be negative"):
        plot_time_series(field_time, ax=ax, show="points", variance=variance)


@pytest.mark.parametrize("n_sigma", [0.0, -1.0, np.nan])
def test_a_bad_n_sigma_is_rejected(ax, field_time, n_sigma):
    """``n_sigma`` must be finite and positive."""
    deviation = field_time.copy(data=np.ones(field_time.shape))
    with pytest.raises(ValueError, match="n_sigma"):
        plot_time_series(
            field_time,
            ax=ax,
            show="points",
            standard_deviation=deviation,
            n_sigma=n_sigma,
        )


# ── missing values ────────────────────────────────────────────────────────────


def test_a_line_gaps_at_a_missing_timestep(ax, field_with_gaps):
    """``NaN`` reaches the drawing, so the curve breaks rather than bridging."""
    plot_time_series(field_with_gaps.isel(member=0), ax=ax)
    assert np.isnan(ax.lines[0].get_ydata()[5])
    assert len(ax.lines[0].get_ydata()) == field_with_gaps.sizes["time"]


def test_a_fan_gaps_where_every_member_is_missing(ax, field_with_gaps):
    """The all-missing timestep splits the bands."""
    plot_time_series(field_with_gaps, ax=ax, levels=(0.5,))
    assert len(ax.collections[0].get_paths()) == 2
    assert np.isnan(ax.lines[0].get_ydata()[5])


def test_a_fan_summarizes_a_partly_missing_timestep(ax, field_with_gaps):
    """A timestep missing in one member is summarized over the others."""
    plot_time_series(field_with_gaps, ax=ax, levels=(0.5,))
    median = ax.lines[0].get_ydata()
    assert np.isfinite(median[9])
    expected = np.nanmedian(field_with_gaps.values[:, 9])
    assert median[9] == pytest.approx(expected)


def test_points_drop_missing_observations(ax, field_with_gaps):
    """``show="points"`` draws only the observed timesteps."""
    one = field_with_gaps.isel(member=0)
    plot_time_series(one, ax=ax, role="obs", show="points")
    drawn = ax.containers[0][0].get_ydata()
    assert len(drawn) == int(one.notnull().sum())
    assert np.isfinite(drawn).all()


# ── the panel's contract ──────────────────────────────────────────────────────


def test_the_given_axes_is_returned(ax, field_time):
    """The return is *ax* itself, so overlays chain."""
    assert plot_time_series(field_time, ax=ax) is ax


def test_no_figure_is_created_when_axes_are_given(ax, field_time, monkeypatch):
    """``pyplot.subplots`` is not called when ``ax`` is passed."""

    def fail(*args, **kwargs):
        raise AssertionError("a figure was created despite ax being given")

    monkeypatch.setattr(plt, "subplots", fail)
    plot_time_series(field_time, ax=ax)


def test_a_figure_is_created_when_no_axes_are_given(field_time):
    """Called without ``ax``, the panel makes its own figure and axes."""
    before = plt.get_fignums()
    returned = plot_time_series(field_time)
    assert returned.figure.number not in before
    plt.close(returned.figure)


def test_the_panel_never_shows_or_saves(ax, field_time, monkeypatch):
    """``pyplot.show`` and ``savefig`` are patched to fail, and are not hit."""

    def fail(*args, **kwargs):
        raise AssertionError("the panel showed or saved a figure")

    monkeypatch.setattr(plt, "show", fail)
    monkeypatch.setattr(type(ax.figure), "savefig", fail)
    plot_time_series(field_time, ax=ax)


def test_a_time_dim_without_a_coordinate_uses_positions(ax):
    """With no ``time`` coordinate the x axis is 0, 1, 2, ..."""
    data = xr.DataArray(
        np.arange(4.0), dims=("time",), attrs={"units": "u", "long_name": "L"}
    )
    plot_time_series(data, ax=ax)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.arange(4))


def test_the_y_label_comes_from_the_attributes(ax, field_time):
    """The y label is the long name and unit, spelled out rather than
    compared against the function that produced it.
    """
    plot_time_series(field_time, ax=ax)
    assert ax.get_ylabel() == "Mean air temperature over the timestep (deg C)"
    assert ax.get_ylabel() == axis_label(field_time)


def test_the_panel_sets_no_title(ax, field_time):
    """Titles belong to the grid."""
    plot_time_series(field_time, ax=ax)
    assert ax.get_title() == ""


def test_an_overlay_adds_to_the_existing_artists(ax, field_member_time, field_time):
    """A second call onto the same axes keeps the first call's artists."""
    plot_time_series(field_member_time, ax=ax, role="posterior")
    lines_after_first = len(ax.lines)
    plot_time_series(field_time, ax=ax, role="obs", show="points")
    assert len(ax.lines) >= lines_after_first
    assert len(ax.collections) == 2
    assert ax.get_legend_handles_labels()[1] == ["posterior", "obs"]
