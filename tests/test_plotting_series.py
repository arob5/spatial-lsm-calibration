"""Tests for :mod:`sipnet_calibration.plotting.series`.

The batch-dim rule -- ``time`` is the x axis and every batch dim is
summarized, while a spatial dim is refused -- is what most of this file exists
to protect, since it is the branch a future change is most likely to break
silently.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.container import ErrorbarContainer

from conftest import make_field
from sipnet_calibration.plotting.primitives import nanquantile
from sipnet_calibration.plotting.series import SHOW_KINDS, plot_time_series
from sipnet_calibration.plotting.style import CURVE_COLORS, ROLES, axis_label


def band_bounds(ax):
    """The lowest and highest y reached by any band on *ax*."""
    vertices = np.concatenate(
        [path.vertices for c in ax.collections for path in c.get_paths()]
    )
    return vertices[:, 1].min(), vertices[:, 1].max()


# ── the batch-dim rule ────────────────────────────────────────────────────────


def test_a_time_only_field_draws_one_line(ax, field_time):
    """``(time,)`` under ``show="auto"`` gives one curve and no bands."""
    plot_time_series(field_time, ax=ax)
    assert len(ax.lines) == 1
    assert len(ax.collections) == 0
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), field_time.values)


def test_a_batch_field_draws_a_fan(ax, field_sample_time):
    """``(sample, time)`` under ``show="auto"`` gives bands over the samples."""
    plot_time_series(field_sample_time, ax=ax, levels=(0.5,))
    (drawn,) = ax.collections
    vertices = drawn.get_paths()[0].vertices
    assert vertices[:, 1].min() == pytest.approx(
        np.percentile(field_sample_time.values, 25, axis=0).min()
    )
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), np.median(field_sample_time.values, axis=0)
    )


def test_a_site_dim_is_refused_with_advice(ax, field_site_time, field_sample_site_time):
    """Sites are not replicates: ``(site, time)`` is refused, not fanned over."""
    for field in (field_site_time, field_sample_site_time):
        with pytest.raises(ValueError, match=r"spatial dim\(s\) \['site'\].*plot_by_site"):
            plot_time_series(field, ax=ax)


def test_one_site_selected_is_drawn(ax, field_sample_site_time):
    """A scalar ``site`` left by ``.sel`` is metadata, not a dim."""
    plot_time_series(field_sample_site_time.sel(site=27), ax=ax, levels=(0.5,))
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), np.median(field_sample_site_time.sel(site=27).values, axis=0)
    )


def test_a_batch_dim_of_any_name_draws_a_fan(ax):
    """The rule is not about a name: ``driver_member`` fans like ``sample``."""
    field = make_field(("driver_member", "time"))
    plot_time_series(field, ax=ax, levels=(0.5,))
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.median(field.values, axis=0))


def two_batch_dims():
    """``(sample, driver_member, time)``, nine curves."""
    return make_field(("sample", "driver_member", "time"))


def test_two_batch_dims_are_stacked(ax):
    """``(sample, driver_member, time)`` takes quantiles over both at once.

    The bounds are the quantiles of the nine stacked curves, not the
    quantiles of per-member quantiles: a quantile of quantiles is not a
    quantile.
    """
    field = two_batch_dims()
    plot_time_series(field, ax=ax, levels=(0.9,))
    stacked = field.values.reshape(-1, field.sizes["time"])
    assert stacked.shape[0] == 9
    lower, upper = band_bounds(ax)
    assert lower == pytest.approx(nanquantile(stacked, 0.05).min())
    assert upper == pytest.approx(nanquantile(stacked, 0.95).max())


def test_stacking_is_not_a_quantile_of_quantiles(ax):
    """The stacked bounds differ from summarizing one dim and then the other."""
    field = two_batch_dims()
    plot_time_series(field, ax=ax, levels=(0.5,))
    stacked_low = nanquantile(field.values.reshape(-1, field.sizes["time"]), 0.25)
    per_member = field.quantile(0.25, dim="sample")
    pooled_low = per_member.quantile(0.25, dim="driver_member").values
    assert not np.allclose(stacked_low, pooled_low)
    assert band_bounds(ax)[0] == pytest.approx(stacked_low.min())


def test_a_field_stored_out_of_order_is_refused(ax, field_sample_time):
    """A field stored as ``(time, sample)`` is not a field, and says so.

    Reshaping without transposing would scramble samples across timesteps: a
    plausible figure of the wrong data. The field contract fixes the order,
    so the plotter refuses the array rather than guessing.
    """
    with pytest.raises(ValueError, match="not in the order"):
        plot_time_series(field_sample_time.transpose("time", "sample"), ax=ax, show="spaghetti")


def labeled_samples(n_sample, n_time=4):
    """``(sample, time)`` whose values say which sample they are, labeled 1..n."""
    values = np.arange(n_sample * n_time, dtype=float).reshape(n_sample, n_time)
    field = make_field(("sample", "time"), n_sample=n_sample, n_time=n_time)
    return field.copy(data=values).assign_coords(sample=np.arange(1, n_sample + 1))


def test_an_explicit_color_overrides_the_per_curve_palette(ax):
    """``color=`` with ``label_by`` wins, as the style precedence says."""
    data = labeled_samples(2)
    plot_time_series(data, ax=ax, show="spaghetti", label_by="sample", color="#123456")
    assert {line.get_color() for line in ax.lines} == {"#123456"}
    assert len(ax.get_legend_handles_labels()[1]) == data.sizes["sample"]


def test_a_field_without_time_is_rejected(ax, field_sample_site):
    """``(sample, site)`` raises, and the message names the dims it found."""
    with pytest.raises(ValueError, match="needs 'time'") as raised:
        plot_time_series(field_sample_site, ax=ax)
    assert "sample" in str(raised.value)


def test_a_dim_outside_the_field_convention_is_rejected(ax, field_sample_time):
    """A ``variable`` dim, labeled with strings, raises rather than being fanned over."""
    data = field_sample_time.rename(sample="variable").assign_coords(variable=["a", "b", "c"])
    with pytest.raises(ValueError, match="neither a batch dim"):
        plot_time_series(data, ax=ax)


def test_something_other_than_a_data_array_is_rejected(ax):
    """A path or a frame is an adapter's input, not a plotter's."""
    with pytest.raises(TypeError, match="DataArray"):
        plot_time_series("data/processed/constraints/modis_leaf_area_index.nc", ax=ax)


# ── show ──────────────────────────────────────────────────────────────────────


def test_show_line_on_an_ensemble_is_rejected(ax, field_sample_time):
    """``show="line"`` with a sample dim raises, not reduces silently."""
    with pytest.raises(ValueError, match="one curve"):
        plot_time_series(field_sample_time, ax=ax, show="line")


def test_show_points_on_an_ensemble_is_rejected(ax, field_sample_time):
    """``show="points"`` draws one series, so a sample dim is an error too."""
    with pytest.raises(ValueError, match="one curve"):
        plot_time_series(field_sample_time, ax=ax, show="points")


@pytest.mark.parametrize("show", ["fan", "spaghetti"])
def test_summarizing_without_a_sample_dim_is_rejected(ax, field_time, show):
    """``show="fan"`` or ``"spaghetti"`` on ``(time,)`` raises."""
    with pytest.raises(ValueError, match="only 'time'"):
        plot_time_series(field_time, ax=ax, show=show)


def test_show_spaghetti_draws_one_curve_per_sample(ax, field_sample_time):
    """The curves' data equal the members, in order."""
    plot_time_series(field_sample_time, ax=ax, show="spaghetti")
    assert len(ax.lines) == field_sample_time.sizes["sample"]
    for artist, expected in zip(ax.lines, field_sample_time.values):
        np.testing.assert_array_equal(artist.get_ydata(), expected)


def test_show_spaghetti_honors_n_max(ax, field_sample_time):
    """``n_max`` reaches the drawing function."""
    plot_time_series(field_sample_time, ax=ax, show="spaghetti", n_max=2)
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


def test_a_fan_also_draws_the_median(ax, field_sample_time):
    """``show="fan"`` draws a curve whose data are the median of the samples."""
    plot_time_series(field_sample_time, ax=ax, show="fan")
    assert len(ax.lines) == 1
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), np.median(field_sample_time.values, axis=0)
    )


def test_a_fan_takes_only_the_color_from_the_role(ax, field_sample_time):
    """Bands get the role's color and not its line width.

    Passing the line keywords to ``fill_between`` instead draws a visible
    edge around every band.
    """
    plot_time_series(field_sample_time, ax=ax, role="posterior")
    default = matplotlib.rcParams["patch.linewidth"]
    assert default != ROLES["posterior"]["linewidth"]
    for collection in ax.collections:
        assert collection.get_linewidth()[0] == pytest.approx(default)


def test_the_fan_legend_entry_is_on_the_median(ax, field_sample_time):
    """Exactly one labeled artist, and it is the median curve."""
    plot_time_series(field_sample_time, ax=ax, role="posterior")
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


def test_label_by_labels_and_colors_each_curve(ax):
    """``label_by="sample"`` labels the curves with the sample labels.

    Each curve takes a distinct color from :data:`CURVE_COLORS` rather than
    the role's single color, which is what makes a panel of a few labeled
    curves readable.
    """
    data = labeled_samples(2)
    plot_time_series(data, ax=ax, show="spaghetti", label_by="sample")
    labels = ax.get_legend_handles_labels()[1]
    assert labels == [f"sample {label}" for label in data["sample"].values]
    colors = [artist.get_color() for artist in ax.lines]
    assert colors == list(CURVE_COLORS[: len(colors)])


def test_label_by_pairs_labels_and_colors_with_the_curves_it_draws(ax):
    """With thinning and colour cycling, each curve keeps its own label.

    Two curves are below both ``n_max`` and the length of the palette, so
    they cannot catch a label taken by drawing position rather than by batch
    index; twelve can.
    """
    data = labeled_samples(12)
    plot_time_series(data, ax=ax, show="spaghetti", label_by="sample", n_max=5)

    drawn = ax.lines
    assert len(drawn) == 5
    for artist in drawn:
        label = int(artist.get_label().removeprefix("sample "))
        np.testing.assert_array_equal(artist.get_ydata(), data.values[label - 1])
    assert [a.get_color() for a in drawn] == list(CURVE_COLORS[:5])


def test_label_by_cycles_the_palette_when_curves_outnumber_it(ax):
    """More curves than colors reuses the palette from the start."""
    n_sample = len(CURVE_COLORS) + 2
    data = labeled_samples(n_sample, n_time=3)
    plot_time_series(data, ax=ax, show="spaghetti", label_by="sample", n_max=n_sample)
    colors = [a.get_color() for a in ax.lines]
    assert colors[: len(CURVE_COLORS)] == list(CURVE_COLORS)
    assert colors[len(CURVE_COLORS) :] == list(CURVE_COLORS[:2])


def test_label_by_requires_spaghetti(ax, field_sample_time):
    """``label_by`` with ``show="fan"`` raises."""
    with pytest.raises(ValueError, match="spaghetti"):
        plot_time_series(field_sample_time, ax=ax, show="fan", label_by="sample")


def test_label_by_rejects_an_unknown_coordinate(ax, field_sample_time):
    """Naming a coordinate that is not there raises."""
    with pytest.raises(ValueError, match="not a coordinate"):
        plot_time_series(field_sample_time, ax=ax, show="spaghetti", label_by="pft")


def test_label_by_rejects_a_coordinate_not_on_a_batch_dim(ax, field_sample_time):
    """Naming ``time`` raises: it cannot tell one curve from another."""
    with pytest.raises(ValueError, match="batch dims"):
        plot_time_series(field_sample_time, ax=ax, show="spaghetti", label_by="time")


def test_label_by_accepts_a_non_dimension_coordinate(ax, field_sample_time):
    """A coordinate on ``sample`` that is not its index can name a curve."""
    data = field_sample_time.assign_coords(chain=("sample", [0, 0, 1]))
    plot_time_series(data, ax=ax, show="spaghetti", label_by="chain")
    assert all("chain" in label for label in ax.get_legend_handles_labels()[1])


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
    from conftest import make_field

    other = make_field(("time",), n_time=field_time.sizes["time"] + 3)
    with pytest.raises(ValueError, match="not aligned"):
        plot_time_series(field_time, ax=ax, show="points", variance=other)


def test_an_error_field_with_other_dims_is_rejected(ax, field_time, field_sample_time):
    """An error field of a different shape raises."""
    with pytest.raises(ValueError, match="dimensions"):
        plot_time_series(
            field_time, ax=ax, show="points", variance=field_sample_time
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
    plot_time_series(field_with_gaps.isel(sample=0), ax=ax)
    assert np.isnan(ax.lines[0].get_ydata()[5])
    assert len(ax.lines[0].get_ydata()) == field_with_gaps.sizes["time"]


def test_a_fan_gaps_where_every_sample_is_missing(ax, field_with_gaps):
    """The all-missing timestep splits the bands."""
    plot_time_series(field_with_gaps, ax=ax, levels=(0.5,))
    assert len(ax.collections[0].get_paths()) == 2
    assert np.isnan(ax.lines[0].get_ydata()[5])


def test_a_fan_summarizes_a_partly_missing_timestep(ax, field_with_gaps):
    """A timestep missing in one sample is summarized over the others."""
    plot_time_series(field_with_gaps, ax=ax, levels=(0.5,))
    median = ax.lines[0].get_ydata()
    assert np.isfinite(median[9])
    expected = np.nanmedian(field_with_gaps.values[:, 9])
    assert median[9] == pytest.approx(expected)


def test_points_drop_missing_observations(ax, field_with_gaps):
    """``show="points"`` draws only the observed timesteps."""
    one = field_with_gaps.isel(sample=0)
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


def test_a_time_dim_without_a_coordinate_is_refused(ax):
    """A field labels its ``time``; an unlabeled one is not a field."""
    data = xr.DataArray(
        np.arange(4.0), dims=("time",), attrs={"units": "1", "long_name": "L"}
    )
    with pytest.raises(ValueError, match="carry no coordinate"):
        plot_time_series(data, ax=ax)


def test_the_y_label_comes_from_the_attributes(ax, field_time):
    """The y label is the long name and unit, spelled out rather than
    compared against the function that produced it.
    """
    plot_time_series(field_time, ax=ax)
    assert ax.get_ylabel() == "Air temperature (degC)"
    assert ax.get_ylabel() == axis_label(field_time)


def test_the_panel_sets_no_title(ax, field_time):
    """Titles belong to the grid."""
    plot_time_series(field_time, ax=ax)
    assert ax.get_title() == ""


def test_an_overlay_adds_to_the_existing_artists(ax, field_sample_time, field_time):
    """A second call onto the same axes keeps the first call's artists."""
    plot_time_series(field_sample_time, ax=ax, role="posterior")
    lines_after_first = len(ax.lines)
    plot_time_series(field_time, ax=ax, role="obs", show="points")
    assert len(ax.lines) >= lines_after_first
    assert len(ax.collections) == 2
    assert ax.get_legend_handles_labels()[1] == ["posterior", "obs"]
