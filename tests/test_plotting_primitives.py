"""Tests for :mod:`sipnet_calibration.plotting.primitives`.

Assertions are on the data and properties of what was drawn, never on rendered
images.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.dates import date2num
from matplotlib.lines import Line2D

from sipnet_calibration.plotting.primitives import band, fan, line, points, spaghetti
from sipnet_calibration.plotting.style import BAND_ALPHAS


@pytest.fixture
def x():
    """A short numeric x axis."""
    return np.arange(6.0)


@pytest.fixture
def samples():
    """An ensemble of 40 curves over six points."""
    return np.random.default_rng(0).normal(size=(40, 6))


def error_bar_bounds(container):
    """The lower and upper end of every bar in an ``ErrorbarContainer``."""
    segments = container.lines[2][0].get_segments()
    return np.array([(segment[0][1], segment[-1][1]) for segment in segments])


# ── line ──────────────────────────────────────────────────────────────────────


def test_line_holds_the_data_it_was_given(ax, x):
    """``get_xydata()`` equals the input."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    drawn = line(ax, x, y)
    assert isinstance(drawn, Line2D)
    np.testing.assert_array_equal(drawn.get_xydata(), np.column_stack([x, y]))


def test_line_accepts_datetime_x(ax, field_time):
    """A ``datetime64`` axis arrives as matplotlib's own date numbers."""
    times = field_time["time"].values
    drawn = line(ax, times, field_time.values)
    np.testing.assert_allclose(drawn.get_xydata()[:, 0], date2num(times))


def test_line_keeps_nan(ax, x):
    """A ``NaN`` in *y* stays in the data, so the curve breaks."""
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    drawn = line(ax, x, y)
    assert np.isnan(drawn.get_xydata()[2, 1])
    assert len(drawn.get_xydata()) == len(y)


def test_line_passes_style_to_the_artist(ax, x):
    """Color and line width given as keywords reach the ``Line2D``."""
    drawn = line(ax, x, x, color="#123456", linewidth=3.0, label="a")
    assert drawn.get_color() == "#123456"
    assert drawn.get_linewidth() == 3.0
    assert drawn.get_label() == "a"


def test_line_rejects_mismatched_lengths(ax, x):
    """*x* and *y* of different lengths raise."""
    with pytest.raises(ValueError, match="same length"):
        line(ax, x, np.arange(3.0))


def test_line_rejects_two_dimensional_input(ax, x):
    """A two-dimensional *y* raises rather than drawing several curves."""
    with pytest.raises(ValueError, match="one-dimensional"):
        line(ax, x, np.zeros((2, 6)))


# ── spaghetti ─────────────────────────────────────────────────────────────────


def test_spaghetti_draws_one_line_per_sample(ax, x, samples):
    """Below ``n_max`` every sample gets its own curve, in order."""
    drawn = spaghetti(ax, x, samples[:5], n_max=25)
    assert len(drawn) == 5
    for artist, expected in zip(drawn, samples[:5]):
        np.testing.assert_array_equal(artist.get_ydata(), expected)


def test_spaghetti_decimates_above_n_max(ax, x, samples):
    """Above ``n_max`` exactly ``n_max`` curves are drawn."""
    assert len(spaghetti(ax, x, samples, n_max=7)) == 7


def test_spaghetti_decimation_is_evenly_spaced_and_keeps_the_ends(ax, x, samples):
    """The curves drawn are the evenly spaced samples, ends included."""
    drawn = spaghetti(ax, x, samples, n_max=5)
    expected = np.linspace(0, len(samples) - 1, 5).round().astype(int)
    for artist, index in zip(drawn, expected):
        np.testing.assert_array_equal(artist.get_ydata(), samples[index])
    assert expected[0] == 0
    assert expected[-1] == len(samples) - 1


def test_spaghetti_curves_share_one_color(ax, x, samples):
    """Every curve takes the same color, so the ensemble reads as one thing."""
    drawn = spaghetti(ax, x, samples, n_max=6, color="#0072B2")
    assert {artist.get_color() for artist in drawn} == {"#0072B2"}


def test_spaghetti_labels_only_the_first_curve(ax, x, samples):
    """One legend entry per ensemble; the rest are ``"_nolegend_"``."""
    drawn = spaghetti(ax, x, samples, n_max=6, label="prior")
    assert [artist.get_label() for artist in drawn] == ["prior"] + ["_nolegend_"] * 5
    assert ax.get_legend_handles_labels()[1] == ["prior"]


def test_spaghetti_without_a_label_adds_no_legend_entry(ax, x, samples):
    """An unlabeled ensemble contributes nothing to the legend."""
    spaghetti(ax, x, samples, n_max=3)
    assert ax.get_legend_handles_labels()[1] == []


def test_spaghetti_keeps_nan(ax, x, samples):
    """A ``NaN`` in a sample stays in that curve's data."""
    gapped = samples[:3].copy()
    gapped[1, 2] = np.nan
    drawn = spaghetti(ax, x, gapped)
    assert np.isnan(drawn[1].get_ydata()[2])


def test_spaghetti_rejects_a_non_positive_n_max(ax, x, samples):
    """``n_max=0`` raises rather than drawing nothing."""
    with pytest.raises(ValueError, match="positive integer"):
        spaghetti(ax, x, samples, n_max=0)


def test_spaghetti_rejects_one_dimensional_samples(ax, x):
    """A single series raises, directing the caller to :func:`line`."""
    with pytest.raises(ValueError, match="two-dimensional"):
        spaghetti(ax, x, np.arange(6.0))


# ── band ──────────────────────────────────────────────────────────────────────


def test_band_spans_the_bounds(ax, x):
    """The path's vertices reach *lower* and *upper*."""
    lower, upper = np.zeros(6), np.ones(6)
    drawn = band(ax, x, lower, upper)
    assert isinstance(drawn, PolyCollection)
    vertices = drawn.get_paths()[0].vertices
    assert vertices[:, 1].min() == pytest.approx(0.0)
    assert vertices[:, 1].max() == pytest.approx(1.0)


def test_band_splits_at_a_gap(ax, x):
    """A ``NaN`` run gives two paths, so the band does not bridge the gap."""
    lower = np.array([0.0, 0.0, np.nan, 0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, np.nan, 1.0, 1.0, 1.0])
    assert len(band(ax, x, lower, upper).get_paths()) == 2


def test_band_passes_style_to_the_artist(ax, x):
    """Opacity given as a keyword reaches the collection."""
    drawn = band(ax, x, np.zeros(6), np.ones(6), alpha=0.25)
    assert drawn.get_alpha() == pytest.approx(0.25)


def test_band_rejects_mismatched_lengths(ax, x):
    """Bounds of a different length from *x* raise."""
    with pytest.raises(ValueError, match="same length"):
        band(ax, x, np.zeros(3), np.ones(6))


# ── fan ───────────────────────────────────────────────────────────────────────


def test_fan_draws_one_collection_per_level(ax, x, samples):
    """``levels=(0.5, 0.9)`` gives two collections."""
    assert len(fan(ax, x, samples, levels=(0.5, 0.9))) == 2


def test_fan_draws_the_widest_band_first(ax, x, samples):
    """The wider level is drawn first, so narrower bands layer on top."""
    drawn = fan(ax, x, samples, levels=(0.5, 0.9))
    widest, narrowest = (path.get_paths()[0].vertices[:, 1] for path in drawn)
    assert widest.max() > narrowest.max()
    assert widest.min() < narrowest.min()


def test_fan_uses_the_central_interval_quantiles(ax, x, samples):
    """Level ``0.5`` bounds are the 25th and 75th percentiles."""
    (drawn,) = fan(ax, x, samples, levels=(0.5,))
    vertices = drawn.get_paths()[0].vertices
    lower_expected = np.nanquantile(samples, 0.25, axis=0)
    upper_expected = np.nanquantile(samples, 0.75, axis=0)
    assert vertices[:, 1].min() == pytest.approx(lower_expected.min())
    assert vertices[:, 1].max() == pytest.approx(upper_expected.max())


def test_fan_opacity_increases_towards_the_narrowest_band(ax, x, samples):
    """Alphas run from ``BAND_ALPHAS[0]`` (widest) to ``BAND_ALPHAS[1]``."""
    drawn = fan(ax, x, samples, levels=(0.5, 0.8, 0.95))
    alphas = [collection.get_alpha() for collection in drawn]
    assert alphas[0] == pytest.approx(BAND_ALPHAS[0])
    assert alphas[-1] == pytest.approx(BAND_ALPHAS[1])
    assert alphas == sorted(alphas)


def test_fan_labels_only_the_narrowest_band(ax, x, samples):
    """One legend entry per fan, on the last band drawn."""
    drawn = fan(ax, x, samples, levels=(0.5, 0.9), label="posterior")
    assert [collection.get_label() for collection in drawn] == [
        "_nolegend_",
        "posterior",
    ]


def test_fan_gaps_where_every_sample_is_missing(ax, x, samples):
    """An all-``NaN`` column gives ``NaN`` bounds and so splits the band."""
    gapped = samples.copy()
    gapped[:, 2] = np.nan
    for collection in fan(ax, x, gapped, levels=(0.5,)):
        assert len(collection.get_paths()) == 2


def test_fan_does_not_warn_on_an_all_missing_column(ax, x, samples):
    """The ``All-NaN slice`` warning is suppressed around the quantile call."""
    gapped = samples.copy()
    gapped[:, 2] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fan(ax, x, gapped, levels=(0.5, 0.9))
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_fan_ignores_missing_samples_within_a_column(ax, x, samples):
    """A column with some members missing takes quantiles over the rest."""
    gapped = samples.copy()
    gapped[:20, 3] = np.nan
    (drawn,) = fan(ax, x, gapped, levels=(0.5,))
    expected = np.nanquantile(gapped[:, 3], 0.75)
    assert np.isfinite(expected)
    vertices = drawn.get_paths()[0].vertices
    assert np.isclose(vertices[:, 1], expected).any()


@pytest.mark.parametrize("levels", [(), (0.5, 0.5), (0.0,), (1.0,), (-0.1,), (1.5,)])
def test_fan_rejects_bad_levels(ax, x, samples, levels):
    """Empty, duplicated or out-of-range levels raise."""
    with pytest.raises(ValueError, match="level"):
        fan(ax, x, samples, levels=levels)


def test_fan_rejects_one_dimensional_samples(ax, x):
    """A single series raises, directing the caller to :func:`line`."""
    with pytest.raises(ValueError, match="line"):
        fan(ax, x, np.arange(6.0))


# ── points ────────────────────────────────────────────────────────────────────


def test_points_returns_a_container_without_error_bars(ax, x):
    """``yerr=None`` still returns an ``ErrorbarContainer``."""
    drawn = points(ax, x, x)
    assert isinstance(drawn, ErrorbarContainer)
    np.testing.assert_array_equal(drawn[0].get_ydata(), x)


def test_points_draws_the_bar_half_lengths_given(ax, x):
    """Each bar reaches ``y - yerr`` and ``y + yerr``."""
    y = np.arange(6.0)
    yerr = np.full(6, 0.5)
    bounds = error_bar_bounds(points(ax, x, y, yerr=yerr))
    np.testing.assert_allclose(bounds[:, 0], y - yerr)
    np.testing.assert_allclose(bounds[:, 1], y + yerr)


def test_points_drops_non_finite_entries(ax, x):
    """A ``NaN`` in *y* drops that point entirely."""
    y = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    drawn = points(ax, x, y)
    np.testing.assert_array_equal(drawn[0].get_xdata(), x[[0, 2, 3, 4, 5]])
    np.testing.assert_array_equal(drawn[0].get_ydata(), y[[0, 2, 3, 4, 5]])


def test_points_drops_an_entry_whose_error_is_missing(ax, x):
    """A point with no usable error is dropped with its bar."""
    drawn = points(ax, x, np.ones(6), yerr=np.array([1.0, np.nan, 1, 1, 1, 1]))
    assert len(drawn[0].get_xdata()) == 5


def test_points_drops_yerr_in_step_with_the_points(ax, x):
    """The surviving bars belong to the surviving points, not to shifted."""
    y = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    yerr = np.arange(6.0) / 10.0
    drawn = points(ax, x, y, yerr=yerr)
    kept = [0, 2, 3, 4, 5]
    bounds = error_bar_bounds(drawn)
    np.testing.assert_allclose(bounds[:, 0], y[kept] - yerr[kept])
    np.testing.assert_allclose(bounds[:, 1], y[kept] + yerr[kept])


def test_points_keeps_datetime_x(ax, field_time):
    """A ``datetime64`` axis survives the dropping of missing values."""
    values = field_time.values.copy()
    values[3] = np.nan
    drawn = points(ax, field_time["time"].values, values)
    assert len(drawn[0].get_xdata()) == len(values) - 1


def test_points_rejects_mismatched_lengths(ax, x):
    """A *yerr* of a different length from *y* raises."""
    with pytest.raises(ValueError, match="same length"):
        points(ax, x, x, yerr=np.ones(3))
