"""Tests for :mod:`sipnet_calibration.plotting.primitives`.

Assertions are on artist data and properties, never on rendered images.
Skipped until the module is implemented; collected so that the intended
assertions are on the record and reviewable.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="primitives.py is a contract skeleton; not yet implemented"
)


# ── line ──────────────────────────────────────────────────────────────────────


def test_line_holds_the_data_it_was_given(ax):
    """``get_xydata()`` equals the input, dates through ``mdates.date2num``."""
    raise NotImplementedError


def test_line_keeps_nan(ax):
    """A ``NaN`` in *y* stays in the artist's data, so the curve breaks."""
    raise NotImplementedError


def test_line_passes_style_to_the_artist(ax):
    """Color and line width given as keywords land on the ``Line2D``."""
    raise NotImplementedError


def test_line_rejects_mismatched_lengths(ax):
    """*x* and *y* of different lengths raise."""
    raise NotImplementedError


def test_line_rejects_two_dimensional_input(ax):
    """A two-dimensional *y* raises rather than drawing several curves."""
    raise NotImplementedError


# ── spaghetti ─────────────────────────────────────────────────────────────────


def test_spaghetti_draws_one_line_per_sample(ax):
    """Below ``n_max`` every sample gets its own ``Line2D``, in order."""
    raise NotImplementedError


def test_spaghetti_decimates_above_n_max(ax):
    """Above ``n_max`` exactly ``n_max`` curves are drawn."""
    raise NotImplementedError


def test_spaghetti_decimation_is_evenly_spaced_and_keeps_the_ends(ax):
    """The curves drawn are the evenly spaced samples, first and last included."""
    raise NotImplementedError


def test_spaghetti_curves_share_one_color(ax):
    """Every curve takes the same color, so the ensemble reads as one thing."""
    raise NotImplementedError


def test_spaghetti_labels_only_the_first_curve(ax):
    """One legend entry per ensemble; the rest are ``"_nolegend_"``."""
    raise NotImplementedError


def test_spaghetti_keeps_nan(ax):
    """A ``NaN`` in a sample stays in that curve's data."""
    raise NotImplementedError


def test_spaghetti_rejects_a_non_positive_n_max(ax):
    """``n_max=0`` raises rather than drawing nothing."""
    raise NotImplementedError


# ── band ──────────────────────────────────────────────────────────────────────


def test_band_spans_the_bounds(ax):
    """The path's vertices reach *lower* and *upper* at each *x*."""
    raise NotImplementedError


def test_band_splits_at_a_gap(ax):
    """A ``NaN`` run gives two paths, so the band does not bridge the gap."""
    raise NotImplementedError


def test_band_rejects_mismatched_lengths(ax):
    """Bounds of a different length from *x* raise."""
    raise NotImplementedError


# ── fan ───────────────────────────────────────────────────────────────────────


def test_fan_draws_one_collection_per_level(ax):
    """``levels=(0.5, 0.9)`` gives two collections."""
    raise NotImplementedError


def test_fan_draws_the_widest_band_first(ax):
    """The wider level is drawn first, so narrower bands layer on top."""
    raise NotImplementedError


def test_fan_uses_the_central_interval_quantiles(ax):
    """Level ``0.5`` bounds are the 25th and 75th percentiles of the samples."""
    raise NotImplementedError


def test_fan_opacity_increases_towards_the_narrowest_band(ax):
    """Alphas run from ``BAND_ALPHAS[0]`` (widest) to ``BAND_ALPHAS[1]``."""
    raise NotImplementedError


def test_fan_labels_only_the_narrowest_band(ax):
    """One legend entry per fan."""
    raise NotImplementedError


def test_fan_gaps_where_every_sample_is_missing(ax):
    """An all-``NaN`` column gives ``NaN`` bounds and so splits the band."""
    raise NotImplementedError


def test_fan_does_not_warn_on_an_all_missing_column(ax, recwarn):
    """The ``All-NaN slice`` warning is suppressed around the quantile call."""
    raise NotImplementedError


def test_fan_ignores_missing_samples_within_a_column(ax):
    """A column with some members missing takes quantiles over the rest."""
    raise NotImplementedError


@pytest.mark.parametrize("levels", [(), (0.5, 0.5), (0.0,), (1.0,), (-0.1,), (1.5,)])
def test_fan_rejects_bad_levels(ax, levels):
    """Empty, duplicated or out-of-range levels raise."""
    raise NotImplementedError


def test_fan_rejects_one_dimensional_samples(ax):
    """A single series raises, directing the caller to :func:`line`."""
    raise NotImplementedError


# ── points ────────────────────────────────────────────────────────────────────


def test_points_returns_a_container_without_error_bars(ax):
    """``yerr=None`` still returns an ``ErrorbarContainer``."""
    raise NotImplementedError


def test_points_draws_the_bar_half_lengths_given(ax):
    """Each bar reaches ``y - yerr`` and ``y + yerr``."""
    raise NotImplementedError


def test_points_drops_non_finite_entries(ax):
    """A ``NaN`` in *x*, *y* or *yerr* drops that point entirely."""
    raise NotImplementedError


def test_points_drops_yerr_in_step_with_the_points(ax):
    """The surviving bars belong to the surviving points, not to shifted ones."""
    raise NotImplementedError


def test_points_rejects_mismatched_lengths(ax):
    """A *yerr* of a different length from *y* raises."""
    raise NotImplementedError
