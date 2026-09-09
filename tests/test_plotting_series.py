"""Tests for :mod:`sipnet_calibration.plotting.series`.

The sample-dim rule -- ``time`` is the x-axis and every other dim is a sample
dim -- is what most of this file exists to protect, since it is the branch a
future change is most likely to break silently.

Skipped until the module is implemented; collected so that the intended
assertions are on the record and reviewable.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="series.py is a contract skeleton; not yet implemented")


# ── the sample-dim rule ───────────────────────────────────────────────────────


def test_a_time_only_field_draws_one_line(ax, field_time):
    """``(time,)`` under ``show="auto"`` gives one ``Line2D`` and no bands."""
    raise NotImplementedError


def test_a_member_field_draws_a_fan(ax, field_member_time):
    """``(member, time)`` under ``show="auto"`` gives bands over the members."""
    raise NotImplementedError


def test_a_site_field_draws_a_fan_over_sites(ax, field_site_time):
    """``(site, time)`` fans over ``site``: the rule is not about ``member``."""
    raise NotImplementedError


def test_two_sample_dims_are_stacked(ax, field_member_site_time):
    """``(member, site, time)`` takes quantiles over member and site at once.

    The bounds equal the quantiles of the six stacked curves, not the
    quantiles of per-site quantiles: a quantile of quantiles is not a
    quantile.
    """
    raise NotImplementedError


def test_a_field_without_time_is_rejected(ax, field_member_site):
    """``(member, site)`` raises, and the message names the dims it found."""
    raise NotImplementedError


def test_a_non_canonical_dim_is_rejected(ax):
    """A field carrying a ``variable`` dim raises."""
    raise NotImplementedError


# ── show ──────────────────────────────────────────────────────────────────────


def test_show_line_on_an_ensemble_is_rejected(ax, field_member_time):
    """``show="line"`` with a sample dim raises rather than reducing silently."""
    raise NotImplementedError


def test_show_fan_without_a_sample_dim_is_rejected(ax, field_time):
    """``show="fan"`` on ``(time,)`` raises."""
    raise NotImplementedError


def test_show_spaghetti_draws_one_curve_per_sample(ax, field_member_time):
    """The curves' data equal the members, in order."""
    raise NotImplementedError


def test_show_spaghetti_honors_n_max(ax, field_member_time):
    """``n_max`` reaches :func:`.primitives.spaghetti`."""
    raise NotImplementedError


def test_show_points_draws_scattered_observations(ax, field_time):
    """``show="points"`` gives an ``ErrorbarContainer`` and no curve."""
    raise NotImplementedError


def test_an_unknown_show_is_rejected(ax, field_time):
    """The message lists :data:`SHOW_KINDS`."""
    raise NotImplementedError


# ── the fan's median ──────────────────────────────────────────────────────────


def test_a_fan_also_draws_the_median(ax, field_member_time):
    """``show="fan"`` draws a line whose data are the median of the samples."""
    raise NotImplementedError


def test_the_fan_legend_entry_is_on_the_median(ax, field_member_time):
    """Exactly one labeled artist, and it is the median line."""
    raise NotImplementedError


# ── roles, labels and style ───────────────────────────────────────────────────


def test_the_role_decides_the_color(ax, field_time):
    """A ``prior`` line and a ``posterior`` line differ in color."""
    raise NotImplementedError


def test_an_explicit_keyword_beats_the_role(ax, field_time):
    """``color=`` overrides the role's color."""
    raise NotImplementedError


def test_the_label_defaults_to_the_role(ax, field_time):
    """With no ``label``, the legend entry is the role's name."""
    raise NotImplementedError


def test_the_label_can_be_suppressed(ax, field_time):
    """``label="_nolegend_"`` leaves the panel with no legend entry."""
    raise NotImplementedError


def test_label_by_labels_and_colors_each_curve(ax, field_site_time):
    """``label_by="site"`` labels the curves with the site ids.

    Each curve takes a distinct color from :data:`.style.CURVE_COLORS` rather
    than the role's single color, which is what makes one panel with a curve
    per site readable.
    """
    raise NotImplementedError


def test_label_by_requires_spaghetti(ax, field_site_time):
    """``label_by`` with ``show="fan"`` raises."""
    raise NotImplementedError


def test_label_by_rejects_a_coordinate_that_is_not_on_a_sample_dim(ax, field_site_time):
    """Naming ``time`` or an absent coordinate raises."""
    raise NotImplementedError


# ── observation error ─────────────────────────────────────────────────────────


def test_variance_becomes_a_standard_deviation_bar(ax, field_time):
    """The bar half-length is ``n_sigma * sqrt(variance)``."""
    raise NotImplementedError


def test_standard_deviation_is_used_as_given(ax, field_time):
    """The bar half-length is ``n_sigma * standard_deviation``."""
    raise NotImplementedError


def test_n_sigma_scales_the_bars(ax, field_time):
    """``n_sigma=2`` doubles the half-lengths."""
    raise NotImplementedError


def test_variance_and_standard_deviation_together_are_rejected(ax, field_time):
    """Giving both raises rather than picking one."""
    raise NotImplementedError


def test_an_error_field_without_show_points_is_rejected(ax, field_time):
    """``variance=`` with ``show="fan"`` raises."""
    raise NotImplementedError


def test_a_misaligned_error_field_is_rejected(ax, field_time):
    """An error field on a different time axis raises rather than aligning."""
    raise NotImplementedError


def test_a_negative_variance_is_rejected(ax, field_time):
    """A negative variance raises rather than producing a ``NaN`` bar."""
    raise NotImplementedError


# ── missing values ────────────────────────────────────────────────────────────


def test_a_line_gaps_at_a_missing_timestep(ax, field_with_gaps):
    """``NaN`` reaches the artist, so the curve breaks rather than bridging."""
    raise NotImplementedError


def test_a_fan_gaps_where_every_member_is_missing(ax, field_with_gaps):
    """The all-missing timestep splits the bands."""
    raise NotImplementedError


def test_a_fan_summarizes_a_partly_missing_timestep(ax, field_with_gaps):
    """A timestep missing in one member is summarized over the others."""
    raise NotImplementedError


def test_points_drop_missing_observations(ax, field_with_gaps):
    """``show="points"`` draws only the observed timesteps."""
    raise NotImplementedError


# ── the panel's contract ──────────────────────────────────────────────────────


def test_the_given_axes_is_returned(ax, field_time):
    """The return is *ax* itself, so overlays chain."""
    raise NotImplementedError


def test_no_figure_is_created_when_axes_are_given(ax, field_time, monkeypatch):
    """``pyplot.subplots`` is not called when ``ax`` is passed."""
    raise NotImplementedError


def test_the_panel_never_shows_or_saves(ax, field_time, monkeypatch):
    """``pyplot.show`` and ``Figure.savefig`` are patched to fail and are not hit."""
    raise NotImplementedError


def test_the_y_label_comes_from_the_attributes(ax, field_time):
    """The y label is :func:`.style.axis_label` of the field."""
    raise NotImplementedError


def test_the_panel_sets_no_title(ax, field_time):
    """Titles belong to the facet grid."""
    raise NotImplementedError


def test_an_overlay_adds_to_the_existing_artists(ax, field_member_time, field_time):
    """A second call onto the same axes keeps the first call's artists."""
    raise NotImplementedError
