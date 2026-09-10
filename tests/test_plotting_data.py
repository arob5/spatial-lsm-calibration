"""The plotting layer against the real driver and constraint products.

The synthetic fixtures exercise the branches; these exercise the seams. They
are the tests that would catch an adapter and a plotter agreeing with each
other but not with the data: the real fields are half missing (drivers) and
ragged (constraints), carry real attributes, and are the fields an experiment
report will actually hand to a panel.

The acceptance criteria of the plotting design spec are written out here as
call sites, each of which should be one to three lines and need no library
function that does not already exist.

Skipped until the plotting modules are implemented; collected so that the
intended assertions are on the record and reviewable.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="the plotting modules are contract skeletons; not yet implemented"
)


# ── drivers ───────────────────────────────────────────────────────────────────


def test_a_driver_ensemble_fans_at_one_site(ax, real_driver_field):
    """One site's real driver ensemble is drawn as quantile bands."""
    raise NotImplementedError


def test_the_fan_gaps_exactly_where_no_driver_file_exists(
    ax, real_driver_field, real_driver_presence
):
    """Half the member-site pairs have no file, so half the curves are missing.

    Where every member of a site is absent the band is ``NaN``; where some are
    present the quantiles are taken over those. The gaps must line up with
    ``driver_present``, not with anything the plotter invented.
    """
    raise NotImplementedError


def test_the_y_label_is_the_real_variable_and_unit(ax, real_driver_field):
    """The label reads ``"... air temperature ... (deg C)"`` from the attrs."""
    raise NotImplementedError


def test_a_driver_field_draws_one_curve_per_site(ax, real_driver_field):
    """One member across both sites: the sample dim is ``site``, not ``member``."""
    raise NotImplementedError


# ── constraints ───────────────────────────────────────────────────────────────


def test_annual_observations_draw_as_points_at_the_snapshot_keys(
    ax, real_constraint_fields
):
    """Only the observed years appear; the unobserved cells are dropped.

    About 38 percent of the ``(site, time)`` cells are observed, so a site's
    record is ragged and a curve would be a claim the data does not support.
    """
    raise NotImplementedError


def test_the_error_bars_are_the_square_root_of_the_real_variances(
    ax, real_constraint_fields
):
    """Each bar's half-length equals ``sqrt`` of that cell's variance."""
    raise NotImplementedError


def test_an_observation_overlay_keeps_the_model_panel(ax, real_constraint_fields):
    """A second call onto the axes adds the observations without replacing anything."""
    raise NotImplementedError


def test_plot_by_variable_over_the_constraint_fields(real_constraint_fields):
    """Four variables with four different units, one panel each, no shared y."""
    raise NotImplementedError


# ── acceptance criteria ───────────────────────────────────────────────────────


def test_acceptance_one_panel_three_aggregations(ax, real_driver_field):
    """Raw, daily and monthly ``par`` at one site, overlaid, in three lines.

    Criterion 1 of the design spec, on ``par`` rather than NEE, which has no
    processed product yet. Like NEE, ``par`` is a per-timestep total, so its
    daily value is a sum.

    The aggregation is written at the call site as ``.resample(time="1D")
    .sum()`` until ``obs_ops.aggregate_time`` exists (issue #6), which takes
    the rule from the variable instead.
    """
    raise NotImplementedError


def test_acceptance_faceted_driver_fan_with_shared_limits(real_driver_field):
    """A driver ensemble fan per site, faceted, shared y-limits, in one line.

    Criterion 2 of the design spec, at the two sites available here rather
    than at six.
    """
    raise NotImplementedError
