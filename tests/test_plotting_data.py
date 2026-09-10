"""The plotting layer against the real driver and constraint products.

The synthetic fixtures exercise the branches; these exercise the seams. They
are the tests that would catch an adapter and a plotting function agreeing
with each other but not with the data: the real arrays are half missing
(drivers) and ragged (constraints), carry real attributes, and are what an
experiment report will actually hand to a plot.

The acceptance criteria of the plotting design spec are written out here as
call sites, each of which should be one to three lines and need no library
function that does not already exist.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sipnet_calibration.plotting import (
    axis_label,
    plot_by_site,
    plot_by_variable,
    plot_time_series,
)


@pytest.fixture
def closing():
    """Close every figure a test made, however it ends."""
    made = []

    def keep(result):
        made.append(result[0])
        return result

    yield keep
    for figure in made:
        plt.close(figure)


def most_observed_site(field):
    """The site id with the most observed values, for a ragged field."""
    counts = field.notnull().sum("time")
    return int(field["site"].values[int(np.argmax(counts.values))])


# ── drivers ───────────────────────────────────────────────────────────────────


def test_a_driver_ensemble_fans_at_one_site(ax, real_driver_field):
    """One site's real driver ensemble is drawn as quantile bands."""
    plot_time_series(real_driver_field.sel(site=1), ax=ax, levels=(0.5, 0.9))
    assert len(ax.collections) == 2
    assert np.isfinite(ax.lines[0].get_ydata()).any()


def test_the_fan_gaps_exactly_where_no_driver_file_exists(
    ax, real_driver_field, real_driver_presence
):
    """Half the member-site pairs have no file, so half the curves are missing.

    Where every member of a site is absent the summary is ``NaN``; where some
    are present the quantiles are taken over those. The gaps have to line up
    with ``driver_present``, not with anything the plot invented.
    """
    for site in real_driver_field["site"].values:
        present = real_driver_presence.sel(site=site).values
        figure, panel = plt.subplots()
        try:
            plot_time_series(real_driver_field.sel(site=site), ax=panel)
            summary = panel.lines[0].get_ydata()
            assert np.isfinite(summary).all() == bool(present.any())
        finally:
            plt.close(figure)
    assert not real_driver_presence.values.all()


def test_a_site_with_no_files_is_entirely_missing(
    ax, real_driver_field, real_driver_presence
):
    """A member with no file for a site contributes nothing to that site."""
    absent = [
        (int(member), int(site))
        for member in real_driver_presence["member"].values
        for site in real_driver_presence["site"].values
        if not bool(real_driver_presence.sel(member=member, site=site))
    ]
    assert absent, "expected at least one missing member-site pair locally"
    member, site = absent[0]
    one = real_driver_field.sel(member=member, site=site)
    plot_time_series(one, ax=ax)
    assert np.isnan(ax.lines[0].get_ydata()).all()


def test_the_y_label_is_the_real_variable_and_unit(ax, real_driver_field):
    """The label comes from the reader's own attributes."""
    plot_time_series(real_driver_field.sel(site=1), ax=ax)
    assert ax.get_ylabel() == axis_label(real_driver_field)
    assert ax.get_ylabel().endswith("(deg C)")


def test_a_driver_field_draws_one_curve_per_site(ax, real_driver_field):
    """One member across both sites: the sample dim is ``site``."""
    one_member = real_driver_field.sel(member=0)
    plot_time_series(one_member, ax=ax, show="spaghetti", label_by="site")
    assert ax.get_legend_handles_labels()[1] == [
        f"site {site}" for site in real_driver_field["site"].values
    ]


# ── constraints ───────────────────────────────────────────────────────────────


def test_annual_observations_draw_as_points_at_the_snapshot_keys(
    ax, real_constraint_fields
):
    """Only the observed years appear; the unobserved cells are dropped.

    A site's record is ragged, so a curve would be a claim the data does not
    support.
    """
    means, _ = real_constraint_fields
    field = means["aboveground_wood_carbon"]
    one = field.sel(site=most_observed_site(field))
    plot_time_series(one, ax=ax, role="obs", show="points")
    drawn = ax.containers[0][0].get_ydata()
    assert len(drawn) == int(one.notnull().sum())
    assert len(drawn) < one.sizes["time"]
    np.testing.assert_allclose(drawn, one.values[np.isfinite(one.values)])


def test_the_error_bars_are_the_square_root_of_the_real_variances(
    ax, real_constraint_fields
):
    """Each bar's half-length equals ``sqrt`` of that cell's variance."""
    means, variances = real_constraint_fields
    name = "aboveground_wood_carbon"
    site = most_observed_site(means[name])
    mean, variance = means[name].sel(site=site), variances[name].sel(site=site)
    plot_time_series(mean, ax=ax, role="obs", show="points", variance=variance)
    segments = ax.containers[0].lines[2][0].get_segments()
    half = np.array([(s[-1][1] - s[0][1]) / 2 for s in segments])
    observed = np.isfinite(mean.values)
    np.testing.assert_allclose(half, np.sqrt(variance.values[observed]))


def test_an_observation_overlay_keeps_the_model_panel(
    ax, real_constraint_fields, real_driver_field
):
    """A second call adds the observations without replacing what was there."""
    means, _ = real_constraint_fields
    field = means["aboveground_wood_carbon"]
    plot_time_series(real_driver_field.sel(site=1), ax=ax, role="posterior")
    bands_before = len(ax.collections)
    plot_time_series(
        field.sel(site=most_observed_site(field)), ax=ax, role="obs", show="points"
    )
    assert len(ax.collections) == bands_before
    assert len(ax.containers) == 1
    assert ax.get_legend_handles_labels()[1] == ["posterior", "obs"]


def test_plot_by_variable_over_the_constraint_fields(closing, real_constraint_fields):
    """Four variables with four different units, one panel each, no shared y."""
    means, _ = real_constraint_fields
    site = most_observed_site(means["aboveground_wood_carbon"])
    one_site = {name: field.sel(site=site) for name, field in means.items()}
    _, axes = closing(
        plot_by_variable(
            one_site,
            lambda data, ax: plot_time_series(data, ax=ax, role="obs", show="points"),
            ncol=2,
        )
    )
    assert len(axes) == len(means)
    units = {axes[i].get_ylabel() for i in range(len(axes))}
    assert len(units) == len(means)


# ── acceptance criteria ───────────────────────────────────────────────────────


@pytest.mark.skip(reason="issue #6: aggregate_time is not implemented yet")
def test_acceptance_one_panel_three_aggregations(ax, real_driver_field):
    """Raw, daily and monthly ``par`` at one site, overlaid, in three lines.

    Criterion 1 of the design spec, on ``par`` rather than NEE, which has no
    processed product yet. Like NEE, ``par`` is a per-timestep total, so its
    daily value is a sum -- and the point of the criterion is that the call
    site does not have to know that. It says ``aggregate_time(par, "1D")``,
    and the rule comes from the variable.

    The sum over the daily curve equals the sum over the 3-hourly one, which
    is what a mean would break.
    """
    from sipnet_calibration.drivers import driver_fields, load_drivers
    from sipnet_calibration.observation_operators import aggregate_time

    par = driver_fields(load_drivers([1], members=[1]))["par"].sel(site=1, member=0)
    plot_time_series(par, ax=ax, role="prior", label="3-hourly")
    plot_time_series(aggregate_time(par, "1D"), ax=ax, role="posterior", label="daily")
    plot_time_series(aggregate_time(par, "MS"), ax=ax, role="truth", label="monthly")

    assert ax.get_legend_handles_labels()[1] == ["3-hourly", "daily", "monthly"]
    lengths = [len(artist.get_ydata()) for artist in ax.lines]
    assert lengths == sorted(lengths, reverse=True)
    assert ax.lines[1].get_ydata().sum() == pytest.approx(par.values.sum())


def test_acceptance_faceted_driver_fan_with_shared_limits(closing, real_driver_field):
    """A driver ensemble per site, faceted, with one y scale, in one line.

    Criterion 2 of the design spec, at the two sites available here rather
    than at six.
    """
    _, axes = closing(plot_by_site(real_driver_field, share="y", ncol=3))
    assert len(axes) == real_driver_field.sizes["site"]
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert [a.get_title() for a in axes] == [
        f"site {site}" for site in real_driver_field["site"].values
    ]
