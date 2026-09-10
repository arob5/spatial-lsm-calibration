"""Tests for the observation-space operations.

The ``sipnet_time_index`` cases are built by hand from SIPNET's
``year``/``day``/``time`` labeling, including the drifting ``time`` column the
``.clim`` drivers carry (issue #9), so that a future change cannot start
trusting that column's value.

The ``aggregate_time`` cases exist mostly to pin down two things that fail
silently. The first is sum versus mean: a per-timestep total aggregated with a
mean is wrong by the number of steps in the period, which for 3-hourly to
daily is a factor of eight, and the result looks entirely plausible on a
figure. The second is an empty period, which ``.resample(...).sum()`` reports
as zero rather than as missing, so a day with no observations reads as zero
flux.

Every expected value is computed with ``numpy`` rather than with the function
under test or with the registry, so that a test cannot agree with a bug by
construction. The fixtures deliberately do **not** all store their dimensions
in the same order: a reduction that assumes an order is a mutation that
survives an entire suite of same-shaped fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sipnet_calibration.observation_operators import (
    AGGREGATION_METHODS,
    aggregate_time,
    aggregation_counts,
    sipnet_time_index,
)

#: Applied to every class whose subject is not implemented yet, so the cases
#: are collected and read as the contract while ``sipnet_time_index``'s own
#: tests keep running.
not_yet = pytest.mark.skip(reason="issue #6: not implemented yet")

REAL_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "drivers"
    / "ERA5_1_1"
    / "ERA5.1.2012-01-01.2024-12-31.clim"
)


def one_day(year: int, day_of_year: int, hours) -> pd.DatetimeIndex:
    n = len(hours)
    return sipnet_time_index([year] * n, [day_of_year] * n, hours)


class TestSipnetTimeIndex:
    def test_builds_the_nominal_grid_from_year_day_and_slot(self):
        index = one_day(2013, 1, [0, 3, 6, 9, 12, 15, 18, 21])
        expected = pd.date_range("2013-01-01", periods=8, freq="3h").as_unit("ns")
        pd.testing.assert_index_equal(index, expected, check_names=False)
        assert index.dtype == np.dtype("datetime64[ns]")

    def test_ignores_the_drift_in_the_time_column(self):
        """A label of 23.00 in the last slot of 31 December still maps to 21:00."""
        drifted = np.arange(8) * 3.000685 + 1.995  # what 31 December looks like
        drifted[-1] = 23.0
        index = one_day(2013, 365, drifted)
        expected = pd.date_range("2013-12-31", periods=8, freq="3h").as_unit("ns")
        pd.testing.assert_index_equal(index, expected, check_names=False)

    def test_leap_day_is_placed_correctly(self):
        assert one_day(2012, 60, [0.0])[0] == pd.Timestamp("2012-02-29")
        assert one_day(2013, 60, [0.0])[0] == pd.Timestamp("2013-03-01")
        assert one_day(2012, 366, [21.0])[0] == pd.Timestamp("2012-12-31 21:00")

    def test_century_rule_for_leap_years(self):
        assert one_day(2000, 366, [0.0])[0] == pd.Timestamp("2000-12-31")
        with pytest.raises(ValueError, match="day_of_year 366 in non-leap year"):
            one_day(2100, 366, [0.0])

    def test_rejects_two_dimensional_input(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            sipnet_time_index([[2013]], [[1]], [[0.0]])

    def test_spans_a_year_boundary_in_row_order(self):
        index = sipnet_time_index([2012, 2012, 2013, 2013], [366, 366, 1, 1], [18, 21, 0, 3])
        expected = pd.DatetimeIndex(
            ["2012-12-31 18:00", "2012-12-31 21:00", "2013-01-01 00:00", "2013-01-01 03:00"]
        ).as_unit("ns")
        pd.testing.assert_index_equal(index, expected, check_names=False)

    def test_accepts_whole_number_floats(self):
        index = sipnet_time_index([2013.0, 2013.0], [5.0, 5.0], [0.0, 3.0])
        assert index[0] == pd.Timestamp("2013-01-05")

    def test_other_timesteps(self):
        daily = sipnet_time_index([2013, 2013], [1, 2], [0.0, 0.0], timestep_hours=24.0)
        pd.testing.assert_index_equal(
            daily, pd.DatetimeIndex(["2013-01-01", "2013-01-02"]).as_unit("ns"), check_names=False
        )
        hourly = sipnet_time_index([2013] * 24, [1] * 24, np.arange(24) + 0.01, timestep_hours=1.0)
        pd.testing.assert_index_equal(
            hourly, pd.date_range("2013-01-01", periods=24, freq="1h").as_unit("ns"), check_names=False
        )

    def test_empty_input_gives_an_empty_index(self):
        index = sipnet_time_index([], [], [])
        assert len(index) == 0 and index.dtype == np.dtype("datetime64[ns]")

    def test_rejects_a_timestep_that_does_not_divide_24(self):
        with pytest.raises(ValueError, match="divide 24"):
            sipnet_time_index([2013], [1], [0.0], timestep_hours=5.0)
        with pytest.raises(ValueError, match="divide 24"):
            sipnet_time_index([2013], [1], [0.0], timestep_hours=0.0)
        with pytest.raises(ValueError, match="divide 24"):
            sipnet_time_index([2013], [1], [0.0], timestep_hours=float("inf"))
        with pytest.raises(ValueError, match="must be a number"):
            sipnet_time_index([2013], [1], [0.0], timestep_hours="three")

    def test_rejects_unequal_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            sipnet_time_index([2013, 2013], [1], [0.0])

    def test_accepts_the_arguments_by_keyword(self):
        index = sipnet_time_index(year=[2013], day_of_year=[2], hours_since_midnight=[6.5])
        assert index[0] == pd.Timestamp("2013-01-02 06:00")

    def test_rejects_non_whole_year_or_day(self):
        with pytest.raises(ValueError, match="day_of_year must hold whole numbers"):
            sipnet_time_index([2013], [1.5], [0.0])
        with pytest.raises(ValueError, match="year must hold whole numbers"):
            sipnet_time_index([2013.5], [1], [0.0])

    def test_rejects_day_outside_the_year(self):
        with pytest.raises(ValueError, match="within 1..366"):
            sipnet_time_index([2013], [0], [0.0])
        with pytest.raises(ValueError, match="within 1..366"):
            sipnet_time_index([2013], [367], [0.0])

    def test_rejects_day_366_in_a_non_leap_year(self):
        with pytest.raises(ValueError, match="day_of_year 366 in non-leap year"):
            sipnet_time_index([2013], [366], [0.0])

    def test_rejects_a_time_outside_the_day(self):
        with pytest.raises(ValueError, match=r"within \[0, 24\)"):
            sipnet_time_index([2013], [1], [24.0])
        with pytest.raises(ValueError, match=r"within \[0, 24\)"):
            sipnet_time_index([2013], [1], [-0.5])
        with pytest.raises(ValueError, match=r"within \[0, 24\)"):
            sipnet_time_index([2013], [1], [np.nan])

    def test_rejects_a_label_that_drifted_into_the_next_slot(self):
        """Two consecutive rows whose labels fall in one slot collide, and are
        refused rather than silently reassigned. A single label cannot reveal
        a drift on its own: a skipped slot is accepted, which is why the
        drivers module asserts the drift model on whole files instead."""
        with pytest.raises(ValueError, match="not strictly increasing"):
            sipnet_time_index([2013, 2013], [1, 1], [3.0, 3.1])
        skipped = sipnet_time_index([2013, 2013], [1, 1], [0.0, 6.0])
        assert list(skipped) == [pd.Timestamp("2013-01-01 00:00"), pd.Timestamp("2013-01-01 06:00")]

    def test_rejects_rows_out_of_order(self):
        with pytest.raises(ValueError, match="not strictly increasing"):
            sipnet_time_index([2013, 2013], [2, 1], [0.0, 0.0])

    @pytest.mark.skipif(not REAL_FILE.exists(), reason="the raw driver file is not present")
    def test_reproduces_the_real_files_grid(self):
        frame = pd.read_csv(REAL_FILE, sep=r"\s+", header=None, usecols=[1, 2, 3], names=["year", "day", "time"])
        index = sipnet_time_index(frame["year"], frame["day"], frame["time"])
        expected = pd.date_range("2012-01-01", "2024-12-31 21:00", freq="3h").as_unit("ns")
        assert len(index) == 37992
        pd.testing.assert_index_equal(index, expected, check_names=False)


# ── aggregate_time ────────────────────────────────────────────────────────────


def three_hourly_field(
    name: str,
    *,
    dims: tuple[str, ...] = ("time",),
    n_days: int = 3,
    start: str = "2013-01-01",
) -> xr.DataArray:
    """A field of *name* on a 3-hourly grid, with a known pattern of values.

    Parameters
    ----------
    name:
        The field's name, which is what ``aggregate_time`` looks up.
    dims:
        The dimensions, in the order they are stored. ``site`` and ``member``
        get two entries each, and ``site`` carries ``lon``/``lat`` as
        non-dimension coordinates. Storing them in the order given, rather
        than in canonical order, is what makes an order-dependent reduction
        visible.
    n_days:
        Whole days of 3-hourly steps.
    start:
        Timestamp of the first row.

    Returns
    -------
    xarray.DataArray
        Values are distinct per cell and are *not* symmetric across a period,
        so a sum, a mean, a first and a last are four different numbers.
    """
    raise NotImplementedError("issue #6")


@not_yet
class TestTheRuleComesFromTheVariable:
    """The highest-value tests in the module: they protect the likelihood."""

    def test_a_per_timestep_total_sums(self):
        """``par`` is a total over the timestep, so its daily value is the sum
        of the day's eight rows. Compared against ``numpy``'s own sum over the
        same rows, never against ``.resample``."""

    def test_an_intensive_variable_means(self):
        """``air_temperature`` describes the timestep rather than accumulating
        over it, so its daily value is the mean."""

    def test_the_sum_and_the_mean_differ_by_the_number_of_steps(self):
        """The same field aggregated under both rules, so the eight-fold error
        is asserted as a ratio and not only as two separate numbers. This is
        the test that fails if the two rules are ever swapped."""

    def test_nee_sums(self):
        """Canonical NEE is a per-timestep total, which is the case the
        registry exists for; a mean here is the error worth the most."""

    def test_precipitation_sums_and_vpd_means(self):
        """A second variable under each rule, so the first two tests cannot
        pass by a coincidence of one entry."""

    def test_the_field_s_own_aggregation_attribute_is_not_consulted(self):
        """A field carrying ``aggregation="mean"`` on a variable the registry
        calls ``"sum"`` is still summed. The drivers write that attribute, and
        which of the two wins has to be pinned rather than assumed."""

    def test_an_unregistered_name_raises(self):
        """Rather than falling back to a default rule, which is how the
        eight-fold error would re-enter."""

    def test_a_field_with_no_name_raises(self):
        """``None`` is what xarray arithmetic leaves behind, so this is the
        realistic way a rule goes missing."""


@not_yet
class TestEmptyPeriods:
    def test_a_period_with_no_observations_is_missing_not_zero(self):
        """A whole day of ``NaN`` under the sum rule. ``.resample().sum()``
        returns 0 there, which reads as zero flux rather than as unobserved,
        and NEE is about 55% missing over site and time."""

    def test_a_period_with_no_observations_is_missing_under_every_rule(self):
        """Sum, mean, first and last, so the guard cannot be attached to the
        sum alone."""

    def test_a_period_that_is_only_partly_missing_uses_what_is_there(self):
        """The complement: the finite values are aggregated, so the guard does
        not throw away a period that has data."""

    def test_the_missing_periods_are_exactly_where_the_data_is_missing(self):
        """Asserted as a pattern over several periods, not on one, so a guard
        that masked everything or nothing would fail."""


@not_yet
class TestPartialPeriods:
    def test_a_partial_period_is_returned_not_scaled(self):
        """Three of a day's eight rows sum to those three values. Scaling
        assumes the absent rows resemble the present ones, which for a diurnal
        flux is false in the worst direction."""

    def test_min_count_drops_a_period_with_too_few_values(self):
        """``min_count=8`` keeps only whole days, which is how a caller states
        a completeness rule."""

    def test_min_count_counts_finite_values_not_rows(self):
        """A period holding eight rows of which three are finite fails
        ``min_count=8``, so the rule cannot be satisfied by missing data."""

    def test_min_count_applies_to_the_mean_rule_too(self):
        """Not only to the sum: ``.mean()`` accepts no ``min_count`` of its
        own, so this is where a sum-only implementation shows."""

    def test_min_count_below_one_raises(self):
        """Zero would mean a period formed from nothing still produces a
        value, which for a sum is the zero the default exists to prevent."""

    def test_min_count_must_be_an_integer(self):
        """``min_count=1.5`` would compare as a float and quietly work."""


@not_yet
class TestStocksAreRefused:
    def test_a_stock_variable_raises(self):
        """``lai`` and the carbon pools are levels at an instant, so neither a
        sum nor a mean is their aggregation."""

    def test_the_message_says_what_to_pass_instead(self):
        """A refusal that does not say ``how="last"`` sends the reader to the
        source."""

    def test_a_stock_aggregates_when_how_is_given(self):
        """The refusal is of the default, not of the operation."""

    def test_instantaneous_cannot_be_passed_as_how(self):
        """It names a refusal rather than a reduction, so it is not among
        :data:`AGGREGATION_METHODS`."""


@not_yet
class TestExplicitHow:
    def test_how_overrides_the_variable_s_rule(self):
        """``how="mean"`` on ``par``, which the registry sums."""

    def test_how_admits_a_field_whose_variable_is_not_registered(self):
        """An observation error variance is named for its variable but is not
        that variable; with ``how`` given the registry is not consulted."""

    def test_last_and_first_take_the_period_s_end_and_start(self):
        """Distinguishable only because the fixture is not symmetric within a
        period."""

    def test_last_skips_a_missing_final_value(self):
        """The last *observed* value of the period, not the last row, which
        would be ``NaN`` whenever a record ends mid-period."""

    def test_an_unknown_how_raises_and_lists_the_methods(self):
        """A typo such as ``how="total"`` would otherwise reach xarray."""


@not_yet
class TestUpsamplingIsRefused:
    def test_a_finer_target_than_the_source_raises(self):
        """``"1h"`` on a 3-hourly field returns a field that is two-thirds
        ``NaN`` with no error, and the emptiness reads as missing data."""

    def test_an_annual_field_refuses_a_monthly_target(self):
        """The case that arises for real: the annual constraints in a report
        that aggregates every variable to one frequency."""

    def test_a_target_equal_to_the_source_spacing_is_allowed(self):
        """Aggregating a daily field to ``"1D"`` is a no-op, not an error."""

    def test_a_coarser_target_of_no_fixed_length_is_allowed(self):
        """``"MS"`` and ``"YS"`` have no fixed length, so the comparison has
        to measure them rather than convert them to a ``Timedelta``."""


@not_yet
class TestWhatSurvivesAggregation:
    def test_the_name_and_the_attributes_are_carried_through(self):
        """``units`` and ``long_name`` above all: the plotting layer builds
        its axis label from them, and ``keep_attrs`` defaults have changed
        between xarray versions."""

    def test_the_aggregation_is_recorded_in_the_attributes(self):
        """``aggregation_applied`` and ``aggregation_freq``. A canonical
        per-timestep total carries the same ``units`` at every resolution, so
        without these two a 3-hourly and a daily NEE field are
        indistinguishable while differing by a factor of eight."""

    def test_the_recorded_method_is_the_one_used_not_the_one_asked_for(self):
        """With ``how`` given it is ``how``; without, it is the registry's
        rule rather than the string ``None``."""

    def test_the_other_dimensions_and_their_coordinates_are_untouched(self):
        """``member``, ``site``, and ``lon``/``lat`` as non-dimension
        coordinates on ``site``: the canonical field convention has to survive
        an aggregation, or the result cannot be plotted or mapped."""

    def test_the_result_is_independent_of_the_stored_dimension_order(self):
        """A ``(time, site)`` field and a ``(site, time)`` field with the same
        values aggregate to the same numbers. An implementation that reshapes
        or transposes without naming dimensions passes every same-shaped
        fixture and scrambles this one."""

    def test_aggregating_over_a_single_period_reproduces_the_reduction(self):
        """The degenerate case, where the answer is one number computed by
        ``numpy`` over the whole array."""


@not_yet
class TestTheTimeAxis:
    def test_a_field_with_no_time_dimension_raises(self):
        """An initial-condition field is ``(member, site)``."""

    def test_a_field_with_a_time_dimension_but_no_coordinate_raises(self):
        """``resample`` needs the coordinate; without this check the failure
        comes out of pandas naming neither the array nor the dimension."""

    def test_an_integer_time_coordinate_raises(self):
        """Not every ``time`` axis is datetime; SIPNET's own output arrives as
        year, day and hour columns until ``sipnet_time_index`` has run."""

    def test_a_non_increasing_time_coordinate_raises(self):
        """Two sources concatenated in the wrong order group into periods that
        overlap, and the result is wrong rather than empty."""

    def test_something_that_is_not_a_dataarray_raises(self):
        """A ``Dataset`` or a ``DataFrame``, which is what an adapter that has
        not run yet would hand over."""

    def test_periods_are_labeled_with_their_own_start(self):
        """xarray's convention, and therefore the one the returned ``time``
        axis follows. The drivers label interval *ends*, so a day's rows
        labeled 00:00 to 21:00 group under that day: the same grouping
        SIPNET's own ``day`` column gives, which is the point."""


@not_yet
class TestAggregationCounts:
    def test_counts_the_finite_values_in_each_period(self):
        """Against ``numpy``'s own count over the same rows."""

    def test_a_period_with_no_observations_counts_zero(self):
        """Zero rather than missing, so a caller can test for it."""

    def test_it_aligns_with_what_aggregate_time_returns(self):
        """Same dimensions and same ``time`` axis, so ``.where(counts == 8)``
        needs no reindexing. This is the whole reason it is public."""

    def test_it_does_not_consult_the_variable_s_rule(self):
        """A stock is counted rather than refused: a count is a property of
        the missingness, not of the variable."""

    def test_it_refuses_the_same_time_axes_aggregate_time_refuses(self):
        """The checks are shared, so a caller cannot get a count for a field
        that cannot be aggregated."""


@not_yet
@pytest.mark.skipif(not REAL_FILE.exists(), reason="the raw driver file is not present")
class TestAgainstTheRealDrivers:
    """The seam. The synthetic cases exercise the branches; these confirm the
    rules hold on the file the model is actually run on."""

    def test_the_daily_par_total_equals_the_sum_over_the_whole_record(self):
        """A sum over days conserves the total, which a mean does not. Read
        from the real file, so an aggregation that dropped or double-counted
        the boundary of a day shows up here."""

    def test_the_daily_air_temperature_is_a_mean_of_the_real_rows(self):
        """Against a ``pandas`` groupby on the same file, so the two paths to
        the same number are independent."""

    def test_every_day_of_the_real_record_holds_eight_rows(self):
        """The driver files have no missing values and no partial days except
        possibly at their ends, which is what makes ``min_count=1`` safe on
        them. Asserted rather than assumed."""
