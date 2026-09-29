"""The experiment's observation sources: observed values prepared from the inputs, each with its operator.

Each observation source's observed values are built from ``inputs`` as
``config`` says, and paired with the operator
``config.OBSERVATION_OPERATORS`` binds to it; the sources make the calibration
and the validation observation vectors. Nothing here converts units or
touches a model: the observed values keep their data source's units, and
each operator predicts them from the model.

Observation sources
-------------------
``nee_night_centered``, ``nee_day_centered``
    Observed NEE, ``umol m-2 s-1`` of CO2, averaged over each of the two
    twelve-hour windows of every UTC day in the period
    (``config.NEE_WINDOWS``) that lies inside the run's record, from the
    series' gap-filled values. A window is
    an observation only when at least ``config.NEE_MINIMUM_MEASURED_FRACTION``
    of its values were measured (quality flag 0); the others are dropped.
    Its ``time`` is the window's end, and ``window_start``/``window_end``
    carry the window.
``modis_leaf_area_index``
    The MODIS composites, as processed, whose labels fall inside the run's
    record.
``landtrendr_aboveground_biomass``
    LandTrendr's annual values in ``config.LANDTRENDR_YEARS``, with their
    year windows, relabeled as dry biomass: the ``constituent`` attribute is
    dropped and a ``comment`` says why (``config.WOOD_CARBON_FRACTION``).
``soilgrids_soil_organic_carbon``
    The one static value, as processed.

The run's record is the interval the prepared drivers cover, which is what
the operators read the model over; every dated observation lies inside it.

Usage
-----
::

    uv run python experiments/single_site_mcmc_vs_eki/observations.py

    import observations
    calibration = observations.calibration_observation_vector()
    calibration.describe()
    calibration.y                         # Flat observations, site-major
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr

import config
import inputs
from sipnet_calibration import net_ecosystem_exchange as nee
from sipnet_calibration.conventions import (
    TIME,
    TIMESTEP_START,
    WINDOW_END,
    WINDOW_START,
)
from sipnet_calibration.observation import ObservationSource, ObservationVector
from sipnet_calibration.observation.time_alignment import reduce_windows

__all__ = [
    "calibration_observation_vector",
    "observed_aboveground_biomass",
    "observed_leaf_area_index",
    "observed_nee_windows",
    "observed_soil_carbon",
    "run_record",
    "validation_observation_vector",
]


def calibration_observation_vector() -> ObservationVector:
    """The observations the calibration conditions on, one source per observed quantity."""
    record = run_record()
    observed_values_by_source = {
        **observed_nee_windows(
            config.CALIBRATION_NEE_SERIES, config.CALIBRATION_NEE_PERIOD, record
        ),
        "modis_leaf_area_index": observed_leaf_area_index(record),
        "landtrendr_aboveground_biomass": observed_aboveground_biomass(record),
        "soilgrids_soil_organic_carbon": observed_soil_carbon(),
    }
    return _observation_vector_of(observed_values_by_source)


def validation_observation_vector() -> ObservationVector:
    """The held-out NEE of the second tower, over the years the calibration does not see."""
    record = run_record()
    return _observation_vector_of(
        observed_nee_windows(
            config.VALIDATION_NEE_SERIES, config.VALIDATION_NEE_PERIOD, record
        )
    )


def run_record() -> pd.Interval:
    """The interval the prepared drivers cover, ``(first step's start, last step's end]``."""
    driver_data = inputs.driver_dataset()
    return pd.Interval(
        pd.Timestamp(driver_data[TIMESTEP_START].values[0]),
        pd.Timestamp(driver_data[TIME].values[-1]),
        closed="right",
    )


def observed_nee_windows(
    series_name: str, period: tuple[int, int], record: pd.Interval
) -> dict[str, xr.DataArray]:
    """One series' NEE averaged over each of ``config.NEE_WINDOWS``, well-measured windows only.

    Parameters
    ----------
    series_name:
        The NEE series, a name in ``sipnet_calibration.net_ecosystem_exchange``.
    period:
        ``(first, last)`` calendar years, inclusive, in UTC.
    record:
        The run's record, :func:`run_record`. Only the windows lying inside it
        are kept: the drivers' record ends three hours before its last day
        does, so that day's last window is not.

    Returns
    -------
    dict
        Observation source name to its observed values on ``(site, time)``,
        in the order of ``config.NEE_WINDOWS``.
    """
    rates = nee.net_ecosystem_exchange_fields([series_name], sites=[config.SITE])[
        series_name
    ]
    flags = nee.net_ecosystem_exchange_quality_flags(
        [series_name], sites=[config.SITE]
    )[series_name]
    measured = _measured_indicator(flags)
    observed_values_by_source = {}
    for name, (start, end) in config.NEE_WINDOWS.items():
        windows = _windows_inside(_daily_windows(period, start, end), record)
        mean_rate = reduce_windows(rates, windows, "mean")
        measured_fraction = reduce_windows(measured, windows, "mean")
        kept = mean_rate.where(
            measured_fraction >= config.NEE_MINIMUM_MEASURED_FRACTION
        )
        kept.attrs = {
            **rates.attrs,
            "comment": (
                f"Mean of {series_name} over the window, kept where at least "
                f"{config.NEE_MINIMUM_MEASURED_FRACTION:g} of its values were "
                "measured rather than gap-filled."
            ),
        }
        observed_values_by_source[name] = _with_windows(kept, windows)
    return observed_values_by_source


def observed_leaf_area_index(record: pd.Interval) -> xr.DataArray:
    """The MODIS composites labeled inside the run's record."""
    field = inputs.constraint_fields()["modis_leaf_area_index"]
    labels = pd.DatetimeIndex(field[TIME].values)
    inside = np.array([label in record for label in labels])
    return field.isel({TIME: np.flatnonzero(inside)})


def observed_aboveground_biomass(record: pd.Interval) -> xr.DataArray:
    """LandTrendr in ``config.LANDTRENDR_YEARS``, relabeled as dry biomass."""
    field = inputs.constraint_fields()["landtrendr_aboveground_biomass"]
    first, last = config.LANDTRENDR_YEARS
    years = pd.DatetimeIndex(field[WINDOW_START].values).year
    field = field.isel({TIME: np.flatnonzero((years >= first) & (years <= last))})
    windows = pd.IntervalIndex.from_arrays(
        pd.DatetimeIndex(field[WINDOW_START].values),
        pd.DatetimeIndex(field[WINDOW_END].values),
        closed="right",
    )
    check_windows_are_inside_the_run(windows, record, "landtrendr_aboveground_biomass")
    attrs = {key: value for key, value in field.attrs.items() if key != "constituent"}
    attrs["comment"] = (
        "Taken by this experiment as dry biomass, not carbon: at the site it is "
        "about twice the carbon density of the initial conditions' aboveground "
        "carbon map, and the spec records the constituent as unconfirmed by "
        "about that factor."
    )
    relabeled = field.copy()
    relabeled.attrs = attrs
    return relabeled


def observed_soil_carbon() -> xr.DataArray:
    """SoilGrids' one static value at the site."""
    return inputs.constraint_fields()["soilgrids_soil_organic_carbon"]


# ── entry point ──


def main() -> int:
    """Build both observation vectors and print what they hold."""
    try:
        calibration = calibration_observation_vector()
        validation = validation_observation_vector()
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"calibration: {calibration!r}")
        print(calibration.describe().to_string())
        print(f"\nvalidation: {validation!r}")
        print(validation.describe().to_string())
        print("\nNEE windows kept, as a fraction of the period's windows, by month")
        for label, vector, period in (
            ("calibration", calibration, config.CALIBRATION_NEE_PERIOD),
            ("validation", validation, config.VALIDATION_NEE_PERIOD),
        ):
            for name in config.NEE_WINDOWS:
                print(f"  {label} {name}: {_kept_by_month(vector[name], period)}")
    return 0


# ── helpers ──


def _observation_vector_of(
    observed_values_by_source: dict[str, xr.DataArray],
) -> ObservationVector:
    """The observation vector of these observed values, each with its configured operator."""
    return ObservationVector(
        observation_sources=[
            ObservationSource(
                observation_source_name=name,
                observed_values=observed_values,
                operator=config.OBSERVATION_OPERATORS[name],
            )
            for name, observed_values in observed_values_by_source.items()
        ]
    )


def _daily_windows(
    period: tuple[int, int], start: pd.Timedelta, end: pd.Timedelta
) -> pd.IntervalIndex:
    """``(day + start, day + end]`` for every UTC day of the years in *period*."""
    first, last = period
    days = pd.date_range(f"{first}-01-01", f"{last}-12-31", freq="D")
    return pd.IntervalIndex.from_arrays(days + start, days + end, closed="right")


def _windows_inside(windows: pd.IntervalIndex, record: pd.Interval) -> pd.IntervalIndex:
    """The *windows* lying wholly inside *record*."""
    inside = (windows.left >= record.left) & (windows.right <= record.right)
    return windows[inside]


def _measured_indicator(flags: xr.DataArray) -> xr.DataArray:
    """1 where a value was measured (quality flag 0), 0 where gap-filled or absent."""
    measured = (flags == 0).astype(float)
    measured.attrs = {"units": "1", "long_name": "measured, not gap-filled"}
    return measured


def _with_windows(field: xr.DataArray, windows: pd.IntervalIndex) -> xr.DataArray:
    """*field*, one value per window, with each window's edges on ``time``."""
    return field.assign_coords(
        {
            WINDOW_START: (TIME, windows.left.to_numpy()),
            WINDOW_END: (TIME, windows.right.to_numpy()),
        }
    )


def _kept_by_month(source: ObservationSource, period: tuple[int, int]) -> list[float]:
    """The fraction of each month's windows in *period* the source keeps."""
    kept = pd.DatetimeIndex(source.observed_values[WINDOW_START].values)
    first, last = period
    days = pd.date_range(f"{first}-01-01", f"{last}-12-31", freq="D")
    kept_counts = pd.Series(kept.month).value_counts()
    all_counts = pd.Series(days.month).value_counts()
    return [
        round(float(kept_counts.get(m, 0) / all_counts[m]), 2) for m in range(1, 13)
    ]


# ── checks ──


def check_windows_are_inside_the_run(
    windows: pd.IntervalIndex, record: pd.Interval, observation_source_name: str
) -> None:
    """Every window of an observation source lies inside the run's record."""
    outside = (windows.left < record.left) | (windows.right > record.right)
    if outside.any():
        first = windows[np.flatnonzero(outside)[0]]
        raise ValueError(
            f"{observation_source_name}: the window {first} is not inside the "
            f"run's record {record}; shorten the period to the drivers' record"
        )


if __name__ == "__main__":
    sys.exit(main())
