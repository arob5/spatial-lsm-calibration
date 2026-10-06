"""The experiment's observation sources: observed values, measurement errors, operators.

Each observation source's observed values and measurement standard
deviations are built from ``inputs`` as ``config`` says, and paired with the
operator ``config.OBSERVATION_OPERATORS`` binds to it; the sources make the
calibration and the validation observation vectors. Nothing here converts
units or touches a model: the observed values keep their data source's
units, and each operator predicts them from the model. ``MODEL.md``, "The
observation model", states each source exactly.

Observation sources
-------------------
``nee_night_centered``, ``nee_day_centered``
    Observed NEE, ``umol m-2 s-1`` of CO2, averaged over each of the two
    twelve-hour windows of every UTC day in the period
    (``config.NEE_WINDOWS``) that lies inside the run's record, from the
    series' gap-filled values. A window is an observation only when at least
    ``config.NEE_MINIMUM_MEASURED_FRACTION`` of its values were measured
    (quality flag 0); the others are dropped. Its ``time`` is the window's
    end, and ``window_start``/``window_end`` carry the window. Its standard
    deviation is :func:`reported_nee_window_standard_deviations`, or the
    source's median where none of the window's values reports one.
``modis_leaf_area_index``
    The MODIS composites, as processed, whose labels fall inside the run's
    record, with their reported standard deviations.
``landtrendr_aboveground_biomass``
    LandTrendr's annual values in ``config.LANDTRENDR_YEARS``, with their
    year windows and reported standard deviations, relabeled as dry
    biomass: the ``constituent`` attribute is dropped and a ``comment`` says
    why (``config.WOOD_CARBON_FRACTION``).
``soilgrids_soil_organic_carbon``
    The one static value, as processed, with its reported standard
    deviation.

The run's record is the interval the prepared drivers cover, which is what
the operators read the model over; every dated observation lies inside it.

Usage
-----
::

    from experiments.single_site_mcmc_vs_eki.model import observations
    calibration = observations.calibration_observation_vector()
    calibration.describe()
    calibration.observed_values_by_component()   # what the posterior conditions on
"""

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration import constraints
from sipnet_calibration import net_ecosystem_exchange as nee
from sipnet_calibration.conventions import (
    TIME,
    TIMESTEP_START,
    WINDOW_END,
    WINDOW_START,
)
from sipnet_calibration.observation import ObservationSource, ObservationVector
from sipnet_calibration.observation.time_alignment import reduce_windows

from .. import config
from . import inputs

__all__ = [
    "aboveground_biomass_source",
    "calibration_observation_vector",
    "leaf_area_index_source",
    "nee_observation_vector",
    "nee_window_sources",
    "reported_nee_window_standard_deviations",
    "run_record",
    "soil_carbon_source",
    "validation_observation_vector",
]


def calibration_observation_vector() -> ObservationVector:
    """The observations the calibration conditions on, one source per observed quantity."""
    record = run_record()
    return ObservationVector(
        observation_sources=[
            *nee_window_sources(
                config.CALIBRATION_NEE_SERIES, config.CALIBRATION_NEE_PERIOD, record
            ),
            leaf_area_index_source(record),
            aboveground_biomass_source(record),
            soil_carbon_source(),
        ]
    )


def validation_observation_vector() -> ObservationVector:
    """The held-out NEE of the second tower, over the years the calibration does not see."""
    return nee_observation_vector(
        config.VALIDATION_NEE_SERIES, config.VALIDATION_NEE_PERIOD
    )


def nee_observation_vector(
    series_name: str, period: tuple[int, int]
) -> ObservationVector:
    """One NEE series' two window sources over *period*, ``(first, last)`` years."""
    return ObservationVector(
        observation_sources=nee_window_sources(series_name, period, run_record())
    )


def run_record() -> pd.Interval:
    """The interval the prepared drivers cover, ``(first step's start, last step's end]``."""
    driver_data = inputs.driver_dataset()
    return pd.Interval(
        pd.Timestamp(driver_data[TIMESTEP_START].values[0]),
        pd.Timestamp(driver_data[TIME].values[-1]),
        closed="right",
    )


def nee_window_sources(
    series_name: str, period: tuple[int, int], record: pd.Interval
) -> list[ObservationSource]:
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
    list of ObservationSource
        One per window of ``config.NEE_WINDOWS``, in its order.

    Raises
    ------
    ValueError
        If no kept window of a source reports an uncertainty.
    """
    rates = nee.net_ecosystem_exchange_fields([series_name], sites=[config.SITE])[
        series_name
    ]
    flags = nee.net_ecosystem_exchange_quality_flags(
        [series_name], sites=[config.SITE]
    )[series_name]
    measured = _measured_indicator(flags)
    sources = []
    for name, (start, end) in config.NEE_WINDOWS.items():
        windows = _windows_inside(_daily_windows(period, start, end), record)
        mean_rate = reduce_windows(rates, windows, "mean")
        is_kept = (
            reduce_windows(measured, windows, "mean")
            >= config.NEE_MINIMUM_MEASURED_FRACTION
        )
        kept = mean_rate.where(is_kept)
        kept.attrs = {
            **rates.attrs,
            "comment": (
                f"Mean of {series_name} over the window, kept where at least "
                f"{config.NEE_MINIMUM_MEASURED_FRACTION:g} of its values were "
                "measured rather than gap-filled."
            ),
        }
        reported = mean_rate.copy(
            data=reported_nee_window_standard_deviations(series_name, windows)[None]
        ).where(is_kept)
        check_some_window_reports_an_uncertainty(reported, name)
        standard_deviation = reported.fillna(float(reported.median()))
        standard_deviation.attrs = {
            **rates.attrs,
            "comment": (
                "The window's measurement error; the source's median where "
                "none of the window's values reports an uncertainty."
            ),
        }
        sources.append(
            ObservationSource(
                observation_source_name=name,
                observed_values=_with_windows(kept, windows),
                standard_deviation=_with_windows(standard_deviation, windows),
                operator=config.OBSERVATION_OPERATORS[name],
            )
        )
    return sources


def reported_nee_window_standard_deviations(
    series_name: str, windows: pd.IntervalIndex
) -> np.ndarray:
    """Each window's measurement-error standard deviation at the site, ``(len(windows),)``.

    With :math:`r` a value's random uncertainty, :math:`j` its joint
    uncertainty and :math:`u = \\sqrt{\\max(j^2 - r^2, 0)}` its u* part, a
    window of :math:`n` values has

    .. math::

        \\sigma_W = \\sqrt{\\overline{r^2} / n + \\bar u^2},

    each mean over the window's values that report it, and :math:`n`
    counting every value of the window: the random part averages down over
    the window, the u* part does not (``MODEL.md``, "Measurement error").
    NaN where none of the window's values reports an uncertainty.
    """
    random = _series_at_site(
        nee.net_ecosystem_exchange_random_uncertainties, series_name
    )
    joint = _series_at_site(nee.net_ecosystem_exchange_joint_uncertainties, series_name)
    # get_indexer needs the times at the windows' precision.
    window_of_value = windows.get_indexer(random.index.as_unit(windows.left.unit))
    in_a_window = window_of_value >= 0
    table = pd.DataFrame(
        {
            "window": window_of_value[in_a_window],
            "random_variance": random.to_numpy()[in_a_window] ** 2,
            "ustar": np.sqrt(
                np.clip(joint.to_numpy() ** 2 - random.to_numpy() ** 2, 0, None)
            )[in_a_window],
        }
    )
    grouped = table.groupby("window")
    counts = grouped.size().reindex(range(len(windows)))
    random_variance = grouped["random_variance"].mean().reindex(range(len(windows)))
    ustar = grouped["ustar"].mean().reindex(range(len(windows)))
    return np.sqrt(random_variance / counts + ustar**2).to_numpy()


def leaf_area_index_source(record: pd.Interval) -> ObservationSource:
    """The MODIS composites labeled inside the run's record."""
    name = "modis_leaf_area_index"
    observed_values, standard_deviation = _constraint_at_site(name)
    labels = pd.DatetimeIndex(observed_values[TIME].values)
    inside = {TIME: np.flatnonzero([label in record for label in labels])}
    return _constraint_source(
        name, observed_values.isel(inside), standard_deviation.isel(inside)
    )


def aboveground_biomass_source(record: pd.Interval) -> ObservationSource:
    """LandTrendr in ``config.LANDTRENDR_YEARS``, relabeled as dry biomass."""
    name = "landtrendr_aboveground_biomass"
    observed_values, standard_deviation = _constraint_at_site(name)
    first, last = config.LANDTRENDR_YEARS
    years = pd.DatetimeIndex(observed_values[WINDOW_START].values).year
    in_years = {TIME: np.flatnonzero((years >= first) & (years <= last))}
    observed_values = observed_values.isel(in_years)
    windows = pd.IntervalIndex.from_arrays(
        pd.DatetimeIndex(observed_values[WINDOW_START].values),
        pd.DatetimeIndex(observed_values[WINDOW_END].values),
        closed="right",
    )
    check_windows_are_inside_the_run(windows, record, name)
    return _constraint_source(
        name,
        _as_dry_biomass(observed_values),
        _as_dry_biomass(standard_deviation.isel(in_years)),
    )


def soil_carbon_source() -> ObservationSource:
    """SoilGrids' one static value at the site."""
    name = "soilgrids_soil_organic_carbon"
    return _constraint_source(name, *_constraint_at_site(name))


# ── helpers ──


def _constraint_at_site(name: str) -> tuple[xr.DataArray, xr.DataArray]:
    """A constraint's values and reported standard deviations at the site."""
    standard_deviations = constraints.constraint_standard_deviations(
        [name], sites=[config.SITE]
    )
    return inputs.constraint_fields()[name], standard_deviations[name]


def _constraint_source(
    name: str, observed_values: xr.DataArray, standard_deviation: xr.DataArray
) -> ObservationSource:
    """A constraint as an observation source, with its configured operator."""
    return ObservationSource(
        observation_source_name=name,
        observed_values=observed_values,
        standard_deviation=standard_deviation,
        operator=config.OBSERVATION_OPERATORS[name],
    )


def _as_dry_biomass(field: xr.DataArray) -> xr.DataArray:
    """LandTrendr's *field* without its ``constituent``, the reason in ``comment``."""
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


def _series_at_site(reader, series_name: str) -> pd.Series:
    """One of a NEE series' companion variables at the site, as a time series."""
    field = reader([series_name], sites=[config.SITE])[series_name]
    return field.squeeze().to_series()


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


def check_some_window_reports_an_uncertainty(
    reported: xr.DataArray, observation_source_name: str
) -> None:
    """At least one kept NEE window reports an uncertainty, to stand for those that do not."""
    if not bool(reported.notnull().any()):
        raise ValueError(
            f"{observation_source_name}: no window reports a random uncertainty "
            "in any value, so no measurement error can be given; check the "
            "series' RANDUNC column"
        )
