"""The diagnostics of a run: the predictive check, NEE's residuals, the two towers.

``MODEL.md``, "Diagnostics", defines every quantity computed here; this
module computes them from an ensemble's predictions and writes nothing.
``scripts/diagnose.py`` runs them on a run and writes the tables, and
``figures/diagnostics.py`` draws them.

Data model
----------
The diagnostics read an ensemble's predictions of one observation vector as
a mapping from observation source name to :class:`SourcePredictions`: the
source's observed values ``y`` ``(n,)``, their window ends ``times`` (``NaT``
for a static source), the ensemble's predictions ``(J, n)``, and the
source's block of ``R`` with its measurement standard deviations. Two
builders make it: :func:`sources_from_flat` from Flat ``y`` and predictions
``(J, N)`` (an EKI run's final step), and :func:`sources_from_predictive`
from a predictive's files (``model/outputs.py``).

Functions
---------
:func:`predictive_check`
    Per source and overall: the misfits of the members, their standardized
    distance from ``n / 2``, the posterior predictive p-value, and coverage.
:func:`nee_residuals`
    Each NEE window's residual from the ensemble's median prediction.
:func:`residual_summary`, :func:`weekly_residuals`,
:func:`residual_autocorrelation`, :func:`slow_fast_split`,
:func:`night_day_correlation`
    The residuals' size beyond measurement error, their recurring seasonal
    part, their autocorrelation against the one ``R`` implies, their split
    into slow and fast parts, and the correlation of one day's night and day
    windows.
:func:`tower_comparison`
    The two towers' window differences, and the representativeness error
    they bound.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

from sipnet_calibration.conventions import SAMPLE, TIME
from sipnet_calibration.observation import ObservationVector

from .. import config
from . import noise, observations
from .discrepancy import NEEDiscrepancy

__all__ = [
    "AUTOCORRELATION_LAGS",
    "COVERAGE_LEVEL",
    "SLOW_MINIMUM_DAYS",
    "SLOW_WINDOW_DAYS",
    "SMOOTHING_WEEKS",
    "SourcePredictions",
    "check_files_hold_the_vector",
    "nee_residuals",
    "night_day_correlation",
    "predictive_check",
    "residual_autocorrelation",
    "residual_summary",
    "slow_fast_split",
    "sources_from_flat",
    "sources_from_predictive",
    "tower_comparison",
    "weekly_residuals",
]

#: The lags, in days, the residuals' autocorrelation is reported at.
AUTOCORRELATION_LAGS = (1, 2, 5, 10, 30)

#: The nominal level of the predictive interval whose coverage is reported.
COVERAGE_LEVEL = 0.9

#: The number of weeks the recurring seasonal part is smoothed over,
#: centered and circular.
SMOOTHING_WEEKS = 5

#: The length, in days, of the centered running mean that is the slow part of
#: a daily series, and the fewest observed days a mean is taken over.
SLOW_WINDOW_DAYS = 31
SLOW_MINIMUM_DAYS = 5

#: The seed of the noise draws the coverage adds to the predictions.
_COVERAGE_SEED = 0


@dataclass(frozen=True, eq=False)
class SourcePredictions:
    """One observation source's observed values and an ensemble's predictions of them.

    ``members_dropped`` counts the members left out of ``predictions`` for a
    prediction that is not finite anywhere in the vector: a failed run, or
    one the forward model failed for leaving pySIPNET's domain.
    """

    times: pd.DatetimeIndex
    y: np.ndarray
    predictions: np.ndarray
    noise_block: np.ndarray
    measurement_standard_deviation: np.ndarray
    members_dropped: int = 0


# ── the builders ──


def sources_from_flat(
    vector: ObservationVector, y, predictions, nee_series_name: str
) -> dict[str, SourcePredictions]:
    """The sources of *vector* from Flat *y* ``(N,)`` and *predictions* ``(J, N)``.

    A member whose predictions are not all finite is left out of every
    source, and counted.
    """
    y, predictions = np.asarray(y), np.asarray(predictions)
    finite = np.isfinite(predictions).all(axis=1)
    predictions = predictions[finite]
    blocks = noise.noise_covariance_blocks(vector, nee_series_name)
    measurement = noise.measurement_standard_deviations(vector, nee_series_name)
    times = vector.index.get_level_values(TIME)
    sources = {}
    for name in vector:
        positions = vector.positions(observation_source_name=name)
        sources[name] = SourcePredictions(
            times=pd.DatetimeIndex(times[positions]),
            y=y[positions],
            predictions=predictions[:, positions],
            noise_block=blocks[name],
            measurement_standard_deviation=np.asarray(measurement[name]),
            members_dropped=int((~finite).sum()),
        )
    return sources


def sources_from_predictive(
    outputs: dict, vector_name: str, vector: ObservationVector, nee_series_name: str
) -> dict[str, SourcePredictions]:
    """The sources of *vector* from a predictive's files (``model/outputs.py``).

    Raises
    ------
    ValueError
        If the files' observed values are not the vector's, in its order.
    """
    predicted = outputs["predicted"]["ensemble"][vector_name]
    y = np.concatenate(
        [
            np.atleast_1d(outputs["observed"][vector_name][name]["value"].to_numpy())
            for name in vector
        ]
    )
    check_files_hold_the_vector(y, vector, vector_name)
    predictions = np.concatenate(
        [
            predicted[name]
            .transpose(SAMPLE, ...)
            .to_numpy()
            .reshape(predicted[name].sizes[SAMPLE], -1)
            for name in vector
        ],
        axis=1,
    )
    return sources_from_flat(vector, y, predictions, nee_series_name)


# ── the predictive check ──


def predictive_check(sources: dict[str, SourcePredictions]) -> pd.DataFrame:
    """The posterior predictive check, per source and for all sources together.

    One row per source and a last row ``all``. The columns are ``n``;
    ``members``, the members scored, and ``members_dropped``, those left out
    for a prediction that is not finite;
    ``misfit_median``, ``misfit_min``, ``misfit_max``, over the members;
    ``ratio``, twice the median misfit over ``n``; ``standardized``,
    ``(median - n / 2) / sqrt(n / 2)``; ``p_value``, the members' mean of
    ``P(chi2_n >= 2 misfit)``; and ``coverage``, the fraction of observations
    inside the ensemble's :data:`COVERAGE_LEVEL` predictive interval with
    noise of the block's diagonal added.
    """
    rng = np.random.default_rng(_COVERAGE_SEED)
    rows, total = [], 0.0
    for name, source in sources.items():
        member_misfits = _member_misfits(source)
        total = total + member_misfits
        rows.append(
            {
                "source": name,
                "members": source.predictions.shape[0],
                "members_dropped": source.members_dropped,
                **_misfit_row(member_misfits, source.y.size),
                "coverage": _coverage(source, rng),
            }
        )
    n_total = sum(source.y.size for source in sources.values())
    first = next(iter(sources.values()))
    rows.append(
        {
            "source": "all",
            "members": first.predictions.shape[0],
            "members_dropped": first.members_dropped,
            **_misfit_row(total, n_total),
            "coverage": np.nan,
        }
    )
    return pd.DataFrame(rows).set_index("source")


# ── NEE's residuals ──


def nee_residuals(sources: dict[str, SourcePredictions]) -> pd.DataFrame:
    """Each NEE window's residual from the ensemble's median prediction.

    One row per window: ``source``, ``time`` (the window's end), ``observed``,
    ``predicted`` (the members' median), ``residual`` (their difference) and
    ``measurement_standard_deviation``.
    """
    frames = []
    for name in config.NEE_WINDOWS:
        if name not in sources:
            continue
        source = sources[name]
        predicted = np.median(source.predictions, axis=0)
        frames.append(
            pd.DataFrame(
                {
                    "source": name,
                    "time": source.times,
                    "observed": source.y,
                    "predicted": predicted,
                    "residual": source.y - predicted,
                    "measurement_standard_deviation": source.measurement_standard_deviation,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def residual_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    """Per NEE source: the residuals' size beyond measurement error, and their seasonality.

    Columns: ``n``; ``residual_variance``; ``measurement_variance``, the mean
    of the windows' measurement variances; ``non_measurement_standard_deviation``, the square
    root of their difference; ``discrepancy_standard_deviation``, the configured
    discrepancy's at one window; ``recurring_share``, the recurring seasonal
    part's share of the residual variance; and the correlation of each year's
    weekly means with the other years' mean, its median, minimum and maximum
    over years; and the mean residual in each season, ``mean_djf``,
    ``mean_mam``, ``mean_jja``, ``mean_son``, by the month each window
    starts in.
    """
    rows = []
    for name, frame in residuals.groupby("source", sort=False):
        residual = _residual_series(frame)
        recurring = _recurring_part(residual)
        excess = residual.var() - np.mean(frame["measurement_standard_deviation"] ** 2)
        correlations = _year_to_year_correlations(residual)
        rows.append(
            {
                "source": name,
                "n": len(residual),
                "residual_variance": residual.var(),
                "measurement_variance": np.mean(
                    frame["measurement_standard_deviation"] ** 2
                ),
                "non_measurement_standard_deviation": np.sqrt(max(excess, 0.0)),
                "discrepancy_standard_deviation": np.sqrt(
                    config.NEE_DISCREPANCY[name].total_variance()
                ),
                "recurring_share": recurring.var() / residual.var(),
                "year_correlation_median": np.median(correlations),
                "year_correlation_min": np.min(correlations),
                "year_correlation_max": np.max(correlations),
                **_seasonal_means(residual),
            }
        )
    return pd.DataFrame(rows).set_index("source")


def weekly_residuals(residuals: pd.DataFrame) -> pd.DataFrame:
    """Per NEE source, week of year and year: the mean residual, and the recurring part.

    One row per source and week, a column per year holding that year's mean
    residual in the week, and ``recurring``, the recurring seasonal part:
    the week's mean over all years, smoothed over :data:`SMOOTHING_WEEKS`
    weeks.
    """
    frames = []
    for name, frame in residuals.groupby("source", sort=False):
        residual = _residual_series(frame)
        table = (
            pd.DataFrame(
                {
                    "residual": residual.to_numpy(),
                    "week": _week_of_year(residual.index),
                    "year": residual.index.year,
                }
            )
            .groupby(["week", "year"])
            .residual.mean()
            .unstack()
            .reindex(range(1, 53))
        )
        table["recurring"] = _smoothed_weekly_means(residual)
        table.insert(0, "source", name)
        frames.append(table)
    return pd.concat(frames)


def residual_autocorrelation(residuals: pd.DataFrame) -> pd.DataFrame:
    """Per NEE source and lag: the residuals' autocorrelation, and the one ``R`` implies.

    Columns ``observed``, the autocorrelation of the residuals over the pairs
    of days both observed; ``remainder``, the same after the recurring
    seasonal part is subtracted; and ``modeled``, the residuals' correlation
    under the configured ``R``, the discrepancy's covariance at the lag over
    its variance plus the mean measurement variance.
    """
    rows = []
    for name, frame in residuals.groupby("source", sort=False):
        residual = _residual_series(frame)
        remainder = residual - _recurring_part(residual)
        modeled = _modeled_correlation(
            config.NEE_DISCREPANCY[name],
            np.mean(frame["measurement_standard_deviation"] ** 2),
        )
        for lag in AUTOCORRELATION_LAGS:
            rows.append(
                {
                    "source": name,
                    "lag_days": lag,
                    "observed": _daily(residual).autocorr(lag),
                    "remainder": _daily(remainder).autocorr(lag),
                    "modeled": modeled(lag),
                }
            )
    return pd.DataFrame(rows).set_index(["source", "lag_days"])


def slow_fast_split(residuals: pd.DataFrame) -> pd.DataFrame:
    """Per NEE source: how much of the misfit is slow and how much fast.

    Each daily series (the observations, the median prediction and their
    residual) splits into a slow part, its centered running mean over
    :data:`SLOW_WINDOW_DAYS` days, and a fast part, what remains. Columns:
    ``slow_variance`` and ``fast_variance`` of the residual;
    ``fast_correlation``, the correlation of the observations' fast part with
    the prediction's; and ``fast_standard_deviation_ratio``, the prediction's
    fast standard deviation over the observations'. ``MODEL.md``,
    "Diagnostics", defines them.
    """
    rows = []
    for name, frame in residuals.groupby("source", sort=False):
        parts = {
            column: _slow_and_fast(_daily(_residual_series(frame, column)))
            for column in ("residual", "observed", "predicted")
        }
        (residual_slow, residual_fast) = parts["residual"]
        observed_fast, predicted_fast = parts["observed"][1], parts["predicted"][1]
        rows.append(
            {
                "source": name,
                "slow_variance": residual_slow.var(),
                "fast_variance": residual_fast.var(),
                "fast_correlation": observed_fast.corr(predicted_fast),
                "fast_standard_deviation_ratio": predicted_fast.std()
                / observed_fast.std(),
            }
        )
    return pd.DataFrame(rows).set_index("source")


def night_day_correlation(residuals: pd.DataFrame) -> pd.Series:
    """The correlation of one UTC day's night-centered and day-centered residuals.

    ``same_day``: the night window ``(d, d + 12 h]`` with the day window
    ``(d + 12 h, d + 24 h]``; ``next_night``: that day window with the next
    night window; with the numbers of pairs.
    """
    by_day = {}
    for name, frame in residuals.groupby("source", sort=False):
        residual = _residual_series(frame)
        by_day[name] = pd.Series(residual.to_numpy(), index=residual.index.normalize())
    night, day = by_day["nee_night_centered"], by_day["nee_day_centered"]
    both = pd.concat({"night": night, "day": day}, axis=1).dropna()
    next_night = night.shift(-1, freq="D")
    later = pd.concat({"day": day, "night": next_night}, axis=1).dropna()
    return pd.Series(
        {
            "same_day": both["night"].corr(both["day"]),
            "same_day_pairs": len(both),
            "next_night": later["day"].corr(later["night"]),
            "next_night_pairs": len(later),
        }
    )


# ── the two towers ──


def tower_comparison(
    first_series: str, second_series: str, period: tuple[int, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two towers' NEE windows where both keep one, and what their differences bound.

    Returns the windows, one row per source and window both towers keep
    (``first``, ``second`` and their measurement standard deviations, and
    ``difference``, first minus second), and a summary per source: ``n``,
    ``first_year``, ``last_year``, ``mean_difference``,
    ``difference_variance``, ``measurement_variance`` (the mean of the two
    towers' summed measurement variances), ``representativeness_standard_deviation`` (the
    per-tower representativeness error the difference bounds, ``MODEL.md``,
    "Diagnostics"), and the mean difference per calendar quarter.
    """
    frames = {}
    for label, series in (("first", first_series), ("second", second_series)):
        vector = observations.nee_observation_vector(series, period)
        sources = sources_from_flat(
            vector, vector.y, np.asarray(vector.y)[None, :], series
        )
        frames[label] = {
            name: pd.DataFrame(
                {
                    label: source.y,
                    f"{label}_standard_deviation": source.measurement_standard_deviation,
                },
                index=source.times,
            )
            for name, source in sources.items()
        }
    windows, rows = [], []
    for name in config.NEE_WINDOWS:
        both = frames["first"][name].join(frames["second"][name], how="inner")
        both["difference"] = both["first"] - both["second"]
        measurement = np.mean(
            both["first_standard_deviation"] ** 2
            + both["second_standard_deviation"] ** 2
        )
        variance = both["difference"].var()
        quarters = both["difference"].groupby(both.index.quarter).mean()
        rows.append(
            {
                "source": name,
                "n": len(both),
                "first_year": both.index.year.min(),
                "last_year": both.index.year.max(),
                "mean_difference": both["difference"].mean(),
                "difference_variance": variance,
                "measurement_variance": measurement,
                "representativeness_standard_deviation": np.sqrt(
                    max((variance - measurement) / 2, 0)
                ),
                **{
                    f"mean_difference_q{q}": quarters.get(q, np.nan)
                    for q in range(1, 5)
                },
            }
        )
        windows.append(both.rename_axis(TIME).reset_index().assign(source=name))
    return pd.concat(windows, ignore_index=True), pd.DataFrame(rows).set_index("source")


# ── helpers ──


def _member_misfits(source: SourcePredictions) -> np.ndarray:
    """Each member's misfit to the source, ``0.5 r' R_k^-1 r``, ``(J,)``."""
    residuals = source.y[None, :] - source.predictions
    factor = cho_factor(source.noise_block, lower=True)
    whitened = cho_solve(factor, residuals.T)
    return 0.5 * np.einsum("ij,ji->i", residuals, whitened)


def _misfit_row(member_misfits: np.ndarray, n: int) -> dict:
    """The misfit columns of :func:`predictive_check` for one row."""
    median = float(np.median(member_misfits))
    return {
        "n": n,
        "misfit_median": median,
        "misfit_min": float(np.min(member_misfits)),
        "misfit_max": float(np.max(member_misfits)),
        "ratio": 2 * median / n,
        "standardized": (median - n / 2) / np.sqrt(n / 2),
        "p_value": float(np.mean(chi2.sf(2 * member_misfits, n))),
    }


def _coverage(source: SourcePredictions, rng: np.random.Generator) -> float:
    """The fraction of the source's observations inside the noisy predictive interval."""
    noise_standard_deviation = np.sqrt(np.diag(source.noise_block))
    noisy = (
        source.predictions
        + rng.standard_normal(source.predictions.shape) * noise_standard_deviation
    )
    tail = (1 - COVERAGE_LEVEL) / 2
    low, high = np.quantile(noisy, [tail, 1 - tail], axis=0)
    return float(np.mean((source.y >= low) & (source.y <= high)))


def _residual_series(frame: pd.DataFrame, column: str = "residual") -> pd.Series:
    """One source's residuals (or another *column*), indexed by the start of each window.

    The seasonal and daily groupings go by the window's start, so a window
    ending at midnight on 1 January belongs to the day and year it covers.
    """
    start, end = config.NEE_WINDOWS[frame["source"].iloc[0]]
    starts = pd.DatetimeIndex(frame["time"]) - (end - start)
    return pd.Series(frame[column].to_numpy(), index=starts)


def _seasonal_means(residual: pd.Series) -> dict[str, float]:
    """The mean residual in each meteorological season."""
    seasons = (residual.index.month % 12) // 3
    means = residual.groupby(seasons).mean()
    return {
        f"mean_{name}": means.get(season, np.nan)
        for season, name in enumerate(("djf", "mam", "jja", "son"))
    }


def _week_of_year(times: pd.DatetimeIndex) -> np.ndarray:
    """The week of the calendar year of each time, ``(day of year - 1) // 7 + 1``,
    its short 53rd week counted as 52, so a week never spans two years."""
    return np.minimum((times.dayofyear.to_numpy() - 1) // 7 + 1, 52)


def _smoothed_weekly_means(residual: pd.Series) -> pd.Series:
    """The mean residual of each week over all years, smoothed circularly, weeks 1-52."""
    weekly = (
        residual.groupby(_week_of_year(residual.index)).mean().reindex(range(1, 53))
    )
    half = SMOOTHING_WEEKS // 2
    padded = np.concatenate(
        [weekly.values[-half:], weekly.values, weekly.values[:half]]
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, SMOOTHING_WEEKS)
    return pd.Series(np.nanmean(windows, axis=1), index=weekly.index)


def _recurring_part(residual: pd.Series) -> pd.Series:
    """The recurring seasonal part at each window: its week's smoothed mean."""
    smoothed = _smoothed_weekly_means(residual)
    return pd.Series(
        smoothed.loc[_week_of_year(residual.index)].to_numpy(), index=residual.index
    )


def _year_to_year_correlations(residual: pd.Series) -> np.ndarray:
    """Each year's weekly mean residuals correlated with the other years' mean."""
    table = (
        pd.DataFrame(
            {
                "residual": residual.to_numpy(),
                "week": _week_of_year(residual.index),
                "year": residual.index.year,
            }
        )
        .groupby(["year", "week"])
        .residual.mean()
        .unstack()
    )
    correlations = []
    for year in table.index:
        both = pd.concat([table.loc[year], table.drop(year).mean()], axis=1).dropna()
        if len(both) > 10:
            correlations.append(both.corr().iloc[0, 1])
    return np.asarray(correlations)


def _daily(residual: pd.Series) -> pd.Series:
    """The residuals on a daily grid, one per day, missing days NaN."""
    return pd.Series(residual.to_numpy(), index=residual.index.normalize()).asfreq("D")


def _slow_and_fast(daily: pd.Series) -> tuple[pd.Series, pd.Series]:
    """A daily series' centered running mean, and what remains."""
    slow = daily.rolling(
        SLOW_WINDOW_DAYS, center=True, min_periods=SLOW_MINIMUM_DAYS
    ).mean()
    return slow, daily - slow


def _modeled_correlation(discrepancy: NEEDiscrepancy, measurement_variance: float):
    """The residuals' correlation at a lag under ``R``, as a function of the lag in days."""
    variance = discrepancy.total_variance() + measurement_variance

    def correlation(lag: float) -> float:
        return float(discrepancy.covariance(np.array([0.0, lag]))[0, 1] / variance)

    return correlation


# ── checks ──


def check_files_hold_the_vector(y, vector: ObservationVector, vector_name: str) -> None:
    """A predictive's observed values are the observation vector's, in its order."""
    if y.shape != np.asarray(vector.y).shape or not np.allclose(y, vector.y):
        raise ValueError(
            f"the predictive's {vector_name} observed values are not the "
            f"{vector_name} vector's; rerun the predictive with the current "
            "observations"
        )
