"""The noise model: the covariance of the observation errors, and the likelihood it defines.

``MODEL.md``, "The observation model", states the model exactly; this module
builds it. The covariance ``R`` of ``y`` is block-diagonal over the
observation sources, and each source's block is its measurement error plus a
model discrepancy:

    R = diag(R_1, ..., R_K),    R_k = Sigma_obs_k + Sigma_delta_k.

The measurement errors are read from the data sources (NEE's random and
joint uncertainties, the constraints' standard deviations); the discrepancy
terms, the floors and the timescales are ``config``'s. ``R`` does not depend
on the parameters, so it is built once and factored once.

Functions
---------
:func:`calibration_likelihood`, :func:`validation_likelihood`
    The Gaussian of ``y`` about the predictions, a ``pyeki.gauss.Gaussian``
    whose ``log_density(predictions)`` is the log likelihood, ``(..., N) ->
    (...)``.
:func:`noise_covariance`
    ``R`` as a pyEKI operator, one dense block per observation source.
:func:`noise_covariance_blocks`
    Each source's block as a NumPy array, for inspection.
:func:`measurement_standard_deviations`
    Each observation's measurement-error standard deviation, before any
    floor or discrepancy.
:func:`noise_summary`
    What each observation source contributes to ``R``.

Usage
-----
::

    from experiments.single_site_mcmc_vs_eki.model import noise, observations
    vector = observations.calibration_observation_vector()
    likelihood = noise.calibration_likelihood(vector)
    likelihood.log_density(predictions)      # predictions (J, N) -> (J,)
"""

from collections.abc import Callable
from datetime import timedelta

import jax.numpy as jnp
import numpy as np
import pandas as pd
from pyeki.gauss import Gaussian
from pyeki.linalg import DensePSD, PSDBlockDiag

from sipnet_calibration import constraints
from sipnet_calibration import net_ecosystem_exchange as nee
from sipnet_calibration.conventions import TIME
from sipnet_calibration.observation import ObservationSource, ObservationVector
from sipnet_calibration.observation.time_alignment import windows_from_observed_values

from .. import config
from . import observations

__all__ = [
    "calibration_likelihood",
    "measurement_standard_deviations",
    "noise_covariance",
    "noise_covariance_blocks",
    "noise_summary",
    "validation_likelihood",
]


def calibration_likelihood(vector: ObservationVector | None = None) -> Gaussian:
    """The calibration's likelihood: ``N(y, R)`` over the calibration vector.

    *vector* defaults to ``observations.calibration_observation_vector()``.
    """
    if vector is None:
        vector = observations.calibration_observation_vector()
    return _gaussian_of(vector, config.CALIBRATION_NEE_SERIES)


def validation_likelihood(vector: ObservationVector | None = None) -> Gaussian:
    """The held-out check's likelihood: ``N(y, R)`` over the validation vector.

    *vector* defaults to ``observations.validation_observation_vector()``.
    """
    if vector is None:
        vector = observations.validation_observation_vector()
    return _gaussian_of(vector, config.VALIDATION_NEE_SERIES)


def noise_covariance(vector: ObservationVector, nee_series_name: str) -> PSDBlockDiag:
    """``R`` for *vector*, one ``DensePSD`` block per observation source, in Flat order.

    Parameters
    ----------
    vector:
        An observation vector of the experiment's observation sources, at one
        site.
    nee_series_name:
        The NEE series the vector's NEE sources were built from, whose
        hourly uncertainties give their measurement errors.

    Raises
    ------
    KeyError
        If an observation source has no block builder here.
    ValueError
        If the vector is not at one site, or a source's observations are not
        one contiguous run of Flat in time order.
    """
    blocks = noise_covariance_blocks(vector, nee_series_name)
    return PSDBlockDiag(
        tuple(DensePSD(jnp.asarray(block)) for block in blocks.values())
    )


def noise_covariance_blocks(
    vector: ObservationVector, nee_series_name: str
) -> dict[str, np.ndarray]:
    """Each observation source's block of ``R``, in Flat order, as NumPy arrays.

    Parameters and Raises as :func:`noise_covariance`.
    """
    check_vector_is_at_one_site(vector)
    standard_deviations = measurement_standard_deviations(vector, nee_series_name)
    blocks = {}
    start = 0
    for name in vector:
        source = vector[name]
        check_source_is_contiguous_in_time_order(vector, name, start)
        blocks[name] = _block_builder(name)(source, standard_deviations[name])
        start += source.n_observations
    return blocks


def measurement_standard_deviations(
    vector: ObservationVector, nee_series_name: str
) -> dict[str, np.ndarray]:
    """Each observation's measurement-error standard deviation, per observation source.

    As ``MODEL.md``, "Measurement error", defines them: for a NEE window, the
    random part averaged down over its values and the u* part not, and the
    median of the source's windows for a window none of whose values reports
    an uncertainty; for a constraint, its data source's own standard
    deviation. In each source's observation order, and in its observed
    values' units.
    """
    return {
        name: (
            _nee_window_standard_deviations(vector[name], nee_series_name)
            if name in config.NEE_WINDOWS
            else _constraint_standard_deviations(vector[name])
        )
        for name in vector
    }


def noise_summary(vector: ObservationVector, nee_series_name: str) -> pd.DataFrame:
    """What each observation source contributes to ``R``, one row per source.

    The columns are the number of observations; ``unreported``, the NEE
    windows none of whose values reports an uncertainty, which take their
    source's median measurement error; the source's units; ``measurement``
    and ``discrepancy``, medians of each observation's standard deviations;
    ``total``, the median of ``sqrt(diag R_k)``; ``effective_n``, how many
    independent observations of the median total variance would constrain a
    shift common to the whole source as tightly, ``(1' R_k^-1 1)`` times that
    variance; and ``R_k``'s smallest eigenvalue.
    """
    measurement = measurement_standard_deviations(vector, nee_series_name)
    blocks = noise_covariance_blocks(vector, nee_series_name)
    rows = []
    for name, block in blocks.items():
        unreported = (
            int(
                np.isnan(
                    _reported_nee_window_standard_deviations(
                        vector[name], nee_series_name
                    )
                ).sum()
            )
            if name in config.NEE_WINDOWS
            else 0
        )
        total = np.sqrt(np.diag(block))
        discrepancy = np.sqrt(np.clip(np.diag(block) - measurement[name] ** 2, 0, None))
        ones = np.ones(block.shape[0])
        effective = float(ones @ np.linalg.solve(block, ones)) * np.median(total) ** 2
        rows.append(
            {
                "observation_source": name,
                "observations": block.shape[0],
                "unreported": unreported,
                "units": vector[name].observed_values.attrs.get("units"),
                "measurement": np.median(measurement[name]),
                "discrepancy": np.median(discrepancy),
                "total": np.median(total),
                "effective_n": effective,
                "smallest_eigenvalue": np.linalg.eigvalsh(block)[0],
            }
        )
    return pd.DataFrame(rows).set_index("observation_source")


# ── the blocks, one builder per kind of observation source ──


def _nee_block(source: ObservationSource, measurement: np.ndarray) -> np.ndarray:
    """Measurement error on the diagonal plus discrepancy correlated in time."""
    discrepancy = config.NEE_DISCREPANCY[source.observation_source_name]
    return np.diag(measurement**2) + discrepancy.covariance(_times_in_days(source))


def _leaf_area_index_block(
    source: ObservationSource, measurement: np.ndarray
) -> np.ndarray:
    """Floored measurement error plus discrepancy correlated within each summer."""
    floored = np.maximum(measurement, config.LAI_STANDARD_DEVIATION_FLOOR)
    times = _times_in_days(source)
    years = pd.DatetimeIndex(source.observed_values[TIME].values).year.to_numpy()
    same_summer = years[:, None] == years[None, :]
    correlation = _exponential_correlation(times, config.LAI_DISCREPANCY_TIMESCALE)
    return (
        np.diag(floored**2)
        + config.LAI_DISCREPANCY_STANDARD_DEVIATION**2 * correlation * same_summer
    )


def _landtrendr_block(source: ObservationSource, measurement: np.ndarray) -> np.ndarray:
    """LandTrendr's error and the carbon fraction's, each shared by every year, plus
    independent discrepancy."""
    values = _observed_values(source)
    relative = config.WOOD_CARBON_FRACTION_UNCERTAINTY / config.WOOD_CARBON_FRACTION
    return (
        np.outer(measurement, measurement)
        + relative**2 * np.outer(values, values)
        + config.LANDTRENDR_DISCREPANCY_STANDARD_DEVIATION**2 * np.eye(values.size)
    )


def _soil_carbon_block(
    source: ObservationSource, measurement: np.ndarray
) -> np.ndarray:
    """Measurement error plus a discrepancy proportional to the stock."""
    values = _observed_values(source)
    discrepancy = config.SOIL_CARBON_DISCREPANCY_FRACTION * values
    return np.diag(measurement**2 + discrepancy**2)


#: The block builder of each constraint observation source; the NEE sources
#: are config.NEE_WINDOWS, which all take the NEE block.
_CONSTRAINT_BLOCK_BUILDERS: dict[str, Callable[..., np.ndarray]] = {
    "modis_leaf_area_index": _leaf_area_index_block,
    "landtrendr_aboveground_biomass": _landtrendr_block,
    "soilgrids_soil_organic_carbon": _soil_carbon_block,
}


def _block_builder(name: str) -> Callable[..., np.ndarray]:
    """The block builder of observation source *name*."""
    if name in config.NEE_WINDOWS:
        return _nee_block
    check_source_has_a_block_builder(name)
    return _CONSTRAINT_BLOCK_BUILDERS[name]


# ── helpers ──


def _gaussian_of(vector: ObservationVector, nee_series_name: str) -> Gaussian:
    """``N(y, R)`` for *vector*."""
    return Gaussian(jnp.asarray(vector.y), noise_covariance(vector, nee_series_name))


def _nee_window_standard_deviations(
    source: ObservationSource, series_name: str
) -> np.ndarray:
    """Each window's measurement error, the source's median where a window reports none."""
    reported = _reported_nee_window_standard_deviations(source, series_name)
    check_some_window_reports_an_uncertainty(reported, source.observation_source_name)
    return np.where(np.isnan(reported), np.nanmedian(reported), reported)


def _reported_nee_window_standard_deviations(
    source: ObservationSource, series_name: str
) -> np.ndarray:
    """``sqrt(mean(r^2) / n + mean(u)^2)`` over each window's values; NaN where none reports.

    ``r`` is the random uncertainty and ``u`` the u* part of the joint
    uncertainty, ``sqrt(max(j^2 - r^2, 0))``; each mean is over the values
    of the window that report it, and ``n`` counts every value of the window.
    """
    random = _series_at_site(
        nee.net_ecosystem_exchange_random_uncertainties, series_name
    )
    joint = _series_at_site(nee.net_ecosystem_exchange_joint_uncertainties, series_name)
    windows = windows_from_observed_values(source.observed_values.squeeze())
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


def _constraint_standard_deviations(source: ObservationSource) -> np.ndarray:
    """The constraint's own standard deviations at the source's observations."""
    constraint_name = source.observation_source_name
    standard_deviations = constraints.constraint_standard_deviations(
        [constraint_name], sites=[config.SITE]
    )[constraint_name]
    observed = source.observed_values
    if TIME in observed.dims:
        standard_deviations = standard_deviations.sel({TIME: observed[TIME]})
    values = np.atleast_1d(standard_deviations.squeeze().to_numpy())
    check_standard_deviations_are_non_negative(values, constraint_name)
    return values


def _series_at_site(reader: Callable[..., dict], series_name: str) -> pd.Series:
    """One of a NEE series' companion variables at the site, as a time series."""
    field = reader([series_name], sites=[config.SITE])[series_name]
    return field.squeeze().to_series()


def _times_in_days(source: ObservationSource) -> np.ndarray:
    """The source's time labels, in days since its first."""
    times = pd.DatetimeIndex(source.observed_values[TIME].values)
    return ((times - times[0]) / pd.Timedelta(days=1)).to_numpy()


def _exponential_correlation(times: np.ndarray, timescale: timedelta) -> np.ndarray:
    """``exp(-|t - t'| / tau)`` between every pair of *times*, in days."""
    tau = timescale / timedelta(days=1)
    return np.exp(-np.abs(times[:, None] - times[None, :]) / tau)


def _observed_values(source: ObservationSource) -> np.ndarray:
    """The source's observed values, in its observation order."""
    return np.atleast_1d(source.observed_values.squeeze().to_numpy())


# ── checks ──


def check_vector_is_at_one_site(vector: ObservationVector) -> None:
    """The noise model is for one site's observations."""
    if len(vector.sites) != 1:
        raise ValueError(
            f"the vector observes {len(vector.sites)} sites; this noise model is "
            "for one site, so select one with vector.select(sites=[...])"
        )


def check_source_is_contiguous_in_time_order(
    vector: ObservationVector, name: str, start: int
) -> None:
    """A source's observations are one run of Flat, from *start*, in time order."""
    positions = vector.positions(observation_source_name=name)
    expected = np.arange(start, start + vector[name].n_observations)
    if not np.array_equal(positions, expected):
        raise ValueError(
            f"{name}'s observations are not Flat positions {start} to "
            f"{expected[-1]} in order; the blocks of R would be misplaced"
        )


def check_source_has_a_block_builder(name: str) -> None:
    """Every observation source has a noise block defined for it."""
    if name not in _CONSTRAINT_BLOCK_BUILDERS:
        raise KeyError(
            f"no noise block is defined for the observation source {name!r}; "
            "add one to model/noise.py and its terms to config.py"
        )


def check_some_window_reports_an_uncertainty(
    reported: np.ndarray, observation_source_name: str
) -> None:
    """At least one NEE window reports a random uncertainty, to stand for those that do not."""
    if np.isnan(reported).all():
        raise ValueError(
            f"{observation_source_name}: no window reports a random uncertainty "
            "in any value, so no measurement error can be given; check the "
            "series' RANDUNC column"
        )


def check_standard_deviations_are_non_negative(values: np.ndarray, name: str) -> None:
    """A constraint's standard deviations are finite and not negative.

    A zero is allowed: every block adds a floor or a discrepancy to it.
    """
    if not (np.isfinite(values).all() and (values >= 0).all()):
        raise ValueError(
            f"{name}: its standard deviations at the observations are not all "
            "finite and non-negative; the noise model cannot use them"
        )
