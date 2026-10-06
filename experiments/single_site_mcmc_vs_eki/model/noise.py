"""The noise model: each observation source's error as a Gaussian noise factor.

``MODEL.md``, "Noise model", states the model exactly; this module declares
it. Source :math:`k`'s observed values :math:`y_k` are Gaussian about its
prediction :math:`m_k`, independent of every other source's,

.. math::

    y_k \\mid m_k \\sim \\mathcal N(m_k,\\ R_k), \\qquad
    R_k = \\Sigma^{\\mathrm{obs}}_k + \\Sigma^{\\delta}_k,

so that :math:`R = \\operatorname{diag}(R_1, \\dots, R_K)`. Each
:math:`R_k` is a covariance spec, the sum of the measurement error
:math:`\\Sigma^{\\mathrm{obs}}_k`, built from the source's standard
deviations (``model/observations.py``), and the model discrepancy
:math:`\\Sigma^{\\delta}_k`, whose values are ``config``'s; a dated source's
is one block per site. The functions below compute each term from the
source's constants
(:meth:`~sipnet_calibration.observation.ObservationVector.constants`), with
times in seconds since the epoch. Nothing reads a parameter, so ``R`` is
built and factored once, when the posterior is conditioned
(``model/calibration.py``).

Functions
---------
:func:`noise_factors`
    One noise factor per source of an observation vector.
:func:`noise_covariance_blocks`, :func:`noise_standard_deviations`
    A posterior's ``R_k``, per source, and the square roots of their
    diagonals as labeled values.
:func:`noise_summary`
    What each source contributes to ``R``.

The remaining public functions are the covariances' terms.

Usage
-----
::

    from experiments.single_site_mcmc_vs_eki.model import calibration, noise
    posterior = calibration.calibration_posterior()
    noise.noise_summary(posterior, calibration.calibration_observation_vector())
"""

from functools import partial

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.observation import STANDARD_DEVIATION, ObservationVector
from sipnet_calibration.observation.model import noise_factor
from sipnet_calibration.probability import (
    BlockDiagonalSpec,
    CovarianceSpec,
    DenseSpec,
    DiagonalSpec,
    FactorSpec,
    LabeledValues,
    Posterior,
    SumSpec,
)

from .. import config
from .discrepancy import NEEDiscrepancy

__all__ = [
    "LANDTRENDR_DISCREPANCY_VARIANCE",
    "SECONDS_PER_DAY",
    "carbon_fraction_error",
    "floored_measurement_variance",
    "leaf_area_index_discrepancy",
    "measurement_variance",
    "nee_discrepancy",
    "noise_covariance_blocks",
    "noise_factors",
    "noise_standard_deviations",
    "noise_summary",
    "shared_measurement_error",
    "soil_carbon_discrepancy",
]

#: Seconds in a day: the sources' times are in seconds, ``config``'s
#: timescales in days.
SECONDS_PER_DAY = 86_400.0

#: The name of LandTrendr's discrepancy variance, a constant of its factor.
LANDTRENDR_DISCREPANCY_VARIANCE = "discrepancy_variance"


def noise_factors(observation_vector: ObservationVector) -> list[FactorSpec]:
    """The noise factor of each source of *observation_vector*, in its order.

    Raises
    ------
    KeyError
        If a source has no noise model here.
    """
    return [
        _noise_factor(observation_vector, name)
        for name in observation_vector.observation_source_names
    ]


def noise_covariance_blocks(posterior: Posterior) -> dict[str, np.ndarray]:
    """Each observed source's ``R_k``, dense, keyed by source, in y's order."""
    likelihood = posterior.gaussian_likelihood()
    return {
        name: np.asarray(block.to_dense())
        for name, block in zip(
            posterior.observations.component_names,
            likelihood.noise_covariance.blocks,
            strict=True,
        )
    }


def noise_standard_deviations(posterior: Posterior) -> LabeledValues:
    """``sqrt(diag R_k)`` for each observed source, labeled as its observations."""
    return posterior.observations.values_to_labeled(
        {
            name: np.sqrt(np.diag(block))
            for name, block in noise_covariance_blocks(posterior).items()
        }
    )


def noise_summary(
    posterior: Posterior, observation_vector: ObservationVector
) -> pd.DataFrame:
    """What each source of *observation_vector* contributes to the posterior's ``R``.

    One row per source: ``observations``; the source's ``units``;
    ``measurement`` and ``discrepancy``, medians of each observation's
    standard deviations, the second
    :math:`\\sqrt{\\max(R_{ii} - \\sigma_i^2, 0)}`; ``total``, the median of
    :math:`\\sqrt{R_{ii}}`; ``effective_n``, how many independent observations
    of the median total variance would constrain a shift common to the
    whole source as tightly, :math:`(\\mathbf 1^\\top R_k^{-1} \\mathbf 1)`
    times that variance; and :math:`R_k`'s smallest eigenvalue.
    """
    rows = []
    for name, block in noise_covariance_blocks(posterior).items():
        measurement = observation_vector.constants(name)[STANDARD_DEVIATION].values
        variance = np.diag(block)
        total = np.sqrt(variance)
        ones = np.ones(block.shape[0])
        rows.append(
            {
                "observation_source": name,
                "observations": block.shape[0],
                "units": observation_vector[name].observed_values.attrs.get("units"),
                "measurement": np.median(measurement),
                "discrepancy": np.median(
                    np.sqrt(np.clip(variance - measurement**2, 0, None))
                ),
                "total": np.median(total),
                "effective_n": float(ones @ np.linalg.solve(block, ones))
                * np.median(total) ** 2,
                "smallest_eigenvalue": np.linalg.eigvalsh(block)[0],
            }
        )
    return pd.DataFrame(rows).set_index("observation_source")


# ── the terms: measurement error ──


def measurement_variance(standard_deviation):
    """:math:`\\Sigma^{\\mathrm{obs}} = \\operatorname{diag}(\\sigma_i^2)`, as its diagonal."""
    return standard_deviation**2


def floored_measurement_variance(standard_deviation):
    """:math:`\\max(\\sigma_i, \\sigma_{\\min})^2`, ``config.LAI_STANDARD_DEVIATION_FLOOR``."""
    return jnp.maximum(standard_deviation, config.LAI_STANDARD_DEVIATION_FLOOR) ** 2


def shared_measurement_error(standard_deviation):
    """:math:`\\sigma \\sigma^\\top`: LandTrendr's error, shared by every year."""
    return jnp.outer(standard_deviation, standard_deviation)


def carbon_fraction_error(observed):
    """:math:`\\kappa^2 y y^\\top`: the carbon fraction's error, shared by
    every year, :math:`\\kappa` its relative uncertainty,
    ``config.WOOD_CARBON_FRACTION_UNCERTAINTY`` over
    ``config.WOOD_CARBON_FRACTION``."""
    relative = config.WOOD_CARBON_FRACTION_UNCERTAINTY / config.WOOD_CARBON_FRACTION
    return relative**2 * jnp.outer(observed, observed)


# ── the terms: discrepancy ──


def nee_discrepancy(time_since_epoch, *, discrepancy: NEEDiscrepancy):
    """:math:`\\Sigma^\\delta_{WW'}` of ``model/discrepancy.py``, in the window ends."""
    return discrepancy.covariance(time_since_epoch / SECONDS_PER_DAY, xp=jnp)


def leaf_area_index_discrepancy(time_since_epoch, calendar_year):
    """The LAI discrepancy, correlated within a summer and independent across summers:

    .. math::

        \\sigma_\\delta^2 \\, e^{-|t - t'| / \\tau} \\,
        \\mathbf 1[\\mathrm{year}(t) = \\mathrm{year}(t')],

    :math:`\\sigma_\\delta` ``config.LAI_DISCREPANCY_STANDARD_DEVIATION`` and
    :math:`\\tau` ``config.LAI_DISCREPANCY_TIMESCALE``.
    """
    lag = jnp.abs(time_since_epoch[:, None] - time_since_epoch[None, :])
    timescale = config.LAI_DISCREPANCY_TIMESCALE.total_seconds()
    same_summer = calendar_year[:, None] == calendar_year[None, :]
    return (
        config.LAI_DISCREPANCY_STANDARD_DEVIATION**2
        * jnp.exp(-lag / timescale)
        * same_summer
    )


def soil_carbon_discrepancy(observed):
    """:math:`(f y)^2`, ``config.SOIL_CARBON_DISCREPANCY_FRACTION``: proportional to the stock."""
    return (config.SOIL_CARBON_DISCREPANCY_FRACTION * observed) ** 2


# ── each source's covariance ──


def _noise_factor(observation_vector: ObservationVector, name: str) -> FactorSpec:
    """Source *name*'s noise factor, with its covariance and provenance."""
    check_source_has_a_noise_model(name)
    if name in config.NEE_WINDOWS:
        discrepancy = config.NEE_DISCREPANCY[name]
        return noise_factor(
            observation_vector,
            name,
            covariance=_per_site(
                DiagonalSpec(measurement_variance),
                DenseSpec(partial(nee_discrepancy, discrepancy=discrepancy)),
            ),
            provenance=(
                "AmeriFlux RANDUNC and JOINTUNC per window; discrepancy "
                f"{discrepancy.provenance}."
            ),
        )
    return _CONSTRAINT_NOISE_FACTORS[name](observation_vector, name)


def _per_site(*terms: CovarianceSpec) -> BlockDiagonalSpec:
    """The sum of *terms*, one block per site."""
    return BlockDiagonalSpec(SumSpec(*terms), by="site")


def _leaf_area_index_factor(observation_vector, name) -> FactorSpec:
    """MODIS LAI: floored measurement error plus discrepancy within each summer, per site."""
    return noise_factor(
        observation_vector,
        name,
        covariance=_per_site(
            DiagonalSpec(floored_measurement_variance),
            DenseSpec(leaf_area_index_discrepancy),
        ),
        provenance=(
            "MCD15A3H LAI_StdDev floored at the reanalysis's "
            f"{config.LAI_STANDARD_DEVIATION_FLOOR} m2 m-2; discrepancy "
            f"{config.LAI_DISCREPANCY_STANDARD_DEVIATION} m2 m-2 over "
            f"{config.LAI_DISCREPANCY_TIMESCALE.days} days within a summer "
            "(reasoned)."
        ),
    )


def _landtrendr_factor(observation_vector, name) -> FactorSpec:
    """LandTrendr: its error and the carbon fraction's, each shared by every
    year, plus independent discrepancy, per site."""
    return noise_factor(
        observation_vector,
        name,
        covariance=_per_site(
            DenseSpec(shared_measurement_error),
            DenseSpec(carbon_fraction_error),
            DiagonalSpec(LANDTRENDR_DISCREPANCY_VARIANCE),
        ),
        constants={
            LANDTRENDR_DISCREPANCY_VARIANCE: xr.DataArray(
                config.LANDTRENDR_DISCREPANCY_STANDARD_DEVIATION**2
            )
        },
        provenance=(
            "LandTrendr's reported error and the carbon fraction's, "
            f"{config.WOOD_CARBON_FRACTION} +/- "
            f"{config.WOOD_CARBON_FRACTION_UNCERTAINTY}, each shared by every "
            "year; independent discrepancy "
            f"{config.LANDTRENDR_DISCREPANCY_STANDARD_DEVIATION} Mg ha-1 (reasoned)."
        ),
    )


def _soil_carbon_factor(observation_vector, name) -> FactorSpec:
    """SoilGrids: measurement error plus a discrepancy proportional to the stock."""
    return noise_factor(
        observation_vector,
        name,
        covariance=SumSpec(
            DiagonalSpec(measurement_variance), DiagonalSpec(soil_carbon_discrepancy)
        ),
        provenance=(
            "SoilGrids' reported error; discrepancy "
            f"{config.SOIL_CARBON_DISCREPANCY_FRACTION:.0%} of the stock, for "
            "the depth and definition SIPNET's soil pool does not share "
            "(reasoned)."
        ),
    )


#: The noise factor of each constraint source; the NEE sources are
#: config.NEE_WINDOWS. Built last: it names the builders above.
_CONSTRAINT_NOISE_FACTORS = {
    "modis_leaf_area_index": _leaf_area_index_factor,
    "landtrendr_aboveground_biomass": _landtrendr_factor,
    "soilgrids_soil_organic_carbon": _soil_carbon_factor,
}


# ── checks ──


def check_source_has_a_noise_model(name: str) -> None:
    """Every observation source has a noise model defined for it."""
    if name not in config.NEE_WINDOWS and name not in _CONSTRAINT_NOISE_FACTORS:
        raise KeyError(
            f"no noise model is defined for the observation source {name!r}; "
            "add one to model/noise.py and its terms to config.py"
        )
