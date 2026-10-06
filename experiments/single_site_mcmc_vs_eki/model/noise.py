"""The noise model: each observation source's error as a Gaussian noise factor.

``MODEL.md``, "The noise model", states the model exactly; this module
declares it. Source :math:`k`'s observed values :math:`y_k` are Gaussian
about its prediction :math:`m_k`, independent of every other source's,

.. math::

    y_k \\mid m_k \\sim \\mathcal N(m_k,\\ s_k C_k), \\qquad
    C_k = \\Sigma^{\\mathrm{obs}}_k + \\Sigma^{\\delta}_k, \\qquad
    s_k \\sim \\mathrm{IG}(a, b),\\ \\operatorname{median} s_k = 1,

so that :math:`R = \\operatorname{diag}(s_1 C_1, \\dots, s_K C_K)`, the scale
:math:`s_k` present for ``config.NOISE_SCALED_SOURCES`` only. Each
:math:`C_k` is a covariance spec, the sum of the measurement error
:math:`\\Sigma^{\\mathrm{obs}}_k`, built from the source's standard
deviations (``model/observations.py``), and the model discrepancy
:math:`\\Sigma^{\\delta}_k`, whose values are ``config``'s and, for NEE, the
error model's (``model/nee_error.py``); a dated source's is one block per
site. A model with fixed noise holds every :math:`s_k` at 1
(``models.py``). The functions below compute each term from the
source's constants
(:meth:`~sipnet_calibration.observation.ObservationVector.constants`), with
times in seconds since the epoch. Nothing reads a parameter, so ``R`` is
built and factored once, when the posterior is conditioned
(``models.py``).

Functions
---------
:func:`noise_factors`, :func:`noise_scale_factors`
    One noise factor per source of an observation vector, under an NEE
    error model; the scales' prior factors.
:func:`noise_scale_name`, :func:`noise_scale_prior_scale`
    A scale's component name; the prior's ``b``, which puts its median at 1.
:func:`noise_covariance_blocks`, :func:`noise_standard_deviations`
    A posterior's ``R_k``, per source, and the square roots of their
    diagonals as labeled values.
:func:`noise_summary`
    What each source contributes to ``R``.

The remaining public functions are the covariances' terms.

Usage
-----
::

    from experiments.single_site_mcmc_vs_eki import models
    from experiments.single_site_mcmc_vs_eki.model import noise, observations
    posterior = models.fixed_posterior(models.Model.parse("long_memory/fixed"))
    noise.noise_summary(posterior, observations.calibration_observation_vector())
"""

from functools import partial

import jax.numpy as jnp
import scipy.stats
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
    POSITIVE,
    ArraySpec,
    LabeledValues,
    Posterior,
    ScaledSpec,
    SumSpec,
    inverse_gamma,
)

from .. import config
from .nee_error import NEE_ERROR_MODELS, Memory

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
    "noise_scale_factors",
    "noise_scale_name",
    "noise_scale_prior_scale",
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


def noise_factors(
    observation_vector: ObservationVector, nee_error_model_name: str
) -> list[FactorSpec]:
    """The noise factor of each source of *observation_vector*, in its order,
    NEE's under error model *nee_error_model_name* (``nee_error.NEE_ERROR_MODELS``).

    A source in ``config.NOISE_SCALED_SOURCES`` reads its scale,
    :func:`noise_scale_name`, whose factor :func:`noise_scale_factors` declares.
    """
    return [
        _noise_factor(observation_vector, name, nee_error_model_name)
        for name in observation_vector.observation_source_names
    ]


def noise_scale_factors(observation_vector: ObservationVector) -> list[FactorSpec]:
    """The prior factor of each scale *observation_vector*'s sources read:
    :math:`s_k \\sim \\mathrm{IG}(a, b)`, ``a = config.NOISE_SCALE_SHAPE`` and
    ``b`` :func:`noise_scale_prior_scale`, so that the median is 1."""
    shape = config.NOISE_SCALE_SHAPE
    return [
        FactorSpec(
            ArraySpec(noise_scale_name(name), units="1", support=POSITIVE),
            law=inverse_gamma(shape=shape, scale=noise_scale_prior_scale()),
            provenance=(
                f"inverse gamma with shape {shape} and median 1, the value a "
                "model with fixed noise holds (reasoned)."
            ),
        )
        for name in observation_vector.observation_source_names
        if name in config.NOISE_SCALED_SOURCES
    ]


def noise_scale_name(source_name: str) -> str:
    """The component name of source *source_name*'s noise scale."""
    return f"{source_name}_noise_scale"


def noise_scale_prior_scale() -> float:
    """The ``b`` of IG(a, b) with median 1: the median of Gamma(a, 1)."""
    return float(scipy.stats.gamma(config.NOISE_SCALE_SHAPE).median())


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


def nee_discrepancy(time_since_epoch, *, variance: float, memory: Memory):
    """:math:`v^\\delta \\rho(t_W - t_{W'})`, ``model/nee_error.py``'s correlation in the window ends."""
    return variance * memory.correlation(time_since_epoch / SECONDS_PER_DAY, xp=jnp)


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


def _noise_factor(
    observation_vector: ObservationVector, name: str, nee_error_model_name: str
) -> FactorSpec:
    """Source *name*'s noise factor, with its covariance and provenance."""
    if name in config.NEE_WINDOWS:
        error_model = NEE_ERROR_MODELS[nee_error_model_name]
        standard_deviation = config.NEE_DISCREPANCY_STANDARD_DEVIATION[name]
        return noise_factor(
            observation_vector,
            name,
            covariance=_per_site(
                name,
                DiagonalSpec(measurement_variance),
                DenseSpec(
                    partial(
                        nee_discrepancy,
                        variance=standard_deviation**2,
                        memory=error_model.memory(name),
                    )
                ),
            ),
            provenance=(
                "AmeriFlux RANDUNC and JOINTUNC per window; discrepancy of "
                f"standard deviation {standard_deviation} with the memory of "
                f"error model {nee_error_model_name!r}: {error_model.provenance}."
            ),
        )
    return _CONSTRAINT_NOISE_FACTORS[name](observation_vector, name)


def _per_site(name: str, *terms: CovarianceSpec) -> BlockDiagonalSpec:
    """The sum of *terms*, times source *name*'s scale if it has one, one block per site."""
    covariance = SumSpec(*terms)
    if name in config.NOISE_SCALED_SOURCES:
        covariance = ScaledSpec(covariance, scale=noise_scale_name(name))
    return BlockDiagonalSpec(covariance, by="site")


def _leaf_area_index_factor(observation_vector, name) -> FactorSpec:
    """MODIS LAI: floored measurement error plus discrepancy within each summer, per site."""
    return noise_factor(
        observation_vector,
        name,
        covariance=_per_site(
            name,
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
            name,
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

