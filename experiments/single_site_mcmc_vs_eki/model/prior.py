"""The calibration's parameterization and prior: the starting point step 4 found.

The parameters' prior factors (what is believed beforehand), the
:class:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap` (how a
value reaches SIPNET), the :class:`~sipnet_calibration.site_dims.SiteDims` of
the one site, and the initial states the map reads that are not calibrated,
declared as inputs. ``model/calibration.py`` joins them with the forward map
and the noise model. ``MODEL.md``, "Parameterization and prior", says how
they were found; every prior factor and fixed value carries its provenance.

**Calibrated** (13 parameters, D = 15 entries of theta), each with its own
factor, so the prior is their product; every law is a Gaussian in theta
pushed through the parameter's bijector:

- photosynthesis: the capacity ``P`` and respiration share ``rho`` of
  :func:`~sipnet_calibration.sipnet_parameter_map.photosynthesis_rules`,
  which replace the four SIPNET parameters that enter only through them
  (``sipnet.c:614, 617, 633``); the optimum temperature, with the minimum
  written at a fixed range; the half-saturation light;
- water: the soil water holding capacity, the one water-limitation direction;
- phenology: the leaf growth at leaf-on and its growing degree-days;
- allocation: a four-part simplex (leaf, wood, fine root, coarse root, the
  last SIPNET's remainder), 3 entries of theta;
- respiration: the wood respiration rate at 10 C (also the coarse roots'),
  the soil respiration flux at 10 C, ``F10 = k10 * SOC0``, and the soil Q10;
- the initial state: initial wood and soil organic carbon, with priors fitted
  to the site's initial-condition ensemble.

Four SIPNET parameters are written by ``Compute`` rules from a value
referenced differently: the minimum photosynthesis temperature, and the wood,
coarse-root and soil base respiration rates.

**Fixed**: every other SIPNET parameter, at the values and with the
justifications in ``fixed_sipnet_parameters.csv``.

**Inputs**: the initial leaf carbon and soil moisture, the site's
initial-condition medians (``sipnet.initial_state``), on ``site``.
"""

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp
import xarray as xr

from sipnet_calibration.probability import (
    NON_NEGATIVE,
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    ArraySpec,
    FactorSpec,
    Interval,
    Posterior,
    condition_on,
    joint,
    log_normal_from_interval,
    log_normal_from_samples,
    logit_normal_from_interval,
    normal,
    softmax_normal,
)
from sipnet_calibration.sipnet_parameter_map import (
    Compute,
    Copy,
    CopySimplex,
    Fixed,
    SIPNETParameterMap,
    ValueRequirement,
    initial_condition_rules,
    photosynthesis_rules,
)
from sipnet_calibration.site_dims import SiteDims
from sipnet_calibration.site_labels import load_site_labels

from . import inputs, sipnet

__all__ = [
    "ALLOCATION_PART",
    "ALLOCATION_PARTS",
    "EXTERNAL_STATE_NAMES",
    "FIXED_SIPNET_PARAMETERS_FILE",
    "SITE_LABELS_NAME",
    "external_inputs",
    "fixed_sipnet_parameters",
    "input_specs",
    "natural_table",
    "prior_alone",
    "prior_center",
    "prior_factors",
    "sipnet_parameter_map",
    "site_dims",
]

#: The site-labels data source the site's deciduousness is read from.
SITE_LABELS_NAME = "reanalysis_3pft"

#: Whether each class of :data:`SITE_LABELS_NAME` is deciduous, which decides
#: whether a run starts with leaves.
DECIDUOUS_BY_SITE_LABEL = {
    "temperate.deciduous.HPDA": True,
    "boreal.coniferous": False,
    "semiarid.grassland_HPDA": False,
}

#: The allocation simplex's element axis, and its labels in order; the last
#: is SIPNET's remainder.
ALLOCATION_PART = "allocation_part"
ALLOCATION_PARTS = ("leaf", "wood", "fine_root", "coarse_root")

#: The fixed SIPNET parameters' values and justifications.
FIXED_SIPNET_PARAMETERS_FILE = (
    Path(__file__).resolve().parent / "fixed_sipnet_parameters.csv"
)

#: The initial states the calibration takes from the site's ensemble medians;
#: initial wood and soil carbon are calibrated instead.
EXTERNAL_STATE_NAMES = ("initial_leaf_carbon", "initial_soil_moisture_saturation")

#: The support of each input, the domain its rule reads it on
#: (``initial_condition_rules``): leaf carbon is a stock, soil moisture a
#: percent of saturation.
EXTERNAL_STATE_SUPPORTS = {
    "initial_leaf_carbon": NON_NEGATIVE,
    "initial_soil_moisture_saturation": Interval(
        0.0, 100.0, low_closed=True, high_closed=True
    ),
}

#: psnTOpt - psnTMin, deg C, held while the optimum is calibrated: the base
#: set's optimum 24 (PEcAn's template; BETY's 43 is implausible) less its
#: minimum 0.04 (the BETY median).
PHOTOSYNTHESIS_TEMPERATURE_RANGE = 24.0 - 0.041553

#: The Q10s the wood and coarse-root base rates are referenced to 10 C with:
#: the BETY medians, fixed (fixed_sipnet_parameters.csv).
WOOD_RESPIRATION_Q10 = 1.80944
COARSE_ROOT_RESPIRATION_Q10 = 3.20614

#: The provenance of each prior factor.
PROVENANCE = {
    "photosynthetic_capacity": (
        "Reasoned. P = aMax (aMaxFrac + baseFolRespFrac) / cFracLeaf. Median 251 nmol "
        "g-1 s-1 (aMax 123, about 8 umol m-2 s-1 at the BETY SLA). The 2.5% end, 140, "
        "is near the BETY Amax x SLA 97.5% (P 170) and below PEcAn's template (228); "
        "the 97.5% end, 450, is near a top-canopy oak leaf's Amax of about 13 umol m-2 "
        "s-1 (approximate literature). BETY's own Amax converts to P of about 58, which "
        "gives GPP near 690 g C m-2 yr-1, far below Harvard's."
    ),
    "respiration_share": (
        "Reasoned from BETY: its leaf-area dark respiration, 0.66 umol m-2 s-1, over a "
        "leaf gross Amax of 6-15 gives 0.04-0.10; the upper end reaches the 0.16 BETY's "
        "baseFolRespFrac implies."
    ),
    "optimum_photosynthesis_temperature": (
        "Reasoned: a temperate deciduous canopy's optimum is about 20-27 C (approximate "
        "literature); BETY's median 43 C is rejected. The minimum follows at the base "
        "set's fixed range of 23.96 C."
    ),
    "half_saturation_light": "BETY temperate deciduous 2.5-97.5% (4.6-26.3 mol m-2 d-1).",
    "soil_water_holding_capacity": (
        "Reasoned placeholder: from the plant-available water of a 1 m stony sandy-loam "
        "rooting zone (about 15 cm, inferred) to past PEcAn's porosity x 2 m (88 cm). "
        "Above about 88 cm the model is water-unlimited at this site, a flat direction."
    ),
    "leaf_on_growth": (
        "BETY's 2.5% (50 g C m-2) kept; the upper end cut from BETY's 249 to 180 (LAI "
        "5.9 at 30.7 g C m-2 of leaf) so the leaf-on pulse does not exceed a full "
        "Harvard canopy (reasoned)."
    ),
    "leaf_on_growing_degree_days": (
        "From the drivers: SIPNET's growing degree-days (base 0 C, sipnet.c:230-235) "
        "reach 500 on about day 123, 700 on 139 and 800 on 145; Harvard's canopy "
        "develops in mid to late May (approximate phenology literature). Median 742."
    ),
    "allocation": (
        "A simplex (leaf, wood, fine root, coarse root = SIPNET's remainder, "
        "sipnet.c:1113-1115), which rules out SIPNET's allocation-sum exit "
        "(sipnet.c:1111-1123). Leaf 0.18 and fine root 0.065 are the BETY medians; wood "
        "is raised from 0.40 to 0.45 so coarse-root allocation is about 0.7 of wood's, "
        "not 0.9 (reasoned; root:shoot about 0.2-0.3). Log-ratio sds 0.25, 0.30, 0.30 "
        "approximate the BETY spreads."
    ),
    "wood_respiration_rate_at_10c": (
        "Reasoned: SIPNET's base rate referenced to 10 C, baseVegResp = r10 / "
        "vegRespQ10 (sipnet.c:1066-1067), so it is decorrelated from the fixed Q10. "
        "Median 0.0148 yr-1, about 130 g C m-2 yr-1 on 8.6 kg C m-2 of wood; BETY's "
        "stem respiration draws are judged unconstrained. It also sets the coarse "
        "roots' rate (coarse roots are wood)."
    ),
    "soil_respiration_flux_at_10c": (
        "A reparameterization of the rate-times-pool ridge: F10 = k10 SOC0, g C m-2 "
        "yr-1 at 10 C and unit moisture, and baseSoilResp = F10 / (1000 SOC0 Q10). "
        "Reasoned: Harvard's heterotrophic respiration is about 300-700 g C m-2 yr-1 "
        "(approximate literature)."
    ),
    "soil_respiration_q10": (
        "Reasoned: an ecosystem-level Q10 converges near 1.4 (Mahecha et al. 2010, "
        "Science 329:838), soil-chamber apparent Q10s are often 2-4; interval 1.3-3.2, "
        "median 2.08."
    ),
    "initial_wood_carbon": (
        "The site's initial-condition ensemble (100 members, Spawn and Gibbs 2020 "
        "aboveground carbon less leaf carbon), a log-normal fitted to its members."
    ),
    "initial_soil_organic_carbon": (
        "The site's initial-condition ensemble (ISCN profiles of the site's ecoregion), "
        "a log-normal fitted to its members; its upper tail, to about 100 kg m-2, is "
        "kept, since the SoilGrids observation constrains it."
    ),
}


def prior_factors() -> list[FactorSpec]:
    """The 13 parameters' prior factors, one each, in theta's order."""
    members = {
        name: np.asarray(field).ravel().astype(float)
        for name, field in inputs.initial_condition_fields().items()
        if name in ("initial_wood_carbon", "initial_soil_organic_carbon")
    }

    def prior_factor(name, units, law, *, support=POSITIVE, **arguments):
        return FactorSpec(
            ArraySpec(name, units=units, support=support, **arguments),
            law=law,
            provenance=PROVENANCE[name],
        )

    return [
        prior_factor(
            "photosynthetic_capacity",
            "nmol g-1 s-1",
            log_normal_from_interval(lower=140.0, upper=450.0),
        ),
        prior_factor(
            "respiration_share",
            "1",
            logit_normal_from_interval(lower=0.04, upper=0.20),
            support=OPEN_UNIT_INTERVAL,
        ),
        prior_factor(
            "optimum_photosynthesis_temperature",
            "degC",
            normal(mean=22.0, standard_deviation=2.5),
            support=REAL,
        ),
        prior_factor(
            "half_saturation_light",
            "mol m-2 d-1",
            log_normal_from_interval(lower=4.6, upper=26.3),
        ),
        prior_factor(
            "soil_water_holding_capacity",
            "cm",
            log_normal_from_interval(lower=15.0, upper=150.0),
        ),
        prior_factor(
            "leaf_on_growth", "g m-2", log_normal_from_interval(lower=50.0, upper=180.0)
        ),
        prior_factor(
            "leaf_on_growing_degree_days",
            "K d",
            log_normal_from_interval(lower=500.0, upper=1100.0),
        ),
        prior_factor(
            "allocation",
            "1",
            softmax_normal(
                center=jnp.array([0.18, 0.45, 0.065, 0.305]),
                logit_sd=jnp.array([0.25, 0.30, 0.30]),
            ),
            support=SIMPLEX,
            element_axes={ALLOCATION_PART: ALLOCATION_PARTS},
        ),
        prior_factor(
            "wood_respiration_rate_at_10c",
            "yr-1",
            log_normal_from_interval(lower=0.006, upper=0.04),
        ),
        prior_factor(
            "soil_respiration_flux_at_10c",
            "g m-2 yr-1",
            log_normal_from_interval(lower=200.0, upper=900.0),
        ),
        prior_factor(
            "soil_respiration_q10", "1", log_normal_from_interval(lower=1.3, upper=3.2)
        ),
        prior_factor(
            "initial_wood_carbon",
            "kg m-2",
            log_normal_from_samples(members["initial_wood_carbon"]),
        ),
        prior_factor(
            "initial_soil_organic_carbon",
            "kg m-2",
            log_normal_from_samples(members["initial_soil_organic_carbon"]),
        ),
    ]


def prior_alone() -> Posterior:
    """The prior factors bound and conditioned on nothing: the prior over theta.

    Its theta is the calibration's, since the parameters are declared in the
    same order; it needs no data beyond the initial-condition ensemble.
    """
    return condition_on(joint(*prior_factors()).bind(coords={}), {})


def prior_center(posterior: Posterior) -> np.ndarray:
    """Theta at the prior's center, ``(D,)``.

    Every prior factor's law is a Gaussian in theta pushed through its
    parameter's bijector (or the Gaussian itself, for the optimum
    temperature), and the center is the Gaussians' means: each parameter at
    its prior median, the allocation at its softmax-normal's center.
    """
    means = []
    for name in posterior.parameter_names:
        law = posterior.model.law(name, given={})
        if isinstance(law, tfp.distributions.TransformedDistribution):
            law = law.distribution
        means.append(np.ravel(law.mean()))
    return np.concatenate(means)


def site_dims() -> SiteDims:
    """The configured site, with its class under :data:`SITE_LABELS_NAME` as ``pft``."""
    return SiteDims(
        site_table=inputs.site_table(),
        site_labels={"pft": load_site_labels(SITE_LABELS_NAME)},
    )


def input_specs() -> list[ArraySpec]:
    """The initial states the map reads and nothing calibrates, on ``site``.

    Each in the units of the initial-condition file (:func:`external_inputs`).
    """
    values = external_inputs()
    return [
        ArraySpec(
            name,
            units=values[name].attrs["units"],
            support=EXTERNAL_STATE_SUPPORTS[name],
            indexed_by=("site",),
        )
        for name in EXTERNAL_STATE_NAMES
    ]


def external_inputs() -> dict[str, xr.DataArray]:
    """The values of :func:`input_specs`, by name: the site's initial-condition medians."""
    initial_state = sipnet.initial_state()
    return {name: initial_state[name] for name in EXTERNAL_STATE_NAMES}


def natural_table(posterior: Posterior, theta) -> pd.DataFrame:
    """Theta's natural values, one row per row of *theta*, ``(J, D)``.

    One column per parameter, and per element of a parameter with element
    axes, named ``<parameter>.<element label>`` (``allocation.leaf``). The
    rows are numbered; a caller labels them.
    """
    labeled = posterior.to_labeled(np.asarray(theta))
    columns = {}
    for name in posterior.parameter_names:
        values = labeled[name].to_pandas()
        if isinstance(values, pd.Series):
            columns[name] = values.to_numpy()
            continue
        for label in values.columns:
            columns[f"{name}.{label}"] = values[label].to_numpy()
    return pd.DataFrame(columns)


def fixed_sipnet_parameters() -> pd.DataFrame:
    """The fixed SIPNET parameters, indexed by pySIPNET flat name."""
    return pd.read_csv(FIXED_SIPNET_PARAMETERS_FILE).set_index("sipnet_parameter_name")


# ── the SIPNET parameter map ──


def sipnet_parameter_map() -> SIPNETParameterMap:
    """The rules for the calibrated values, and every other SIPNET parameter fixed."""
    copies = (
        "optimum_photosynthesis_temperature",
        "half_saturation_light",
        "soil_water_holding_capacity",
        "leaf_on_growth",
        "leaf_on_growing_degree_days",
        "soil_respiration_q10",
    )
    rules = [
        *photosynthesis_rules(
            capacity_value_name="photosynthetic_capacity",
            respiration_share_value_name="respiration_share",
        ),
        *(Copy(value_name=name, sipnet_parameter_name=name) for name in copies),
        *_reference_rules(),
        CopySimplex(
            value_name="allocation",
            sipnet_parameter_names=(
                "leaf_allocation",
                "wood_allocation",
                "fine_root_allocation",
            ),
        ),
        *initial_condition_rules(deciduous=DECIDUOUS_BY_SITE_LABEL),
    ]
    with FIXED_SIPNET_PARAMETERS_FILE.open() as file:
        fixed = [
            Fixed(
                sipnet_parameter_name=row["sipnet_parameter_name"],
                value=float(row["value"]),
                provenance=row["justification"],
            )
            for row in csv.DictReader(file)
        ]
    return SIPNETParameterMap(rules=rules, fixed=fixed)


def _reference_rules() -> list[Compute]:
    """The four SIPNET parameters written from a value referenced differently."""
    return [
        Compute(
            sipnet_parameter_name="min_photosynthesis_temperature",
            values_read={
                "optimum_photosynthesis_temperature": ValueRequirement("degC", REAL)
            },
            function=_minimum_photosynthesis_temperature,
            provenance=(
                "psnTMin = psnTOpt less the base set's fixed range of "
                f"{PHOTOSYNTHESIS_TEMPERATURE_RANGE} C"
            ),
        ),
        Compute(
            sipnet_parameter_name="base_wood_respiration_rate",
            values_read={
                "wood_respiration_rate_at_10c": ValueRequirement("yr-1", POSITIVE)
            },
            function=_base_wood_respiration_rate,
            provenance="baseVegResp = r10 / vegRespQ10, sipnet.c:1066-1067",
        ),
        Compute(
            sipnet_parameter_name="base_coarse_root_respiration_rate",
            values_read={
                "wood_respiration_rate_at_10c": ValueRequirement("yr-1", POSITIVE)
            },
            function=_base_coarse_root_respiration_rate,
            provenance=(
                "baseCoarseRootResp = r10 / coarseRootQ10: coarse roots respire "
                "at the wood rate at 10 C"
            ),
        ),
        Compute(
            sipnet_parameter_name="base_soil_respiration_rate",
            values_read={
                "soil_respiration_flux_at_10c": ValueRequirement(
                    "g m-2 yr-1", POSITIVE
                ),
                "initial_soil_organic_carbon": ValueRequirement("kg m-2", POSITIVE),
                "soil_respiration_q10": ValueRequirement("1", POSITIVE),
            },
            function=_base_soil_respiration_rate,
            provenance=(
                "F10 = k10 SOC0 referenced to 10 C and unit moisture: "
                "baseSoilResp = F10 / (1000 SOC0 Q10)"
            ),
        ),
    ]


# ── the Compute rules' formulas, JAX-traceable ──


def _minimum_photosynthesis_temperature(optimum_photosynthesis_temperature):
    return optimum_photosynthesis_temperature - PHOTOSYNTHESIS_TEMPERATURE_RANGE


def _base_wood_respiration_rate(wood_respiration_rate_at_10c):
    return wood_respiration_rate_at_10c / WOOD_RESPIRATION_Q10


def _base_coarse_root_respiration_rate(wood_respiration_rate_at_10c):
    return wood_respiration_rate_at_10c / COARSE_ROOT_RESPIRATION_Q10


def _base_soil_respiration_rate(
    soil_respiration_flux_at_10c, initial_soil_organic_carbon, soil_respiration_q10
):
    # F10 = k10 SOC0 with SOC0 in g m-2, and k10 = baseSoilResp Q10.
    return soil_respiration_flux_at_10c / (
        1000.0 * initial_soil_organic_carbon * soil_respiration_q10
    )
