"""Write the calibration references the equivalence tests read.

Overview
--------
Two calibrations are evaluated at fixed thetas and their results stored: the
layout of theta, the prior's log density, the natural values and the SIPNET
parameter fields. ``tests/test_equivalence.py`` holds the parameter layer to
them, to 1e-12.

Input data
----------
None. The calibrations are built here:

- ``example``: :func:`sipnet_calibration.calibration.example_calibration` at
  sites 1, 27 and 4711, whose PFTs are deciduous, conifer and deciduous;
- ``single_site``: a copy of the single-site Harvard Forest calibration of
  ``experiments/single_site_mcmc_vs_eki`` at site 4977, with its thirteen
  parameters, its four derived parameters (the minimum photosynthesis
  temperature and three base respiration rates referenced to 10 C), its
  allocation simplex, its initial-condition rule and the initial leaf carbon
  and soil moisture as external inputs. Its fixed values are the handful the
  rules read and two more; the two initial-carbon priors are fitted to fixed
  intervals rather than to the site's ensemble.

Output data
-----------
``calibration_reference_<name>.nc`` beside this script, one per calibration:

- ``theta`` on ``(sample, entry)``, with the entry index's levels as the
  string coordinates ``entry_parameter``, ``entry_dim``, ``entry_dim_label``
  and ``entry_unconstrained_name`` on ``entry``;
- ``log_prob`` on ``sample``;
- ``natural:<variable>`` for each variable of the labeled form of theta;
- ``sipnet:<name>`` for each SIPNET parameter field.

Notes
-----
It builds the calibrations with ``parameter_vector``, ``prior`` and
``sipnet_parameter_map`` as they stood at commit 175cd8d, the last before
the ``parameters`` subpackage replaced them, and runs only there:

    git worktree add --detach /tmp/references 175cd8d
    cd /tmp/references && uv sync && uv run python tests/data/write_calibration_references.py

Usage
-----
::

    uv run python tests/data/write_calibration_references.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.calibration import example_calibration
from sipnet_calibration.parameter_vector import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    DerivedParameter,
    Parameter,
    ParameterVector,
)
from sipnet_calibration.prior import (
    Prior,
    PriorTerm,
    log_normal_from_interval,
    logit_normal_from_interval,
    softmax_normal,
)
from sipnet_calibration.sipnet_parameter_map import (
    ComputeInitialConditions,
    ComputePhotosynthesisRates,
    Copy,
    CopySimplex,
    Fixed,
    SIPNETParameterMap,
)

HERE = Path(__file__).resolve().parent

#: The example calibration's sites and their PFTs.
EXAMPLE_SITES = (1, 27, 4711)
EXAMPLE_PFT = ("deciduous", "conifer", "deciduous")

#: The single-site calibration's site and its class.
SINGLE_SITE = 4977
SINGLE_SITE_LABEL = "temperate.deciduous.HPDA"

#: The single-site calibration's constants, as the experiment has them.
PHOTOSYNTHESIS_TEMPERATURE_RANGE = 24.0 - 0.041553
WOOD_RESPIRATION_Q10 = 1.80944
COARSE_ROOT_RESPIRATION_Q10 = 3.20614
DECIDUOUS_BY_SITE_LABEL = {
    "temperate.deciduous.HPDA": True,
    "boreal.coniferous": False,
    "semiarid.grassland_HPDA": False,
}

#: The single-site calibration's fixed values: those its rules read, and two more.
SINGLE_SITE_FIXED = {
    "daily_mean_photosynthesis_fraction": 0.860623,
    "leaf_carbon_fraction": 0.466075,
    "fine_root_fraction": 0.03,
    "coarse_root_fraction": 0.16,
    "leaf_carbon_per_area": 30.7092,
    "wood_respiration_q10": WOOD_RESPIRATION_Q10,
    "vapor_pressure_deficit_exponent": 2.0,
}

#: The single-site calibration's external inputs: initial leaf carbon (kg m-2)
#: and soil moisture (percent of saturation).
SINGLE_SITE_EXTERNAL = {"initial_leaf_carbon": (0.19, "kg m-2"), "initial_soil_moisture_saturation": (42.0, "percent")}

#: The thetas: prior draws, then theta = 0, then draws of N(0, I).
N_PRIOR_DRAWS, N_STANDARD_DRAWS = 6, 2
SEED = 20260930


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    example = example_calibration(site_table(EXAMPLE_SITES), EXAMPLE_PFT)
    write("example", *example, external_inputs=None)
    write("single_site", *single_site_calibration(), external_inputs=single_site_external_inputs())


# ── the steps ─────────────────────────────────────────────────────────────────


def single_site_calibration() -> tuple[ParameterVector, Prior, SIPNETParameterMap]:
    """The single-site calibration's three objects."""
    vector = ParameterVector(
        parameters=[
            Parameter(name="photosynthetic_capacity", support=POSITIVE, units="nmol g-1 s-1"),
            Parameter(name="respiration_share", support=OPEN_UNIT_INTERVAL, units="1"),
            Parameter(name="optimum_photosynthesis_temperature", support=REAL, units="degC"),
            Parameter(name="half_saturation_light", support=POSITIVE, units="mol m-2 d-1"),
            Parameter(name="soil_water_holding_capacity", support=POSITIVE, units="cm"),
            Parameter(name="leaf_on_growth", support=POSITIVE, units="g m-2"),
            Parameter(name="leaf_on_growing_degree_days", support=POSITIVE, units="K d"),
            Parameter(name="allocation", support=SIMPLEX, units="1",
                      natural_names=("leaf", "wood", "fine_root", "coarse_root")),
            Parameter(name="wood_respiration_rate_at_10c", support=POSITIVE, units="yr-1"),
            Parameter(name="soil_respiration_flux_at_10c", support=POSITIVE, units="g m-2 yr-1"),
            Parameter(name="soil_respiration_q10", support=POSITIVE, units="1"),
            Parameter(name="initial_wood_carbon", support=POSITIVE, units="kg m-2"),
            Parameter(name="initial_soil_organic_carbon", support=POSITIVE, units="kg m-2"),
        ],
        derived_parameters=[
            DerivedParameter(
                name="min_photosynthesis_temperature", units="degC", support=REAL,
                derived_from=("optimum_photosynthesis_temperature",),
                compute=lambda dim_index, site_table, optimum_photosynthesis_temperature:
                    optimum_photosynthesis_temperature - PHOTOSYNTHESIS_TEMPERATURE_RANGE,
            ),
            DerivedParameter(
                name="base_wood_respiration_rate", units="yr-1", support=POSITIVE,
                derived_from=("wood_respiration_rate_at_10c",),
                compute=lambda dim_index, site_table, wood_respiration_rate_at_10c:
                    wood_respiration_rate_at_10c / WOOD_RESPIRATION_Q10,
            ),
            DerivedParameter(
                name="base_coarse_root_respiration_rate", units="yr-1", support=POSITIVE,
                derived_from=("wood_respiration_rate_at_10c",),
                compute=lambda dim_index, site_table, wood_respiration_rate_at_10c:
                    wood_respiration_rate_at_10c / COARSE_ROOT_RESPIRATION_Q10,
            ),
            DerivedParameter(
                name="base_soil_respiration_rate", units="yr-1", support=POSITIVE,
                derived_from=("soil_respiration_flux_at_10c", "initial_soil_organic_carbon",
                              "soil_respiration_q10"),
                compute=lambda dim_index, site_table, soil_respiration_flux_at_10c,
                initial_soil_organic_carbon, soil_respiration_q10: soil_respiration_flux_at_10c
                / (1000.0 * initial_soil_organic_carbon * soil_respiration_q10),
            ),
        ],
        site_table=site_table((SINGLE_SITE,)),
        site_labels={"pft": [SINGLE_SITE_LABEL]},
    )

    def term(distribution):
        return PriorTerm(distribution, provenance="Reference fixture.")

    prior = Prior(vector, {
        "photosynthetic_capacity": term(log_normal_from_interval(lower=140.0, upper=450.0)),
        "respiration_share": term(logit_normal_from_interval(lower=0.04, upper=0.20)),
        "optimum_photosynthesis_temperature": term(tfp.distributions.Normal(jnp.float64(22.0), jnp.float64(2.5))),
        "half_saturation_light": term(log_normal_from_interval(lower=4.6, upper=26.3)),
        "soil_water_holding_capacity": term(log_normal_from_interval(lower=15.0, upper=150.0)),
        "leaf_on_growth": term(log_normal_from_interval(lower=50.0, upper=180.0)),
        "leaf_on_growing_degree_days": term(log_normal_from_interval(lower=500.0, upper=1100.0)),
        "allocation": term(softmax_normal(center=jnp.array([0.18, 0.45, 0.065, 0.305]),
                                          logit_sd=jnp.array([0.25, 0.30, 0.30]))),
        "wood_respiration_rate_at_10c": term(log_normal_from_interval(lower=0.006, upper=0.04)),
        "soil_respiration_flux_at_10c": term(log_normal_from_interval(lower=200.0, upper=900.0)),
        "soil_respiration_q10": term(log_normal_from_interval(lower=1.3, upper=3.2)),
        "initial_wood_carbon": term(log_normal_from_interval(lower=5.0, upper=15.0)),
        "initial_soil_organic_carbon": term(log_normal_from_interval(lower=5.0, upper=60.0)),
    })
    copies = (
        "optimum_photosynthesis_temperature", "min_photosynthesis_temperature", "half_saturation_light",
        "soil_water_holding_capacity", "leaf_on_growth", "leaf_on_growing_degree_days",
        "base_wood_respiration_rate", "base_coarse_root_respiration_rate", "base_soil_respiration_rate",
        "soil_respiration_q10",
    )
    sipnet_map = SIPNETParameterMap(
        rules=[
            ComputePhotosynthesisRates(capacity_value_name="photosynthetic_capacity",
                                       respiration_share_value_name="respiration_share"),
            *(Copy(value_name=name, sipnet_parameter_name=name) for name in copies),
            CopySimplex(value_name="allocation",
                        sipnet_parameter_names=("leaf_allocation", "wood_allocation", "fine_root_allocation")),
            ComputeInitialConditions(deciduous=DECIDUOUS_BY_SITE_LABEL),
        ],
        fixed=[
            Fixed(sipnet_parameter_name=name, value=value, provenance="Reference fixture.")
            for name, value in SINGLE_SITE_FIXED.items()
        ],
    )
    return vector, prior, sipnet_map


def single_site_external_inputs() -> xr.Dataset:
    """The initial leaf carbon and soil moisture, on ``site``."""
    return xr.Dataset(
        {
            name: ("site", np.array([value], dtype=np.float64), {"units": units})
            for name, (value, units) in SINGLE_SITE_EXTERNAL.items()
        },
        coords={"site": np.array([SINGLE_SITE], dtype=np.int32)},
    )


def write(
    name: str,
    vector: ParameterVector,
    prior: Prior,
    sipnet_map: SIPNETParameterMap,
    *,
    external_inputs: xr.Dataset | None,
) -> None:
    """One calibration's reference file."""
    theta = thetas(vector, prior)
    natural = vector.dataset(theta)
    fields = sipnet_map.sipnet_parameter_fields(vector, theta, external_inputs=external_inputs)
    index = vector.index
    reference = xr.Dataset(
        {
            "theta": (("sample", "entry"), np.asarray(theta)),
            "log_prob": ("sample", np.asarray(prior.log_prob(theta))),
            **{f"natural:{v}": natural[v] for v in natural.data_vars},
            **{f"sipnet:{v}": fields[v] for v in fields.data_vars},
        },
        coords={
            "entry": list(vector.entry_names),
            **{f"entry_{level}": ("entry", [str(v) for v in index.get_level_values(level)])
               for level in index.names},
        },
    )
    for variable in reference.variables.values():
        variable.attrs = {}
    path = HERE / f"calibration_reference_{name}.nc"
    reference.to_netcdf(path, engine="h5netcdf")
    print(f"wrote {path.name}: D = {vector.dimension}, {len(theta)} thetas, {len(fields.data_vars)} fields")


def thetas(vector: ParameterVector, prior: Prior) -> np.ndarray:
    """Prior draws, theta = 0, and draws of N(0, I)."""
    draws = np.asarray(prior.sample(jax.random.key(SEED), N_PRIOR_DRAWS))
    standard = np.random.default_rng(SEED).standard_normal((N_STANDARD_DRAWS, vector.dimension))
    return np.concatenate([draws, np.zeros((1, vector.dimension)), standard])


# ── helpers ───────────────────────────────────────────────────────────────────


def site_table(site_ids: tuple[int, ...]) -> pd.DataFrame:
    """``site_id``, ``lon`` and ``lat``, as ``conftest.site_table_of`` makes them."""
    ids = np.asarray(site_ids, dtype=np.int32)
    return pd.DataFrame({"site_id": ids, "lon": -100.0 - ids / 100, "lat": 40.0 + ids / 100})


if __name__ == "__main__":
    main()
