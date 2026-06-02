"""Shared SIPNET parameter helpers for all experiments.

Experiments define their own priors directly using ProbPipe distributions
(see experiments/<name>/config.py). This module contains only helpers that
are truly shared across experiments.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysipnet import SIPNETParametersV1


def default_base_params() -> SIPNETParametersV1:
    """Standard temperate-forest SIPNET baseline parameter set.

    Returns a fully-specified SIPNETParametersV1 suitable for use as
    base_params in a SIPNETModel. Individual experiments override specific
    parameters by passing them as keyword arguments to the model callable.
    """
    from pysipnet import SIPNETParametersV1
    from pysipnet.parameters import (
        InitialConditions,
        PhotosynthesisParams,
        PhenologyParams,
        RespirationParams,
        AllocationParams,
        WaterParams,
        LeafPhysiologyParams,
    )

    return SIPNETParametersV1(
        initial_conditions=InitialConditions(
            plant_wood=30000.0,
            lai=0.0,
            soil=10000.0,
            soil_water_frac=0.5,
            snow=1.0,
            fine_root_frac=0.05,
            coarse_root_frac=0.15,
        ),
        photosynthesis=PhotosynthesisParams(
            a_max=120.0,
            a_max_frac=0.76,
            base_fol_resp_frac=0.1,
            psn_t_min=2.0,
            psn_t_opt=20.0,
            d_vpd_slope=0.05,
            d_vpd_exp=1.0,
            half_sat_par=200.0,
            attenuation=0.5,
        ),
        phenology=PhenologyParams(
            leaf_off_day=270.0,
            gdd_leaf_on=100.0,
            leaf_growth=50.0,
            frac_leaf_fall=0.95,
            leaf_allocation=0.25,
            leaf_turnover_rate=1.0,
        ),
        respiration=RespirationParams(
            base_veg_resp=0.5,
            veg_resp_q10=2.0,
            growth_resp_frac=0.0,
            frozen_soil_fol_r_eff=0.5,
            frozen_soil_threshold=-1.0,
            base_fine_root_resp=0.5,
            base_coarse_root_resp=0.1,
            fine_root_q10=2.0,
            coarse_root_q10=2.0,
            base_soil_resp=0.3,
            soil_resp_q10=2.2,
            soil_resp_moist_effect=1.5,
        ),
        allocation=AllocationParams(
            fine_root_allocation=0.35,
            wood_allocation=0.30,
            fine_root_turnover_rate=1.0,
            coarse_root_turnover_rate=0.1,
            wood_turnover_rate=0.02,
        ),
        water=WaterParams(
            water_remove_frac=0.1,
            frozen_soil_eff=0.1,
            wue_const=10.0,
            soil_whc=12.0,
            litter_whc=5.0,
            immed_evap_frac=0.1,
            fast_flow_frac=0.1,
            snow_melt=0.15,
            rd_const=100.0,
            r_soil_const1=3.0,
            r_soil_const2=2.0,
        ),
        leaf=LeafPhysiologyParams(
            leaf_c_sp_wt=32.0,
            c_frac_leaf=0.45,
        ),
    )
