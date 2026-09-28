"""A calibration's three objects together: the record of them, and an example.

A calibration is specified by a
:class:`~sipnet_calibration.parameter_vector.ParameterVector` (what is
calibrated), a :class:`~sipnet_calibration.prior.Prior` (what is believed
beforehand) and a
:class:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap` (how a
value reaches SIPNET). They hold functions, so they cannot be serialized;
:func:`describe_calibration` is the table an experiment writes beside every
run instead. :func:`example_calibration` is the worked example and test
fixture.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from sipnet_calibration.parameter_vector import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    SIMPLEX,
    Parameter,
    ParameterVector,
)
from sipnet_calibration.prior import (
    Prior,
    PriorTerm,
    iid_over_dim,
    log_normal,
    log_normal_from_interval,
    logit_normal,
    logit_normal_from_interval,
    softmax_normal,
    term_name,
)
from sipnet_calibration.sipnet_parameter_map import (
    ComputePhotosynthesisRates,
    Copy,
    CopySimplex,
    Fixed,
    SIPNETParameterMap,
)

__all__ = ["describe_calibration", "example_calibration"]


def describe_calibration(
    parameter_vector: ParameterVector, prior: Prior, sipnet_parameter_map: SIPNETParameterMap
) -> pd.DataFrame:
    """One row per parameter, then per derived parameter, indexed by
    ``parameter``: the three objects' descriptions joined.

    Columns: ``dim``, ``support``, ``units``, ``bijector`` and
    ``derived_from`` (the vector); ``term`` (the name of the term covering
    the parameter, a joint term's names joined with ``"+"``), ``prior``,
    ``given`` and ``provenance`` (the prior, empty for a derived parameter,
    which has none); and ``sipnet_parameters`` and ``rules`` (the map: what
    the parameter or derived parameter reaches, comma-separated, and by
    which rules).

    Raises
    ------
    KeyError
        If the prior lacks a term for one of the vector's parameters.
    """
    vector = parameter_vector.describe()[["dim", "support", "units", "bijector", "derived_from"]]
    terms = prior.describe()[["prior", "given", "provenance"]]
    covering = {
        name: term_name(key) for key in prior.terms for name in ((key,) if isinstance(key, str) else key)
    }
    rows = {}
    for name in vector.index:
        if name in parameter_vector.derived_parameter_names:
            rows[name] = {"term": "", "prior": "", "given": "", "provenance": ""}
        else:
            rows[name] = {"term": covering[name], **terms.loc[covering[name]].to_dict()}
    reached = {name: ([], []) for name in vector.index}
    for rule in sipnet_parameter_map.rules:
        for name in rule.values_read:
            if name in reached:
                reached[name][0].extend(rule.sipnet_parameter_names_written)
                reached[name][1].append(type(rule).__name__)
    links = pd.DataFrame(
        {
            "sipnet_parameters": {n: ", ".join(r[0]) for n, r in reached.items()},
            "rules": {n: ", ".join(dict.fromkeys(r[1])) for n, r in reached.items()},
        }
    )
    return vector.join(pd.DataFrame.from_dict(rows, orient="index")).join(links).rename_axis("parameter")


def example_calibration(
    site_table: pd.DataFrame, pft: Any
) -> tuple[ParameterVector, Prior, SIPNETParameterMap]:
    """A small example calibration. **Not the calibration, and not a reviewed
    prior.**

    Its parameters are a shared photosynthetic capacity and respiration
    share, an allocation simplex and a base soil respiration rate per PFT, a
    shared leaf fall fraction, and an initial soil carbon per site. Each prior
    traces to the BETY reanalysis trait posteriors or says it is a
    placeholder. The map leaves SIPNET parameters unset, which a run's base
    parameter set supplies
    (:attr:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap.unset_sipnet_parameter_names`).

    Parameters
    ----------
    site_table:
        The sites, as :class:`~sipnet_calibration.parameter_vector.ParameterVector`
        takes them.
    pft:
        The site labels, named ``"pft"`` in the vector.

    Returns
    -------
    tuple
        ``(parameter_vector, prior, sipnet_parameter_map)``.
    """
    vector = ParameterVector(
        parameters=[
            Parameter(name="photosynthetic_capacity", support=POSITIVE, units="nmol g-1 s-1"),
            Parameter(name="respiration_share", support=OPEN_UNIT_INTERVAL, units="1"),
            Parameter(name="allocation", support=SIMPLEX, units="1", dim="pft",
                      natural_names=("leaf", "wood", "fine_root", "coarse_root")),
            Parameter(name="base_soil_respiration", support=POSITIVE, units="yr-1", dim="pft"),
            Parameter(name="leaf_fall_fraction", support=OPEN_UNIT_INTERVAL, units="1"),
            Parameter(name="initial_soil_carbon", support=POSITIVE, units="g m-2", dim="site"),
        ],
        site_table=site_table,
        site_labels={"pft": pft},
    )
    fixture = "Example fixture, not a reviewed prior. "
    # Temperate-deciduous BETY medians, with aMaxFrac and cFracLeaf fixed:
    # P = aMax (aMaxFrac + baseFolRespFrac) / cFracLeaf and
    # rho = baseFolRespFrac / (aMaxFrac + baseFolRespFrac).
    a_max_frac, c_frac_leaf, a_max, fol_resp = 0.76, 0.466, 58.0, 0.17
    prior = Prior(vector, {
        "photosynthetic_capacity": PriorTerm(
            log_normal(median=a_max * (a_max_frac + fol_resp) / c_frac_leaf, geometric_sd=1.75),
            provenance=fixture + "Median from the temperate-deciduous BETY posterior medians aMax "
            "58 nmol g-1 s-1 and baseFolRespFrac 0.17, with aMaxFrac 0.76 and cFracLeaf 0.466. "
            "Geometric sd 1.75 is about twice, on the log scale, the 1.32 the BETY aMax "
            "2.5-97.5% range 28-83 implies.",
        ),
        "respiration_share": PriorTerm(
            logit_normal_from_interval(lower=0.10 / (a_max_frac + 0.10), upper=0.39 / (a_max_frac + 0.39)),
            provenance=fixture + "From the BETY baseFolRespFrac 2.5-97.5% range 0.10-0.39 at "
            "fixed aMaxFrac.",
        ),
        "allocation": PriorTerm(
            iid_over_dim(softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5)),
            provenance=fixture + "Center is the temperate-deciduous BETY allocation posterior "
            "medians; logit sd 0.5 is a placeholder. One value per PFT, all with this prior.",
        ),
        "base_soil_respiration": PriorTerm(
            iid_over_dim(log_normal_from_interval(lower=0.004, upper=0.020)),
            provenance=fixture + "BETY som_respiration_rate posterior, 2.5-97.5% quantiles "
            "0.004-0.020 yr-1, for every PFT.",
        ),
        "leaf_fall_fraction": PriorTerm(
            logit_normal(median=0.5, logit_sd=1.7),
            provenance=fixture + "A near-flat logit-normal on (0, 1): a placeholder until a "
            "prior is fitted to BETY fracLeafFall.",
        ),
        "initial_soil_carbon": PriorTerm(
            iid_over_dim(log_normal(median=30_000.0, geometric_sd=2.0)),
            provenance=fixture + "Median 30 kg C m-2, the center of the 12-75 kg C m-2 the "
            "ISCN-derived initial soil carbon spans at the first test sites; a placeholder "
            "until priors are fitted per site to the initial condition ensemble.",
        ),
    })
    sipnet_map = SIPNETParameterMap(
        rules=[
            ComputePhotosynthesisRates(
                capacity_value_name="photosynthetic_capacity",
                respiration_share_value_name="respiration_share",
            ),
            CopySimplex(value_name="allocation", sipnet_parameter_names=(
                "leaf_allocation", "wood_allocation", "fine_root_allocation")),
            Copy(value_name="base_soil_respiration", sipnet_parameter_name="base_soil_respiration_rate"),
            Copy(value_name="leaf_fall_fraction", sipnet_parameter_name="leaf_off_fall_fraction"),
            Copy(value_name="initial_soil_carbon", sipnet_parameter_name="soil_carbon"),
        ],
        fixed=[
            Fixed(
                sipnet_parameter_name="daily_mean_photosynthesis_fraction", value=a_max_frac,
                provenance=fixture + "0.76, within the BETY per-PFT posterior median range "
                "0.75-0.86; fixed because aMaxFrac spans one of the two exactly degenerate "
                "photosynthesis directions (sipnet.c:614, 617, 633).",
            ),
            Fixed(
                sipnet_parameter_name="leaf_carbon_fraction", dim="pft",
                value=dict.fromkeys(vector.dim_index("pft"), c_frac_leaf),
                provenance=fixture + "The BETY leafC posterior median for temperate deciduous, "
                "0.466, applied to every PFT here; fixed for the same reason as aMaxFrac.",
            ),
            Fixed(
                sipnet_parameter_name="vapor_pressure_deficit_exponent", value=2.0,
                provenance="Braswell et al. (2005) fix the exponent at 2.",
            ),
        ],
    )
    return vector, prior, sipnet_map

