"""A calibration's objects together: the record of them, and an example.

A calibration over sites is specified by a
:class:`~sipnet_calibration.parameters.ParameterVector` (what is
calibrated), its :class:`~sipnet_calibration.parameters.DerivedParameters`
(what is computed from it), a :class:`~sipnet_calibration.parameters.Prior`
(what is believed beforehand), a
:class:`~sipnet_calibration.site_dims.SiteDims` (the sites) and a
:class:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap` (how
the values at a site become SIPNET parameters). They hold functions, so
they cannot be serialized; :func:`describe_calibration` is the record an
experiment writes beside every run instead. :func:`example_calibration` is
the worked example and test fixture.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from sipnet_calibration.parameters import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    SIMPLEX,
    DerivedParameters,
    Parameter,
    ParameterVector,
    Prior,
    PriorTerm,
    iid_over_dim,
    log_normal,
    log_normal_from_interval,
    logit_normal,
    logit_normal_from_interval,
    softmax_normal,
)
from sipnet_calibration.sipnet_parameter_map import (
    Copy,
    CopySimplex,
    Fixed,
    SIPNETParameterMap,
    photosynthesis_rules,
)
from sipnet_calibration.site_dims import SiteDims

__all__ = ["ROLES", "describe_calibration", "example_calibration"]

#: The roles of a SIPNET parameter written, as :func:`describe_calibration`
#: has them: it depends on a parameter or derived parameter (``calibrated``),
#: on external inputs only (``propagated``), on nothing, being a rule of
#: constants and fixed values (``constant``), or is held (``fixed``).
ROLES: tuple[str, ...] = ("calibrated", "propagated", "constant", "fixed")


def describe_calibration(
    parameter_vector: ParameterVector, prior: Prior, sipnet_parameter_map: SIPNETParameterMap
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The record of a calibration: one table per parameter, and one per
    SIPNET parameter written.

    Parameters
    ----------
    parameter_vector, prior, sipnet_parameter_map:
        The calibration's objects; the derived parameters are the prior's
        (``prior.derived_parameters``).

    Returns
    -------
    tuple of pandas.DataFrame
        ``(parameter_table, sipnet_parameter_table)``.

        ``parameter_table``, one row per parameter then per derived parameter,
        indexed by ``parameter``: ``indexed_by``, ``shape``, ``support``,
        ``units`` and ``bijector`` (the vector's; a derived parameter's
        support is the one it declares, if any, and it has no bijector);
        ``parameter_names`` (what a derived parameter is computed from);
        ``term`` (the name of the term covering the parameter, a joint
        term's names joined with ``"+"``), ``prior``, ``given`` and
        ``provenance`` (the prior's, empty for a derived parameter); and
        ``sipnet_parameter_names``, the SIPNET parameters depending on it,
        directly or through derived parameters, comma-separated.

        ``sipnet_parameter_table``, the map's
        :meth:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap.describe`
        with ``role``, one of :data:`ROLES`.

    Raises
    ------
    KeyError
        If the prior lacks a term for one of the vector's parameters.
    """
    derived = prior.derived_parameters
    derived_parameters = () if derived is None else derived.derived_parameters
    dependencies = sipnet_parameter_map.dependencies()
    covering = {name: term.name for term in prior.terms for name in term.parameter_names}
    terms = prior.describe()[["prior", "given", "provenance"]]
    rows = []
    for piece in (*parameter_vector.parameters, *derived_parameters):
        is_parameter = isinstance(piece, Parameter)
        row = {
            "parameter": piece.name,
            "indexed_by": ", ".join(piece.indexed_by),
            "shape": piece.shape,
            "support": "" if piece.support is None else piece.support.name,
            "units": piece.units,
            "bijector": piece.bijector.name if is_parameter else "",
            "parameter_names": "" if is_parameter else ", ".join(piece.parameter_names),
            "term": "", "prior": "", "given": "", "provenance": "",
            "sipnet_parameter_names": ", ".join(
                n for n, depends in dependencies.items() if depends & _influenced(piece.name, derived)
            ),
        }
        if is_parameter:
            check_prior_covers_the_parameter(piece.name, covering)
            row |= {"term": covering[piece.name], **terms.loc[covering[piece.name]].to_dict()}
        rows.append(row)
    calibrated = {*parameter_vector.parameter_names, *(d.name for d in derived_parameters)}
    sipnet_parameter_table = sipnet_parameter_map.describe()
    sipnet_parameter_table.insert(1, "role", [
        _role(set_by, dependencies[name], calibrated)
        for name, set_by in sipnet_parameter_table["set_by"].items()
    ])
    return pd.DataFrame(rows).set_index("parameter"), sipnet_parameter_table


def example_calibration(site_dims: SiteDims) -> tuple[ParameterVector, Prior, SIPNETParameterMap]:
    """A small example calibration over the sites of *site_dims*, which
    labels them by ``"pft"``. **Not the calibration, and not a reviewed
    prior.**

    Its parameters are a shared photosynthetic capacity and respiration
    share, an allocation simplex and a base soil respiration rate per PFT, a
    shared leaf fall fraction, and an initial soil carbon per site. Each prior
    traces to the BETY reanalysis trait posteriors or says it is a
    placeholder. The map leaves SIPNET parameters unset, which a run's base
    parameter set supplies
    (:attr:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap.unset_sipnet_parameter_names`).

    Returns
    -------
    tuple
        ``(parameter_vector, prior, sipnet_parameter_map)``.

    Raises
    ------
    KeyError
        If *site_dims* has no site labels ``"pft"``.
    """
    pft = site_dims.coords["pft"]
    vector = ParameterVector(
        parameters=[
            Parameter(name="photosynthetic_capacity", support=POSITIVE, units="nmol g-1 s-1"),
            Parameter(name="respiration_share", support=OPEN_UNIT_INTERVAL, units="1"),
            Parameter(name="allocation", support=SIMPLEX, units="1", shape=(4,), indexed_by=("pft",),
                      element_labels={"allocation_part": ("leaf", "wood", "fine_root", "coarse_root")}),
            Parameter(name="base_soil_respiration", support=POSITIVE, units="yr-1", indexed_by=("pft",)),
            Parameter(name="leaf_fall_fraction", support=OPEN_UNIT_INTERVAL, units="1"),
            Parameter(name="initial_soil_carbon", support=POSITIVE, units="g m-2", indexed_by=("site",)),
        ],
        coords={"pft": pft, "site": site_dims.coords["site"]},
    )
    fixture = "Example fixture, not a reviewed prior. "
    # Temperate-deciduous BETY medians, with aMaxFrac and cFracLeaf fixed:
    # P = aMax (aMaxFrac + baseFolRespFrac) / cFracLeaf and
    # rho = baseFolRespFrac / (aMaxFrac + baseFolRespFrac).
    a_max_frac, c_frac_leaf, a_max, fol_resp = 0.76, 0.466, 58.0, 0.17
    prior = Prior(vector, [
        PriorTerm(
            parameter_names=("photosynthetic_capacity",),
            distribution=log_normal(median=a_max * (a_max_frac + fol_resp) / c_frac_leaf, geometric_sd=1.75),
            provenance=fixture + "Median from the temperate-deciduous BETY posterior medians aMax "
            "58 nmol g-1 s-1 and baseFolRespFrac 0.17, with aMaxFrac 0.76 and cFracLeaf 0.466. "
            "Geometric sd 1.75 is about twice, on the log scale, the 1.32 the BETY aMax "
            "2.5-97.5% range 28-83 implies.",
        ),
        PriorTerm(
            parameter_names=("respiration_share",),
            distribution=logit_normal_from_interval(lower=0.10 / (a_max_frac + 0.10), upper=0.39 / (a_max_frac + 0.39)),
            provenance=fixture + "From the BETY baseFolRespFrac 2.5-97.5% range 0.10-0.39 at "
            "fixed aMaxFrac.",
        ),
        PriorTerm(
            parameter_names=("allocation",),
            distribution=iid_over_dim(softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5)),
            provenance=fixture + "Center is the temperate-deciduous BETY allocation posterior "
            "medians; logit sd 0.5 is a placeholder. One value per PFT, all with this prior.",
        ),
        PriorTerm(
            parameter_names=("base_soil_respiration",),
            distribution=iid_over_dim(log_normal_from_interval(lower=0.004, upper=0.020)),
            provenance=fixture + "BETY som_respiration_rate posterior, 2.5-97.5% quantiles "
            "0.004-0.020 yr-1, for every PFT.",
        ),
        PriorTerm(
            parameter_names=("leaf_fall_fraction",),
            distribution=logit_normal(median=0.5, logit_sd=1.7),
            provenance=fixture + "A near-flat logit-normal on (0, 1): a placeholder until a "
            "prior is fitted to BETY fracLeafFall.",
        ),
        PriorTerm(
            parameter_names=("initial_soil_carbon",),
            distribution=iid_over_dim(log_normal(median=30_000.0, geometric_sd=2.0)),
            provenance=fixture + "Median 30 kg C m-2, the center of the 12-75 kg C m-2 the "
            "ISCN-derived initial soil carbon spans at the first test sites; a placeholder "
            "until priors are fitted per site to the initial condition ensemble.",
        ),
    ])
    sipnet_map = SIPNETParameterMap(
        rules=[
            *photosynthesis_rules(
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
                sipnet_parameter_name="leaf_carbon_fraction",
                value=pd.Series(dict.fromkeys(pft, c_frac_leaf)).rename_axis("pft").to_xarray(),
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


# ── helpers ───────────────────────────────────────────────────────────────────


def _influenced(name: str, derived: DerivedParameters | None) -> set[str]:
    """*name* and every derived parameter computed from it, directly or
    through others."""
    out = {name}
    if derived is None:
        return out
    for derived_name in derived.names:  # dependency order: inputs come first
        if out & set(derived[derived_name].parameter_names):
            out.add(derived_name)
    return out


def _role(set_by: str, depends_on: frozenset[str], calibrated: set[str]) -> str:
    """A SIPNET parameter's role, one of :data:`ROLES`."""
    if set_by == "fixed":
        return "fixed"
    if depends_on & calibrated:
        return "calibrated"
    return "propagated" if depends_on else "constant"


# ── checks ────────────────────────────────────────────────────────────────────


def check_prior_covers_the_parameter(name: str, covering: Mapping[str, str]) -> None:
    """The prior has a term for each of the vector's parameters: a prior over
    another vector would describe the wrong parameters."""
    if name not in covering:
        raise KeyError(
            f"the prior has no term for parameter {name!r}; describe a prior built on this vector, "
            "prior.parameter_vector."
        )
