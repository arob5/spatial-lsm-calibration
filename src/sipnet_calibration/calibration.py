"""A calibration's objects together: the record of them, and an example.

A calibration over sites is specified by a model of the probability layer,
conditioned on its observations
(:class:`~sipnet_calibration.probability.Posterior`), a
:class:`~sipnet_calibration.site_dims.SiteDims` (the sites) and a
:class:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap` (how
the values at a site become SIPNET parameters), which the model reads
through a :class:`~sipnet_calibration.forward.SIPNETSimulator`. They hold
functions, so they cannot be serialized; :func:`describe_calibration` is the
record an experiment writes beside every run instead.
:func:`example_calibration` is the worked example and test fixture.
"""

from __future__ import annotations


import pandas as pd

from sipnet_calibration.probability import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    SIMPLEX,
    ArraySpec,
    DeterministicSpec,
    FactorSpec,
    Posterior,
    Simulator,
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
#: has them: it depends on a parameter of the posterior, directly or through
#: a deterministic (``calibrated``), on other values only, such as a model's
#: inputs or external inputs (``propagated``), on nothing, being a rule of
#: constants and fixed values (``constant``), or is held (``fixed``).
ROLES: tuple[str, ...] = ("calibrated", "propagated", "constant", "fixed")


def describe_calibration(
    posterior: Posterior, sipnet_parameter_map: SIPNETParameterMap
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The record of a calibration: one table per component and input of
    its model, and one per SIPNET parameter written.

    Parameters
    ----------
    posterior:
        The model conditioned on its observations.
    sipnet_parameter_map:
        The map its SIPNET simulator runs through.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(component_table, sipnet_parameter_table)``.

        ``component_table``, one row per component and input of the model in
        declaration order, the inputs last, but for a simulator's outputs,
        indexed by ``name``: ``role``, ``part`` and ``class``
        (:meth:`Posterior.describe
        <sipnet_calibration.probability.Posterior.describe>`'s);
        ``indexed_by``, ``shape``, ``support``, ``units`` and ``bijector``
        (the component's :class:`~sipnet_calibration.probability.ArraySpec`);
        ``law``, ``given`` and ``provenance`` (the declaring part's, as
        :meth:`ModelSpec.describe
        <sipnet_calibration.probability.ModelSpec.describe>` has them, empty
        for an input); and ``sipnet_parameter_names``, the SIPNET parameters
        depending on it, directly or through deterministics,
        comma-separated.

        ``sipnet_parameter_table``, the map's
        :meth:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap.describe`
        with ``role``, one of :data:`ROLES`.

    Raises
    ------
    TypeError
        If *posterior* is not a ``Posterior``.
    """
    check_posterior_is_a_posterior(posterior)
    spec = posterior.model.spec
    roles = posterior.describe()
    parts = spec.describe()
    dependencies = sipnet_parameter_map.dependencies()
    simulator_outputs = {c.name for p in spec.parts if isinstance(p, Simulator) for c in p.outputs}
    rows = []
    for name, described in roles.iterrows():
        if name in simulator_outputs:
            continue
        component = spec.component_spec(name)
        part = parts.loc[described["part"]] if described["part"] else None
        influenced = _influenced(spec, {name})
        rows.append({
            "name": name,
            "role": described["role"],
            "part": described["part"],
            "class": described["class"],
            "indexed_by": ", ".join(component.indexed_by),
            "shape": component.shape,
            "support": component.support.name,
            "units": component.units,
            "bijector": component.bijector.name,
            "law": "" if part is None else part["law"],
            "given": "" if part is None else part["given"],
            "provenance": "" if part is None else part["provenance"],
            "sipnet_parameter_names": ", ".join(n for n, depends in dependencies.items() if depends & influenced),
        })
    calibrated = _influenced(spec, set(posterior.parameter_names))
    sipnet_parameter_table = sipnet_parameter_map.describe()
    sipnet_parameter_table.insert(1, "role", [
        _role(set_by, dependencies[name], calibrated)
        for name, set_by in sipnet_parameter_table["set_by"].items()
    ])
    return pd.DataFrame(rows).set_index("name"), sipnet_parameter_table


def example_calibration(site_dims: SiteDims) -> tuple[tuple[FactorSpec, ...], SIPNETParameterMap]:
    """A small example calibration over the sites of *site_dims*, which
    labels them by ``"pft"``. **Not the calibration, and not a reviewed
    prior.**

    Its parameters are a shared photosynthetic capacity and respiration
    share, an allocation simplex and a base soil respiration rate per PFT, a
    shared leaf fall fraction, and an initial soil carbon per site. Each
    prior traces to the BETY reanalysis trait posteriors or says it is a
    placeholder. The map leaves SIPNET parameters unset, which a run's base
    parameter set supplies
    (:attr:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap.unset_sipnet_parameter_names`).

    Returns
    -------
    tuple
        ``(prior_factors, sipnet_parameter_map)``: one factor per
        parameter, which a model joins with a simulator and noise factors
        and binds at ``site_dims.coords``.

    Raises
    ------
    KeyError
        If *site_dims* has no site labels ``"pft"``.
    """
    pft = site_dims.coords["pft"]
    fixture = "Example fixture, not a reviewed prior. "
    # Temperate-deciduous BETY medians, with aMaxFrac and cFracLeaf fixed:
    # P = aMax (aMaxFrac + baseFolRespFrac) / cFracLeaf and
    # rho = baseFolRespFrac / (aMaxFrac + baseFolRespFrac).
    a_max_frac, c_frac_leaf, a_max, fol_resp = 0.76, 0.466, 58.0, 0.17
    prior_factors = (
        FactorSpec(
            ArraySpec("photosynthetic_capacity", units="nmol g-1 s-1", support=POSITIVE),
            law=log_normal(median=a_max * (a_max_frac + fol_resp) / c_frac_leaf, geometric_sd=1.75),
            provenance=fixture + "Median from the temperate-deciduous BETY posterior medians aMax "
            "58 nmol g-1 s-1 and baseFolRespFrac 0.17, with aMaxFrac 0.76 and cFracLeaf 0.466. "
            "Geometric sd 1.75 is about twice, on the log scale, the 1.32 the BETY aMax "
            "2.5-97.5% range 28-83 implies.",
        ),
        FactorSpec(
            ArraySpec("respiration_share", units="1", support=OPEN_UNIT_INTERVAL),
            law=logit_normal_from_interval(lower=0.10 / (a_max_frac + 0.10), upper=0.39 / (a_max_frac + 0.39)),
            provenance=fixture + "From the BETY baseFolRespFrac 2.5-97.5% range 0.10-0.39 at "
            "fixed aMaxFrac.",
        ),
        FactorSpec(
            ArraySpec("allocation", units="1", support=SIMPLEX, indexed_by=("pft",),
                      element_axes={"allocation_part": ("leaf", "wood", "fine_root", "coarse_root")}),
            law=iid_over_dim(softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5)),
            provenance=fixture + "Center is the temperate-deciduous BETY allocation posterior "
            "medians; logit sd 0.5 is a placeholder. One value per PFT, all with this prior.",
        ),
        FactorSpec(
            ArraySpec("base_soil_respiration", units="yr-1", support=POSITIVE, indexed_by=("pft",)),
            law=iid_over_dim(log_normal_from_interval(lower=0.004, upper=0.020)),
            provenance=fixture + "BETY som_respiration_rate posterior, 2.5-97.5% quantiles "
            "0.004-0.020 yr-1, for every PFT.",
        ),
        FactorSpec(
            ArraySpec("leaf_fall_fraction", units="1", support=OPEN_UNIT_INTERVAL),
            law=logit_normal(median=0.5, logit_sd=1.7),
            provenance=fixture + "A near-flat logit-normal on (0, 1): a placeholder until a "
            "prior is fitted to BETY fracLeafFall.",
        ),
        FactorSpec(
            ArraySpec("initial_soil_carbon", units="g m-2", support=POSITIVE, indexed_by=("site",)),
            law=iid_over_dim(log_normal(median=30_000.0, geometric_sd=2.0)),
            provenance=fixture + "Median 30 kg C m-2, the center of the 12-75 kg C m-2 the "
            "ISCN-derived initial soil carbon spans at the first test sites; a placeholder "
            "until priors are fitted per site to the initial condition ensemble.",
        ),
    )
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
    return prior_factors, sipnet_map


# ── helpers ───────────────────────────────────────────────────────────────────


def _influenced(spec, names: set[str]) -> set[str]:
    """*names* and every component a deterministic or simulator computes
    from them, directly or through others."""
    out = set(names)
    computing = [p for p in spec.parts if isinstance(p, (DeterministicSpec, Simulator))]
    grew = True
    while grew:
        grew = False
        for part in computing:
            declared = {c.name for c in part.outputs}
            if out & set(part.given) and not declared <= out:
                out |= declared
                grew = True
    return out


def _role(set_by: str, depends_on: frozenset[str], calibrated: set[str]) -> str:
    """A SIPNET parameter's role, one of :data:`ROLES`."""
    if set_by == "fixed":
        return "fixed"
    if depends_on & calibrated:
        return "calibrated"
    return "propagated" if depends_on else "constant"


# ── checks ────────────────────────────────────────────────────────────────────


def check_posterior_is_a_posterior(posterior: object) -> None:
    """The calibration described is a :class:`~sipnet_calibration.probability.Posterior`."""
    if not isinstance(posterior, Posterior):
        raise TypeError(
            f"describe_calibration takes the calibration's Posterior, got {type(posterior).__name__}; "
            "condition the model on its observations first (condition_on)."
        )
