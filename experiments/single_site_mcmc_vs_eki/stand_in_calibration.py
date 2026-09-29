"""A stand-in calibration for step 3: enough to run the machinery, not the calibration.

**Not the parameterization and not a reviewed prior.** Step 4 replaces this
module with the calibration proper. It exists so that step 3 can run SIPNET
once and as an ensemble through the same three objects the calibration will
use: a :class:`~sipnet_calibration.parameter_vector.ParameterVector` (what
is calibrated), a :class:`~sipnet_calibration.prior.Prior` (what is believed
beforehand) and a
:class:`~sipnet_calibration.sipnet_parameter_map.SIPNETParameterMap` (how a
value reaches SIPNET).

- **Parameters**: three, each one value for the site. The photosynthetic
  capacity and respiration share of
  :class:`~sipnet_calibration.sipnet_parameter_map.ComputePhotosynthesisRates`,
  and the base soil respiration rate. Their priors are
  :func:`sipnet_calibration.calibration.example_calibration`'s, which trace to
  the temperate-deciduous BETY posteriors.
- **The initial state**: the site's initial conditions, as external inputs
  (``runs.initial_state``), through
  :class:`~sipnet_calibration.sipnet_parameter_map.ComputeInitialConditions`;
  the site is deciduous by its reanalysis site label, so it starts leafless.
- **Held fixed**: the two photosynthesis parameters the capacity and share
  leave degenerate, the VPD exponent, and the five values the initial state
  and a deciduous canopy need. Every other SIPNET parameter is the base
  parameter set's (``config.BASE_SIPNET_PARAMETER_FILE``).
"""

import pandas as pd

from sipnet_calibration.parameter_vector import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    Parameter,
    ParameterVector,
)
from sipnet_calibration.prior import (
    Prior,
    PriorTerm,
    log_normal,
    log_normal_from_interval,
    logit_normal_from_interval,
)
from sipnet_calibration.sipnet_parameter_map import (
    ComputeInitialConditions,
    ComputePhotosynthesisRates,
    Copy,
    Fixed,
    SIPNETParameterMap,
)

__all__ = ["DECIDUOUS_BY_SITE_LABEL", "SITE_LABELS_NAME", "stand_in_calibration"]

#: The site-labels data source the site's deciduousness is read from.
SITE_LABELS_NAME = "reanalysis_3pft"

#: Whether each class of :data:`SITE_LABELS_NAME` is deciduous, which decides
#: whether a run starts with leaves.
DECIDUOUS_BY_SITE_LABEL = {
    "temperate.deciduous.HPDA": True,
    "boreal.coniferous": False,
    "semiarid.grassland_HPDA": False,
}

#: The provenance every stand-in value starts with.
_STAND_IN = "Stand-in for step 3, not a reviewed value. "

#: The temperate-deciduous BETY posterior medians the photosynthesis prior is
#: built on, as in example_calibration: aMaxFrac, cFracLeaf, aMax
#: (nmol g-1 s-1) and baseFolRespFrac.
_A_MAX_FRACTION, _LEAF_CARBON_FRACTION, _A_MAX, _FOLIAR_RESPIRATION = (
    0.76,
    0.466,
    58.0,
    0.17,
)


def stand_in_calibration(
    site_table: pd.DataFrame, site_labels: pd.Series
) -> tuple[ParameterVector, Prior, SIPNETParameterMap]:
    """The stand-in parameter vector, prior and SIPNET parameter map.

    Parameters
    ----------
    site_table:
        The one-row site table of the site (``inputs.site_table()``).
    site_labels:
        The site's class under :data:`SITE_LABELS_NAME`, one per site of
        *site_table*, in its order.

    Returns
    -------
    tuple
        ``(parameter_vector, prior, sipnet_parameter_map)``.
    """
    vector = ParameterVector(
        parameters=[
            Parameter(
                name="photosynthetic_capacity", support=POSITIVE, units="nmol g-1 s-1"
            ),
            Parameter(name="respiration_share", support=OPEN_UNIT_INTERVAL, units="1"),
            Parameter(name="base_soil_respiration", support=POSITIVE, units="yr-1"),
        ],
        site_table=site_table,
        site_labels={"pft": list(site_labels)},
    )
    return vector, _stand_in_prior(vector), _stand_in_map()


def _stand_in_prior(vector: ParameterVector) -> Prior:
    """example_calibration's priors for the three parameters."""
    capacity = _A_MAX * (_A_MAX_FRACTION + _FOLIAR_RESPIRATION) / _LEAF_CARBON_FRACTION
    return Prior(
        vector,
        {
            "photosynthetic_capacity": PriorTerm(
                log_normal(median=capacity, geometric_sd=1.75),
                provenance=_STAND_IN
                + "example_calibration's: the median from the temperate-deciduous "
                "BETY medians aMax 58 nmol g-1 s-1 and baseFolRespFrac 0.17, with "
                "aMaxFrac 0.76 and cFracLeaf 0.466; geometric sd 1.75.",
            ),
            "respiration_share": PriorTerm(
                logit_normal_from_interval(
                    lower=0.10 / (_A_MAX_FRACTION + 0.10),
                    upper=0.39 / (_A_MAX_FRACTION + 0.39),
                ),
                provenance=_STAND_IN
                + "example_calibration's: from the BETY baseFolRespFrac 2.5-97.5% "
                "range 0.10-0.39 at fixed aMaxFrac.",
            ),
            "base_soil_respiration": PriorTerm(
                log_normal_from_interval(lower=0.004, upper=0.020),
                provenance=_STAND_IN
                + "example_calibration's: the BETY som_respiration_rate posterior, "
                "2.5-97.5% quantiles 0.004-0.020 yr-1.",
            ),
        },
    )


def _stand_in_map() -> SIPNETParameterMap:
    """The rules for the three parameters and the initial state, and the fixed values."""
    return SIPNETParameterMap(
        rules=[
            ComputePhotosynthesisRates(
                capacity_value_name="photosynthetic_capacity",
                respiration_share_value_name="respiration_share",
            ),
            Copy(
                value_name="base_soil_respiration",
                sipnet_parameter_name="base_soil_respiration_rate",
            ),
            ComputeInitialConditions(deciduous=DECIDUOUS_BY_SITE_LABEL),
        ],
        fixed=[
            Fixed(
                sipnet_parameter_name="daily_mean_photosynthesis_fraction",
                value=_A_MAX_FRACTION,
                provenance=_STAND_IN
                + "example_calibration's 0.76; aMaxFrac spans one of the two "
                "degenerate photosynthesis directions (sipnet.c:614, 617, 633).",
            ),
            Fixed(
                sipnet_parameter_name="leaf_carbon_fraction",
                value=_LEAF_CARBON_FRACTION,
                provenance=_STAND_IN
                + "example_calibration's 0.466, the BETY temperate-deciduous leafC "
                "median; the other degenerate direction.",
            ),
            Fixed(
                sipnet_parameter_name="vapor_pressure_deficit_exponent",
                value=2.0,
                provenance="Braswell et al. (2005) fix the exponent at 2.",
            ),
            Fixed(
                sipnet_parameter_name="leaf_carbon_per_area",
                value=40.0,
                provenance=_STAND_IN
                + "About 80 g of dry leaf mass per m2 of leaf at 0.47 carbon, the "
                "order of broadleaf deciduous leaves; the base set's 270 is a "
                "conifer's.",
            ),
            Fixed(
                sipnet_parameter_name="leaf_off_fall_fraction",
                value=1.0,
                provenance=_STAND_IN
                + "A deciduous canopy drops every leaf at leaf-off; the base set's "
                "0 is an evergreen's.",
            ),
            Fixed(
                sipnet_parameter_name="leaf_on_growth",
                value=200.0,
                provenance=_STAND_IN
                + "A full canopy at leaf-on, LAI about 5 at 40 g C m-2 of leaf. "
                "SIPNET grows leaves at leaf-on only by this amount (sipnet.c:819), "
                "so a deciduous run starting leafless with the base set's 0, an "
                "evergreen's, never grows a leaf.",
            ),
            Fixed(
                sipnet_parameter_name="fine_root_fraction",
                value=0.2,
                provenance=_STAND_IN
                + "The base set's value, read by the initial state.",
            ),
            Fixed(
                sipnet_parameter_name="coarse_root_fraction",
                value=0.2,
                provenance=_STAND_IN
                + "The base set's value, read by the initial state.",
            ),
        ],
    )
