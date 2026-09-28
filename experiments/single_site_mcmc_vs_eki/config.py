"""Single-site calibration of SIPNET, MCMC against EKI: the configuration.

The one source of truth for this experiment. Every script here reads its
choices from this module and nothing else; change the site, the driver member
or an observation source here, and every step follows.

What is configured so far (step 1): the site, the driver member and how its
file is corrected, the data sources the calibration reads and the ones it
leaves out, and where the experiment writes. The observation operators, the
parameterization, the priors and the algorithm settings are added by later
steps.
"""

from datetime import timedelta
from pathlib import Path

from frozendict import frozendict

from sipnet_calibration.conventions import data_root

__all__ = [
    "CALIBRATION_NEE_SERIES",
    "CONSTRAINT_NAMES",
    "DRIVER_SOURCE_INDEX",
    "DRIVER_TIME_ZONE",
    "EXCLUDED_CONSTRAINTS",
    "EXPERIMENT_DIRECTORY",
    "OUTPUT_DIRECTORY",
    "PREPARED_DRIVERS_ROOT",
    "RAW_DRIVERS_ROOT",
    "SITE",
    "SOIL_TEMPERATURE_TIMESCALE",
    "VALIDATION_NEE_PERIOD",
    "VALIDATION_NEE_SERIES",
]

# ── the site ──

#: The site calibrated, by its id in the site pool: Harvard Forest, where the
#: AmeriFlux towers US-Ha1 (hourly) and US-xHA (NEON, half-hourly) both stand.
SITE = 4977

# ── the drivers ──

#: The ERA5 ensemble member that drives every run, by its 1-based index in the
#: driver directory names (``ERA5_<site>_<index>``). One member keeps the
#: forward model deterministic, which the MCMC comparison needs.
DRIVER_SOURCE_INDEX = 1

#: The clock the prepared driver file's labels are on, declared to pySIPNET.
#: ``prepare_drivers.py`` relabels each row with the UTC start of the step its
#: values describe, so the model's time axis is UTC, the observed NEE's clock.
DRIVER_TIME_ZONE = "UTC"

#: The timescale of the exponential filter of air temperature that
#: ``prepare_drivers.py`` computes soil temperature with. It is PEcAn's
#: ``met2model.SIPNET`` choice; there the filter averages the following weeks,
#: here the preceding ones.
SOIL_TEMPERATURE_TIMESCALE = timedelta(days=15)

#: Where the raw driver files are: ``ERA5_<site>_<index>/`` directories of
#: ``.clim`` files, as copied from the SCC. Never edited.
RAW_DRIVERS_ROOT = data_root() / "raw" / "drivers"

# ── the observations ──

#: The NEE series calibrated against: US-Ha1's hourly record, with the u*
#: threshold estimated per year (``NEE_VUT_REF``). It covers 2012-2020 of the
#: drivers' 2012-2024. The variable-threshold series exists at more of the
#: pool's sites than the constant one, and the two differ little here.
CALIBRATION_NEE_SERIES = "ameriflux_nee_hourly_ustar_variable"

#: The NEE series held out for an out-of-sample check: US-xHA's half-hourly
#: record, a second tower at the same site, over the years the calibration
#: series does not cover.
VALIDATION_NEE_SERIES = "ameriflux_nee_half_hourly_ustar_variable"

#: The years of :data:`VALIDATION_NEE_SERIES` held out, inclusive, as
#: ``(first, last)`` calendar years in UTC.
VALIDATION_NEE_PERIOD = (2021, 2024)

#: The constraints calibrated against, by name in
#: :data:`sipnet_calibration.constraints.CONSTRAINTS`.
CONSTRAINT_NAMES = (
    "modis_leaf_area_index",
    "landtrendr_aboveground_biomass",
    "gedi_aboveground_biomass",
    "soilgrids_soil_organic_carbon",
)

#: The constraints left out, each with the reason.
EXCLUDED_CONSTRAINTS = frozendict(
    {
        "smap_soil_moisture": (
            "SMAP L4 total-profile volumetric water content, in percent, one "
            "3-hourly snapshot on 07-15 each year on a 9 km grid. It has no "
            "defensible counterpart in SIPNET's soil water, a bucket whose "
            "capacity is porosity times the soil file's depth: the reanalysis "
            "compared it with the bucket's fraction of capacity, which is a "
            "degree of saturation, not a volumetric content, and the two "
            "differ by the porosity. About ten values at one site would not "
            "repay an operator built on an assumed depth and porosity."
        ),
    }
)

# ── where the experiment writes ──

#: This experiment's directory.
EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent

#: Everything the experiment writes, untracked.
OUTPUT_DIRECTORY = EXPERIMENT_DIRECTORY / "output"

#: The driver file the runs read: the raw file of :data:`SITE` and
#: :data:`DRIVER_SOURCE_INDEX`, corrected by ``prepare_drivers.py``, laid out
#: as the raw ones.
PREPARED_DRIVERS_ROOT = OUTPUT_DIRECTORY / "drivers"
