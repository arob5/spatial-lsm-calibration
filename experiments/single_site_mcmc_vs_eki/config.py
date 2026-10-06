"""Single-site calibration of SIPNET, MCMC against EKI: the configuration.

The one source of truth for this experiment. Every script here reads its
choices from this module and nothing else; change the site, the driver member
or an observation source here, and every step follows.

Sections, in order:

- **the site**: which site is calibrated;
- **paths**: where the experiment reads raw inputs and writes everything else;
- **the drivers**: which driver member runs, and how its file is corrected;
- **the model**: what SIPNET simulates and its base parameters, which change
  results;
- **running SIPNET**: how the runs are executed, which does not;
- **the observations**: the NEE series and how they are windowed, the pool
  constraints, the data sources left out, and the operator that predicts each
  observation source;
- **the prior predictive**: the ensemble's size and seed, and what it
  writes;
- **the noise model**: the measurement-error floors, the discrepancy terms,
  NEE's discrepancy size and the noise scales' prior (the error models are
  ``model/nee_error.py``'s, the models ``models.py``'s);
- **EKI**: where runs write, the ensemble, the tempering ladder, the seeds;
- **importance sampling and SMC after EKI**, and **MCMC**: their settings.

The parameterization and the prior are ``model/prior.py``'s.
"""

import os
from datetime import timedelta
from pathlib import Path

from frozendict import frozendict
from pysipnet import niwot_reference_files
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import ClimateStaging

from sipnet_calibration.conventions import data_root
from sipnet_calibration.observation import DEFAULT_OBS_OPS, ReduceOverRun

from .model.operators import AverageRateOverWindows, ComputeAbovegroundBiomass

__all__ = [
    "BASE_SIPNET_PARAMETER_FILE",
    "CALIBRATION_NEE_PERIOD",
    "CALIBRATION_NEE_SERIES",
    "CLIMATE_STAGING",
    "CONSTRAINT_NAMES",
    "DRIVER_SOURCE_INDEX",
    "DRIVER_TIME_ZONE",
    "EKI_ENSEMBLE_SIZE",
    "EKI_ESS_FRACTION",
    "EKI_SEED",
    "EXCLUDED_CONSTRAINTS",
    "EXPERIMENT_DIRECTORY",
    "FIGURE_DIRECTORY",
    "LAI_DISCREPANCY_STANDARD_DEVIATION",
    "LAI_DISCREPANCY_TIMESCALE",
    "LAI_STANDARD_DEVIATION_FLOOR",
    "LANDTRENDR_DISCREPANCY_STANDARD_DEVIATION",
    "LANDTRENDR_YEARS",
    "MCMC_ADAPTATION_STEPS",
    "MCMC_CHAINS_PER_WORKER",
    "MCMC_DISCARDED_STEPS",
    "MCMC_SEED",
    "MCMC_STEPS",
    "MODEL_FLAGS",
    "NEE_DISCREPANCY_STANDARD_DEVIATION",
    "NOISE_SCALE_SHAPE",
    "NOISE_SCALED_SOURCES",
    "NEE_MINIMUM_MEASURED_FRACTION",
    "NEE_WINDOWS",
    "N_WORKERS",
    "OBSERVATION_OPERATORS",
    "OUTPUT_DIRECTORY",
    "PREPARED_DRIVERS_ROOT",
    "PRIOR_PREDICTIVE_DIRECTORY",
    "PRIOR_PREDICTIVE_ENSEMBLE_SIZE",
    "PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES",
    "PRIOR_PREDICTIVE_SEED",
    "RAW_DRIVERS_ROOT",
    "REWEIGHTING_COVARIANCE_INFLATION",
    "REWEIGHTING_DEFENSIVE_FRACTION",
    "REWEIGHTING_DEGREES_OF_FREEDOM",
    "REWEIGHTING_SEED",
    "IMPORTANCE_SAMPLE_SIZE",
    "RUNS_DIRECTORY",
    "SMC_SAMPLE_SIZE",
    "SIPNET_TIMEOUT",
    "SITE",
    "SOIL_CARBON_DISCREPANCY_FRACTION",
    "SOIL_TEMPERATURE_TIMESCALE",
    "SYNTHETIC_TRUTH_SEED",
    "VALIDATION_NEE_PERIOD",
    "VALIDATION_NEE_SERIES",
    "WOOD_CARBON_FRACTION",
    "WOOD_CARBON_FRACTION_UNCERTAINTY",
]

# ── the site ──

#: The site calibrated, by its id in the site pool: Harvard Forest, where the
#: AmeriFlux towers US-Ha1 (hourly) and US-xHA (NEON, half-hourly) both stand.
SITE = 4977

# ── paths ──

#: This experiment's directory.
EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent

#: Everything the experiment writes, untracked.
OUTPUT_DIRECTORY = EXPERIMENT_DIRECTORY / "output"

#: Where the raw driver files are: ``ERA5_<site>_<index>/`` directories of
#: ``.clim`` files, as copied from the SCC. Never edited.
RAW_DRIVERS_ROOT = data_root() / "raw" / "drivers"

#: The driver file the runs read: the raw file of :data:`SITE` and
#: :data:`DRIVER_SOURCE_INDEX`, corrected by ``run/prepare_drivers.py``,
#: laid out as the raw ones.
PREPARED_DRIVERS_ROOT = OUTPUT_DIRECTORY / "drivers"

# ── the drivers ──

#: The ERA5 ensemble member that drives every run, by its 1-based index in the
#: driver directory names (``ERA5_<site>_<index>``). One member keeps the
#: forward model deterministic, which the MCMC comparison needs.
DRIVER_SOURCE_INDEX = 1

#: The clock the prepared driver file's labels are on, declared to pySIPNET.
#: ``run/prepare_drivers.py`` relabels each row with the UTC start of the
#: step its values describe, so the model's time axis is UTC, the observed
#: NEE's clock.
DRIVER_TIME_ZONE = "UTC"

#: The timescale of the exponential filter of air temperature that
#: ``run/prepare_drivers.py`` computes soil temperature with. It is
#: PEcAn's ``met2model.SIPNET`` choice; there the filter averages the
#: following weeks, here the preceding ones.
SOIL_TEMPERATURE_TIMESCALE = timedelta(days=15)

# ── the model ──

#: Which of SIPNET's optional processes are on: pySIPNET's standard set, the
#: binary's own defaults. Leaf-on is decided by growing degree-days (not soil
#: temperature); snow and the soil-moisture limit on decomposition are on; the
#: litter pool is off, so litter goes straight into ``soil_carbon``, which is
#: what the soil carbon operator reads; growth respiration and leaf water are
#: off. The flags also decide which SIPNET parameters must be given.
MODEL_FLAGS = ModelFlags.standard()

#: The base SIPNET parameter set: the value of every SIPNET parameter a
#: calibration's map leaves unset. pySIPNET's Niwot Ridge reference parameters,
#: a subalpine conifer forest, the only complete set on hand. The calibration's
#: map (``model/prior.py``) sets every SIPNET parameter, its fixed ones from
#: the temperate deciduous values of ``model/fixed_sipnet_parameters.csv``, so
#: no value of this set reaches a calibrated run.
BASE_SIPNET_PARAMETER_FILE = niwot_reference_files().param

# ── running SIPNET ──
#
# How the runs are executed. None of this changes a result. Left at
# pySIPNET's defaults: the binary, found by pySIPNET's own lookup
# ($PYSIPNET_BINARY, then its cache), and checked against the pinned SIPNET
# tag before the first run; each run's working directory, a fresh one under
# the system temporary directory ($TMPDIR on the SCC), deleted afterwards;
# and the output, parsed into memory rather than kept as SIPNET's text file.
# Results worth keeping are written by the experiment, under OUTPUT_DIRECTORY.

#: The longest one SIPNET run may take before it is abandoned and its row
#: marked failed. A run over the whole record takes about a second.
SIPNET_TIMEOUT = timedelta(seconds=60)

#: How the driver file reaches each run's working directory: linked, not
#: copied, since every run reads the same file.
CLIMATE_STAGING = ClimateStaging.SYMLINK

#: How many SIPNET runs execute at once: ``$SIPNET_WORKERS``, which an SCC
#: job sets to its slots, else 3 on the laptop, a fixed budget beside other
#: sessions' ensembles. Each worker holds about 0.3 GB and the calling
#: process about 0.8 GB.
N_WORKERS = int(os.environ.get("SIPNET_WORKERS", "3"))

# ── the observations: NEE ──

#: The NEE series calibrated against: US-Ha1's hourly record, with the u*
#: threshold estimated per year (``NEE_VUT_REF``). It covers 2012-2020 of the
#: drivers' 2012-2024. The variable-threshold series exists at more of the
#: pool's sites than the constant one, and the two differ little here.
CALIBRATION_NEE_SERIES = "ameriflux_nee_hourly_ustar_variable"

#: The years of :data:`CALIBRATION_NEE_SERIES` calibrated against, inclusive,
#: as ``(first, last)`` calendar years in UTC: all of US-Ha1's record inside
#: the drivers'.
CALIBRATION_NEE_PERIOD = (2012, 2020)

#: The NEE series held out for an out-of-sample check: US-xHA's half-hourly
#: record, a second tower at the same site, over the years the calibration
#: series does not cover.
VALIDATION_NEE_SERIES = "ameriflux_nee_half_hourly_ustar_variable"

#: The years of :data:`VALIDATION_NEE_SERIES` held out, inclusive, as
#: ``(first, last)`` calendar years in UTC.
VALIDATION_NEE_PERIOD = (2021, 2024)

#: The windows observed NEE is averaged over, each an observation source:
#: name to ``(start, end)``, offsets from 00:00 UTC of each day. Each day is
#: split in two twelve-hour windows, one centered on the day and one on the
#: night (07:00-19:00 and 19:00-07:00 local standard time at Harvard Forest,
#: UTC-5). A whole day would need both halves measured, which keeps too few
#: summer days: calm summer nights fail the u* filter. Both edges are
#: timestep edges of the three-hourly model, so a window averages whole steps.
NEE_WINDOWS = frozendict(
    {
        "nee_night_centered": (timedelta(hours=0), timedelta(hours=12)),
        "nee_day_centered": (timedelta(hours=12), timedelta(hours=24)),
    }
)

#: The fewest of a window's NEE values that must be measured rather than
#: gap-filled (quality flag 0) for the window to be an observation, as a
#: fraction of the window's values. Below it the window's mean is mostly
#: marginal distribution sampling, a model of its own.
NEE_MINIMUM_MEASURED_FRACTION = 0.5

# ── the observations: pool constraints ──

#: The constraints calibrated against, by name in
#: :data:`sipnet_calibration.constraints.CONSTRAINTS`.
CONSTRAINT_NAMES = (
    "modis_leaf_area_index",
    "landtrendr_aboveground_biomass",
    "soilgrids_soil_organic_carbon",
)

#: The years of LandTrendr biomass calibrated against, inclusive: those whose
#: values and uncertainties are LandTrendr's own. From 2018 the uncertainties
#: come from a random-forest prediction beside the product, and both they and
#: the values change at that boundary (``data/README.md`` open question 19).
LANDTRENDR_YEARS = (2012, 2017)

#: The mass fraction of dry wood that is carbon, which turns SIPNET's wood
#: carbon into the dry biomass LandTrendr estimates: the IPCC default for
#: temperate broadleaf wood, 0.48 (2006 IPCC Guidelines, Volume 4, Table 4.3).
#: LandTrendr is taken to be dry biomass, although its spec records carbon,
#: because it is at the site about twice the carbon density of Spawn and
#: Gibbs (2020), the initial conditions' aboveground carbon map, and its spec
#: says the constituent is unconfirmed by about that factor.
WOOD_CARBON_FRACTION = 0.48

# ── the observations: data sources left out ──

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
        "gedi_aboveground_biomass": (
            "Its units are not established, carbon or dry biomass "
            "(data/README.md open question 9), and its three values at the "
            "site fall from 145 to 78 over 2019-2022 while LandTrendr's change "
            "by a few percent, so three values of unknown meaning would add "
            "little but their disagreement."
        ),
    }
)

# ── the observations: operators ──

#: Which operator predicts each observation source, by observation source
#: name. NEE is a mean rate over its window; leaf area index is the
#: library's default, leaf carbon over leaf carbon per area at the step
#: containing each composite's label; biomass is wood carbon over the carbon
#: fraction, averaged over each year's window; soil carbon, a single static
#: value, is SIPNET's soil pool averaged over the run.
OBSERVATION_OPERATORS = frozendict(
    {
        **{
            name: AverageRateOverWindows("net_ecosystem_exchange")
            for name in NEE_WINDOWS
        },
        "modis_leaf_area_index": DEFAULT_OBS_OPS["modis_leaf_area_index"],
        "landtrendr_aboveground_biomass": ComputeAbovegroundBiomass(
            WOOD_CARBON_FRACTION
        ),
        "soilgrids_soil_organic_carbon": ReduceOverRun("soil_carbon", how="mean"),
    }
)

# ── the prior predictive ──
#
# The prior center's run and an ensemble of prior draws, the model against the
# data before calibration; run/prior_predictive.py runs them and
# figures/prior_predictive.py draws them.

#: Where the prior predictive writes its runs.
PRIOR_PREDICTIVE_DIRECTORY = OUTPUT_DIRECTORY / "prior_predictive"

#: Where the figures go.
FIGURE_DIRECTORY = OUTPUT_DIRECTORY / "figures"

#: How many prior draws the ensemble runs.
PRIOR_PREDICTIVE_ENSEMBLE_SIZE = 200

#: The seed of the ensemble's prior draws.
PRIOR_PREDICTIVE_SEED = 20260929

#: The model output variables the prior predictive keeps as daily trajectories,
#: for the figures: what the observation operators read, and the fluxes and
#: pools behind them.
PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES = (
    "net_ecosystem_exchange",
    "gross_primary_production",
    "ecosystem_respiration",
    "leaf_carbon",
    "wood_carbon",
    "soil_carbon",
)

# ── the noise model ──
#
# The covariance R of the observation errors, block-diagonal over the
# observation sources, each block measurement error plus model discrepancy;
# MODEL.md, "Noise model", states it exactly and model/noise.py declares it. The
# discrepancy terms carry most of the weight: they set how much each source
# constrains the calibration. NEE's discrepancy has the three terms of
# model/discrepancy.py; the LAI timescale is that of the exponential
# correlation exp(-|t - t'| / tau).

#: The standard deviation of NEE's discrepancy at one window, per NEE source,
#: in umol m-2 s-1 of CO2: the same in every error model, so that the error
#: models differ only in memory (model/nee_error.py). It is the first
#: calibration's residual standard deviation beyond measurement error,
#: MODEL.md's s_hat_k.
NEE_DISCREPANCY_STANDARD_DEVIATION = frozendict(
    {"nee_night_centered": 1.43, "nee_day_centered": 2.99}
)

#: The sources whose noise covariance has an unknown scale s_k, R_k = s_k C_k,
#: in a model with inferred noise; LandTrendr (6 values) and SoilGrids (1)
#: are too few to inform one and keep theirs fixed.
NOISE_SCALED_SOURCES = (
    "nee_night_centered",
    "nee_day_centered",
    "modis_leaf_area_index",
)

#: The shape a of each scale's inverse-gamma prior, IG(a, b); b is set so
#: that the prior median is 1, the scale a model with fixed noise holds.
NOISE_SCALE_SHAPE = 2.0

#: The smallest standard deviation a MODIS LAI observation is given, in
#: m2 m-2: the reanalysis's floor (data/README.md open question 22). The
#: product's own is a spread among retrieval solutions and reaches 0.1.
LAI_STANDARD_DEVIATION_FLOOR = 0.66

#: The standard deviation of LAI's model discrepancy, in m2 m-2.
LAI_DISCREPANCY_STANDARD_DEVIATION = 0.5

#: The timescale of LAI's discrepancy correlation within one summer; composites
#: of different summers are uncorrelated.
LAI_DISCREPANCY_TIMESCALE = timedelta(days=30)

#: The uncertainty of :data:`WOOD_CARBON_FRACTION`: half the IPCC range for
#: temperate broadleaf wood, 0.46-0.50. Its error is shared by every
#: LandTrendr year.
WOOD_CARBON_FRACTION_UNCERTAINTY = 0.02

#: The standard deviation of LandTrendr's year-to-year model discrepancy, in
#: Mg ha-1, independent between years.
LANDTRENDR_DISCREPANCY_STANDARD_DEVIATION = 5.0

#: The standard deviation of the soil carbon discrepancy, as a fraction of
#: the observed stock: for the depth and definition SIPNET's single soil pool
#: does not share with a 0-200 cm stock (data/README.md open question 21).
SOIL_CARBON_DISCREPANCY_FRACTION = 0.25

# ── EKI ──
#
# Ensemble Kalman inversion in its sampling form: an ensemble drawn from the
# prior, moved up the tempering ladder from beta = 0 (the prior) to beta = 1
# (the posterior) by EnsKit's perturbed-observation (Matheron) update, the
# increments chosen adaptively. algorithms/eki.py runs it (run/calibrate.py).

#: Where every calibration run writes: ``<nee error model>/<noise>/<algorithm>/``.
#: ``$HF_RUNS_DIRECTORY`` moves it, for a smoke test.
RUNS_DIRECTORY = Path(os.environ.get("HF_RUNS_DIRECTORY", OUTPUT_DIRECTORY / "runs"))

#: The number of ensemble members, J. Every iterate lies in the affine span of
#: the initial ensemble, so J - 1 must exceed theta's 15 entries with room to
#: spare.
EKI_ENSEMBLE_SIZE = 100

#: The seed of the initial ensemble's prior draw and of the run's own key,
#: which the updates' perturbed observations are drawn with.
EKI_SEED = 20260930

#: The effective sample size, as a fraction of J, the adaptive ladder's
#: increments target (EnsKit's AdaptiveESSSchedule and its default).
EKI_ESS_FRACTION = 0.5

#: The seed of the synthetic truth: one prior draw theta*, and the noise
#: realization added to its predictions to make the synthetic observations.
SYNTHETIC_TRUTH_SEED = 20260931

# ── importance sampling and SMC after EKI ──
#
# A Student-t fitted to an EKI run's final ensemble, mixed with the prior,
# is the base density q; importance sampling draws from q once, tempered SMC
# moves from q to the posterior (MODEL.md, "IS and SMC after EKI").

#: The degrees of freedom of the Student-t fitted to the EKI ensemble.
REWEIGHTING_DEGREES_OF_FREEDOM = 5.0

#: The factor the EKI ensemble's covariance is inflated by in the Student-t.
REWEIGHTING_COVARIANCE_INFLATION = 1.5

#: The prior's share of the base density, which bounds the weights.
REWEIGHTING_DEFENSIVE_FRACTION = 0.1

#: The number of draws importance sampling makes.
IMPORTANCE_SAMPLE_SIZE = 1000

#: The number of samples tempered SMC carries.
SMC_SAMPLE_SIZE = 500

#: The seed of the base density's draws and of SMC's moves.
REWEIGHTING_SEED = 20261006

# ── MCMC ──
#
# Parallel random-walk Metropolis on the posterior of theta (the scales
# integrated out where they are inferred), started from an importance-
# sampling run's resampled draws, its proposal covariance theirs, adapted
# from the chains during the first steps and then frozen.

#: Chains per worker: the chains advance together, one batch of runs a step.
MCMC_CHAINS_PER_WORKER = 2

#: The number of steps each chain takes.
MCMC_STEPS = 2000

#: The steps during which the proposal covariance is re-estimated.
MCMC_ADAPTATION_STEPS = 500

#: The steps discarded from the start of every chain, adaptation included.
MCMC_DISCARDED_STEPS = 1000

#: The seed of the chains' proposals and starting draws.
MCMC_SEED = 20261007
