"""The AIMES talk: every number behind its figures.

The one source of truth for the talk's computations. The modules under
``toy/``, ``inference/`` and ``illustrations/`` define *how* things are
computed and take every number as an argument; the scripts under ``run/``
pass them the values below. Change a value here and rerun the scripts
(``README.md`` lists the commands), and every figure follows.

Sections, in order:

- **paths**: where results and figures are written;
- **the toy region**: the sites, the climate covariate and the drivers;
- **the toy carbon model**: the forward model's fixed constants;
- **the spatial prior**: how the two parameters vary over the region;
- **the observations**: which sites see what, and how noisy it is;
- **site-by-site calibration**: the grid each site's posterior is computed on;
- **the spatial reference posterior**: the MCMC settings;
- **the scalable method (IPL)**: ensemble size and iterations;
- **the illustrations**: the North America prior draws and uncertainty, and
  the calibration options.

Units in the toy are deliberately generic (carbon in "units of C", time in
months); it is a stand-in for SIPNET, not a model of a real ecosystem.
"""

from pathlib import Path

import numpy as np

# ── paths ───────────────────────────────────────────────────────────────────

TALK_DIRECTORY = Path(__file__).resolve().parent
#: Computed results (``.npz``), written by ``run/`` and read by ``plots.py``.
#: Untracked: regenerate with the commands in ``README.md``.
OUTPUT_DIRECTORY = TALK_DIRECTORY / "output"
#: The figures the slides show, written by ``run/draw_figures.py``.
FIGURE_DIRECTORY = TALK_DIRECTORY / "figures"
#: The brand palette and fonts, shared with the slides' theme.
BRAND_FILE = TALK_DIRECTORY / "theme" / "_brand.yml"

TOY_RESULTS_FILE = OUTPUT_DIRECTORY / "toy_results.npz"
SITE_POSTERIORS_FILE = OUTPUT_DIRECTORY / "site_posteriors.npz"
CALIBRATION_OPTIONS_FILE = OUTPUT_DIRECTORY / "calibration_options.npz"
UNCERTAINTY_FILE = OUTPUT_DIRECTORY / "uncertainty.npz"
NORTH_AMERICA_FILE = OUTPUT_DIRECTORY / "north_america_prior.npz"

# ── the toy region ──────────────────────────────────────────────────────────
# A square region, coordinates in [0, 1]^2. Sites are scattered uniformly at
# random. "Training" sites carry data; "held-out" sites carry none, and are
# where we check how well each method predicts a site it has not seen.

#: Seed for everything random in the toy's setup and synthetic truth.
TOY_SEED = 20261027
N_TRAINING_SITES = 200
N_HELD_OUT_SITES = 50
#: The maps of the parameter field are drawn on a regular grid of this many
#: points per side.
MAP_GRID_SIZE = 50

#: Mean annual temperature (deg C) over the region is
#:     MAT(x) = MAT_CENTER + MAT_SCALE * z(x),
#:     z(x) = 2 (0.5 - x2) + 0.35 sin(2 pi x1),
#: so it is colder to the north (large x2) with a gentle east-west wave.
#: z(x) is also the covariate in the prior mean (spatial prior section).
MAT_CENTER = 10.0
MAT_SCALE = 6.0

#: Monthly drivers at each site, over N_MONTHS months:
#:     temperature(t) = MAT(x) + SEASONAL_AMPLITUDE sin(2 pi (t - 3) / 12)
#:                      + weather(t),
#: where weather is an AR(1) series with this coefficient and innovation SD,
#: drawn independently at each site;
#:     light(t) = 1 + (0.3 + 0.4 x2) sin(2 pi (t - 2.5) / 12),
#: so the light's seasonal swing is larger in the north.
N_MONTHS = 60
SEASONAL_AMPLITUDE = 9.0
WEATHER_AUTOCORRELATION = 0.6
WEATHER_INNOVATION_SD = 1.0

# ── the toy carbon model ────────────────────────────────────────────────────
# One carbon pool C, monthly (toy/forward_model.py has the equations).
# The two calibrated parameters are productivity and turnover; everything
# below is held fixed.

#: The pool at the start of the record. Well below the pool's steady state,
#: so the five years trace a recovering, still-growing pool; see
#: forward_model.py for why that matters.
INITIAL_POOL = 0.5
#: Photosynthesis switches on with temperature as
#: sigmoid((T - TEMPERATURE_HALF_SATURATION) / TEMPERATURE_SATURATION_WIDTH).
TEMPERATURE_HALF_SATURATION = 5.0
TEMPERATURE_SATURATION_WIDTH = 3.0
#: Respiration rises with temperature as exp(Q (T - REFERENCE_TEMPERATURE)).
RESPIRATION_TEMPERATURE_SENSITIVITY = 0.09
REFERENCE_TEMPERATURE = 10.0

# ── the spatial prior ───────────────────────────────────────────────────────
# Each parameter p (on the log scale) is a field over the region:
#     theta_p(x) = BETA[p, 0] + BETA[p, 1] z(x)     climate trend
#                + FIELD_SD[p] u_p(x)                smooth variation
#                + SITE_NOISE_SD[p] eta_p(x)         site-to-site noise
# u_p is a Gaussian process with unit variance and a Matern-3/2 correlation
# of length scale LENGTH_SCALE[p] (correlation 0.48 at that distance), and
# eta_p is independent at every site. The two parameters are a priori
# independent. Every number here is treated as known by every method.

PARAMETER_NAMES = ("productivity", "turnover")
#: How the parameters are labeled on figures (theta is the log).
PARAMETER_LABELS = ("log productivity", "log turnover rate")
#: Rows are parameters, columns (intercept, slope on z). Warmer sites are
#: both more productive and faster to turn over.
BETA = np.array([
    [np.log(0.10), 0.30],
    [np.log(0.012), 0.20],
])
FIELD_SD = np.array([0.70, 0.70])
LENGTH_SCALE = np.array([0.35, 0.20])
SITE_NOISE_SD = np.array([0.17, 0.17])

# ── the observations ────────────────────────────────────────────────────────
# Every training site observes the pool at the end of each year (think
# biomass or soil carbon inventories, or a remote-sensing product). A few
# "tower" sites also observe monthly net ecosystem exchange (NEE).

#: The fraction of training sites that are towers.
TOWER_FRACTION = 0.10
#: Observation noise SDs, Gaussian and independent, fixed per site from the
#: model's output at that site's prior mean parameters (not the truth):
#: the pool's SD is POOL_RELATIVE_NOISE x the mean pool there, and NEE's is
#: NEE_RELATIVE_NOISE x the root-mean-square NEE there.
POOL_RELATIVE_NOISE = 0.10
NEE_RELATIVE_NOISE = 0.15

# ── site-by-site calibration ────────────────────────────────────────────────
# Each site alone, with its own prior (its marginal under the spatial prior,
# ignoring every other site). With two parameters its posterior is computed
# exactly on a grid, so no sampler is involved.

#: Grid points per parameter. The first grid spans the prior mean +- the
#: first half-width in prior SDs. A tower's posterior is far narrower than
#: its prior, so the grid is then re-centered on the posterior mean and
#: narrowed to +- the second half-width in posterior SDs, this many times.
SITE_GRID_SIZE = 161
SITE_GRID_HALF_WIDTH_IN_PRIOR_SDS = 4.5
SITE_GRID_HALF_WIDTH_IN_POSTERIOR_SDS = 6.0
SITE_GRID_REFINEMENTS = 2

# ── the spatial reference posterior ─────────────────────────────────────────
# The exact posterior of the spatial model, every training site jointly,
# sampled by NUTS (gradient-based MCMC). It needs gradients of the model,
# which the toy has and SIPNET does not: it is the yardstick, not a method
# we could run on SIPNET.

MCMC_SEED = 1
MCMC_CHAINS = 4
MCMC_WARMUP = 500
MCMC_SAMPLES = 500

# ── the scalable method (IPL) ───────────────────────────────────────────────
# Iterated posterior linearization: at each iteration, IPL_ENSEMBLE_SIZE
# model runs per site, then one global Gaussian update (inference/ipl.py).
# It stops when no site's mean moved by more than IPL_CONVERGENCE_TOLERANCE
# prior SDs in the last iteration, or after IPL_MAX_ITERATIONS.

IPL_SEED = 2
IPL_ENSEMBLE_SIZE = 30
IPL_MAX_ITERATIONS = 15
IPL_CONVERGENCE_TOLERANCE = 0.01

# ── the site posteriors ─────────────────────────────────────────────────────
# Two sites' posteriors as contours (run/site_posteriors.py): the exact
# site-by-site density, and a Gaussian KDE of the spatial posterior's MCMC
# draws on a KDE_GRID_SIZE x KDE_GRID_SIZE grid, checked as
# inference/density.py describes.

#: The probability inside each contour drawn.
CONTOUR_PROBABILITIES = (0.5, 0.9)
KDE_GRID_SIZE = 200
#: The KDE's bandwidth as a multiple of Scott's rule. At 1 (Scott's rule)
#: the contours are smooth, and the reproduction check shows its regions
#: holding a few percent more than they should: slightly too wide, which can
#: only understate how much the spatial posterior narrows. Smaller factors
#: shrink that bias but draw the MCMC draws' noise as wiggly contours.
KDE_BANDWIDTH_FACTOR = 1.0
KDE_CHECK_SEED = 5

# ── the illustrations ───────────────────────────────────────────────────────

#: Prior draws of one parameter over North America, as continuous maps: a
#: climate trend in mean annual temperature plus a Matern-3/2 Gaussian
#: process in the chordal distance through the Earth
#: (illustrations/north_america.py). Lengths in meters, areas in square
#: meters. The field is drawn exactly on a grid NORTH_AMERICA_GRID_SPACING
#: apart and shown on a raster NORTH_AMERICA_DISPLAY_SPACING apart, within
#: NORTH_AMERICA_LAND_RADIUS of a site, with enclosed holes smaller than
#: NORTH_AMERICA_MAX_HOLE_AREA filled.
NORTH_AMERICA_SEED = 3
NORTH_AMERICA_N_DRAWS = 3
NORTH_AMERICA_LENGTH_SCALE = 800e3
NORTH_AMERICA_FIELD_SD = 0.6
NORTH_AMERICA_TREND_SLOPE = 0.5
NORTH_AMERICA_GRID_SPACING = 60e3
NORTH_AMERICA_DISPLAY_SPACING = 15e3
NORTH_AMERICA_LAND_RADIUS = 90e3
NORTH_AMERICA_MAX_HOLE_AREA = 3e10

#: The calibration options compared on one slide: a parameter drawn
#: independently at each site, and one drawn per PFT (the pft_16class site
#: labels), each N(0, CALIBRATION_OPTIONS_PARAMETER_SD^2)
#: (illustrations/calibration_options.py). The spatial option is a North
#: America prior draw.
CALIBRATION_OPTIONS_SEED = 4
CALIBRATION_OPTIONS_PARAMETER_SD = 0.6
CALIBRATION_OPTIONS_SITE_LABELS = "pft_16class"

#: The prior and posterior uncertainty of the North America field
#: (illustrations/uncertainty.py), with its prior's length scale and SD: every
#: site measures the parameter with noise SD UNCERTAINTY_SITE_NOISE_SD (weak
#: data), and UNCERTAINTY_N_TOWERS spread-out tower sites with
#: UNCERTAINTY_TOWER_NOISE_SD (strong data). The site noise is large because
#: thousands of sites fall within one length scale and their data pool: at
#: 3 the posterior is low everywhere and the towers' effect is lost; at 10
#: the drop away from towers is modest and the towers stand out.
UNCERTAINTY_N_TOWERS = 20
UNCERTAINTY_TOWER_NOISE_SD = 0.05
UNCERTAINTY_SITE_NOISE_SD = 10.0
