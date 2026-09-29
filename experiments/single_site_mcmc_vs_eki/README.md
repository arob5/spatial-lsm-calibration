# Single-site calibration: MCMC against EKI

Calibrate SIPNET at one site, Harvard Forest by default, to observed NEE and
the pool constraints, and compare the posterior an MCMC sampler finds with the
one EKI finds. `config.py` is the one source of truth; nothing else here makes
a choice. `MODEL.md` states the model exactly: the observation operators, the
noise covariance, and the parameterization and prior.

## Layout

The experiment is a package, `experiments.single_site_mcmc_vs_eki`, run from
the repository root with `python -m`. Its parts depend one way:

| Part | What it holds | Imports |
|---|---|---|
| `config.py` | every choice, in sections: the site, paths, the drivers, the model (SIPNET's process flags), running SIPNET (timeout, staging, workers), the observations (NEE windows, pool constraints, sources left out, operators), the prior predictive, and the noise model (floors, discrepancy terms, timescales) | the library, `model/operators.py` |
| `model/` | the calibration's definitions: nothing here writes a file or has a `main` | `config`, the library |
| `scripts/` | the entry points, which run the model and write under `output/` | `model`, `config` |
| `figures/` | drawing what the scripts wrote; nothing here runs a model | `model`, `config` |
| `exploration/` | tools and records that found the model, not needed to reproduce it | anything; nothing imports it |

| File | What it does |
|---|---|
| `model/inputs.py` | one loader per data source, restricted to the site |
| `model/operators.py` | the two observation operators the library does not have: a mean rate over each window (NEE), and dry aboveground biomass from wood carbon (LandTrendr) |
| `model/observations.py` | the observed values of every observation source, prepared from `inputs` as `config` says, and the calibration and validation observation vectors |
| `model/noise.py` | the noise covariance $R$ and the Gaussian likelihood it defines, for both vectors |
| `model/sipnet.py` | what running SIPNET needs, built from `config`: the base parameters, the runner and model, the drivers, the site's initial state, and the forward model over PyEns |
| `model/prior.py` | **the calibration's parameterization and prior**: the parameter vector, the prior and the SIPNET parameter map, with every prior term's provenance |
| `model/fixed_sipnet_parameters.csv` | every SIPNET parameter the calibration does not calibrate: its value and justification |
| `scripts/prepare_drivers.py` | writes the site's driver file, corrected for four known defects of the ERA5 driver files, to `output/drivers/`, which the runs read; its docstring says what each defect is and how it is corrected |
| `scripts/describe.py` | prints what the inputs, the observation vectors and the noise model hold; the check that everything is found and builds |
| `scripts/prior_predictive.py` | one SIPNET run by hand at the prior mean and an ensemble of prior draws through the forward model, predicting both observation vectors, scored under the likelihood; writes to `output/prior_predictive/` |
| `scripts/provenance.py` | the `provenance.json` each script writes beside its outputs |
| `figures/common.py` | reading the scripts' outputs, and the shared legend |
| `figures/prior_predictive.py` | the prior predictive's figures: NEE windows, pool constraints and daily trajectories, and the slide figures (prior marginals, NEE's seasonal cycle, annual NEE against both towers, coverage per source); writes to `output/figures/` |
| `exploration/parameter_analysis/` | the evidence for the prior and the fixed values: the parameter-structure analysis, the base set, the sensitivity screening and the prior-predictive checks |
| `exploration/fast_forward.py` | a fast forward path for exploration, SIPNET in a process pool with the predictions by index arithmetic; it equals the library's to 1e-11, and is not the calibration's forward model |
| `output/` | everything the experiment writes; untracked |

## Running it

From the repository root, in an environment synced as `CLAUDE.md` describes,
with the site's driver directories under `data/raw/drivers/` and the
processed files built (`scripts/ingest_*.py`):

```bash
uv run python -m experiments.single_site_mcmc_vs_eki.scripts.prepare_drivers
uv run python -m experiments.single_site_mcmc_vs_eki.scripts.describe
uv run python -m experiments.single_site_mcmc_vs_eki.scripts.prior_predictive   # about an hour; --ensemble-size for fewer draws
uv run python -m experiments.single_site_mcmc_vs_eki.figures.prior_predictive
```

In a notebook started from the repository root, import the parts the same
way: `from experiments.single_site_mcmc_vs_eki import config` and
`from experiments.single_site_mcmc_vs_eki.model import observations, prior`.

## The record of a run

The code will move on after a result is reported, so a result is recorded
rather than the code frozen. Each script writes `provenance.json` beside its
outputs (`scripts/provenance.py`): the repository commit and whether the tree
was dirty, each companion package's installed commit, the SIPNET pin and
binary, the command, every `config` constant, and the path, size and MD5 of
every input file. A result worth keeping is then the commit it names, tagged,
with its `output/` directory archived beside the tag.

## Decisions

Each records what was chosen and why, so a result can be read against it.

- **The site** is 4977, Harvard Forest, where two AmeriFlux towers stand:
  US-Ha1 (hourly) and the NEON tower US-xHA (half-hourly).
- **One driver member**, so that the forward model is deterministic, as the
  MCMC comparison needs.
- **The driver file is corrected before any run**, so that every row means
  what SIPNET's format says: labeled with the UTC start of its three-hour
  step, and every value a total or a mean over that step. The four
  corrections, in `scripts/prepare_drivers.py`: the drifting hour labels are rebuilt
  from position (`data/README.md` Note 15); every label moves back three hours,
  since radiation and precipitation cover the three hours *ending* at the
  label (Note 16); the snapshot columns (temperature, humidity, wind) become
  the mean of their two edge values (Note 16); and soil temperature, which
  PEcAn built with a filter that averages the *following* weeks of air
  temperature, is recomputed with the same filter run forward in time. The
  model's time axis is then UTC, the observed NEE's clock.
- **NEE: US-Ha1 to calibrate, US-xHA to check.** US-Ha1 covers nine years of
  the drivers' record, US-xHA six, so the longer record calibrates. US-xHA's
  years after US-Ha1's record ends are then an out-of-sample check, by a tower
  whose footprint the calibration never saw; how much the two towers disagree
  where they overlap is a measure of how well one tower stands for the site.
- **NEE: the u\* threshold estimated per year** (`NEE_VUT_REF`), over one
  threshold for the whole record. The two differ little at this site; the
  per-year series exists at more of the pool's sites, so the choice carries
  over to a multi-site calibration.
- **SMAP soil moisture is left out.** The reason is
  `config.EXCLUDED_CONSTRAINTS`: SMAP L4's total-profile volumetric water
  content has no defensible counterpart in SIPNET's bucket, and the
  reanalysis's comparison, with the bucket's fraction of capacity, compares a
  volumetric content with a degree of saturation.
- **NEE is averaged over two twelve-hour windows a day**, 00-12 and 12-24 UTC
  (night- and day-centered at Harvard Forest), each an observation only when
  at least half its values were measured rather than gap-filled. A whole day
  would need both halves measured, which keeps almost no summer days, since
  calm summer nights fail the u\* filter, and biases the kept days toward
  less uptake. The model is reduced over the same windows, whose edges are
  its timestep edges, as a mean rate.
- **LandTrendr is dry biomass, 2012-2017 only.** Its spec records carbon, but
  at the site it is about twice the initial conditions' aboveground carbon
  map, so the model's wood carbon is divided by a carbon fraction of 0.48.
  From 2018 its uncertainties come from a separate model and change with its
  values (`data/README.md` open question 19).
- **GEDI is left out**: its units are not established, and its three values
  at the site disagree with each other and with LandTrendr
  (`config.EXCLUDED_CONSTRAINTS`).
