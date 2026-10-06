# Single-site calibration: MCMC against EKI

Calibrate SIPNET at one site, Harvard Forest by default, to observed NEE and
the pool constraints, and compare the posterior an MCMC sampler finds with the
one EKI finds. `config.py` is the one source of truth; nothing else here makes
a choice. `MODEL.md` states the model exactly: the observation operators, the
noise covariance, the parameterization and prior, the diagnostics every run
is checked by, NEE's error, and what the EKI setups show about the error
model and a trade-off within SIPNET.

## Pipeline

Every entry point is a module of `run/`, run from the repository root as
`uv run python -m experiments.single_site_mcmc_vs_eki.run.<module>`. In the
order a calibration runs them:

| Stage | `run.<module>` | Writes | Then |
|---|---|---|---|
| setup | `prepare_drivers` | the site's corrected driver file, `output/drivers/` | |
| setup | `check_inputs` | nothing: prints what the inputs, observation vectors and noise model hold | |
| 1 | `prior_predictive` | `output/prior_predictive/` | draws its figures, and diagnoses it |
| 2 | `eki --data synthetic` or `--data observed` | `output/eki/<setup>/<data>/` | draws its ladder and marginals (and, on synthetic data, the recovery of the truth) |
| 3 | `posterior_predictive --data <data>` | `output/eki/<setup>/<data>/posterior_predictive/` | draws its figures, and diagnoses the run |
| 4 | `compare_setups` | `output/figures/comparison/` | |

`<setup>` is `config.EKI_RUN_NAME`. A diagnosis is the same for every run, so
runs compare directly: the tables under the run's `diagnostics/` and their
figures, as `MODEL.md`, "Diagnostics", defines them. `--no-diagnose` skips it.

For a run already written:

| `run.<module>` | What it does |
|---|---|
| `diagnose --run <run>` | rediagnoses it, under `config`'s current noise model; `<run>` is `prior`, `synthetic` or `observed` |
| `draw_figures --run <run>` | redraws its figures, after a change to `figures/` |
| `fit_nee_discrepancy --data <data>` | fits NEE's discrepancy to an EKI run's residuals; adopting a fit is copying it into `config` |

## Layout

The experiment is a package, `experiments.single_site_mcmc_vs_eki`. Its
parts depend one way:

| Part | What it holds | Imports |
|---|---|---|
| `config.py` | every choice, in sections: the site, paths, the drivers, the model (SIPNET's process flags), running SIPNET (timeout, staging, workers), the observations (NEE windows, pool constraints, sources left out, operators), the prior predictive, the noise model (floors, NEE's discrepancy, the other discrepancy terms), and EKI (the setup's name, ensemble, seeds, ladder) | the library, `model/operators.py`, `model/discrepancy.py` |
| `model/` | the calibration's definitions: nothing here writes a file or has a `main` | `config`, the library |
| `run/` | the entry points, and nothing else: each public module has a `main`, and the underscore modules are the machinery they share | `model`, `figures`, `config` |
| `figures/` | the figures, as functions of what the runs wrote; the runs call them, and nothing here runs a model or has a `main` | `model`, `config` |
| `report/` | `report.qmd`, a Quarto document walking through the setup and the results, reading the runs' outputs | `model`, `figures`, `config`; nothing imports it |
| `exploration/` | tools and records that found the model, not needed to reproduce it | anything; nothing imports it |

| File | What it does |
|---|---|
| `model/inputs.py` | one loader per data source, restricted to the site |
| `model/operators.py` | the two observation operators the library does not have: a mean rate over each window (NEE), and dry aboveground biomass from wood carbon (LandTrendr) |
| `model/observations.py` | every observation source, its observed values and measurement standard deviations prepared from `inputs` as `config` says, and the calibration and validation observation vectors |
| `model/discrepancy.py` | the form of NEE's model discrepancy: a short, a long and a recurring term; `config.NEE_DISCREPANCY` holds its values |
| `model/noise.py` | the noise model: one Gaussian noise factor per source, its covariance a measurement term plus a discrepancy term, as covariance specs; a posterior's $R$ blocks, and their summary |
| `model/sipnet.py` | what running SIPNET needs, built from `config`: the base parameters, the runner and model, the drivers, the site's initial state, and the SIPNET runs over PyEns |
| `model/prior.py` | **the calibration's parameterization and prior**: one prior factor per parameter, with its provenance, and the SIPNET parameter map; the initial states nothing calibrates, as inputs; the site's `SiteDims`; the prior's center in theta; theta's natural values as a table |
| `model/calibration.py` | **the model, assembled in one place**: the prior factors, the forward map as a `SIPNETSimulator` and the noise factors, joined, bound at the site and conditioned on the observations, as the posterior every algorithm reads; the calibration and validation posteriors |
| `model/outputs.py` | reading what the runs wrote: a predictive, an EKI run, a run's diagnostics |
| `model/diagnostics.py` | the diagnostics of a run, as `MODEL.md`, "Diagnostics", defines them: the posterior predictive check, NEE's residuals (size, recurring seasonal part, autocorrelation against $R$'s, slow and fast parts, night-day correlation), and the two towers |
| `model/fixed_sipnet_parameters.csv` | every SIPNET parameter the calibration does not calibrate: its value and justification |
| `run/prepare_drivers.py` | the site's driver file, corrected for four known defects of the ERA5 driver files; its docstring says what each defect is and how it is corrected |
| `run/check_inputs.py` | prints what the inputs, the observation vectors and the noise model hold; the check that everything is found and builds |
| `run/prior_predictive.py` | one SIPNET run by hand at the prior's center and an ensemble of prior draws |
| `run/eki.py` | EKI in its sampling form, on observed or synthetic data: EnsKit's driver on `inference.eki_problem(posterior)`: a prior ensemble moved up an adaptive tempering ladder to beta = 1 by the perturbed-observation update, every step checkpointed and resumable (`--resume`) |
| `run/posterior_predictive.py` | an EKI run's final ensemble through the same predictive |
| `run/compare_setups.py` | the EKI setups' figures side by side |
| `run/diagnose.py` | a run's diagnosis: its tables under `diagnostics/`, and their figures |
| `run/draw_figures.py` | a stored run's figures, redrawn |
| `run/fit_nee_discrepancy.py` | fits NEE's discrepancy to an EKI run's residuals by maximum marginal likelihood, several variants per source, checked against the towers and the held-out tower; writes `nee_discrepancy_fit.csv` beside the run |
| `run/_predictive.py` | what the prior and posterior predictives share: an ensemble (and optionally one run by hand) evaluated under the calibration and validation posteriors and run once more for daily output, written in one layout |
| `run/_provenance.py` | the `provenance.json` each run writes beside its outputs, and the calibration's record: `calibration_parts.csv`, `calibration_components.csv` and `calibration_sipnet_parameters.csv` |
| `figures/common.py` | panel titles, the shared legend, and saving a figure |
| `figures/prior_predictive.py` | the prior predictive's figures: NEE windows, pool constraints and daily trajectories, and the slide figures (prior marginals, NEE's seasonal cycle, annual NEE against both towers, coverage per source); into `output/figures/` |
| `figures/eki.py` | an EKI run's figures: the ladder, prior against posterior marginals, on synthetic data recovery of the truth, and the posterior predictive's figures; into `output/figures/<setup>/` |
| `figures/diagnostics.py` | a run's diagnostic figures, for slides: the predictive check per source, the weekly residuals and their recurring part, the residuals' autocorrelation against $R$'s, and the two towers |
| `figures/comparison.py` | the EKI setups compared, for slides: each setup's posterior predictive seasonal cycle, the parameters the error model moves, and the daytime residual's slow and fast parts; into `output/figures/comparison/` |
| `exploration/parameter_analysis/` | the evidence for the prior and the fixed values: the parameter-structure analysis, the base set, the sensitivity screening and the prior-predictive checks |
| `exploration/fast_forward.py` | a fast forward path for exploration, SIPNET in a process pool with the predictions by index arithmetic; it equals the library's to 1e-11, and is not the calibration's forward map |
| `output/` | everything the experiment writes; untracked |

## Running it

From the repository root, in an environment synced as `CLAUDE.md` describes,
with the site's driver directories under `data/raw/drivers/` and the
processed files built (`scripts/ingest_*.py`):

```bash
uv run python -m experiments.single_site_mcmc_vs_eki.run.prepare_drivers
uv run python -m experiments.single_site_mcmc_vs_eki.run.check_inputs
uv run python -m experiments.single_site_mcmc_vs_eki.run.prior_predictive   # about 35 minutes on the default 3 workers; --ensemble-size for fewer draws
uv run python -m experiments.single_site_mcmc_vs_eki.run.eki --data synthetic   # --resume continues an interrupted run
uv run python -m experiments.single_site_mcmc_vs_eki.run.posterior_predictive --data synthetic
uv run python -m experiments.single_site_mcmc_vs_eki.run.eki --data observed
uv run python -m experiments.single_site_mcmc_vs_eki.run.posterior_predictive --data observed
uv run python -m experiments.single_site_mcmc_vs_eki.run.compare_setups
```

To refit NEE's discrepancy to an EKI run's residuals, once its posterior
predictive has run (the held-out scores and the tower floor read it and its
diagnosis):

```bash
uv run python -m experiments.single_site_mcmc_vs_eki.run.fit_nee_discrepancy --data observed
```

**EKI setups.** Each EKI setup, a noise model or an algorithm, has a name,
`config.EKI_RUN_NAME`, and its runs and figures are kept under it
(`output/eki/<setup>/`, `output/figures/<setup>/`), so a new setup leaves
the earlier runs in place. A run is diagnosed while `config` holds the setup
it ran with, since the diagnostics score it under `config`'s $R$; the
diagnosis says when the run's `provenance.json` records other noise
settings. The first calibration's setup is `single_term_discrepancy`, the
second `three_term_discrepancy`, and the current one `two_term_discrepancy`.

To render the report, with Quarto pointed at the worktree's interpreter:

```bash
export QUARTO_PYTHON=$(git rev-parse --show-toplevel)/.venv/bin/python
```

```bash
quarto preview experiments/single_site_mcmc_vs_eki/report/report.qmd
```

In a notebook started from the repository root, import the parts the same
way: `from experiments.single_site_mcmc_vs_eki import config` and
`from experiments.single_site_mcmc_vs_eki.model import calibration`, whose
`calibration_posterior()` is the whole model.

## The record of a run

The code will move on after a result is reported, so a result is recorded
rather than the code frozen. Each run writes `provenance.json` beside its
outputs (`run/_provenance.py`): the repository commit and whether the tree
was dirty, each companion package's installed commit, the SIPNET pin and
binary, the command, every `config` constant, and the path, size and MD5 of
every input file. Beside it, a run of the calibration writes the
calibration's own record: one table per part of the model with its law and
provenance, one per component with its role in the posterior, and one per
SIPNET parameter with its rule or fixed value. A result worth keeping is then
the commit it names, tagged,
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
  corrections, in `run/prepare_drivers.py`: the drifting hour labels are rebuilt
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
- **NEE's discrepancy has two terms, fitted to the first calibration.**
  The first calibration, with one exponential term of two days, failed its
  posterior predictive check by about 30 standard deviations, and its NEE
  residuals showed a bias recurring every year and a memory of weeks
  (`MODEL.md`, "NEE error"). A three-term discrepancy with a term recurring
  every year passed the check but let the calibration give up summer
  daytime uptake as a shared seasonal bias, so the recurring term was
  dropped. The discrepancy is a short and a long term per NEE source, fitted
  by maximum marginal likelihood to the first calibration's residuals and
  checked against the two towers' floor and the held-out tower. Together the
  three setups show that the error model chooses between two fits SIPNET
  cannot make at once, the summer uptake and the day-to-day variation of
  daytime NEE, so $R$ is part of the calibration's specification and is not
  refitted to a calibration's residuals (`MODEL.md`, "The error model and
  the fast-slow trade-off").
