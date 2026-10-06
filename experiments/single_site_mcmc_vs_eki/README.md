# Single-site calibration: error models and algorithms

Calibrate SIPNET at one site, Harvard Forest by default, to observed NEE and
the pool constraints, under three NEE error models, each with fixed or
inferred noise, and compare what EKI, EKI with Gibbs noise updates,
importance sampling and SMC after EKI, and MCMC find. `config.py` holds every
setting, `models.py` defines the models, and `MODEL.md` states the
mathematics.

## The models and the algorithms

A **model** is `<nee error model>/<noise>`:

| NEE error model | memory of NEE's discrepancy |
|---|---|
| `short_memory` | errors forget within days |
| `long_memory` | plus season-long anomalies |
| `recurring_bias` | plus a bias recurring every year |

and `fixed` (each noise scale held at its prior median, 1) or `inferred`
(the scales of NEE by night, NEE by day and MODIS LAI unknown, inverse-gamma
priors). Every model shares the prior, the forward map and the other sources'
noise.

| Algorithm (`--algorithm`) | for | run directory |
|---|---|---|
| `eki` | fixed noise | `eki` |
| `eki_gibbs_common`, `eki_gibbs_per_particle` | inferred noise | the same |
| `is`, `smc` `--from <eki run>` | either | `<eki run>_is`, `<eki run>_smc` |
| `mcmc` `--from <is run>` | either | `mcmc` |

Every run writes under `output/runs/<nee error model>/<noise>/<run>/`, in one
format (`algorithms/records.py`): `samples.nc`, `natural_values.csv`,
`cost.json` (SIPNET runs, forward-map calls and wall time per phase, a
seeded run's seed phases included), `history.csv` and `provenance.json`;
`run.predict` adds `predictive/`.

## Layout

| Part | What it holds |
|---|---|
| `config.py` | every setting: the site, paths, drivers, SIPNET, observations, the noise model's constants, and each algorithm's settings |
| `models.py` | the models: `Model`, `MODEL_NAMES`, and the posteriors every algorithm reads (`fixed_posterior`, `heldout_posterior`, `marginal_posterior`) |
| `model/` | the calibration's definitions: `prior.py` (the prior and the SIPNET parameter map), `nee_error.py` (the NEE error models), `noise.py` (the noise factors and scales), `likelihood.py` (the likelihood from predictions, fixed or with the scales integrated out, and the scales' conditionals), `observations.py`, `operators.py`, `sipnet.py`, `inputs.py` |
| `algorithms/` | `eki.py`, `eki_gibbs.py`, `reweighting.py` (IS and SMC), `mcmc.py`, and `records.py` (the run format and its cost) |
| `run/` | the entry points: `calibrate`, `predict`, `diagnose`, `compare`, `fit_nee_discrepancy`, `prepare_drivers`, `check_inputs`, `prior_predictive` |
| `analysis/` | `compare.py`: the cross-run tables (cost, parameters, agreement with a reference run, noise scales, reweighting, held-out scores) |
| `scc/` | the SCC jobs: `submit_all.sh`, `model_job.sh`, `mcmc_job.sh`, `smoke_job.sh`, `env.sh` |
| `tests/` | the numerical pieces on small problems, no SIPNET |
| `model/diagnostics.py`, `model/outputs.py` | a run's diagnostics from its predictive, and reading what the runs wrote |
| `figures/`, `report/`, `exploration/` | figures (the prior predictive's and a run's diagnostics), the report, and the records that found the model |

## Running it

From the repository root, in an environment synced as `CLAUDE.md`
describes:

```bash
uv run python -m experiments.single_site_mcmc_vs_eki.run.prepare_drivers
uv run python -m experiments.single_site_mcmc_vs_eki.run.calibrate --model long_memory/fixed --algorithm eki
uv run python -m experiments.single_site_mcmc_vs_eki.run.calibrate --model long_memory/fixed --algorithm is --from eki
uv run python -m experiments.single_site_mcmc_vs_eki.run.predict --model long_memory/fixed --run eki_is
uv run python -m experiments.single_site_mcmc_vs_eki.run.diagnose --model long_memory/fixed --run eki_is
uv run python -m experiments.single_site_mcmc_vs_eki.run.compare
uv run pytest experiments/single_site_mcmc_vs_eki/tests
```

On the laptop a run uses 3 workers; `SIPNET_WORKERS` changes that, and
`HF_RUNS_DIRECTORY` moves the runs (a smoke test).

## Running on the SCC

Every write stays under `/projectnb/dietzelab/arober`: `scc/env.sh` points
the caches, temporary files and the storage-backed data
(`SIPNET_CALIBRATION_DATA=/projectnb/dietzelab/arober/hf_data`) there, and
the job logs go to `/projectnb/dietzelab/arober/hf_runs/logs`. Each job runs
on one node, its SIPNET runs on `$NSLOTS` local workers.

```bash
bash experiments/single_site_mcmc_vs_eki/scc/submit_all.sh
```

submits, per model, a `seed` job (EKI and importance sampling), then a
`rest` job (SMC and the predictives) and, for `long_memory`, an MCMC job,
both held on `seed`. MCMC checkpoints every 25 steps and resubmitting it
continues it. `scc/smoke_job.sh` checks the setup first in about 15 minutes.

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
- **NEE's error model is a choice the experiment varies, not a fit it
  adopts.** Three earlier EKI setups showed that the memory of NEE's
  discrepancy decides which features of daytime NEE the calibration
  reproduces (`MODEL.md`, "The NEE error model"). The experiment therefore
  compares three error models of the same per-window size and different
  memory (`model/nee_error.py`), each with its noise scale fixed at the
  prior median or inferred.
