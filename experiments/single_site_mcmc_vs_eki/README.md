# Single-site calibration: MCMC against EKI

Calibrate SIPNET at one site, Harvard Forest by default, to observed NEE and
the pool constraints, and compare the posterior an MCMC sampler finds with the
one EKI finds. `config.py` is the one source of truth; nothing else here makes
a choice.

## Files

| File | What it does |
|---|---|
| `config.py` | the site, the driver member and its clock, the data sources read and left out, where the experiment writes |
| `prepare_drivers.py` | writes the site's driver file with regular hour labels to `output/drivers/`, which the runs read |
| `inputs.py` | one loader per data source, restricted to the site; run it to check that every input is found |
| `output/` | everything the experiment writes; untracked |

## Running it

From the repository root, in an environment synced as `CLAUDE.md` describes,
with the site's driver directories under `data/raw/drivers/` and the
processed files built (`scripts/ingest_*.py`):

```bash
uv run python experiments/single_site_mcmc_vs_eki/prepare_drivers.py
uv run python experiments/single_site_mcmc_vs_eki/inputs.py
```

## Decisions

Each records what was chosen and why, so a result can be read against it.

- **The site** is 4977, Harvard Forest, where two AmeriFlux towers stand:
  US-Ha1 (hourly) and the NEON tower US-xHA (half-hourly).
- **One driver member**, so that the forward model is deterministic, as the
  MCMC comparison needs. The drivers' hour labels are regularized, and their
  clock is declared as start-of-step on UTC+03:00 (`data/README.md` Notes 15
  and 16; `config.DRIVER_TIME_ZONE`).
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
