# AIMES talk: calibrating land surface models as maps

An eight-minute talk for a land data assimilation audience (ecologists, not
statisticians): why calibrate parameters spatially rather than site by site,
what a spatial prior looks like, and a toy experiment showing what it buys
and that it can be computed with few model runs and no gradients.

## Layout

The package is organized by role, so a choice has one home and a figure can
be traced back to every assumption behind it.

| Path | Role |
|---|---|
| `config.py` | **every number**: sizes, seeds, model constants, prior, noise, algorithm settings |
| `toy/domain.py` | data setup: the sites, the map grid, the climate covariate, the drivers |
| `toy/forward_model.py` | the model: a one-pool carbon model standing in for SIPNET |
| `toy/spatial_prior.py` | the model: the prior over the parameter fields, and kriging to new locations |
| `toy/observations.py` | the model: what each site observes, the noise, synthetic data, the likelihood |
| `inference/site_by_site.py` | algorithm: each site alone, exact on a grid |
| `inference/spatial_mcmc.py` | algorithm: the exact spatial posterior by NUTS (the reference) |
| `inference/ipl.py` | algorithm: iterated posterior linearization (the scalable method) |
| `inference/moments.py`, `inference/evaluation.py` | the common form of a result; predictions and scores |
| `illustrations/` | pictures that are not experiments: the 1D transect, prior draws over North America |
| `results.py` | the schema of each results file, shared by writer and reader |
| `run/` | the scripts: each a short, high-level workflow over the pieces above |
| `plots.py`, `style.py` | the figures (reading results, computing nothing) and their brand style |
| `slides.qmd`, `slides.css`, `theme/` | the deck; `theme/` is the dissertation's brand and reveal.js theme |

Results go to `output/` (untracked); figures to `figures/`.

## Reproducing everything

From the repository root, in the worktree's environment:

```bash
uv run python -m experiments.presentation_aimes_talk.run.toy_experiment   # about 10 minutes (NUTS)
uv run python -m experiments.presentation_aimes_talk.run.illustrations    # seconds
uv run python -m experiments.presentation_aimes_talk.run.draw_figures
export QUARTO_PYTHON=$(git rev-parse --show-toplevel)/.venv/bin/python
quarto render experiments/presentation_aimes_talk/slides.qmd
```

The North America illustration needs the processed site table
(`uv run scripts/ingest_sites.py`, which runs from tracked inputs).

## The toy experiment

`run/toy_experiment.py` reads top to bottom as the workflow: build the
problem, draw a truth and data, calibrate three ways, predict and score.

**The problem.** A unit square with 200 training sites and 50 held-out sites,
scattered uniformly. A climate covariate `z(x)` (mean annual temperature,
warmer to the south) drives both the monthly drivers and the prior's trend.
At each site a one-pool carbon model runs monthly for five years:
photosynthesis `a L_t sigmoid((T_t - 5)/3)`, respiration of a fraction
`1 - exp(-k_t)` of the pool with `k_t = k exp(0.09 (T_t - 10))`. The two
calibrated parameters are `log a` (productivity) and `log k` (turnover).

**The prior.** Each parameter is a field: climate trend + a Matern-3/2
Gaussian process + independent site noise, the two parameters independent
(`toy/spatial_prior.py`). **Every prior number is known to every method**: no
hyperparameters are estimated, which a real calibration would have to do.

**The data.** Every training site observes its pool at the end of each year
(10% noise). 10% of training sites are towers, which also observe monthly NEE
(15% noise). The noise scale is set from the model at each site's prior mean.
The truth is one draw from the prior, so the prior is correct by
construction.

**The methods.**

- *Site by site*: each site with its own marginal prior (keeping the climate
  trend, which is generous), computed exactly on a refined 2D grid. At a site
  with no data it can say nothing beyond the prior.
- *Spatial, exact*: NUTS on all 400 site parameters jointly, in whitened
  coordinates. It differentiates the model, which SIPNET cannot be; it is the
  yardstick. Its cost is the number of leapfrog steps, each one model run
  (with gradient) at every site.
- *Spatial, scalable (IPL)*: per iteration, 30 runs at each site from the
  current approximation, a linear regression of outputs on parameters per
  site, and one Gaussian update over all sites with the spatial prior. It
  stops when no site's mean moves by more than 1% of a prior SD.

**What the figures show** (`run/draw_figures.py` lists each figure, the
results it reads and the function that draws it):

| Figure | Shows |
|---|---|
| `toy_decomposition` | the true turnover field split into the prior's three terms |
| `toy_world` | the true turnover map with the sites; a tower's NEE and a sparse site's pools |
| `equifinality_*` | one sparse site's posterior: alone (contours, 50% and 90%), then spatial (reference draws) |
| `maps_estimate_*` | truth, site-by-site posterior means, spatial posterior mean of the smooth field |
| `maps_uncertainty_turnover` | posterior SDs: site by site at each site, spatial everywhere |
| `held_out_errors` | RMSE at the held-out sites, per parameter and method |
| `cost_bars` | model runs per site: NUTS (warmup and sampling, all chains) against IPL |
| `ipl_agreement_turnover` | exact and IPL posterior mean maps |
| `ipl_convergence` | IPL's median distance from the exact posterior after each iteration |

**Choices made in the figures.** The zoomed-in sparse site
(`plots.showcase_sites`) is the 90th percentile of how much the spatial
posterior narrows turnover over site by site: a clear case, not the most
extreme one. The aggregate figures (maps, held-out errors) show every site.

**Caveats to keep in mind when quoting it.** One synthetic dataset, one seed;
the prior is the truth's own distribution with every hyperparameter known;
the model is far simpler than SIPNET. It shows the mechanism and the cost
scaling, not the size of the benefit on real data.

## The illustrations

- `transect`: a 1D Gaussian process, four direct noisy measurements of the
  parameter, exact posterior. A schematic of how information spreads.
- `north_america`: three prior draws of one generic parameter over North
  America, as continuous maps: a trend in standardized mean annual
  temperature (the site covariate table's `MAT`, carried to each grid point
  from its nearest site) plus a Matern-3/2 Gaussian process (800 km) of the
  chordal distance. The field is drawn exactly on a 60 km grid in the
  display projection and shown on a 15 km raster interpolated from it,
  within 90 km of a site (small enclosed gaps filled), which traces the land
  the 8000-site pool covers; the sites are drawn over it. The tower sites (for the site map) are the
  tower table's primary towers. The numbers are chosen to look plausible,
  not fitted.
