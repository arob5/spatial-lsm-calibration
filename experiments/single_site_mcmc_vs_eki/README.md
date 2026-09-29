# Single-site calibration: MCMC against EKI

Calibrate SIPNET at one site, Harvard Forest by default, to observed NEE and
the pool constraints, and compare the posterior an MCMC sampler finds with the
one EKI finds. `config.py` is the one source of truth; nothing else here makes
a choice.

## Files

| File | What it does |
|---|---|
| `config.py` | every choice, in sections: the site, paths, the drivers, the model (SIPNET's process flags), running SIPNET (timeout, staging, workers), the observations (NEE windows, pool constraints, sources left out, operators), and the noise model (floors, discrepancy terms, timescales) |
| `prepare_drivers.py` | writes the site's driver file, corrected for four known defects of the ERA5 driver files, to `output/drivers/`, which the runs read; its docstring says what each defect is and how it is corrected |
| `inputs.py` | one loader per data source, restricted to the site; run it to check that every input is found |
| `operators.py` | the two observation operators the library does not have: a mean rate over each window (NEE), and dry aboveground biomass from wood carbon (LandTrendr) |
| `observations.py` | the observed values of every observation source, prepared from `inputs` as `config` says, and the calibration and validation observation vectors; run it to see what each holds |
| `noise.py` | the noise covariance $R$ and the Gaussian likelihood it defines, for both vectors; run it to see what each observation source contributes |
| `runs.py` | what running SIPNET needs, built from `config`: the base parameters, the runner and model, the drivers, the site's initial state, and the forward model over PyEns |
| `stand_in_calibration.py` | step 3's stand-in parameter vector, prior and SIPNET parameter map: three parameters, enough to run the machinery; **not** the calibration, which step 4 writes |
| `forward_check.py` | one SIPNET run by hand and an ensemble of prior draws through the forward model, predicting both observation vectors, scored under the likelihood; the calibration's prior by default, or step 3's stand-in; writes to `output/forward_check/<calibration>/` |
| `plots.py` | the figures, from what the scripts wrote; writes to `output/figures/` |
| `calibration_prior.py` | **the calibration's parameterization and prior** (step 4): the parameter vector, the prior and the SIPNET parameter map, with every prior term's provenance |
| `fixed_sipnet_parameters.csv` | every SIPNET parameter the calibration does not calibrate: its value and justification |
| `parameter_analysis/` | the evidence for both: the parameter-structure analysis, the base set, the sensitivity screening and the prior-predictive checks |
| `fast_forward.py` | a fast forward path for exploration, SIPNET in a process pool with the predictions by index arithmetic; it equals the library's to 1e-11, and is not the calibration's forward model |
| `output/` | everything the experiment writes; untracked |

## Running it

From the repository root, in an environment synced as `CLAUDE.md` describes,
with the site's driver directories under `data/raw/drivers/` and the
processed files built (`scripts/ingest_*.py`):

```bash
uv run python experiments/single_site_mcmc_vs_eki/prepare_drivers.py
uv run python experiments/single_site_mcmc_vs_eki/inputs.py
uv run python experiments/single_site_mcmc_vs_eki/observations.py
uv run python experiments/single_site_mcmc_vs_eki/noise.py
uv run python experiments/single_site_mcmc_vs_eki/forward_check.py   # a few minutes; --calibration stand_in for step 3's
uv run python experiments/single_site_mcmc_vs_eki/plots.py           # the same --calibration
```

## The observation model

The calibration conditions on a vector $y \in \mathbb{R}^N$ through

$$
y = \mathcal{H}\big(\mathcal{M}(\theta)\big) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, R),
$$

where $\mathcal{M}(\theta)$ is one SIPNET run over the prepared drivers at
parameters $\theta$, $\mathcal{H}$ stacks the observation operators of the
$K = 5$ observation sources below, and $R$ is the noise covariance. $y$, its
order (site-major, then source in the order below, then time) and the
operators are `observations.calibration_observation_vector()`; each operator
is bound to its source in `config.OBSERVATION_OPERATORS`. This section states
each source's data, its operator and its measurement error exactly, and then
the noise model assembling $R$, which `noise.py` builds.

### Notation

The model runs on three-hourly timesteps $s = 1, \dots, S$ covering the
intervals $(a_s, b_s]$ in UTC, of length $\Delta_s = b_s - a_s$ (in days), as
pySIPNET's `timestep_start` and `time` give them. The record is
$(a_1, b_S] = $ (2011-12-31 21:00, 2024-12-31 21:00]. SIPNET reports two kinds
of output variable:

- a **per-step total** $F_s$ over $(a_s, b_s]$, such as net ecosystem exchange
  $\mathrm{NEE}_s$ in g C m⁻² per step (`timestep_total`);
- a **state** $X_s$ at the end of the step $b_s$, such as a carbon pool in
  g C m⁻² (`timestep_end_state`).

A **window** is an interval $W = (w^-, w^+]$. The model steps a window reduces
over are those whose end falls inside it, $S(W) = \{s : b_s \in W\}$; every
window below has its edges on step edges, so these steps tile it exactly. The
step-length-weighted mean of a quantity $Z$ over a window is

$$
\langle Z \rangle_W = \frac{\sum_{s \in S(W)} Z_s \, \Delta_s}{\sum_{s \in S(W)} \Delta_s}.
$$

For an observation labeled at a time $t$, $s(t)$ is the step containing it,
$a_{s(t)} < t \le b_{s(t)}$.

### NEE: `nee_night_centered` and `nee_day_centered`

**Data.** The hourly series `ameriflux_nee_hourly_ustar_variable` of US-Ha1
(AmeriFlux FLUXNET FULLSET `NEE_VUT_REF`, the gap-filled NEE with the u\*
threshold estimated per year), in µmol CO₂ m⁻² s⁻¹, positive to the
atmosphere. Hour $h$ is the mean rate $\mathrm{NEE}^{\mathrm{obs}}_h$ over
$(h - 1\,\mathrm{h}, h]$ UTC, with quality flag $q_h$ (`NEE_VUT_REF_QC`: 0
measured, 1-3 gap-filled by marginal distribution sampling of decreasing
quality).

**Windows.** Each UTC day $d$ of the calibration years 2012-2020 gives two
windows, $W^{\mathrm{night}}_d = (d, d + 12\,\mathrm{h}]$ and
$W^{\mathrm{day}}_d = (d + 12\,\mathrm{h}, d + 24\,\mathrm{h}]$
(`config.NEE_WINDOWS`). Each holds $n_W = 12$ hours
$H(W) = \{h : (h - 1\,\mathrm{h}, h] \subset W\}$.

**Observation.** Window $W$ is an observation if and only if at least half its
hours were measured (`config.NEE_MINIMUM_MEASURED_FRACTION`),

$$
\frac{1}{n_W} \sum_{h \in H(W)} \mathbf{1}[q_h = 0] \;\ge\; 0.5,
$$

and its value is the mean of all its hours, measured and gap-filled,

$$
y_W = \frac{1}{n_W} \sum_{h \in H(W)} \mathrm{NEE}^{\mathrm{obs}}_h .
$$

It is labeled with the window's end $w^+$ and carries the window.

**Operator** (`operators.AverageRateOverWindows("net_ecosystem_exchange")`).
The per-step total divided by the step length is a rate in g C m⁻² d⁻¹,
averaged over the window and converted:

$$
\mathcal{H}_W(\mathcal{M}) = c \left\langle \frac{\mathrm{NEE}}{\Delta} \right\rangle_W
= c \, \frac{\sum_{s \in S(W)} \mathrm{NEE}_s}{w^+ - w^-},
\qquad
c = \frac{10^6}{M_{\mathrm{C}} \cdot 86400} \approx 0.96362,
$$

with $M_{\mathrm{C}} = 12.011$ g mol⁻¹, turning g C m⁻² d⁻¹ into
µmol CO₂ m⁻² s⁻¹ (one mole of CO₂ per mole of C). The conversion is the
observation vector's, through `pysipnet.units`. Here $S(W)$ is four steps.

### MODIS leaf area index: `modis_leaf_area_index`

**Data.** MCD15A3H v061 4-day composites at the site, June-August only, with
the product's flagged retrievals dropped at ingest (`sd > 20`, which is
exactly the `qc == "001"` flag). Composite $i$ has leaf area index $y_i$
(m² m⁻²), standard deviation $\sigma_i$, and a date label $t_i$ at 00:00 UTC;
whether the label is the composite's first day is unconfirmed
(`data/README.md` open question 23).

**Observation.** $y_i$ for every composite with $a_1 < t_i \le b_S$, which
drops the 2011 composites.

**Operator** (the library's `ComputeLeafAreaIndex`, SIPNET's own
`plantLeafC / leafCSpWt`). Leaf carbon at the end of the step containing the
label, over the run's leaf carbon per area $\lambda$ (SIPNET's `leafCSpWt`,
g C per m² of leaf, a parameter):

$$
\mathcal{H}_i(\mathcal{M}) = \frac{C^{\mathrm{leaf}}_{s(t_i)}}{\lambda}.
$$

A label at 00:00 falls in the step ending then, so this is the leaf carbon at
the label itself.

### LandTrendr biomass: `landtrendr_aboveground_biomass`

**Data.** LandTrendr's annual aboveground biomass at the site, `agb_mean` in
Mg ha⁻¹, with its `agb_sd`, for the years 2012-2017
(`config.LANDTRENDR_YEARS`). Year $j$ carries the window
$W_j = (\text{1 January } j, \text{1 January } j+1]$.

**Observation.** $y_j$, taken as **dry biomass**, not carbon (its processed
file records carbon; the reasons are under Decisions).

**Operator** (`operators.ComputeAbovegroundBiomass(0.48)`). Wood carbon, a
state, averaged over the year and divided by the carbon fraction of dry wood
$f_{\mathrm{C}} = 0.48$ (`config.WOOD_CARBON_FRACTION`), converted from
g m⁻² to Mg ha⁻¹:

$$
\mathcal{H}_j(\mathcal{M}) = 10^{-2} \, \frac{\langle C^{\mathrm{wood}} \rangle_{W_j}}{f_{\mathrm{C}}}.
$$

$C^{\mathrm{wood}}$ is SIPNET's `plantWoodC`, its aboveground wood; coarse and
fine roots are pools of their own, and leaves are not counted.

### SoilGrids soil carbon: `soilgrids_soil_organic_carbon`

**Data.** One static value $y$, the SoilGrids soil organic carbon stock over
0-200 cm in Mg C ha⁻¹, with its `sd`.

**Operator** (the library's `ReduceOverRun("soil_carbon", how="mean")`). The
soil carbon pool averaged over the whole run, converted from g C m⁻² to
Mg C ha⁻¹:

$$
\mathcal{H}(\mathcal{M}) = 10^{-2} \, \langle C^{\mathrm{soil}} \rangle_{(a_1, b_S]} .
$$

With the litter pool off (`config.MODEL_FLAGS`), litter goes straight into
$C^{\mathrm{soil}}$.

### Measurement error

What each source's data say about their own error, and nothing more. Each is
a standard deviation per observation; how they are combined, correlated and
added to is the noise model below.

- **NEE.** Two hourly columns of the FULLSET file, both in µmol m⁻² s⁻¹:
  the random uncertainty $r_h$ (`NEE_VUT_REF_RANDUNC`, estimated from the
  measured data) and the joint uncertainty $j_h$ (`NEE_VUT_REF_JOINTUNC`),
  defined by ONEFlux as $j_h^2 = r_h^2 + \big((\mathrm{NEE}_{84,h} -
  \mathrm{NEE}_{16,h}) / 2\big)^2$, the second term the spread of NEE across
  the ensemble of u\* thresholds. Their difference is the u\*-filtering part,
  $u_h = \sqrt{\max(j_h^2 - r_h^2, 0)}$. For a window,

  $$
  \big(\sigma^{\mathrm{obs}}_W\big)^2 = \frac{1}{n_W} \, \overline{r^2}_W + \big(\bar{u}_W\big)^2,
  $$

  where $\overline{r^2}_W$ and $\bar u_W$ are the means over the hours of $W$
  that report them. The random part is independent from hour to hour, so it
  averages down with $n_W$; the u\* part shifts every hour of a window
  together, so it does not. Where some hours report no uncertainty, the mean
  over those that do stands for all twelve. Where none does, as for whole
  days in which the file carries neither column (about 2% of the windows,
  most in November-December 2020), the window takes the median
  $\sigma^{\mathrm{obs}}_W$ of its source's other windows.
- **MODIS LAI.** $\sigma_i$ is the product's `LaiStdDev_500m` (times its 0.1
  scale factor). The retrieval finds every canopy its radiative transfer
  model accepts as consistent with the observed reflectances; $y_i$ is their
  mean LAI and $\sigma_i$ their standard deviation, which the MOD15 user guide
  (V6.1) calls "a measure of the solution accuracy". It is a spread among
  solutions, not a comparison with ground measurements. It can be exactly 0
  (two composites at the site, both of LAI 1.3), which the floor below
  replaces.
- **LandTrendr.** $\sigma_j$ is the raw file's `agb_sd`, which for 2012-2017
  `data/README.md` records as LandTrendr's own; what it quantifies is not
  documented in the file.
- **SoilGrids.** $\sigma$ is the raw file's `sd`, carried with the value
  through the assembly; how it was derived from SoilGrids is not recorded.

### Noise model

$R$ is fixed, not a function of $\theta$, so that MCMC and EKI target the same
posterior, and block-diagonal over the sources, no two sources' errors
correlated:

$$
R = \operatorname{diag}(R_1, \dots, R_K), \qquad R_k = \Sigma^{\mathrm{obs}}_k + \Sigma^{\delta}_k .
$$

$\Sigma^{\mathrm{obs}}_k$ is built from the measurement errors above;
$\Sigma^{\delta}_k$ is model discrepancy and representation error, how far a
point SIPNET can be from the observed quantity at the best $\theta$. The log
likelihood is

$$
\log p(y \mid \theta) = -\tfrac12 \big(y - \mathcal{H}(\mathcal{M}(\theta))\big)^{\!\top} R^{-1} \big(y - \mathcal{H}(\mathcal{M}(\theta))\big) - \tfrac12 \log\det R - \tfrac{N}{2} \log 2\pi,
$$

scored as `noise.calibration_likelihood().log_density(predictions)`, a
`pyeki.gauss.Gaussian` whose covariance is a `PSDBlockDiag` of one `DensePSD`
block per source, each factored once; a failed run's NaN row is the caller's
to score $-\infty$. The constants below are `config`'s "noise model"
section. The temporal correlation below is
$c_\tau(t, t') = \exp(-|t - t'| / \tau)$, with $t$ in days.

- **NEE**, for each of the two sources separately, the two independent:

  $$
  R_{WW'} = \big(\sigma^{\mathrm{obs}}_W\big)^2 \mathbf{1}[W = W'] + \sigma_\delta^2 \, c_\tau(w^+, w'^+),
  $$

  with $\sigma_\delta = 1.0$ µmol m⁻² s⁻¹ for `nee_night_centered` and $1.8$
  for `nee_day_centered`, and $\tau = 2$ days. $\sigma_\delta$ is 0.7 times the
  standard deviation of the observed windows' anomalies from their seasonal
  cycle (1.43 and 2.55), that is, SIPNET assumed to explain about half the
  day-to-day variance; $\tau$ gives a correlation of 0.61 between consecutive
  days, where the observed anomalies' is 0.45-0.50.
- **MODIS LAI**:

  $$
  R_{ii'} = \max(\sigma_i, 0.66)^2 \, \mathbf{1}[i = i'] + \sigma_\delta^2 \, c_{\tau}(t_i, t_{i'}) \, \mathbf{1}[\mathrm{year}(t_i) = \mathrm{year}(t_{i'})],
  $$

  with $\sigma_\delta = 0.5$ m² m⁻² and $\tau = 30$ days: correlated within a
  summer, independent across summers. The 0.66 floor is the reanalysis's
  (`data/README.md` open question 22).
- **LandTrendr**, with $\sigma = (\sigma_j)_j$ and $y = (y_j)_j$ as vectors:

  $$
  R = \sigma \sigma^{\top} + \kappa^2 \, y y^{\top} + \sigma_\delta^2 I,
  \qquad \kappa = \frac{0.02}{0.48},\ \sigma_\delta = 5\ \mathrm{Mg\,ha^{-1}}.
  $$

  The first term is LandTrendr's error, taken as shared by every year, since
  its trajectories are fitted per pixel; the second is the carbon fraction's
  uncertainty, the IPCC range 0.46-0.50 about 0.48, also shared; the third is
  independent year-to-year discrepancy.
- **SoilGrids**: $R = \sigma^2 + (0.25\, y)^2$, the second term for the depth
  and definition SIPNET's single soil pool does not share with a 0-200 cm
  stock (`data/README.md` open question 21).

## Parameterization and prior

`calibration_prior.calibration()` returns the three objects; this section says
how they were found. The records are in `parameter_analysis/`
(`phase1_report.md`, `phase2_report.md` and their tables). It is a
**starting point**: every prior is set from the literature, the traits or the
site's data, and none was tuned to the observations beyond checking that the
prior predictive covers them.

### Method

1. **The parameter structure, from the SIPNET source.** Every claim of the
   earlier structure analysis was checked against the pinned SIPNET
   (tag v2.2.0-alpha.1, commit 41fa853e), with line numbers
   (`phase1_report.md`). The photosynthesis parameters `aMax`, `aMaxFrac`,
   `baseFolRespFrac` and `cFracLeaf` enter only through
   $P = a_{\max}(f + r)/c$ and $a_{\max} r / c$ (`sipnet.c:614, 617, 633`):
   two exactly flat directions.
2. **A temperate deciduous base set.** Every SIPNET parameter got a value:
   the temperate deciduous BETY trait posterior's median, converted to SIPNET's
   units as PEcAn's `write.configs.SIPNET.R` converts it, where that is sound;
   else PEcAn's template; else a cited value (`base_parameter_candidates.csv`).
   Three BETY traits were rejected (below). With it, Harvard Forest is a
   modest sink, where the step-3 conifer stand-in made it a strong source.
3. **Screening.** Morris elementary effects over 29 candidates, 20
   trajectories (600 runs, none failed), on each observation source's misfit
   and on annual GPP, respiration, NEE, peak LAI and the change of wood and
   soil carbon (`morris_relative_mu_star.csv`).
4. **The parameterization.** The most influential parameters, reparameterized
   to remove the known ridges, and the initial wood and soil carbon, whose
   ensemble spread at the site is wide. Flat or redundant directions are fixed.
5. **Priors, then the prior predictive, iterated.** Each prior from its
   provenance; then 250 draws through the forward model, checked for coverage
   (the fraction of observations inside the 90% prior-predictive interval of
   prediction plus noise, per source and season) and for ecological
   plausibility (annual GPP, NEE, LAI, pool changes, respiration ratios). Five
   iterations, each change for a stated reason (`phase2_report.md`, section 4).

### The parameters

Thirteen parameters, $D = 15$ entries of $\theta$ (the allocation simplex is
three):

| Parameter | Prior | 5 / 50 / 95% | Reaches SIPNET as |
|---|---|---|---|
| photosynthetic capacity $P$ (nmol g⁻¹ s⁻¹) | log-normal, 95% in 140-450 | 161 / 252 / 402 | `aMax`, by `ComputePhotosynthesisRates` |
| respiration share $\rho$ | logit-normal, 95% in 0.04-0.20 | 0.05 / 0.09 / 0.18 | `baseFolRespFrac`, same rule |
| optimum photosynthesis temperature (°C) | normal(22, 2.5) | 17.6 / 21.8 / 26.4 | `psnTOpt`; `psnTMin` derived at a fixed range |
| half-saturation light (mol m⁻² d⁻¹) | log-normal, 4.6-26.3 | 5.0 / 10.1 / 21.4 | `halfSatPar` |
| soil water holding capacity (cm) | log-normal, 15-150 | 18 / 52 / 123 | `soilWHC` |
| leaf growth at leaf-on (g C m⁻²) | log-normal, 50-180 | 60 / 91 / 149 | `leafGrowth` |
| leaf-on growing degree-days | log-normal, 500-1100 | 552 / 741 / 1008 | `gddLeafOn` |
| allocation (leaf, wood, fine root, coarse root) | softmax-normal about (0.18, 0.45, 0.065, 0.305) | leaf 0.11-0.25, wood 0.34-0.57 | `leafAllocation`, `woodAllocation`, `fineRootAllocation` |
| wood respiration rate at 10 °C (yr⁻¹) | log-normal, 0.006-0.04 | 0.007 / 0.015 / 0.035 | `baseVegResp` and `baseCoarseRootResp`, each $r_{10}/Q_{10}$ |
| soil respiration flux at 10 °C, $F_{10}$ (g C m⁻² yr⁻¹) | log-normal, 200-900 | 230 / 423 / 855 | `baseSoilResp` $= F_{10} / (1000\, C_{s,0}\, Q_{10})$ |
| soil respiration $Q_{10}$ | log-normal, 1.3-3.2 | 1.37 / 2.08 / 2.91 | `soilRespQ10` |
| initial wood carbon (kg C m⁻²) | log-normal fitted to the site's initial-condition members | 2.9 / 7.3 / 23.5 | `plantWoodInit`, by `ComputeInitialConditions` |
| initial soil carbon $C_{s,0}$ (kg C m⁻²) | the same | 4.5 / 17.9 / 64 | `soilInit`, same rule |

"95% in a-b" is the interval the log-normal or logit-normal is fitted to.
Every term is independent: the ridges were removed by reparameterizing, and
there is no evidence for a correlation among the rest. Each term's full
provenance is in `calibration_prior.PROVENANCE` and `prior_proposal.csv`.

**The reparameterizations**, each against a ridge the analysis found:

- **$P$ and $\rho$** for the four photosynthesis parameters, with `aMaxFrac`
  and `cFracLeaf` fixed at the BETY medians: the model sees nothing else.
- **Respiration rates at 10 °C.** SIPNET's base rates are at 0 °C
  (`sipnet.c:1066-1067, 1073-1078`), so a base rate and its $Q_{10}$ trade
  against each other over the observed temperatures. A rate at 10 °C is
  nearly decorrelated from its $Q_{10}$. The coarse roots take the wood's rate,
  since they are wood.
- **The soil respiration flux $F_{10} = k_{10} C_{s,0}$**, not the rate
  $k_{10}$. Heterotrophic respiration depends on the product, so a rate prior
  independent of a wide initial-pool prior put 67% of the first iteration's
  runs at NEE > 0; the NEE data inform $F_{10}$, and the soil carbon
  observation informs $C_{s,0}$.
- **Allocation as a simplex**, which also rules out SIPNET's exit on
  fractions summing past one (`sipnet.c:1111-1123`).

**Fixed** (`fixed_sipnet_parameters.csv`): everything else. That includes
four parameters that scored high but are flat or redundant. Light
attenuation and leaf carbon per area enter the fluxes only as their ratio, and
BETY pins leaf carbon per area to ±12%. Water use efficiency and water removal
duplicate the soil water direction.

**Values that depart from PEcAn's**, with the reason:

- The BETY temperate deciduous optimum photosynthesis temperature, 43 °C
  (27.6-67.0), is implausible; the prior is centered on 22 °C.
- BETY's `Amax` converts to $P \approx 58$, which gives GPP near
  690 g C m⁻² yr⁻¹, half of Harvard's; the prior on $P$ is set from leaf
  physiology instead.
- PEcAn converts stem and root respiration to per-day rates, but SIPNET reads
  them as per-year and divides by 365 again (`write.configs.SIPNET.R:461-491`
  against `sipnet.c:1873, 1902`). The fine-root rate is fixed at the BETY
  median in SIPNET's units, 0.186 yr⁻¹, 365 times PEcAn's.
- BETY's leaf turnover, 0.75 yr⁻¹, counts leaf fall twice, since leaves fall
  at leaf-off by `fracLeafFall`; the template's 0.13 is used.

### The prior predictive

250 draws, none failed (`final_coverage.csv`, `final_plausibility.csv`):

| Observation source | Inside the 90% interval | Below | Above |
|---|---|---|---|
| NEE, night-centered | 0.86 | 0.02 | 0.12 |
| NEE, day-centered | 0.91 | 0.08 | 0.02 |
| MODIS LAI | 0.87 | 0.12 | 0.01 |
| LandTrendr biomass | 1.00 | 0 | 0 |
| SoilGrids soil carbon | 1.00 | 0 | 0 |

The median run: GPP 1451, NEE −203 g C m⁻² yr⁻¹, June-August LAI 4.7,
autotrophic respiration 0.59 of GPP. The tails are broad (GPP above 1800 in 26%
of runs, NEE above 0 in 34%), mostly from $P$, the half-saturation light and
the two initial-condition priors, which keep the ensemble's own spread.

**Structure no prior removes**, for the calibration and the model to answer:
summer daytime uptake is too weak at the median, and winter and spring night
respiration too low. SIPNET keeps allocating to leaves until leaf-off, so the
annual peak LAI exceeds 8 in 30% of runs. The fine-root pool settles near
allocation times NPP over turnover whatever it starts at.

### Recommended, not adopted

- **Leaf-on by soil temperature.** At the observed onset of daytime uptake,
  the soil temperature is 14.3 ± 0.7 °C across years, where the growing
  degree-days are 847 ± 119: soil temperature is the steadier trigger
  (`phenology_evidence.csv`). But the evidence is the calibration data
  themselves, so check it against Harvard's phenology record before switching
  `config.MODEL_FLAGS`.
- **Screen the low MODIS LAI values.** 14 of the 89 summer composites are 0.7-2.2,
  which a closed deciduous canopy does not reach in midsummer; likely cloud or
  quality contamination. A screening rule should come from the product's
  quality layers rather than a floor chosen by eye.
- **The NEE discrepancy.** The best prior runs' residuals are about twice the
  noise model's standard deviations and seasonally structured, which a
  two-day correlation cannot represent. Revisit `config.NEE_DISCREPANCY_*`
  after a first calibration, not against the prior.

## Decisions

Each records what was chosen and why, so a result can be read against it.

- **The site** is 4977, Harvard Forest, where two AmeriFlux towers stand:
  US-Ha1 (hourly) and the NEON tower US-xHA (half-hourly).
- **One driver member**, so that the forward model is deterministic, as the
  MCMC comparison needs.
- **The driver file is corrected before any run**, so that every row means
  what SIPNET's format says: labeled with the UTC start of its three-hour
  step, and every value a total or a mean over that step. The four
  corrections, in `prepare_drivers.py`: the drifting hour labels are rebuilt
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
