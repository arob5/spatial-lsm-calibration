# The calibration's model: observations, noise and prior

The mathematics of the single-site calibration: what is observed and how the
model predicts it, the noise covariance, the parameterization and prior, the
diagnostics every run is checked by, and NEE's error: the first
calibration's test of it and the revised model fitted in its place.
`README.md` says how to run it and records the decisions; the code is in
`model/`, and every constant named here is `config.py`'s.

## The observation model

The calibration conditions on a vector $y \in \mathbb{R}^N$ through

$$
y = \mathcal{H}\big(\mathcal{M}(\theta)\big) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, R),
$$

where $\mathcal{M}(\theta)$ is one SIPNET run over the prepared drivers at
parameters $\theta$, $\mathcal{H}$ stacks the observation operators of the
$K = 5$ observation sources below, and $R$ is the noise covariance. $y$, its
order (site-major, then source in the order below, then time) and the
operators are `model/observations.py`'s `calibration_observation_vector()`; each operator
is bound to its source in `config.OBSERVATION_OPERATORS`. This section states
each source's data, its operator and its measurement error exactly, and then
the noise model assembling $R$, which `model/noise.py` builds.

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

scored as `noise.calibration_likelihood().log_density(predictions)` (`model/noise.py`), a
`pyeki.gauss.Gaussian` whose covariance is a `PSDBlockDiag` of one `DensePSD`
block per source, each factored once; a failed run's NaN row is the caller's
to score $-\infty$. The constants below are `config`'s "noise model"
section. The temporal correlation below is
$c_\tau(t, t') = \exp(-|t - t'| / \tau)$, with $t$ in days.

- **NEE**, for each of the two sources separately, the two independent: the
  measurement error on the diagonal plus a discrepancy of two terms,

  $$
  R_{WW'} = \big(\sigma^{\mathrm{obs}}_W\big)^2 \mathbf{1}[W = W'] + \big(\Sigma^\delta\big)_{WW'},
  $$

  $\Sigma^\delta$ a short and a long exponential term in the window ends
  $t_W = w^+$ (`model/discrepancy.py`), whose form, parameters and fit are
  "NEE error" below. The parameters are `config.NEE_DISCREPANCY`'s (EKI
  setup `two_term_discrepancy`):

  | Source | $\sigma_{\mathrm s}$ | $\tau_{\mathrm s}$ (d) | $\sigma_\ell$ | $\tau_\ell$ (d) |
  |---|---|---|---|---|
  | `nee_night_centered` | 0.603 | 0.740 | 1.20 | 57.2 |
  | `nee_day_centered` | 1.96 | 1.72 | 2.30 | 36.5 |

  with the standard deviations in µmol m⁻² s⁻¹. Two earlier setups are
  recorded in "NEE error". The first calibration (`single_term_discrepancy`)
  ran with the short term alone, $\sigma_\delta^2 \, c_\tau(w^+, w'^+)$,
  $\sigma_\delta = 1.0$ at night and 1.8 by day and $\tau = 2$ days:
  $\sigma_\delta$ 0.7 times the standard deviation of the observed windows'
  anomalies from their seasonal cycle (1.43 and 2.55), $\tau$ a correlation
  of 0.61 between consecutive days, where the observed anomalies' is
  0.45-0.50. That calibration rejected it. The second
  (`three_term_discrepancy`) added a term recurring every year, which was
  dropped for the reason "The three-term run" gives.
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

`model/prior.py`'s `calibration()` returns the three objects; this section
says how they were found. The records are in `exploration/parameter_analysis/`
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
   modest sink, where the Niwot conifer set made it a strong source.
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
provenance is in `model/prior.py`'s `PROVENANCE` and `prior_proposal.csv`.

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

**Fixed** (`model/fixed_sipnet_parameters.csv`): everything else. That includes
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

250 draws through the exploration's fast path, none failed
(`final_coverage.csv`, `final_plausibility.csv`), with the noise model's
standard deviations added:

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

The recorded prior predictive, `scripts/prior_predictive.py`'s 200 draws
through the forward model, a separate set of draws, gives the same picture;
its coverage is `figures/prior_predictive.py`'s `prior_predictive_coverage`
figure.

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
  after a first calibration, not against the prior: "NEE error" below does.

## Diagnostics

Every run of the experiment is diagnosed the same way, so that runs compare
directly: `scripts/diagnose.py --run <run>` computes the quantities below
(`model/diagnostics.py`) and writes them as tables under the run's
`diagnostics/`, and `figures/diagnostics.py --run <run>` draws them. A run
is the prior predictive (`prior`) or an EKI run (`synthetic`, `observed`)
of the current setup, `config.EKI_RUN_NAME`. $R$ is the one `config` sets
when the diagnostics run; the script says when that differs from what the
run's `provenance.json` records, since a run diagnosed under another $R$
than it ran with is scored against the wrong reference.

### Notation

A diagnosis reads an ensemble $\theta_1, \dots, \theta_J$ and its
predictions of one observation vector: the final EKI ensemble, of the
calibration vector from its last step and of the validation vector from the
posterior predictive, or the prior predictive's draws. The residual of
observation $i$ at parameters $\theta$, the misfit of the whole vector and
of source $k$ are

$$
\rho_i(\theta) = y_i - \mathcal H_i\big(\mathcal M(\theta)\big),
\qquad
\Phi(\theta) = \tfrac12\, \rho(\theta)^{\top} R^{-1} \rho(\theta),
\qquad
\Phi_k(\theta) = \tfrac12\, \rho_k(\theta)^{\top} R_k^{-1} \rho_k(\theta),
$$

with $\rho_k$ the entries of source $k$, $n_k$ of them, and
$\Phi = \sum_k \Phi_k$, $R$ being block-diagonal over the sources. For an
NEE source, $\mathcal W_k$ is its set of windows and $t_W = w^+$ the end of
window $W$ in days. The ensemble's median prediction is
$\hat g_i = \operatorname{median}_j \mathcal H_i(\mathcal M(\theta_j))$,
with residual $\hat\rho_i = y_i - \hat g_i$; it estimates the error
$\varepsilon_i$ up to the fitted directions of $\theta$, at most $D = 15$ of
$n_k \ge 800$ for an NEE source.

### The posterior predictive check

`predictive_check_<vector>.csv`, per source and for all sources together.
The check is a posterior predictive check with a realized discrepancy
(Gelman, Meng and Stern 1996, *Statistica Sinica* 6, 733-807), whose test
quantity is the misfit, $T(y, \theta) = \Phi(\theta; y)$, a function of the
data and the parameters. For each posterior draw $\theta_j$, the realized
discrepancy $T(y, \theta_j)$ is compared with the discrepancy of data
replicated under the model at that draw,

$$
y^{\mathrm{rep}}_j \sim \mathcal N\big(G(\theta_j), R\big),
\qquad
T\big(y^{\mathrm{rep}}_j, \theta_j\big) = \tfrac12 \big(y^{\mathrm{rep}}_j - G(\theta_j)\big)^{\!\top} R^{-1} \big(y^{\mathrm{rep}}_j - G(\theta_j)\big),
$$

and the posterior predictive p-value is

$$
p = \Pr\Big(T\big(y^{\mathrm{rep}}, \theta\big) \ge T(y, \theta) \;\Big|\; y\Big),
\qquad \theta \sim p(\theta \mid y),\;\; y^{\mathrm{rep}} \mid \theta \sim \mathcal N\big(G(\theta), R\big).
$$

For this test quantity the replicated discrepancy's distribution is known
exactly: at any fixed $\theta$, $R^{-1/2}\big(y^{\mathrm{rep}} - G(\theta)\big)$
is standard normal, so $2\,T(y^{\mathrm{rep}}, \theta) \sim \chi^2_N$, whatever
$\theta$ is. No replicated data need be drawn, and with $J$ draws

$$
p \approx \frac1J \sum_{j=1}^{J} \Pr\Big(\chi^2_N \ge 2\,\Phi(\theta_j)\Big).
$$

Since $R$ is block-diagonal, the same holds for each source alone, with
$\Phi_k$ and $\chi^2_{n_k}$ in place of $\Phi$ and $\chi^2_N$. Under the
model, the realized misfits lie where $\tfrac12\chi^2_N$ puts its mass:
near $N/2$, within a few multiples of $\sqrt{N/2}$. (That the realized
discrepancy, and not only the replicated one, is centered there follows from
Bayes' rule: $\theta^\star$ drawn from the prior and $y$ from the model at it
have joint density $p(\theta^\star)\,p(y \mid \theta^\star) = p(y)\,p(\theta^\star
\mid y)$, the joint density of $y$ with a posterior draw $\theta$, so
averaged over data sets $\Phi(\theta)$ has the distribution of
$\Phi(\theta^\star)$, which is $\tfrac12\chi^2_N$.)

The table reports, per source, $n_k$; the median, minimum and maximum over
members of $\Phi_k(\theta_j)$; the ratio $2 \operatorname{median}_j
\Phi_k(\theta_j) / n_k$, near 1 under the model; the standardized misfit
$\big(\operatorname{median}_j \Phi_k(\theta_j) - n_k/2\big) / \sqrt{n_k/2}$,
within about $\pm 2$ under the model; the p-value $p_k$; and the coverage,
the fraction of the source's observations inside the ensemble's 90%
predictive interval, the 5% and 95% quantiles over $j$ of
$\mathcal H_i(\mathcal M(\theta_j)) + \epsilon_{ij}$,
$\epsilon_{ij} \sim \mathcal N(0, R_{ii})$ independent (seed 0). The row
`all` sums the misfits over sources per member.

Four cautions come with the check:

- **It is conservative.** The data form the posterior and are then checked
  against it, so under the model the p-value is not uniform: it concentrates
  around $\tfrac12$, and small values are rarer than their nominal rate. A
  p-value near 0 is strong evidence against the model; a moderate one is
  weak evidence for it. On held-out data, which did not form the posterior,
  it is not conservative.
- **The draws are approximate.** The $\theta_j$ are an EKI ensemble, which
  approximates the posterior. The same check on synthetic data, where the
  model is right by construction, shows whether that approximation can fail
  the check by itself.
- **On the prior predictive** the same computation, with prior draws, is a
  prior predictive check, which asks whether the prior's draws could have
  produced the data, not whether the fitted model describes them.
- **It tests consistency with $R$, and nothing more.** An $R$ that makes a
  bias cheap, such as one shared by every year, passes a fit that has the
  bias. The check is read with the coverage, the seasonal means of the
  residuals and the weekly residuals ("NEE's residuals"), never alone: "The
  three-term run" in "NEE error" is a calibration that passed it and missed
  the summer uptake by half.

### NEE's residuals

For the calibration vector's NEE sources, from the residuals $\hat\rho_W$ of
the median prediction. A window is placed in time by its start,
$w^- = t_W - 12$ h, so that a window ending at midnight on 1 January belongs
to the day and year it covers. `nee_residuals.csv` holds every window's
$y_W$, $\hat g_W$, $\hat\rho_W$ and $\sigma^{\mathrm{obs}}_W$.

**Size** (`nee_residual_summary.csv`). With
$\bar v_k = \frac{1}{n_k}\sum_{W \in \mathcal W_k} \big(\sigma^{\mathrm{obs}}_W\big)^2$
the mean measurement variance of source $k$, the standard deviation of the
residuals beyond measurement error is

$$
\hat s_k = \Big(\widehat{\operatorname{Var}}_{W \in \mathcal W_k}(\hat\rho_W) - \bar v_k\Big)^{1/2},
$$

reported beside the configured discrepancy's standard deviation at one
window, $\big(\Sigma^\delta_{WW}\big)^{1/2}$, and the seasonal means of
$\hat\rho_W$ (December-February, March-May, June-August,
September-November, by the month of $w^-$).

**The recurring seasonal part** (`nee_weekly_residuals.csv`, and
`recurring_share` and `year_correlation_*` in the summary). Let
$\omega(W) = \min\big(\lfloor (\mathrm{doy}(w^-) - 1) / 7 \rfloor + 1,\ 52\big)$
be the week of the calendar year of $w^-$, $\mathrm{doy}$ its day of the
year, the last one or two days counted in week 52, so that a week never
spans two years, and $\bar\rho_\omega$ the mean of $\hat\rho_W$ over the windows of
week $\omega$ in all years. Smoothed circularly over five weeks,

$$
\bar s(\omega) = \operatorname{mean}\big\{\bar\rho_{\omega + i} : i = -2, \dots, 2,\ \bar\rho_{\omega + i} \text{ defined}\big\}
\quad (\text{indices mod } 52),
$$

it is the part of the residual that recurs at the same season every year,
$\hat\rho^{\mathrm{seas}}_W = \bar s(\omega(W))$; the remainder is
$\hat\rho^{\mathrm{rem}}_W = \hat\rho_W - \hat\rho^{\mathrm{seas}}_W$. The
summary reports its share of the residual variance,
$\operatorname{Var}(\hat\rho^{\mathrm{seas}}) / \operatorname{Var}(\hat\rho)$,
and, for each year with more than ten weeks observed, the correlation over
weeks of that year's weekly means with the mean of the other years' weekly
means: its median, minimum and maximum over years. The weekly table holds
each year's weekly means and $\bar s$.

**Autocorrelation** (`nee_autocorrelation.csv`). The residuals on a daily
grid, one per day $w^-$, days without an observation missing, and their
sample autocorrelation at lags $\ell \in \{1, 2, 5, 10, 30\}$ days: the
Pearson correlation of the pairs of days $\ell$ apart that are both
observed, for $\hat\rho$ (`observed`) and for $\hat\rho^{\mathrm{rem}}$
(`remainder`). Beside them, the correlation of the residuals that $R$
implies, the measurement errors being independent,

$$
c^{R}_k(\ell) = \frac{\Sigma^\delta_k(\ell)}{\Sigma^\delta_k(0) + \bar v_k},
$$

with $\Sigma^\delta_k(\ell)$ the discrepancy's covariance at a separation of
$\ell$ days (`modeled`). Under the model, `observed` follows `modeled`.

**Night and day** (`nee_night_day.csv`). The correlation over days of the
residual of the night-centered window $(d, d + 12\,\mathrm h]$ with that of
the day-centered window $(d + 12\,\mathrm h, d + 24\,\mathrm h]$
(`same_day`), and of the day-centered window with the next night's
(`next_night`), with the numbers of pairs. $R$ makes the two sources
independent; these say how far that holds.

### The two towers

`nee_towers.csv` and `nee_towers_summary.csv`, the same for every run.
US-Ha1 and US-xHA stand in the same pixel. Both towers' NEE windows are
built as the observations are, over 2012-2024, and paired where both keep a
window. For tower $i$, write its window mean as

$$
y^{(i)}_W = f_W + e^{(i)}_W + r^{(i)}_W ,
$$

with $f_W$ the pixel's mean NEE over $W$, $e^{(i)}_W$ the tower's random
measurement error, of variance $\big(\sigma^{\mathrm{obs},(i)}_W\big)^2$ as
under "Measurement error", and $r^{(i)}_W$ its representativeness error:
footprint, u\* filtering and processing. If the four errors are independent
and each $r^{(i)}_W$ has variance $\sigma_r^2$, the difference
$d_W = y^{(1)}_W - y^{(2)}_W = e^{(1)}_W - e^{(2)}_W + r^{(1)}_W - r^{(2)}_W$
has

$$
\operatorname{Var}(d_W) = \mathbb E\Big[\big(\sigma^{\mathrm{obs},(1)}_W\big)^2 + \big(\sigma^{\mathrm{obs},(2)}_W\big)^2\Big] + 2\sigma_r^2,
\qquad
\hat\sigma_r^2 = \tfrac12\Big(\widehat{\operatorname{Var}}(d_W) - \overline{\big(\sigma^{\mathrm{obs},(1)}\big)^2 + \big(\sigma^{\mathrm{obs},(2)}\big)^2}\Big),
$$

the second term's bar the mean over the paired windows. If the footprints
overlap, $r^{(1)}$ and $r^{(2)}$ are correlated with some $\kappa \ge 0$,
$\operatorname{Var}(r^{(1)} - r^{(2)}) = 2\sigma_r^2(1 - \kappa)$, and
$\hat\sigma_r^2$ estimates $\sigma_r^2(1 - \kappa) \le \sigma_r^2$: a lower
bound (`representativeness_standard_deviation`). The summary also reports the mean
difference, overall and per calendar quarter, since part of $r^{(i)}$ can be
a bias.

What the difference cannot see is SIPNET's structural error. The
calibration residual is

$$
\rho_W(\theta) = \big(f_W - \mathcal H_W(\mathcal M(\theta))\big) + e^{(1)}_W + r^{(1)}_W ,
$$

and the model's error $f_W - \mathcal H_W(\mathcal M(\theta))$, common to
both towers, cancels in $d_W$. The towers give a floor on the discrepancy,
not an estimate of it.

## NEE error

The first calibration, EKI on the observed data under the single-term NEE
discrepancy (EKI setup `single_term_discrepancy`, recorded in
`output/eki/single_term_discrepancy/` with its `provenance.json`), tests
the noise model, and its NEE terms fail the test. This section gives the
diagnostics of that run and of its synthetic twin, all under the $R$ they
ran with, says what they show is wrong, and states the revised model for
NEE's error, how its parameters are estimated, and the fit, which
`config.NEE_DISCREPANCY` now holds.

### The first calibration's predictive check

With $N = 2715$, $N/2 = 1357.5$ and $\sqrt{N/2} = 36.8$. Per source,
$\Phi_k$ is the median over the $J = 100$ members, and $p$ the average of
"The posterior predictive check":

| | $n_k$ | $n_k/2$ | synthetic: $\Phi_k$ | $p$ | observed: $\Phi_k$ | $2\Phi_k/n_k$ | $p$ |
|---|---|---|---|---|---|---|---|
| NEE, night-centered | 800 | 400 | 419 | 0.17 | 558 | 1.39 | 1 × 10⁻¹⁰ |
| NEE, day-centered | 1819 | 909.5 | 887 | 0.77 | 1708 | 1.88 | 1 × 10⁻⁹⁵ |
| MODIS LAI | 89 | 44.5 | 40.5 | 0.70 | 218 | 4.89 | 1 × 10⁻⁴⁵ |
| LandTrendr biomass | 6 | 3 | 1.3 | 0.79 | 7.6 | 2.52 | 0.04 |
| SoilGrids soil carbon | 1 | 0.5 | 0.2 | 0.60 | 0.4 | 0.72 | 0.41 |
| all sources | 2715 | 1357.5 | 1343 to 1361 | 0.59 | 2476 to 2509 | 1.84 | 1 × 10⁻¹³⁵ |

The last row's $\Phi$ is the range over members. On synthetic data every
source passes, with p-values near $\tfrac12$ as the first caution leads one
to expect, and the ensemble's misfits bracket the truth's, 1344: the
approximation does not fail the check by itself. On the observed data the
check fails, overall and for NEE by day and night and for LAI, by margins no
conservativeness or approximation accounts for: every member's misfit is
about 31 standard deviations above $N/2$, and the whitened residuals
$R^{-1/2}\rho(\theta_j)$ have mean square 1.82 to 1.85 where the model says
1. On the held-out tower, US-xHA in 2021-2024, the day-centered windows fail
too (943 windows in all; by day $2\Phi_k/n_k = 1.86$, $p = 1 \times
10^{-31}$), and the night-centered ones narrowly ($1.18$, $p = 0.02$).

So no $\theta$ the calibration reached makes the residuals as small as $R$
says they are. For the likelihood, "$R$ is too small" and "SIPNET cannot
reproduce the data at any such $\theta$" are the same statement; the
discrepancy $\Sigma^\delta$ is where it is accounted for. One qualification:
every EKI iterate lies in the affine span of the initial ensemble, so a
$\theta$ outside it could fit better; an MCMC run on the same problem would
settle it. NEE by day and LAI fail by the most; this section is about NEE,
whose windows are 96% of $y$.

### What the residuals show

"NEE's residuals" in "Diagnostics", of the first calibration.

**Size.** In µmol m⁻² s⁻¹ for standard deviations and their squares for
variances:

| Source | $\widehat{\operatorname{Var}}(\hat\rho_W)$ | $\bar v_k$ | $\hat s_k$ | $\sigma_\delta$ of the run |
|---|---|---|---|---|
| night-centered | 2.51 | 0.47 | 1.43 | 1.0 |
| day-centered | 10.01 | 1.05 | 2.99 | 1.8 |

**A recurring seasonal part.**

| Source | mean of $\hat\rho_W$, DJF / MAM / JJA / SON | recurring share of the variance | year-to-year correlation, median (range) |
|---|---|---|---|
| night-centered | 0.02 / 0.84 / 1.38 / 0.88 | 0.18 | 0.37 (0.11 to 0.54) |
| day-centered | −0.27 / −1.40 / −1.94 / −0.46 | 0.10 | 0.36 (0.23 to 0.60) |

NEE is positive to the atmosphere, so from spring to autumn the observed
night respiration exceeds the model's and the observed daytime uptake
exceeds the model's, the same way in every year.

**Long memory.** The residuals' autocorrelation, with and without the
recurring part, against the correlation the run's $R$ implies:

| Lag $\ell$ (days) | 1 | 2 | 5 | 10 | 30 |
|---|---|---|---|---|---|
| night-centered, $\hat\rho$ | 0.62 | 0.54 | 0.54 | 0.48 | 0.34 |
| night-centered, $\hat\rho^{\mathrm{rem}}$ | 0.50 | 0.36 | 0.33 | 0.35 | 0.20 |
| night-centered, $c^R(\ell)$ | 0.41 | 0.25 | 0.06 | 0.005 | 0.000 |
| day-centered, $\hat\rho$ | 0.63 | 0.53 | 0.39 | 0.30 | 0.12 |
| day-centered, $\hat\rho^{\mathrm{rem}}$ | 0.57 | 0.46 | 0.31 | 0.22 | 0.13 |
| day-centered, $c^R(\ell)$ | 0.46 | 0.28 | 0.06 | 0.005 | 0.000 |

The correlation drops over the first day or two and then decays over weeks:
at least two timescales, where the run's term has one of two days.

**Night and day.** The two sources' residuals of one day correlate at 0.10
(643 days), and a day's with the next night's at 0.17 (631 days): weakly,
so the two blocks stay independent.

**Why the structure matters, not only the size.** A shift of the residuals
along a direction $v$ (the shape a parameter such as $Q_{10}$ changes the
seasonal cycle by) is informed by $v^{\top} R^{-1} v$. If the error has a
component along $v$ of variance $\sigma^2$, $R = A + \sigma^2 v v^{\top}$ with
$A$ positive definite and $v$ of unit norm, then by the Sherman-Morrison
formula

$$
v^{\top} R^{-1} v = \frac{a}{1 + \sigma^2 a} < \frac{1}{\sigma^2},
\qquad a = v^{\top} A^{-1} v,
$$

however many windows there are. An error that recurs every year, or persists
for weeks, bounds the evidence the data carry in its directions. The run's
$R$ has no such component: its correlation vanishes within about a week, so
each week, and each year's repeat of the same seasonal bias, counts as new
evidence, and the evidence grows with $n_k$. That is how nine springs of one
structural bias can move $Q_{10}$ to 1.0 and the optimum photosynthesis
temperature to 30 °C, both far outside their priors.

**A data point to review.** Five day-centered windows of January-February
2015 hold observed NEE of 9.7 to 17 µmol m⁻² s⁻¹, where the season's
other windows are near 1-2 and the model's 1.5; they are measured windows,
not gap-filled ones. They are kept: they are a few percent of the
day-centered residuals' sum of squares, and whether to screen them is a
question about the data, not the noise model.

### What the two towers show

"The two towers" in "Diagnostics". Over the windows both towers keep, which
fall in 2019-2020 only:

| Source | windows | $\widehat{\operatorname{Var}}(d_W)$ | mean $\sum_i \big(\sigma^{\mathrm{obs},(i)}_W\big)^2$ | $\hat\sigma_r$ | $\hat\sigma_r^2 / \hat s_k^2$ |
|---|---|---|---|---|---|
| night-centered | 52 | 1.22 | 0.42 | 0.63 | 0.20 |
| day-centered | 135 | 5.12 | 1.73 | 1.30 | 0.19 |

The difference is also systematic: by day in July-September its mean is
$+1.75$ µmol m⁻² s⁻¹ (47 windows), US-xHA measuring more uptake than
US-Ha1, so part of $r^{(i)}$ is a seasonal bias of each tower. By the last
column, the towers account for about a fifth of the non-measurement
variance; the rest is the model's, which the towers cannot see.

### A revised error model for NEE

The measurement error $\Sigma^{\mathrm{obs}}_k$ stays as it is. The
discrepancy of each NEE source becomes the sum of three terms, one per
feature above:

$$
\big(\Sigma^\delta_k\big)_{WW'} =
\sigma_{\mathrm s}^2 \, e^{-|t_W - t_{W'}| / \tau_{\mathrm s}}
+ \sigma_{\ell}^2 \, e^{-|t_W - t_{W'}| / \tau_{\ell}}
+ \sigma_{\mathrm p}^2 \exp\!\Big(-\frac{2 \sin^2\!\big(\pi (t_W - t_{W'}) / P\big)}{\lambda^2}\Big),
\qquad P = 365.25 \text{ days},
$$

each term's parameters separate for the two sources, which stay independent
of each other:

- a **short** term, $\tau_{\mathrm s}$ of order a day: the weather-scale
  errors behind the drop in correlation over the first lags;
- a **long** term, $\tau_\ell$ of order weeks: the anomalies of one year,
  such as the timing of leaf-on or a dry spell, that persist but do not
  recur;
- a **recurring** term, periodic in the day of the year: the seasonal bias
  that repeats every year. $\lambda$ sets its width in the season (for small
  $\lambda$, the correlation falls to $e^{-1/2}$ at a separation of about
  $\lambda P / (2\pi)$ days of the year); two windows a whole number of years
  apart are perfectly correlated. Its Fourier expansion has a constant term,
  so it also carries a bias common to every season.

The parameters of source $k$ are
$\phi_k = (\sigma_{\mathrm s}, \tau_{\mathrm s}, \sigma_\ell, \tau_\ell,
\sigma_{\mathrm p}, \lambda)$. It is compared with two variants that bracket
it: the first calibration's single term, the special case $\sigma_\ell = \sigma_{\mathrm
p} = 0$, refitted; and a recurring term that drifts from year to year, its
periodic factor multiplied by $e^{-|t_W - t_{W'}|/\tau_{\mathrm p}}$ with
$\tau_{\mathrm p}$ in years, of which the model above is the limit
$\tau_{\mathrm p} \to \infty$.

The two NEE blocks stay independent of each other: the night and day
windows of one day could share a respiration error, but their residuals
correlate at only 0.10 ("What the residuals show").

### Estimating the parameters

$R$ must be fixed for EKI, and fixed alike for MCMC so the two target one
posterior, so $\phi_k$ is estimated before the calibration that uses it, by
maximum marginal likelihood (type-II maximum likelihood) on the residuals of
a first calibration, as follows.

1. **Residuals.** From the current calibration's final ensemble,
   $\hat\rho_k = y_k - \hat g_k$ for each NEE source.
2. **Fit.** With $R_k(\phi) = \Sigma^{\mathrm{obs}}_k + \Sigma^\delta_k(\phi)$,
   maximize the Gaussian log likelihood of the residuals,

   $$
   \hat\phi_k = \arg\max_{\phi} \; \ell_k(\phi),
   \qquad
   \ell_k(\phi) = -\tfrac12\, \hat\rho_k^{\top} R_k(\phi)^{-1} \hat\rho_k - \tfrac12 \log\det R_k(\phi) - \tfrac{n_k}{2} \log 2\pi ,
   $$

   over the logarithms of the parameters, by L-BFGS-B with the gradient from
   JAX, within bounds on each (`scripts/fit_nee_discrepancy.py`'s `BOUNDS`),
   from the starts of its `VARIANTS` (three for the three-term model, their
   standard deviations fractions of $\hat s_k$ and their timescales from
   half a day to two months), keeping the best. Each evaluation is one Cholesky
   factorization of $R_k$, of side $n_k \le 1819$; no SIPNET run is needed.
   The residuals are modeled as zero-mean, their bias carried by the
   recurring term, whose constant Fourier term is a bias common to every
   season.
3. **The tower floor.** The total discrepancy variance of each source is at
   least the towers' estimate of the representativeness error alone,

   $$
   \sigma_{\mathrm s}^2 + \sigma_\ell^2 + \sigma_{\mathrm p}^2 \;\ge\; \hat\sigma_r^2 ,
   $$

   checked at $\hat\phi_k$. With $\hat s_k$ five times $\hat\sigma_r$ in
   variance, it is not expected to bind; if it does, the fit is redone with
   it as a constraint.
4. **The check on held-out data.** $\hat\phi_k$ is fitted to the residuals of
   the data the calibration saw, so it is judged on data neither saw: the
   validation vector, US-xHA in 2021-2024, with $R^{\mathrm{val}}_k$ built
   from $\hat\phi_k$ and US-xHA's own measurement errors. The posterior
   predictive check of "Diagnostics", applied to the validation vector,
   should pass:
   $2\,\Phi^{\mathrm{val}}_k / n^{\mathrm{val}}_k$ near 1, within about
   $\sqrt{2 / n^{\mathrm{val}}_k}$, and the predictive intervals covering at
   their nominal rates. On held-out data the check is not conservative, since
   those data did not form the posterior. The three variants are ranked by
   the held-out log predictive density, over the members of the calibration
   whose residuals were fitted,

   $$
   \log p\big(y^{\mathrm{val}} \mid y\big) \approx \log \frac1J \sum_{j=1}^{J} \mathcal N\big(y^{\mathrm{val}};\, G^{\mathrm{val}}(\theta_j),\, R^{\mathrm{val}}\big),
   $$

   and by $\ell_k(\hat\phi_k)$ with a penalty for the parameters added
   (AIC), which have to agree for the choice to stand.
5. **One iteration.** The residuals of step 1 came from a calibration with
   the old $R$. EKI is rerun with $R(\hat\phi)$, and steps 1-2 repeated on its
   residuals; if every $\hat\sigma$ and $\hat\tau$ moves by less than about
   10%, $\hat\phi$ stands, and otherwise the loop runs once more. The loop
   is not safe on its own: a term that can absorb a bias the parameters
   should explain lets the calibration leave the bias in the residuals, the
   refit then asks for more of the term, and the discrepancy grows with each
   round. Each round is therefore checked on the residuals themselves, their
   seasonal means and coverage, before its fit is adopted ("The three-term
   run").

This fits the discrepancy to the same data the calibration then conditions
on, which the held-out check guards against. The principled version infers
$\phi$ with $\theta$, $p(\theta, \phi \mid y) \propto p(y \mid \theta, \phi)
\, p(\theta)\, p(\phi)$: given the predictions $G(\theta)$, the likelihood's
dependence on $\phi$ costs only the Cholesky factorizations above, so an
MCMC sampler can update $\phi$ at no extra SIPNET cost. EKI conditions on a
fixed $R$ and cannot. That is a natural part of the MCMC-against-EKI
comparison.

What this changes and what it does not: it makes $R$ describe the errors
the residuals show, so the posterior's width reflects them and a recurring
bias counts as one piece of evidence rather than nine. It does not remove
the bias. The model's spring respiration and its abrupt leaf-on, against
the observed gradual onset of uptake, remain structural errors of SIPNET or
of its configuration (see "Recommended, not adopted").

### The fit

`scripts/fit_nee_discrepancy.py --data observed`, on the first
calibration's residuals, recorded in its run directory as
`nee_discrepancy_fit.csv`. The held-out columns score each fitted $R$ on the
first calibration's posterior predictive of US-xHA: they rank the error
models given that one posterior, not posteriors formed under each.

| Source | variant | $k$ | $\ell_k(\hat\phi_k)$ | AIC | held-out $\log p(y^{\mathrm{val}} \mid y)$ | held-out $2\Phi/n$ |
|---|---|---|---|---|---|---|
| night-centered | single | 2 | −1239.6 | 2483.1 | −422.1 | 1.11 |
| | two-term | 4 | −1197.5 | 2403.0 | −408.3 | 0.92 |
| | three-term | 6 | −1184.2 | 2380.3 | −399.8 | 0.88 |
| | drifting | 7 | −1181.1 | 2376.2 | −401.3 | 0.92 |
| day-centered | single | 2 | −4144.2 | 8292.3 | −1359.4 | 1.14 |
| | two-term | 4 | −4094.8 | 8197.5 | −1351.5 | 1.15 |
| | three-term | 6 | −4073.0 | 8158.1 | −1335.5 | 1.12 |
| | drifting | 7 | −4070.9 | 8155.9 | −1334.6 | 1.12 |

The refitted single term is $\sigma_\delta = 1.39$ over $\tau = 13.7$ days at
night and 3.07 over 5.5 days by day: larger and longer than the first
calibration's, and still far worse than three terms. Against it the
three-term model gains 55 and 71 in log likelihood for four parameters, and
22 and 24 nats on the held-out tower; the two criteria agree. Between the
three-term model and the drifting one they do not separate: at night AIC
favors the drifting variant by 4.2 and the held-out density the three-term
one by 1.4 nats, by day the drifting variant by 2.2 and 0.9, and its fitted
drift is slow, $\tau_{\mathrm p}$ of 7.9 years at night and 4.7 by day. The
three-term model was adopted for both sources, the simpler of two the data
do not tell apart, and run as setup `three_term_discrepancy`. Every fit holds
the towers' floor: the three-term discrepancy's standard deviation at one
window is 1.28 at night and 2.93 by day, against $\hat\sigma_r$ of 0.63 and
1.30. The two-term variant, the short and long terms without the recurring
one, was fitted after that run, by `--setup single_term_discrepancy`: it
fits the residuals less well than three terms and better than one, and is
the one adopted now, for the reason below.

### The three-term run

EKI under the three-term $R$ (setup `three_term_discrepancy`, observed data)
passed the posterior predictive check and gave the summer uptake up:

| | first calibration | three-term run |
|---|---|---|
| all sources, $2\Phi/N$ ($p$) | 1.84 ($10^{-135}$) | 1.05 (0.03) |
| NEE day, $2\Phi_k/n_k$; coverage | 1.88; 0.74 | 0.86; 0.67 |
| NEE night, $2\Phi_k/n_k$; coverage | 1.39; 0.83 | 1.10; 0.88 |
| held-out tower, $2\Phi/n$ | 1.61 | 0.85 |
| NEE day, residual variance; non-measurement sd | 10.0; 2.99 | 20.0; 4.36 |
| NEE day, mean residual MAM / JJA / SON | −1.40 / −1.94 / −0.46 | −2.23 / −8.91 / −3.02 |
| NEE day, recurring share of the residual variance | 0.10 | 0.65 |

The posterior predictive puts summer daytime NEE near −6 µmol m⁻² s⁻¹
where the weekly observed means reach −17. The posterior moved to less
uptake: photosynthetic capacity 182-228 (the first calibration's 255-283),
half-saturation light 23-37 (prior 90% 6-26), the foliar respiration share
0.028-0.054 (prior 0.051-0.164), and the wood respiration rate at 10 °C
0.026-0.044 (prior 0.006-0.030). Soil $Q_{10}$ moved off 1.0 (0.97-1.61) and
the optimum photosynthesis temperature stayed out of its prior (28.3-30.3
°C). The refit on its residuals kept the night terms and asked, by day, for
a recurring standard deviation of 4.56 over a width $\lambda$ of 0.95, 3.5
and 4.6 times the run's.

The recurring term is perfectly correlated across years, so by the argument
of "Why the structure matters" the evidence against a bias that repeats in
every summer is capped at about $1/\sigma_{\mathrm p}^2$ in its direction,
however many summers there are. That is what the term was meant to do for
the first calibration's small recurring bias; it also made a large one
cheap, and the calibration took it, trading the summer uptake for the rest
of the fit. The refit then read the larger bias as a larger recurring term,
the runaway step 5 now warns of. This is the confounding of a discrepancy
with the parameters (Brynjarsdóttir and O'Hagan 2014, *Inverse Problems* 30,
114007): a discrepancy flexible enough to absorb what the parameters should
explain needs an informative prior, or it takes the signal.

The recurring term is therefore dropped. The long term remains: with
$\tau_\ell$ of 36-57 days it correlates the windows of one season but not
the same season of different years, so each year's miss of the summer
uptake counts as its own evidence. The run under the two-term $R$ is setup
`two_term_discrepancy`. The principled form, a discrepancy inferred with
$\theta$ under an informative prior on its parameters, is for MCMC, which
can update $\phi$ at no extra SIPNET cost.
