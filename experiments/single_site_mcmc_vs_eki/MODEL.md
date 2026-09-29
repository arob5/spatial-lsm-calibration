# The calibration's model: observations, noise and prior

The mathematics of the single-site calibration: what is observed and how the
model predicts it, the noise covariance, the parameterization and prior, and
the first calibration's test of NEE's error, with a revised model for it.
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

- **NEE**, for each of the two sources separately, the two independent:

  $$
  R_{WW'} = \big(\sigma^{\mathrm{obs}}_W\big)^2 \mathbf{1}[W = W'] + \sigma_\delta^2 \, c_\tau(w^+, w'^+),
  $$

  with $\sigma_\delta = 1.0$ µmol m⁻² s⁻¹ for `nee_night_centered` and $1.8$
  for `nee_day_centered`, and $\tau = 2$ days. The first calibration rejects
  this term; "NEE error" below states the test and proposes its
  replacement. $\sigma_\delta$ is 0.7 times the
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

## NEE error

The first calibration, EKI on the observed data (`scripts/eki.py --data
observed`, recorded in `output/eki/observed/` with its `provenance.json`),
tests the noise model above, and its NEE terms fail the test. This section
states the test, the two analyses that say what is wrong, and a revised
model for NEE's error with how its parameters are to be estimated. The
revised model is a proposal; `config.py` and `model/noise.py` still hold the
model of "Noise model" above.

### Notation

For NEE source $k \in \{\mathrm{night}, \mathrm{day}\}$, $\mathcal W_k$ is its
set of windows, $n_k = |\mathcal W_k|$, and $t_W = w^+$ is the end of window
$W$, in days. The residual of window $W$ at parameters $\theta$ is

$$
\rho_W(\theta) = y_W - \mathcal H_W\big(\mathcal M(\theta)\big),
$$

and $\rho_k(\theta) = (\rho_W(\theta))_{W \in \mathcal W_k}$. The misfit of
the whole vector and of source $k$ are

$$
\Phi(\theta) = \tfrac12\, \rho(\theta)^{\top} R^{-1} \rho(\theta),
\qquad
\Phi_k(\theta) = \tfrac12\, \rho_k(\theta)^{\top} R_k^{-1} \rho_k(\theta),
$$

and $\Phi = \sum_k \Phi_k$, $R$ being block-diagonal over the $K$ sources.
The final EKI ensemble is $\theta_1, \dots, \theta_J$, $J = 100$, and
$\hat g_W = \operatorname{median}_j \mathcal H_W(\mathcal M(\theta_j))$ is its
median prediction, with residual $\hat\rho_W = y_W - \hat g_W$.

### The test: a posterior predictive check

The test is a posterior predictive check with a realized discrepancy (Gelman,
Meng and Stern 1996, *Statistica Sinica* 6, 733-807), whose test quantity is
the misfit itself, $T(y, \theta) = \Phi(\theta; y)$, a function of the data
and the parameters. For each posterior draw $\theta_j$, the realized
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
$\Phi_k$ and $\chi^2_{n_k}$ in place of $\Phi$ and $\chi^2_N$; the per-source
checks are independent given $\theta$. Under the model, the realized misfits
lie where $\tfrac12\chi^2_N$ puts its mass: near $N/2$, within a few
multiples of $\sqrt{N/2}$. (That the realized discrepancy, and not only the
replicated one, is centered there follows from Bayes' rule: $\theta^\star$
drawn from the prior and $y$ from the model at it have joint density
$p(\theta^\star)\,p(y \mid \theta^\star) = p(y)\,p(\theta^\star \mid y)$, the
joint density of $y$ with a posterior draw $\theta$, so averaged over data
sets $\Phi(\theta)$ has the distribution of $\Phi(\theta^\star)$, which is
$\tfrac12\chi^2_N$.)

Two cautions come with the check:

- **It is conservative.** The data form the posterior and are then checked
  against it, so under the model the p-value is not uniform: it concentrates
  around $\tfrac12$, and small values are rarer than their nominal rate. A
  p-value near 0 is strong evidence against the model; a moderate one is
  weak evidence for it.
- **The draws are approximate.** The $\theta_j$ are the final EKI ensemble,
  which approximates the posterior. The same check on synthetic data, where
  the model is right by construction, shows whether that approximation can
  fail the check by itself.

With $N = 2715$, $N/2 = 1357.5$ and $\sqrt{N/2} = 36.8$. Per source, $\Phi_k$
is the median over the $J = 100$ members, and $p$ the average above:

| | $n_k$ | $n_k/2$ | synthetic: $\Phi_k$ | $p$ | observed: $\Phi_k$ | $2\Phi_k/n_k$ | $p$ |
|---|---|---|---|---|---|---|---|
| NEE, night-centered | 800 | 400 | 419 | 0.17 | 558 | 1.39 | 1 × 10⁻¹⁰ |
| NEE, day-centered | 1819 | 909.5 | 887 | 0.77 | 1708 | 1.88 | 1 × 10⁻⁹⁵ |
| MODIS LAI | 89 | 44.5 | 40.5 | 0.70 | 218 | 4.89 | 1 × 10⁻⁴⁵ |
| LandTrendr biomass | 6 | 3 | 1.3 | 0.79 | 7.6 | 2.53 | 0.04 |
| SoilGrids soil carbon | 1 | 0.5 | 0.2 | 0.60 | 0.4 | 0.8 | 0.41 |
| all sources | 2715 | 1357.5 | 1343 to 1361 | 0.59 | 2476 to 2509 | 1.83 | 1 × 10⁻¹³⁵ |

The last row's $\Phi$ is the range over members. On synthetic data every
source passes, with p-values near $\tfrac12$ as the first caution leads one
to expect, and the ensemble's misfits bracket the truth's, 1344: the
approximation does not fail the check by itself. On the observed data the
check fails, overall and for NEE by day and night and for LAI, by margins no
conservativeness or approximation accounts for: every member's misfit is
about 31 standard deviations above $N/2$, and the whitened residuals
$R^{-1/2}\rho(\theta_j)$ have mean square 1.82 to 1.85 where the model says 1.

So no $\theta$ the calibration reached makes the residuals as small as $R$
says they are. For the likelihood, "$R$ is too small" and "SIPNET cannot
reproduce the data at any such $\theta$" are the same statement; the
discrepancy $\Sigma^\delta$ is where it is accounted for. One qualification:
every EKI iterate lies in the affine span of the initial ensemble, so a
$\theta$ outside it could fit better; an MCMC run on the same problem would
settle it. NEE by day and LAI fail by the most; this section is about NEE,
whose windows are 96% of $y$.

### What the residuals show

The residuals of the median prediction, $\hat\rho_W$, estimate the errors
$\varepsilon_W$ up to the fitted directions of $\theta$, at most $D = 15$ of
$n_k \ge 800$, which is negligible here.

**Size.** With $\bar v_k = \frac{1}{n_k}\sum_W \big(\sigma^{\mathrm{obs}}_W\big)^2$
the mean measurement variance of source $k$, the variance of the residuals
beyond measurement error is
$\hat s_k^2 = \widehat{\operatorname{Var}}(\hat\rho_W) - \bar v_k$:

| Source | $\widehat{\operatorname{Var}}(\hat\rho_W)$ | $\bar v_k$ | $\hat s_k$ | $\sigma_\delta$ now |
|---|---|---|---|---|
| night-centered | 2.51 | 0.47 | 1.43 | 1.0 |
| day-centered | 10.01 | 1.05 | 2.99 | 1.8 |

with variances in (µmol m⁻² s⁻¹)² and standard deviations in µmol m⁻² s⁻¹.

**A recurring seasonal part.** Let $\omega(W) \in \{1, \dots, 52\}$ be the
ISO week of $t_W$, week 53 counted as 52, and $\bar\rho_\omega$ the mean of
$\hat\rho_W$ over the windows of week $\omega$ in all nine years. Smoothing
it circularly over five weeks,
$\bar s(\omega) = \frac15 \sum_{i=-2}^{2} \bar\rho_{\omega + i}$ (indices
mod 52), gives the part of the residual that recurs at the same season every
year, $\hat\rho^{\mathrm{seas}}_W = \bar s(\omega(W))$, and the remainder
$\hat\rho^{\mathrm{rem}}_W = \hat\rho_W - \hat\rho^{\mathrm{seas}}_W$.

| Source | seasonal mean of $\hat\rho_W$, DJF / MAM / JJA / SON | $\operatorname{Var}(\hat\rho^{\mathrm{seas}}) / \operatorname{Var}(\hat\rho)$ | correlation of one year's weekly means with the other years' mean, median (range) |
|---|---|---|---|
| night-centered | 0.02 / 0.84 / 1.38 / 0.88 | 0.17 | 0.37 (0.04 to 0.58) |
| day-centered | −0.25 / −1.43 / −1.91 / −0.47 | 0.10 | 0.42 (0.12 to 0.58) |

NEE is positive to the atmosphere, so from spring to autumn the observed
night respiration exceeds the model's and the observed daytime uptake
exceeds the model's, the same way in every year.

**Long memory.** The remainder's autocorrelation, over the pairs of days
both observed, against the correlation the current discrepancy term assumes,
$c_\tau(\ell) = e^{-\ell/\tau}$ with $\tau = 2$ days:

| Lag $\ell$ (days) | 1 | 2 | 5 | 10 | 30 |
|---|---|---|---|---|---|
| night-centered, $\hat\rho^{\mathrm{rem}}$ | 0.50 | 0.36 | 0.33 | 0.35 | 0.20 |
| day-centered, $\hat\rho^{\mathrm{rem}}$ | 0.58 | 0.46 | 0.31 | 0.22 | 0.13 |
| $c_\tau(\ell)$, $\tau = 2$ d | 0.61 | 0.37 | 0.08 | 0.007 | 3 × 10⁻⁷ |

The correlation drops quickly over the first day or two and then decays over
weeks: at least two timescales, where the current term has one of two days.

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
for weeks, bounds the evidence the data carry in its directions. The current
$R$ has no such component: its correlation vanishes within about a week, so
each week, and each year's repeat of the same seasonal bias, counts as new
evidence, and the evidence grows with $n_k$. That is how nine springs of one
structural bias can move $Q_{10}$ to 1.0 and the optimum photosynthesis
temperature to 30 °C, both far outside their priors.

### What the two towers show

US-Ha1 and US-xHA stand in the same pixel. For tower $i$, write its window
mean as

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
\hat\sigma_r^2 = \tfrac12\Big(\widehat{\operatorname{Var}}(d_W) - \overline{\big(\sigma^{\mathrm{obs},(1)}\big)^2 + \big(\sigma^{\mathrm{obs},(2)}\big)^2}\Big).
$$

If the footprints overlap, $r^{(1)}$ and $r^{(2)}$ are correlated with some
$\kappa \ge 0$, $\operatorname{Var}(r^{(1)} - r^{(2)}) = 2\sigma_r^2(1 - \kappa)$,
and $\hat\sigma_r^2$ estimates $\sigma_r^2(1 - \kappa) \le \sigma_r^2$: a
lower bound. Over the windows both towers keep, which fall in 2019-2020 only:

| Source | windows | $\widehat{\operatorname{Var}}(d_W)$ | mean $\sum_i \big(\sigma^{\mathrm{obs},(i)}_W\big)^2$ | $\hat\sigma_r$ | $\hat\sigma_r^2 / \hat s_k^2$ |
|---|---|---|---|---|---|
| night-centered | 52 | 1.22 | 0.42 | 0.63 | 0.20 |
| day-centered | 135 | 5.12 | 1.73 | 1.30 | 0.19 |

The difference is also systematic: by day in July-September its mean is
$+1.75$ µmol m⁻² s⁻¹ (47 windows), US-xHA measuring more uptake than
US-Ha1, so part of $r^{(i)}$ is a seasonal bias of each tower.

What the difference cannot see is SIPNET's structural error. The calibration
residual is

$$
\rho_W(\theta) = \big(f_W - \mathcal H_W(\mathcal M(\theta))\big) + e^{(1)}_W + r^{(1)}_W ,
$$

and the model's error $f_W - \mathcal H_W(\mathcal M(\theta))$, common to
both towers, cancels in $d_W$. By the last column, the towers account for
about a fifth of the non-measurement variance; the rest is the model's. The
towers therefore give a floor on the discrepancy, not an estimate of it,
from two years of overlap only.

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
it: the current single term, the special case $\sigma_\ell = \sigma_{\mathrm
p} = 0$, refitted; and a recurring term that drifts from year to year, its
periodic factor multiplied by $e^{-|t_W - t_{W'}|/\tau_{\mathrm p}}$ with
$\tau_{\mathrm p}$ in years, of which the model above is the limit
$\tau_{\mathrm p} \to \infty$.

The two NEE blocks stay independent of each other. Whether they should be,
the night and day windows of one day sharing a respiration error, is checked
by the cross-correlation of their residuals before the model is adopted.

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

   over the logarithms of the six parameters, by L-BFGS with the gradient
   from JAX, from the current values and the scales above (for example
   $\tau_{\mathrm s} = 1$, $\tau_\ell = 30$ days, $\lambda = 0.5$) and from a
   few perturbed starts, keeping the best. Each evaluation is one Cholesky
   factorization of $R_k$, of side $n_k \le 1819$; no SIPNET run is needed.
   The residuals are taken with mean zero: the recurring term carries the
   bias.
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
   predictive check above, applied to the validation vector, should pass:
   $2\,\Phi^{\mathrm{val}}_k / n^{\mathrm{val}}_k$ near 1, within about
   $\sqrt{2 / n^{\mathrm{val}}_k}$, and the predictive intervals covering at
   their nominal rates. On held-out data the check is not conservative, since
   those data did not form the posterior. The three variants are ranked by
   the held-out log predictive density,

   $$
   \log p\big(y^{\mathrm{val}} \mid y\big) \approx \log \frac1J \sum_{j=1}^{J} \mathcal N\big(y^{\mathrm{val}};\, G^{\mathrm{val}}(\theta_j),\, R^{\mathrm{val}}\big),
   $$

   and by $\ell_k(\hat\phi_k)$ with a penalty for the parameters added
   (AIC), which have to agree for the choice to stand.
5. **One iteration.** The residuals of step 1 came from a calibration with
   the old $R$. EKI is rerun with $R(\hat\phi)$, and steps 1-2 repeated on its
   residuals; if every $\hat\sigma$ and $\hat\tau$ moves by less than about
   10%, $\hat\phi$ stands, and otherwise the loop runs once more.

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
