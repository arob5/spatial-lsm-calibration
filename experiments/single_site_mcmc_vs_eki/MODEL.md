# The calibration's models: observations, error models, noise, prior and algorithms

The mathematics of the single-site calibration: what is observed and how the
model predicts it, the NEE error models, the noise model with its scales,
the six models, the parameterization and prior, the algorithms, the
diagnostics and metrics every run is checked by, and how the earlier
calibrations led to this design. `README.md` says how to run it; the code is
in `models.py`, `model/` and `algorithms/`, and every constant named here is
`config.py`'s.

## The observation model

The calibration conditions on a vector $y \in \mathbb{R}^N$ through

$$
y = \mathcal{H}\big(\mathcal{M}(\theta)\big) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}\big(0, R(s)\big),
$$

where $\mathcal{M}(\theta)$ is one SIPNET run over the prepared drivers at
parameters $\theta$, $\mathcal{H}$ stacks the observation operators of the
$K = 5$ observation sources below, and $R(s)$ is the noise covariance, block
diagonal over the sources with a scale per source ("The noise model"). The
sources, their observed values and measurement errors, and the operators are
`model/observations.py`'s `calibration_observation_vector()`; each operator
is bound to its source in `config.OBSERVATION_OPERATORS`. $y$ is ordered by
source in the order below, then by site and time.

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

## The NEE error model

This section builds the error model of one NEE source one assumption at a
time, with each formula beside its reason. "The noise model" below assembles
every source's.

### Two sources of error

For a window $W$, let $f_W$ be the true mean NEE of the tower's footprint
over $W$. The residual at parameters $\theta$ splits as

$$
y_W - \mathcal H_W\big(\mathcal M(\theta)\big)
= \underbrace{\big(y_W - f_W\big)}_{e_W:\ \text{measurement}}
+ \underbrace{\big(f_W - \mathcal H_W(\mathcal M(\theta))\big)}_{\delta_W:\ \text{discrepancy}} .
$$

Both are taken Gaussian, independent of each other, and independent of
$\theta$ near the best $\theta$, so $C = \Sigma^{\mathrm{obs}} +
\Sigma^\delta$. The measurement error is known from the tower ("Measurement
error" above: diagonal, a median of about 0.7 µmol m⁻² s⁻¹ at night and 1.0
by day). The residuals are about 1.6 and 3.2, so almost all of the error is
discrepancy.

### Discrepancy: a process in time

$\delta_W$ is what SIPNET gets wrong: its structure, the drivers, and the
footprint against the pixel. Those errors come from states that change
slowly (leaf area, phenology timing, soil water) and from processes that
follow the weather, so nearby windows err alike. The simplest model with
that property is a stationary Gaussian process in the window ends $t_W$, in
days:

$$
\operatorname{Cov}(\delta_W, \delta_{W'}) = v^\delta\, \rho(t_W - t_{W'}), \qquad \rho(0) = 1,
$$

$v^\delta$ the discrepancy variance of one window and $\rho$ its memory,
a weighted sum of

$$
\kappa_\tau(\Delta) = e^{-|\Delta|/\tau}
\qquad\text{and}\qquad
\kappa^{\mathrm{per}}_\lambda(\Delta) = \exp\!\Big(-\frac{2\sin^2(\pi\Delta/P)}{\lambda^2}\Big),\ P = 365.25\ \text{d}:
$$

a short exponential ($\tau$ about a day) for weather-scale error, a long one
($\tau$ of weeks) for an anomaly that lasts a season, and the periodic
kernel for a bias that recurs at the same time every year.

### Why memory matters

If $n$ windows share one bias $b \sim \mathcal N(0, \sigma_b^2)$ on top of
independent errors of variance $\sigma_e^2$, then $C = \sigma_e^2 I +
\sigma_b^2 \mathbf 1\mathbf 1^\top$ and the information about a constant
shift of the predictions is

$$
\mathbf 1^\top C^{-1} \mathbf 1 = \frac{n}{\sigma_e^2 + n\sigma_b^2} \;\xrightarrow[n\to\infty]{}\; \frac{1}{\sigma_b^2}:
$$

however many windows there are, the posterior variance of the shift stays
above $\sigma_b^2$. The same holds along any direction $v$ in which a
parameter moves the predictions, such as the seasonal shape $Q_{10}$
changes: an error of variance $\sigma^2$ along $v$ caps $v^\top C^{-1} v$
at $\|v\|^2/\sigma^2$ (Sherman–Morrison). The effective number of windows
along $v$,

$$
n_{\mathrm{eff}}(v) = \bar\sigma^2\, \frac{v^\top C^{-1} v}{\|v\|^2/n},
\qquad \bar\sigma^2 = \operatorname{mean}_W C_{WW},
$$

is the number of independent windows of the same variance that would carry
the same information. For the day-centered source ($n = 1819$), at the same
per-window variance, a seasonal cosine counts as about 850 windows under
short memory, 115 under short and long memory, and 50 with the recurring
term added.

The same idea by frequency: for a stationary error the likelihood's
quadratic form is approximately (Whittle)

$$
r^\top C^{-1} r \approx \sum_j \frac{|\hat r(\omega_j)|^2}{S(\omega_j)},
\qquad S_{v\kappa_\tau}(\omega) = \frac{2 v \tau}{1 + \omega^2\tau^2},
$$

so a misfit is cheap at the frequencies where the error's spectrum $S$ is
large. A short exponential's spectrum is nearly flat, and a slow misfit (a
missed seasonal amplitude) costs as much as a fast one of the same energy; a
long term raises the spectrum at low frequencies and makes slow misfits
cheap; the periodic term makes a misfit repeating every year cheaper still.
SIPNET cannot match both the summer amplitude and the day-to-day variation
of daytime NEE ("How we got here"), so the error model decides which one the
calibration gives up.

### The three NEE error models

For NEE source $k$ under error model $L$,

$$
C^L_k = \Sigma^{\mathrm{obs}}_k + v^\delta_k P^L_k, \qquad
(P^L_k)_{WW'} = \rho^L_k(t_W - t_{W'}), \qquad
\rho^L_k = \sum_m w_m \kappa_m,\ \ w_m = \frac{\sigma_m^2}{\sum_{m'}\sigma_{m'}^2},
$$

with $v^\delta_k$ the same for every $L$, $1.43^2$ at night and $2.99^2$ by
day (`config.NEE_DISCREPANCY_STANDARD_DEVIATION`: the first calibration's
residual standard deviation beyond measurement error, $\hat s_k$ of "How we
got here"), so the error models differ in memory alone
(`model/nee_error.py`). The terms' standard deviations $\sigma_m$ and
timescales are the maximum-likelihood fits of `run/fit_nee_discrepancy.py`
to the first calibration's residuals, used for their shape only:

| Error model | night: terms ($\sigma$, $\tau$ or $\lambda$) | day: terms | what it says |
|---|---|---|---|
| `short_memory` | short ($\tau$ 0.74 d) | short (1.72 d) | errors forget within days |
| `long_memory` | short (0.60, 0.74 d), long (1.20, 57.2 d) | short (1.96, 1.72 d), long (2.30, 36.5 d) | plus season-long anomalies |
| `recurring_bias` | short (0.69, 1.17 d), long (0.79, 64.2 d), recurring (0.72, $\lambda$ 0.437) | short (2.02, 1.78 d), long (1.68, 46.6 d), recurring (1.30, 0.208) | plus a bias repeating every year |

The weights are then 0.20 and 0.80 (night) and 0.42 and 0.58 (day) for
`long_memory`, and 0.29, 0.39, 0.32 (night) and 0.48, 0.33, 0.20 (day) for
`recurring_bias`. The models nest: the short term of `short_memory` is
`long_memory`'s.

## The noise model

$R$ is block-diagonal over the sources, no two sources' errors correlated,
and each block is a reference covariance $C_k$ times a scale:

$$
R(s) = \operatorname{diag}\big(s_1 C_1, \dots, s_K C_K\big), \qquad
s_k \overset{\mathrm{ind}}{\sim} \mathrm{IG}(a, b),\ a = 2,\ b = \operatorname{median}\mathrm{Gamma}(2, 1) = 1.678,
$$

so that $\operatorname{median} s_k = 1$, with 90% of the prior in $[0.35,
4.7]$ (the noise standard deviation multiplied by 0.59 to 2.2). The scales
are those of NEE by night, NEE by day and MODIS LAI
(`config.NOISE_SCALED_SOURCES`); LandTrendr (6 values) and SoilGrids (1) are
too few to inform one, and their $s_k$ is 1. A **model with fixed noise**
holds every $s_k$ at 1, its prior median; a **model with inferred noise**
leaves them unknown. In the code (`model/noise.py`, `models.py`) each source
is a Gaussian noise factor whose covariance is a covariance spec,
`ScaledSpec` over the sum of its terms, one block per site, and holding a
scale is observing it at 1.

The reference covariances:

- **NEE**: $C^L_k = \Sigma^{\mathrm{obs}}_k + v^\delta_k P^L_k$, above, in
  the window ends $t_W = w^+$.
- **MODIS LAI**:
  $C_{ii'} = \max(\sigma_i, 0.66)^2\, \mathbf 1[i = i'] + 0.5^2\,
  e^{-|t_i - t_{i'}|/30}\, \mathbf 1[\mathrm{year}(t_i) = \mathrm{year}(t_{i'})]$:
  the floored product spread, plus discrepancy correlated within a summer
  (the 0.66 floor is the reanalysis's, `data/README.md` open question 22).
- **LandTrendr**, with $\sigma = (\sigma_j)_j$ and $y = (y_j)_j$:
  $C = \sigma\sigma^\top + \kappa^2 yy^\top + 5^2 I$, $\kappa = 0.02/0.48$:
  LandTrendr's error and the carbon fraction's uncertainty (IPCC 0.46-0.50),
  each shared by every year, plus independent discrepancy.
- **SoilGrids**: $C = \sigma^2 + (0.25\, y)^2$, the second term for the depth
  and definition SIPNET's single soil pool does not share with a 0-200 cm
  stock (`data/README.md` open question 21).

### Two closed forms

Write $r_k = y_k - \mathcal G_k(\theta)$ and $q_k(\theta) = r_k^\top
C_k^{-1} r_k$. With the likelihood raised to a tempering power $\varphi
\in [0, 1]$,

$$
\mathcal N(y_k; \mathcal G_k, s_k C_k)^{\varphi}\, \mathrm{IG}(s_k; a, b)
\propto s_k^{-(a + \varphi n_k/2) - 1} \exp\!\Big(-\frac{b + \varphi q_k/2}{s_k}\Big)
\;\Rightarrow\;
s_k \mid \theta, y \sim \mathrm{IG}\Big(a + \frac{\varphi n_k}{2},\ b + \frac{\varphi q_k(\theta)}{2}\Big),
$$

an exact Gibbs step that needs no SIPNET run, and

$$
p(y_k \mid \theta) = \int \mathcal N(y_k; \mathcal G_k, s C_k)\,\mathrm{IG}(s; a, b)\,ds
= t_{2a}\Big(y_k;\ \mathcal G_k(\theta),\ \frac{b}{a} C_k\Big)
\propto |C_k|^{-1/2}\Big(b + \frac{q_k(\theta)}{2}\Big)^{-(a + n_k/2)},
$$

the likelihood of $\theta$ with the scale integrated out, so that the
posterior of $\theta$ alone ($D = 15$) is the target of importance
sampling, SMC and MCMC under inferred noise. `model/likelihood.py` computes
both from the predictions; its values equal the library's own posteriors'
(the fixed posterior's Gaussian and `FactoredDistribution.marginalize`'s
Student-t) to rounding.

The data dominate each scale: its conditional has shape $a + n_k/2$ (402 at
night, 912 by day, 46 for LAI), a relative standard deviation of about
$\sqrt{2/n_k}$. So a model with inferred noise behaves much like one fixed
at $\hat s_k \approx q_k(\hat\theta)/n_k$: it rescales each source's weight
by $1/\hat s_k$, widening or narrowing the posterior and shifting the
balance between NEE, LAI and the pools, but it does not change which
timescales are cheap, which is $\rho^L$'s.

An inverse-Wishart prior on a whole $C_k$ is not used: with one realization
of $y_k$ it updates to $\mathrm{IW}(\nu + 1, \Psi + r_k r_k^\top)$, a
rank-one change to a matrix of side up to 1819. A scale on the discrepancy
term alone, $\Sigma^{\mathrm{obs}}_k + s_k v^\delta_k P_k$, would be more
physical (the measurement error is known) but is not conjugate; $s_k$ here
also scales the measurement error, about 19% of $C_{WW}$ at night and 10% by
day.

## The models

A model is an NEE error model crossed with a noise treatment
(`models.py`), six in all:

$$
\theta \sim \pi_0, \qquad
y_k \mid \theta, s \sim \mathcal N\big(\mathcal G_k(\theta),\ s_k C^L_k\big),
\qquad s_k \sim \mathrm{IG}(a, b) \text{ (inferred) or } s_k = 1 \text{ (fixed)},
$$

named `<error model>/<noise>`, such as `long_memory/inferred`. Every model
shares the prior $\pi_0$ ("Parameterization and prior"), the forward map
$\mathcal G = \mathcal H \circ \mathcal M$, and the noise of the unscaled
sources. The held-out vector, US-xHA's NEE windows of 2021-2024, has the
same error model with its own measurement errors.

## The algorithms

`algorithms/`; every run writes its samples, their weights and its cost in
one format (`algorithms/records.py`).

### EKI (fixed noise)

An ensemble of $J = 100$ prior draws moves through the tempered posteriors
$\pi_\varphi(\theta) \propto \pi_0(\theta)\, \mathcal N(y; \mathcal
G(\theta), R)^{\varphi}$ by the perturbed-observation update

$$
\theta_j \leftarrow \theta_j + \hat C_{\theta g}\big(\hat C_{gg} + R/\Delta\varphi\big)^{-1}\big(y - \mathcal G(\theta_j) - \eta_j\big),
\qquad \eta_j \sim \mathcal N(0, R/\Delta\varphi),
$$

each increment the largest $\Delta\varphi \le 1 - \varphi$ with
$\mathrm{ESS}(e^{-\Delta\varphi\, \Phi_j}) \ge J/2$, until $\varphi = 1$
(EnsKit's driver, `AdaptiveESSSchedule` and `Matheron`). The final ensemble
is evaluated once more, for its predictions.

### EKI with Gibbs noise updates (inferred noise)

The tempered target is joint,

$$
\pi_\varphi(\theta, s) \propto \pi_0(\theta) \prod_k \mathrm{IG}(s_k; a, b) \prod_k \mathcal N\big(y_k; \mathcal G_k(\theta), s_k C_k\big)^{\varphi},
$$

and each stage $\varphi \to \varphi' = \varphi + \Delta\varphi$, at an
ensemble $(\theta_j, s_j)$ with known predictions $g_j$ (Botha et al. 2023,
*Inverse Problems* 39, 125014, with their noise step made an exact draw):

1. **Increment**, from the ratio of the $s$-marginal targets: with
   $A_k(\varphi) = a + \varphi n_k/2$ and $B_{jk}(\varphi) = b + \varphi
   q_{jk}/2$,
   $$
   \log w_j = \sum_{k\ \mathrm{scaled}}\big[A_k(\varphi)\log B_{jk}(\varphi) - A_k(\varphi')\log B_{jk}(\varphi')\big] - \frac{\Delta\varphi}{2}\sum_{k\ \mathrm{unscaled}} q_{jk},
   $$
   $\Delta\varphi$ the largest with $\mathrm{ESS} \ge J/2$. The weights
   are the ratio $\pi_{\varphi'}(\theta_j)/\pi_\varphi(\theta_j)$ of the
   targets actually tempered, so they carry the $\log\det$ terms that
   per-particle scales bring; EnsKit's schedule on whitened residuals would
   drop them.
2. **Theta step**, the perturbed-observation update over $\Delta\varphi$,
   in one of two versions (below).
3. **Evaluate** $g_j = \mathcal G(\theta_j)$: the stage's one batch of runs.
4. **Scale step**, exact Gibbs for $\pi_{\varphi'}$:
   $s_{jk} \sim \mathrm{IG}(a + \varphi' n_k/2,\ b + \varphi' q_{jk}/2)$.

At $\varphi = 0$ the ensemble is a prior draw of both.

**The two theta steps.** Write $T$ and $U$ for the centered ensembles of
$\theta$ and $g$ over $\sqrt{J-1}$ (rows are particles), so $\hat C_{\theta
g} = T^\top U$ and $\hat C_{gg} = U^\top U$:

$$
\textbf{common:}\ \ \theta_j' = \theta_j + T^\top U\big(U^\top U + R(\bar s)/\Delta\varphi\big)^{-1}(y - g_j - \eta_j),\ \ \eta_j \sim \mathcal N\big(0, R(\bar s)/\Delta\varphi\big),\ \ \bar s_k = \Big(\tfrac1J\textstyle\sum_j s_{jk}^{-1}\Big)^{-1};
$$

$$
\textbf{per-particle:}\ \ \theta_j' = \theta_j + T^\top U\big(U^\top U + R(s_j)/\Delta\varphi\big)^{-1}(y - g_j - \eta_j),\ \ \eta_j \sim \mathcal N\big(0, R(s_j)/\Delta\varphi\big),
$$

the latter with the parts of $T$ and $U$ explained by $\log s_j$ first
regressed out. Neither is exact even for a linear $\mathcal G$: for fixed
$s$ the update carries $\pi_\varphi(\theta \mid s)$ exactly to
$\pi_{\varphi'}(\theta \mid s)$ only with a gain built from
$\operatorname{Cov}(\theta, g \mid s)$, and the ensemble estimates the
marginal covariance, which by the law of total covariance exceeds
$\mathbb E[\operatorname{Cov}(\theta \mid s)]$ by
$\operatorname{Cov}(\mathbb E[\theta \mid s])$. The per-particle gain uses
the right $R(s_j)$ but too large a covariance, and particles with different
$s_j$ move by different amounts, so it tends to over-disperse; regressing
out $\log s$ estimates the conditional covariance. The common gain ignores
the spread of $s$ and tends to under-disperse; $\bar s$ is the mean
precision because the gain reads $R^{-1}$. The two coincide as
$\operatorname{Var}(s \mid y)$ shrinks, which here is fast. Both share a
lag: after the theta step the pairs are near $\pi_\varphi(s)\,
\pi_{\varphi'}(\theta \mid s)$, and one Gibbs draw moves them toward
$\pi_{\varphi'}(\theta, s)$ but not all the way. Importance sampling
afterwards corrects what remains.

By the push-through identity both are computed per particle with a $J
\times J$ solve,

$$
\theta_j' = \theta_j + T^\top\Big(I_J + \Delta\varphi \sum_k \frac{1}{s_{jk}} M_k\Big)^{-1}
\Delta\varphi \sum_k \frac{1}{s_{jk}}\, U_k C_k^{-1} d_{jk},
\qquad M_k = U_k C_k^{-1} U_k^\top,\ d_j = y - g_j - \eta_j,
$$

$U_k$ source $k$'s columns of $U$ and $s_{jk} = \bar s_k$ for the common
gain.

### Importance sampling and SMC after EKI

EKI's ensemble is not a draw from a known density, so a density is fitted to
it and fresh draws are corrected. The target is the model's posterior of
$\theta$, $\pi(\theta) \propto \pi_0(\theta)\, p(y \mid \theta)$, Gaussian
at $s = 1$ for fixed noise and the Student-t marginal for inferred noise.
The base density is

$$
q = 0.9\; t_5\big(\hat m, 1.5\, \hat C\big) + 0.1\; \pi_0,
$$

the Student-t fitted to the seed EKI run's valid final ensemble, mixed with
the prior, which bounds the weights by $10\,\pi/\pi_0$. **Importance
sampling** draws $\theta_m \sim q$, $m = 1, \dots, 1000$, with

$$
\log \tilde w_m = \log\pi_0(\theta_m) + \log p(y \mid \theta_m) - \log q(\theta_m),
\qquad W_m = \tilde w_m / \textstyle\sum_{m'} \tilde w_{m'},
$$

and reports $\mathrm{ESS} = 1/\sum W_m^2$, the PSIS $\hat k$ and
$\log\hat Z = \log\frac1M\sum\tilde w_m$. **Tempered SMC** (500 samples)
moves through $\pi_\beta \propto q^{1-\beta}(\pi_0 L)^\beta$ from the same
$q$, each increment keeping the conditional ESS at half, each stage
resampling and then moving every sample by Metropolis-Hastings (an
independent $t_5$ or a random walk) until 99% have moved (`smc.py`).
Under inferred noise the scales' posterior follows from the stored
predictions with no new run,
$p(s_k \mid y) \approx \sum_m W_m\, \mathrm{IG}(s_k; a + n_k/2, b + q_k(\theta_m)/2)$,
and each sample carries one draw of it.

### MCMC

The target is the same $\pi(\theta)$. $C$ chains ($2 \times$ the workers)
advance in lockstep, so each step is one batch of runs:

$$
\theta'_c = \theta_c + \frac{2.38}{\sqrt D} L z_c,\quad z_c \sim \mathcal N(0, I),
\qquad \text{accepted with probability } \min\big(1, \pi(\theta'_c)/\pi(\theta_c)\big).
$$

The chains start from an importance-sampling run's draws, resampled by
weight among those whose runs succeeded, and $LL^\top$ starts as that run's
weighted covariance. During the first 500 steps it is re-estimated every 50
from the second half of all chains' history, then frozen; the first 1000 of
the 2000 steps are discarded, so the kept draws are a Metropolis chain with
a fixed kernel. Under inferred noise each kept draw carries one draw of the
scales from their conditional.

### Cost

Every run records, per phase, the SIPNET runs, the forward-map calls (each
one batch through PyEns), the wall time, the workers and the node
(`cost.json`). A run seeded by another (IS and SMC by EKI, MCMC by IS)
copies its seed's phases, so its total is the whole pipeline's and the
seed's share is separate. SIPNET runs are the primary measure, since wall
time depends on the node; every SCC job asks for the same 28 slots.

## Parameterization and prior

`model/prior.py` declares the parameters, a prior factor for each, and the
SIPNET parameter map; this section says how they were found. The records are in `exploration/parameter_analysis/`
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
| photosynthetic capacity $P$ (nmol g⁻¹ s⁻¹) | log-normal, 95% in 140-450 | 161 / 252 / 402 | `aMax`, by `photosynthesis_rules` |
| respiration share $\rho$ | logit-normal, 95% in 0.04-0.20 | 0.05 / 0.09 / 0.18 | `baseFolRespFrac`, same rule |
| optimum photosynthesis temperature (°C) | normal(22, 2.5) | 17.6 / 21.8 / 26.4 | `psnTOpt`; `psnTMin` at a fixed range, by a `Compute` rule |
| half-saturation light (mol m⁻² d⁻¹) | log-normal, 4.6-26.3 | 5.0 / 10.1 / 21.4 | `halfSatPar` |
| soil water holding capacity (cm) | log-normal, 15-150 | 18 / 52 / 123 | `soilWHC` |
| leaf growth at leaf-on (g C m⁻²) | log-normal, 50-180 | 60 / 91 / 149 | `leafGrowth` |
| leaf-on growing degree-days | log-normal, 500-1100 | 552 / 741 / 1008 | `gddLeafOn` |
| allocation (leaf, wood, fine root, coarse root) | softmax-normal about (0.18, 0.45, 0.065, 0.305) | leaf 0.11-0.25, wood 0.34-0.57 | `leafAllocation`, `woodAllocation`, `fineRootAllocation` |
| wood respiration rate at 10 °C (yr⁻¹) | log-normal, 0.006-0.04 | 0.007 / 0.015 / 0.035 | `baseVegResp` and `baseCoarseRootResp`, each $r_{10}/Q_{10}$, by `Compute` rules |
| soil respiration flux at 10 °C, $F_{10}$ (g C m⁻² yr⁻¹) | log-normal, 200-900 | 230 / 423 / 855 | `baseSoilResp` $= F_{10} / (1000\, C_{s,0}\, Q_{10})$, by a `Compute` rule |
| soil respiration $Q_{10}$ | log-normal, 1.3-3.2 | 1.37 / 2.08 / 2.91 | `soilRespQ10` |
| initial wood carbon (kg C m⁻²) | log-normal fitted to the site's initial-condition members | 2.9 / 7.3 / 23.5 | `plantWoodInit`, by `initial_condition_rules` |
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

The recorded prior predictive, `run/prior_predictive.py`'s 200 draws
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

## Diagnostics

Every run of the experiment is diagnosed the same way, so that runs compare
directly, under its own model's $R$: for inferred noise, at the posterior
median of each scale. A run is diagnosed from its posterior predictive
(`run/predict.py`): `run/diagnose.py --model <model> --run <run>` computes
the quantities below (`model/diagnostics.py`) and writes them as tables
under the run's `diagnostics/`; `--run prior` diagnoses the prior
predictive.

### Notation

A diagnosis reads an ensemble $\theta_1, \dots, \theta_J$ and its
predictions of one observation vector, from a run's posterior predictive:
up to 300 of its samples (drawn by weight from a weighted run), or the prior
predictive's draws. The residual of
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
- **The draws may be approximate.** EKI's $\theta_j$ approximate the
  posterior; importance sampling, SMC and MCMC correct them, and comparing
  the check across a model's runs shows how much the approximation moves
  it.
- **On the prior predictive** the same computation, with prior draws, is a
  prior predictive check, which asks whether the prior's draws could have
  produced the data, not whether the fitted model describes them.
- **It tests consistency with $R$, and nothing more.** An $R$ that makes a
  bias cheap, such as one shared by every year, passes a fit that has the
  bias. The check is read with the coverage, the seasonal means of the
  residuals and the weekly residuals ("NEE's residuals"), never alone: the
  three-term run of "How we got here" passed it and missed the summer
  uptake by half.

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

**Slow and fast** (`nee_slow_fast.csv`). Each daily series $x_d$, one value
per day $d$ of $w^-$ (the observations $y$, the median prediction $\hat g$,
and their residual $\hat\rho$), splits into a slow part, its centered
running mean over 31 days,

$$
x^{\mathrm{slow}}_d = \operatorname{mean}\{x_{d'} : |d' - d| \le 15,\ x_{d'} \text{ observed}\}
\quad (\text{defined where at least 5 are}),
\qquad
x^{\mathrm{fast}}_d = x_d - x^{\mathrm{slow}}_d ,
$$

the fast part being the day-to-day variation about it. The table reports
$\operatorname{Var}(\hat\rho^{\mathrm{slow}})$ and
$\operatorname{Var}(\hat\rho^{\mathrm{fast}})$, where the misfit sits; the
correlation of $y^{\mathrm{fast}}$ with $\hat g^{\mathrm{fast}}$, how well the
model follows the day-to-day variation; and
$\operatorname{sd}(\hat g^{\mathrm{fast}}) / \operatorname{sd}(y^{\mathrm{fast}})$,
how large the model's day-to-day variation is against the observed. None of
these depends on $R$.

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

### The held-out score

The expected log predictive density of US-xHA's NEE windows of 2021-2024,
which no calibration saw, is a proper score, so it ranks models and
algorithms. For held-out source $k$ and the $K$ samples a predictive runs,

$$
\mathrm{ELPD}^{\mathrm{val}}_k = \log \frac1K \sum_{m=1}^{K} \mathcal N\big(y^{\mathrm{val}}_k;\ \mathcal G^{\mathrm{val}}_k(\theta_m),\ s_{mk} C^{L,\mathrm{val}}_k\big),
$$

$s_{mk} = 1$ under fixed noise, reported per source with its Monte Carlo
standard error (`predictive/heldout_scores.csv`).

### Comparing algorithms

Against a reference run of the same model (MCMC, else SMC), with mean
$m^{\mathrm{ref}}$ and covariance $\Sigma^{\mathrm{ref}}$: per parameter,
in natural units, the mean error $|m_i - m^{\mathrm{ref}}_i|/\sigma^{\mathrm{ref}}_i$
and the ratio $\sigma_i/\sigma^{\mathrm{ref}}_i$; over all of $\theta$, the
Gaussian Kullback-Leibler divergence

$$
\mathrm{KL}\big(\mathcal N(m^{\mathrm{ref}}, \Sigma^{\mathrm{ref}}) \,\|\, \mathcal N(m, \Sigma)\big)
= \tfrac12\Big[\operatorname{tr}(\Sigma^{-1}\Sigma^{\mathrm{ref}}) + (m - m^{\mathrm{ref}})^\top \Sigma^{-1}(m - m^{\mathrm{ref}}) - D + \log\frac{|\Sigma|}{|\Sigma^{\mathrm{ref}}|}\Big];
$$

beside them the cost ("The algorithms", "Cost"), and for importance sampling
and SMC the ESS, $\hat k$ and $\log\hat Z$.


## How we got here

Three EKI calibrations with fixed $R$, before the present models, each
recorded with its `provenance.json`, showed why the NEE error model is a
choice the experiment varies. Their numbers are the evidence for the error
models above.

**The first calibration failed its check.** Its NEE discrepancy was a single
exponential term, $\sigma_\delta$ of 1.0 at night and 1.8 by day over
$\tau = 2$ days. On the observed data every member's misfit was about 31
standard deviations above $N/2$ ($2\Phi/N = 1.84$, $p \approx 10^{-135}$),
while its synthetic twin passed: no $\theta$ EKI reached made the residuals
as small as $R$ said. The residuals, in µmol m⁻² s⁻¹:

| Source | $\widehat{\operatorname{Var}}(\hat\rho)$ | mean measurement variance | $\hat s_k$ | mean residual DJF / MAM / JJA / SON | autocorrelation at 1, 5, 30 days |
|---|---|---|---|---|---|
| night-centered | 2.51 | 0.47 | 1.43 | 0.02 / 0.84 / 1.38 / 0.88 | 0.62, 0.54, 0.34 |
| day-centered | 10.01 | 1.05 | 2.99 | −0.27 / −1.40 / −1.94 / −0.46 | 0.63, 0.39, 0.12 |

The error was about twice the model's, biased the same way every spring to
autumn (observed night respiration and day uptake both above the model's),
and correlated over weeks where $R$'s correlation vanished within a week.
The two towers, US-Ha1 and US-xHA in one pixel, put a floor of 0.63 and 1.30
on the representativeness error, a fifth of $\hat s_k^2$: the rest is the
model's.

**Fitted discrepancies, and the trade-off they exposed.** Short, long and
recurring terms were fitted by maximum likelihood to those residuals
(`run/fit_nee_discrepancy.py`; three terms beat one by 55 and 71 in log
likelihood and by 22 and 24 nats on the held-out tower). Calibrating under
the three-term fit (setup `three_term_discrepancy`) and the two-term fit
(`two_term_discrepancy`) then passed the predictive check ($2\Phi/N$ 1.05
and 1.08) and gave up most of the summer daytime uptake:

| | one term | three terms | two terms |
|---|---|---|---|
| day residual, mean over June-August | −1.9 | −8.9 | −7.4 |
| day residual, $\operatorname{Var}(\hat\rho^{\mathrm{slow}})$ / $\operatorname{Var}(\hat\rho^{\mathrm{fast}})$ | 3.8 / 5.7 | 16.5 / 3.7 | 11.9 / 3.9 |
| day, $\operatorname{sd}(\hat g^{\mathrm{fast}})/\operatorname{sd}(y^{\mathrm{fast}})$ | 1.09 | 0.68 | 0.76 |
| photosynthetic capacity $P$, median | 268 | 207 | 221 |
| half-saturation light, median | 15.4 | 29.0 | 25.7 |

One parameter set does not give SIPNET both the observed summer uptake and
the observed day-to-day variation of daytime NEE. Short memory weighs slow
and fast errors alike, and the calibration fits the seasonal amplitude at
the cost of daily NEE that varies more than the observed; long or recurring
memory makes slow errors cheap, and the calibration damps the light
response to fit the fast variation, at the cost of the summer mean. The
recurring term, perfectly correlated across years, made a shared summer
bias nearly free, and the refit on that run's residuals asked for 3.5 times
more of it: a discrepancy flexible enough to absorb what the parameters
should explain takes the signal (Brynjarsdóttir and O'Hagan 2014, *Inverse
Problems* 30, 114007). In every setup, the model's day-to-day variation of
night NEE stayed a fifth of the observed and the optimum photosynthesis
temperature near 29-30 °C, far above its prior: properties of SIPNET's
configuration at the site, not of $R$.

**Consequences, which the present design follows.** $R$ is part of the
model's specification, stated for what the calibration should reproduce,
and is not refitted to a calibration's residuals. The experiment therefore
compares error models of one per-window size and different memory, takes
their shapes from the first calibration's fits, and lets the data set the
size only through a scale with a prior.
