# Phase 2: the starting parameterization and prior for Harvard Forest (site 4977)

Scripts are `phase2_*.py` in this directory. Intermediate outputs are in `phase2/`.
Deliverables: `prior_proposal.csv`, `fixed_parameters.csv`, `prior_predictive.npz`.
To reproduce, run `phase2_setup.py` (it caches the vector and noise sds), then
`phase2_morris.py 20` and `phase2_morris_analyse.py`, then
`phase2_prior_iterate.py i1..i5` with `phase2_report_iter.py`,
`phase2_diagnose.py` and `phase2_criteria.py`, then `phase2_final.py 250`,
`phase2_item5.py` and `phase2_annotate_fixed.py`. The side studies are
`phase2_phenology.py` and `phase2_water_probe.py`, and the library check is
`phase2_check_library.py`.

## Method

- **Forward runs.** Every run used the fast path, `fast_forward.run_batch`,
  with the full phase-1 base set plus the initial state. The fast predictions
  match the library's to 2.6e-12 on a base run.
- **Scoring.**
  - Misfit is 0.5 Σ z² per source, with z = (y − prediction)/sd and sd =
    sqrt(diag) of that source's noise block.
  - Coverage is the fraction of y inside the 5-95% interval of prediction +
    N(0, sd²) noise, taken over the J runs. The tables also give the fraction
    below and above that interval, and the mean z at the predictive median.
- **Screening.** Morris elementary effects: 29 candidates, 20 trajectories,
  4 levels, Δ = 2/3, so 600 runs, with no failures. Positive parameters with
  wide ranges were sampled on a log scale. Each candidate's range and its
  provenance are in `phase2_morris.py:CANDIDATES`.
- **Re-referenced rates.** A rate "at 10 C" is the SIPNET base rate × Q10.
  SIPNET's own reference is 0 C: `sipnet.c:1066-1067` for wood, 1073-1078 for
  roots, and `depeffects.c` for soil.

## 1. Sensitivity ranking

The table gives mu*, relative to the most influential parameter of each output
(`phase2/morris_relative_mu_star.csv`; absolute mu* and sigma are in
`morris_mu_star.csv`). The "m_" columns are the misfit per source. Score is the
maximum across the columns.

| parameter | m_night | m_day | m_lai | m_agb | m_soc | GPP | Reco | NEE | LAI | Δwood | Δsoil | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| photosynthetic_capacity P | .83 | .79 | .28 | .25 | .12 | 1 | .95 | .57 | .38 | .41 | .33 | 1 |
| initial_soil_organic_carbon | .37 | .20 | 0 | 0 | 1 | 0 | .59 | .53 | 0 | 0 | .59 | 1 |
| soil_respiration_rate_at_10c | .47 | .35 | 0 | 0 | .56 | 0 | 1 | .89 | 0 | 0 | 1 | 1 |
| leaf_on_growth | 1 | 1 | .86 | .36 | .11 | .40 | .61 | .43 | .96 | .68 | .20 | 1 |
| respiration_share rho | .62 | .64 | 1 | .27 | .13 | .21 | .81 | 1 | 1 | 1 | .41 | 1 |
| initial_wood_carbon | .19 | .13 | .13 | 1 | .07 | .04 | .28 | .27 | .15 | .51 | .16 | 1 |
| leaf_allocation | .58 | .28 | .73 | .21 | .06 | .25 | .39 | .22 | .96 | .16 | .16 | .96 |
| half_saturation_light | .33 | .60 | .44 | .30 | .11 | .68 | .45 | .56 | .66 | .48 | .31 | .68 |
| soil_water_holding_capacity | .30 | .39 | .48 | .12 | .15 | .65 | .39 | .59 | .57 | .46 | .30 | .65 |
| wood_allocation | .02 | .03 | .13 | .14 | .02 | .02 | .07 | .08 | .06 | .52 | .11 | .52 |
| light_extinction_coefficient | .12 | .49 | .21 | .06 | .05 | .24 | .13 | .22 | .28 | .15 | .12 | .49 |
| optimum_photosynthesis_temperature | .37 | .32 | .25 | .07 | .05 | .28 | .46 | .20 | .31 | .29 | .13 | .46 |
| water_removal_fraction | .13 | .45 | .21 | .05 | .08 | .19 | .11 | .28 | .21 | .13 | .20 | .45 |
| wood_respiration_rate_at_10c | .21 | .17 | .16 | .16 | .04 | .10 | .44 | .42 | .27 | .44 | .14 | .44 |
| water_use_efficiency | .42 | .39 | .41 | .20 | .05 | .42 | .42 | .25 | .43 | .22 | .12 | .43 |
| leaf_carbon_per_area | .11 | .29 | .35 | .06 | .03 | .15 | .10 | .12 | .43 | .10 | .06 | .43 |
| vapor_pressure_deficit_slope | .11 | .30 | .06 | .05 | .04 | .14 | .07 | .16 | .10 | .12 | .09 | .30 |
| wood_turnover_rate | .02 | .01 | .01 | .10 | .08 | .01 | .03 | .03 | .03 | .29 | .15 | .29 |
| coarse_root_respiration_rate_at_10c | .08 | .06 | .10 | .13 | .03 | .03 | .23 | .24 | .16 | .22 | .07 | .24 |

The remaining candidates all scored ≤ 0.22: the wood and soil Q10s, the
coarse-root turnover rate, gddLeafOn, the soil moisture exponent, the
coarse-root fraction, the leaf turnover rate, leafOffDay, the fine-root
turnover rate and the fine-root allocation. Every sigma/mu* is ≥ 0.85, and
most are 1.2-2, so the interactions are strong.

Two cautions:
- The misfit mu* are dominated by the extreme corners of the design.
- **The screening underrated gddLeafOn.** On a 4-level grid the effect of this
  step function washes out. The prior-predictive diagnosis (section 4) showed
  leaf-on timing to be the largest May error.

## 2. The parameterization (D = 15)

The calibrated parameters and how each reaches SIPNET:
- **photosynthetic_capacity P and respiration_share rho**, through
  ComputePhotosynthesisRates. These replace the four degenerate SIPNET
  parameters, with aMaxFrac 0.861 and cFracLeaf 0.466 fixed at the BETY
  medians.
- **optimum_photosynthesis_temperature.** psnTMin is a DerivedParameter at a
  fixed psnTOpt − psnTMin = 23.96 C, which is phase 1's recommendation 4.
- **half_saturation_light.**
- **soil_water_holding_capacity**, the single water-limitation direction. wue
  and waterRemoveFrac are fixed.
- **leaf_on_growth.**
- **leaf_on_growing_degree_days.**
- **allocation**, a 4-simplex (leaf, wood, fine root, coarse root) through
  CopySimplex. It counts 3 dimensions, and it makes the `ensureAllocation`
  exit (`sipnet.c:1111-1123`) impossible.
- **wood_respiration_rate_at_10c.** Two DerivedParameters set baseVegResp =
  r10/vegRespQ10 and baseCoarseRootResp = r10/coarseRootQ10. Tying the two
  follows vault note 4.3: coarse roots are wood, so they get the same per-C
  rate at 10 C.
- **soil_respiration_flux_at_10c F10 = k10·SOC0**, in g C m-2 yr-1, with
  baseSoilResp = F10/(1000·SOC0·Q10). This reparameterizes the k × SOC ridge:
  the NEE data inform F10, and the soil C observation and soil change inform
  SOC0. With the two independent, (k10, SOC0) put 67% of prior runs at NEE > 0
  (i1).
- **soil_respiration_q10.**
- **initial_wood_carbon and initial_soil_organic_carbon**, through
  ComputeInitialConditions. Initial leaf carbon and moisture stay external
  inputs at the ensemble medians.

What is fixed, and why (`fixed_parameters.csv` gives each value, source and
Morris score):
- **Left out despite scores of 0.43-0.49, as flat or redundant directions:**
  - attenuation and leafCSpWt. They enter only as attenuation/leafCSpWt on the
    flux side, and BETY pins leafCSpWt to ±12%.
  - waterRemoveFrac and wueConst, which duplicate the soilWHC direction.
- **Left out for low scores:** everything scoring ≤ 0.3.
- **One fixed value changed from phase 1:** baseFineRootResp = 0.000509 × 365
  = 0.186 yr-1. This undoes PEcAn's per-day slip (R:461-468) against SIPNET's
  `/365` (`sipnet.c:1902`), as phase 1 found.

`phase2_check_library.py` confirms two things:
- The library map writes the same SIPNET overrides as the NumPy mapping used
  in the iterations. The only differences are the fine-root rate (see open
  issue 1) and rounding of soilWFracInit.
- `Prior.log_prob` is finite.

## 3. Priors

Quantiles are natural values from 250 draws of `Prior.sample`. The full
provenance and the mapping of each parameter are in `prior_proposal.csv`.

| parameter | family (args) | 5 / 50 / 95% | provenance |
|---|---|---|---|
| P (nmol g-1 s-1) | log-normal, interval 140-450 | 161 / 252 / 402 | reasoned: from the BETY Amax·SLA 97.5% (P 170) to top-canopy oak Amax ~13 umol m-2 s-1 (approximate); the template's 228 is inside |
| rho | logit-normal, 0.04-0.20 | .049 / .095 / .178 | BETY area Rd 0.66 over gross Amax 6-15, up to BETY's 0.16 |
| psnTOpt (C) | Normal(22, 2.5) | 17.6 / 21.8 / 26.4 | literature ~20-27 (approximate); BETY 43 rejected |
| halfSatPar | log-normal, 4.6-26.3 | 5.0 / 10.1 / 21.4 | BETY 2.5-97.5% |
| soilWHC (cm) | log-normal, 15-150 | 18 / 52 / 123 | reasoned placeholder; above ~88 there is no water limit (flat) |
| leafGrowth (g m-2) | log-normal, 50-180 | 60 / 91 / 149 | BETY lower end; upper end cut at LAI 5.9 (reasoned) |
| gddLeafOn (K d) | log-normal, 500-1100 | 552 / 741 / 1008 | from the drivers plus approximate Harvard phenology (see below) |
| allocation (leaf, wood, fine, coarse) | softmax_normal, center (.18, .45, .065, .305), sd (.25, .3, .3) | leaf .11/.17/.25; wood .34/.45/.57; fine .04/.065/.10; coarse .24/.31/.37 | BETY leaf and fine; wood raised so that coarse:wood ≈ 0.7 (reasoned) |
| wood rate at 10 C (yr-1) | log-normal, 0.006-0.04 | .007 / .015 / .035 | reasoned, ~130 g C m-2 yr-1 at 8.6 kg C m-2 |
| F10 (g C m-2 yr-1) | log-normal, 200-900 | 230 / 423 / 855 | reasoned from Harvard Rh ~300-700 (approximate literature) |
| soil Q10 | log-normal, 1.3-3.2 | 1.37 / 2.08 / 2.91 | Mahecha et al. 2010 (ecosystem Q10 ~1.4) against the chamber-typical 2-4 |
| initial wood (kg m-2) | log_normal_from_samples(IC ensemble) | 2.9 / 7.3 / 23.5 | site IC ensemble |
| initial SOC (kg m-2) | log_normal_from_samples(IC ensemble) | 4.5 / 17.9 / 64 | site IC ensemble |

Every term is independent. No joint prior is used: the ridges were removed by
reparameterizing (P and rho, F10), and I have no evidence for a correlation
between the remaining terms.

The prior on k10 is implied rather than stated. Its pushforward baseSoilResp is
0.0018-0.061 yr-1 (2.5-97.5%), wider than BETY's 0.004-0.020, because the SOC
ensemble is wide.

## 4. Iteration log

J = 250 in every iteration, and no run failed in any of them. Coverage is the
fraction inside the 90% interval. "ok" is the fraction of runs meeting every
plausibility bound (GPP, NEE, annual peak LAI, and wood within 0.8-2 times its
initial value).

| iter | change (why) | night | day | day JJA | LAI | GPP 5/50/95 | NEE 5/50/95 | NEE>0 | ok |
|---|---|---|---|---|---|---|---|---|---|
| i1 | screening-based starting prior, (k10, SOC) independent | .92 | .78 | .48 | .92 | 509/1210/2013 | −572/+251/2112 | .67 | .13 |
| i2 | F10 reparameterization (the Rh ridge set NEE); P 150-500 (the top-canopy Amax argument); rho 0.04-0.20; soil Q10 calibrated; leafCSpWt fixed | .84 | .81 | .69 | .87 | 853/1611/2368 | −815/−138/591 | .37 | .19 |
| i3 | gddLeafOn calibrated at 500-1100: May daytime NEE was −9.5 against −3.1 observed, because leaf-on came ~25 d early (below); soil Q10 1.3-3.2 (the Mahecha argument) | .86 | .88 | .75 | .87 | 733/1411/2193 | −736/−64/610 | .43 | .22 |
| i4 | soilWHC 15-150: in the probe, WHC 10 gave −5.3 in JJA against −14.8 observed, and 2016 observed uptake shows no drought collapse; leafGrowth ≤ 180; wood rate 0.006-0.04 (Ra/GPP > 1 in 12%) | .83 | .91 | .85 | .87 | 760/1538/2470 | −940/−234/433 | .30 | .22 |
| i5 | P 120-400 (to test the GPP tail) | .85 | .88 | .75 | .87 | 589/1255/2148 | −786/−77/569 | .44 | .21 |
| final | i4 with P 140-450; the library objects; baseFineRootResp fixed as intended | .86 | .91 | .86 | .87 | 789/1451/2635 | −973/−203/562 | .34 | — |

i5 lost JJA daytime coverage without making the plausibility any better, so
the final keeps i4's centre and trims only P's upper tail.

## 5. Final checks

**Coverage** (`phase2/final_coverage.csv`). z is the mean z at the predictive
median.

| group | n | inside | below | above | z |
|---|---|---|---|---|---|
| NEE night | 800 | .86 | .02 | .12 | +0.57 |
| NEE day | 1819 | .91 | .08 | .02 | −0.87 |
| MODIS LAI | 89 | .87 | .12 | .01 | +0.07 |
| LandTrendr AGB | 6 | 1.00 | 0 | 0 | +0.85 |
| SoilGrids SOC | 1 | 1.00 | 0 | 0 | −0.58 |
| night DJF / MAM / JJA / SON | | .93/.83/.94/.78 | | .07/.17/.05/.15 | +.65/+1.10/−.23/+.22 |
| day DJF / MAM / JJA / SON | | .97/.93/.86/.88 | .00/.05/.14/.10 | | +.24/−.70/−2.43/−.26 |

**Plausibility** (`final_plausibility.csv` and `final_criteria.csv`), as
5/50/95% over the runs:
- **Carbon fluxes (g C m-2 yr-1):** GPP 789/1451/2635; Reco 768/1266/2041;
  NEE −973/−203/562.
- **Respiration ratios:** Ra/GPP 0.34/0.59/0.95; Rroot/Rsoil 0.07/0.19/0.38.
- **LAI:** JJA-mean LAI 2.7/4.7/7.8; annual peak LAI 3.9/6.5/10.8.
- **Pool changes over 13 yr (g C m-2):** wood −2327/+1830/+6238; soil
  −5059/+295/+4858.
- **Root pools (g C m-2):** fine roots 13-69; coarse roots 980-3980.

The fraction of runs outside each bound:

| bound | fraction |
|---|---|
| GPP < 1000 | .16 |
| GPP > 1800 | .26 |
| NEE > 0 | .34 |
| NEE < −700 | .14 |
| JJA LAI outside 3-8 | .12 |
| annual peak LAI > 8 | .30 |
| wood < 0.8× initial | .03 |
| wood > 2× initial | .12 |
| Ra > GPP | .04 |
| soil change > 20% | .24 |

What the checks show:
- **The data lie in the bulk.** Every source has 86-100% of its observations
  inside the 90% interval, and every median is inside the plausible ranges.
- **The tails are broad.** They come mostly from P, halfSatPar and the two
  initial-condition priors, which I kept at the ensemble's honest spread.
- **Residual structure that no prior removed:**
  - Summer daytime uptake is still too weak at the median (JJA z −2.4).
  - Winter and spring night respiration is too low (DJF +0.65, MAM +1.1),
    and autumn is also low.
  - Inference: the autumn part points to the litter pool being off. Leaf fall
    goes straight into a ~15 kg m-2 soil pool, so there is no fresh-litter
    pulse.
  - Inference: the whole pattern also fits a respiration seasonal amplitude
    that is too large. The observed night ratio from Jan to Aug is 3.1; the
    model's is ~7 at the prior median.

## 6. Item 5 notes (not implemented)

**Low MODIS LAI.**
- 14 of the 89 June-August values are below 2.5 (0.7-2.2), for example
  2015-06-22 at 1.0 and 2021-08-13 at 0.7.
- At those dates the prior-median prediction is 3.2-6.2.
- A closed deciduous canopy cannot fall to LAI 1 in midsummer, so these are
  probably cloud or QC contamination (inference).
- They raise the LAI residual RMS of the best 5% of runs from 1.53 to 2.12.
- The evidence supports screening them, by QC flag or by a documented floor.

**Leaf-on trigger** (`phase2/phenology_evidence.csv`).
- SIPNET's GDD (base 0 C, `sipnet.c:230-235`) reaches 500 on DOY 123 ± 7. The
  observed onset of daytime uptake (3 days < −5 umol m-2 s-1) is DOY 148 ± 5
  over 2012-2020.
- At onset, cumulative GDD is 847 ± 119 (CV 14%) and soil temperature is
  14.3 ± 0.7 C (CV 5%).
- So soil temperature is the more consistent trigger, which supports trying
  `soil_phenol` with soilTempLeafOn ≈ 14 C (template 12). This is inferred
  from the calibration data themselves; check it against the Harvard
  phenology record (HF003) before adopting it.

**Discrepancy sd.**
- The best 5% of prior runs have residual RMS 2.0 at night and 4.2 by day,
  against median total sds of 1.07 and 1.90 (χ²/obs 2.5 and 3.6). LAI is
  2.1 against 0.94.
- These runs are not calibrated, so this is an upper bound on the posterior
  residual scale.
- The residuals are also seasonally structured: the same seasonal signs
  appear in every iteration.
- A discrepancy with a 2-day timescale for NEE cannot represent a seasonal
  bias. Inference: the NEE discrepancy sds (1.0 and 1.8) are likely too small
  unless calibration halves the residuals; either an explicit seasonal term
  or a larger sd with a longer timescale is worth considering.
- The LAI discrepancy (0.5) is small against the low-value outliers. Screen
  those first.

## 7. Open issues and oddities

1. **My iteration script had a bug.** In `phase2_prior_iterate.to_overrides`,
   `pc.overrides` recomputed baseFineRootResp from its default, so i1-i5 ran
   with the PEcAn value 0.000509 rather than 0.186. The final run uses 0.186.
   The effect is small, because the fine-root pool is 13-69 g C m-2: Rroot/Rsoil
   rose from ~0.10 to 0.19.
2. **The simplex is not a Dataset variable.** A parameter vector's Dataset holds
   a simplex as `allocation.leaf`, `allocation.wood` and so on, so
   `"allocation" in dataset` is False. My first final script silently dropped
   it. This is worth documenting at `ParameterVector.dataset`.
3. **Structural issues in SIPNET:**
   - Leaves are allocated NPP all season until leafOffDay, so the annual
     maximum LAI exceeds the MODIS JJA level (peak LAI > 8 in 30% of runs,
     JJA LAI in only 4%).
   - The fine-root pool collapses to BETY allocation × NPP / turnover.
   - Root respiration comes almost entirely from the coarse roots.
   - Harvard trenching puts roots at ~1/3 of soil respiration (Bowden et al.
     1993, approximate); the model gives ~0.19.
4. **Two tails come from the initial-condition ensemble.** Its SOC upper tail
   (to ~100 kg m-2) and its wood lower tail (~1.5 kg m-2) drive most of the
   soil and wood implausibility. The log-normal fit also moves the wood median
   from 8.63 to 7.3. If the ensemble mixes soil types (inference), a robust
   interquartile fit would be a defensible alternative.
5. **The soilWHC likelihood is flat above ~88 cm.** Nothing is
   water-limited there, so the posterior will follow the prior's upper tail.
6. **The fine-root allocation dimension of the simplex is nearly flat** (Morris
   0.02), and only its prior controls it.
7. **The MODIS LAI record is longer than the NEE record.** The calibration
   vector's MODIS dates run past 2020 (e.g., 2021-2022), while the calibration
   NEE is 2012-2020. This is noted, not changed.
