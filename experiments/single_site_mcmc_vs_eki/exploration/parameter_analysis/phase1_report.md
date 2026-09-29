# Phase 1: SIPNET parameters for Harvard Forest (site 4977)

Pinned SIPNET source: `/private/tmp/claude-501/-Users-andrewroberts-Desktop-git-repos-spatial-lsm-calibration/9b51c9fb-5443-4450-a72f-9dd172cfac22/scratchpad/sipnet_pinned`
(tag v2.2.0-alpha.1, `git log -1` = 41fa853e7131f542c52fcc0f4e3ea76892b52eda).
All `sipnet.c:N` references below are to `src/sipnet/sipnet.c` at that commit, and
`R:N` to PEcAn `models/sipnet/R/write.configs.SIPNET.R` (develop, 523a430c).
Scripts: `scratchpad/base_set.py`, `sanity_run.py`, `write_csv.py`. Per-run
annual tables: `run_*.csv` in this directory.

## 1. Identifiability structure, checked against the source

**Photosynthetic amplitude group (vault note 3.1): verified.** Every model use of
the four symbols: `aMax` at 614 and 617; `aMaxFrac` at 617; `baseFolRespFrac` at
614; `cFracLeaf` at 633, plus the divide-by-zero guard at 404-405 (the other hits
are the reads at 306-308 and 350). Then `potGrossPsn = grossAMax*dTemp*dVpd*dLight*conversion`
(636), and `baseFolResp = respPerGram*conversion` (640). So the model depends on
them only through P = aMax(aMaxFrac+baseFolRespFrac)/cFracLeaf and
F = aMax*baseFolRespFrac/cFracLeaf. That leaves two exact flat directions.

**leafCSpWt x attenuation (note 3.2): verified, with one correction.**
`leafCSpWt` is used at 633 (`conversion`, where it cancels against
`lai = plantLeafC/leafCSpWt`), 1274 (the `lai` definition), 1887 (C_L(0) = laiInit*leafCSpWt)
and the guard at 413-414. `attenuation` is used only at 546
(`exp(-attenuation*cumLai)`, with `cumLai = lai*layer/NUM_LAYERS` at 542).
`laiInit` is used only at 1887. The correction: line 873 does not use
`leafCSpWt` directly. It uses `lai`, and only inside the `leaf_water` branch
(869-878). With the default flag, `immedEvap = rain*immedEvapFrac` (880). So under
the default flags only attenuation/leafCSpWt is identifiable from C and water
outputs. For our deciduous run, which starts leafless (laiInit = 0), the laiInit
leg of the scaling does nothing. `leafGrowth` is in g C m-2, so it does not
break the scaling either. The LAI operator (C_L/leafCSpWt) is the only thing that
separates the two.

**Other claims, verified:**
- `psnTMax = 2*psnTOpt - psnTMin` (1880-1881).
- Foliar respiration uses `vegRespQ10^((tair-psnTOpt)/10)` (1055-1056). So
  `psnTOpt` does two jobs, as note 4.1 says.
- Wood respiration is `baseVegResp*getTotalWoodC()*vegRespQ10^(tair/10)`
  (1066-1067). `getTotalWoodC = plantWoodC + plantCAccountingDelta`
  (`state.c:17-19`), which is aboveground wood only, not roots.
- The per-year to per-day rate conversions are at 1873-1877 and 1898-1902.
- `ensureAllocation` exits the process (1111-1123).
- `litterInit` is inert with `litter_pool` off (1889-1893).
- `leafOnDay` is inert under `gdd`, because the gdd branch returns first
  (705-731).
- Both phenology thresholds are step functions: `gddLeafOn` at 715 and
  `leafOffDay` at 735-737.

One point in note 3.4 is not true of this repository: the silent-death hazard of
fineRootFrac + coarseRootFrac >= 1 at 1883-1885 is real in SIPNET, but
`to_sipnet_initial_conditions` refuses such values.

**Where each process is controlled:**
- **Phenology (gdd).**
  - Leaf-on happens when cumulative GDD >= `gddLeafOn` (705-716). A one-step pulse
    `leafGrowth/length` is added then (816). The pulse is capped at
    (plantWoodC+coarseRootC)*`leafOnReallocFrac` (`limitations.c:26-27`) and
    drawn from wood and coarse roots in proportion (819-822, 1599-1601,
    1673-1677).
  - Leaf-off happens when DOY >= `leafOffDay`, which drops `plantLeafC*fracLeafFall`
    (733-737, 833).
  - Between the two, leaf creation is `leafAllocation`*(5-day mean NPP) and
    litter is `leafTurnoverRate`*C_L (768-777).
- **Autotrophic respiration.** Foliar: F, `vegRespQ10`, `psnTOpt`,
  `frozenSoilFolREff`/`frozenSoilThreshold` (1055-1061). Wood: `baseVegResp`,
  `vegRespQ10` (1066-1067). Roots: `calcRootResp` = base*pool*Q10^(tsoil/10)
  (1073-1078), called with the fine and coarse `base*RootResp` and `*RootQ10`
  (1192-1195).
- **Heterotrophic respiration.** soilC*`baseSoilResp`*M(w)*`soilRespQ10`^(tsoil/10)
  (1146, `depeffects.c:73`). M = f_whc^`soilRespMoistEffect`, and 1 when
  tsoil < 0 (`depeffects.c:23-43`).
- **Allocation.** The leaf, wood, fine-root and coarse-root fractions of the
  5-day mean NPP, with coarse roots taking the remainder (777-778, 1185-1186,
  1113-1115). The difference between actual NPP and what the mean-NPP
  allocations use goes into wood (`plantCAccountingDelta`, 1597), so wood is the
  residual sink. A negative leaf creation is moved to wood
  (`limitations.c:129-146`).
- **Turnover.** Wood (765), leaves (768), roots (1178-1179). All are per year,
  converted to per day at 1876-1877 and 1898-1899.
- **The initial root split.** C_W(0) = (1-fine-coarse)*plantWoodInit, and the
  root pools are fraction*plantWoodInit (1883-1885, 1921-1922).
- **Water limitation.** `removableWater = min(soilWater, soilWHC)*waterRemoveFrac`
  (682), and dWater = trans/potTrans (677, 697).

## 2. The base set (`base_parameter_candidates.csv`, 58 rows)

The CSV has one row for every pySIPNET parameter, and every proposed value is
inside its pySIPNET domain. Where the source is BETY, the value is the
temperate-deciduous posterior median converted per sample exactly as R:336-617
does. Where it is not, the value comes from `template.param_v2`, which is what
PEcAn uses for SIPNET >= 2 (R:117, 276-284). The Niwot values were never needed,
since the template covers every parameter BETY does not. The notable choices and
flags:

- **psnTOpt = 24 (template).** The BETY median is 43.0 C (2.5-97.5%: 27.6-67.0),
  which is implausible for this canopy (literature optimum ~20-27 C, approximate).
- **aMax = 112 nmol g-1 s-1 (template).** BETY gives Amax 3.8 umol m-2 s-1, and
  Amax*SLA (R:358-361) = 58 nmol g-1 s-1 (28-83). That is low for temperate
  deciduous leaves (inferred: typical light-saturated rates are ~8-15
  umol m-2 s-1). With it, GPP was ~690 and the site a net source. At the BETY SLA,
  112 is 7.4 umol m-2 s-1.
- **baseFolRespFrac = 0.089.** This keeps BETY's area-based Rd (0.66) with the
  new aMax, using R:379-381's own formula. The BETY-exact value is 0.167.
- **leafCSpWt = 30.7 g C m-2** (1000*leafC/SLA, R:348-349; 27.4-35.1), which is
  LMA ~66 g m-2. cFracLeaf is 0.466.
- **leafGrowth = 113.8 g C m-2** (BETY; 50.7-248.6). It is greater than 0, as a
  leafless start requires (816), and gives LAI ~3.7 at leaf-on. Leaf-on fell on
  DOY 108-133 in the run and leaf-off on DOY 285 (template).
- **leafTurnoverRate = 0.13 (template).** BETY's 0.75 yr-1 double-counts leaf
  loss that `fracLeafFall` (0.995) already does at leaf-off.
- **Root split: fineRootFrac 0.03, coarseRootFrac 0.16 (inferred).** These
  follow from coarse roots ~0.2 x aboveground wood and fine roots ~300 g C m-2,
  and should be checked against Harvard allometry. They give fine 320, coarse
  1705 and aboveground 8633 g C m-2. The stand-in's 0.2/0.2 gave ~2880 each. Either
  way the fine-root pool collapses to ~25-90 within two years, because it
  equilibrates at fineRootAllocation*NPP/fineRootTurnoverRate (0.065 x ~600 / 1.55).
- **Root and stem respiration: a unit slip in PEcAn.** R:461-468 and R:484-491
  convert umol CO2 kg-1 s-1 to **g C g-1 d-1** under a comment saying the value is
  read "as per-year". SIPNET does read it as yr-1 and divides by 365 (1873, 1902).
  So the value in effect is 365x below the trait.
  - baseFineRootResp becomes 0.00051 yr-1, so root respiration is essentially
    only the coarse roots (template 0.006).
  - baseVegResp becomes 0.0118 yr-1, the template's order. The trait taken
    literally would be 4.3 yr-1, which is absurd. The BETY stem draws (3-97,
    median 50) look unconstrained.
  - Both are kept BETY-exact but flagged.
- **soilWHC = 88.1 cm.** This is PEcAn's own formula: porosity x thickness over
  2 m (R:783-812). data/README.md:1284-1306 gives the median 88.1 over a
  100-site sample. This site's own value is not available here, and 2 m of
  porosity overstates plant-available water.
- **dVpdExp = 2** (Braswell; comment at 625). The BETY value is 1.50, and the
  change made no difference to the run.
- **Other flags:**
  - Coarse-root allocation is the remainder, 0.36, which is large (BETY sums
    are 0.43-0.85, never >= 1).
  - Inert under the default flags: growthRespFrac, leafPoolDepth, litterInit,
    litterBreakdownRate, fracLitterRespired, leafOnDay and soilTempLeafOn.
  - litterInit and snowInit are the pySIPNET defaults (0) that
    `to_sipnet_initial_conditions` sets.
  - The BETY gddLeafOn (458-542) and cFracLeaf draws are near point masses.

## 3. The sanity runs

Every run is one SIPNET run over 2012-2024, from `runs.sipnet_model()` with all
58 parameters overridden. Numbers are annual means in g C m-2 yr-1. NEE is
positive for a source. Ra_ag is `above_ground_respiration`, Rroot
`root_respiration` and Rh `heterotrophic_respiration`. Pools are start -> end.

| run | change | GPP | Reco | Ra_ag | Rroot | Rh | NEE | NPP | peak LAI | wood | soil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | rule-based set | 687 | 1057 | 522 | 42 | 494 | **+371** | 123 | 3.9 | 8633->7336 | 15000->12392 |
| a1 | aMax 112, baseFolRespFrac 0.089 | 1104 | 1145 | 625 | 59 | 462 | +41 | 420 | 5.2 | ->8863 | ->14167 |
| a2 | + leafTurnoverRate 0.13 | 1125 | 1186 | 671 | 57 | 458 | +61 | 397 | 6.2 | ->8746 | ->14105 |
| a3 | + dVpdExp 2 | 1126 | 1186 | 671 | 57 | 458 | +61 | 397 | 6.2 | ->8748 | ->14101 |
| d4 | + soilWHC 100 (diagnostic only) | 1456 | 1369 | 749 | 72 | 548 | -87 | 634 | 7.5 | ->9946 | ->14068 |
| **a5** | a3 + soilWHC 88.1 (**proposed**) | **1456** | **1361** | 749 | 72 | 540 | **-95** | 634 | **7.5** | ->9946 | ->14174 |

What the runs showed:
- **v0 is a net source**, with low GPP and a soil pool falling ~200 per year.
  Fine roots fall from 320 to ~5.
- **Foliar respiration is ~400-450 of the a1-a5 Ra_ag.** Wood respiration is
  ~220, computed offline from the formula at 1066-1067 with the run's wood pool
  and tair.
- **Water is the lever the a1-a3 GPP was missing.** With soilWHC = 12, GPP
  dropped by 20-35% in drier years (soil wetness ~0.73-0.75 in 2016, 2020 and
  2022). Removing the water limit raised mean GPP by ~330.
- **a5 is a modest sink.** Wood gains ~100 per year, and soil C falls ~65 per
  year. Peak LAI (7-8) is somewhat above MODIS (5-7). NEE (-95) is at the weak end
  of the observed -110 to -540.
- **Components still out of line with the site** (literature values
  approximate): below-ground autotrophic respiration is ~70, far below typical
  temperate forest root respiration, and Rh carries almost all soil
  respiration. Ra/GPP is ~0.51.

## 4. Recommendations

**Most important to calibrate:**
- **Daytime NEE / GPP:** P (the aMax group), the water-side product
  (soilWHC with waterRemoveFrac or wueConst), and halfSatPar. psnTOpt is second
  order.
- **Night NEE / Reco:** foliar respiration F with its Q10 and reference,
  baseVegResp, baseSoilResp and soilRespQ10. Root respiration as currently
  parameterized is negligible. It needs either the unit fix or a prior on
  total below-ground respiration.
- **LAI:** leafGrowth, leafAllocation and leafTurnoverRate (the level), and
  gddLeafOn and leafOffDay (the timing, which are step functions, note 4.6).
  leafCSpWt converts carbon to LAI and is identifiable only through LAI.
- **Wood biomass:** woodAllocation (and with it the coarse-root remainder),
  woodTurnoverRate, baseVegResp, and the initial wood pool.
- **Soil carbon:** baseSoilResp against soilInit (note 4.2), and the litter
  inputs (turnover rates).

**Reparameterize:**
1. Calibrate (P, F), or aMax and baseFolRespFrac with cFracLeaf and aMaxFrac
   fixed. The stand-in's ComputePhotosynthesisRates already does this.
2. With LAI data, calibrate attenuation/leafCSpWt and leafCSpWt, or fix
   leafCSpWt from SLA and leafC.
3. Re-reference every base-rate x Q10 pair to a site temperature of about 10 C
   (note 4.1): foliar, wood, both roots, and soil. For foliage, decouple the
   reference from psnTOpt.
4. Use (psnTOpt, psnTOpt - psnTMin) in place of (psnTMin, psnTOpt).
5. Put the allocation fractions on a simplex.
6. Tie the coarse-root and wood respiration amplitudes (note 4.3), or fix
   the coarse-root one.
7. Fix dVpdExp at 2.
8. Treat soilWHC x waterRemoveFrac (x wueConst) as one water-limitation
   direction. At this site it matters (~330 g C yr-1 of GPP), so it should not be
   left at the template's 12 cm.
9. Fix the inert parameters and the transient initial values.
10. Decide the PEcAn root and stem respiration unit question before the
    BETY-derived priors are used.
