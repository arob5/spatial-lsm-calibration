# Parameter analysis (step 4)

The evidence behind `../../model/prior.py` and
`../../model/fixed_sipnet_parameters.csv`. The experiment's `MODEL.md`,
"Parameterization and prior", summarizes it;
these are the records it cites.

| File | What it is |
|---|---|
| `phase1_report.md` | SIPNET's parameter structure checked against the pinned source (tag v2.2.0-alpha.1), the temperate deciduous base set, and its sanity runs |
| `base_parameter_candidates.csv` | the base set: every pySIPNET parameter, its proposed value, source, the BETY quantiles after PEcAn's conversion, and flags |
| `phase2_report.md` | the Morris screening, the parameterization, the priors, the prior-predictive iterations and the final checks |
| `prior_proposal.csv` | the prior, one row per calibrated parameter, with quantiles and provenance |
| `morris_mu_star.csv`, `morris_relative_mu_star.csv` | the screening's elementary effects, per output |
| `final_coverage.csv`, `final_plausibility.csv`, `final_criteria.csv` | the final prior predictive (250 draws): coverage per observation source and season, and the ecological plausibility checks |
| `phenology_evidence.csv` | leaf-on timing: growing degree-days and soil temperature at the observed onset of uptake, per year |

The exploration scripts that produced these (`phase2_*.py`, the per-iteration
tables and the prior-predictive draws, `prior_predictive.npz`) are in the
untracked `../../output/analysis/`, as run; they used `../fast_forward.py`,
when it and the model modules sat at the experiment's top level, so they
import it as `fast_forward` and would need their imports updated to rerun.
They are the record, not maintained code, so they are not tracked.
