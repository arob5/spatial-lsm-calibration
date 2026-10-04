# Probability-layer refactor: handoff

This file is the state of the refactor that replaces `sipnet_calibration.parameters`
with `sipnet_calibration.probability`:
- which PRs are merged, open and next;
- what each session learned;
- the deviations from the design;
- the questions waiting for Andrew.

The design is `design.html` beside this file, and the workflow every session
follows is CLAUDE.md's "The probability-layer refactor". Sessions update this
file in their own PR (CLAUDE.md, step 6).

## The PRs

Statuses: `next`, `open #n`, `merged #n`, `waiting` (its dependencies are not
merged). The next PR is the first row that is neither merged nor open. Scopes
and proofs are the design's §12; the column below is a summary.

| PR | Branch | Status | Needs | Scope |
|---|---|---|---|---|
| P0 | `docs/probability-refactor-workflow` | merged #76 | — | The workflow in CLAUDE.md; this file; `design.html` |
| P1 | `refactor/probability-p1-references` | open #77 | P0 | A script writing reference values from today's code: the prior draws and densities (PR #69's prior transcribed, `example_calibration`, a hierarchy, a copula, a per-PFT simplex), and `ForwardModel` predictions on `test_forward`'s fake runners |
| P2 | `refactor/probability-p2-foundations` | next | P0 | Supports, with `PositiveDefinite`; `ArraySpec`; `labels` v2; `Layout`; encode and decode; shims left in `parameters` (split into P2a and P2b if large) |
| P3 | `refactor/probability-p3-prior-model` | waiting | P1, P2 | Laws, families, builders; `FactorSpec`, `DeterministicSpec`, decorators; `joint`, `bind`, `FactoredDistribution`; `condition_on` and `Posterior` without simulators |
| P4 | `refactor/probability-p4-adapter-prep` | waiting | P2 | F8; the SIPNET map on dicts and `ArraySpec`s; `ObservationSource.standard_deviation`; the observation dims and constants; `observation.model` |
| P5 | `refactor/probability-p5-simulator` | waiting | P3, P4 | The `Simulator` seam; `SIPNETRuns`, with today's `ForwardModel` delegating to it; `SIPNETSimulator`; F6, F7 |
| P6 | `refactor/probability-p6-gaussian` | waiting | P5 | Covariance specs, `GaussianSpec`, `noise_factor`, `gaussian_likelihood`, through the `probability/_linalg.py` shim over today's pyEKI |
| P7 | `refactor/probability-p7-inference` | waiting | P6 | The `inference` package on today's pyEKI |
| #69 | `feat/single-site-mcmc-vs-eki` | not this refactor's | P7 | PR #69 migrates in its own session |
| P8 | `refactor/probability-p8-conjugacy` | waiting | P6 | `marginalize`, `full_conditional`, `theta_with` |
| P9 | `refactor/probability-p9-foreign-laws` | waiting | P3 | numpyro and EnsKit `Gaussian` adapters; GPJax as an optional test group |
| R1 | `refactor/probability-r1-removal` | waiting | #69 migrated | Delete `parameters`, today's `ForwardModel`, the Flat API, the old `describe_calibration`; move the vocabulary into CLAUDE.md's glossary |
| E1–E3 | `refactor/probability-e<k>-enskit` | waiting | EnsKit PRs 1, 2 and 4, 7 | The pyEKI-to-EnsKit rename, EnsKit's `linalg` and `Gaussian`, EnsKit's EKI driver |

The plan allows P2 and P4 to run in parallel with P1. The workflow takes one PR
at a time, in table order, unless Andrew starts a parallel session himself.

## Decisions

The design's §13 lists decisions D1–D24, each marked "agreed" or "recommend".
Until Andrew overrides one, a session implements the design as written,
recommendations included, and reports any recommendation it finds doubtful.

## Open questions for Andrew

- **Pin pyEKI?** EnsKit's PR 1 renames `pyeki` to `enskit`. A session elsewhere
  that runs CLAUDE.md's standard companion upgrade would then break every pyEKI
  import, PR #69's included. The refactor's sessions do not upgrade pyEKI.
  Pinning its revision in `[tool.uv.sources]` would protect everyone.
- **The recommended decisions** (§13): D1, D2, D5, D7, D8, D9, D10, D11, D12,
  D13, D14, D18–D24.

## Notes for implementers

These come from the design's reviews; each was checked against the code or its
data.

- **Parity (P3).** Bit-identical prior draws need five things:
  - theta's order equals today's term order;
  - `ParameterVector`'s default `order`;
  - `log_prob` sums factors in declaration order;
  - pushforward terms keep being evaluated by their base density;
  - `pushforward` of a TFP base returns an exact `tfd.TransformedDistribution`.
- **`labels` (P2).**
  - `as_coords` refuses a MultiIndex today.
  - `check_labels_are_covered` is quadratic (about 2 s for 2k labels against
    20k).
  - `pd.date_range` gives `datetime64[us]` in this environment, where the
    design asks for `[ns]`.
- **The keyword rule (P3).** A function reads the names of its parameters
  without defaults. Refuse `*args`, `**kwargs` and positional-only parameters;
  a `functools.partial`'s bound arguments are defaults.
- **Imports.**
  - `probability` imports nothing of the package.
  - `observation` must stay free of TFP, so its probability-facing functions
    go in `observation.model`, which `observation/__init__` does not import.
  - `conventions` must not import `probability`: both hold the literal
    `"sample"`, and a test checks they agree.
  - The parameter-layer import test bans pyEKI; the `_linalg.py` shim needs an
    exception there.
- **Data (P4, P6).** About 10% of MODIS LAI standard deviations, and some
  LandTrendr ones, are zero in the local processed files. A
  `standard_deviation` is non-negative, and the covariance built from it must
  be positive definite.
- **PR #69** touches only `experiments/` and has no tests. Every PR before R1
  must leave its imports working (`parameters`, `ForwardModel`,
  `ObservationVector.y`, `positions`, `flat`).

## Session log

Each session adds an entry, newest last: the date, the PR, what was done, the
deviations, and what the next session must know.

### 2026-10-04: P0 (design session)

The design went through five revisions and four review rounds. Revision 5 is
`design.html`, also published as a private Artifact. This PR adds the workflow
and this file; no code changes.

### 2026-10-04: P1, the references

**Done.** `tests/data/write_probability_references.py` writes six files to
`tests/data/probability_references/`, through `io.write_checked_together`,
from today's `parameters` and `ForwardModel`.
`tests/test_probability_references.py` reruns the script into a temporary
directory and compares bytes; it takes about 30 s. There is no library change.
Tests: 2474 passed and 94 skipped before, 2475 and 94 after.

The cases:

- `single_site_mcmc_vs_eki`: PR #69's prior at 7949640, D = 15. Its two
  initial carbons are fitted to site 4977's members, read from the tracked raw
  file (`read_raw`, by each spec's `source_name`). The processed file is a
  transpose of the raw one, so these are the members the experiment reads.
- `example_calibration`: the example's prior at sites 1, 27 and 4711.
- `hierarchy`: `pft_log_mean` on `pft`, `spread`, `soil_carbon` on `site`
  given the derived `site_log_mean` (through a `pft_of_site` membership) and
  `spread`, and `offsets` on `site` given `spread` through `iid_over_dim`.
  Two given terms, so the per-draw `vmap` sampling path is covered.
- `copula`: `gaussian_copula` over a log-normal and two logit-normals.
- `per_pft_simplex`: `independent_over_dim(softmax_normal, ...)` with a
  per-PFT `center` constant on `(pft, allocation_part)`.
- `forward_example`: `test_forward`'s setup (the example at sites 1 and 27,
  site 27's drivers cut to 40 steps, two observation sources), with four
  prior draws and theta = 0 through `ForwardModel.evaluate`. Every row is
  valid; there is no failing row, since P5 has its own failing-run test.

**Deviations from the design.** None. Choices the design left open: the
script lives beside PR #72's `write_calibration_references.py` rather than in
`scripts/`, because it imports `tests/conftest.py`'s stand-in SIPNET. The seed
is 20261004.

**What P3 and P5 must know.**

- A prior file's `draws` are `prior.sample(jax.random.key(seed), n_draws)`,
  with both values in the file's attributes. `theta` is the draws, then
  theta = 0, then `n_standard` rows of N(0, I). `log_prob` is at `theta`.
  `natural:<name>` and `derived:<name>` are the labeled values at `theta`.
  `constant:<name>` is what the case needs beyond code: PR #69's members, and
  the simplex's centers.
- Entries are labeled by today's unconstrained `entry_names`, with the index
  levels as string coordinates `entry_<level>` (empty where a level does not
  apply). A simplex's unconstrained element labels are its first `k - 1`
  labels.
- A forward file's `observation_site`, `observation_source` and
  `observation_time` are today's observation index, in its order.
- The bytes depend on the platform and on `uv.lock`. If a companion upgrade
  (pySIPNET's Niwot data, say) or a JAX or TFP bump breaks the byte test,
  rerun the script and commit the new files, provided `parameters` and
  `ForwardModel` are still today's code. After P5 delegates `ForwardModel` to
  `SIPNETRuns`, rewrite `forward_example` only from a commit before P5.

**Housekeeping.** At this session's start, the auto-mode classifier refused
removal of P0's stale worktree, and also refused reading `design.html`, until
Andrew allowed both. A worktree-isolated session's hook allows writes only in
its own worktree, so a session enters the new PR's worktree with
`EnterWorktree(path=...)` before writing.
