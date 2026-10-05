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
| P1 | `refactor/probability-p1-references` | merged #77 | P0 | A script writing reference values from today's code: the prior draws and densities (PR #69's prior transcribed, `example_calibration`, a hierarchy, a copula, a per-PFT simplex), and `ForwardModel` predictions on `test_forward`'s fake runners |
| P2 | `refactor/probability-p2-foundations` | merged #78 | P0 | Supports, with `PositiveDefinite`; `ArraySpec`; `labels` v2; `Layout`; encode and decode; shims left in `parameters` (split into P2a and P2b if large) |
| P3 | `refactor/probability-p3-prior-model` | merged #79 | P1, P2 | Laws, families, builders; `FactorSpec`, `DeterministicSpec`, decorators; `joint`, `bind`, `FactoredDistribution`; `condition_on` and `Posterior` without simulators |
| P4 | `refactor/probability-p4-adapter-prep` | merged #80 | P2 | F8; the SIPNET map on dicts and `ArraySpec`s; `ObservationSource.standard_deviation`; the observation dims and constants; `observation.model` |
| P5 | `refactor/probability-p5-simulator` | merged #81 | P3, P4 | The `Simulator` seam; `SIPNETRuns`, with today's `ForwardModel` delegating to it; `SIPNETSimulator`; F6, F7 |
| P6 | `refactor/probability-p6-gaussian` | merged #83 | P5 | Covariance specs, `GaussianSpec`, `noise_factor`, `gaussian_likelihood`, through the `probability/_linalg.py` shim over today's pyEKI |
| P7 | `refactor/probability-p7-inference` | merged #84 | P6 | The `inference` package on today's pyEKI |
| P8 | `refactor/probability-p8-conjugacy` | merged #85 | P6 | `marginalize`, `full_conditional`, `theta_with` |
| P9 | `refactor/probability-p9-foreign-laws` | merged #86 | P3 | numpyro and EnsKit `Gaussian` adapters; GPJax as an optional test group |
| E1 | `refactor/probability-e1-enskit` | open #88 | P9 | Re-pin from pyEKI to EnsKit's `main` (`TARPS-group/EnsKit`, package `enskit`), which moves JAX 0.8 to 0.10 and TFP's nightly with it; the `_linalg` shim on `enskit.linalg` and `enskit.distribution`; `GaussianLaw` and `as_law` on EnsKit's block `Gaussian`; `test_inference` and `test_smc` off the deleted `pyeki.eki`; P1's references rewritten if their bytes move; CLAUDE.md's companion table, upgrade command and pyEKI facts |
| E2 | `refactor/probability-e2-enskit-eki` | waiting | E1 | EnsKit's EKI driver: `eki_problem` in the terms of `enskit.algorithms.eki`; `initial_ensemble` an EnsKit `Ensemble` |
| #69 | `feat/single-site-mcmc-vs-eki` | not this refactor's | P7, E2 | PR #69 migrates in its own session, onto the probability layer and EnsKit together |
| R1 | `refactor/probability-r1-removal` | waiting | #69 migrated | Delete `parameters`, today's `ForwardModel`, the Flat API, the old `describe_calibration`; move the vocabulary into CLAUDE.md's glossary |

The plan allows P2 and P4 to run in parallel with P1. The workflow takes one PR
at a time, in table order, unless Andrew starts a parallel session himself.
E1 and E2 replace the plan's E1–E3, which were to follow EnsKit's rewrite
piece by piece; it is complete (`design.html` §12, "Revised 2026-10-05").

## Decisions

The design's §13 lists decisions D1–D24, each marked "agreed" or "recommend".
Until Andrew overrides one, a session implements the design as written,
recommendations included, and reports any recommendation it finds doubtful.

Decided by Andrew:

- **Track EnsKit's `main` (2026-10-05).** EnsKit's rewrite of pyEKI is
  complete, and the project re-pins to its `main` in E1, after P9. Until E1
  merges, sessions still do not upgrade pyEKI; after it, `enskit` is upgraded
  with the other companions. This settles the earlier question of pinning
  pyEKI.
  - E1 comes before #69's migration so that #69 moves once. Its experiment
    imports `pyeki.gauss`, `pyeki.linalg` and `pyeki.eki`, and EnsKit's
    `main` has none of the first and last. Once E1 is on `main`, #69's branch
    breaks on rebasing or re-locking until its migration ports those
    imports; #69's session should know before it starts.
  - E1's environment was resolved and smoke-tested in a scratch copy on
    2026-10-05: EnsKit requires `jax>=0.10.1`, which locks JAX 0.10.2,
    `tfp-nightly` 0.26.0.dev20261005 and NumPy 2.5. TFP's JAX substrate,
    numpyro, GPJax and EnsKit imported and evaluated there. EnsKit's
    `linalg` keeps every operator name the shim imports; its `Gaussian`
    (`enskit.distribution`) is block-structured, `Gaussian(means, *,
    factors=, block_covs=, latent_dim=)`.
  - P1's byte test depends on the JAX and TFP builds. If E1 moves the bytes,
    rerun the script and commit the new files, keeping the notes under P1:
    `forward_example` is rewritten only from code before P5. Pre-P5 `src/`
    imports no pyEKI, so it can run in E1's environment.

## Open questions for Andrew

- **The recommended decisions** (§13): D1, D2, D5, D7, D8, D9, D10, D11, D12,
  D13, D14, D18–D24.
- **TFP's Gamma sampling under JAX 0.10 (E1, found in review).** TFP's
  nightly calls a JAX shape function with `None`, which JAX 0.10 deprecates
  ("will be an error"); `bind`'s support check samples every Gamma-family
  and Dirichlet law, so a future JAX that makes it an error breaks `bind`
  for those laws until TFP fixes it. The lock pins JAX, so nothing breaks
  today. Recommended: leave it, and check the warning (the E1 entry says
  how) before any JAX upgrade; the alternative is a `filterwarnings` entry
  that would hide it.
- **An EnsKit Gaussian with a factor row and no independent term (E1).**
  `as_law` refuses it, since a `GaussianLaw` whitens its covariance and
  EnsKit holds `F F^T` as the factor alone. When the factor has at least as
  many columns as the block has entries, as an `Ensemble.project()` with
  more particles than dimensions does, the law has a density, which EnsKit
  scores by a QR of the factor. Recommended: keep the refusal until a factor
  needs such a law; adapting it means a `GaussianLaw` that scores through
  EnsKit's `log_density` for every case, and draws from the factor.
- **An observation's calendar year (P4).** `calendar_year` and
  `year_label_map` take the year of the `time` label. For a source labeled
  at its window's end, such as observed NEE, the step ending
  2013-01-01T00:00 then belongs to 2013, though its window lies in 2012.
  The annual constraints, labeled January 1, are unaffected. Recommended:
  keep the label's year (the design's definition, and what PR #69's LAI
  block assumes), and revisit when a noise factor groups NEE by year;
  the alternative is the year of the window's start where there is one.
- **A simulator downstream of another (P5, found in review).** The core
  passes a downstream simulator `NaN` at the samples where the upstream one
  failed, and the `Simulator` contract has no way to say "skip these". A
  `SIPNETSimulator` under `out_of_domain="raise"` would then refuse the
  whole batch. No model chains simulators today. Recommended: leave it, and
  add a `computed` mask to `Simulator.__call__` when a chain is needed;
  meanwhile `condition_on` skips `check_given` for a simulator with a
  simulator upstream, which would otherwise run that one at every corner.
- **`predict` runs SIPNET again (P5, found in review).** `predict(key,
  theta)` re-runs the simulator outputs barren factors read, including any
  `evaluate` already computed, so a predictive pass after an evaluation
  costs a second batch of runs. Recommended: keep the design's signature
  for P7, and add `predict(key, evaluation)` if a run's cost makes it
  matter.
- **Element axes are read by position in the SIPNET map (P4, found in
  review; predates the refactor).** `CopySimplex` and every rule read a
  value's element axes in the order the array holds them, so a simplex
  given in another `allocation_part` order writes the wrong allocations
  without an error, through a Dataset or labeled values alike. A `Layout`
  always produces the spec's order, so the probability layer's own path is
  safe. Recommended: P5's `check_given` checks the given `ArraySpec`s'
  element labels, and the map transposes by label where a value carries
  them; or leave it, the risk being a hand-built Dataset.
- **Correlated sources (P6).** A Gaussian whose event spans several
  components, each on its own observation dim, is refused at `GaussianSpec`
  and `noise_factor`. Building it needs: `FactorSpec`'s joint-factor rule
  relaxed for a Gaussian; per-component index shapes in `BoundFactor`; a
  rule for a constant read on several dims (the design's "concatenated in
  event order" says nothing of how one constant names several); and the
  permuted residual of the design's §14, with the EKI view holding the
  block dense. Recommended: build it when an experiment correlates two
  sources' errors; no current one does.
- **"Depends on theta" for a covariance (P6).** D21 says "has a target
  ancestor". P6 holds a covariance when it is computable from the held
  values (observed values, inputs, constants and their deterministics), so
  one reading an observed value that itself has a target ancestor is held.
  That is exact, the value being held in the conditional, and more
  permissive than D21. Recommended: keep it.
- **An asymmetric `DenseSpec` matrix (P6, found in review).** pyEKI's
  `DensePSD` factors the symmetric part, so a matrix function returning an
  asymmetric matrix is used as `(A + A^T)/2` without an error. The docstring
  says so. Recommended: add a symmetry check on concrete matrices (bind and
  held covariances) if it bites; it cannot run under the per-draw trace.
- **`initial_points`' batches (P7, found in review).** It evaluates the
  prior draws in order, in batches the size of the shortfall, so it runs no
  draw past the `n`-th finite one. At a 50% failure rate, `n = 5` takes five
  simulator calls of 5, 2, 2, 2 and 1 samples; for SIPNET each is a PyEns
  dispatch, so start-up grows like `log2(n)` calls with a serial tail.
  Asking for more, the shortfall divided by the valid fraction seen so far,
  chooses the same points in fewer calls at the cost of runs past the
  `n`-th. Recommended: keep it until a SIPNET start-up is measured.
- **`initial_points`' records (P7, found in review).** The evaluation it
  returns stitches its batches' rows, and holds per simulator a tuple of
  every batch's record, where `PosteriorEvaluation` documents one record.
  A record describes a whole batch and cannot be cut to the kept rows.
  Recommended: keep the tuple, documented on `initial_points`; the
  alternative is to drop the records, or to say on `PosteriorEvaluation`
  that a stitched one holds tuples.
- **Another posterior's evaluation (P8, found in review).** A
  `FullConditional` reads its residuals from a `PosteriorEvaluation`, which
  does not record its posterior. One from another posterior with the same
  `D` is accepted, and mixes that posterior's means with this one's
  observations; `log_density_given` (P5) has the same gap. Recommended:
  leave it, documented; the alternative is a `posterior` field on
  `PosteriorEvaluation`, checked by both.
- **Names that now cover a Student-t (P8, found in review).**
  `check_gaussian_mean_matches_its_event`,
  `check_gaussian_event_is_one_vector_on_the_reals`,
  `check_gaussian_mean_has_its_events_shape` and
  `check_gaussian_covariance_is_positive_definite` check Student-t factors
  too; their messages and docstrings say so, their names do not.
  Recommended: rename them `check_centered_*` in a module cleanup, not in a
  density PR.
- **`StudentTSpec(shape=, scale=)` (P8, found in review).** The glossary's
  *shape* is one value's array shape; these are the inverse gamma's shape
  and scale parameters, named as `inverse_gamma(shape=, scale=)` names
  them. Recommended: keep them matching the family; the alternative is
  `concentration` in both, which renames P3's family.
- **numpyro and GPJax are not dependencies (P9).** The layer recognizes
  numpyro's distributions by class name and imports neither package;
  numpyro is in the `dev` group for the tests, GPJax in an optional `gpjax`
  group, so the GPJax test is skipped after a plain `uv sync`.
  Recommended: keep it so until an experiment uses one of them; the
  alternative is GPJax in `dev`, which adds about ten packages (equinox,
  lineax, optax, ...) to every environment.
- **The builders refuse numpyro's laws (P9).** `iid_over_dim` and
  `independent_over_dim` wrap a law in TFP's `Sample` or `Independent`, so
  they refuse one from another package, and a numpyro law over a block is
  written as a law function (`nd.Normal(...).expand([n]).to_event(1)`).
  Recommended: keep it; repeating a numpyro law through its own `expand`
  and `to_event` is a small addition if an experiment wants it.

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

### 2026-10-04: P2, the foundations

**Done.** One PR; no split was needed. The new package
`sipnet_calibration.probability` holds:

- `support`, moved from `parameters` with `PositiveDefinite` and
  `POSITIVE_DEFINITE` added (`closure()` is the semi-definite matrices);
- `names`, the layer's own names and `RESERVED_NAMES`;
- `labels` v2: coords with stacked dims (a `MultiIndex`, times converted to
  `datetime64[ns]`), constants with `own_dims`, label maps into dims or
  element axes, every lookup by `get_indexer` (`indexer`);
- `spec.ArraySpec`;
- `layout`: `Layout`, `ValuesByName`, `LabeledValues`, their validators,
  `encode_labeled_values` and `decode_labeled_values`.

`_validation` and `_probes` moved too. `parameters.support`,
`parameters._validation` and `parameters._probes` are re-export shims, and
`tests/test_package.py` lets `parameters` import `probability` (and nothing
else) and checks that `probability` imports nothing of the package.
Tests: 2475 passed and 94 skipped at P1's merge; 2651 and 94 after.

**Review.** One Standard round (code, mutation testing, docs). It found,
and this PR fixes:
- a missing string label accepted (pandas 3 holds `None` among strings as
  `str`);
- `datetime64` selectors turned into integers or dates by coercion;
- labeled values whose draws differ in number or order across components
  paired wrongly;
- an own dim that is also a coords dim passed unaligned;
- integer labels held as `object` breaking `Layout.index`;
- repeated labels in a constant or label map raising pandas' own error;
- a stacked dim ordered two ways reaching `xr.Dataset` unchecked;
- a stacked dim's tuples matched across kinds (`620.0` for `620`);
- time-zone-aware and out-of-range times failing in pandas' words.

Tests now cover each, and the gaps mutation testing found, with one
exception: the round-trip half of `ArraySpec`'s custom-bijector check, which
no TFP bijector at hand violates while passing the other half.

**Deviations from the design.** None in substance. Choices it left open:

- theta's element axis for a positive-definite value is named
  `"<row axis>_<column axis>_cholesky"` (recorded in `design.html` §7.1);
- the probes `-10 * 1` and `-20 * 1` make positive-definite values too
  ill-conditioned to invert (`e^-10` on the diagonal under off-diagonals of
  -10), so `ArraySpec`'s custom-bijector check round-trips only where the
  default bijector inverts its own image;
- `parameters.labels` was not shimmed: v2 checks more (finite constants,
  label kinds) and `derived` and `prior` read v1's memberships, so v1 stays,
  quadratic check included, until R1;
- the conversions are named `values_to_labeled`, `labeled_to_values`,
  `flat_to_labeled` and `labeled_to_flat`; `Layout` adds `slice_of(name)` and
  `level_names`;
- `validate_labeled_values` ignores keys that are not components, as
  `validate_values_by_name` does (`ParameterDataset`'s refused them), so a
  `to_labeled` holding `"theta"` converts as it is;
- entry names carry no transform name (`log(x)`), there being no
  `long_name` (D6); `describe()` has the bijector;
- the encoding's attribute is `stacked_dims`, a JSON object
  `{dim: [level, ...]}`; it refuses values that label a shared dim
  differently rather than letting xarray outer-join them.

**What P3 and later must know.**

- **Order.** A layout's block is in C order over `indexed_by`;
  `ParameterVector`'s default order sorts by the coords' dims instead. They
  agree when every parameter's dims are in the coords' order, which holds for
  all of P1's cases; `bind` should pass coords in the order the parameters
  use them. `test_a_block_is_in_c_order_whatever_the_coords_order` pins the
  difference.
- **Coords at bind.** `Layout` refuses a coords dim no component uses, as
  `ParameterVector` did, so `bind` filters `site_dims.coords` to the dims in
  use.
- **Selection.** `Layout.select` refuses a selection that leaves a kept
  component's dim without a label (a source not observing the selected
  site); `positions` does not. `FactoredDistribution.select` must drop such
  components or say so.
- **Positive-definite probes.** Any probe-point check of a law or
  deterministic on a positive-definite component (P3, P8) meets the same
  ill-conditioning at `-10 * 1` and `-20 * 1`; compare in the closure, and
  skip what the default bijector cannot invert.
- **Constants** must be finite at the labels in use, checked when
  `aligned_constants` reads them, the earliest point that knows the labels.
- Performance: binding 8,000 sites and 80,000 stacked labels, with a
  constant and a label map on them, takes well under a second
  (`test_eighty_thousand_labels_bind_in_under_a_second`).

### 2026-10-04: P3, the prior model

**Done.** Six modules added to `sipnet_calibration.probability`:

- `laws`: `Law`, `as_law`, `pushforward` (TFP bases only);
- `families`: moved from `parameters`, with `normal`, `inverse_gamma` and
  `InverseWishart`/`inverse_wishart` added;
- `builders`: moved from `parameters.prior_functions`, with a `Builder` base
  exposing `.law` and `.reads`;
- `parts`: `FactorSpec`, `DeterministicSpec`, `@factor`, `@deterministic`,
  and the keyword rule (`_keywords`);
- `model`: `joint`, `ModelSpec`, `FactoredDistribution` (`law`, `select`,
  `sample`, `log_prob`, `describe`), `block_at_labels`;
- `posterior`: `condition_on`, `Posterior` (`sample_prior`, `log_prior`,
  `log_likelihood`, `log_density`, `natural_values`, `to_labeled`,
  `describe`).

The private `_bound` holds a part bound to the labels in use: its law,
density and draws, the bind checks ported from today's prior, and the
log-Jacobian, now on positive-definite matrices too.
`parameters.families`, `parameters.prior_functions` and
`parameters._distributions` are re-export shims, and P1's byte test still
passes through them.

`tests/test_probability_parity.py` redeclares P1's five prior cases as
specs and shows bit-identical theta order, draws, `log_prior` and natural
values, the hierarchy's deterministic included. The families are checked
against SciPy, `InverseWishart` among them. Graph tests cover barren nodes,
`O_c`, cycles and nothing observed. Tests: 2651 passed and 94 skipped at
P2's merge; 2795 and 94 after. One test checks that observing a
hyperparameter equals declaring it an input (Proposition 3.3): the same
draws and densities, and a different `log_constant`.

**Review.** One Deep round, with four reviewers: numerics, edge cases,
mutation testing and docs. It found no density that is wrong; A1 checked
them against SciPy, closed forms and grid integration. This PR fixes:

- a law class (`law=tfd.Normal`) taken for a law object; it is now read
  as a function of its arguments (`laws.is_law`, which also replaces three
  copies of the same test);
- booleans and strings accepted as input and observed values;
- `FactoredDistribution.log_prob` returning NaN outside a support, where
  the posterior gives `-inf`;
- a model of deterministics alone accepted;
- a float32 law refused with a message that never named the dtype;
- an empty provenance raising TypeError rather than ValueError;
- `select`'s message for an unknown selector advising `component=`;
- the inverse Wishart's scale factor formed with an explicit inverse;
- the builders' messages and check names still in the parameter layer's
  words, with three of its tests' `match=` strings updated to match;
- Raises sections and `__all__` lists that did not match the code.

Tests now cover the `-inf` mapping, a chain of two deterministics between
factors, the base-event-shape guard of the pushforward test, and the
autodiff log-Jacobian on the simplex. The last normalizes a Dirichlet
under `IteratedSigmoidCentered` by grid integration. Mutation testing's
`[:-1]` against `[1:]` in that Jacobian is an equivalent mutant: the
coordinate projections of the simplex share one Jacobian. The joint
factor's guard has no test, since no TFP bijector at hand reaches it.

**Deviations from the design**, each recorded in `design.html`:

- `Posterior.log_likelihood` and `log_density` are traced functions of
  theta, `(..., D) -> (...)`, until P5 routes them through `evaluate`;
- `Posterior.observations` is `None` when nothing in the likelihood is
  observed, since a `Layout` holds at least one component; `y` is then
  `(0,)`;
- a factor that reads a value varying by draw is built and checked at two
  ancestral draws, as today's prior does. The design's "probe points pushed
  through the deterministics" is applied to deterministics' outputs only:
  pushing probes into a law's parameters (a spread of `e^20`) would fail
  the draw-based support check on sound hierarchies;
- `DeterministicSpec` takes `own_dims=` too.

**Choices the design left open:**

- "a factor that reads no component draws at once" is read as "reads
  nothing that varies by draw". Inputs, held observed values, and what is
  computed from them alone are built once, and the factor draws `n` values
  at once.
- A constant or label map the function never reads is refused when the
  part is declared.
- `FactoredDistribution.sample(component_names=)` returns only those names.
  `log_prob` refuses deterministic components and inputs (`ValueError`) and
  unknown names (`KeyError`).
- `FactoredDistribution.law` is in P3 for TFP and protocol laws; P6 adds
  the Gaussian.
- `log_prior` maps a non-finite factor to `-inf`, factor by factor, so a
  finite value is unchanged bit for bit.
- The once-only check of §4.7, "a target kernel finite at the observed
  values", is made at theta = 0, for each target factor that reads an
  observed value.
- `Posterior(model, observed)` and `FactoredDistribution(spec, coords=,
  inputs=)` are also constructible directly; `condition_on` and `bind` call
  them.
- The moved builders keep the parameter layer's messages ("prior", "term"),
  which today's tests match.
- `InverseWishart`'s own default event-space bijector refuses float64 input
  in the pinned TFP. The bind check of a law's own bijector treats such a
  bijector as absent and relies on the draw-based check.

**What P4 and later must know.**

- Everything simulator-shaped is P5's: `Posterior.evaluate`,
  `PosteriorEvaluation`, `predict`, `replicate`, `simulator_inputs`,
  `log_density_given`, `simulator_free_positions` and the corner points.
  Nothing yet stops a target factor from reading a simulator output. That
  refusal belongs in `condition_on` once `Simulator` exists, as part of
  `_roles` in `posterior.py`.
- `FactoredDistribution` exposes its internals to `Posterior` through
  underscore names: `_factors`, `_deterministics`, `_fixed` (inputs and
  what is computed from them alone), `_ancestral` and `_computed`. A
  simulator node joins `_ancestral`'s and `_computed`'s loops in
  topological order.
- `BoundFactor.law_at` dispatches on the law's form; a `GaussianSpec` (P6)
  is a fourth branch there and in `parts._law_reads`.
- `parameters.prior` still runs on its own copy of the logic; only the
  families and builders are shared. R1 deletes it.

**Decided by Andrew after the review: laws with no density on their
support are refused.** Binding checks a law only at probe points and
draws, so it accepted a law singular or discrete on its support and
evaluated it as a density: `tfd.LKJ` on a positive-definite component, or
`tfd.Poisson` on `POSITIVE`. `_bound.check_law_has_a_density` now refuses,
by class, discrete laws, point masses and the LKJ laws. It looks inside
`Sample`, `Independent`, a pushforward's base, mixtures' components and
joint laws' parts, but not at a mixture's choice of component. The list is
`_bound._LAWS_WITHOUT_A_DENSITY`, by name. A law from another package that
is singular is not caught; when the foreign-law adapters arrive (P9),
theirs need the same check.

### 2026-10-04: P4, the adapters' preparation

**Done.** Additive throughout; the Flat API, `ForwardModel` and every old
test are unchanged.

- `ObservationSource(standard_deviation=)`: optional observed values, read
  by label at the observed values' labels (a larger grid is fine), in units
  that convert to theirs by exactly 1, finite and non-negative at every
  observation (zero allowed), stored on the observed grid with `NaN` where
  they are. It follows `select` and `restrict_to_sites`.
  `ObservationSource.observation_labels` is the source's `(site[, time])`
  `MultiIndex` (`site` `int32`, `time` `datetime64[ns]`).
- `ObservationVector`: `coords` (one observation dim per source,
  `"<source>_observation"`), `observation_dim_name`, `prediction_name`
  (`"predicted_<source>"`), `constants(source)` (`observed`,
  `standard_deviation`, `time_since_epoch`, `calendar_year`,
  `window_length`, each as the design lists), `year_label_map`,
  `observed_values_by_component`, `with_observed_values` and `to_fields`.
  The names are module constants exported by `observation`.
- `observation.model`: `observed_components` and `prediction_components`,
  `ArraySpec`s on `REAL` in the observed values' units. The package does
  not import it, and a test says so.
- The SIPNET map takes labeled values (a dict of DataArrays) wherever it
  took a Dataset, reading only what its rules name and refusing arrays
  that label a shared dim differently. `check_sipnet_parameter_map_fits`
  accepts `ArraySpec`s. F8: a `ValueRequirement` may omit `units` and
  `domain` (`FROM_SIPNET_SPEC`), resolved by `Compute` against the SIPNET
  parameter it writes; `Copy.same_names(names)`.
- `conventions.ReadOnlyCopies(default=None)`, for the optional standard
  deviation.

A test binds and conditions a small model over the observed and prediction
components, reading the vector's constants, and checks its density against
SciPy. Another shows a dict of labeled values from a `Layout` gives the map
the same fields and domain report as today's Dataset.
Tests: 2795 passed and 94 skipped at P3's merge; 2858 and 94 after.

**Review.** One Standard round (code, mutation testing, docs). It found,
and this PR fixes:

- an explicit `ValueRequirement(units, FROM_SIPNET_SPEC)` losing its domain
  (the domain's default is now its own sentinel);
- `with_observed_values` accepting values with other windows, locations or
  units, which would have changed the constants silently;
- `to_fields` accepting an observation twice;
- a custom rule's non-`ValueRequirement` raising `AttributeError`, and
  conflicting coordinates in labeled values escaping as xarray's
  `MergeError`;
- a standard deviation lacking an observed label raising `ValueError`
  rather than CLAUDE.md's `KeyError` (the design's stub said `ValueError`;
  noted there);
- messages and docs: a check without a `message_name`, an ungrammatical
  message, "predictions" used for the components, CLAUDE.md's layout
  missing `observation_dim_name`, unwrapped lines.

Mutation testing killed 28 of 31 mutants; tests now kill the other three
(a resolved requirement's shape, the standard deviation's own attributes,
and its constant's), and unequal windows pin `window_length`'s order.
Two findings are questions below, not fixes.

**Deviations from the design**, recorded in `design.html`:

- F8's "the SIPNET spec" is read as the spec of the SIPNET parameter the
  rule writes. A requirement that states its units and omits its domain
  keeps today's meaning, no domain, so `ValueRequirement("degC")` and the
  old tests are unchanged; a field inherits when omitted with the units,
  or when given as `FROM_SIPNET_SPEC`. A map refuses a custom rule that
  leaves a field unresolved.
- The design named no methods for the constants or the dim's name:
  `constants(source)` and `observation_dim_name(source)` are added, and
  `ObservationSource.observation_labels`.
- An observed component carries the observed values' `units` but not their
  `constituent`, which an `ArraySpec` has no field for.

**Choices the design left open:**

- `to_fields` skips an array on none of the vector's observation dims
  (`theta`, a parameter), so a posterior's whole `to_labeled` can be
  passed; it refuses a label that is no observation (`KeyError`), a dim
  that is not a batch dim, and other levels.
- `with_observed_values` requires the new values to observe exactly the
  same `(site[, time])` pairs, `NaN` elsewhere, with the same coordinates
  (locations, windows) and units; a source not named is kept as it is, the
  same object.
- `calendar_year` is a `float64` constant (constants are `float64` or
  `bool`); `year_label_map` holds the year as a string, for a `year`
  element axis.
- The standard deviation's units are compared by `conversion_factor == 1`,
  as the map compares a value's, constituents included.
- No check refuses a source name whose derived names collide (a source
  `predicted_x` beside `x`); `joint` refuses duplicate component names.

**What P5 and P6 must know.**

- P5: `SIPNETSimulator.check_given` is `check_sipnet_parameter_map_fits(map,
  given_specs)` plus today's corner check. `sipnet_parameter_fields` and
  `out_of_domain` take `LabeledValues` directly; the inputs (initial states)
  are not in `posterior.to_labeled`, so `SIPNETRuns.evaluate` must merge
  them in. `prediction_components(observation_vector)` are the simulator's
  `outputs`; a prediction's Flat-to-labeled conversion is by label through
  `observation_labels`, never by Flat position.
- P6: `noise_factor` reads `observation_vector.constants(source)` and, for a
  grouping by site, the `site` level of the observation dim. About 10% of
  MODIS LAI standard deviations are zero locally, which the source accepts;
  the covariance's positive-definiteness is `condition_on`'s to check.
- Two stacked dims with a `site` level cannot share an `xr.Dataset` or one
  DataArray (xarray refuses the shared level), which is why labeled values
  are a dict.

### 2026-10-05: P5, the simulator seam

**Done.**

- **The core.** `probability.parts` gains `Simulator`, an abstract base:
  `name`, `given`, `outputs`, `__call__(labeled values) -> SimulatorOutput`,
  `at(coords, outputs)`, and an optional `check_given`. It also gains
  `SimulatorOutput`: values `(J, *block)`, validity `(J,)` and a record.
  - `joint` takes simulators.
  - `bind` binds each simulator with `at`, every output, and runs none.
  - `FactoredDistribution.sample` and `log_prob` run each simulator once
    per batch. What is drawn or computed from a failed output is `NaN`,
    and a density that reads one is `-inf`.
- **The posterior.** `condition_on` does three new things:
  - it refuses a target factor, and a factor of `O_c`, with a simulator
    among its ancestors;
  - it restricts each simulator to the outputs the likelihood reads;
  - it calls `check_given` at the corner points, which are the core's now
    (`_probes.corner_points`, `CORNERS`).

  `Posterior` gains `evaluate -> PosteriorEvaluation`, `log_density_given`,
  `predict`, `replicate`, `simulator_inputs`, `simulators` and
  `simulator_free_positions`. `log_likelihood` and `log_density` go through
  `evaluate` when a simulator is upstream of the likelihood, and stay
  traced otherwise.
- **The adapter.** `forward.SIPNETRuns` holds what `ForwardModel` ran. Its
  `evaluate(values, observation_vectors=, output_variable_names=, freq=,
  external_inputs=)` makes one pass for several reductions (F6) and returns
  a `SIPNETRunsEvaluation`. `ForwardModel` delegates to it; one private
  `test_forward` test was updated, the rest are unchanged.
  `forward.SIPNETSimulator` is the forward map as a `Simulator`:
  - it runs only the sites its vector observes;
  - it marks a source's prediction invalid only where a run at one of that
    source's sites failed;
  - `at` restricts it to fewer sources and sites;
  - `check_given` is the map's fit check plus, under `"raise"`, the corner
    domain check.
- **F7.** It needed no code. An external initial state is an input the
  simulator reads, and a name declared both ways is refused by `joint`.
  Both are tested.
- **Tests.** 2858 passed and 94 skipped at P4's merge; 2909 and 94 after.
  - The SIPNET simulator's predictions equal P1's `forward_example` bit for
    bit, and P1's byte test still passes through the delegating
    `ForwardModel`.
  - A failing run invalidates one source and not the other.
  - One pass equals three calls.
- **Shared test helpers.** `tests/conftest.py` gains
  `two_source_observation_vector` (P1's script now uses it) and
  `example_calibration_factors` (the parity test now uses it).

**Review.** One Standard round: code, mutation testing, docs. Fixed:

- a deterministic downstream of a failed output was not masked in
  `_computed`, so `log_prob` and `PosteriorEvaluation.values` could be
  finite at a failed sample; `log_prob` now also maps each factor to `-inf`
  where what it reads failed;
- `condition_on` ran an upstream simulator at every corner point for a
  chain of simulators;
- a traced theta raised JAX's `TracerArrayConversionError` rather than a
  `ValueError` naming the cause;
- `sample(key, 0)` called the simulator with no samples and crashed;
- `log_density_given` had `NaN`, not zero, gradients at failed rows;
- `simulator_free_positions` counted simulators only barren factors read;
- `SimulatorOutput` was neither keyword-only nor frozen, as documented;
- Raises sections, stale annotations and docstrings, CLAUDE.md's
  forward-model bullet, messages without a fix, and a stale check name.

Mutation testing found ten survivors. Tests now kill the masking, the
two-outputs and finiteness rules, the theta mask and `log_density_given`'s
`-inf`. Three remain:

- the per-source `ran` and `in_domain` terms, which `NaN` placement already
  implies, so the mutants are equivalent;
- `_block_order`, the identity for every vector the public API builds;
- `NaN` in a row out of the domain whose runs succeeded, which needs a
  contrived `fail_row` map.

**Deviations from the design**, each recorded in `design.html`:

- A factor downstream of a simulator is checked at bind at a placeholder
  (each output's bijector's image of 0) for its form only, not with
  `jax.eval_shape`.
- `SIPNETRuns.evaluate` returns a `SIPNETRunsEvaluation`, since
  `ForwardEvaluation` keeps `theta` for `ForwardModel` and PR #69 until R1.
- `SIPNETSimulator.at(coords, outputs)` takes the core's signature, and
  `SIPNETRuns.select(sites=)` is added for it.
- No factor of `O_c` may have a simulator ancestor.
- `PosteriorEvaluation.values` holds the deterministics the likelihood
  reads and those computable without a simulator.

**Choices the design left open:**

- `log_likelihood` and `log_density` stay traced without a simulator.
- `replicate` draws the `O_theta` factors ancestrally.
- `predict` computes its own pruned outputs.
- A run plan is built once by `ForwardModel` and `SIPNETSimulator`, and
  per call by `SIPNETRuns.evaluate`.
- A combined pass checks the union of its output variables finite.
- `ForwardModel` imports the core's private `_probes` until R1.

**What P6 and later must know.**

- A Gaussian noise factor (P6) reads `predicted_<source>` from
  `SIPNETSimulator`, whose block is in `observation_vector.coords` order,
  by label.
- `BoundFactor.law_at` for a `GaussianSpec` must cope with the placeholder
  predictions at bind: the bound checks are structural only downstream of
  a simulator.
- `Posterior._likelihood_at` sums the O_theta factors and reports
  finiteness. `gaussian_likelihood` (P6) and the inference adapters (P7)
  read `evaluate`'s `valid`, which `simulator_valid` refines (NaN for
  SMC's failed runs; `-inf` where only the traced part failed).
- `FactoredDistribution._computed`, `_ancestral` and `_simulate` take a
  `simulators` mapping, so a caller can pass pruned ones, and record each
  run's `SimulatorOutput` in `runs`.

### 2026-10-05: P6, the Gaussian observation model

**Done.**

- `probability/_linalg.py`: the one shim over pyEKI, re-exporting
  `Gaussian`, `PSDLinOp`, `PSDDiagonal`, `DensePSD`, `PSDScaled`,
  `PSDBlockDiag`, `block_diag` and `UnsupportedOpError`. `tests/test_package.py`
  now allows pyEKI in `parameters` and `probability` only through it, and
  checks it is the one file importing pyEKI.
- `probability/covariance.py`: `CovarianceSpec` and `DiagonalSpec`,
  `DenseSpec`, `SumSpec`, `ScaledSpec`, `BlockDiagonalSpec`,
  `SubmatrixSpec`. Each is bound to its scope at `bind` (`_at(scope)`: the
  groups are fixed then) and builds a pyEKI operator, or a dense matrix for
  a sum, per draw.
- `probability.parts.GaussianSpec(mean=, covariance=)`, a factor's fourth
  law form; `probability.laws.GaussianLaw`, what it evaluates to, a `Law`
  over the block holding the operator and the pyEKI `Gaussian`; `as_law`
  turns a `pyeki.gauss.Gaussian` into one.
- Binding (`model.py`, `_bound.py`): a Gaussian factor's covariance spec is
  bound with it; `joint` checks its mean matches its event in layout and
  units; `describe` says `"Gaussian"`.
- `Posterior`: a Gaussian factor's covariance computable from the held
  values is built and factored once at `condition_on` (refused there if not
  positive definite), and `gaussian_likelihood() -> GaussianLikelihood`
  (`y`, `noise_covariance` block-diagonal in y's order, `forward(theta) ->
  (predictions, valid, evaluation)`, `mean_names`, `posterior`).
- `observation.model.noise_factor`: one source's Gaussian factor, holding
  the source constants its covariance reads.
- Tests: 2909 passed and 94 skipped at P5's merge; 2987 and 95 after (one more
  skip: the real-data test below, without processed files).
  - Every covariance spec against the dense matrix it stands for, on a
    stacked dim of three sites with ragged times, and its density against
    SciPy; nested groupings, per-site scales, submatrices of an
    inverse-Wishart component.
  - PR #69's one-site `R`: its four block builders and `config.py` values
    transcribed, against noise factors on four synthetic sources shaped as
    its own, equal to 1e-12; its pyEKI `Gaussian`'s log density equals the
    posterior's log likelihood. A second test repeats the three constraint
    blocks on the local processed files at Harvard Forest (4977), skipped
    without them; it passes with `SIPNET_CALIBRATION_DATA` pointed at the
    root's `data/`.
  - The Gaussian likelihood's forward map through `SIPNETSimulator` equals
    P1's `forward_example` predictions, by label, in y's order.

**Review.** One Deep round, four reviewers: numerics, edge cases,
mutation testing, docs. The numerics reviewer found no wrong density,
ordering or held-covariance decision. Fixed:

- binding refused a model whose covariance reads a parameter and is not
  positive definite at one of the two ancestral draws (a correlation with a
  wide prior); such a covariance is now checked per sample only, and its
  draws are `NaN` where it fails, not "a value on its support's boundary";
- a block-diagonal inside a sum's scaled or nested term was summed as a
  dense matrix over the whole event; it is refused;
- a per-site constant could not be read at each group's label unless some
  component was indexed by `site`; a factor's `own_dims` now label it, its
  constants on one own dim sharing their labels;
- `BlockDiagonalSpec(by=[])` and a repeated level failed in pyEKI's words;
- `noise_factor` with a covariance that is not a spec, or constants that
  are not a mapping, raised Python's errors; one reading a constant its
  source lacks (a static source's times, a standard deviation not given)
  is now refused there, by name;
- a Gaussian mean naming a constant raised a bare `KeyError`; a mean of the
  wrong shape was reported as a wrong covariance;
- docs: `laws`' claim about pyEKI's Gaussian, `DenseSpec`'s symmetric part,
  helper names (`_slice_to_group`, `_evaluate_argument`), import order,
  method order, a "held" glossary row.

Mutation testing made fifteen mutations, of which six survived; tests now
kill four of them (a scaled term
in a dense sum, a matrix-shaped `GaussianLaw` block, a scale on
`NON_NEGATIVE`, a group split by one entry). Left: the early return for a
simulator behind a covariance in `_with_held_covariance`, a guard that
keeps `condition_on` from running a simulator reading only inputs; and the
finiteness term of `SIPNETSimulator`'s validity, which P6 did not change.
Judgment calls are the open questions above (correlated sources, the
reading of D21, asymmetric matrices). Decided by Andrew after the review:
`noise_factor` takes one source, `observation_source_name: str`, against
the design's `str | Sequence[str]`; a sequence is a `TypeError`.

**Deviations from the design**, each recorded in `design.html` ("As built
in P6"):

- one mean, and an event of one component on `REAL` indexed by one dim at
  most with no element axes; correlated sources are refused (open question);
- `noise_factor(observation_vector, observation_source_name, /, ...)`,
  singular;
- `GaussianLaw` is the layer's own `Law` over pyEKI's operators;
- a Gaussian factor gets no probe-point checks; its covariance is checked
  positive definite at bind only when it reads nothing that varies by draw;
- "depends on theta" is "not computable from the held values";
- `BlockDiagonalSpec`'s groups must be contiguous; `SumSpec` sums dense
  matrices and factors once (a term may be semi-definite);
- `GaussianLikelihood` holds `posterior` and `mean_names`; its
  `noise_covariance` over one factor is that factor's operator.

**What P7 and later must know.**

- P7's `eki_problem` reads `gaussian_likelihood()`: `y`, `noise_covariance`
  (a pyEKI `PSDLinOp`, ready for `pyeki.eki`), and `forward(theta)`, which
  makes one `evaluate` and returns `NaN` rows where invalid. Keep the
  evaluation it returns as `last_evaluation`.
- A block-diagonal covariance over many sites builds one block per site in
  a Python loop per draw; at hundreds of sites, group equal sizes and vmap
  (the design's §14 risk).
- `FactorSpec(law=pyeki_gaussian)` is still refused: `as_law` adapts one
  only inside a law function. P9's foreign-law work should recognize it,
  and EnsKit's `Gaussian` (E2) through the same shim.
- `_with_held_covariance` replaces target and likelihood bounds; the
  model's own `_factors` keep per-draw covariances, so `sample`,
  `replicate` and `predict` build them per draw.

### 2026-10-05: P7, the inference adapters

**Done.** A new package, `sipnet_calibration.inference`, reading a
`Posterior` and nothing of SIPNET:

- `eki`: `EKIProblem` and `eki_problem(posterior)`. The problem holds the
  posterior and its `GaussianLikelihood`, `y`, `noise_covariance` (pyEKI's
  operator), `forward(theta) -> (J, N)` with `NaN` rows where invalid,
  which replaces `last_evaluation`, and `initial_ensemble(key, n)`, theta
  `(n, D)` from the prior.
- `tempering`: `PriorBaseDensity` (the prior in theta as an
  `smc.BaseDensity`) and `tempering_problem(posterior, *, base=None)`. Its
  log likelihood is `NaN` where `simulator_valid` is false and `-inf` where
  only the traced part failed; with a Gaussian likelihood the predictions
  travel as `smc`'s auxiliary values.
- `mcmc`: `batched_log_density` (`(J, D) -> (J,)`), `log_density`
  (`(D,) -> float`), and `initial_points(posterior, key, n, *, max_draws=)`.

The package imports `probability`, `smc` and `validation` only, and no
algorithm package; `tests/test_package.py` checks it.
Tests: 2988 passed and 95 skipped at P6's merge; 3019 and 95 after.
`tests/test_inference.py` declares a linear-Gaussian toy in the
probability layer (a correlated Gaussian prior on three coefficients, a
simulator `m = A u` that can fail on a half-space, `y ~ N(m, R)` with `R`
dense) and checks, against closed forms:

- EKI from an ensemble with the prior's exact moments is the posterior to
  1e-8 (pyEKI's affine-Gaussian claim), and from `initial_ensemble`'s
  draws within Monte Carlo error;
- tempered SMC from the prior and importance sampling from a Student-t
  give the posterior's moments and the evidence; tempered SMC from the
  Student-t gives the posterior truncated where the simulator fails, and
  its evidence;
- the MCMC densities are SciPy's log posterior, `-inf` where it fails;
  `initial_points` picks the first finite draws in order.

Each Monte Carlo tolerance is about twice the largest error over eight
seeds. SMC from the prior misses the evidence by 0.05 to 0.12 on every
seed; `test_smc` measures 0.113 for the same kind of run, so it is the
tempered estimator's, not the adapter's.

**Review.** One Standard round: code, mutation testing, docs. The code
reviewer found no bug: the failure semantics, the evidence's
normalization, the stitched rows and the seeding held, with and without a
simulator, with nothing observed and with a non-Gaussian likelihood.
Fixed:

- a base density with no `dimension` raised `ValueError` naming
  `None`, and a boolean one passed; both are now `TypeError`s;
- `initial_ensemble` accepted fewer than two members, which pyEKI then
  refused in its own words;
- a check of two invariants (rank and `D`) split into two and a group;
- docs: the evidence's relation to `log_constant`, which adapters run a
  simulator, the dependency diagram missing `validation`, a Notes section
  for `initial_points`' batching, Usage snippets missing `import jax`, the
  design's §9 bullets, CLAUDE.md's "Where shared things live", and the test
  module's description of its runs.

Mutation testing made twenty mutants, of which eleven survived; tests now
kill eight of them: the batch sizes (the old test's tenth finite draw
ended a batch of ten either way), the stitched `log_prior`,
`log_likelihood` and a partial shortfall, the non-Gaussian branch's
traced failure, a base of smaller dimension, the base ignoring its
generator, and a wrong `D` given to `log_density`. The rest are
equivalent: every kept row is valid, so `valid`, `simulator_valid` and a
finite log density agree there, and swapping or dropping one changes
nothing. Two findings are questions above, not
fixes (`initial_points`' batches and records).

**Deviations from the design**, each recorded in `design.html` ("As built
in P7"):

- the package imports no algorithm package, where §9's introduction calls
  it the one place that does: pyEKI and emcee are the experiment's imports;
- `initial_ensemble` returns theta `(n, D)`, today's `EKIState` input,
  until EnsKit's `Ensemble` (E3), and refuses `n < 2`; `EKIProblem` also
  holds `posterior` and `likelihood`;
- the tempering log likelihood carries auxiliary values only when the
  likelihood is Gaussian.

**Choices the design left open:**

- `tempering_problem` finds the Gaussian likelihood by calling
  `gaussian_likelihood()` and treating its `ValueError` as "none", which
  today is raised for exactly the three reasons a posterior has none;
- `PriorBaseDensity.sample` seeds its JAX key with one 63-bit integer drawn
  from the generator;
- `batched_log_density` and `log_density` go through `evaluate` even with
  no simulator, unjitted, so validity is the same on both paths;
- `initial_points` keeps a draw where it is valid and its log density
  finite, in batches the size of the shortfall (open question above).

**What the next sessions must know.**

- PR #69's migration reads `eki_problem(posterior)`; its `_RecordingForward`
  becomes `problem.forward` and `problem.last_evaluation`. A run's terminal
  `pyeki.eki.evaluate` also goes through `problem.forward`, so
  `last_evaluation` is then the final ensemble's.
- Today's EKI started from `prior_gaussian`; `initial_ensemble` draws the
  prior itself, so the same seed gives a different start (design §10.2).
- E3 replaces `initial_ensemble`'s array with EnsKit's `Ensemble`; nothing
  else in `eki.py` names pyEKI.
- P8 (conjugacy) is next in table order; it needs P6 only.

### 2026-10-05: P8, conjugacy

**Done.** The design's §7.13 rules, which the code calls the **scale
rule** (R1, an inverse gamma on a covariance scale) and the **block rule**
(R2, an inverse Wishart on a covariance block), since R1 is also the
removal PR's name:

- `probability/scale_mixtures.py`: `StudentTSpec` and `MatrixStudentTSpec`,
  law forms a factor may have, as `GaussianSpec` is, evaluated to
  `StudentTLaw` and `MatrixStudentTLaw`; `inverse_wishart_log_prob` and
  `sample_inverse_wishart`, unchecked and traceable.
- `probability/conjugacy.py`: `conjugate_rule(model, name) ->
  ConjugateRule` (the matched factors, the marginal law, and the full
  conditional's parameters from residuals), `marginalize`,
  `InverseWishartGivenRows`, and the rule names `INVERSE_GAMMA_SCALE` and
  `INVERSE_WISHART`.
- `FactoredDistribution.marginalize(names)`;
  `Posterior.full_conditional(name) -> FullConditional` (`law(evaluation,
  sample)`, `sample(key, evaluation)`), in `posterior`;
  `Posterior.theta_with(theta, values)`.
- `parts.CENTERED_LAW_SPECS`: `GaussianSpec` and the two Student-t forms,
  which binding, the event check, the mean check and the ancestral draws
  treat alike.

Tests: 3019 passed and 95 skipped at P7's merge; 3072 and 95 after.
`tests/test_probability_conjugacy.py`:
- the scale rule's marginal against quadrature over the scale (a scale
  per site, one shared by the sites, one over the whole covariance,
  per-label priors), and its draws' moments;
- its full conditional against the joint density in the scale, its closed
  form (summed over groups for one scale; placed by label whatever the
  coords' order; a site observing nothing keeps its prior), and a
  60,000-step random-walk Metropolis chain;
- the block rule's marginal against Monte Carlo over the prior, with every
  row observed and with some, and its draws' covariance;
- its full conditional against the joint density in the matrix, and its
  closed-form moments, the unobserved rows drawn from the prior given the
  observed;
- a Gibbs step that draws from an evaluation with no new simulator run,
  `NaN` where the mean was not computed; `theta_with`; the refusals.

The shared models are bound once per module (`functools.cache`): binding
checks each factor with thousands of draws, and the file took five minutes
without it.

**Review.** One Deep round, four reviewers: numerics, edge cases, mutation
testing, docs. The numerics reviewer found no wrong density: every closed
form and sampler agreed with SciPy, quadrature or Monte Carlo, the
completion of the matrix from the prior included. Fixed:

- the block rule's full conditional raised at a sample whose mean was not
  computed, when the groups observe every row, building a checked
  `InverseWishart` from `NaN`; it is now always an
  `InverseWishartGivenRows`, which is the inverse Wishart itself when
  every row is observed, `NaN` there;
- `inverse_wishart_log_prob` and the matrix Student-t took `slogdet` and
  dropped its sign, giving a finite value for an indefinite scale; they now
  factor it, `NaN` there;
- `theta_with` let JAX raise for a batch that does not broadcast to
  theta's, and took a boolean;
- `marginalize` took a name twice; `full_conditional` raised `KeyError` for
  a name that is not a string;
- the prior's reads were checked before its law, so a Gaussian factor was
  refused with advice about a shape and scale; a prior reading only inputs
  is now accepted, its parameters fixed;
- `StudentTSpec` refused a 0-d array and reached pandas' error for repeated
  labels;
- docs: an unused pyEKI shim import, the dependency diagram, stale
  "Gaussian" text, "R1" colliding with the removal PR, `shared_rows`
  public, a third copy of `check_seed_is_given` (now one, in `laws`).

Mutation testing made 17 mutants, of which 4 survived; tests now kill
three (the shape summed over groups for one scale, each group's place by
label, a scale that is also the mean). The fourth, dropping the sort of
the observed rows, is equivalent: each group's entries are already in row
order. Three findings are questions above.

**Deviations from the design**, each recorded in `design.html` ("As built
in P8"):

- the marginal factor's law, which the design left unnamed, is a new law
  form holding the prior's numbers as labeled arrays, so a marginal model
  selected at fewer labels is the marginal of the selection;
- `FullConditional` is in `posterior` (§6's module list) and holds
  `posterior`; the rules are a new `conjugacy` module, with a public
  `ConjugateRule`;
- the scale rule recognizes its prior from the bound law; a scalar scale
  under a block-diagonal grouping gives one Student-t over the event;
- the block rule's full conditional is checked against the joint density
  and its moments rather than a chain over a matrix.

**Choices the design left open:**

- names are integrated out in order, each from the model the last left;
  the marginal factor takes the reader's place in declaration order, and
  its provenance joins the reader's and the prior's;
- a prior reading only inputs counts as "reading no component";
- a Student-t's covariance is built per evaluation, not held at
  `condition_on` as a Gaussian's is;
- `theta_with` checks a value has a theta only when it is concrete, so it
  stays traceable.

**What the next sessions must know.**

- A Gibbs sampler alternates `posterior.evaluate(theta)` (one simulator
  batch), `full_conditional(name).sample(key, evaluation)` and
  `theta_with(theta, {name: draw})`. Rows where the evaluation is invalid
  draw `NaN`, which `theta_with` refuses: keep the valid rows.
- `CENTERED_LAW_SPECS` is where a new law centered on a mean joins the
  binding (P9's EnsKit `Gaussian` adapter may want it).
- P9 (foreign laws) is next in table order; it needs P3 only. #69's
  migration and R1 follow.

### 2026-10-05: P9, laws from other packages

**Done.** Additive, in `probability`:

- `laws`: `as_law` adapts a numpyro distribution (GPJax's
  `GaussianDistribution` among them) to a new `NumpyroLaw`, and a
  `pyeki.gauss.Gaussian` to a `GaussianLaw`, as P6 began; `is_law` answers
  for both. `NumpyroLaw` reorders `sample`'s arguments and vmaps `log_prob`
  over the axes in front of its batch and event, which GPJax's does not
  broadcast. `pushforward` of a TFP base is unchanged (an exact
  `tfd.TransformedDistribution`); of any other base it is a new
  `PushforwardLaw`.
- `_numpyro` (private): numpyro's classes recognized by the qualified names
  in their MRO, with no import; which have no density; what a wrapper
  wraps.
- `_bound`: a law function's result is adapted; a `PushforwardLaw` through
  the components' own bijectors is evaluated by its base density; the
  no-density check looks inside adapters and numpyro's wrappers and
  mixtures; numpyro's `Dirichlet` (under `Independent` or `expand` too) is
  accepted on the simplex; a structure change inside an adapter is caught.
- `parts.FactorSpec` holds a bare law adapted, so `FactorSpec(law=gaussian)`
  with a pyEKI `Gaussian` works (P6's open item). `builders` refuse a law
  that is not TFP's.
- `pyproject.toml`: numpyro in `dev`, GPJax in a new optional `gpjax` group;
  `uv.lock` only adds packages.

Tests: 3072 passed and 95 skipped at P8's merge; 3101 and 95 after, with the
`gpjax` group installed (one more skipped without it). `tests/test_probability_law_adapters.py` checks
each adapted law against TFP or SciPy (densities in theta and at natural
values, draws), the §10.3 GPJax field against TFP's `MaternThreeHalves`
process, and the refusals; `tests/test_package.py` checks that no file
imports numpyro or GPJax.

**Review.** One Standard round: code, mutation testing, docs. Fixed:

- `PushforwardLaw.log_prob` summed the bijector's log-Jacobian over the
  base's rank rather than the value's, so any non-TFP base on
  `POSITIVE_DEFINITE` (`FillScaleTriL`, rank 1 to 2) raised;
- `NumpyroLaw.log_prob` mis-shaped a value with fewer axes than the
  distribution's batch;
- a numpyro law with `float32` parameters passed the `float64` check, its
  draws being `float64` under x64; `NumpyroLaw.dtype` now reports the
  parameter's;
- `PushforwardLaw.dtype` raised `AttributeError` for a base with no `dtype`;
- an expanded numpyro `Dirichlet` was refused on the simplex;
- advice to wrap a numpyro law in `iid_over_dim`, which refuses it;
- docs: the glossary's *law*, `pushforward`'s Raises, stale "TFP only"
  messages, the test module's name ("foreign" is not a glossary word) and
  docstring, the design's account of discrete laws.

Mutation testing made 24 mutants, of which 5 survived; tests now kill all
of them (a structure change inside a `NumpyroLaw` and a `PushforwardLaw`,
the unwrapping of `Independent`, `MixtureGeneral`'s components).
P3's `test_pushforward_takes_a_tfp_base` pinned the restriction this PR
lifts; it is now `test_pushforward_takes_a_law`.

**Deviations from the design**, each recorded in `design.html` ("As built
in P9"):

- the project still pins pyEKI from before EnsKit's rename, so pyEKI's
  `Gaussian` is adapted, through the `_linalg` shim; E1 moves the shim and
  the adapter to EnsKit's;
- numpyro is recognized by class name, not imported, and is not a
  dependency of the package;
- `pushforward` of a non-TFP base returns a `PushforwardLaw`, the design
  having said only "a law";
- the builders stay TFP's and refuse other laws.

**Choices the design left open:** numpyro's `Delta`, `Unit` and LKJ laws are
refused by class and its discrete laws by their support, beside TFP's list;
a numpyro discrete law draws integers, so the dtype check refuses it first.
`CENTERED_LAW_SPECS` was not needed: a pyEKI `Gaussian` given as a law is a
fixed law over its block, not one centered on another component.

**What the next sessions must know.**

- The GPJax test runs only after `uv sync --group gpjax`; a plain
  `uv sync` removes the group again.
- numpyro's own deprecation warning (`is_prng_key`) comes from GPJax's
  sampler, not from the adapter.
- GPJax's `GaussianDistribution` flattens to a pytree with no leaves, so a
  structure check cannot see its parameters; the adapter's class and the
  distribution's class are what it compares.
- E1 is next, once this PR merges: the re-pin to EnsKit's `main` (see
  "Decided by Andrew"). Then E2, then #69's migration in its own session,
  then R1.

### 2026-10-05: E1, the re-pin to EnsKit

**Done.** The project tracks EnsKit's `main` (`TARPS-group/EnsKit`, package
`enskit`) in place of pyEKI.

- `pyproject.toml`: the dependency `pyeki` is `enskit`. `uv.lock` adds
  EnsKit 0.1.0 at 38df902, moves JAX 0.8.3 to 0.10.2 (EnsKit requires
  `>= 0.10.1`) and `tfp-nightly` to its 2026-10-05 build, the one the plan's
  scratch check used, and removes pyEKI. Nothing else moves: NumPy stays
  2.4.6, the scratch check's 2.5 having come from a fresh lock.
- `probability/_linalg.py` imports the same names from `enskit.linalg` and
  `enskit.distribution`; the operators' constructors are unchanged.
- `GaussianLaw` holds an EnsKit `Gaussian` of one block, its covariance the
  block's independent term, and scores through its `log_density`; it still
  draws `m + L z` itself. `as_law` adapts an EnsKit `Gaussian` of one block
  through `Gaussian.cov`, a factor row included, and refuses one of several
  blocks and a block with no independent term (two new checks).
- Tests: `test_inference`, `test_smc` and `test_forward`'s EKI test run
  EnsKit's driver with `kalman.SymmetricSquareRoot()`, the counterpart of
  pyEKI's default `TransformUpdate`; the tests that built pyEKI's
  `Gaussian(mean, cov)` build `Gaussian.independent(name=(mean, cov))`;
  `test_package` names `enskit`, and its type-hint check skips names
  re-exported from another package.
- P1's references: JAX 0.10 and the TFP build moved five of the six files by
  round-off (at most 1e-15 relative: draws, `log_prob`, allocations;
  `forward_example`'s predictions did not move, only its theta). They were
  rewritten from P4's merge (dd396fd, before P5) in this environment, and
  today's code writes the same bytes, so P1's rule for `forward_example`
  holds. `copula` did not move.
- Docs: CLAUDE.md's companion table, upgrade command, workflow step 4,
  glossary rows, layout and an "EnsKit" section of facts (each checked
  against the installed package) replacing the pyEKI facts; the README;
  every docstring naming pyEKI; `design.html` "As built in E1" and the
  sections that described pyEKI as current.

Tests: 3100 passed and 96 skipped before; 3102 and 96 after (the two new
`as_law` tests). The GPJax test passes under JAX 0.10 after
`uv sync --group gpjax`. EKI from prior draws through EnsKit's driver, over
eight seeds: at most 0.028 posterior sds in the mean and 0.5% in an sd, so
the tolerances (0.06, 1.5%) keep their margin.

**Review.** One Standard round: code, mutation testing, docs. No bug in the
changed mathematics or control flow. Fixed:

- the refusal of a block with no independent term said its covariance has
  no density, which is false when the factor has at least as many columns as
  the block has entries (an `Ensemble.project()` with `J > d`); it now says
  a `GaussianLaw` cannot whiten the factor alone;
- the private block name, which EnsKit's debug messages print, is
  `"GaussianLaw event"` rather than `"block"`;
- a mutant flattening a matrix block's mean in Fortran order survived; the
  matrix-block test now has a nonzero mean;
- docs: the design's stale present-tense pyEKI passages, a circular phrase,
  "particle" for a sample, a `#:` comment, the JAX requirement's wording,
  what an inf row scores, a release number that would go stale.

Not acted on (nits): a vmapped EnsKit family is refused by EnsKit's guard
rather than ours; `log_density` formats `repr(self)` per call, negligible
under `jit`; no test passes `float32` values to `GaussianLaw.log_prob`
(pre-existing).

**Deviations from the design.** None. Choices the design left open: the
EKI tests use the deterministic rule; `as_law` refuses rather than reads a
multi-block Gaussian, as P6 refused correlated sources.

**What the next sessions must know.**

- Run CLAUDE.md's standard companion upgrade now: `enskit` is upgraded with
  pySIPNET and PyEns.
- JAX 0.10 makes TFP's Gamma sampler, which `Dirichlet` uses, emit
  "shape requires ndarray or scalar arguments, got NoneType ... will be an
  error" from `bind`'s support check (`_bound._draws_lie_in_the_support`).
  It is TFP's code; a JAX bump that makes it an error breaks `bind` for any
  Gamma-family or Dirichlet law until TFP's nightly fixes it. Run
  `pytest tests/test_probability_law_adapters.py -W "error:shape requires
  ndarray:DeprecationWarning"` to see whether it has.
- `typing.get_type_hints` cannot resolve EnsKit's `Gaussian` (`Array` is
  imported for type checking only); EnsKit's to fix, if anyone reports it.
- Under `enskit.linalg.set_debug_checks(True)`, a `GaussianLaw` with a
  non-finite mean raises at construction, where pyEKI's scored NaN. Debug
  mode is opt-in and the layer never turns it on.
- E2 is next: `eki_problem` in the terms of `enskit.algorithms.eki`, with
  `initial_ensemble` an `Ensemble`, and §9.1 of the design rewritten. Then
  #69's migration in its own session, then R1. #69's branch breaks on
  rebasing or re-locking until it ports its `pyeki` imports.
