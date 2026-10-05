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
| P6 | `refactor/probability-p6-gaussian` | open #PR | P5 | Covariance specs, `GaussianSpec`, `noise_factor`, `gaussian_likelihood`, through the `probability/_linalg.py` shim over today's pyEKI |
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
- **`noise_factor`'s source argument (P6, found in review).** The design
  writes `noise_factor(vector, observation_source_names: str |
  Sequence[str], /, ...)`, and P6 keeps it, accepting one name or a
  sequence of one. CLAUDE.md's rule refuses a bare string for a sequence
  argument, and a name argument is singular. Recommended: make it
  `observation_source_name: str` now, and add the plural form with the
  correlated-sources factor (open item below); the alternative keeps the
  design's signature, which breaks the convention until then.
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
Judgment calls are the open questions above (`noise_factor`'s argument,
correlated sources, the reading of D21, asymmetric matrices).

**Deviations from the design**, each recorded in `design.html` ("As built
in P6"):

- one mean, and an event of one component on `REAL` indexed by one dim at
  most with no element axes; correlated sources are refused (open question);
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
