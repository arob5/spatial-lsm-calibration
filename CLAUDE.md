# CLAUDE.md — spatial-lsm-calibration

## Project purpose

This repository develops and applies **scalable Bayesian algorithms for parameter calibration of the SIPNET land-surface model**, with an emphasis on multi-site inference that exploits spatial structure (going beyond plant functional types). It is a research codebase, not a package — the deliverables are calibrated parameter ensembles, diagnostic outputs, and reusable inference machinery.

The long-term vision:
- Many sites, many runs, potentially many algorithms
- Real flux-tower / ERA5 data (symlinked from storage)
- Spatial hierarchy: partial-pooling priors or GP-based spatial priors over sites
- Full reproducibility: every run's configuration is one source of truth

## Companion packages (read-only — do not edit)

| Package | Source | Role |
|---------|--------|------|
| `pySIPNET` | `TARPS-group/pySIPNET` | SIPNET model interface; `SIPNETModel(**overrides)` |
| `PyEns` | `arob5/PyEns` | Parallel ensemble execution via `ProcessPoolExecutor`, and the xarray-to-Grid bridge (`pyens.xarray`) |
| `EnsKit` | `TARPS-group/EnsKit` (package `enskit`; formerly pyEKI) | Ensemble Kalman methods: structured linear operators (`enskit.linalg`), Gaussians and ensembles over named blocks (`enskit.distribution`), and the EKI driver (`enskit.algorithms.eki`) |
| `ProbPipe` | `TARPS-group/prob-pipe` (also on PyPI) | **Not currently a dependency** — API in flux; planned migration target for inference. See below. |

The first three are dependencies, installed from git rather than from sibling
directories: `[tool.uv.sources]` tracks each repository's `main` branch and
`uv.lock` pins an exact commit. A sibling clone of any of them is **not** what
gets installed — `uv` fetches the pinned commit from GitHub — so nothing about
a local checkout reaches this project, and unpushed work in one is invisible
here. Never modify their source from here.

The boundary between them: PyEns owns the shape of an ensemble and the
translation of labeled data into and out of it; pySIPNET owns one SIPNET run's
inputs and outputs; EnsKit owns the ensemble Kalman update; this repository owns
what varies and why. A function belongs in pySIPNET only if deleting SIPNET from
it leaves nothing.

**All three are under active development, so start any work here by taking
their current `main`:**

```bash
uv lock --upgrade-package pysipnet --upgrade-package pyens --upgrade-package enskit
uv sync
```

`uv sync` on its own never re-reads a companion's branch: it installs the
commit `uv.lock` already pins. The `--upgrade-package` line is what goes back
to each `main`. Run both in the checkout or worktree whose `.venv` you are
using, since each has its own.

When a companion has not moved this is a no-op and leaves `uv.lock` untouched,
so it does not dirty every branch. When one has, `uv.lock` changes — and that
change is a commit like any other, the record of which version of each package
a run used, so commit it rather than carrying it. If you were only testing and
want it gone, `git checkout origin/main -- uv.lock` restores the base version
(ask first if others share the checkout, per the destructive-command rule
below), and the venv keeps the newer package until the next `uv sync`.

Two consequences worth knowing. A companion's change reaches this project only
once it is **pushed** to that repository's `main`. And upgrading pySIPNET does
not install a SIPNET binary: see the pySIPNET notes under "Key API facts".

For tight iteration on a companion, where pushing before every check is too
slow, the README covers overlaying an editable install on top of the synced
environment. It makes a local clone live, at the cost of a venv that no longer
matches `uv.lock` — which is how a checkout drifts without anyone noticing, so
undo it with `uv sync` when done.

## Data

**`data/README.md` is the authority on data formats, provenance, units, and the
open questions.** Read it before writing ingest or adapter code, and do not
duplicate its content here — add data facts there instead.

Facts specific to this working copy, which the README deliberately does not carry:

- **Only a subset of `data/raw/` is present locally.** Drivers exist for
  `ERA5_1_1`, `ERA5_1_2` and `ERA5_27_5`; of PEcAn's initial condition source
  files, site 1 members 1 and 2 and site 27 member 94, under
  `data/raw/initial_conditions/files/`; of the soil texture ensemble, sites 1
  and 27 only, three files of 769,300, which is why
  `scripts/survey_soil_texture.py` needs `--no-check` here. The two converted
  NEE time series (`data/raw/net_ecosystem_exchange/*.nc`, not tracked, copied
  from the SCC) and AmeriFlux's site listing beside them, the NEE tower table
  (tracked), the five constraint files (tracked), the converted initial condition ensemble
  (tracked, all 8000 sites x 100 members), the assembled `.Rdata` pair retained
  for validation, the site shapefile, both leaf phenology CSVs, and the two
  site-labels data sources and the covariate table (all tracked) are complete.
  The full dataset lives on Boston University's SCC. Anything about the drivers
  that needs to hold across all 8000 sites cannot be verified here.
- In the root checkout the storage-backed inputs are real copies, not symlinks;
  on SCC, and in a worktree that links them from the root, they are symlinks.
  The five constraint files and the site shapefile are tracked either way.
- **A tracked raw input is found from the repository, not from
  `conventions.data_root()`.** `data_root()` (and `$SIPNET_CALIBRATION_DATA`)
  says where the storage-backed part of `data/` is, which on the SCC or with
  the variable set is another tree; a tracked file is always in the checkout.
  The tests (`conftest.REPOSITORY`), the Natural Earth scripts,
  `split_site_pft_16class.py` and the library's own `default_raw_directory()`
  resolvers for tracked directories (through
  `conventions.tracked_data_root()`, which falls back to `data_root()` under a
  non-editable install, where there is no checkout) follow this.
- R is available on this machine (`Rscript`), which is how the `.Rdata` files can
  be inspected; `pyreadr` is not installed and would not handle their nesting.
- **`pyproj` installs here.** Issue #4 recorded that it could not, on the
  grounds that every arm64 wheel targets macOS 14 or newer; the machine has
  since been upgraded past that, and `pyproj` is now a dependency. `cartopy`
  also installs now (0.26.0 ships a cp314 arm64 wheel) but is deliberately not
  a dependency: the maps draw on plain `Axes` over a vendored basemap. No R
  spatial package is installable.

Operational rules that follow from the data and are easy to get wrong in code:

- Never open one of PEcAn's initial condition source files with CF time
  decoding on: the `time` units are an unparseable template (README note 5).
  Use `initial_conditions.read_source_file`, or `decode_times=False` with
  `engine="scipy"`. Everything downstream reads the tracked converted file.
- Never build a timestamp yourself from a `.clim` or SIPNET-output row label.
  The time axis is pySIPNET's (`ClimateDrivers.xarray`, `SIPNETOutput`), and driver and
  model fields both carry it. The local ERA5 drivers' hour column drifts
  (README Note 15, issue #9), so pySIPNET refuses them until they are
  corrected; the tests read them through `tests/conftest.py`'s
  `regular_drivers_root`, which rewrites only that column.
- Never parse an AmeriFlux FLUXNET timestamp through a time zone, the
  session's default included (R's `tz = ""`, pandas `tz_localize`): the stamps
  are local standard time with no daylight saving, and a zone with DST drops or
  duplicates the half-hours at each transition. Read them as naive
  `YYYYMMDDHHMM` and shift by the tower's fixed `utc_offset_hours` from the
  tower table. Every NEE file from before the AmeriFlux ingest broke this rule,
  which is why none of them is used (`data/README.md`, Net ecosystem exchange).
- A tower's pool site comes from the tower table, never from a nearest-point
  search; the table records how each match was made.
- Never renumber the 1-8000 site ids; they are a shared key with collaborators.
- The site table is `data/raw/sites/pts.*` (tracked) and, after ingest,
  `data/processed/sites/sites.csv`. There is no other site source.
- Do not assume rectangular coverage: observed NEE is mostly missing over
  site x time, and every constraint is ragged over site x time.
- Constraints keep their **source units** and their source's own time
  labels. Nothing is converted or aligned at ingest; the observation operator
  does both. See the processed-data conventions below.
- Observed NEE keeps its source units too, with **one deliberate exception to
  the time labels**: the processed files are on UTC, each tower's local
  standard time shifted by its fixed `utc_offset_hours`. That is a relabeling
  of the clock, not an alignment: every value keeps its own half-hour or hour,
  and the offset each tower was shifted by is a coordinate of the file.
- The initial conditions' processed file is in the source files' units, negative
  wood and leaf draws included, and applies **no state-to-parameter mapping**:
  three of the four SIPNET initial parameters depend on calibrated parameters,
  so the mapping is applied per proposal, by the SIPNET parameter map's
  `initial_condition_rules`. Each spec's `pecan_conversion` says what
  PEcAn did.

## Code conventions

These are project-wide and apply to new code without being restated. They
follow `logs/2026-09-25_Repository Design Model.md` in the vault, the approved
design every change is measured against; a change that needs a convention not
written here adds it there and here first. The code is being brought into line
with it by a series of PRs: PR 1 (the foundation: shared constants, coercion,
file writing, site-table functions), PR 2 (batch dims), PR 3 (vocabulary
renames), PR 4 (contracts and vectors: the aliases and their validators, the
vector conventions), then module cleanups. Where the code does not follow a
rule below yet, the rule says which PR changes it.

### Glossary

One concept, one word; one word, one concept. A word this glossary does not
list is either added here or not used for a project concept. The retired words
are gone from the code, except where a rule below names the PR that retires
them.

**Space and sites.**

| Word | Meaning | Retires / not to be confused with |
|---|---|---|
| **site** | a location where the model is run and data are indexed: one row of the site table in use, whichever pool that is; the `site` dim and coordinate hold its site id | "location" (only for lon/lat prose); a **point** |
| **site id** | the integer (`int32`) label of a site, unique within its site table; `site_id` is the site table's column. Ids of a pool shared with collaborators are never renumbered | |
| **site ids** / `sites` | a sequence of site ids, **always** | `sites` for a DataFrame |
| **site table** / `site_table` | the pandas site table (`sites.load_sites`), keyed or indexed on `site_id`, with `lon`/`lat` | `sites=`, `sites_table=` for a table |
| **site pool** (prose) | the sites of the site table in use. Code working on a site table counts them from the table in hand (`n_sites`). `sites.N_SITES` is defined once, as the size of the pool the raw inputs define, and is read only by raw-data code that checks its inputs have that size: the ingest scripts, the raw-data specs (`site_labels`' `expected_rows`), the survey scripts and the split script | `POOL`, `POOL_SIZE`, literal 8000s |
| **site labels** | the site-labels data source: a class per site, such as PFT | "label" alone for it |
| **grid cell** | a cell of `sites.SITE_GRID` | |
| **point** | a location that is not a site, such as a spatial prediction target; the `point` dim carries integer labels with no meaning beyond the field, and `lon`/`lat` coordinates | a site (a point has no site id) |

**Time.**

| Word | Meaning | Retires |
|---|---|---|
| **timestep** (prose and identifiers) | one SIPNET step, `(timestep_start, time]`; pySIPNET's coordinates are `timestep_start`/`timestep_length` (`conventions.TIMESTEP_START`/`TIMESTEP_LENGTH`) | "time step", `time_step` |
| **time label** | a value of the `time` coordinate | |
| **window** | the interval an observation's value covers, which model timesteps are reduced over (`pd.IntervalIndex`); on an observation's `time` as the coordinates `conventions.WINDOW_START`/`WINDOW_END`, built from the CF `time_bounds` variable the processed file stores; their values are `window_start`/`window_end` | "time bounds" and `time_bounds_start`/`time_bounds_end` for these coordinates |
| **cell** | **only the CF sense**: the interval or area one value represents (a calendar resampling cell, a grid cell, a raster cell, a map renderer's site cell) | an element of y, a `(sample, site)` pair, a CSV field, a figure slot |
| **record** | a data source's full time extent (a run's record, a site's driver record); also one raw CSV row | |

**Data sources and data.**

| Word | Meaning | Retires |
|---|---|---|
| **data source** | a producer's dataset, raw or processed (MODIS LAI, the ERA5 drivers, PEcAn's initial conditions, the site table) | "product" for this |
| **observation source** | a data source used as observational constraint data; one `ObservationSource` | "product" in the observation package |
| **processed file** | the file an ingest script writes under `data/processed/` | "processed product" |
| **upstream product** | the producer's own product name (MCD15A3H v061); the spec field and netCDF attribute `upstream_product` | the spec field and attribute `product` (a processed file written before the rename carries it until re-ingested) |
| **raw** | a data source as it arrives, never edited | |
| **constraint** | one of the five constraint data sources of `constraints.py` | |
| **observation** | one entry of **y**: one observed value of one observation source at one site (and time) | "cell", "observed cell" |
| **model output** | SIPNET's output as a labeled `xr.Dataset`, one run or a stack | `dataset` as a variable name for it |
| **run** | one SIPNET execution: one site, one sample, and one member of each driving data source | "cell", "slot", "pair" |
| **run index** | the rows of an evaluation of the runs: its `batch_dim` and any crossed dims of the external inputs, `SIPNETRunsEvaluation.run_index` | |

**The calibration.** Two layers and one seam. The **probability layer**,
`sipnet_calibration.probability`, declares a model in specs, binds it to
labels and conditions it on data, and knows nothing of sites, SIPNET or the
rest of the package. The **adapter layer** is `SiteDims` (what each site
is), `SIPNETParameterMap` (how the values at a site become SIPNET parameters
there), and `SIPNETRuns` and `SIPNETSimulator` (SIPNET run for a batch of
values, and the forward map as a model's simulator). The seam between them
is the **labeled values**, a dict of `xr.DataArray`s, one per component.

| Word | Meaning | Retires / not to be confused with |
|---|---|---|
| **component** | a named array a draw of a model holds, declared by an `ArraySpec` (`probability.spec`): a parameter, a derived value, an observed value, a prediction | "component" for an element; not a **field** |
| **parameter** | a component of the target, what theta holds (`Posterior.parameter_names`): an array-valued quantity with a support, units, one value's shape, and the dims it is indexed by; always distinct from a **SIPNET parameter**, which is always called by its full name | "calibration parameter"; the `Parameter` class |
| **spec** | a declaration, holding no labels and no numbers; its class ends in `Spec` (`ArraySpec`, `FactorSpec`, `DeterministicSpec`, `ModelSpec`) | the distribution it becomes once bound |
| **dim** / **dim label** | a dim a component is indexed by (`ArraySpec.indexed_by`), and one of its labels, in the coords a model is bound at; in the adapter layer a dim is `site` or a site-labels name, whose labels are the site ids or the classes some site carries (`SiteDims.coords`) | "group", "copy", `varies_by`, `group_dim`, `dim=`, `dim_index` |
| **stacked dim** | a dim whose labels are a `pandas.MultiIndex` with named levels (integers, strings or `datetime64[ns]`), how a ragged set of labels, such as the `(site, time)` pairs a source observes, becomes one dim; a level may be named wherever a dim may, and merges with a plain dim of its name in a layout's index | a batch dim |
| **value** | a component's value at one tuple of labels of its dims, of shape `ArraySpec.shape`; `shape` is always one value's, never the block's | |
| **element** / **element axes** | one number of one value; a value's axes are its element axes, named, with their labels (strings) or a length (`ArraySpec.element_axes`), which name the element axes of the labeled forms | "natural" / "unconstrained size and names", `k` and `e`, `natural_names`, `element_labels`, "component" for an element |
| **block** | all of a component's values, one at each tuple of labels of its dims: **block shape** `(*index shape, *shape)`, the **index shape** `[len(coords[d]) for d in indexed_by]`, the labels in use; values by name put the batch in front, `(*batch, *block shape)` | "block" for one value |
| **event** | the components a factor declares (`FactorSpec.event`). TFP's sense, the axes one draw of a distribution covers, is always written "TFP event shape"; one value's law, which `iid_over_dim` repeats over a block, has the element axes as its TFP event shape; a support's or bijector's `event_ndims` is how many trailing element axes it constrains jointly (0 on an interval, 1 on the simplex) | "event" for element axes |
| **support** | a set of values, a `Support` (`Interval`, `Simplex`, `PositiveDefinite`, ...): the set a component's values lie in, whose default bijector is `bijector_for(support)`; a rule's domain is one too (`ValueRequirement.domain`) | "the open set", `Bounds`, `OpenInterval` |
| **constant** | a value a law, deterministic or SIPNET rule reads that is the same in every draw: an `xr.DataArray`, `float64` or `bool`, scalar or keyed by label on dims of the coords or element axes, read at the labels in use (`probability.labels` defines it) | |
| **own dim** | a dim of a constant that is neither a dim of the coords nor an element axis, passed whole, which its reader declares in `own_dims=` | |
| **label map** | a one-dimensional `xr.DataArray` on a dim of the coords, named for its target (a dim or an element axis), whose values are the target's labels (a site's PFT, a PFT's biome: `SiteDims.labels`); a function receives it as `int64` positions (`probability.labels`) | "membership", `memberships`, `site_positions`, `dim_label_positions` |
| **layout** | named arrays as one flat vector, a `Layout` (`probability.layout`): components in declaration order, each block in C order, with no `order` argument; theta's and y's | the parameter vector |
| **values by name** / `ValuesByName` | a layout's structured, traceable form: `{name: (*batch, *block shape)}`, the form every law, deterministic and rule function computes on | `ValuesByParameter`, `NaturalValues` |
| **labeled values** / `LabeledValues` | a layout's labeled form: a `dict` of one `xr.DataArray` per component on `(*batch dims, *indexed_by, *element axes)`, a dict because two stacked dims with a `site` level cannot share a Dataset; with the external inputs, what the SIPNET parameter map reads; `encode_labeled_values` makes the Dataset netCDF holds | `ParameterDataset`, "labeled natural values" |
| **part** / **factor** / **deterministic** | a part is a factor (a conditional law over its event, a `FactorSpec`) or a deterministic (components computed by a pure function, a `DeterministicSpec`, or by a simulator) | a factor: "prior term", `PriorTerm`; a deterministic: "derived parameter", `DerivedParameter`, `derived_from`, `compute`, "pointwise" |
| **law** | the concrete distribution a factor evaluates to for one draw of what it reads: a TFP distribution, or an object implementing `probability.laws.Law`, such as a `GaussianLaw`; an EnsKit `Gaussian` of one block or a numpyro distribution (GPJax's among them) is adapted to one by `as_law` (`GaussianLaw`, `NumpyroLaw`), and `pushforward` of a base that is not TFP's is a `PushforwardLaw` | a factor, which declares one |
| **law function** | a function returning a factor's law, `f(**given, **constants)`, its TFP event shape coming from what it reads; the builders `iid_over_dim`, `independent_over_dim` and `gaussian_copula` return ones the layer also passes the index shape, privately | "prior function", `PriorFunction`, `f(index_shape, ...)` for a user's function |
| **given** | the components and inputs a part's function reads, inferred from its keywords (the keyword rule), never stated | `given=` |
| **input** | a node with no parents and no law, declared by an `ArraySpec` in `joint(..., inputs=)` and bound by `bind(..., inputs=)` | an external input, the SIPNET adapter's word |
| **bind** | give a model spec the labels of its dims and its inputs' values, making a `FactoredDistribution` | |
| **target** / **barren** / **observed** | after `condition_on`: observed factors are conditioned on; barren ones are unobserved with no observed descendant, dropped; the rest are the target, whose components are theta's; an observed factor with no target ancestor is constant (`O_c`) | |
| **draw** | one joint value of every component; a batch of draws has batch dim `sample` | |
| **simulator** | a deterministic computed outside JAX for a whole batch of samples, which may fail at some of them: a `Simulator` (`probability.parts`), called once per batch with labeled values and returning a `SimulatorOutput` (its outputs, and at which samples each was computed); `SIPNETSimulator` is the forward map as one | the forward model's runs, `SIPNETRuns`; `ForwardModel` |
| **valid** (a sample) | every simulator output the likelihood reads was computed at it and every likelihood factor's density is finite there; an invalid sample's log likelihood is `-inf` (`PosteriorEvaluation.valid`) | `SIPNETRunsEvaluation.valid`, a row whose runs all succeeded in the domain |
| **observation dim** | a source's stacked dim, `"<source>_observation"`, whose labels are the `(site, time)` pairs it observes (`(site,)` if static), sorted by site, then time: `ObservationVector.coords`, `ObservationSource.observation_labels`, `observation_dim_name()` | a batch dim |
| **observed component** | a source's observed values as a component, named for the source, on its observation dim (`observation.model.observed_components`, values `observed_values_by_component()`) | |
| **prediction** | the forward model's value of an observed quantity, on the source's observation dim and in its units: `predicted_<source>` (`ObservationVector.prediction_name`, `observation.model.prediction_components`) | "predictions", the Flat `(J, N)`, which keeps its meaning |
| **Gaussian factor** | a factor whose law is a `GaussianSpec` (`probability.parts`): its one component, on `REAL` and indexed by one dim at most, centered on a mean component, with a covariance declared by a **covariance spec** (`probability.covariance`); its law at a draw is a `GaussianLaw`, holding one of EnsKit's operators | a law function returning a Gaussian, which is opaque |
| **covariance spec** / **scope** | a declaration of a covariance as structure over labels (`DiagonalSpec`, `DenseSpec`, `SumSpec`, `ScaledSpec`, `BlockDiagonalSpec`, `SubmatrixSpec`); its scope is the entries it covers, the event or one group of a `BlockDiagonalSpec`, whose groups are the entries sharing labels at a level (`by=`) | "cell" for a block |
| **noise factor** | a source's Gaussian factor, its observed component centered on its prediction (`observation.model.noise_factor`), holding the source's constants its covariance reads | |
| **held** (values) | what a posterior fixes whatever theta is: the observed values, the inputs, the constants, and the deterministics computed from them alone with no simulator; a Gaussian factor's covariance computable from held values is a **held covariance**, built and factored once at `condition_on` | |
| **Gaussian likelihood** | the likelihood written as `y ~ N(G(theta), R)` when every `O_theta` factor is Gaussian with a covariance the held values fix (`Posterior.gaussian_likelihood() -> GaussianLikelihood`): `R` block-diagonal over the factors in y's order, `G` their means (`forward`) | |
| **conjugate rule** / **marginal** | the scale rule or the block rule (`probability.conjugacy`, the design's §7.13 R1 and R2): a component whose inverse-gamma or inverse-Wishart prior is conjugate to the one Gaussian factor reading it; integrated out (`FactoredDistribution.marginalize`), that factor becomes its marginal, a Student-t or a matrix Student-t (`probability.scale_mixtures`) | |
| **full conditional** | a parameter's closed-form law given every other component, by a conjugate rule (`Posterior.full_conditional -> FullConditional`), drawn from an evaluation's residuals with no new simulator run | |
| **site dims** | the sites and the dims they define, a `SiteDims`: each site's id, location, site covariates and site labels | the parameter vector's `site_table`, `site_labels`, `sites` |
| **site covariate** | a `float64` column of the site table, named in `SiteDims(covariate_names=)`, read as a constant (`SiteDims.covariate`) | `site_covariate_names` |
| **external input** | an uncertain value a SIPNET rule reads that is propagated, not calibrated, paired with the samples by dim name (`sipnet_parameter_map.ExternalInputs`), given to `SIPNETRuns.evaluate`; a model's own fixed values are its inputs | the `to_sipnet_parameter_fields` hook |
| **role** | what a SIPNET parameter written depends on (`calibration.ROLES`): `calibrated` (a parameter, directly or through deterministics), `propagated` (other values only: inputs, external inputs), `constant` (a rule of constants and fixed values), or `fixed` | |

**Representations.**

| Word | Meaning | Retires |
|---|---|---|
| **field** | one `xr.DataArray` holding one variable under the field contract below | "canonical field", a Dataset called a field |
| **Fields** (a representation) | an observation vector's fields, a `dict[str, Field]` by source: what `predict` returns and `to_fields` makes from labeled values | Fields for labeled values, which are not fields; `SiteDims.site_fields()` is their per-site view |
| **Flat** | a layout's unlabeled numeric form: one vector, theta `(D,)` or y `(N,)`, or a batch `(n_samples, D)`/`(n_samples, N)`, called "batched Flat" where the shape matters | "block" for it |
| **entry** | one position of a Flat vector | "column" (of theta), "cell" (of y) |
| **segment** | the contiguous labels of one site in an observation dim, or the contiguous entries of one component in Flat | "block" in `forward.py` |
| **SIPNET parameter fields** / `sipnet_parameter_fields` | the `xr.Dataset` of SIPNET parameter values, `fields.SIPNETParameterFields` (its form is in `fields.py`'s data model) | "SIPNET table", `table` for a Dataset |
| **SIPNET overrides** / `sipnet_overrides` | one run's flat `dict[str, float]`, the keywords `SIPNETModel` takes | `sipnet_parameters` for this |
| **SIPNET parameters** / `sipnet_parameters` | **only** a pySIPNET `SIPNETParameters` | the operator keyword of that name |
| **table** | a pandas DataFrame, only | an `xr.Dataset` called a table |

**Ensembles and batches.**

| Word | Meaning | Retires |
|---|---|---|
| **batch dim** | any dim of a field other than its spatial dim and `time` whose index coordinate holds integers: an axis of independent replicates | "member dim", "sample dim" |
| **sample** | the default name of the batch dim made from the rows of batched Flat; one row of theta is one sample (`conventions.SAMPLE`) | `member` for theta rows |
| **J** (mathematics only) | the number of samples; in code `n_samples`, or `batch_size` for a batch dim of any name | `J`, `n_members` as identifiers |
| **member** | one member of a data source's own ensemble, only inside a name that says which source: `initial_condition_member`, `driver_member` | bare `member` as a dim name |
| **batch shape** | TFP's term, only in TFP code, always written "TFP batch shape" | "batch member" for a dim label |

**Sampling.** The samples of tempered SMC and importance sampling are
samples, as above; "particle" is not used.

| Word | Meaning | Retires |
|---|---|---|
| **base density** | the normalized density `q` a tempered SMC run or an importance sampler draws from, `smc.BaseDensity`: the prior, or a Student-t fitted to an ensemble | "proposal" for it: a proposal is a move's |
| **stage** | one tempering increment of SMC, with its resampling and moves (`SMCState.stage`) | "iteration" |
| **weight** | a sample's normalized importance weight `W`; `log_weights` in code | |

**Narrowed words.** *label*: an xarray coordinate label, text on a figure, or
"site labels" the data source. *kind*: only pySIPNET's variable kind. *source*:
only "data source" and its attributes (`source_file`, `source_column`).
*identity*: not used. *row*: a table row, or a row of batched Flat (a sample);
never a timestep or a site position. *grid*: `SITE_GRID`, a raster grid, or a
PyEns `Grid` (always "PyEns grid").

**Notation.** `J` samples, `D` the dimension of theta
(`posterior.dimension`), `N` the dimension of y
(`posterior.gaussian_likelihood().y`), `S` sites (`n_sites`), `K`
observation sources; `theta` Flat unconstrained parameters, `y` Flat
observations, `G` the forward map (`GaussianLikelihood.forward`, through a
`SIPNETSimulator`), `M_s` SIPNET at site s, `H_k` the observation operator of
source k, `T` the unconstrained-to-natural transform. "Predictions" is
always a Flat `(N,)`/`(J, N)`; a source's labeled form is its prediction,
`predicted_<source>`, and its fields `ObservationVector.to_fields`'.

### Field contract

A **field** is one `xr.DataArray` holding one variable, checked by
`fields.validate_field`; it is a convention, not a wrapper class. The
contract, the batch dims and the forms that cross a module boundary are
defined once, in `fields.py`'s module docstring: read it before writing code
that makes or reads a field. The rules most easily broken:

- **A structural axis is never a dim** of a field (variable, component,
  quantile, bounds, PFT class): split it into a `dict` or `Dataset` of
  fields.
- **One name means one index.** Two batch dims with the same name are the
  same index (they zip in PyEns and align in xarray); different names are
  different indices (they cross), so unrelated ensembles are kept apart by
  distinct names, and a deliberate pairing is spelled by giving two dims one
  name.
- **Flat has at most one batch dim.** A field with several is reduced, or
  stacked with `fields.stack_batch_dims(field, new_batch_dim="run")` under a
  new name, since the stacked dim is a new index.
- **A data source's member dim is named for its source**
  (`conventions.DATA_SOURCE_MEMBER_NAMES`), never bare `member`.

**The aliases.** One per form that crosses a module boundary, each exactly
one type with one validator beside it (PEP 695 `type` statements); each
form is stated in its home module's data model:

| Alias | Home | Validator |
|---|---|---|
| `Field` | `fields` | `validate_field` |
| `ModelOutput` | `fields` | `validate_model_output` |
| `ObservedValues` | `observation.source` | `validate_observed_values` |
| `SIPNETParameterFields` | `fields` | `validate_sipnet_parameter_fields` |
| `SIPNETOverrides` | `fields` | `validate_sipnet_overrides` |
| `ValuesByName` | `probability.layout` | `validate_values_by_name` |
| `LabeledValues` | `probability.layout` | `validate_labeled_values` |
| `ExternalInputs` | `sipnet_parameter_map` | `validate_external_inputs` |

The validators of the field forms are strict: each requires everything
`validate_field` does, `lon`/`lat` included; `validate_sipnet_overrides`
checks a mapping of pySIPNET flat parameter names to numbers.
`SIPNETParameterFields` and `SIPNETOverrides` live in `fields` beside
`ModelOutput`, the model's input beside its output, so `initial_conditions`
and `observation` never import `probability` (and with it TFP).

### Vector-like classes

`ObservationVector`, `Layout` and their pieces follow one convention. An
observation vector's pieces are its `ObservationSource`s; a layout's are its
components' `ArraySpec`s (`layout[name]`, `in`, `iter`, `len`). A model's
parts, its binding and its conditioning are the probability layer's
(`ModelSpec`, `FactoredDistribution`, `Posterior`), and the SIPNET parameter
map is a separate object (`sipnet_parameter_map.py`) a `SIPNETRuns` holds.

| Aspect | Convention |
|---|---|
| Construction | `@dataclass(frozen=True, eq=False, kw_only=True)`; validation in `__post_init__` through one grouped check (`check_observation_vector_is_valid`, `check_observation_source_is_valid`); nothing mutable reachable: mappings frozen (`frozendict`, which pickles and hashes), arrays copied and read-only (`conventions.ReadOnlyCopies` for xarray data) |
| Pieces | `vector[name]`, `name in vector` (`False` for anything else, an unhashable value included), `iter(vector)` and `reversed(vector)` (piece names), `len(vector)` (number of pieces), `<piece>_names` |
| Size | `size` on a layout, theta having `D = posterior.parameters.unconstrained.size` entries; an observation vector has none, y's order and size being its posterior's |
| Entries | a layout's `index`: a `pd.MultiIndex` over the entries (`(component, *dims, *levels, element)`), and `positions(**selectors) -> int64 array`, an unknown label a `KeyError` as in `select`; an observation vector's observations are its `coords`, one observation dim per source |
| Sites | the observation vector's `sites` (ids, ascending; each source's observations are sorted by site and time as a normalization, since their order carries nothing) and `site_table` (from its observed values' `lon`/`lat`, which its sources must agree on); a model's sites are its `SiteDims`' (`sites`, ascending, refused if unsorted), with their locations, site covariates and site labels |
| Selection | `select(...)`: an unknown label raises `KeyError`; the result keeps the declared order whatever the request order, so Flat order never changes by selection; duplicates are refused. A layout's is `select(component=[...], <dim or level>=[...])`, and `FactoredDistribution.select` binds the model again at fewer labels; the observation vector's is `select(*, observation_source_names=None, sites=None, time=None)`, and `restrict_to_sites(sites)` the intersecting form |
| Representations | a layout: Flat, values by name and labeled values, converted by `<source>_to_<target>` (`flat_to_values`, `values_to_labeled`, ...), with `to_natural`/`to_unconstrained` between the spaces; the observation vector: `observed_values_by_component()` and `constants()` on the observation dims, `to_fields(labeled) -> Fields` back |
| Flat's array type | JAX everywhere: the layout, the posterior and the inference adapters return `jax.Array` Flat and accept any array-like; internals that fill arrays in place work in NumPy and convert on return. 64-bit JAX is on for the whole package |
| Description | `describe()`: one row per piece; `__repr__` one summary line. `calibration.describe_calibration(posterior, sipnet_parameter_map)` joins the descriptions into the record written beside a run: one table per component and input, and one per SIPNET parameter, with its role |
| Directions | where a map, a rule or an operator lists SIPNET parameter names, the name says which way: `sipnet_parameter_names_written`, `sipnet_parameter_names_read` |
| Section comments | `# ── identity ──`, `# ── selection ──`, `# ── representations ──` or `# ── coordinates ──`, `# ── evaluation ──` |

`SIPNETRuns` is a regular class with read-only properties (its arguments, so
the run machinery cannot go stale; its climate a `frozendict`);
`SIPNETRunsEvaluation` is `frozen, eq=False` with JAX `predictions` (one
read-only mapping per observation vector, by prediction name), `in_domain`
and `valid`. A data source's ensemble enters `SIPNETRuns.evaluate` as
external inputs, crossed with the samples by PyEns when its dim is not
`batch_dim`, and the reduction over them is the experiment's;
`SIPNETSimulator` takes none, a model's own fixed values being its inputs.
No base class is shared by the
vectors: they share an interface, not an implementation, and their shared
coercion lives in `validation.py`.

### Where shared things live

- **`conventions.py`** holds every name constant two modules share (dims, the
  data source member dims `INITIAL_CONDITION_MEMBER` and `DRIVER_MEMBER`
  (`DATA_SOURCE_MEMBER_NAMES`), coordinates, `SOURCE_INDEX`, the `site_id`
  column, the `time_bounds` variable and its `BOUNDS` dim,
  `NON_BATCH_DIM_NAMES`, `SIPNET_ROW_LABEL_NAMES`, `RESERVED_NAMES` (what
  no component, element axis, external input, site-labels name or site
  covariate may be named), the attributes of
  `site`/`lon`/`lat`/`sample` and of a data source's member dim
  (`DATA_SOURCE_MEMBER_ATTRIBUTES`), `SITE_DTYPE`, `BATCH_LABEL_DTYPE`,
  `NAME_PATTERN`, `STALE_TIME_ATTRIBUTE_NAMES`, `CF_CONVENTIONS`,
  `DATA_ROOT_ENV_VAR`, `data_root()`, `tracked_data_root()`); and
  `read_only_copy` and `ReadOnlyCopies`, the read-only copies of xarray data a
  frozen class keeps and hands out, copied on assignment so nothing a caller
  holds is frozen (`ReadOnlyCopies(default=None)` for an optional
  one). Read-only mappings are `frozendict`s (the `frozendict`
  package): a `dict` subclass, so pandas and `json` read one as a dict, and it
  pickles and hashes. Every module-level mapping constant of the package is
  one (the scripts' own tables are not the package's), and one is handed to
  xarray as it is, since xarray copies attrs; pandas' `agg`, which refills the
  mapping it is given, takes a `dict(...)` copy. A module imports these; it
  never defines its own copy and never re-exports one. A name only one module
  uses lives in that module: `RAW_MEMBER` in `initial_conditions.names`.
- **`validation.py`** holds the argument coercion two modules need, each
  `as_<thing>(value, *, message_name) -> thing`: `as_site_ids`, `as_site_id`,
  `as_integer`, `as_positive_integer`, `as_bounded_integer`,
  `as_positive_integers`, `as_batch_label`, `as_batched_flat` (with
  `is_one_vector`),
  `as_bbox`, `as_sequence`, `as_names`, `as_frozen_mapping`; the `check_*`
  functions they are written with; and `truncated(items)` for messages and
  `range_summary(values)` for reports. One rule for every argument of a kind:
  - **a sequence argument** (site ids, names, source indices, site-label
    classes) is a sequence, always: one bare id or one string is a
    `TypeError` naming the fix ("pass [27]"), a `set` and a mapping are
    refused (no order; a mapping's keys are passed as `m.keys()`), order is
    kept, and `dict.keys()` and NumPy, JAX, xarray and pandas arrays are
    accepted. Site-id arguments go through `as_site_ids` (duplicates
    refused), names arguments through `as_names`, and a sequence of items of
    any type through `as_sequence`;
  - **an integer** (a site id, a source index, a count) refuses a boolean, a
    missing value and a float, even an integral one ("cast it with int()");
  - **a site the data lacks** is a `KeyError` (the vectors' `select`
    included; the observation vector's `restrict_to_sites` is the
    intersecting form).
  `as_batched_flat(values, dimension, *, message_name)` returns the 2-D
  batch alone, `float64`, JAX when given JAX; a caller that must know a
  one-vector input was given asks `is_one_vector(values)` (not
  `np.ndim`, which a list of JAX tracers refuses under `jax.jit`), and a
  caller's further rules (at least one row, finite) are its own checks.
- **`io.py`** holds writing a file safely (`write_checked`, and
  `write_checked_together` for files that belong together), `file_md5` and
  `utc_timestamp`.
- **`sites.py`** holds everything that reads the site table: `load_sites`,
  `select_sites`, `site_lookup`, `site_locations`, `site_coordinates` (the
  `site`, `lon` and `lat` coordinates a processed file's `site` dim carries),
  the site-table checks (one invariant each, grouped as
  `check_site_table_locates_the_sites` and
  `check_site_table_is_keyed_on_site_ids`), the pool checks a raw data source's
  sites are held to (`check_site_table_lists_the_sites`,
  `check_sites_are_the_site_table`), and `N_SITES`. No lookup is written as a
  hand `set_index("site_id")`; `site_lookup` is the keyed form.
- **`probability/`**, the probability layer, imports nothing of the package
  outside itself; `tests/test_package.py` enforces it, and that
  EnsKit is imported by `probability/_linalg.py` alone, the one shim over
  its operators and `Gaussian`, and numpyro and GPJax by no file: the layer
  adapts their distributions recognized by class name
  (`probability/_numpyro.py`), and neither is a dependency (numpyro is in
  the `dev` group, GPJax in the optional `gpjax` group). The
  probability layer keeps its own private coercion helpers and its own
  reserved names (`probability.names`, whose `SAMPLE` a test holds equal to
  `conventions.SAMPLE`), and the project's reserved names, the site table
  and SIPNET are the adapter layer's (`site_dims.py`,
  `sipnet_parameter_map.py`, `forward.py`).
- **`inference/`**, the inference adapters, imports `probability`, `smc` and
  `validation` only, and no algorithm package (tested): of EnsKit, only
  `inference/eki.py` imports anything, its `Ensemble`. It reads a
  `Posterior` and nothing of SIPNET, so the experiment imports EnsKit's
  driver or emcee and hands it what an adapter returns.
- **`tests/conftest.py`** holds every fixture or builder more than one test
  file uses (some Niwot stacks and observation builders are still per file,
  until the module cleanups, PR 5).
- The package's `__init__.py` documents every module and the direction the
  dependencies run, re-exports nothing, and turns on 64-bit JAX, its one
  import-time side effect.

### Naming in processed data

Raw variable names are not ours to choose; processed ones are.

- **`lower_case_with_underscores`** for every variable, coordinate and column of
  a processed file.
- **Avoid abbreviations** unless they are universal. So `soil_organic_carbon`,
  not `soc`; `aboveground_biomass`, not `agb`; `standard_deviation`, not `sd`.
  `lai` is fine, and so are `lon`/`lat`, which the field convention
  fixes.
- The correspondence from source to processed belongs in **one explicit spec**
  in the library beside the schema, not spread across a script. See
  `ConstraintSpec` in `sipnet_calibration.constraints`, whose `raw_file`,
  `value_column` and `sd_column` name the source and whose fields are written
  into the processed file as `source_file` and `source_column`, so the
  correspondence is never guesswork.
- A constraint is named for its **raw file's stem** (`modis_leaf_area_index`,
  not `lai`): the name is then the spec's key, the raw file and the output file
  at once, it keeps two data sources of one quantity apart, and it asserts
  nothing the source does not (biomass, not carbon).
- Renaming is safe only where a record carries its own identity. Where the
  source pairs values *positionally*, the positional read stays in source names
  and the rename happens after the data is self-describing.
- **The `VARIABLES` registry is keyed on processed names**, so a field's
  `name` is a processed name. `validate_field()` reads neither the name nor
  the registry.

### Naming in code

The rules that keep a name from having to be looked up. The first four are
the ones most often broken.

- **A name and the thing are named differently.** A string or tuple of
  strings is `<thing>_name` / `<thing>_names`; the things themselves are the
  plural noun. `parameter_names` and `sipnet_parameter_names` are names;
  `sipnet_parameters` are values. `variable` and `parameter` never mean a
  string. A module-level tuple of names is `*_NAMES` (`TIME_COORD_NAMES`,
  `SPATIAL_DIM_NAMES`).
- **A variable holding one representation of a concept says which**, in the
  glossary's words. A posterior's unconstrained Flat is `theta` and its
  natural Flat `natural_flat`, its values by name `values_by_name`, its
  labeled form `labeled_values`; SIPNET parameter fields are
  `sipnet_parameter_fields`; the observations' Flat is `y`. A pySIPNET object
  is named for its class (`sipnet_result`, `sipnet_output`,
  `sipnet_parameters`); the labeled xarray of a run's output is
  `model_output`. Nobody should have to ask whether a value is a layout, an
  `ArraySpec` or a SIPNET parameter.
- **`sipnet_` prefixes anything in pySIPNET's vocabulary**: names, values,
  objects. The bare word or `calibration_` is this repository's vocabulary.
  `pysipnet_` is not used: this repository reaches SIPNET only through
  pySIPNET, so a second prefix would have no second referent.
- **An `xr.Dataset` when the variables share one grid, a
  `dict[str, DataArray]` when they do not.** One run's or one stack's model
  output shares a time axis and is a Dataset; the constraint data sources have
  three time structures and are a dict. A dict is named for what it holds and
  its key (`observed_values_by_source`, keyed by observation source name;
  `run_outputs_by_sample_site` where the key order matters).
- **A field is one thing**, as the field contract above defines it. The word
  "canonical" is not used with it; `fields.py` holds the generic operations on
  fields and nothing else.
- **A class whose call is its whole interface may be named as an imperative
  verb**, for what the call does (`SelectTimestep`, `ComputeLeafAreaIndex`).
  This is not a rule for every callable class: a protocol, a record, or an
  object with an interface of its own beyond the call is a noun
  (`ObservationOperator`, `SIPNETRuns`, a model object).
- **Private helpers are named for what they do or what they return**: a verb
  phrase (`_sort_by_site_and_time`, `_drop_padding_rows`) or a noun phrase
  (`_read_only_float64_array`, `_observation_restricted_to`). Never a bare
  participle (`_selected`, `_frozen`, `_aggregated`) and never a name that
  hides a side effect (a `_with_...` that drops, a `_sort_...` that
  partitions). The
  retired names still in the code are renamed by the module cleanups (PR 5).
- **A function that validates and converts is `as_<thing>`**; a check never
  returns a value (below).
- **Constants are documented with `#:` comments** above them, and **every
  module has `__all__`** listing its public API (the package's own is empty;
  the docstring-only `plotting/registry.py` and `plotting/diagnostics.py`
  stubs are the exceptions, until PR 5e).
- **No abbreviations** beyond the universal ones, as above:
  `constraint_standard_deviations`, not `constraint_sds`. Names that are
  pandas', xarray's or pySIPNET's own (`how`, `freq`, `coords`, `dims`) stay,
  because matching `pysipnet.resample(how=)` or `DataArray.coords` is worth
  more than spelling them out; so does `DEFAULT_OBS_OPS`, which the author
  chose.
- **Units are not in names** (`radius_km`, `interval_ms` are retired):
  quantities are SI, meters and seconds, with the unit in the docstring.
- **An argument is named for what it is for**, never for where it sits: not
  `at`, not `data`, not `x`. A site table is `site_table=` everywhere, and
  `sites` is always a sequence of site ids.

Existing code is brought into line by the PRs of the design model, with the
tests renamed alongside; `logs/2026-09-25_Repository Design Model.md` in the
vault lists what changes in which.

### File organization

- **Public first, private last.** Public functions, classes and constants at the
  top of a file; helpers and anything underscore-prefixed below them. The one
  exception is a constant whose construction runs module functions, such as a
  spec registry, whose specs run their checks when built: it follows them,
  last in the file, under a one-line comment saying why.
- Data processing scripts follow the section order
  `entry point` -> `the steps, in the order main calls them` ->
  `supporting types and helpers` -> `checks`, with `# ── ... ──` section
  comments. `scripts/ingest_sites.py` and `scripts/ingest_constraints.py` are
  the worked examples.
- Keep functions short enough that the top-level one reads as a summary of the
  work. If it stops reading that way, pull a step out as a helper. Roughly 40
  lines is where to start looking for the seam, not a hard limit.

### Checks, errors and messages

These apply to the library and the scripts alike.

- **A check is a public-named `check_<subject>_<predicate>` function** with a
  one-line docstring, saying what it checks: `check_covariances_were_diagonal`,
  `check_site_table_lists_each_site_once`. Not `validate`, not an inline
  `assert` buried in a transformation. It checks one invariant, and raises or
  returns `None`; it never returns a value. A group of checks always called
  together becomes one check that calls them in order, whose one-line
  docstring says what the group checks; its body is the list of checks, so
  the docstring does not repeat it (`check_site_table_locates_the_sites`).
  Checks used from another module are in `__all__`.
- The `check_*` functions live together in the **`# ── checks ──` section at
  the bottom** of the file.
- **Error types**: `TypeError` for a wrong type, including a boolean where a
  number is wanted and one string where a sequence is; `ValueError` for a
  wrong value; `KeyError` for a name or label that is not there (an unknown
  parameter, source, site or site-table row); `FileNotFoundError` for a
  missing file. The coercers of `validation.py` raise by this rule already;
  plotting and projection, which still raise `ValueError` for some wrong
  types, are brought into line by the module cleanups.
- **Messages** start lowercase, name the invariant that broke, then `;` and
  what to do about it; name the subject through a `message_name` argument,
  passed last; and truncate a list to ten items with one helper,
  `validation.truncated(items)`. `validation.py` follows this throughout;
  elsewhere the older messages are brought into line by the module cleanups
  (PR 5).
- No inline `assert` in library code, and no inline `raise` in a function
  that also has `check_*` calls (`validation.py`, `load_drivers` and the
  functions PR 1 rewrote follow this; the rest is PR 5's). A script's `main`
  turns its checks' errors into a reported error rather than a traceback.

### What does and does not belong in documentation

- **Do not write volatile measurements into documentation.** Row counts, element
  counts, file sizes, per-variable coverage, "929 of them are zero" — these
  describe one snapshot of the data and go stale silently. A numeric property
  the code depends on is **checked programmatically**: an assertion in
  the ingest script, a constant in the library, or a test. Documentation says
  what the property *is* and where it is checked, not what it currently
  measures. Where a run's numbers are genuinely useful, print them.
  `data/README.md` is the exception, since recording measured characteristics of
  the raw data is its job — but even there, anything the code relies on is
  asserted in code as well, not just written down.
- **Code does not cite the vault.** Docstrings, comments and provenance strings
  never point at the Obsidian vault, a design log or the readiness report: those
  live outside the repository, so a reader of the code cannot follow the
  reference, and they move. A citation in code names a primary source -- a
  paper, a line of the SIPNET source, a data producer or upstream product (BETY,
  ISCN), a pySIPNET module. Design reasoning goes the other way: the vault cites
  the code.
- **Verbosity must earn its keep.** A docstring is clear and precise, and no
  longer: it says what a thing is, takes, returns and raises (the exception
  types and when, in a sentence each), and does not restate what the body
  plainly does. A validator does not list the checks it runs; it says what it
  checks, points to where that is defined, and keeps a short Raises. A
  definition with one home, such as a constant, a list of names or a
  contract, is named and pointed to, never copied: the field contract lives
  in `fields.py`'s docstring, the reserved names in `conventions`, an alias's
  form beside the alias.
- **Keep low-level design reasoning out of docstrings.** A docstring says what
  something is, what it takes and what it returns. Why a design was chosen over
  an alternative, what bug it avoids, what would break if it were done the other
  way — that belongs in a **Notes section at the end**, if it belongs in the
  docstring at all. Otherwise put it in an implementation comment beside the
  code it explains, or in the design log in the vault. A top-level docstring is
  read by someone trying to use the thing, not to review its design.

### Docstrings for modules that define a data model

A module that owns how some data is represented should answer four questions,
because these are what someone opens it to find out:

1. **Where it sits in the pipeline** — which scripts produce the data it reads,
   and which way the dependency runs.
2. **What it reads** — the inputs, named, with what each is for.
3. **The data model** — for an xarray data source, the dims, the data variables
   and their dtypes, the coordinates and which dims they are on, the attributes,
   and what missing means. State it plainly; do not make the reader infer it
   from the validation code. State it once, in its home module; another
   module that uses the form names it and points there, and states only what
   it adds (as `sipnet_parameter_map` does for SIPNET parameter fields, whose
   form `fields` owns).
4. **The functions it provides** — the public entry points and what each one
   does with that model.

Then Notes for the design reasoning, then Usage for how to call the public
functions. `sipnet_calibration.constraints` is the worked example.

### Docstrings for data processing scripts

File-level docstrings use these sections, in this order:

1. **Overview** — a couple of sentences on what the script does.
2. **Input data** — the assumed format of what it reads. Clear and precise, but
   not every detail.
3. **Output data** — the same for what it writes.
4. **Notes** — anything else that matters: traps, why a step exists, what a
   choice depends on. Omit if there is nothing to say.
5. **Usage** — the command lines.

Function and module docstrings elsewhere are ordinary NumPy style.

### Processed data conventions

- **Processed files follow the Climate and Forecast conventions, CF-1.11**, as
  pySIPNET's model output does since its PR #38, so model and observation read
  alike: `Conventions = "CF-1.11"` on the dataset; `standard_name`, `axis` and,
  where present, `bounds` on `time`; `standard_name` and `units` on `lon`/`lat`;
  no `_FillValue` on any coordinate. Where a value's support is documented it is
  a CF `time_bounds(time, bounds)` coordinate, as pySIPNET writes one whose two
  edges are `timestep_start` and `time`. Where CF has no vocabulary for what a
  label means, the meaning goes in words (`time_reference`, `comment`), never in
  a `cell_methods` that is not literally true.
- **Ingest changes structure, never values.** No unit conversion, no temporal
  alignment, no choice of which record stands for a year. Units are the raw
  file's; the observation operator converts through Pint
  (`pysipnet.units.unit_registry`) and decides the alignment.
- **One flat spec per variable, from which everything is derived**, in the shape
  of pySIPNET's `VariableSpec`: the spec's `xarray_attributes()` is what the
  processed file stores, so a netCDF describes itself and there is no separate
  processed schema to keep in step. `ConstraintSpec` is the worked example.
- **A unit that is inferred is recorded as inferred**, in a `units_provenance`
  sentence on the spec and the processed file, not as a status enum.

### Processed files and their readers

- **Schema constants and the reader live in the library**, not the script, so
  the writer and the reader of a processed file cannot drift apart
  (`SITE_COLUMNS` in `sites.py`, `CONSTRAINTS` in `constraints.py`).
  A script's own round-trip check calls the library loader, never a parallel
  reader.
- **Write through `io.write_checked(path, write, check)`**: the file goes to a
  `.partial` path beside its destination and is renamed only after the checks
  pass, so a failed run cannot leave a corrupt file at the canonical path.
  On a failure the `.partial` file is **kept for inspection and its path
  printed**; it cannot be mistaken for the processed file, and a rerun
  overwrites it (a stale `.partial` is removed before a run writes). A write
  must write the path it is given and nothing else, which is checked before
  any check runs; a directory made for the destination is removed if a
  failure leaves it empty. Every
  script that writes a processed file, a tracked raw input or a generated
  definition does this, with no protocol of its own; files that belong
  together go through `io.write_checked_together`, which refuses one
  destination given twice and moves none of them unless all were written and
  checked. A survey script's `--out` report is
  not such a file.

## Writing conventions

- **American English spelling throughout**: `center`, not `centre`; `color`,
  `behavior`, `labeled`, `modeling`, `meter`, `organize`, `recognize`,
  `normalize`, `summarize`. This applies to prose, code comments, docstrings,
  identifiers, commit messages, and issue and pull-request text alike.
- Wrap prose in Markdown and docstrings at roughly 80 columns, matching the
  surrounding file.

## Working alongside other sessions

Several sessions often work in this repository at once, on separate branches
and separate pull requests. Unless each has its own checkout they share one
working tree, one index and one `HEAD`, and a number of ordinary git commands
then do something other than what they appear to do.

**Work in your own worktree, created from an explicit start point.** The root
checkout is nobody's workspace; leave it on `main` and clean.
`.claude/worktrees/` is already ignored.

```bash
git worktree add -b feat/<topic> .claude/worktrees/<topic> origin/main
```

- **Always name the start point.** `git checkout -b <name>`, and
  `git worktree add` with its trailing `<commit-ish>` omitted, base the new
  branch on whatever is currently checked out: `<commit-ish>` defaults to
  `HEAD`. A branch created while another session's work is checked out is
  rooted on that session's commit, which yields a pull request carrying
  someone else's commit that cannot merge until theirs does. Neither command
  needs a clean tree, so nothing warns you. Naming `origin/main`, or whatever
  the base really is, is the whole fix.
- **Stage explicit paths.** `git add -A` and `git add .` stage every dirty
  file in the tree, including the ones another session is still editing.
  `git add <path> <path>` cannot.
- **Do not switch branches in a checkout you do not own.** `git switch` and
  `git checkout <branch>` move `HEAD` for every session using that tree.
  Read-only commands are always safe: `status`, `log`, `diff`, `show`,
  `reflog`, `worktree list`.
- **Read `git status --short` before every commit**, and confirm that every
  file it lists is yours.

Naming the start point and staging explicit paths are not alternatives, and
neither is "commit before switching branches". Committing first prevents
neither wrong base, since the branch point is chosen when the branch is
created, whatever the state of the tree; and explicit staging never touches
the branch point. One habit protects the index, the other the base.

Two checks catch a wrong base or a stray file after the fact. They compare
against `origin/main` rather than local `main`, which in a fresh worktree is
only as current as the last fetch:

```bash
git rev-list --count origin/main..HEAD     # more commits than you made?
git diff --name-only origin/main...HEAD    # files you did not touch?
```

Destructive commands discard work that may belong to another session:
`git checkout -- <path>`, `git restore`, `git reset --hard`, `git clean`,
`git stash`. Ask before running one in a shared checkout, and name specific
paths rather than a whole tree. Never run one on another session's behalf:
permission belongs to the session whose work it affects, and routing a denied
command through a peer is not a way to get it approved.

`git stash` needs its own warning, because a worktree gives no protection from
it. The stack is a single repository-wide `refs/stash` shared by the root and
every worktree, and `git stash pop` takes the top of it whoever pushed it.
Prefer a temporary commit for setting work aside. If you must stash, label it
with `git stash push -u -m "<label>"`, note its SHA from
`git stash list --format='%H %gs'`, restore with `git stash apply <sha>`
rather than `pop`, and drop that entry afterwards.

A wrong base is usually repaired by rebasing and force-pushing. Force-push
only your own branch, with `--force-with-lease`, and check first whether
anyone has based a branch on yours: `git branch --contains <old-tip>`. If one
has, tell that session before you push — rewriting a branch moves the base of
everything stacked on it, and they will have to rebase too.

**A worktree isolates git, but it starts with no Python environment.** Give it
its own, taking the companions' current `main` as above while you are there:

```bash
uv lock --upgrade-package pysipnet --upgrade-package pyens --upgrade-package enskit
uv sync
uv run pytest
```

The companion packages come from git rather than from sibling paths, so
`uv sync` needs nothing next to the worktree, and the `.venv` it creates has
`sipnet_calibration` installed editable against **that worktree's** `src/`.

Do not reach for the root's interpreter instead. Its `sipnet_calibration` is
editable against the **root's** `src/`, so it imports whatever branch the root
checkout is on rather than your own. That failure is loud only when a module
exists on your branch alone — `ModuleNotFoundError` for something you are
looking at in your editor. For a module that exists on both, the tests pass
while exercising the root's copy, which is the case worth remembering.

## The probability-layer refactor

`sipnet_calibration.parameters`, the parameter layer, was replaced, PR by
PR, by a generic probability layer, `sipnet_calibration.probability`, from
which the prior and the observation model are both built; R1 removed the
parameter layer, today's `ForwardModel` and the observation vector's Flat
API, and is the plan's last PR. Three documents govern it:

- `docs/probability-refactor/design.html`, the design (open it in a browser).
  Its §12 is the PR plan and its §13 the decisions. Its §6 vocabulary is in
  the glossary above since R1.
- `docs/probability-refactor/HANDOFF.md`, the state of the refactor: which PRs
  are merged, open or next, what each session learned, the deviations from the
  design, and the questions waiting for Andrew. It is the source of truth for
  what to do next.
- This section, the workflow.

**A session started to continue the refactor** (its prompt says so) follows
these steps without asking. It stops only where a step says to.

1. **Find the next PR.** `git fetch -q origin`, then read `HANDOFF.md` on
   `origin/main` (`git show origin/main:docs/probability-refactor/HANDOFF.md`).
   Statuses lag, since a PR's own row is written before it is merged: check
   every `open #n` row with `gh pr view <n> --json state`, and treat a merged
   one as merged (this session's PR updates its row). The next PR is the first
   row whose status is neither merged nor open. If every row is merged, the
   refactor is complete: clean up as step 3 says, tell Andrew, and stop.
2. **Check that what it needs is merged.** For each PR in its "needs" column,
   `gh pr view <number> --json state,mergedAt`. If any is not `MERGED`, **stop**,
   and tell Andrew which PR is waiting and on what. Do nothing else.
3. **Clean up stale worktrees, and only those.** A worktree is stale when its
   branch is this refactor's (`refactor/probability-*` or
   `docs/probability-refactor-*`), its PR is merged or closed, and
   `git -C <path> status --porcelain` prints nothing. For each one,
   `git worktree remove <path>`, then `git branch -D <branch>`. Never touch any
   other worktree or branch: they belong to other sessions. Report what was
   removed.
4. **Implement in a new worktree.**
   - Create it:
     `git worktree add -b refactor/probability-<id>-<topic> .claude/worktrees/probability-<id> origin/main`.
   - Set up its environment:
     `uv lock --upgrade-package pysipnet --upgrade-package pyens --upgrade-package enskit`,
     then `uv sync`, then `uv run pytest` for the baseline count. EnsKit
     is tracked like the other companions since E1 re-pinned the project to
     its `main`.
   - Implement the row's scope from the design, nothing more. Write reference
     values from today's code before changing behavior that must be preserved.
   - When implementation shows the design must change, make the smallest
     change. Update `design.html` in the same PR, and record the change in
     `HANDOFF.md`.
   - Anything out of scope goes in `HANDOFF.md`'s open items, not in the PR.
5. **Review once.** Run the `review-pr` skill on the branch, for one round:
   - the Standard tier;
   - the Deep tier only for a PR whose diff computes densities, likelihoods or
     inference results (P3, P6, P8);
   - Solo for P0, P1 and R1's deletions.

   Fix what it finds. Judgment calls go in the report, not in more rounds.
6. **Update `HANDOFF.md` in the PR.** Set the row's status to
   `open #<number>`, and add a session entry: what was done, the deviations from
   the design, the decisions taken, the open questions for Andrew, and what the
   next session must know. Update CLAUDE.md's layout, alias and glossary entries
   for what the PR adds.
7. **Open the PR, never merge it.**
   - Read `git status --short` and stage explicit paths.
   - Commit, push the branch, then `gh pr create --base main`. The description
     gives the scope, the deviations, the test counts before and after, and the
     review's outcome.
   - Only Andrew merges.
8. **Report and hand off.**
   - Tell Andrew:
     - the PR's link;
     - the important findings;
     - any change to the design;
     - the open questions.
   - Then call the `spawn_task` tool (the session chip in the Claude desktop
     app), so that one click starts the next session:
     - title: "Probability refactor: next PR";
     - prompt: "Continue the probability-layer refactor. Follow 'The
       probability-layer refactor' in CLAUDE.md.";
     - tldr: the PR just opened, and which PR the next session will take.
   - The next session waits, at step 2, for this PR to be merged. When the PR
     just opened is the plan's last row, there is no next PR, and no chip.

The rules elsewhere in this file still hold. In particular:
- stage explicit paths, and never switch the root checkout's branch;
- the companion packages are read-only.

## Running on the SCC

Two caches have to be moved off the home directory, which is small and
quota-limited: `uv`'s own, and the one a SIPNET binary is installed into.
A `uv` git clone is usually what fails first, with `Disk quota exceeded`, so
set these before anything else rather than as a tuning step:

```bash
export UV_CACHE_DIR=/projectnb/dietzelab/<user>/uv_cache
export PYSIPNET_CACHE_DIR=/projectnb/dietzelab/<user>/pysipnet_cache
export TMPDIR=/projectnb/dietzelab/<user>/tmp
```

`pysipnet install-sipnet` compiles there rather than downloading, because the
published Linux SIPNET binary is built against a newer glibc than the SCC
provides; `pysipnet info` reports the reason it refused the prebuilt, and
`gcc`, `make` and `git` on a login node are all the compile needs. The binary
then lives under `$PYSIPNET_CACHE_DIR`, and any environment exporting that
variable finds it. A `qsub` script has to export all three itself, and
`#$ -P dietzelab` with `#$ -l buyin` is the queue. For PyEns jobs,
`compute.scc_backend` writes those directives and a `-v` exporting the
cache, binary and data variables; its module docstring says why `TMPDIR` is
left to Grid Engine.

## Repository layout

The layout below is the **agreed target**, specified in
`logs/2026-08-28_Plotting Design Spec.md` in the Obsidian vault. The src-layout
reorg has landed, so the paths below are the real ones; `sites.py`,
`constraints.py`, `initial_conditions/`, `drivers.py`, `projection.py`, the
`probability/` package, `site_dims.py`, `sipnet_parameter_map.py`,
`calibration.py`, `site_labels.py`, `forward.py`, `compute.py`, `smc.py`,
the `inference/` package and the `observation/` package are implemented, `fields.py` has the model-output
adapters, the plotting package has series, maps and grids, and the other
modules carry the contract each is to satisfy.
`initial_conditions` is a package rather than a module: it spans several
artifacts, and giving each its own file keeps that artifact's schema, writer,
reader and checks together. `probability` is a package for another reason:
it is a layer, which imports nothing of the rest.

```
pyproject.toml            # name = "sipnet-calibration"; src layout
src/sipnet_calibration/
  __init__.py             # the module map and the dependency direction; no
                          # re-exports; turns on 64-bit JAX
  sites.py                # SITE_GRID + grid conversions and cell_of(),
                          # load_sites(), select_sites(site_ids=, bbox=, where=,
                          # n_random=, seed=), EXTENTS (named lon/lat boxes),
                          # N_SITES; site_lookup(), site_locations(),
                          # site_coordinates(), the site-table and pool checks;
                          # default_site_table_path()
  projection.py           # SITE_PROJECTION (LAEA 50 N, 100 W) over pyproj:
                          # forward(), projected_bounds(), factors()
  projections/            # the stored definition, generated from the dataclass
  constraints.py          # ConstraintSpec + CONSTRAINTS, one per raw file;
                          # read_raw(), build_constraint(), load_constraint(),
                          # constraint_fields() -> one field per constraint
  conventions.py          # every shared name constant: SITE, TIME, SAMPLE,
                          # INITIAL_CONDITION_MEMBER, DRIVER_MEMBER,
                          # the reserved spatial names, TIMESTEP_START/LENGTH,
                          # WINDOW_START/END, TIME_BOUNDS, SITE_ID, SOURCE_INDEX,
                          # RESERVED_NAMES; the attributes of site/lon/lat/sample and a source
                          # member; SITE_DTYPE, BATCH_LABEL_DTYPE, NAME_PATTERN;
                          # CF_CONVENTIONS, data_root() and
                          # tracked_data_root(); read_only_copy(),
                          # ReadOnlyCopies
  validation.py           # argument coercion: as_site_ids, as_site_id,
                          # as_integer, as_positive_integer, as_bounded_integer,
                          # as_positive_integers, as_batch_label,
                          # as_batched_flat, as_bbox,
                          # as_names, as_frozen_mapping; its checks; truncated(),
                          # range_summary()
  io.py                   # write_checked() and write_checked_together() (the
                          # .partial protocol), file_md5(), utc_timestamp()
  initial_conditions/     # one module per artifact; __init__ re-exports them all
    __init__.py           # curated exports + the processed file's data model
    names.py              # RAW_MEMBER, the two file
                          # names, the path helpers
    source_files.py       # SOURCE (the PEcAn file format), read_source_file()
    specs.py              # InitialConditionSpec + INITIAL_CONDITIONS
    raw.py                # build_raw(), raw_encoding(), read_raw()
    processed.py          # build_initial_conditions(), load_initial_conditions(),
                          # netcdf_encoding(), initial_condition_fields()
    sipnet_parameters.py  # to_sipnet_initial_conditions() and, for an ensemble,
                          # to_sipnet_initial_condition_fields()
  net_ecosystem_exchange/ # observed NEE, laid out as initial_conditions/ is
    __init__.py           # curated exports + the processed files' data model
    names.py              # the raw files' dims, the raw (local standard time)
                          # and processed (UTC) axes per resolution, the raw
                          # and tracked files' paths
    source_files.py       # SOURCE (the AmeriFlux FULLSET CSV), read_source_file()
    raw.py                # build_raw(), raw_encoding(), read_raw(): every tower on
                          # one local-standard-time axis per resolution
    towers.py             # the tower table: exact tower-to-site matching, one
                          # primary tower per site and resolution,
                          # recover_utc_offset() from SW_IN_POT,
                          # shortwave_lag_steps(); read_tower_table()
    specs.py              # NetEcosystemExchangeSpec + NET_ECOSYSTEM_EXCHANGE, one
                          # per series (ameriflux_nee_<resolution>_ustar_<variable|constant>)
    sources.py            # SOURCE_READERS: raw file -> (tower, time) UTC series
    processed.py          # net_ecosystem_exchange_path(),
                          # build_net_ecosystem_exchange(),
                          # load_net_ecosystem_exchange(),
                          # net_ecosystem_exchange_fields() and the companions
  drivers.py              # load_drivers(): raw .clim files read by pySIPNET's
                          # ClimateDrivers, stacked into (driver_member, site, time) on
                          # pySIPNET's axis; no processed file exists
  site_labels.py          # SiteLabelsSpec + SITE_LABELS, one per raw file; a
                          # site-labels data source is site_id -> class, one
                          # processed file per source; read_raw(),
                          # build_site_labels(),
                          # load_site_labels(), site_labels_field() -> CF flags
  probability/            # the probability layer: imports nothing of the
                          # package outside itself
                          # (tested); __init__ re-exports it
    support.py            # Support (Interval, Simplex, PositiveDefinite:
                          # contains, closure), REAL, POSITIVE, NON_NEGATIVE,
                          # OPEN_UNIT_INTERVAL, UNIT_INTERVAL, SIMPLEX,
                          # POSITIVE_DEFINITE; DEFAULT_BIJECTORS, bijector_for
    names.py              # SAMPLE, COMPONENT_LEVEL, ELEMENT_LEVEL, THETA,
                          # THETA_ENTRY, RESERVED_NAMES
    labels.py             # coords (stacked dims), constants (own dims), label
                          # maps (into dims or element axes): as_coords,
                          # as_constants, as_label_maps, aligned_constants,
                          # aligned_label_maps; indexer() by hashing
    spec.py               # ArraySpec: name, units, support, indexed_by,
                          # element_axes (labels or a length), bijector;
                          # shape, unconstrained()
    layout.py             # Layout: index (component, *dims, *levels,
                          # element), select()/positions() by dim or level,
                          # Flat, values by name and labeled values converted
                          # by <source>_to_<target>, unconstrained,
                          # to_natural(), to_unconstrained(), contains();
                          # ValuesByName, LabeledValues;
                          # encode_labeled_values/decode_labeled_values
    laws.py               # Law (the protocol), as_law (TFP; an EnsKit
                          # Gaussian, a numpyro distribution, GPJax's too),
                          # pushforward (any base); GaussianLaw (a Gaussian
                          # over a block holding a structured covariance),
                          # NumpyroLaw, PushforwardLaw
    covariance.py         # CovarianceSpec: DiagonalSpec, DenseSpec, SumSpec,
                          # ScaledSpec, BlockDiagonalSpec (by= a level or the
                          # dim; groups contiguous), SubmatrixSpec; each bound
                          # to its scope at bind, building an EnsKit operator
                          # per draw
    families.py           # one value's law: log_normal, logit_normal
                          # (support=), their _from_* forms, softmax_normal;
                          # normal, inverse_gamma, InverseWishart/inverse_wishart
    builders.py           # Builder (.law, .reads); iid_over_dim,
                          # independent_over_dim, gaussian_copula
    scale_mixtures.py     # StudentTSpec (a Student-t per group) and
                          # MatrixStudentTSpec (an inverse Wishart block
                          # integrated out), a factor's law as GaussianSpec
                          # is; StudentTLaw, MatrixStudentTLaw;
                          # inverse_wishart_log_prob, sample_inverse_wishart
    parts.py              # FactorSpec (a conditional law over its event),
                          # GaussianSpec (mean=, covariance=: a factor's
                          # law), CENTERED_LAW_SPECS (it and the Student-t
                          # forms), DeterministicSpec; @factor, @deterministic; given
                          # read off the function's keywords (the keyword rule);
                          # Simulator (name, given, outputs, __call__, at,
                          # check_given) and SimulatorOutput
    model.py              # joint -> ModelSpec (the graph, topological order);
                          # bind -> FactoredDistribution: law(), select(),
                          # marginalize(), sample() keyed by crc32(name), log_prob(),
                          # simulators, describe(); block_at_labels; a
                          # simulator runs once per batch, never at bind
    posterior.py          # condition_on -> Posterior: target, barren and
                          # O_c from the graph, each simulator pruned to the
                          # outputs the likelihood reads and checked at the
                          # corner points; sample_prior, log_prior, evaluate
                          # -> PosteriorEvaluation, log_likelihood,
                          # log_density, log_density_given, predict,
                          # replicate, simulator_inputs, natural_values,
                          # theta_with, to_labeled, gaussian_likelihood ->
                          # GaussianLikelihood (y, noise_covariance R,
                          # forward), full_conditional -> FullConditional
                          # (law, sample from an evaluation), describe; a
                          # Gaussian factor's covariance the held values fix
                          # is built and factored once here
    conjugacy.py          # the conjugate rules, the scale rule (an inverse
                          # gamma on a covariance scale) and the block rule
                          # (an inverse Wishart on a
                          # covariance block): conjugate_rule ->
                          # ConjugateRule, marginalize,
                          # InverseWishartGivenRows
    _bound.py, _keywords.py, _probes.py, _validation.py, _linalg.py,
    _numpyro.py           # private: a part at the labels in use (its law,
                          # density, draws, the bind checks, the log-Jacobian
                          # against each support's reference measure); the
                          # keyword rule; the probe and corner points;
                          # coercion; the one shim over EnsKit's operators
                          # and Gaussian; numpyro's distributions recognized
                          # by class name, with no import
  site_dims.py            # SiteDims: the sites and the dims they define; coords,
                          # labels() (label maps), covariate(), at_sites(),
                          # site_fields(), select()
  sipnet_parameter_map.py # SIPNETParameterMap: how the values at a site become
                          # SIPNET parameters, from labeled values (the
                          # probability layer's dict, or a Dataset) and a
                          # SiteDims. Rules (Copy, Copy.same_names, CopySimplex,
                          # Compute; photosynthesis_rules,
                          # initial_condition_rules) reading
                          # values by name with a ValueRequirement (units, a
                          # Support domain, shape; omitted, FROM_SIPNET_SPEC:
                          # the written SIPNET parameter's) and constants; Fixed;
                          # dependencies(), sipnet_parameter_names_depending_on();
                          # ExternalInputs; sipnet_parameter_fields(),
                          # out_of_domain(); support_from_sipnet_domain; the fit
                          # check, against ArraySpecs
  calibration.py          # describe_calibration(posterior, map) (two tables:
                          # per component and input, per SIPNET parameter with
                          # its role), example_calibration() -> (factors, map)
  forward.py              # SIPNETRuns: labeled values -> SIPNET once per run
                          # through PyEns, the observation operators applied on
                          # the worker, predictions per observation vector and
                          # model output from one pass (SIPNETRunsEvaluation,
                          # with its run index and predictions per source);
                          # SIPNETSimulator, the forward map as a Simulator;
                          # the failure split; the corner check
  compute.py              # scc_backend(): the SCC GridEngineBackend preset
  smc.py                  # tempered SMC from a base density q to the
                          # posterior, importance sampling its one-step case;
                          # knows nothing of SIPNET. MultivariateStudentT,
                          # DefensiveMixture, fit_student_t; TemperingProblem,
                          # SMCSettings, SMCState; initial_state(), run_smc(),
                          # save_state()/load_state(); next_increment() (CESS),
                          # pareto_k() (ArviZ's PSIS), systematic_resample()
  inference/              # a Posterior as each algorithm reads it; imports
                          # probability, smc and validation only (tested), and
                          # no algorithm package; __init__ re-exports it
    eki.py                # EKIProblem (forward, y, noise_covariance,
                          # last_evaluation, initial_ensemble -> an EnsKit
                          # Ensemble of one block, theta), eki_problem(): what
                          # enskit.algorithms.eki's driver reads
    tempering.py          # PriorBaseDensity, tempering_problem() -> an
                          # smc.TemperingProblem, predictions as auxiliary
    mcmc.py               # batched_log_density(), log_density(),
                          # initial_points()
    _validation.py        # private: the checks the three share
  fields.py               # the field contract: validate_field(), batch_dims(),
                          # stack_batch_dims()/unstack_batch_dims(),
                          # batch_coordinate(), scalar_batch_labels(),
                          # is_categorical(); the Field, ModelOutput and
                          # SIPNETParameterFields aliases and their validators;
                          # to_model_output() (a SIPNETResult, a SIPNETOutput or
                          # its Dataset with site/lon/lat and batch labels: the
                          # model output the observation operators read),
                          # stack_model_outputs() (runs to one on
                          # (*batch, site, time)), in_field_layout(),
                          # resolve_output_variable_names(), message_name(),
                          # window_coordinates()
  observation/            # the observation side of the inverse problem
    __init__.py           # curated exports
    time_alignment.py     # aggregate_time, reduce_windows, select_timestep_at,
                          # windows_from_observed_values, run_window, the counts;
                          # the verb a caller applies before plotting
    operators.py          # ObservationOperator protocol; SelectTimestep,
                          # ReduceOverWindows, ReduceOverRun,
                          # ComputeLeafAreaIndex; DEFAULT_OBS_OPS;
                          # check_operator and the contract's checks the
                          # vector shares
    source.py             # ObservedValues + validate_observed_values();
                          # ObservationSource, one source's fields, operator
                          # and optional standard_deviation;
                          # observation_labels
    vector.py             # ObservationVector: select(), restrict_to_sites(),
                          # predict(); for the probability layer, coords (one
                          # observation dim per source),
                          # observation_dim_name(), constants(),
                          # prediction_name(), year_label_map(),
                          # observed_values_by_component(),
                          # with_observed_values(), to_fields()
    model.py              # observed_components(), prediction_components(),
                          # noise_factor() (one source's Gaussian factor):
                          # imports the probability layer, so __init__ does
                          # not import it
  plotting/
    __init__.py           # curated exports
    style.py              # ROLES, rcParams
    registry.py           # VARIABLES
    primitives.py         # L1: (ax, plain numpy, **style) -> artist
    series.py             # L2 time series panels
    maps.py               # L2 plot_map (points/cells/triangles renderers,
                          # rasters, classes), summarize_batch, animate_map
    basemap.py            # coastlines/borders/graticule on projected Axes
    basemap_data/         # the built Natural Earth basemap, tracked
    facet.py              # L3 build_plot_grid + by_site/by_variable and the
                          # map grids: plot_map_grid/_by/_quantiles
    diagnostics.py        # L5 EKI history, marginals, coverage
scripts/                  # ingest: data/raw/ -> data/processed/; and
                          # build_basemap.py: raw/natural_earth -> basemap_data
  survey_*.py             # NOT the pipeline: answer a question about raw data
                          # and write nothing under data/. The phenology and
                          # soil texture ones also assert what data/README.md
                          # records and exit non-zero when it no longer holds.
  raw_sources/            # NOT the pipeline: code that *makes* a raw input,
                          # run once. SCC-only except the Natural Earth download
                          # and the NEE tower table, which needs only the raw
                          # NEE files.
experiments/<task>/       # config.py (source of truth) + plots.py (L4 reports)
data/raw/                 # never edited; raw/sites/, raw/constraints/,
                          # raw/initial_conditions/, raw/site_labels/,
                          # raw/covariates/ and raw/natural_earth/ are tracked,
                          # and raw/net_ecosystem_exchange/'s tower table,
                          # tower list and provenance (not its time series)
data/processed/           # ingest output == the plotting input; untracked;
                          # constraints/<name>.nc is one CF-1.11 netCDF per
                          # constraint; net_ecosystem_exchange/<name>.nc one per
                          # NEE series
tests/                    # conftest.py: every fixture or builder two files use;
                          # storage-backed data found through
                          # conventions.data_root(), tracked inputs through
                          # conftest.REPOSITORY
```

Conventions:

- One directory per experiment under `experiments/`, with `config.py` as the
  single source of truth for that experiment (parameters, transforms, data
  paths, algorithm settings).
- Heavy computation lives in scripts, not notebooks. Notebooks are for
  exploration and plotting only, and load results from disk.
- Raw inputs are symlinked into `data/raw/` and never edited; ingest scripts
  convert them to `data/processed/`, whose format **is** the canonical format
  used throughout the project. The drivers are the one exception: nothing is
  written under `processed/` for them, and `drivers.load_drivers` produces the
  canonical form from `data/raw/drivers/` on demand; `data/README.md` says why.

### Plotting and field conventions

Read `logs/2026-08-28_Plotting Design Spec.md` in the vault before writing
plotting code. The load-bearing rules:

- **Field**: as the field contract under "Code conventions" defines it. One
  `DataArray` per variable; facet-by-variable takes `dict[str, DataArray]`.
- Every plotter calls `fields.validate_field` first and branches on **presence
  of a batch dim**, found with `fields.batch_dims`, never on a mode keyword.
  `plot_time_series` fans over batch dims of any name and refuses a `site` dim
  (select a site, or facet with `plot_by_site`); `plot_map` refuses a batch dim
  with advice naming it (`plot_map_by`, `plot_map_quantiles(batch_dim=)`,
  `summarize_batch(field, stat, batch_dim=)`); `plot_map_by` and
  `plot_map_quantiles` refuse a batch dim besides the one their panels are over
  with the same advice, and the map grids and `animate_map` check every panel
  is a map (`maps.check_field_is_a_map`) before a shared scale reads its
  values. No plotter types a dim name: `SITE`, `TIME`, `LON`, `LAT` and
  `SAMPLE` come from `conventions`.
- **Temporal aggregation lives in `observation/time_alignment.py`**: the
  observation operators are written with it, and it is the verb a caller
  applies before plotting, so a predictive-check figure cannot disagree with
  what the likelihood consumed. The plotting layer imports nothing from it:
  `plot_time_series(aggregate_time(f, "1D"))`, never a plotter keyword.
- **The variable's kind says which resampling methods are valid; the caller
  may name one.** pySIPNET owns the first half: since its PR #38 every
  variable has a `kind`, `RESAMPLING_METHODS_FOR_KIND` says what may be done
  with it, and `pysipnet.resample.resample(ds, freq, how=...)` requires `how`,
  weights means by step length and refuses a method the kind does not support
  (a pool is not additive; a per-step total is not averaged until it is a
  rate). `observation.time_alignment.aggregate_time(field, freq, how=None)`
  is that operation for a field, through `resample` itself: a field carrying
  pySIPNET's interval coordinates is combined by its steps, keeping batch and
  `site` dims (pySIPNET PR #49), and one without them, such as observed
  values, on calendar cells alone (PR #51), and the module applies
  pySIPNET's public rules (`drop_padding`, `check_frequency`,
  `check_resampling_method`, `resampled_attributes`, `variable_kind`)
  wherever their inputs allow. With no `how` it takes **the method that
  leaves the variable the kind it already is**, read off pySIPNET's
  `RESAMPLED_KIND` rather than written down: a total sums, a step mean or a
  rate means, a pool or a running total takes its last value. Kept
  deliberately beside pySIPNET's: `last` is `NaN` in a cell holding a gap
  anywhere, where pySIPNET's reads only the cell's last value; a field with
  neither a kind nor interval coordinates is combined as told through a
  stand-in kind, keeping its own attributes and its `time`'s; the window
  mean's equal-spacing check, with pySIPNET's `STEP_TOLERANCE`; and the
  padding and empty-record checks, worded for every reader rather than for
  `resample`.
  SIPNET's `net_ecosystem_exchange` is `g m-2` of C per timestep, so 3-hourly
  to daily is a **sum**, and a mean is wrong by 8x while looking plausible;
  the default is there so that omission cannot reach that error, and `how=` is
  for asking deliberately for something else, such as the time-weighted mean
  of a pool. An invalid pair is refused in pySIPNET's own words.
- **An observation operator is a callable checked at the boundary, not a
  grammar.** `observation.ObservationOperator` is a protocol, and
  `observation/operators.py`'s module docstring is its contract: the call,
  what it declares, the checks at the boundary, the verbs an operator is
  written with and the default binding. Which operator reads an observation
  source is a modeling decision an experiment writes in `config.py`.
- **The forward model is runs and a simulator over existing pieces.**
  `forward.SIPNETRuns(sipnet_model, sipnet_parameter_map=, site_dims=,
  climate=, backend=, out_of_domain=)` runs SIPNET once per sample and site
  for labeled values and returns, from one pass, predictions per observation
  vector and model output (`evaluate`); `SIPNETSimulator(runs,
  observation_vector=)` is the forward map as the probability layer's
  `Simulator`, which never sees theta, runs only the sites its vector
  observes, and marks a source's prediction invalid only where a run at one
  of the source's sites failed. The module docstring says how the pieces
  compose: the labeled values (the layers' seam) merged with the external
  inputs, read at the site dims' sites by the map. The rules a session can
  get wrong: the observation operators run **on the worker**, each run
  receiving only its site's slice of the observation vector and returning
  that slice's predictions per source, which the calling process writes in
  the site's segment of each source's observation dim (right because an
  observation dim is sorted by site, which the plan checks per site and
  source); a run at a site no observation source observes returns nothing,
  so an observation source costs nothing at the sites it does not observe.
  External inputs pair with the samples by dim name: one on `batch_dim` zips
  with the samples (labels `0` to `J - 1`), one on any other batch dim is
  crossed, and PyEns enumerates the runs; each run's output is placed by its
  coordinate on the run index `(batch_dim, *crossed dims)`, so the row order
  is the repository's, not PyEns's. Values outside their domains (a rule
  input outside its requirement's, a SIPNET parameter outside pySIPNET's)
  raise before anything runs, and `SIPNETSimulator.check_given` checks the
  map at the corners of the target when a posterior is built;
  `out_of_domain="fail_row"` marks their rows out of the domain instead, a
  truncation of the prior that the model's factors do not know. A run that
  fails at its parameters (`SIPNETRunError`, pydantic's `ValidationError`, a
  timeout, or a non-finite value in a read variable,
  `ModelOutputNotFiniteError`; across a process boundary matched on PyEns's
  fully qualified `RemoteError.type_name`) makes its predictions NaN, the
  rest of its row kept, and anything else a worker returns is the machinery
  failing and is raised with the collected runs on the error's `evaluation`,
  as is a batch asking for model output in which every run failed. The model
  output is stacked by `fields.stack_model_outputs`, so it carries no
  `time_bounds` or SIPNET row labels; `freq=` aggregates each run's variables
  one at a time with `observation.aggregate_time` by the method that keeps
  its kind, as a predictive-check figure does, the Dataset gaining
  pySIPNET's `resampling_frequency` and `timestep_length_source`. Under any
  backend but `SequentialBackend` the drivers must be file-backed.
  `compute.scc_backend` is the SCC preset.
- **The observation vector holds no order of y.** Each source's
  observations are the labels of its observation dim (`coords`), sorted by
  site and then time, and the noise factors
  (`observation.model.noise_factor`) read the vector's observation dims and
  `constants()`; the posterior conditioned on `observed_values_by_component()`
  owns y's order, in which `gaussian_likelihood().y` and its
  `noise_covariance` are. A source may carry its measurement standard
  deviations (`standard_deviation=`).
  A batch dim on an observation source's values is
  refused: the experiment reduces an ensemble of observed values before it
  enters; a scalar batch label is metadata and is kept. An `ObservationSource`
  keeps only the sites and time labels it observes, so its operator never reads
  the model elsewhere, and a `select(sites=...)` slice's operators read the
  model only inside the kept sites' records.
- **An annual constraint's field carries its windows**, read from the
  processed file's CF `time_bounds`, as the 1-D coordinates
  `window_start`/`window_end` on `time` (`constraint_fields` adds them,
  through `fields.window_coordinates`; `conventions.WINDOW_START` and
  `WINDOW_END`), which is what `ReduceOverWindows` reads; a dated or static
  constraint documents no interval. An observed NEE field
  (`net_ecosystem_exchange_fields`) carries each step's window the same way:
  its values are mean rates over `(window_start, window_end]`, with `time` the
  window's end, and it carries no `timestep_*` coordinate, a timestep being
  SIPNET's.
- **Model and driver fields carry pySIPNET's names, units, kinds and time axis
  unchanged.** `fields.to_model_output` adds `site`, batch labels and
  `lon`/`lat` to a run's output; `drivers.driver_fields` does the same for the
  drivers, read through `ClimateDrivers`. The registry names are already
  `lower_case_with_underscores`, so they are the processed names. Both keep
  `conventions.TIME_COORD_NAMES`, pySIPNET's axis: `time` at the step end, with
  `timestep_start` beside it, so the interval a value covers is
  `(timestep_start, time]`, whose two edges are the pair pySIPNET writes as
  its CF `time_bounds` variable, and a run's output and the drivers it ran on share one axis by
  construction. `time_bounds` itself cannot ride on a field, its `bounds`
  dimension being no field dimension, so it and the `time` attribute naming it
  are dropped, along with SIPNET's `year`/`day_of_year`/`hour_of_day` row
  labels, which `timestep_start` already is.
- **Model and observed NEE are not in the same units.** Observed NEE is
  `umol CO2 m-2 s-1` (a rate); SIPNET's is `g C m-2` per timestep (a total).
  Observation sources keep their source units; the observation operator
  converts the model into the observed values' units, through Pint, before a
  residual or an overlay is formed. Plotting the two on one axis without
  converting fails silently, by orders of magnitude.
- **L1 primitives** take `(ax, plain numpy, **style)` and return artists: no
  pandas, no xarray, no figure creation. **No plotter** calls `plt.show()` or
  `savefig`, creates a figure implicitly, or accepts a `SIPNETResult` or a path
  (that is an adapter's job).
- **Anything that knows an experiment/task name belongs in
  `experiments/<task>/plots.py`, not the library.**
- Style comes from the `VARIABLES` registry and `ROLES` palette, not per-call
  keywords. `center=0.0` for signed fluxes such as NEE is correctness, not
  cosmetics.
- **Maps never interpolate unless asked.** Site values go through a
  `SiteRenderer`: `Points` (the default) and `Cells` (nearest site within a
  fixed radius, blank beyond it) draw only sites' own values, so a sparse set
  looks sparse; `Triangles` interpolates and is opt-in, continuous only. The
  radius is fixed, never scaled to site spacing, which would be interpolation
  by another name. `plot_map` refuses a batch or `time` dim rather than
  reducing it: use `summarize_batch`, `plot_map_by`, `plot_map_quantiles` or
  `animate_map`. A GP is not fitted in plotting; its predictions are a
  `(lat, lon)` raster or site values, mapped like any field. A field is
  categorical by `fields.is_categorical`, the field contract's one rule, and is
  colored by class position so a class keeps its
  color across figures; an optional `flag_display_names`
  tuple (ours, not CF's) is what the legend shows, and `site_labels_field`
  sets it from the spec's `display_names`. Sites are 8000 **irregular points**
  spanning 7-82 deg N, so a real projection is required and CONUS-only
  assumptions are wrong.
- **The display projection is settled**: a Lambert Azimuthal Equal Area
  centered at 50 N, 100 W on WGS 84, held as `SITE_PROJECTION` in
  `sipnet_calibration.projection`, which provides `forward()`,
  `projected_bounds()` for axes limits, `factors()` for local distortion, and
  the PROJJSON and PROJ string that PROJ serializes from it under
  `src/sipnet_calibration/projections/`. Plotting code projects through
  that module and never defines projection parameters of its own. Named
  lon/lat extents (`CONUS`, `NORTH_AMERICA`, `ALASKA`) are `EXTENTS` in
  `sipnet_calibration.sites`, beside the selection that takes the same form.
  ESRI:102003, which the published reanalysis figures used, was rejected on
  measured distortion. This projection holds angular deformation under 14
  degrees and anisotropy under 1.3 over the whole pool, both asserted against
  the real site table in `tests/test_projection.py`; `data/README.md` and issue
  #4 carry the comparison.
- **PROJ does the projection arithmetic**, through `pyproj`. What
  `projection.py` owns is the project's choice of projection, the serialized
  definition, and the shape the rest of the code consumes it in — not any
  formula. `Projection.factors()` exposes PROJ's own distortion measures,
  which is how a caller converts the long-edge mask threshold between a
  projected length and a ground distance, and how it learns that projected
  north rotates by about 150 degrees across the domain. Maps are plain `Axes`
  in projected meters, with limits from `projected_bounds`, never cartopy's
  `set_extent`, which gives a badly wrong frame on this projection. The
  basemap is Natural Earth 1:50m, clipped to 100 degrees of arc around the
  center (`Projection.angular_distance`) so nothing nears the antipode.
- `site` is the integer site id of the site table in use (`int32`; the shared
  pool's ids are never renumbered); `ameriflux_site_id` is a non-dimension coord
  on `site`. PFT is **not** site metadata and is not a column of the site table:
  which site labels to use is an experimental choice, so site labels are their
  own data source at `data/processed/site_labels/<name>.csv`, keyed on
  `site_id`, and a caller joins one on before selecting. A batch label is an
  integer, created `int64` (any integer dtype is accepted), meaningful only
  within its dim's name. See the Data section above for the rules these imply.

## Key API facts (hard-won from source reading)

### pySIPNET
- `SIPNETModel(runner, base_params=..., base_climate=...)(**overrides) -> SIPNETResult`
- `ClimateStaging` is in `pysipnet.runner`, not `pysipnet.climate`
- `SIPNETRunner(climate_staging=ClimateStaging.SYMLINK)` — staging goes on the runner, not the model
- Parameter override keys are flat snake_case leaf names
  (`max_photosynthesis_rate`, not `photosynthesis.max_photosynthesis_rate`)
- `SIPNETOutput` selects with `out["nee"]` (a `DataArray`) and `out[["nee", "gpp"]]`
  (a `Dataset`); aliases resolve. `result.nee()` and `to_xarray()` are gone (pySIPNET PR #36).
- The output `Dataset` is CF-1.11: `time` is the **end** of each step, `timestep_start` and
  `timestep_length` are coordinates, and `time_bounds = [timestep_start, time]` (PR #38;
  PR #52 renamed them from `time_step_*`, keeping no aliases, as it did
  `SIPNETOutput.timestep_length` and the `timestep_length_source` attribute).
  `pysipnet.resample.resample(ds, freq, how=...)` requires `how`.
- `pysipnet.variables.OUTPUT_VARIABLES` / `CLIMATE_VARIABLES` own the names, UDUNITS `units`,
  `constituent` and `kind` of every column. `pysipnet.units.validate_units` refuses a substance
  token inside a unit string: `"g C m-2"` is wrong, `"g m-2"` + `constituent="C"` is right.
- **`pysipnet.units` converts units** (pySIPNET PRs #46, #47). `conversion_factor(*, units,
  constituent="", to_units, to_constituent=None)` returns a float; `convert_units(values, *,
  units, ...)` takes the same keywords and scales unlabeled values whose units the caller
  states; `convert_dataarray_units(array, *, to_units, to_constituent=None)` reads `units` and
  `constituent` from `array.attrs`, refuses an array with no `units` attr, scales the data, sets
  the new `units`/`constituent`, and drops the unit-dependent attrs (`output_decimals`, and on
  climate variables `sipnet_internal_units` and `sipnet_internal_conversion`). Where a
  conversion needs a molar mass or a density, the constituent qualifies the **first** unit
  token, which must then be an amount or a mass of it, or for H2O also a depth or a volume. The
  tables are `MOLAR_MASS`, `DENSITY` and `ATOMS_PER_MOLECULE`. For example, `g m-2 d-1` of C to
  `umol m-2 s-1` of CO2 is 0.96362, and `Mg ha-1` to `g m-2` is 100. A per-step total such as
  SIPNET's `nee` (`g m-2`) is refused against a rate until it is divided by its step length.
- **`pysipnet.arithmetic` combines labeled arrays** (pySIPNET PR #48): `multiply_with_units`,
  `divide_with_units`, `add_with_units`, `subtract_with_units` return a `DataArray` whose
  `units`, `constituent` and `kind` (with `time_reference`, `cell_methods`) are true of the
  result, named `None`, with a `derivation` attr naming the operands. At most one operand of a
  product or quotient has a kind, never the denominator; a `timestep_total` over a time is a
  `daily_rate` and back (`KIND_AFTER_TIME_POWER` in `pysipnet.variables`); every other change of
  time dimension is refused. Index coordinates must match exactly, and conflicting non-index
  coordinates are refused. `step_length(data, units="d")` is the `timestep_length` coordinate
  as a float array, so `divide_with_units(nee, step_length(nee))` is `g m-2 d-1` of C.
  `parameter_dataarray(name, values, dims=, coords=)` and `SIPNETParameters.dataarray(name)`
  label a parameter's values from `ParameterSpec.xarray_attributes()` (no `kind`) and refuse
  values outside its domain (`ParameterDomain.contains`, sharing the Pydantic bounds).
- **`ClimateDrivers` owns the `.clim` format** (PRs #43, #45). It reads either layout, detected
  from the file (there is no `n_columns` argument for a file), validates once on load, and
  refuses labels that disagree with the declared step lengths: an overlap, or a drift from
  the running sum of lengths beyond `pysipnet.dataset.DRIFT_TOLERANCE` (5 minutes). `head(n)`
  gives a prefix; `to_file(path)` writes one. `time_zone="UTC"` or `"UTC±HH:MM"` declares the
  labels' clock as metadata only, and is `"undeclared"` otherwise: SIPNET has no clock of its
  own.
- `ClimateDrivers.from_file(path, *, time_zone=None)` and
  `pysipnet.io.clim_io.read_clim_file(path, *, time_zone=None)` (not re-exported from
  `pysipnet.io`) read and validate at once, so both refuse the local ERA5 files.
  `ClimateDrivers.from_path(path, *, time_zone=None)` only opens the file: a SIPNET run on it
  completes, and the refusal comes at the first read (`.xarray`, `.pandas`, or the output's
  Dataset). The tests use the `regular_drivers_root` fixture in `tests/conftest.py`.
- `pysipnet.dataset.assemble_time_coords(*, start, end, length, attributes_for, length_source,
  time_zone)` is keyword-only and `time_zone=` is required.
- **An output's time axis is its drivers'** (PR #43): the runner passes `climate=` to
  `SIPNETOutput`, and `SIPNETOutput.from_dataframe(df, *, climate=None, flags=None,
  run_id=None)` does the same by hand. Without drivers the axis
  falls back to the printed labels, which SIPNET rounds to 0.01 h; the Dataset's
  `time_axis_source` says which was used.
- **The Niwot reference data ships inside the package** (PR #40), so real SIPNET inputs and
  real SIPNET output are available with no pySIPNET checkout: `niwot_reference_output()`
  (a `SIPNETOutput`, no binary needed), `niwot_reference_climate()`,
  `niwot_reference_files()` (`.param` / `.clim` / `.output` / `.readme` paths), and
  `niwot_reference_parameters()` (the `SIPNETParameters` the reference output was run with,
  PR #54). The tests use all but the paths.
- **A `.param` file is read by pySIPNET** (PR #54): `SIPNETParameters.from_param_file(path)`,
  or `pysipnet.io.param_io.read_parameters(path)`. It refuses what SIPNET would misread
  (a value that is not a number, a duplicate name, an overlong line) and a file missing a
  parameter SIPNET always requires; a name it cannot hold is dropped with an
  `UnknownParameterWarning`. `param_io.read_param_file(path)` is the flat `{name: value}` dict.
- **Each parameter says when SIPNET requires it** (PR #54): `ParameterSpec.required_when`,
  `"always"` or a condition on the flags (`"snow"`, `"not gdd and not soil_phenol"`), which
  `validate_for_flags` and the reader both read. So `soil_respiration_moisture_exponent` is
  `float | None`, required only under `water_hresp`, and `leaf_on_day`/`leaf_off_day` are
  `NON_NEGATIVE`, since 0 switches that trigger off.
- **The binary is found, not assumed** (PR #41). `pysipnet.build.find_binary()` returns `None`
  when there is none and `missing_binary_message()` says where it looked. The search is
  `$PYSIPNET_BINARY`, then a binary bundled in the wheel, then — only when pySIPNET is
  running from a checkout — that checkout's cache, then `$PYSIPNET_CACHE_DIR` or the
  platform user cache. Both caches are keyed by the pinned SIPNET commit, so bumping the
  pin looks in a new, empty directory. **A git install has none of these until
  `pysipnet install-sipnet` runs** — this project installs pySIPNET from git, not editable,
  so that command is the setup step, not a fallback, and the checkout candidate never
  applies. `SIPNETRunner` verifies the binary matches the pinned tag before the first run.
  `pysipnet info` prints the pin, every path searched, and what was found.

### TensorFlow Probability (JAX substrate)
- `tfd.LogNormal`, `tfd.LogitNormal` and any `TransformedDistribution` expose `.distribution`
  (the base) and `.bijector`. The coordinates are the component's, never the law's:
  `ArraySpec.bijector` is its support's default unless overridden, and a bound factor
  (`probability._bound`) is evaluated by its law's base density only when the law's
  bijector agrees with the component's at the probe points, by change of variables
  otherwise.
- Bijectors are compared by their images, never by equality:
  `tfb.Sigmoid() == tfb.Sigmoid(low=0., high=1.)` is `False`.
- `LogNormal` and `LogitNormal` default to `Exp` and `Sigmoid` as their event-space
  bijector, `Gamma` and `HalfNormal` to `Softplus`; `MixtureSameFamily` has none
  (`None`), which is why `bind`'s support check also draws.
- `tfd.GaussianProcess(kernel, index_points, mean_fn)` with a `tfp.math.psd_kernels` kernel is a
  multivariate normal over the index points (event `(S,)`, batch `()`, analytic `.mean()` and
  `.covariance()`), so a GP prior needs no other package (issue #33). Built without x64 it is
  `float32`, which `bind` refuses; importing `sipnet_calibration` turns x64 on for the
  process, the package's one import-time side effect.
- TFP bijectors **cache** forward/inverse pairs: `b.forward(b.inverse(x))` hands `x` back
  unchanged, so a check that an input lies in a bijector's image must re-apply `forward` to a
  fresh copy of the array.
- Several TFP distributions (`MultivariateNormalTriL`, `Weibull`, `Gumbel`, ...) are
  `TransformedDistribution` subclasses over an internal reparameterization; their `.bijector` is
  not a map from unconstrained space. A bound factor reads `.distribution`/`.bijector` only off
  an exact `TransformedDistribution`, `LogNormal` or `LogitNormal` (`laws.CARRIES_ITS_BIJECTOR`).
- A law over a block puts the bijector outside the batch: `iid_over_dim` builds
  `TransformedDistribution(Sample(base, index_shape), b)`, whose base density is exact in theta.
- With the pinned build, a distribution built from `TransformedDistribution` pickles but fails
  `pickle.loads`; `LogNormal` and `LogitNormal` round-trip. Send PyEns workers plain data
  (SIPNET parameter fields through `pyens.xarray.fields_from_dataset`), never a model or a law.
- Moments do **not** pass through a non-affine bijector: `TransformedDistribution(...).mean()`
  raises `NotImplementedError`. Take them from `.distribution`.
- On the simplex, densities differ by their reference measure. `SoftmaxCentered`'s
  `forward_log_det_jacobian` is against the embedded volume element, `0.5 * logdet(J^T J)`,
  which differs from `log|det J|` of the first `k - 1` rows by `0.5 log k`; a `Dirichlet`'s
  density is against Lebesgue measure on the first `k - 1` coordinates.
  The probability layer's private log-Jacobian (`probability._bound.log_jacobian`) takes the
  latter, by autodiff, whatever the bijector.
- `IteratedSigmoidCentered` rounds a simplex coordinate to exactly 0 at `theta = 20 * 1` in
  float64, so the probe checks accept a bijector's image in the support's closure and skip
  probes that land on its boundary.
- In float64, `Gamma(0.02)` draws exactly 0, whose log is `-inf`, about once in 10^4 draws,
  depending on TFP's batch shape; `bind`'s draw-based support check catches it.
- Sampling takes `seed=` a `jax.random` key; `jax.random.fold_in(key, zlib.crc32(name))` is
  how the probability layer keys a factor by its name.
- `probability.families.InverseWishart` (a `TransformedDistribution` subclass over
  `WishartTriL`) matches `scipy.stats.invwishart`, but its
  `experimental_default_event_space_bijector()` refuses float64 input with the pinned build;
  the bind check of a law's own bijector treats such a bijector as absent and relies on draws.
  TFP's `Chain([CholeskyOuterProduct(), FillScaleTriL(diag_bijector=Exp(), diag_shift=None)])`
  `forward_log_det_jacobian(theta, event_ndims=1)` equals the log-Jacobian against the lower
  triangle that `probability._bound.log_jacobian` computes by autodiff.

### PyEns
- `EnsembleRunner(model, LocalBackend(n_workers=N)).run(EnsembleSpec(inputs=...))` — `model` must be defined at module level (pickling)
- `sipnet_member_fields(members_axis, **{param_name: list_of_floats})` from `pysipnet.ensemble` builds `Grid` specs
- `result.succeeded` is a list of `RunRecord`; access output via `rec.output`
- **`pyens.xarray`** (PyEns PR #7; the `xarray` extra, declared here as
  `pyens[xarray]`) builds specs from labeled data: `axes_of(obj)`,
  `field_from_dataarray(array, *, along=None, axes=None)`,
  `fields_from_dataset(dataset, *, along=None, axes=None)` and
  `dataset_as_field(dataset, *, along, axes=None)`. A dim with a coordinate
  becomes `Axis(dim, labels=[...])`, one without becomes `Axis(dim, size=n)`;
  datetime labels become ISO strings; a 0-d variable becomes `Fixed`. On
  SIPNET parameter fields from `SIPNETParameterMap.sipnet_parameter_fields`
  the int32 `site` and int64 `sample` coordinates become plain `int` labels,
  a variable is a grid along its own dims only (a fixed value on `(site,)`
  along the site axis alone), and a hand-built label-keyed
  `Grid({site_id: drivers}, along=Axis("site", labels=[...]))` zips with the
  result.
- **The pairing rule is the dim name.** PyEns makes one axis per dim, named for
  it: two batch dims of **one name zip** (paired label by label) and two of
  **different names cross** (every combination, multiplying the runs). So the
  SIPNET parameter fields' `sample`, the drivers' `driver_member` and the
  initial conditions' `initial_condition_member` cross, and pairing two
  ensembles deliberately is spelled by giving their dims one name. Same-named
  axes must be equal or PyEns raises: `Axis("sample", size=J)` is not equal to
  `Axis("sample", labels=[0, ..., J-1])` ("two axes named 'sample' have
  different structures"). `fields_from_dataset` makes the labeled form from a
  coordinate, so a `Grid` built by hand beside it must use an equal `Axis`;
  passing the same object is simplest (`fields_from_dataset` accepts `axes=`;
  `SIPNETRuns` builds its site axis once and passes it to every grid).
  `tests/test_fields.py` pins both halves of the rule against PyEns.

### EnsKit
- EnsKit is pyEKI rewritten and renamed (package `enskit`); `pyeki.gauss` and
  `pyeki.eki` are gone. It requires `jax>=0.10.1`, and importing it turns on 64-bit JAX, as
  importing this package does.
- **Distributions are over named blocks**, each a 1-D vector. `Gaussian(means, *, factors=,
  block_covs=, latent_dim=)` holds per block a mean, an optional row of a shared factor and an
  optional independent term; `Gaussian.independent(y=(mean, cov))` is one block with
  covariance `cov`. `log_density(y=values)` takes `(*batch, N)` values and returns `batch`;
  `sample(key, n)` returns an `Ensemble` and refuses `n < 2`.
  `Ensemble({"theta": array})` holds `(n_particles, d)` per block, read as `ensemble["theta"]`.
- There is no log-likelihood helper: `Gaussian.independent(y=(y, noise_cov)).log_density(
  y=predictions)` scores a batch of predictions by the symmetry of the density in point and
  mean, and equals `-eki.misfits(y, predictions, noise_cov) - (logdet R + N log 2 pi) / 2`
  (`from enskit.algorithms import eki`). Build `noise_cov` with `enskit.linalg.DensePSD(R)`,
  which factorizes the symmetric part `(R + R^T) / 2`; a factor already computed is passed by
  keyword, `DensePSD(L=L)`, and must be the **lower** Cholesky factor. Outside debug mode
  (`enskit.linalg.set_debug_checks(True)` turns it on), a row holding a NaN scores NaN, a row
  holding an inf scores `-inf`, and every row scores NaN when `noise_cov` is singular; the
  caller maps NaN to `-inf`.
- **The EKI driver** is `eki.run(eki.EKIState(ensemble, key=key), forward, y, noise_cov, *,
  update_rule=, schedule=, on_failure="raise")`. The state's ensemble is an unweighted
  `Ensemble` whose blocks are the parameters; `forward` receives one positional array per
  block and returns `(J, N)`. `update_rule` is required: `enskit.kalman.SymmetricSquareRoot()`
  (deterministic, exact in moments for the linear-Gaussian case) or `kalman.Matheron()`
  (stochastic). `on_failure="repair"` moves a particle whose prediction is not finite to the
  valid particles' center and warns once at the end of the run. A run that ends on one of
  EnsKit's schedules never evaluates its final ensemble: `result.last_evaluation.ensemble`
  holds the particles before the last update (after inflation and repair), and
  `EKIProblem.last_evaluation` the theta its forward map was handed then (before repair). A
  forward map returning a dtype wider than the ensemble's is refused, so a `float64` forward
  map needs a `float64` ensemble.
- `typing.get_type_hints` cannot resolve EnsKit's `Gaussian` (its `Array` annotation is
  imported for type checking only), so `tests/test_package.py`'s hint check skips names
  re-exported from another package.

### ProbPipe (deferred — not a current dependency)

ProbPipe was removed as a dependency on 2026-08-20 because its API is still in
flux; the plan is to migrate back once it stabilizes. The facts below were
verified against the source at that time and are kept for that migration —
**re-verify before relying on any of them.**

- Use `ProductDistribution(**{name: dist})` for named independent joint priors (not `Record`)
- `LogNormal(loc, scale, name=name)`, `Normal(loc, scale, name=name)` — `name` is a required keyword
- RWMH method name in `condition_on` is `"tfp_rwmh"` (not `"rwmh"`)
- Inside `log_likelihood`, `params` is a **flat 1-D JAX array** — call `prior.unflatten_value(params)` to get a `NumericRecord`, then access fields via `record["field_name"]`
- `prior._sample(key, (n,))` returns `NumericRecordArray`; fields via `arr["name"]` give shape `(n,)` arrays
- `posterior.chains` is a list of flat `(n_draws, n_params)` arrays
- `posterior.inference_data` is an ArviZ `InferenceData` object
- Type for a prior with named fields: `RecordDistribution` from `probpipe.core._record_distribution` (exported from `probpipe`)

#### ProbPipe sharp edges (note for ProbPipe developer)
- **RWMH is a pure Python for-loop** — no `jax.jit`, no `jax.vmap`. `log_likelihood` is called
  once per step with a single `(n_params,)` array; return value is wrapped in `float()`.
  Python side-effects (subprocess, file I/O) are safe. This is good for SIPNET but means
  RWMH cannot exploit JAX's parallelism or JIT compilation. A future `jax.pure_callback`-based
  variant could enable JIT and true chain parallelism via `vmap`.
- **No batching contract on `log_likelihood`** — ProbPipe's Likelihood protocol only requires
  scalar output. There is no vectorized/batched variant expected by any current sampler. This
  means subclasses cannot inadvertently trigger batched calls, but also means there is no
  path to a vectorized likelihood (e.g., for HMC with batched proposals) without a protocol change.

## Running notebooks

Notebooks must be run with the project venv's jupyter (not `uv run jupyter`):

```bash
.venv/bin/jupyter lab
# or to execute headlessly:
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 \
    --output <output.ipynb> <input.ipynb>
```
