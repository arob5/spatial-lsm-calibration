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
| `PyEns` | `arob5/PyEns` | Parallel ensemble execution via `ProcessPoolExecutor` |
| `pyEKI` | `TARPS-group/pyEKI` | Solving inverse problems with ensemble Kalman methods |
| `ProbPipe` | `TARPS-group/prob-pipe` (also on PyPI) | **Not currently a dependency** — API in flux; planned migration target for inference. See below. |

The first three are dependencies, installed from git rather than from sibling
directories: `[tool.uv.sources]` tracks each repository's `main` branch and
`uv.lock` pins an exact commit, so `uv sync` needs nothing beside the checkout
and no companion moves until someone upgrades it. Take new work from one with
`uv lock --upgrade-package <pysipnet|pyens|pyeki>`; the README also covers the
editable-overlay workflow for developing one locally. Never modify their
source from here.

## Data

**`data/README.md` is the authority on data formats, provenance, units, and the
open questions.** Read it before writing ingest or adapter code, and do not
duplicate its content here — add data facts there instead.

Facts specific to this working copy, which the README deliberately does not carry:

- **Only a subset of `data/raw/` is present locally.** Drivers exist for
  `ERA5_1_1`, `ERA5_1_2` and `ERA5_27_5`; initial conditions for site 1 members
  1 and 2 and site 27 member 94. The NEE csv, the AGB/LAI `.Rdata` files and the
  site shapefile are complete. The full dataset lives on Boston University's
  SCC. Anything that needs to hold across all 8000 sites cannot be verified
  here.
- The local files are real copies, not symlinks. On SCC they should be symlinks.
- R is available on this machine (`Rscript`), which is how the `.Rdata` files can
  be inspected; `pyreadr` is not installed and would not handle their nesting.
- Neither `pyproj` nor any R spatial package is installable here, so CRS and
  projection definitions cannot be validated locally (issue #4).

Operational rules that follow from the data and are easy to get wrong in code:

- Open the IC netCDFs with `decode_times=False` (README note 5).
- Never build a timestamp from the `.clim` or SIPNET-output `time` column; it
  drifts (README note 15, issue #9). Use `obs_ops.sipnet_time_index`, which
  takes only the slot from it.
- Drop the NEE csv's `ens_mean` column; never admit it to the `member` dim.
- Never renumber the 1-8000 site ids; they are a shared key with collaborators.
- The site table is `data/raw/sites/pts.*` (tracked) and, after ingest,
  `data/processed/sites/sites.csv`. There is no other site source.
- Do not assume rectangular coverage: NEE is ~55% missing over site x time, and
  the AGB/LAI constraints are ragged over site x year x variable.

## Code conventions

These are project-wide and apply to new code without being restated.

### Naming in processed data

Raw variable names are not ours to choose; processed ones are.

- **`lower_case_with_underscores`** for every variable, coordinate and column of
  a processed product.
- **Avoid abbreviations** unless they are universal. So `total_soil_carbon`, not
  `TotSoilCarb`; `aboveground_wood_carbon`, not `AbvGrndWood`;
  `soil_moisture_percent`, not `SoilMoistFrac`. `lai` is fine, and so are
  `lon`/`lat`, which the canonical field convention fixes.
- The rename from source to processed name belongs in **one explicit mapping**
  in the library beside the schema, not spread across a script. See
  `SOURCE_VARIABLE_NAMES` in `sipnet_calibration.constraints`. Keep the source
  name in the product's attributes so the correspondence is never guesswork.
- Renaming is safe only where a record carries its own identity. Where the
  source pairs values *positionally*, the positional read stays in source names
  and the rename happens after the data is self-describing.
- **The `VARIABLES` registry is keyed on processed names**, so a canonical
  field's `name` is a processed name. That is what makes `validate_field()`
  usable against anything an adapter produces.

### File organization

- **Public first, private last.** Public functions, classes and constants at the
  top of a file; helpers and anything underscore-prefixed below them.
- Data processing scripts follow the section order
  `entry point` -> `the steps, in the order main calls them` ->
  `supporting types and helpers` -> `checks`, with `# ── ... ──` section
  comments. `scripts/ingest_sites.py` and `scripts/ingest_constraints.py` are
  the worked examples.
- Keep functions short enough that the top-level one reads as a summary of the
  work. If it stops reading that way, pull a step out as a helper. Roughly 40
  lines is where to start looking for the seam, not a hard limit.

### Data validation in processing scripts

- Every validation check is its own helper named **`check_*`**, saying what it
  checks: `check_covariances_were_diagonal`, `check_no_duplicate_triples`,
  `check_sites_are_in_the_site_table`. Not `validate`, not an inline `assert`
  buried in a transformation.
- The `check_*` helpers live together in the **`checks` section at the bottom**
  of the file.
- A check raises with a message naming the invariant that broke and, where
  possible, what to do about it. The script's `main` turns those into a reported
  error rather than a traceback.

### What does and does not belong in documentation

- **Do not write volatile measurements into documentation.** Row counts, cell
  counts, file sizes, per-variable coverage, "929 of them are zero" — these
  describe one snapshot of the data and go stale silently. A numeric property
  the code depends on is **checked programmatically**: an assertion in
  the ingest script, a constant in the library, or a test. Documentation says
  what the property *is* and where it is checked, not what it currently
  measures. Where a run's numbers are genuinely useful, print them.
  `data/README.md` is the exception, since recording measured characteristics of
  the raw data is its job — but even there, anything the code relies on is
  asserted in code as well, not just written down.
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
3. **The data model** — for an xarray product, the dims, the data variables and
   their dtypes, the coordinates and which dims they are on, the attributes,
   and what missing means. State it plainly; do not make the reader infer it
   from the validation code.
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

### Products and their readers

- **Schema constants and the reader live in the library**, not the script, so
  the writer and the reader of a product cannot drift apart
  (`SITE_COLUMNS` in `sites.py`, `CONSTRAINT_VARIABLES` in `constraints.py`).
  A script's own round-trip check calls the library loader, never a parallel
  reader.
- **Write to a `.partial` path and rename only after the checks pass**, so a
  failed run cannot leave a corrupt file at the canonical path.

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
its own, which takes one command and nothing beside it:

```bash
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

## Repository layout

The layout below is the **agreed target**, specified in
`logs/2026-08-28_Plotting Design Spec.md` in the Obsidian vault. The src-layout
reorg has landed, so the paths below are the real ones; `sites.py`,
`constraints.py` and `drivers.py` are implemented, `obs_ops.py` has
`sipnet_time_index`, and the other modules carry the contract each is to
satisfy.

```
pyproject.toml            # name = "sipnet-calibration"; src layout
src/sipnet_calibration/
  sites.py                # SITE_GRID + grid conversions, load_sites(),
                          # select_sites(ids=, bbox=, where=, sample=, seed=)
  constraints.py          # annual constraint schema, load_constraints(),
                          # constraint_fields() -> canonical per-variable view
  drivers.py              # driver schema, load_drivers() reading raw .clim files
                          # into (member, site, time); no processed file exists
  fields.py               # canonical field convention, validate_field(), adapters
  obs_ops.py              # sipnet_time_index (done); aggregate_time (issue #6) —
                          # shared with the likelihood
  plotting/
    __init__.py           # curated exports
    style.py              # ROLES, rcParams
    registry.py           # VARIABLES
    primitives.py         # L1: (ax, plain numpy, **style) -> artist
    series.py             # L2 time series panels
    maps.py               # L2 spatial panels + SpatialRenderer implementations
    facet.py              # L3 the one generic facet function
    diagnostics.py        # L5 EKI history, marginals, coverage
scripts/                  # ingest: data/raw/ -> data/processed/
experiments/<task>/       # config.py (source of truth) + plots.py (L4 reports)
data/raw/                 # never edited; only raw/sites/ (the site shapefile) is tracked
data/processed/           # ingest output == canonical plotting input; untracked
tests/
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

- **Canonical field**: an `xr.DataArray` with dims a *subset* of
  `(member, site, time)`, `lon`/`lat` as non-dimension coords on `site`, and
  units/`long_name` in `attrs`. It is a **convention plus `validate_field()`**,
  not a wrapper class — a wrapper would fight xarray's `.sel`/`.resample`/
  `.quantile`, which are the three operations this project needs. One
  `DataArray` per variable; facet-by-variable takes `dict[str, DataArray]`.
- Plotters branch on **presence of the `member` dim**, never on a mode keyword.
- **Temporal aggregation lives in `obs_ops.py`** and is imported by both the
  observation operator and the plotting layer, so a predictive-check figure
  cannot disagree with what the likelihood consumed. Aggregation is a verb the
  caller applies — `series_panel(agg(f, "1D"))` — never a plotter keyword.
- **The aggregation rule is a property of the variable, carried in `VARIABLES`
  as `agg`.** SIPNET's `nee` is `g C m-2 per timestep` — extensive — so
  3-hourly to daily is a **sum**; a mean is wrong by 8x and looks plausible.
  `tair`/`vpd` are intensive (mean); `par`/`precip` are per-timestep totals
  (sum); carbon pools and `aboveground_wood_carbon`/`lai` are stocks
  (instantaneous).
  `aggregate_time` reads the registry; `how=` is an override, not the input.
- **Model and observed NEE are not in the same units.** Observed NEE is
  `umol CO2 m-2 s-1` (a rate); SIPNET's is `g C m-2` per timestep (a total).
  Adapters convert into the one canonical unit named in `VARIABLES`, and
  `validate_field()` checks `attrs["units"]` against it. Plotting the two on one
  axis without converting fails silently, by orders of magnitude.
- **L1 primitives** take `(ax, plain numpy, **style)` and return artists: no
  pandas, no xarray, no figure creation. **No plotter** calls `plt.show()` or
  `savefig`, creates a figure implicitly, or accepts a `SIPNETResult` or a path
  (that is an adapter's job).
- **Anything that knows an experiment/task name belongs in
  `experiments/<task>/plots.py`, not the library.**
- Style comes from the `VARIABLES` registry and `ROLES` palette, not per-call
  keywords. `center=0.0` for signed fluxes such as NEE is correctness, not
  cosmetics.
- Spatial rendering goes through the `SpatialRenderer` protocol (default
  `tripcolor` on the Delaunay triangulation, masking long edges; GP renderer
  later). Sites are 8000 **irregular points** spanning 7-82 deg N, so a real
  projection is required and CONUS-only assumptions are wrong.
- **No projection library is installable here** (issue #4): every pyproj arm64
  wheel targets macOS 14+, on every Python version, so downgrading Python does
  not help. `cartopy` is commented out of `pyproject.toml`; do not re-add it
  expecting it to work locally. `plotting/maps.py` is blocked on that decision.
  The source CRS is settled (WGS 84 geographic) and the grid is `SITE_GRID` in
  `sipnet_calibration.sites`; what is open is only the display projection.
- `site` is the integer 1-8000; `ameriflux_site_id` is a non-dimension coord on
  `site`. PFT is **not** site metadata and is not a column of the site table: a
  labeling is an experimental choice, so labelings are their own product at
  `data/processed/labelings/<name>.csv`, keyed on `site_id`, and a caller joins
  one on before selecting. `member` is a 0-based integer, meaningful only within
  one source. See the Data section above for the rules these imply.

## Key API facts (hard-won from source reading)

### pySIPNET
- `SIPNETModel(runner, base_params=..., base_climate=...)(**overrides) -> SIPNETResult`
- `ClimateStaging` is in `pysipnet.runner`, not `pysipnet.climate`
- `SIPNETRunner(climate_staging=ClimateStaging.SYMLINK)` — staging goes on the runner, not the model
- Parameter override keys are flat snake_case leaf names (`a_max`, not `photosynthesis.a_max`)
- `result.nee()` returns a `pd.Series` of one value per climate timestep (sub-daily if climate is sub-daily)
- `ClimateDrivers` has no `slice()` or `to_path()` — slice by reading/writing raw text lines

### PyEns
- `EnsembleRunner(model, LocalBackend(n_workers=N)).run(EnsembleSpec(inputs=...))` — `model` must be defined at module level (pickling)
- `sipnet_member_fields(members_axis, **{param_name: list_of_floats})` from `pysipnet.ensemble` builds `Grid` specs
- `result.succeeded` is a list of `RunRecord`; access output via `rec.output`

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
