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
| `pyEKI` | `TARPS-group/pyEKI` | Solving inverse problems with ensemble Kalman methods |
| `ProbPipe` | `TARPS-group/prob-pipe` (also on PyPI) | **Not currently a dependency** — API in flux; planned migration target for inference. See below. |

The first three are dependencies, installed from git rather than from sibling
directories: `[tool.uv.sources]` tracks each repository's `main` branch and
`uv.lock` pins an exact commit. A sibling clone of any of them is **not** what
gets installed — `uv` fetches the pinned commit from GitHub — so nothing about
a local checkout reaches this project, and unpushed work in one is invisible
here. Never modify their source from here.

The boundary between them: PyEns owns the shape of an ensemble and the
translation of labeled data into and out of it; pySIPNET owns one SIPNET run's
inputs and outputs; pyEKI owns the ensemble Kalman update; this repository owns
what varies and why. A function belongs in pySIPNET only if deleting SIPNET from
it leaves nothing.

**All three are under active development, so start any work here by taking
their current `main`:**

```bash
uv lock --upgrade-package pysipnet --upgrade-package pyens --upgrade-package pyeki
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
  `scripts/survey_soil_texture.py` needs `--no-check` here. The NEE csv, the
  five constraint files (tracked), the converted initial condition ensemble
  (tracked, all 8000 sites x 100 members), the assembled `.Rdata` pair retained
  for validation, the site shapefile, both leaf phenology CSVs, and the two
  site-labels products and the covariate table (all tracked) are complete. The
  full dataset lives on Boston University's SCC. Anything about the drivers that
  needs to hold across all 8000 sites cannot be verified here.
- In the root checkout the storage-backed inputs are real copies, not symlinks;
  on SCC, and in a worktree that links them from the root, they are symlinks.
  The five constraint files and the site shapefile are tracked either way.
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
- Drop the NEE csv's `ens_mean` column; never admit it to the `member` dim.
- Never renumber the 1-8000 site ids; they are a shared key with collaborators.
- The site table is `data/raw/sites/pts.*` (tracked) and, after ingest,
  `data/processed/sites/sites.csv`. There is no other site source.
- Do not assume rectangular coverage: NEE is ~55% missing over site x time, and
  every constraint product is ragged over site x time.
- Constraint products keep their **source units** and their source's own time
  labels. Nothing is converted or aligned at ingest; the observation operator
  does both. See the processed-data conventions below.
- The initial condition product is in the source files' units, negative wood and
  leaf draws included, and applies **no state-to-parameter mapping**: three of
  the four SIPNET initial parameters depend on calibrated parameters, so the
  mapping is the experiment layer's, per proposed parameter vector. Each spec's
  `pecan_conversion` says what PEcAn did.

## Code conventions

These are project-wide and apply to new code without being restated.

### Naming in processed data

Raw variable names are not ours to choose; processed ones are.

- **`lower_case_with_underscores`** for every variable, coordinate and column of
  a processed product.
- **Avoid abbreviations** unless they are universal. So `soil_organic_carbon`,
  not `soc`; `aboveground_biomass`, not `agb`; `standard_deviation`, not `sd`.
  `lai` is fine, and so are `lon`/`lat`, which the field convention
  fixes.
- The correspondence from source to processed belongs in **one explicit spec**
  in the library beside the schema, not spread across a script. See
  `ConstraintSpec` in `sipnet_calibration.constraints`, whose `raw_file`,
  `value_column` and `sd_column` name the source and whose fields are written
  into the product as `source_file` and `source_column`, so the correspondence
  is never guesswork.
- A constraint product is named for its **raw file's stem**
  (`modis_leaf_area_index`, not `lai`): the name is then the spec's key, the
  raw file and the output file at once, it keeps two products of one quantity
  apart, and it asserts nothing the source does not (biomass, not carbon).
- Renaming is safe only where a record carries its own identity. Where the
  source pairs values *positionally*, the positional read stays in source names
  and the rename happens after the data is self-describing.
- **The `VARIABLES` registry is keyed on processed names**, so a
  field's `name` is a processed name. That is what will make `validate_field()`
  (owed, issue #6) usable against anything an adapter produces.

### Naming in code

The rules that keep a name from having to be looked up. The first four are
the ones most often broken.

- **A name and the thing are named differently.** A string or tuple of
  strings is `<thing>_name` / `<thing>_names`; the things themselves are the
  plural noun. `parameter_names` and `sipnet_parameter_names` are names;
  `sipnet_parameters` are values. `variable` and `parameter` never mean a
  string.
- **A variable holding one representation of a concept says which.** The
  parameter vector's Flat is `theta`, its Fields is `<thing>_fields`, its
  SIPNET table is `sipnet_table`; the observation vector's Flat is `y`. A
  pySIPNET object is named for its class (`sipnet_result`, `sipnet_output`,
  `sipnet_parameters`); the labeled xarray of a run's output is
  `model_output`. Nobody should have to ask whether a value is a
  `ParameterVector`, a `CalibrationParameter` or a SIPNET parameter.
- **`sipnet_` prefixes anything in pySIPNET's vocabulary**: names, values,
  objects. The bare word or `calibration_` is this repository's vocabulary.
  `pysipnet_` is not used: this repository reaches SIPNET only through
  pySIPNET, so a second prefix would have no second referent.
- **An `xr.Dataset` when the variables share one grid, a
  `dict[str, DataArray]` when they do not.** One run's or one stack's model
  output shares a time axis and is a Dataset; the constraint products have
  three time structures and are a dict. A dict is named for what it holds and
  its key (`observed_values`, keyed by product name).
- **A field is one thing.** An `xr.DataArray` holding one variable, whose
  dims are a subset of `(member, site, time)`, with `lon`/`lat` on `site` and `units`/`long_name`
  in its attributes. The word "canonical" is not used with it; `fields.py`
  holds the generic operations on fields and nothing else.
- **A class whose call is its whole interface may be named as an imperative
  verb**, for what the call does (`SelectTimestep`, `ComputeLeafAreaIndex`).
  This is not a rule for every callable class: a protocol, a record, or an
  object with an interface of its own beyond the call is a noun
  (`ObservationOperator`, `Observation`, a model object).
- **No abbreviations** beyond the universal ones, as above:
  `constraint_standard_deviations`, not `constraint_sds`. Names that are
  pandas', xarray's or pySIPNET's own (`how`, `freq`, `coords`, `dims`) stay,
  because matching `pysipnet.resample(how=)` or `DataArray.coords` is worth
  more than spelling them out.
- **An argument is named for what it is for**, never for where it sits: not
  `at`, not `data`, not `x`.

Existing code is brought into line by mechanical rename PRs, one module at a
time, with the tests renamed alongside; `logs/2026-09-24_Naming
Conventions.md` in the vault lists what still changes.

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
- **Code does not cite the vault.** Docstrings, comments and provenance
  strings never point at the Obsidian vault, a design log or the readiness
  report: those live outside the repository, so a reader of the code cannot
  follow the reference, and they move. A citation in code names a primary
  source -- a paper, a line of the SIPNET source, a data producer or product
  (BETY, ISCN), a pySIPNET module. Design reasoning goes the other way: the
  vault cites the code.
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

### Processed data conventions

- **Processed products follow the Climate and Forecast conventions, CF-1.11**,
  as pySIPNET's model output does since its PR #38, so model and observation
  read alike: `Conventions = "CF-1.11"` on the dataset; `standard_name`,
  `axis` and, where present, `bounds` on `time`; `standard_name` and `units`
  on `lon`/`lat`; no `_FillValue` on any coordinate. Where a value's support
  is documented it is a CF `time_bounds(time, bounds)` coordinate, as
  pySIPNET writes `time_bounds = [time_step_start, time]`. Where CF has no
  vocabulary for what a label means, the meaning goes in words
  (`time_reference`, `comment`), never in a `cell_methods` that is not
  literally true.
- **Ingest changes structure, never values.** No unit conversion, no temporal
  alignment, no choice of which record stands for a year. Units are the raw
  file's; the observation operator converts through Pint
  (`pysipnet.units.unit_registry`) and decides the alignment.
- **One flat spec per variable, from which everything is derived**, in the
  shape of pySIPNET's `VariableSpec`: the spec's `xarray_attributes()` is what
  the product stores, so a netCDF describes itself and there is no separate
  processed schema to keep in step. `ConstraintSpec` is the worked example.
- **A unit that is inferred is recorded as inferred**, in a `units_provenance`
  sentence on the spec and the product, not as a status enum.

### Products and their readers

- **Schema constants and the reader live in the library**, not the script, so
  the writer and the reader of a product cannot drift apart
  (`SITE_COLUMNS` in `sites.py`, `CONSTRAINTS` in `constraints.py`).
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
its own, taking the companions' current `main` as above while you are there:

```bash
uv lock --upgrade-package pysipnet --upgrade-package pyens --upgrade-package pyeki
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
`#$ -P dietzelab` with `#$ -l buyin` is the queue.

## Repository layout

The layout below is the **agreed target**, specified in
`logs/2026-08-28_Plotting Design Spec.md` in the Obsidian vault. The src-layout
reorg has landed, so the paths below are the real ones; `sites.py`,
`constraints.py`, `initial_conditions/`, `drivers.py`, `projection.py`,
`parameter_vector.py` and `site_labels.py` are implemented, `obs_ops.py` has
`aggregate_time`, `fields.py` has the model-output
adapters, the plotting package has series, maps and grids, and the other
modules carry the contract each is to satisfy.
`initial_conditions` is a package rather than a module: it spans several
artifacts, and giving each its own file keeps that artifact's schema, writer,
reader and checks together.

```
pyproject.toml            # name = "sipnet-calibration"; src layout
src/sipnet_calibration/
  sites.py                # SITE_GRID + grid conversions, load_sites(),
                          # select_sites(ids=, bbox=, where=, sample=, seed=),
                          # EXTENTS (named lon/lat boxes)
  projection.py           # SITE_PROJECTION (LAEA 50 N, 100 W) over pyproj:
                          # forward(), projected_bounds(), factors()
  projections/            # the stored definition, generated from the dataclass
  constraints.py          # ConstraintSpec + CONSTRAINTS, one per raw file;
                          # read_raw(), build_constraint(), load_constraint(),
                          # constraint_fields() -> one field per product
  conventions.py          # CF_CONVENTIONS and data_root(): the settings every
                          # product has to agree on
  initial_conditions/     # one module per artifact; __init__ re-exports them all
    __init__.py           # curated exports + the product's data model
    names.py              # SITE/MEMBER, the two file names, the path helpers
    source_files.py       # SOURCE (the PEcAn file format), read_source_file()
    specs.py              # InitialConditionSpec + INITIAL_CONDITIONS
    raw.py                # build_raw(), raw_encoding(), read_raw()
    processed.py          # build_initial_conditions(), load_initial_conditions(),
                          # netcdf_encoding(), initial_condition_fields()
    sipnet_parameters.py  # to_sipnet_initial_conditions() and its table form
  drivers.py              # load_drivers(): raw .clim files read by pySIPNET's
                          # ClimateDrivers, stacked into (member, site, time) on
                          # pySIPNET's axis; no processed file exists
  site_labels.py          # SiteLabelsSpec + SITE_LABELS, one per raw file; a
                          # site-labels product is site_id -> class, one product
                          # per source; read_raw(), build_site_labels(),
                          # load_site_labels(), site_labels_field() -> CF flags
  parameter_vector.py     # ParameterVector: a named random vector over sites, built
                          # from CalibrationParameters (TFP prior in natural space,
                          # varies_by, SIPNETMap) and FixedParameters; select(),
                          # sample/log_prior/gaussian_prior on Flat (J, D); three
                          # value representations with named conversions:
                          # fields() <-> flat() (Fields: Dataset of
                          # fields on (member, site), attrs["space"]), and
                          # sipnet_table() -> sipnet_overrides() / pyens_grids();
                          # example_parameter_vector()
  fields.py               # field convention; from_sipnet_output(),
                          # stack_sipnet_outputs() over SIPNETOutput.select,
                          # site_lookup(); validate_field() and the adapters
                          # for the other sources (issue #6)
  obs_ops.py              # aggregate_time — shared with the likelihood;
                          # obs_index (issue #6)
  plotting/
    __init__.py           # curated exports
    style.py              # ROLES, rcParams
    registry.py           # VARIABLES
    primitives.py         # L1: (ax, plain numpy, **style) -> artist
    series.py             # L2 time series panels
    maps.py               # L2 plot_map (points/cells/triangles renderers,
                          # rasters, classes), member_summary, animate_map
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
  raw_sources/            # NOT the pipeline: code that *makes* a tracked raw
                          # input, run once. SCC-only except the Natural
                          # Earth download.
experiments/<task>/       # config.py (source of truth) + plots.py (L4 reports)
data/raw/                 # never edited; raw/sites/, raw/constraints/,
                          # raw/initial_conditions/, raw/site_labels/,
                          # raw/covariates/ and raw/natural_earth/ are tracked
data/processed/           # ingest output == the plotting input; untracked;
                          # constraints/<name>.nc is one CF-1.11 netCDF per constraint
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

- **Field**: an `xr.DataArray` with dims a *subset* of
  `(member, site, time)`, `lon`/`lat` as non-dimension coords on `site`, and
  units/`long_name` in `attrs`. It is a **convention plus `validate_field()`**,
  not a wrapper class — a wrapper would fight xarray's `.sel`/`.resample`/
  `.quantile`, which are the three operations this project needs. One
  `DataArray` per variable; facet-by-variable takes `dict[str, DataArray]`.
- Plotters branch on **presence of the `member` dim**, never on a mode keyword.
- **Temporal aggregation lives in `obs_ops.py`** and is imported by both the
  observation operator and the plotting layer, so a predictive-check figure
  cannot disagree with what the likelihood consumed. Aggregation is a verb the
  caller applies — `plot_time_series(aggregate_time(f, "1D"))` — never a
  plotter keyword.
- **The variable's kind says which resampling methods are valid; the caller
  may name one.** pySIPNET owns the first half: since its PR #38 every
  variable has a `kind`, `RESAMPLING_METHODS_FOR_KIND` says what may be done
  with it, and `pysipnet.resample.resample(ds, freq, how=...)` requires `how`,
  weights means by step length and refuses a method the kind does not support
  (a pool is not additive; a per-step total is not averaged until it is a
  rate). `obs_ops.aggregate_time(field, freq, how=None)` is that operation for
  a field — a field may have `member` and `site` dims, which
  `resample` does not reduce over — and it adds one thing: with no `how` it
  takes **the method that leaves the variable the kind it already is**, read
  off pySIPNET's `RESAMPLED_KIND` rather than written down. A total sums, a
  step mean or a rate means, a pool or a running total takes its last value.
  SIPNET's `net_ecosystem_exchange` is `g m-2` of C per timestep, so 3-hourly
  to daily is a **sum**, and a mean is wrong by 8x while looking plausible;
  the default is there so that omission cannot reach that error, and `how=` is
  for asking deliberately for something else, such as the time-weighted mean
  of a pool. An invalid pair is refused in pySIPNET's own words.
- **Model and driver fields carry pySIPNET's names, units, kinds and time axis
  unchanged.** `fields.from_sipnet_output` adds `site`, `member` and
  `lon`/`lat` to a run's output; `drivers.driver_fields` does the same for the
  drivers, read through `ClimateDrivers`. The registry names are already
  `lower_case_with_underscores`, so they are the processed names. Both keep
  `fields.TIME_COORDS`, pySIPNET's axis: `time` at the step end, with
  `time_step_start` beside it, so the interval a value covers is
  `[time_step_start, time]` — the pair pySIPNET writes as its CF `time_bounds`
  variable — and a run's output and the drivers it ran on share one axis by
  construction. `time_bounds` itself cannot ride on a field, its `bounds`
  dimension being no field dimension, so it and the `time` attribute naming it
  are dropped, along with SIPNET's `year`/`day_of_year`/`hour_of_day` row
  labels, which `time_step_start` already is.
- **Model and observed NEE are not in the same units.** Observed NEE is
  `umol CO2 m-2 s-1` (a rate); SIPNET's is `g C m-2` per timestep (a total).
  Observation products keep their source units; the observation operator
  converts the model into the observation's units, through Pint, before a
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
  by another name. `plot_map` refuses a `member` or `time` dim rather than
  reducing it: use `member_summary`, `plot_map_by`, `plot_map_quantiles` or
  `animate_map`. A GP is not fitted in plotting; its predictions are a
  `(lat, lon)` raster or site values, mapped like any field. Categorical
  fields are CF `flag_values`/`flag_meanings`, colored by class position so a
  class keeps its color across figures; an optional `flag_display_names`
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
- `site` is the integer 1-8000; `ameriflux_site_id` is a non-dimension coord on
  `site`. PFT is **not** site metadata and is not a column of the site table:
  which site labels to use is an experimental choice, so site labels are their
  own product at `data/processed/site_labels/<name>.csv`, keyed on `site_id`,
  and a caller joins one on before selecting. `member` is a 0-based integer,
  meaningful only within one source. See the Data section above for the rules
  these imply.

## Key API facts (hard-won from source reading)

### pySIPNET
- `SIPNETModel(runner, base_params=..., base_climate=...)(**overrides) -> SIPNETResult`
- `ClimateStaging` is in `pysipnet.runner`, not `pysipnet.climate`
- `SIPNETRunner(climate_staging=ClimateStaging.SYMLINK)` — staging goes on the runner, not the model
- Parameter override keys are flat snake_case leaf names
  (`max_photosynthesis_rate`, not `photosynthesis.max_photosynthesis_rate`)
- `SIPNETOutput` selects with `out["nee"]` (a `DataArray`) and `out[["nee", "gpp"]]`
  (a `Dataset`); aliases resolve. `result.nee()` and `to_xarray()` are gone (pySIPNET PR #36).
- The output `Dataset` is CF-1.11: `time` is the **end** of each step, `time_step_start` and
  `time_step_length` are coordinates, and `time_bounds = [time_step_start, time]` (PR #38).
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
  run_id=None)` does the same by hand. `time_step_length=` is gone. Without drivers the axis
  falls back to the printed labels, which SIPNET rounds to 0.01 h; the Dataset's
  `time_axis_source` says which was used.
- **The Niwot reference data ships inside the package** (PR #40), so real SIPNET inputs and
  real SIPNET output are available with no pySIPNET checkout: `niwot_reference_output()`
  (a `SIPNETOutput`, no binary needed), `niwot_reference_climate()`,
  `niwot_reference_files()` (`.param` / `.clim` / `.output` / `.readme` paths). The tests here
  use the first; there is still no public `.param` reader (pySIPNET issue #19), so
  `tests/conftest.py` carries a small one.
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
  (the unconstrained base) and `.bijector`; `sipnet_calibration.parameter_vector` stores one
  prior per calibration parameter and reads both off it.
- `tfd.GaussianProcess(kernel, index_points, mean_fn)` with a `tfp.math.psd_kernels` kernel is a
  multivariate normal over the index points (event `(S,)`, batch `()`, analytic `.mean()` and
  `.covariance()`), so a GP prior needs no other package; `ParameterVector` admits it as a
  *joint* prior over a scalar calibration parameter's groups (issue #33). Built without x64 it is
  `float32`, which the vector refuses.
- Batch slicing by an **integer** index array (`prior[np.array([0, 2])]`) works for `LogNormal`
  and `LogitNormal` but **fails for a batched `softmax_normal`** once two or more indices are kept
  (TFP's slicing of the `MultivariateNormalDiag` base), so `parameter_vector` rebuilds that family
  from its sliced moments. Slicing by a **boolean** mask silently returns the wrong shape. TFP does
  not slice event dimensions, so a joint prior is restricted by taking the Gaussian marginal of
  its base.
- TFP bijectors **cache** forward/inverse pairs: `b.forward(b.inverse(x))` hands `x` back
  unchanged, so a check that an input lies in a bijector's image must re-apply `forward` to a
  fresh copy of the array.
- Several TFP distributions (`MultivariateNormalTriL`, `Weibull`, `Gumbel`, ...) are
  `TransformedDistribution` subclasses over an internal reparameterization; their `.bijector` is
  not a map from unconstrained space. `parameter_vector` reads `.distribution`/`.bijector` only
  off an exact `TransformedDistribution`, `LogNormal` or `LogitNormal`.
- With the pinned build, a distribution built from `TransformedDistribution` or `Blockwise`
  (the simplex and product priors) pickles but fails `pickle.loads`; `LogNormal` and `LogitNormal`
  round-trip. Send PyEns workers plain data (`pyens_grids`), never a `ParameterVector`.
- Moments do **not** pass through a non-affine bijector: `TransformedDistribution(...).mean()`
  raises `NotImplementedError`. Take them from `.distribution`.
- `SoftmaxCentered`'s density on the simplex is against the embedded volume element,
  `0.5 * logdet(J^T J)`, which differs from `log|det J|` of the first `k - 1` rows by `0.5 log k`.
- Sampling takes `seed=` a `jax.random` key; `BatchBroadcast(dist, to_shape=(n,))` batches a
  shared prior over groups.

### PyEns
- `EnsembleRunner(model, LocalBackend(n_workers=N)).run(EnsembleSpec(inputs=...))` — `model` must be defined at module level (pickling)
- `sipnet_member_fields(members_axis, **{param_name: list_of_floats})` from `pysipnet.ensemble` builds `Grid` specs
- `result.succeeded` is a list of `RunRecord`; access output via `rec.output`
- **`pyens.xarray`** (PyEns PR #7; the `xarray` extra, declared here as `pyens[xarray]`)
  builds specs from labeled data: `axes_of(obj)`,
  `field_from_dataarray(array, *, along=None, axes=None)`,
  `fields_from_dataset(dataset, *, along=None, axes=None)` and
  `dataset_as_field(dataset, *, along, axes=None)`. A dim with a coordinate becomes
  `Axis(dim, labels=[...])`, one without becomes `Axis(dim, size=n)`; datetime labels become
  ISO strings; a 0-d variable becomes `Fixed`. On a SIPNET table from `example_parameter_vector`
  the int32 `site` and int16 `member` coordinates become plain `int` labels, a non-dimension
  coordinate such as `pft` is ignored, and a hand-built label-keyed
  `Grid({site_id: drivers}, along=Axis("site", labels=[...]))` zips with the result.
- **Trap:** `Axis("member", size=J)` is not equal to `Axis("member", labels=[0, ..., J-1])`, and
  an `EnsembleSpec` holding both raises "two axes named 'member' have different structures".
  `fields_from_dataset` makes the labeled form from a `member` coordinate, while
  `parameter_vector.pyens_grids` is documented with the sized form built by hand, so grids made
  the two ways cannot share a spec unless both are given the same `Axis` (`pyens_grids`
  accepts `axes_of(table)["member"]`; `fields_from_dataset` accepts `axes=`). Build each axis
  once and pass that object everywhere it is used. Equal axes zip, so two sources that both
  put a 0-based `member` coordinate on their ensemble dim (the SIPNET table, the drivers, the
  initial conditions) are paired member by member, silently, whenever their sizes match.

### pyEKI
- There is deliberately no log-likelihood helper (as of pyEKI PR #31).
  `pyeki.gauss.Gaussian(y, noise_cov)`, with `noise_cov` a `PSDLinOp`, scores a batch of
  predictions with `log_density(predictions)`, `(..., N) -> (...)`, by the symmetry of the
  density in point and mean. It equals
  `-pyeki.eki.misfits(y, predictions, noise_cov) - (logdet R + N log 2 pi) / 2` and requires
  `noise_cov` to support `whiten` and `logdet`. Build `noise_cov` with
  `DensePSD.from_matrix(R)`: `DensePSD(L)` takes the **lower** Cholesky factor and checks only
  that it is square and finite, so passing `R` or an upper factor gives a wrong density
  silently. Outside debug mode (`pyeki.linalg.set_debug_checks(True)` turns it on), a row
  holding a NaN or an inf scores NaN, and so does every row when `noise_cov` is singular or
  holds a NaN; the caller maps that to `-inf`.

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
