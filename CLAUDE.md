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
- **A tracked raw input is found from the repository, not from
  `conventions.data_root()`.** `data_root()` (and `$SIPNET_CALIBRATION_DATA`)
  says where the storage-backed part of `data/` is, which on the SCC or with
  the variable set is another tree; a tracked file is always in the checkout.
  The tests (`conftest.REPOSITORY`), the Natural Earth scripts and
  `split_site_pft_16class.py` follow this;
  the library's own `default_raw_dir()` resolvers for tracked directories
  still go through `data_root()` until the data-source cleanup (PR 5d).
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
- Drop the NEE csv's `ens_mean` column; never admit it to the NEE ensemble's
  member dim (`nee_member`, when NEE is ingested).
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

These are project-wide and apply to new code without being restated. They
follow `logs/2026-09-25_Repository Design Model.md` in the vault, the approved
design every change is measured against; a change that needs a convention not
written here adds it there and here first. The code is being brought into line
with it by a series of PRs: PR 1 (the foundation: shared constants, coercion,
file writing, site-table functions), PR 2 (batch dims), PR 3 (vocabulary
renames), PR 4 (contracts and vectors), then module cleanups. Where the code
does not follow a rule below yet, the rule says which PR changes it.

### Glossary

One concept, one word; one word, one concept. A word this glossary does not
list is either added here or not used for a project concept. The retired words
are still in the code until the PR that renames them (mostly PR 3).

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
| **timestep** (prose and identifiers) | one SIPNET step, `(timestep_start, time]`. pySIPNET's coordinates are `time_step_start`/`time_step_length` today; it is renaming them, and `conventions.TIMESTEP_START`/`TIMESTEP_LENGTH` then follow | "time step", `time_step` |
| **time label** | a value of the `time` coordinate | |
| **window** | the interval an observation's value covers, which model timesteps are reduced over (`pd.IntervalIndex`); on an observation's `time` as the coordinates `conventions.WINDOW_START`/`WINDOW_END`, built from the CF `time_bounds` variable the processed file stores. Their values are still `time_bounds_start`/`time_bounds_end` until PR 3 renames them `window_start`/`window_end` | "time bounds" for these coordinates |
| **cell** | **only the CF sense**: the interval or area one value represents (a calendar resampling cell, a grid cell, a raster cell, a map renderer's site cell) | an element of y, a `(sample, site)` pair, a CSV field, a figure slot |
| **record** | a data source's full time extent (a run's record, a site's driver record); also one raw CSV row | |

**Data sources and data.**

| Word | Meaning | Retires |
|---|---|---|
| **data source** | a producer's dataset, raw or processed (MODIS LAI, the ERA5 drivers, PEcAn's initial conditions, the site table) | "product" for this |
| **observation source** | a data source used as observational constraint data; one `ObservationSource` | "product" in the observation package |
| **processed file** | the file an ingest script writes under `data/processed/` | "processed product" |
| **upstream product** | the producer's own product name (MCD15A3H v061); the spec field and netCDF attribute `upstream_product` | the spec field and attribute `product` (renaming it needs a re-ingest) |
| **raw** | a data source as it arrives, never edited | |
| **constraint** | one of the five constraint data sources of `constraints.py` | |
| **observation** | one entry of **y**: one observed value of one observation source at one site (and time) | "cell", "observed cell" |
| **model output** | SIPNET's output as a labeled `xr.Dataset`, one run or a stack | `dataset` as a variable name for it |
| **run** | one SIPNET execution: one site, one sample, and one member of each driving data source | "cell", "slot", "pair" |

**Representations.**

| Word | Meaning | Retires |
|---|---|---|
| **field** | one `xr.DataArray` holding one variable under the field contract below | "canonical field", a Dataset called a field |
| **Fields** (a representation) | a vector's labeled form: an `xr.Dataset` of fields for the parameter vector, a `dict[str, Field]` for the observation vector | |
| **Flat** | a vector's unlabeled numeric form: one vector `(D,)`/`(N,)`, or a batch `(n_samples, D)`/`(n_samples, N)`, called "batched Flat" where the shape matters | "block" |
| **entry** | one position of a Flat vector | "column" (parameter vector), "cell" (observation vector) |
| **segment** | the contiguous entries of one site, or one piece, in Flat | "block" in `forward.py` |
| **SIPNET parameter fields** / `sipnet_parameter_fields` | the `xr.Dataset` of SIPNET parameter values over `(sample, site)` or `(site,)` | "SIPNET table", `table` for a Dataset |
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
| **batch shape** | TFP's term, only in TFP code, always written "TFP batch shape"; prior groups are "groups" | "batch member" for a group |

**Narrowed words.** *label*: an xarray coordinate label, text on a figure, or
"site labels" the data source. *kind*: only pySIPNET's variable kind. *source*:
only "data source" and its attributes (`source_file`, `source_column`).
*identity*: not used. *row*: a table row, or a row of batched Flat (a sample);
never a timestep or a site position. *grid*: `SITE_GRID`, a raster grid, or a
PyEns `Grid` (always "PyEns grid").

**Notation.** `J` samples, `D` the dimension of theta
(`ParameterVector.dimension`), `N` the dimension of y
(`ObservationVector.dimension`), `S` sites (`n_sites`), `K` observation sources;
`theta` Flat unconstrained calibration parameters, `y` Flat observations, `G`
the forward map (`ForwardModel`), `M_s` SIPNET at site s, `H_k` the observation
operator of source k, `T` the unconstrained-to-natural transform.
"Predictions" is always a Flat `(N,)`/`(J, N)`; its labeled form is
`predicted_fields`.

### Field contract

A **field** is one `xr.DataArray` holding one variable. It is a convention
checked by one validator, not a wrapper class: a wrapper would fight xarray's
`.sel`/`.resample`/`.quantile`, which are the operations this project needs.

- **dims**: zero or more batch dims, then at most one spatial dim, then `time`
  if present: `(*batch, space, time)`. A dim that is none of these is not
  allowed.
- **The spatial dim** is one of `site` (site ids of the site table in use);
  `point` (arbitrary locations); or a raster pair `lat`, `lon` (or projected
  `y`, `x`). These names are reserved (`conventions.SPATIAL_DIM_NAMES`) and are
  never batch dims. `site` and `point` carry `float64` `lon`/`lat` coordinates
  with the CF attributes of `conventions.LON_ATTRIBUTES`/`LAT_ATTRIBUTES`
  (`sites.site_locations` makes them); a scalar `site` coordinate marks a field
  of one site.
- **`site`**: `int32` site ids (`conventions.SITE_DTYPE`), unique.
- **`time`**: naive `datetime64[ns]`, strictly increasing, no `NaT`; pySIPNET's
  timestep coordinates on `time` alone when the field is model output or
  drivers (`conventions.TIME_COORD_NAMES`); the window coordinates on `time`
  when the values are attributed to windows (observations).
- **attrs**: `units` (validated by pySIPNET), `constituent` where the quantity
  has one, `long_name`; `kind` where pySIPNET's kind applies.
- A structural axis (variable, component, quantile, bounds, PFT class) is never
  a dim of a field: split it into a `dict` or `Dataset` of fields.
- **Batch dims.** A batch dim is every dim other than the spatial dim and
  `time` whose index coordinate holds integers (`fields.batch_dims` finds
  them); a dim with string or float labels, or none, is refused, which is what
  keeps `variable`, `quantile`, `pft` and `bounds` off a field. Two batch dims
  with the same name are the same index (they zip in PyEns and align in
  xarray); different names are different indices (they cross), so unrelated
  ensembles are kept apart by distinct names, and a deliberate pairing is
  spelled by giving two dims one name. Batch dims come first.
- **The batch dim made from batched Flat** is `sample` (`conventions.SAMPLE`)
  by default, overridable by `batch_dim=` on every function that creates one
  (`ParameterVector.fields` and `.sipnet_table`, `ObservationVector.fields`,
  `ForwardModel`, whose `sipnet_table`, `model_output`, `run_succeeded` and
  `failures` column all carry that one name), labeled `0..n_samples-1` in row
  order. A created batch dim may not take a spatial name, `time`, a
  site-labels name or a calibration parameter name.
- **A data source's own ensemble** is a batch dim named for the source:
  `initial_condition_member`, `driver_member` (and `nee_member` when NEE is
  ingested), with its 1-based file index beside it as
  `conventions.SOURCE_INDEX` (`source_index`). The tracked raw initial
  condition file keeps its own `member` dim, since raw data is never edited.
- **Labels** are `int64` (`conventions.BATCH_LABEL_DTYPE`), any distinct
  integers, with no cap on their number. `ForwardModel` alone requires its
  table's labels to be `0..J-1` in row order, because it places each run in
  the row its label names.
- **Flat has at most one batch dim.** A field with several is reduced, or
  stacked with `fields.stack_batch_dims(field, into=SAMPLE)`, which labels the
  stacked dim `0..n-1` and keeps each original dim's labels as a
  `<dim>_label` coordinate on it; `fields.unstack_batch_dims` reverses it.
- **A scalar coordinate is not a dim.** A field whose batch dim was selected
  away with `.isel(sample=k)` has no batch dim; its scalar label is metadata,
  and `flat` gives one vector. An observation refuses a batch dim, not a
  scalar batch label.

One validator, `fields.validate_field(field, *, message_name=None)`, checks
all of this. Every plotter calls it first, as `stack_batch_dims` and
`unstack_batch_dims` do; the time-alignment verbs, the observation class and
the parameter vector still check their own parts of it until PR 4 and the
module cleanups route them through it. Time labels may be in any
`datetime64` unit (pandas and xarray make microseconds), since numpy compares
them across units. **Model output** is an `xr.Dataset` of pySIPNET-named
variables on one shared time axis, each variable a field; one run's has a
scalar `site` and scalar batch labels (`fields.label_run(dataset, site=,
batch={"sample": 3})`), a stack has `site` and batch dims
(`fields.stack_model_outputs(runs, key_dims=("sample", "site"))`, keys in
`key_dims` order).

### Vector-like classes

`ParameterVector`, `ObservationVector` and their pieces (`CalibrationParameter`,
`FixedParameter`, the observation vector's per-source piece) follow one
convention (approved; **being implemented in PR 4**, except where noted):

| Aspect | Convention |
|---|---|
| Construction | `@dataclass(frozen=True, eq=False, kw_only=True)`; validation in `__post_init__` through one grouped check; nothing mutable reachable: mappings frozen (`conventions.FrozenMapping`, which pickles), arrays copied and read-only. `FixedParameter`'s value, `ParameterVector.site_labels` and its lon/lat are frozen already (PR 1); one grouped check in `__post_init__` is PR 4's |
| Pieces | `vector[name]`, `name in vector`, `iter(vector)` (piece names), `len(vector)` (number of pieces), `<piece>_names` |
| Size | `dimension` (D or N) |
| Entries | `index`: a `pd.MultiIndex` over the entries; `positions(**selectors) -> int64 array` on both |
| Sites | `sites` (ids, ascending, refused if unsorted on input), `site_table` on both |
| Selection | `select(*, <piece>_names=None, sites=None, ...)`: an unknown label raises `KeyError` (PR 4; `ObservationVector.select(sites=)` ignores an unknown site until then); the result keeps vector order whatever the request order; duplicates are refused (PR 1, through `validation.as_site_ids`); `restrict_to_sites(sites)` is the intersecting form |
| Representations | `flat(fields) -> Flat`, `fields(flat_values, *, batch_dim=SAMPLE) -> Fields` |
| Flat's array type | JAX everywhere: both vectors and `ForwardModel` return `jax.Array` Flat and accept any array-like; internals that fill arrays in place work in NumPy and convert on return. 64-bit JAX is on for the whole package (PR 1) |
| Description | `describe()`: one row per piece; `index`: one row per entry; `__repr__` one summary line |
| Directions | where a vector and its pieces list SIPNET parameter names, the name says which way: `sipnet_parameter_names_written`, `sipnet_parameter_names_read` |
| Section comments | `# ── identity ──`, `# ── selection ──`, `# ── representations ──`, `# ── evaluation ──` |

`ForwardModel` is a regular class with read-only properties (PR 4; its
attributes are still plain and reassignable); `ForwardEvaluation` is
`frozen, eq=False` (PR 1). No base class is shared by
the vectors: they share an interface, not an implementation, and their shared
coercion lives in `validation.py`.

### Where shared things live

- **`conventions.py`** holds every name constant two modules share (dims,
  coordinates, `SOURCE_INDEX`, the `site_id` column, the `time_bounds`
  variable, the attributes of `site`/`lon`/`lat`/`sample` and of a data
  source's member dim, `SITE_DTYPE`, `BATCH_LABEL_DTYPE`, `NAME_PATTERN`,
  `STALE_TIME_ATTRIBUTE_NAMES`, `CF_CONVENTIONS`, `DATA_ROOT_ENV_VAR`,
  `data_root()`), and `FrozenMapping`, the one read-only mapping type: a
  `dict` subclass whose mutators (a second `__init__` included) raise, so
  pandas and `json` read it as a dict, and which pickles and hashes. Every
  module-level mapping constant of the package is one (the scripts' own
  tables are not the package's), and one is handed to xarray as it is, since
  xarray copies attrs; pandas' `agg`, which refills the mapping it is given,
  takes a `dict(...)` copy. A module imports these; it never defines its own
  copy and never re-exports one. A name only one module uses lives in that
  module: `DRIVER_MEMBER` in `drivers`, `INITIAL_CONDITION_MEMBER` and
  `RAW_MEMBER` in `initial_conditions.names`.
- **`validation.py`** holds the argument coercion two modules need, each
  `as_<thing>(value, *, message_name) -> thing`: `as_site_ids`, `as_site_id`,
  `as_integer`, `as_positive_integer`, `as_bounded_integer`,
  `as_positive_integers`, `as_batched_flat` (with `is_one_vector`),
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
  - **a site the data lacks** is a `KeyError` (the vectors' `select` from
    PR 4, which aligns their selection: `ObservationVector.select(sites=)`
    still ignores a site it does not observe).
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
  `site`, `lon` and `lat` coordinates a product's `site` dim carries), the
  site-table checks (one invariant each, grouped as
  `check_site_table_locates_the_sites` and
  `check_site_table_is_keyed_on_site_ids`), the pool checks a raw data
  source's sites are held to (`check_site_table_lists_the_sites`,
  `check_sites_are_the_site_table`), and `N_SITES`. No lookup is written as a
  hand `set_index("site_id")`; `site_lookup` is the keyed form.
- **`tests/conftest.py`** holds every fixture or builder more than one test
  file uses (some Niwot stacks and observation builders are still per file,
  until the module cleanups, PR 5).
- The package's `__init__.py` documents every module and the direction the
  dependencies run, re-exports nothing, and turns on 64-bit JAX, its one
  import-time side effect.

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
  field's `name` is a processed name, which is what makes `validate_field()`
  usable against anything an adapter produces.

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
  glossary's words. The parameter vector's Flat is `theta`, its Fields is
  `<thing>_fields`, its SIPNET parameter fields are `sipnet_parameter_fields`
  (still `sipnet_table` in the code until PR 3); the observation vector's Flat
  is `y`. A pySIPNET object is named for its class (`sipnet_result`,
  `sipnet_output`, `sipnet_parameters`); the labeled xarray of a run's output
  is `model_output`. Nobody should have to ask whether a value is a
  `ParameterVector`, a `CalibrationParameter` or a SIPNET parameter.
- **`sipnet_` prefixes anything in pySIPNET's vocabulary**: names, values,
  objects. The bare word or `calibration_` is this repository's vocabulary.
  `pysipnet_` is not used: this repository reaches SIPNET only through
  pySIPNET, so a second prefix would have no second referent.
- **An `xr.Dataset` when the variables share one grid, a
  `dict[str, DataArray]` when they do not.** One run's or one stack's model
  output shares a time axis and is a Dataset; the constraint data sources have
  three time structures and are a dict. A dict is named for what it holds and
  its key (`observed_values`, keyed by source name; `run_outputs_by_sample_site`
  where the key order matters).
- **A field is one thing**, as the field contract above defines it. The word
  "canonical" is not used with it; `fields.py` holds the generic operations on
  fields and nothing else.
- **A class whose call is its whole interface may be named as an imperative
  verb**, for what the call does (`SelectTimestep`, `ComputeLeafAreaIndex`).
  This is not a rule for every callable class: a protocol, a record, or an
  object with an interface of its own beyond the call is a noun
  (`ObservationOperator`, `ForwardModel`, a model object).
- **Private helpers are named for what they do or what they return**: a verb
  phrase (`_sort_by_site_and_time`, `_drop_padding_rows`) or a noun phrase
  (`_read_only_copy`, `_observation_restricted_to`). Never a bare participle
  (`_selected`, `_frozen`, `_aggregated`) and never a name that hides a side
  effect (a `_with_...` that drops, a `_sort_...` that partitions). The
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
  `at`, not `data`, not `x`. A site table is `site_table=` everywhere (PR 3
  renames the `sites=` and `sites_table=` keywords that still take one).

Existing code is brought into line by the PRs of the design model, with the
tests renamed alongside; `logs/2026-09-25_Repository Design Model.md` in the
vault lists what changes in which.

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

### Checks, errors and messages

These apply to the library and the scripts alike.

- **A check is a public-named `check_<subject>_<predicate>` function** with a
  one-line docstring, saying what it checks: `check_covariances_were_diagonal`,
  `check_site_table_lists_each_site_once`. Not `validate`, not an inline
  `assert` buried in a transformation. It checks one invariant, and raises or
  returns `None`; it never returns a value. A group of checks always called
  together becomes one check that calls them in order, whose docstring's
  first line says what the group checks and whose rest names the checks it
  runs (`check_site_table_locates_the_sites`). Checks used from another
  module are in `__all__`.
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
  pySIPNET writes one whose two edges are `time_step_start` and `time`. Where CF has no
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
`#$ -P dietzelab` with `#$ -l buyin` is the queue. For PyEns jobs,
`compute.scc_backend` writes those directives and a `-v` exporting the
cache, binary and data variables; its module docstring says why `TMPDIR` is
left to Grid Engine.

## Repository layout

The layout below is the **agreed target**, specified in
`logs/2026-08-28_Plotting Design Spec.md` in the Obsidian vault. The src-layout
reorg has landed, so the paths below are the real ones; `sites.py`,
`constraints.py`, `initial_conditions/`, `drivers.py`, `projection.py`,
`parameter_vector.py`, `site_labels.py`, `forward.py`, `compute.py` and the
`observation/` package are implemented, `fields.py` has the model-output
adapters, the plotting package has series, maps and grids, and the other
modules carry the contract each is to satisfy.
`initial_conditions` is a package rather than a module: it spans several
artifacts, and giving each its own file keeps that artifact's schema, writer,
reader and checks together.

```
pyproject.toml            # name = "sipnet-calibration"; src layout
src/sipnet_calibration/
  __init__.py             # the module map and the dependency direction; no
                          # re-exports; turns on 64-bit JAX
  sites.py                # SITE_GRID + grid conversions, load_sites(),
                          # select_sites(ids=, bbox=, where=, sample=, seed=),
                          # EXTENTS (named lon/lat boxes), N_SITES; site_lookup(),
                          # site_locations(), site_coordinates(), the site-table
                          # and pool checks
  projection.py           # SITE_PROJECTION (LAEA 50 N, 100 W) over pyproj:
                          # forward(), projected_bounds(), factors()
  projections/            # the stored definition, generated from the dataclass
  constraints.py          # ConstraintSpec + CONSTRAINTS, one per raw file;
                          # read_raw(), build_constraint(), load_constraint(),
                          # constraint_fields() -> one field per product
  conventions.py          # every shared name constant: SITE, TIME, SAMPLE,
                          # the reserved spatial names, TIMESTEP_START/LENGTH,
                          # WINDOW_START/END, TIME_BOUNDS, SITE_ID, SOURCE_INDEX;
                          # the attributes of site/lon/lat/sample and a source
                          # member; SITE_DTYPE, BATCH_LABEL_DTYPE, NAME_PATTERN;
                          # CF_CONVENTIONS and data_root(); FrozenMapping
  validation.py           # argument coercion: as_site_ids, as_site_id,
                          # as_integer, as_positive_integer, as_bounded_integer,
                          # as_positive_integers, as_batched_flat, as_bbox,
                          # as_names, as_frozen_mapping; its checks; truncated(),
                          # range_summary()
  io.py                   # write_checked() and write_checked_together() (the
                          # .partial protocol), file_md5(), utc_timestamp()
  initial_conditions/     # one module per artifact; __init__ re-exports them all
    __init__.py           # curated exports + the product's data model
    names.py              # INITIAL_CONDITION_MEMBER/RAW_MEMBER, the two file
                          # names, the path helpers
    source_files.py       # SOURCE (the PEcAn file format), read_source_file()
    specs.py              # InitialConditionSpec + INITIAL_CONDITIONS
    raw.py                # build_raw(), raw_encoding(), read_raw()
    processed.py          # build_initial_conditions(), load_initial_conditions(),
                          # netcdf_encoding(), initial_condition_fields()
    sipnet_parameters.py  # to_sipnet_initial_conditions() and its table form
  drivers.py              # load_drivers(): raw .clim files read by pySIPNET's
                          # ClimateDrivers, stacked into (driver_member, site, time) on
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
                          # fields on (sample, site), attrs["space"]), and
                          # sipnet_table() -> sipnet_overrides(), and PyEns grids
                          # through pyens.xarray.fields_from_dataset;
                          # example_parameter_vector()
  forward.py              # ForwardModel: theta (J, D) -> predictions (J, N),
                          # SIPNET once per (sample, site) through PyEns, the
                          # observation operators applied on the worker;
                          # ForwardEvaluation; the failure split
  compute.py              # scc_backend(): the SCC GridEngineBackend preset
  fields.py               # the field contract: validate_field(), batch_dims(),
                          # stack_batch_dims()/unstack_batch_dims(); label_run()
                          # (a run's Dataset with site/lon/lat and batch labels:
                          # the model_output the observation operators read),
                          # from_sipnet_output(), stack_sipnet_outputs() over
                          # SIPNETOutput.select, stack_model_outputs() (runs'
                          # Datasets to one on (*batch, site, time)),
                          # resolve_output_variable_names(), field_label()
  observation/            # the observation side of the inverse problem
    __init__.py           # curated exports
    time_alignment.py     # aggregate_time, reduce_windows, select_timestep_at,
                          # windows_from_time_bounds, run_window, the counts;
                          # the verb a caller applies before plotting
    operators.py          # ObservationOperator protocol; SelectTimestep,
                          # ReduceOverTimeBounds, ReduceOverRun,
                          # ComputeLeafAreaIndex; DEFAULT_OBS_OPS;
                          # check_operator and the contract's checks the
                          # vector shares
    vector.py             # Observation, ObservationVector: index (site,
                          # product, time), y, flat()/fields(), positions(),
                          # predict()
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
  raw_sources/            # NOT the pipeline: code that *makes* a tracked raw
                          # input, run once. SCC-only except the Natural
                          # Earth download.
experiments/<task>/       # config.py (source of truth) + plots.py (L4 reports)
data/raw/                 # never edited; raw/sites/, raw/constraints/,
                          # raw/initial_conditions/, raw/site_labels/,
                          # raw/covariates/ and raw/natural_earth/ are tracked
data/processed/           # ingest output == the plotting input; untracked;
                          # constraints/<name>.nc is one CF-1.11 netCDF per constraint
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
- Every plotter calls `fields.validate_field` first and branches on
  **presence of a batch dim**, found with `fields.batch_dims`, never on a mode
  keyword. `plot_time_series` fans over batch dims of any name and refuses a
  `site` dim (select a site, or facet with `plot_by_site`); `plot_map` refuses
  a batch dim with advice naming it (`plot_map_by`,
  `plot_map_quantiles(batch_dim=)`, `summarize_batch(field, stat,
  batch_dim=)`). No plotter types a dim name: `SITE`, `TIME`, `LON`, `LAT`
  and `SAMPLE` come from `conventions`.
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
  is that operation for a field: a field carrying pySIPNET's interval
  coordinates goes through `resample` itself, which keeps batch and `site`
  dims (pySIPNET PR #49), and one without them, such as an observation, is
  combined on the same calendar cells here. It adds one thing: with no
  `how` it takes **the method that leaves the variable the kind it already
  is**, read off pySIPNET's `RESAMPLED_KIND` rather than written down. A
  total sums, a step mean or a rate means, a pool or a running total takes
  its last value.
  SIPNET's `net_ecosystem_exchange` is `g m-2` of C per timestep, so 3-hourly
  to daily is a **sum**, and a mean is wrong by 8x while looking plausible;
  the default is there so that omission cannot reach that error, and `how=` is
  for asking deliberately for something else, such as the time-weighted mean
  of a pool. An invalid pair is refused in pySIPNET's own words.
- **An observation operator is a callable checked at the boundary, not a
  grammar.** `observation.ObservationOperator` is a protocol:
  `operator(model_output: xr.Dataset, observed_values: xr.DataArray, *,
  sipnet_parameters=None) -> xr.DataArray` on the observation's own
  `(site[, time])` grid, declaring `output_variable_names` and
  `sipnet_parameter_names`, and pointwise in `site` and in every batch dim so
  it can run on a worker (`check_operator` slices each to test that). A
  SIPNET table's every dim is selected at the model output's labels or
  refused, so a table dim is never broadcast into a run. It returns whatever units
  it produces, with `units`/`constituent` attrs; `ObservationVector.predict`
  converts through `pysipnet.units.convert_dataarray_units` and refuses a
  wrong dimension, grid or site set through the checks `check_operator` also
  applies (`check_model_output_carries_what_is_read`,
  `check_result_is_on_the_observation_grid`; `Observation` applies
  `check_operator_declares_names` on construction), and a NaN where the run
  succeeded. The verbs it is written with carry pySIPNET's attributes: its
  arithmetic is `pysipnet.arithmetic` (`divide_with_units`, `step_length`,
  ...), a SIPNET
  parameter is labeled by `pysipnet.parameters.model.parameter_dataarray`,
  and the time-alignment verbs are `select_timestep_at` (the model step
  whose `(time_step_start, time]` contains the label), `reduce_windows` (a
  step belongs to the window its end falls in; means weighted by step
  length; a gap makes the window NaN) and `windows_from_time_bounds`
  (`observation.time_alignment`). `ReduceOverTimeBounds` refuses a window
  reaching a step or more beyond the model record, so a partial year never
  passes for a year (`check_run_spans_the_windows`). The library binds a
  default operator only where the construction is established from a primary
  source (`DEFAULT_OBS_OPS`, today MODIS LAI as SIPNET's own
  `plantLeafC / leafCSpWt`, `sipnet.c`); which operator reads a product is a
  modeling decision an experiment writes in `config.py`.
- **The forward model is one class over existing pieces.**
  `forward.ForwardModel(model, parameter_vector, climate=, backend=,
  observation_vector=)` is pyEKI's `(J, D) -> (J, N)`; its module docstring
  says how the pieces compose. The rules a session can get wrong: the
  observation operators run **on the worker**, each run receiving only its
  site's slice of the observation vector and returning that slice's Flat,
  which the calling process writes at `positions(site=)` (right because the
  vector is site-major, which `__init__` checks per site); a run at a site no
  product observes returns nothing, so a product costs nothing at the sites
  it does not observe. The SIPNET table a `to_sipnet_table` hook returns must
  be on exactly `(batch_dim, site)` (`sample` unless the model's `batch_dim=`
  says otherwise) with every variable on both, labels `0` to `J - 1` in
  `theta`'s row order, sites in the parameter vector's, and each
  variable named by pySIPNET's flat parameter name (an alias such as `aMax`
  passes pySIPNET's lookup but `SIPNETModel` refuses it on every run). A run
  that fails at its parameters (`SIPNETRunError`, pydantic's
  `ValidationError`, a timeout, or a non-finite value in a read variable,
  `ModelOutputNotFiniteError`; across a process boundary matched on PyEns's
  fully qualified `RemoteError.type_name`) makes the **whole sample's** row
  NaN, and anything else a worker returns is the machinery failing and is
  raised with the collected runs on the error's `evaluation`, as is a
  prior-predictive batch in which every run failed. The prior-predictive
  output is stacked by `fields.stack_model_outputs`, so it carries no
  `time_bounds` or SIPNET row labels; `freq=` is for that path only, and
  aggregates each run's variables one at a time with
  `observation.aggregate_time` by the method that keeps its kind, as a
  predictive-check figure does, the Dataset gaining pySIPNET's
  `resampling_frequency` and `time_step_length_source`. Under any backend
  but `SequentialBackend` the drivers must be file-backed.
  `compute.scc_backend` is the SCC preset.
- **The observation vector is site-major.** `ObservationVector.index` is a
  `(site, product, time)` MultiIndex over the observed (not-NaN) cells,
  sites ascending, then products in declaration order, then times, with
  `NaT` for a static product; `y` is Flat in that order, `flat()`/`fields()`
  convert, and `positions()` finds a site's or a product's block. No
  standard deviation, covariance or likelihood lives in the package; the
  inference layer builds those from `y`, `index` and `positions`. A batch
  dim on an observation is refused: the experiment reduces an observation
  ensemble before it enters; a scalar batch label is metadata and is kept. An
  `Observation` keeps only the sites and time labels it observes, so its
  operator never reads the model elsewhere, and a `select(sites=...)` slice's
  operators read the model only inside the kept sites' records.
- **An annual constraint's array carries its `time_bounds`** as the 1-D
  coordinates `time_bounds_start`/`time_bounds_end` on `time`
  (`constraint_fields` adds them; the names are `conventions.WINDOW_START`
  and `WINDOW_END`, whose values PR 3 renames `window_*`), which is what
  `ReduceOverTimeBounds` reads; a dated or static product documents no
  interval.
- **Model and driver fields carry pySIPNET's names, units, kinds and time axis
  unchanged.** `fields.from_sipnet_output` adds `site`, batch labels and
  `lon`/`lat` to a run's output; `drivers.driver_fields` does the same for the
  drivers, read through `ClimateDrivers`. The registry names are already
  `lower_case_with_underscores`, so they are the processed names. Both keep
  `conventions.TIME_COORD_NAMES`, pySIPNET's axis: `time` at the step end, with
  `time_step_start` beside it, so the interval a value covers is
  `(time_step_start, time]`, whose two edges are the pair pySIPNET writes as
  its CF `time_bounds` variable, and a run's output and the drivers it ran on share one axis by
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
  by another name. `plot_map` refuses a batch or `time` dim rather than
  reducing it: use `summarize_batch`, `plot_map_by`, `plot_map_quantiles` or
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
- `site` is the integer site id of the site table in use (`int32`; the
  shared pool's ids are never renumbered); `ameriflux_site_id` is a non-dimension coord on
  `site`. PFT is **not** site metadata and is not a column of the site table:
  which site labels to use is an experimental choice, so site labels are their
  own product at `data/processed/site_labels/<name>.csv`, keyed on `site_id`,
  and a caller joins one on before selecting. A batch label is an `int64`,
  meaningful only within its dim's name. See the Data section above for the
  rules these imply.

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
- **`pysipnet.arithmetic` combines labeled arrays** (pySIPNET PR #48): `multiply_with_units`,
  `divide_with_units`, `add_with_units`, `subtract_with_units` return a `DataArray` whose
  `units`, `constituent` and `kind` (with `time_reference`, `cell_methods`) are true of the
  result, named `None`, with a `derivation` attr naming the operands. At most one operand of a
  product or quotient has a kind, never the denominator; a `timestep_total` over a time is a
  `daily_rate` and back (`KIND_AFTER_TIME_POWER` in `pysipnet.variables`); every other change of
  time dimension is refused. Index coordinates must match exactly, and conflicting non-index
  coordinates are refused. `step_length(data, units="d")` is the `time_step_length` coordinate
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
  `float32`, which the vector refuses; importing `sipnet_calibration` turns x64 on for the
  process, the package's one import-time side effect.
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
  round-trip. Send PyEns workers plain data (a SIPNET table through
  `pyens.xarray.fields_from_dataset`), never a `ParameterVector`.
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
  the int32 `site` and int64 `sample` coordinates become plain `int` labels, a non-dimension
  coordinate such as `pft` is ignored, and a hand-built label-keyed
  `Grid({site_id: drivers}, along=Axis("site", labels=[...]))` zips with the result.
- **The pairing rule is the dim name.** PyEns makes one axis per dim, named for it: two
  batch dims of **one name zip** (paired label by label) and two of **different names cross**
  (every combination, multiplying the runs). So the SIPNET table's `sample`, the drivers'
  `driver_member` and the initial conditions' `initial_condition_member` cross, and pairing two
  ensembles deliberately is spelled by giving their dims one name. Same-named axes must be equal
  or PyEns raises: `Axis("sample", size=J)` is not equal to `Axis("sample", labels=[0, ...,
  J-1])` ("two axes named 'sample' have different structures"). `fields_from_dataset` makes the
  labeled form from a coordinate, so a `Grid` built by hand beside it must use an equal `Axis`;
  passing the same object is simplest (`fields_from_dataset` accepts `axes=`; `ForwardModel`
  builds its site axis once and passes it to every grid). `tests/test_fields.py` pins both
  halves of the rule against PyEns.

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
