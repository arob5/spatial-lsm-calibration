# CLAUDE.md — spatial-lsm-calibration

## Project purpose

This repository develops and applies **scalable Bayesian algorithms for parameter calibration of the SIPNET land-surface model**, with an emphasis on multi-site inference that exploits spatial structure (going beyond plant functional types). It is a research codebase, not a package — the deliverables are calibrated parameter ensembles, diagnostic outputs, and reusable inference machinery.

The long-term vision:
- Many sites, many runs, potentially many algorithms
- Real flux-tower / ERA5 data (symlinked from storage)
- Spatial hierarchy: partial-pooling priors or GP-based spatial priors over sites
- Full reproducibility: every run's configuration is one source of truth

## Companion packages (read-only — do not edit)

| Package | Path | Role |
|---------|------|------|
| `pySIPNET` | `../pySIPNET` | SIPNET model interface; `SIPNETModel(**overrides)` |
| `ProbPipe` | `../prob-pipe` (also on PyPI) | **Not currently a dependency** — API in flux; planned migration target for inference. See below. |
| `PyEns` | `../PyEns` | Parallel ensemble execution via `ProcessPoolExecutor` |

Install them as editable locals via `uv sync` (see `pyproject.toml`). Never modify their source.

## Data

**`data/README.md` is the authority on data formats, provenance, units, and the
open questions.** Read it before writing ingest or adapter code, and do not
duplicate its content here — add data facts there instead.

Facts specific to this working copy, which the README deliberately does not carry:

- **Only a subset of `data/raw/` is present locally.** Drivers and initial
  conditions exist for site 1, member 1 only; the NEE csv and the AGB/LAI
  `.Rdata` files are complete. The full dataset lives on Boston University's SCC.
  Anything that needs to hold across all 8000 sites cannot be verified here.
- The local files are real copies, not symlinks. On SCC they should be symlinks.
- R is available on this machine (`Rscript`), which is how the `.Rdata` files can
  be inspected; `pyreadr` is not installed and would not handle their nesting.
- Neither `pyproj` nor any R spatial package is installable here, so CRS and
  projection definitions cannot be validated locally (issue #4).

Operational rules that follow from the data and are easy to get wrong in code:

- Open the IC netCDFs with `decode_times=False` (README note 5).
- Drop the NEE csv's `ens_mean` column; never admit it to the `member` dim.
- Never renumber the 1-8000 site ids; they are a shared key with collaborators.
- The site table is `data/raw/sites/pts.*` (tracked) and, after ingest,
  `data/processed/sites/sites.csv`. There is no other site source.
- Do not assume rectangular coverage: NEE is ~55% missing over site x time, and
  the AGB/LAI constraints are ragged over site x year x variable.

## Repository layout

The layout below is the **agreed target**, specified in
`logs/2026-08-28_Plotting Design Spec.md` in the Obsidian vault. As of
2026-08-28 the reorg is in progress: the repo still has a flat
`sipnet_calibration/` package containing only the superseded `plotting.py`.
Build new code at the target paths; do not extend the old ones.

```
pyproject.toml            # name = "sipnet-calibration"; src layout
src/sipnet_calibration/
  sites.py                # site table + select_sites(pft=, bbox=, has_nee=, ids=, sample=)
  fields.py               # canonical field convention, validate_field(), adapters
  obs_ops.py              # aggregate_time, sipnet_time_index — shared with the likelihood
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
data/raw/                 # symlinked, never edited
data/processed/           # ingest output == canonical plotting input
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
  used throughout the project.

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
  (sum); carbon pools and `AbvGrndWood`/`LAI` are stocks (instantaneous).
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
  The source CRS of the site coordinates is recorded in `data/site_crs.json`.
- `site` is the integer 1-8000; Ameriflux `Site_ID` and `pft` are non-dimension
  coords on `site`. `member` is a 0-based integer, meaningful only within one
  source. See the Data section above for the rules these imply.

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
