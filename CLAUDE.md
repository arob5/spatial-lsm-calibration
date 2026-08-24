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

## Repository layout

**Cleared 2026-08-20** — the ProbPipe/RWMH single-site demo was removed to start
fresh for real multi-site data. Only `sipnet_calibration/plotting.py` survives.
The new layout has not been written yet; do not assume a structure here.

Conventions retained from the previous layout (still the intended design):

- One directory per experiment under `experiments/`, with `config.py` as the
  single source of truth for that experiment (parameters, transforms, data
  paths, algorithm settings).
- Heavy computation lives in scripts, not notebooks. Notebooks are for
  exploration and plotting only, and load results from disk.
- Raw inputs are symlinked into `data/raw/` (gitignored) and never edited;
  an ingest pipeline converts them to a canonical format.

## Docstring conventions

These apply to all docstrings, and strictly to module-, class-, and
public-function-level ones.

**Write for the person calling the code.** Lead with what the thing is and how
to use it. Explain behaviour, arguments, return values, and errors — not the
reasoning that led to the implementation.

**Use clear, precise language and no unnecessary jargon.** Prefer a plain
description over a compressed technical phrase. Do not editorialize about the
design: sentences like "the split is load-bearing rather than cosmetic" state a
low-level design judgement and do not belong at the top of an API.

**Organize with sections.** Use standard headings — `Parameters`, `Returns`,
`Raises`, `Notes` — and tables when listing several classes or functions. A
reader should be able to skim the structure.

**Put design rationale in a `Notes` section, or leave it out.** Consequential
lower-level decisions are worth recording when they are non-obvious or easy to
undo by accident, but they go at the bottom under `Notes`, never in the
opening description.

**Scope each level distinctly; do not repeat yourself.**
- *Module*: what the module provides, an index of its contents, and any
  convention shared across everything in it.
- *Class*: what this class represents and its parameters. Do not restate
  module-level conventions.
- *Method/function*: what this call does, its arguments and return value. Do
  not restate class-level context.

**Keep docstrings self-contained.** Do not reference files outside the
repository — Obsidian logs, design docs, external notes. A reader with only
the source must be able to follow them. Cross-reference other modules and
classes within the repo freely.

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
