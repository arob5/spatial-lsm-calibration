# Spatial Parameter Calibration for Land Surface Models (LSMs)

Scalable Bayesian algorithms for parameter calibration of the **SIPNET**
land-surface model, with an emphasis on multi-site inference that exploits
spatial structure rather than treating plant functional types as the only source
of pooling.

This is a **research codebase, not a library**. The deliverables are calibrated
parameter ensembles, diagnostic outputs, and reusable inference machinery — not
a published package. Interfaces change when the science requires it.

## Status

Early. The inference substrate lives in a separate package (`pyEKI`, below) and
the first multi-site calibration run (`test1`) is not yet configured. What
exists here today:

- the `src/sipnet_calibration/` module skeleton, with contract docstrings and
  no implementation yet;
- site metadata for the 8000-site pool (`data/site_ids.csv`) and the Ameriflux
  ID map (`data/site_id_map.csv`);
- a single site's drivers and initial conditions pulled down locally as a
  format reference.

Design records — the model definition, notation, algorithm design, and the
plotting specification — live in an Obsidian vault outside this repository, not
in the code. `CLAUDE.md` documents code conventions only.

## Scope of the problem

| | |
|---|---|
| Sites | 8000 irregular points, 7–82° N and 178° W–20° W (~3640 inside CONUS) |
| Period | 2012–2024 |
| Drivers | ERA5, 3-hourly, ensemble |
| Initial conditions | per-site, per-member netCDF |
| Constraint data | NEE (3-hourly, 25-member, 209 Ameriflux sites of which 165 map to site ids), AGB, LAI, soil C and moisture (annual) |

Every input arrives in ensemble form. Note the sites are **scattered points, not
a grid**, and the extent is North America rather than CONUS — assumptions to the
contrary are wrong.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The interpreter is pinned to 3.14 in
`.python-version`; `requires-python` is only a floor, so use the pin.

```bash
uv sync
```

Two companion packages are installed as editable locals from sibling
directories, so they must be checked out alongside this repository:

| Package | Expected path | Role |
|---|---|---|
| [`pySIPNET`](https://github.com/TARPS-group/pySIPNET) | `../pySIPNET` | SIPNET model interface — `SIPNETModel(**overrides)` |
| `PyEns` | `../PyEns` | parallel ensemble execution |

Two further packages are related but **not currently dependencies**:

- **`pyEKI`** — the ensemble-Kalman inference substrate (structured linear
  operators, Gaussian conditioning, EKI). Developed independently; not yet wired
  into this repo's dependency list.
- **`ProbPipe`** — a planned migration target for inference, removed as a
  dependency on 2026-08-20 because its API is in flux.

### Known setup caveat

`cartopy` is required by the spatial plotting design but is **commented out of
`pyproject.toml`**: it has no installable wheel on macOS 12 arm64, and the
blocker is the operating system rather than the Python version. `uv sync` is
clean without it, but `plotting/maps.py` is unimplemented pending the decision
in [#4](https://github.com/arob5/spatial-lsm-calibration/issues/4). On Linux
(BU's SCC) cartopy installs normally.

## Layout

```
src/sipnet_calibration/
  sites.py                # site table + select_sites(pft=, bbox=, has_nee=, ids=, sample=)
  fields.py               # canonical field convention, validate_field(), adapters
  obs_ops.py              # aggregate_time, sipnet_time_index — shared with the likelihood
  plotting/               # style, registry, primitives, series, maps, facet, diagnostics
scripts/                  # ingest: data/raw/ -> data/processed/
experiments/<task>/       # config.py (source of truth) + plots.py (report figures)
data/raw/                 # symlinked from storage, never edited
data/processed/           # ingest output == the canonical format used throughout
tests/
```

Data formats, provenance, and the coordinate reference system are documented in
[`data/README.md`](data/README.md) and [`data/site_crs.json`](data/site_crs.json).

## Conventions

- **One directory per experiment** under `experiments/`, with `config.py` as the
  single source of truth for that experiment: parameters, transforms, data
  paths, algorithm settings.
- **Heavy computation lives in scripts, not notebooks.** Notebooks are for
  exploration and plotting, and load results from disk.
- **Raw inputs are never edited.** `data/raw/` is symlinked from storage; ingest
  scripts convert it to `data/processed/`, whose format *is* the canonical
  format used by the rest of the project — including as the input format for
  plotting.
- **Calibration runs are tagged** `c_<task_id>_<run_name>_<run_index>`, where
  `c` marks a calibration run, `<task_id>` names the calibration task,
  `<run_name>` a run within it, and `<run_index>` increments on re-runs.

### Plotting

The plotting suite is layered so that data provenance and display style stay
independent — adapters map each data source to one canonical form, and plotters
consume only that form. The rules are summarised in `CLAUDE.md` and specified in
full in the vault. The two that most affect how it is used:

- A **canonical field** is an `xarray.DataArray` with dims a *subset* of
  `(member, site, time)` and `lon`/`lat` as non-dimension coords on `site`.
  Plotters branch on presence of the `member` dim, so the same function serves a
  single deterministic run and a posterior predictive ensemble.
- **Temporal aggregation is a verb the caller applies**, not a plotter keyword,
  and it lives in `obs_ops.py` shared with the observation operator — so a
  predictive-check figure cannot disagree with what the likelihood consumed.

## Running notebooks

Use the project venv's Jupyter directly, **not** `uv run jupyter`:

```bash
.venv/bin/jupyter lab
```

To execute headlessly:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --output out.ipynb in.ipynb
```

## Open issues

Data and environment problems currently tracked:

- [#3](https://github.com/arob5/spatial-lsm-calibration/issues/3) — the initial-condition
  netCDFs carry an unsubstituted `[year]` template in their time units, so they
  cannot be opened with CF decoding enabled. Adapters must use
  `decode_times=False`.
- [#4](https://github.com/arob5/spatial-lsm-calibration/issues/4) — no installable
  projection library on the development workstation; the spatial plotting layer
  is blocked on choosing an approach.
