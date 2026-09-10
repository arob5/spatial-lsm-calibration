# Spatial Parameter Calibration for Land Surface Models (LSMs)

Scalable Bayesian algorithms for parameter calibration of the **SIPNET**
land surface model (LSM), with an emphasis on multi-site inference that exploits
spatial structure rather than treating plant functional types as the only source
of spatial variability.

This is a research codebase, not a library. 

## Getting set up

Requires [uv](https://docs.astral.sh/uv/). One piece of the data ingest step requires
`Rscript` as well.

**1. Clone this repository.** Nothing else needs cloning: the three companion
packages are installed from git, and are described under
[Companion packages](#companion-packages) below.

```bash
git clone https://github.com/arob5/spatial-lsm-calibration.git
cd spatial-lsm-calibration
```

**2. Sync the environment.** This creates `.venv` from `uv.lock`, using the
interpreter pinned in `.python-version` (3.14). `requires-python` is only a
floor, so use the pin.

```bash
uv sync
```

**3. Activate the virtual environment.**

```bash
source .venv/bin/activate
```

All code should be run from within this virtual environment. Running 
`which python` should print a path ending in `.venv/bin/python`. Alternatively,
`uv run <command>` can be utilized for a one-off command without activating the venv.
See [Running notebooks](#running-notebooks) for more details regarding running Jupyter
notebooks.

**4. Check the environment works.**

```bash
uv run pytest
```

### Companion packages

Three packages are developed alongside this project and are dependencies of it:

| Package | Role |
|---|---|
| [`pySIPNET`](https://github.com/TARPS-group/pySIPNET) | the SIPNET model interface |
| [`PyEns`](https://github.com/arob5/PyEns) | running ensembles |
| [`pyEKI`](https://github.com/TARPS-group/pyEKI) | the ensemble Kalman inference substrate |

`[tool.uv.sources]` in `pyproject.toml` tracks the `main` branch of each, and
`uv.lock` records the **exact commit** resolved from it. So `uv sync` installs
the same three commits for everyone, and none of them moves until someone
upgrades it deliberately. Ordinary use needs no local checkout of any of them.

#### Upgrading a companion package

New work on `main` in one of these repositories does **not** reach this project
until the lock is refreshed. To take it:

```bash
uv lock --upgrade-package pysipnet
uv sync
```

The distribution names are `pysipnet`, `pyens` and `pyeki`; name several in one
command to upgrade them together. `uv lock --upgrade` upgrades everything
including the third-party dependencies, which is usually not what you want
here.

The only file that changes is `uv.lock`, and its diff shows which commit each
package moved to. **Commit that change**, since it is the record of which
version of each package a calibration run used.

#### Developing a companion package

To work on one of them and have this project pick up the edits immediately,
clone it anywhere and overlay an editable install on top of the synced
environment:

```bash
git clone https://github.com/TARPS-group/pySIPNET.git ../pySIPNET
uv pip install -e ../pySIPNET
```

Note that **any later `uv sync` silently replaces the overlay** with the
commit pinned in `uv.lock` — including `uv sync --inexact`, and including the
sync that another step of some workflow happens to run. Nothing warns you; the
symptom is that your edits stop having any effect. `uv pip show pysipnet` says
which one is installed: an `Editable project location` line means the local
checkout, and no such line means the pinned commit. Re-run the
`uv pip install -e` after any sync.

Once the work is pushed to `main`, upgrade as above and drop the overlay.

## Running the data processing

A one-time step per checkout. Ingest converts `data/raw/` into
`data/processed/`, whose format **is** the canonical format the rest of the
project reads.

It requires the raw data to be present under `data/raw/` in the layout
[`data/README.md`](data/README.md) specifies. **The data for this project is
housed on Boston University's Shared Computing Cluster (SCC).** A fresh clone
has almost none of it — only the site shapefile and the Ameriflux identifier map
are tracked — so this section is mostly about running on the SCC. Step 1 is the
exception and runs anywhere.

**The order below is load-bearing**, because each step reads what an earlier one
wrote. A single top-level helper that runs the whole sequence is wanted
eventually but does not exist yet, so for now it is these calls, in this order.

**1. The site table.** Writes `data/processed/sites/sites.csv`.

```bash
python scripts/ingest_sites.py
```

This must run first: `ingest_constraints.py` and `ingest_ic.py` both read that
file for the site axis and the `lon`/`lat` coordinates.

**2. Flatten the constraint `.Rdata` files.** Writes a long CSV and a JSON
manifest of what R checked.

```bash
Rscript scripts/export_constraints.R --out long.csv --manifest manifest.json
```

These are the only inputs that need R (`data.table` and `jsonlite`). Both
outputs are scratch rather than products, so write them outside
`data/processed/`.

**3. The annual constraints product.** Writes
`data/processed/constraints_annual.nc`.

```bash
python scripts/ingest_constraints.py --long-table long.csv --manifest manifest.json
```

Must follow step 2, whose two files are its input, and step 1.

**4. The initial conditions product.** Writes `data/processed/ic.nc`.

```bash
python scripts/ingest_ic.py --jobs 16
```

`--root` defaults to `data/raw/initial_conditions`. The read is I/O bound, so
`--jobs` is worth raising on a networked filesystem.

There is also an `--allow-gaps` flag, and it is not a default. It fills missing
`(site, member)` pairs with `NaN` and records them in `ic_present` instead of
stopping, and exists so the script is runnable in a checkout holding only part
of the ensemble. On the SCC, with the full ensemble present, it should not be
needed — reaching for it there means something is wrong with the data or with
`--root`, and using it anyway yields a silently partial product.

Two things are deliberately absent from that sequence:

- **Net ecosystem exchange.** `scripts/ingest_nee.py` does not exist yet;
  `data/README.md` records it as intended.
- **Drivers.** There is no driver ingest step at all, and nothing is written
  under `data/processed/` for them.
  `sipnet_calibration.drivers.load_drivers` parses the raw `.clim` files into
  the canonical form on demand.

[`data/README.md`](data/README.md) is the authority on the per-product detail —
the expected layout, provenance, units, and what each script reads and writes.
This section is only the order in which to call them.

### Optional: raw data survey

Two diagnostics that answer questions about the raw data which can only be
answered where the files are. They are **optional**, they run **before**
processing, and they write a JSON summary rather than any processed product.
Neither is part of the ingest pipeline.

```bash
python3 scripts/survey_ic_variables.py --root <IC root> --jobs 16 --out ic_survey.json
```

*Which variables do the initial condition files actually carry, and is the
`(site, member)` ensemble a complete rectangle?* The files are not all alike, and
`ingest_ic.py` treats an unregistered variable as fatal, so this is how to find
out what is there first. It parses the netCDF-3 headers directly and imports
nothing third-party, so it runs under a bare `python3` with no environment
activated. `--sample N` surveys a random sample of sites instead of all of them.

```bash
python scripts/survey_drivers.py --root <drivers root> --jobs 16 --out drivers_survey.json
```

*Does the driver directory template cover every site and member, and does every
`.clim` file pass the reader's own checks?* It applies
`sipnet_calibration.drivers.read_clim_file` to each file, so unlike the survey
above it needs the project environment. There are around 80,000 files at roughly
a tenth of a second each, which is what `--jobs` is for.

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

## Layout

```
src/sipnet_calibration/
  sites.py                # SITE_GRID, load_sites(), select_sites(ids=, bbox=, where=, ...)
  fields.py               # canonical field convention, validate_field(), adapters
  obs_ops.py              # aggregate_time, sipnet_time_index — shared with the likelihood
  plotting/               # style, registry, primitives, series, maps, facet, diagnostics
scripts/                  # ingest: data/raw/ -> data/processed/
experiments/<task>/       # config.py (source of truth) + plots.py (report figures)
data/raw/                 # inputs, never edited; only raw/sites/ is tracked
data/processed/           # ingest output == the canonical format used throughout
tests/
```

`CLAUDE.md` records the code conventions, and
[`data/README.md`](data/README.md) documents the inputs — their formats,
provenance, and coordinate reference system. The model definition, notation,
algorithm design and plotting specification are maintained outside this
repository and are not published with it.

## Conventions

- **One directory per experiment** under `experiments/`, with `config.py` as the
  single source of truth for that experiment: parameters, transforms, data
  paths, algorithm settings.
- **Heavy computation lives in scripts, not notebooks.** Notebooks are for
  exploration and plotting, and load results from disk.
- **Raw inputs are never edited.** Ingest scripts convert `data/raw/` to
  `data/processed/`, whose format *is* the canonical format used by the rest of
  the project — including as the input format for plotting.
- **Calibration runs are tagged** `c_<task_id>_<run_name>_<run_index>`, where
  `c` marks a calibration run, `<task_id>` names the calibration task,
  `<run_name>` a run within it, and `<run_index>` increments on re-runs.

### Plotting

The plotting suite is layered so that data provenance and display style stay
independent — adapters map each data source to one canonical form, and plotters
consume only that form. `CLAUDE.md` records the rules in full; the two that most
affect how the suite is used are:

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

- [#3](https://github.com/arob5/spatial-lsm-calibration/issues/3) — the initial condition
  netCDFs carry an unsubstituted `[year]` template in their time units, so they
  cannot be opened with CF decoding enabled. Adapters must use
  `decode_times=False`.
- [#4](https://github.com/arob5/spatial-lsm-calibration/issues/4) — no installable
  projection library on the development workstation; the spatial plotting layer
  is blocked on choosing an approach.
