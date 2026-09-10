# Spatial Parameter Calibration for Land Surface Models (LSMs)

Scalable Bayesian algorithms for parameter calibration of the **SIPNET**
land surface model (LSM), with an emphasis on multi-site inference that exploits
spatial structure rather than treating plant functional types as the only source
of spatial variability.

This is a research codebase, not a library. 

## Contents

- [Quick start](#quick-start)
  - [Generating the processed data](#generating-the-processed-data)
- [Advanced setup](#advanced-setup)
  - [Known setup caveat](#known-setup-caveat)
  - [Companion packages](#companion-packages)
  - [Raw data survey](#raw-data-survey)
  - [Running notebooks](#running-notebooks)
- [Scope of the problem](#scope-of-the-problem)
- [Layout](#layout)
- [Conventions](#conventions)

## Quick start

Requires [uv](https://docs.astral.sh/uv/). One data processing step below also
needs `Rscript`.

```bash
git clone https://github.com/arob5/spatial-lsm-calibration.git
cd spatial-lsm-calibration
uv sync                     # create .venv from uv.lock, on the pinned 3.14
source .venv/bin/activate   # run everything from inside this environment
pytest                      # check it works
```

Nothing else needs cloning: the three companion packages are installed from
git, and are described under [Companion packages](#companion-packages).

### Generating the processed data

Raw inputs arrive from several sources with differing conventions, so most of
them are converted once into a processed form that everything downstream reads.
This needs the raw data present under `data/raw/` in the layout
[`data/README.md`](data/README.md) specifies. That data is housed on Boston
University's Shared Computing Cluster (SCC), and a fresh clone has almost none
of it, so in practice this runs on the SCC. Raw inputs too large to be worth
copying, the meteorological drivers especially, are left alone; helper
functions query them in place instead.

Run these in order; each reads what an earlier one wrote. A top-level helper
will eventually replace the sequence with a single command.

```bash
python scripts/ingest_sites.py                                                       # -> data/processed/sites/sites.csv
Rscript scripts/export_constraints.R --out long.csv --manifest manifest.json         # scratch, not products
python scripts/ingest_constraints.py --long-table long.csv --manifest manifest.json  # -> data/processed/constraints_annual.nc
python scripts/ingest_ic.py --jobs 16                                                # -> data/processed/ic.nc
```

[`data/README.md`](data/README.md) is the authority on the per-product detail:
the expected layout, provenance, units, and what each script reads and writes.
Every script takes `--help`, which documents its inputs, its outputs and the
flags for pointing it at data that is not where it expects.

## Advanced setup

### Known setup caveat

`pyproj` sets the platform floor: its wheels need macOS 14 on Apple silicon,
macOS 13 on Intel, and glibc 2.28 — RHEL 8 or newer — on Linux. An older
machine will not resolve this environment.

`cartopy` is not a dependency. Its arm64 wheels stop at cp313 while the
development venv is on 3.14, so adopting it would mean bounding the interpreter
from above, which `requires-python = ">=3.12"` does not do today. Nothing is
blocked on it: the display projection is settled and applied through `pyproj`
in `sipnet_calibration.projection`. What cartopy would still supply is the
coastlines and gridline labels that `plotting/maps.py` otherwise needs a
vendored basemap for.

### Companion packages

Three packages are developed alongside this project and are dependencies of it:

| Package | Role |
|---|---|
| [`pySIPNET`](https://github.com/TARPS-group/pySIPNET) | the SIPNET model interface |
| [`PyEns`](https://github.com/arob5/PyEns) | running ensembles |
| [`pyEKI`](https://github.com/TARPS-group/pyEKI) | solving inverse problems with ensemble Kalman methods |

`[tool.uv.sources]` in `pyproject.toml` tracks the `main` branch of each, and
`uv.lock` records the **exact commit** resolved from it. So `uv sync` installs
the same three commits for everyone, and none of them moves until someone
upgrades it deliberately. Ordinary use needs no local checkout of any of them.

#### Upgrading a companion package

New work on `main` in one of these repositories does **not** reach this project
until the lock is refreshed. To refresh:

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
sync that another step of some workflow happens to run. `uv pip show pysipnet` says
which one is installed: an `Editable project location` line means the local
checkout, and no such line means the pinned commit. Re-run the
`uv pip install -e` after any sync.

Once the work is pushed to `main`, upgrade as above and drop the overlay.

### Raw data survey

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

### Running notebooks

Use the project venv's Jupyter directly, **not** `uv run jupyter`:

```bash
.venv/bin/jupyter lab
```

To execute headlessly:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --output out.ipynb in.ipynb
```

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
