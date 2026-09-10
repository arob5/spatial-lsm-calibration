# Spatial Parameter Calibration for Land Surface Models (LSMs)

Scalable Bayesian algorithms for parameter calibration of the **SIPNET**
land-surface model, with an emphasis on multi-site inference that exploits
spatial structure rather than treating plant functional types as the only source
of pooling.

This is a **research codebase, not a library**. The deliverables are calibrated
parameter ensembles, diagnostic outputs, and reusable inference machinery — not
a published package. Interfaces change when the science requires it.

## Getting set up

Requires [uv](https://docs.astral.sh/uv/). The constraints ingest additionally
needs `Rscript`; nothing else does.

**1. Clone this repository and its two companion packages as siblings.**
`pyproject.toml` installs `pysipnet` and `pyens` as editable locals from
`../pySIPNET` and `../PyEns`, so the sibling layout is required rather than a
convention. `pySIPNET` is the SIPNET model interface; `PyEns` runs ensembles in
parallel.

```bash
git clone https://github.com/arob5/spatial-lsm-calibration.git
git clone https://github.com/TARPS-group/pySIPNET.git
git clone https://github.com/arob5/PyEns.git
cd spatial-lsm-calibration
```

**2. Sync the environment.** This creates `.venv` from `uv.lock`, using the
interpreter pinned in `.python-version` (3.14). `requires-python` is only a
floor, so use the pin.

```bash
uv sync
```

**3. Activate it, and stay in it for everything below.**

```bash
source .venv/bin/activate
```

Scripts, tests and notebooks alike import `sipnet_calibration` and its
dependencies from `.venv`, so a system `python` fails on the first import rather
than doing something subtly different; `which python` should print a path ending
in `.venv/bin/python`. `uv run <command>` is the equivalent for a one-off
without activating. Notebooks are the one case needing more than this; see
[Running notebooks](#running-notebooks).

**4. Check the environment works.**

```bash
uv run pytest
```

Two further packages are related but **not currently dependencies**, and nothing
above installs them:

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

Data formats, provenance, and the coordinate reference system are documented in
[`data/README.md`](data/README.md).

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

## Status

Early. The inference substrate lives in a separate package (`pyEKI`, above) and
the first multi-site calibration run (`test1`) is not yet configured. What
exists here today:

- the `src/sipnet_calibration/` module layout, whose modules carry the contract
  each is to satisfy; `sites.py` is implemented, the rest are not;
- `SITE_GRID` and the conversions between coordinates and grid indices, in
  `sipnet_calibration.sites`, with tests;
- `load_sites()` and `select_sites()` over the processed site table, with tests;
- site metadata for the 8000-site pool, as a point shapefile under
  `data/raw/sites/`, and the Ameriflux ID map (`data/site_id_map.csv`);
- `scripts/ingest_sites.py`, which turns those two into
  `data/processed/sites/sites.csv`.

The model definition, notation, algorithm design and plotting specification are
maintained outside this repository and are not published with it. `CLAUDE.md`
records the code conventions; `data/README.md` documents the inputs.

## Open issues

Data and environment problems currently tracked:

- [#3](https://github.com/arob5/spatial-lsm-calibration/issues/3) — the initial condition
  netCDFs carry an unsubstituted `[year]` template in their time units, so they
  cannot be opened with CF decoding enabled. Adapters must use
  `decode_times=False`.
- [#4](https://github.com/arob5/spatial-lsm-calibration/issues/4) — no installable
  projection library on the development workstation; the spatial plotting layer
  is blocked on choosing an approach.
