# Data

This directory holds the calibration inputs: raw files as provided by collaborators
in `raw/`, and the converted, canonical form in `processed/`.

This document explains assumptions about the structure of the data, where the datasets
came from, and how to convert the raw data in the processed form for the purposes of 
this project. Note that the terms *raw* and *processed* is used with respect to any 
data manipulation done as part of this project. The *raw* data ingested here in 
reality has been processed by others as part of previous analyses. Details are 
given below.

**Verification is a work in progress.** Everything below is either
`VERIFIED` (checked directly against a file present in this repo), `DOCUMENTED`
(stated by an upstream source), or `UNVERIFIED` (from notes, or not checkable from
the local subset). Nothing is inferred silently — where a value merely *looks*
consistent with a unit, that is called out as an open question rather than
recorded as fact. Ongoing communications with colleagues who are sourcing the 
data will continue to fill in the gaps. Once everything is sufficiently verified
this message and the verification tags will be removed.

---

## Contents

- [Site metadata](#site-metadata) — tracked in git
- [Coordinate reference system](#coordinate-reference-system) — tracked in git
- [Raw inputs](#raw-inputs) — not tracked
- [Conversion: raw to processed](#conversion-raw-to-processed)
- [Processed format](#processed-format) — not tracked
- [Open questions](#open-questions)

---

## Site metadata

Two small files, tracked in git because they are the shared keys everything
else joins on.

### `site_ids.csv` — VERIFIED

8000 rows, 4 columns: `site_id`, `lon`, `lat`, `site_name`.

- `site_id` is the integer 1–8000, handed down from Dongchen Zhang's
  `site_info.Rdata`. These are a **shared key with collaborators' files and must
  never be renumbered.**
- `lon`/`lat` are degrees. See [Coordinate reference system](#coordinate-reference-system).
- Extent: lon −178.754 to −20.013, lat 7.013 to 82.546. The corners are
  site 731 in the western Aleutians, site 1 in northeast Greenland, and site 8000
  at 7.01°N in northern South America.
- Only 3640 of 8000 fall inside a CONUS bounding box
  (24–50°N, 125–66°W).

| Region | Sites |
|---|---|
| above 60°N | 2517 |
| above 70°N | 475 |
| above 80°N | 36 |
| below 20°N | 412 |
| west of 140°W | 1289 |
| east of 50°W | 31 |

> **`site_name` is MISALIGNED with `lon`/`lat` — do not use it for anything.** — VERIFIED
>
> The column has 724 distinct values, dominated by `weighted_sample` (6907), then
> `ameriflux` (190) and `soil_core_alaska` (100). But it does not correspond to
> the coordinates in its own row:
>
> | Label | n | Actual lat range | Where that name implies |
> |---|---|---|---|
> | `soil_core_alaska` | 100 | 33.3 – 36.0 | Alaska, 55–72°N |
> | `Hawaii` | 4 | 36.6 – 36.7, lon −119.6 to −86.1 | ~19–22°N, −155 to −160 |
> | names containing `(CR-*)`, `(PA-*)` | 6 | 45.6 – 45.8 | Costa Rica / Panama, 7–12°N |
> | names containing `(MX-*)` | 9 | 46.0 – 46.6 | Mexico, 14–33°N |
> | `ameriflux` | 190 | 8.4 – 33.3 (contiguous) | scattered continent-wide |
>
> Every label group occupies a **contiguous latitude band**, which is the
> signature of an ordering mismatch rather than of individual bad rows: the
> coordinates are sorted by descending latitude while the name vector appears to
> retain a different order.
>
> **`site_id`, `lon` and `lat` are consistent with each other** — only
> `site_name` is decoupled. Verified by spot-checking the id map against known
> site locations: `US-PF*` → −90.2/45.9 (Park Falls, WI), `US-xWR` → −121.95/45.82
> (Wind River, WA), `US-Syv` → −89.35/46.24 (Sylvania, MI), `US-UMB` → −84.71/45.56
> (UMBS). All correct.
>
> **Where the misalignment was introduced is unknown** — it could be upstream in
> `site_info.Rdata`, or in this repo's extraction of that file to csv. Since
> `site_info` is four parallel vectors, an extraction that reordered one of them
> would produce exactly this. Checking that is the cheapest next step: reload the
> `.Rdata` and confirm `site_name[i]` agrees with `lat[i]`.

### `site_id_map.csv` — VERIFIED

185 rows, 2 columns: `Site_ID` (Ameriflux, e.g. `US-xDC`), `index` (the 1–8000
site id, range 4102–7418 in this file). Produced by exact matching.

> **Only 165 sites are usable for NEE.** — VERIFIED
>
> The NEE product covers **209** distinct Ameriflux sites and this map covers
> **185**, but the intersection is **165**. That is: 44 sites have NEE data but no
> site id, and 20 sites have a site id but no NEE data. Any NEE-constrained
> calibration is limited to those 165 unless the mapping is extended. (Not a
> case or whitespace artifact — re-checked case-insensitively, same counts.)
>
> **The map is CONUS-only, which explains most of the 44.** All 185 mapped target
> sites fall within lat 25.35–47.16 and lon −122.33 to −68.74, entirely inside a
> CONUS box, and they address the contiguous id block **4102–7418**. Since site
> ids are ordered by descending latitude, that block *is* a latitude band. The NEE
> product, by contrast, spans the continent. Of the 44 unmapped sites, 12 are
> non-US (11 `CA-*`, 1 `CR-*`) and 15 carry codes belonging to Alaskan clusters
> (Bonanza Creek, Imnavait Creek, Poker Flat, Rosie Creek, Yukon-Kuskokwim, and
> NEON Alaska) — so ~27 of 44 are outside CONUS. The remaining 17 are
> unexplained. *(Locations of the 44 are read from site codes, not verified
> against coordinates — no Ameriflux coordinates are available locally.)*
>
> **The 20 map-only sites remain genuinely unexplained.** They include `US-Ha1`
> (Harvard Forest), `US-MMS` (Morgan Monroe) and `US-Ne1`/`US-Ne2` (Mead) — among
> the most data-rich sites in Ameriflux, all inside CONUS, all present in the map.
> Their total absence from a gap-filled EC product is not explained by geography.
> The likeliest reading is that the id map and the gap-fill product are two
> separate deliverables built from different input site lists; worth one question
> to Gu Yang.

---

## Coordinate reference system

Recorded in **[`site_crs.json`](site_crs.json)** (tracked in git), which is the
authoritative machine-readable definition. Summary:

- **Source CRS of the `lon`/`lat` columns: EPSG:4326 (WGS 84 geographic)** —
  DOCUMENTED by the ORNL DAAC dataset guide.
- The coordinates are **cell centres of a 30 arcsecond (1/120°) geographic
  grid**: extent −179→−20 lon, 7→85 lat, 19080 × 9360 cells. VERIFIED — all 8000
  sites land on cell centres to within 1.02e-6° (≈0.11 m).
- The grid is **not equal-area**: 30 arcsec is ~928 m in latitude everywhere, but
  in longitude ranges from ~921 m at 7°N to **~121 m at 82.5°N**.
- The file stores **longitude first**, which is the GDAL/PROJ traditional order
  and the opposite of EPSG:4326's formal axis order. Transform with
  `always_xy=True` or the equivalent.

**The map projection for plotting is not specified here, by design.** It is our
choice, is not yet decided (repo issue #4), and will be documented separately once
it is. `site_crs.json` records only what gives the raw coordinates their meaning.
For context, the upstream product's own figures used ESRI:102003 (USA Contiguous
Albers Equal Area Conic); that note lives in the JSON's `provenance` block, clearly
marked as context rather than as a property of our data.

---

## Raw inputs

**Not tracked in git** (`.gitignore` excludes `data/raw/`), and never edited.

The full dataset lives on BU's SCC; see `Data Notes.md` for the SCC paths. What
is present locally is a **small format-reference subset** — a single site's
drivers and initial conditions, plus the complete NEE and AGB/LAI constraint
files. On SCC these should be symlinks rather than copies; locally they are real
copies of the subset.

### Provenance

The 8000 sites, the driver and initial-condition ensembles, and the AGB/LAI
constraints all originate from Dongchen Zhang's North American land carbon
reanalysis:

> Zhang, D., J. Huggins, Q. Li, S. Ramachandran, S. P. Serbin, C. Webb, Z. Zuo,
> and M. Dietze. 2026. *North American Land Carbon Reanalysis, 2012–2024.* ORNL
> DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2507

Preprint: https://www.biorxiv.org/content/10.64898/2026.02.25.708030v1.abstract
Dataset guide: https://daacweb-prod.ornl.gov/CMS/guides/Land_C_Reanalysis_NorthAmerica.html

The NEE constraint data comes separately, from Gu Yang's gap-filling work.

### Expected layout

```
raw/
  drivers/ERA5_<site_id>_<member>/ERA5.<member>.2012-01-01.2024-12-31.clim
  initial_conditions/<site_id>/IC_site_<site_id>_<member>.nc
  constraints/
    nee/ens_ec_3h.csv
    sda_8k_site_rdata/obs.mean.Rdata
    sda_8k_site_rdata/obs.cov.Rdata
```

The two path templates are **UNVERIFIED** as general patterns: only
`ERA5_1_1/ERA5.1.2012-01-01.2024-12-31.clim` and
`initial_conditions/1/IC_site_1_1.nc` exist locally, so the templates are
inferred from those two examples plus `Data Notes.md`. Whether all 8000 site
directories follow them has not been checked.

### Drivers — `.clim` — VERIFIED (one file)

SIPNET climate format, space-delimited, **no header**. From
`ERA5.1.2012-01-01.2024-12-31.clim`:

- **37,992 rows**, **14 columns** (uniform across all rows).
- 3-hourly, 2012-01-01 through 2024 day 366. Row count is exactly consistent:
  13 years with 4 leap years = 4749 days × 8 steps = 37,992.
- Columns are pySIPNET's 14-column layout:
  `loc, year, day, time, length, tair, tsoil, par, precip, vpd, vpd_soil, vpress, wspd, soil_wetness`.
- `loc` is constant `0`. `length` is constant `0.125` (days = 3 h), which confirms
  the timestep. `soil_wetness` is constant `0.6`; per pySIPNET, SIPNET ignores it.
- **No datetime column** — time is `year`, `day`, `time`.

Units are as documented by pySIPNET's `clim_io` module rather than by the data
itself. Note two of its warnings apply here: `par` is a **total** over the
timestep (not a rate), and SIPNET silently clamps non-positive `vpd`/`wspd`.

Driver **ensemble size is UNVERIFIED** — only member 1 is present locally.

### Initial conditions — `.nc` — VERIFIED (one file)

From `IC_site_1_1.nc` (712 bytes):

| Variable | Dims | Value | `units` attr | `long_name` attr |
|---|---|---|---|---|
| `AbvGrndWood` | `(time: 1)` | 0.05823572 | `kg C m-2` | Above ground woody biomass |
| `wood_carbon_content` | `(time: 1)` | 0.05823572 | `kg C m-2` | Wood Carbon Content |
| `soil_organic_carbon_content` | `(time: 1)` | 13.08545431 | `kg C m-2` | Soil Organic Carbon Content by Layer |
| `time` | `(time: 1)` | 1.0 | `days since [year]-01-01 00:00:00 UTC` | Time middle averaging period |

No global attributes. Note `AbvGrndWood` and `wood_carbon_content` are *equal* in
this file.

> **These files cannot be opened with CF time decoding** — repo issue #3.
>
> The `time` units attribute is a literal unsubstituted template: `[year]` was
> never replaced. No calendar library can parse it, and **installing `cftime`
> does not help** (verified with 1.6.5). Open with `decode_times=False`. The
> `time` dimension is length 1 and carries no information — these are static
> initial conditions.

`Data Notes.md` records that other IC files carry additional variables
(`leaf_carbon_content`, `SoilMoistFrac`) — **UNVERIFIED locally**, as is the
ensemble size. The upstream product is DOCUMENTED as having 100 ensemble members
plus mean and standard deviation, which is consistent with the `IC_site_1_100.nc`
path appearing in Dongchen's XML, but the IC ensemble size has not been directly
confirmed.

### NEE constraint — `ens_ec_3h.csv` — VERIFIED

1.7 GB. **3,547,921 data rows, 28 columns.**

- `Site_ID` — Ameriflux id (e.g. `CA-ARB`). **209 distinct sites**; see the
  165-site caveat under [`site_id_map.csv`](#site_id_mapcsv--verified).
- `utc` — ISO timestamps, 3-hourly, **2012-01-01T03:00:00Z to 2025-01-01T03:00:00Z**.
- `ens01` … `ens25` — 25 ensemble members.
- `ens_mean` — **verified to be exactly the mean of `ens01`…`ens25`** (max
  discrepancy 7.8e-14 at `US-UMB`). It is therefore redundant, and **must never be
  treated as a 26th ensemble member**: admitting it to the member dimension
  corrupts every quantile computed from the ensemble.

> **24.7% of rows have zero across-member spread** — VERIFIED at `US-UMB`
> (7,225 of 29,225 rows have all 25 members identical; median spread on the rest
> is 0.116). A plausible reading is that zero spread marks a *measured* rather
> than gap-filled value, which would make it a free quality flag and would matter
> a great deal for the observation-error model. **Unconfirmed — worth asking.**

> **Coverage is highly ragged in time** — VERIFIED
>
> The row count implies far less data than 209 sites × full period would give. A
> complete 3-hourly series over 2012-01-01 to 2025-01-01 is ~37,992 timesteps, but
> per-site row counts run **min 2,921, median 17,537, mean 16,975, max 37,993** —
> a mean of **45% of the full period**.
>
> | Coverage | Sites |
> |---|---|
> | ≥99% | 17 |
> | ≥90% | 28 |
> | ≥50% | 62 |
> | ≥25% | 144 |
>
> So only 17 of 209 sites have a near-complete record. A dense
> `(member, site, time)` array would be **~55% NaN**. That is an acceptable price
> for keeping the canonical rectangular convention — NaN compresses well in zarr —
> but it must be a deliberate choice, and any per-site statistic has to account for
> wildly unequal sample sizes.

> **Units are an OPEN QUESTION and this matters.**
>
> At `US-UMB` (UMBS, Michigan; 29,225 rows) values run **−37.2 to +13.3**, with a
> July diurnal cycle peaking near **−21** at local midday. Those magnitudes are
> characteristic of **µmol CO₂ m⁻² s⁻¹** — a **rate** — and are hard to reconcile
> with any other common NEE unit; a value of −21 g C m⁻² per 3 h would be roughly
> an order of magnitude beyond anything physical. This is strong evidence, but it
> is still inference from magnitude and is **not recorded here as fact**.
>
> SIPNET's own `nee` is `g C m⁻² per timestep` — a **total**. If the two are
> plotted or compared without conversion the result is wrong by orders of
> magnitude, with no visual cue.
>
> The **sign convention, by contrast, is settled empirically** — VERIFIED. At
> `US-UMB`, July means by UTC hour run +6.2 at night and −20.8 at local midday,
> and monthly means are negative June–September, positive October–May. So
> **negative = uptake, positive = release**, matching SIPNET's documented
> `+ = to atmosphere`.
>
> `Data Notes.md` quotes Meng's README as `x_kgC_m2_s <- x_umolCO2_m2_s * 12e-9`,
> which does not by itself establish which unit this file is in. **Confirm with
> Gu Yang before using NEE as a constraint.**

### AGB / LAI / soil constraints — `.Rdata` — VERIFIED

Inspected directly in R. Both files are `list` of length **13**, keyed by date
`2012-07-15` … `2024-07-15`; each year is a list of length **8000** named
`"1"` … `"8000"`.

**`obs.mean.Rdata`** — object `obs.mean`. Each site-year is a **1-row
data.frame** whose columns are the variables present. Column count ranges over
{0, 1, 2, 3, 4}; columns appear in alphabetical order
(`AbvGrndWood`, `LAI`, `SoilMoistFrac`, `TotSoilCarb`). No entry is `NULL`, but
**6 sites in 2012 have a 0-column data.frame** — i.e. no observations at all.

**Coverage is strongly year-dependent**, which is a sharper constraint than "not
every site has every variable":

| Year | TotSoilCarb | LAI | AbvGrndWood | SoilMoistFrac |
|---|---|---|---|---|
| 2012 | 7990 | 7670 | 3281 | **0** |
| 2013 | 7990 | 7668 | 3281 | **0** |
| 2014 | 7990 | 7677 | 3281 | **0** |
| 2015 | 7990 | 7674 | 3281 | 7974 |
| 2016–2023 | 7990 | 7649–7678 | 3262–3281 | 7974 |
| 2024 | 7990 | 7663 | **0** | 7974 |

So `SoilMoistFrac` is absent before 2015, `AbvGrndWood` is absent in 2024, and
`AbvGrndWood` covers only ~41% of sites in the years it exists. **Ingest must not
assume a rectangular site × year × variable array.**

**`obs.cov.Rdata`** — object `obs.cov`. Same nesting. Each site-year is a
`numeric` (1×1 case) or a `matrix`. Dimensions in 2012: 6 empty, 283 1×1,
4475 2×2, 3236 3×3 — matching `obs.mean`'s column counts exactly, verified for
all 8000 sites in 2015 with zero mismatches.

> Two corrections to `Data Notes.md`, both VERIFIED:
>
> 1. `obs.cov`'s site-level lists **are** named `"1"`…`"8000"` (the notes say
>    there are no names). What is genuinely absent is **dimnames on the
>    matrices**, so it is the **variable order**, not the site order, that must be
>    taken from `obs.mean`'s columns.
> 2. The directory is named `Rdata_with_attributes`, but there are **no
>    non-standard attributes anywhere** — not on the outer list, the year lists,
>    or any of the 8000 data.frames. This bears on the open question of how it
>    differs from the sibling `Rdata/` directory.

**Units** — DOCUMENTED by the ORNL DAAC dataset guide for the published product:

| Variable | Unit | Measured range (2015) |
|---|---|---|
| `AbvGrndWood` | Mg C ha⁻¹ | 0 – 459 |
| `LAI` | m² m⁻² | 0.1 – 6.9 |
| `SoilMoistFrac` | percent | 0.99 – 92.89 |
| `TotSoilCarb` | kg C m⁻² | 5.79 – 144.4 |

This resolves `Data Notes.md`'s puzzle that "Meng's README says Mg/ha, but this
can't be true for all variables" — only `AbvGrndWood` is. It also confirms
`SoilMoistFrac` is a **percent, not a fraction**, from documentation rather than
from its range.

One caveat, deliberately not glossed: the guide documents the **published
reanalysis output**, whereas `obs.mean.Rdata` is the SDA **observation input**.
The four variable names and all 13 July-15 timestamps match exactly, so they
almost certainly share definitions — but that is inference. **Worth one
confirmation.** The July-15 timestamps are DOCUMENTED as the product's annual
snapshot convention, which answers a standing question in `Data Notes.md`.

---

## Conversion: raw to processed

Ingest scripts live in `../scripts/`. Each reads from `raw/`, writes to
`processed/`, and never modifies its input. Run `--help` on a script for usage;
this table records only what each one does.

| Script | Reads | Writes |
|---|---|---|
| `ingest_sites.py` | `site_ids.csv`, `site_id_map.csv`, PFT csv | `processed/sites.parquet` |
| `ingest_ic.py` | `raw/initial_conditions/` | `processed/ic.nc` |
| `ingest_constraints.py` | R export of `obs.mean`/`obs.cov` | `processed/agb_lai.nc` |
| `ingest_nee.py` | `raw/constraints/nee/ens_ec_3h.csv` | `processed/nee.zarr` |
| `ingest_drivers.py` | `raw/drivers/` | `processed/drivers.zarr` |

**The `.Rdata` files require R.** `obs.mean` is a list-of-lists-of-data.frames,
and `pyreadr` handles data.frames only, so flattening must happen in a one-time R
script that writes netCDF or csv. These are the only inputs needing R.

Two conversions are load-bearing and must not be left implicit:

- **NEE units.** Whatever the resolution of the open question above, the adapter
  converts into the one canonical unit named in the code's `VARIABLES` registry.
  Nothing downstream reconciles units.
- **`ens_mean` is dropped**, not carried as a member.

`processed/` is regenerable from `raw/` and is therefore not tracked in git. It
does not exist on a fresh clone; ingest scripts create it.

---

## Processed format

**PROPOSED** — the ingest scripts are not yet written. The rationale for these
choices is in the plotting design spec; the short version is that the processed
format *is* the canonical format used by the rest of the project, including as
plotting input, so it is chosen to be directly loadable as such.

The canonical in-memory form is an `xarray.DataArray` with dims a subset of
`(member, site, time)`, `lon`/`lat` as non-dimension coordinates on `site`, and
units in `attrs`. Formats are chosen by *shape*:

| Product | Format | Dims | Rationale |
|---|---|---|---|
| `sites.parquet` | Parquet | table | tabular, tiny, joins |
| `ic.nc` | netCDF | `(member, site)` | ~8000 × 100 × 3 ≈ 19 MB, whole-file |
| `agb_lai.nc` | netCDF | `(site, time)` per variable, plus covariances | 8000 × 13, tiny |
| `nee.zarr` | Zarr, chunked on `site` | `(member, site, time)` | 25 × 165 × 38k ≈ 630 MB f32 dense, ~55% NaN; lazy per-site reads |
| `drivers.zarr` | Zarr, chunked on `(site, time)` | `(member, site, time)` | full array ~15 GB per member; site subsets only, locally |

Zarr rather than Parquet for the `(member, site, time)` blocks because Zarr *is*
the canonical xarray convention on disk: `open_zarr(...).sel(site=[...])` is the
adapter, with no reshape step. Lazy reads are dask-backed.

Conventions that apply to every product:

- `site` is the integer 1–8000. **Never renumbered.** Ameriflux `Site_ID` and
  `pft` are non-dimension coordinates on `site`, absent where unknown.
- `member` is a 0-based integer, and is **source-local** — see the open question
  about cross-source member alignment.
- Time is a real datetime index. SIPNET's `year`/`day`/`time` triple is converted
  at the boundary.
- Ragged coverage is represented honestly. The AGB/LAI product in particular is
  not rectangular over site × year × variable.

**Open format questions:** the canonical NEE unit; whether covariances are stored
as full matrices per site-year or in a sparse form; whether `processed/` carries
its own spatially-meaningful site ordering as an extra coordinate (see below).

---

## Open questions

Data questions, in rough order of how much they block.

1. **What unit is `ens_ec_3h.csv` in, and what is its sign convention?** Blocks
   any comparison of modelled and observed NEE. Ask Gu Yang.
2. **Is `obs.mean.Rdata` in the same units as the published reanalysis product?**
   Near-certain but inferred; one confirmation from Dongchen Zhang would settle it.
3. **Can the NEE site mapping be extended past 165 sites?** 44 sites have NEE
   data but no site id. `Data Notes.md` also records an unresolved concern that
   earlier mappings were distance-based rather than exact.
4. **Is the `member` index meaningful across sources?** Were driver member *i*,
   IC member *i*, and the calibration ensemble drawn jointly or independently?
   xarray aligns on coordinate values automatically, so this determines whether
   cross-source arithmetic is a feature or a silent bug.
5. **Should `processed/` carry its own spatially-meaningful site ordering?** The
   1–8000 ids must not be renumbered, but a Hilbert or Morton rank as an extra
   coordinate would help triangulation and chunk locality.
6. **Why is `site_name` misaligned with the coordinates, and where did it happen?**
   Upstream in `site_info.Rdata`, or in this repo's csv extraction? Until answered,
   `site_name` is unusable and any conclusion resting on it is void.
7. **Why do 20 mapped sites have no NEE data, including Harvard Forest and Morgan
   Monroe?** And can the 17 unexplained unmapped CONUS sites be recovered? Ask
   Gu Yang.
8. **Which PFT assignment?** `Data Notes.md` describes two files with 16 and 3
   distinct PFTs respectively. Not resolved, and no PFT file is present locally.
9. **How do the sibling observation directories differ?** `Rdata_with_attributes`
   versus `Rdata` versus the XML's `Obs` output directory — and given that
   `Rdata_with_attributes` carries no attributes, the naming is actively
   misleading.
10. **What reference year was intended for the IC `time` units?** Repo issue #3.
   Not needed for calibration, since the dimension is degenerate.
11. **Should the time horizon extend beyond 2012–2024?** Note the NEE file already
   runs to 2025-01-01 while the drivers end at 2024 day 366.
