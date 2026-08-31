# Data

This directory holds the inputs used for SIPNET parameter calibration: the files
as received from their original sources, under `raw/`, and the converted form
used throughout the project, under `processed/`.

The terms *raw* and *processed* refer only to data manipulation carried out as
part of this project. The raw data ingested here has already been processed by
others as part of earlier analyses; the sources are documented for each dataset
below.

Points that are unresolved or awaiting confirmation from the groups that produced
the data are marked inline with a numbered note, and described in more detail
under [Open questions](#open-questions).

## Contents

- [Sources](#sources)
- [Directory layout](#directory-layout)
- [Site metadata](#site-metadata)
- [Coordinate reference system](#coordinate-reference-system)
- [Raw inputs](#raw-inputs)
- [Conversion to processed form](#conversion-to-processed-form)
- [Processed format](#processed-format)
- [Open questions](#open-questions)

## Sources

Three sources are referenced throughout this document.

- **[NALCR]** Zhang, D., J. Huggins, Q. Li, S. Ramachandran, S. P. Serbin,
  C. Webb, Z. Zuo, and M. Dietze (2026). *North American Land Carbon Reanalysis,
  2012-2024.* ORNL DAAC, Oak Ridge, Tennessee, USA.
  [doi:10.3334/ORNLDAAC/2507](https://doi.org/10.3334/ORNLDAAC/2507).
  See also the
  [dataset guide](https://daacweb-prod.ornl.gov/CMS/guides/Land_C_Reanalysis_NorthAmerica.html)
  and the
  [preprint](https://www.biorxiv.org/content/10.64898/2026.02.25.708030v1.abstract).
- **[pySIPNET]** The pySIPNET interface package, whose `pysipnet.climate` and
  `pysipnet.io.clim_io` modules define the SIPNET climate file format.
- **[GAPFILL]** Gap-filled eddy-covariance net ecosystem exchange, produced by
  Gu Yang at Boston University. No published reference at present.

---

## Directory layout

Ingest expects the following structure. Site identifiers are the integers 1-8000
described under [Site metadata](#site-metadata), and `<member>` indexes an
ensemble member.

```
data/
  site_ids.csv                  site table
  site_id_map.csv               Ameriflux identifier map
  site_crs.json                 coordinate reference system
  raw/
    drivers/
      ERA5_<site_id>_<member>/ERA5.<member>.2012-01-01.2024-12-31.clim
    initial_conditions/
      <site_id>/IC_site_<site_id>_<member>.nc
    constraints/
      nee/ens_ec_3h.csv
      sda_8k_site_rdata/obs.mean.Rdata
      sda_8k_site_rdata/obs.cov.Rdata
  processed/                    ingest output, created by the ingest scripts
```

> **Note 1.** The two per-site directory templates are inferred from a single
> example of each rather than confirmed across all 8000 sites.

The three files at the top level are tracked in version control, since they are
small and are the keys every other dataset joins on. Neither `raw/` nor
`processed/` is tracked. Files under `raw/` are treated as read-only; all
conversion happens on the way into `processed/`, which is regenerable and absent
on a fresh clone.

---

## Site metadata

### `site_ids.csv`

8000 rows, one per site.

| Column | Type | Description |
|---|---|---|
| `site_id` | integer | Site identifier, 1-8000 |
| `lon` | float | Longitude, degrees east |
| `lat` | float | Latitude, degrees north |
| `site_name` | string | Free-text label; see Note 2 |

The identifiers originate with [NALCR] and are shared with collaborators' files,
so they are treated as fixed and are never renumbered. Rows are ordered by
descending latitude.

Coordinates span -178.754 to -20.013 in longitude and 7.013 to 82.546 in
latitude. The extremes are site 731 in the western Aleutians, site 1 in northeast
Greenland, and site 8000 at 7.01 degrees north in northern South America. 3640 of
the 8000 sites fall inside a conterminous-US bounding box of 24-50 north and
125-66 west, so analyses restricted to that region use a little under half the
pool. The rest are distributed as follows.

| Region | Sites |
|---|---|
| north of 60 | 2517 |
| north of 70 | 475 |
| north of 80 | 36 |
| south of 20 | 412 |
| west of 140 | 1289 |
| east of 50 west | 31 |

> **Note 2.** The `site_name` column does not correspond to the coordinates in
> its own row and should not be used. For example, the 100 rows labelled
> `soil_core_alaska` have latitudes between 33.3 and 36.0. The `site_id`, `lon`
> and `lat` columns are mutually consistent.

### `site_id_map.csv`

185 rows mapping Ameriflux site identifiers onto the integer site identifiers, by
exact match.

| Column | Type | Description |
|---|---|---|
| `Site_ID` | string | Ameriflux site identifier, for example `US-xDC` |
| `index` | integer | Corresponding `site_id`; 4102-7418 in this file |

All mapped sites lie within 25.35-47.16 north and 122.33-68.74 west, so the map
covers the conterminous US only. Because site identifiers are ordered by
descending latitude, the mapped range 4102-7418 is a contiguous latitude band.

The map and the [GAPFILL] net ecosystem exchange file do not cover the same
sites: the NEE file contains 209 Ameriflux sites and the map contains 185, with
165 in common. Calibration constrained by NEE is therefore limited to 165 sites
unless the mapping is extended.

> **Note 3.** 44 sites have NEE data but no site identifier, and 20 sites have a
> site identifier but no NEE data.

---

## Coordinate reference system

The `lon` and `lat` columns of `site_ids.csv` are geographic coordinates on the
WGS 84 datum (EPSG:4326), as documented in the [NALCR]
[dataset guide](https://daacweb-prod.ornl.gov/CMS/guides/Land_C_Reanalysis_NorthAmerica.html).
The full definition, together with the grid the coordinates fall on, is recorded
in machine-readable form in [`site_crs.json`](site_crs.json).

Site coordinates are cell centres of the approximately 1 km geographic grid used
by [NALCR]: 0.008333 degree, or 30 arcsecond, resolution in both latitude and
longitude, spanning 179 west to 20 west and 7 north to 85 north as a
19080 by 9360 array. All 8000 sites fall on cell centres to within
1.02e-6 degrees, about 0.11 m.

Two consequences are worth noting.

- The grid is not equal-area. Thirty arcseconds is about 928 m in latitude
  everywhere, but in longitude it ranges from roughly 921 m at 7 north to 121 m
  at 82.5 north. Density and per-area calculations must account for this.
- The file stores longitude before latitude, which is the traditional GDAL and
  PROJ ordering rather than the axis order EPSG:4326 formally declares.
  Coordinate transformations should be configured accordingly, for instance with
  pyproj's `always_xy=True`.

`site_crs.json` records the coordinate system of the input data only. The map
projection used for plotting is a separate choice, tracked in
[issue #4](https://github.com/arob5/spatial-lsm-calibration/issues/4), and will
be documented once settled.

---

## Raw inputs

### Meteorological drivers

**Format.** ERA5 reanalysis written in the SIPNET climate format: space-delimited
text with no header row, one row per timestep. The example file has 37,992 rows
and 14 columns, covering 2012-01-01 to the end of 2024 on a 3-hourly timestep.
That row count is consistent with 4749 days, being thirteen years including four
leap years, at eight timesteps per day.

Columns follow the 14-column layout defined by [pySIPNET].

| # | Column | Unit | Description |
|---|---|---|---|
| 1 | `loc` | | Location index; constant, and required by SIPNET to be so |
| 2 | `year` | | Integer year |
| 3 | `day` | | Integer day of year, 1 = 1 January |
| 4 | `time` | hours | Fractional hours at the start of the timestep |
| 5 | `length` | days | Timestep duration; 0.125, that is 3 hours |
| 6 | `tair` | deg C | Mean air temperature |
| 7 | `tsoil` | deg C | Mean soil temperature |
| 8 | `par` | mol m-2 | Photosynthetically active radiation, integrated over the timestep |
| 9 | `precip` | mm | Total precipitation over the timestep |
| 10 | `vpd` | Pa | Vapour pressure deficit |
| 11 | `vpd_soil` | Pa | Soil-air vapour pressure deficit |
| 12 | `vpress` | Pa | Vapour pressure in the canopy airspace |
| 13 | `wspd` | m s-1 | Mean wind speed |
| 14 | `soil_wetness` | | Legacy column, ignored by SIPNET; constant 0.6 |

**Interpretation.** There is no datetime column; time is given by the `year`,
`day` and `time` triple and must be assembled during ingest. Two columns are
integrated quantities rather than rates: `par` and `precip` are totals over the
timestep, so temporal aggregation of either is a sum rather than a mean. SIPNET
requires `vpd` and `wspd` to be strictly positive and silently clamps values that
are not, so non-positive entries are better caught at ingest.

> **Note 4.** The number of driver ensemble members is not established.

**Source.** ERA5, prepared for the 8000-site pool as part of [NALCR].

### Initial conditions

**Format.** One netCDF file per site and ensemble member, holding scalar initial
values for the model's carbon pools. Each variable has a single `time` element.
The example file contains the following.

| Variable | Units | Long name | Example value |
|---|---|---|---|
| `AbvGrndWood` | kg C m-2 | Above ground woody biomass | 0.05823572 |
| `wood_carbon_content` | kg C m-2 | Wood Carbon Content | 0.05823572 |
| `soil_organic_carbon_content` | kg C m-2 | Soil Organic Carbon Content by Layer | 13.08545431 |
| `time` | see Note 5 | Time middle averaging period | 1.0 |

**Interpretation.** These are static initial conditions, so the length-1 `time`
dimension carries no information and can be dropped on read. In the example file
`AbvGrndWood` and `wood_carbon_content` hold the same value. Files for other
sites are reported to include `leaf_carbon_content` and `SoilMoistFrac` as well,
so ingest should treat the variable set as varying between files rather than
fixed.

> **Note 5.** The `time` units attribute is the unsubstituted template
> `days since [year]-01-01 00:00:00 UTC`, which no calendar library can parse.
> These files must be opened with CF time decoding disabled, for example
> `xarray.open_dataset(path, decode_times=False)`.

> **Note 6.** The number of initial-condition ensemble members, and which
> variables appear in which files, are not established.

**Source.** Initial conditions prepared for the 8000-site pool as part of
[NALCR], which reports 100 ensemble members together with ensemble mean and
standard deviation.

### Net ecosystem exchange

**Format.** A single CSV file of about 1.7 GB, with 3,547,921 data rows and 28
columns.

| Column | Type | Description |
|---|---|---|
| `Site_ID` | string | Ameriflux site identifier; 209 distinct values |
| `utc` | timestamp | ISO 8601, 3-hourly, 2012-01-01T03:00:00Z to 2025-01-01T03:00:00Z |
| `ens01` to `ens25` | float | Twenty-five ensemble members |
| `ens_mean` | float | Mean of `ens01` to `ens25` |

**Interpretation.** `ens_mean` is exactly the arithmetic mean of the 25 member
columns, agreeing to within 8e-14 in the site checked, and is therefore
redundant. It must not be carried as a 26th ensemble member, since that biases
any quantile computed across members.

The sign convention is that positive values denote flux to the atmosphere and
negative values denote uptake, matching SIPNET's own convention for net ecosystem
exchange. At `US-UMB`, July means by hour run from +6.2 overnight to -20.8 at
local midday, and monthly means are negative from June to September and positive
from October to May.

Temporal coverage is uneven across sites, and sparser than the row count alone
suggests. A complete 3-hourly record over the file's date range would be about
37,992 rows per site, but per-site counts range from 2,921 to 37,993 with a median
of 17,537, so the average site covers about 45% of the period.

| Coverage of the full period | Sites |
|---|---|
| at least 99% | 17 |
| at least 90% | 28 |
| at least 50% | 62 |
| at least 25% | 144 |

Only 17 of the 209 sites have a near-complete record. Representing the file as a
dense array over site and time therefore leaves roughly 55% of entries missing,
and any per-site statistic must account for very unequal sample sizes.

> **Note 7.** The units of the ensemble columns are not confirmed. Values at
> `US-UMB` range from -37.2 to +13.3 and peak near -21 at midsummer midday, which
> is characteristic of umol CO2 m-2 s-1, a rate. SIPNET reports net ecosystem
> exchange as a per-timestep total in g C m-2, so a conversion is required and its
> direction depends on this answer.

> **Note 8.** In about a quarter of rows, all 25 ensemble members are identical:
> 7225 of 29,225 rows at `US-UMB`, where the median spread elsewhere is 0.116. The
> meaning of these rows is not established.

**Source.** [GAPFILL].

### Biomass, leaf area and soil constraints

**Format.** Two R data files, each holding a single object that nests as year,
then site, then observation. Both are lists of length 13 keyed by date from
`2012-07-15` to `2024-07-15`, and each element is a list of length 8000 named
`"1"` to `"8000"`.

In `obs.mean.Rdata`, the object `obs.mean` gives each site-year as a single-row
data frame whose columns are the variables observed there. Between zero and four
columns appear, in alphabetical order.

| Variable | Unit | Observed range, 2015 |
|---|---|---|
| `AbvGrndWood` | Mg C ha-1 | 0 to 459 |
| `LAI` | m2 m-2 | 0.1 to 6.9 |
| `SoilMoistFrac` | percent | 0.99 to 92.89 |
| `TotSoilCarb` | kg C m-2 | 5.79 to 144.4 |

In `obs.cov.Rdata`, the object `obs.cov` gives the corresponding observation error
covariances with the same nesting, as a bare numeric in the single-variable case
and a matrix otherwise. In 2012 the dimensions are 6 empty, 283 of 1x1, 4475 of
2x2 and 3236 of 3x3, matching the column counts in `obs.mean` exactly.

**Interpretation.** The July 15 keys are the annual snapshot convention of the
source product, not observation dates. No site-year entry is null, but some are
empty data frames with zero columns, denoting a site-year with no observations at
all; six such entries occur in 2012.

Coverage varies by variable and by year, so the data are not rectangular over
site, year and variable.

| Year | `TotSoilCarb` | `LAI` | `AbvGrndWood` | `SoilMoistFrac` |
|---|---|---|---|---|
| 2012 | 7990 | 7670 | 3281 | 0 |
| 2013 | 7990 | 7668 | 3281 | 0 |
| 2014 | 7990 | 7677 | 3281 | 0 |
| 2015 | 7990 | 7674 | 3281 | 7974 |
| 2016-2023 | 7990 | 7649-7678 | 3262-3281 | 7974 |
| 2024 | 7990 | 7663 | 0 | 7974 |

`SoilMoistFrac` is absent before 2015 and `AbvGrndWood` in 2024, and
`AbvGrndWood` covers about 41% of sites in the years where it is present.

The covariance matrices carry no dimension names, so the variable each row and
column refers to must be taken from the column order of the corresponding
`obs.mean` entry. The site-level lists in both objects are named, so sites can be
addressed by name rather than by position.

> **Note 9.** These files are the observation inputs to the source product's data
> assimilation, whereas the units above are documented for its published output.

> **Note 10.** Two further directories of `obs.mean` and `obs.cov` files exist
> alongside this one, and the relationship between them is not established.

**Source.** Observation inputs to the state data assimilation described in
[NALCR]. Underlying products include LandTrendr aboveground biomass.

---

## Conversion to processed form

Ingest scripts live in [`../scripts/`](../scripts). Each reads from `raw/`,
writes to `processed/`, and leaves its input unmodified. Run a script with
`--help` for usage.

| Script | Reads | Writes |
|---|---|---|
| `ingest_sites.py` | `site_ids.csv`, `site_id_map.csv`, plant functional type table | `processed/sites.parquet` |
| `ingest_ic.py` | `raw/initial_conditions/` | `processed/ic.nc` |
| `ingest_constraints.py` | R export of `obs.mean` and `obs.cov` | `processed/agb_lai.nc` |
| `ingest_nee.py` | `raw/constraints/nee/ens_ec_3h.csv` | `processed/nee.zarr` |
| `ingest_drivers.py` | `raw/drivers/` | `processed/drivers.zarr` |

Reading the R data files requires R. The `obs.mean` object is a list of lists of
data frames, which `pyreadr` does not support, so it is flattened by a one-off R
script that writes netCDF. These are the only inputs that require R.

Two conversions are applied during ingest rather than downstream: net ecosystem
exchange is converted to a single canonical unit, so nothing later in the pipeline
has to reconcile units, and the redundant `ens_mean` column is dropped.

> **Note 11.** Two plant functional type tables exist for the 8000 sites, with 16
> and 3 distinct classes respectively, and the one to use has not been chosen.

---

## Processed format

The ingest scripts are not yet written; this section records the intended output.

The processed form is also the form used throughout the rest of the project, so it
is chosen to load directly as such: an `xarray.DataArray` per variable, with
dimensions drawn from `member`, `site` and `time`, longitude and latitude as
non-dimension coordinates on `site`, and units recorded in the array's attributes.
Formats are chosen according to the shape of each product.

| Product | Format | Dimensions | Approximate size |
|---|---|---|---|
| `sites.parquet` | Parquet | table | negligible |
| `ic.nc` | netCDF | `(member, site)` | 19 MB |
| `agb_lai.nc` | netCDF | `(site, time)` per variable, plus covariances | negligible |
| `nee.zarr` | Zarr, chunked on `site` | `(member, site, time)` | 630 MB dense, about 55% missing |
| `drivers.zarr` | Zarr, chunked on `site` and `time` | `(member, site, time)` | 15 GB per member; site subsets in practice |

Zarr is used for the arrays indexed by member, site and time because it maps
directly onto the in-memory representation: `xarray.open_zarr(...).sel(site=...)`
reads only the requested sites, with no reshaping step. Lazy reads are backed by
dask.

The following conventions apply to every product.

- `site` is the integer identifier 1-8000, never renumbered. Ameriflux
  identifiers and plant functional type are non-dimension coordinates on `site`,
  and are absent where unknown.
- `member` is a zero-based integer index, meaningful only within a single source.
- Time is stored as a datetime index; SIPNET's `year`, `day` and `time` triple is
  converted at the boundary.
- Uneven coverage is preserved rather than filled. The biomass and leaf area
  product in particular is not rectangular over site, year and variable.

> **Note 12.** Whether ensemble member *i* of one source corresponds to member
> *i* of another is not established.

> **Note 13.** Whether the processed form should carry an additional
> spatially-ordered site coordinate is undecided.

---

## Open questions

Numbered notes above refer to the corresponding entry here.

**1. Per-site directory templates.** The driver and initial-condition path
templates are inferred from `ERA5_1_1/ERA5.1.2012-01-01.2024-12-31.clim` and
`initial_conditions/1/IC_site_1_1.nc`. Whether all 8000 site directories follow
them has not been checked, and ingest should fail loudly on any that do not.

**2. Misalignment of `site_name`.** Every label in `site_name` occupies a
contiguous band of latitudes, which suggests the column retains an ordering
different from the one applied to the coordinates, rather than that individual
rows are wrong. Besides `soil_core_alaska` at 33.3 to 36.0 north, four rows
labelled `Hawaii` lie between 119.6 and 86.1 west, and rows whose labels embed
Costa Rican, Panamanian and Mexican site codes all lie near 46 north. That the
coordinates themselves are sound was confirmed by joining `site_id_map.csv` to
`site_ids.csv` and checking against known locations for Park Falls, Wind River,
Sylvania and the University of Michigan Biological Station. It remains to be
determined whether the misalignment originates in the source file or in the
extraction to CSV; the latter is the cheaper possibility to rule out first.

**3. Coverage gaps between the NEE file and the identifier map.** Of the 44 sites
with NEE data but no identifier, 12 have non-US prefixes and a further 15 carry
codes belonging to Alaskan site clusters, consistent with the map covering only
the conterminous US. The remaining 17 are unexplained. The 20 sites with an
identifier but no NEE data are harder to account for, since they include
`US-Ha1`, `US-MMS`, `US-Ne1` and `US-Ne2`, all long-running sites well inside the
mapped region. The likeliest explanation is that the identifier map and the
gap-filled product were assembled from different site lists.

**4. Driver ensemble size.** Only one member is available locally, so the size of
the driver ensemble is unknown.

**5. Reference year for the initial-condition time coordinate.** The units
attribute is an unsubstituted template, so the intended reference year cannot be
recovered from the file. This does not affect calibration, since the dimension is
degenerate, but it does mean the files cannot be used for anything time-aware.
Tracked as
[issue #3](https://github.com/arob5/spatial-lsm-calibration/issues/3).

**6. Initial-condition variable sets and ensemble size.** The variable set is
reported to differ between files, with `leaf_carbon_content` and `SoilMoistFrac`
appearing in some. Only one file is available locally, so neither the full set of
combinations nor the ensemble size has been confirmed against the data.

**7. Units of the NEE ensemble columns.** This blocks any comparison between
modelled and observed net ecosystem exchange in either direction. The evidence for
umol CO2 m-2 s-1 is the magnitude of the values: a midsummer midday value of -21
in g C m-2 per three hours would be far outside the physical range, whereas -21
umol CO2 m-2 s-1 is unremarkable for a productive temperate forest. A conversion
factor between kg C m-2 s-1 and umol CO2 m-2 s-1 has been noted elsewhere, but
does not by itself establish which unit this file uses.

**8. Rows with identical ensemble members.** One reading is that zero spread marks
a directly measured value and non-zero spread a gap-filled one, which would make
the ensemble spread usable as a quality flag and would matter for the observation
error model. This has not been confirmed, and the alternative, that the
gap-filling simply produces no spread under some conditions, has quite different
consequences.

**9. Units of the assimilation inputs.** The units given for the four variables
are documented for the published reanalysis output, whereas `obs.mean.Rdata` holds
the observation inputs to that reanalysis. All four variable names and all
thirteen annual keys agree, so the two almost certainly share definitions, but
this has not been confirmed.

**10. Relationship between the observation directories.** Two further directories
of `obs.mean` and `obs.cov` files exist alongside the one used here. It is not
known how they differ or which is authoritative. The directory in use is named as
though its contents carry variable attributes, but no attributes are present on
any object within it.

**11. Choice of plant functional type table.** Two tables exist for the 8000
sites, distinguishing 16 and 3 classes respectively. Neither is present in this
repository, and the choice between them is open.

**12. Correspondence of ensemble members across sources.** Whether driver member
*i*, initial-condition member *i* and the calibration ensemble were drawn jointly
or independently determines whether arithmetic that pairs them is meaningful.
Because xarray aligns on coordinate values automatically, an incorrect assumption
here would combine unrelated members without any error being raised.

**13. Site ordering in the processed form.** The 1-8000 identifiers are fixed, but
a spatially coherent ordering, a Hilbert or Morton rank for instance, would
improve locality for triangulation and for chunked reads. Such an ordering would
be added as an additional coordinate rather than by renumbering.
