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
- **[pySIPNET]** [pySIPNET](https://github.com/TARPS-group/pySIPNET), the SIPNET
  interface package, whose `pysipnet.climate` and `pysipnet.io.clim_io` modules
  define the SIPNET climate file format.
- **[GAPFILL]** Gap-filled eddy-covariance net ecosystem exchange, produced by
  Yang Gu at Boston University. Unpublished; no citable reference at present.

### Relationship to the reanalysis

This project shares a number of inputs with [NALCR]: the 8000-site pool and the
grid it is defined on, the ERA5 driver ensembles, the initial condition
ensembles, and the biomass, leaf area and soil constraint files. It does not take
the reanalysis output as an input. The two analyses draw on overlapping inputs
rather than one feeding the other, and they make different use of them: [NALCR]
assimilates these data into a state reanalysis, whereas the work here uses them
to calibrate model parameters.

[NALCR] is referenced throughout because it is where the site pool originates and
because its documentation is the fullest available description of several of the
shared inputs. Where a statement below rests on documentation of the reanalysis
output rather than on the input files themselves, that is marked with a note.

---

## Directory layout

Ingest expects the following structure. Site identifiers are the integers 1-8000
described under [Site metadata](#site-metadata), and `<member>` indexes an
ensemble member.

```
data/
  site_id_map.csv               Ameriflux identifier map
  raw/
    sites/                      site table, tracked in version control
      pts.shp, pts.shx, pts.dbf, pts.prj, pts.cpg
    drivers/
      ERA5_<site_id>_<member>/ERA5.<member>.2012-01-01.2024-12-31.clim
    initial_conditions/
      <site_id>/IC_site_<site_id>_<member>.nc
    constraints/
      nee/ens_ec_3h.csv
      sda_8k_site_rdata/obs.mean.Rdata
      sda_8k_site_rdata/obs.cov.Rdata
  processed/                    ingest output, created by the ingest scripts
    sites/sites.csv
    ic.nc
    agb_lai.nc
    nee.zarr/
    drivers.zarr/
```

> **Note 1.** The two per-site directory templates are inferred from a single
> example of each rather than confirmed across all 8000 sites.

Files under `raw/` are treated as read-only; all conversion happens on the way
into `processed/`, which is regenerable and absent on a fresh clone. Neither
directory is tracked in version control, with one exception: `raw/sites/` holds
the site shapefile, which is small, is a primary source rather than a pipeline
output, and is the one input without which the repository carries no site
information at all. `site_id_map.csv` is tracked for the same reason.

---

## Site metadata

The site pool is defined by a point shapefile under `raw/sites/`, described here
alongside the Ameriflux identifier map. The pool and its identifiers were defined
for the model runs underlying [NALCR] and are shared with collaborators' files, so
the identifiers are treated as fixed and are never renumbered.

### `raw/sites/pts.shp` and companions

The site table as an ESRI point shapefile: 8000 `Point` records in `pts.shp`, with
`pts.shx`, `pts.dbf`, `pts.prj` and `pts.cpg` alongside. Record *N* corresponds to
site identifier *N*. This is the only site source in the repository, and the only
file under `raw/` that is tracked in version control: it is small, it is a primary
source rather than a pipeline output, and without it the repository carries no site
information at all.

Geometry is in geographic coordinates on the WGS 84 datum, declared by `pts.prj`.
Records are ordered by descending latitude. Coordinates span -178.754 to -20.013
in longitude and 7.013 to 82.546 in latitude; the extremes are site 731 in the
western Aleutians, site 1 in northeast Greenland, and site 8000 at 7.01 north in
northern South America. 3640 of the 8000 sites fall inside a conterminous-US
bounding box of 24-50 north and 125-66 west, so analyses restricted to that region
use a little under half the pool. The remainder are distributed as follows.

| Region | Sites |
|---|---|
| north of 60 | 2517 |
| north of 70 | 475 |
| north of 80 | 36 |
| south of 20 | 412 |
| west of 140 | 1289 |
| east of 50 west | 31 |

The attribute table in `pts.dbf` holds five fields.

| Field | Type | Description |
|---|---|---|
| `site_id` | numeric | Site identifier, 1-8000, in record order |
| `site_names` | character | Site label |
| `site_order` | numeric | 0 for sampled points, 1-1093 for named sites |
| `cluster` | numeric | Sampling stratum, six classes; see Note 2 |
| `landcover` | numeric | Land cover class, eight classes; see Note 2 |

`site_order` distinguishes the two kinds of site in the pool: 1093 named
locations, being flux towers, research stations and soil cores, and 6907 points
labelled `weighted_sample` drawn to fill out the sample. Named sites carry values
forming a permutation of 1 to 1093; sampled points carry 0.

`cluster` and `landcover` together appear to define the strata the sampled points
were drawn from. Their cross-tabulation populates 35 of 48 cells, and the empty
cells form a staircase rather than being scattered: clusters 1 to 3 span all eight
land cover classes, cluster 4 lacks class 5, cluster 5 holds only classes 1, 3 and
8, and cluster 6 only class 1. Several columns hold near-equal counts across
clusters, land cover class 3 standing at 40 or 41 in each of clusters 1 to 5,
which is the signature of a per-stratum sampling target truncated where too few
candidate cells were available.

`cluster` is not a geographic partition. Mean within-cluster pairwise distance
ranges from 1636 to 3455 km against 3098 km for the pool as a whole, so cluster 4
is more dispersed than the pool average. Whatever was clustered was not location.
Cluster 6 is the one exception, confined to 25.8-49.1 north and 124-76 west, and
to a single land cover class. `landcover`, by contrast, behaves as a land cover
classification should: class 3 appears only between 42.8 and 69.0 north, class 7
is spatially compact, and class 6 spans the full latitude range.

> **Note 2.** The variables underlying `cluster`, and the classification scheme
> behind `landcover`, are not documented in the available sources.

### `site_id_map.csv`

185 rows mapping Ameriflux site identifiers onto the integer site identifiers, by
exact match.

| Column | Type | Description |
|---|---|---|
| `Site_ID` | string | Ameriflux site identifier, for example `US-xDC` |
| `index` | integer | Corresponding `site_id` |

All mapped sites lie within 25.35-47.16 north and 122.33-68.74 west, so the map
covers the conterminous US only. Because site identifiers run in descending
latitude, the mapped range 4102-7418 is a contiguous latitude band.

This file and the [GAPFILL] net ecosystem exchange file do not cover the same
sites: the NEE file contains 209 Ameriflux sites and this map contains 185, with
165 in common, so calibration constrained by NEE is limited to those 165 sites.
The producer attributes the omissions to three causes: some sites did not pass a
validation test, some lacked the half-hourly data the gap-filling requires, and
some resolved to the same model site identifier as another site. An updated
release of the product covers more sites; see Note 7.

> **Note 3.** Because more than one Ameriflux site can fall within a single grid
> cell, several may resolve to the same model site identifier. This file contains
> no repeated identifiers, so such cases appear to have been dropped rather than
> merged; how they should be handled is undecided.

---

## Coordinate reference system

Site coordinates are geographic, on the WGS 84 datum (EPSG:4326). Two independent
statements agree on this: the [NALCR]
[dataset guide](https://daacweb-prod.ornl.gov/CMS/guides/Land_C_Reanalysis_NorthAmerica.html),
and `raw/sites/pts.prj`, which declares WGS 84 and no projected coordinate system.
Note that `pts.prj` is written in the Esri flavour of WKT and carries no
`AUTHORITY` token, so a reader that resolves codes strictly will not recognise it
as EPSG:4326 without matching on names.

Site coordinates are cell centres of the approximately 1 km geographic grid on
which [NALCR] is also defined: 0.008333 degree, or 30 arcsecond, resolution in
both latitude and longitude, spanning 179 west to 20 west and 7 north to 85 north
as a 19080 by 9360 array. Those figures are mutually consistent to the cell:
(20 - 179) x 120 is exactly 19080 and (85 - 7) x 120 exactly 9360. Cell centres
lie at

    lon = -179 + (lon_idx + 0.5)/120,   lat = 7 + (lat_idx + 0.5)/120

for zero-based indices, which is where the `lon_idx` and `lat_idx` columns of the
processed site table come from.

All 8000 sites fall on cell centres, but only to within 1.02e-6 degrees, about
0.11 m. The residual is consistent with the coordinates having passed through
32-bit floating point somewhere upstream: site 1's stored latitude is
82.5458343506 where the exact centre is 82.5458333333. The integer indices are
therefore the exact representation of a site's position and the stored
coordinates are a lossy rendering of it, which is why both are carried.

The grid is defined in code as `SITE_GRID` in
[`sipnet_calibration.sites`](../src/sipnet_calibration/sites.py), together with
the conversions between coordinates and indices. It lives there rather than in a
data file because the constants and the two functions that use them must not be
able to disagree.

Two consequences are worth noting.

- The grid is not equal-area. Thirty arcseconds is about 928 m in latitude
  everywhere, but in longitude it ranges from roughly 921 m at 7 north to 121 m
  at 82.5 north. Density and per-area calculations must account for this.
- Coordinates are stored longitude before latitude, which is the traditional GDAL
  and PROJ ordering rather than the axis order EPSG:4326 formally declares.
  Coordinate transformations should be configured accordingly, for instance with
  pyproj's `always_xy=True`.

The map projection used for plotting is a separate choice from the coordinate
system of the input data. It is tracked in
[issue #4](https://github.com/arob5/spatial-lsm-calibration/issues/4) and will be
documented once settled.

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

**Source.** ERA5, prepared for the 8000-site pool for the model runs underlying
[NALCR]. The same driver files are used here.

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

**Source.** Initial condition ensembles prepared for the 8000-site pool for the
model runs underlying [NALCR]. The same files are used here. The published
reanalysis output carries 100 ensemble members together with ensemble mean and
standard deviation; whether the initial condition files use the same ensemble
size has not been confirmed against the data, and is the subject of Note 6.

### Net ecosystem exchange

**Format.** A single CSV file of about 1.7 GB, with 3,547,921 data rows and 28
columns.

| Column | Type | Description |
|---|---|---|
| `Site_ID` | string | Ameriflux site identifier; 209 distinct values |
| `utc` | timestamp | ISO 8601, 3-hourly, 2012-01-01T03:00:00Z to 2025-01-01T03:00:00Z |
| `ens01` to `ens25` | float | Twenty-five ensemble members |
| `ens_mean` | float | Mean of `ens01` to `ens25` |

**Interpretation.** Values are in micromoles of CO2 per square metre per second
(umol CO2 m-2 s-1), a rate, as confirmed by the producer. SIPNET reports net
ecosystem exchange as a per-timestep total in g C m-2, so ingest must convert
between the two.

Each of the 25 member columns is the gap-filled series obtained from one member
of a driver ensemble. The spread across members therefore reflects the
propagation of driver uncertainty through the gap-filling procedure, and not
measurement error or uncertainty in the gap-filling model itself. `ens_mean` is
the average of the 25 members, exactly so to within 8e-14 in the site checked,
and is therefore redundant. It must not be carried as a 26th ensemble member,
since that biases any quantile computed across members.

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

> **Note 7.** An updated release of this product covers 217 sites, including
> several absent here, but carries only the ensemble mean. Which release to use is
> undecided.

> **Note 8.** In about a quarter of rows, all 25 ensemble members are identical:
> 7225 of 29,225 rows at `US-UMB`, where the median spread elsewhere is 0.116.
> Since the members differ only in the driver realisation used for gap-filling,
> this is consistent with a directly measured timestep, which requires no
> gap-filling, but that has not been confirmed.

**Source.** [GAPFILL].

### Biomass, leaf area and soil constraints

**Format.** Two R data files, each holding a single object that nests as year,
then site, then observation. Both are lists of length 13 keyed by date from
`2012-07-15` to `2024-07-15`, and each element is a list of length 8000 named
`"1"` to `"8000"`.

In `obs.mean.Rdata`, the object `obs.mean` gives each site-year as a single-row
data frame whose columns are the variables observed there. Between zero and four
columns appear, in alphabetical order. The units below are those documented in
the [NALCR] dataset guide for the corresponding variables of the reanalysis
output; see Note 9.

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

> **Note 9.** The units above are documented for the reanalysis output, whereas
> these files are the observation inputs to that reanalysis.

> **Note 10.** Two further directories of `obs.mean` and `obs.cov` files exist
> alongside this one, and the relationship between them is not established.

**Source.** The observation files assimilated by [NALCR]; the same files are used
here as calibration constraints. Underlying products include LandTrendr
aboveground biomass.

---

## Conversion to processed form

Ingest scripts live in [`../scripts/`](../scripts). Each reads from `raw/`,
writes to `processed/`, and leaves its input unmodified. Run a script with
`--help` for usage.

| Script | Reads | Writes |
|---|---|---|
| `ingest_sites.py` | `raw/sites/pts.*`, `site_id_map.csv`, plant functional type table | `processed/sites/sites.csv` |
| `ingest_ic.py` | `raw/initial_conditions/` | `processed/ic.nc` |
| `ingest_constraints.py` | R export of `obs.mean` and `obs.cov` | `processed/agb_lai.nc` |
| `ingest_nee.py` | `raw/constraints/nee/ens_ec_3h.csv` | `processed/nee.zarr` |
| `ingest_drivers.py` | `raw/drivers/` | `processed/drivers.zarr` |

Reading the R data files requires R. The `obs.mean` object is a list of lists of
data frames, which `pyreadr` does not support, so it is flattened by a one-off R
script that writes netCDF. These are the only inputs that require R.

Two conversions are applied during ingest rather than downstream. Net ecosystem
exchange is converted from umol CO2 m-2 s-1 to the canonical unit used
throughout, so that nothing later in the pipeline has to reconcile units; and the
redundant `ens_mean` column is dropped.

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
| `sites/sites.csv` | CSV | table | ~1 MB |
| `ic.nc` | netCDF | `(member, site)` | 19 MB |
| `agb_lai.nc` | netCDF | `(site, time)` per variable, plus covariances | negligible |
| `nee.zarr` | Zarr, chunked on `site` | `(member, site, time)` | 630 MB dense, about 55% missing |
| `drivers.zarr` | Zarr, chunked on `site` and `time` | `(member, site, time)` | 15 GB per member; site subsets in practice |

Zarr is used for the arrays indexed by member, site and time because it maps
directly onto the in-memory representation: `xarray.open_zarr(...).sel(site=...)`
reads only the requested sites, with no reshaping step. Lazy reads are backed by
dask. The site table is CSV instead because it is small, tabular and read by
people as often as by code.

`sites/sites.csv` carries every field of the shapefile, so that nothing is lost in
translation, together with the grid indices:

| Column | Type | Description |
|---|---|---|
| `site_id` | integer | Site identifier, 1-8000 |
| `lon`, `lat` | float | Coordinates, written at 17 significant digits |
| `lon_idx`, `lat_idx` | integer | Zero-based indices on the 1/120 degree grid |
| `site_name` | string | From the shapefile's `site_names` |
| `site_order` | integer | 0 for sampled points, 1-1093 for named sites |
| `cluster`, `landcover` | integer | Sampling stratum and land cover class |

The grid indices are the exact representation of a site's position: reconstructing
`lon` and `lat` from them differs from the stored floats by up to 1.0e-6 degrees,
which is the departure of the stored values from true cell centres rather than an
error in the reconstruction. Ingest writes floats at full round-trip precision and
asserts that reading them back reproduces the shapefile values exactly, since CSV
formatting is the one place this table can silently lose information.

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
> *i* of another is not established, though the net ecosystem exchange members are
> known to derive from a driver ensemble.

> **Note 13.** Whether the processed form should carry an additional
> spatially-ordered site coordinate is undecided.

---

## Open questions

Numbered notes above refer to the corresponding entry here.

**1. Per-site directory templates.** The driver and initial-condition path
templates are inferred from `ERA5_1_1/ERA5.1.2012-01-01.2024-12-31.clim` and
`initial_conditions/1/IC_site_1_1.nc`. Whether all 8000 site directories follow
them has not been checked, and ingest should fail loudly on any that do not.

**2. Meaning of the `cluster` and `landcover` fields.** Neither is documented in
the sources available. The evidence that they define sampling strata is
circumstantial but consistent: their cross-tabulation leaves 13 of 48 cells empty
in a staircase pattern rather than at random, several columns hold near-equal
counts across clusters, and the 6907 non-named sites are labelled
`weighted_sample`. Against a geographic reading, mean within-cluster pairwise
distance runs from 1636 to 3455 km where the pool as a whole averages 3098, so
cluster 4 is more dispersed than the pool and the grouping cannot be spatial.

The reanalysis preprint describes the pool as "8,000 pre-selected 1km²
locations", which suggests the selection procedure is documented in earlier work
or in supplementary material rather than in that paper. `landcover` resolves eight
classes, fewer than the seventeen of the IGBP scheme, so it is likely an
aggregation. Confirming both would take one question to the group that produced
the site pool.

**3. Sites resolving to the same model identifier.** The 8000-site pool is a
subsample of a roughly 1 km grid, so two eddy-covariance towers close together can
fall in the same cell and resolve to one model site. The producer of [GAPFILL]
gives this as one reason sites were omitted from the identifier map, and this file
contains no repeated identifiers, which suggests such cases were dropped. If a
later release retains them, the calibration has to decide what two observation
series attached to a single model prediction mean: whether both enter the
likelihood, whether they are averaged first, and how the observation error
covariance should treat them. This is a modelling question rather than a data one,
and is unresolved.

**4. Driver ensemble size.** Only one member is available locally, so the size of
the driver ensemble has not been confirmed. The gap-filling behind [GAPFILL] used
25 driver members, and the reanalysis output carries 100, so neither figure can be
assumed for the driver files themselves.

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

**7. Which release of the gap-filled product to use.** An updated release exists,
combining the identifier map and the observations in a single file covering 217
eddy-covariance sites. It includes sites absent from the release documented here,
among them `US-Ha1`, and its site matching supersedes `site_id_map.csv`. It
carries only the ensemble mean, however, so adopting it exchanges the ensemble
spread for wider site coverage.

That trade matters because the spread is currently the only per-observation
uncertainty available for net ecosystem exchange, and an observation error
covariance has to come from somewhere. Whether the ensemble can be obtained for
the updated results, rather than only its mean, would settle the question; failing
that, the choice is between coverage and a quantified uncertainty. Neither file is
present in this repository.

**8. Rows with identical ensemble members.** The members differ only in the
driver realisation used for gap-filling, so a timestep that was measured directly,
and therefore needed no gap-filling, would be expected to take the same value in
every member. That would make zero spread a usable flag for measured values, which
matters for the observation error model, since measured and imputed values should
not carry equal weight. The producer has not confirmed this reading, and it does
not carry over to the updated release, which has no ensemble.

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
sites, distinguishing 16 and 3 classes respectively, and neither is present in
this repository. The `landcover` field of the site shapefile is a third
classification, with eight classes, and is present. Which to use, and how the
three relate, is open; see also Note 2.

**12. Correspondence of ensemble members across sources.** Whether driver member
*i*, initial-condition member *i* and the calibration ensemble were drawn jointly
or independently determines whether arithmetic that pairs them is meaningful.
Because xarray aligns on coordinate values automatically, an incorrect assumption
here would combine unrelated members without any error being raised. One
connection is known: the members of [GAPFILL] are gap-filled series driven by
successive members of a driver ensemble, so if that is the same ensemble used here,
net ecosystem exchange member *i* and driver member *i* would share a realisation.
Whether it is the same ensemble has not been established.

**13. Site ordering in the processed form.** The 1-8000 identifiers are fixed, but
a spatially coherent ordering, a Hilbert or Morton rank for instance, would
improve locality for triangulation and for chunked reads. Such an ordering would
be added as an additional coordinate rather than by renumbering.
