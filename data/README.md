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
```

The drivers have no processed form. SIPNET reads the raw `.clim` files itself,
so `sipnet_calibration.drivers.load_drivers` produces the canonical
`(member, site, time)` form from `raw/drivers/` on demand instead; see
[Meteorological drivers](#meteorological-drivers) and
[Processed format](#processed-format).

> **Note 1.** The per-site directory templates are inferred from three driver
> directories and three initial-condition files rather than confirmed across
> all 8000 sites.

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
in longitude and 7.013 to 82.546 in latitude. The four extremes are site 731 at
68.51 N on the Chukchi coast (westernmost, 178.75 W), site 115 at 77.09 N in
northeast Greenland (easternmost, 20.01 W), site 1 at 82.55 N, also in northeast
Greenland (northernmost), and site 8000 at 7.01 N in northern South America
(southernmost). 3640 of the 8000 sites fall inside a conterminous-US
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
labeled `weighted_sample` drawn to fill out the sample. Named sites carry values
forming a permutation of 1 to 1093; sampled points carry 0.

Three properties of the attribute table matter to anything that reads it, and
each is quiet when it goes wrong.

`pts.cpg` declares UTF-8, and exactly two of the 8000 names carry non-ASCII
bytes: site 7176 `Rayón (MX-Ray)` and site 7813
`Estación Experimental Forestal Horizontes`. Read as latin-1 neither raises;
both decode to a string that still looks like a plausible site label. The
declared encoding is therefore asserted at ingest rather than left to a library
default.

Eight sites are named literally `NA`: 3392, 7484, 7542, 7589, 7595, 7607, 7616
and 7617. A CSV reader using its default missing-value strings turns these into
nulls, so the site table is read with `keep_default_na=False`.

`cluster`, `landcover` and `site_order` are declared in the `.dbf` as numerics
with 15 decimal places, so a reader returns them as floats even though every
value is a whole number. They are cast at ingest rather than written through.

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

#### Co-located instruments

**A mapped identifier can stand for a location where several towers operate, and
the tower this file selects is not always the one the site name suggests.**

The shapefile's `site_names` field also carries Ameriflux identifiers. 57 of the
185 mapped sites embed one in parentheses; 53 match this file and 4 do not.

| `site_id` | `site_name` | This file | Embedded in the name |
|---|---|---|---|
| 4326 | Park Falls WLEF (US-PFa) | `US-PFp` | `US-PFa` |
| 4705 | Bartlett Experimental Forest (US-Bar) | `US-xBR` | `US-Bar` |
| 5584 | Blandy Experimental Farm (US-Bef) | `US-xBL` | `US-Bef` |
| 5758 | Sherman Island (US-Snd) | `US-Sne` | `US-Snd` |

These are co-located instruments, not transcription errors. 35 of the 185 mapped
identifiers begin `US-x`, as do two of the four disagreements; taking that prefix
to mean NEON, those two are a NEON tower and an Ameriflux tower sharing a cell.
The other two pair Ameriflux towers at one location. A further 163 unmapped sites
embed an identifier in their name, so the name field is not a fallback for the
mapping.

The choice of instrument is made here, per site, and recorded nowhere else. See
Note 3, and Note 7 on the release that supersedes this map.

> **Note 3.** Because more than one Ameriflux site can fall within a single grid
> cell, several may resolve to the same model site identifier. This file contains
> no repeated identifiers, so such cases appear to have been dropped rather than
> merged; how they should be handled is undecided. The table above is the visible
> trace: cells where a choice between co-located towers was made silently.

---

## Coordinate reference system

Site coordinates are geographic, on the WGS 84 datum (EPSG:4326). Two independent
statements agree on this: the [NALCR]
[dataset guide](https://daacweb-prod.ornl.gov/CMS/guides/Land_C_Reanalysis_NorthAmerica.html),
and `raw/sites/pts.prj`, which declares WGS 84 and no projected coordinate system.
Note that `pts.prj` is written in the Esri flavor of WKT and carries no
`AUTHORITY` token, so a reader that resolves codes strictly will not recognize it
as EPSG:4326 without matching on names.

Site coordinates are cell centers of the approximately 1 km geographic grid on
which [NALCR] is also defined: 0.008333 degree, or 30 arcsecond, resolution in
both latitude and longitude, spanning 179 west to 20 west and 7 north to 85 north
as a 19080 by 9360 array. Those figures are mutually consistent to the cell:
(20 - 179) x 120 is exactly 19080 and (85 - 7) x 120 exactly 9360. Cell centers
lie at

    lon = -179 + (lon_index + 0.5)/120,   lat = 7 + (lat_index + 0.5)/120

for zero-based indices, which is where the `lon_index` and `lat_index` columns
of the processed site table come from.

All 8000 sites fall on cell centers, but only to within 1.02e-6 degrees, about
0.11 m. The residual is consistent with the coordinates having passed through
32-bit floating point somewhere upstream: site 1's stored latitude is
82.5458343506 where the exact center is 82.5458333333. The integer indices are
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

### Display projection

The projection used for spatial figures is a separate choice from the coordinate
system of the input data, and is settled: a **Lambert Azimuthal Equal Area
centered at 50 N, 100 W**, on WGS 84, in meters, with no false origin.

    +proj=laea +lat_0=50 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs

The definition lives in code, as `SITE_PROJECTION` in
[`sipnet_calibration.projection`](../src/sipnet_calibration/projection.py),
which also provides the forward transform and writes the same definition as
PROJJSON and as a PROJ string under
`src/sipnet_calibration/projections/` for tools outside this package. Those
files are generated from the dataclass and checked against it by the test
suite, so they cannot drift from the transform; regenerate them with
`python -m sipnet_calibration.projection --write`.

Three points about the choice, with the full analysis and the measured
distortion over all 8000 sites recorded in
[issue #4](https://github.com/arob5/spatial-lsm-calibration/issues/4).

- **It is not the projection the reanalysis figures used.** Those used the USA
  Contiguous Albers Equal Area Conic (ESRI:102003), which is area-true
  everywhere but is defined for a region of predominant east-west expanse. Over
  this site pool, which spans 75 degrees of latitude, its shape distortion
  reaches 107 degrees of angular deformation and a 9:1 local anisotropy at the
  northernmost sites. The projection adopted here holds angular deformation
  under 14 degrees and anisotropy under 1.3 everywhere; both ceilings are
  asserted against this site table in `tests/test_projection.py`.
- **One projection serves both the conterminous-US and the full-domain
  figures**, so that panels are comparable. It costs the CONUS figure almost
  nothing relative to 102003, and 102003 costs the full-domain figure a great
  deal.
- **No datum transformation is involved.** The projection's base CRS is WGS 84,
  matching the site coordinates above, so nothing is shifted. Had a NAD83-based
  definition been adopted, the mismatch would have amounted to the roughly 2 m
  between the two datums, which is about 1e-4 of a pixel at the width of these
  figures.

Named longitude/latitude boxes for the regions the figures use — `CONUS`,
`NORTH_AMERICA` and `ALASKA` — are `EXTENTS` in
[`sipnet_calibration.sites`](../src/sipnet_calibration/sites.py), beside the
site selection that takes the same form, so a figure and the sites it plots
cannot disagree about what a region means.

---

## Raw inputs

### Meteorological drivers

**Format.** ERA5 reanalysis written in the SIPNET climate format: text with no
header row, one row per timestep, the fields separated by tabs and padded with
spaces. [pySIPNET] describes the format as space-delimited, and SIPNET itself
accepts either. Each of the three files present has 37,992 rows and 14 columns,
covering 2012-01-01 to the end of 2024 on a 3-hourly timestep. That row count is
4749 days, being thirteen years including four leap years, at eight timesteps
per day, and the `year`, `day`, `time` and `length` columns are identical across
the three files.

Columns follow the 14-column layout defined by [pySIPNET].

| # | Column | Unit | Description |
|---|---|---|---|
| 1 | `loc` | | Location index; constant, and required by SIPNET to be so |
| 2 | `year` | | Integer year |
| 3 | `day` | | Integer day of year, 1 = 1 January |
| 4 | `time` | hours | Hour-of-day label of the timestep; drifts and is not a timestamp, see Note 15 |
| 5 | `length` | days | Timestep duration; 0.125, that is 3 hours |
| 6 | `tair` | deg C | Mean air temperature |
| 7 | `tsoil` | deg C | Mean soil temperature |
| 8 | `par` | mol m-2 | Photosynthetically active radiation, integrated over the timestep |
| 9 | `precip` | mm | Total precipitation over the timestep |
| 10 | `vpd` | Pa | Vapor pressure deficit |
| 11 | `vpd_soil` | Pa | Soil-air vapor pressure deficit |
| 12 | `vpress` | Pa | Vapor pressure in the canopy airspace |
| 13 | `wspd` | m s-1 | Mean wind speed |
| 14 | `soil_wetness` | | Legacy column, ignored by SIPNET; constant 0.6 |

**Interpretation.** There is no datetime column; time is given by the `year`,
`day` and `time` triple. The `time` column cannot be used as written (Note 15),
so timestamps are assembled from `year`, `day` and the row's position within its
day by `sipnet_calibration.obs_ops.sipnet_time_index`. What clock those labels
are on, and whether a label marks the start or the end of its three hours, is
inferred rather than documented (Note 16). Two columns are integrated
quantities rather than rates: `par` and `precip` are totals over the timestep,
so temporal aggregation of either is a sum rather than a mean. SIPNET requires
`vpd` and `wspd` to be strictly positive and silently clamps values that are
not; the files also hold small negative excursions of `par` and `precip` around
zero (Note 17). `sipnet_calibration.drivers.load_drivers` leaves all of these
unchanged and counts them in the variable attributes. The column units are the
ones the format documents, which is what SIPNET
assumes when it reads the file; the producer has not confirmed them, and the
NALCR guide describes the forcing differently (Note 18).

> **Note 15.** The `time` column is hour-of-day computed by reducing a
> whole-year `linspace` modulo 24 with an off-by-one endpoint: for a year of
> `n` days, `linspace(0, 24 n - 1, 8 n) % 24` reproduces it to 5e-7 h in all
> three files. The label steps by 3.000685 h rather than 3, so it is two hours
> late by the last slot of each year, resets at the year boundary, and is not
> monotone within a year. Tracked as
> [issue #9](https://github.com/arob5/spatial-lsm-calibration/issues/9).

> **Note 16.** The drivers are on a longitude-tracking clock consistent with
> UTC, and the value in the row labeled hour *h* covers the three hours ending
> at *h*. Both statements are inferred from the diurnal PAR cycle at sites 1
> and 27, 54 degrees of longitude apart, not confirmed by the producer.

> **Note 17.** `par` takes exactly two negative values, -1.926e-15 and
> -1.374e-05, the latter in about 1350 rows per file, almost all between
> October and March at these two polar sites; `precip` negatives are all
> exactly -6.939e-15. `vpd_soil` is exactly zero in about 30 percent of rows
> in every file. What produces these is not known.

> **Note 18.** The [NALCR] dataset guide says the reanalysis ran SIPNET on
> "hourly meteorological forcing from the ERA5 atmospheric reanalysis". These
> files are 3-hourly.

> **Note 4.** The driver ensemble has **10 members**. The three
> directories present locally are members 1, 2 and 5, so this cannot be
> seen from this checkout; the figure is confirmed for the project rather
> than inferred from the files.

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

> **Note 6.** The initial-condition ensemble has **100 members**, the same size
> as the published reanalysis output. Confirmed for the project rather than
> inferred from the files: this checkout holds three of the 800,000, and the
> highest member index among them is 94. Which variables appear in which files
> is still not established; that can only be answered where the files are.

**Source.** Initial condition ensembles prepared for the 8000-site pool for the
model runs underlying [NALCR]. The same files are used here. The published
reanalysis output carries 100 ensemble members together with ensemble mean and
standard deviation, and the initial condition files use the same ensemble size.
That could not be confirmed against the data available here; see Note 6.

### Net ecosystem exchange

**Format.** A single CSV file of about 1.7 GB, with 3,547,921 data rows and 28
columns.

| Column | Type | Description |
|---|---|---|
| `Site_ID` | string | Ameriflux site identifier; 209 distinct values |
| `utc` | timestamp | ISO 8601, 3-hourly, 2012-01-01T03:00:00Z to 2025-01-01T03:00:00Z |
| `ens01` to `ens25` | float | Twenty-five ensemble members |
| `ens_mean` | float | Mean of `ens01` to `ens25` |

**Interpretation.** Values are in micromoles of CO2 per square meter per second
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

> **Note 8.** In about a third of rows, all 25 ensemble members are identical:
> 10,273 of 29,225 rows at `US-UMB`. Across the remaining rows the median
> standard deviation over members is 0.147 and the median range is 0.565; over
> all rows the median standard deviation is 0.067.
> Since the members differ only in the driver realization used for gap-filling,
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
Summed over all thirteen snapshots the observation counts are `TotSoilCarb`
103,870, `LAI` 99,632, `SoilMoistFrac` 79,740 and `AbvGrndWood` 39,273, or
322,515 observations in total.

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

Ingest scripts live in [`../scripts/`](../scripts). Each reads from `raw/`
(and, for `ingest_sites.py`, the tracked `site_id_map.csv` beside it), writes to
`processed/`, and leaves its input unmodified. Run a script with
`--help` for usage.

| Script | Reads | Writes |
|---|---|---|
| `ingest_sites.py` | `raw/sites/pts.*`, `site_id_map.csv` | `processed/sites/sites.csv` |
| `export_constraints.R` | `raw/constraints/sda_8k_site_rdata/obs.{mean,cov}.Rdata` | a long CSV and a JSON manifest |
| `ingest_constraints.py` | that CSV and manifest, `processed/sites/sites.csv` | `processed/constraints_annual.nc` |
| `ingest_ic.py` | `raw/initial_conditions/` | `processed/ic.nc` |
| `ingest_nee.py` | `raw/constraints/nee/ens_ec_3h.csv` | `processed/nee.zarr` |

The drivers have no ingest script. SIPNET runs read the raw `.clim` files, so
converting 80,000 of them into a store would produce a large copy the model
never reads; `sipnet_calibration.drivers.load_drivers` parses the raw files for
the sites a caller names and returns the canonical form directly. A cached
subset, where a workflow wants one, is the caller's `to_zarr`.

Reading the R data files requires R, and they are the only inputs that do.
`obs.mean` is a list of lists of data frames, which `pyreadr` does not support,
and R's `ncdf4` is not installed on the development machine, so R cannot write
the netCDF either. `export_constraints.R` therefore does only what R must --
read the objects and flatten them to one row per observed
`(snapshot, site, variable)` triple -- and `ingest_constraints.py` makes every
schema decision. The intermediate CSV is 322,515 rows and about 19 MB; it is
scratch, not a product, and belongs outside `processed/`.

Alongside the CSV, `export_constraints.R` writes a JSON manifest of what it
checked: per-snapshot per-variable row counts, the exact extremes per variable,
the empty site-snapshots, and the largest off-diagonal covariance element it
saw. `ingest_constraints.py` checks the CSV against that manifest and refuses to
write if the two disagree. The manifest exists because **the diagonality of the
covariances can only be checked in R** -- by the time the CSV exists the
off-diagonal is gone -- and the processed form stores variances rather than
matrices, which is lossless exactly when they are diagonal.

Conversions applied during ingest rather than downstream:

- **Constraints.** The covariance matrices are reduced to their diagonals. This
  is lossless and asserted, not assumed. Zero variances are written through
  unchanged; 929 `AbvGrndWood` variances are exactly zero. 925 of those sit
  where the observation is zero too, but four assert a non-zero value with no
  uncertainty at all: sites 5664 (2014), 6558 (2016) and 7167 (2015 and 2016),
  all with a mean of 1.0. Either way flooring them is a modeling decision that
  would be hidden if an ingest script made it. See open question 14.
- **Net ecosystem exchange.** Converted from umol CO2 m-2 s-1 to the canonical
  unit used throughout, so that nothing later has to reconcile units, and the
  redundant `ens_mean` column is dropped.

> **Note 11.** Plant functional type is not site metadata and is not a column
> of the site table. Which labeling a calibration uses, and how many exist, is
> an experimental choice; see Note 11 under Open questions.

---

## Processed format

`ingest_sites.py` and `ingest_constraints.py` are written, and the driver
reader in `sipnet_calibration.drivers` is implemented; the rest of this section
records the intended output of scripts not yet written.

The processed form is also the form used throughout the rest of the project, so it
is chosen to load directly as such: an `xarray.DataArray` per variable, with
dimensions drawn from `member`, `site` and `time`, longitude and latitude as
non-dimension coordinates on `site`, and units recorded in the array's attributes.
Formats are chosen according to the shape of each product.

| Product | Format | Dimensions | Approximate size |
|---|---|---|---|
| `sites/sites.csv` | CSV | table | ~1 MB |
| `constraints_annual.nc` | netCDF | `(site, time, variable)` for the mean and the variance | 2.2 MB |
| `ic.nc` | netCDF | `(member, site)` | 32 MB at 100 members |
| `nee.zarr` | Zarr, chunked on `site` | `(member, site, time)` | 630 MB dense, about 55% missing |
| drivers | no file; `load_drivers()` over `raw/drivers/` | `(member, site, time)` | about 2.4 MB per site-member in memory |

Zarr is used for the arrays indexed by member, site and time because it maps
directly onto the in-memory representation: `xarray.open_zarr(...).sel(site=...)`
reads only the requested sites, with no reshaping step. Lazy reads are backed by
dask. The site table is CSV instead because it is small, tabular and read by
people as often as by code.

`sites/sites.csv` carries every field of the shapefile, so that nothing is lost in
translation, together with the grid indices and the Ameriflux identifier:

| Column | Type | Description |
|---|---|---|
| `site_id` | int32 | Site identifier, 1-8000, in shapefile record order |
| `lon`, `lat` | float64 | Coordinates, at full round-trip precision |
| `lon_index`, `lat_index` | int32 | Zero-based indices on the 1/120 degree grid |
| `site_name` | string | From the shapefile's `site_names`, renamed to the singular |
| `site_order` | int32 | 0 for sampled points, 1-1093 for named sites |
| `cluster`, `landcover` | int8 | Sampling stratum and land cover class |
| `ameriflux_site_id` | string | From `site_id_map.csv`; empty for the 7815 unmapped sites |

A `.dbf` null becomes the empty string in `site_name`, matching what an empty
`ameriflux_site_id` means, and is an error in any numeric column: the integer
columns cannot hold a missing value, and none of them has a spare code for one.

There is deliberately no `pft` column; see Note 11. The Ameriflux column is
renamed from that file's `Site_ID`, which is opaque about which of the two
identifiers it means, and is provisional in that a newer release supersedes the
map it comes from; see open question 7.

The grid indices are the exact representation of a site's position: reconstructing
`lon` and `lat` from them differs from the stored floats by up to 1.02e-6 degrees,
which is the departure of the stored values from true cell centers rather than an
error in the reconstruction. Ingest writes floats at full round-trip precision and
asserts that reading them back reproduces the shapefile values exactly, since CSV
formatting is the one place this table can silently lose information.

Two details of that round trip are load-bearing, and both were found by the
assertion rather than by inspection, so `sipnet_calibration.sites.load_sites`
exists to keep the reader and the writer in agreement rather than leaving the
settings to each caller.

- Coordinates are read with `float_precision="round_trip"`. **This, not the
  write format, is what makes the round trip exact.** The C parser
  `pandas.read_csv` uses by default is inexact for either candidate format:
  1496 of the 8000 longitudes come back wrong from the `repr` output actually
  written, and 1632 from `float_format="%.17g"`, in both cases by about
  1.4e-14 degrees. `repr` is written because it is shortest and is exact under
  Python's own `float()`, so the file is right for any reader that parses
  correctly.
- Text columns are read with `keep_default_na=False`. Eight of the 8000 sites
  are named literally `NA`, which a default read turns into a null, and an
  unmapped `ameriflux_site_id` is an empty string rather than a missing value.

`constraints_annual.nc` carries the annual biomass, leaf area and soil
constraints on a dense grid, with `NaN` where a site-snapshot-variable was not
observed:

| Variable | Dims | Type | Description |
|---|---|---|---|
| `observation_mean` | `(site, time, variable)` | float64 | The observation |
| `observation_variance` | `(site, time, variable)` | float64 | Its error variance |

`site` is the full 1-8000 pool, whether or not a site was ever observed; `time`
is the thirteen July 15 snapshot keys. `lon` and `lat` are non-dimension
coordinates on `site`, joined from the site table.

The `variable` coordinate holds **processed** names. The source names are not
ours to choose, but the processed ones follow the project convention of lower
case with underscores and no unnecessary abbreviation, and each variable's
source name is kept in the file's attributes as
`variable_<name>_source_name`:

| Source | Processed | Unit |
|---|---|---|
| `AbvGrndWood` | `aboveground_wood_carbon` | Mg C ha-1 |
| `LAI` | `lai` | m2 m-2 |
| `SoilMoistFrac` | `soil_moisture_percent` | percent |
| `TotSoilCarb` | `total_soil_carbon` | kg C m-2 |

The rename is applied by `ingest_constraints.py`, from a single mapping in
`sipnet_calibration.constraints`. It happens there rather than in R because the
intermediate long table names the variable on every row, so a row carries its
own identity and the rename cannot mis-pair a variance with a variable. In the
source the pairing is positional, which is why the R side keeps the source names
and the source order. 8000 x 13 x 4 is 416,000 cells per array, of
which 322,515 are observed, so the file is 2.2 MB compressed. Dense is chosen
over a ragged encoding because the raggedness costs nothing to represent this
way and dense is far easier to reason about.

Three points about the layout.

- **Variances, not covariance matrices.** Every source covariance is diagonal,
  so nothing is lost. Of the 104,000 site-snapshot entries, 103,029 are
  matrices whose off-diagonal is checked element by element at every export;
  953 are single-variable scalars, which have no off-diagonal; 18 are empty.
  The check runs at every export, not once, because it is what makes the
  choice lossless.
- **`variable` is a dimension.** That is not a canonical field, whose dims must
  be a subset of `(member, site, time)`. It is stored this way because the
  observation operator indexes observations by exactly `(site, variable, time)`,
  so flattening to the observation vector is a stack rather than a join, and
  because all four variables share one `(site, time)` grid here.
  `sipnet_calibration.constraints.constraint_fields` returns the canonical
  per-variable view -- one `DataArray` per variable with dims `(site, time)` --
  so the plotting layer and the likelihood are each served without reshaping the
  other's form.
- **The snapshot key is labeled `nominal`.** The July 15 dates are the source
  product's annual bookkeeping convention, not observation dates, so the label
  is neither an instant nor an interval boundary. The file records
  `time_label = "nominal"` with a note saying so, rather than claiming one of
  the interval conventions the other products use.

Each variable's unit is a dataset attribute, `variable_<name>_units`, beside
`_long_name` and `_source_name`. The two data variables carry
`units_status = "unconfirmed"` and a provenance string, because the units are
documented for the reanalysis *output* rather than for these observation
*inputs*. See open question 9.

The **drivers** are served by `sipnet_calibration.drivers.load_drivers(sites,
...)`, which parses the raw `.clim` files for the named sites and returns an
`xarray.Dataset` with one `float64` variable per consumed column on
`(member, site, time)`, `lon` and `lat` on `site`, and a `source_member_index`
coordinate on `member` holding the 1-based index from the directory name. The
`loc`, `length` and `soil_wetness` columns are asserted constant and not
carried; `length` becomes the `timestep_days` attribute. The processed names
follow the same convention as the constraints:

| Source | Processed | Unit | Aggregation |
|---|---|---|---|
| `tair` | `air_temperature` | deg C | mean |
| `tsoil` | `soil_temperature` | deg C | mean |
| `par` | `par` | mol m-2 | sum |
| `precip` | `precipitation` | mm | sum |
| `vpd` | `vpd` | Pa | mean |
| `vpd_soil` | `soil_vpd` | Pa | mean |
| `vpress` | `vapor_pressure` | Pa | mean |
| `wspd` | `wind_speed` | m s-1 | mean |

Every variable carries `units_status = "format_documented"` with a provenance
string: the units are what the `.clim` format documents and SIPNET assumes, not
units the producer has confirmed. The `time` coordinate holds the nominal
`year`/`day`/`3 x slot` instants and carries `time_zone = "UTC"`,
`time_label = "interval_end"` and `clock_status = "inferred"`, per Note 16;
keeping the nominal labels means a daily resample groups exactly the eight rows
SIPNET itself calls one day. A requested `(site, member)` pair with no file is
an error unless `allow_missing=True`, which fills it with `NaN` and adds a
boolean `driver_present(member, site)`. The three local files are such a case:
site 1 has members 1 and 2, site 27 has member 5.

The following conventions apply to every product.

- `site` is the integer identifier 1-8000, never renumbered. The Ameriflux
  identifier is a non-dimension coordinate on `site`, absent where unknown.
- `member` is a zero-based integer index, meaningful only within a single source.
- Time is stored as a datetime index; SIPNET's `year`, `day` and `time` triple is
  converted at the boundary by `sipnet_calibration.obs_ops.sipnet_time_index`,
  which uses the `time` column only to identify a row's slot within its day
  (Note 15). This applies to SIPNET output as well as to the drivers, since
  SIPNET copies the column into its output verbatim.
- Each product is stored at the temporal resolution its source arrives in.
  Aggregation is the observation operator's business, specified per variable at
  model-specification time, so that different constraints can be used at
  different time scales without a re-ingest.
- Uneven coverage is preserved rather than filled. The annual constraints in
  particular are not rectangular over site, snapshot and variable, and
  unobserved cells are `NaN` rather than zero -- a zero there would be an
  observation of no biomass, which is a different and real statement.

> **Note 12.** Whether ensemble member *i* of one source corresponds to member
> *i* of another is not established, though the net ecosystem exchange members are
> known to derive from a driver ensemble.

> **Note 13.** Whether the processed form should carry an additional
> spatially-ordered site coordinate is undecided.

---

## Open questions

Numbered notes above refer to the corresponding entry here.

**1. Per-site directory templates.** The driver template
`ERA5_<site>_<member>/ERA5.<member>.<start>.<end>.clim` holds for the three
directories present and the initial-condition template
`initial_conditions/<site>/IC_site_<site>_<member>.nc` for the three files
present. Whether all 8000 site directories follow
them has not been checked. The driver reader raises on any file it is asked
for that departs from the template, and on a directory whose member disagrees
with its file name; whether the 8000 x 10 set is complete can only be surveyed
where the files are.

**2. Meaning of the `cluster` and `landcover` fields.** Neither is documented in
the sources available. The evidence that they define sampling strata is
circumstantial but consistent: their cross-tabulation leaves 13 of 48 cells empty
in a staircase pattern rather than at random, several columns hold near-equal
counts across clusters, and the 6907 non-named sites are labeled
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
covariance should treat them. This is a modeling question rather than a data one,
and is unresolved.

**4. Driver ensemble size.** *Resolved: the driver ensemble has 10 members.*
Only three driver directories are available locally (`ERA5_1_1`, `ERA5_1_2`,
`ERA5_27_5`, so members 1, 2 and 5 across sites 1 and 27), which is not enough to
see this from the files, and neither of the two figures nearby applies: the
gap-filling behind [GAPFILL] used 25 driver members, and the reanalysis output
carries 100. Kept numbered so the surrounding references do not shift. What
remains open is member correspondence across sources, which is question 12.

**5. Reference year for the initial-condition time coordinate.** The units
attribute is an unsubstituted template, so the intended reference year cannot be
recovered from the file. This does not affect calibration, since the dimension is
degenerate, but it does mean the files cannot be used for anything time-aware.
Tracked as
[issue #3](https://github.com/arob5/spatial-lsm-calibration/issues/3).

**6. Initial-condition variable sets.** *The ensemble size is resolved: 100
members, as Note 6 records.* What remains open is the variable set, which is
reported to differ between files, with `leaf_carbon_content` and `SoilMoistFrac`
appearing in some. Three files are available locally — site 1 members 1 and 2,
and site 27 member 94 — and none of the three carries either variable, so the
full set of combinations is still unconfirmed, and can only be surveyed on the
SCC, where the files are.

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
driver realization used for gap-filling, so a timestep that was measured directly,
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

**11. Where plant functional type labelings live, and which to use.** Two
tables exist for the 8000 sites, distinguishing 16 and 3 classes respectively,
and neither is present in this repository. The `landcover` field of the site
shapefile is a third classification, with eight classes, and is present.

This is no longer a question about `ingest_sites.py`, which deliberately does
not join a PFT table. A labeling is not an intrinsic property of a site: some
calibrations will not use PFTs at all, others will use different labelings, and
the labeling is likely to be varied experimentally, so carrying one in the site
table would bake an experimental choice into a key shared with collaborators.

The agreed destination is a separate processed product, one file per labeling at
`processed/labelings/<name>.csv`, keyed on `site_id`, so that several can
coexist and a calibration names the one it used. Nothing is implemented yet
because the tables are not in the repository. What remains open is which
labelings to pull down and how the three classifications relate; see also
Note 2.

**12. Correspondence of ensemble members across sources.** Whether driver member
*i*, initial-condition member *i* and the calibration ensemble were drawn jointly
or independently determines whether arithmetic that pairs them is meaningful.
Because xarray aligns on coordinate values automatically, an incorrect assumption
here would combine unrelated members without any error being raised. One
connection is known: the members of [GAPFILL] are gap-filled series driven by
successive members of a driver ensemble, so if that is the same ensemble used here,
net ecosystem exchange member *i* and driver member *i* would share a realization.
Whether it is the same ensemble has not been established.

**13. Site ordering in the processed form.** The 1-8000 identifiers are fixed, but
a spatially coherent ordering, a Hilbert or Morton rank for instance, would
improve locality for triangulation and for chunked reads. Such an ordering would
be added as an additional coordinate rather than by renumbering.

**14. Observations with an error variance of exactly zero.** 929 `AbvGrndWood`
site-years carry a variance of 0. In 925 of them the observation is 0 as well,
which reads as "no biomass, and no uncertainty about that" and is at least
self-consistent. The other four assert a non-zero value with no uncertainty at
all: sites 5664 (2014), 6558 (2016) and 7167 (2015 and 2016), each with a mean
of exactly 1.0.

A zero variance is unusable either way. A Gaussian likelihood weights a residual
by `1/variance`, so these four contribute an infinite weight to a value of 1.0,
and any code that forms a precision matrix or sums a log-likelihood over them
returns `inf` or `NaN` for that site-year rather than a large number. The 925
are the same arithmetic but at least encode a plausible intent.

Nothing floors them at ingest, deliberately: the floor is a modeling choice.
But whether these are real, a placeholder, or an artifact of the source
processing is a question for the producer, and it bears on whether the four
should be dropped rather than floored.

**15. The `time` column of the driver files.** Filed as
[issue #9](https://github.com/arob5/spatial-lsm-calibration/issues/9), which
identifies the generator artifact exactly. Nothing in the project uses the
column's value: `sipnet_time_index` takes the slot from `floor(time / 3)`, which
is correct because the drift is never negative and never reaches a full step,
and the driver reader asserts the drift model per file so that a regenerated
file without it is noticed. The open question is for the producer: is the
series intended to be exactly 3-hourly?

**16. The driver clock and interval labeling.** The two sites are 54 degrees
of longitude apart, so a UTC clock requires the diurnal PAR cycle to shift by
3.6 h between them and a fixed local clock requires no shift. Measured, the
first harmonic of the summer PAR cycle shifts by 3.6 h, and the PAR-centroid
method of issue #6's comment by 3.4 h; either reading excludes a fixed local
clock. A PAR-centroid test at both sites places each row's total over the three
hours *ending* at its nominal label, one step from the "start of timestep" that
[pySIPNET] documents. "UTC with end-of-interval labels" and "UTC-3 with
start-of-interval labels" describe the same intervals and cannot be told apart
from the data; the reader records the former with `clock_status = "inferred"`.
Confirmation from the producer would settle it, and matters most for the
sub-daily comparison against net ecosystem exchange, whose own clock is the
subject of [issue #8](https://github.com/arob5/spatial-lsm-calibration/issues/8).

**17. Values that are not physical.** The negative `par` value -1.374e-05
recurs in about 1350 rows per file with no variation, and the tiny negatives of
`par` and `precip` look like floating-point residue from the generator; the 30
percent of rows with `vpd_soil` exactly zero is a larger fraction than the 0 to
7 rows with `vpd` zero. None of this is altered on read: the reader asserts
that negative excursions stay within 1e-4 of zero and records the counts in the
variable attributes, since clamping would hide an upstream artifact and a
value of -1e-15 mm harms nothing. What produces them is a question for the
producer. Also unexplained: in every file the mean of `tsoil` equals the mean
of `tair` to about 3e-5 deg C, as though `tsoil` were a mean-preserving filter
of `tair`.

**18. Hourly or 3-hourly forcing.** The [NALCR] dataset guide describes the
reanalysis as run on hourly ERA5 forcing, while these files are 3-hourly.
Whether these are the files the reanalysis used, and whether they were
aggregated from hourly, is not documented. The same guide says nothing about
the driver ensemble size, variables, units or clock, so the units recorded here
rest on the format definition alone.
