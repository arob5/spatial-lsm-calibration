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
    initial_conditions/         tracked: the converted ensemble and its record
      pecan_pool_initial_conditions.nc
      provenance.md
      files/                    SCC only: symlink to PEcAn's 800,000 source files,
                                <site_id>/IC_site_<site_id>_<member>.nc
    labelings/                site labelings, tracked in version control
      reanalysis_site_pft.csv
      site_pft_16class.csv
      provenance.md
    covariates/               per-site predictors, tracked in version control
      site_covariates_pft_assignment.csv
      provenance.md
    phenology/                MODIS leaf phenology
      leaf_phenology_8k.csv
      leaf_phenology_neon.csv
    soil_texture/             soil property ensemble
      <site_id>/Soil_params_0-<site_id>_<member>.nc
    constraints/              observations, tracked in version control
      landtrendr_aboveground_biomass.csv.gz
      gedi_aboveground_biomass.csv.gz
      modis_leaf_area_index.csv.gz
      smap_soil_moisture.csv.gz
      soilgrids_soil_organic_carbon.csv.gz
      provenance.md
      nee/ens_ec_3h.csv
      sda_8k_site_rdata/obs.mean.Rdata    retained for validation
      sda_8k_site_rdata/obs.cov.Rdata     retained for validation
  processed/                    ingest output, created by the ingest scripts
    sites/sites.csv
    labelings/<name>.csv        one per labeling, keyed on site_id
    constraints/<name>.nc       one per constraint, <name> the raw file's stem
    initial_conditions.nc
    nee.zarr/
```

The drivers have no processed form. SIPNET reads the raw `.clim` files itself,
so `sipnet_calibration.drivers.load_drivers` produces the canonical
`(member, site, time)` form from `raw/drivers/` on demand instead; see
[Meteorological drivers](#meteorological-drivers) and
[Processed format](#processed-format).

> **Note 1.** The driver directory template is inferred from three driver
> directories rather than confirmed across all 8000 sites. The initial
> condition template is confirmed: the conversion read all 800,000 files.

Files under `raw/` are treated as read-only; all conversion happens on the way
into `processed/`, which is regenerable and absent on a fresh clone. Neither
directory is tracked in version control, with five exceptions: three small
primary sources and two derived inputs, none of them a pipeline output and
all of them inputs the repository cannot do without. `raw/sites/` holds the site shapefile, without which the
repository carries no site information at all; `site_id_map.csv` is tracked for
the same reason. `raw/constraints/` holds the five per-variable observation
files, which are tracked because the upstream copies are edited and moved in
place, so a symlink is not a stable input; see
[Constraint observations](#constraint-observations). `raw/initial_conditions/`
holds the initial condition ensemble converted from PEcAn's 800,000
per-member source files into one 26 MB array, which is the only form in which it
exists off the SCC; see [Initial conditions](#initial-conditions).
`raw/labelings/` and `raw/covariates/` hold the site labelings and the per-site
predictors, a few megabytes in all and, like the initial conditions, held
nowhere else off the SCC; see [Site labelings](#site-labelings) and
[Site covariates](#site-covariates). Everything else under `raw/`, including the
much larger drivers, eddy-covariance files, phenology and soil texture files,
lives on storage and is symlinked.

---

## Site metadata

The site pool is defined by a point shapefile under `raw/sites/`, described here
alongside the Ameriflux identifier map. The pool and its identifiers were defined
for the model runs underlying [NALCR] and are shared with collaborators' files, so
the identifiers are treated as fixed and are never renumbered.

### `raw/sites/pts.shp` and companions

The site table as an ESRI point shapefile: 8000 `Point` records in `pts.shp`, with
`pts.shx`, `pts.dbf`, `pts.prj` and `pts.cpg` alongside. Record *N* corresponds to
site identifier *N*. This is the only site source in the repository, and one of the
three inputs tracked under `raw/` (see [Directory layout](#directory-layout)):
it is small, it is a primary source rather than a pipeline output, and without
it the repository carries no site information at all.

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
system of the input data: a **Lambert Azimuthal Equal Area centered at
50 N, 100 W**, on WGS 84, in meters, with no false origin.

    +proj=laea +lat_0=50 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs

The parameters live in code, as `SITE_PROJECTION` in
[`sipnet_calibration.projection`](../src/sipnet_calibration/projection.py),
which builds a `pyproj.CRS` from them and provides the transform. PROJ
serializes that CRS to the PROJJSON and PROJ string stored under
`src/sipnet_calibration/projections/`, for tools outside this package; the test
suite checks them against the parameters, so a drifted file is a test failure.
Regenerate them with `python -m sipnet_calibration.projection --write`.

This bears on the data above in two ways. The projection's base CRS is
WGS 84, matching the site coordinates, so **no datum transformation is
involved** and nothing here is shifted. And because it is equal-area, a density
or per-area figure is honest in a way the 1 km geographic grid is not — that
grid, as noted above, is not equal-area.

The choice of projection is a plotting decision rather than a property of these
inputs, so the argument for it is not repeated here. It is recorded in
[issue #4](https://github.com/arob5/spatial-lsm-calibration/issues/4), with the
distortion of every candidate measured over all 8000 sites, and summarized in
the module's own documentation. The short version: the projection the reanalysis
figures used, ESRI:102003, the USA Contiguous Albers Equal Area Conic, is
area-true everywhere but is intended for a region of predominant east-west
expanse, and it degrades in shape far from its standard parallels. This site
pool reaches 82.5 N, where it distorts shape severely — 107 degrees of angular
deformation, against under 14 for the projection adopted here. The ceiling for
the adopted projection is asserted against this table in
`tests/test_projection.py`; the 102003 figure is a one-off measurement recorded
on issue #4, since this package implements only the one projection.

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

**What is tracked.** `pecan_pool_initial_conditions.nc`, one netCDF holding
the whole ensemble on `(site, member)`: 8000 sites by 100 members, five
`float64` variables in the source files' names with their `units` and
`long_name` strings, `NaN` where none of a site's files carries the variable,
and the source files' 1-based member index as `member`. It was made once from
those files by `scripts/raw_sources/convert_initial_conditions.py` on the SCC,
bit for bit and with nothing renamed, converted or masked;
[`raw/initial_conditions/provenance.md`](raw/initial_conditions/provenance.md)
records the run and its md5. The conversion refuses any file off the source
template below, so the template is confirmed for all 800,000 files, not
inferred from three.

**The source files.** One netCDF-3 classic file per site and ensemble
member, `<site>/IC_site_<site>_<member>.nc`, written by PEcAn's
`pool_ic_list2netcdf`. No global attributes; one unlimited `time` dimension of
length 1; a `time` variable with value 1.0 and the attributes of Note 5; and
three to five scalar `float64` variables on `("time",)`, each with exactly the
attributes `_FillValue = -999.0`, `long_name` and `units` from PEcAn's
`standard_vars.csv`. No file holds the fill or a non-finite value. Which
variables a file carries depends on the site only, never on the member, and
comes in four combinations:

| Variables present | Sites | Where |
|---|---|---|
| all five | 7113 | median latitude 47 N |
| all but `SoilMoistFrac` | 551 | median 48 N |
| all but `leaf_carbon_content` | 271 | median 63 N |
| the three carbon pools only | 65 | median 78 N; the three local files are here |

| Variable | Units attribute | Long name | Sites |
|---|---|---|---|
| `AbvGrndWood` | `kg C m-2` | Above ground woody biomass | 8000 |
| `wood_carbon_content` | `kg C m-2` | Wood Carbon Content | 8000 |
| `leaf_carbon_content` | `kg C m-2` | Leaf Carbon Content | 7664 |
| `soil_organic_carbon_content` | `kg C m-2` | Soil Organic Carbon Content by Layer | 8000 |
| `SoilMoistFrac` | `(-)` | Average Layer Fraction of Saturation | 7384 |

**How they were made.** The PEcAn preparation script is
`/projectnb/dietzelab/dongchen/anchorSites/IC_prep_anchorSites.R` (Dongchen
Zhang, 2024-03-27), the same code as
`modules/assim.sequential/inst/anchor/IC_prep_anchorSites.Rmd` on PEcAn
`develop` and the script Cami Webb's `IC_prep_guide.md` names for pool initial
conditions. It draws 100 members per site at the nominal date **2011-07-15**
(`time_poimt <- as.Date("2011-07-15")`, spelled as the script spells it), from
these sources, with these PEcAn
functions (fetched from `develop` on 2026-09-20):

| Variable | Source product | Draw | Units on arrival |
|---|---|---|---|
| `AbvGrndWood` | Spawn and Gibbs (2020), *Global Aboveground and Belowground Biomass Carbon Density Maps for the Year 2010*, ORNL DAAC, [doi:10.3334/ORNLDAAC/1763](https://doi.org/10.3334/ORNLDAAC/1763), 300 m; the mean and uncertainty bands of `NA_runs/IC/AGB/agb_2010_global.tif` | `Prep_AGB_IC_from_2010_global`: normal with the pixel's mean and uncertainty (an uncertainty of 0 replaced by 0.1), negatives set to 0, then `ud_convert(x, "Mg ha-1", "kg m-2")` | Mg C ha-1 to kg C m-2; carbon by the product's definition |
| `leaf_carbon_content` | MODIS MCD15A3H leaf area index via `MODIS_LAI_prep`: the composite nearest 2011-07-15 within 30 days, `sd >= 20` dropped. The extraction is byte-identical (md5 `873441b4...`) to the source of `constraints/modis_leaf_area_index.csv.gz` | normal with the composite's LAI and standard deviation, divided by one draw from the site PFT's 100 SLA samples (`SDA_8k_site/samples.Rdata`) | LAI over SLA in m2 per kg leaf mass, labeled kg C m-2; the leaf carbon fraction (about 0.48) is not applied |
| `wood_carbon_content` | the two above | `AbvGrndWood - leaf_carbon_content` where a leaf draw exists, else `AbvGrndWood` | kg C m-2 |
| `soil_organic_carbon_content` | ISCN generation 3 (`PEcAn.data.land::iscn_soc`: 200 profile stocks in g cm-2 for each of 43 CEC level-2 ecoregions); the site's ecoregion by point-in-polygon | `IC_ISCN_SOC`: 100 draws with replacement from the ecoregion's 200 values, then `ud_convert(x, "g cm-2", "kg m-2")` | g C cm-2 to kg C m-2; integration depth undocumented |
| `SoilMoistFrac` | Copernicus C3S / ESA CCI *Soil moisture gridded data from 1978 to present*, [doi:10.24381/cds.d7782f18](https://doi.org/10.24381/cds.d7782f18), active sensor, daily, CDR v202212, 0.25 degree; variable `sm`, whose own attributes read `units = "percent"`, `long_name = "Percent of Saturation Soil Moisture"`, valid range 0-100 | `extract_SM_CDS`: the first day with a retrieval from 2011-07-15 forward within 30 days; normal with the retrieval and its uncertainty, negatives set to 0 | percent of saturation of the 2-5 cm surface layer; the files' `(-)` is the `standard_vars` string, not the data's |

Two consequences of that construction are visible in the values and are
asserted or reported by the ingest:

- `wood_carbon_content` equals `AbvGrndWood` bit for bit wherever
  `leaf_carbon_content` is absent, and equals `AbvGrndWood -
  leaf_carbon_content` bit for bit elsewhere. `ingest_initial_conditions.py`
  asserts the identity; a break means the source changed.
- The leaf draw exceeds the biomass draw in a fifth of the members that have
  one, so **`wood_carbon_content` is negative there**, at 5990 of the 8000
  sites and at every member of 52 of them; `leaf_carbon_content` is itself
  negative at 11,572 members, all at grassland sites, matching the share of
  negative values in that PFT's SLA sample. The values are written through
  unchanged and counted; see Note 6 for what PEcAn did with them.

**How PEcAn used them.** `write.config.SIPNET` (the `poolinitcond` branch,
through `PEcAn.data.land::prepare_pools`) turned each file into SIPNET
initial parameters as follows. This is the mapping the experiment layer has to
reproduce or consciously depart from; the ingest applies none of it.

| File variable | SIPNET parameter | pySIPNET field | Conversion in PEcAn |
|---|---|---|---|
| `wood_carbon_content` | `plantWoodInit` | `total_wood_carbon` | `1000 x wood / (1 - fineRootFrac - coarseRootFrac)` since PEcAn commit `913dcec66` (2025-09-02, merged in PR #3544); `1000 x wood` before it. Which version ran the 8000-site reanalysis is open question 24. |
| `leaf_carbon_content` | `laiInit` | `leaf_area_index` | `leaf x SLA` with the run's own SLA draw; then 0 if the PFT is deciduous (`fracLeafFall > 0.5`) and the run starts outside leaf-on |
| `soil_organic_carbon_content` | `soilInit` | `soil_carbon` | `1000 x soil` |
| `SoilMoistFrac` | `soilWFracInit` | `soil_wetness_fraction` | `SoilMoistFrac / 100`; SIPNET defines the parameter as a fraction of water holding capacity, a different fraction |
| `AbvGrndWood` | none | none | unused: `prepare_pools` prefers `wood_carbon_content`, and would use `AbvGrndWood` only with a coarse-root pool, which no file carries |

Two of the four conversions read a parameter the calibration proposes -- the
root fractions and the specific leaf weight -- which is why the ingest applies
none of them (see [Processed format](#processed-format)). A third,
`soilWFracInit`, takes no proposed parameter but is a fraction of a water
holding capacity the calibration also proposes, so its meaning moves as well.
`sipnet_calibration.initial_conditions.to_pysipnet_initial_conditions` applies
the mapping to one `(member, site)` cell for one proposed parameter vector, and
`to_pysipnet_initial_conditions_table` does it over a whole `(member, site)`
ensemble. Both write the leaf row with SIPNET's own `leafCSpWt` rather than
PEcAn's SLA draw, and both guard `fineRootFrac + coarseRootFrac < 1`, which
pySIPNET does not check and SIPNET runs to completion without.

> **Note 5.** The `time` units attribute is the unsubstituted template
> `days since [year]-01-01 00:00:00 UTC`, which no calendar library can parse,
> with `long_name = "Time middle averaging period"` and the value 1.0; the
> strings are PEcAn's `standard_vars` entry for the `time` dimension, verbatim.
> The conversion asserts the template in every file, so a substituted year
> upstream is noticed rather than averaged away, and both netCDFs carry the
> three values as `source_time_*` attributes. Anyone reading a source file
> directly must disable CF time decoding; `read_source_file` in
> `sipnet_calibration.initial_conditions` parses them with `scipy.io.netcdf_file`
> and does not decode time at all. Tracked as
> [issue #3](https://github.com/arob5/spatial-lsm-calibration/issues/3).

> **Note 6.** The ensemble has **100 members** at every one of the 8000 sites,
> confirmed by the conversion. `prepare_pools` accepts a pool only if it is
> numeric, not `NA` and not negative, so a member with negative
> `wood_carbon_content` or `leaf_carbon_content` was **silently skipped and
> SIPNET kept the template default** for that parameter in the reanalysis. What
> this project should do with those members is a modeling decision recorded in
> open question 24, not something the ingest settles.

**Source.** Initial condition ensembles prepared for the 8000-site pool for the
model runs underlying [NALCR]. The same files are used here. The script above
targets the 343 anchor sites and writes elsewhere; the 8000-site files were
written on 2025-07-23 by a run whose script was not found (a 6400-site
predecessor, `NA_runs/IC/IC_pre`, is dated 2025-04-10). The files match the
script's construction exactly, so it is recorded as the template for that run
with that caveat; see open question 24.

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

### Constraint observations

**Format.** Five gzipped CSV files under `raw/constraints/`, one per source
product. Each is a flat table addressed by `site_id`, and `site_id` is the
1-8000 identifier of the site table. These files are **copied into the
repository rather than symlinked**: the upstream copies are edited and moved in
place, so a symlink is not a stable input.
[`provenance.md`](raw/constraints/provenance.md) records the source path, size,
checksum and copy date of each, and the conversion applied.

| File | Source product | Rows | Columns |
|---|---|---|---|
| `landtrendr_aboveground_biomass.csv.gz` | LandTrendr | 96,000 | `site_id`, `year`, `agb_mean`, `agb_sd` |
| `gedi_aboveground_biomass.csv.gz` | GEDI | 48,000 | `year`, `site_id`, `agb`, `sd` |
| `modis_leaf_area_index.csv.gz` | MODIS | 1,404,385 | `date`, `site_id`, `lat`, `lon`, `lai`, `sd`, `qc` |
| `smap_soil_moisture.csv.gz` | SMAP | 79,740 | `date`, `site_id`, `lat`, `lon`, `smp`, `sd` |
| `soilgrids_soil_organic_carbon.csv.gz` | SoilGrids | 104,000 | `site_id`, `soc`, `sd`, `year` |

The product attribution is the producer's own, taken from a dictionary in the
assembly code rather than inferred.

Two of the five are byte-verbatim copies of the producer's CSVs, gzipped and
otherwise untouched. The other three were `.Rdata` objects and were serialized
to CSV at **17 significant digits**, which round-trips a float64 exactly; the
column names are the source's own. Read them with
`pandas.read_csv(..., float_precision="round_trip")`, for the same reason
`sites.csv` does (see [Processed format](#processed-format)).

Where a column duplicates the site table -- `lat` and `lon` in the MODIS and
SMAP files -- it is retained deliberately, as a check that a file's `site_id`
means the same thing the site table means. That is not hypothetical: a second,
6400-site pool exists upstream whose site 1 is a different location.

#### `landtrendr_aboveground_biomass.csv.gz`

8000 sites x 12 years, 2012-2023; 39,273 rows carry values. The unit is
`Mg C ha-1` (Note 9).

`agb_mean` is **integer-valued throughout**, taking 399 distinct values from 0
to 502. Coverage is **US land only**: 3,281 sites, all within CONUS. 2024 is
absent because the assembly code assigns it `NA` explicitly, not because data
is missing.

`agb_sd` comes from **two different sources either side of 2018**, and the two
halves are not the same kind of quantity (Note 19). The file is the
concatenation of an object holding 2012-2017 standard deviations, which are
integer-valued and are LandTrendr's own, and an object named `agb.pred` holding
2018-2023, which sit beside a 430 MB random-forest model. The means come from a
single object spanning 2012-2023 and are continuous in provenance, yet still
show a level-dependent discontinuity at the 2017/2018 boundary (Note 19).

The source names this quantity aboveground **biomass** throughout -- directory,
file and columns -- while the state variable it feeds is named aboveground
**wood**. No numerical conversion is applied between them (Note 20).

#### `gedi_aboveground_biomass.csv.gz`

8000 sites x 6 years, 2019-2024; 12,596 rows carry values. Units are not
established (Note 9).

**This product is not part of the constraint set the reanalysis used**, and is
absent from `obs.mean.Rdata`. It is kept as a candidate additional biomass
constraint: it is independent of LandTrendr, and it covers 2024, which
LandTrendr does not.

#### `modis_leaf_area_index.csv.gz`

322 observation dates from 2011-06-02 to 2024-08-28 across 7,705 sites, so
unlike the other files it is **not one row per site-year**. `lai` runs 0 to 7
and `sd` 0 to 24.8, both exact multiples of 0.1, which is the native MODIS
scaling.

The 322 dates are 23 per year on a 4-day lattice inside a June 1 to August 29
window, so the file is a summer-window extraction of the 4-day MCD15A3H
composites, not a year-round record. The `date` is the composite's label as
the extraction returned it; whether it marks the first day of the 4-day period
is not confirmed (Note 23).

`qc` is a **three-character string**, `"000"` or `"001"`, and **`qc == "001"`
is exactly equivalent to `sd > 20`**: 221,659 rows satisfy each, and no row
satisfies one but not the other. Within that flagged set, **`lai == 0` is a
no-data sentinel** -- all 60,016 such rows carry `sd` of exactly 24.8, which
is the product's fill value for `LaiStdDev_500m` (248) times its 0.1 scale
factor, and all are flagged. After dropping flagged rows the minimum `lai` is
0.1 and no zeros remain. The ingest drops the flagged rows and records how
many in the product's attributes.

The assembler's selection rule is fully reproduced: keep the unflagged rows,
keep those within **30 days** of July 15, take the nearest, and on a tie
between an earlier and a later composite take the **earlier**. That
reproduces the `LAI` means in `obs.mean.Rdata` exactly, at all 99,632 of them,
and `max(sd, 0.66)` reproduces every variance. The tie-break is load-bearing:
within the window 4,840 site-years tie, the two candidates differ in 4,003 of
them, and the later-date rule fails on exactly those. The window excludes 398
site-years whose nearest unflagged composite is 31-45 days out. Nearest-date selection without the flag filter
reproduces only 85.9%. The rule is what the PEcAn prep code does
(`MODIS_LAI_prep.R`: `search_window = 30`, rows with `sd >= 20` dropped,
`which.min` taking the first minimum), and it is encoded in
`tests/test_constraints.py`, not in any product.

#### `smap_soil_moisture.csv.gz`

7,974 sites x 10 years, 2015-2024, complete; the variable is absent before
2015. Values run 0.99 to 92.95, so the scale is 0-100 despite the source
variable being named a fraction (Note 9).

The `date` column holds **only the July 15 snapshot label** -- ten distinct
values, all July 15 -- so the acquisition date of the underlying retrieval is
not recoverable from this file.

The source grid is coarser than the 1 km site grid: 529 pairs of sites carry
bitwise-identical values in all ten years, with a median separation of 4.6 km
and a maximum of 25.9 km. Sites sharing a source cell do not carry independent
observations.

#### `soilgrids_soil_organic_carbon.csv.gz`

8000 sites x 13 years, 2012-2024; 103,870 rows carry values. `soc` and `sd` are
**ten times** the corresponding values in `obs.mean.Rdata`, which are declared
`kg C m-2`; this file is therefore in `Mg C ha-1` (Note 9). The processed
product keeps that unit; the factor is the observation operator's to apply.

The values integrate **0-200 cm** (Note 21).

**The values are constant in time.** Every site carries a bitwise-identical
`soc` and `sd` in all 13 years, so the 103,870 rows are 7,990 distinct
observations repeated thirteen times. The assembly code confirms the mechanism:
soil carbon is read once and reused for every snapshot. Anything that treats
site-years as independent will weight this variable thirteen times too heavily.

#### Retained for validation: `sda_8k_site_rdata/obs.{mean,cov}.Rdata`

The nested assimilation inputs remain symlinked. No script reads them any
longer; they are kept because they are what the reanalysis actually
assimilated, which the per-variable files above cannot show, and because the
questions going to their producers are still being formulated.
`tests/test_constraints.py` checks that the new products reproduce them, via
`processed/constraints_annual.nc`, the last output of the retired R and Python
pipeline (repository commit `d533dfd` and earlier). No script writes that file
any longer; the check runs where a copy is present and skips otherwise.

Two R data files, each a single object nesting as year, then site, then
observation: lists of length 13 keyed by date from `2012-07-15` to
`2024-07-15`, each element a list of length 8000 named `"1"` to `"8000"`. In
`obs.mean`, each site-year is a single-row data frame whose columns are the
variables observed there, between zero and four, in alphabetical order. In
`obs.cov`, the corresponding error covariances, as a bare numeric in the
single-variable case and a matrix otherwise. The covariance matrices carry no
dimension names, so the variable each row refers to must be taken from the
column order of the corresponding `obs.mean` entry.

The July 15 keys are the annual snapshot convention of the source product, not
observation dates. Some site-years are empty data frames with zero columns,
denoting no observations at all; six such entries occur in each of 2012, 2013
and 2014.

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

Summed over all thirteen snapshots the observation counts are `TotSoilCarb`
103,870, `LAI` 99,632, `SoilMoistFrac` 79,740 and `AbvGrndWood` 39,273, or
322,515 observations in total.

**These files do not agree with the per-variable sources on `LAI`.** Their
`AbvGrndWood`, `SoilMoistFrac` and `TotSoilCarb` values reproduce the
per-variable files exactly. Their `LAI` standard deviations are the source
values **floored at 0.66**, which affects 82.4% of observations and is not
documented anywhere upstream (Note 22). Whoever uses these files for validation
must account for that floor.

> **Note 9.** The units of all five products are documented for the published
> reanalysis output, or inferred from the assembled observation files, rather
> than stated by any attribute in the raw data. None has been confirmed by the
> producer. GEDI's are not established at all.

> **Note 10.** The directory the symlink points at is named as though its
> contents carry variable attributes, but no attributes are present on any
> object within it, at either the data-frame or the column level. The assembly
> code shows attributes were intended, naming the source product of each
> variable; the files in place do not carry them.

> **Note 19.** `AbvGrndWood` standard deviations before and after 2018 come
> from different objects and are not the same kind of quantity; the later half
> appears to be model-predicted. Separately, the means show a level-dependent
> discontinuity at that boundary despite coming from a single object.

> **Note 20.** The source calls the biomass product aboveground biomass; the
> state variable is aboveground wood. The values pass through unchanged.

> **Note 21.** The 0-200 cm depth is established by correlation, not by an
> attribute: against a SoilGrids table carrying both intervals, `soc` matches
> the 0-200 cm column at 0.9970 and the 0-30 cm column at 0.8147. The match is
> close but not exact, so the depth interval is settled while the precise
> extraction is not.

> **Note 22.** The 0.66 floor on the `LAI` standard deviations appears only in
> the assembled covariance file, not in any per-variable source. The script
> that applies it is PEcAn's `MODIS_LAI_prep.R`; whether it is intended for
> assimilation is a question for the producer.

> **Note 23.** Whether the MODIS composite date labels the first day of the
> 4-day period is not confirmed, so the processed product carries the date as
> written and writes no `time_bounds`.

**Source.** The observation inputs assimilated by [NALCR]. The per-variable
files are the upstream sources from which those inputs were assembled.

---

### Site labelings

A **labeling** maps sites to classes -- every site, where its spec says so.
Plant functional type is the only kind held so far, and it is not site metadata:
which labeling a calibration uses is an experimental choice, so each is its own
product rather than a column of the site table. See Note 11.

**Format.** `reanalysis_site_pft.csv`: two columns, `site` and `pft`, one row
per site, no missing values. The header and the class names are quoted; the
identifiers are not, which is one of the things that tells this file from the
older pool's (see below). `site` is the 1-8000 identifier shared with the
rest of the project; `pft` is one of three class names.

| Class | Sites | `landcover` classes | Latitude (min / median / max) |
|---|---|---|---|
| `boreal.coniferous` | 2369 | 1, 2 | 7.0 / 48.4 / 67.8 |
| `temperate.deciduous.HPDA` | 1537 | 3, 4 | 9.2 / 44.1 / 69.0 |
| `semiarid.grassland_HPDA` | 4094 | 5, 6, 7, 8 | 7.2 / 52.5 / 82.5 |

**The labeling is an exact aggregation of `landcover`.** Every one of the 8000
sites follows the rule in the third column, with no exception in either
direction; the cross-tabulation has no off-diagonal cell. That does not settle
Note 2, which asks what `cluster` and `landcover` mean, but it is evidence about
`landcover`: whatever its eight classes are, the reanalysis read 1-2 as one
group, 3-4 as a second and 5-8 as a third, which is consistent with an ordering
by needleleaf, broadleaf-deciduous and non-forest. It still does not name the
scheme. Whether the aggregation rule is the intended one, and what the eight
classes are, is open question 24(k). The relation is
measured rather than stated, so `ingest_labelings.py` asserts it and refuses a
raw file that departs from it.

**The classes are coarse, and two of the three names mislead.** Over a pool
spanning 7-82 degrees north, neither name constrains what it says it does.
`boreal.coniferous` carries no latitude restriction -- 805 of its 2369 sites lie
south of 40 N -- and it is not reliably coniferous either: it holds Vaira Ranch
(site 5692), a California annual grassland with a few oaks.
`semiarid.grassland_HPDA` is the catch-all for everything non-forest, so 992 of
its sites lie north of the Arctic Circle and are neither semiarid nor grassland.
It does not hold every Arctic site either: of the 1107 sites above 66.5633 N,
113 are `temperate.deciduous.HPDA` and 2 are `boreal.coniferous`.
A prior built on these labels inherits that coarseness, which is the argument for
a class offset in the mean plus a smooth spatial residual rather than pooling on
class alone.

**Two upstream files are named `site_pft.csv`.** One directory above the source
sits a sibling with the same header and the same three class names, labeling the
older 6400-site pool with identifiers 1-6400. Nothing inside either file says
which it is, so the discriminator is the row count: the expected count is a field
of each labeling's spec in `sipnet_calibration.labelings`, and the ingest refuses
a mismatch naming the other pool.
[`raw/labelings/provenance.md`](raw/labelings/provenance.md) tabulates both.

**Source.** [NALCR]'s 8000-site state data assimilation, whose per-PFT trait
posteriors are indexed by these class names -- which is why the names are kept
verbatim rather than renamed to this project's convention.

#### `site_pft_16class.csv`, the labeling this project calibrates under

Sixteen classes over the same 8000 sites, assembled for this calibration by a
colleague in the Dietze lab from MODIS land cover refined by clustering on
climate, vegetation structure, soil and biogeography. It **supersedes** the
three reanalysis classes for calibration; those are kept because the trait
posteriors are indexed by them.

**Format.** `index` is the 1-8000 site identifier, complete and unique;
`final_pft` is the class, never missing. Sixteen further columns record how
each label was arrived at. The producer supplied a display name per class,
which `sipnet_calibration.labelings` carries on the spec; the internal names
are the join keys and are never renamed.

| Class | Sites | Display name |
|---|---|---|
| `Open_Vegetation_Complex_P1` | 1681 | High latitude grassland |
| `Open_Vegetation_Complex_P2` | 998 | High seasonal open woodland |
| `Open_Vegetation_Complex_P4` | 787 | Arid grassland |
| `Open_Vegetation_Complex_P3` | 774 | Greener open woodland |
| `Open_Shrublands__P1` | 666 | Cold Shrublands |
| `CroplandPool__Cereal_Croplands` | 633 | Cereal Croplands |
| `Evergreen_Needleleaf_Forest__P2` | 369 | Closed Long-season ENF |
| `CroplandPool__Broad_Croplands` | 336 | Broad Croplands |
| `Open_Shrublands__P2` | 332 | Warm Shrublands |
| `Deciduous_Broadleaf_Forest__P1_P2_P3` | 263 | Strongly Seasonal High C-N DBF |
| `Mixed_Forest__P2` | 262 | Closed Weakly Seasonal MF |
| `Evergreen_Broadleaf_Forest` | 216 | Evergreen Broadleaf Forest |
| `Deciduous_Broadleaf_Forest__P4_P5` | 212 | Weakly Seasonal Low C-N DBF |
| `Permanent_Wetlands` | 169 | Permanent Wetlands |
| `Evergreen_Needleleaf_Forest__P1` | 161 | Open Cold-seasonal ENF |
| `Mixed_Forest__P1` | 141 | Open Strongly Seasonal MF |

**It does not nest inside the three reanalysis classes.** Every one of the
sixteen draws sites from at least two of the three, and twelve from all three.
The old `boreal.coniferous` is the clearest case: of its 2369 sites only 483
are needleleaf forest here, and 207 are evergreen **broadleaf** forest. A prior
cannot be carried from the coarse labeling to this one by inheritance, which is
what Note 11 records.

**363 sites were assigned by proxy**, not directly: nearest median profile over
up to fourteen ecological variables. Of those, 292 have a `distance_margin` at
or below 0.02 against a mean nearest distance of 0.095, so the runner-up class
is nearly as close as the one chosen. They concentrate in the Arctic classes.
`second_nearest_final_pft` is kept in the file so a result's sensitivity to
them can be measured rather than guessed at.

**It is one half of a larger table.** The other half is
[Site covariates](#site-covariates) below, which is also where the split, and
what stands in for the md5 check it forfeits, are described.

---

### Site covariates

Per-site predictors: neither an observation to fit nor a labeling to pool over.
Nothing reads them yet. They are here because a spatial prior that puts a
smooth residual on top of a class offset needs predictors for that residual,
and these are the ones already assembled for this pool.

**Format.** `site_covariates_pft_assignment.csv`, 8000 rows by 43 columns,
keyed on `index`, which is the only column it shares with the labeling half.

| Group | Columns |
|---|---|
| Position | `lat`, `lon` |
| MODIS land cover | `LC_Type1`, `LC_Type1_name`, `LC_Type1_name_original`, `MODIS_LC_year`, `LC_Type3`, `LC_Prob3`, `LC_source_hdf`, `LC2`, `LC2_name`, `LC2_group` |
| Climate | `KGC`, `MAT`, `T_warmest_q`, `MAP`, `P_seasonality`, `MaxCWD`, `GSL_median` |
| Vegetation structure | `VCF_tree`, `LAI_max`, `NDVI_cv`, `EVI_min`, `SWIR`, `agb` |
| Soil and terrain | `Soil_AWC`, `TWI`, `twi_was_na`, `PH`, `Sand`, `SOC`, `N` |
| Biogeography | `BIOME_NAME`, `BIOME_NUM`, `REALM`, `ECO_ID`, `ECO_NAME`, `NNH`, `NNH_NAME` |
| Disturbance and period | `Fire_frequency`, `start_date`, `end_date` |

**No units, long names or source products are recorded** for any of them, in
the file or anywhere else this repository has found. That is open question 25,
and it is why there is no ingest for this file: a processed product whose units
are unknown would assert something nobody has checked.

**Coverage is ragged.** `LC_Type1_name_original` is absent for 3997 sites,
`LC2_group` for 7047, `Fire_frequency` for 475 and the biome columns for about
30. Eighteen numeric columns are complete over all 8000 sites.

**These are the variables the sixteen classes were derived from**, so a model
carrying both a class effect and these covariates relates the two by
construction rather than by coincidence.

**The split.** This file and `raw/labelings/site_pft_16class.csv` are one
60-column source table cut in two by
`scripts/raw_sources/split_site_pft_16class.py`. Neither half is byte-verbatim,
so neither can be checked against the upstream md5; what replaces that check is
recorded in [`raw/covariates/provenance.md`](raw/covariates/provenance.md) and
enforced by the script's own assertions, chief among them that re-joining the
halves reproduces the source cell for cell.

---

### Leaf phenology

Two tables of satellite-derived leaf-on and leaf-off dates. Nothing in the
project reads them yet: they are a prior-specification input for `leafOffDay`,
recorded here because they are present and because how they can be used is not
obvious from the files.

**Format.** One row per site-year, with the same eight columns in both files:

| Column | Type | Description |
|---|---|---|
| `year` | integer | Calendar year |
| `site_id` | integer | Site identifier; see below on which |
| `lat`, `lon` | float | Coordinates, to four decimal places |
| `leafonday` | integer or `NA` | Leaf-on day of year |
| `leafoffday` | integer or `NA` | Leaf-off day of year |
| `leafon_qa`, `leafoff_qa` | 0-3 | Quality of the day beside it |

`leaf_phenology_8k.csv` is 96,000 rows, a complete rectangle of 8000 sites by
the twelve years 2012-2023 with no duplicate site-year.
`leaf_phenology_neon.csv` is 390 rows, 39 sites by 2012-2021.

**Quality, and what is missing.** The flags are 0 "best", 1 "good", 2 "fair" and
3 "poor", from the MCD12Q2 Collection 6 user guide by way of the comment in the
extraction function. In both files a flag of 3 coincides **exactly** with a
missing day, in both directions and in both columns, because the extraction sets
the day to `NA` where the flag is 3. In the 8000-site file 62,475 leaf-on days
are flagged best and 29,169 are missing; 6599 sites carry at least one leaf-on
day and 6600 at least one leaf-off day. Median leaf-on is day 149 and median
leaf-off day 262.

**The two columns invert on 731 site-years, and the file is right to.** Across
308 sites, `leafonday` is at or after `leafoffday`, with a median `leafoffday` of
42. This is not corruption: the MODIS bands are days since 1970-01-01, the
extraction guards against leaf-on falling after leaf-off **on that scale**, and
only then converts each with `lubridate::yday`, which discards the year. A
leaf-off falling in the following January therefore returns as a small day of
year, past a guard that was correct where it ran. Anything that differences the
two columns has to handle it.

**Which site identifiers.** `leaf_phenology_8k.csv` is keyed on **our** 1-8000
identifiers, and its coordinates agree with the site table to the four decimal
places it prints. `leaf_phenology_neon.csv` is keyed on identifiers around
1000004875-1000004945 and shares none of ours, so it is **not joinable without a
map**. Measured, each of its 39 sites has a nearest site in our pool at most
0.0049 degrees away, under one 1/120-degree cell, and the match is unambiguous:
at every one of the 39 the second-nearest pool site is at least 1.36 times
further, a gap of at least 0.0013 degrees. So a nearest-site join is available
and no NEON tower is a close call between two pool sites. What is *not*
established is that the nearest site is the intended correspondence. That is an
inference from proximity, and only the producer or a published map settles it.

**How PEcAn used it.** `write.configs.SIPNET.R` writes the **start year's**
`leafonday` and `leafoffday` to the SIPNET parameters `leafOnDay` and
`leafOffDay`, one value for the whole run, skipping either where it is `NA`.
Nothing multi-year is used.

**Under this project's defaults SIPNET does not read `leafOnDay`.**
[pySIPNET]'s `ModelFlags.gdd` defaults to `True`, and `leaf_on_day` is used only
when `gdd` and `soil_phenol` are both off; leaf-on is then the growing
degree-day threshold `gddLeafOn` instead. `leaf_off_day` has no default and is
always read. So of the two columns only `leafoffday` feeds a parameter as things
stand, and using both means running with `gdd = False`. Which build and which
compile-time options the reanalysis itself ran is open question 24(j).

**Source.** `leaf_phenology_8k.csv` is byte-identical to
`SDA_8k_site/leaf_phenology.csv` in the reanalysis's own directory. Both files
were produced by `PEcAn.data.remote::extract_phenology_MODIS` from **MODIS
MCD12Q2 v061**, the Land Cover Dynamics product, taking leaf-on from the
`MidGreenup.Num_Modes_01` band, leaf-off from `MidGreendown.Num_Modes_01` and the
flags from the 2nd and 6th of the seven **two-bit fields** packed into
`QA_Detailed.Num_Modes_01` -- bits 2-3 and 10-11 counting from zero, which is
what a 0-3 value needs and a single bit could not give; fill values of 32767 and
flag-3 records become `NA`. The driver script is
`anchorSites/NA_runs/MODIS_Phenology/script.R`, which requested 2012-2024 and
returned twelve years. The upstream path of the NEON companion was not found.
What remains unconfirmed is whether the flag-3 rule and the year-discarding
conversion are intended; see open question 24(o).

**Checked by** `scripts/survey_phenology.py`, which measures everything recorded
above and exits non-zero if one of those characteristics no longer holds.

---

### Soil texture

An ensemble of soil physical properties by depth, one netCDF per site and
member. As with the phenology, nothing in the project reads it yet: it is a
prior-specification input for `soilWHC`.

**Format.** `<site_id>/Soil_params_0-<site_id>_<member>.nc`, members 1-100. The
`0-` is the billions component of the site identifier: the producer's
`soil_params_ensemble.R` builds the name from
`paste0(siteid %/% 1e+09, "-", siteid %% 1e+09)`, so it is 0 here only because
this pool's identifiers are 1-8000, and an identifier such as 1000004875 would
give `1-4875`. Each file has a single
`depth` dimension of six, whose values are **layer bottoms in meters** --
0.05, 0.15, 0.3, 0.6, 1.0, 2.0 -- with the first layer's top at the surface, and
one `float32` variable per property on that dimension. No global attributes.
The depth semantics are the producer's, not an inference: the extraction script
below lists the source layers as `0-5cm`, `5-15cm`, `15-30cm`, `30-60cm`,
`60-100cm` and `100-200cm`, whose bottoms are exactly these six values.

| Variable | Units |
|---|---|
| `fraction_of_sand_in_soil`, `fraction_of_silt_in_soil`, `fraction_of_clay_in_soil` | 1 |
| `soil_type` | `string` (a numeric class code despite the attribute) |
| `soil_hydraulic_b`, `soil_water_potential_at_saturation` | 1, m |
| `soil_hydraulic_conductivity_at_saturation` | m s-1 |
| `volume_fraction_of_water_in_soil_at_saturation` | m3 m-3 |
| `volume_fraction_of_water_in_soil_at_field_capacity` | m3 m-3 |
| `volume_fraction_of_condensed_water_in_soil_at_wilting_point` | m3 m-3 |
| `volume_fraction_of_condensed_water_in_dry_soil` | m3 m-3 |
| `thcond0`, `thcond1`, `thcond2`, `thcond3` | W m-1 K-1, W m-1 K-1, 1, 1 |
| `soil_thermal_conductivity`, `soil_thermal_conductivity_at_saturation` | W m-1 K-1 |
| `soil_albedo`, `soil_bulk_density`, `soil_thermal_capacity` | 1, kg m-3, J kg-1 K-1 |

The three texture fractions sum to one in every layer, to within about 4e-8,
which is float32 rounding on values of order one. Most files carry all twenty
variables; a small minority carry seventeen, lacking `soil_albedo`,
`soil_bulk_density` and `soil_thermal_capacity` -- 20 of the 10,000 files in the
surveyed sample. In files that do carry them, those three are `NaN` in the top
layer of no file and in about 20 percent of files at each of the five layers
below it, with no concentration at depth.

**Coverage.** 7693 of the 8000 sites have a directory, each holding exactly 100
files and every name on the template: 769,300 files in all. **307 sites are
absent.** What happened at those is open question 24(n).

**How PEcAn used it, and what that implies.** `write.configs.SIPNET.R` takes
layer thickness as `c(depth[1], diff(depth))` -- so 5, 10, 15, 30, 40 and 100 cm
-- and sets

- `soilWHC` (cm) to `volume_fraction_of_water_in_soil_at_saturation` times
  thickness, summed over the profile: **porosity integrated over 2 m**;
- `litterWHC` to that product for the top layer alone, taken only where the top
  layer is no deeper than 10 cm;
- `litWaterDrainRate` to the top layer's
  `soil_hydraulic_conductivity_at_saturation`, converted to cm day-1.

**This matters more than its deferral suggests.** Over a 100-site, 10,000-file
sample the `soilWHC` that formula gives runs **72.6 to 100.7 cm**, median 88.1.
[pySIPNET]'s reference fixture and PEcAn's own `template.param` both use **12
cm**. That is a factor of six to eight (median 7.3) in the single parameter
setting how often the model is water-limited, and it also changes what
`soilWFracInit` means, since that is a fraction of the bucket whose size this
parameter is (open question 24(f)). Sharper still: `template.param` gives
`soilWHC` a range of 0.1 to **36 cm**, so every value the ensemble implies lies
above the top of PEcAn's own prior for it. A run set up for comparison with the
reanalysis should use a value of order 70-100 cm, not 12. Whether the 2 m
porosity integral is the intended `soilWHC` is part of open question 24(n).

**Source.** [NALCR], at
`anchorSites/NA_runs/soil_nc/soil_texture_output/soil_texture_ensemble`,
symlinked into `raw/soil_texture/`. The files themselves carry no attribute
naming an upstream product, but the code that made them does, in two places:
`anchorSites/NA_runs/soilgrids_texture_extract.R`, in the reanalysis's own
directory, declares **SoilGrids250m version 2.0** (soilgrids.org) and lists the
six depth intervals, and PEcAn's `soil_params_ensemble.R`, which turns the
extracted texture into this ensemble, documents the same input. So the product,
its version and the depth semantics are all established from the producer's
code rather than inferred, and `write.configs.SIPNET.R`'s own comment -- which
calls the layer-bottom reading an assumption -- is not what this rests on.

**Checked by** `scripts/survey_soil_texture.py`, which exits non-zero if a
characteristic it records no longer holds. What it records is the coverage
above, the depth profile, and that every file opened is readable and has a
complete porosity. What it measures but does **not** assert is everything that
moves with `--sample`: the variable and unit table, the 17-variable minority,
the fraction-sum tolerance and the `soilWHC` range. Those four are reported on
every run and are as good as the sample behind them, which for the figures above
was 100 sites. Its coverage pass is a directory listing and is quick; opening
files is sampled by default. On a partial copy, pass `--no-check`.

---

## Conversion to processed form

Ingest scripts live in [`../scripts/`](../scripts). Each reads from `raw/`
(and, for `ingest_sites.py`, the tracked `site_id_map.csv` beside it), writes to
`processed/`, and leaves its input unmodified. Run a script with
`--help` for usage.

| Script | Reads | Writes |
|---|---|---|
| `ingest_sites.py` | `raw/sites/pts.*`, `site_id_map.csv` | `processed/sites/sites.csv` |
| `ingest_labelings.py` | `raw/labelings/*.csv`, `processed/sites/sites.csv` | `processed/labelings/<name>.csv`, one per labeling |
| `ingest_constraints.py` | `raw/constraints/*.csv.gz`, `processed/sites/sites.csv` | `processed/constraints/<name>.nc`, one per constraint |
| `ingest_initial_conditions.py` | `raw/initial_conditions/pecan_pool_initial_conditions.nc`, `processed/sites/sites.csv` | `processed/initial_conditions.nc` |
| `ingest_nee.py` | `raw/constraints/nee/ens_ec_3h.csv` | `processed/nee.zarr` |

The drivers have no ingest script. SIPNET runs read the raw `.clim` files, so
converting 80,000 of them into a store would produce a large copy the model
never reads; `sipnet_calibration.drivers.load_drivers` parses the raw files for
the sites a caller names and returns the canonical form directly. A cached
subset, where a workflow wants one, is the caller's `to_zarr`.

No ingest needs R. Each constraint's raw file is described by a
`ConstraintSpec` in `sipnet_calibration.constraints` -- the columns that carry
the site, the time, the value and its standard deviation, the units, the time
structure and an optional quality flag -- and `ingest_constraints.py` reads the
file, runs one generic set of checks driven by that spec, places the records on
the site pool and writes the netCDF with the spec's fields as attributes.
`python scripts/ingest_constraints.py --describe` prints every spec. The
initial conditions follow the same pattern with an `InitialConditionSpec` per
variable in `sipnet_calibration.initial_conditions`, and
`ingest_initial_conditions.py --describe` prints the specs. Their raw file is
itself made by a script, which is **not** a pipeline step; see
[Making a raw input](#making-a-raw-input) below.

### Surveys, which are not the pipeline either

Three scripts under [`../scripts/`](../scripts) answer a question about raw
data and write nothing under `data/`: `survey_drivers.py`,
`survey_phenology.py` and `survey_soil_texture.py`. Nothing under `processed/`
depends on one, and no ingest calls one.

The two added with the phenology and soil texture sections above do one thing a
diagnostic does not: each carries a `RECORDED` table, compares its measurements
against it, and **exits non-zero if one no longer holds**. That is what keeps
those numbers from going stale in silence -- the property is described here and
checked there, so a re-copied or regenerated input that changed is refused
rather than absorbed. Where the two disagree, re-measure, then change this
document and the script's table together.

`RECORDED` is not every number in the sections above, and the difference
matters. A characteristic is asserted only where it is exact over the whole
input or structural over any sample; a distributional figure drawn from a
sample is reported and never asserted, because it would fail on a different
`--sample` without anything having changed. Each section says which of its
numbers fall on which side. `survey_phenology.py` reads whole files, so
everything it reports is asserted; `survey_soil_texture.py` asserts its coverage
and the structural facts, and reports the rest.

| Script | Surveys | Needs the SCC |
|---|---|---|
| `survey_drivers.py` | `raw/drivers/` coverage, format and value ranges | for coverage |
| `survey_phenology.py` | `raw/phenology/*.csv` shape, flags and day distributions | no |
| `survey_soil_texture.py` | `raw/soil_texture/` coverage, variables and `soilWHC` | for coverage |

### Making a raw input

Two scripts are **not** part of the pipeline above and live apart from it, in
[`../scripts/raw_sources/`](../scripts/raw_sources): each *creates* a raw input
rather than processing one. The initial conditions arrive as 800,000 per-member
netCDFs that exist only on the SCC, so they are laid on `(site, member)` once,
bit for bit, and
the result is tracked here as
`raw/initial_conditions/pecan_pool_initial_conditions.nc`; see
[Initial conditions](#initial-conditions) for why, and
`raw/initial_conditions/provenance.md` for the run. A normal working copy never
runs it: it needs the SCC, and it is re-run only if the source files change.

`split_site_pft_16class.py` is the second. The 16-class labeling arrives as one
60-column table holding two different things, a labeling and the covariates it
was derived from, so the script cuts it along an explicit column partition and
writes both halves. That forfeits the md5 check every other tracked raw input
gets -- neither half can be compared against the upstream file -- so the script
asserts instead that the halves partition the source, that both are keyed on
the whole pool, and that **re-joining them reproduces the source cell for
cell**, comparing as text so no float is reparsed. See
[Site covariates](#site-covariates).

| Script | Reads | Writes |
|---|---|---|
| `raw_sources/convert_initial_conditions.py` | `raw/initial_conditions/files/` (SCC only) | `raw/initial_conditions/pecan_pool_initial_conditions.nc`, tracked |
| `raw_sources/split_site_pft_16class.py` | the producer's 60-column PFT table (SCC only) | `raw/labelings/site_pft_16class.csv` and `raw/covariates/site_covariates_pft_assignment.csv`, both tracked |

Conversions applied during ingest rather than downstream:

- **Constraints.** None to the values: the ingest changes structure, never
  values. Units stay the raw file's (so SoilGrids soil carbon is `Mg ha-1`,
  not the `kg m-2` of the assembled files), no record is chosen to stand for a
  year, and nothing is aligned in time. Two structural steps are declared by
  the spec and counted in the product's attributes: MODIS rows failing the
  producer's quality flag are dropped, and SoilGrids' identical yearly copies
  are collapsed to one value per site after a check that they are identical.
  Zero standard deviations are written through unchanged; 929 LandTrendr
  records carry one, 925 of them where the observation is zero too, but four
  assert a non-zero value with no uncertainty at all: sites 5664 (2014), 6558
  (2016) and 7167 (2015 and 2016), all with a mean of 1.0. Flooring them is a
  modeling decision that would be hidden if an ingest script made it. See open
  question 14.
- **Initial conditions.** None to the values. The conversion is a re-layout
  in the source files' names and units strings; the ingest renames the
  variables to the spec names, renumbers `member` from their 1-based index to
  the project's 0-based one keeping the original as `source_member`, and drops
  the degenerate `time`. No state-to-parameter conversion and no unit
  conversion: three of the four SIPNET initial parameters depend on parameters
  the calibration proposes (see [Initial conditions](#initial-conditions)), so
  the mapping is evaluated per proposed parameter vector in the experiment
  layer. Negative wood and leaf draws pass through and are counted in the run
  report.
- **Site labelings.** None to the class names: they are the join key to the
  reanalysis's per-PFT trait tables, so they are written exactly as the producer
  wrote them, abbreviations and dots included. What changes is the column names,
  which are this project's to choose (`site` and `pft` become `site_id` and
  `label`), and the row order, which becomes ascending by `site_id`. The
  `landcover` relation is asserted, not applied: the classes come from the file,
  and a raw file departing from the relation is refused rather than corrected.
- **Net ecosystem exchange.** None to the values, as for the constraints: the
  product keeps the producer's umol CO2 m-2 s-1 and the observation operator
  converts the model into it (the 2026-09-15 observation-operator design
  decision). The redundant `ens_mean` column is dropped.

> **Note 11.** Plant functional type is not site metadata and is not a column
> of the site table. Which labeling a calibration uses, and how many exist, is
> an experimental choice; see Note 11 under Open questions.

---

## Processed format

`ingest_sites.py`, `ingest_constraints.py` and `ingest_initial_conditions.py`
are written, and the driver reader in `sipnet_calibration.drivers` is
implemented; the rest of this section records the intended output of scripts
not yet written.

The processed form is also the form used throughout the rest of the project, so it
is chosen to load directly as such: an `xarray.DataArray` per variable, with
dimensions drawn from `member`, `site` and `time`, longitude and latitude as
non-dimension coordinates on `site`, and units recorded in the array's attributes.
Formats are chosen according to the shape of each product.

| Product | Format | Dimensions | Approximate size |
|---|---|---|---|
| `sites/sites.csv` | CSV | table | ~1 MB |
| `labelings/<name>.csv` | CSV, one per labeling | table | ~0.2 MB each |
| `constraints/<name>.nc` | netCDF, one per constraint | `(site, time)`, or `(site,)` for the static soil carbon | 0.2 to 4.8 MB each |
| `initial_conditions.nc` | netCDF | `(member, site)` | 26 MB compressed |
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

There is deliberately no `pft` column; see Note 11. A labeling is joined on
instead, from `processed/labelings/`. The Ameriflux column is
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

The **labelings** are one CSV each under `processed/labelings/`, named by the
labeling rather than by its raw file, with two columns:

| Column | Type | Description |
|---|---|---|
| `site_id` | int32 | Site identifier, ascending |
| `label` | category | The class, exactly as the producer wrote it |

The column is `label` rather than `pft` so that every labeling has one schema:
code that pools over classes indexes `label` without knowing which labeling it
was handed, and a labeling that is not a plant functional type labeling needs no
schema change. Which kind of class a labeling holds is a field of its spec in
`sipnet_calibration.labelings`, which also fixes the order the classes are
indexed in -- `load_labeling` returns `label` as a categorical over exactly the
spec's classes, in that order, so a class axis is stable and an undeclared class
is an error rather than a new category.

There are no other columns: coordinates and `landcover` are site metadata, and a
caller joins `load_sites()` on `site_id`. Nothing is missing, either -- a site a
labeling does not label is absent from its file rather than carrying a null
class, and a labeling whose spec says it covers the pool is refused at ingest if
it leaves a site out.

The **constraints** are five files under `processed/constraints/`, one per
constraint, named by the raw file's stem. Each is an `xarray.Dataset` of two
`float64` variables, `NaN` where a site (and time) was not observed and in the
same cells of both:

| Variable | Dims | Description |
|---|---|---|
| `value` | `(site, time)`, or `(site,)` | The observation, in the raw file's units |
| `standard_deviation` | the same | The standard deviation the source reports beside it |

`site` is the full 1-8000 pool in every file, whether or not a site was ever
observed, so any two products align on `site` without a join; `lon` and `lat`
are non-dimension coordinates on it. `time` is each product's own:

| Constraint | Time structure | `time` | `time_bounds` | Units |
|---|---|---|---|---|
| `landtrendr_aboveground_biomass` | annual | January 1 of 2012-2023 | the calendar year | `Mg ha-1`, constituent `C` |
| `gedi_aboveground_biomass` | annual | January 1 of 2019-2024 | the calendar year | `Mg ha-1` |
| `modis_leaf_area_index` | dated | the composite dates, 2011-2024 | none | `m2 m-2` |
| `smap_soil_moisture` | dated | the July 15 keys, 2015-2024 | none | `percent` |
| `soilgrids_soil_organic_carbon` | static | no time dimension | none | `Mg ha-1`, constituent `C` |

An **annual** product labels each value with January 1 of its year -- a key,
not an acquisition time -- and states the calendar year the value is
attributed to as CF `time_bounds`. A **dated** product carries the source's
own date label exactly as written, with no bounds: what the label marks (a
4-day composite, a snapshot key) is documented but its exact placement is not,
and the `comment` on `value` says what is known. A **static** product has no
time dimension; the raw file's yearly copies were checked to be identical and
collapsed. Which record stands for a model time, and how, is the observation
operator's decision, not the product's.

**The processed files follow the Climate and Forecast conventions, CF-1.11**,
as pySIPNET's model output does, so the two sides read alike. The dataset
declares `Conventions = "CF-1.11"`; `time` carries `standard_name = "time"`,
`axis = "T"` and, where present, `bounds = "time_bounds"`; `lon` and `lat`
carry `standard_name` and `units`; no coordinate is encoded with a
`_FillValue`. No observation carries `cell_methods`, because CF has no
vocabulary for "the nearest composite" or "an annual map"; the meaning of the
label is written in words in the `time_reference` attribute, following
pySIPNET's use of a `comment` where `cell_methods` cannot speak.

Every attribute on `value` comes from the constraint's `ConstraintSpec`:
`units`, `constituent` (where the unit is of a substance), `long_name`,
`description`, `product`, `source_file`, `source_column`, `time_reference`,
`units_provenance` and, where set, `sign_convention` and `comment`. The dataset
counts what the ingest did to the rows: `rows_read`,
`rows_dropped_by_quality_flag` and `rows_collapsed_as_copies`. The units
are the raw file's, unchanged, and every one is inferred or documented for
something adjacent rather than confirmed by the producer; `units_provenance`
says which, in a sentence. See open question 9.

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

`initial_conditions.nc` carries the initial condition ensemble on
`(member, site)`, in the source files' units, read through
`sipnet_calibration.initial_conditions.load_initial_conditions` and split into
canonical fields by `initial_condition_fields`:

| Variable | Source variable | Units |
|---|---|---|
| `initial_aboveground_biomass_carbon` | `AbvGrndWood` | `kg m-2`, constituent `C` |
| `initial_wood_carbon` | `wood_carbon_content` | `kg m-2`, constituent `C` |
| `initial_leaf_carbon` | `leaf_carbon_content` | `kg m-2`, constituent `C` |
| `initial_soil_organic_carbon` | `soil_organic_carbon_content` | `kg m-2`, constituent `C` |
| `initial_soil_moisture_saturation` | `SoilMoistFrac` | `percent` |

`site` is the whole pool with `lon`/`lat`; `member` is 0-based with
`source_member` carrying the source files' 1-based index; there is no `time`,
and what the source's degenerate one claimed is kept in the `source_time_*`
attributes. `NaN` has one meaning, that no source file for the site carries
the variable, uniform over the site's members and asserted on load. Each variable
carries its spec's fields as attributes: `units`, `long_name`, `description`,
`product`, `source_name`, `source_units`, `source_long_name`,
`sipnet_initial_condition` (the `pysipnet.parameters.InitialConditions` field
PEcAn fed it into), `pecan_conversion`, `units_provenance` and, where set,
`constituent` and `comment`. The dataset records the PEcAn preparation script
and its caveat as `source_script` and `source_script_note`, the nominal date
2011-07-15 with where it comes from, `member_source =
"ic"` and `member_correspondence` (Note 12). The names carry `initial_` because
the product is the model's starting state -- PEcAn calls the format
`pool_initial_conditions` -- and so that no name collides with a constraint
product's; `biomass` rather than the file's `woody` because the Spawn and
Gibbs product is total aboveground biomass carbon.

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
- Uneven coverage is preserved rather than filled. The constraint products in
  particular are ragged over site and time, and
  unobserved cells are `NaN` rather than zero -- a zero there would be an
  observation of no biomass, which is a different and real statement.

> **Note 12.** Whether ensemble member *i* of one source corresponds to member
> *i* of another is not established, though the net ecosystem exchange members are
> known to derive from a driver ensemble. Every product with a `member`
> dimension records `member_source` and `member_correspondence` attributes
> saying so, because xarray aligns integer member labels silently.

> **Note 13.** Whether the processed form should carry an additional
> spatially-ordered site coordinate is undecided.

---

## Open questions

Numbered notes above refer to the corresponding entry here.

**1. Per-site directory templates.** *Resolved for the initial conditions:*
the conversion read all 800,000 files and found exactly 8000 site directories
of 100 files each, every name on the template
`<site>/IC_site_<site>_<member>.nc` with the directory's site. The driver
template `ERA5_<site>_<member>/ERA5.<member>.<start>.<end>.clim` still holds
only for the three directories present. The driver reader raises on any file
it is asked for that departs from the template, and on a directory whose
member disagrees with its file name; whether the 8000 x 10 set is complete can
only be surveyed where the files are.

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

*Evidence from the PFT labeling, on the `landcover` half only.* The
reanalysis's own three-class labeling is an exact function of `landcover`,
grouping 1-2, 3-4 and 5-8; see [Site labelings](#site-labelings). So `landcover`
is at least ordered by something its producer read as needleleaf,
broadleaf-deciduous and non-forest, which is consistent with an aggregation of a
standard scheme. It names neither the eight classes nor anything about
`cluster`, so both halves of this note stand; what it adds is asked as question
24(k).

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

**5. Reference year for the initial condition time coordinate.** The units
attribute is an unsubstituted template, so the intended reference year cannot be
recovered from the file. This does not affect calibration, since the dimension is
degenerate, but it does mean the files cannot be used for anything time-aware.
The conversion asserts the template in every file and both netCDFs keep the
strings verbatim; the sampling date 2011-07-15 is known from the PEcAn script
rather than from the files and travels as `nominal_date`. Tracked as
[issue #3](https://github.com/arob5/spatial-lsm-calibration/issues/3).

**6. Initial condition variable sets.** *Resolved.* The ensemble has 100
members at every site, and the variable set varies by site only, in the four
combinations tabulated under [Initial conditions](#initial-conditions): the
two carbon pools and the soil carbon everywhere, `leaf_carbon_content` at 7664
sites and `SoilMoistFrac` at 7384, thinning toward the Arctic. All five are
specified in `sipnet_calibration.initial_conditions`. What the absences mean
upstream (no MODIS composite passed quality control; no CCI retrieval) follows
from PEcAn's code and is recorded in each spec's `description`.

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

**9. Units of the constraint observations.** No unit is stated by any attribute
in any of the five raw files. `Mg C ha-1` for LandTrendr biomass and `m2 m-2`
for MODIS leaf area index are documented for the published reanalysis output
rather than for these inputs; `Mg C ha-1` for SoilGrids soil carbon is inferred
from its being exactly ten times the assembled values, which are themselves
declared `kg C m-2` on the same unconfirmed basis. The SMAP scale is 0-100
despite the source variable being named a fraction, and what it is a fraction of
-- saturation, porosity, water holding capacity -- is not established, which
matters because SIPNET's `soilWFracInit` is a fraction of water holding
capacity. GEDI's units are not established at all, and biomass against carbon
differs there by about a factor of two.

**10. Provenance of the assembled observation files.** The `obs.mean.Rdata` we
hold is byte-identical to a file in a sibling directory dated ten months
earlier, while the `obs.cov.Rdata` beside it matches none of the twelve other
covariance files upstream and differs from its sibling only by the `LAI` floor
of question 22. No script producing it has been found, so whether the two are an
intended pair is unknown. The directory is named as though its contents carry
variable attributes; they do not (Note 10).

**11. Where plant functional type labelings live, and which to use.**
*Resolved for the three-class table, open for the finer one.* A labeling is not
an intrinsic property of a site: some calibrations will not use PFTs at all,
others will use different labelings, and the labeling is likely to be varied
experimentally, so carrying one in the site table would bake an experimental
choice into a key shared with collaborators. Labelings are therefore a separate
processed product, one file per labeling at `processed/labelings/<name>.csv`
keyed on `site_id`, so several coexist and a calibration names the one it used.

The three-class table the reanalysis assimilated under is now held as
`raw/labelings/reanalysis_site_pft.csv` and ingested to
`processed/labelings/reanalysis_3pft.csv`; see
[Site labelings](#site-labelings). It turns out to be an exact aggregation of
the shapefile's eight `landcover` classes, which answers how those two
classifications relate. Whether the aggregation rule is the intended one is open
question 24(k).

*The 16-class table has since arrived* and is tracked as
`raw/labelings/site_pft_16class.csv`; it is the labeling this project intends
to calibrate under, and nothing in the design changed when it came, since
`sipnet_calibration.labelings` takes a second spec. What it settles is that the
two labelings **do not nest**: every one of its sixteen classes draws sites from
at least two of the three reanalysis classes, and twelve from all three. So the
planned transfer of priors from coarse classes to fine ones has no parent class
to inherit from and has to be reconsidered. What remains open is how, if at all,
the reanalysis's per-PFT trait posteriors map onto the sixteen.

**12. Correspondence of ensemble members across sources.** Whether driver member
*i*, initial condition member *i* and the calibration ensemble were drawn jointly
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

**19. The 2018 discontinuity in aboveground biomass.** The standard deviations
before and after 2018 come from different objects: 2012-2017 are LandTrendr's
own and are integer-valued, 2018-2023 come from an object named `agb.pred`
sitting beside a 430 MB random-forest model. Whether the later half is a model
prediction, and whether a predicted uncertainty should be assimilated on equal
footing with a measured one, is a question for the producer. Separately, the
means come from a single continuous object yet still jump at that boundary:
24% of sites move by more than 10 Mg C ha-1, against 0.6-3.2% at every other
year boundary, and the shift runs the wrong way for growth -- low sites up,
high sites down, with the change correlating at -0.47 with the 2017 level.

**20. Biomass or wood.** The source names the product aboveground biomass in
its directory, file and column names; the state variable it populates is named
aboveground wood. No numerical conversion is applied between them. Either the
product is already wood-only despite its name, or the relabeling is unconverted
-- which matters by whatever the wood fraction is.

**21. Soil carbon depth and extraction.** The 0-200 cm interval is established
by correlation rather than by an attribute (Note 21), and the ~2% residual says
the extraction differs in some way from the SoilGrids table used for the
comparison. Whether the values are organic carbon only, and whether they include
litter and roots, is also unconfirmed.

**22. The leaf area index floor, and which version is authoritative.** The
assembled covariance file floors the `LAI` standard deviations at 0.66,
affecting 82.4% of observations; no per-variable source carries the floor. The
script that applies it is PEcAn's `MODIS_LAI_prep.R`, so what remains open is
whether the floor is intended for assimilation. Separately, a later revision of
the MODIS extraction exists upstream which disagrees with the assembled file at 16,770 of
99,112 observations, spread evenly across all thirteen years and by as much as
6.5 leaf area index units. Which extraction is authoritative determines what a
future ingest should produce.

**23. The MODIS composite date.** `modis_leaf_area_index.csv.gz` labels each
value with a date on a 4-day lattice, and MCD15A3H is a 4-day composite, but
neither the file nor the product's catalog page says whether the label is the
first day of the compositing period. Until that is confirmed the processed
product carries the label as written and writes no `time_bounds`.

**24. The reanalysis's inputs: how they were produced and used.** Questions for
the producer, recorded here rather than asked yet. (a) to (i) are about the
initial condition ensemble; (j), (k), (n) and (o) about the build and the other
shared inputs. The gap is deliberate: (l) and (m), on the per-PFT trait samples,
are recorded in the project's design log rather than here, because nothing in
this repository reads those files yet. (a) Which script wrote the
8000-site files on 2025-07-23? The anchor-site `IC_prep_anchorSites.R` is the
template, and the files match its construction, but the run was not found; how
does it differ from the 6400-site `IC_pre` of 2025-04-10? (b) Which PEcAn
version ran the 8000-site reanalysis: with or without the root-fraction
division in `plantWoodInit` (commit `913dcec66`, 2025-09-02)? (c)
`wood_carbon_content = AbvGrndWood - leaf_carbon_content` is negative in a
fifth of the members with leaf carbon and at every member of 52 sites, and
`prepare_pools` then kept SIPNET's template default. Was that intended, and
should this project treat those members as missing, floor them, or fall back to
`AbvGrndWood`? (d) Leaf carbon is LAI over an SLA in m2 per kg leaf mass, so
the `kg C m-2` label omits the leaf carbon fraction (about 0.48); intended? (e)
The grassland PFT's SLA sample includes negative values, which produce the
11,572 negative leaf-carbon members. (f) `SoilMoistFrac` is CCI percent of
saturation of the top 2-5 cm, while SIPNET's `soilWFracInit` is a fraction of
the water holding capacity of its whole bucket; is dividing by 100 the intended
mapping? (g) What depth do the ISCN stocks integrate over, and how was the
200 x 43 `iscn_soc` table built? Its members reach 1794 kg C m-2. (h) The
nominal date is 2011-07-15 for runs starting 2012-01-01; intended? (i) Which
biomass raster fed the 8000-site files, the 300 m `agb_2010_global.tif` or the
1 km resample beside it?

(j) Which SIPNET build ran the 8000-site assimilation, and with which
compile-time options -- growing-degree-day against `leafOnDay` phenology, the
litter pool, water-limited heterotrophic respiration? `pecan.xml` says
`revision ssr` and the binary is a pre-v2 tree, so the options are not
recoverable from either. It decides whether the `leafonday` column of
[Leaf phenology](#leaf-phenology) fed anything at all.

(k) `site_pft.csv` sends `landcover` 1-2, 3-4 and 5-8 to the three classes,
exactly, over all 8000 sites. Is that the intended rule, and what are the eight
land cover classes? The second half is the outstanding part of Note 2.

(n) The soil texture ensemble covers 7693 sites. What happened at the other
307, and is the 2 m porosity integral the intended `soilWHC`? The integral
itself is not in doubt -- the source layers and their depths are declared in
`soilgrids_texture_extract.R` -- so what is being asked is whether integrating
the whole 2 m profile is what the parameter was meant to receive, given that
the value it gives is six to eight times SIPNET's template default and above
the top of the range that template allows; see
[Soil texture](#soil-texture).

(o) *Answered in part.* `leaf_phenology_8k.csv` comes from
`PEcAn.data.remote::extract_phenology_MODIS` over MODIS MCD12Q2, with the
bands and QA bits named under [Leaf phenology](#leaf-phenology), so the product
and the QA vocabulary are established from the code. What is not: whether
discarding the day flagged "poor" rather than passing the flag through is
intended, and whether the `yday` conversion that inverts 731 site-years is
known to the producer.

**25. Units, provenance and vintage of the site covariates.** Nothing records
what any column of `site_covariates_pft_assignment.csv` is in, where it came
from, or what period it describes. `MAT` is plainly a temperature and `MAP` a
precipitation total, but whether `MAP` is mm per year, what `SWIR` is a
reflectance of, what `Soil_AWC` is a fraction or depth of, what `agb` and `SOC`
are per unit area, and which product and epoch each was drawn from, are all
unestablished. `start_date` and `end_date` are 2012-01-01 and 2024-12-31 in
every row, which suggests the covariates are meant as period averages over the
run window, but that is an inference from two constant columns.

This is why there is no ingest for the file. The project's rule is that a
processed product carries its source units and records where they came from; a
product built from these would have nothing to record. Four questions to the
producer would settle it: the unit of every numeric column, the source product
and version of each, the period each summarizes, and what the fill convention
is for the ragged columns. Until then the file is tracked and read by nothing.
