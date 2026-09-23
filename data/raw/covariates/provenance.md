# Site covariate provenance

Where each file in this directory came from, what was done to it, and how to
tell whether it has drifted from its source.

A **covariate table** is per-site data that is neither an observation to fit
nor a class to pool over: climate, soil, terrain, vegetation structure,
biogeography. Nothing in the project reads one yet. They are here because a
spatial prior that puts a smooth residual on top of a class offset needs
predictors for that residual, and these are the ones already assembled for this
site pool.

These files are **copied into version control rather than symlinked**, for the
same reason as the site labels: they are a few megabytes, and the copy here is
the only form in which they exist off the Boston University SCC.

## What is here

| File | Sites | Columns | Size | md5 |
|---|---|---|---|---|
| `site_covariates_pft_assignment.csv` | 8000 | 43 | 4,166,530 | `51e3e97cef68c45827fda250be588791` |

## `site_covariates_pft_assignment.csv`

**This file is not a verbatim copy, and that is the one thing to know about
it.** It is one half of a 60-column table the producer assembled to derive the
16-class plant functional type assignment; the other half is the site labels
themselves, at
[`raw/site_labels/site_pft_16class.csv`](../site_labels/site_pft_16class.csv).
The two together are the source, column for column and cell for cell.

The split is made by `scripts/raw_sources/split_site_pft_16class.py`, which is
**not** part of the ingest pipeline: like
`convert_initial_conditions.py` beside it, it *creates* raw inputs rather than
processing them, and a normal working copy never runs it.

| | |
|---|---|
| Source | `/projectnb/dietzelab/guYANG/SIPNET_Model_Calibration/Final_PFT_assignment_v4/final_8000_sites_with_final_pft_v4.csv` |
| Source size | 5,902,793 |
| Source md5 | `02a26a77525d62bc2836953d184c72fc` |
| Source date | 2025-06-23 |
| Split on | 2026-09-22 |

**Why the split, and what replaces the md5 check.** Every other tracked raw
input here is byte-verbatim, so a one-line `md5sum` tells you whether it still
matches upstream. Cutting a table in two forfeits that: neither half can be
compared against the source. What stands in for it is the script and the four
checks it makes before writing anything --

- the two column sets **partition** the source, sharing only the `index` key,
  so no column is lost and none is in both;
- both halves are keyed on the **whole 1-8000 pool**, once each;
- no site has an empty class;
- **re-joining the halves reproduces the source cell for cell**, compared as
  text.

and the source md5 above, against which the split can be re-run and the outputs
compared. The script reads and writes every cell as a string, so no float is
parsed and none can come back at a different precision.

```bash
# Re-derive both halves and confirm they match what is committed.
python scripts/raw_sources/split_site_pft_16class.py --source <the source above>
md5sum data/raw/covariates/site_covariates_pft_assignment.csv  # 51e3e97c...
```

## What the columns are

`index` is this project's 1-8000 site identifier, complete and unique. It is
the only column shared with the site-labels half, and the one to join on.

| Group | Columns |
|---|---|
| Position | `lat`, `lon` |
| MODIS land cover | `LC_Type1`, `LC_Type1_name`, `LC_Type1_name_original`, `MODIS_LC_year`, `LC_Type3`, `LC_Prob3`, `LC_source_hdf`, `LC2`, `LC2_name`, `LC2_group` |
| Climate | `KGC` (Koppen-Geiger), `MAT`, `T_warmest_q`, `MAP`, `P_seasonality`, `MaxCWD`, `GSL_median` |
| Vegetation structure and greenness | `VCF_tree`, `LAI_max`, `NDVI_cv`, `EVI_min`, `SWIR`, `agb` |
| Soil and terrain | `Soil_AWC`, `TWI`, `twi_was_na`, `PH`, `Sand`, `SOC`, `N` |
| Biogeography | `BIOME_NAME`, `BIOME_NUM`, `REALM`, `ECO_ID`, `ECO_NAME`, `NNH`, `NNH_NAME` |
| Disturbance and period | `Fire_frequency`, `start_date`, `end_date` |

No units, no long names and no source product are recorded for any of them,
either in the source file or anywhere this repository has found. Note that this
is a statement about *these* columns: where the reanalysis's own inputs are
concerned the producer's extraction scripts often do document the source, as
`soilgrids_texture_extract.R` does for the soil texture ensemble. Nothing
equivalent has been found for this table. Establishing the units and sources is
the first piece of work for whoever ingests it; see `data/README.md` open
question 25.

## Caveats carried by the source itself

- **Coverage is ragged.** `LC_Type1_name_original` is absent for 3997 sites,
  `LC2_group` for 7047, `Fire_frequency` for 475, and the WWF biome columns for
  about 30. Eighteen numeric columns are complete over all 8000 sites.
- **`lat` and `lon` duplicate the site table**, which is the only redundancy in
  the file. Treat `data/processed/sites/sites.csv` as authoritative and use
  these as a check on the join, not as a source of coordinates.
- **Nothing here is independent of the site labels.** These are the variables the
  16 classes were derived from, so using both a class effect and these
  covariates in one model means the two are related by construction, not
  coincidence.

## Re-copying

Writes nothing on the cluster.

```bash
C=/projectnb/dietzelab/guYANG/SIPNET_Model_Calibration/Final_PFT_assignment_v4
scp <host>:$C/final_8000_sites_with_final_pft_v4.csv /tmp/source.csv
python scripts/raw_sources/split_site_pft_16class.py --source /tmp/source.csv
```
