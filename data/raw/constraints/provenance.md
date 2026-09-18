# Constraint file provenance

Where each file in this directory came from, what was done to it, and how to
tell whether it has drifted from its source.

These files are **copied into version control rather than symlinked**. The
upstream copies are edited and moved in place by their producer -- the
directory holding them contains four generations of the same product under
different names -- so a symlink is not a stable input.

Copied on **2026-09-17** from the Boston University SCC. All source paths are
relative to:

```
/projectnb/dietzelab/dongchen/anchorSites/NA_runs/SDA_8k_site/observation
```

## What was copied

| File in this directory | Source path | Source size | Source md5 | Source date |
|---|---|---|---|---|
| `landtrendr_aboveground_biomass.csv.gz` | `AGB/agb_final.Rdata` | 1,920,275 | `3e792fb7ab0bd0de31ea8049a2c88684` | 2025-07-16 |
| `gedi_aboveground_biomass.csv.gz` | `GEDI_AGB/gedi_final.Rdata` | 1,344,266 | `828ebd6f858163df235bfa65485618d9` | 2026-04-06 |
| `modis_leaf_area_index.csv.gz` | `LAI/LAI.csv` | 93,528,221 | `873441b48b48c98e7a194630354d443c` | 2025-07-12 |
| `smap_soil_moisture.csv.gz` | `SM/SMAP.csv` | 6,934,499 | `be3ee4844bcabbf639d3d803a32d3d63` | 2025-07-13 |
| `soilgrids_soil_organic_carbon.csv.gz` | `SOC/soc_final.Rdata` | 2,912,266 | `21991fdd39a141651d007e273f2c6a69` | 2025-07-16 |

## What was done to each

**`modis_leaf_area_index.csv.gz` and `smap_soil_moisture.csv.gz`** are
**byte-verbatim** copies of the producer's CSVs, gzipped and otherwise
untouched. Decompressing either reproduces the source md5 above exactly, which
is the check to run if you suspect drift:

```bash
gzip -dc modis_leaf_area_index.csv.gz | md5sum   # 873441b4...
gzip -dc smap_soil_moisture.csv.gz    | md5sum   # be3ee484...
```

**The other three** were `.Rdata` objects holding a single data frame, and were
serialized to CSV. Every numeric field was written with `sprintf("%.17g", ...)`
and missing values as the literal `NA`; column names are the source's own and
row names were dropped. Seventeen significant digits round-trips a float64
exactly, so no precision was lost -- but that only holds if the reader asks for
it:

```python
pandas.read_csv(path, float_precision="round_trip")
```

The default C parser is inexact and will not reproduce these values. This is
the same trap documented for `sites.csv` in the repository's `CLAUDE.md`.

Verified after copying: reading each of the three back with
`float_precision="round_trip"` and re-serializing at `%.17g` reproduces the
committed file byte for byte.

| File in this directory | md5 as committed | md5 of the uncompressed content |
|---|---|---|
| `landtrendr_aboveground_biomass.csv.gz` | `faac188c243c9db5e6468f01ae851ad6` | `1d2098b550b43590803c1e5b1ed192ff` |
| `gedi_aboveground_biomass.csv.gz` | `cef0d2d1705566ef4e28ccffb6e9bbde` | `00bfbe4a0907484c6bbd36d0585abc63` |
| `modis_leaf_area_index.csv.gz` | `c54c81634d97e78b506446f175455ccf` | `873441b48b48c98e7a194630354d443c` |
| `smap_soil_moisture.csv.gz` | `267ac6fe87451fc45adb4ba4f022d98e` | `be3ee4844bcabbf639d3d803a32d3d63` |
| `soilgrids_soil_organic_carbon.csv.gz` | `03ff9fe851fb7ec897c0eb6d366076be` | `e69ec2ffea52ef46e8d937a7953737db` |

The committed md5 is a record of the bytes in this repository, not something to
compare against the source: gzip output is not reproducible across versions.
The uncompressed md5 is the one that carries meaning.

## Why these files and not the assembled ones

The observations the reanalysis assimilated live upstream as
`obs.mean.Rdata` / `obs.cov.Rdata`, which remain symlinked under
`sda_8k_site_rdata/`. Nothing reads them any longer; they are kept for the
questions still going to their producers, and `tests/test_constraints.py`
checks the products built from the files here against what was built from
them. Those files
combine all four variables into a nested structure of 13 snapshots x 8000
sites, carry no attributes despite their directory's name, key their covariance
matrices positionally, and apply at least one undocumented modification: the
`LAI` standard deviations in them are floored at 0.66, affecting 82.4% of
observations.

The per-variable files here are the upstream sources those were assembled from.
Each was checked against the assembled file:

| Variable | Check | Result |
|---|---|---|
| `AbvGrndWood` | `agb_mean`, `agb_sd` vs the assembled values | 39,273 of 39,273 exact |
| `SoilMoistFrac` | `smp`, `sd` vs the assembled values | 79,740 of 79,740 exact |
| `TotSoilCarb` | `soc/10`, `sd/10` vs the assembled values | 103,870 of 103,870 exact |
| `LAI` | nearest date to July 15 among rows with `sd <= 20` | 99,632 of 99,632 exact |

Maximum absolute difference zero in every case. GEDI is not in the assembled
files at all and has nothing to check against.

## Caveats carried by the sources themselves

Recorded here because they are properties of these files, not of our
processing. `data/README.md` documents them in full.

- **Units are unconfirmed for all five products**, and unestablished for GEDI.
- **`soilgrids_soil_organic_carbon.csv.gz` is ten times** the unit used
  downstream, and its values are **constant across all thirteen years**.
- **`landtrendr_aboveground_biomass.csv.gz` changes uncertainty source at
  2018**, and its means jump at that boundary.
- **`modis_leaf_area_index.csv.gz` is not one row per site-year** -- it holds
  322 observation dates -- and its `qc == 1` rows are no-data, not measurements.
- **`smap_soil_moisture.csv.gz`'s `date` column is a snapshot label**, not an
  acquisition date, and neighboring sites share source cells.

## Re-copying

The commands used, for reproduction. They write nothing on the cluster.

```bash
B=/projectnb/dietzelab/dongchen/anchorSites/NA_runs/SDA_8k_site/observation

# verbatim
ssh <host> "gzip -c $B/LAI/LAI.csv"  > modis_leaf_area_index.csv.gz
ssh <host> "gzip -c $B/SM/SMAP.csv"  > smap_soil_moisture.csv.gz

# converted: load the single object, format every numeric at 17 digits, emit CSV
cat > conv.R <<'R'
e <- new.env(); nm <- load("<source .Rdata>", envir = e); d <- get(nm[1], envir = e)
fmt <- function(v) { if (!is.numeric(v)) return(as.character(v))
                     o <- sprintf("%.17g", v); o[is.na(v)] <- "NA"; o }
d[] <- lapply(d, fmt)
write.table(d, stdout(), sep = ",", quote = FALSE, row.names = FALSE, na = "NA")
R
ssh <host> 'R --vanilla --slave | gzip -c' < conv.R > <output>.csv.gz
```
