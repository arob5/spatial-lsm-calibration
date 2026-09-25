# Net ecosystem exchange raw file provenance

Where the raw NEE files came from, how they were made, and how to tell whether
they have drifted from their source.

The observations are AmeriFlux FLUXNET (ONEFlux) FULLSET files, one CSV per
tower, which exist in this project only on the Boston University SCC. They take
44 GB unzipped, most of it columns the calibration never reads, so the kept
columns of every tower are **converted once into one array per resolution**.
The conversion changes structure only: values are the source text parsed to the
nearest double, column names are the source's, and the stamps are the source's
local standard time. The two arrays, at about a gigabyte, are copied into a
checkout rather than tracked; the tower table made from them is tracked.

## Source

```
/projectnb/dietzelab/guYANG/FLUXNET_Sites_NH/AMF_<tower>_FLUXNET_FULLSET_{HH,HR}_<first>-<last>_<version>.csv
```

symlinked as `data/raw/net_ecosystem_exchange/fluxnet/` in the SCC checkout,
each beside the zip it came out of. Downloaded by Yang Gu on 2025-08-19 with
`amerifluxr` under the CC-BY-4.0 data policy: 241 towers, 237 half-hourly and 4
hourly (`US-Ha1`, `US-MMS`, `US-Ne1`, `US-Ne2`), site versions from 3-5 to 5-7.
An older download of 193 of the same towers, mostly at earlier versions, sits in
`FLUXNET_Sites_NA/` and is not used.

Each tower's dataset has its own DOI, recorded in the tower table's `doi`
column; citing it is a condition of the license
(https://ameriflux.lbl.gov/data/data-policy/).

## What was done

Converted at **2026-09-25T00:58:55Z** (half-hourly) and **01:01:10Z** (hourly),
the files' `converted` attributes, on the SCC's dietzelab buy-in queue (job
7730477 on `geo-int`), with

```bash
qsub -v UV_CACHE_DIR=...,PYSIPNET_CACHE_DIR=... scripts/raw_sources/convert_ameriflux_nee.qsub
```

from this repository at commit `c527722`, which introduced the script. Before reading
any CSV the conversion checked each against the size and CRC-32 its zip records;
all 241 match, so the CSVs are AmeriFlux's distribution as downloaded. It then
parsed every file with `sipnet_calibration.net_ecosystem_exchange.read_source_file`,
which asserts, per file:

- the name is a FULLSET half-hourly or hourly name, one file per tower;
- every kept column is present, except the constant-u*-threshold (`NEE_CUT_*`)
  group, which is present whole or not at all;
- `TIMESTAMP_START` and `TIMESTAMP_END` are `YYYYMMDDHHMM`, one step apart in
  every row, contiguous, and tile exactly the whole years the name declares;
- `-9999` is the only fill, no value is non-finite, and every flag is in its
  vocabulary;
- an NEE value is present wherever its quality flag is.

The stamps are read as naive integers, never through a time zone.

| File | Towers | Size | md5 |
|---|---|---|---|
| `ameriflux_nee_half_hourly.nc` | 237 | 1160.8 MB | `0c814ba05e33797ec84d50ddef21d7f6` |
| `ameriflux_nee_hourly.nc` | 4 | 20.9 MB | `19097a400296740bfca9040deb8443ad` |

53 of the 241 files carry no `NEE_CUT_*` columns; those columns are stored as
missing for them and named in the per-tower `absent_columns` coordinate. Each
tower's source file, version, years and md5 are coordinates of the raw file
too, so the table above is the whole record; the full per-tower report the
script printed is reproduced by running it again.

## The other inputs

| File | Tracked | Source | md5 |
|---|---|---|---|
| `Unmatched_Sites.csv` | yes | `/projectnb/dietzelab/dongchen/anchorSites/downscale/extra_site/Unmatched_Sites.csv` (2025-06-30), the list of AmeriFlux towers the reanalysis's site selection (`anchorSites/downscale/downscale_anchorsites.Rmd`) added to the pool; byte-identical to Yang Gu's `Validation/validation/Unmatched_Sites.csv` | `96875b30f6e4797858c02f01db085560` |
| `ameri_sites.tsv` | no | `/projectnb/dietzelab/guYANG/FLUXNET_Sites_NH/ameri_sites.tsv` (2025-08-19), AmeriFlux's site listing downloaded with the FLUXNET files: coordinates, IGBP class and DOI per site. Not tracked because it carries principal investigators' contact details | `cfbb633a246a73f1e15961c8c9c14d64` |

## The tower table

`ameriflux_towers.csv` is made from the two raw files and the inputs above by

```bash
python scripts/raw_sources/build_ameriflux_towers.py
```

It is the one record of each tower's pool site, match basis, primary status,
UTC offset and exclusion; its diff is the review of any change to them. The
offsets are recovered from each tower's `SW_IN_POT`
(`utc_offset_source = "recovered from SW_IN_POT"`): AmeriFlux publishes the
site `UTC_OFFSET` only in its BADM, to account holders. Where a BADM value is
known it agrees: `US-CRT` and `US-Ha1`, both -5.

## Detecting drift

- The CSVs: rerun the conversion; it refuses a CSV that no longer matches its
  zip, and each tower's `source_md5` coordinate records the file read.
- The raw files: compare their md5 with the table above.
- The tower list and site listing: compare their md5 with the table above.
