# Initial condition file provenance

Where `pecan_pool_initial_conditions.nc` came from, how it was made, and how to
tell whether it has drifted from its source.

The source initial conditions are 800,000 netCDF-3 files written by PEcAn, one
per site and ensemble member, on the Boston University SCC. SIPNET never reads them: they
are a PEcAn intermediate that PEcAn's `write.config.SIPNET` turns into
parameters. They take 816 MB and 800,000 inodes for 32 MB of values, one file
open per `(site, member)` cell, and they exist nowhere but the SCC. So they
are **converted once into one array and the result is tracked in version
control**, as the constraint CSVs are. The conversion changes structure only:
the values are copied bit for bit, the variable names and the `units` and
`long_name` strings are the source files', and the member index is their
1-based file index.

## Source

```
/projectnb/dietzelab/dongchen/anchorSites/NA_runs/IC/IC/<site>/IC_site_<site>_<member>.nc
```

symlinked as `data/raw/initial_conditions/files/` in the SCC checkout. Every
file is dated 2025-07-23. The tree holds exactly 8000 site directories, 1 to
8000, of exactly 100 files each, members 1 to 100; the conversion refuses any
other layout.

The script that drew the ensemble is described in `data/README.md` under
Initial conditions, with the caveat that the copy found targets the anchor
sites and the run that wrote these 8000-site files was not located.

## What was done

Converted at **2026-09-21T01:16:15Z** (the evening of 2026-09-20 in Boston; the
file's `converted` attribute) on the SCC's dietzelab buy-in queue (`qsub -P
dietzelab -l buyin`, which landed on `geo-int`, job 7670299), with

```bash
qsub scripts/raw_sources/convert_initial_conditions.qsub   # --jobs 16
```

from this repository at the commit that introduced the script. The conversion
parsed every file with `sipnet_calibration.initial_conditions.read_source_file`,
which asserts the source template for each one -- netCDF-3 classic, no global
attributes, an unlimited length-1 `time` whose variable carries exactly
`units = "days since [year]-01-01 00:00:00 UTC"`,
`long_name = "Time middle averaging period"` and the value 1.0, and scalar
`float64` data variables with exactly the attributes `_FillValue = -999.0`,
`long_name` and `units` that PEcAn's `standard_vars` prescribes -- and asserted
across files that the `(site, member)` set is a complete rectangle and that a
variable present for one member of a site is present for all 100. No file held
the `-999.0` fill or a non-finite value.

| File in this directory | Source | Files read | md5 of the file as committed |
|---|---|---|---|
| `pecan_pool_initial_conditions.nc` | the tree above | 800,000 | `8591d0429e63585e114a321939dc6316` |

The run report, which the script prints:

```
variable                       sites   min          median       max          negative
AbvGrndWood                     8000   1.42235e-08  0.496581     56.9659      0
wood_carbon_content             8000   -9.14241     0.318134     137.094      160987
leaf_carbon_content             7664   -135.275     0.176845     9.77886      11572
soil_organic_carbon_content     8000   0.00268813   21.6         1793.6       0
SoilMoistFrac                   7384   0.000108681  60.5462      100          0
variable sets:
   711300 files: ['AbvGrndWood', 'SoilMoistFrac', 'leaf_carbon_content', 'soil_organic_carbon_content', 'wood_carbon_content']
    55100 files: ['AbvGrndWood', 'leaf_carbon_content', 'soil_organic_carbon_content', 'wood_carbon_content']
    27100 files: ['AbvGrndWood', 'SoilMoistFrac', 'soil_organic_carbon_content', 'wood_carbon_content']
     6500 files: ['AbvGrndWood', 'soil_organic_carbon_content', 'wood_carbon_content']
```

## How to detect drift

The md5 above is of the committed bytes. Re-running the conversion writes a new
`converted` timestamp into the file, so a fresh conversion will not reproduce
the md5 even from identical sources; compare the *values* instead:

```python
import xarray as xr
old = xr.open_dataset("pecan_pool_initial_conditions.nc")
new = xr.open_dataset("<fresh conversion>")
assert all(old[v].equals(new[v]) for v in old.data_vars)   # bit for bit, NaN included
```

If the source tree changes -- a file re-dated, a site gaining a variable --
the conversion's checks either refuse it or the comparison above fails, which
is the signal to re-open the questions in `data/README.md` before adopting the
new file.
