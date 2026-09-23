# Site labeling provenance

Where each file in this directory came from, what was done to it, and how to
tell whether it has drifted from its source.

A labeling maps every site to a class. It is **not** site metadata and is not a
column of the site table: which labeling to use is an experimental choice, so
each one is its own product. `data/README.md` sets out that reasoning under
[Site labelings](../../README.md#site-labelings) and open question 11.

These files are **held in version control rather than symlinked**. Together
they are a couple of megabytes, and they are the only form in which these
labelings exist off the Boston University SCC.

The two got here differently, and the difference matters for how each is
checked.

## `reanalysis_site_pft.csv`, a verbatim copy

Copied on **2026-09-21**.

| Source path | Source size | Source md5 | Source date |
|---|---|---|---|
| `/projectnb/dietzelab/dongchen/anchorSites/NA_runs/SDA_8k_site/site_pft.csv` | 234,229 | `31ceffd6a37b67b0b24b62b83df22ea1` | 2025-07-17 |

**Byte-verbatim.** Nothing was parsed, reformatted or renamed; the file here
has the source md5. Its values are integers and short strings, so none of the
float precision care that
[`raw/constraints/provenance.md`](../constraints/provenance.md) describes
applies.

```bash
md5sum reanalysis_site_pft.csv   # 31ceffd6a37b67b0b24b62b83df22ea1
```

## `site_pft_16class.csv`, one half of a split

The labeling this project calibrates under, superseding the three reanalysis
classes. Produced by a colleague in the Dietze lab for this calibration and
sent to us on 2026-09-22; the source file is dated 2025-06-23 and is version 4
of their assignment. It sits in a directory of intermediate products
(`final_pft_counts_v4.csv`, `uncovered_sites_nearest_final_pft_assignment_v4.csv`
and others) that are not copied here.

**It is not a verbatim copy.** The source is a 60-column table holding two
different things: this labeling and the covariates it was derived from.
`scripts/raw_sources/split_site_pft_16class.py` cuts it in two, writing the
labeling here and the covariates to
[`raw/covariates/`](../covariates/provenance.md). Split on **2026-09-22**.

| Source path | Source size | Source md5 | Source date |
|---|---|---|---|
| `/projectnb/dietzelab/guYANG/SIPNET_Model_Calibration/Final_PFT_assignment_v4/final_8000_sites_with_final_pft_v4.csv` | 5,902,793 | `02a26a77525d62bc2836953d184c72fc` | 2025-06-23 |

| File in this directory | Rows | Columns | Size | md5 |
|---|---|---|---|---|
| `site_pft_16class.csv` | 8000 | 18 | 1,775,162 | `497718d8fda222051cac9fc66be68c13` |

**What replaces the md5 check.** A split half cannot be compared against the
upstream file, so the script asserts instead, before writing anything, that the
two column sets partition the source sharing only the key, that both halves are
keyed on the whole 1-8000 pool once each, that no site has an empty class, and
that **re-joining the halves reproduces the source cell for cell**. Every cell
is read and written as text, so no float is parsed and none can return at a
different precision. `tests/test_split_site_pft_16class.py` drives each of
those assertions to failure.

```bash
# Re-derive both halves from the source and compare.
python scripts/raw_sources/split_site_pft_16class.py --source <the source above>
md5sum site_pft_16class.csv   # 497718d8fda222051cac9fc66be68c13
```

**The key is `index`, and it is our site identifier**: the integers 1-8000, all
present, no duplicates. **The class is `final_pft`**, one of sixteen values,
never missing. The sixteen further columns record how each label was arrived
at; `sipnet_calibration.labelings` declares the whole header, and the processed
product keeps only `site_id` and `label`.

The sixteen `final_pft` values are internal names; the producer supplied the
display names alongside them, and the pairing is theirs, not ours:

| `final_pft` | Display name |
|---|---|
| `Evergreen_Needleleaf_Forest__P1` | Open Cold-seasonal ENF |
| `Evergreen_Needleleaf_Forest__P2` | Closed Long-season ENF |
| `Evergreen_Broadleaf_Forest` | Evergreen Broadleaf Forest |
| `Deciduous_Broadleaf_Forest__P1_P2_P3` | Strongly Seasonal High C-N DBF |
| `Deciduous_Broadleaf_Forest__P4_P5` | Weakly Seasonal Low C-N DBF |
| `Mixed_Forest__P1` | Open Strongly Seasonal MF |
| `Mixed_Forest__P2` | Closed Weakly Seasonal MF |
| `Open_Shrublands__P1` | Cold Shrublands |
| `Open_Shrublands__P2` | Warm Shrublands |
| `Open_Vegetation_Complex_P1` | High latitude grassland |
| `Open_Vegetation_Complex_P2` | High seasonal open woodland |
| `Open_Vegetation_Complex_P3` | Greener open woodland |
| `Open_Vegetation_Complex_P4` | Arid grassland |
| `CroplandPool__Broad_Croplands` | Broad Croplands |
| `CroplandPool__Cereal_Croplands` | Cereal Croplands |
| `Permanent_Wetlands` | Permanent Wetlands |

**How a site got its class**, from `final_pft_assignment_method`: 7637 sites
directly, from the producer's own clustering of a land cover class; 363 by
nearest median profile over up to fourteen ecological variables, with
`nearest_ecological_distance`, `second_nearest_final_pft` and `distance_margin`
recording how close the call was. Those three columns are populated only for
the 363, which is how to tell the two populations apart.

**Caveats carried by the source itself.**

- **The labeling does not nest inside the three reanalysis classes.** All
  sixteen classes draws from at least two of the three, and twelve from all
  three, so it is a different partition of the pool rather than a refinement of
  the old one. Anything that assumed a coarse-to-fine transfer between them has
  to be reconsidered.
- **292 of the 363 proxy-assigned sites have a `distance_margin` at or below
  0.02**, against a mean nearest distance of 0.095 -- the runner-up class is
  nearly as close as the chosen one. `second_nearest_final_pft` is in the file,
  so the sensitivity of a result to those sites can be measured rather than
  guessed at.
- **`LC_Type1_name_original` is absent for 3997 sites**, `LC2_group` for 7047
  and `Fire_frequency` for 475; the WWF biome columns are absent for about 30.
  None of that touches `final_pft`, which is complete.

## Which `site_pft.csv`

**There are two, and the wrong one is the easier to reach.** One directory
above the source path sits
`/projectnb/dietzelab/dongchen/anchorSites/NA_runs/site_pft.csv`, with the same
name, the same two-column header and the same three class names. It labels the
**older 6400-site pool**, whose identifiers run 1-6400, and is not the pool
this project uses.

Nothing inside either file announces which it is. The discriminators are the
row count and the identifier set, so `sipnet_calibration.labelings` carries the
expected row count on the spec and `scripts/ingest_labelings.py` refuses a file
that does not match, naming the older pool in the message.

| | 8000-site pool (used) | 6400-site pool (refused) |
|---|---|---|
| Path | `SDA_8k_site/site_pft.csv` | `site_pft.csv` |
| Data rows | 8000 | 6400 |
| Identifiers | 1-8000 | 1-6400 |
| `boreal.coniferous` | 2369 | 1877 |
| `temperate.deciduous.HPDA` | 1537 | 1252 |
| `semiarid.grassland_HPDA` | 4094 | 3271 |
| Identifiers quoted in the file | no | yes |

## What the file contains

Two columns, `site` and `pft`, one row per site, header quoted. `site` is the
1-8000 identifier shared with the rest of the project; `pft` is one of three
class names, written verbatim as the producer wrote them. Those names are the
join key to the reanalysis's per-PFT trait sample tables, so they are data and
are never renamed to this project's naming convention.

## Caveats carried by the source itself

Recorded here because they are properties of this file, not of our processing.
`data/README.md` documents them in full.

- **The three classes are an exact aggregation of the site shapefile's eight
  `landcover` classes**: 1-2 to `boreal.coniferous`, 3-4 to
  `temperate.deciduous.HPDA`, 5-8 to `semiarid.grassland_HPDA`. Measured over
  all 8000 rows, with no row off the relation. That the rule is the intended
  one is open question 24(k); the ingest asserts it either way.
- **The classes are coarse, and two of the three names mislead.**
  `boreal.coniferous` is not reliably needleleaf -- it holds Vaira Ranch, a
  California annual grassland -- and carries no latitude restriction, 805 of its
  2369 sites lying south of 40 N, median 48 N. `semiarid.grassland_HPDA` is the
  catch-all for everything non-forest, so 992 of its sites lie north of the
  Arctic Circle, median 52 N. A prior built on these labels inherits that
  coarseness.
- **No documentation of the file was found** beyond the reanalysis's own code.

## Re-copying

Writes nothing on the cluster.

```bash
B=/projectnb/dietzelab/dongchen/anchorSites/NA_runs/SDA_8k_site
scp <host>:$B/site_pft.csv reanalysis_site_pft.csv

C=/projectnb/dietzelab/guYANG/SIPNET_Model_Calibration/Final_PFT_assignment_v4
scp <host>:$C/final_8000_sites_with_final_pft_v4.csv /tmp/source.csv
python scripts/raw_sources/split_site_pft_16class.py --source /tmp/source.csv
```
