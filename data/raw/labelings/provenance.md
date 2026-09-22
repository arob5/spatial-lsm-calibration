# Site labeling provenance

Where each file in this directory came from, what was done to it, and how to
tell whether it has drifted from its source.

A labeling maps every site to a class. It is **not** site metadata and is not a
column of the site table: which labeling to use is an experimental choice, so
each one is its own product. `data/README.md` sets out that reasoning under
[Site labelings](../../README.md#site-labelings) and open question 11.

These files are **copied into version control rather than symlinked**. Each is
a few hundred kilobytes, and the copy here is the only form in which the
labeling exists off the Boston University SCC.

Copied on **2026-09-21**.

## What was copied

| File in this directory | Source path | Source size | Source md5 | Source date |
|---|---|---|---|---|
| `reanalysis_site_pft.csv` | `/projectnb/dietzelab/dongchen/anchorSites/NA_runs/SDA_8k_site/site_pft.csv` | 234,229 | `31ceffd6a37b67b0b24b62b83df22ea1` | 2025-07-17 |

**Byte-verbatim.** Nothing was parsed, reformatted or renamed; the file in this
directory has the source md5 above. The values are integers and short strings,
so none of the float precision care that
[`raw/constraints/provenance.md`](../constraints/provenance.md) describes
applies here.

```bash
md5sum reanalysis_site_pft.csv   # 31ceffd6a37b67b0b24b62b83df22ea1
```

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
  `boreal.coniferous` is any needleleaf cover, with a median latitude of 48 N
  and members as far south as 7 N; `semiarid.grassland_HPDA` is the catch-all
  for everything non-forest and holds every Arctic site, with a median latitude
  of 52 N. A prior built on these labels inherits that coarseness.
- **No documentation of the file was found** beyond the reanalysis's own code.

## Re-copying

Writes nothing on the cluster.

```bash
B=/projectnb/dietzelab/dongchen/anchorSites/NA_runs/SDA_8k_site
scp <host>:$B/site_pft.csv reanalysis_site_pft.csv
```
