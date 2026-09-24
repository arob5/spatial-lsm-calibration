# Natural Earth provenance

Where each archive in this directory came from, and how to tell whether it has
drifted from its source.

These are the geographic layers the map basemap is drawn from: coastlines,
lakes, country boundaries and state/province boundaries, at Natural Earth's
1:50m scale. They are not project data. They are tracked so that the basemap
can be rebuilt without a network connection, and so that the bytes it was built
from are on record.

Natural Earth is in the public domain
(<https://www.naturalearthdata.com/about/terms-of-use/>).

## How they got here

`scripts/raw_sources/download_natural_earth.py` downloads each archive from
Natural Earth's CDN, `https://naciscdn.org/naturalearth/50m/<theme>/<file>`,
the location the download links on naturalearthdata.com resolve to, and
refuses one whose md5 differs from the value recorded in the script's
`SOURCES`. The archives are kept byte for byte as served: nothing is unpacked
or edited.

`scripts/build_basemap.py` reads them in place and writes
`src/sipnet_calibration/plotting/basemap_data/natural_earth_50m.npz`, which is
what the plotting layer loads. `tests/test_basemap.py` checks both that the
archives here have the recorded md5s and that the tracked basemap is what they
build to.

## What is here

| File | Theme | Layer | Natural Earth version | md5 |
|---|---|---|---|---|
| `ne_50m_coastline.zip` | physical | `coastline` | 4.1.0 | `7639330d2519efa1005eac0407172130` |
| `ne_50m_lakes.zip` | physical | `lakes` | 5.0.0 | `93de5a3d32451e7c18c75425b2f071dd` |
| `ne_50m_admin_0_boundary_lines_land.zip` | cultural | `borders` | 5.1.0 | `2c6695791ef99755162094e7bfad97b2` |
| `ne_50m_admin_1_states_provinces_lines.zip` | cultural | `states` | 5.1.0 | `4f1373b05294a5848886e72f0f6a30a5` |

The version is the one in each archive's own `*.VERSION.txt`. Downloaded on
2026-09-23.

## Adopting a new release

Natural Earth republishes a layer in place under the same URL, so a changed
md5 on download means a new release rather than a corrupt transfer. To adopt
one deliberately:

```bash
uv run python scripts/raw_sources/download_natural_earth.py --no-check
# record the printed md5s in SOURCES and in the table above, then
uv run python scripts/build_basemap.py
uv run pytest tests/test_basemap.py
```
