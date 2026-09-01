"""The site table and site selection.

The site table is read from ``data/processed/sites/sites.csv``, produced by
``scripts/ingest_sites.py`` from the point shapefile in ``data/raw/sites/``, the
PFT assignment table, and ``data/site_id_map.csv`` (Ameriflux ``Site_ID`` ->
integer site id, exact matching). See ``data/README.md`` for the column set.

``select_sites`` is the most-reused operation in the project and is deliberately
not a plotting concern: subsetting by PFT, bounding box, data availability, or
random sample happens once and the result is passed to adapters and plotters
alike.

Note the sites are 8000 *irregular points* spanning 7-82 deg N and
178 W-20 W. Only ~3640 fall inside a CONUS bounding box, so CONUS-only
assumptions are wrong.
"""
