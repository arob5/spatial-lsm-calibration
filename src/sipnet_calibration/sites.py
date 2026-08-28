"""The site table and site selection.

The site table is the join of ``data/raw/site_ids.csv`` (8000 sites: id, lon,
lat, name), the PFT assignment csv, and ``data/raw/site_id_map.csv`` (185
Ameriflux ``Site_ID`` -> integer site id, exact matching).

``select_sites`` is the most-reused operation in the project and is deliberately
not a plotting concern: subsetting by PFT, bounding box, data availability, or
random sample happens once and the result is passed to adapters and plotters
alike.

Note the sites are 8000 *irregular points* spanning 7-82 deg N and
178 W-20 W. Only ~3640 fall inside a CONUS bounding box, so CONUS-only
assumptions are wrong.
"""
