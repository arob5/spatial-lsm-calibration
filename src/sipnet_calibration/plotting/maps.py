"""L2 -- spatial panels, and the renderers behind them.

``map_panel(field, ax=None, *, stat="mean", render=None, extent="CONUS",
norm=None) -> Axes``

Requires the field to reduce to ``(site,)``. ``stat`` reduces ``member`` only; if
``time`` is still present, **raise** and direct the caller to select or aggregate
first -- consistent with aggregation being the caller's verb everywhere else.

Interpolation sits behind a protocol, because GP-based interpolation is expected
to replace the default::

    class SpatialRenderer(Protocol):
        def __call__(self, lon, lat, values, *, extent) -> Artist: ...

* ``TriRenderer`` (default) -- ``ax.tripcolor`` on the Delaunay triangulation.
  Cheap, bakes in no interpolation choice, honest about where data actually is.
  **Must mask triangles whose longest edge exceeds a threshold**, or fills
  appear across the Gulf of Mexico and across regions with no sites.
* ``GridRenderer`` -- ``scipy.interpolate.griddata`` onto a raster, for a smooth
  field.
* ``GPRenderer`` -- later. The real motivation for the seam: a GP gives a
  posterior sd panel alongside the mean, which ``tripcolor`` structurally
  cannot show.

Sites are 8000 irregular points spanning 7-82 deg N and 178 W-20 W, with only
~3640 inside a CONUS box, so CONUS-only assumptions are wrong and unprojected
lon/lat is not acceptable. Extent presets ``CONUS`` / ``NA`` / ``ALASKA``, plus a
raw ``bbox``. Colour scales must be shareable across a facet grid (common
``vmin``/``vmax``), centred for signed quantities, and log-scaled for the skewed
carbon pools.

BLOCKED -- do not implement yet
------------------------------
1. **No projection library is installable here** (issue #4). Neither ``cartopy``
   nor ``pyproj`` has a usable wheel on macOS 12 arm64: every pyproj arm64 wheel
   targets ``macosx_14_0`` (macOS 14+), on *every* Python version, so this is a
   platform incompatibility and downgrading Python does not help. ``cartopy`` is
   commented out of ``pyproject.toml``; do not re-add it expecting it to work
   locally. The leading option is to implement the Albers Equal Area forward
   transform directly (closed form, ~30 lines, testable against published
   reference coordinates) plus a vendored Natural Earth GeoJSON.
2. **The source projection of the site coordinates is not yet recorded.** The
   ``lon``/``lat`` in ``data/site_ids.csv`` were produced under a specific
   projection; an equal-area replot must start from the correct source CRS, and
   assuming plain WGS84 would be subtly wrong in a way that is invisible at a
   glance.

``basemap()`` in :mod:`sipnet_calibration.plotting.primitives` is the seam for
both, so neither blocks the renderers or the rest of the layer.
"""
