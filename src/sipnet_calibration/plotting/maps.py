"""L2 -- spatial panels, and the renderers behind them.

``map_panel(field, ax=None, *, stat="mean", render=..., extent=..., norm=...)``.

Interpolation sits behind a protocol, because GP-based interpolation is
expected to replace the default::

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

Sites are irregular points spanning 7-82 deg N, so cartopy projections are
required (Albers Equal Area for CONUS) and unprojected lon/lat is not
acceptable. Extent presets ``CONUS`` / ``NA`` / ``ALASKA``, plus a raw ``bbox``.

Colour scales must be shareable across a facet grid, centred for signed
quantities, and log-scaled for the skewed carbon pools.
"""
