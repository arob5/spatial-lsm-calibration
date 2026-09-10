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

Sites are 8000 irregular points spanning 7-82 deg N and 178 W-20 W, with a
little under half inside a CONUS box (``data/README.md``), so CONUS-only
assumptions are wrong and unprojected lon/lat is not acceptable. Extent presets come from
:data:`sipnet_calibration.sites.EXTENTS` -- ``CONUS``, ``NORTH_AMERICA`` and
``ALASKA``, the middle one spelled out rather than ``NA`` -- plus a raw
``bbox``. Color scales must be shareable across a facet grid (common
``vmin``/``vmax``), centered for signed quantities, and log-scaled for the skewed
carbon pools.

The projection
--------------
**Settled, and not this module's to define.** The display projection is a
Lambert Azimuthal Equal Area centered at 50 N, 100 W, held as
:data:`sipnet_calibration.projection.SITE_PROJECTION`. Project coordinates with
:meth:`~sipnet_calibration.projection.Projection.forward` and get axes limits
for an extent from
:meth:`~sipnet_calibration.projection.Projection.projected_bounds`, which
samples the box boundary rather than its corners. Do not define projection
parameters here, and do not reach for degrees: the choice, the alternatives and
the distortion measured over all 8000 sites are recorded on issue #4.

Two properties of it bear on the renderers. It is equal-area, which is what
makes a density or heatmap panel honest. And its anisotropy stays under 1.3
across the pool, which matters because ``TriRenderer`` is to triangulate
*after* projecting -- a Delaunay triangulation is not affine-invariant, so under
a strongly anisotropic projection the mesh would be an artifact of the
projection rather than of where the sites are -- and because the long-edge mask
threshold is a projected length, which only means one ground distance where the
local scale is close to isotropic.

Still outstanding -- the basemap
-------------------------------
Nothing here is blocked on the projection any more. What is missing is a
vendored Natural Earth coastline and states GeoJSON, small enough to track,
projected with the same forward transform. ``basemap()`` in
:mod:`sipnet_calibration.plotting.primitives`, which is a name in that module's
contract rather than a function yet, is the seam for it, so the renderers and
the rest of this layer can be built before it exists.

**No projection *library* is installable here** (issue #4), which is why the
forward transform is implemented in numpy rather than through cartopy. Neither
``cartopy`` nor ``pyproj`` has a usable wheel on macOS 12 arm64: every pyproj
arm64 wheel targets ``macosx_14_0`` (macOS 14+), on *every* Python version, so
this is a platform incompatibility and downgrading Python does not help.
``cartopy`` is commented out of ``pyproject.toml``; do not re-add it expecting
it to work locally.
"""
