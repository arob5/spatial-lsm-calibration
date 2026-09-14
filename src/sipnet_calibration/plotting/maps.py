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
:data:`sipnet_calibration.projection.SITE_PROJECTION` and applied by PROJ.
Project coordinates with
:meth:`~sipnet_calibration.projection.Projection.forward`, get axes limits for
an extent from
:meth:`~sipnet_calibration.projection.Projection.projected_bounds`, which
samples the box boundary rather than its corners, and get local distortion from
:meth:`~sipnet_calibration.projection.Projection.factors`. Do not define
projection parameters here, and do not reach for degrees: the choice, the
alternatives and the distortion measured over all 8000 sites are recorded on
issue #4.

Set ``ax.set_aspect("equal")``. Without it the equal-area property, which is
the whole reason for this projection, does not survive to the page.

Two properties of it bear on the renderers. It is equal-area, which is what
makes a density or heatmap panel honest. And its anisotropy stays under 1.3
across the pool, which matters because ``TriRenderer`` is to triangulate
*after* projecting -- a Delaunay triangulation is not affine-invariant, so under
a strongly anisotropic projection the mesh would be an artifact of the
projection rather than of where the sites are -- and because the long-edge mask
threshold is a projected length, which only means one ground distance where the
local scale is close to isotropic --
:meth:`~sipnet_calibration.projection.Projection.factors` is what converts
between the two, and ``tissot_semimajor`` is the bound to use.

North is not up. Over the site pool projected north rotates from -71 degrees on
the Chukchi coast to +75 in northeast Greenland, and over the whole
``NORTH_AMERICA`` extent from -78 to +79, so a single north arrow on a
full-domain panel is wrong nearly everywhere on it; draw the graticule instead.
``factors`` reports the rotation at a point as ``meridian_convergence``, in
PROJ's sign convention rather than the one the eye reads.

Still outstanding: the basemap
------------------------------
Nothing here is blocked on the projection any more. What is missing is a
vendored Natural Earth coastline and states GeoJSON, small enough to track,
projected with the same forward transform. A ``basemap()`` primitive, which
:mod:`sipnet_calibration.plotting.primitives` does not name yet, is the seam for
it, so the renderers and the rest of this layer can be built before it exists.

``cartopy`` is still absent, but not for the reason issue #4 gives: that issue
assumed macOS 12, where no pyproj arm64 wheel could be installed, and the
workstation has since moved past macOS 14. ``pyproj`` is now a dependency.
``cartopy``'s own arm64 wheels stop at cp313 while the development venv is on
3.14, so adopting it would mean bounding the interpreter from above, which
``requires-python`` does not do today -- worth weighing when the basemap is
built, since cartopy would supply the coastlines and the gridline labels
outright.
"""
