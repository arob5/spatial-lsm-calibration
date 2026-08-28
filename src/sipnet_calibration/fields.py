"""The canonical field convention and the adapters that produce it.

A **canonical field** is an :class:`xarray.DataArray` with

* dims a *subset* of ``(member, site, time)``;
* ``lon`` and ``lat`` as *non-dimension* coordinates on ``site``;
* ``units`` and ``long_name`` in ``attrs``.

This is a convention plus :func:`validate_field`, deliberately **not** a wrapper
class: the three operations this project needs most are ``.quantile(dim="member")``,
``.resample(time=...)`` and ``.sel(site=...)``, and a wrapper would spend the
project re-exporting them.

Dims are a subset by design. A deterministic single run is ``(time,)``; an IC
map is ``(member, site)``; the NEE observation ensemble is
``(member, site, time)``. Plotters branch on *presence of the* ``member`` *dim*,
never on a mode keyword.

One DataArray per variable, not one aligned Dataset -- NEE is 3-hourly, AGB/LAI
are annual July-15 snapshots, ICs are static, and forcing a shared ``time``
index costs NaN padding for nothing. Facet-by-variable takes
``dict[str, DataArray]``.

Adapters live here so that no plotter ever accepts a ``SIPNETResult`` or a path.

API details verified against the installed stack (xarray 2026.4, zarr 3.3):

* The EKI adapter must build the ``(site, variable, time)`` MultiIndex
  coordinate explicitly --
  ``xr.Coordinates.from_pandas_multiindex(midx, "obs")`` -- and pass that to
  ``assign_coords``. Passing a bare ``pd.MultiIndex`` still works but raises a
  ``FutureWarning``: implicit promotion to multiple indexed coordinates is being
  removed.
* ``lon``/``lat`` survive a zarr round-trip as non-dimension coordinates, so the
  processed store needs no side-car for site geometry.
* The IC netCDFs must be opened with ``decode_times=False``. Their ``time``
  units attribute is a literal unsubstituted template,
  ``"days since [year]-01-01 00:00:00 UTC"``, which no calendar library can
  parse. The dim is length-1 and meaningless for static ICs regardless.
"""
