"""The canonical field convention and the adapters that produce it.

A **canonical field** is an :class:`xarray.DataArray` with

* dims a *subset* of ``(member, site, time)``;
* ``lon`` and ``lat`` as *non-dimension* coordinates on ``site``, whenever
  ``site`` is a dim;
* ``units`` and ``long_name`` in ``attrs``;
* a ``name`` that is a key in the ``VARIABLES`` registry.

This is a convention plus :func:`validate_field`, deliberately **not** a wrapper
class: the three operations this project needs most are ``.quantile(dim="member")``,
``.resample(time=...)`` and ``.sel(site=...)``, and a wrapper would spend the
project re-exporting them.

Dims are a subset by design. A deterministic single run is ``(time,)``; an IC
map is ``(member, site)``; the NEE observation ensemble is
``(member, site, time)``; a per-site calibrated parameter is ``(member, site)``.
Plotters branch on *presence of the* ``member`` *dim*, never on a mode keyword.

One DataArray per variable, not one aligned Dataset -- NEE is 3-hourly, AGB/LAI
are annual July-15 snapshots, ICs are static, and forcing a shared ``time``
index costs NaN padding for nothing. Facet-by-variable takes
``dict[str, DataArray]``, which is also what multi-variable adapters return
(``from_clim`` covers 12 variables).

Identifiers
-----------
* ``site`` is the **handed-down integer id, 1-8000**. It has to be: only 185 of
  the 8000 sites are Ameriflux sites. These ids are a shared key with
  collaborators' files -- **never renumber them.** A spatially meaningful
  ordering, if wanted, is a *separate* coordinate (e.g. a Hilbert rank), not a
  renumbering.
* Ameriflux ``Site_ID`` and ``pft`` are non-dimension coords on ``site``, NaN
  for sites lacking them.
* ``member`` is a 0-based integer. Two traps: the NEE csv's ``ens_mean`` column
  must **never** become a member (it would corrupt every quantile), and integer
  labels let xarray silently align members across unrelated sources -- whether
  that pairing is meaningful is still an open question.

Adapter notes
-------------
Adapters live here so that no plotter ever accepts a ``SIPNETResult`` or a path.
They are also where unit conversion happens: nothing downstream reconciles
units, and :func:`validate_field` checks ``attrs["units"]`` against the registry.

* **The IC adapter must pass** ``decode_times=False``. The IC netCDFs carry an
  unsubstituted template, ``units = "days since [year]-01-01 00:00:00 UTC"``,
  which no calendar library can parse -- installing ``cftime`` does not help.
  The ``time`` dim there is length 1 and carries no information.
* ``from_eki_predictions`` unstacks a ``(J, N)`` block using the MultiIndex from
  :func:`sipnet_calibration.obs_ops.obs_index` -- the same object the
  observation operator used to build ``y``::

      (xr.DataArray(predictions, dims=("member", "obs"))
         .assign_coords(obs=obs_index)
         .unstack("obs"))
"""
