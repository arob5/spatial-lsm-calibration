"""L1 -- axes-level primitives.

Every function here has signature ``(ax, <plain numpy>, **style) -> artist``.
No pandas, no xarray, no figure creation. That is what makes them reusable
everywhere and trivial to test.

Planned: ``band``, ``fan``, ``spaghetti``, ``line``, ``points``,
``step_intervals``, ``map_points``, ``map_raster``, ``basemap``.

``fan`` takes *symmetric interval levels* -- ``(0.5, 0.9)`` -- not four raw
quantiles: it generalises to n bands, reads correctly in a legend, and cannot be
handed a non-monotone tuple.

``spaghetti`` takes ``n_max`` and decimates. A single site is 38k timesteps x 25
members; ``fan`` collapses that via quantiles but ``spaghetti`` does not, and
the guardrail belongs here rather than at every call site.
"""
