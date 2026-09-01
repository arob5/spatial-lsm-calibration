"""Per-variable display metadata *and* semantics.

``VARIABLES[name] -> VarSpec(label, units, agg, cmap, center, sign, transform)``

This registry is what replaces per-variable plotting functions (``plot_nee``,
``plot_gpp``, ...) -- the combinatorial trap this suite exists to avoid.

Three fields are correctness, not cosmetics:

* ``agg`` -- the temporal aggregation rule. ``"sum"`` for per-timestep flux
  totals (``nee``, ``gpp``, ``par``, ``precip``), ``"mean"`` for intensive state
  (``tair``, ``vpd``), instantaneous for stocks (carbon pools, ``AbvGrndWood``,
  ``LAI``). ``obs_ops.aggregate_time`` reads this. Defaulting everything to
  ``"mean"`` makes every NEE aggregation in the project wrong by 8x.
* ``units`` -- the single canonical unit for the variable. Adapters convert into
  it; ``validate_field()`` checks ``attrs["units"]`` against it. This is the
  guard against plotting model NEE (a per-timestep total) against observed NEE
  (apparently a rate) on one axis, which fails by orders of magnitude with no
  visual cue. The canonical NEE unit is still an open question.
* ``center`` -- ``0.0`` for signed fluxes such as NEE, so maps get a diverging
  colormap centred correctly. A sequential colormap on a signed flux is a
  genuinely misleading figure.

``sign`` records the direction convention (SIPNET is ``+ = to atmosphere``;
gap-filled eddy-covariance products vary and must be confirmed per product).
"""
