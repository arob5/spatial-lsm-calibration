"""Per-variable display metadata *and* semantics.

``VARIABLES[name] -> VarSpec(label, units, cmap, center, sign, transform)``

This registry is what replaces per-variable plotting functions (``plot_nee``,
``plot_gpp``, ...) -- the combinatorial trap this suite exists to avoid.

**The temporal aggregation rule is no longer this registry's.** It was to be an
``agg`` field here; it is pySIPNET's ``kind`` instead, and
:func:`sipnet_calibration.obs_ops.aggregate_time` reads that, falling back to
pySIPNET's own variable registries by name. Nothing about aggregation belongs
here.

Two fields are correctness, not cosmetics:

* ``units`` -- the single canonical unit for the variable. Adapters convert into
  it; ``validate_field()`` checks ``attrs["units"]`` against it. This is the
  guard against plotting model NEE (a per-timestep total) against observed NEE
  (apparently a rate) on one axis, which fails by orders of magnitude with no
  visual cue. The canonical NEE unit is still an open question.
* ``center`` -- ``0.0`` for signed fluxes such as NEE, so maps get a diverging
  colormap centered correctly. A sequential colormap on a signed flux is a
  genuinely misleading figure.

``sign`` records the direction convention (SIPNET is ``+ = to atmosphere``;
gap-filled eddy-covariance products vary and must be confirmed per product).
"""
