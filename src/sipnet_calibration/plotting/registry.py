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

* ``units`` -- the unit a field of this variable is expected to be in, for
  ``validate_field()`` to check ``attrs["units"]`` against. Nothing converts on
  the way in: a model field keeps pySIPNET's units and an observation product
  keeps its source's, and the **observation operator** is what converts one to
  the other. This is the guard against plotting model NEE (a per-timestep
  total) against observed NEE (a rate) on one axis, which fails by orders of
  magnitude with no visual cue.
* ``center`` -- ``0.0`` for signed fluxes such as NEE, so maps get a diverging
  colormap centered correctly. A sequential colormap on a signed flux is a
  genuinely misleading figure.

``sign`` records the direction convention (SIPNET is ``+ = to atmosphere``;
gap-filled eddy-covariance products vary and must be confirmed per product).
"""
