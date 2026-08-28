"""Per-variable display metadata.

``VARIABLES[name] -> VarSpec(label, units, cmap, center, transform)``.

This registry is what replaces per-variable plotting functions (``plot_nee``,
``plot_gpp``, ...) -- the combinatorial trap this suite exists to avoid.

``center=0.0`` on signed fluxes such as NEE is correctness rather than
cosmetics: a sequential colormap on a signed quantity is a genuinely misleading
map, and it should be right by default rather than remembered per figure.
"""
