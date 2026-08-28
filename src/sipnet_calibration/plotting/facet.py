"""L3 -- the one generic facet function.

``facet(items, panel_fn, *, ncol, share=..., common_lims=..., legend="dedup")``
owns figure and axes construction, shared limits, and legend de-duplication,
with thin ``by_site`` / ``by_variable`` / ``by_aggregation`` wrappers.

Most of the repetitive figure code in a project like this is
figure/axes/limit/legend plumbing. Centralising it here is where the bulk of the
saving comes from, and it is the reason L2 panels never create their own figure.
"""
