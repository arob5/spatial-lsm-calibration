"""Helpers for constructing ProbPipe priors over SIPNET parameters.

Note: ProbPipe distribution constructors require `name` as a keyword argument.
LogNormal uses `loc`/`scale` (mean and std of the underlying normal),
not `mu`/`sigma` as in some other libraries.
"""
from __future__ import annotations


def build_prior(param_specs: dict) -> "ProductDistribution":
    """Build a ProbPipe ProductDistribution prior from a dict of (name, spec) pairs.

    Parameters
    ----------
    param_specs:
        dict mapping param_name -> dict with keys:
            "prior_type": "lognormal" | "normal" | "beta"
            For lognormal: "loc" (mean of log), "scale" (std of log)
            For normal: "loc" (mean), "scale" (std)
            For beta: "alpha", "beta"

    Returns
    -------
    ProductDistribution
        Independent joint prior over all parameters.
    """
    from probpipe import ProductDistribution, LogNormal, Normal, Beta

    dist_map = {}
    for name, spec in param_specs.items():
        kind = spec["prior_type"]
        if kind == "lognormal":
            dist_map[name] = LogNormal(spec["loc"], spec["scale"], name=name)
        elif kind == "normal":
            dist_map[name] = Normal(spec["loc"], spec["scale"], name=name)
        elif kind == "beta":
            dist_map[name] = Beta(spec["alpha"], spec["beta"], name=name)
        else:
            raise ValueError(f"Unknown prior type: {kind!r}")
    return ProductDistribution(**dist_map)
