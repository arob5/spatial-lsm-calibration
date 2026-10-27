"""The toy spatial calibration problem.

A square region with scattered sites, a one-pool carbon model run
independently at each site, a spatial prior over the model's two parameters,
and observations that are rich at a few tower sites and sparse elsewhere.
Nothing here performs inference; ``inference`` does that.

- ``domain``: the sites, the map grid, the climate covariate and the drivers.
- ``forward_model``: the carbon model, ``CarbonModel.simulate``.
- ``spatial_prior``: the prior over the parameter fields, and predicting the
  fields away from the sites.
- ``observations``: which sites observe what, the noise, the synthetic truth
  and data, and the log likelihood.
"""
