"""The algorithms that calibrate the toy, and the summaries that compare them.

Each method returns a ``PosteriorMoments`` over the training sites'
parameters, so prediction and scoring (``evaluation``) are the same for all
of them.

- ``moments``: ``PosteriorMoments``, the common form of a result.
- ``site_by_site``: each site calibrated alone, exactly, on a grid.
- ``spatial_mcmc``: the exact spatial posterior by NUTS, the reference.
- ``ipl``: iterated posterior linearization, the scalable method: model
  runs at each site, combined by a Gaussian update over all sites.
- ``evaluation``: maps and held-out predictions from any method's moments,
  and the error summaries the slides quote.
- ``density``: a site's 2D posterior as a density on a grid: its contours,
  the KDE of draws, and the checks of that KDE.
"""
