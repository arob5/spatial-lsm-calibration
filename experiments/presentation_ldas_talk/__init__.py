"""The LDAS talk: slides on spatial calibration, and the toy experiment behind them.

The package is organized by role, so each kind of choice has one home:

- ``config``: every number (sizes, seeds, model constants, settings).
- ``toy``: the toy *problem*: the region and its drivers (``domain``), the
  carbon model standing in for SIPNET (``forward_model``), the spatial prior
  over its parameters (``spatial_prior``), and what each site observes
  (``observations``).
- ``inference``: the *algorithms* that calibrate the toy: site by site,
  the exact spatial posterior by MCMC, and the scalable method (IPL); and
  the summaries that compare them.
- ``illustrations``: computations that are pictures, not experiments: the
  1D transect and prior draws over the real North American site pool.
- ``results``: the files the computations write and the figures read.
- ``run``: the scripts, each a short workflow over the pieces above.
- ``plots`` and ``style``: the figures, which read results and compute
  nothing.

Importing the package turns on 64-bit JAX, which the MCMC reference and the
model's gradients need for accurate log densities.
"""

import jax

jax.config.update("jax_enable_x64", True)
