"""Shared helpers for spatial SIPNET parameter calibration.

Enables float64 before any array is created. JAX defaults to float32, and
float32 is not adequate for the Gaussian-conditioning algebra downstream --
anomaly centring and the whitened SVD both lose accuracy visibly.

Two caveats, both verified:

* Arrays created *before* this runs stay float32 and are never retroactively
  promoted. Importing this package first is therefore load-bearing.
* ``ProcessPoolExecutor`` workers (as used by ``PyEns``) do **not** inherit the
  setting. Set ``JAX_ENABLE_X64=1`` in the environment for anything that forks
  workers; the config call here only covers the parent process.
"""
import jax

jax.config.update("jax_enable_x64", True)
