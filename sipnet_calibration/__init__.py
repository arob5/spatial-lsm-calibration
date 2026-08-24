"""Shared helpers for spatial SIPNET parameter calibration.

Importing this package enables JAX float64. JAX defaults to float32, which is
not accurate enough for the Gaussian-conditioning algebra this code performs.

Import it before creating any array: arrays made beforehand stay float32 and
are not promoted afterwards.

Notes
-----
The setting applies to the current process only. ``ProcessPoolExecutor``
workers, such as those used by ``PyEns``, do not inherit it; set the
environment variable ``JAX_ENABLE_X64=1`` for anything that forks workers.
"""
import jax

jax.config.update("jax_enable_x64", True)
