"""The toy carbon model: SIPNET's stand-in, one carbon pool run monthly at a site.

With parameters theta = (log a, log k), productivity ``a`` and turnover rate
``k``, and a site's monthly temperature ``T_t`` and light ``L_t``:

    GPP_t = a L_t sigmoid((T_t - T_half) / T_width)       photosynthesis
    k_t   = k exp(Q (T_t - T_ref))                        turnover, warmer is faster
    R_t   = C_t (1 - exp(-k_t))                           respiration
    C_t+1 = C_t + GPP_t - R_t,   C_0 = initial pool
    NEE_t = R_t - GPP_t                                   net ecosystem exchange

Respiration takes the fraction of the pool that decays within the month at
rate ``k_t``, ``1 - exp(-k_t)``, rather than ``k_t C_t``: the two agree to
first order in ``k_t``, and this form keeps the pool positive for every
parameter value.

The model's output at a site is its 60 monthly NEE values followed by the
pool at the end of each of the 5 years (``nee_indices`` and
``annual_pool_indices``).

Notes
-----
Why a tower matters, in this model. Monthly NEE resolves GPP's seasonal
cycle, which pins ``a`` directly. The annual pools alone mostly constrain a
*combination* of ``a`` and ``k``: more productive and faster turnover can
give the same pool. So at a site with pools alone the posterior is a curved
ridge, which is equifinality. The pool starts well below its steady state,
so the ridge bends rather than being a straight line in theta, which makes
it a fair test of Gaussian approximations.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["CarbonModel"]


@dataclass(frozen=True)
class CarbonModel:
    """The toy model's fixed constants; ``simulate`` runs it."""

    n_months: int
    initial_pool: float
    temperature_half_saturation: float
    temperature_saturation_width: float
    respiration_temperature_sensitivity: float
    reference_temperature: float

    # ── outputs ──

    @property
    def n_years(self) -> int:
        return self.n_months // 12

    @property
    def n_outputs(self) -> int:
        return self.n_months + self.n_years

    @property
    def nee_indices(self) -> np.ndarray:
        """Where the monthly NEE values sit in the output vector."""
        return np.arange(self.n_months)

    @property
    def annual_pool_indices(self) -> np.ndarray:
        """Where the end-of-year pools sit in the output vector."""
        return self.n_months + np.arange(self.n_years)

    # ── running the model ──

    def simulate(self, theta, temperature, light) -> jax.Array:
        """Run the model; returns ``(..., n_outputs)``.

        Parameters
        ----------
        theta:
            ``(..., 2)``: log productivity and log turnover rate.
        temperature, light:
            ``(..., n_months)``, broadcast against ``theta``'s leading dims,
            so one site's drivers can run a whole ensemble of parameters.

        Notes
        -----
        Written in JAX so the MCMC reference can differentiate it; the
        site-by-site and IPL computations only evaluate it.
        """
        theta = jnp.asarray(theta)
        productivity = jnp.exp(theta[..., 0])[..., None]
        turnover = jnp.exp(theta[..., 1])[..., None]

        temperature_limitation = jax.nn.sigmoid(
            (temperature - self.temperature_half_saturation) / self.temperature_saturation_width
        )
        gpp = productivity * light * temperature_limitation
        monthly_turnover = turnover * jnp.exp(
            self.respiration_temperature_sensitivity * (temperature - self.reference_temperature)
        )
        respired_fraction = 1.0 - jnp.exp(-monthly_turnover)
        gpp, respired_fraction = jnp.broadcast_arrays(gpp, respired_fraction)

        def one_month(pool, month_inputs):
            gpp_t, respired_fraction_t = month_inputs
            respiration_t = pool * respired_fraction_t
            next_pool = pool + gpp_t - respiration_t
            return next_pool, (respiration_t - gpp_t, next_pool)

        # lax.scan steps over the leading axis, so time goes first.
        initial_pool = jnp.full(gpp.shape[:-1], self.initial_pool, dtype=gpp.dtype)
        _, (nee, pool) = jax.lax.scan(
            one_month,
            initial_pool,
            (jnp.moveaxis(gpp, -1, 0), jnp.moveaxis(respired_fraction, -1, 0)),
        )
        nee = jnp.moveaxis(nee, 0, -1)
        # pool[t] is the pool at the end of month t; months 11, 23, ... end a year.
        annual_pool = jnp.moveaxis(pool[11::12], 0, -1)
        return jnp.concatenate([nee, annual_pool], axis=-1)
