"""Replicated observations: a predictive's draws of the data, noise included.

At each sample :math:`m` a predictive ran, the replicated data of source
:math:`k` are

.. math::

    y^{\\mathrm{rep}}_{mk} = \\mathcal G_k(\\theta_m) + \\sqrt{s_{mk}}\\, L_k z_{mk},
    \\qquad z_{mk} \\sim \\mathcal N(0, I),\\quad L_k L_k^\\top = C_k,

with :math:`C_k` the source's reference covariance under the model's error
model and :math:`s_{mk}` the sample's noise scale: its draw from the scale's
posterior where the noise is inferred (``predictive/samples.csv``), 1 where
it is fixed or the source has no scale. A prior predictive passes draws of
the scales' prior instead. So :math:`y^{\\mathrm{rep}}` is a draw of the
posterior (or prior) predictive of the data, to be set against the observed
values.
"""

from collections.abc import Mapping

import numpy as np
import xarray as xr

from sipnet_calibration.conventions import SAMPLE

from ..model.likelihood import NoiseModel

__all__ = ["check_observed_values_are_the_noise_models", "replicated_observations"]


def replicated_observations(
    predicted: Mapping[str, xr.DataArray],
    observed: Mapping[str, xr.Dataset],
    noise_model: NoiseModel,
    scales: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """:math:`y^{\\mathrm{rep}}` of each source of *noise_model*, ``(K, n_k)``.

    *predicted* holds each source's predictions on ``(sample, ...)``, as a
    predictive writes them, and *observed* its observed values, which must be
    the noise model's (they fix the order of a source's entries). *scales*
    holds each scaled source's scale per sample, ``(K,)``; a source missing
    from it has scale 1.

    Raises
    ------
    ValueError
        If a source's observed values are not the noise model's ``y``.
    """
    replicated = {}
    for source in noise_model.sources:
        check_observed_values_are_the_noise_models(observed[source.name], noise_model, source)
        values = predicted[source.name].transpose(SAMPLE, ...).to_numpy()
        values = values.reshape(values.shape[0], -1)
        scale = np.asarray(scales.get(source.name, np.ones(len(values))), dtype=float)
        noise = rng.standard_normal(values.shape) @ source.cholesky.T
        replicated[source.name] = values + np.sqrt(scale)[:, None] * noise
    return replicated


# ── checks ──


def check_observed_values_are_the_noise_models(observed: xr.Dataset, noise_model, source) -> None:
    """A source's observed values, in the predictive's order, are the noise
    model's ``y`` for it, so the predictions and the covariance line up."""
    values = np.ravel(observed["value"].to_numpy())
    if not np.allclose(values, noise_model.y[source.positions], equal_nan=True):
        raise ValueError(
            f"the observed values of {source.name!r} are not the noise model's, "
            "or not in its order; rerun the predictive with the current observations"
        )
