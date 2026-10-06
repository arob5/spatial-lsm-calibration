"""EKI with Gibbs updates of the noise scales, for a model with inferred noise.

The tempered target is joint in theta and the scales :math:`s`,

.. math::

    \\pi_\\varphi(\\theta, s) \\propto \\pi_0(\\theta) \\prod_k \\mathrm{IG}(s_k; a, b)
        \\prod_k \\mathcal N\\big(y_k; \\mathcal G_k(\\theta), s_k C_k\\big)^{\\varphi},

with :math:`s_k \\equiv 1` for an unscaled source. Each stage
:math:`\\varphi \\to \\varphi' = \\varphi + \\Delta\\varphi`, at an ensemble
:math:`(\\theta_j, s_j)` whose predictions :math:`g_j` are known:

1. **Increment.** The largest :math:`\\Delta\\varphi \\le 1 - \\varphi` whose
   weights keep :math:`\\mathrm{ESS} \\ge` ``config.EKI_ESS_FRACTION``
   :math:`J`, the weights the ratio of the scale-marginal targets,

   .. math::

       \\log w_j = \\sum_{k \\text{ scaled}} \\big[A_k(\\varphi) \\log B_{jk}(\\varphi)
           - A_k(\\varphi') \\log B_{jk}(\\varphi')\\big]
           - \\tfrac{\\Delta\\varphi}{2} \\sum_{k \\text{ unscaled}} q_{jk},

   :math:`A_k(\\varphi) = a + \\varphi n_k/2`, :math:`B_{jk}(\\varphi) = b +
   \\varphi q_{jk}/2`, :math:`q_{jk} = r_{jk}^\\top C_k^{-1} r_{jk}`.
2. **Theta step** (:func:`kalman_step`): the perturbed-observation update
   over :math:`\\Delta\\varphi` with noise covariance :math:`R(s_j)` for
   particle :math:`j` (``gain="per_particle"``), or :math:`R(\\bar s)` for
   every particle, :math:`\\bar s_k = (\\frac1J \\sum_j s_{jk}^{-1})^{-1}`
   (``gain="common"``).
3. **Evaluate** :math:`g_j = \\mathcal G(\\theta_j)`: the stage's one batch of
   SIPNET runs.
4. **Scale step**, exact Gibbs for :math:`\\pi_{\\varphi'}`:
   :math:`s_{jk} \\sim \\mathrm{IG}(a + \\varphi' n_k/2,\\ b + \\varphi' q_{jk}/2)`.

At :math:`\\varphi = 0` the ensemble is a prior draw of both. A particle whose
run fails has weight 0 and is moved to the valid particles' mean before the
next evaluation, as EnsKit's ``on_failure="repair"`` does.

The theta step, by the push-through identity
:math:`U(U^\\top U + R')^{-1} = (I + U R'^{-1} U^\\top)^{-1} U R'^{-1}`, with
:math:`T, U` the centered ensembles of theta and :math:`g` over
:math:`\\sqrt{J-1}` (rows are particles) and :math:`M_k = U_k C_k^{-1}
U_k^\\top`:

.. math::

    \\theta_j' = \\theta_j + T^\\top \\Big(I_J + \\Delta\\varphi \\sum_k
        \\tfrac{1}{s_{jk}} M_k\\Big)^{-1} \\Delta\\varphi \\sum_k
        \\tfrac{1}{s_{jk}} U_k C_k^{-1} d_{jk},
    \\qquad d_j = y - g_j - \\eta_j,\\ \\eta_{jk} \\sim \\mathcal N(0, s_{jk} C_k/\\Delta\\varphi).

For ``per_particle`` the parts of :math:`T` and :math:`U` explained by
:math:`\\log s` are first regressed out, so the gain uses an estimate of
:math:`\\mathbb E[\\operatorname{Cov}(\\theta, g \\mid s)]`, not the marginal
covariance (``MODEL.md``, "Common gain or per-particle gain").
"""

import jax
import numpy as np
import scipy.linalg

from sipnet_calibration.inference import eki_problem

from .. import config
from ..models import Model
from .records import Cost, append_history, save_samples

__all__ = ["GAINS", "kalman_step", "run_eki_gibbs", "scale_marginal_increment"]

#: The two versions of the theta step.
GAINS = ("common", "per_particle")


def run_eki_gibbs(
    model: Model,
    posterior,
    noise_model,
    directory,
    *,
    gain: str,
    ensemble_size: int,
) -> None:
    """Run EKI with Gibbs scale updates on *model* and write the run.

    *posterior* is the model's fixed posterior, whose forward map and prior
    it reads; *noise_model* its ``NoiseModel`` with inferred scales.
    """
    problem = eki_problem(posterior)
    cost = Cost()
    forward = cost.counted(problem.forward)
    rng = np.random.default_rng(config.EKI_SEED)
    theta = np.asarray(posterior.sample_prior(jax.random.key(config.EKI_SEED), ensemble_size))
    scales = noise_model.draw_prior_scales(rng, ensemble_size)
    phi, stage = 0.0, 0
    with cost.phase("eki"):
        predictions = np.asarray(forward(theta))
        while phi < 1.0:
            forms = noise_model.quadratic_forms(predictions)
            valid = ~np.isnan(predictions).any(axis=1)
            increment, ess = scale_marginal_increment(noise_model, forms, valid, phi)
            theta = kalman_step(
                noise_model, theta, predictions, scales, valid, increment, rng, gain=gain
            )
            predictions = np.asarray(forward(theta))
            phi = min(phi + increment, 1.0)
            forms = noise_model.quadratic_forms(predictions)
            scales = noise_model.draw_scales(rng, forms, phi)
            stage += 1
            _record_stage(directory, stage, phi, increment, ess, predictions, forms, scales)
    valid = ~np.isnan(predictions).any(axis=1)
    save_samples(
        directory,
        posterior,
        theta,
        model_name=model.name,
        algorithm=f"eki_gibbs_{gain}",
        log_prior=np.asarray(posterior.log_prior(theta)),
        log_likelihood=noise_model.log_likelihood(predictions),
        valid=valid,
        predictions=predictions,
        extra={
            **{f"quadratic_form_{name}": forms[name] for name in noise_model.scaled_names},
            **{f"{name}_noise_scale": values for name, values in scales.items()},
        },
    )
    cost.write(directory)


def scale_marginal_increment(noise_model, forms, valid, phi: float) -> tuple[float, float]:
    """The stage's increment and the ESS it keeps, by bisection on the
    scale-marginal weights of the module docstring; invalid particles weigh 0."""
    n_valid = int(valid.sum())
    target = config.EKI_ESS_FRACTION * n_valid

    def log_weights(increment: float) -> np.ndarray:
        after = phi + increment
        total = np.zeros(len(valid))
        for source in noise_model.sources:
            q = np.where(valid, forms[source.name], 0.0)
            if source.scaled:
                a, b, n = noise_model.shape, noise_model.prior_scale, source.size
                total += (a + phi * n / 2) * np.log(b + phi * q / 2) - (
                    a + after * n / 2
                ) * np.log(b + after * q / 2)
            else:
                total -= increment * q / 2
        return np.where(valid, total, -np.inf)

    def ess(increment: float) -> float:
        weights = np.exp(log_weights(increment) - np.max(log_weights(increment)))
        return weights.sum() ** 2 / (weights**2).sum()

    if n_valid < 2:
        raise ValueError(f"only {n_valid} particles are valid; EKI cannot continue")
    if ess(1.0 - phi) >= target:
        return 1.0 - phi, ess(1.0 - phi)
    low, high = 0.0, 1.0 - phi
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if ess(middle) >= target else (low, middle)
    return low, ess(low)


def kalman_step(
    noise_model, theta, predictions, scales, valid, increment, rng, *, gain: str
) -> np.ndarray:
    """The theta step of the module docstring, under *gain*; invalid
    particles move to the valid particles' mean."""
    J = len(theta)
    per_particle = _scales_by_particle(noise_model, scales, J, gain=gain)
    T = (theta[valid] - theta[valid].mean(0)) / np.sqrt(valid.sum() - 1)
    U = (predictions[valid] - predictions[valid].mean(0)) / np.sqrt(valid.sum() - 1)
    if gain == "per_particle" and noise_model.scaled_names:
        T, U = _regressed_on_log_scales(T, U, per_particle, valid, noise_model)
    new = np.tile(theta[valid].mean(0), (J, 1))
    system = np.zeros((J, valid.sum(), valid.sum()))
    right = np.zeros((J, valid.sum()))
    for source in noise_model.sources:
        Uk = U[:, source.positions]
        CiU = scipy.linalg.cho_solve((source.cholesky, True), Uk.T)          # (n_k, J_valid)
        M = Uk @ CiU
        eta = (source.cholesky @ rng.normal(size=(source.size, J))).T
        eta *= np.sqrt(per_particle[source.name] / increment)[:, None]
        d = noise_model.y[source.positions] - predictions[:, source.positions] - eta
        weight = increment / per_particle[source.name]                      # (J,)
        system += weight[:, None, None] * M[None]
        right += weight[:, None] * np.nan_to_num(d @ CiU)
    identity = np.eye(valid.sum())
    for j in np.flatnonzero(valid):
        new[j] = theta[j] + T.T @ np.linalg.solve(identity + system[j], right[j])
    return new


def _scales_by_particle(noise_model, scales, J: int, *, gain: str) -> dict[str, np.ndarray]:
    """Each source's scale per particle: its own, the mean-precision one, or 1."""
    by_particle = {}
    for source in noise_model.sources:
        if not source.scaled:
            by_particle[source.name] = np.ones(J)
        elif gain == "common":
            by_particle[source.name] = np.full(J, 1.0 / np.nanmean(1.0 / scales[source.name]))
        else:
            by_particle[source.name] = np.asarray(scales[source.name], float)
    return by_particle


def _regressed_on_log_scales(T, U, scales, valid, noise_model):
    """*T* and *U* less their least-squares fit on the centered log scales."""
    Z = np.column_stack([np.log(scales[name][valid]) for name in noise_model.scaled_names])
    Z = Z - Z.mean(0)
    projection = Z @ np.linalg.pinv(Z)
    return T - projection @ T, U - projection @ U


def _record_stage(directory, stage, phi, increment, ess, predictions, forms, scales) -> None:
    """One stage's row of ``history.csv``, printed."""
    valid = ~np.isnan(predictions).any(axis=1)
    row = {
        "stage": stage,
        "beta": phi,
        "increment": increment,
        "ess": ess,
        "n_valid": int(valid.sum()),
        **{f"median_{name}_noise_scale": float(np.nanmedian(v)) for name, v in scales.items()},
        **{f"mean_quadratic_form_{name}": float(np.nanmean(q)) for name, q in forms.items()},
    }
    append_history(directory, row)
    print(
        f"stage {stage}: beta {phi:.4g} (+{increment:.3g}), ESS {ess:.1f}, "
        f"{row['n_valid']} valid, median scales "
        + ", ".join(f"{np.nanmedian(v):.3g}" for v in scales.values()),
        flush=True,
    )
