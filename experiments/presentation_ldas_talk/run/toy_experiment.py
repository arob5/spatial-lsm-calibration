"""Run the toy spatial calibration experiment and save every result.

Overview
--------
Build the toy problem, draw a synthetic truth and data from it, calibrate
with three methods, and predict maps and held-out sites from each:

1. **problem**: the region, its drivers, the carbon model and the spatial
   prior (``toy``);
2. **truth and data**: one draw of the parameter fields from the prior, the
   observation design, and noisy observations of the model at the truth;
3. **calibration**: site by site (exact, on a grid), the spatial posterior
   by NUTS (the exact reference), and IPL (the scalable method);
4. **prediction and scoring**: maps of the field and predictions at the
   held-out sites from each posterior, scored against the truth, and IPL's
   iterations scored against the reference.

Every number comes from ``config``.

Output data
-----------
``config.TOY_RESULTS_FILE``: a ``results.ToyResults``.

Usage
-----
::

    uv run python -m experiments.presentation_ldas_talk.run.toy_experiment
"""

import time

import numpy as np

from .. import config, results
from ..inference.evaluation import (
    Predictions,
    held_out_scores,
    predict_from_posterior,
    prior_predictions,
    sd_ratio_to_reference,
    standardized_error_to_reference,
)
from ..inference.ipl import IPLResult, run_ipl
from ..inference.moments import PosteriorMoments
from ..inference.site_by_site import site_by_site_posteriors
from ..inference.spatial_mcmc import sample_spatial_posterior
from ..toy.domain import Region, build_region, simulate_drivers
from ..toy.forward_model import CarbonModel
from ..toy.observations import design_observations, simulate_observations
from ..toy.spatial_prior import SpatialPrior

# ── entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    rng = np.random.default_rng(config.TOY_SEED)

    # 1. The problem.
    region = build_region(
        rng,
        n_training_sites=config.N_TRAINING_SITES,
        n_held_out_sites=config.N_HELD_OUT_SITES,
        map_grid_size=config.MAP_GRID_SIZE,
    )
    drivers = simulate_drivers(
        region.training_locations,
        rng,
        n_months=config.N_MONTHS,
        mat_center=config.MAT_CENTER,
        mat_scale=config.MAT_SCALE,
        seasonal_amplitude=config.SEASONAL_AMPLITUDE,
        weather_autocorrelation=config.WEATHER_AUTOCORRELATION,
        weather_innovation_sd=config.WEATHER_INNOVATION_SD,
    )
    model = CarbonModel(
        n_months=config.N_MONTHS,
        initial_pool=config.INITIAL_POOL,
        temperature_half_saturation=config.TEMPERATURE_HALF_SATURATION,
        temperature_saturation_width=config.TEMPERATURE_SATURATION_WIDTH,
        respiration_temperature_sensitivity=config.RESPIRATION_TEMPERATURE_SENSITIVITY,
        reference_temperature=config.REFERENCE_TEMPERATURE,
    )
    prior = SpatialPrior(
        beta=config.BETA,
        field_sd=config.FIELD_SD,
        length_scale=config.LENGTH_SCALE,
        site_noise_sd=config.SITE_NOISE_SD,
    )

    # 2. The truth and the data.
    all_site_locations = np.vstack([region.training_locations, region.held_out_locations])
    truth_at_sites, truth_on_map = prior.draw(rng, all_site_locations, region.map_grid_locations)
    true_theta = truth_at_sites.value
    true_theta_training = true_theta[: config.N_TRAINING_SITES]
    design = design_observations(
        model,
        drivers,
        prior.trend(region.training_locations),
        rng,
        tower_fraction=config.TOWER_FRACTION,
        pool_relative_noise=config.POOL_RELATIVE_NOISE,
        nee_relative_noise=config.NEE_RELATIVE_NOISE,
    )
    data = simulate_observations(model, true_theta_training, drivers, design, rng)

    # 3. Calibration, three ways.
    with _timed("site by site"):
        site_by_site = site_by_site_posteriors(
            model,
            drivers,
            design,
            data,
            prior.trend(region.training_locations),
            prior.site_marginal_sd(),
            grid_size=config.SITE_GRID_SIZE,
            half_width_in_prior_sds=config.SITE_GRID_HALF_WIDTH_IN_PRIOR_SDS,
            half_width_in_posterior_sds=config.SITE_GRID_HALF_WIDTH_IN_POSTERIOR_SDS,
            n_refinements=config.SITE_GRID_REFINEMENTS,
        )
    with _timed("spatial reference (NUTS)"):
        reference = sample_spatial_posterior(
            model,
            drivers,
            design,
            data,
            prior,
            region.training_locations,
            seed=config.MCMC_SEED,
            n_chains=config.MCMC_CHAINS,
            n_warmup=config.MCMC_WARMUP,
            n_samples=config.MCMC_SAMPLES,
        )
    print(
        f"  NUTS: {reference.model_runs_per_site} model runs per site, "
        f"max R-hat {reference.max_r_hat:.3f}, min ESS {reference.min_effective_sample_size:.0f}"
    )
    with _timed("IPL"):
        ipl = run_ipl(
            model,
            drivers,
            design,
            data,
            prior,
            region.training_locations,
            ensemble_size=config.IPL_ENSEMBLE_SIZE,
            max_iterations=config.IPL_MAX_ITERATIONS,
            convergence_tolerance=config.IPL_CONVERGENCE_TOLERANCE,
            seed=config.IPL_SEED,
        )
    print(f"  IPL: {len(ipl.history)} iterations, {ipl.runs_per_site[-1]} model runs per site")

    # 4. Prediction and scoring.
    true_theta_held_out = true_theta[config.N_TRAINING_SITES :]
    no_map = np.full((len(region.map_grid_locations), prior.n_parameters), np.nan)
    toy_results = results.ToyResults(
        training_locations=region.training_locations,
        held_out_locations=region.held_out_locations,
        map_grid_locations=region.map_grid_locations,
        map_grid_size=region.map_grid_size,
        is_tower=design.is_tower,
        observed=design.observed,
        noise_sd=design.noise_sd,
        data=data,
        true_model_output=np.asarray(model.simulate(true_theta_training, drivers.temperature, drivers.light)),
        temperature=drivers.temperature,
        light=drivers.light,
        truth=_truth_results(truth_at_sites, truth_on_map, config.N_TRAINING_SITES),
        prior=_method_results(
            posterior=PosteriorMoments.independent_sites(
                prior.trend(region.training_locations),
                np.repeat(np.diag(prior.site_marginal_sd() ** 2)[None], config.N_TRAINING_SITES, axis=0),
            ),
            map_predictions=prior_predictions(prior, region.map_grid_locations, include_site_noise=False),
            held_out_predictions=prior_predictions(prior, region.held_out_locations, include_site_noise=True),
            true_theta_held_out=true_theta_held_out,
        ),
        # Site by site makes no map, and predicts a site with no data by its prior.
        site_by_site=_method_results(
            posterior=site_by_site.moments,
            map_predictions=Predictions(mean=no_map, sd=no_map),
            held_out_predictions=prior_predictions(prior, region.held_out_locations, include_site_noise=True),
            true_theta_held_out=true_theta_held_out,
        ),
        spatial_reference=_spatial_method_results(prior, region, reference.moments, true_theta_held_out),
        ipl=_spatial_method_results(prior, region, ipl.final, true_theta_held_out),
        site_grids=results.SiteGrids(grid_axes=site_by_site.grid_axes, density=site_by_site.density),
        ipl_history=_ipl_history(ipl, reference.moments, design.is_tower),
        reference_samples=reference.samples,
        reference_model_runs_per_site=reference.model_runs_per_site,
        reference_max_r_hat=reference.max_r_hat,
        reference_min_effective_sample_size=reference.min_effective_sample_size,
    )
    results.save(toy_results, config.TOY_RESULTS_FILE)
    _report(toy_results)
    print(f"wrote {config.TOY_RESULTS_FILE}")


# ── the steps' supporting functions ─────────────────────────────────────────


def _spatial_method_results(
    prior: SpatialPrior,
    region: Region,
    posterior: PosteriorMoments,
    true_theta_held_out: np.ndarray,
) -> results.MethodResults:
    """A spatial method's results: its maps and held-out predictions come from kriging."""
    return _method_results(
        posterior=posterior,
        map_predictions=predict_from_posterior(
            prior, region.training_locations, posterior, region.map_grid_locations, include_site_noise=False
        ),
        held_out_predictions=predict_from_posterior(
            prior, region.training_locations, posterior, region.held_out_locations, include_site_noise=True
        ),
        true_theta_held_out=true_theta_held_out,
    )


def _method_results(
    *,
    posterior: PosteriorMoments,
    map_predictions: Predictions,
    held_out_predictions: Predictions,
    true_theta_held_out: np.ndarray,
) -> results.MethodResults:
    scores = held_out_scores(held_out_predictions, true_theta_held_out)
    return results.MethodResults(
        posterior_mean=posterior.mean,
        posterior_site_covariance=posterior.site_covariances(),
        map_mean=map_predictions.mean,
        map_sd=map_predictions.sd,
        held_out_mean=held_out_predictions.mean,
        held_out_sd=held_out_predictions.sd,
        held_out_root_mean_square_error=scores.root_mean_square_error,
        held_out_interval_coverage=scores.interval_coverage,
    )


def _truth_results(truth_at_sites, truth_on_map, n_training_sites: int) -> results.TruthResults:
    training = slice(0, n_training_sites)
    return results.TruthResults(
        theta_training=truth_at_sites.value[training],
        theta_held_out=truth_at_sites.value[n_training_sites:],
        trend_training=truth_at_sites.trend[training],
        smooth_variation_training=truth_at_sites.smooth_variation[training],
        site_noise_training=truth_at_sites.site_noise[training],
        trend_map=truth_on_map.trend,
        smooth_variation_map=truth_on_map.smooth_variation,
    )


def _ipl_history(ipl: IPLResult, reference: PosteriorMoments, is_tower: np.ndarray) -> results.IPLHistory:
    errors = np.stack([standardized_error_to_reference(moments, reference) for moments in ipl.history])
    sd_ratios = np.stack([sd_ratio_to_reference(moments, reference) for moments in ipl.history])
    return results.IPLHistory(
        runs_per_site=ipl.runs_per_site,
        mean=np.stack([moments.mean for moments in ipl.history]),
        sd=np.stack([moments.sd for moments in ipl.history]),
        median_standardized_error_towers=np.median(errors[:, is_tower], axis=1),
        median_standardized_error_sparse=np.median(errors[:, ~is_tower], axis=1),
        median_sd_ratio_towers=np.median(sd_ratios[:, is_tower], axis=1),
        median_sd_ratio_sparse=np.median(sd_ratios[:, ~is_tower], axis=1),
    )


def _report(toy_results: results.ToyResults) -> None:
    """Print the scores the slides quote, so a run's numbers can be checked."""
    print("held-out sites: RMSE / 90% interval coverage, per parameter")
    for name in ("prior", "site_by_site", "spatial_reference", "ipl"):
        method = getattr(toy_results, name)
        print(
            f"  {name:18s} RMSE {np.round(method.held_out_root_mean_square_error, 3)}"
            f"  coverage {np.round(method.held_out_interval_coverage, 2)}"
        )
    history = toy_results.ipl_history
    print("IPL vs reference: median |mean - ref| / ref SD (towers, sparse), by iteration")
    for i, runs in enumerate(history.runs_per_site):
        print(
            f"  {runs:4d} runs/site  towers {np.round(history.median_standardized_error_towers[i], 3)}"
            f"  sparse {np.round(history.median_standardized_error_sparse[i], 3)}"
            f"  SD ratio sparse {np.round(history.median_sd_ratio_sparse[i], 3)}"
        )


class _timed:
    """Print how long a step took."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> None:
        self.start = time.perf_counter()
        print(f"{self.label} ...", flush=True)

    def __exit__(self, *exc) -> None:
        print(f"  {self.label}: {time.perf_counter() - self.start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
