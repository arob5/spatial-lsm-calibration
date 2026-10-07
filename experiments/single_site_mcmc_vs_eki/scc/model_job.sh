#!/bin/bash -l
#$ -P dietzelab
#$ -l buyin
#$ -l avx2
#$ -pe omp 28
#$ -l h_rt=12:00:00
#$ -j y
# One model's pipeline, in two parts so that MCMC can start after the first:
#   PART=seed: its EKI run(s) and importance sampling from each;
#   PART=rest: SMC from the first EKI run, and the posterior predictive of
#              every run but MCMC.
# MODEL, PART and REPO come from qsub -v.
source "$REPO/experiments/single_site_mcmc_vs_eki/scc/env.sh"
echo "model $MODEL, part $PART, on $(hostname) with $SIPNET_WORKERS workers"
if [[ $MODEL == */fixed ]]; then
    seeds=(eki)
else
    seeds=(eki_gibbs_common eki_gibbs_per_particle)
fi
if [[ $PART == seed ]]; then
    for seed in "${seeds[@]}"; do
        step "$seed" calibrate --model "$MODEL" --algorithm "$seed"
        step "${seed}_is" calibrate --model "$MODEL" --algorithm is --from "$seed"
    done
else
    step "${seeds[0]}_smc" calibrate --model "$MODEL" --algorithm smc --from "${seeds[0]}"
    for run in "${seeds[@]}" "${seeds[@]/%/_is}" "${seeds[0]}_smc"; do
        step "predict $run" predict --model "$MODEL" --run "$run"
    done
fi
