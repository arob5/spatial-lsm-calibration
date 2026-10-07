#!/bin/bash -l
#$ -P dietzelab
#$ -l buyin
#$ -l avx2
#$ -pe omp 28
#$ -l h_rt=12:00:00
#$ -j y
# MCMC for one model, started from an importance-sampling run (FROM); it
# resumes from its checkpoint if one exists, so resubmitting continues it.
# MODEL, FROM and REPO come from qsub -v.
source "$REPO/experiments/single_site_mcmc_vs_eki/scc/env.sh"
echo "model $MODEL on $(hostname) with $SIPNET_WORKERS workers"
step "mcmc" calibrate --model "$MODEL" --algorithm mcmc --from "$FROM" --resume
step "predict mcmc" predict --model "$MODEL" --run mcmc
