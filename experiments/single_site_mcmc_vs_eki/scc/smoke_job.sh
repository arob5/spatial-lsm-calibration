#!/bin/bash -l
#$ -P dietzelab
#$ -l buyin
#$ -pe omp 8
#$ -l h_rt=01:00:00
#$ -j y
# Every algorithm end to end with tiny sizes, written to a scratch directory,
# to check the SCC setup before the real runs (about 15 minutes):
#
#   qsub -N hf_smoke -o /projectnb/dietzelab/arober/hf_runs/logs -v REPO=$PWD \
#       experiments/single_site_mcmc_vs_eki/scc/smoke_job.sh
source "$REPO/experiments/single_site_mcmc_vs_eki/scc/env.sh"
export HF_RUNS_DIRECTORY=$SCC_ROOT/hf_runs/smoke
rm -rf "$HF_RUNS_DIRECTORY"
echo "smoke test on $(hostname) with $SIPNET_WORKERS workers, writing $HF_RUNS_DIRECTORY"
step "eki_gibbs_per_particle" calibrate --model long_memory/inferred --algorithm eki_gibbs_per_particle --ensemble-size 8
step "is" calibrate --model long_memory/inferred --algorithm is --from eki_gibbs_per_particle --samples 16
step "mcmc" calibrate --model long_memory/inferred --algorithm mcmc --from eki_gibbs_per_particle_is --chains 8 --steps 4
step "eki" calibrate --model short_memory/fixed --algorithm eki --ensemble-size 8
step "smc" calibrate --model short_memory/fixed --algorithm smc --from eki --samples 16
step "predict" predict --model short_memory/fixed --run eki_smc --samples 8
find "$HF_RUNS_DIRECTORY" -name cost.json | sort
