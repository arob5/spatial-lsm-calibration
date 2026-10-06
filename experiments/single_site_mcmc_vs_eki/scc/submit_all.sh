#!/bin/bash
# Submit every model's pipeline, and MCMC for the models in $MCMC_MODELS:
#
#   bash experiments/single_site_mcmc_vs_eki/scc/submit_all.sh
#   MODELS="long_memory/fixed" MCMC_MODELS="" bash .../submit_all.sh   # a subset
#
# Per model, three jobs: "seed" (EKI and importance sampling), then "rest"
# (SMC and the predictives) and "mcmc" (for the MCMC models), both held until
# "seed" has finished.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SCC=$REPO/experiments/single_site_mcmc_vs_eki/scc
LOGS=/projectnb/dietzelab/arober/hf_runs/logs
mkdir -p "$LOGS"
MODELS=${MODELS:-"short_memory/fixed short_memory/inferred long_memory/fixed long_memory/inferred recurring_bias/fixed recurring_bias/inferred"}
MCMC_MODELS=${MCMC_MODELS-"long_memory/fixed long_memory/inferred"}
submit() {  # submit <name> <script> <variables> [qsub options...]
    local name=$1 script=$2 variables=$3
    shift 3
    qsub -terse -N "$name" -o "$LOGS" -wd "$REPO" "$@" -v "REPO=$REPO,$variables" "$SCC/$script"
}
declare -A seed_job
for model in $MODELS; do
    tag=${model/\//_}
    seed_job[$model]=$(submit "hf_seed_$tag" model_job.sh "MODEL=$model,PART=seed")
    rest=$(submit "hf_rest_$tag" model_job.sh "MODEL=$model,PART=rest" -hold_jid "${seed_job[$model]}")
    echo "$model: seed job ${seed_job[$model]}, rest job $rest"
done
for model in $MCMC_MODELS; do
    if [[ $model == */fixed ]]; then from=eki_is; else from=eki_gibbs_common_is; fi
    hold=()
    if [[ -n ${seed_job[$model]:-} ]]; then hold=(-hold_jid "${seed_job[$model]}"); fi
    job=$(submit "hf_mcmc_${model/\//_}" mcmc_job.sh "MODEL=$model,FROM=$from" -l h_rt=20:00:00 "${hold[@]}")
    echo "$model: mcmc job $job"
done
echo "logs in $LOGS; watch with: qstat -u $USER"
