# Sourced by every job: everything a run writes stays under $SCC_ROOT.
# REPO (the worktree) is set by submit_all.sh through qsub -v.
export SCC_ROOT=/projectnb/dietzelab/arober
export SIPNET_CALIBRATION_DATA=$SCC_ROOT/hf_data
export UV_CACHE_DIR=$SCC_ROOT/uv_cache
export PYSIPNET_CACHE_DIR=$SCC_ROOT/pysipnet_cache
export TMPDIR=$SCC_ROOT/tmp
export XDG_CACHE_HOME=$SCC_ROOT/tmp/xdg_cache
export MPLCONFIGDIR=$SCC_ROOT/tmp/matplotlib
export JAX_COMPILATION_CACHE_DIR=$SCC_ROOT/jax_cache
# One thread per process: the SIPNET workers are the parallelism.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SIPNET_WORKERS=${NSLOTS:-4}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$REPO"
PYTHON="$REPO/.venv/bin/python"
EXPERIMENT=experiments.single_site_mcmc_vs_eki

# step <label> <module> <arguments...>: run one step, log its time, and keep
# going if it fails, so that one failed run does not cost the rest.
step() {
    local label=$1
    shift
    local start
    start=$(date +%s)
    echo "=== $(date '+%F %T') $label"
    if "$PYTHON" -m "$EXPERIMENT.run.$@"; then
        echo "=== $label done in $(($(date +%s) - start)) s"
    else
        echo "=== $label FAILED after $(($(date +%s) - start)) s"
    fi
}
