#!/bin/bash
#SBATCH --job-name=eth3d_geom_acc_non_h200_cpu
#SBATCH --output=eth3d_geom_acc_log_cpu.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=amd

export NUMEXPR_MAX_THREADS=64 # Keep in sync with --cpus-per-task!
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

# Slurm redirects stdout/stderr to a file, which makes Python block-buffer its
# output so log lines only appear at exit (or are lost if the job is killed).
# Force unbuffered output so logs stream into the slurm log in real time.
export PYTHONUNBUFFERED=1

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/ivd_splat"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/main/common_vars.sh"


#######################################################################
# CPU stage: score the ETH3D reconstructions written by the GPU stage
# (15_eth3d_geom_acc_gpu.sh) with the external ETH3DMultiViewEvaluation tool, and
# render the LaTeX table. Must be submitted from the same directory as the GPU
# stage so the shared intermediate directory resolves.
TMP_DIR=/data/temporary/ivd_splat_geom_accuracy_eval/
mkdir -p $TMP_DIR

INTERMEDIATE_ETH3D=eth3d_geom_acc_intermediate

ml GCC

python -u "$REPO_PATH/experiments/main/eval_final_recon_accuracy_cpu.py" \
    --intermediate-dir "$INTERMEDIATE_ETH3D" \
    --output final_recon_accuracy_eth3d.json \
    --temp-dir-override $TMP_DIR \
    --max-cpu-threads 64
