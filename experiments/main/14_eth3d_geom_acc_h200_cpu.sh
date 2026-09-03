#!/bin/bash
#SBATCH --job-name=geom_acc_cpu
#SBATCH --output=eth3d_geom_acc_log_h200_cpu.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=amd

export NUMEXPR_MAX_THREADS=32 # Keep in sync with --cpus-per-task!
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

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
# (14_eth3d_geom_acc_h200_gpu.sh) with the external ETH3DMultiViewEvaluation
# tool, and render the LaTeX table. Must be submitted from the same directory as
# the GPU stage so the shared intermediate directory resolves.
TMP_DIR=/data/temporary/ivd_splat_geom_accuracy_eval/
mkdir -p $TMP_DIR

INTERMEDIATE_ETH3D=eth3d_geom_acc_intermediate_h200

ml GCC

python -u "$REPO_PATH/experiments/main/eval_final_recon_accuracy_cpu.py" \
    --intermediate-dir "$INTERMEDIATE_ETH3D" \
    --output final_recon_accuracy_eth3d_h200.json \
    --temp-dir-override $TMP_DIR \
    --max-cpu-threads 32
