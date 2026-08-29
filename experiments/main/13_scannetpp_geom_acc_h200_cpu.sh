#!/bin/bash
#SBATCH --job-name=geom_acc_cpu
#SBATCH --output=geom_acc_log_h200_cpu.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1G
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
# CPU stage: score the reconstructions written by the GPU stage
# (13_scannetpp_geom_acc_h200_gpu.sh) with the KD-tree F-score, and render the
# LaTeX table. Reads the shared intermediate directories produced there; must be
# submitted from the same directory as the GPU stage.
TMP_DIR=/data/temporary/ivd_splat_geom_accuracy_eval/
mkdir -p $TMP_DIR

INTERMEDIATE_MAIN=geom_acc_intermediate_h200/main
INTERMEDIATE_LASER_SCAN_NO_SPARSE=geom_acc_intermediate_h200/laser_scan_no_sparse

python -u "$REPO_PATH/experiments/main/eval_final_recon_accuracy_cpu.py" \
    --intermediate-dir "$INTERMEDIATE_MAIN" \
    --output final_recon_accuracy_h200.json \
    --temp-dir-override $TMP_DIR \
    --max-cpu-threads 64

python -u "$REPO_PATH/experiments/main/eval_final_recon_accuracy_cpu.py" \
    --intermediate-dir "$INTERMEDIATE_LASER_SCAN_NO_SPARSE" \
    --output final_recon_accuracy_h200_laser_scan_no_sparse.json \
    --temp-dir-override $TMP_DIR \
    --max-cpu-threads 64
