#!/bin/bash
#SBATCH --job-name=gs_init_compare
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=amdgpufast
#SBATCH --array=0-9

export NUMEXPR_MAX_THREADS=10 # Keep in sync with --cpus-per-task!

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/ivd_splat"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/main/common_vars.sh"

#######################################################################
    
# No eth3d for now due to lack of non-uniform image sizes support

init_runner --datasets $ALL_DATASETS_EXCEPT_ETH3D \
    --method da3 \
    --output-dir $RESULTS_DIR \
    --configs "max_num_images={300}"

init_runner --datasets $ALL_DATASETS_EXCEPT_ETH3D \
    --method edgs \
    --output-dir $RESULTS_DIR

init_runner --datasets $ALL_DATASETS_EXCEPT_ETH3D \
    --method monodepth \
    --output-dir $RESULTS_DIR \
    --extra-args="--ignore-depth-cache=True" # So runtimes are real.

