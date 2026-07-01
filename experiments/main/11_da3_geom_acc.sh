#!/bin/bash
#SBATCH --job-name=gs_init_compare
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=amdgpufast

export NUMEXPR_MAX_THREADS=10 # Keep in sync with --cpus-per-task!

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/ivd_splat"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/main/common_vars.sh"


#######################################################################

python eval_da3_geometry_accuracy.py \
    --results-dir $RESULTS_DIR \
    --datasets mipnerf360 tanksandtemples \
    --config-a floater_removal=True \
    --config-b output_gaussians=True_max_num_images=150 \
    --output ./da3_geom_acc_results.json

