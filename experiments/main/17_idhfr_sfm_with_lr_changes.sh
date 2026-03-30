#!/bin/bash
#SBATCH --job-name=gs_init_compare
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=amdgpu
#SBATCH --array=0-9

export NUMEXPR_MAX_THREADS=10 # Keep in sync with --cpus-per-task!

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/monocular_depth_tsdf_fusion"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/gt_pointclouds/common_vars.sh"

# Regarding hyperparameters, we adjusted position lr init to 0.00004 and position lr final to 0.000002.
# Original:                                                  0.00016                          0.0000016             
ivd_splat_runner --datasets $ALL_DATASETS \
    --method ivd-splat \
    --init_methods sfm \
    --output-dir $RESULTS_DIR \
    --configs "strategy={IDHFRStrategy} means_lr_init={0.00004} means_lr_final={0.000002}" \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE

