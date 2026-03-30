#!/bin/bash
#SBATCH --job-name=gs_init_compare
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=amdgpufast

export NUMEXPR_MAX_THREADS=10 # Keep in sync with --cpus-per-task!

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/monocular_depth_tsdf_fusion"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/gt_pointclouds/common_vars.sh"


#######################################################################

python real_init_get_min_num_pts_per_scene.py \
    --results-dir $RESULTS_DIR \
    --datasets $ALL_DATASETS \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --gaussian_cap_fraction 1.0 \
    --output $REAL_INIT_NUM_POINTS_PER_SCENE_FILE

