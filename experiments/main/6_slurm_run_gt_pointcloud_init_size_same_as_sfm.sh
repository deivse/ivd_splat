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
REPO_PATH="$HOME/ivd_splat"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/gt_pointclouds/common_vars.sh"

# ADC
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --configs "strategy={DefaultWithGaussianCapStrategy} strategy.grow_grad2d={$ABSGRAD_GRAD_THRESH}" \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $INITIAL_NUM_POINTS_PER_SCENE_FILE \

# IDHFR
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --configs "strategy={IDHFRStrategy}" \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $INITIAL_NUM_POINTS_PER_SCENE_FILE \

for dataset in $GT_DATASETS; do
    # If contains scannet++, use custom opacity reg for MCMC since default is too high and causes all points to be removed.
    if [[ $dataset == *"scannet++"* ]]; then
        opacity_reg_config="opacity_reg={$SCANNETPP_MCMC_CUSTOM_OPACITY_REG}"
    else
        opacity_reg_config=""
    fi

    # 3DGS MCMC with various init fractions.
    ivd_splat_runner --datasets $dataset \
        --method ivd-splat \
        --init_method laser_scan \
        --output-dir $RESULTS_DIR \
        --configs "strategy={MCMCStrategy} $opacity_reg_config" \
        --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
        --init_size_per_scene_file $INITIAL_NUM_POINTS_PER_SCENE_FILE \
done
