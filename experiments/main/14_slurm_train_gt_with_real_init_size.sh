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

#######################################################################
INIT_METHOD=laser_scan

# No ADC
ivd_splat_runner --datasets $GT_DATASETS_EXCEPT_ETH3D \
    --method ivd-splat \
    --output-dir $RESULTS_DIR \
    --configs "strategy={DefaultWithoutADCStrategy}" \
    --init_method $INIT_METHOD \
    --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE

# ADC (with absgrad by default) 
ivd_splat_runner --datasets $GT_DATASETS_EXCEPT_ETH3D \
    --method ivd-splat \
    --output-dir $RESULTS_DIR \
    --configs "strategy.grow_grad2d={$ABSGRAD_GRAD_THRESH}" \
    --init_method $INIT_METHOD \
    --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE

# IDHFR
ivd_splat_runner --datasets $GT_DATASETS_EXCEPT_ETH3D \
    --method ivd-splat \
    --output-dir $RESULTS_DIR \
    --configs "strategy={IDHFRStrategy}" \
    --init_method $INIT_METHOD \
    --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE

# MCMC
for dataset in $GT_DATASETS_EXCEPT_ETH3D; do
    # If contains scannet++, use custom opacity reg for MCMC since default causes uncontrollable growth of gaussians
    if [[ $dataset == *"scannet++"* ]]; then
        opacity_reg_config="opacity_reg={$SCANNETPP_MCMC_CUSTOM_OPACITY_REG}"
    else
        opacity_reg_config=""
    fi

    ivd_splat_runner --datasets $dataset \
        --method ivd-splat \
        --output-dir $RESULTS_DIR \
        --configs "strategy={MCMCStrategy} $opacity_reg_config" \
        --init_method $INIT_METHOD \
        --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
        --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE
done
