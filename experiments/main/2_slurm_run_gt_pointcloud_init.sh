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
POS_NOISE_SCALES="0.01, 0.1"
INIT_FRACTIONS="0.5, 0.75"
# SAMPLING_TYPES="uniform, adaptive" # Just uniform for now
SAMPLING_TYPES="uniform"
#######################################################################


# No ADC
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "strategy={DefaultWithoutADCStrategy} dense_init.sampling={$SAMPLING_TYPES}"

# ADC (with absgrad by default) 
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "dense_init.sampling={$SAMPLING_TYPES} strategy.grow_grad2d={$ABSGRAD_GRAD_THRESH}" 

# IDHFR
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "dense_init.sampling={$SAMPLING_TYPES} strategy={IDHFRStrategy}"

# With various init sizes and ADC (with absgrad by default)
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "dense_init.sampling={$SAMPLING_TYPES} dense_init.target_points_fraction={$INIT_FRACTIONS} strategy.grow_grad2d={$ABSGRAD_GRAD_THRESH}" 

# With various init sizes and IDHFR
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "dense_init.sampling={$SAMPLING_TYPES} dense_init.target_points_fraction={$INIT_FRACTIONS} strategy={IDHFRStrategy}" 

#######################################################################
## With position noise

# No ADC with noise
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "strategy={DefaultWithoutADCStrategy} dense_init.sampling={$SAMPLING_TYPES} init.position_noise_std={$POS_NOISE_SCALES}"

# ADC (with absgrad by default) with noise at 0.5 init size
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "dense_init.target_points_fraction={0.5} init.position_noise_std={$POS_NOISE_SCALES} strategy.grow_grad2d={$ABSGRAD_GRAD_THRESH}" 

# IDHFR with noise at 0.5 init size
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "dense_init.target_points_fraction={0.5} init.position_noise_std={$POS_NOISE_SCALES} strategy={IDHFRStrategy}" 

# # ADC (with absgrad by default) with noise
# ivd_splat_runner --datasets $GT_DATASETS \
#     --method ivd-splat \
#     --init_method laser_scan \
#     --output-dir $RESULTS_DIR \
#     --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
#     --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
#     --configs "dense_init.sampling={$SAMPLING_TYPES} init.position_noise_std={$POS_NOISE_SCALES} strategy.grow_grad2d={$ABSGRAD_GRAD_THRESH}" 

# # With various init sizes and ADC (with absgrad by default) with noise
# ivd_splat_runner --datasets $GT_DATASETS \
#     --method ivd-splat \
#     --init_method laser_scan \
#     --output-dir $RESULTS_DIR \
#     --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
#     --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
#     --configs "dense_init.sampling={$SAMPLING_TYPES} init.position_noise_std={$POS_NOISE_SCALES} dense_init.target_points_fraction={$INIT_FRACTIONS} strategy.grow_grad2d={$ABSGRAD_GRAD_THRESH}" 
