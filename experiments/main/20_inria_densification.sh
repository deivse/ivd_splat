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
source "$REPO_PATH/experiments/main/common_vars.sh"

POS_NOISE_SCALES="0.01, 0.1"
INIT_FRACTIONS="0.5, 0.75"

# SfM

ivd_splat_runner --datasets $ALL_DATASETS \
    --method ivd-splat \
    --init_method sfm \
    --output-dir $RESULTS_DIR \
    --configs "strategy={INRIAStrategy}" \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE

    
# Laser Scan

## Same as SfM init size
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --configs "strategy={INRIAStrategy}" \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $INITIAL_NUM_POINTS_PER_SCENE_FILE \

## Fractions of G_max
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "strategy={INRIAStrategy} dense_init.target_points_fraction={$INIT_FRACTIONS}" 

## Full
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "strategy={INRIAStrategy}" 

## Noise and half G_max
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "strategy={INRIAStrategy} dense_init.target_points_fraction={0.5} init.position_noise_std={$POS_NOISE_SCALES}" 

# Gaussian cap fractions with SfM and GT init at same size as SfM.
GAUSSIAN_CAP_FRACTIONS="0.75 1.25"
for fract in $GAUSSIAN_CAP_FRACTIONS; do
    ivd_splat_runner --datasets $GT_DATASETS \
        --method ivd-splat \
        --init_method sfm \
        --output-dir $RESULTS_DIR \
        --configs "strategy={INRIAStrategy}" \
        --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
        --gaussian_cap_fraction=${fract}

    
    # 3DGS MCMC with various init fractions.
    ivd_splat_runner --datasets $GT_DATASETS \
        --method ivd-splat \
        --init_method laser_scan \
        --output-dir $RESULTS_DIR \
        --configs "strategy={INRIAStrategy} dense_init.target_points_fraction={0.5}" \
        --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
        --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
        --gaussian_cap_fraction=${fract}
done

# Monodepth, EDGS, and Laser Scan at same init size as those.
INIT_METHODS="monodepth edgs laser_scan"

for init_method in $INIT_METHODS; do
	ivd_splat_runner --datasets $GT_DATASETS_EXCEPT_ETH3D \
        --method ivd-splat \
        --output-dir $RESULTS_DIR \
        --configs "strategy={INRIAStrategy}" \
        --init_method $init_method \
        --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
        --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE
    
    if [ "$init_method" == "edgs" ]; then
        ivd_splat_runner --datasets $OTHER_DATASETS \
            --method ivd-splat \
            --output-dir $RESULTS_DIR \
            --configs "strategy={INRIAStrategy}" \
            --init_method $init_method \
            --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
            --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
            --init-method-config "full_sh_init=True"
    fi
done

INIT_METHODS="monodepth edgs"

for init_method in $INIT_METHODS; do
	ivd_splat_runner --datasets $OTHER_DATASETS \
        --method ivd-splat \
        --output-dir $RESULTS_DIR \
        --configs "strategy={INRIAStrategy}" \
        --init_method $init_method \
        --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
        --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE

    if [ "$init_method" == "edgs" ]; then
        ivd_splat_runner --datasets $OTHER_DATASETS \
            --method ivd-splat \
            --output-dir $RESULTS_DIR \
            --configs "strategy={INRIAStrategy}" \
            --init_method $init_method \
            --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
            --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
            --init-method-config "full_sh_init=True"
    fi
done
