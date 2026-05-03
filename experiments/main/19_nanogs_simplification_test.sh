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


#######################################################################
INIT_FRACTIONS="0.75"
NANOGS_CONFIG="nanogs_simplify_iter={500}"
#######################################################################

# ==== LASER SCAN INIT ====

# AbsGS
ivd_splat_runner --datasets $GT_DATASETS \
    --method ivd-splat \
    --init_method laser_scan \
    --output-dir $RESULTS_DIR \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --configs "$NANOGS_CONFIG dense_init.target_points_fraction={$INIT_FRACTIONS} strategy.grow_grad2d={$ABSGRAD_GRAD_THRESH}" 

# MCMC
for dataset in $GT_DATASETS; do
    # If contains scannet++, use custom opacity reg for MCMC since default is too high and causes all points to be removed.
    if [[ $dataset == *"scannet++"* ]]; then
        opacity_reg_config="opacity_reg={$SCANNETPP_MCMC_CUSTOM_OPACITY_REG}"
    else
        opacity_reg_config=""
    fi

    ivd_splat_runner --datasets $dataset \
        --method ivd-splat \
        --init_method laser_scan \
        --output-dir $RESULTS_DIR \
        --configs "$NANOGS_CONFIG dense_init.target_points_fraction={$INIT_FRACTIONS} strategy={MCMCStrategy} $opacity_reg_config" \
        --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
        --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE
done

# ==== REAL INIT ====

# NOTE: did not run Laset with same size as Real Init for now...

INIT_METHODS="edgs monodepth"

for init_method in $INIT_METHODS; do
    # ADC
    ivd_splat_runner --datasets $ALL_DATASETS_EXCEPT_ETH3D \
        --method ivd-splat \
        --output-dir $RESULTS_DIR \
        --configs "$NANOGS_CONFIG strategy.grow_grad2d={$ABSGRAD_GRAD_THRESH}" \
        --init_method $init_method \
        --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
        --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE

    # MCMC
    for dataset in $ALL_DATASETS_EXCEPT_ETH3D; do
        # If contains scannet++, use custom opacity reg for MCMC
        if [[ $dataset == *"scannet++"* ]]; then
            opacity_reg_config="opacity_reg={$SCANNETPP_MCMC_CUSTOM_OPACITY_REG}"
        else
            opacity_reg_config=""
        fi

        ivd_splat_runner --datasets $dataset \
            --method ivd-splat \
            --output-dir $RESULTS_DIR \
            --configs "$NANOGS_CONFIG strategy={MCMCStrategy} $opacity_reg_config" \
            --init_method $init_method \
            --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
            --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE
    done
done

# ==== SfM INIT ====

# AbsGS
ivd_splat_runner --datasets $ALL_DATASETS \
    --method ivd-splat \
    --init_method sfm \
    --output-dir $RESULTS_DIR \
    --configs "$NANOGS_CONFIG strategy.grow_grad2d={$ABSGRAD_GRAD_THRESH}" \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE # Ensure max is the same as without NanoGS (it is not a given)

# MCMC
for dataset in $ALL_DATASETS; do
    # If contains scannet++, use custom opacity reg for MCMC
    if [[ $dataset == *"scannet++"* ]]; then
        opacity_reg_config="opacity_reg={$SCANNETPP_MCMC_CUSTOM_OPACITY_REG}"
    else
        opacity_reg_config=""
    fi

    ivd_splat_runner --datasets $dataset \
        --method ivd-splat \
        --init_method sfm \
        --output-dir $RESULTS_DIR \
        --configs "$NANOGS_CONFIG strategy={MCMCStrategy} $opacity_reg_config" \
        --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE
done
