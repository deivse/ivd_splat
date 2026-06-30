#!/bin/bash
#SBATCH --job-name=extra_practical
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=amdgpulong
#SBATCH --array=0-7

export NUMEXPR_MAX_THREADS=10 # Keep in sync with --cpus-per-task!

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/ivd_splat"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/main/common_vars.sh"
source "$REPO_PATH/experiments/main/main_training_funcs.sh"


EXTRA_PRACTICAL_EVAL_STRATEGIES="DefaultWithGaussianCapStrategy MCMCStrategy IDHFRStrategy"
EXTRA_PRACTICAL_INIT_FRACTIONS="0.5"

function da3_splat_init() {
    local extra_config=$(prepend_space_if_set "$1")

    train_strat_with_practical_init_method da3 "$ALL_DATASETS_EXCEPT_ETH3D" \
                                           "output_gaussians=True_max_num_images=150" \
                                           "splat_init.increase_scale_with_fewer_splats=False$extra_config"
}

for strategy in $EXTRA_PRACTICAL_EVAL_STRATEGIES; do
    ## Include sparse
    # First for datasets where we have laser scan data
    run_practical_init_methods_no_ablations_no_da3_splat "$GT_DATASETS_EXCEPT_ETH3D" "monodepth da3 laser_scan" "dense_init.include_sparse={True}"
    # Datasets without laser scan data, so just monodepth and EDGS.
    run_practical_init_methods_no_ablations_no_da3_splat "$OTHER_DATASETS" "monodepth da3" "dense_init.include_sparse={True}"
done

for strategy in $EXTRA_PRACTICAL_EVAL_STRATEGIES; do
    echo "Training with strategy: $strategy"
    
    da3_splat_init "splat_init.init_scale_with_knn=True"
    da3_splat_init "splat_init.init_scale_isotropic_mean=True"
    da3_splat_init "splat_init.opacity_uniform_override=0.1"
    da3_splat_init "splat_init.rotation_noise_angle_std_deg=45"
    da3_splat_init "splat_init.color_noise_std=0.5"
done

for strategy in $EXTRA_PRACTICAL_EVAL_STRATEGIES; do
    echo "Training with strategy: $strategy"
    run_practical_init_methods_no_ablations "$GT_DATASETS_EXCEPT_ETH3D" "monodepth da3" "dense_init.target_points_fraction={$EXTRA_PRACTICAL_INIT_FRACTIONS}"
    run_practical_init_methods_no_ablations "$GT_DATASETS_EXCEPT_ETH3D" "laser_scan" "dense_init.include_sparse={True} dense_init.target_points_fraction={$EXTRA_PRACTICAL_INIT_FRACTIONS}"
    run_practical_init_methods_no_ablations "$OTHER_DATASETS" "monodepth da3" "dense_init.target_points_fraction={$EXTRA_PRACTICAL_INIT_FRACTIONS}"
done
