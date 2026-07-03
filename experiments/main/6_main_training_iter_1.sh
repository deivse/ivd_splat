#!/bin/bash
#SBATCH --job-name=gs_init_compare
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

ITERATION_ARG="--eval_iter 1"

for strategy in $ALL_STRATEGIES; do
    echo "Training with strategy: $strategy"

    ############### SfM ###############

    # Need to re-run training with SfM with new random seed for eval_iter=1
    train_strat_with_sfm
        
    ############### Laser Scan ###############

    if [ "$strategy" != "DefaultWithoutADCStrategy" ]; then
        ## Same as SfM init size
        train_strat_with_laser_scan $INITIAL_NUM_POINTS_PER_SCENE_FILE
    fi

    ## Init size at fractions of G_max
    train_strat_with_laser_scan $FINAL_NUM_POINTS_PER_SCENE_FILE "dense_init.target_points_fraction={$INIT_FRACTIONS}"

    ## Init size == G_max
    train_strat_with_laser_scan $FINAL_NUM_POINTS_PER_SCENE_FILE

    ## With include sparse (no eth3d, as it doesn't help there)
    train_strat_with_laser_scan $FINAL_NUM_POINTS_PER_SCENE_FILE \
        "dense_init.include_sparse={True}" \
        "$GT_DATASETS_EXCEPT_ETH3D"
    train_strat_with_laser_scan $FINAL_NUM_POINTS_PER_SCENE_FILE \
        "dense_init.include_sparse={True} dense_init.target_points_fraction={$INIT_FRACTIONS}" \
        "$GT_DATASETS_EXCEPT_ETH3D"

    ## With include sparse for practical init eval (mostly overlaps with above)
    train_strat_with_laser_scan $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
        "dense_init.include_sparse={True}" \
        "$GT_DATASETS_EXCEPT_ETH3D"

    if [ "$strategy" != "DefaultWithoutADCStrategy" ]; then
        ############### Laser Scan + Noise ###############
        train_strat_with_laser_scan \
            $FINAL_NUM_POINTS_PER_SCENE_FILE \
            "dense_init.target_points_fraction={$NOISE_EVAL_INIT_FRACTIONS} init.position_noise_std={$POS_NOISE_SCALES}"

        ############### Varying cap fractions ###############
        for fract in $GAUSSIAN_CAP_FRACTIONS; do
            train_strat_with_cap_fraction sfm $fract
            train_strat_with_cap_fraction laser_scan $fract "dense_init.target_points_fraction={$GAUSSIAN_CAP_EVAL_INIT_FRACTIONS}"
        done
    fi

    ############### Practical Init Methods (+ Laser Scan at same size as those) ###############

    # First for datasets where we have laser scan data
    run_practical_init_methods_no_ablations "$GT_DATASETS_EXCEPT_ETH3D" "monodepth edgs da3 laser_scan"
    # Datasets without laser scan data, so just monodepth, EDGS and DA3.
    run_practical_init_methods_no_ablations "$OTHER_DATASETS" "monodepth edgs da3"

    # TODO: With sparse?
done
