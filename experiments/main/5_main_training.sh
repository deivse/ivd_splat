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

for strategy in $ALL_STRATEGIES; do
    echo "Training with strategy: $strategy"

    ############### SfM ###############

    # DefaultWithGaussianCapStrategy already trained to get G_max in 0_slurm_run_sfm.sh.
    if [ "$strategy" != "DefaultWithGaussianCapStrategy" ]; then
        train_strat_with_sfm
    fi
        
    ############### Laser Scan ###############

    if [ "$strategy" != "DefaultWithoutADCStrategy" ]; then
        ## Same as SfM init size
        train_strat_with_laser_scan $INITIAL_NUM_POINTS_PER_SCENE_FILE
    fi

    ## Init size at fractions of G_max
    train_strat_with_laser_scan $FINAL_NUM_POINTS_PER_SCENE_FILE "dense_init.target_points_fraction={$INIT_FRACTIONS}"

    ## Init size == G_max
    train_strat_with_laser_scan $FINAL_NUM_POINTS_PER_SCENE_FILE

    ## Tmp test adpative subsample
    if [ "$strategy" == "MCMCStrategy" || "$strategy" == "DefaultWithGaussianCapStrategy" ]; then
        train_strat_with_laser_scan $FINAL_NUM_POINTS_PER_SCENE_FILE "dense_init.target_points_fraction={$INIT_FRACTIONS} dense_init.sampling={adaptive}"
        train_strat_with_laser_scan $FINAL_NUM_POINTS_PER_SCENE_FILE "dense_init.sampling={adaptive}"
    fi

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
    run_practical_init_methods_with_ablations "$GT_DATASETS_EXCEPT_ETH3D" "monodepth edgs da3 laser_scan"
    # Datasets without laser scan data, so just monodepth and EDGS.
    run_practical_init_methods_no_ablations "$OTHER_DATASETS" "monodepth edgs da3"
done
