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

strategy=MCMCModStrategy
echo "Training with strategy: $strategy"

############### Practical Init Methods (+ Laser Scan at same size as those) ###############

# TODO: edgs if this works...
practical_methods="monodepth da3" 

# Datasets without laser scan data, so just monodepth and EDGS.
run_practical_init_methods_no_ablations "$OTHER_DATASETS" "$practical_methods"
run_practical_init_methods_no_ablations "$OTHER_DATASETS" "$practical_methods" "mcmc_mod.init_scale_mult=0.5"
run_practical_init_methods_no_ablations "$OTHER_DATASETS" "$practical_methods" "mcmc_mod.opacity_reg=0.2 mcmc_mod.scale_reg=0.2"

# First for datasets where we have laser scan data
# run_practical_init_methods_no_ablations "$GT_DATASETS_EXCEPT_ETH3D" "$practical_methods laser_scan"

# ----------------------
# ----------------------
# ----------------------

# ############### SfM ###############

# # DefaultWithGaussianCapStrategy already trained to get G_max in 0_slurm_run_sfm.sh.
# if [ "$strategy" != "DefaultWithGaussianCapStrategy" ]; then
#     train_strat_with_sfm
# fi
    
# ############### Laser Scan ###############

# if [ "$strategy" != "DefaultWithoutADCStrategy" ]; then
#     ## Same as SfM init size
#     train_strat_with_laser_scan $INITIAL_NUM_POINTS_PER_SCENE_FILE
# fi

# ## Init size at fractions of G_max
# train_strat_with_laser_scan $FINAL_NUM_POINTS_PER_SCENE_FILE "dense_init.target_points_fraction={$INIT_FRACTIONS}"

# ## Init size == G_max
# train_strat_with_laser_scan $FINAL_NUM_POINTS_PER_SCENE_FILE
