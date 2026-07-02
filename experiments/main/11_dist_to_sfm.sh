#!/bin/bash
#SBATCH --job-name=gs_init_compare
#SBATCH --output=dist_to_sfm_log.out
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=amdfast

export NUMEXPR_MAX_THREADS=10 # Keep in sync with --cpus-per-task!

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/ivd_splat"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/main/common_vars.sh"


#######################################################################

python eval_closeness_to_sfm.py \
    --results-dir $RESULTS_DIR \
    --datasets mipnerf360 tanksandtemples \
    --init-methods edgs=default monodepth=default da3=floater_removal=True da3=max_num_images=150_output_gaussians=True \
    --debug-export-dir ./debug_closeness_to_sfm

