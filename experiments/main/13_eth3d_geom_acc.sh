#!/bin/bash
#SBATCH --job-name=eth3d_geom_acc
#SBATCH --output=eth3d_geom_acc_log.out
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=amdgpufast

export NUMEXPR_MAX_THREADS=10 # Keep in sync with --cpus-per-task!
# Slurm redirects stdout/stderr to a file, which makes Python block-buffer its
# output so log lines only appear at exit (or are lost if the job is killed).
# Force unbuffered output so logs stream into the slurm log in real time.
export PYTHONUNBUFFERED=1

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/ivd_splat"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/main/common_vars.sh"


#######################################################################


python -u experiments/main/eval_final_recon_accuracy.py \
    --init-methods sfm=default edgs=default monodepth=default da3=floater_removal=True da3=output_gaussians=True_max_num_images=150 laser_scan=default \
    --datasets eth3d \
    --ivd_splat_configs "strategy=DefaultWithGaussianCapStrategy" "strategy=INRIAStrategy" "strategy=MCMCStrategy" "strategy=IDHFRStrategy" "strategy=DefaultWithoutADCStrategy" \
    --ivd_splat_configs_suffix laser_scan=default dense_init.include_sparse=True da3=output_gaussians=True_max_num_images=150 splat_init.increase_scale_with_fewer_splats=False \
    --extra_tags gsplat_version=bfa5e98 \
    --gaussian_cap_per_scene_file $GAUSSIAN_CAP_PER_SCENE_FILE  \
    --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
    --debug-export-dir ./debug_geom_acc_eth3d \
    --output final_recon_accuracy_eth3d.json

