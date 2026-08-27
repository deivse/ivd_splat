#!/bin/bash
#SBATCH --job-name=geom_acc
#SBATCH --output=geom_acc_log_h200.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=h200

export NUMEXPR_MAX_THREADS=64 # Keep in sync with --cpus-per-task!
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

# Slurm redirects stdout/stderr to a file, which makes Python block-buffer its
# output so log lines only appear at exit (or are lost if the job is killed).
# Force unbuffered output so logs stream into the slurm log in real time.
export PYTHONUNBUFFERED=1

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/ivd_splat"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/main/common_vars.sh"


#######################################################################
TMP_DIR=/data/temporary/ivd_splat_geom_accuracy_eval/

mkdir -p $TMP_DIR

python -u "$REPO_PATH/experiments/main/eval_final_recon_accuracy.py" \
    --results-dir "$RESULTS_DIR" \
    --init-methods sfm=default edgs=default monodepth=default da3=floater_removal=True da3=output_gaussians=True_max_num_images=150 laser_scan=default \
    --dataset eth3d \
    --ivd_splat_configs "strategy=RevDGSStrategy" "strategy=DefaultWithGaussianCapStrategy" "strategy=INRIAStrategy" "strategy=MCMCStrategy" "strategy=IDHFRStrategy" "strategy=DefaultWithoutADCStrategy" \
    --ivd_splat_configs_suffix laser_scan=default dense_init.include_sparse=True da3=output_gaussians=True_max_num_images=150 splat_init.increase_scale_with_fewer_splats=False \
    --extra_tags gsplat_version=bfa5e98 \
    --output final_recon_accuracy_h200.json \
    --gaussian_cap_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
    --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
    --temp-dir-override $TMP_DIR \
    --max-cpu-threads 64 \
    --far-plane=50 \
    --tsdf-backend gpu
