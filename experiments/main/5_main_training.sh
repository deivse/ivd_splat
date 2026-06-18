#!/bin/bash
#SBATCH --job-name=gs_init_compare
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=amdgpulong
#SBATCH --array=0-9

export NUMEXPR_MAX_THREADS=10 # Keep in sync with --cpus-per-task!

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/ivd_splat"
source "$REPO_PATH/experiments/common_slurm_setup.sh"
source "$REPO_PATH/experiments/main/common_vars.sh"

INIT_FRACTIONS="0.5, 0.75"

POS_NOISE_SCALES="0.01, 0.1" # Scales for noise eval
NOISE_EVAL_INIT_FRACTIONS="0.5" # Init size fraction for noise eval tests

GAUSSIAN_CAP_FRACTIONS="0.75 1.25" # For experiment where we vary Gaussian cap fraction with SfM and laser scan init
GAUSSIAN_CAP_EVAL_INIT_FRACTIONS="0.5" # Init size fraction for Gaussian cap fraction eval tests

EXTRA_TAGS="--extra_tags gsplat_version=bfa5e98"

function prepend_space_if_set {
    local extra_config=$1

    if [ -z "$extra_config" ]; then
        echo ""
    else
        echo " $extra_config"
    fi
}

function get_cap_max_param {
    local cap_max_file=$1

    if [ "$strategy" != "DefaultWithoutADCStrategy" ]; then
        echo "--gaussian_cap_per_scene_file $cap_max_file"
    else
        echo ""
    fi
}


function train_strat_with_sfm {
    local extra_config=$(prepend_space_if_set "$1")

    ivd_splat_runner --datasets $ALL_DATASETS \
        --method ivd-splat \
        --init_method sfm \
        --output-dir $RESULTS_DIR \
        --configs "strategy={$strategy}$extra_config" \
        $(get_cap_max_param $FINAL_NUM_POINTS_PER_SCENE_FILE) \
        $EXTRA_TAGS
}

function train_strat_with_laser_scan {
    local init_size_per_scene_file=$1
    local extra_config=$(prepend_space_if_set "$2")

    ivd_splat_runner --datasets $GT_DATASETS \
        --method ivd-splat \
        --init_method laser_scan \
        --output-dir $RESULTS_DIR \
        --configs "strategy={$strategy}$extra_config" \
        $(get_cap_max_param $FINAL_NUM_POINTS_PER_SCENE_FILE) \
        --init_size_per_scene_file $init_size_per_scene_file \
        $EXTRA_TAGS
}

function train_strat_with_cap_fraction {
    local init_method=$1
    local fract=$2
    local extra_config=$(prepend_space_if_set "$3")

    ivd_splat_runner --datasets $GT_DATASETS \
        --method ivd-splat \
        --init_method $init_method \
        --output-dir $RESULTS_DIR \
        --configs "strategy={$strategy}$extra_config" \
        $(get_cap_max_param $FINAL_NUM_POINTS_PER_SCENE_FILE) \
        --init_size_per_scene_file $FINAL_NUM_POINTS_PER_SCENE_FILE \
        --gaussian_cap_fraction=${fract} \
        $EXTRA_TAGS
}

function train_strat_with_practical_init_method {
    local init_method=$1
    local datasets=$2
    # default to "default" if not set
    local init_method_config="${3:-default}"
    local extra_config=$(prepend_space_if_set "$4")

    ivd_splat_runner --datasets $datasets \
        --method ivd-splat \
        --output-dir $RESULTS_DIR \
        --configs "strategy={$strategy}$extra_config" \
        --init_method $init_method \
        --init_size_per_scene_file $REAL_INIT_NUM_POINTS_PER_SCENE_FILE \
        $(get_cap_max_param $FINAL_NUM_POINTS_PER_SCENE_FILE) \
        --init-method-config $init_method_config \
        $EXTRA_TAGS
}

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
    INIT_METHODS="monodepth edgs da3 laser_scan"

    for init_method in $INIT_METHODS; do
        train_strat_with_practical_init_method $init_method "$GT_DATASETS_EXCEPT_ETH3D"
        
        # EDGS with full SH init
        if [ "$init_method" == "edgs" ]; then
            train_strat_with_practical_init_method $init_method "$GT_DATASETS_EXCEPT_ETH3D" "full_sh_init=True"
        fi
        if [ "$init_method" == "da3" ]; then
            # DA3 with floater removal
            train_strat_with_practical_init_method $init_method "$GT_DATASETS_EXCEPT_ETH3D" "floater_removal=True"
            # DA3 with splat init
            train_strat_with_practical_init_method $init_method "$GT_DATASETS_EXCEPT_ETH3D" \
                                                   "output_gaussians=True_max_num_images=150" \
                                                   "splat_init.increase_scale_with_fewer_splats=False"
        fi
    done

    # Datasets without laser scan data, so just monodepth and EDGS.
    INIT_METHODS="monodepth edgs da3"

    for init_method in $INIT_METHODS; do
        train_strat_with_practical_init_method $init_method "$OTHER_DATASETS"
        
        # EDGS with full SH init
        if [ "$init_method" == "edgs" ]; then
            train_strat_with_practical_init_method $init_method "$OTHER_DATASETS" "full_sh_init=True"
        fi
        if [ "$init_method" == "da3" ]; then
            # DA3 with floater removal
            train_strat_with_practical_init_method $init_method "$OTHER_DATASETS" "floater_removal=True"
            # DA3 with splat init
            train_strat_with_practical_init_method $init_method "$OTHER_DATASETS" \
                                                   "output_gaussians=True_max_num_images=150" \
                                                   "splat_init.increase_scale_with_fewer_splats=False"
        fi
    done
done
