# Warning: These rely on global variables available in the training scripts, 
# such as RESULTS_DIR, FINAL_NUM_POINTS_PER_SCENE_FILE, REAL_INIT_NUM_POINTS_PER_SCENE_FILE, EXTRA_TAGS, etc.
# Importantly, they use $strategy to determine the densification strategy to train with.


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
        $EXTRA_TAGS $ITERATION_ARG
}

function train_strat_with_laser_scan {
    local init_size_per_scene_file=$1
    local extra_config=$(prepend_space_if_set "$2")
    local datasets=${3:-$GT_DATASETS}

    ivd_splat_runner --datasets $datasets \
        --method ivd-splat \
        --init_method laser_scan \
        --output-dir $RESULTS_DIR \
        --configs "strategy={$strategy}$extra_config" \
        $(get_cap_max_param $FINAL_NUM_POINTS_PER_SCENE_FILE) \
        --init_size_per_scene_file $init_size_per_scene_file \
        $EXTRA_TAGS $ITERATION_ARG
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
        $EXTRA_TAGS $ITERATION_ARG
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
        $EXTRA_TAGS $ITERATION_ARG
}

function run_practical_init_methods_with_ablations {
    local datasets=$1
    local init_methods=$2
    local extra_config=$(prepend_space_if_set "$3")

    for init_method in $init_methods; do
            train_strat_with_practical_init_method $init_method "$datasets" "default" "$extra_config"
            
            # EDGS with full SH init
            if [ "$init_method" == "edgs" ]; then
                train_strat_with_practical_init_method $init_method "$datasets" "full_sh_init=True" "$extra_config"
            fi
            if [ "$init_method" == "da3" ]; then
                # DA3 with floater removal
                train_strat_with_practical_init_method $init_method "$datasets" "floater_removal=True" "$extra_config"
                # DA3 with splat init
                train_strat_with_practical_init_method $init_method "$datasets" \
                                                    "output_gaussians=True_max_num_images=150" \
                                                    "splat_init.increase_scale_with_fewer_splats=False$extra_config"
            fi
    done
}

function run_practical_init_methods_no_ablations_no_da3_splat {
    local datasets=$1
    local init_methods=$2
    local extra_config=$(prepend_space_if_set "$3")

    for init_method in $init_methods; do
        if [ "$init_method" == "da3" ]; then
            # DA3 with floater removal
            train_strat_with_practical_init_method $init_method "$datasets" "floater_removal=True" "$extra_config"
        else
            # No DA3 without floater removal, that's our default config for eval iters > 0
            train_strat_with_practical_init_method $init_method "$datasets" "default" "$extra_config"
        fi
    done
}


function run_practical_init_methods_no_ablations {
    local datasets=$1
    local init_methods=$2
    local extra_config=$(prepend_space_if_set "$3")
    local extra_config_no_space=$3

    run_practical_init_methods_no_ablations_no_da3_splat "$datasets" "$init_methods" "$extra_config_no_space"

    # DA3 with splat init
    train_strat_with_practical_init_method da3 "$datasets" \
                                           "output_gaussians=True_max_num_images=150" \
                                           "splat_init.increase_scale_with_fewer_splats=False$extra_config"
}
