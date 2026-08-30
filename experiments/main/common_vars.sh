export GT_DATASETS_EXCEPT_ETH3D="scannet++ eval_on_train_set_scannet++"
export GT_DATASETS="$GT_DATASETS_EXCEPT_ETH3D eth3d"

export OTHER_DATASETS="mipnerf360 tanksandtemples"
export SPARSIFIED_DATASETS="mipnerf360-sparsified tanksandtemples-sparsified"

export BASE_DATASETS="$GT_DATASETS $OTHER_DATASETS"
export ALL_DATASETS="$BASE_DATASETS $SPARSIFIED_DATASETS"
export BASE_DATASETS_EXCEPT_ETH3D="$GT_DATASETS_EXCEPT_ETH3D $OTHER_DATASETS"
export ALL_DATASETS_EXCEPT_ETH3D="$BASE_DATASETS_EXCEPT_ETH3D $SPARSIFIED_DATASETS"
export ALL_STRATEGIES="DefaultWithoutADCStrategy INRIAStrategy DefaultWithGaussianCapStrategy MCMCStrategy IDHFRStrategy RevDGSStrategy"

export EXPERIMENT_NAME="main"
export FINAL_NUM_POINTS_PER_SCENE_FILE="$RESULTS_DIR/num_points_per_scene.json"
export INITIAL_NUM_POINTS_PER_SCENE_FILE="$RESULTS_DIR/init_sfm_pts_per_scene.json"
export REAL_INIT_NUM_POINTS_PER_SCENE_FILE="$RESULTS_DIR/real_init_num_points_per_scene.json"

export MLFLOW_EXPERIMENT_NAME=${EXPERIMENT_NAME}

export INIT_FRACTIONS="0.5, 0.75"

export POS_NOISE_SCALES="0.01, 0.1" # Scales for noise eval
export NOISE_EVAL_INIT_FRACTIONS="0.5" # Init size fraction for noise eval tests

export GAUSSIAN_CAP_FRACTIONS="0.75 1.25" # For experiment where we vary Gaussian cap fraction with SfM and laser scan init
export GAUSSIAN_CAP_EVAL_INIT_FRACTIONS="0.5" # Init size fraction for Gaussian cap fraction eval tests

export EXTRA_TAGS="--extra_tags gsplat_version=bfa5e98"
