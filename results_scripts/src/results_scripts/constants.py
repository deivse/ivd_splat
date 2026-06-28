from matplotlib import pyplot as plt

TRACKING_URI = "http://localhost:6069"

SCANNETPP_SCENE_SELECTION = (
    "c5439f4607,bcd2436daf,b0a08200c9,6115eddb86,f3d64c30f8,"
    "3f15a9266d,5eb31827b7,3db0a1c8f3,40aec5fffa,9071e139d9,"
    "e7af285f7d,bde1e479ad,5748ce6f01,825d228aec,7831862f02"
).split(",")
TANKSANDTEMPLES_SCENE_SELECTION = (
    "auditorium,ballroom,palace,temple,family,horse,lighthouse,m60,train,"
    "barn,caterpillar,church,meetingroom,truck"
).split(",")
MIPNERF360_SCENE_SELECTION = (
    "garden,bicycle,flowers,treehill,stump,kitchen,bonsai,counter,room"
).split(",")

DATASET_SCENE_SELECTION = {
    "eth3d": {
        "eth3d/pipes",
        "eth3d/kicker",
        "eth3d/terrace",
        "eth3d/relief",
        "eth3d/relief_2",
        "eth3d/terrains",
        "eth3d/office",
    },
    "scannet++": {f"scannet++/{scene}" for scene in SCANNETPP_SCENE_SELECTION},
    "eval_on_train_set_scannet++": {
        f"eval_on_train_set_scannet++/{scene}" for scene in SCANNETPP_SCENE_SELECTION
    },
    "mipnerf360": {f"mipnerf360/{scene}" for scene in MIPNERF360_SCENE_SELECTION},
    "tanksandtemples": {
        f"tanksandtemples/{scene}" for scene in TANKSANDTEMPLES_SCENE_SELECTION
    },
}

DEFAULT_MEANS_LR_INIT = 0.00016

STRATEGY_NAMES = {
    "DefaultWithGaussianCapStrategy": "AbsGS",
    "INRIAStrategy": "INRIA",
    "MCMCStrategy": "MCMC",
    "MCMCModStrategy": "MCMC Mod",
    "BiasedMCMCStrategy": "Biased MCMC",
    "IDHFRStrategy": "IDHFR",
    "RevDGSStrategy": "RevDGS",
    "DefaultWithoutADCStrategy": "No D.",
}

STRATEGIES_EXCLUDED_FROM_DEFAULT = ["BiasedMCMCStrategy", "MCMCModStrategy"]
ALL_STRATEGIES = [
    k for k in STRATEGY_NAMES.keys() if k not in STRATEGIES_EXCLUDED_FROM_DEFAULT
]
ALL_STRATEGIES_EXCEPT_NO_D = [
    strategy for strategy in ALL_STRATEGIES if strategy != "DefaultWithoutADCStrategy"
]

COMMON_DEFAULT_ARGS = {
    "nanogs_simplify_iter": "-1",
    "means_lr_init": str(DEFAULT_MEANS_LR_INIT),
    "dense_init.sampling": "uniform",
}

DEFAULT_STRATEGY_ARGS = {
    "DefaultWithGaussianCapStrategy": {
        "strategy": "DefaultWithGaussianCapStrategy",
        "strategy.grow_grad2d": "0.0004",
        **COMMON_DEFAULT_ARGS,
    },
    "DefaultWithoutADCStrategy": {
        "strategy": "DefaultWithoutADCStrategy",
        **COMMON_DEFAULT_ARGS,
    },
    "MCMCStrategy": {
        "strategy": "MCMCStrategy",
        "opacity_reg": "0.01",
        "init.scale_mult": "0.1",
        **COMMON_DEFAULT_ARGS,
    },
    "BiasedMCMCStrategy": {
        "strategy": "BiasedMCMCStrategy",
        **COMMON_DEFAULT_ARGS,
    },
    "MCMCModStrategy": {
        "strategy": "MCMCModStrategy",
        **COMMON_DEFAULT_ARGS,
    },
    "IDHFRStrategy": {
        "strategy": "IDHFRStrategy",
        **COMMON_DEFAULT_ARGS,
    },
    "INRIAStrategy": {
        "strategy": "INRIAStrategy",
        **COMMON_DEFAULT_ARGS,
    },
    "RevDGSStrategy": {
        "strategy": "RevDGSStrategy",
        **COMMON_DEFAULT_ARGS,
    },
}

GT_DATASETS = ["scannet++", "eval_on_train_set_scannet++", "eth3d"]
GT_DATASETS_WITHOUT_ETH3D = [dataset for dataset in GT_DATASETS if dataset != "eth3d"]
OTHER_DATASETS = ["mipnerf360", "tanksandtemples"]
ALL_DATASETS = GT_DATASETS + OTHER_DATASETS
ALL_DATASETS_WITHOUT_ETH3D = [dataset for dataset in ALL_DATASETS if dataset != "eth3d"]

DATASET_NAMES = {
    "scannet++": "ScanNet++",
    "eval_on_train_set_scannet++": "ScanNet++ (On-Trajectory)",
    "eth3d": "ETH3D",
    "mipnerf360": "Mip-NeRF 360",
    "tanksandtemples": "Tanks and Temples",
}

DEFAULT_TABLE_METRICS = [
    f"eval-all-test/{metric}" for metric in ["psnr", "ssim", "lpips"]
] + [
    "train/num-gaussians",
    "train/total-train-time",
]

METRIC_PRETTY_NAMES = {
    "eval-all-test/psnr": "PSNR ↑",
    "eval-all-test/ssim": "SSIM ↑",
    "eval-all-test/lpips": "LPIPS ↓",
    "train/num-gaussians": "Num Gaussians",
    "train/total-train-time": "Train Time (min)",
}
PER_SCENE_VARYING_PARAMS = {"scene", "dense_init.target_num_points", "strategy.cap_max"}

PLOT_RANGES_PER_METRIC_SCANNETPP = {
    "eval-all-test/psnr": (15, 35.5),
    "eval-all-test/ssim": (0.7, 1.0),
    "eval-all-test/lpips": (0.0, 0.4),
    "train/num-gaussians": (0, 5e6),
    "train/total-train-time": (0, 90),
    "PSNR per 1M Gaussians": (5, 18),
    "SSIM per 1M Gaussians": (0.25, 0.5),
    "(1 - LPIPS) per 1M Gaussians": (0.15, 0.4),
}

PLOT_RANGES_PER_METRIC_ETH3D = {
    "eval-all-test/psnr": (15, 35),
    "eval-all-test/ssim": (0.7, 1.0),
    "eval-all-test/lpips": (0.0, 0.4),
    "train/num-gaussians": (0, 8e6),
    "train/total-train-time": (0, 90),
    "PSNR per 1M Gaussians": (1, 14),
    "SSIM per 1M Gaussians": (0.1, 0.35),
    "(1 - LPIPS) per 1M Gaussians": (0.05, 0.3),
}

PLOT_RANGES_PER_METRIC = {
    "scannet++": PLOT_RANGES_PER_METRIC_SCANNETPP,
    "eval_on_train_set_scannet++": PLOT_RANGES_PER_METRIC_SCANNETPP,
    "eth3d": PLOT_RANGES_PER_METRIC_ETH3D,
}

METRIC_NAME_MAP = {
    "eval-all-test/psnr": "PSNR",
    "eval-all-test/ssim": "SSIM",
    "eval-all-test/lpips": "LPIPS",
    "train/num-gaussians": "#Gaussians",
    "train/total-train-time": "Train Time",
}

DENSE_INIT_METRICS = DEFAULT_TABLE_METRICS + [
    "dense_init.target_num_points",
    "dense_init.target_points_fraction",
]

INIT_METHOD_COLORS = [plt.get_cmap("tab10")(index) for index in range(10)]

# Plot start values shared by the EDGS/monodepth/DA3 real-init bar charts.
REAL_INIT_PLOT_STARTS = {
    "scannet++": {
        "eval-all-test/psnr": 20,
        "eval-all-test/ssim": 0.825,
        "eval-all-test/lpips": 0.2,
    },
    "eval_on_train_set_scannet++": {
        "eval-all-test/psnr": 30.0,
        "eval-all-test/ssim": 0.9,
        "eval-all-test/lpips": 0.05,
    },
    "mipnerf360": {
        "eval-all-test/psnr": 25.0,
        "eval-all-test/ssim": 0.77,
        "eval-all-test/lpips": 0.1,
    },
    "tanksandtemples": {
        "eval-all-test/psnr": 20.75,
        "eval-all-test/ssim": 0.77,
        "eval-all-test/lpips": 0.12,
    },
}

# Plot start values shared by the laser-scan line charts (main + NanoGS).
LINE_CHART_PLOT_STARTS = {
    "scannet++": {
        "eval-all-test/psnr": 21,
        "eval-all-test/ssim": 0.8525,
        "eval-all-test/lpips": 0.22,
        "train/total-train-time": 6.5,
    },
    "eval_on_train_set_scannet++": {
        "eval-all-test/psnr": 31.5,
        "eval-all-test/ssim": 0.935,
        "eval-all-test/lpips": 0.07,
        "train/total-train-time": 6.5,
    },
    "eth3d": {
        "eval-all-test/psnr": 20.5,
        "eval-all-test/ssim": 0.76,
        "eval-all-test/lpips": 0.16,
        "train/total-train-time": 7.5,
    },
}

TABLE_ROUNDING_PER_METRIC = {
    "eval-all-test/psnr": 2,
    "eval-all-test/ssim": 3,
    "eval-all-test/lpips": 3,
    "train/total-train-time": 1,
}


def get_default_strategy_args(strategy: str, dataset: str) -> dict[str, str]:
    args = DEFAULT_STRATEGY_ARGS[strategy]
    return {
        key: (value[dataset] if isinstance(value, dict) else value)
        for key, value in args.items()
    }
