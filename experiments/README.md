This directory contains the exact code that was used to run all experiments.
It is not really portable, as it was intended to run in a specific environment on a 
compute cluster using SLURM, however we provide it for transparency an replicability.

# File list
<pre>
├── main
|   ├── ...                                                  - various scripts for running individual configurations
│   ├── common_vars.sh                                       - also included in every script
│   ├── get_num_pts_per_scene.py                             - used to get G_max with SfM init and AbsGS densification
│   ├── get_sfm_init_pts_per_scene.py                        - used to get the number of SfM points per scene
│   └── real_init_get_min_num_pts_per_scene.py               - used to get target init size when running initialization with EDGS*, Monodepth
├── common_slurm_setup.sh - included in every script and defines common variables and does other setup
└── mlflow_server.sh      - the sbatch script used to run the mlflow server on a separate node
