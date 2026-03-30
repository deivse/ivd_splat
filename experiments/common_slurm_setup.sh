# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mnt/appl/software/Anaconda3/2024.02-1/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh" ]; then
        . "/mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/appl/software/Anaconda3/2024.02-1/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

ml Anaconda3
conda activate main

# Repo path can be set in parent script or environment
export REPO_PATH="${REPO_PATH:-$HOME/monocular_depth_tsdf_fusion}"

SCANNETPP_LOADER_DIR=$(python -c "import scannetpp_nerfbaselines_loader; import os; print(os.path.dirname(scannetpp_nerfbaselines_loader.__file__))")
ETH3D_LOADER_DIR=$(python -c "import eth3d_nerfbaselines_loader; import os; print(os.path.dirname(eth3d_nerfbaselines_loader.__file__))")
export NERFBASELINES_REGISTER="$REPO_PATH/src/nerfbaselines_register.py:$SCANNETPP_LOADER_DIR/register_scannetpp_loader.py:$ETH3D_LOADER_DIR/register_eth3d_loader.py:$NERFBASELINES_REGISTER"

export RESULTS_DIR="results/"

export SLURM_MLFLOW_SERVER_NODE_FILE="$HOME/.mlflow_server_node"
export SLURM_MLFLOW_SERVER_NODE_IP_FILE="$HOME/.mlflow_server_node_ip"
export SLURM_MLFLOW_SERVER_PORT=6069

if [ -f "$SLURM_MLFLOW_SERVER_NODE_FILE" ] && [ ! "${SLURM_MLFLOW_IS_SERVER_NODE}" ]; then
    export MLFLOW_TRACKING_URI="http://$(cat $SLURM_MLFLOW_SERVER_NODE_IP_FILE):${SLURM_MLFLOW_SERVER_PORT}"
fi


export SCANNETPP_PATH="ivd_splat_scannetpp_integration/processed/"
export SCANNETPP_SCENES="c5439f4607,bcd2436daf,b0a08200c9,6115eddb86,f3d64c30f8,3f15a9266d,5eb31827b7,3db0a1c8f3,40aec5fffa,9071e139d9,e7af285f7d,bde1e479ad,5748ce6f01,825d228aec,7831862f02"

export ETH3D_PATH="ivd_splat_eth3d_integration/eth3d_dataset/"
export ETH3D_SCENES="pipes,kicker,terrace,relief,relief_2,terrains,office"

export TANKSANDTEMPLES_SCENES="auditorium, ballroom, palace, temple, family, horse, lighthouse, m60, train, barn, caterpillar, church, meetingroom, truck"

