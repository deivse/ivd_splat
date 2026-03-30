#!/bin/bash
#SBATCH --job-name=mlflow-server
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --partition=amdextralong
#SBATCH --output=/home/desiaiva/logs/mlflow_server.log

# Can't use script_dir here because location changes when running via slurm
REPO_PATH="$HOME/monocular_depth_tsdf_fusion"

export SLURM_MLFLOW_IS_SERVER_NODE=1
source "$REPO_PATH/experiments/common_slurm_setup.sh"

# Write server url to file for other nodes to read
echo $(hostname) > $SLURM_MLFLOW_SERVER_NODE_FILE
ifconfig eth0 | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' > $SLURM_MLFLOW_SERVER_NODE_IP_FILE

mlflow server \
  --host 0.0.0.0 \
  --port $SLURM_MLFLOW_SERVER_PORT \
  --backend-store-uri sqlite:////$RESULTS_DIR/mlflow.db \
  --default-artifact-root $RESULTS_DIR/mlruns
