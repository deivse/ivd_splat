#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python "$SCRIPT_DIR/create_linked_dataset.py" \
  mipnerf360 \
  mipnerf360-sparsified \
  customized_dataset_loaders.sparsifying_colmap_loader:load_and_sparsify_colmap_dataset

python "$SCRIPT_DIR/create_linked_dataset.py" \
  tanksandtemples \
  tanksandtemples-sparsified \
  customized_dataset_loaders.sparsifying_colmap_loader:load_and_sparsify_colmap_dataset
