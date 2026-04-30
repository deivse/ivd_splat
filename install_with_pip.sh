#!/bin/bash

# Use this script to install this project and its dependencies.
# Unfortunately a clean solution with a single pyproject.toml was not tenable, as some dependencies require --no-build-isolation.
# The main project and eval_scripts are installed in editable mode, so that you can edit the code and have the changes reflected without reinstalling.

git submodule update --init --recursive

pip install -e .                         \
            -e packages/eval_scripts/    \
            -v packages/native_modules/  \
            submodules/RoMa \
            git+https://github.com/deivse/NanoGS.git # Using fork for now, can switch to main if PR is merged.

pip install --no-build-isolation "fused_ssim @ git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" \
            --no-build-isolation third-party/diff-gaussian-rasterization-idhfr/
