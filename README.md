# Official implementation for "The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting".
This is the official implementation for the paper "The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting" (https://arxiv.org/abs/2603.20714), which deals with benchmarking 3DGS performance under different initialization and densification strategies.


# Installation
External requirements:
- **CUDA 12.6** - other versions would most likely work but require changing torch wheel sources in `pyproject.toml`
- **C++ compiler with C++20 support** - used to compile some native code and `third-party/diff-gaussian-rasterization-idhfr`.

Use the `install_with_pip.sh` script to install this project and its dependencies. The script will also clone git submodules.
Unfortunately a clean solution with a single pyproject.toml was not tenable, as some dependencies require `--no-build-isolation`.

# Project structure

- `src/`
    - `ivd_splat/` - main gsplat-based Gaussian Splatting implementation
        - `strategies/` - implementation of individual densification strategies
        - ...
    - `monodepth/` - Monodepth initialization, can be executed with `monodepth` command when installed (prefer using `init_runner`).
    - `da3/` - DA3 initialization, can be executed with `da3_init` command when installed (prefer using `init_runner`).
    - `edgs/` - EDGS* initialization, can be executed with `edgs` command when installed (prefer using `init_runner`).
    - `shared/` - some functionality that is shared by multiple parts of the project.
    - `mlflow_nerfbaselines_logger` - mlflow integration for NerfBaselines to enable logging training statistics and metrics to an mlflow server instance.
    - `nerfbaselines_register.py` - utility file to register `ivd_splat` and proxy dataset readers of initialization method implementations with nerfbaselines.
- `packages/` - first-party code that is structured as separate python projects with their own pyproject.toml files    
    - `eval_scripts` includes orchestrator scripts to easily invoke initialization and training on any datasets/scenes with mlflow logging integration (`ivd_splat_runner`, `init_runner`), as well some results processing utilities.
    - `native_modules` is the catch all for any native code used by the implementation, but currently only contains a C++ LO-RANSAC implementation used in `src/monodepth`.
- `experiments` - SLURM job scheduler scripts used to run all experiments reported in the paper. See [experiments/README.md](experiments/README.md) for details.
- `submodules/` - git submodules
- `third-party/` - third-party code that could not be included as git submodules 
    - `diff-gaussian-rasterization-idhfr` - modified version of diff-gaussian-rasterization used by official IDHFR implementation (https://github.com/XiaoBin2001/Improved-GS/ - `submodules.zip`)

### Related repositories
- ScanNet++ dataset integration - https://github.com/deivse/ivd_splat_scannetpp_integration
- ETH3D dataset integration - https://github.com/deivse/ivd_splat_eth3d_integration
- NerfBaselines fork used in this repository, which supports additional fields for datasets - https://github.com/deivse/nerfbaselines


# Usage
Once installed, the easiest way to replicate reported experiment results is to use the `ivd_splat_runner` and `init_runner` utilities provided by `packages/eval_scripts`. Once the
module is installed, they are added to path and can be invoked directly, e.g. try running `ivd_splat_runner --help`.

Both utilities take an `--output-dir` CLI argument which specifies a directory where initialization and training output will be written. Both utilities understand a common directory structure, so as long as the same `--output-dir` is provided, it is easy to use the initialization data produced with `init_runner` for training with `ivd_splat_runner`, for example:
```shell
# 1. Run monodepth initialization for all scenes of mipnerf360 dataset and save output to ./results
init_runner --output-dir results --method monodepth --datasets mipnerf360
# 2. Run training using 3DGS-MCMC using the monodepth initialization data from ./results
ivd_splat_runner --output-dir results --datasets mipnerf360 --configs "strategy={MCMCStrategy}" --init-methods monodepth
```

Both `init_runner` and `ivd_splat` runner are set up to log information to mlflow, and will respect the "MLFLOW_TRACKING_URI" environment variable. If that variable is not set, logging will default to using a local sqlite database in `<output_dir>/mlflow.db`.

`ivd_splat` runner accepts a config parameter, which allows to specify any number of configurations that will be ran using a specialized syntax. For example, `--configs "strategy={MCMCStrategy, IDHFRStrategy}"` will result in two runs for each scene, one using MCMC and one using IDHFR.

For advanced examples of using the runner scripts, please consult `--help` output of the two runner utilities, and see the real invocations in `experiments/main` used to produce the results in the paper.

# Future

This is an initial code release, in the future, we hope to add detailed instructions on adding new initialization methods and strategies.

# License 
Note: This is really messy, but this is the only way I was able to abide by the licensing terms of all third-party code (to the best of my knowledge) while keeping the main license as open as possible.

The core project is licensed under the terms of the non-commercial license specified in the root `LICENSE` file.
Some of the other code included in the project or aggregated in this repository is licensed under different terms:
- We license the code under `src/shared` which is included in many other parts of the project under the MIT license (see [src/shared/LICENSE](src/shared/LICENSE))
- The `diff-gaussian-rasterization` implementation included in `third-party/diff-gaussian-rasterization-idhfr` is licensed under the Gaussian-Splatting License, which prohibits commercial use.
- The Monodepth implementation in `src/monodepth` is governed by the terms of GPLv3 ([src/monodepth/LICENSE](src/monodepth/LICENSE)), as it uses some GPLv3 code. It is used as a separate utility.
- The EDGS implementation in `src/edgs` follows the original EDGS license (`src/edgs/LICENSE`) which prohibits commercial use. It is used as a separate utility.
- RoMA, included as a git submodule in `submodules/RoMA`, is licensed under the MIT license.
- The ivd_splat code and some of the strategies are based on gsplat examples and the core gsplat library code, licensed under Apache 2.0. Attribution is provided at the top of individual files where applicable, but the modified work is licensed under the same terms as the core project.
- The IDHFR strategy implementation is based on the official implementation which seems to use the Gaussian Splatting license.
