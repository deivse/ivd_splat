# Official implementation for "The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting".
This is the official implementation for the paper "The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting" (https://arxiv.org/abs/2603.20714), which deals with benchmarking 3DGS performance under different initialization and densification strategies.


# Installation
External requirements:
- **CUDA 12.6** - other versions would most likely work but require changing torch wheel sources in `pyproject.toml`
- **C++ compiler with C++20 support** - used to compile some native code and `third-party/diff-gaussian-rasterization-idhfr`.

Use the `install_with_pip.sh` script to install this project and its dependencies. The script will also clone git submodules.
Unfortunately a clean solution with a single pyproject.toml was not tenable, as some dependencies require `--no-build-isolation`.


# Usage

## Running initialization and training
Once installed, the easiest way to replicate reported experiment results is to use the `ivd_splat_runner` and `init_runner` utilities provided by `packages/eval_scripts`. Once the
module is installed, they are added to path and can be invoked directly, e.g. try running `ivd_splat_runner --help`.

Both utilities take an `--output-dir` CLI argument which specifies a directory where initialization and training output will be written. Both utilities understand a common directory structure, so as long as the same `--output-dir` is provided, it is easy to use the initialization data produced with `init_runner` for training with `ivd_splat_runner`, for example:
```shell
# 1. Run monodepth initialization for all scenes of mipnerf360 dataset and save output to ./results
init_runner --output-dir results --method monodepth --datasets mipnerf360
# 2. Run training using 3DGS-MCMC using the monodepth initialization data from ./results
ivd_splat_runner --output-dir results --datasets mipnerf360 --configs "strategy={MCMCStrategy}" --init-method monodepth
```

Both `init_runner` and `ivd_splat` runner are set up to log information to mlflow, and will respect the "MLFLOW_TRACKING_URI" environment variable. If that variable is not set, logging will default to using a local sqlite database in `<output_dir>/mlflow.db`.

`ivd_splat_runner` and `init_runner` accept config string parameters, which allow to specify configuration overrides for ivd_splat or the initialization method respectively. Each config string can specify multiple values for a parameter, and the runner will execute initialization or training once for each possible parameter combination. Both ofg them also use a common scene ID format. See [docs/reference.md](docs/reference.md) for more information.
Additionally, if running via SLURM in an array context, the runners will automatically calculate which array job should run which tasks and only run the tasks (configurations) assigned to their job id (the allocations are static, nothing fancy like work stealing).

For advanced examples of using the runner scripts, please consult the `--help` output of the two runner utilities, and see the real invocations in `experiments/main` used to produce the results in the paper.

## Adding new initialization and densification methods

The code base supports plug-and-play integration of additional initialization and densification methods without modifying ivd_splat itself. Please see [docs/custom_methods_integration.md](docs/custom_methods_integration.md) for detailed instructions.

# Project structure

This repository contains most of the code required to replicate the results in the paper. See [Directories & Files Reference](docs/reference.md#directories-and-files) for a detailed introduction to the project structure. Dataset integration code for ScanNet++ and ETH3D is contained in separate repositories, listed below.

### Related repositories
- ScanNet++ dataset integration - https://github.com/deivse/ivd_splat_scannetpp_integration
- ETH3D dataset integration - https://github.com/deivse/ivd_splat_eth3d_integration
- NerfBaselines fork used in this repository, which supports additional fields for datasets - https://github.com/deivse/nerfbaselines (we plan to integrate a feature into `nerfbaselines` upstreamd that will allow to drop this fork).

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
