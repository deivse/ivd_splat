# Detailed Reference

This file contains detailed descriptions of specific user-facing parts of the code.

## Scene IDs
We use a slightly different way to identify scenes than nerfbaselines.
For any supported dataset (currently, any dataset natively supported by nerfbaselines, as well as ScanNet++ and ETH3D), a scene can be specified as follows `<dataset_id>/<scene_id>`, where:
- `<dataset_id>` is any valid nerfbaselines dataset name (without "external://"), or one of `scannet++`, `eval_on_train_set_scannet++`, `eth3d`, assuming the corresponding dataset is properly installed (see [ScanNet++ Integration](https://github.com/deivse/ivd_splat_scannetpp_integration), [ETH3D Integration](https://github.com/deivse/ivd_splat_eth3d_integration)).
- `<scene_id>` is the scene identifier.

For example, for the nerfbaselines scene ID `external://mipnerf360/garden`, our scene ID is simply `mipnerf360/garden`. For datasets that are natively supported by nerfbaselines, this simply removes the
"external://" prefix. For ScanNet++, ETH3D, and any future custom dataset integrations, this allows to specify just the dataset name, and as long as the environment is set up correctly, `init_runner`, `ivd_splat_runner`, or any other code using `eval_scripts.dataset_scenes.scene_id_to_nerfbaselines_data_value()` will be able to find the dataset location and pass the absolute path to `nerfbaselines`.

## Config Strings

Config strings serve as a simple and compact way to specify a set of possible parameter values, which are then used to invoke a certain method/executable with *all combinations of these parameters*. 

For example, these are used by `ivd_splat_runner` to allow to specify arbitrary parameters for `ivd_splat`. Invoking , `ivd_splat_runner --configs "strategy={MCMCStrategy, IDHFRStrategy}"` will result in two runs for each scene, one using MCMC and one using IDHFR. If we specify a config string with 2 parameters with 2 possible values each, e.g., for
```bash
ivd_splat_runner ... --configs "paramA={a, b} paramB={c, d}"
```
`ivd_splat_runner` would execue `ivd_splat` with 4 different combinations of CLI arguments:
```bash
ivd_splat ... --paramA="a" --paramB="c"
ivd_splat ... --paramA="a" --paramB="d"
ivd_splat ... --paramA="b" --paramB="c"
ivd_splat ... --paramA="b" --paramB="d"
```

### Syntax
A config string is a list of `<parameter_spec>` separated by spaces.

A `<parameter_spec>` is `<parameter>={<value>, ...}`, where curly brackets `{}` are required even if there's only one value.
If an executable (`ivd_splat` or an init method) accepts `--parameter-name` on its CLI, then `<parameter>` can be either `parameter-name` or `--parameter-name`.

## Directories and Files

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
- `experiments/` - SLURM job scheduler scripts used to run all experiments reported in the paper. See [experiments/README.md](experiments/README.md) for details.
- `results_scripts/` - contains ipynb used to produce all results in the paper (obtains data from mlflow) and related files.
- `submodules/` - git submodules
- `third-party/` - third-party code that could not be included as git submodules 
    - `diff-gaussian-rasterization-idhfr` - modified version of diff-gaussian-rasterization used by official IDHFR implementation (https://github.com/XiaoBin2001/Improved-GS/ - `submodules.zip`)
