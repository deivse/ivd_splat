# Training with User-Defined Initialization/Densification

## Densification

Densification strategies can be defined as python classes that implement the interface defined by the `abc`-based `IVDSplatBaseStrategy` base class ([src/ivd_splat/strategies/base.py](/src/ivd_splat/strategies/base.py)). We include a strategy discovery mechanism that works by searching the directory pointed to by the `IVD_SPLAT_ADDITIONAL_STRATEGIES_DIR` environment variable for strategy implementations.

To add a custom strategy, do the following:
1. Create a new directory, where you will put your implementation e.g. `ivd_splat_strategies`, and set the environment variable `IVD_SPLAT_ADDITIONAL_STRATEGIES_DIR` to the absolute path to that directory (must be set when `ivd_splat` is invoked.)
2. Create a file under that directory with any name, containing a python class implementing the strategy. As a first step, we recommended to familiarize yourself with one of the existing implementations in `src/ivd_splat/strategies/`. A given strategy class must:
    - Derive from `IVDSplatBaseStrategy` (hard requirement for discovery)
    - Be a dataclass (hard requirement for logging config parameters to mlflow)
    - Implement all `IVDSplatBaseStrategy` abstract methods
    - Implement any methods with default behaviour you want to override (please see `IVDSplatBaseStrategy` documentation comments)
    - Additionally, you might want to add the following to your class definition:
        ```python
        CONFIG_SERIALIZATION_IGNORED_FIELDS: typing.ClassVar[set[str]] = {
            # ... list of fields that should not be logged as run parameters to mlflow.
        }
        ```

With these steps complete, you can train with the new strategy using `ivd_splat_runner` by passing `strategy={YourStrategyClassName}` in the config string. You can also override any parameters specific to your strategy with `strategy.param_name={value, ...}`.

## Initialization

To add a new initialization method, one simply needs to create a standalone program that, given a scene ID, and an output directory, loads the original nerfbaselines dataset, runs the initialization method, and saves a so-called proxy dataset, as well as some additional metadata to the output directory.
While any program that follows the CLI API contract, and produces a proxy dataset, as well as corresponding ivd_splat init metadata, can be used with `init_runner` (and its output with `ivd_splat_runner`), this is easiest to implement as a python script added to PATH, which is the case for the provided `monodepth` and `edgs` initialization methods. Overall, a valid init method executable must conform to the following:
1. Accepts `--scene` and `--output-dir` arguments.
2. Creates a nerfbaselines (proxy) dataset in the output directory.
3. Creates `init_info.json` in the output directory.

See below for details.


### CLI API
An initialization method executable must accept the following mandatory CLI arguments:
- `--output-dir` - directory where output will be saved, e.g. `results/mipnerf360/garden/init_method/default`
- `--scene` - scene identifier in the format used by our code base. You should use the `eval_scripts.dataset_scenes.scene_id_to_nerfbaselines_data_value()` function to convert our scene ID into a value that can be passed directly to `nerfbaselines.load_dataset()`.

It may also accept other arbitrary configuration via CLI. These can then be set via config strings (see [Config Strings Reference](reference.md#config-strings)).

### Proxy datasets

The initialization method executable should create a so-called proxy dataset in the directory pointed to by `--output-dir`. When the path to this directory is passed as a dataset to `nerfbaselines`, it should load the original dataset data corresponding to `--scene`, but can modify or extend it (usually by replacing the `points3D_xyz` and `points3D_rgb` dataset features). This is achieved by:
1. storing any produced points/splats in the `--output-dir`, and 
2. saving a custom `nb-info.json` that includes a reference to the original dataset, and names a custom nerfbaselines loader that will load the original dataset using `nerfbaselines.load_dataset` and modify it as needed.

Please see [src/monodepth/proxy_dataset.py](/src/monodepth/proxy_dataset.py) for a concrete example. *Make sure to register your custom dataset loader with nerfbaselines.*

### Init Metadata

Finally, your executable should create a file named `init_info.json` to the output directory. We suggest using [`shared.save_init_info.save_init_info_json()`](/src/shared/save_init_info.py). The created file should have the following fields:
```json
{
    // Initialization type, as accepted by `ivd_splat`. In practice, this will be either
    // "dense" or "splat" for init methods that produce a point cloud or splat data directly.
    "init_type": "dense", 
    // A list of files that are required to use the output proxy dataset and indicate a complete output. 
    // The paths are relative to the output directory. These are used by `init_runner` to check if output for
    // a certain scene and config string is complete, or if it requires overwriting.
    "required_files": [
        "points.ply",
        "nb-info.json"
    ]
}
```





