_name = "ivd-splat"

IVD_SPLAT_METHOD_SPEC = {
    "id": _name,
    "method_class": "ivd_splat.nerfbaselines_integration.method:IVDSplat",
    "conda": {
        "environment_name": "ivd_splat_nerfbaselines",
        "python_version": "3.10",
        "install_script": r"""unimplemented""",
    },
    "metadata": {
        "name": "ivd_splat",
        "description": """""",
    },
    "presets": {
        "blender": {
            "@apply": [{"dataset": "blender"}],
            "init_type": "random",
            "background_color": (1.0, 1.0, 1.0),
            "random_init.extent": 0.5,
        },
        "phototourism": {
            "@apply": [{"dataset": "phototourism"}],
            "app_opt": True,  # Enable appearance optimization
            "steps_scaler": 3.333334,  # 100k steps
        },
    },
    "implementation_status": {},
    "required_features": frozenset(
        (
            "points3D_xyz",
            "points3D_rgb",
            "points3D_normals",
            "color",
            "images_points3D_indices",
        )
    ),
    "supported_camera_models": frozenset(("pinhole",)),
    "supported_outputs": ("color", "depth", "accumulation"),
}
