from nerfbaselines import register
from monodepth.proxy_dataset import PROXY_DATASET_ID as MONODEPTH_PROXY_DATASET_ID
from edgs.proxy_dataset import PROXY_DATASET_ID as EDGS_PROXY_DATASET_ID
from da3.proxy_dataset import PROXY_DATASET_ID as DA3_PROXY_DATASET_ID
from ivd_splat.nerfbaselines_integration.method_spec import IVD_SPLAT_METHOD_SPEC

# Register with nerfbaselines. For this to work, NERFBASELINES_REGISTER needs to contain a path to this file.
register(
    {
        "id": MONODEPTH_PROXY_DATASET_ID,
        "load_dataset_function": "monodepth.proxy_dataset:monodepth_proxy_dataset_loader",
    }
)
register(
    {
        "id": EDGS_PROXY_DATASET_ID,
        "load_dataset_function": "edgs.proxy_dataset:edgs_proxy_dataset_loader",
    }
)

register(
    {
        "id": DA3_PROXY_DATASET_ID,
        "load_dataset_function": "da3.proxy_dataset:da3_proxy_dataset_loader",
    }
)


register(
    {
        "id": "mlflow",
        "logger_class": "mlflow_nerfbaselines_logger.logger:MLflowLogger",
    }
)

register(IVD_SPLAT_METHOD_SPEC)


def download_not_implemented(*args, **kwargs):
    raise NotImplementedError(
        "This dataset does not have a download function implemented."
    )


register(
    {
        "id": "mipnerf360-sparsified",
        "download_dataset_function": "customized_dataset_loaders.sparsifying_colmap_loader:download_not_implemented",
        "evaluation_protocol": "nerf",
        "metadata": {
            "id": "mipnerf360-sparsified",
            "name": "Mip-NeRF 360 Sparsified",
            "scenes": [
                {"id": scene, "name": scene.title()}
                for scene in [
                    "bicycle",
                    "bonsai",
                    "counter",
                    "garden",
                    "kitchen",
                    "room",
                    "stump",
                    "treehill",
                    "flowers",
                ]
            ],
        },
    }
)

register(
    {
        "id": "tanksandtemples-sparsified",
        "download_dataset_function": "customized_dataset_loaders.sparsifying_colmap_loader:download_not_implemented",
        "evaluation_protocol": "default",
        "metadata": {
            "id": "tanksandtemples-sparsified",
            "name": "Tanks and Temples Sparsified",
            "scenes": [
                {"id": scene, "name": scene.title()}
                for scene in [
                    "auditoriumballroom",
                    "courtroom",
                    "museum",
                    "m60",
                    "panther",
                    "church",
                    "meetingroom",
                    "playground",
                    "palace",
                    "temple",
                    "family",
                    "francis",
                    "horse",
                    "lighthouse",
                    "train",
                    "barn",
                    "caterpillar",
                    "courthouse",
                    "ignatius",
                    "truck",
                ]
            ],
        },
    }
)
