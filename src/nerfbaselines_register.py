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

register(IVD_SPLAT_METHOD_SPEC)

register(
    {
        "id": "mlflow",
        "logger_class": "mlflow_nerfbaselines_logger.logger:MLflowLogger",
    }
)
