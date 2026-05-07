from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from gsplat.exporter import export_splats as gsplat_export_splats  # type: ignore
import numpy as np
import torch


@dataclass
class SplatData:
    means: torch.Tensor  # (num_splats, 3)
    scales: torch.Tensor  # (num_splats, 3) before activations
    quats: torch.Tensor  # (num_splats, 4)
    opacities: torch.Tensor  # (num_splats,) before activations
    sh0: torch.Tensor  # (num_splats, 1, 3)
    shN: torch.Tensor  # (num_splats, (N + 1) ** 2 - 1, 3)

    def select_random_subset_inplace(self, num_points: int):
        indices = torch.randperm(self.means.shape[0])[:num_points]

        self.means = self.means[indices]
        self.scales = self.scales[indices]
        self.sh0 = self.sh0[indices]
        self.shN = self.shN[indices]
        self.opacities = self.opacities[indices]
        self.quats = self.quats[indices]


def load_splat_ply(path: Path | str) -> SplatData:
    path = Path(path)
    with path.open("rb") as f:
        ply_bytes = f.read()
    return _parse_splat_ply_bytes(ply_bytes)


def export_splat_ply(path: Path | str, splats: SplatData):
    gsplat_export_splats(
        splats.means,
        splats.scales,
        splats.quats,
        splats.opacities,
        # The gsplat export function expects very specific shape.
        splats.sh0.reshape(
            splats.sh0.shape[0], 1, splats.sh0.shape[-1]
        ),  # (num_splats, 1, X)
        splats.shN,
        format="ply",
        save_to=path,
    )


def _parse_splat_ply_bytes(ply_bytes: bytes) -> SplatData:
    buffer = BytesIO(ply_bytes)
    num_splats = None
    properties = []

    while True:
        line = buffer.readline()
        if not line:
            raise ValueError("Invalid PLY header.")
        line_str = line.decode("ascii").strip()

        if line_str.startswith("format") and "binary_little_endian" not in line_str:
            raise ValueError("Only binary little-endian PLY format is supported.")
        elif line_str.startswith("element vertex"):
            num_splats = int(line_str.split()[-1])
        elif line_str.startswith("property"):
            dtype, name = line_str.split()[1:3]
            assert dtype == "float", f"Unsupported property type: {dtype}"
            properties.append(name)
        elif line_str == "end_header":
            break

    if num_splats is None:
        raise ValueError("Missing vertex count in PLY header.")

    num_props = len(properties)
    data = np.frombuffer(buffer.read(), dtype=np.dtype(np.float32).newbyteorder("<"))
    if data.size < num_splats * num_props:
        raise ValueError("PLY data is incomplete.")

    data = data[: num_splats * num_props].reshape(num_splats, num_props)

    mean_indices = [i for i, p in enumerate(properties) if p in ("x", "y", "z")]
    assert (
        mean_indices == sorted(mean_indices) and len(mean_indices) == 3
    ), "Expected x, y, z properties for means."
    means = data[:, mean_indices]

    sh0_indices = [i for i, p in enumerate(properties) if p.startswith("f_dc")]
    assert len(sh0_indices) == 3, "Expected 3 SH0 properties."
    sh0 = data[:, sh0_indices].reshape(num_splats, 1, 3)

    shN_indices = [i for i, p in enumerate(properties) if p.startswith("f_rest_")]
    # It is safer to sort these to ensure they are in order (f_rest_0, f_rest_1...)
    shN_indices.sort(key=lambda x: int(properties[x].split('_')[-1]))
    
    num_sh_per_channel = len(shN_indices) // 3
    # Reshape to (N, 3, 15) then transpose to (N, 15, 3)
    shN = data[:, shN_indices].reshape(num_splats, 3, num_sh_per_channel).transpose(0, 2, 1)

    opacity_indices = [i for i, p in enumerate(properties) if p.startswith("opacity")]
    assert len(opacity_indices) == 1, "Expected 1 opacity property."
    opacities = data[:, opacity_indices[0]]

    scale_indices = [i for i, p in enumerate(properties) if p.startswith("scale_")]
    assert len(scale_indices) == 3, "Expected 3 scale properties."
    scales = data[:, scale_indices]

    quat_indices = [i for i, p in enumerate(properties) if p.startswith("rot_")]
    assert len(quat_indices) == 4, "Expected 4 quaternion properties."
    quats = data[:, quat_indices]

    # Make copies since original buffer is not writable which leads to undefined behaviour for torch tensors.
    retval_data = {
        "means": torch.from_numpy(means.copy()),
        "sh0": torch.from_numpy(sh0.copy()),
        "shN": torch.from_numpy(shN.copy()),
        "opacities": torch.from_numpy(opacities.copy()),
        "scales": torch.from_numpy(scales.copy()),
        "quats": torch.from_numpy(quats.copy()),
    }

    return SplatData(**retval_data)
