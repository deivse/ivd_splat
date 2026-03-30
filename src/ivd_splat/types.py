from typing import NamedTuple

import torch


class InputImage(NamedTuple):
    data: torch.Tensor
    name: str
    cam2world: torch.Tensor
    K: torch.Tensor
