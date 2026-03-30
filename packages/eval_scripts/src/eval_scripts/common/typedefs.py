from enum import Enum


class InitMethod(str, Enum):
    sfm = "sfm"
    monodepth = "monodepth"
    edgs = "edgs"
    da3 = "da3"
    gt_pointcloud = "gt_pointcloud"
