import glob
import os.path
from typing import Union

import numpy as np

from datasets.base import BaseDataset


class KittiRawDataset(BaseDataset):
    def __init__(self, data_path,
                 point_cloud_transform=None,
                 drop_invalid_point_function=None,
                 n_points=None,
                 apply_pillarization=True,
                 point_features=3):
        super().__init__(data_path=data_path,
                         point_cloud_transform=point_cloud_transform,
                         drop_invalid_point_function=drop_invalid_point_function,
                         n_points=n_points,
                         apply_pillarization=apply_pillarization,
                         point_features=point_features)

        self.files = glob.glob(os.path.join(data_path, "*/*/pairs/*.npz"))
        self.frame = None

    def __len__(self):
        return len(self.files)

    def get_pointcloud_t0(self, index: int) -> np.ndarray:
        self.frame = np.load(self.files[index])
        return self.frame['pcl_t0']

    def get_pointcloud_t1(self, index: int) -> Union[np.ndarray, None]:
        self.frame = np.load(self.files[index])
        return self.frame['pcl_t1']

    def get_pose_transform(self, index: int) -> Union[np.ndarray, None]:
        self.frame = np.load(self.files[index])
        return np.linalg.inv(self.frame['odom_t0_t1'])

    def get_global_transform(self, index: int) -> Union[np.ndarray, None]:
        return self.frame['global_pose']

    def get_flow(self, index: int) -> Union[np.ndarray, None]:
        return None
