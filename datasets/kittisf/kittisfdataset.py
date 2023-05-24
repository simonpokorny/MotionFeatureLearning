import glob
import os.path

import numpy as np
from typing import Union

from datasets.base import BaseDataset


class KittiSceneFlowDataset(BaseDataset):
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

        self.files = glob.glob((os.path.join(data_path, "*.npz")))
        self.frame = None

    def __len__(self):
        """
        For each dataset should be separetly written.
        Returns:
            length of the dataset
        """
        return len(self.files)

    def get_pointcloud_t0(self, index: int) -> np.ndarray:
        self.frame = np.load(self.files[index])

        x, z, y = self.frame['pc1'].T
        #y = y - 35
        pc1 = np.stack((x, y, z)).T
        return pc1

    def get_pointcloud_t1(self, index: int) -> Union[np.ndarray, None]:
        self.frame = np.load(self.files[index])

        x, z, y = self.frame['pc2'].T
        #y = y - 35
        pc2 = np.stack((x, y, z)).T
        return pc2

    def get_global_transform(self, index: int) -> Union[np.ndarray, None]:
        return self.frame['global_pose']

    def get_flow(self, index: int) -> Union[np.ndarray, None]:
        self.frame = np.load(self.files[index])

        x, z, y = self.frame['flow'].T
        flow = np.stack((x, y, z)).T
        return flow
