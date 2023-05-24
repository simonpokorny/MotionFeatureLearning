import glob
import os.path

import numpy as np
from typing import Union

from datasets.base import BaseDataset


class NuScenesDataset(BaseDataset):
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

        self.files = glob.glob(os.path.join(data_path, "*.npz"))
        self.frame = None

        scanning_frequency = 20
        self.m_thresh = 0.05 * (10 / scanning_frequency)

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
        return self.frame["flow_t0_t1"]

    def get_classes(self, index: int) -> Union[np.ndarray, None]:
        self.frame = np.load(self.files[index])

        flow = self.frame["flow_t0_t1"]
        frame = self.frame['pcl_t0']
        odometry = np.linalg.inv(self.frame['odom_t0_t1'])
        static_flow = self.compute_gt_stat_flow(frame, odometry)
        labels = (np.linalg.norm(flow - static_flow, axis=1) > (self.m_thresh)).reshape(-1,1)
        return labels

    @staticmethod
    def cart2hom(pcl):
        assert pcl.ndim == 2 and pcl.shape[1] == 3, "PointCloud should be in shape [N, 3]"
        N, _ = pcl.shape
        return np.concatenate((pcl, np.ones((N, 1))), axis=1)

    @staticmethod
    def hom2cart(pcl):
        assert pcl.ndim == 2 and pcl.shape[1] == 4, "PointCloud should be in shape [N, 4]"
        return pcl[:, :3] / pcl[:, 3:4]

    def compute_gt_stat_flow(self, pcl_t0, odometry):
        pcl_t0 = pcl_t0[:, :3]
        pcl_t1 = self.cart2hom(pcl_t0).T
        pcl_t1 = (odometry @ pcl_t1).T
        pcl_t1 = self.hom2cart(pcl_t1)
        flow = pcl_t1 - pcl_t0
        return flow
