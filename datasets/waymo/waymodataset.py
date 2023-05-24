import os.path
import pickle
from typing import Union

import numpy as np

from datasets.base import BaseDataset


class WaymoDataset(BaseDataset):
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

        # Config parameters
        metadata_path = os.path.join(data_path, 'metadata')
        # It has information regarding the files and transformations

        self.data_path = data_path
        scanning_frequency = 10
        self.m_thresh = 0.05 * (10 / scanning_frequency)

        try:
            with open(metadata_path, 'rb') as metadata_file:
                self.metadata = pickle.load(metadata_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Metadata not found, please create it by running preprocess.py. file : {metadata_path}")

        self._n_points = n_points

    def __len__(self):
        """
        For each dataset should be separetly written.
        Returns:
            length of the dataset
        """
        return len(self.metadata['look_up_table'])

    def get_pointcloud_t0(self, index: int) -> np.ndarray:
        """
        Read from disk the current and previous point cloud given an index
        """
        # In the lookup table entries with (current_frame, previous_frame) are stored
        frame_t0 = np.load(os.path.join(self.data_path,
                                              self.metadata['look_up_table'][index][0][0]))['frame'][:, :5]
        return frame_t0

    def get_pointcloud_t1(self, index: int) -> Union[np.ndarray, None]:
        frame_t1 = np.load(os.path.join(self.data_path,
                                              self.metadata['look_up_table'][index][1][0]))['frame'][:, :5]
        return frame_t1

    def get_pose_transform(self, index: int) -> Union[np.ndarray, None]:
        frame_t0 = self.metadata['look_up_table'][index][0][1]
        frame_t1 = self.metadata['look_up_table'][index][1][1]

        # G_T_C -> Global_TransformMatrix_Current (current frame to global frame)
        G_T_P = np.reshape(np.array(frame_t0), [4, 4])

        # G_T_P -> Global_TransformMatrix_Previous (previous frame to global frame)
        G_T_C = np.reshape(np.array(frame_t1), [4, 4])

        # Transformation matrix Previous (t0) to Current (t1)
        C_T_P = np.linalg.inv(G_T_C) @ G_T_P

        return C_T_P

    def get_global_transform(self, index: int) -> Union[np.ndarray, None]:
        previous_frame_pose = self.metadata['look_up_table'][index][1][1]

        # G_T_P -> Global_TransformMatrix_Previous
        G_T_P = np.reshape(np.array(previous_frame_pose), [4, 4])
        return G_T_P

    def get_flow(self, index: int) -> Union[np.ndarray, None]:
        previous_frame = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][0][0]))['frame']
        flows = - (previous_frame[:, -4:-1])
        odometry = self.get_pose_transform(index)
        gt_static_flow = self.compute_gt_stat_flow(previous_frame, odometry)
        flows = flows / 10 + gt_static_flow
        return flows

    def get_classes(self, index: int) -> Union[np.ndarray, None]:
        previous_frame = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][0][0]))['frame']
        flows = - (previous_frame[:, -4:-1])
        odometry = self.get_pose_transform(index)

        gt_static_flow = self.compute_gt_stat_flow(previous_frame, odometry)
        flows = flows / 10 + gt_static_flow

        label = np.linalg.norm(flows - gt_static_flow, axis=1) > self.m_thresh
        return label.reshape((-1, 1))

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
