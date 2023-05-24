from typing import Union, Callable

import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base class for all datasets.
    """

    def __init__(self, data_path,
                 point_features: int,
                 point_cloud_transform: Callable = None,
                 drop_invalid_point_function: Callable = None,
                 n_points: int = None,
                 apply_pillarization: bool = False):
        """
        Args:
            data_path (str): Path to the point cloud data.
            point_cloud_transform (Callable): Transformation function to apply to the point cloud.
            drop_invalid_point_function (Callable): Function to drop invalid points from the point cloud.
            n_points (int): Number of points to process.
            apply_pillarization (bool): Flag indicating whether to apply pillarization.
                When visualizing, set this to False to display the points without pillarization.
            point_features (int): Number of features to use from the point cloud.
                For example, if we don't want to use intensities, num_features = 6.

        Returns:
            None
        """
        super().__init__()
        self._point_features = point_features
        self._n_points = n_points

        self.data_path = data_path

        # Optional functions to apply to point clouds
        self._drop_invalid_point_function = drop_invalid_point_function
        self._point_cloud_transform = point_cloud_transform
        self._apply_pillarization = apply_pillarization

    def __getitem__(self, index):
        """
        For appropriate function it is mandatory to overwrite  _get_point_cloud_pair and _get_pose_transform functions.

        Args:
            - index: Index for getitem

        Returns:
            - Tuple[t0_frame, t1_frame], where t0_frame is Tuple[pcl, bev_projecticion]
              and t1_frame is Tuple[pcl, bev_projecticion].
            - flows: in shape [N, 4].
            - t0_to_t1: transformation matrix from t0 to t1.

        """

        # Mandatory
        t0_frame = self.get_pointcloud_t0(index)[:, :self._point_features]
        t1_frame = self.get_pointcloud_t1(index)[:, :self._point_features]
        assert type(t0_frame) == np.ndarray and t0_frame.ndim == 2
        assert type(t1_frame) == np.ndarray and t1_frame.ndim == 2

        t0_to_t1 = self.get_pose_transform(index)
        if type(t0_to_t1) == np.ndarray:
            assert t0_to_t1.shape == (4, 4), "Matrix in custom dataset must be in shape (4, 4)"
        else:
            assert t0_to_t1 is None
            t0_to_t1 = np.zeros((4, 4))

        # (optional) ground truth flows from t0 to t1
        flows = self.get_flow(index)
        if type(flows) == np.ndarray:
            assert (flows.shape == t0_frame[:, :3].shape), "Flows should be equal to frame t0"
        else:
            assert flows is None
            flows = np.ones(t0_frame[:, :3].shape) * -100

        classes = self.get_classes(index)
        if type(classes) == np.ndarray:
            assert (classes.shape == t0_frame[:, :1].shape), "Flows should be equal to frame t0"
        else:
            assert classes is None
            classes = np.ones(t0_frame[:, :1].shape) * -100

        # Drop invalid points according to the method supplied
        if self._drop_invalid_point_function is not None:
            t0_frame, mask = self._drop_invalid_point_function(t0_frame)
            t1_frame, _ = self._drop_invalid_point_function(t1_frame)

            flows = flows[mask]
            classes = classes[mask]

        # Subsample points based on n_points
        if self._n_points is not None:
            t0_frame, mask = self._subsample_points(t0_frame)
            t1_frame, _ = self._subsample_points(t1_frame)

            flows = flows[mask]
            classes = classes[mask]

        # Perform the pillarization of the point_cloud
        # Pointcloud after pillarization is in shape [N, 6 + num features]
        if self._point_cloud_transform is not None and self._apply_pillarization:
            t0_frame = self._point_cloud_transform(t0_frame)
            t1_frame = self._point_cloud_transform(t1_frame)
        else:
            # output must be a tuple
            t0_frame = (t0_frame, None)
            t1_frame = (t1_frame, None)

        return (t0_frame, t1_frame), (flows, classes), t0_to_t1

    def __len__(self):
        """
        For each dataset should be separetly written.
        Returns:
            length of the dataset
        """
        raise NotImplementedError()

    def get_pointcloud_t0(self, index: int) -> np.ndarray:
        """
        For each dataset should be separetly written. Returns two consecutive point clouds.
        Args:
            index:

        Returns:
            t0_frame: pointcloud in shape [N, features]
        """

        raise NotImplementedError()

    def get_pointcloud_t1(self, index: int) -> Union[np.ndarray, None]:
        r"""
        Optional method.

        For each dataset should be separetly written. Returns pointcloud frame (N, num_features) from time *t+1*.

        Args:
            index: Indexing the frame in the dataset

        Returns:
            t1_frame: pointcloud in shape [N, features]
        """
        return None

    def get_pose_transform(self, index: int) -> Union[np.ndarray, None]:
        """
        For each dataset should be separetly written. Returns transforamtion from t0 to t1
        Returns:
            t0_to_t1: in shape [4, 4]
        """
        return None

    def get_global_transform(self, index: int) -> Union[np.ndarray, None]:
        """
        Optional. For each dataset should be separetly written. Returns transforamtion from t0 to
        global coordinates system.
        """
        return None

    def get_flow(self, index: int) -> Union[np.ndarray, None]:
        """
        Optional. For each dataset should be separetly written. Returns gt flow in shape [N, channels].
        """
        return None

    def get_classes(self, index: int) -> Union[np.ndarray, None]:
        """
        Optional. For each dataset should be separetly written. Returns gt classes in shape [N, 1].
        """
        return None

    def _subsample_points(self, frame):
        if frame.shape[0] > self._n_points:
            indexes_previous_frame = np.linspace(0, frame.shape[0] - 1, num=self._n_points).astype(int)
            frame = frame[indexes_previous_frame, :]
        return frame, indexes_previous_frame


if __name__ == "__main__":
    from datasets.waymo.waymodatamodule import WaymoDataModule

    # COMPUTE NUM POINTS FOR MOVING DYNAMIC THRESHOLDS

    import numpy as np
    from tqdm import tqdm

    grid_cell_size = 0.109375
    dataset_path = "../../data/waymo/"

    data_module = WaymoDataModule(
        dataset_directory=dataset_path,
        grid_cell_size=grid_cell_size,
        x_min=-35,
        x_max=35,
        y_min=-35,
        y_max=35,
        z_min=0.25,
        z_max=10,
        batch_size=1,
        has_test=False,
        num_workers=0,
        n_pillars_x=640,
        n_pillars_y=640,
        n_points=None,
        apply_pillarization=True)

    data_module.setup()
    train_dl = data_module.train_dataloader()

    SUM = np.array([0]).astype('Q')
    SUM_min = np.array([0]).astype('Q')

    for x, flow, T_gt in tqdm(train_dl):
        # Create pcl from features vector
        num_points = x[0][0].shape[1]

        if num_points < 10:
            SUM_min += 1

        SUM += num_points

    print(f"Num points in train dataset : {SUM}")
    print(f"Num samples in train dataset : {len(train_dl)}")
    print(f"Num samples with points below 10 : {len(SUM_min)}")
