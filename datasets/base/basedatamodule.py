from pathlib import Path
from typing import Optional, Union, List, Dict

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from datasets.base.collate_batch import custom_collate_batch
from geometry import Pillarization, DropPoints


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, dataset,
                 dataset_directory: str,
                 # These parameters are specific to the dataset
                 grid_cell_size: float = 0.109375,
                 x_min: float = -35.,
                 x_max: float = 35.,
                 y_min: float = -35.,
                 y_max: float = 35,
                 z_min: float = -10,
                 z_max: float = 10,
                 n_pillars_x: int = 640,
                 n_pillars_y: int = 640,
                 batch_size: int = 32,
                 has_test=False,
                 num_workers=1,
                 n_points=None,
                 apply_pillarization=True,
                 shuffle_train=True,
                 point_features=3,
                 ):
        """
        This class defines a PyTorch Lightning DataModule that loads and preprocesses data from a specified dataset
        directory using the provided arguments. For correct using, it is neccesery to implement a class dataset for
        specific data based on BasaDataset.

        Args:
            dataset (class): A PyTorch Dataset class that inherits from `BaseDataset`.
            dataset_directory (str): Path to the directory containing the train, val and test dataset.
            grid_cell_size (float): The size of the grid cell for the pillarization step.
            x_min (float): The minimum x-coordinate which will be used.
            x_max (float): The maximum x-coordinate which will be used.
            y_min (float): The minimum y-coordinate which will be used.
            y_max (float): The maximum y-coordinate which will be used.
            z_min (float): The minimum z-coordinate which will be used.
            z_max (float): The maximum z-coordinate which will be used.
            n_pillars_x (int): The number of pillars along the x-axis for the pillarization step.
            batch_size (int, optional): The batch size to use for the DataLoader (default: 32).
            has_test (bool, optional): Whether the dataset has a test set (default: False).
            num_workers (int, optional): The number of worker processes to use for loading the data (default: 1).
            n_points (int, optional): The number of points to sample from each point cloud (default: None, which means to use all points).
            apply_pillarization (bool, optional): Whether to apply pillarization to the point clouds (default: True).
            shuffle_train (bool, optional): Whether to shuffle the training set (default: True).

        """
        super(BaseDataModule, self).__init__()
        self._dataset = dataset
        self._dataset_directory = Path(dataset_directory)
        self._batch_size = batch_size
        self._point_features = point_features
        self._train_ = None
        self._val_ = None
        self._test_ = None
        self._shuffle_train = shuffle_train
        self.apply_pillarization = apply_pillarization

        self._pillarization_transform = Pillarization(grid_cell_size=grid_cell_size, x_min=x_min,
                                                      y_min=y_min, z_min=z_min, z_max=z_max,
                                                      n_pillars_x=n_pillars_x)

        # This returns a function that removes points that should not be included in the pillarization.
        # It also removes the labels if given.
        self._drop_points_function = DropPoints(x_min=x_min,
                                                x_max=x_max, y_min=y_min, y_max=y_max,
                                                z_min=z_min, z_max=z_max)
        self._has_test = has_test
        self._num_workers = num_workers

        # add zeros to pad the batch to same shape
        self._collate_fn = custom_collate_batch
        self._n_points = n_points

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup of the datasets. Called on every GPU in distributed training.
        Do splits and build model internals here.
        :param stage: either 'fit', 'validate', 'test' or 'predict'
        :return: None
        """

        self._train_ = self._dataset(self._dataset_directory.joinpath("train"),
                                     point_cloud_transform=self._pillarization_transform,
                                     drop_invalid_point_function=self._drop_points_function,
                                     n_points=self._n_points, apply_pillarization=self.apply_pillarization,
                                     point_features=self._point_features)
        self._val_ = self._dataset(self._dataset_directory.joinpath("valid"),
                                   point_cloud_transform=self._pillarization_transform,
                                   drop_invalid_point_function=self._drop_points_function,
                                   apply_pillarization=self.apply_pillarization,
                                   n_points=self._n_points, point_features=self._point_features)
        if self._has_test:
            self._test_ = self._dataset(self._dataset_directory.joinpath("test"),
                                        point_cloud_transform=self._pillarization_transform,
                                        drop_invalid_point_function=self._drop_points_function,
                                        apply_pillarization=self.apply_pillarization,
                                        point_features=self._point_features)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for training
        :return: the dataloader to use
        """
        return DataLoader(self._train_, self._batch_size, num_workers=self._num_workers,
                          shuffle=self._shuffle_train,
                          collate_fn=self._collate_fn)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for validation
        :return: the dataloader to use
        """
        return DataLoader(self._val_, self._batch_size, shuffle=False, num_workers=self._num_workers,
                          collate_fn=self._collate_fn)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for testing
        :return: the dataloader to use
        """
        if not self._has_test:
            raise RuntimeError("No test dataset specified. Maybe set has_test=True in DataModule init.")
        return DataLoader(self._test_, self._batch_size, shuffle=False, num_workers=self._num_workers,
                          collate_fn=self._collate_fn)
