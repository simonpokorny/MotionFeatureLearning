from datasets.waymo.waymodataset import WaymoDataset
from datasets.base import BaseDataModule


class WaymoDataModule(BaseDataModule):
    """
    Data module to prepare and load the waymo dataset.
    Using a data module streamlines the data loading and preprocessing process.
    """
    def __init__(self, dataset_directory,
                 grid_cell_size, x_min, x_max, y_min, y_max, z_min, z_max, n_pillars_x, n_pillars_y,
                 batch_size: int = 32,
                 has_test=False,
                 num_workers=1,
                 n_points=None,
                 apply_pillarization=True,
                 shuffle_train=True,
                 point_features=3):
        super().__init__(dataset=WaymoDataset,
                         dataset_directory=dataset_directory,
                         grid_cell_size=grid_cell_size,
                         x_min=x_min,
                         x_max=x_max,
                         y_min=y_min,
                         y_max=y_max,
                         z_min=z_min,
                         z_max=z_max,
                         n_pillars_x=n_pillars_x,
                         n_pillars_y=n_pillars_y,
                         batch_size=batch_size,
                         has_test=has_test,
                         num_workers=num_workers,
                         n_points=n_points,
                         apply_pillarization=apply_pillarization,
                         shuffle_train=shuffle_train,
                         point_features=point_features)