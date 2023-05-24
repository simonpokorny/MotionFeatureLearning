from datasets.base import BaseDataModule
from datasets.kitti.kittidataset import KittiRawDataset


class KittiDataModule(BaseDataModule):
    def __init__(self,
                 dataset_directory: str = "~/data/rawkitti/prepared/",
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
                 batch_size: int = 1,
                 has_test=False,
                 num_workers=1,
                 n_points=None,
                 apply_pillarization=True,
                 shuffle_train=True,
                 point_features=7):

        super().__init__(dataset=KittiRawDataset,
                         dataset_directory=dataset_directory,
                         grid_cell_size=grid_cell_size,
                         x_min=x_min,
                         x_max=x_max,
                         y_min=y_min,
                         y_max=y_max,
                         z_min=z_min,
                         z_max=z_max,
                         n_pillars_x=n_pillars_x,
                         batch_size=batch_size,
                         has_test=has_test,
                         num_workers=num_workers,
                         n_points=n_points,
                         apply_pillarization=apply_pillarization,
                         shuffle_train=shuffle_train,
                         point_features=point_features)
