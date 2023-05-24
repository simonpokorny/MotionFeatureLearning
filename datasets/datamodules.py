from typing import Union

from datasets import WaymoDataModule, KittiDataModule, KittiSceneFlowDataModule, NuScenesDataModule


def get_datamodule(cfg: dict, data_path: Union[str, None], name=str):
    data_cfg = cfg["data"][name]
    grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["n_pillars_x"]

    if name == 'waymo':
        dataset_path = data_path if data_path is not None else \
            "/Users/simonpokorny/Developer/motion_learning/data/waymoflow"
        data_module = WaymoDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    elif name == 'rawkitti':
        dataset_path = data_path if data_path is not None else \
            "/Users/simonpokorny/Developer/motion_learning/data/rawkitti/"
        data_module = KittiDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    elif name == "kittisf":
        dataset_path = data_path if data_path is not None else \
            "/Users/simonpokorny/Developer/motion_learning/data/kittisf/"
        data_module = KittiSceneFlowDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size,
                                               **data_cfg)
    elif name == "nuscenes":
        dataset_path = data_path if data_path is not None else \
            "/Users/simonpokorny/Developer/motion_learning/data/nuscenes/"
        data_module = NuScenesDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)

    else:
        raise ValueError('Dataset {} not available yet'.format(name))
    return data_module
