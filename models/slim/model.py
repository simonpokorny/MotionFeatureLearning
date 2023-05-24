import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from models.slim.decoder.slimDecoder import SlimDecoder
from models.slim.encoder import Encoder
from .backbone import RaftBackbone


class SLIM(pl.LightningModule):
    def __init__(self,
                 n_pillars_x=640,
                 n_pillars_y=640,
                 point_features=3,
                 iters=6,
                 corr_levels=4,
                 corr_radius=3,
                 feature_downsampling_factor=8,
                 use_moving_average_thresholds=False):
        """
        Args:
            config (dict): Config is based on configs from configs/slim.yaml
            dataset (str): type of the dataset
        """
        super(SLIM, self).__init__()
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams

        self.n_features = point_features + 3
        self.n_pillars_x = n_pillars_x
        self.n_pillars_y = n_pillars_y

        self._encoder = Encoder(n_pillars_x=640, n_pillars_y=640, n_features=3)
        self._encoder.apply(init_weights)

        # Raft init weights is done internally.
        self._backbone = RaftBackbone(iters=iters,
                                      corr_levels=corr_levels,
                                      corr_radius=corr_radius,
                                      feature_downsampling_factor=feature_downsampling_factor)

        # self._moving_dynamicness_threshold = MovingAverageThreshold(**config["moving_threshold"][dataset])

        # no learnable params in decoder
        self._decoder = SlimDecoder(use_moving_average_thresholds=use_moving_average_thresholds)

    def forward(self, previous_batch, current_batch, transforms_matrice):
        """
        The usual forward pass function of a torch module
        :param previous_batch:
            previous_pcl[0] - Tensor of point clouds after pillarization in shape (BS, num points, num features)
            previous_pcl[1] - Grid indices in flattened mode in shape (BS, num points)
            previous_pcl[2] - Boolean mask tensor in shape (BS, num points) indicating if the point is valid.
        :param current_batch:
            previous_pcl[0] - Tensor of point clouds after pillarization in shape (BS, num points, num features)
            previous_pcl[1] - Grid indices in flattened mode in shape (BS, num points)
            previous_pcl[2] - Boolean mask tensor in shape (BS, num points) indicating if the point is valid.
        :param transforms_matrice: A tensor for the transformation matrix from previous to current point cloud in shape (BS, 4, 4).

        :return:
            - predictions_fw: Forward predictions
            - predictions_bw: Backward predictions
            - previous_batch_pc: Tensor of previous point cloud batch in shape (BS, num points, num features)
            - current_batch_pc: Tensor of current point cloud batch in shape (BS, num points, num features)

        """
        # 1. Do scene encoding of each point cloud to get the grid with pillar embeddings
        # P_T_C is transform from t0 to t1
        P_T_C = transforms_matrice.type(self.dtype)
        C_T_P = torch.linalg.inv(P_T_C).type(self.dtype)

        previous_batch_pc, previous_batch_grid, previous_batch_mask = previous_batch
        current_batch_pc, current_batch_grid, current_batch_mask = current_batch
        assert current_batch_pc.shape[2] == self.n_features and previous_batch_pc.shape[2] == self.n_features

        # Get point-wise voxel coods in shape [BS, num points, 2],
        current_voxel_coordinates = get_pointwise_pillar_coords(current_batch_grid, self.n_pillars_x, self.n_pillars_y)
        previous_voxel_coordinates = get_pointwise_pillar_coords(previous_batch_grid, self.n_pillars_x,
                                                                 self.n_pillars_y)

        # Create bool map of filled/non-filled pillars
        current_batch_pillar_mask = create_bev_occupancy_grid(current_voxel_coordinates, current_batch_mask,
                                                              self.n_pillars_x, self.n_pillars_y)
        previous_batch_pillar_mask = create_bev_occupancy_grid(previous_voxel_coordinates, previous_batch_mask,
                                                               self.n_pillars_x, self.n_pillars_y)

        # 1. Encoder
        pillar_embeddings = self._encoder(
            previous_batch_pc, previous_batch_mask, previous_batch_grid,
            current_batch_pc, current_batch_mask, current_batch_grid)

        # 2. RAFT Encoder with motion flow backbone
        # Output for forward pass and backward pass
        # Each of the output is a list of num_iters x [1, 9, n_pillars_x, n_pillars_x]
        # logits, static_flow, dynamic_flow, weights are concatinate in channels in shapes [4, 2, 2, 1]
        outputs_fw, outputs_bw = self._backbone(pillar_embeddings)

        # 3. SLIM Decoder
        predictions_fw = []
        predictions_bw = []
        for it, (raft_output_0_1, raft_output_1_0) in enumerate(zip(outputs_fw, outputs_bw)):
            prediction_fw = self._decoder_fw(
                network_output=raft_output_0_1,
                dynamicness_threshold=self._moving_dynamicness_threshold.value().to(self.device),
                pc=previous_batch_pc,
                pointwise_voxel_coordinates_fs=previous_voxel_coordinates,
                pointwise_valid_mask=previous_batch_mask,
                filled_pillar_mask=previous_batch_pillar_mask.type(torch.bool),
                odom=P_T_C,
                inv_odom=C_T_P,
                it=it)

            prediction_bw = self._decoder_bw(
                network_output=raft_output_1_0,
                dynamicness_threshold=self._moving_dynamicness_threshold.value().to(self.device),
                pc=current_batch_pc,
                pointwise_voxel_coordinates_fs=current_voxel_coordinates,
                pointwise_valid_mask=current_batch_mask,
                filled_pillar_mask=current_batch_pillar_mask.type(torch.bool),
                odom=C_T_P,
                inv_odom=P_T_C,
                it=it)

            predictions_fw.append(prediction_fw)
            predictions_bw.append(prediction_bw)
        return predictions_fw, predictions_bw, previous_batch_pc, current_batch_pc


def get_pointwise_pillar_coords(batch_grid, n_pillars_x=640, n_pillars_y=640):
    """
    A method that takes a batch of grid indices in flatten mode and returns the corresponding 2D grid
    coordinates. The method calculates the x and y indices of the grid points using the number of
    pillars in the x and y dimensions, respectively, and then concatenates them along the second dimension.

    :param:
        - batch_grid: torch tensor of shape (BS, num_points)
        - n_pillars_x: size in x in bev
        - n_pillars_y: size in y in bev
    :return:
        - grid: torch tensor in shape (BS, num_points, 2)
    """
    assert batch_grid.ndim == 2

    grid = torch.cat(((batch_grid // n_pillars_x).unsqueeze(1),
                      (batch_grid % n_pillars_y).unsqueeze(1)), dim=1)
    return grid.transpose(-1, 1)


def create_bev_occupancy_grid(pointwise_pillar_coords, batch_mask, n_pillars_x=640, n_pillars_y=640):
    """
    A method that takes a batch of grid indices and masks and returns a tensor with a 1 in the location
    of each grid point and a 0 elsewhere. The method creates a tensor of zeros with the same shape as
    the voxel grid, and then sets the locations corresponding to the grid points in the batch to 1.
    """
    assert pointwise_pillar_coords.ndim == 3
    assert batch_mask.ndim == 2

    bs = pointwise_pillar_coords.shape[0]
    # pillar mask
    pillar_mask = torch.zeros((bs, 1, n_pillars_x, n_pillars_y), device=pointwise_pillar_coords.device)

    x = pointwise_pillar_coords[batch_mask][..., 0]
    y = pointwise_pillar_coords[batch_mask][..., 1]
    pillar_mask[:, :, x, y] = 1
    return pillar_mask


def init_weights(m) -> None:
    """
    Apply the weight initialization to a single layer.
    Use this with your_module.apply(init_weights).
    The single layer is a module that has the weights parameter. This does not yield for all modules.
    :param m: the layer to apply the init to
    :return: None
    """
    if type(m) in [torch.nn.Linear, torch.nn.Conv2d]:
        # Note: There is also xavier_normal_ but the paper does not state which one they used.
        torch.nn.init.xavier_uniform_(m.weight)


if __name__ == "__main__":
    import sys
    sys.path.append("../../")


    DATASET = "nuscenes"
    trained_on = "waymo"
    assert DATASET in ["waymo", "rawkitti", "kittisf", "nuscenes"]
    from datasets import get_datamodule
    from configs.utils import load_config

    model = SLIM()

    cfg = load_config(path="configs/datasets/rawkitti.yaml")
    data_module = get_datamodule(cfg=cfg, data_path=None)

    loggers = TensorBoardLogger(save_dir="", log_graph=True, version=0)
    trainer = pl.Trainer(fast_dev_run=True, num_sanity_val_steps=0,
                         logger=loggers)  # , callbacks=callbacks)  # Add Trainer hparams if desired

    # trainer.fit(model, data_module)
    trainer.test(model, data_module)
    # trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print("Done")
