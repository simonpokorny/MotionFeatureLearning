import torch

#from visualization.plot import save_trans_pcl, save_tensor_per_channel


class StaticAggregatedFlow(torch.nn.Module):
    def __init__(self, use_eps_for_weighted_pc_alignment=False):
        """
        Initializes a StaticAggregatedFlow module.

        Args:
            use_eps_for_weighted_pc_alignment (bool): If True, adds a small epsilon to the denominator
                of the weighted point cloud alignment computation to avoid division by zero errors.
                Defaults to False.
        """
        super().__init__()
        self.use_eps_for_weighted_pc_alignment = use_eps_for_weighted_pc_alignment

    def forward(
            self,
            static_flow,
            staticness,
            pc,
            pointwise_voxel_coordinates_fs,
            pointwise_valid_mask,
            voxel_center_metric_coordinates,
    ):
        """
        Computes the forward pass of the StaticAggregatedFlow module.

        Args:
            static_flow (torch.Tensor): A tensor of shape (batch_size, 2, H, W) representing the
                static flow field.
            staticness (torch.Tensor): A tensor of shape (batch_size, H, W) representing the
                staticness field.
            pc (torch.Tensor): A tensor of shape (batch_size, N, 3) representing the input point
                clouds.
            pointwise_voxel_coordinates_fs (torch.Tensor): A tensor of shape (batch_size, N, 2)
                representing the 2D voxel coordinates of each point in the static flow field.
            pointwise_valid_mask (torch.Tensor): A tensor of shape (batch_size, N) containing boolean
                values indicating whether each point in the input point clouds is valid.
            voxel_center_metric_coordinates (torch.Tensor): A tensor of shape (H, W, 3) representing
                the 3D metric coordinates of the centers of the voxels in the static flow field.

        Returns:
            A tuple containing:
            - static_aggr_flow (torch.Tensor): A tensor of shape (batch_size, N, 2) representing the
              static aggregated flow field.
            - trafo (torch.Tensor): A tensor of shape (batch_size, 4, 4) representing the transformation
              matrix used to align the input point clouds to the static flow field.
            - not_enough_points (torch.Tensor): A tensor of shape (batch_size,) containing boolean values
              indicating whether there were not enough valid points in the input point clouds to perform
              the weighted point cloud alignment.
        """
        assert len(static_flow.shape) == 4
        assert static_flow.shape[1] == 2
        assert (pointwise_voxel_coordinates_fs >= 0).all()

        # To static flow is also add a third coord z with value 0
        static_3d_flow_grid = torch.cat([static_flow, torch.zeros_like(static_flow[:, :1])], dim=1)
        # static_grid_flow_3d [1,3,640,640]
        # pointwise_voxel_coordinates_fs [1, num points in pcl, 2]
        # representing the coordinates of each point in the BEV grid
        bs, ch, h, w = static_3d_flow_grid.shape

        ### GET POINTWISE TENSORS ###
        # getting x,y coordinates where the individual point belongs in bev image
        x, y = pointwise_voxel_coordinates_fs[:, :, 0], pointwise_voxel_coordinates_fs[:, :, 1]
        pointwise_flow = static_3d_flow_grid[torch.arange(bs)[:, None], :, x, y]
        # Probability of being static for each point in pcl
        pointwise_staticness = staticness[torch.arange(bs)[:, None], :, x, y]
        # Masking unfilled pillars with zeros
        pointwise_staticness = torch.where(
            pointwise_valid_mask.unsqueeze(-1),
            pointwise_staticness,
            torch.zeros_like(pointwise_staticness))


        # change order from [BS, CH, H, W] -> [BS, H, W, CH]
        static_3d_flow_grid = static_3d_flow_grid.permute(0, 2, 3, 1)

        # Create meshgrid (x,y,z)
        pc0_grid = torch.cat([voxel_center_metric_coordinates, torch.zeros_like(
            voxel_center_metric_coordinates[..., :1])], dim=-1).to(voxel_center_metric_coordinates.device)
        #save_tensor_per_channel(pc0_grid.permute(2, 0, 1)[None, :], "", labels=["X", "Y", "Z"], name="pc0_grid",show=True)

        assert pc0_grid.shape == static_3d_flow_grid.shape[1:]
        grid_shape = static_3d_flow_grid.shape
        batched_pc0_grid = pc0_grid.expand(grid_shape)
        # batched_pc0_grid is in shape [BS, 640, 640, 3]

        # pc_xyz is in shape [BS, Num Points, 3]
        pc_xyz = pc[:, :, :3] + pc[:, :, 3:6]

        # Computing of weighted kabsch algorithm
        transformation = find_weighted_rigid_alignment(
                                            A=pc_xyz,
                                            B=(pc_xyz + pointwise_flow),
                                            weights=pointwise_staticness[:, :, 0],
                                            use_epsilon_on_weights=self.use_eps_for_weighted_pc_alignment)
        transformation = transformation.float()

        homogeneous_batched_pc0_grid = torch.cat([batched_pc0_grid,
                                                  torch.ones_like(batched_pc0_grid[..., 0][..., None])], dim=-1)
        # Constructing of static_aggr_flow
        # print(transformation.dtype, pc.dtype)
        static_aggr_flow = torch.einsum(
            "bij,bhwj->bhwi",
            transformation.double() - torch.eye(4, device=pc.device, dtype=torch.double),
            homogeneous_batched_pc0_grid)[..., 0:2]

        # Change static aggr flow to default [BS CH H W]
        static_aggr_flow = static_aggr_flow.permute(0, 3, 1, 2).float()
        return static_aggr_flow, transformation


def find_rigid_alignment(A, B):
    """
    Calculates the rigid transformation that aligns two sets of points.

    Args:
        A (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the first set of points.
        B (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the second set of points.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 4, 4) containing the rigid transformation matrix that aligns A to B.
    """

    a_mean = A.mean(axis=1)
    b_mean = B.mean(axis=1)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.transpose(1, 2) @ B_c
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V @ U.transpose(1, 2)
    # Translation vector
    t = b_mean[:, None].transpose(1, 2) -(R @ a_mean[:, None].transpose(1, 2))

    T = torch.cat((R, t), dim=2)
    T = torch.cat((T, torch.ones((1, 1, 4), device=A.device)), dim=1)
    return T


def find_weighted_rigid_alignment(A, B, weights, use_epsilon_on_weights=False):
    """
    Calculates the weighted rigid transformation that aligns two sets of points.

    Args:
        A (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the first set of points.
        B (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the second set of points.
        weights (torch.Tensor): A tensor of shaep (batch_size, num_points) containing weights.
        use_epsilon_on_weights (bool): A condition if to use eps for weights.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 4, 4) containing the rigid transformation matrix that aligns A to B.
    """
    assert (weights >= 0.0).all(), "Negative weights found"
    if use_epsilon_on_weights:
        weights += torch.finfo(weights.dtype).eps
        count_nonzero_weighted_points = torch.sum(weights > 0.0, dim=-1)
        not_enough_points = count_nonzero_weighted_points < 3
    else:
        # Add eps if not enough points with weight over zero
        count_nonzero_weighted_points = torch.sum(weights > 0.0, dim=-1)
        not_enough_points = count_nonzero_weighted_points < 3
        eps = not_enough_points.float() * torch.finfo(weights.dtype).eps
        weights += eps.unsqueeze(-1)
    assert not not_enough_points, f"pcl0 shape {A.shape}, pcl1 shape {B.shape}, points {count_nonzero_weighted_points}"

    weights = weights.unsqueeze(-1)
    sum_weights = torch.sum(weights, dim=1)

    A_weighted = A * weights
    B_weighted = B * weights

    a_mean = A_weighted.sum(axis=1) / sum_weights.unsqueeze(-1)
    b_mean = B_weighted.sum(axis=1) / sum_weights.unsqueeze(-1)

    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = ((A_c * weights).transpose(1, 2) @ B_c) / sum_weights
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V @ U.transpose(1, 2)
    # Translation vector
    t = b_mean.transpose(1, 2) - (R @ a_mean.transpose(1, 2))

    T = torch.cat((R, t), dim=2)
    T = torch.cat((T, torch.tensor([[[0,0,0,1]]], device=A.device)), dim=1)
    return T

if __name__ == "__main__":

    ### DATAMODULE ###
    from datasets.kitti import KittiDataModule
    # dataset_path = "/Users/simonpokorny/mnt/data/waymo/raw/processed/training"
    grid_cell_size = 0.109375

    dataset_path =  "../../../data/rawkitti"
    data_module = KittiDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size,
                                  x_min=-35,
                                  x_max=35,
                                  y_min=-35,
                                  y_max=35,
                                  z_min=-10,
                                  z_max=10,
                                  batch_size=1,
                                  has_test=False,
                                  num_workers=0,
                                  n_pillars_x=640,
                                  n_points=None, apply_pillarization=True)

    data_module.setup()
    train_dl = data_module.train_dataloader()

    def _mask_z_point_cloud(pcl, z_min=-1.4):
        mask = pcl[:, :, 2] > z_min
        return mask

    for x, _, T_gt in train_dl:

        # Create pcl from features vector
        pcl_0 = torch.tensor((x[0][0][:, :, :3] + x[0][0][:, :, 3:6]))
        pcl_1 = torch.tensor((x[1][0][:, :, :3] + x[1][0][:, :, 3:6]))

        # Mask it in z coordinates
        mask_0 = _mask_z_point_cloud(pcl_0)
        mask_1 = _mask_z_point_cloud(pcl_1)

        # Masking
        pcl_0 = pcl_0[mask_0][None, :]
        pcl_1 = pcl_1[mask_1][None, :]

        num_points_0 = pcl_0.shape[1]

        # Create flow from odometry
        only_flow = (T_gt @ torch.cat((pcl_0, torch.ones_like(pcl_0)[:, :, 0:1]),
                                 dim=2).transpose(1, 2)).transpose(1, 2)[:, :, :3]

        # Create flow first half, second half pcl 0
        first_flow_then_pcl_0 = torch.cat((only_flow[:, :int(num_points_0/2)], pcl_0[:, int(num_points_0/2):]), dim=1)


        # Weights
        weights_only = torch.ones((1, num_points_0))
        weights_first_half = torch.zeros((1, num_points_0))
        weights_first_half[:, :int(num_points_0/2)] = 1
        weights_second_half = torch.zeros((1, num_points_0))
        weights_second_half[:, int(num_points_0/2):] = 1
        weights_in_the_middle = torch.zeros((1, num_points_0))
        weights_in_the_middle[:, int(num_points_0/4):int(num_points_0*3/4)] = 1

        #T, _ = kabsch(pcl_0.double(), flow.double(), weights.double())

        '''
        T = find_rigid_alignment(pcl_0, only_flow)
        save_trans_pcl(P_T_C=T, P=pcl_0, C=pcl_1, path=" ", name="all points", show=True)

        T = find_rigid_alignment(pcl_0, first_flow_then_pcl_0)
        save_trans_pcl(P_T_C=T, P=pcl_0, C=pcl_1, path=" ", name="half of the flow", show=True)

        save_trans_pcl(P_T_C=torch.eye(4)[None, :].double(), P=pcl_0, C=pcl_1, path=" ", name="no tranforms", show=True)

        save_trans_pcl(P_T_C=torch.eye(4)[None, :].double(), P=pcl_0, C=pcl_1, path=" ", name="", show=True)
        # Check ground truth odometry that it is transformation from t0 to t1
        save_trans_pcl(P_T_C=torch.eye(4)[None, :].double(), P=only_flow, C=pcl_1, path=" ", name="", show=True)
        '''

        T = find_weighted_rigid_alignment(pcl_0, only_flow, weights_only)
        save_trans_pcl(P_T_C=T, P=pcl_0, C=pcl_1, path=" ", name="all weights", show=True)

        T = find_weighted_rigid_alignment(pcl_0, first_flow_then_pcl_0, weights_first_half)
        save_trans_pcl(P_T_C=T, P=pcl_0, C=pcl_1, path=" ", name="fitst half of weigths", show=True)

        T = find_weighted_rigid_alignment(pcl_0, only_flow, weights_first_half)
        save_trans_pcl(P_T_C=T, P=pcl_0, C=pcl_1, path=" ", name="half of the flow", show=True)

        T = find_weighted_rigid_alignment(pcl_0, first_flow_then_pcl_0, weights_second_half)
        save_trans_pcl(P_T_C=T, P=pcl_0, C=pcl_1, path=" ", name="half of the flow", show=True)

        T = find_weighted_rigid_alignment(pcl_0, first_flow_then_pcl_0, weights_in_the_middle)
        save_trans_pcl(P_T_C=T, P=pcl_0, C=pcl_1, path=" ", name="half of the flow", show=True)





