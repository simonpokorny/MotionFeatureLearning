import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_pillars_x=640, n_pillars_y=640, n_features=3):
        super(Encoder, self).__init__()

        self._point_feature_net = PointFeatureNet(in_features=n_features, out_features=64)
        self._point_scatter_net = PillarFeatureNetScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y)

    def forward(self,
                previous_pcl, previous_mask, previous_grid,
                current_pcl, current_mask, current_grid):
        # Pass the whole batch of point clouds to get the embedding for each point in the cloud
        # Input pc is (batch_size, max_n_points, features_in)
        previous_batch_pc_embedding = self._transform_point_cloud_to_embeddings(previous_pcl, previous_mask)
        # previous_batch_pc_embedding = [n_batch, N, 64]
        # Output pc is (batch_size, max_n_points, embedding_features)
        current_batch_pc_embedding = self._transform_point_cloud_to_embeddings(current_pcl,
                                                                               current_mask).type(self.dtype)

        # Now we need to scatter the points into their 2D matrix
        # No learnable params in this part
        previous_pillar_embeddings = self._pillar_feature_net(previous_batch_pc_embedding, previous_grid)
        current_pillar_embeddings = self._pillar_feature_net(current_batch_pc_embedding, current_grid)
        # pillar_embeddings = (batch_size, 64, 640, 640)

        # Concatenate the previous and current batches along a new dimension.
        # This allows to have twice the amount of entries in the forward pass
        # of the encoder which is good for batch norm.
        pillar_embeddings = torch.stack((previous_pillar_embeddings, current_pillar_embeddings), dim=1)
        # This is now (batch_size, 2, 64, 640, 640) large
        pillar_embeddings = pillar_embeddings.flatten(0, 1)
        # Flatten into (batch_size * 2, 64, 512, 512) for encoder forward pass.

        pass

    def _transform_point_cloud_to_embeddings(self, pc, mask):
        """
         A method that takes a point cloud and a mask and returns the corresponding embeddings.
         The method flattens the point cloud and mask, applies the point feature network,
         and then reshapes the embeddings to their original dimensions.
        """
        pc_flattened = pc.flatten(0, 1)
        mask_flattened = mask.flatten(0, 1)
        # Init the result tensor for our data. This is necessary because the point net
        # has a batch norm and this needs to ignore the masked points
        batch_pc_embedding = torch.zeros((pc_flattened.size(0), 64), device=pc.device, dtype=pc.dtype)
        # Flatten the first two dimensions to get the points as batch dimension
        batch_pc_embedding[mask_flattened] = self._point_feature_net(pc_flattened[mask_flattened])
        # This allows backprop towards the MLP: Checked with backward hooks. Gradient is present.
        # Output is (batch_size * points, embedding_features)
        # Retransform into batch dimension (batch_size, max_points, embedding_features)
        batch_pc_embedding = batch_pc_embedding.unflatten(0, (pc.size(0), pc.size(1)))
        # 241.307 MiB    234
        return batch_pc_embedding


class PillarFeatureNetScatter(torch.nn.Module):
    """
    Transform the raw point cloud data of shape (n_points, 3) into a representation of shape (n_points, 6).
    Each point consists of 6 features: (x_c, y_c, z_c, x_delta, y_delta, z_delta, laser_feature1, laser_feature2).
    x_c, y_c, z_c: Center of the pillar to which the point belongs.
    x_delta, y_delta, z_delta: Offset from the pillar center to the point.

    References
    ----------
    .. [PointPillars] Alex H. Lang and Sourabh Vora and  Holger Caesar and Lubing Zhou and Jiong Yang and Oscar Beijbom
       PointPillars: Fast Encoders for Object Detection from Point Clouds
       https://arxiv.org/pdf/1812.05784.pdf
    """

    def __init__(self, n_pillars_x, n_pillars_y):
        super().__init__()
        self.n_pillars_x = n_pillars_x
        self.n_pillars_y = n_pillars_y

    def forward(self, x, indices):
        # pc input is (batch_size, N_points, 64) with 64 being the embedding dimension of each point
        # in indices we have (batch_size, N_points) which contains the index in the grid
        # We want to scatter into a n_pillars_x and n_pillars_y grid
        # Thus we should allocate a tensor of the desired shape (batch_size, n_pillars_x, n_pillars_y, 64)

        # The grid indices are (batch_size, max_points) long. But we need them as
        # (batch_size, max_points, feature_dims) to work. Features are in all necessary cases 64.
        # Expand does only create multiple views on the same datapoint and not allocate extra memory
        indices = indices.unsqueeze(-1).expand(-1, -1, 64)

        # Init the matrix to only zeros
        # Construct the desired tensor
        grid = torch.zeros((x.size(0), self.n_pillars_x * self.n_pillars_y, x.size(2)), device=x.device, dtype=x.dtype)
        # And now perform the infamous scatter_add_ that changes the grid in place
        # the source (x) and the indices matrix are now 2 dimensional with (batch_size, points)
        # The batch dimension stays the same. But the cells are looked up using the index
        # thus: grid[batch][index[batch][point]] += x[batch][point]
        grid.scatter_add_(1, indices, x)
        # Do a test if actually multiple
        # the later 2D convolutions want (batch, channels, H, W) of the image
        # thus make the grid (batch, channels, cells)
        grid = grid.permute((0, 2, 1))
        # and make the cells 2D again
        grid = grid.unflatten(2, (self.n_pillars_x, self.n_pillars_y))

        return grid


class PointFeatureNet(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PointFeatureNet, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.batch_norm = torch.nn.BatchNorm1d(out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Encode all points into their embeddings
        :param x: (n_points, in_features)
        :return: (n_points, out_features)
        """
        # linear transformation
        x = self.linear(x)  # 8 * 64 = 512 weights + 64 biases = 576 Params
        x = self.batch_norm(x)  # Small number of weights for affine transform per channel
        x = self.relu(x)
        return x
