import torch
from pytorch3d.ops.points_alignment import iterative_closest_point


def kabsch_algorithm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Calculates the rigid transformation that aligns two sets of points.

    Args:
        A (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the first set of points.
        B (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the second set of points.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 4, 4) containing the rigid transformation matrix that aligns A to B.
    """
    assert A.ndim == 3 and B.ndim == 3 and A.shape == B.shape

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
    t = b_mean[:, None].transpose(1, 2) - (R @ a_mean[:, None].transpose(1, 2))

    T = torch.cat((R, t), dim=2)
    T = torch.cat((T, torch.ones((1, 1, 4), device=A.device)), dim=1)
    return T


def weighted_kabsch_algorithm(A: torch.Tensor,
                              B: torch.Tensor,
                              weights: torch.Tensor,
                              use_epsilon_on_weights: bool = False)  -> torch.Tensor:
    """
    Calculates the weighted rigid transformation that aligns two sets of points.

    Args:
        A (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the first set of points.
        B (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the second set of points.
        weights (torch.Tensor): A tensor of shaep (batch_size, num_points) containing weights.
        use_epsilon_on_weights (bool): A condition if to use eps for weights.

    Returns:
        T (torch.Tensor): A tensor of shape (batch_size, 4, 4) containing the rigid transformation matrix that aligns A to B.
    """
    assert (weights >= 0.0).all(), "Negative weights found"
    assert A.ndim == 3 and B.ndim == 3 and A.shape == B.shape

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
    T = torch.cat((T, torch.tensor([[[0, 0, 0, 1]]], device=A.device)), dim=1)
    return T


def ICP(A: torch.Tensor, B: torch.Tensor, verbose=False):
    assert A.shape[2] == B.shape[2] and A.shape[0] == B.shape[0]
    assert A.device == B.device

    raise NotImplementedError

    device = A.device

    out = iterative_closest_point(A, B, allow_reflection=True, verbose=verbose)
    R = out.RTs.R[0]
    T = out.RTs.T[0]
    transformed_pts = out.Xt[0]

    T_mat = torch.eye(4)
    T_mat[:3,:3] = R
    T_mat[:3,-1] = T

    return T_mat, transformed_pts
