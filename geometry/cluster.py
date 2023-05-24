import numpy as np
import torch
from sklearn.cluster import DBSCAN

__all__ = ["DBSCAN_pytorch"]


def DBSCAN_pytorch(X: torch.Tensor, eps: float = 0.5, min_samples: int = 10):
    """
    Apply the DBSCAN clustering algorithm on a 3D tensor of point cloud data.

    Args:
        X (torch.Tensor): Input tensor of shape (BS, num_points, CH) representing the point cloud data.
        eps (float, optional): The maximum distance between two samples for them to be considered as in
         the same neighborhood. Defaults to 0.5.
        min_samples (int, optional): The minimum number of samples in a neighborhood for a point to be
         considered as a core point. Defaults to 10.

    Returns:
        torch.Tensor: Cluster labels for each point in the input tensor (BS, num_points).
    """
    assert type(X) == torch.Tensor and X.ndim == 3
    device = X.device
    BS, num_points, CH = X.shape
    X = X.detach().cpu().numpy()
    labels = np.zeros((BS, num_points))
    for idx, x in enumerate(X):
        labels[idx, :] = DBSCAN(eps=eps, min_samples=min_samples).fit(x).labels_
    labels = torch.from_numpy(labels).to(device)
    return labels


if __name__ == "__main__":
    x = torch.rand((2, 100, 3))
    labels = DBSCAN_pytorch(x)
