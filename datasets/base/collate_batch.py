import numpy as np
from torch.utils.data._utils.collate import default_collate

__all__ = [
    "custom_collate_batch"
]

def _pad_batch(batch):
    # Get the number of points in the largest point cloud
    true_number_of_points = [e[0].shape[0] for e in batch]
    max_points_prev = np.max(true_number_of_points)

    # We need a mask of all the points that actually exist
    zeros = np.zeros((len(batch), max_points_prev), dtype=bool)
    # Mark all points that ARE NOT padded
    for i, n in enumerate(true_number_of_points):
        zeros[i, :n] = 1

    # resize all tensors to the max points size
    # Use np.pad to perform this action. Do not pad the second dimension and pad the first dimension AFTER only
    return [
        [np.pad(entry[0], ((0, max_points_prev - entry[0].shape[0]), (0, 0))),
         np.pad(entry[1], (0, max_points_prev - entry[1].shape[0])) if entry[1] is not None
         else np.empty(shape=(max_points_prev,)),  # set empty array, if there is None entry in the tuple
         # (for baseline, we do not have grid indices, therefore this tuple entry is None)
         zeros[i]] for i, entry in enumerate(batch)
    ]


def _pad_targets(batch):
    true_number_of_points = [e.shape[0] for e in batch]
    max_points = np.max(true_number_of_points)
    return [
        np.pad(entry, ((0, max_points - entry.shape[0]), (0, 0)))
        for entry in batch
    ]


def custom_collate_batch(batch):
    """
    This version of the collate function create the batch necessary for the input to the network.

    Take the list of entries and batch them together.
        This means a batch of the previous images and a batch of the current images and a batch of flows.
    Because point clouds have different number of points the batching needs the points clouds with less points
        being zero padded.
    Note that this requires to filter out the zero padded points later on.

    :param batch: batch_size long list of ((prev, cur), flows) pointcloud tuples with flows.
        prev and cur are tuples of (point_cloud, grid_indices, mask)
         point clouds are (N_points, features) with different N_points each
    :return: ((batch_prev, batch_cur), batch_flows)
    """
    # Build numpy array with data

    # Only convert the points clouds from numpy arrays to tensors
    # entry[0, 0] is the previous (point_cloud, grid_index) entry
    batch_previous = [entry[0][0] for entry in batch]
    batch_previous = _pad_batch(batch_previous)

    batch_current = [entry[0][1] for entry in batch]
    batch_current = _pad_batch(batch_current)

    # For the targets we can only transform each entry to a tensor and not stack them
    batch_targets_flow = [entry[1][0] for entry in batch]
    batch_targets_flow = _pad_targets(batch_targets_flow)

    batch_targets_classes = [entry[1][1] for entry in batch]
    batch_targets_classes = _pad_targets(batch_targets_classes)


    batch_transform = [entry[2] for entry in batch]
    batch_transform = _pad_targets(batch_transform)

    # Call the default collate to stack everything
    batch_previous = default_collate(batch_previous)
    batch_current = default_collate(batch_current)
    batch_targets_flow = default_collate(batch_targets_flow)
    batch_targets_classes = default_collate(batch_targets_classes)

    batch_transform = default_collate(batch_transform)

    # Return a tensor that consists of
    # the data batches consist of batches of tensors
    #   1. (batch_size, max_n_points, features) the point cloud batch
    #   2. (batch_size, max_n_points) the 1D grid_indices encoding to map to
    #   3. (batch_size, max_n_points) the 0-1 encoding if the element is padded
    #   4. (batch_size, 4, 4) transformation matrix from frame to global coords
    # Batch previous for the previous frame
    # Batch current for the current frame

    # The targets consist of
    #   (batch_size, max_n_points, target_features). should by 4D x,y,z flow and class id

    return (batch_previous, batch_current), (batch_targets_flow, batch_targets_classes), batch_transform
