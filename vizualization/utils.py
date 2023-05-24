import numpy as np
import open3d as o3d

RED = [0.9, 0.1, 0.1]
BLUE = [0.1, 0.1, 0.7]


def create_o3d_pcl(pcl, color=None):
    """
    Creates an Open3D PointCloud object from a 2D numpy array of shape (n_points, 3).
    :param pcl: numpy array of shape (n_points, 3) containing the point cloud coordinates
    :param color: optional color for the point cloud. If provided, should be a list of 3 floats between 0 and 1
    :return: Open3D PointCloud object
    """
    assert pcl.ndim == 2 and pcl.shape[1] == 3
    frame = o3d.geometry.PointCloud()
    frame.points = o3d.utility.Vector3dVector(pcl)
    if color is not None:
        frame.paint_uniform_color(color)
    return frame


def show_flow(t0_frame, t1_frame, flow):
    """
    Visualizes the optical flow between two point clouds.
    :param t0_frame: numpy array of shape (3, n_points) containing the coordinates of the first point cloud
    :param t1_frame: numpy array of shape (3, n_points) containing the coordinates of the second point cloud
    :param flow: numpy array of shape (3, n_points) containing the optical flow vectors
    :return: None
    """
    assert t0_frame.shape[1] == 3 and type(t0_frame) == np.ndarray
    assert t1_frame.shape[1] == 3 and type(t1_frame) == np.ndarray
    assert t0_frame.shape == flow.shape and type(flow) == np.ndarray

    flow = np.concatenate([t0_frame, t0_frame + flow], axis=0)
    indices = np.arange(t0_frame.shape[0] * 2).reshape((2, -1)).T

    t0_frame_o3d = create_o3d_pcl(t0_frame, [0.1, 0.1, 0.7])
    t1_frame_o3d = create_o3d_pcl(t1_frame, [0.9, 0.1, 0.1])

    o3d_flow = o3d.geometry.LineSet()
    o3d_flow.points = o3d.utility.Vector3dVector(flow)
    o3d_flow.lines = o3d.utility.Vector2iVector(indices)

    o3d.visualization.draw_geometries([t0_frame_o3d, t1_frame_o3d, o3d_flow])


def show_pcl(t0_frame, t1_frame):
    """
    Visualizes two point clouds.
    :param t0_frame: numpy array of shape (3, n_points) containing the coordinates of the first point cloud
    :param t1_frame: numpy array of shape (3, n_points) containing the coordinates of the second point cloud
    :return: None
    """
    assert t0_frame.shape[1] == 3 and type(t0_frame) == np.ndarray
    assert t1_frame.shape[1] == 3 and type(t1_frame) == np.ndarray

    t0_frame_o3d = create_o3d_pcl(t0_frame, BLUE)
    t1_frame_o3d = create_o3d_pcl(t1_frame, RED)
    o3d.visualization.draw_geometries([t0_frame_o3d, t1_frame_o3d])
