import torch
from torchmetrics import Metric


class AEE(Metric):
    """
    Calculates the Average Endpoint Error (AEE) between two flows. The AEE is the mean L2 distance between each endpoint
    of the predicted flow and the corresponding endpoint of the ground truth flow.

    """

    def __init__(self):
        super().__init__()
        self.add_state("errorL2", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, flow: torch.Tensor, gt_flow: torch.Tensor):
        """
        Updates the accumulated L2 distance error and total number of elements with the given flows.

        Args:
            flow (torch.Tensor): The predicted flow, with shape [BS, N, 3].
            gt_flow (torch.Tensor): The ground truth flow, with shape [BS, N, 3].
        """
        assert flow.shape == gt_flow.shape, f"Predicted flow have different shape in comparison with gt flow"
        # Computing the error only in xy coordinates
        err = torch.linalg.vector_norm(((gt_flow - flow))[:, :, :2], ord=2, dim=2)

        self.errorL2 += err.sum().long()
        self.total += gt_flow.shape[1]

    def compute(self):
        """
        Computes the Average Endpoint Error (AEE) as the ratio of the accumulated L2 distance error and the total number
        of elements.

        Returns:
            The AEE as a float32 tensor.
        """
        return self.errorL2.float() / self.total


class AEE_50_50(Metric):
    """
    AEE 50-50 as the average between the AEE measured separately on stationary and on moving points. Ground truth
    odometry is used to compute the non-rigid flow component $\bold{f}_{nr,gt,i} = \bold{f}_{gt,i} − (O_1 − I_4)pi$.
    Points with a non-rigid t→t+1 flow larger than mthresh = 5cm (corresponds to 1.8 $\frac{km}{h}$) are labeled as
    dynamic, the rest as stationary.

    Args:
        scanning_frequency (int): The scanning frequency in Hz. Default is 10.
    """

    def __init__(self, scanning_frequency: int = 10):
        super().__init__()
        self.add_state("stat_err", default=torch.tensor(0, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("dyn_err", default=torch.tensor(0, dtype=torch.double), dist_reduce_fx="sum")

        self.add_state("stat_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("dyn_total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.m_thresh = 0.05 * (10 / scanning_frequency)

    def cart2hom(self, pcl):
        assert pcl.ndim == 3 and pcl.shape[2] == 3, "PointCloud should be in shape [BS, N, 3]"
        BS, N, _ = pcl.shape
        return torch.cat((pcl, torch.ones((BS, N, 1), device=self.device)), dim=2)

    @staticmethod
    def hom2cart(pcl):
        assert pcl.ndim == 3 and pcl.shape[2] == 4, "PointCloud should be in shape [BS, N, 4]"
        return pcl[:, :, :3] / pcl[:, :, 3:4]

    def compute_gt_stat_flow(self, pcl_t0: torch.Tensor, odometry: torch.Tensor):
        pcl_t1 = self.cart2hom(pcl_t0).transpose(1, 2)
        pcl_t1 = (odometry @ pcl_t1).transpose(1, 2)
        pcl_t1 = self.hom2cart(pcl_t1)
        flow = pcl_t1 - pcl_t0
        return flow

    def update(self, flow: torch.Tensor, gt_flow: torch.Tensor, odometry: torch.Tensor, pcl_t0: torch.Tensor):
        """
        Updates the metric with the given predicted flow, ground truth flow, odometry, and point cloud at time t0.

        Args:
            flow (torch.Tensor): The predicted flow from t0 to t1, with shape [BS, N, 3].
            gt_flow (torch.Tensor): The ground truth flow, with shape [BS, N, 3].
            odometry (torch.Tensor): The ground truth odometry, with shape [BS, 4, 4].
            pcl_t0 (torch.Tensor): The point cloud at time t, with shape [BS, N, 3].
        """

        assert flow.shape == gt_flow.shape, f"Predicted flow have different shape in comparison with gt flow"
        assert pcl_t0.ndim == 3 and pcl_t0.shape[2] == 3, "PointCloud should be in shape [BS, N, 3]"
        assert odometry.ndim == 3 and odometry.shape[2] == 4, "Odometry should be in shape [BS, 4, 4]"

        gt_static_flow = self.compute_gt_stat_flow(pcl_t0=pcl_t0, odometry=odometry)

        gt_static_flow = gt_static_flow[:, :, :2]
        gt_flow = gt_flow[:, :, :2]
        flow = flow[:, :, :2]

        gt_err = torch.linalg.vector_norm((gt_static_flow - gt_flow), ord=2, dim=2)
        static_mask = gt_err < self.m_thresh
        dynamic_mask = torch.logical_not(static_mask)

        err = torch.linalg.vector_norm((gt_flow - flow), ord=2, dim=2)
        stat_err = err[static_mask]
        dyn_err = err[dynamic_mask]

        self.stat_err += stat_err.sum()
        self.dyn_err += dyn_err.sum()

        self.stat_total += static_mask.sum()
        self.dyn_total += dynamic_mask.sum()

    def compute(self):
        """
        Computes the metric as the average of the AEE errors measured separately on the stationary and dynamic points.

        Returns:
            Tuple of three torch.Tensors: The total AEE error, the AEE error for stationary points, and the AEE error
            for dynamic points.
        """
        static_error = self.stat_err.float() / self.stat_total
        dynamic_error = self.dyn_err.float() / self.dyn_total
        return (static_error + dynamic_error) / 2, static_error, dynamic_error

    def compute_total(self):
        total = self.stat_total + self.dyn_total
        return self.stat_total / total, self.dyn_total / total


class AccS(Metric):
    """
    Calculates the point ratio where the endpoint error (EE) or the relative error is less than 0.05.

    The endpoint error (EE) is the L2 distance between each endpoint of the predicted flow and the corresponding
    endpoint of the ground truth flow. The relative error is the EE divided by the L2 norm of the ground truth flow.
    The AccS is the ratio of the number of points where either the EE or the relative error is less than 0.05 and
    the total number of points in the ground truth flow.
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, flow: torch.Tensor, gt_flow: torch.Tensor):
        """
        Args:
            flow (torch.Tensor): The predicted flow, with shape [BS, N, 3].
            gt_flow (torch.Tensor): The ground truth flow, with shape [BS, N, 3].
        """
        assert flow.shape == gt_flow.shape, f"Predicted flow have different shape in comparison with gt flow"

        # Computing the error only in xy coordinates
        err = torch.linalg.vector_norm(((gt_flow - flow))[:, :, :2], ord=2, dim=2)
        relative_err = err / torch.linalg.vector_norm(gt_flow[:, :, :2], ord=2, dim=2)

        self.correct += torch.logical_or(err < 0.05, relative_err < 0.05).sum()
        self.total += gt_flow.shape[1]

    def compute(self):
        """
        Returns:
            The AccS as a float32 tensor.
        """
        return self.correct.float() / self.total


class AccR(Metric):
    """
    Calculates the point ratio where the endpoint error (EE) or the relative error is less than 0.1.

    The endpoint error (EE) is the L2 distance between each endpoint of the predicted flow and the corresponding
    endpoint of the ground truth flow. The relative error is the EE divided by the L2 norm of the ground truth flow.
    The AccS is the ratio of the number of points where either the EE or the relative error is less than 0.1 and
    the total number of points in the ground truth flow.
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, flow: torch.Tensor, gt_flow: torch.Tensor):
        """
        Args:
            flow (torch.Tensor): The predicted flow, with shape [BS, N, 3].
            gt_flow (torch.Tensor): The ground truth flow, with shape [BS, N, 3].
        """
        assert flow.shape == gt_flow.shape, f"Predicted flow have different shape in comparison with gt flow"

        # Computing the error only in xy coordinates
        err = torch.linalg.vector_norm(((gt_flow - flow))[:, :, :2], ord=2, dim=2)
        relative_err = err / torch.linalg.vector_norm(gt_flow, ord=2, dim=2)

        self.correct += torch.logical_or(err < 0.1, relative_err < 0.1).sum()
        self.total += gt_flow.shape[1]

    def compute(self):
        """
        Returns:
            The AccS as a float32 tensor.
        """
        return self.correct.float() / self.total


class Outl(Metric):
    """
    Calculates the point ratio where the endpoint error (EE) or the relative error is bigger than 0.3.

    The endpoint error (EE) is the L2 distance between each endpoint of the predicted flow and the corresponding
    endpoint of the ground truth flow. The relative error is the EE divided by the L2 norm of the ground truth flow.
    The AccS is the ratio of the number of points where either the EE or the relative error is higher than 0.3 and
    the total number of points in the ground truth flow.
    """

    def __init__(self):
        super().__init__()
        self.add_state("outliers", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, flow: torch.Tensor, gt_flow: torch.Tensor):
        """
        Args:
            flow (torch.Tensor): The predicted flow, with shape [BS, N, 3].
            gt_flow (torch.Tensor): The ground truth flow, with shape [BS, N, 3].
        """
        assert flow.shape == gt_flow.shape, f"Predicted flow have different shape in comparison with gt flow"

        # Computing the error only in xy coordinates
        err = torch.linalg.vector_norm(((gt_flow - flow))[:, :, :2], ord=2, dim=2)
        relative_err = err / torch.linalg.vector_norm(gt_flow[:, :, :2], ord=2, dim=2)

        self.outliers += torch.logical_or(err > 0.3, relative_err > 0.3).sum()
        self.total += gt_flow.shape[1]

    def compute(self):
        """
        Returns:
            The AccS as a float32 tensor.
        """
        return self.outliers.float() / self.total


class ROutl(Metric):
    """
    Calculates the point ratio where the endpoint error (EE) and the relative error is bigger than 0.3.

    The endpoint error (EE) is the L2 distance between each endpoint of the predicted flow and the corresponding
    endpoint of the ground truth flow. The relative error is the EE divided by the L2 norm of the ground truth flow.
    The AccS is the ratio of the number of points where the EE and the relative error is higher than 0.3 and
    the total number of points in the ground truth flow.
    """

    def __init__(self):
        super().__init__()
        self.add_state("outliers", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, flow: torch.Tensor, gt_flow: torch.Tensor):
        """
        Args:
            flow (torch.Tensor): The predicted flow, with shape [BS, N, 3].
            gt_flow (torch.Tensor): The ground truth flow, with shape [BS, N, 3].
        """
        assert flow.shape == gt_flow.shape, f"Predicted flow have different shape in comparison with gt flow"

        # Computing the error only in xy coordinates
        err = torch.linalg.vector_norm(((gt_flow - flow))[:, :, :2], ord=2, dim=2)
        relative_err = err / torch.linalg.vector_norm(gt_flow[:, :, :2], ord=2, dim=2)

        self.outliers += torch.logical_and(err > 0.3, relative_err > 0.3).sum()
        self.total += gt_flow.shape[1]

    def compute(self):
        """
        Returns:
            The AccS as a float32 tensor.
        """
        return self.outliers.float() / self.total
