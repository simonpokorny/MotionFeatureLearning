import open3d as o3d
import torch.utils.data as data
import numpy as np

from vizualization.utils import show_flow, show_pcl


class Visualizer(data.DataLoader):
    def __init__(self, dataloader, visualize, shuffle=False, dims=3, model=None):
        super().__init__(dataloader.dataset, batch_size=1, shuffle=shuffle, collate_fn=dataloader.collate_fn)
        assert dims in [2, 3], "Visualizer can only visualize in 2d or 3d space"

        self.model = model
        self.vis_type = visualize
        types = ["flow3d", "sync3d", "seg3d"]
        if visualize == "flow3d":
            self.visualize = self.visualize3Dflow
        elif visualize == "seg3d":
            self.visualize = self.visualize3Dseg
        elif visualize == "sync3d":
            self.visualize = self.visualize3Dsync
        else:
            raise ValueError(f"visualize must be in {types}")

        self.colors = np.array(
            [[1.0, 0.0, 0.0],  # RED
             [0.0, 1.0, 0.0],  # Green
             [0.0, 0.0, 1.0],  # Blue
             [1.0, 1.0, 0.0]])  # Yellow

    def __iter__(self):
        # Get the original data loader iterator
        data_iter = super(Visualizer, self).__iter__()

        # iterate through the batches and plot them
        for batch_idx, batch_data in enumerate(data_iter):
            # Unwrap data
            pcl, labels, trans = batch_data

            t0_frame, t1_frame = pcl
            flow, seg_lab = labels

            t0_frame = t0_frame[0].detach().cpu().numpy()[0]
            t1_frame = t1_frame[0].detach().cpu().numpy()[0]

            flow = flow.detach().cpu().numpy()[0]
            seg_lab = seg_lab.detach().cpu().numpy()[0]

            trans = trans.detach().cpu().numpy()[0]

            if self.dataset._apply_pillarization == True:
                t0_frame = t0_frame[:, :3] + t0_frame[:, 3:6]
                t1_frame = t1_frame[:, :3] + t1_frame[:, 3:6]

            # Utilize the flow from model
            if self.model is not None and "flow" in self.vis_type:
                self.model.eval()
                prediction = self.model(pcl, trans)
                flow = prediction[0][-1][0]["aggregated_flow"].detach().cpu().numpy()[0]
                # flow = self.model.refinement_module(prediction[2], prediction[3], prediction[0][-1][0])

            batch_data = ((t0_frame, t1_frame), (flow, seg_lab), trans)
            self.visualize(batch_data)
            yield batch_data

    def visualize3Dseg(self, batch):

        (t0_frame, _), (_, seg_labels), _ = batch

        frame = o3d.geometry.PointCloud()
        frame.points = o3d.utility.Vector3dVector(t0_frame)

        colors = np.take_along_axis(self.colors, seg_labels.astype(int), axis=0)
        frame.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([frame])

    def visualize3Dflow(self, batch):

        (t0_frame, t1_frame), (flow, seg_lab), t0_to_t1 = batch
        assert (flow.shape == t0_frame.shape), "Flows should be equal to frame t0"
        assert t0_to_t1.shape == (4, 4), "Matrix in custom dataset must be in shape (4, 4)"
        show_flow(t0_frame, t1_frame, flow)

    def visualize3Dsync(self, batch):
        """
        Visualizes a pair of point clouds with flow between them.

        Args:
            index (int): The index of the frame pair to visualize.

        Returns:
            None
        """
        x, flow, t0_to_t1 = batch

        t0_frame = x[0][0].detach().cpu().numpy()[0]
        t1_frame = x[1][0].detach().cpu().numpy()[0]

        t0_to_t1 = t0_to_t1.detach().cpu().numpy()[0]
        assert t0_to_t1.shape == (4, 4), "Matrix in custom dataset must be in shape (4, 4)"

        # Get only xyz coordinates without any features (unpillarization)
        t0_frame = t0_frame[:, :3] + t0_frame[:, 3:6]
        t1_frame = t1_frame[:, :3] + t1_frame[:, 3:6]

        t0_frame = self.cart2hom(t0_frame).T
        t0_frame = (t0_to_t1 @ t0_frame).T
        t0_frame = self.hom2cart(t0_frame)

        show_pcl(t0_frame, t1_frame)

    @staticmethod
    def cart2hom(pcl):
        assert pcl.ndim == 2 and pcl.shape[1] == 3, "PointCloud should be in shape [N, 3]"
        N, _ = pcl.shape
        return np.concatenate((pcl, np.ones((N, 1))), axis=1)

    @staticmethod
    def hom2cart(pcl):
        assert pcl.ndim == 2 and pcl.shape[1] == 4, "PointCloud should be in shape [N, 4]"
        return pcl[:, :3] / pcl[:, 3:4]


if __name__ == "__main__":
    from datasets.waymo.waymodatamodule import WaymoDataModule

    # COMPUTE NUM POINTS FOR MOVING DYNAMIC THRESHOLDS

    import numpy as np

    grid_cell_size = 0.109375
    dataset_path = "../data/waymo/"

    data_module = WaymoDataModule(
        dataset_directory=dataset_path,
        grid_cell_size=grid_cell_size,
        x_min=-35,
        x_max=35,
        y_min=-35,
        y_max=35,
        z_min=0.25,
        z_max=10,
        batch_size=1,
        has_test=False,
        num_workers=0,
        n_pillars_x=640,
        n_pillars_y=640,
        n_points=None,
        apply_pillarization=True)

    data_module.setup()
    train_dl = data_module.train_dataloader()

    vis = Visualizer(train_dl, visualize="flow3d")

    for t in vis:
        a = None
