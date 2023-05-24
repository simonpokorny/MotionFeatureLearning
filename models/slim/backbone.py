import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    print("For faster inference compile alt_cuda_corr from official RAFT repo")
    pass


from .updateBlock import UpdateBlock
from .resnetEncoder import ResnetEncoder


class RaftBackbone(pl.LightningModule):
    def __init__(self,
                 iters=6,
                 corr_levels=4,
                 corr_radius=3,
                 feature_downsampling_factor=8):
        """
         Initialize the RAFT modul.

         """
        super(RaftBackbone).__init__()
        self.feature_downsampling_factor = feature_downsampling_factor
        self.corr_search_radius = corr_radius
        self.corr_num_levels = corr_levels
        self.iters = iters

        # Encoder for correlation and input to RAFT-S
        self._features_encoder_net = ResnetEncoder(input_dim=64, output_dim=128, norm_fn='instance', dropout=0.0)
        # Encoder for context in shape hidden_dim=96 + context_dim=64 -> 160
        self._context_encoder_net = ResnetEncoder(input_dim=64, output_dim=160, norm_fn='none', dropout=0.0)
        # RAFT update block, SLIM is using smaller version RAFT-S
        self._update_block = UpdateBlock()

    def forward(self, x):
        """
        Forward pass of the RAFT model.

        Args:
            x (torch.Tensor): pillar embeddings in the shape of (bs * 2, 640, 640, 64)

        Returns:
            A tuple of two dicts containing forward and backward flow maps and classes.
        """

        assert x.shape == (2, 64, 640, 640)
        # RAFT Encoder step
        # Note that we are using two encoder, one is for features extraction
        # and second one is for context extraction.
        # Feature encoder:
        batch_size = int(x.shape[0] / 2)
        pillar_features = self._features_encoder_net(x)
        t0_features, t1_features = torch.split(pillar_features, [batch_size, batch_size], dim=0)
        # frames features are in shape [BS, 128, 80, 80]
        previous_pillar_embeddings, current_pillar_embeddings = torch.split(x, [batch_size, batch_size], dim=0)
        # pillar embeddings are in shape [BS, 64, 640, 640]

        # RAFT motion flow backbone
        # Note that we predict motion from t0 to t1 and t1 to t0
        retvals_forward = self._predict_single_flow_and_classes(previous_pillar_embeddings=previous_pillar_embeddings,
                                                                t0_features=t0_features,
                                                                t1_features=t1_features)

        retvals_backward = self._predict_single_flow_and_classes(previous_pillar_embeddings=current_pillar_embeddings,
                                                                 t0_features=t1_features,
                                                                 t1_features=t0_features)
        return retvals_forward, retvals_backward

    def _predict_single_flow_and_classes(self, previous_pillar_embeddings, t0_features, t1_features):
        """
        Predicts a single flow and classes.

        Args:
            previous_pillar_embeddings (torch.Tensor): pillar embeddings at time step t-1 in shape (bs, 64, 640, 640)
            t0_features (torch.Tensor): features at time step t-1 in shape (bs, 128, 80, 80).
            t1_features (torch.Tensor): features at time step t in shape (bs, 128, 80, 80).

        Returns:
            A dictionary containing flow maps and classes.
        """
        # 3. RAFT Flow Backbone

        # Initialization of the flow
        coords_t0, coords_t1 = self._initialize_flow(previous_pillar_embeddings, indexing="ij")
        bs, c, h, w = coords_t0.shape
        device = previous_pillar_embeddings.device

        # Initialization for logits and weights for Kabsch algorithm
        logits = torch.zeros([bs, 4, h, w], device=device)
        weights = torch.zeros([bs, 1, h, w], device=device)

        corr_fn = CorrBlock(fmap1=t0_features,
                            fmap2=t1_features,
                            num_levels=self.corr_num_levels,
                            radius=self.corr_search_radius,
                            indexing="ij")

        # Context encoder
        cnet = self._context_encoder_net(previous_pillar_embeddings)  # context features shape [BS, 160, 80, 80]
        net, inp = torch.split(cnet, [self.hdim, self.cdim], dim=1)
        net = torch.tanh(net)
        inp = nn.functional.relu(inp)

        intermediate_flow_predictions = []
        for _i in range(self.iters):

            coords_t1 = coords_t1.detach()
            logits = logits.detach()
            weights = weights.detach()

            corr = corr_fn(coords_t1)  # index correlation volume
            flow = coords_t1 - coords_t0

            net, delta_flow, delta_logits, mask, delta_weights = \
                self._update_block(net, inp, corr, flow, logits, weights)

            coords_t1 = coords_t1 + delta_flow
            flow = flow + delta_flow
            logits = logits + delta_logits
            weights = weights + delta_weights

            upsampled_flow = self._upsample(flow, n=self.feature_downsampling_factor)
            upsampled_flow_usfl_convention = self._change_flow_convention_from_raft2usfl(upsampled_flow)

            # Upsample the weight logits for static aggregation and logits
            upsampled_weights = self._upsample(weights, n=self.feature_downsampling_factor)
            upsampled_logits = self._upsample(logits, n=self.feature_downsampling_factor)

            # Concatenate the upsampled logits, flow convention, and weight logits to the network output
            conc_out = torch.cat([upsampled_logits,
                                  upsampled_flow_usfl_convention,
                                  upsampled_flow_usfl_convention,
                                  upsampled_weights], dim=1)

            intermediate_flow_predictions.append(conc_out)

        return intermediate_flow_predictions


    def _change_flow_convention_from_raft2usfl(self, flow):
        """
        Converts the optical flow representation from RAFT convention to USFL convention.

        "RAFT" is a popular optical flow method that outputs optical flow in a convention where
        positive X component indicates movement to the right, and positive Y component indicates
        movement downwards. This convention is used in popular computer vision benchmarks such as KITTI.

        "USFL" is short for "up-sampled flow", and it refers to a convention used in the Lyft Level 5 AV dataset,
        where positive X component indicates movement to the right, and positive Y component indicates movement upwards.

        Args:
        flow (torch.Tensor): A tensor of shape (batch_size, 2, height, width) representing the optical flow in RAFT convention.

        Returns:
        torch.Tensor: A tensor of shape (batch_size, 2, height, width) representing the optical flow in USFL convention.
        """
        # x,y - resolution of bev map
        resolution_adapter = torch.tensor([70 / 640, 70 / 640], dtype=torch.float32, device=self.device).reshape((1, -1, 1, 1))
        flow_meters = flow * resolution_adapter

        x = flow_meters[:, 0:1]
        y = flow_meters[:, 1:2]

        return torch.cat((y,x), dim=1)

    def _upsample(self, flow, n, mode='bilinear'):
        """
        Upsamples the input optical flow tensor by a factor of 8 or different.

        Args:
        flow (torch.Tensor): A tensor of shape (batch_size, 2, height, width) representing the optical flow.
        n (int): Upsampling factor.
        mode (str): Interpolation mode to be used. Defaults to 'bilinear'.

        Returns:
        torch.Tensor: A tensor of shape (batch_size, 2, n*height, n*width) representing the upsampled optical flow.
        """
        new_size = (n * flow.shape[2], n * flow.shape[3])
        return n * torch.nn.functional.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    def _initialize_flow(self, img, indexing="xy"):
        """
        Initializes the optical flow tensor.

        Args:
        img (torch.Tensor): A tensor of shape (batch_size, num_channels, height, width) representing the input image.
            indexing (str): The type of coordinate indexing to be used. Can be set to 'xy' or 'ij'. Defaults to 'xy'.

        Returns:
        Tuple[torch.Tensor]: A tuple of two tensors of shape (batch_size, 2, height / feature_downsampling_factor,
            width / feature_downsampling_factor) representing the initial and
            final coordinate grids for computing optical flow.
        """
        coords0 = coords_grid(batch=img, downscale_factor=self.feature_downsampling_factor, device=img.device, indexing=indexing)
        coords1 = coords_grid(batch=img, downscale_factor=self.feature_downsampling_factor, device=img.device, indexing=indexing)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, indexing="ij"):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.indexing = indexing

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing=self.indexing), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())



def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, downscale_factor=8, device="cpu", indexing="xy"):
    bs, ch, h, w = batch.shape
    coords = torch.meshgrid(torch.arange(h // downscale_factor, device=device),
                            torch.arange(w // downscale_factor, device=device), indexing=indexing)
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(bs, 1, 1, 1)

