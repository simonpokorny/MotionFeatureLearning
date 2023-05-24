"""
This is update block for SLIM with GRU, motion encoder ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=96, out_dim=2, hidden_dim=256):
        super(FlowHead, self).__init__()
        assert out_dim in [2, 3, 4], \
            "choose out_dims=2 for flow or out_dims=4 for classification or 3 if the paper DL is dangerously close"
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=96, input_dim=192 + 96):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (3, 3), padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (3, 3), padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (3, 3), padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SlimMotionEncoder(nn.Module):
    def __init__(self):
        """
        Args:
            predict_logits (bool): Whether to predict logits.
        """
        super(SlimMotionEncoder, self).__init__()
        self.conv_stat_corr1 = nn.Conv2d(196, 96, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv_flow1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=3)
        self.conv_flow2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.conv_class1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(1, 1), padding=3)
        self.conv_class2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.conv = nn.Conv2d(160, 80, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, flow, corr, L_cls):
        corr = F.relu(self.conv_stat_corr1(corr))

        flow = F.relu(self.conv_flow1(flow))
        flow = F.relu(self.conv_flow2(flow))

        concat_vals = [corr, flow]

        L_cls = F.relu(self.conv_class1(L_cls))
        L_cls = F.relu(self.conv_class2(L_cls))
        concat_vals.append(L_cls)

        cor_flo_logits = torch.cat(concat_vals, dim=1)
        out = F.relu(self.conv(cor_flo_logits))

        return torch.cat([out, L_cls, flow], dim=1)


class UpdateBlock(nn.Module):
    def __init__(self):
        super(UpdateBlock, self).__init__()

        hidden_dim = 96
        num_stat_flow_head_channels = 3
        self.static_flow_head = FlowHead(input_dim=hidden_dim, out_dim=num_stat_flow_head_channels, hidden_dim=256)

        self.motion_encoder = SlimMotionEncoder()
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=208)
        self.classification_head = FlowHead(input_dim=hidden_dim, out_dim=4, hidden_dim=256)

    def forward(self, net, inp, corr, flow, L_cls, L_wgt):
        """
        Args:
            net (torch.Tensor): The hidden state tensor.
            inp (torch.Tensor): The input tensor.
            corr (torch.Tensor): The correlation tensor.
            flow (torch.Tensor): The optical flow tensor.
            logits (torch.Tensor): The segmentation logits tensor.
            weights (float): The weight for the segmentation logits tensor.

        Returns:
            net (torch.Tensor): The hidden state tensor of the ConvGRU.
            delta_static_flow (torch.Tensor): The static flow tensor.
            delta_logits (torch.Tensor): The logit tensor.
            mask (torch.Tensor): The upsampling mask tensor.
            delta_weights (torch.Tensor): The weight delta tensor.

        """

        assert L_cls.shape == (1, 3, 80, 80)
        assert L_wgt.shape == (1, 1, 80, 80)
        assert corr.shape == (1, 196, 80, 80)
        assert flow.shape == (1, 2, 80, 80)
        assert inp.shape == (1, 64, 80, 80)
        assert net.shape == (1, 96, 80, 80)

        motion_features = self.motion_encoder(torch.cat([flow, L_wgt], dim=1), corr, L_cls)
        inp = torch.cat([inp, motion_features], dim=1)

        # Iteration in GRU block
        net = self.gru(h=net, x=inp)

        delta = self.static_flow_head(net)
        delta_static_flow = delta[:, :2]
        delta_weights = delta[:, -1].unsqueeze(1)
        delta_logits = self.classification_head(net)
        return net, delta_static_flow, delta_logits, delta_weights,
