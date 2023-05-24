import torch
import torch.nn as nn
import torch.functional as F


class FocalLoss2D(nn.Module):
    def __init__(self, gamma=2, ce_weights=(1, 1), reduction='mean'):
        super(FocalLoss2D, self).__init__()
        self.gamma = gamma
        self.ce_weights = ce_weights
        self.reduction = reduction

        self.CE = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights), ignore_index=-1, reduction='none')

    def forward(self, logits, target):
        # Logits: B, N, C, but to CrossEntropy it needs to be B, C, N
        # Target: B, N
        CE_loss = self.CE(logits, target)
        logits_soft = F.log_softmax(logits, dim=1)

        max_logits = torch.max(logits_soft, dim=1)[
            0]  # values, CE loss should be 0 on -1 index and therefore cancels this

        loss = (1 - max_logits) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == "none":
            return loss


class FocalLoss3D(nn.Module):
    def __init__(self, gamma=2, ce_weights=(1, 1), reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce_weights = ce_weights
        self.reduction = reduction

        self.CE = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights), ignore_index=-1, reduction='none')

    def forward(self, logits, target):
        # Logits: B, N, C, but to CrossEntropy it needs to be B, C, N
        # Target: B, N
        # for speed, there can be only one softmax I guess
        logits = logits.permute(0, 2, 1)

        CE_loss = self.CE(logits, target)

        logits = F.log_softmax(logits, dim=1)

        pt = logits.permute(0, 2, 1)

        pt = pt.flatten(start_dim=0, end_dim=1)
        target_gather = target.flatten()

        ignore_index = -1
        valid_mask = target_gather != ignore_index
        valid_target = target_gather[valid_mask]
        valid_pt = pt[valid_mask]
        CE_loss = CE_loss.flatten()[valid_mask]

        valid_target = valid_target.tile(2, 1).permute(1, 0)  # get the same shape as pt
        only_probs_as_target = torch.gather(valid_pt, 1, valid_target)[:, 0]

        loss = (1 - only_probs_as_target) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == "none":
            return loss
