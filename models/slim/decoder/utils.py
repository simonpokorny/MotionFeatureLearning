from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn



def shapes_broadcastable(shape_a, shape_b):
    """
    Check if two shapes are broadcastable for element-wise operations.

    Args:
        shape_a (tuple of int): Shape of the first tensor.
        shape_b (tuple of int): Shape of the second tensor.

    Returns:
        bool: True if the shapes are broadcastable, False otherwise.

    Raises:
        ValueError: If the number of dimensions of the shapes are different.
    """
    return all(sa == 1 or sb == 1 or sa == sb for sa, sb in zip(shape_a, shape_b))


def numerically_stable_quotient_lin_comb_exps(
    *, num_exps, num_weights, denom_exps, denom_weights
):
    # evaluates in numerically stable form expression of
    # (sum_i num_weight_i * exp(num_exp_i))  /  (sum_i denom_weight_i * exp(denom_exp_i))
    max_denom_exp = denom_exps[0]
    for i in range(1, len(denom_exps)):
        max_denom_exp = torch.max(max_denom_exp, denom_exps[i])
    # divide numerator and denominator by exp(max_denom_exp)
    # by subtracting in every exponent
    for i in range(len(num_exps)):
        num_exps[i] = num_exps[i] - max_denom_exp
    for i in range(len(denom_exps)):
        denom_exps[i] = denom_exps[i] - max_denom_exp
    # now max denom exp is 0 and therefore denominator is well-behaved
    # in the case where all num exps are contained in the denom exps we also have a well-behaved numerator
    # and everything is fine (quite common case, e.g. exp(a)/(exp(a)+exp(b))))
    num_elems = [w * torch.exp(e) for w, e in zip(num_weights, num_exps)]
    denom_elems = [w * torch.exp(e) for w, e in zip(denom_weights, denom_exps)]
    return torch.sum(torch.stack(num_elems), dim=0) / torch.sum(torch.stack(denom_elems), dim=0)


def numerically_stable_quotient_lin_comb_exps_across_axis(num_exps, num_weights, denom_exps, denom_weights, axis=-1):
    assert (
        num_exps.ndim == num_weights.ndim == denom_exps.ndim == denom_weights.ndim
    )
    # evaluates in numerically stable form expression of
    # (sum_i num_weight_i * exp(num_exp_i))  /  (sum_i denom_weight_i * exp(denom_exp_i))
    weight_masked_denom_exps = torch.where(
        denom_weights != 0.0,
        denom_exps,
        torch.min(denom_exps, dim=axis, keepdim=True).values,
    )
    max_denom_exp = torch.max(weight_masked_denom_exps, dim=axis, keepdim=True).values
    # divide numerator and denominator by exp(max_denom_exp)
    # by subtracting in every exponent
    num_exps = num_exps - max_denom_exp
    denom_exps = denom_exps - max_denom_exp
    # now max denom exp is 0 and therefore denominator is well-behaved
    # in the case where all num exps are contained in the denom exps we also have a well-behaved numerator
    # and everything is fine (quite common case, e.g. exp(a)/(exp(a)+exp(b))))
    num = num_weights * torch.exp(num_exps)
    denom = denom_weights * torch.exp(denom_exps)
    return torch.sum(num, dim=axis) / torch.sum(denom, dim=axis)


def normalized_sigmoid_sum(logits, mask=None):
    # sigmoid(x) = exp(-relu(-x)) * sigmoid(abs(x))
    logits = logits.double() # TODO done because somehow convergence values to zeros
    neg_logit_part = -torch.nn.functional.relu(-logits)
    weights = torch.sigmoid(torch.abs(logits))
    if mask is not None:
        neg_logit_part = torch.where(mask, neg_logit_part, torch.zeros_like(neg_logit_part))
        weights = torch.where(mask, weights, torch.zeros_like(weights))
    return numerically_stable_quotient_lin_comb_exps_across_axis(
        num_exps=neg_logit_part[..., :, None],
        num_weights=weights[..., :, None],
        denom_exps=neg_logit_part[..., None, :],
        denom_weights=weights[..., None, :],
    )


def scale_gradient(tensor, scaling):
    """
    Scale the gradient of a tensor by a factor.

    Args:
        tensor: The tensor whose gradient should be scaled.
        scaling: The scaling factor to apply to the gradient. If `scaling == 1.0`, the gradient is unchanged. If
            `scaling == 0.0`, the gradient is set to zero. If `scaling > 0.0`, the gradient is scaled by the factor
            `(scaling - 1)`.

    Returns:
        The input tensor with the gradient scaled as specified by `scaling`.
    """
    if scaling == 1.0:
        return tensor
    if scaling == 0.0:
        return tensor.detach()
    assert scaling > 0.0
    return tensor * scaling - tensor.detach() * (scaling - 1.0)




