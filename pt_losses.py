import torch
from torch import nn
from torch.nn import functional as F


def policy_loss(target: torch.Tensor, output: torch.Tensor):
    # Illegal moves are marked by a value of -1 in the labels - we mask these with large negative values
    output.masked_fill_(target < 0, -1e10)
    # The large negative values will still break the loss, so we replace them with 0 once we finish masking
    target = F.relu(target, inplace=True)
    log_prob = F.log_softmax(output.type(torch.float32), dim=1)
    nll = -(target.type(torch.float32) * log_prob).sum() / output.shape[0]
    return nll


def value_loss(target: torch.Tensor, output: torch.Tensor):
    log_prob = F.log_softmax(output.type(torch.float32), dim=1)
    value_nll = -(target.type(torch.float32) * log_prob)
    return value_nll.sum() / output.shape[0]


def moves_left_loss(target: torch.Tensor, output: torch.Tensor):
    # Scale the loss to similar range as other losses.
    scale = 20.0
    target = target.type(torch.float32) / scale
    output = output.type(torch.float32) / scale
    return F.huber_loss(output, target, reduction='mean', delta=10.0 / scale)
