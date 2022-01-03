from pt_layers import (
    ConvBlock,
    ResidualBlock,
    ConvolutionalPolicyHead,
    ConvolutionalValueOrMovesLeftHead,
    DensePolicyHead,
    DenseValueOrMovesLeftHead,
)
from pt_losses import policy_loss, value_loss, moves_left_loss
import torch
from torch import nn
from collections import OrderedDict
from typing import Optional, NamedTuple


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor
    moves_left: torch.Tensor
    policy_loss: Optional[torch.Tensor]
    value_loss: Optional[torch.Tensor]
    moves_left_loss: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]


class LeelaZeroNet(nn.Module):
    def __init__(
        self,
        num_filters,
        num_residual_blocks,
        se_ratio,
        policy_loss_weight,
        value_loss_weight,
        moves_left_loss_weight,
        q_ratio,
    ):
        super().__init__()
        self.input_block = ConvBlock(
            input_channels=112, filter_size=3, output_channels=num_filters
        )
        residual_blocks = OrderedDict(
            [
                (f"residual_block_{i}", ResidualBlock(num_filters, se_ratio))
                for i in range(num_residual_blocks)
            ]
        )
        self.residual_blocks = nn.Sequential(residual_blocks)
        self.policy_head = ConvolutionalPolicyHead(num_filters=num_filters)
        # The value head has 3 dimensions for estimating the likelihood of win/draw/loss (WDL)
        self.value_head = ConvolutionalValueOrMovesLeftHead(
            input_dim=num_filters,
            output_dim=3,
            num_filters=32,
            hidden_dim=128,
            relu=False,
        )
        # Moves left cannot be less than 0, so we use relu to clamp
        self.moves_left_head = ConvolutionalValueOrMovesLeftHead(
            input_dim=num_filters,
            output_dim=1,
            num_filters=8,
            hidden_dim=128,
            relu=True,
        )
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.moves_left_loss_weight = moves_left_loss_weight
        self.q_ratio = q_ratio

    def forward(
        self,
        input_planes: torch.Tensor,
        policy_target: torch.Tensor,
        wdl_target: torch.Tensor,
        q_target: torch.Tensor,
        moves_left_target: torch.Tensor,
    ) -> ModelOutput:
        flow = input_planes.reshape(-1, 112, 8, 8)
        flow = self.input_block(flow)
        flow = self.residual_blocks(flow)
        policy_out = self.policy_head(flow)
        value_out = self.value_head(flow)
        moves_left_out = self.moves_left_head(flow)
        value_target = q_target * self.q_ratio + wdl_target * (1 - self.q_ratio)
        p_loss = policy_loss(policy_target, policy_out)
        v_loss = value_loss(value_target, value_out)
        ml_loss = moves_left_loss(moves_left_target, moves_left_out)
        total_loss = (
            self.policy_loss_weight * p_loss
            + self.value_loss_weight * v_loss
            + self.moves_left_loss_weight * ml_loss
        )
        return ModelOutput(
            policy_out, value_out, moves_left_out, p_loss, v_loss, ml_loss, total_loss
        )
