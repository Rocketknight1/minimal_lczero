from pt_layers import (
    ConvBlock,
    ResidualBlock,
    ConvolutionalPolicyHead,
    ConvolutionalValueOrMovesLeftHead,
)
from pt_losses import policy_loss, value_loss, moves_left_loss
import torch
from torch import nn
from collections import OrderedDict
from typing import Optional, NamedTuple
import pytorch_lightning as pl
from math import prod, sqrt


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor
    moves_left: torch.Tensor


class LeelaZeroNet(pl.LightningModule):
    def __init__(
        self,
        num_filters,
        num_residual_blocks,
        se_ratio,
        policy_loss_weight,
        value_loss_weight,
        moves_left_loss_weight,
        q_ratio,
        optimizer,
        learning_rate
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
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, input_planes: torch.Tensor) -> ModelOutput:
        flow = input_planes.reshape(-1, 112, 8, 8)
        flow = self.input_block(flow)
        flow = self.residual_blocks(flow)
        policy_out = self.policy_head(flow)
        value_out = self.value_head(flow)
        moves_left_out = self.moves_left_head(flow)
        return ModelOutput(policy_out, value_out, moves_left_out)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            for param in self.parameters():
                if getattr(param, "clamp_weights", False):
                    fan_in = prod(param.shape[1:])
                    fan_out = param.shape[0]
                    n_dims = fan_in * fan_out
                    scale = sqrt(2 / (fan_in + fan_out))
                    desired_norm = scale * sqrt(n_dims)
                    # clip_grad_norm does in-place weight norm clamping for us
                    torch.nn.utils.clip_grad_norm_(param, max_norm=desired_norm)
        inputs, policy_target, wdl_target, q_target, moves_left_target = batch
        policy_out, value_out, moves_left_out = self(inputs)
        value_target = q_target * self.q_ratio + wdl_target * (1 - self.q_ratio)
        p_loss = policy_loss(policy_target, policy_out)
        v_loss = value_loss(value_target, value_out)
        ml_loss = moves_left_loss(moves_left_target, moves_left_out)
        total_loss = (
            self.policy_loss_weight * p_loss
            + self.value_loss_weight * v_loss
            + self.moves_left_loss_weight * ml_loss
        )
        self.log("policy_loss", p_loss)
        self.log("value_loss", v_loss)
        self.log("moves_left_loss", ml_loss)
        self.log("total_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
