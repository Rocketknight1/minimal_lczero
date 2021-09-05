import torch
from torch import nn
from torch.nn import functional as F
import lc0_az_policy_map


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, se_ratio):
        super().__init__()
        self.se_ratio = se_ratio
        self.pooler = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Sequential(
            nn.Linear(channels, channels // se_ratio, bias=False),
            nn.ReLU()
        )
        self.expand = nn.Linear(channels // se_ratio, channels * 2, bias=False)
        self.channels = channels
        nn.init.xavier_normal_(self.squeeze[0].weight)
        nn.init.xavier_normal_(self.expand.weight)

    def forward(self, x):
        pooled = self.pooler(x).view(-1, self.channels)
        squeezed = self.squeeze(pooled)
        expanded = self.expand(squeezed).view(-1, self.channels * 2, 1, 1)
        gammas, betas = torch.split(expanded, self.channels, dim=1)
        gammas = torch.sigmoid(gammas)
        return gammas * x + betas


class ConvBlock(nn.Module):
    def __init__(self, input_channels,  output_channels, filter_size):
        super().__init__()
        self.conv_layer = nn.Conv2d(input_channels, output_channels, filter_size, bias=False, padding='same')
        self.batchnorm = nn.BatchNorm2d(output_channels)
        nn.init.xavier_normal_(self.conv_layer.weight)

    def forward(self, inputs):
        out = self.conv_layer(inputs)
        out = self.batchnorm(out)
        return F.relu(out)


class ResidualBlock(nn.Module):
    def __init__(self, channels, se_ratio):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels,
                                            3,
                                            bias=False,
                                            padding='same',
                            )
        self.batch_norm = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels,
                                            3,
                                            bias=False,
                                            padding='same',
                             )
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        self.squeeze_excite = SqueezeExcitation(channels, se_ratio)

    def forward(self, inputs):
        out1 = self.conv1(inputs)
        out1 = F.relu(self.batch_norm(out1))
        out2 = self.conv2(out1)
        out2 = self.squeeze_excite(out2)
        return F.relu(inputs + out2)


class ConvolutionalPolicyHead(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv_block = ConvBlock(filter_size=3, input_channels=num_filters, output_channels=num_filters)
        # No l2_reg on the final convolution, because it's not going to be followed by a batchnorm
        self.conv = nn.Conv2d(num_filters,
            80,
            3,
            bias=True,
            padding='same')
        nn.init.xavier_normal_(self.conv.weight)
        self.fc1 = nn.parameter.Parameter(torch.tensor(lc0_az_policy_map.make_map(), requires_grad=False, dtype=torch.float32), requires_grad=False)

    def forward(self, inputs):
        flow = self.conv_block(inputs)
        flow = self.conv(flow)
        h_conv_pol_flat = flow.reshape(-1, 80 * 8 * 8)
        return h_conv_pol_flat @ self.fc1.type(h_conv_pol_flat.dtype)


class DensePolicyHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # No l2_reg on the final layer, because it's not going to be followed by a batchnorm
        self.fc_final = nn.Linear(hidden_dim, 1858)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc_final.weight)

    def forward(self, inputs):
        # Flatten input before proceeding
        inputs = inputs.reshape(inputs.shape[0], -1)
        out = F.relu(self.fc1(inputs))
        return self.fc_final(out)


class ConvolutionalValueOrMovesLeftHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_filters, hidden_dim, relu):
        super().__init__()
        self.num_filters = num_filters
        self.conv_block = ConvBlock(input_channels=input_dim, filter_size=1, output_channels=num_filters)
        # No l2_reg on the final layers, because they're not going to be followed by a batchnorm
        self.fc1 = nn.Linear(
            num_filters * 64,
            hidden_dim,
            bias=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=True)
        self.relu = relu
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc_out.weight)

    def forward(self, inputs):
        flow = self.conv_block(inputs)
        flow = flow.reshape(-1, self.num_filters * 8 * 8)
        flow = self.fc1(flow)
        flow = self.fc_out(flow)
        if self.relu:
            flow = F.relu(flow)
        return flow


class DenseValueOrMovesLeftHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, relu):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = relu
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc_out.weight)

    def forward(self, inputs):
        if inputs.dim() > 2:
            # Flatten input before proceeding
            inputs = inputs.reshape(inputs.shape[0], -1)
        flow = F.relu(self.fc1(inputs))
        flow = self.fc_out(flow)
        if self.relu:
            flow = F.relu(flow)
        return flow
