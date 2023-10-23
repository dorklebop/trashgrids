import torch.nn as nn

import torch

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

import torch.nn.functional as F


class CKBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout_rate: float,
        norm_class: type[nn.Module],
        conv_class: type[
            nn.Module
        ],  # (dwromero) For now we can assume a normal conv. We can at some point
        # also replace it with a CKConv or other things.
        dropout_class: type[nn.Module],
        nonlinear_class: type[nn.Module],
        cfg,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.padding = "same"
        self.norm = norm_class(num_features=in_channels)
        self.conv = conv_class(
            in_channels=in_channels,  # TODO(dwromero) is this pointwise?
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=in_channels,
        )  # For depth-wise convolutions
        self.channel_mixer = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.nonlinear = nonlinear_class()
        self.dropout = dropout_class(p=dropout_rate)
        self.linear = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        # kernel dispatch can be optimized by merging operations.
        out = self.nonlinear(self.channel_mixer(self.conv(self.norm(x))))
        out = self.dropout(out)
        out = self.nonlinear(self.linear(out))

        return out + shortcut



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class ConvNeXtBlock3D(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout_rate: float,
        norm_class: type[nn.Module],
        conv_class: type[
            nn.Module
        ],  # (dwromero) For now we can assume a normal conv. We can at some point
        dropout_class: type[nn.Module],
        nonlinear_class: type[nn.Module],
        cfg,
    ):
        super().__init__()


        layer_scale_init_value = cfg.net.block.layer_scale_init_value #1e-6,
        bottleneck_factor = cfg.net.block.bottleneck_factor
        drop_path = cfg.net.block.drop_path

        self.padding = "same"


        # self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)  # depthwise conv

        self.dwconv = conv_class(
            in_channels=in_channels,  # TODO(dwromero) is this pointwise?
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=in_channels,
        )  # For depth-wise convolutions

        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(in_channels, bottleneck_factor * out_channels)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(bottleneck_factor * out_channels, out_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)


        x = self.shortcut(input) + self.drop_path(x)


        return x