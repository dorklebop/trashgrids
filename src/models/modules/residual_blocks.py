import torch.nn as nn
import gconv.gnn as gnn


class GResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        group_kernel_size,
        kernel_size=3,
        padding=0,
        group_mode="rbf",
        spatial_mode="bilinear",
        permute=False,
        mask=False
    ) -> None:
        super().__init__()

        self.gconv1 = gnn.GSeparableConvSE3(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            group_sampling_mode=group_mode,
            spatial_sampling_mode=spatial_mode,
            group_kernel_size=group_kernel_size,
            permute_output_grid=permute,
            mask=mask,
        )
        self.gconv2 = gnn.GSeparableConvSE3(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            group_sampling_mode=group_mode,
            spatial_sampling_mode=spatial_mode,
            group_kernel_size=group_kernel_size,
            permute_output_grid=permute,
            mask=mask
        )

        self.bn1 = gnn.GInstanceNorm3d(out_channels)
        self.bn2 = gnn.GInstanceNorm3d(out_channels)
        self.bn3 = gnn.GBatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.residual_mapping = gnn.GSeparableConvSE3(
            in_channels,
            out_channels,
            kernel_size=1,
            group_sampling_mode=group_mode,
            bias=False,
            permute_output_grid=False,
        )

    def forward(self, x, H):
        residual, residual_H = x, H

        x, H = self.gconv1(x, H)
        x, H = self.bn1(x, H)
        x = self.relu(x)

        x, H = self.gconv2(x, H)
        x, H = self.bn2(x, H)
        x = self.relu(x)

        residual, _ = self.residual_mapping(residual, residual_H, H)

        x = x + residual
        x, H = self.bn3(x, H)

        return x, H

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
