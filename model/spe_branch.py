
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class SpectralEncoder(nn.Module):

    def __init__(self, in_channels: int, patch_size: int = 9, d_out: int = 128, inter_size: int = 24):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_out = d_out
        self.inter_size = inter_size

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=self.inter_size,
            kernel_size=(7, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
            bias=True
        )
        self.bn1 = nn.BatchNorm3d(self.inter_size)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=self.inter_size,
            out_channels=self.inter_size,
            kernel_size=(7, 1, 1),
            stride=(1, 1, 1),
            padding=(3, 0, 0),
            bias=True
        )
        self.bn2 = nn.BatchNorm3d(self.inter_size)
        self.act2 = nn.ReLU(inplace=True)


        spectral_depth_after_conv1 = ((self.in_channels - 7 + 2 * 1) // 2) + 1
        self.conv4 = nn.Conv3d(
            in_channels=self.inter_size,
            out_channels=self.d_out,
            kernel_size=(spectral_depth_after_conv1, 1, 1),
            bias=True
        )
        self.bn4 = nn.BatchNorm3d(self.d_out)
        self.act4 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, d_out]
        """
        # [B, C, H, W] -> [B, 1, C, H, W]
        x = x.unsqueeze(1)

        # conv1: [B, 1, C, H, W] -> [B, inter_size, C1, H, W]
        x = self.conv1(x)
        x = self.act1(self.bn1(x))

        # conv2: [B, inter_size, C1, H, W] -> [B, inter_size, C1, H, W]
        x = self.conv2(x)
        x = self.act2(self.bn2(x))

        # conv4: [B, inter_size, C1, H, W] -> [B, d_out, 1, H, W]
        x = self.conv4(x)
        x = self.act4(self.bn4(x))

        # 去掉光谱深度维
        x = x.squeeze(2)  # [B, d_out, H, W]

        # 空间全局池化
        x = self.avgpool(x)  # [B, d_out, 1, 1]
        x = x.flatten(1)  # [B, d_out]

        return x



