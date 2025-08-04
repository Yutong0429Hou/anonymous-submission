import torch
import torch.nn as nn
import torch.nn.functional as F

class MAFMv2(nn.Module):

    def __init__(self, in_channels):
        super(MAFMv2, self).__init__()


        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3),
        ])


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        )


        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)


        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual = x
        multi_scale_feats = [conv(x) for conv in self.multi_scale]
        x = sum(multi_scale_feats) / len(multi_scale_feats)


        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_attention = torch.sigmoid(avg_out + max_out)
        x = x * channel_attention

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        x = x * spatial_attention

        x = self.fusion(x + residual)
        return x
