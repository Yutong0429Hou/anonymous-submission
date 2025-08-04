import torch
import torch.nn as nn
import torch.nn.functional as F

class AddCoords(nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        xx_channel = torch.arange(w).repeat(1, h, 1).float().to(x.device)
        yy_channel = torch.arange(h).repeat(1, w, 1).transpose(1, 2).float().to(x.device)

        xx_channel = xx_channel / (w - 1)
        yy_channel = yy_channel / (h - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(b, 1, 1, 1)
        yy_channel = yy_channel.repeat(b, 1, 1, 1)

        return torch.cat([x, xx_channel, yy_channel], dim=1)

class SimplifiedDetailEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimplifiedDetailEncoder, self).__init__()

        self.coord_add = AddCoords()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.residual_proj = nn.Sequential(
            nn.Conv2d(in_channels + 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x_coord = self.coord_add(x)
        out = self.conv(x_coord)
        res = self.residual_proj(x_coord)
        return F.relu(out + res)
