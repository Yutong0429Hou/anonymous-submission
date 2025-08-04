# Spiral patch scan utility
import torch
from torch import nn
from torch.nn import functional as F


def spiral_indices(h, w, clockwise=True):
    visited = [[False]*w for _ in range(h)]
    dx = [0, 1, 0, -1] if clockwise else [1, 0, -1, 0]
    dy = [1, 0, -1, 0] if clockwise else [0, 1, 0, -1]
    x = y = d = 0
    result = []
    for _ in range(h * w):
        result.append((x, y))
        visited[x][y] = True
        nx, ny = x + dx[d], y + dy[d]
        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny]:
            x, y = nx, ny
        else:
            d = (d + 1) % 4
            x += dx[d]
            y += dy[d]
    return result


def spiral_patch_sequence(x, patch_size=4, clockwise=True):
    B, C, H, W = x.shape
    ph, pw = H // patch_size, W // patch_size
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, ph, pw, -1)  # [B, ph, pw, C*ps*ps]
    idx = spiral_indices(ph, pw, clockwise=clockwise)
    idx_flat = [i * pw + j for i, j in idx]
    patches = patches.view(B, ph * pw, -1)
    patches = patches[:, idx_flat, :]
    return patches  # [B, N, D]


# Example BiMamba2_1D modified to support [B, D, N] input
class GatedUnit(nn.Module):
    def __init__(self, input_channels):
        super(GatedUnit, self).__init__()
        self.gate_fc = nn.Linear(input_channels, input_channels)

    def forward(self, x):
        gate_weights = torch.sigmoid(self.gate_fc(x.transpose(1, 2)))
        return x * gate_weights.transpose(1, 2)


class BiMamba2_1D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.fc_in = nn.Linear(cin, cout, bias=False)
        self.fc_out = nn.Linear(cout, cin, bias=False)
        self.gated_unit = GatedUnit(cout)
        self.mamba = nn.Sequential(
            nn.Conv1d(cout, cout, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(cout, cout, 1)
        )

    def forward(self, x):  # x: [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        x = self.fc_in(x)
        x = x.transpose(1, 2)  # [B, D, N]
        y = self.mamba(x)
        y = self.gated_unit(y)
        y = y.transpose(1, 2)
        y = self.fc_out(y).transpose(1, 2)
        return y  # [B, D, N]


# Updated wrapper to accept cin, cout, d_model
class BiMamba2SpiralWrapper(nn.Module):
    def __init__(self, cin, cout, d_model=None, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Conv2d(cin, cout, 1)
        self.mamba = BiMamba2_1D(cout * patch_size * patch_size, cout * patch_size * patch_size)
        self.reproject = nn.Conv2d(cout, cin, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_embed = self.embed(x)
        x_seq = spiral_patch_sequence(x_embed, self.patch_size)  # [B, N, D]
        x_seq = x_seq.transpose(1, 2)  # [B, D, N]
        y_seq = self.mamba(x_seq)
        y_seq = y_seq.transpose(1, 2)  # [B, N, D]

        ph, pw = H // self.patch_size, W // self.patch_size
        y = y_seq.view(B, ph, pw, -1).permute(0, 3, 1, 2).contiguous()
        y = F.pixel_shuffle(y, self.patch_size)
        y = self.reproject(y)
        return y
