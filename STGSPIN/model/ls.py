import math
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_flat = Rearrange('c_out c_in k1 k2 -> c_out c_in (k1 k2)')(conv_weight)
        conv_weight_cd = torch.zeros(conv_shape[0], conv_shape[1], 9, device=conv_weight.device)
        conv_weight_cd[:, :, :] = conv_weight_flat
        conv_weight_cd[:, :, 4] = conv_weight_flat[:, :, 4] - conv_weight_flat.sum(2)
        conv_weight_cd = Rearrange('c_out c_in (k1 k2) -> c_out c_in k1 k2', k1=3, k2=3)(conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_out c_in k1 k2 -> c_out c_in (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_out c_in (k1 k2) -> c_out c_in k1 k2', k1=3, k2=3)(conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_rd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        if math.fabs(self.theta - 0.0) < 1e-8:
            return self.conv(x)
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            conv_weight = Rearrange('c_out c_in k1 k2 -> c_out c_in (k1 k2)')(conv_weight)
            conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 25, device=conv_weight.device)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            return F.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias,
                            stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.zeros(conv_shape[0], conv_shape[1], 9, device=conv_weight.device)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_out c_in (k1 k2) -> c_out c_in k1 k2', k1=3, k2=3)(conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.zeros(conv_shape[0], conv_shape[1], 9, device=conv_weight.device)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_out c_in (k1 k2) -> c_out c_in k1 k2', k1=3, k2=3)(conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class DEConv_2(nn.Module):
    def __init__(self, dim):
        super(DEConv_2, self).__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.conv1 = nn.Conv2d(dim * 5, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        if isinstance(x, list):
            x = x[-1]

        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias


        w_sum = w1 + w2 + w3 + w4 + w5
        b_sum = b1 + b2 + b3 + b4 + b5


        w_cat = torch.cat([w1, w2, w3, w4, w5], dim=1)
        w_fuse = self.conv1(w_cat)


        res1 = F.conv2d(x, w_sum, bias=b_sum, stride=1, padding=1)
        res2 = F.conv2d(x, w_fuse, bias=b_sum, stride=1, padding=1)
        a = self.sigmoid(res1 + res2)
        out = x + a * res1 + (1 - a) * res2
        return out
