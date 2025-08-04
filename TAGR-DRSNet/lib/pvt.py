from lib.pvtv2 import pvt_v2_b2
import os

import torch.nn as nn


import torch
import torch.nn.functional as F

def feature_entropy(feature_map):
    """
    计算特征图的熵，用于衡量特征图中包含的信息量。
    熵越高，信息越杂乱；熵越低，信息越有序。
    """
    p = F.softmax(feature_map, dim=1)
    log_p = torch.log(p + 1e-8)
    entropy = -torch.sum(p * log_p, dim=1, keepdim=True)
    entropy_mean = entropy.mean(dim=[2, 3], keepdim=True)
    return entropy_mean

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(self.pool(x))
        return x * weight


class EAFFM(nn.Module):
    def __init__(self, channel):
        super(EAFFM, self).__init__()
        self.se2 = SEBlock(channel)
        self.se3 = SEBlock(channel)
        self.se4 = SEBlock(channel)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_fuse = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x4, x3, x2):
        x4_se = self.se4(x4)
        x3_se = self.se3(x3)
        x2_se = self.se2(x2)
        x4_up = self.upsample(self.upsample(x4_se))
        x3_up = self.upsample(x3_se)
        entropy_x2 = feature_entropy(x2_se)
        entropy_x3 = feature_entropy(x3_up)
        entropy_x4 = feature_entropy(x4_up)
        entropy = torch.cat([entropy_x2, entropy_x3, entropy_x4], dim=1)
        weights = F.softmax(-entropy, dim=1)
        fused_feature = weights[:, 0:1] * x2_se + weights[:, 1:2] * x3_up + weights[:, 2:] * x4_up
        fused_feature = self.conv_fuse(fused_feature)
        output = fused_feature + x2
        return output

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class AGRM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(AGRM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = AdaptiveMultiLayerGCN(num_state=self.num_s, num_node=self.num_n, num_layers=2)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.no_spatial = no_spatial
        self.cw = self.AttentionGate()
        self.hc = self.AttentionGate()
        if not no_spatial:
            self.hw = self.AttentionGate()

    class AttentionGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            return self.conv(x)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3)  # C,H,W -> H,C,W
        x_out1 = self.cw(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3)

        x_perm2 = x.permute(0, 3, 2, 1)  # C,H,W -> W,H,C
        x_out2 = self.hc(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1)

        if not self.no_spatial:
            x_out3 = self.hw(x)

            out = (x_out1 + x_out2 + x_out3) / 3
        else:
            out = (x_out1 + x_out2) / 2

        return out * x

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, reduction=4):
        super(DynamicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction,
                      out_channels * in_channels * kernel_size * kernel_size, 1)
        )

    def forward(self, x):
        B, C, H, W = x.size()
        weight = self.fc(self.pool(x))
        weight = weight.view(B * self.out_channels, C, self.kernel_size, self.kernel_size)
        x = x.view(1, B * C, H, W)
        out = F.conv2d(x, weight, bias=None, padding=self.padding, groups=B)
        out = out.view(B, self.out_channels, H, W)
        return out

class EnhancedGCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False, dropout=0.1):
        super(EnhancedGCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)
        self.layer_norm = nn.LayerNorm(num_state)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        h = self.dropout(h)
        h = self.layer_norm(h + residual)
        return h


class MultiLayerEnhancedGCN(nn.Module):
    def __init__(self, num_state, num_node, num_layers=2, bias=False, dropout=0.1):
        super(MultiLayerEnhancedGCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()

        for i in range(num_layers):
            self.gcn_layers.append(EnhancedGCN(num_state, num_node, bias, dropout))

    def forward(self, x):
        for layer in self.gcn_layers:
            x = layer(x)
        return x


class AdaptiveMultiLayerGCN(nn.Module):
    def __init__(self, num_state, num_node, num_layers=2, bias=False, dropout=0.1):
        super(AdaptiveMultiLayerGCN, self).__init__()
        self.adjacency_learner = nn.Parameter(torch.randn(num_node, num_node))
        self.multi_gcn = MultiLayerEnhancedGCN(num_state, num_node, num_layers, bias, dropout)

    def forward(self, x):
        adj = torch.softmax(self.adjacency_learner, dim=-1)
        x_adapted = torch.matmul(adj, x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.multi_gcn(x_adapted)

class MuralDamageNet(nn.Module):
    def __init__(self, channel=32):
        super(MuralDamageNet, self).__init__()
        self.backbone = pvt_v2_b2()
        path = './pretrained_pth/pretrained.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.EAFFM = EAFFM(channel)
        self.coord_att = CoordAtt(64, 64)
        self.triplet = TripletAttention()
        self.texture_enhancer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.AGRM = AGRM()
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)  # 下采样0.5倍
        self.out_AGRM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

    def forward(self, x):

        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x1 = self.coord_att(x1)
        x1 = self.triplet(x1)
        texture_mask = self.texture_enhancer(x1)
        cim_feature = texture_mask * x1

        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4)  
        cfm_feature = self.EAFFM(x4_t, x3_t, x2_t)

        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.AGRM(cfm_feature, T2)

        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_AGRM(sam_feature)
        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
        return prediction1_8, prediction2_8


if __name__ == '__main__':
    model = MuralDamageNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
