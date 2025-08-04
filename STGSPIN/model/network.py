from .base_function import *
from .external_function import SpectralNorm
import torch.nn.functional as F
import torch
from torchvision import models
import functools
import os
import torchvision.transforms as transforms
# from LocalDetailEncoder import SimplifiedDetailEncoder
from model.LocalDetailEncoder import SimplifiedDetailEncoder


from model.ls import DEConv_2
from model.image_evaluator import MuralQualityEvaluator
from model.repair_input_composer import RepairInputComposer
from model.segmentation_evaluator import SegmentationEvaluator
##############################################################################################################
# Network function
##############################################################################################################
def cat_with_resize(out, tensor, dim=1):
    """
    Concatenates two tensors along a specified dimension after resizing the second tensor
    to match the spatial dimensions of the first tensor.

    Args:
        out (torch.Tensor): The first tensor.
        tensor (torch.Tensor): The second tensor to be concatenated.
        dim (int): The dimension along which to concatenate.

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    if out.size(2) != tensor.size(2) or out.size(3) != tensor.size(3):
        tensor = F.interpolate(tensor, size=(out.size(2), out.size(3)), mode='bilinear', align_corners=False)
    return torch.cat([out, tensor], dim=dim)




def define_se(init_type='normal', gpu_ids=[]):
    net = SegmentationEvaluator(in_channels=3)
    return init_net(net, init_type, gpu_ids)



def define_eq(init_type='normal', gpu_ids=[]):

    net = MuralQualityEvaluator()
    return init_net(net, init_type, gpu_ids)

def define_ric(init_type='normal', gpu_ids=[]):
    net = RepairInputComposer()
    return init_net(net, init_type, gpu_ids)

def define_es(init_type='normal', gpu_ids=[]):

    net = Encoder_S()

    return init_net(net, init_type, gpu_ids)

def define_et(init_type='normal', gpu_ids=[]):
    d_model_dict = {
        'enc1': 64,
        'enc3': 128,
    }
    net = Encoder_T(d_model_dict)
    return init_net(net, init_type, gpu_ids)

def define_feature_align_loss():
    return FeatureAlignLoss()

def define_contrastive_loss():
    return ContrastiveLoss(temperature=0.1)

def define_de(init_type='orthogonal', gpu_ids=[]):

    net = Decoder()

    return init_net(net, init_type, gpu_ids)


def define_dis_g(init_type='orthogonal', gpu_ids=[]):

    net = GlobalDiscriminator()

    return init_net(net, init_type, gpu_ids)

def define_attn(init_type='orthogonal', gpu_ids=[], in_channels_s=512, in_channels_t=256):
    net = DMGCACrossAttention(in_channels_s=in_channels_s, in_channels_t=in_channels_t)
    return init_net(net, init_type, gpu_ids)



def define_fuse_s(init_type='orthogonal', gpu_ids=[]):

    net = Trans_conv_s()

    return init_net(net, init_type, gpu_ids)

def define_fuse_t(init_type='orthogonal', gpu_ids=[]):

    net = Trans_conv_t()

    return init_net(net, init_type, gpu_ids)

def define_G2(init_type='orthogonal', gpu_ids=[]):

    net = refine_G2()

    return init_net(net, init_type, gpu_ids)


#############################################################################################################
# Network structure
#############################################################################################################
import torch
import torch.nn as nn


class Encoder_S(nn.Module):
    def __init__(self):
        super(Encoder_S, self).__init__()

        self.nonlinearity = nn.LeakyReLU(0.1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.gate_nonlinearity = nn.Sigmoid()


        self.encoder1 = SimplifiedDetailEncoder(6, 32)
        self.gate1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=4, stride=2, padding=1), self.gate_nonlinearity)

        self.encoder2 = SimplifiedDetailEncoder(32, 64)
        self.gate2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), self.gate_nonlinearity)

        self.encoder3 = SimplifiedDetailEncoder(64, 128)
        self.gate3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), self.gate_nonlinearity)

        self.encoder4 = SimplifiedDetailEncoder(128, 256)
        self.gate4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), self.gate_nonlinearity)

        self.encoder5 = SimplifiedDetailEncoder(256, 512)
        self.gate5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), self.gate_nonlinearity)

        self.encoder6 = SimplifiedDetailEncoder(512, 512)
        self.gate6 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), self.gate_nonlinearity)

    def forward(self, x):
        features = []

        x = self.pool(self.encoder1(x)) * self.gate1(x)
        features.append(x)  # [B, 32, H/2, W/2]

        x = self.pool(self.encoder2(x)) * self.gate2(x)
        features.append(x)  # [B, 64, H/4, W/4]

        x = self.pool(self.encoder3(x)) * self.gate3(x)
        features.append(x)  # [B, 128, H/8, W/8]

        x = self.pool(self.encoder4(x)) * self.gate4(x)
        features.append(x)  # [B, 256, H/16, W/16]

        x = self.pool(self.encoder5(x)) * self.gate5(x)
        features.append(x)  # [B, 512, H/32, W/32]

        x = self.pool(self.encoder6(x)) * self.gate6(x)
        features.append(x)  # [B, 512, H/64, W/64]


        q_mu, q_std = torch.chunk(x, 2, dim=1)
        distribution = [[q_mu, F.softplus(q_std)]]

        return distribution, features




import torch.nn as nn
from model.Spiral_patch import BiMamba2SpiralWrapper

class Encoder_T(nn.Module):
    def __init__(self, d_model_dict):
        super(Encoder_T, self).__init__()

        def depthwise_separable_conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.encoder1 = depthwise_separable_conv(6, 64)
        self.bi_mamba_enc1 = BiMamba2SpiralWrapper(cin=64, cout=64, d_model=d_model_dict['enc1'])

        self.encoder2 = depthwise_separable_conv(64, 128)
        self.bi_mamba_enc2 = BiMamba2SpiralWrapper(cin=128, cout=128, d_model=d_model_dict['enc3'])

        self.encoder3 = depthwise_separable_conv(128, 256)
        self.encoder4 = depthwise_separable_conv(256, 256)

        self.feature_adapt1 = FeatureAdaptBlock(64, 32)
        self.feature_adapt2 = FeatureAdaptBlock(128, 64)
        self.feature_adapt3 = FeatureAdaptBlock(256, 128)


    def forward(self, x):
        feature = []
        distribution = []

        x = self.pool(self.encoder1(x))           # [B, 64, H/2, W/2]
        x = self.bi_mamba_enc1(x)
        feature.append(self.feature_adapt1(x))    # → [B, 32, ...]

        x = self.pool(self.encoder2(x))           # [B, 128, H/4, W/4]
        x = self.bi_mamba_enc2(x)
        feature.append(self.feature_adapt2(x))    # → [B, 64, ...]

        x = self.pool(self.encoder3(x))           # [B, 256, H/8, W/8]
        feature.append(self.feature_adapt3(x))    # → [B, 128, ...]

        x = self.pool(self.encoder4(x))           # [B, 256, H/16, W/16]
        feature.append(x)

        q_mu, q_std = torch.split(x, 128, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])

        return distribution, feature

class FeatureAlignLoss(nn.Module):
    def __init__(self, temperature=0.07, use_cosine=True):
        super().__init__()
        self.temperature = temperature
        self.use_cosine = use_cosine

    def forward(self, feat_s_list, feat_t_list):
        loss = 0
        for fs, ft in zip(feat_s_list, feat_t_list):
            B, C, H, W = fs.shape
            fs = fs.view(B, -1)
            ft = ft.view(B, -1)
            fs = F.normalize(fs, dim=1)
            ft = F.normalize(ft, dim=1)

            if self.use_cosine:
                sim = F.cosine_similarity(fs, ft, dim=1)
                loss += 1 - sim.mean()
            else:
                loss += F.mse_loss(fs, ft)
        return loss / len(feat_s_list)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, feat_q, feat_k):
        """
        feat_q: student features [B, C, H, W]
        feat_k: teacher features [B, C, H, W]
        """

        B, C, H, W = feat_q.shape

        # Flatten and normalize
        q = F.normalize(feat_q.view(B, C, -1), dim=1)   # [B, C, HW]
        k = F.normalize(feat_k.view(B, C, -1), dim=1)   # [B, C, HW]

        # global average pooling
        q = q.mean(dim=2)  # [B, C]
        k = k.mean(dim=2)  # [B, C]

        # similarity matrix: B x B
        logits = torch.matmul(q, k.T) / self.temperature
        labels = torch.arange(B, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        kwargs_out = {'kernel_size': 3, 'padding': 0, 'bias': True}
        kwargs_fuse_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)
        self.gate_nonlinearity = nn.Sigmoid()


        self.s1_proj = SpectralNorm(nn.Conv2d(512, 256, kernel_size=1))
        self.s2_proj = SpectralNorm(nn.Conv2d(512, 256, kernel_size=1))
        self.s3_proj = SpectralNorm(nn.Conv2d(256, 256, kernel_size=1))


        # decoder1
        self.conv1 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.shortcut1 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.model1 = nn.Sequential(self.nonlinearity, self.conv1,
                                    self.nonlinearity, self.conv2)
        self.gateconv1 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.gate1 = nn.Sequential(self.gateconv1, self.gate_nonlinearity)

        # decoder2
        self.conv3 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.shortcut2 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.model2 = nn.Sequential(self.norm(512), self.nonlinearity, self.conv3,
                                    self.norm(512), self.nonlinearity, self.conv4)
        self.gateconv2 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.gate2 = nn.Sequential(self.gateconv2, self.gate_nonlinearity)

        # decoder3
        self.conv5 = SpectralNorm(nn.Conv2d(512, 256, **kwargs))
        self.conv6 = SpectralNorm(nn.ConvTranspose2d(256, 128, **kwargs_short))
        self.shortcut3 = SpectralNorm(nn.ConvTranspose2d(512, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.norm(512), self.nonlinearity, self.conv5,
                                    self.norm(256), self.nonlinearity, self.conv6)
        self.gateconv3 = SpectralNorm(nn.ConvTranspose2d(512, 128, **kwargs_short))
        self.gate3 = nn.Sequential(self.gateconv3, self.gate_nonlinearity)

        # out1
        self.conv_out1 = SpectralNorm(nn.Conv2d(128, 3, **kwargs_out))
        self.model_out1 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1),
                                        self.conv_out1, nn.Tanh())

        # decoder4
        self.conv7 = SpectralNorm(nn.Conv2d(259, 64, **kwargs))
        self.conv8 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs_short))
        self.shortcut4 = SpectralNorm(nn.ConvTranspose2d(259, 64, **kwargs_short))
        self.model4 = nn.Sequential(self.norm(259), self.nonlinearity, self.conv7,
                                    self.norm(64), self.nonlinearity, self.conv8)
        self.gateconv4 = SpectralNorm(nn.ConvTranspose2d(259, 64, **kwargs_short))
        self.gate4 = nn.Sequential(self.gateconv4, self.gate_nonlinearity)

        # out2
        self.conv_out2 = SpectralNorm(nn.Conv2d(64, 3, **kwargs_out))
        self.model_out2 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1),
                                        self.conv_out2, nn.Tanh())

        # decoder5
        self.conv9 = SpectralNorm(nn.Conv2d(131, 32, **kwargs))
        self.conv10 = SpectralNorm(nn.ConvTranspose2d(32, 32, **kwargs_short))
        self.shortcut5 = SpectralNorm(nn.ConvTranspose2d(131, 32, **kwargs_short))
        self.model5 = nn.Sequential(self.norm(131), self.nonlinearity, self.conv9,
                                    self.norm(32), self.nonlinearity, self.conv10)
        self.gateconv5 = SpectralNorm(nn.ConvTranspose2d(131, 32, **kwargs_short))
        self.gate5 = nn.Sequential(self.gateconv5, self.gate_nonlinearity)

        # out3
        self.conv_out3 = SpectralNorm(nn.Conv2d(32, 3, **kwargs_out))
        self.model_out3 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1),
                                        self.conv_out3, nn.Tanh())

        # decoder6
        self.conv11 = SpectralNorm(nn.Conv2d(67, 16, **kwargs))
        self.conv12 = SpectralNorm(nn.ConvTranspose2d(16, 16, **kwargs_short))
        self.shortcut6 = SpectralNorm(nn.ConvTranspose2d(67, 16, **kwargs_short))
        self.model6 = nn.Sequential(self.norm(67), self.nonlinearity, self.conv11,
                                    self.norm(16), self.nonlinearity, self.conv12)
        self.gateconv6 = SpectralNorm(nn.ConvTranspose2d(67, 16, **kwargs_short))
        self.gate6 = nn.Sequential(self.gateconv6, self.gate_nonlinearity)

        # out4
        self.conv_out4 = SpectralNorm(nn.Conv2d(16, 3, **kwargs_out))
        self.model_out4 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1),
                                        self.conv_out4, nn.Tanh())

        # decoder7
        self.conv13 = SpectralNorm(nn.Conv2d(19, 16, **kwargs))
        self.conv14 = SpectralNorm(nn.Conv2d(16, 16, **kwargs))
        self.shortcut7 = SpectralNorm(nn.ConvTranspose2d(19, 16, **kwargs))
        self.model7 = nn.Sequential(self.norm(19), self.nonlinearity, self.conv13,
                                    self.norm(16), self.nonlinearity, self.conv14)
        self.gateconv7 = SpectralNorm(nn.ConvTranspose2d(19, 16, **kwargs))
        self.gate7 = nn.Sequential(self.gateconv7, self.gate_nonlinearity)

        # out5
        self.conv_out5 = SpectralNorm(nn.Conv2d(16, 3, **kwargs_out))
        self.model_out5 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1),
                                        self.conv_out5, nn.Tanh())


        self.fuse_conv1 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_conv2 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_conv3 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_down1 = nn.Sequential(self.fuse_conv1, self.nonlinearity,
                                        self.fuse_conv2, self.nonlinearity,
                                        self.fuse_conv3, self.nonlinearity)

        self.fuse_conv4 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_conv4_1 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_down2 = nn.Sequential(self.fuse_conv4, self.nonlinearity,
                                        self.fuse_conv4_1, self.nonlinearity)

        self.fuse_conv5 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_down3 = nn.Sequential(self.fuse_conv5, self.nonlinearity)

        self.fuse_conv6 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.fuse_up1 = nn.Sequential(self.fuse_conv6, self.norm(128), self.nonlinearity)

        self.fuse_conv8 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.fuse_conv10 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.fuse_conv11 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs_short))
        self.fuse_up2 = nn.Sequential(self.fuse_conv8, self.norm(128), self.nonlinearity,
                                      self.fuse_conv10, self.norm(64), self.nonlinearity,
                                      self.fuse_conv11, self.norm(64), self.nonlinearity)

        self.fuse_conv12 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.fuse_conv13 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.fuse_conv14 = SpectralNorm(nn.Conv2d(64, 32, **kwargs))
        self.fuse_conv15 = SpectralNorm(nn.ConvTranspose2d(32, 32, **kwargs_short))
        self.fuse_conv16 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.fuse_conv17 = SpectralNorm(nn.ConvTranspose2d(32, 32, **kwargs_short))
        self.fuse_up3 = nn.Sequential(self.fuse_conv12, self.norm(128), self.nonlinearity,
                                      self.fuse_conv13, self.norm(64), self.nonlinearity,
                                      self.fuse_conv14, self.norm(32), self.nonlinearity,
                                      self.fuse_conv15, self.norm(32), self.nonlinearity,
                                      self.fuse_conv16, self.norm(32), self.nonlinearity,
                                      self.fuse_conv17, self.norm(32), self.nonlinearity)

    # -------------------------------------------------------------
    # forward
    # -------------------------------------------------------------
    def forward(self, x, fuse_s, fuse_t, s_feats=None, t_feats=None):

        if s_feats is not None:
            s_1 = self.s1_proj(s_feats[-1])  # (B,256,H/64,W/64)
            s_2 = self.s2_proj(s_feats[-2])  # (B,256,H/32,W/32)
            s_3 = self.s3_proj(s_feats[-3])  # (B,256,H/16,W/16)
        else:
            s_1 = self.fuse_down1(fuse_s)  # 256
            s_2 = self.fuse_down2(fuse_s)  # 256
            s_3 = self.fuse_down3(fuse_s)  # 256


        t_1 = self.fuse_up1(fuse_t)  # 128  (H/64)
        t_2 = self.fuse_up2(fuse_t)  # 64   (H/32)
        t_3 = self.fuse_up3(fuse_t)  # 32   (H/16)


        results = []
        out = x

        # ---------- Stage-1 ----------
        out = cat_with_resize(out, s_1, dim=1)
        out = self.nonlinearity(self.model1(out) + self.shortcut1(out)) * self.gate1(out)

        # ---------- Stage-2 ----------
        out = cat_with_resize(out, s_2, dim=1)
        out = self.nonlinearity(self.model2(out) + self.shortcut2(out)) * self.gate2(out)

        # ---------- Stage-3 ----------
        out = cat_with_resize(out, s_3, dim=1)
        out = self.nonlinearity(self.model3(out) + self.shortcut3(out)) * self.gate3(out)

        # ---------- Output-1 ----------
        output = self.model_out1(out)  # 3 ch
        results.append(output)

        # ---------- Stage-4  (128+3+128 = 259) ----------
        out = cat_with_resize(out, output, dim=1)
        out = cat_with_resize(out, t_1, dim=1)
        out = self.nonlinearity(self.model4(out) + self.shortcut4(out)) * self.gate4(out)

        # ---------- Output-2 ----------
        output = self.model_out2(out)  # 3 ch
        results.append(output)

        # ---------- Stage-5  (64+3+64 = 131) ----------
        out = cat_with_resize(out, output, dim=1)
        out = cat_with_resize(out, t_2, dim=1)
        out = self.nonlinearity(self.model5(out) + self.shortcut5(out)) * self.gate5(out)

        # ---------- Output-3 ----------
        output = self.model_out3(out)
        results.append(output)

        # ---------- Stage-6 ----------
        out = cat_with_resize(out, output, dim=1)
        out = cat_with_resize(out, t_3, dim=1)
        out = self.nonlinearity(self.model6(out) + self.shortcut6(out)) * self.gate6(out)

        # ---------- Output-4 ----------
        output = self.model_out4(out)  # 3 ch
        results.append(output)

        # ---------- Stage-7 ----------
        out = cat_with_resize(out, output, dim=1)
        out = self.nonlinearity(self.model7(out) + self.shortcut7(out)) * self.gate7(out)

        # ---------- Output-5 (final) ----------
        output = self.model_out5(out)
        results.append(output)

        return results


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # encoder0
        self.conv1 = SpectralNorm(nn.Conv2d(3, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.bypass1 = SpectralNorm(nn.Conv2d(3, 32, **kwargs_short))
        self.model1 = nn.Sequential(self.conv1, self.nonlinearity, self.conv2, self.pool)
        self.shortcut1 = nn.Sequential(self.pool, self.bypass1)

        # encoder1
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs))
        self.bypass2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_short))
        self.model2 = nn.Sequential(self.nonlinearity, self.conv3, self.nonlinearity, self.conv4)
        self.shortcut2 = nn.Sequential(self.bypass2)

        # encoder2
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 128, **kwargs))
        self.bypass3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.nonlinearity, self.conv5, self.nonlinearity, self.conv6)
        self.shortcut3 = nn.Sequential(self.bypass3)

        # encoder3
        self.conv7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass4 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model4 = nn.Sequential(self.nonlinearity, self.conv7, self.nonlinearity, self.conv8)
        self.shortcut4 = nn.Sequential(self.bypass4)

        # encoder4
        self.conv9 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model5 = nn.Sequential(self.nonlinearity, self.conv9, self.nonlinearity, self.conv10)
        self.shortcut5 = nn.Sequential(self.bypass5)

        # encoder5
        self.conv11 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv12 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass6 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model6 = nn.Sequential(self.nonlinearity, self.conv11, self.nonlinearity, self.conv12)
        self.shortcut6 = nn.Sequential(self.bypass6)

        self.concat = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 1, kernel_size=3, padding=1)),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.model1(x) + self.shortcut1(x)
        x = self.pool(self.model2(x)) + self.pool(self.shortcut2(x))
        x = self.pool(self.model3(x)) + self.pool(self.shortcut3(x))
        out = self.pool(self.model4(x)) + self.pool(self.shortcut4(x))
        out = self.pool(self.model5(out)) + self.pool(self.shortcut5(out))
        out = self.model6(out) + self.shortcut6(out)
        out = self.concat(self.nonlinearity(out))
        return out



class DMGCACrossAttention(nn.Module):
    def __init__(self, in_channels_s=512, in_channels_t=256, reduction=8):
        super(DMGCACrossAttention, self).__init__()
        self.reduction = reduction

        self.scale_extractor_t = DEConv_2(in_channels_t)

        self.query_convs = nn.Sequential(
            nn.Conv2d(in_channels_t, in_channels_s // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.key_convs = nn.Sequential(
            nn.Conv2d(in_channels_s, in_channels_s // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.value_conv = nn.Conv2d(in_channels_s, in_channels_s, kernel_size=1, bias=False)

        self.gate = nn.Parameter(torch.ones(1, in_channels_s, 1, 1))


        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_s, x_t):

        if isinstance(x_s, list):
            x_s = x_s[-1]
        if isinstance(x_t, list):
            x_t = x_t[-1]

        B, C_s, H_s, W_s = x_s.size()

        t_feat = self.scale_extractor_t([x_t]*5)


        if t_feat.shape[2:] != x_s.shape[2:]:
            t_feat = F.interpolate(t_feat, size=(H_s, W_s), mode='bilinear', align_corners=False)

        # query, key, value
        q = self.query_convs(t_feat).view(B, -1, H_s * W_s).permute(0, 2, 1)  # [B, N, C_q]
        k = self.key_convs(x_s).view(B, -1, H_s * W_s)                        # [B, C_k, N]
        v = self.value_conv(x_s).view(B, -1, H_s * W_s)                      # [B, C_v, N]

        # Attention
        attention = torch.bmm(q, k)  # [B, N, N]
        attention = self.softmax(attention)

        out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C_v, N]
        out = out.view(B, C_s, H_s, W_s)

        out = self.gate * out + x_s
        return out


class Trans_conv_s(nn.Module):
    def __init__(self):
        super(Trans_conv_s, self).__init__()
        kwargs_short = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)


        self.conv1 = SpectralNorm(nn.Conv2d(32, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.model1 = nn.Sequential(self.conv1, self.norm(128), self.nonlinearity, self.conv2, self.norm(256), self.nonlinearity)

        self.conv3 = SpectralNorm(nn.Conv2d(64, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.model2 = nn.Sequential(self.conv3, self.norm(256), self.nonlinearity)

        self.conv4 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.model3 = nn.Sequential(self.conv4, self.norm(256), self.nonlinearity)

        self.conv5 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv6 = SpectralNorm(nn.ConvTranspose2d(256, 256,**kwargs_short))
        self.model4 = nn.Sequential(self.conv5, self.norm(256), self.nonlinearity, self.conv6, self.norm(256), self.nonlinearity)

        self.conv7 = SpectralNorm(nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv8 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv9 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv10 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.model5 = nn.Sequential(self.conv7, self.norm(256), self.nonlinearity, self.conv8, self.norm(256),self.nonlinearity,
                                    self.conv9, self.norm(256), self.nonlinearity, self.conv10, self.norm(256),self.nonlinearity)

        self.conv11 = SpectralNorm(nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv12 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv13 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv14 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv15 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv16 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.model6 = nn.Sequential(self.conv11, self.norm(256), self.nonlinearity, self.conv12, self.norm(256),self.nonlinearity,
                                    self.conv13, self.norm(256), self.nonlinearity, self.conv14, self.norm(256),self.nonlinearity,
                                    self.conv15, self.norm(256), self.nonlinearity, self.conv16, self.norm(256),self.nonlinearity)

        self.conv17 = SpectralNorm(nn.Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1)))
        self.down = nn.Sequential(self.conv17, self.norm(256), self.nonlinearity)
    def forward(self, feature1):
        x1 = self.model1(feature1[0])
        x2 = self.model2(feature1[1])
        x3 = self.model3(feature1[2])
        x4 = self.model4(feature1[3])
        x5 = self.model5(feature1[4])
        x6 = self.model6(feature1[5])
        x_fuse = torch.cat([x1, x2, x3, x4, x5, x6], 1)
        x = self.down(x_fuse)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn.utils import spectral_norm as SpectralNorm

class Trans_conv_t(nn.Module):
    def __init__(self):
        super(Trans_conv_t, self).__init__()
        kwargs_short = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)

        self.model1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(32, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(128, affine=True),
            self.nonlinearity,
            SpectralNorm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(256, affine=True),
            self.nonlinearity
        )

        self.model2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(256, affine=True),
            self.nonlinearity
        )

        self.model3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)),
            nn.InstanceNorm2d(256, affine=True),
            self.nonlinearity
        )

        self.model4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(256, 256, kernel_size=1, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(256, affine=True),
            self.nonlinearity,
            SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short)),
            nn.InstanceNorm2d(256, affine=True),
            self.nonlinearity
        )


        self.conv17 = SpectralNorm(nn.Conv2d(1024, 256, kernel_size=1, stride=1))
        self.down = nn.Sequential(
            self.conv17,
            nn.InstanceNorm2d(256, affine=True),
            self.nonlinearity
        )

    def forward(self, feature1):
        x1 = self.model1(feature1[0])
        x2 = self.model2(feature1[1])
        x3 = self.model3(feature1[2])
        x4 = self.model4(feature1[3])

        target_size = (256, 256)
        x1 = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=target_size, mode='bilinear', align_corners=False)

        x_fuse = torch.cat([x1, x2, x3, x4], dim=1)  # [B, 1024, 256, 256]
        out = self.down(x_fuse)  # [B, 256, 256, 256]
        return out


class refine_G2(nn.Module):
    def __init__(self):
        super(refine_G2, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_3 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_5 = {'kernel_size': 5, 'stride': 1, 'padding': 2}
        kwargs_7 = {'kernel_size': 7, 'stride': 1, 'padding': 3}
        kwargs_up = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.gate_nonlinearity = nn.Sigmoid()
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)

        # encoder1
        self.conv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.shortcut1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))
        self.model1 = nn.Sequential(self.conv1, self.norm(32), self.nonlinearity, self.conv2)
        self.gateconv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))
        self.gate1 = nn.Sequential(self.gateconv1, self.gate_nonlinearity)

        # encoder2
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_down))
        self.shortcut2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_down))
        self.model2 = nn.Sequential(self.norm(32), self.nonlinearity, self.conv3, self.norm(32), self.nonlinearity, self.conv4)
        self.gateconv2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_down))
        self.gate2 = nn.Sequential(self.gateconv2, self.gate_nonlinearity)

        # encoder3
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.shortcut3 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.model3 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv5, self.norm(64), self.nonlinearity, self.conv6)
        self.gateconv3 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.gate3 = nn.Sequential(self.gateconv3, self.gate_nonlinearity)

        # encoder4
        self.conv7 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_down))
        self.shortcut4 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_down))
        self.model4 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv7, self.norm(64), self.nonlinearity, self.conv8)
        self.gateconv4 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_down))
        self.gate4 = nn.Sequential(self.gateconv4, self.gate_nonlinearity)

        # encoder5
        self.conv9 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.model5 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv9, self.norm(128), self.nonlinearity, self.conv10)
        self.gateconv5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.gate5 = nn.Sequential(self.gateconv5, self.gate_nonlinearity)

        #Multi_conv
        self.conv_3 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_3))
        self.multi_3 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv_3)
        self.conv_5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_5))
        self.multi_5 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv_5)
        self.conv_7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_7))
        self.multi_7 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv_7)

        # decoder1
        self.de_conv1 = SpectralNorm(nn.Conv2d(384, 384, **kwargs))
        self.de_conv2 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.de_shortcut1 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.de_model1 = nn.Sequential(self.norm(384), self.nonlinearity, self.de_conv1, self.norm(384), self.nonlinearity, self.de_conv2)
        self.de_gateconv1 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.de_gate1 = nn.Sequential(self.de_gateconv1, self.gate_nonlinearity)

        # decoder2
        self.de_conv3 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.de_conv4 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs))
        self.de_shortcut2 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs))
        self.de_model2 = nn.Sequential(self.norm(128), self.nonlinearity, self.de_conv3, self.norm(128), self.nonlinearity,
                                    self.de_conv4)
        self.de_gateconv2 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs))
        self.de_gate2 = nn.Sequential(self.de_gateconv2, self.gate_nonlinearity)

        self.attn1 = Auto_Attn(128)

        # decoder3
        self.de_conv5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.de_conv6 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_up))
        self.de_shortcut3 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_up))
        self.de_model3 = nn.Sequential(self.norm(128), self.nonlinearity, self.de_conv5, self.norm(128), self.nonlinearity,
                                    self.de_conv6)
        self.de_gateconv3 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_up))
        self.de_gate3 = nn.Sequential(self.de_gateconv3, self.gate_nonlinearity)

        # decoder4
        self.de_conv7 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.de_conv8 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs))
        self.de_shortcut4 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs))
        self.de_model4 = nn.Sequential(self.norm(64), self.nonlinearity, self.de_conv7, self.norm(64), self.nonlinearity,
                                    self.de_conv8)
        self.de_gateconv4 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs))
        self.de_gate4 = nn.Sequential(self.de_gateconv4, self.gate_nonlinearity)

        self.attn2 = Auto_Attn(64)

        # decoder5
        self.de_conv9 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.de_conv10 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_up))
        self.de_shortcut5 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_up))
        self.de_model5 = nn.Sequential(self.norm(64), self.nonlinearity, self.de_conv9, self.norm(64), self.nonlinearity,
                                    self.de_conv10)
        self.de_gateconv5 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_up))
        self.de_gate5 = nn.Sequential(self.de_gateconv5, self.gate_nonlinearity)

        self.out = SpectralNorm(nn.Conv2d(32, 3, **kwargs))
        self.model_out = nn.Sequential(self.nonlinearity, self.out, nn.Tanh())

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        feature = []
        x = self.nonlinearity(self.model1(x) + self.shortcut1(x)) * self.gate1(x)
        feature.append(x)
        x = self.nonlinearity(self.model2(x) + self.shortcut2(x)) * self.gate2(x)
        feature.append(x)
        x = self.nonlinearity(self.model3(x) + self.shortcut3(x)) * self.gate3(x)
        feature.append(x)
        x = self.nonlinearity(self.model4(x) + self.shortcut4(x)) * self.gate4(x)
        feature.append(x)
        x = self.nonlinearity(self.model5(x) + self.shortcut5(x)) * self.gate5(x)
        feature.append(x)
        multi1 = self.multi_3(x)
        multi2 = self.multi_5(x)
        multi3 = self.multi_7(x)
        fuse = torch.cat([multi1, multi2, multi3], 1)
        x = self.nonlinearity(self.de_model1(fuse) + self.de_shortcut1(fuse)) * self.de_gate1(fuse)
        x = self.nonlinearity(self.de_model2(x) + self.de_shortcut2(x)) * self.de_gate2(x)
        x = self.attn1(x, feature[3], mask)
        x = self.nonlinearity(self.de_model3(x) + self.de_shortcut3(x)) * self.de_gate3(x)
        x = self.nonlinearity(self.de_model4(x) + self.de_shortcut4(x)) * self.de_gate4(x)
        x = self.attn2(x, feature[1], mask)
        x = self.nonlinearity(self.de_model5(x) + self.de_shortcut5(x)) * self.de_gate5(x)
        x = self.model_out(x)


        return x


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
class FeatureAdaptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(FeatureAdaptBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        attn = self.attention(x)
        return x * attn