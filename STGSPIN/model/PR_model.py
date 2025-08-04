import torch
from .base_model import BaseModel
import torch.nn.functional as F
import torch.nn as nn
from . import network, base_function, external_function
from util import task
import itertools
from torch.optim.lr_scheduler import LambdaLR
import math


def warmup_cosine_lr_lambda(current_step):
    warmup_steps = 1000
    total_steps = 100000
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    # Cosine decay after warmup
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1. + math.cos(math.pi * min(progress, 1.0)))

class PR(BaseModel):
    def name(self):
        return "PR Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=5, help='# of number of the output scale')
        # 添加 quality_threshold 默认值
        parser.add_argument('--quality_threshold', type=float, default=10.0,
                            help='Threshold to decide whether to use simple G2 path based on image quality score')
        if is_train:
            parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_kl', type=float, default=1.0, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')
            parser.add_argument('--lambda_align', type=float, default=1.0, help='weight for feature alignment loss')
            parser.add_argument('--lambda_contrast', type=float, default=5.0, help='weight for contrastive loss')
        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)

        self.visual_names = ['img_m', 'img_truth', 'img_out', 'merged_image', 'img_out2']
        self.value_names = ['u_m', 'sigma_m', 'u_prior', 'sigma_prior']
        self.model_names = ['ET', 'ES', 'G', 'D', 'cross_attenstion', 'fuse_s', 'fuse_t', 'G2']
        self.loss_names = ['kl_s', 'kl_t', 'app_G2', 'app_G1', 'img_dg', 'ad_l', 'G', 'align', 'contrast','coarse_percep']
        self.distribution = []

        self.net_EQ = network.define_eq(init_type='orthogonal', gpu_ids=opt.gpu_ids)

        self.net_ET = network.define_et(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_ES = network.define_es(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.feature_align_loss = network.define_feature_align_loss()
        self.contrastive_loss = network.define_contrastive_loss()

        self.net_G = network.define_de(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_D = network.define_dis_g(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_cross_attenstion = network.define_attn(init_type='orthogonal', gpu_ids=opt.gpu_ids, in_channels_s=512, in_channels_t=256)
        self.net_fuse_s = network.define_fuse_s(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_fuse_t = network.define_fuse_t(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_G2 = network.define_G2(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.compress_conv = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)
        # （global encoder + attention + teacher fuse）
        for param in self.net_ET.parameters():
            param.requires_grad = False
        for param in self.net_fuse_t.parameters():
            param.requires_grad = False
        for param in self.net_cross_attenstion.parameters():
            param.requires_grad = False
        self.lossNet = network.VGG16FeatureExtractor()
        self.lossNet.cuda(opt.gpu_ids[0])


        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            # define the optimizer
            # self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters()),
            #                                                     filter(lambda p: p.requires_grad, self.net_ET.parameters()),
            #                                                     filter(lambda p: p.requires_grad, self.net_cross_attenstion.parameters()),
            #                                                     filter(lambda p: p.requires_grad, self.net_fuse_s.parameters()),
            #                                                     filter(lambda p: p.requires_grad,self.net_fuse_t.parameters()),
            #                                                     # filter(lambda p: p.requires_grad,self.net_fuse.parameters()),
            #                                                     filter(lambda p: p.requires_grad, self.net_G2.parameters()),
            #                                                     filter(lambda p: p.requires_grad, self.net_ES.parameters())),
            #                                     lr=opt.lr, betas=(0.0, 0.999))

            self.optimizer_G = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_G.parameters()),
                filter(lambda p: p.requires_grad, self.net_fuse_s.parameters()),
                filter(lambda p: p.requires_grad, self.net_G2.parameters()),
                filter(lambda p: p.requires_grad, self.net_ES.parameters())
            ), lr=opt.lr, betas=(0.0, 0.999))
            self.lr_scheduler_G = LambdaLR(self.optimizer_G, lr_lambda=warmup_cosine_lr_lambda)
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


        # load the pretrained model and schedulers
        self.setup(opt)

    def set_input(self, input):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = input['mask']


        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])

        self.img_truth = self.img * 2 - 1

        self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)
        self.scale_mask = task.scale_pyramid(self.mask, self.opt.output_scale)
        self.img_m = self.img_truth * (1 - self.mask) + self.mask

    def test(self):


        self.save_results(self.img_truth, data_name='truth')
        self.save_results(self.img_m, data_name='mask')
        self.image = torch.cat([self.img_m, self.mask], dim=1)

        with torch.no_grad():
            _, t_feats = self.net_ET(self.image)
            fuse_t = self.net_fuse_t(t_feats)

        _, s_feats = self.net_ES(self.image)
        fuse_s = self.net_fuse_s(s_feats)  # [B,256,H/64,W/64]

        distribution = self.net_cross_attenstion(
            s_feats[-1], [f.detach() for f in t_feats]
        )
        mu, sigma = torch.chunk(distribution, 2, dim=1)
        sigma = F.softplus(sigma).clamp(min=1e-6)
        normal = torch.distributions.Normal(mu, sigma)

        for i in range(self.opt.nsampling):

            z = normal.rsample()

            self.img_g = list(
                self.net_G(
                    z,
                    fuse_s, fuse_t,  # fused global / local features
                    s_feats,
                    [f.detach() for f in t_feats]
                )
            )


            merged = self.mask * self.img_g[-1].detach() + (1 - self.mask) * self.img_m
            img_out = self.net_G2(merged, self.mask)

            self.save_results(self.img_g[-1], i, data_name='out1')  # coarse
            self.save_results(img_out, i, data_name='out2')  # refined

        return

    def get_distribution(self, distributions):
        # get distribution
        q_distribution, kl = 0, 0
        self.distribution = []
        for distribution in distributions:
            q_mu, q_sigma = distribution
            m_distribution = torch.distributions.Normal(torch.zeros_like(q_mu), torch.ones_like(q_sigma))
            q_distribution = torch.distributions.Normal(q_mu, q_sigma)

            # kl divergence
            kl += torch.distributions.kl_divergence(q_distribution, m_distribution)
            self.distribution.append([torch.zeros_like(q_mu), torch.ones_like(q_sigma), q_mu, q_sigma])

        return kl

    def forward(self):
        """Run forward processing (two-stage with fuse_s + fuse_t + teacher-skip)."""
        self.loss_align = torch.tensor(0.0, device=self.img.device, requires_grad=True)
        self.loss_contrast = torch.tensor(0.0, device=self.img.device, requires_grad=True)

        with torch.no_grad():
            self.I_mp, self.image_type, self.quality_scores = self.net_EQ(self.img_truth)

        self.refined_mask = self.mask
        if self.refined_mask.shape[-2:] != self.I_mp.shape[-2:]:
            self.refined_mask = F.interpolate(self.refined_mask, size=self.I_mp.shape[-2:],
                                              mode='bilinear', align_corners=False)
        self.image = torch.cat([self.I_mp, self.refined_mask], dim=1)

        quality_threshold = getattr(self.opt, 'quality_threshold', 10.0)
        if self.quality_scores[0] <= quality_threshold:
            self.use_two_stage = False
            self.image = F.interpolate(self.image, size=self.img_truth.shape[2:], mode='bilinear', align_corners=False)
            self.refined_mask = F.interpolate(self.refined_mask, size=self.img_truth.shape[2:], mode='bilinear',
                                              align_corners=False)

            if self.image.shape[1] != self.img_truth.shape[1]:
                self.compress_conv = self.compress_conv.to(self.image.device)
                self.image = self.compress_conv(self.image)

            self.merged_image = self.img_truth * (1 - self.refined_mask) + self.image * self.refined_mask
            self.img_out2 = self.net_G2(self.merged_image, self.refined_mask)
            return

        self.use_two_stage = True
        s_x, self.output2 = self.net_ES(self.image)

        with torch.no_grad():
            t_x, self.output1 = self.net_ET(self.image)

        if self.isTrain:
            self.loss_contrast = self.contrastive_loss(self.output2[-1], self.output1[-1].detach())
        self.loss_align = self.feature_align_loss(self.output2, [f.detach() for f in self.output1])

        fuse_s = self.net_fuse_s(self.output2)
        with torch.no_grad():
            fuse_t = self.net_fuse_t(self.output1)

        self.kl_g_s = self.get_distribution(s_x)
        with torch.no_grad():
            self.kl_g_t = self.get_distribution(t_x)

        distribution = self.net_cross_attenstion(self.output2[-1], [f.detach() for f in self.output1])
        mu, sigma_raw = torch.split(distribution, 256, dim=1)
        sigma = F.softplus(sigma_raw).clamp(min=1e-6)
        z = torch.distributions.Normal(mu, sigma).rsample()

        self.img_g = list(self.net_G(z, fuse_s, fuse_t, self.output2, [f.detach() for f in self.output1]))
        self.img_out = F.interpolate(self.img_g[-1].detach(), size=self.img_truth.shape[2:], mode='bilinear',
                                     align_corners=False)

        self.refined_mask = F.interpolate(self.refined_mask, size=self.img_truth.shape[2:], mode='bilinear',
                                          align_corners=False)
        self.merged_image = self.img_truth * (1 - self.refined_mask) + self.img_out * self.refined_mask
        self.img_out2 = self.net_G2(self.merged_image, self.refined_mask)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # global
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        self.loss_img_dg = self.backward_D_basic(self.net_D, self.img_truth, self.img_g[-1])


    def backward_G(self):
        """Calculate training loss for the generator"""

        total_loss = torch.tensor(0.0, device=self.img.device, requires_grad=True)

        if not getattr(self, 'use_two_stage', True):
            self.loss_kl_s = torch.tensor(0.0, device=self.img.device)
            self.loss_kl_t = torch.tensor(0.0, device=self.img.device)
            self.loss_ad_l = torch.tensor(0.0, device=self.img.device)
            self.loss_app_G1 = torch.tensor(0.0, device=self.img.device)
            self.loss_align = torch.tensor(0.0, device=self.img.device)
            self.loss_contrast = torch.tensor(0.0, device=self.img.device)
            self.loss_coarse_percep = torch.tensor(0.0, device=self.img.device)
        else:
            self.loss_kl_s = self.kl_g_s.mean() * self.opt.lambda_kl * self.opt.output_scale
            self.loss_kl_t = self.kl_g_t.mean() * self.opt.lambda_kl * self.opt.output_scale

            base_function._freeze(self.net_D)
            D_fake_g = self.net_D(self.img_g[-1])
            D_real_g = self.net_D(self.img_truth)
            self.loss_ad_l = self.L2loss(D_fake_g, D_real_g) * self.opt.lambda_g

            loss_app_hole, loss_app_context = 0, 0
            for img_fake_i, img_real_i, mask_i in zip(self.img_g, self.scale_img, self.scale_mask):
                target_size = img_fake_i.shape[2:]
                img_real_i = F.interpolate(img_real_i, size=target_size, mode='bilinear', align_corners=False)
                mask_i = F.interpolate(mask_i, size=target_size, mode='nearest')
                loss_app_hole += self.L1loss(img_fake_i * mask_i, img_real_i * mask_i)
                loss_app_context += self.L1loss(img_fake_i * (1 - mask_i), img_real_i * (1 - mask_i))

            self.loss_app_G1 = (loss_app_hole + loss_app_context) * self.opt.lambda_rec

        # ─── G2 refined output ──────────────────────────────────────────────
        loss_app_hole2 = self.L1loss(self.img_out2 * self.refined_mask, self.img_truth * self.refined_mask)
        loss_app_context2 = self.L1loss(self.img_out2 * (1 - self.refined_mask),
                                        self.img_truth * (1 - self.refined_mask))
        self.loss_app_G2 = (loss_app_hole2 + loss_app_context2) * self.opt.lambda_rec

        real_feats2 = self.lossNet(self.img_truth)
        fake_feats2 = self.lossNet(self.img_out2)
        comp_feats2 = self.lossNet(self.merged_image)
        self.loss_G_style = base_function.style_loss(real_feats2, fake_feats2) + \
                            base_function.style_loss(real_feats2, comp_feats2)

        self.loss_G_content = base_function.perceptual_loss(real_feats2, fake_feats2) + \
                              base_function.perceptual_loss(real_feats2, comp_feats2)

        self.loss_G = 0.1 * self.loss_G_content + 100 * self.loss_G_style

        # ─── NEW: perceptual loss for img_out (coarse) ──────────────────────
        fake_feats_coarse = self.lossNet(self.img_out.detach())
        self.loss_coarse_percep = base_function.perceptual_loss(real_feats2, fake_feats_coarse) * 0.1
        # ─────────────────────────────────────────────────────────────────────

        if getattr(self, 'use_two_stage', True):
            self.loss_align *= getattr(self.opt, 'lambda_align', 1.0)
            self.loss_contrast *= getattr(self.opt, 'lambda_contrast', 1.0)
        else:
            self.loss_align = torch.tensor(0.0, device=self.img.device)
            self.loss_contrast = torch.tensor(0.0, device=self.img.device)

        # 最终 loss 列表加 coarse perceptual loss
        loss_list = [self.loss_kl_s, self.loss_kl_t, self.loss_ad_l, self.loss_app_G1,
                     self.loss_app_G2, self.loss_G, self.loss_align, self.loss_contrast,
                     self.loss_coarse_percep]

        total_loss = sum(loss_list)
        total_loss.backward()

        for loss in loss_list:
            loss.detach_()

    def optimize_parameters(self):
        """update network weights"""
        self.forward()

        self.optimizer_D.zero_grad(set_to_none=True)
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad(set_to_none=True)
        self.backward_G()
        self.optimizer_G.step()
        self.lr_scheduler_G.step()
