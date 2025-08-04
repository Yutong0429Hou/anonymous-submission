import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from model.enlightengan import _EnlightenGAN as EG
from model.NIQE import niqe
from piq.brisque import brisque
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def get_similarity_scores(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(img1_gray, img2_gray, data_range=255)
    psnr_score = psnr(img1, img2, data_range=255)
    return ssim_score, psnr_score


class EnlightenGAN:
    def __init__(self, device='cuda'):
        self.model = EG().to(device)
        ckpt_path = os.path.join(os.path.dirname(__file__), 'EnlightenGAN.pth')
        print('加载权重：', ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, low_img: np.ndarray):
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        low_img = (torch.from_numpy(low_img).permute(2, 0, 1) / 255.0).float().to(self.device)
        low_img = (low_img - 0.5) / 0.5
        r, g, b = low_img[0] + 1, low_img[1] + 1, low_img[2] + 1
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        A_gray = A_gray.unsqueeze(0).unsqueeze(0)
        low_img = low_img.unsqueeze(0)
        out_img = self.model(low_img, A_gray)[0]
        out_img = out_img * 0.5 + 0.5
        out_img = out_img.cpu().numpy() * 255
        out_img = out_img.transpose(1, 2, 0).astype(np.uint8)
        out_img = np.clip(out_img, 0, 255)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        return out_img


class MuralQualityEvaluator(nn.Module):
    def __init__(self, device='cuda'):
        super(MuralQualityEvaluator, self).__init__()
        self.device = device
        self.enhancer = EnlightenGAN(device=device)

        self.to_pil = transforms.ToPILImage()
        self.img_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def compute_quality_scores_np(self, img_np):
        img_np_resized = cv2.resize(img_np, (256, 256))
        gray = cv2.cvtColor(img_np_resized, cv2.COLOR_RGB2GRAY)
        gray_np = gray.astype(np.float32) / 255.0
        rgb_tensor = torch.tensor(img_np_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        niqe_score = niqe(gray_np).item()
        brisque_score = brisque(rgb_tensor).item()
        return niqe_score + brisque_score

    def forward(self, image_tensor):

        batch_size = image_tensor.shape[0]
        selected_tensors = []
        image_types = []
        quality_scores = []

        for i in range(batch_size):
            img = image_tensor[i].detach().cpu()
            img_np = ((img.permute(1, 2, 0) + 1) * 127.5).numpy().astype(np.uint8)


            enhanced_np = self.enhancer(img_np)
            ssim_score, _ = get_similarity_scores(img_np, enhanced_np)

            if ssim_score > 0.85:
                selected_np = enhanced_np
                image_type = 'Imz'
            else:
                selected_np = img_np
                image_type = 'Imd'

            selected_tensor = self.img_transform(self.to_pil(selected_np)).unsqueeze(0).to(self.device)
            selected_tensors.append(selected_tensor)
            image_types.append(image_type)


            total_quality_score = self.compute_quality_scores_np(selected_np)
            quality_scores.append(total_quality_score)

        output_tensor = torch.cat(selected_tensors, dim=0)
        return output_tensor, image_types, quality_scores
