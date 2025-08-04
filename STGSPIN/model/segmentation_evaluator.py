import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationEvaluator(nn.Module):
    def __init__(self, in_channels=3):

        super(SegmentationEvaluator, self).__init__()


        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )


        self.quality_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.mask_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, xl: torch.Tensor, xm: torch.Tensor):

        assert xl.shape[0] == xm.shape[0], f"Batch size mismatch: {xl.shape[0]} vs {xm.shape[0]}"


        xm_refined_1c = self.refine_conv(xm)
        xm_refined = xm_refined_1c.repeat(1, 3, 1, 1)

        quality_score = self.quality_mlp(xm_refined_1c)
        mask_type_logits = self.mask_classifier(xm_refined_1c)
        mask_type_pred = torch.argmax(mask_type_logits, dim=1)

        return {
            'refined_mask': xm_refined,         # [B, 3, H, W]
            'mask_type': mask_type_pred,        # [B]
            'quality_score': quality_score      # [B, 1]
        }
