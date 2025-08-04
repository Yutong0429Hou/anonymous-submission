import torch
import torch.nn as nn
import torch.nn.functional as F

class RepairInputComposer(nn.Module):
    def __init__(self):
        super(RepairInputComposer, self).__init__()

    def forward(self, image: torch.Tensor,
                image_type,              # 'Imd', 'Imz', 'Img'
                mask: torch.Tensor = None,
                mask_type = None         # 'Xm_s', 'Xm_m' or 0, 1 or tensor
               ):


        if isinstance(image_type, torch.Tensor):
            if image_type.numel() == 1:
                image_type = image_type.item()
            else:
                image_type = image_type[0].item()
        image_type = str(image_type)


        if mask is None:
            return image, image_type


        if mask_type is None:
            mask_type = 'Xm_s'
        elif isinstance(mask_type, torch.Tensor):
            mask_type = mask_type.item()
        if isinstance(mask_type, int):
            mask_type = 'Xm_m' if mask_type == 0 else 'Xm_s'

        if mask_type not in ['Xm_s', 'Xm_m']:
            raise ValueError(f"无效的 mask_type：{mask_type}，应为 'Xm_s' 或 'Xm_m'")


        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask, size=image.shape[-2:], mode='bilinear', align_corners=False)


        out = torch.cat([image, mask], dim=1)
        output_type = f"{image_type}+{mask_type}"
        return out, output_type
