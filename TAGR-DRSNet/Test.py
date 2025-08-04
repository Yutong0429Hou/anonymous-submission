import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc
from lib.pvt import MuralDamageNet
from utils.dataloader import test_dataset
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/MuralDamageNet/PolypPVT.pth', help='path to the trained model')
    opt = parser.parse_args()

    # Load model
    model = MuralDamageNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    # Set test data paths
    image_root = './data/test/images/'
    gt_root = './data/test/masks/'
    save_path = './result_map/PolypPVT/test/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.cuda()
        P1, P2 = model(image)
        res = F.upsample(P1 + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        cv2.imwrite(os.path.join(save_path, name), res * 255)

    print('Testing finished!')
