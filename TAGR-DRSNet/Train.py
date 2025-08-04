import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import MuralDamageNet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import matplotlib.pyplot as plt

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = os.path.join(data_path, 'images')
    gt_root = os.path.join(data_path, 'masks')
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1 = model(image)
        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        smooth = 1
        dice = (2 * (input * target).sum() + smooth) / (input.sum() + target.sum() + smooth)
        DSC += float(f'{dice:.4f}')

    return DSC / num1

def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            P1, P2 = model(images)
            loss = structure_loss(P1, gts) + structure_loss(P2, gts)
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_P2_record.update(loss.item(), opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], Loss: {loss_P2_record.show():.4f}')

    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}PolypPVT.pth'))

    global dict_plot
    test1path = './dataset/TestDataset/'
    if (epoch + 1) % 1 == 0:
        for dataset in ['MuralTest', 'WallMaskEval']:
            dice = test(model, test1path, dataset)
            logging.info(f'epoch: {epoch}, dataset: {dataset}, dice: {dice}')
            print(dataset, ':', dice)
            dict_plot[dataset].append(dice)
        meandice = test(model, test_path, 'test')
        dict_plot['test'].append(meandice)
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), os.path.join(save_path, 'PolypPVT.pth'))
            torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}PolypPVT-best.pth'))
            print('########## Best Updated:', best)
            logging.info(f'########## Best Dice Updated: {best}')

def plot_train(dict_plot=None, name=None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    transfuse = {'MuralTest': 0.902, 'WallMaskEval': 0.88, 'test': 0.83}
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        plt.axhline(y=transfuse.get(name[i], 0.8), color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Training Evaluation')
    plt.legend()
    plt.savefig('eval.png')

if __name__ == '__main__':
    dict_plot = {'MuralTest': [], 'WallMaskEval': [], 'test': []}
    name = ['MuralTest', 'WallMaskEval', 'test']
    model_name = 'MuralDamageNet'

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=160)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--augmentation', default=False)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=50)
    parser.add_argument('--train_path', type=str, default='./dataset/TrainDataset')
    parser.add_argument('--test_path', type=str, default='./dataset/TestDataset')
    parser.add_argument('--train_save', type=str, default=f'./model_pth/{model_name}/')

    opt = parser.parse_args()

    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    torch.cuda.set_device(0)
    model = MuralDamageNet().cuda()
    print(model)

    best = 0
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4) if opt.optimizer == 'AdamW' \
                else torch.optim.SGD(model.parameters(), opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = os.path.join(opt.train_path, 'images')
    gt_root = os.path.join(opt.train_path, 'masks')
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.test_path)

    # plot_train(dict_plot, name)
