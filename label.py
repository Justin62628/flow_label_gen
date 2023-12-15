import os
import cv2
import math
import torch
import numpy as np
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab
from model.RIFE import Model
from lpips import LPIPS
import _util.distance_transform_v0 as udist
from pandas import DataFrame
from tqdm import tqdm
import glob
import argparse


parser = argparse.ArgumentParser(description='Process scales for image processing.')
parser.add_argument('--scales', nargs='+', help='List of scales in the format width,height. Example: --scales 256,512 192,384')

args = parser.parse_args()

# 解析scales参数
if args.scales:
    scale_list = [tuple(map(int, scale.split(','))) for scale in args.scales]
else:
    scale_list = [(832, 1664), (704, 1408), (576, 1152), (448, 896), (320, 640), (256, 512), (192, 384)]
resize_param = (960, 576)  # fixed, corresponding to `model.reuse`


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

model = Model()
model.load_model('train_log', -1)
model.eval()
model.device()

lpips_loss = LPIPS(net='vgg').to(device)
chamfer_loss = udist.ChamferDistance2dMetric(t=2.0, sigma=1.0).to(device)

with open("paths.txt", "r", encoding='utf-8') as r:
    paths = r.readlines()


def path2tensor(path: str):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, resize_param)
    tensor = (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.).float().unsqueeze(0)
    return tensor


for scale in tqdm(scale_list):
    
    # chamfer_list = []
    # lpips_list = []
    # ssim_list = []
    # psnr_list = []
    loss_list = list()
    
    for p in tqdm(paths):
        p = p.strip()
        if os.path.exists(p + '/frame1.jpg'):
            imgpaths = [p + '/frame1.jpg', p + '/frame2.jpg', p + '/frame3.jpg']
        elif os.path.exists(p + '/im0.png'):
            imgpaths = [p + '/im0.png', p + '/im1.png', p + '/im2.png']
        else:
            continue

        img0 = path2tensor(imgpaths[0])
        img1 = path2tensor(imgpaths[2])
        gt = path2tensor(imgpaths[1])

        with torch.autocast(device_type='cuda'):
            reuse_things = model.reuse(img0, img1, scale)
            pred = model.inference(img0, img1, reuse_things)

            chamfer = chamfer_loss(pred, gt).detach().cpu().numpy().mean()
            # chamfer_list.append(chamfer)

            lpips = lpips_loss.forward((pred.flip(0) - 0.5) / 0.5, (gt.flip(0) - 0.5) / 0.5).detach().cpu().numpy().mean()
            # lpips_list.append(lpips)

            ssim = ssim_matlab(torch.round(gt * 255.) / 255., torch.round(pred * 255) / 255.).detach().cpu().numpy().mean()
            # ssim_list.append(ssim)

            pred = (torch.round(pred[0] * 255.) / 255.).detach().cpu().numpy().transpose(1, 2, 0)  # H, W, C
            gt = (torch.round(gt[0] * 255.) / 255.).detach().cpu().numpy().transpose(1, 2, 0)
            psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
            # psnr_list.append(psnr)

        loss_list.append({'scale': scale, 'path': p, 'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 'chamfer': chamfer})

    loss_df = DataFrame(loss_list)
    loss_df.to_csv(f'resize_{resize_param}_scale_{scale}_log.csv')
