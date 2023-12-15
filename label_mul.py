import os
import cv2
import math
import torch
import numpy as np
from model.pytorch_msssim import ssim_matlab
from model.RIFE import Model
from lpips import LPIPS
import _util.distance_transform_v0 as udist
from pandas import DataFrame
from tqdm import tqdm
import glob
import multiprocessing as mp
from multiprocessing import Pool
import tqdm.contrib.concurrent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# 定义path2tensor函数
def path2tensor(path: str, device):
    img = cv2.imread(path)
    img = cv2.resize(img, (960, 576))  # fixed resize_param
    tensor = (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.).float().unsqueeze(0)
    return tensor

def proc_init():
    global model, lpips_loss, chamfer_loss
    model = Model()
    model.load_model('train_log', -1)
    model.eval()
    model.device()
    
    lpips_loss = LPIPS(net='vgg').to(device)
    chamfer_loss = udist.ChamferDistance2dMetric(t=2.0, sigma=1.0).to(device)


# 定义处理单个图像的函数
def process_image(args):
    global model, lpips_loss, chamfer_loss
    scale, p = args
    # 在每个进程中加载模型
    

    if os.path.exists(p + '/frame1.jpg'):
        imgpaths = [p + '/frame1.jpg', p + '/frame2.jpg', p + '/frame3.jpg']
    elif os.path.exists(p + '/im0.png'):
        imgpaths = [p + '/im0.png', p + '/im1.png', p + '/im2.png']
    else:
        return None

    img0 = path2tensor(imgpaths[0], device)
    img1 = path2tensor(imgpaths[2], device)
    gt = path2tensor(imgpaths[1], device)

    with torch.autocast(device_type='cuda'):
        reuse_things = model.reuse(img0, img1, scale)
        pred = model.inference(img0, img1, reuse_things)

        chamfer = chamfer_loss(pred, gt).detach().cpu().numpy().mean()
        lpips = lpips_loss.forward((pred.flip(0) - 0.5) / 0.5, (gt.flip(0) - 0.5) / 0.5).detach().cpu().numpy().mean()
        ssim = ssim_matlab(torch.round(gt * 255.) / 255., torch.round(pred * 255) / 255.).detach().cpu().numpy().mean()
        pred = (torch.round(pred[0] * 255.) / 255.).detach().cpu().numpy().transpose(1, 2, 0)  # H, W, C
        gt = (torch.round(gt[0] * 255.) / 255.).detach().cpu().numpy().transpose(1, 2, 0)
        psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())

    return {'scale': scale, 'path': p, 'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 'chamfer': chamfer}

# 主函数
def main():
    dirs = ['/root/autodl-tmp/animedata/anime_dataset', 
            '/root/autodl-tmp/animedata/test_2k_original', 
            '/root/autodl-tmp/animedata/train_10k']
    paths = list()
    for dir in dirs:
        paths.extend(glob.glob(f"{dir}/*"))

    scale_list = [(832, 1664), (704, 1408), (576, 1152), (448, 896), (320, 640), (256, 512), (192, 384)]
    scale_list = [(256, 512), (192, 384)]  # test

    # 使用多进程
    mp.set_start_method('spawn')
    with Pool(processes=2, initializer=proc_init) as pool:
        results = []
        for scale in scale_list:
            # 使用tqdm.contrib.concurrent.process_map来处理进度条
            scale_results = tqdm.contrib.concurrent.process_map(process_image, [(scale, p) for p in paths], chunksize=2, pool=pool)
            results.extend([res for res in scale_results if res is not None])

    # 保存结果
    loss_df = DataFrame(results)
    loss_df.to_csv('resize_log.csv')

if __name__ == '__main__':
    main()
