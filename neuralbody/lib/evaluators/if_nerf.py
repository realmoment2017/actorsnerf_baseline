#import numpy as np
#from lib.config import cfg
#from skimage.measure import compare_ssim
#import os
#import cv2
#from termcolor import colored
#
#
#class Evaluator:
#    def __init__(self):
#        self.mse = []
#        self.psnr = []
#        self.ssim = []
#
#    def psnr_metric(self, img_pred, img_gt):
#        mse = np.mean((img_pred - img_gt)**2)
#        psnr = -10 * np.log(mse) / np.log(10)
#        return psnr
#
#    def ssim_metric(self, img_pred, img_gt, batch):
#        #if not cfg.eval_whole_img:
#        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
#        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
#        mask_at_box = mask_at_box.reshape(H, W)
#        # crop the object region
#        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
#        img_pred = img_pred[y:y + h, x:x + w]
#        img_gt = img_gt[y:y + h, x:x + w]
#
#        result_dir = os.path.join(cfg.result_dir, 'comparison')
#        os.system('mkdir -p {}'.format(result_dir))
#        frame_index = batch['frame_index'].item()
#        view_index = batch['cam_ind'].item()
#        cv2.imwrite(
#            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
#                                                   view_index),
#            (img_pred[..., [2, 1, 0]] * 255))
#        cv2.imwrite(
#            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
#                                                      view_index),
#            (img_gt[..., [2, 1, 0]] * 255))
#
#        # compute the ssim
#        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
#        return ssim
#
#    def evaluate(self, output, batch):
#        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
#        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
#
#        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
#        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
#        mask_at_box = mask_at_box.reshape(H, W)
#        # convert the pixels into an image
#        white_bkgd = int(cfg.white_bkgd)
#        img_pred = np.zeros((H, W, 3)) + white_bkgd
#        img_pred[mask_at_box] = rgb_pred
#        img_gt = np.zeros((H, W, 3)) + white_bkgd
#        img_gt[mask_at_box] = rgb_gt
#
#        #if cfg.eval_whole_img:
#        rgb_pred = img_pred
#        rgb_gt = img_gt
#
#        mse = np.mean((rgb_pred - rgb_gt)**2)
#        self.mse.append(mse)
#
#        psnr = self.psnr_metric(rgb_pred, rgb_gt)
#        self.psnr.append(psnr)
#
#        rgb_pred = img_pred
#        rgb_gt = img_gt
#        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
#        self.ssim.append(ssim)
#
#    def summarize(self):
#        result_dir = cfg.result_dir
#        print(
#            colored('the results are saved at {}'.format(result_dir),
#                    'yellow'))
#
#        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
#        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
#        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
#        np.save(result_path, metrics)
#        print('mse: {}'.format(np.mean(self.mse)))
#        print('psnr: {}'.format(np.mean(self.psnr)))
#        print('ssim: {}'.format(np.mean(self.ssim)))
#        self.mse = []
#        self.psnr = []
#        self.ssim = []



import numpy as np
from lib.config import cfg
from skimage.metrics import structural_similarity
import os
import cv2
import matplotlib.pyplot as plt
from termcolor import colored

from third_parties.lpips import LPIPS
import torch

def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.

class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.lpips_model = LPIPS(net='vgg').cuda()

        # result_dir = os.path.join(cfg.result_dir,
        #                           'epoch_' + str(cfg.test.epoch),
        #                           cfg.exp_folder_name)
        # print(
        #     colored('the results are saved at {}'.format(result_dir),
        #             'yellow'))

    def psnr_metric(self, img_pred, img_gt):

        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def lpips_metric(self, img_pred, img_gt):
        img_pred = torch.tensor(img_pred[None, ...]).float().cuda()
        img_gt = torch.tensor(img_gt[None, ...]).float().cuda()
        lpips_loss = self.lpips_model(scale_for_lpips(img_pred.permute(0, 3, 1, 2)),
                                    scale_for_lpips(img_gt.permute(0, 3, 1, 2)))
        lpips_loss = torch.mean(lpips_loss)
        return lpips_loss.item()

        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch):

        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                      view_index),
            (img_gt[..., [2, 1, 0]] * 255))

        # compute the ssim
        ssim = structural_similarity(img_pred, img_gt, multichannel=True)
        return ssim


    def evaluate(self, output, batch):

        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image
        white_bkgd = int(cfg.white_bkgd)
        img_pred = np.zeros((H, W, 3)) + white_bkgd
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3)) + white_bkgd
        img_gt[mask_at_box] = rgb_gt

        rgb_pred = img_pred
        rgb_gt = img_gt

        mse = np.mean((rgb_pred - rgb_gt) ** 2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)

        lpips = self.lpips_metric(rgb_pred, rgb_gt)
        self.lpips.append(lpips)

        mse_str = 'mse: {}'.format(np.mean(self.mse))
        psnr_str = 'psnr: {}'.format(np.mean(self.psnr))
        ssim_str = 'ssim: {}'.format(np.mean(self.ssim))
        lpips_str = 'lpips: {}'.format(np.mean(self.lpips))

        print(mse_str)
        print(psnr_str)
        print(ssim_str)
        print(lpips_str)

    def summarize(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim, 'lpips': self.lpips}
        np.save(result_path, metrics)
        print('mse: {}'.format(np.mean(self.mse)))
        print('psnr: {}'.format(np.mean(self.psnr)))
        print('ssim: {}'.format(np.mean(self.ssim)))
        print('lpips: {}'.format(np.mean(self.lpips)))
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []
