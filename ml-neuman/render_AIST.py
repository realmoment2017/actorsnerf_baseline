#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Render test views, and report the metrics
'''

import os
import argparse

import imageio
import torch
import numpy as np
import skimage
from skimage.metrics import structural_similarity as ssim
import lpips

loss_fn_alex = lpips.LPIPS(net='vgg')

from models import human_nerf
from utils import render_utils, utils
from data_io import neuman_helper, AIST_helper_eval
from options import options

import re
import time
from utils.evaluator import Evaluator

def eval_metrics(gts, preds):
    results = {
        'ssim': [],
        'psnr': [],
        'lpips': []
    }
    for gt, pred in zip(gts, preds):
        results['ssim'].append(ssim(pred, gt, multichannel=True))
        results['psnr'].append(skimage.metrics.peak_signal_noise_ratio(gt, pred))
        results['lpips'].append(
            float(loss_fn_alex(utils.np_img_to_torch_img(pred[None])/127.5-1, utils.np_img_to_torch_img(gt[None])/127.5-1)[0, 0, 0, 0].data)
        )
    for k, v in results.items():
        results[k] = np.mean(v)
    return results


def optimize_pose_with_nerf(opt, cap, net, iters=1000, save_every=10):
    # TODO: optimize the pose with the trained NeRF
    pass


def main(opt):
    # _, _, test_split = neuman_helper.create_split_files(opt.scene_dir)
    # test_views = neuman_helper.read_text(test_split)
    test_views = list(range(301, 550))
    test_views = test_views[::10]

    preds = []
    gts = []

    evaluator = Evaluator()
    for exclude_TABU_CAMS in range(22): 
        TABU_CAMS = list(range(23))
        TABU_CAMS.pop(exclude_TABU_CAMS+1)
        scene = AIST_helper_eval.ZjuMocapReader.read_scene(
            opt.scene_dir,
            tgt_size=None,
            TABU_CAMS=TABU_CAMS,
        )
        if opt.geo_threshold < 0:
            bones = []
            for i in range(len(scene.captures)):
                bones.append(np.linalg.norm(scene.smpls[i]['joints_3d'][3] - scene.smpls[i]['joints_3d'][0]))
            opt.geo_threshold = np.mean(bones)
            net = human_nerf.HumanNeRF(opt)
            weights = torch.load(opt.weights_path, map_location='cpu')
            utils.safe_load_weights(net, weights['hybrid_model_state_dict'])

        for view_name in test_views:
            cap = scene[view_name]
            i = cap.frame_id['frame_id']
            #if exclude_TABU_CAMS+1==4:
            #    continue
            # out = render_utils.render_hybrid_nerf(
            out = render_utils.render_smpl_nerf(
                net,
                cap,
                scene.verts[i],
                scene.faces,
                scene.Ts[i],
                # rays_per_batch=opt.rays_per_batch,
                samples_per_ray=opt.samples_per_ray,
                white_bkg=True,
                geo_threshold=opt.geo_threshold,
                return_depth=False
            )
            breakpoint()
            file_path = re.split('/', opt.weights_path)[-2]
            save_path = os.path.join('./demo', f'test_views/{file_path}', f'{str(exclude_TABU_CAMS+1)}_frame_{str(i).zfill(6)}.png')
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            imageio.imsave(save_path, out)
            print(f'image saved: {save_path}')
            # preds.append(imageio.imread(save_path))
            # gts.append(cap.image)
            gt_image = cap.image.copy() / 255.0
            gt_image[cap.mask<1] = 1
            evaluator.evaluate(out, gt_image)
            evaluator.summarize()
    evaluator.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options.set_general_option(parser)
    opt, _ = parser.parse_known_args()

    options.set_nerf_option(parser)
    options.set_pe_option(parser)
    options.set_render_option(parser)
    options.set_trajectory_option(parser)
    parser.add_argument('--scene_dir', required=True, type=str, help='scene directory')
    parser.add_argument('--image_dir', required=False, type=str, default=None, help='image directory')
    parser.add_argument('--out_dir', default='./out', type=str, help='weights dir')
    parser.add_argument('--offset_scale', default=1.0, type=float, help='scale the predicted offset')
    parser.add_argument('--geo_threshold', default=-1, type=float, help='')
    parser.add_argument('--normalize', default=True, type=options.str2bool, help='')
    parser.add_argument('--bkg_range_scale', default=3, type=float, help='extend near/far range for background')
    parser.add_argument('--human_range_scale', default=1.5, type=float, help='extend near/far range for human')
    parser.add_argument('--num_offset_nets', default=1, type=int, help='how many offset networks')
    parser.add_argument('--offset_scale_type', default='linear', type=str, help='no/linear/tanh')

    opt = parser.parse_args()
    assert opt.geo_threshold == -1, 'please use auto geo_threshold'
    if opt.render_h is None:
        opt.render_size = None
    else:
        opt.render_size = (opt.render_h, opt.render_w)
    main(opt)
