import os

import torch
import numpy as np
from tqdm import tqdm

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter_Category, to_8b_image, to_8b3ch_image

from configs import cfg, args
from core.evaluator import Evaluator

from moviepy.editor import *
# import kornia

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'sub_idx', 'cam_idx'
                       'img_width', 'img_height', 'ray_mask']


def load_network():
    model = create_network()
    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    for m1, _ in model.named_parameters():
        if m1 not in ckpt['network'].keys():
            raise Exception("model not match")
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda().deploy_mlps_to_secondary_gpus()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image


# def img_process(img, mask, bg_color):
#     mask = mask / 255.
#     img = mask * img + (1.0 - mask) * bg_color[None, None, :]
#     img = torch.tensor(img[None, ...]).permute(0, 3, 1, 2)
#     if cfg.resize_img_scale != 1.:
#         H, W = int(img.shape[-2]*cfg.resize_img_scale), int(img.shape[-1]*cfg.resize_img_scale)
#         img = kornia.geometry.transform.resize(img, (H, W), interpolation='bilinear')
#     img = img / 255.0
#     return img.permute(0, 2, 3, 1).contiguous()[0].float()

def _eval(
        data_type=None,
        dataset_mode='ins_level',
        folder_name=None):
    cfg.perturb = 0.

    model = load_network()
    test_loader = create_dataloader(data_type, dataset_mode)

    # if cfg.use_data_cross_pose:
    #     test_loader_enc = create_dataloader(data_type, dataset_mode='cat_level')
    # elif cfg.use_data_cross_view:
    #     test_loader_enc = create_dataloader(data_type='multiview', dataset_mode='cat_level')
    # else:
    #     test_loader_enc = None

    writer = ImageWriter_Category(
                output_dir=os.path.join(cfg.logdir, cfg.load_net),
                exp_name=folder_name)

    if data_type in ['eval_novel_view', 'eval_novel_pose']:
        evaluator = Evaluator()
    model.eval()
    for batch in tqdm(test_loader):
        #batch = batch[0]
        for k, v in batch.items():
            if k=="data_enc" or k=='txyz_dic' or k=='pxyz_dic' or k=='dst_bbox':
                batch[k] = v
            else:
                batch[k] = v[0]

        # render_sub_idx = batch['sub_idx']
        # render_cam_idx = batch['cam_idx'].item()
        # render_frame_name = batch['frame_name']
        # render_frame_name = 'frame_000001'

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        # if cfg.use_data_cross_pose:
        #     data_enc = {}
        #     num_data_enc_frames = 1
        #     for aug_idx in range(num_data_enc_frames):
        #         frame_idx = idx.item()
        #         data_enc[aug_idx] = {}
        #         data_enc_batch = test_loader_enc.dataset[frame_idx][0]
        #         data_enc[aug_idx]['img'] = data_enc_batch['img']
        #         data_enc[aug_idx]['img'] = img_process(data_enc_batch['img'], data_enc_batch['mask'], data_enc_batch['bgcolor'])
        #         data_enc[aug_idx]['cam_K'] = data_enc_batch['cam_K']
        #         data_enc[aug_idx]['cam_R'] = data_enc_batch['cam_R']
        #         data_enc[aug_idx]['cam_T'] = data_enc_batch['cam_T']
        #         data_enc[aug_idx]['dst_Rs'] = data_enc_batch['dst_Rs']
        #         data_enc[aug_idx]['dst_Ts'] = data_enc_batch['dst_Ts']
        #         data_enc[aug_idx]['cnl_gtfms'] = data_enc_batch['cnl_gtfms']
        #         for k in data_enc[aug_idx]:
        #             data_enc[aug_idx][k] = torch.tensor(data_enc[aug_idx][k]).cuda()
        #     data['data_enc'] = data_enc

        # elif cfg.use_data_cross_view:
        #     data_enc = {}
        #     cam_idxs = [6, 12, 18]
        #     for aug_idx in range(len(cam_idxs)):
        #         frame_idx = test_loader_enc.dataset.frame_to_idx[(render_sub_idx, cam_idxs[aug_idx], render_frame_name)]
        #         data_enc[aug_idx] = {}
        #         data_enc_batch = test_loader_enc.dataset[frame_idx][0]
        #         data_enc[aug_idx]['img'] = data_enc_batch['img']
        #         data_enc[aug_idx]['img'] = img_process(data_enc_batch['img'], data_enc_batch['mask'], data_enc_batch['bgcolor'])
        #         data_enc[aug_idx]['cam_K'] = data_enc_batch['cam_K']
        #         data_enc[aug_idx]['cam_R'] = data_enc_batch['cam_R']
        #         data_enc[aug_idx]['cam_T'] = data_enc_batch['cam_T']
        #         data_enc[aug_idx]['dst_Rs'] = data_enc_batch['dst_Rs']
        #         data_enc[aug_idx]['dst_Ts'] = data_enc_batch['dst_Ts']
        #         data_enc[aug_idx]['cnl_gtfms'] = data_enc_batch['cnl_gtfms']
        #         for k in data_enc[aug_idx]:
        #             data_enc[aug_idx][k] = torch.tensor(data_enc[aug_idx][k]).cuda()
        #     data['data_enc'] = data_enc

        data['iter_val'] = torch.full((1,), 1000000).cuda()
        with torch.no_grad():
            net_output = model(**data)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']
        target_rgbs = batch.get('target_rgbs', None)

        rgb_img, alpha_img, _ = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy())

        imgs = [rgb_img]
        if cfg.show_truth and target_rgbs is not None:
            target_rgbs = to_8b_image(target_rgbs.numpy())
            imgs.append(target_rgbs)
        if cfg.show_alpha:
            imgs.append(alpha_img)

        img_out = np.concatenate(imgs, axis=1)
        writer.append(batch, img_out)

        if data_type in ['eval_novel_view', 'eval_novel_pose']:
            H, W, _ = batch['target_rgbs'].shape
            mask = ray_mask.detach().cpu().numpy().reshape(H, W)
            if cfg.task=='zju_mocap':
                eval_pred_rgb = np.zeros((H, W, 3))
            if cfg.task=='AIST_mocap':
                eval_pred_rgb = np.ones((H, W, 3))
            eval_pred_rgb[mask>0] = rgb.detach().cpu().numpy()
            eval_gt_rgb = batch['target_rgbs'].detach().cpu().numpy()
            evaluator.evaluate(eval_pred_rgb, eval_gt_rgb)

    if data_type in ['eval_novel_view', 'eval_novel_pose']:
        evaluator.summarize()

    writer.finalize()


def _eval_freeview(
        data_type=None,
        dataset_mode='ins_level',
        folder_name=None):
    cfg.perturb = 0.

    model = load_network()
    test_loader = create_dataloader(data_type, dataset_mode)

    writer = ImageWriter_Category(
                output_dir=os.path.join(cfg.logdir, cfg.load_net),
                exp_name=folder_name)

    model.eval()
    imgs = []
    for batch in tqdm(test_loader):
        #batch = batch[0]
        for k, v in batch.items():
            if k=="data_enc" or k=='txyz_dic' or k=='pxyz_dic' or k=='dst_bbox':
                batch[k] = v
            else:
                batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        data['iter_val'] = torch.full((1,), 1000000).cuda()
        with torch.no_grad():
            net_output = model(**data)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']
        target_rgbs = batch.get('target_rgbs', None)

        rgb_img, alpha_img, _ = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy())

        writer.append(batch, rgb_img)
        imgs.append(rgb_img)
        img_out = np.stack(imgs, axis=0)

    writer.finalize()
    save_path = os.path.join(cfg.logdir, cfg.load_net, folder_name)
    import torchvision
    torchvision.io.write_video(save_path+'/result.mp4', torch.tensor(img_out), fps=10, video_codec='h264', options={'crf': '10'})
    videoClip  = VideoFileClip(save_path+'/result.mp4')
    videoClip.write_gif(save_path+'/result.gif')



def run_eval_novel_view():
    _eval(
        data_type='eval_novel_view',
        dataset_mode='cat_level',
        folder_name=f"eval_novel_view")

def run_eval_novel_pose():
    _eval(
        data_type='eval_novel_pose',
        dataset_mode='cat_level',
        folder_name=f"eval_novel_pose")

def run_tpose():
    _eval(
        data_type='tpose',
        dataset_mode='cat_level',
        folder_name=f"tpose")

def run_eval_freeview():
    _eval_freeview(
        data_type='eval_freeview',
        dataset_mode='cat_level',
        folder_name=f"eval_freeview")
        
if __name__ == '__main__':
    globals()[f'run_{args.type}']()
