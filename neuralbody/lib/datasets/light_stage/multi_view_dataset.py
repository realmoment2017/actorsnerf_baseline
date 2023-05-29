import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        test_view = [i for i in range(num_cams) if i not in cfg.training_view]
        view = cfg.training_view if split == 'train' else test_view
        if len(view) == 0:
            view = [0]

        train_few_shot = False
        if train_few_shot:
            ni = 11
            i_intv = 30
            view = [0]
        else:
            cfg.num_train_frame = 301

        # prepare input images
        i = 0
        i = i + cfg.begin_ith_frame
        if not train_few_shot:
            i_intv = cfg.frame_interval
            ni = cfg.num_train_frame
        if cfg.test_novel_pose:
            #i = (i + cfg.num_train_frame) * i_intv
            #ni = cfg.num_novel_pose_frame
            #if self.human == 'CoreView_390':
            #    i = 0
            i = 301
            ni = 25
            i_intv = 10
            if self.human in ['d16', 'd17', 'd18', 'd19', 'd20']: # for AIST++
                i = 301
                ni = 1000
                i_intv = 10

        if train_few_shot:
            self.ims = np.array([
                np.array(ims_data['ims'])[view]
                for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
            ]).ravel()[1:]
            self.cam_inds = np.array([
                np.arange(len(ims_data['ims']))[view]
                for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
            ]).ravel()[1:]

            if self.human in ['d16', 'd17', 'd18', 'd19', 'd20']: # for AIST++
                ni = 300
                i_intv = 1
                skip = 30
                if skip==60:
                    i_intv = 75
                    ni = 5
                    if self.human == 'd16':
                        render_framelist = ['c01/000200.jpg', 'c01/000564.jpg', 'c01/000824.jpg', 'c01/000832.jpg', 'c01/000840.jpg']
                    if self.human == 'd17':
                        render_framelist = ['c01/000200.jpg', 'c01/000564.jpg', 'c01/001228.jpg', 'c01/001240.jpg', 'c01/001252.jpg']
                    if self.human == 'd18':
                        render_framelist = ['c01/000200.jpg', 'c01/000500.jpg', 'c01/000708.jpg', 'c01/000736.jpg', 'c01/000780.jpg']
                    if self.human == 'd19':
                        render_framelist = ['c01/000212.jpg', 'c01/000280.jpg', 'c01/000408.jpg', 'c01/000556.jpg', 'c01/000564.jpg']
                    if self.human == 'd20':
                        render_framelist = ['c01/000200.jpg', 'c01/000240.jpg' , 'c01/000644.jpg', 'c01/000700.jpg', 'c01/000712.jpg']
                    self.ims = np.array(render_framelist)
                    self.cam_inds = np.array([
                            np.arange(len(ims_data['ims']))[view]
                            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
                        ]).ravel()
    
                else:
                    maxframes = 300
                    framelist = np.array([
                        np.array(ims_data['ims'])[view]
                        for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
                    ]).ravel()[:-3]
                    if self.human == 'd16':
                        render_framelist = ['c01/000204.jpg', 'c01/000832.jpg', 'c01/000844.jpg']
                    if self.human == 'd17':
                        render_framelist = ['c01/000564.jpg', 'c01/001228.jpg', 'c01/001252.jpg']
                    if self.human == 'd18':
                        render_framelist = ['c01/000204.jpg', 'c01/000708.jpg', 'c01/000736.jpg']
                    if self.human == 'd19':
                        render_framelist = ['c01/000212.jpg', 'c01/000408.jpg', 'c01/000556.jpg']
                    if self.human == 'd20':
                        render_framelist = ['c01/000204.jpg', 'c01/000700.jpg', 'c01/000712.jpg']
                    self.ims = np.array(render_framelist + list(framelist))
 
                    self.cam_inds = np.array([
                        np.arange(len(ims_data['ims']))[view]
                        for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
                    ]).ravel()
            
        else:
            self.ims = np.array([
                np.array(ims_data['ims'])[view]
                for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
            ]).ravel()
            self.cam_inds = np.array([
                np.arange(len(ims_data['ims']))[view]
                for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
            ]).ravel()

        self.num_cams = len(view)

        self.nrays = cfg.N_rand
        # print(self.ims)

    def get_mask(self, index):
        if self.human in ['d16', 'd17', 'd18', 'd19', 'd20']:
            msk_path = os.path.join(self.data_root, 'mask',
                                    self.ims[index])[:-4] + '.png'
            msk_cihp = imageio.imread(msk_path)[..., 0]
        else:
            msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'
            msk_cihp = imageio.imread(msk_path)
        msk = (msk_cihp != 0).astype(np.uint8)

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk


    # def data_loader_check(self, min_xyz, max_xyz, vertices):
    #     import torch
    #     import wis3d

    #     def min_max_to_bbox_8points(min_coord, max_coord):
    #         a, b, c = min_coord
    #         d, e, f = max_coord
    #         points = torch.zeros((8,3))
    #         points[0] = torch.tensor([a, b, c])
    #         points[1] = torch.tensor([d, b, c])
    #         points[2] = torch.tensor([d, b, f])
    #         points[3] = torch.tensor([a, b, f])
    #         points[4] = torch.tensor([a, e, c])
    #         points[5] = torch.tensor([d, e, c])
    #         points[6] = torch.tensor([d, e, f])
    #         points[7] = torch.tensor([a, e, f])
    #         return points

    #     from wis3d import Wis3D
    #     wis_dir = "/data/jiteng/gdrive"
    #     wis3d = Wis3D(wis_dir, 'figures')
    #     pbbox = min_max_to_bbox_8points(min_coord=min_xyz, max_coord=max_xyz)
    #     wis3d.add_boxes(pbbox, name='pbbox', labels='pbbox')
    #     colors = torch.tensor([[255,0,0]]).repeat(6890,1)
    #     wis3d.add_point_cloud(vertices, colors, name='tsmpl')
    #     breakpoint()

    # def check_input(self, i, K, RT, msk):
    #     # read xyz, normal, color from the ply file
    #     vertices_path = os.path.join(self.data_root, cfg.vertices,
    #                                  '{}.npy'.format(i))
    #     xyz = np.load(vertices_path).astype(np.float32)
    #     nxyz = np.zeros_like(xyz).astype(np.float32)

    #     # obtain the original bounds for point sampling
    #     min_xyz = np.min(xyz, axis=0)
    #     max_xyz = np.max(xyz, axis=0)
    #     if cfg.big_box:
    #         min_xyz -= 0.05
    #         max_xyz += 0.05
    #     else:
    #         min_xyz[2] -= 0.05
    #         max_xyz[2] += 0.05
    #     can_bounds = np.stack([min_xyz, max_xyz], axis=0)

    #     # transform smpl from the world coordinate to the smpl coordinate
    #     params_path = os.path.join(self.data_root, cfg.params,
    #                                '{}.npy'.format(i))
    #     params = np.load(params_path, allow_pickle=True).item()
    #     # self.data_loader_check(min_xyz, max_xyz, xyz)
    #     breakpoint()
    #     import matplotlib.pyplot as plt
    #     from lib.utils.base_utils import project
    #     plt.imshow(msk)
    #     plt.plot(project(xyz, K, RT)[:,0], project(xyz, K, RT)[:,1], 'ro')
    #     plt.savefig('plot.png')
    #     Rh = params['Rh']
    #     R = cv2.Rodrigues(Rh)[0].astype(np.float32)
    #     Th = params['Th'].astype(np.float32)
    #     xyz = np.dot(xyz - Th, R)

    #     # obtain the bounds for coord construction
    #     min_xyz = np.min(xyz, axis=0)
    #     max_xyz = np.max(xyz, axis=0)
    #     if cfg.big_box:
    #         min_xyz -= 0.05
    #         max_xyz += 0.05
    #     else:
    #         min_xyz[2] -= 0.05
    #         max_xyz[2] += 0.05
    #     bounds = np.stack([min_xyz, max_xyz], axis=0)

    #     # construct the coordinate
    #     dhw = xyz[:, [2, 1, 0]]
    #     min_dhw = min_xyz[[2, 1, 0]]
    #     max_dhw = max_xyz[[2, 1, 0]]
    #     voxel_size = np.array(cfg.voxel_size)
    #     coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

    #     # construct the output shape
    #     out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
    #     x = 32
    #     out_sh = (out_sh | (x - 1)) + 1

    #     return coord, out_sh, can_bounds, bounds, Rh, Th

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return coord, out_sh, can_bounds, bounds, Rh, Th

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        img = cv2.resize(img, (cfg.W, cfg.H))
        msk = self.get_mask(index)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        K[:2] = K[:2] * cfg.ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        elif self.human in ['d16', 'd17', 'd18', 'd19', 'd20']:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        # coord, out_sh, can_bounds, bounds, Rh, Th = self.check_input(
        #     i, K, np.concatenate((R, T), axis=1), msk)
        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            i)

        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, can_bounds, self.nrays, self.split)

        ret = {
            'coord': coord,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = (frame_index - cfg.begin_ith_frame) // cfg.frame_interval
        if cfg.test_novel_pose:
            latent_index = cfg.num_train_frame - 1
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
