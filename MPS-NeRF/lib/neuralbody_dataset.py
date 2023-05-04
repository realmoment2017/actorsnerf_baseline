from torch.utils.data import DataLoader, dataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import imageio
import cv2
import time
import lib.if_nerf_data_utils as if_nerf_dutils


class NeuBodyDatasetBatch(Dataset):
    def __init__(self, data_root, split='test', view_num=23, border=5, N_rand=1024*32, multi_person=False, num_instance=1, 
                start=0, interval=5, poses_num=100, image_scaling=1.0, finetune_subject='None'):
        super(NeuBodyDatasetBatch, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = os.path.join(data_root, 'CoreView_387', 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        # if view_num==23:
        #     self.train_view = [x for x in range(23)] # [x for x in range(24)] # [0,1,2,3]
        # else:
        #     self.train_view = [0, 2, 6, 9, 12, 15, 18, 21] # [0, 3, 6, 9, 12, 15, 18, 21]
        # self.input_view =  [0, 2, 6, 9, 12, 15, 18, 21] # [0, 3, 6, 9, 12, 15, 18, 21]
        if view_num==8:
            self.input_view = [0,1, 6,7, 12,13, 18,19] # [x for x in range(24)] # [0,1,2,3] # 3,4,6
        elif view_num==6:
            self.input_view = [0,1, 6, 12,13, 18] # [0,1,2,3] # [x for x in range(20)]
        elif view_num==4:
            self.input_view = [0, 6, 12, 18] # [0, 4, 10, 15]
        elif view_num==3:
            self.input_view = [4, 10, 16]
        self.input_view = [0]

        # TODO For finetune
        # self.train_view =  [0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] # self.input_view# [x for x in range(23)] # remove 3,4,6 / 2,3,5
        self.train_view =  self.input_view # self.input_view# [x for x in range(23)] # remove 3,4,6 / 2,3,5
        
        # TODO
        self.test_view = [x for x in range(20)] # [0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ] # , 21, 22] # [0,6,12,18]#
        # self.test_view =  [0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] # , 21, 22] # [0,6,12,18]#
        self.output_view = self.train_view if split == 'train' else self.test_view
        
        """
        annots = {
            'cams':{
                'K':[],#N arrays, (3, 3)
                'D':[],#(5, 1), all zeros
                'R':[],#(3, 3)
                'T':[] #(3, 1)
            },

            'ims':[
                # {'ims':['54138969/000000.jpg', '55011271/000000.jpg''60457274/000000.jpg']}, # same pose different views
                # {'ims':[]} 
                #  repeat time is number of poses
            ]
        }
        """
        self.i = start # start index 0
        self.i_intv = interval # interval 1
        self.ni = poses_num # number of used poses 30

        self.finetune_subject = finetune_subject
        if finetune_subject == 'None':
            human_dirs = [
                "CoreView_313", "CoreView_315", "CoreView_377", "CoreView_386",
                "CoreView_390", "CoreView_392", "CoreView_396"
            ] # "CoreView_377", CoreView_313, CoreView_315, "CoreView_396" 
        else:
            human_dirs = [finetune_subject]

        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)

        self.nrays = N_rand
        self.border = border
        self.image_scaling = image_scaling
        self.num_instance = num_instance
        self.multi_person = multi_person
        all_human_data_root = os.path.join(os.path.dirname(data_root))
        

        self.root_list = [data_root] if not multi_person else [os.path.join(all_human_data_root, x.strip()) for x in human_dirs]
        print(self.root_list, len(self.ims), self.ims)
        

    def get_mask(self, index, view_index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index][view_index])[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp.copy()

        border = self.border
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        # msk[(msk_dilate - msk_erode) == 1] = 100

        kernel_ = np.ones((border+3, border+3), np.uint8)
        msk_dilate_ = cv2.dilate(msk.copy(), kernel_)

        msk[(msk_dilate - msk_erode) == 1] = 100
        msk[(msk_dilate_ - msk_dilate) == 1] = 200

        return msk, msk_cihp

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        
        feature = cxyz
        # feature = np.ones((6890,1)) * 0.01

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, bounds

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'new_vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz
        # nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        # nxyz = nxyz.astype(np.float32)
        # feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)
        feature = cxyz

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params # , center, rot, trans

    def update(self, data_root):
        offset = 0
        if os.path.basename(self.data_root) == 'CoreView_396':
            offset = 810
        if os.path.basename(self.data_root) == 'CoreView_390':
            offset = 600
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i + offset:self.i + offset + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)

    def __getitem__(self, pose_index):
        
        data_root_i = np.random.randint(len(self.root_list)) if self.multi_person else 0
        self.data_root = self.root_list[data_root_i]
        self.update(self.data_root)
        pose_index = pose_index % self.ni
        # print(self.input_view)
        img_all, K_all, R_all, T_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, img_ray_d_all = [], [], []
        
        for idx, view_index in enumerate(self.output_view):
            # Load image, mask, K, D, R, T
            img_path = os.path.join(self.data_root, self.ims[pose_index][idx].replace('\\', '/'))
            img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
            msk, origin_msk = np.array(self.get_mask(pose_index, idx))

            K = np.array(self.cams['K'][view_index])
            D = np.array(self.cams['D'][view_index])
            # TODO
            # img = cv2.undistort(img, K, D)
            # msk = cv2.undistort(msk, K, D)
            R = np.array(self.cams['R'][view_index])
            T = np.array(self.cams['T'][view_index]) / 1000.
            img[origin_msk == 0] = 0

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = self.image_scaling
            if ratio != 1.:
                H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                K[:2] = K[:2] * ratio
            
            # Prepare the smpl input, including the current pose and canonical pose
            if view_index == self.output_view[0]:
                # TODO
                # print(os.path.basename(img_path))
                if os.path.basename(self.data_root) == 'CoreView_313' or os.path.basename(self.data_root) == 'CoreView_315':
                    i = int(os.path.basename(img_path)[-32:-28])
                else:
                    i = int(os.path.basename(img_path)[:-4])
                # i = int(os.path.basename(img_path)[-32:-28])
                feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params = self.prepare_input(i)                
                # path = os.path.join(self.data_root, '45_big_pose_tvertices.npy') # before_200_45_big_pose_tvertices # 45_big_pose_tvertices # posing_tvertices
                path = os.path.join(self.data_root, 'lbs/tvertices.npy') # before_200_45_big_pose_tvertices # 45_big_pose_tvertices # posing_tvertices
                smpl_R = cv2.Rodrigues(Rh)[0].astype(np.float32)
                params["R"] = cv2.Rodrigues(params['Rh'])[0].astype(np.float32)
                t_vertices = np.load(path)
                t_feature, t_coord, t_out_sh, t_bounds = self.prepare_input_t(t_vertices_path=path)
            
            # Sample rays in target space world coordinate
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk,  img_ray_d = if_nerf_dutils.sample_ray_neubody_batch(
                    img, msk, K, R, T, world_bounds, self.nrays, self.split, ratio=0.6)

            # Pack all inputs of all views
            img = np.transpose(img, (2,0,1))
            img_ray_d = np.transpose(img_ray_d, (2,0,1))
            if view_index in self.input_view:
                img_all.append(img)
                img_ray_d_all.append(img_ray_d)
                K_all.append(K)
                R_all.append(R)
                T_all.append(T)
            msk_all.append(msk)
            rgb_all.append(rgb)
            ray_o_all.append(ray_o)
            ray_d_all.append(ray_d)
            near_all.append(near)
            far_all.append(far)
            mask_at_box_all.append(mask_at_box)
            bkgd_msk_all.append(bkgd_msk)

        img_all = np.stack(img_all, axis=0)
        img_ray_d_all = np.stack(img_ray_d_all, axis=0)
        msk_all = np.stack(msk_all, axis=0)
        K_all = np.stack(K_all, axis=0)
        R_all = np.stack(R_all, axis=0)
        T_all = np.stack(T_all, axis=0)
        # if self.split == "train":
        rgb_all = np.stack(rgb_all, axis=0)
        ray_o_all = np.stack(ray_o_all, axis=0)
        ray_d_all = np.stack(ray_d_all, axis=0)
        near_all = np.stack(near_all, axis=0)[...,None]
        far_all = np.stack(far_all, axis=0)[...,None]
        mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)
        

        ret = {
            'pose_index': pose_index,
            "instance_idx": data_root_i, 
            'R': smpl_R, # smpl global Rh Th # 
            'Th': Th,
            'gender': 2,

            "params": params, # smpl params including R, Th
            'vertices': vertices, # world vertices
            'feature': feature, # smpl pose space xyz
            'coord': coord,    # smpl pose space coords
            'bounds': bounds, # smpl pose space bounds
            'out_sh': out_sh,   # smpl pose space out_sh

            't_vertices': t_vertices,
            't_feature': t_feature, # smpl pose space xyz
            't_coord': t_coord,    # smpl pose space coords
            't_bounds': t_bounds, # smpl pose space bounds
            't_out_sh': t_out_sh,   # smpl pose space out_sh
            
            # world space
            'img_all': img_all,
            'img_ray_d_all': img_ray_d_all,
            'msk_all': msk_all,
            'K_all': K_all,
            'R_all': R_all,
            'T_all': T_all,
            'rgb_all': rgb_all,
            'ray_o_all': ray_o_all,
            'ray_d_all': ray_d_all,
            'near_all': near_all,
            'far_all': far_all,
            'mask_at_box_all': mask_at_box_all,
            'bkgd_msk_all': bkgd_msk_all
        }


        return ret

    def __len__(self):
        return len(self.ims) * self.num_instance# * len(self.output_view)


class NeuBodyDatasetBatch_Source(Dataset):
    def __init__(self, data_root, split='test', view_num=23, border=5, N_rand=1024*32, multi_person=False, num_instance=1, 
                start=0, interval=5, poses_num=100, image_scaling=1.0, finetune_subject='None'):
        super(NeuBodyDatasetBatch_Source, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = os.path.join(data_root, 'CoreView_387', 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        # if view_num==23:
        #     self.train_view = [x for x in range(23)] # [x for x in range(24)] # [0,1,2,3]
        # else:
        #     self.train_view = [0, 2, 6, 9, 12, 15, 18, 21] # [0, 3, 6, 9, 12, 15, 18, 21]
        # self.input_view =  [0, 2, 6, 9, 12, 15, 18, 21] # [0, 3, 6, 9, 12, 15, 18, 21]
        if view_num==8:
            self.input_view = [0,1, 6,7, 12,13, 18,19] # [x for x in range(24)] # [0,1,2,3] # 3,4,6
        elif view_num==6:
            self.input_view = [0,1, 6, 12,13, 18] # [0,1,2,3] # [x for x in range(20)]
        elif view_num==4:
            self.input_view = [0, 6, 12, 18] # [0, 4, 10, 15]
        elif view_num==3:
            self.input_view = [4, 10, 16]
        self.input_view = [0]

        # TODO For finetune
        # self.train_view =  [0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] # self.input_view# [x for x in range(23)] # remove 3,4,6 / 2,3,5
        self.train_view =  self.input_view # self.input_view# [x for x in range(23)] # remove 3,4,6 / 2,3,5
        
        # TODO
        self.test_view = [x for x in range(20)] # [0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ] # , 21, 22] # [0,6,12,18]#
        # self.test_view =  [0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] # , 21, 22] # [0,6,12,18]#
        self.output_view = self.train_view if split == 'train' else self.test_view
        
        """
        annots = {
            'cams':{
                'K':[],#N arrays, (3, 3)
                'D':[],#(5, 1), all zeros
                'R':[],#(3, 3)
                'T':[] #(3, 1)
            },

            'ims':[
                # {'ims':['54138969/000000.jpg', '55011271/000000.jpg''60457274/000000.jpg']}, # same pose different views
                # {'ims':[]} 
                #  repeat time is number of poses
            ]
        }
        """
        self.i = start # start index 0
        self.i_intv = interval # interval 1
        self.ni = poses_num # number of used poses 30

        self.finetune_subject = finetune_subject
        if finetune_subject == 'None':
            human_dirs = [
                "CoreView_313", "CoreView_315", "CoreView_377", "CoreView_386",
                "CoreView_390", "CoreView_392", "CoreView_396"
            ] # "CoreView_377", CoreView_313, CoreView_315, "CoreView_396" 
        else:
            human_dirs = [finetune_subject]

        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)

        self.nrays = N_rand
        self.border = border
        self.image_scaling = image_scaling
        self.num_instance = num_instance
        self.multi_person = multi_person
        all_human_data_root = os.path.join(os.path.dirname(data_root))
        

        self.root_list = [data_root] if not multi_person else [os.path.join(all_human_data_root, x.strip()) for x in human_dirs]
        print(self.root_list)


    def get_mask(self, index, view_index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index][view_index])[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp.copy()

        border = self.border
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        # msk[(msk_dilate - msk_erode) == 1] = 100

        kernel_ = np.ones((border+3, border+3), np.uint8)
        msk_dilate_ = cv2.dilate(msk.copy(), kernel_)

        msk[(msk_dilate - msk_erode) == 1] = 100
        msk[(msk_dilate_ - msk_dilate) == 1] = 200

        return msk, msk_cihp

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        
        feature = cxyz
        # feature = np.ones((6890,1)) * 0.01

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, bounds

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'new_vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz
        # nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        # nxyz = nxyz.astype(np.float32)
        # feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)
        feature = cxyz

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params # , center, rot, trans

    def update(self, data_root):
        offset = 0
        if os.path.basename(self.data_root) == 'CoreView_396':
            offset = 810
        if os.path.basename(self.data_root) == 'CoreView_390':
            offset = 600
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        if self.finetune_subject != 'None':
            if self.finetune_subject == 'CoreView_387':
                self.ims = np.array([['Camera_B1/000075.jpg'],['Camera_B1/000225.jpg'],['Camera_B1/000300.jpg']])
            if self.finetune_subject == 'CoreView_393':
                self.ims = np.array([['Camera_B1/000075.jpg'],['Camera_B1/000150.jpg'],['Camera_B1/000300.jpg']])
            if self.finetune_subject == 'CoreView_394':
                self.ims = np.array([['Camera_B1/000000.jpg'],['Camera_B1/000150.jpg'],['Camera_B1/000300.jpg']])

    def __getitem__(self, instance_idx):
        
        # data_root_i = np.random.randint(len(self.root_list)) if self.multi_person else 0
        data_root_i  = instance_idx
        self.data_root = self.root_list[data_root_i]
        self.update(self.data_root)
        # pose_index = pose_index % self.ni
        # print(self.input_view)
        img_all, K_all, R_all, T_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, img_ray_d_all = [], [], []
        params_all = []
        
        finetune_idx = 0


        for idx, _ in enumerate(list(range(3))):
            view_index = 0
            idx = 0
            pose_index = np.random.randint(len(self.ims))
            # Load image, mask, K, D, R, T
            if self.finetune_subject=='None':
                img_path = os.path.join(self.data_root, self.ims[pose_index][idx].replace('\\', '/'))
            else:
                img_path = os.path.join(self.data_root, self.ims[finetune_idx][idx].replace('\\', '/'))
            img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
            # if os.path.basename(self.data_root) == 'CoreView_396':
            #     pose_index= pose_index + 810
            # if os.path.basename(self.data_root) == 'CoreView_390':
            #     pose_index= pose_index+ 1150
            msk, origin_msk = np.array(self.get_mask(pose_index, idx))

            K = np.array(self.cams['K'][view_index])
            D = np.array(self.cams['D'][view_index])
            # TODO
            # img = cv2.undistort(img, K, D)
            # msk = cv2.undistort(msk, K, D)
            R = np.array(self.cams['R'][view_index])
            T = np.array(self.cams['T'][view_index]) / 1000.
            img[origin_msk == 0] = 0

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = self.image_scaling
            if ratio != 1.:
                H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                K[:2] = K[:2] * ratio
            
            # Prepare the smpl input, including the current pose and canonical pose
            if view_index == self.output_view[0]:
                # TODO
                # print(os.path.basename(img_path))
                if os.path.basename(self.data_root) == 'CoreView_313' or os.path.basename(self.data_root) == 'CoreView_315':
                    i = int(os.path.basename(img_path)[-32:-28])
                else:
                    i = int(os.path.basename(img_path)[:-4])
                # i = int(os.path.basename(img_path)[-32:-28])
                feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params = self.prepare_input(i)                
                # path = os.path.join(self.data_root, '45_big_pose_tvertices.npy') # before_200_45_big_pose_tvertices # 45_big_pose_tvertices # posing_tvertices
                path = os.path.join(self.data_root, 'lbs/tvertices.npy') # before_200_45_big_pose_tvertices # 45_big_pose_tvertices # posing_tvertices
                smpl_R = cv2.Rodrigues(Rh)[0].astype(np.float32)
                params["R"] = cv2.Rodrigues(params['Rh'])[0].astype(np.float32)
                t_vertices = np.load(path)
                t_feature, t_coord, t_out_sh, t_bounds = self.prepare_input_t(t_vertices_path=path)
            
            # # Sample rays in target space world coordinate
            # rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk,  img_ray_d = if_nerf_dutils.sample_ray_neubody_batch(
            #         img, msk, K, R, T, world_bounds, self.nrays, self.split, ratio=0.6)

            # Pack all inputs of all views
            img = np.transpose(img, (2,0,1))
            # img_ray_d = np.transpose(img_ray_d, (2,0,1))
            if view_index in self.input_view:
                img_all.append(img)
                # img_ray_d_all.append(img_ray_d)
                K_all.append(K)
                R_all.append(R)
                T_all.append(T)
            msk_all.append(msk)
            # rgb_all.append(rgb)
            # ray_o_all.append(ray_o)
            # ray_d_all.append(ray_d)
            # near_all.append(near)
            # far_all.append(far)
            # mask_at_box_all.append(mask_at_box)
            # bkgd_msk_all.append(bkgd_msk)
            params_all.append(params)

            finetune_idx = finetune_idx + 1
        
        params_all_ = {}
        for k in params.keys():
            lst = []
            for j in range(3):
                lst.append(params_all[j][k])
            params_all_[k] = np.stack(lst, axis=0)

        img_all = np.stack(img_all, axis=0)
        # img_ray_d_all = np.stack(img_ray_d_all, axis=0)
        msk_all = np.stack(msk_all, axis=0)
        K_all = np.stack(K_all, axis=0)
        R_all = np.stack(R_all, axis=0)
        T_all = np.stack(T_all, axis=0)
        # if self.split == "train":
        # rgb_all = np.stack(rgb_all, axis=0)
        # ray_o_all = np.stack(ray_o_all, axis=0)
        # ray_d_all = np.stack(ray_d_all, axis=0)
        # near_all = np.stack(near_all, axis=0)[...,None]
        # far_all = np.stack(far_all, axis=0)[...,None]
        # mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        # bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)
        


        ret = {
            'pose_index': pose_index,
            "instance_idx": data_root_i, 
            'R': smpl_R, # smpl global Rh Th # 
            'Th': Th,
            'gender': 2,

            "params": params_all_, # smpl params including R, Th
            # 'vertices': vertices, # world vertices
            # 'feature': feature, # smpl pose space xyz
            # 'coord': coord,    # smpl pose space coords
            # 'bounds': bounds, # smpl pose space bounds
            # 'out_sh': out_sh,   # smpl pose space out_sh

            # 't_vertices': t_vertices,
            # 't_feature': t_feature, # smpl pose space xyz
            # 't_coord': t_coord,    # smpl pose space coords
            # 't_bounds': t_bounds, # smpl pose space bounds
            # 't_out_sh': t_out_sh,   # smpl pose space out_sh
            
            # world space
            'img_all': img_all,
            # 'img_ray_d_all': img_ray_d_all,
            'msk_all': msk_all,
            'K_all': K_all,
            'R_all': R_all,
            'T_all': T_all,
            # 'rgb_all': rgb_all,
            # 'ray_o_all': ray_o_all,
            # 'ray_d_all': ray_d_all,
            # 'near_all': near_all,
            # 'far_all': far_all,
            # 'mask_at_box_all': mask_at_box_all,
            # 'bkgd_msk_all': bkgd_msk_all
        }


        return ret

    def __len__(self):
        return len(self.ims) * self.num_instance# * len(self.output_view)