'''
zju mocap dataset
'''

import os
import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm
import PIL

from geometry import pcd_projector
from cameras.camera_pose import CameraPose
from cameras.pinhole_camera import PinholeCamera
from cameras import captures as captures_module, contents
from scenes import scene as scene_module
from utils import debug_utils, utils, ray_utils
from models.smpl import SMPL, batch_rodrigues

MAX_VIEWS = 10
EVERY_K=1


'''
SMPL meshes and cameras are aligned.
But SMPL meshes has translation and rotation, but only 1 single canonical space.
We need to inverse the transformation, and apply the inverse to the ray.
'''

# TABU_CAMS = [3]
TABU_CAMS = [0, 1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

class ZjuCapture(captures_module.RigRGBPinholeCapture):
    def __init__(self, image_path, mask_path, pinhole_cam, cam_pose, view_id, cam_id):
        captures_module.RigRGBPinholeCapture.__init__(self, image_path, pinhole_cam, cam_pose, view_id, cam_id)
        self.captured_image = contents.UndistortedCapturedImage(
            image_path,
            pinhole_cam.intrinsic_matrix,
            pinhole_cam.radial_dist
        )
        self.captured_mask = contents.UndistortedCapturedImage(
            mask_path,
            pinhole_cam.intrinsic_matrix,
            pinhole_cam.radial_dist
        )

    def read_image_to_ram(self) -> int:
        return self.captured_image.read_image_to_ram() + self.captured_mask.read_image_to_ram()

    @property
    def mask(self):
        _mask = self.captured_mask.image
        assert _mask.shape[0:2] == self.pinhole_cam.shape, f'mask does not match with camera model: mask shape: {_mask.shape}, pinhole camera: {self.pinhole_cam}'
        _mask[_mask > 0] = 1
        return _mask

    @property
    def binary_mask(self):
        _mask = self.mask.copy()
        _mask[_mask > 0] = 1
        return _mask


class ResizedZjuCapture(captures_module.ResizedRigRGBPinholeCapture):
    def __init__(self, image_path, mask_path, pinhole_cam, cam_pose, tgt_size, view_id, cam_id):
        captures_module.ResizedRigRGBPinholeCapture.__init__(self, image_path, pinhole_cam, cam_pose, tgt_size, view_id, cam_id)
        '''
        Note: we pass in the original intrinsic and distortion matrix, NOT the resized intrinsic
        '''
        self.captured_image = contents.ResizedUndistortedCapturedImage(
            image_path,
            pinhole_cam.intrinsic_matrix,
            pinhole_cam.radial_dist,
            tgt_size)
        self.captured_mask = contents.ResizedUndistortedCapturedImage(
            mask_path,
            pinhole_cam.intrinsic_matrix,
            pinhole_cam.radial_dist,
            tgt_size,
            sampling=PIL.Image.NEAREST)

    def read_image_to_ram(self) -> int:
        return self.captured_image.read_image_to_ram() + self.captured_mask.read_image_to_ram()

    @property
    def mask(self):
        _mask = self.captured_mask.image
        assert _mask.shape[0:2] == self.pinhole_cam.shape, f'mask does not match with camera model: mask shape: {_mask.shape}, pinhole camera: {self.pinhole_cam}'
        return _mask

    @property
    def binary_mask(self):
        _mask = self.mask.copy()
        _mask[_mask > 0] = 1
        return _mask


def get_cams(scene_dir):
    intri = cv2.FileStorage(os.path.join(scene_dir, 'intri.yml'), cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(os.path.join(scene_dir, 'extri.yml'), cv2.FILE_STORAGE_READ)
    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    for i in range(23):
        if i in TABU_CAMS:
            continue
        cams['K'].append(intri.getNode('K_Camera_B{}'.format(i + 1)).mat())
        cams['D'].append(
            intri.getNode('dist_Camera_B{}'.format(i + 1)).mat().T)
        cams['R'].append(extri.getNode('Rot_Camera_B{}'.format(i + 1)).mat())
        cams['T'].append(extri.getNode('T_Camera_B{}'.format(i + 1)).mat())
    return cams


def get_img_paths(scene_dir):
    all_ims = []
    for i in range(23):
        if i in TABU_CAMS:
            continue
        data_root = os.path.join(scene_dir, 'Camera_B{}'.format(i + 1))
        ims = glob.glob(os.path.join(data_root, '*.jpg'))
        ims = np.array(sorted(ims))
        all_ims.append(ims)
    num_img = min([len(ims) for ims in all_ims])
    all_ims = [ims[:num_img] for ims in all_ims]
    all_ims = np.stack(all_ims, axis=1)
    return all_ims


def get_mask_paths(scene_dir):
    all_masks = []
    for i in range(23):
        if i in TABU_CAMS:
            continue
        data_root = os.path.join(scene_dir, 'mask_cihp/Camera_B{}'.format(i + 1))
        ims = glob.glob(os.path.join(data_root, '*.png'))
        ims = np.array(sorted(ims))
        all_masks.append(ims)
    num_img = min([len(ims) for ims in all_masks])
    all_masks = [ims[:num_img] for ims in all_masks]
    all_masks = np.stack(all_masks, axis=1)
    return all_masks


def create_split_files(scene_dir):
    # 10% as test set
    # 10% as validation set
    # 80% as training set
    dummy_scene = ZjuMocapReader.read_scene(scene_dir)
    scene_length = len(dummy_scene.captures)
    num_val = scene_length // 5
    length = int(1 / (num_val) * scene_length)
    offset = length // 2
    val_list = list(range(scene_length))[offset::length]
    train_list = list(set(range(scene_length)) - set(val_list))
    test_list = val_list[:len(val_list) // 2]
    val_list = val_list[len(val_list) // 2:]
    assert len(train_list) > 0
    assert len(test_list) > 0
    assert len(val_list) > 0
    splits = []
    for l, split in zip([train_list, val_list, test_list], ['train', 'val', 'test']):
        output = []
        save_path = os.path.join(scene_dir, f'{split}_split.txt')
        for i, cap in enumerate(dummy_scene.captures):
            if i in l:
                output.append(os.path.basename(cap.image_path))
        with open(save_path, 'w') as f:
            for item in output:
                f.write("%s\n" % item)
        splits.append(save_path)
    return splits


class ZjuMocapReader():
    def __init__(self):
        pass

    @classmethod
    def read_scene(cls, scene_dir, tgt_size=None):
        # assert tgt_size is None
        captures, num_views, num_cams = cls.read_captures(scene_dir, tgt_size)
        scene = scene_module.RigCameraScene(captures, num_views, num_cams)
        smpls, world_verts, static_verts, Ts, alignments = cls.read_smpls(scene_dir)
        assert scene.num_views == len(smpls) == len(world_verts) == len(Ts)
        scene.smpls, scene.verts, scene.Ts, scene.alignments = smpls, world_verts, Ts, alignments
        scene.static_vert = static_verts
        _, uvs, faces = utils.read_obj('./sample_data/smplx/smpl_uv.obj')
        scene.uvs, scene.faces = uvs, faces

        # compute the near and far
        for view_id in tqdm(range(scene.num_views), total=scene.num_views, desc='Computing near/far'):
            for cam_id in range(scene.num_cams):
                cur_cap = scene.get_capture_by_view_cam_id(view_id, cam_id)
                # cur_cap.near = 0
                # cur_cap.far = 5
                SCALE = 1.5
                pcd_2d = pcd_projector.project_point_cloud_at_capture(scene.verts[view_id], cur_cap, render_type='pcd')
                near = pcd_2d[:, 2].min()
                far = pcd_2d[:, 2].max()
                center = (near + far) / 2
                length = (far - near) * SCALE
                cur_cap.near = {'bkg': 0.0, 'human': max(0.0, float(center - length / 2))}
                cur_cap.far = {'bkg': 10.0, 'human': float(center + length / 2)}
                cur_cap.frame_id = {'frame_id': view_id, 'total_frames': scene.num_views}
        scene.scale = 1
        return scene

    @classmethod
    def read_captures(cls, scene_dir, tgt_size):
        cams = get_cams(scene_dir)
        T = np.array(cams['T'])
        R = np.array(cams['R'])
        RT = np.concatenate([R, T], axis=2)
        lower_row = np.array([[0., 0., 0., 1.]])
        cam_poses = []
        for rt in RT:
            w2c = np.concatenate([rt, lower_row], axis=0)
            cam_poses.append(CameraPose.from_world_to_camera(w2c))
        K = np.array(cams['K']).astype(np.float32)
        D = np.array(cams['D']).astype(np.float32)
        intrins = []
        for k, d in zip(K, D):
            temp_cam = PinholeCamera.from_intrinsic(1024, 1024, k)
            temp_cam.radial_dist = d
            intrins.append(temp_cam)
        caps = []
        img_paths = get_img_paths(scene_dir)
        mask_maths = get_mask_paths(scene_dir)
        num_views = min(MAX_VIEWS, img_paths.shape[0])
        num_cams = img_paths.shape[1]
        assert num_cams==1
        counter = 0
        for view_id in range(num_views):
            if view_id % EVERY_K != 0:
                continue
            for cam_id in range(num_cams):
                if tgt_size is None:
                    temp = ZjuCapture(
                        img_paths[view_id, cam_id],
                        mask_maths[view_id, cam_id],
                        intrins[cam_id],
                        cam_poses[cam_id],
                        counter, #view_id,
                        cam_id
                    )
                else:
                    temp = ResizedZjuCapture(
                        img_paths[view_id, cam_id],
                        mask_maths[view_id, cam_id],
                        intrins[cam_id],
                        cam_poses[cam_id],
                        tgt_size,
                        counter, #view_id,
                        cam_id
                    )
                temp.id = counter
                counter += 1
                caps.append(temp)
        num_views = len(caps)
        return caps, num_views, num_cams

    @classmethod
    def read_smpls(cls, scene_dir):
        device = torch.device('cpu')
        body_model = SMPL('./sample_data/smplx/smpl',
                               gender='neutral',
                               device=device,
                               )
        smpls = []
        world_verts = []
        static_verts = []
        Ts = []
        alignments = []
        data_root = os.path.join(scene_dir, 'new_params')
        smpl_paths = glob.glob(os.path.join(data_root, '*.npy'))
        smpl_paths = sorted(smpl_paths,key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))[:MAX_VIEWS]
        for i, path in enumerate(smpl_paths):
            if i % EVERY_K != 0:
                continue
            temp_smpl = (
                np.load(path, allow_pickle=True).item()
            )
            temp_smpl['pose'] = temp_smpl.pop('poses')[0]
            temp_smpl['betas'] = temp_smpl.pop('shapes')[0]

            rot_4x4 = np.eye(4)[None]
            rot_4x4[:, :3, :3] = batch_rodrigues(torch.from_numpy(temp_smpl['Rh'])).numpy()
            transl_4x4 = np.eye(4)[None]
            transl_4x4[:, :3, 3] = temp_smpl['Th']
            temp_alignment = np.matmul(transl_4x4, rot_4x4)[0].T
            alignments.append(temp_alignment)

            # 大 pose
            da_smpl = np.zeros_like(temp_smpl['pose'][None])
            da_smpl = da_smpl.reshape(-1, 3)
            da_smpl[1] = np.array([ 0, 0, 1.0])
            da_smpl[2] = np.array([ 0, 0, -1.0])
            da_smpl = da_smpl.reshape(1, -1)

            _, T1 = body_model.verts_transformations(
                return_tensor=False,
                poses=temp_smpl['pose'][None],
                betas=temp_smpl['betas'][None],
                concat_joints=True
                )

            _, T2 = body_model.verts_transformations(
                return_tensor=False,
                poses=da_smpl,
                betas=temp_smpl['betas'][None],
                concat_joints=True
                )
            T = T1 @ np.linalg.inv(T2)
            T = (temp_alignment.T @ T)

            da_pose_verts, da_pose_joints = body_model(return_tensor=False,
                               return_joints=True,
                               poses=da_smpl,
                               betas=temp_smpl['betas'][None]
                               )
            temp_world_verts = np.einsum('BNi, Bi->BN', T, ray_utils.to_homogeneous(np.concatenate([da_pose_verts, da_pose_joints], axis=0)))[:, :3].astype(np.float32)
            temp_world_verts, temp_world_joints = temp_world_verts[:6890, :], temp_world_verts[6890:, :]
            temp_smpl['joints_3d'] = temp_world_joints
            smpls.append(temp_smpl)
            Ts.append(T)
            static_verts.append(da_pose_verts)
            world_verts.append(temp_world_verts)

        return smpls, world_verts, static_verts, Ts, alignments