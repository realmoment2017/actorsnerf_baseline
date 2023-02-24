import os

import torch
import numpy as np
from tqdm import tqdm
import mcubes
import trimesh

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

from configs import cfg, args

# import kornia

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'sub_idx', 'cam_idx'
                       'img_width', 'img_height', 'ray_mask']

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']

def load_network():
    model = create_network()
    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    for m1, _ in model.named_parameters():
        if m1 not in ckpt['network'].keys():
            raise Exception("model not match")
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    # return model.cuda().deploy_mlps_to_secondary_gpus()
    return torch.nn.DataParallel(model.cuda())


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
    test_loader = create_dataloader(data_type)

    model.eval()
    for batch in tqdm(test_loader):

        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        # iter_val = torch.full((batch['img'].shape[0],), cfg.eval_iter)
        with torch.no_grad():
            net_output = model(**data)

        alpha = net_output['alpha']
        print(alpha.min(), alpha.max(), alpha.mean())

        cnl_bbox_min_xyz = data['cnl_bbox_min_xyz']
        cnl_bbox_max_xyz = data['cnl_bbox_max_xyz']
        x = torch.arange(cnl_bbox_min_xyz[0], cnl_bbox_max_xyz[0] + cfg.voxel_size[0],
                    cfg.voxel_size[0])
        y = torch.arange(cnl_bbox_min_xyz[1], cnl_bbox_max_xyz[1] + cfg.voxel_size[1],
                    cfg.voxel_size[1])
        z = torch.arange(cnl_bbox_min_xyz[2], cnl_bbox_max_xyz[2] + cfg.voxel_size[2],
                    cfg.voxel_size[2])
        pts = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        sh = pts.shape[:-1]
        cube = torch.zeros(sh)
        cube = alpha.view(sh).cpu().detach().numpy()

        # cube = np.pad(cube, 10, mode='constant')
        # vertices, triangles = mcubes.marching_cubes(cube, 5)
        # vertices = (vertices - 10) * cfg.voxel_size[0]
        # vertices = vertices + cnl_bbox_min_xyz[0, 0].detach().cpu().numpy()

        # mesh = trimesh.Trimesh(vertices,
        #                         triangles,
        #                         process=False)

        # result_dir = os.path.join(cfg.logdir, cfg.load_net)
        # os.system('mkdir -p {}'.format(result_dir))
        # mesh_path = os.path.join(result_dir, 'tpose_mesh.ply')
        # mesh.export(mesh_path)
        # print("mesh exported")

        result_dir = os.path.join(cfg.logdir, cfg.load_net)
        os.system('mkdir -p {}'.format(result_dir))
        mesh_path = os.path.join(result_dir, 'tpose.mrc')
        import mrcfile
        with mrcfile.new_mmap(mesh_path, overwrite=True, shape=cube.shape, mrc_mode=2) as mrc:
                mrc.data[:] = cube

        break

def run_vis_mesh():
    _eval(
        data_type='train',
        dataset_mode='cat_level',
        folder_name=f"vis_mesh")

def run_tpose():
    _eval(
        data_type='tpose',
        dataset_mode='cat_level',
        folder_name=f"tpose")
        
if __name__ == '__main__':
    globals()[f'run_{args.type}']()
