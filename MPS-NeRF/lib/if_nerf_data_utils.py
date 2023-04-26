import numpy as np
import lib.base_utils as base_utils
import cv2
# from lib.config import cfg
import random
import trimesh
import imageio

pred_image_ = 0

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = base_utils.project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    ray_d[ray_d==0.0] = 1e-8
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    # TODO
    # mask_at_box = p_mask_at_box.sum(-1) >= 1

    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box


def sample_ray_grid(img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    img[bound_mask != 1] = 0

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = cfg.face_sample_ratio
        body_sample_ratio = cfg.body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        # n_body = int((nrays - nsampled_rays) * body_sample_ratio)
        # n_face = int((nrays - nsampled_rays) * face_sample_ratio)
        # n_rand = (nrays - nsampled_rays) - n_body - n_face

        # sample rays on body

        human_fg = (msk != 0).astype(int)
        human_bg = (1-((bound_mask==1).astype(int)* (msk != 0).astype(int))) * ((bound_mask==1).astype(int))
        human_fg_idx = np.argwhere(human_fg)
        human_bg_idx = np.argwhere(human_bg)
        human_idx = np.concatenate([human_fg_idx, human_bg_idx], axis=0)

        prob_list = [cfg.sample_fg_ratio/len(human_fg_idx)]*len(human_fg_idx) + [(1-cfg.sample_fg_ratio)/len(human_bg_idx)]*len(human_bg_idx)
        sample_idx = np.random.choice(np.arange(len(human_idx)), 1, p=prob_list)
        sample_center = human_idx[sample_idx]
        h, w = int(nrays**0.5), int(nrays**0.5)

        def gen_grid(h, w):
            x_ = np.arange(0, h)
            y_ = np.arange(0, w)
            x, y = np.meshgrid(x_, y_, indexing='ij')
            grid = np.stack([x, y], axis=-1).reshape(-1, 2)
            center = -1*np.array([h//2, w//2]).astype(grid.dtype)
            grid += center
            return grid

        coord = sample_center + gen_grid(h, w)

        # move the patch inside the image
        img_h, img_w = img.shape[:2]
        border = np.max([0 - coord[:, 0], coord[:, 0] - (img_h - 1)])
        border = max(border, 0)
        coord[:, 0] = coord[:, 0] - border
        border = np.max([0 - coord[:, 1], coord[:, 1] - (img_w - 1)])
        border = max(border, 0)
        coord[:, 1] = coord[:, 1] - border

        ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
        ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
        rgb_ = img[coord[:, 0], coord[:, 1]]

        near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)
        def pad(input):
            pad_array = np.zeros(nrays,)
            pad_array[:len(input)] = input
            return pad_array
        near_ = pad(near_)
        far_ = pad(far_)

        ray_o_list.append(ray_o_)
        ray_d_list.append(ray_d_)
        rgb_list.append(rgb_)
        near_list.append(near_)
        far_list.append(far_)
        coord_list.append(coord)
        mask_at_box_list.append(mask_at_box)
        nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def sample_ray(img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    img[bound_mask != 1] = 0
    msk = msk * bound_mask

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = cfg.face_sample_ratio
        body_sample_ratio = cfg.body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_face

            # sample rays on body
            coord_body = np.argwhere(msk != 0)
            coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]
            # sample rays on face
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]
            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split, ratio=0.8):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)
    img_ray_d = ray_d.copy()
    img_ray_d = img_ray_d / np.linalg.norm(img_ray_d, axis=-1, keepdims=True)
    pose = np.concatenate([R, T], axis=1)
    
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
    # if split == 'test':
    #     bound_mask = np.ones_like(bound_mask)
    mask_bkgd = True

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    
    if mask_bkgd:
        img[bound_mask != 1] = 0

    if split == 'train':
        
        nsampled_rays = 0
        body_sample_ratio = ratio #0.8
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []
        bkgd_msk_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            index = np.random.randint(0, len(coord_body), n_body)
            # index.sort()
            coord_body = coord_body[index]
            fgd_msk = np.ones_like(coord_body[:,1:])
            # sample rays in the bound mask
            # coord = np.argwhere(bound_mask == 1)
            coord = np.argwhere((bound_mask==1) & (msk!=1))
            index2 = np.random.randint(0, len(coord), n_rand)
            # index2.sort()
            coord = coord[index2]
            bkgd_msk = np.zeros_like(coord[:,1:])
            coord = np.concatenate([coord_body, coord], axis=0)
            bkgd_msk = np.concatenate([fgd_msk, bkgd_msk], axis=0)
            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)
            # mask_at_box = np.ones_like(mask_at_box)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            bkgd_msk_list.append(bkgd_msk[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)
            # nsampled_rays += len(ray_o_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        bkgd_msk = np.concatenate(bkgd_msk_list)
        mask_at_box = np.concatenate(mask_at_box_list)
        
        """
        # Sample all
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)
        """
        # random sample n_rays
        # index = np.random.randint(0, len(ray_o), 4096) # len(ray_o) 4503
        # index = np.array(random.sample(range(0,len(ray_o)), 4096))
        # # index.sort()
        # ray_o = ray_o[index]
        # ray_d = ray_d[index]
        # rgb = rgb[index]
        # near = near[index]
        # far = far[index]
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)
        # bkgd_msk = np.ones_like(msk)
        # bkgd_msk[msk!=100] = msk[msk!=100]
        bkgd_msk = msk.reshape(-1,1)[mask_at_box]

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, bkgd_msk, img_ray_d

def sample_ray_h36m_batch(img, msk, K, R, T, bounds, nrays, split, ratio=0.8, img_path=1):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)
    img_ray_d = ray_d.copy()
    img_ray_d = img_ray_d / np.linalg.norm(img_ray_d, axis=-1, keepdims=True)
    pose = np.concatenate([R, T], axis=1)
    
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    mask_bkgd = True

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    bound_mask[msk == 200] = 0
    
    if mask_bkgd:
        img[bound_mask != 1] = 0
    # img[bound_mask != 1] = 0

    if split == 'train':
        
        nsampled_rays = 0
        body_sample_ratio = ratio #0.8
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []
        bkgd_msk_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_rand = int(((nrays - nsampled_rays) - n_body) * 0.5)
            n_rand_2 = ((nrays - nsampled_rays) - n_body - n_rand)
            # n_rand = (nrays - nsampled_rays) - n_body

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            index = np.random.randint(0, len(coord_body), n_body)
            # index.sort()
            coord_body = coord_body[index]
            fgd_msk = np.ones_like(coord_body[:,1:])
            
            # sample rays in the second background
            # coord = np.argwhere(bound_mask == 1)
            coord = np.argwhere((bound_mask==1) & (msk!=1))
            if len(coord) <= n_rand:
                print(img_path)
                imageio.imwrite("./objs/msk.png", msk)
                imageio.imwrite("./objs/bound_mask.png", bound_mask*255)
                imageio.imwrite("./objs/img.png", img*255)
            index2 = np.random.randint(0, len(coord), n_rand)
            # index2.sort()
            coord = coord[index2]
            bkgd_msk = np.zeros_like(coord[:,1:])

            # sample rays in the first background
            coord_3 = np.argwhere((bound_mask==0) & (msk==200))
            index3 = np.random.randint(0, len(coord_3), n_rand_2)
            # index2.sort()
            coord_3 = coord_3[index3]
            bkgd_msk_3 = np.zeros_like(coord_3[:,1:])
            coord = np.concatenate([coord_body, coord, coord_3], axis=0)
            bkgd_msk = np.concatenate([fgd_msk, bkgd_msk, bkgd_msk_3], axis=0)

            # coord = np.concatenate([coord_body, coord], axis=0)
            # bkgd_msk = np.concatenate([fgd_msk, bkgd_msk], axis=0)
            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)
            # mask_at_box = np.ones_like(mask_at_box)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            bkgd_msk_list.append(bkgd_msk[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)
            # nsampled_rays += len(ray_o_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        bkgd_msk = np.concatenate(bkgd_msk_list)
        mask_at_box = np.concatenate(mask_at_box_list)
        
        
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)

        near_all = np.zeros_like(ray_o[:,0])
        far_all = np.ones_like(ray_o[:,0])
        near_all[mask_at_box] = near 
        far_all[mask_at_box] = far 
        near = near_all
        far = far_all


        # rgb = rgb[mask_at_box]
        # ray_o = ray_o[mask_at_box]
        # ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64) # no use
        bkgd_msk = np.ones_like(msk) # no use
        # bkgd_msk[msk!=100] = msk[msk!=100]
        # bkgd_msk = msk.reshape(-1,1)[mask_at_box]

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, bkgd_msk, img_ray_d

def sample_ray_neubody_batch(img, msk, K, R, T, bounds, nrays, split, ratio=0.8):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)
    img_ray_d = ray_d.copy()
    img_ray_d = img_ray_d / np.linalg.norm(img_ray_d, axis=-1, keepdims=True)
    pose = np.concatenate([R, T], axis=1)
    
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    mask_bkgd = True

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    # bound_mask[msk == 200] = 0
    
    # if mask_bkgd:
    #     img[bound_mask != 1] = 0

    if split == 'train':
        
        nsampled_rays = 0
        body_sample_ratio = ratio #0.8
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []
        bkgd_msk_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            # n_rand = int(((nrays - nsampled_rays) - n_body) * 0.5)
            # n_rand_2 = ((nrays - nsampled_rays) - n_body - n_rand)
            n_rand = (nrays - nsampled_rays) - n_body

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            index = np.random.randint(0, len(coord_body), n_body)
            # index.sort()
            coord_body = coord_body[index]
            fgd_msk = np.ones_like(coord_body[:,1:])
            
            # sample rays in the second background
            # coord = np.argwhere(bound_mask == 1)
            coord = np.argwhere((bound_mask==1) & (msk!=1))
            index2 = np.random.randint(0, len(coord), n_rand)
            # index2.sort()
            coord = coord[index2]
            bkgd_msk = np.zeros_like(coord[:,1:])

            # sample rays in the first background
            # coord_3 = np.argwhere((bound_mask==0) & (msk==200))
            # index3 = np.random.randint(0, len(coord_3), n_rand_2)
            # # index2.sort()
            # coord_3 = coord_3[index3]
            # bkgd_msk_3 = np.zeros_like(coord_3[:,1:])
            # coord = np.concatenate([coord_body, coord, coord_3], axis=0)
            # bkgd_msk = np.concatenate([fgd_msk, bkgd_msk, bkgd_msk_3], axis=0)

            coord = np.concatenate([coord_body, coord], axis=0)
            bkgd_msk = np.concatenate([fgd_msk, bkgd_msk], axis=0)
            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)
            # mask_at_box = np.ones_like(mask_at_box)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            bkgd_msk_list.append(bkgd_msk[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)
            # nsampled_rays += len(ray_o_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        bkgd_msk = np.concatenate(bkgd_msk_list)
        mask_at_box = np.concatenate(mask_at_box_list)
        
        
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)

        near_all = np.zeros_like(ray_o[:,0])
        far_all = np.ones_like(ray_o[:,0])
        near_all[mask_at_box] = near 
        far_all[mask_at_box] = far 
        near = near_all
        far = far_all


        # rgb = rgb[mask_at_box]
        # ray_o = ray_o[mask_at_box]
        # ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64) # no use
        bkgd_msk = np.ones_like(msk) # no use
        # bkgd_msk[msk!=100] = msk[msk!=100]
        # bkgd_msk = msk.reshape(-1,1)[mask_at_box]

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, bkgd_msk, img_ray_d


def sample_ray_THuman_batch(img, msk, K, R, T, bounds, nrays, split, ratio=0.8):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)
    img_ray_d = ray_d.copy()
    img_ray_d = img_ray_d / np.linalg.norm(img_ray_d, axis=-1, keepdims=True)
    pose = np.concatenate([R, T], axis=1)
    
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    mask_bkgd = True

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    
    if mask_bkgd:
        img[bound_mask != 1] = 0
    # img[bound_mask != 1] = 0

    if split == 'train':
        
        nsampled_rays = 0
        body_sample_ratio = ratio #0.8
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []
        bkgd_msk_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            index = np.random.randint(0, len(coord_body), n_body)
            # index.sort()
            coord_body = coord_body[index]
            fgd_msk = np.ones_like(coord_body[:,1:])
            
            # sample rays in the second background
            # coord = np.argwhere(bound_mask == 1)
            coord = np.argwhere((bound_mask==1) & (msk!=1))
            index2 = np.random.randint(0, len(coord), n_rand)
            # index2.sort()
            coord = coord[index2]
            bkgd_msk = np.zeros_like(coord[:,1:])

            coord = np.concatenate([coord_body, coord], axis=0)
            bkgd_msk = np.concatenate([fgd_msk, bkgd_msk], axis=0)
            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)
            # mask_at_box = np.ones_like(mask_at_box)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            bkgd_msk_list.append(bkgd_msk[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)
            # nsampled_rays += len(ray_o_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        bkgd_msk = np.concatenate(bkgd_msk_list)
        mask_at_box = np.concatenate(mask_at_box_list)
        
        
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)

        near_all = np.zeros_like(ray_o[:,0])
        far_all = np.ones_like(ray_o[:,0])
        near_all[mask_at_box] = near 
        far_all[mask_at_box] = far 
        near = near_all
        far = far_all


        # rgb = rgb[mask_at_box]
        # ray_o = ray_o[mask_at_box]
        # ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64) # no use
        bkgd_msk = np.ones_like(msk) # no use
        # bkgd_msk[msk!=100] = msk[msk!=100]
        # bkgd_msk = msk.reshape(-1,1)[mask_at_box]

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, bkgd_msk, img_ray_d

def sample_ray_THuman(img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)
    img_ray_d = ray_d.copy()
    img_ray_d = img_ray_d / np.linalg.norm(img_ray_d, axis=-1, keepdims=True)
    pose = np.concatenate([R, T], axis=1)
    
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
    # all image sample instead of bounding box sample
    # bound_mask = np.ones_like(bound_mask)
    # mask_bkgd = True

    # msk = msk * bound_mask
    # bound_mask[msk == 100] = 0
    
    # if mask_bkgd:
    #     img[bound_mask != 1] = 0

    if split == 'train':
        
        nsampled_rays = 0
        body_sample_ratio = 0.8 #0.8
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []
        bkgd_msk_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            index = np.random.randint(0, len(coord_body), n_body)
            # index.sort()
            coord_body = coord_body[index]
            fgd_msk = np.ones_like(coord_body[:,1:])
            # sample rays in the bound mask
            coord = np.argwhere((bound_mask == 1) & (msk!=1))
            # coord = np.argwhere(msk!=1)
            index2 = np.random.randint(0, len(coord), n_rand)
            # index2.sort()
            coord = coord[index2]
            bkgd_msk = np.zeros_like(coord[:,1:])
            coord = np.concatenate([coord_body, coord], axis=0)
            bkgd_msk = np.concatenate([fgd_msk, bkgd_msk], axis=0)
            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            # near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_) # mask_at_box are all one during training
            mask_at_box = np.ones_like(ray_o_[...,0]).astype(np.int)
            near_ = np.ones_like(ray_o_[...,0]) * 2.
            far_ = np.ones_like(ray_o_[...,0]) * 3.

            ray_o_list.append(ray_o_) # [mask_at_box])
            ray_d_list.append(ray_d_) # [mask_at_box])
            rgb_list.append(rgb_) # [mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord) # [mask_at_box])
            bkgd_msk_list.append(bkgd_msk) # [mask_at_box])
            mask_at_box_list.append(mask_at_box) # [mask_at_box])
            nsampled_rays += len(near_)
            # nsampled_rays += len(ray_o_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        bkgd_msk = np.concatenate(bkgd_msk_list)
        mask_at_box = np.concatenate(mask_at_box_list)
        
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d) # mask_at_box are not all one during test
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        # mask_at_box = np.ones_like(bound_mask).reshape(-1).astype(np.int)
        # mask_at_box = np.ones_like(ray_o[...,0]).astype(np.int)
        
        near_all = np.zeros_like(ray_o[:,0])
        far_all = np.ones_like(ray_o[:,0])
        near_all[mask_at_box] = near 
        far_all[mask_at_box] = far 
        near = near_all
        far = far_all
        
        # rgb = rgb[mask_at_box]
        # ray_o = ray_o[mask_at_box]
        # ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)
        bkgd_msk = np.ones_like(msk) # For acc loss, useless in test
        # bkgd_msk[msk!=100] = msk[msk!=100]

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, bkgd_msk, img_ray_d


def get_smpl_data(ply_path):
    ply = trimesh.load(ply_path)
    xyz = np.array(ply.vertices)
    nxyz = np.array(ply.vertex_normals)

    if cfg.add_pointcloud:
        # add random points
        xyz_, ind_ = trimesh.sample.sample_surface_even(ply, 5000)
        nxyz_ = ply.face_normals[ind_]
        xyz = np.concatenate([xyz, xyz_], axis=0)
        nxyz = np.concatenate([nxyz, nxyz_], axis=0)

    xyz = xyz.astype(np.float32)
    nxyz = nxyz.astype(np.float32)

    return xyz, nxyz


def get_acc(coord, msk):
    acc = msk[coord[:, 0], coord[:, 1]]
    acc = (acc != 0).astype(np.uint8)
    return acc


def rotate_smpl(xyz, nxyz, t):
    """
    t: rotation angle
    """
    xyz = xyz.copy()
    nxyz = nxyz.copy()
    center = (np.min(xyz, axis=0) + np.max(xyz, axis=0)) / 2
    xyz = xyz - center
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    R = R.astype(np.float32)
    xyz[:, :2] = np.dot(xyz[:, :2], R.T)
    xyz = xyz + center
    # nxyz[:, :2] = np.dot(nxyz[:, :2], R.T)
    return xyz, nxyz, center


def transform_can_smpl(xyz):
    center = np.array([0, 0, 0]).astype(np.float32)
    rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
    rot = rot.astype(np.float32)
    trans = np.array([0, 0, 0]).astype(np.float32)
    rot_ratio = 0.
    if np.random.uniform() > rot_ratio:
        return xyz, center, rot, trans

    xyz = xyz.copy()

    # rotate the smpl
    rot_range = np.pi / 32
    t = np.random.uniform(-rot_range, rot_range)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot = rot.astype(np.float32)
    center = np.mean(xyz, axis=0)
    xyz = xyz - center
    xyz[:, [0, 2]] = np.dot(xyz[:, [0, 2]], rot.T)
    xyz = xyz + center

    # translate the smpl
    x_range = 0.05
    z_range = 0.025
    x_trans = np.random.uniform(-x_range, x_range)
    z_trans = np.random.uniform(-z_range, z_range)
    trans = np.array([x_trans, 0, z_trans]).astype(np.float32)
    xyz = xyz + trans

    return xyz, center, rot, trans


def sample_ray_ohem(img, loss_img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = 0
        body_sample_ratio = 0
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_face

            # sample rays on body
            coord_body = np.argwhere(msk != 0)
            coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]
            # sample rays on face
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]

            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            loss = loss_img[coord[:, 0], coord[:, 1]]
            loss = loss / loss.sum()
            n_rand0 = int(n_rand * 0.8)
            ind = np.random.choice(np.arange(len(coord)),
                                   n_rand0,
                                   replace=False,
                                   p=loss)
            coord0 = coord[ind]
            coord1 = coord[np.random.randint(0, len(coord), n_rand - n_rand0)]
            coord = np.concatenate([coord0, coord1], axis=0)

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box)
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def unproject(depth, K, R, T):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xyz = xy1 * depth[..., None]
    pts3d = np.dot(xyz, np.linalg.inv(K).T)
    pts3d = np.dot(pts3d - T.ravel(), R)
    return pts3d


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(poses, joints, parents):
    """
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    """
    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints
    transforms = transforms.astype(np.float32)

    return transforms
