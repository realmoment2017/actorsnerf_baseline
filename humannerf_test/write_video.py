from moviepy.editor import *
import cv2
import glob
import torchvision
import torch
from PIL import Image
import numpy as np
import natsort

folder_name = 'experiments/human_nerf/AIST_mocap/d19/300frame-ablation_skip3/iter_200000/eval_freeview'

def load_image(path, to_rgb=True):
    img = Image.open(path)
    return img.convert('RGB') if to_rgb else img

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def to_8b_image(image):
    return (255.* np.clip(image, 0., 1.)).astype(np.uint8)

def to_3ch_image(image):
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        assert image.shape[2] == 1
        return np.concatenate([image, image, image], axis=-1)
    else:
        print(f"to_3ch_image: Unsupported Shapes: {len(image.shape)}")
        return image

def to_8b3ch_image(image):
    return to_3ch_image(to_8b_image(image))

def write_freeview(folder_name=None):

    imgs = []
    img_paths = natsort.natsorted(glob.glob(os.path.join(folder_name, '*.png')))
    for img_path in img_paths:
        img = load_image(img_path)
        imgs.append(img)
    img_out = np.stack(imgs, axis=0)

    save_path = os.path.join(folder_name)
    torchvision.io.write_video(save_path+'/result.mp4', torch.tensor(img_out), fps=10, video_codec='h264', options={'crf': '10'})
    videoClip  = VideoFileClip(save_path+'/result.mp4')
    videoClip.write_gif(save_path+'/result.gif')

write_freeview(folder_name)
