import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset

SCANNET_FAR = 2.0


class ScanNetDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.unpad = 24

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        K = np.loadtxt(os.path.join(self.root_dir, "./intrinsic/intrinsic_color.txt"))[:3, :3]
        H, W = 968 - 2 * self.unpad, 1296 - 2 * self.unpad
        K[:2, 2] -= self.unpad
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(H, W, self.K)
        self.img_wh = (W, H)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'train':
            with open(os.path.join(self.root_dir, "train.txt"), 'r') as f:
                frames = f.read().strip().split()
                frames = frames[:800]
        else:
            with open(os.path.join(self.root_dir, f"{split}.txt"), 'r') as f:
                frames = f.read().strip().split()
                frames = frames[:80]

        cam_bbox = np.loadtxt(os.path.join(self.root_dir, f"cam_bbox.txt"))
        sbbox_scale = (cam_bbox[1] - cam_bbox[0]).max() + 2 * SCANNET_FAR
        sbbox_shift = cam_bbox.mean(axis=0)

        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            c2w = np.loadtxt(os.path.join(self.root_dir, f"pose/{frame}.txt"))[:3]

            # add shift
            c2w[0, 3] -= sbbox_shift[0]
            c2w[1, 3] -= sbbox_shift[1]
            c2w[2, 3] -= sbbox_shift[2]
            c2w[:, 3] /= sbbox_scale

            self.poses += [c2w]

            try:
                img_path = os.path.join(self.root_dir, f"color/{frame}.jpg")
                img = read_image(img_path, self.img_wh, unpad=self.unpad)
                self.rays += [img]
            except: pass

        if len(self.rays)>0:
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
