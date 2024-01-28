import json
import os

import BboxTools as bbt
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from ..utils import subcate_to_cate


class SuperCLEVRTest(Dataset):
    def __init__(self, dataset_path, prefix, anno_path=None, image_h=352, image_w=520, crop_object=False,
                 bg_path=None, category=None, rotate=None, subcategory=None, transform=None):
        super().__init__()
        self.img_path = os.path.join(dataset_path, 'images')
        self.scene_path = os.path.join(dataset_path, 'scenes')
        self.obj_mask_path = os.path.join(dataset_path, 'masks')
        self.anno_path = anno_path
        self.prefix = prefix
        self.category = category
        self.subcategory = subcategory
        self.transform = transform
        self.image_h = image_h
        self.image_w = image_w
        self.crop_object = crop_object
        self.bg_path = bg_path
        self.rotate = rotate

        assert (category is None and subcategory is not None) or (subcategory is None and category is not None)

        self.prepare()

    def prepare(self):
        self.file_list = sorted(
            [x.split('.')[0] for x in os.listdir(self.img_path) if x.startswith(self.prefix) and x.endswith('.png')])

    def __getitem__(self, item):
        img_name = self.file_list[item]
        img = Image.open(os.path.join(self.img_path, img_name + '.png'))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        with open(os.path.join(self.scene_path, img_name + '.json')) as f:
            scene = json.load(f)

        anno = dict(np.load(os.path.join(self.anno_path, img_name + '.npz'), allow_pickle=True))
        object = anno['objects']
        obj = object[0]
        all_distance = obj['distance']
        cub_mask = obj['obj_mask']
        kp = obj['kp']
        kpvis = obj['kpvis']
        principal = obj['principal']

        mw = np.array(scene['matrix_world'])
        mw_inv = np.array(scene['matrix_world_inverted'])
        proj_mat = np.array(scene['projection_matrix'])
        cam_loc = np.array(scene['camera_location'])

        objects = scene['objects']
        anno = []
        for obj in objects:
            anno.append({
                'location': obj['3d_coords'],
                'size_r': obj['size_r'],
                'pixel_coords': obj['pixel_coords'],
                'shape': obj['shape'],
                'theta': obj['theta'],
                'color': obj['color'],
                'category': self.category,
                'subcategory': obj['shape']
            })
        theta = 0
        obj_mask = Image.open(os.path.join(self.obj_mask_path, img_name + '.png'))
        obj_mask = np.array(obj_mask)
        mask = np.zeros_like(obj_mask[:, :, 0])
        mask = np.where(obj_mask[:, :, 0] == 64, 0, 1)
        mask = np.where(obj_mask[:, :, 1] == 64, mask, 1)
        mask = np.where(obj_mask[:, :, 2] == 64, mask, 1)
        img = np.array(img)
        if self.crop_object:

            x_0 = np.where(np.sum(mask, axis=0) != 0)[0][0]
            x_1 = np.where(np.sum(mask, axis=0) != 0)[0][-1]
            y_0 = np.where(np.sum(mask, axis=1) != 0)[0][0]
            y_1 = np.where(np.sum(mask, axis=1) != 0)[0][-1]

            bbox = [x_0, y_0, x_1, y_1]
            resize_rate = float(all_distance / 6)
            center = principal
            center = [int(center[1] * resize_rate), int(center[0] * resize_rate)]
            box = bbt.from_numpy(bbox, sorts=('x0', 'y0', 'x1', 'y1'))
            box_ori = box.copy()

            out_shape = [self.image_h, self.image_w]
            box_ori = box_ori.set_boundary(img.shape[0:2])
            box *= resize_rate
            dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))
            img = cv2.resize(img, dsize=dsize)
            cub_mask = cv2.resize(cub_mask, dsize=dsize)
            mask = mask.astype('float32')
            mask = cv2.resize(mask, dsize=dsize)
            box1 = bbt.box_by_shape(out_shape, center)
            all_distance /= resize_rate

            if out_shape[0] // 2 - center[0] > 0 or out_shape[1] // 2 - center[1] > 0 or out_shape[0] // 2 + center[0] - \
                    img.shape[0] > 0 or out_shape[1] // 2 + center[1] - img.shape[1] > 0:
                if len(img.shape) == 2:
                    padding = (
                        (max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                        (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)))
                else:
                    padding = (
                        (max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                        (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)),
                        (0, 0))

                img = np.pad(img, padding, mode='constant')
                box = box.shift([padding[0][0], padding[1][0]])
                box1 = box1.shift([padding[0][0], padding[1][0]])
                mask = np.pad(mask, padding[0:2], mode='constant')
                cub_mask = np.pad(cub_mask, padding[0:2], mode='constant')

            img_box = box1.set_boundary(img.shape[0:2])
            box_in_cropped = img_box.box_in_box(box)
            img = img_box.apply(img)
            mask = img_box.apply(mask)
            cub_mask = img_box.apply(cub_mask)

            proj_foo = bbt.projection_function_by_boxes(box_ori, box_in_cropped, compose=False)

            for i in range(len(kp)):
                kp[i][0] = proj_foo[1](kp[i][0])
                kp[i][1] = proj_foo[0](kp[i][1])
            principal[0] = proj_foo[1](principal[0])
            principal[1] = proj_foo[0](principal[1])

            kpvis = np.logical_and(kpvis, np.all(kp >= np.zeros_like(kp), axis=1))
            kpvis = np.logical_and(kpvis, np.all(kp < np.array([self.image_w, self.image_h]), axis=1))
            kp = np.max([np.zeros_like(kp), kp], axis=0)
            kp = np.min([np.ones_like(kp) * (np.array([self.image_w, self.image_h]) - 1), kp], axis=0)

        if self.rotate is not None:
            # theta = random.gauss(0, self.rotate)
            theta = self.rotate[item]
            img = Image.fromarray(img)
            cub_mask = Image.fromarray(cub_mask)
            mask = Image.fromarray(np.uint8(mask))
            img = img.rotate(theta)
            cub_mask = cub_mask.rotate(theta)
            mask = mask.rotate(theta)
            img = np.array(img)
            cub_mask = np.array(cub_mask)
            mask = np.array(mask)

            mat = np.array([[np.cos(theta / 180 * np.pi), -np.sin(theta / 180 * np.pi)],
                            [np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi)]])

            kp = ((kp - principal) @ mat).astype('int') + principal

            kpvis = np.logical_and(kpvis, np.all(kp >= np.zeros_like(kp), axis=1))
            kpvis = np.logical_and(kpvis, np.all(kp < np.array([self.image_w, self.image_h]), axis=1))
            kp = np.max([np.zeros_like(kp), kp], axis=0)
            kp = np.min([np.ones_like(kp) * (np.array([self.image_w, self.image_h]) - 1), kp], axis=0)

        if self.bg_path:
            bg_names = os.listdir(self.bg_path)
            idx = np.random.randint(len(bg_names))
            bg = Image.open(os.path.join(self.bg_path, bg_names[idx]))
            bg = bg.resize([img.shape[1], img.shape[0]])
            bg = np.array(bg)
            img_bg = img.copy()
            for j in range(3):
                img_bg[:, :, j] = np.where(mask == 0, bg[:, :, j], img[:, :, j])
            img = img_bg

        img = Image.fromarray(img)

        sample = {
            'img_name': img_name,
            'img': img,
            'mw': mw,
            'mw_inv': mw_inv,
            'kp': kp,
            'kpvis': kpvis,
            'rotate': theta / 180 * np.pi,
            'proj_mat': proj_mat,
            'cam_loc': cam_loc,
            'objects': anno,
            'obj_mask': cub_mask,
            'distance': all_distance
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.file_list)
