import math
import os

import BboxTools as bbt
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.utils import MESH_FACE_BREAKS_1000, load_off, plot_mesh


def get_anno(record, *args, idx=0):
    out = []
    for key_ in args:
        if key_ == 'class':
            out.append(record['objects'][0, 0]['class'][0, idx])
        elif key_ == 'difficult':
            out.append(record['objects'][0, 0]['difficult'][0, idx])
        elif key_ == 'height':
            out.append(record['imgsize'][0, 0][0][1])
        elif key_ == 'width':
            out.append(record['imgsize'][0, 0][0][0])
        elif key_ == 'bbox':
            out.append(record['objects'][0, 0]['bbox'][0, idx][0])
        elif key_ == 'cad_index':
            if len(record['objects'][0, 0]['cad_index'][0, idx][0]) == 0:
                out.append(None)
            else:
                out.append(record['objects'][0, 0]['cad_index'][0, idx][0, 0])
        elif key_ == 'principal':
            if len(record['objects'][0, 0]['viewpoint'][0, idx][0]) == 0:
                out.append(np.array([None, None]))
            else:
                px = record['objects'][0, 0]['viewpoint'][0, idx]['px'][0, 0][0, 0]
                py = record['objects'][0, 0]['viewpoint'][0, idx]['py'][0, 0][0, 0]
                out.append(np.array([px, py]))
        elif key_ in ['theta', 'azimuth', 'elevation']:
            if len(record['objects'][0, 0]['viewpoint'][0, idx][0]) == 0:
                out.append(None)
            else:
                out.append(record['objects'][0, 0]['viewpoint'][0, idx][key_][0, 0][0, 0] * math.pi / 180)
        else:
            if len(record['objects'][0, 0]['viewpoint'][0, idx][0]) == 0:
                out.append(None)
            else:
                out.append(record['objects'][0, 0]['viewpoint'][0, idx][key_][0, 0][0, 0])

    if len(out) == 1:
        return out[0]

    return tuple(out)


def project(params, pts):
    azimuth, elevation, theta, distance, px, py = params

    C = np.zeros((3, 1))
    C[0] = distance * math.cos(elevation) * math.sin(azimuth)
    C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
    C[2] = distance * math.sin(elevation)

    azimuth = -azimuth
    elevation = - (math.pi / 2 - elevation)

    Rz = np.array([
        [math.cos(azimuth), -math.sin(azimuth), 0],
        [math.sin(azimuth), math.cos(azimuth), 0],
        [0, 0, 1],
    ])  # rotation by azimuth
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(elevation), -math.sin(elevation)],
        [0, math.sin(elevation), math.cos(elevation)],
    ])  # rotation by elevation
    R_rot = np.dot(Rx, Rz)
    R = np.hstack((R_rot, np.dot(-R_rot, C)))
    R = np.vstack((R, [0, 0, 0, 1]))

    P = np.array([[3000, 0, 0],
                  [0, 3000, 0],
                  [0, 0, -1]]).dot(R[:3, :4])

    x3d = pts
    x3d_ = np.hstack((x3d, np.ones((len(x3d), 1)))).T

    x2d = np.dot(P, x3d_)  # 3x4 * 4x8 = 3x8
    x2d[0, :] = x2d[0, :] / x2d[2, :]
    x2d[1, :] = x2d[1, :] / x2d[2, :]
    x2d = x2d[0:2, :]

    R2d = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    x2d = np.dot(R2d, x2d).T

    x2d[:, 1] *= -1

    x2d[:, 0] += px * 520
    x2d[:, 1] += py * 352

    # x2d = x2d - (x2d[8] - pts2d[8])
    # return x2d

    return x2d


class PASCAL3DPOcc(Dataset):
    def __init__(self, img_path, anno_path, dist=6.0, image_h=352, image_w=520,
                 transform=None, bg_path=None):
        super().__init__()
        self.img_path = img_path
        self.anno_path = anno_path
        self.image_h = image_h
        self.image_w = image_w
        self.transform = transform
        self.dist = dist
        self.file_list = [file.split('.')[0] for file in os.listdir(self.img_path)]
        self.bg_path = bg_path

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        fname = self.file_list[item]
        img = Image.open(os.path.join(self.img_path, fname + '.JPEG'))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        record = sio.loadmat(os.path.join(self.anno_path, fname.split('.')[0] + '.mat'))['record']

        azimuth = get_anno(record, 'azimuth', idx=0)
        elevation = get_anno(record, 'elevation', idx=0)
        theta = get_anno(record, 'theta', idx=0)
        # resize_rate = anno['distance'].item() / self.dist
        resize_rate = float(get_anno(record, 'distance') / self.dist)

        # bbox = anno['bbox'].astype(int)
        bbox = get_anno(record, 'bbox', idx=0)
        box = bbt.from_numpy(bbox, sorts=('x0', 'y0', 'x1', 'y1'))
        box *= resize_rate

        dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))
        if dsize[0] != 0 and dsize[1] != 0:
            img = cv2.resize(img, dsize=dsize)

        center = get_anno(record, 'principal', idx=0)
        center = [int(center[1] * resize_rate), int(center[0] * resize_rate)]
        out_shape = [self.image_h, self.image_w]

        box1 = bbt.box_by_shape(out_shape, center)
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
        img_box = box1.set_boundary(img.shape[0:2])
        img_cropped = img_box.apply(img)

        if self.bg_path:
            bg_names = os.listdir(self.bg_path)
            idx = np.random.randint(len(bg_names))
            bg = Image.open(os.path.join(self.bg_path, bg_names[idx]))
            bg = bg.resize([img_cropped.shape[1], img_cropped.shape[0]])
            bg = np.array(bg)
            mask = np.where(img_cropped[:, :, 0] == 0, 0, 1)
            mask = np.where(img_cropped[:, :, 1] == 0, mask, 1)
            mask = np.where(img_cropped[:, :, 2] == 0, mask, 1)

            img_bg = img_cropped.copy()

            for j in range(3):
                img_bg[:, :, j] = np.where(mask == 0, bg[:, :, j], img_cropped[:, :, j])
            img_cropped = img_bg

        sample = {
            'img': img_cropped,
            'img_name': fname,
            'azimuth': azimuth,
            'elevation': elevation,
            'theta': theta,
            'distance': self.dist,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def debug(self, item, fname='debug.png'):
        sample = self.__getitem__(item)
        print(sample['img_name'])
        img = sample['img']
        Image.fromarray(img).save(fname)
