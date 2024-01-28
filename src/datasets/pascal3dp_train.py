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


class PASCAL3DPTrain(Dataset):
    def __init__(self, img_path, anno_path, list_file, mesh_path, category, dist=6, image_h=352, image_w=520,
                 crop_object=False,
                 transform=None, bg_path=None, pseudo_label=None):
        super().__init__()
        self.img_path = img_path
        self.anno_path = anno_path
        self.list_file = list_file
        self.category = category
        self.image_h = image_h
        self.image_w = image_w
        self.crop_object = crop_object
        self.transform = transform
        self.file_list = [l.strip() for l in open(self.list_file).readlines()]
        self.file_list = sorted(self.file_list)
        self.bg_path = bg_path
        self.mesh_path = os.path.join(mesh_path, category)
        self.dist = dist
        self.pseudo_label = pseudo_label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        fname = self.file_list[item]
        img = Image.open(os.path.join(self.img_path, fname + '.JPEG'))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        record = sio.loadmat(os.path.join(self.anno_path, fname.split('.')[0] + '.mat'))['record']

        if self.crop_object:
            resize_rate = float(get_anno(record, 'distance') / self.dist)
        else:
            resize_rate = float(min(self.image_h / img.shape[0], self.image_w / img.shape[1]))

        bbox = get_anno(record, 'bbox', idx=0)
        if get_anno(record, 'distance') == 0:
            bbox = np.array([0, 0, self.image_w - 1, self.image_h - 1])
            resize_rate = 1
        box = bbt.from_numpy(bbox, sorts=('x0', 'y0', 'x1', 'y1'))
        box_ori = box.copy()
        box_ori = box_ori.set_boundary(img.shape[0:2])
        box *= resize_rate

        dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))

        if dsize[0] != 0 and dsize[1] != 0:
            img = cv2.resize(img, dsize=dsize)

        if self.crop_object:
            center = get_anno(record, 'principal', idx=0)
            center = [int(center[1] * resize_rate), int(center[0] * resize_rate)]
            if get_anno(record, 'distance') == 0:
                center = (img.shape[0] // 2, img.shape[1] // 2)
        else:
            center = (img.shape[0] // 2, img.shape[1] // 2)
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
        box_in_cropped = img_box.box_in_box(box)

        img_cropped = img_box.apply(img)
        proj_foo = bbt.projection_function_by_boxes(box_ori, box_in_cropped, compose=False)

        principal = get_anno(record, 'principal', idx=0)

        if get_anno(record, 'distance') == 0:
            principal = np.array([self.image_h // 2, self.image_w // 2])

        principal[0] = proj_foo[1](principal[0])
        principal[1] = proj_foo[0](principal[1])

        azimuth = get_anno(record, 'azimuth', idx=0)
        elevation = get_anno(record, 'elevation', idx=0)
        theta = get_anno(record, 'theta', idx=0)
        distance_orig = get_anno(record, 'distance', idx=0)
        distance = self.dist
        fine_cad_idx = get_anno(record, 'cad_index', idx=0)

        if get_anno(record, 'distance') == 0:
            azimuth = 0
            elevation = 0
            theta = 0
            distance_orig = 0
            fine_cad_idx = 0

        if self.pseudo_label:
            labels = self.pseudo_label
            azimuth = labels[fname]['azimuth']
            theta = labels[fname]['theta']
            elevation = labels[fname]['elevation']
            # distance = labels[fname]['distance']

        if self.bg_path:
            bg_names = os.listdir(self.bg_path)
            idx = np.random.randint(len(bg_names))
            bg = Image.open(os.path.join(self.bg_path, bg_names[idx]))
            bg = bg.resize([img_cropped.shape[1], img_cropped.shape[0]])
            bg = np.array(bg)
            mask = np.zeros_like(img_cropped)
            mask = np.where(img_cropped[:, :, 0] == 0, 0, 1)
            mask = np.where(img_cropped[:, :, 1] == 0, mask, 1)
            mask = np.where(img_cropped[:, :, 2] == 0, mask, 1)

            img_bg = img_cropped.copy()

            for j in range(3):
                img_bg[:, :, j] = np.where(mask == 0, bg[:, :, j], img_cropped[:, :, j])
            img_cropped = img_bg

        (y1, y2), (x1, x2) = box_in_cropped.bbox

        xvert, xface = load_off(os.path.join(self.mesh_path, '01.off'), to_torch=True)

        azimuth = (azimuth - np.pi) % (2 * np.pi)
        # azimuth = np.pi * 5 / 6
        # elevation = np.pi / 9
        # theta = 0
        # kp_high_score = np.loadtxt('/home/jiahao/pretrain_6d_pose/test_pascal3d/kp_score_car.txt').astype(int)[0:100]

        params = [azimuth, elevation, theta, distance, 0.5, 0.5]
        pts_2d = project(params, xvert)
        vis = [0, 0, 0, 0, 0, 0]

        if elevation > 0:
            vis[1] = 1
        else:
            vis[0] = 1
        if azimuth > 0 and azimuth < np.pi:
            vis[5] = 1
        else:
            vis[4] = 1
        if azimuth > np.pi / 2 and azimuth < np.pi * 3 / 2:
            vis[3] = 1
        else:
            vis[2] = 1
        #
        breaks = [0] + MESH_FACE_BREAKS_1000[self.category]
        kpvis = np.zeros((breaks[-1],), dtype=np.uint8)
        for i in range(6):
            if vis[i] == 1:
                kpvis[breaks[i]:breaks[i + 1]] = 1

        # kpvis = np.zeros((breaks[-1],), dtype=np.uint8)
        # kpvis[kp_high_score] = 1

        kpvis = np.logical_and(kpvis, np.all(pts_2d >= np.zeros_like(pts_2d), axis=1))
        kpvis = np.logical_and(kpvis, np.all(pts_2d < np.array([self.image_w, self.image_h]), axis=1))
        pts_2d = np.max([np.zeros_like(pts_2d), pts_2d], axis=0)
        pts_2d = np.min([np.ones_like(pts_2d) * (np.array([self.image_w, self.image_h]) - 1), pts_2d], axis=0)

        mask = plot_mesh(np.zeros_like(img_cropped),
                         os.path.join(self.mesh_path, '01.off'), size=(1, 1, 1),
                         focal_length=3000,
                         azimuth=np.float32(azimuth), elevation=np.float32(elevation), theta=np.float32(theta),
                         distance=np.float32(distance),
                         principal=principal.astype(np.float32), down_sample_rate=8)

        obj_mask = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
        obj_mask = np.where(obj_mask == 0, 0, 1)
        #obj_mask = mask

        sample = {
            'img': img_cropped,
            'img_name': fname,
            'azimuth': azimuth,
            'elevation': elevation,
            'theta': theta,
            'distance_orig': distance_orig,
            'distance': distance,
            'fine_cad_idx': fine_cad_idx,
            'resize_rate': resize_rate,
            'principal': principal,
            'bbox': [x1, y1, x2, y2],
            'kp': pts_2d,
            'kpvis': kpvis,
            'obj_mask': obj_mask
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def debug(self, item, fname='debug.png'):
        sample = self.__getitem__(item)
        print(sample['img_name'])
        img = sample['img']
        #img = sample['obj_mask']
        kp = sample['kp']
        kpvis = sample['kpvis']
        p1, p2 = sample['principal']
        # img = cv2.circle(img, (int(p1), int(p2)), 2, (0, 255, 0), -1)
        [x1, y1, x2, y2] = sample['bbox']
        # img = cv2.line(img, (int(x1), int(y1)), (int(x1), int(y2)), (0, 255, 0), 2)
        # img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y1)), (0, 255, 0), 2)
        # img = cv2.line(img, (int(x2), int(y2)), (int(x1), int(y2)), (0, 255, 0), 2)
        # img = cv2.line(img, (int(x2), int(y2)), (int(x2), int(y1)), (0, 255, 0), 2)
        for i in range(len(kp)):
            if kpvis[i] == 1:
                cv2.circle(img, [int(kp[i, 0]), int(kp[i, 1])], radius=1, color=(0, 255, 0))
        Image.fromarray(img).save(fname)
        # mask = sample['obj_mask']
        # Image.fromarray(np.uint8(mask) * 255).save(fname.split('.')[0] + '_mask.png')
