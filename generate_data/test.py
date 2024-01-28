import _init_paths

import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from src.utils import subcate_to_cate, load_off, MESH_FACE_BREAKS_1000


def get_rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


"""
def str_to_mask(s, h, w):
    lis = [int(x) for x in s.split(',')]
    img = []
    cur = 0
    for l in lis:
        img += [cur] * l
        cur = 1 - cur
    img = np.array(img, dtype=np.uint8).reshape(h, w)
    return img
"""


"""
def get_transform(scale, theta, loc):
    rotate_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    scale_mat = np.array([
        [scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, 1]
    ])
    translate_mat = np.array([
        [0, 0, 0, loc[0]], [0, 0, 0, loc[1]], [0, 0, 0, loc[2]], [0, 0, 0, 1]
    ])
    return np.dot(translate_mat, np.dot(scale_mat, rotate_mat))
"""


def transform(xvert, theta, scale, loc):
    rotate_mat = get_rot_z(theta / 180. * math.pi)
    xvert = (rotate_mat @ xvert.transpose((1, 0))).transpose((1, 0))
    xvert = xvert * scale + loc
    return xvert


def get_nocs_colors(xvert):
    xvert = xvert.copy()
    xvert[:, 0] -= (np.min(xvert[:, 0]) + np.max(xvert[:, 0]))/2.0
    xvert[:, 1] -= (np.min(xvert[:, 1]) + np.max(xvert[:, 1]))/2.0
    xvert[:, 2] -= (np.min(xvert[:, 2]) + np.max(xvert[:, 2]))/2.0
    xvert[:, 0] /= np.max(np.abs(xvert[:, 0])) * 2.0
    xvert[:, 1] /= np.max(np.abs(xvert[:, 1])) * 2.0
    xvert[:, 2] /= np.max(np.abs(xvert[:, 2])) * 2.0
    xvert += 0.5
    xvert = np.rint(np.clip(xvert * 255.0, 0, 255)).astype(np.uint8)
    return xvert


def dist(loc, cam_loc):
    return np.sqrt(np.sum((np.array(loc)-np.array(cam_loc))**2))


def get_face_centers(xvert):
    xmin, xmax = np.min(xvert[:, 0]), np.max(xvert[:, 0])
    ymin, ymax = np.min(xvert[:, 1]), np.max(xvert[:, 1])
    zmin, zmax = np.min(xvert[:, 2]), np.max(xvert[:, 2])
    centers = np.array([
        [(xmin+xmax)*0.5, (ymin+ymax)*0.5, zmin],
        [(xmin+xmax)*0.5, (ymin+ymax)*0.5, zmax],
        [(xmin+xmax)*0.5, ymin, (zmin+zmax)*0.5],
        [(xmin+xmax)*0.5, ymax, (zmin+zmax)*0.5],
        [xmin, (ymin+ymax)*0.5, (zmin+zmax)*0.5],
        [xmax, (ymin+ymax)*0.5, (zmin+zmax)*0.5],
    ])
    return centers


def get_visibility(face_centers, cam_loc, cate):
    vis = [0, 1, 0, 0, 0, 0]

    dist_list = [dist(c, cam_loc) for c in face_centers]
    if dist_list[2] <= dist_list[3]:
        vis[2] = 1
    else:
        vis[3] = 1
    if dist_list[4] <= dist_list[5]:
        vis[4] = 1
    else:
        vis[5] = 1
    
    print(vis)

    breaks = [0] + MESH_FACE_BREAKS_1000[cate]
    kpvis = np.zeros((breaks[-1],), dtype=np.uint8)
    for i in range(6):
        if vis[i] == 1:
            kpvis[breaks[i]:breaks[i+1]] = 1
    return kpvis


with open('/mnt/sdf/wufei/data/SuperCLEVR_20220317/scenes/superCLEVR_train_000003.json') as f:
    scene = json.load(f)
img = Image.open('/mnt/sdf/wufei/data/SuperCLEVR_20220317/images/superCLEVR_train_000003.png')
img = np.array(img.convert('RGB'))

mw = np.array(scene['matrix_world'])
mw_inv = np.array(scene['matrix_world_inverted'])
proj_mat = np.array(scene['projection_matrix'])
cam_loc = np.array(scene['camera_location'])

# mask = str_to_mask(scene['obj_mask_box']['0']['obj'][1], 320, 448)
# img[mask == 1] = [255, 255, 255]

objects = scene['objects']
dist_list = [dist(obj['location'], cam_loc) for obj in objects]
objects = [obj for _, obj in sorted(zip(dist_list, objects))]

occ = np.zeros((320, 448), dtype=np.uint8)
for obj_idx in range(len(objects)):
    obj = objects[obj_idx]
    theta = obj['rotation']
    obj_name = obj['shape']
    size_name = obj['size']
    scale = obj['size_r']
    x, y = obj['3d_coords'][:2]
    loc = obj['location']

    xvert, xface = load_off(os.path.join('/home/wufei/Documents/pretrain_6d_pose/data/CAD_cate', subcate_to_cate[obj_name], '01.off'))
    face_centers = get_face_centers(xvert)
    colors = get_nocs_colors(xvert)

    xvert = transform(xvert, theta, scale, loc)
    pts_3d = np.concatenate([xvert.transpose((1, 0)), np.ones((1, len(xvert)), dtype=np.float32)], axis=0)
    face_centers = transform(face_centers, theta, scale, loc)
    kpvis = get_visibility(face_centers, cam_loc, subcate_to_cate[obj_name])

    P = np.dot(proj_mat, mw_inv)
    pts_2d = np.dot(P, pts_3d)
    pts_2d[0, :] = pts_2d[0, :] / pts_2d[3, :] * 224 + 224
    pts_2d[1, :] = -pts_2d[1, :] / pts_2d[3, :] * 160 + 160
    pts_2d = np.rint(pts_2d.transpose((1, 0))[:, :2]).astype(np.int32)

    if obj_idx >= 0:
        for i, p in enumerate(pts_2d):
            if kpvis[i] > 0:
                c = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
                if p[1] >= 0 and p[1] < 320 and p[0] >= 0 and p[0] < 448 and occ[p[1], p[0]] == 0:
                    img = cv2.circle(img, (p[0], p[1]), 1, c, -1)
    
    hull = ConvexHull(pts_2d).vertices
    img = cv2.polylines(img, [pts_2d[hull].reshape(-1, 1, 2)], True, (255, 255, 255), 1)
    occ = cv2.fillPoly(occ, [pts_2d[hull].reshape(-1, 1, 2)], 1)

Image.fromarray(img).save('kps.png')
Image.fromarray(occ*255).save('occ.png')
