import _init_paths

import argparse
import concurrent.futures
import json
import math
import os
import random
import shutil

import cv2
import multiprocessing
import numpy as np
from PIL import Image
import scipy.io as sio
from scipy.spatial import ConvexHull
from tqdm import tqdm

from src.models.calculate_point_direction import cal_point_weight, direction_calculator
from src.models.calculate_occ import cal_occ_one_image
from src.utils import str2bool, subcate_to_cate, load_off, MESH_FACE_BREAKS_1000


def parse_args():
    parser = argparse.ArgumentParser('Create SuperCLEVR 6D pose training set')
    parser.add_argument('--save_path', type=str, default='/mnt/sdf/wufei/data')
    parser.add_argument('--dataset_name', type=str, default='superclevr_20220506_train_subtype')
    parser.add_argument('--workers', type=int, default=10)

    parser.add_argument('--categories', type=str, default=['car'])

    # Data args
    parser.add_argument('--image_h', type=int, default=320)
    parser.add_argument('--image_w', type=int, default=448)
    parser.add_argument('--mesh_path', type=str, default='/home/wufei/Documents/pretrain_6d_pose/data/CGParts_remesh_consistent')
    parser.add_argument('--dataset_path', type=str, default='/mnt/sdf/wufei/data/SuperCLEVR_20220506')
    parser.add_argument('--filename_prefix', type=str, default='superCLEVR')

    return parser.parse_args()


def get_rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


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


def transform(xvert, theta, scale, loc):
    rotate_mat = get_rot_z(theta / 180. * math.pi)
    xvert = (rotate_mat @ xvert.transpose((1, 0))).transpose((1, 0))
    xvert = xvert * scale + loc
    return xvert


def dist(loc, cam_loc):
    return np.sqrt(np.sum((np.array(loc)-np.array(cam_loc))**2))


def prepare_object(obj_dict, cam_loc, occ, proj_mat, mw_inv, args):
    theta = obj_dict['rotation']
    subcate = obj_dict['shape']
    cate = subcate_to_cate[subcate]
    scale = obj_dict['size_r']
    x, y = obj_dict['3d_coords'][:2]
    loc = obj_dict['location']

    if not os.path.isfile(os.path.join(args.mesh_path, subcate, 'mesh.off')):
        return {'category': cate, 'sub_category': subcate, 'theta': theta, 'loc': loc, 'scale': scale, 'kp': None, 'kpvis': None,
                'obj_mask': None, 'distance': dist(loc, cam_loc)}, occ

    xvert, xface = load_off(os.path.join(args.mesh_path, subcate, 'mesh.off'))
    xvert = transform(xvert, theta, scale, loc)
    pts_3d = np.concatenate([xvert.transpose((1, 0)), np.ones((1, len(xvert)), dtype=np.float32)], axis=0)
    P = np.dot(proj_mat, mw_inv)
    pts_2d = np.dot(P, pts_3d)
    pts_2d[0, :] = pts_2d[0, :] / pts_2d[3, :] * (args.image_w//2) + (args.image_w//2)
    pts_2d[1, :] = -pts_2d[1, :] / pts_2d[3, :] * (args.image_h//2) + (args.image_h//2)
    pts_2d = pts_2d.transpose((1, 0))[:, :2]
    pts_2d = np.rint(pts_2d).astype(np.int32)

    distance = np.sum((xvert - cam_loc[np.newaxis, :])**2, axis=1)**0.5
    distance = (distance - distance.min()) / (distance.max() - distance.min())

    kpvis = cal_occ_one_image(points_2d=pts_2d, distance=distance, triangles=xface, image_size=(args.image_h, args.image_w))

    kpvis = np.logical_and(kpvis, np.all(pts_2d >= np.zeros_like(pts_2d), axis=1))
    kpvis = np.logical_and(kpvis, np.all(pts_2d < np.array([args.image_w, args.image_h]), axis=1))
    pts_2d = np.max([np.zeros_like(pts_2d), pts_2d], axis=0)
    pts_2d = np.min([np.ones_like(pts_2d) * (np.array([args.image_w, args.image_h]) - 1), pts_2d], axis=0)

    for i, p in enumerate(pts_2d):
        if kpvis[i] > 0 and occ[p[1], p[0]] == 1:
            kpvis[i] = 0
    
    obj_mask = np.zeros((args.image_h, args.image_w), dtype=np.uint8)
    for tri in xface:
        obj_mask = cv2.fillPoly(obj_mask, [pts_2d[tri].reshape(-1, 1, 2)], 1)
    obj_mask = obj_mask & (1-occ)
    occ = occ | obj_mask

    return {
        'category': cate,
        'sub_category': subcate,
        'theta': theta,
        'scale': scale,
        'loc': loc,
        'kp': pts_2d,
        'kpvis': kpvis,
        'obj_mask': obj_mask,
        'distance': dist(loc, cam_loc)
    }, occ


def process_sample(fname, image_path, scene_path, args):
    with open(scene_path) as f:
        scene = json.load(f)
    
    mw = np.array(scene['matrix_world'])
    mw_inv = np.array(scene['matrix_world_inverted'])
    proj_mat = np.array(scene['projection_matrix'])
    cam_loc = np.array(scene['camera_location'])
    
    objects = scene['objects']
    dist_list = [dist(obj['location'], cam_loc) for obj in objects]
    objects = [obj for _, obj in sorted(zip(dist_list, objects))]

    occ = np.zeros((args.image_h, args.image_w), dtype=np.uint8)
    all_objects = []
    for i in range(len(objects)):
        obj, occ = prepare_object(objects[i], cam_loc, occ, proj_mat, mw_inv, args)
        all_objects.append(obj)

    return {
        'img_name': fname,
        'matrix_world': mw,
        'matrix_world_inv': mw_inv,
        'projection_matrix': proj_mat,
        'camera_location': cam_loc,
        'obj_mask': occ,
        'objects': all_objects
    }


def worker(split, save_path, args, start_idx=0, end_idx=500):
    print(f'Start preparing images for {split} from {start_idx} to {end_idx}')
    out_str = '\n'
    save_img_path = os.path.join(save_path, split, 'images')
    save_anno_path = os.path.join(save_path, split, 'annotations')

    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_anno_path, exist_ok=True)

    img_path = os.path.join(args.dataset_path, 'images')
    scene_path = os.path.join(args.dataset_path, 'scenes')

    fnames = [f'{args.filename_prefix}_{split}_{i:06d}' for i in range(start_idx, end_idx)]
    for fname in fnames:
        save_dict = process_sample(fname, os.path.join(img_path, fname+'.png'), os.path.join(scene_path, fname+'.json'), args)

        shutil.copyfile(os.path.join(img_path, fname+'.png'), os.path.join(save_img_path, fname+'.png'))
        np.savez(os.path.join(save_anno_path, fname), **save_dict)


def main():
    args = parse_args()
    print(args)

    save_path = os.path.join(args.save_path, args.dataset_name)
    os.makedirs(save_path, exist_ok=True)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)

    objs = []

    """
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=0, end_idx=500))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=500, end_idx=1000))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=1000, end_idx=1500))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=1500, end_idx=2000))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=2000, end_idx=2500))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=2500, end_idx=3000))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=3000, end_idx=3500))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=3500, end_idx=4000))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=4000, end_idx=4500))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=4500, end_idx=5000))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=5000, end_idx=5500))
    objs.append(threadpool_executor.submit(worker, 'train', save_path, args, start_idx=5500, end_idx=6000))
    """

    for i in range(20):
        objs.append(threadpool_executor.submit(worker, 'trainLower', save_path, args, start_idx=500*i, end_idx=500*i+500))

    for obj in tqdm(objs):
        obj.result()


def test():
    args = parse_args()
    print(args)

    save_path = os.path.join(args.save_path, args.dataset_name)
    os.makedirs(save_path, exist_ok=True)

    worker('train', save_path, args, start_idx=0, end_idx=5)


if __name__ == '__main__':
    main()
    # test()
