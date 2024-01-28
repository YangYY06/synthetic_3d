import _init_paths

import argparse
import math
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras, camera_position_from_spherical_angles
from scipy.optimize import least_squares
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm

from src.datasets import PASCAL3DPMulti, SuperCLEVRTest
from src.models import NetE2E, NearestMemoryManager, mask_remove_near, MeshInterpolateModule
from src.optim import pre_compute_kp_coords
from src.utils import str2bool, MESH_FACE_BREAKS_1000, load_off, normalize_features, center_crop_fun, plot_score_map, \
    keypoint_score, plot_mesh, plot_loss_landscape, create_cuboid_pts, pose_error, plot_multi_mesh, notation_blender_to_pyt3d, \
    add_3d, flow_warp

from solve_pose_multi import solve_pose_multi_obj, save_segmentation_maps


def parse_args():
    parser = argparse.ArgumentParser(description='Inference 6D pose estimation on SuperCLEVR')

    parser.add_argument('--category', type=str, default='car')
    parser.add_argument('--ckpt', type=str, default='/home/wufei/Documents/pretrain_6d_pose/experiments/may04_superclevr_car_s6/ckpts/saved_model_25000.pth')
    # parser.add_argument('--ckpt', type=str, default='/home/wufei/Documents/pretrain_6d_pose/experiments/may04_superclevr_car_s3/ckpts/saved_model_25000.pth')
    parser.add_argument('--mesh_path', type=str, default='/home/wufei/Documents/pretrain_6d_pose/data/CAD_cate')

    parser.add_argument('--img_path', type=str, default='/mnt/sdd/wufei/data/PASCAL3D+_release1.1/Images')
    parser.add_argument('--anno_path', type=str, default='/mnt/sdd/wufei/data/PASCAL3D+_release1.1/Annotations')
    parser.add_argument('--list_file', type=str, default='/mnt/sdd/wufei/data/PASCAL3D+_release1.1/PASCAL/VOCdevkit/VOC2012/ImageSets/Main')

    parser.add_argument('--test_index', type=int, default=2)
    parser.add_argument('--save_results', type=str, default='vis_outputs')
    parser.add_argument('--metrics',type=str, nargs='+', default=['pose_error'])
    parser.add_argument('--thr', type=float, default=30.0)
    parser.add_argument('--pre_filter', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')

    # Data args
    parser.add_argument('--image_h', type=int, default=320)
    parser.add_argument('--image_w', type=int, default=448)
    parser.add_argument('--prefix', type=str, default='superCLEVR')

    # Model args
    parser.add_argument('--backbone', type=str, default='resnetext')
    parser.add_argument('--d_feature', type=int, default=128)
    parser.add_argument('--local_size', type=int, default=1)
    parser.add_argument('--separate_bank', type=str2bool, default=False)
    parser.add_argument('--max_group', type=int, default=512)
    parser.add_argument('--num_noise', type=int, default=0)
    parser.add_argument('--adj_momentum', type=float, default=0.0)

    # Render args
    parser.add_argument('--down_sample_rate', type=int, default=8)
    parser.add_argument('--blur_radius', type=float, default=0.0)
    parser.add_argument('--num_faces', type=int, default=1)
    parser.add_argument('--gamma', type=int, default=0.01)
    parser.add_argument('--mode', type=str, default='bilinear')

    # Optimization args
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--azimuth_sample', type=int, default=18)
    parser.add_argument('--elevation_sample', type=int, default=7)
    parser.add_argument('--theta_sample', type=int, default=5)
    parser.add_argument('--distance_sample', type=int, default=6)
    parser.add_argument('--px_sample', type=int, default=21)
    parser.add_argument('--py_sample', type=int, default=21)
    parser.add_argument('--loss_type', type=str, default='with_clutter')
    parser.add_argument('--adam_beta_0', type=float, default=0.4)
    parser.add_argument('--adam_beta_1', type=float, default=0.6)

    args = parser.parse_args()

    args.mesh_path = os.path.join(args.mesh_path, args.category)
    args.save_results = args.save_results + f'_{args.test_index}'

    args.img_path = os.path.join(args.img_path, f'{args.category}_pascal')
    args.anno_path = os.path.join(args.anno_path, f'{args.category}_pascal')
    args.list_file = os.path.join(args.list_file, f'{args.category}_trainval.txt')

    return args


def keypoint_score2(feature_map, memory, nocs, clutter_score=None, device='cuda:0'):
    if not torch.is_tensor(feature_map):
        feature_map = torch.tensor(feature_map, device=device).unsqueeze(0) # (1, C, H, W)
    if not torch.is_tensor(memory):
        memory = torch.tensor(memory, device=device) # (nkpt, C)
    
    nkpt, c = memory.size()
    feature_map = feature_map.expand(nkpt, -1, -1, -1)
    memory = memory.view(nkpt, c, 1, 1)

    kpt_map = torch.sum(feature_map * memory, dim=1) # (nkpt, H, W)
    kpt_map, idx = torch.max(kpt_map, dim=0)

    nocs_map = nocs[idx, :].view(kpt_map.shape[0], kpt_map.shape[1], 3).to(device)

    nocs_map = nocs_map * kpt_map.unsqueeze(2)

    if clutter_score is not None:
        nocs_map[kpt_map < clutter_score] = 0.0

    return nocs_map.detach().cpu().numpy()


def get_nocs_features(xvert):
    xvert = xvert.clone()
    xvert[:, 0] -= (torch.min(xvert[:, 0]) + torch.max(xvert[:, 0]))/2.0
    xvert[:, 1] -= (torch.min(xvert[:, 1]) + torch.max(xvert[:, 1]))/2.0
    xvert[:, 2] -= (torch.min(xvert[:, 2]) + torch.max(xvert[:, 2]))/2.0
    xvert[:, 0] /= torch.max(torch.abs(xvert[:, 0])) * 2.0
    xvert[:, 1] /= torch.max(torch.abs(xvert[:, 1])) * 2.0
    xvert[:, 2] /= torch.max(torch.abs(xvert[:, 2])) * 2.0
    xvert += 0.5
    return xvert


def get_rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def get_gt_poses(sample):
    gt_poses = []
    for i in range(len(sample['principal'])):
        gt_poses.append({
            'azimuth': sample['azimuth'][i],
            'elevation': sample['elevation'][i],
            'theta': sample['theta'][i],
            'distance': sample['distance'][i],
            'principal': sample['principal'][i]
        })
    return gt_poses


def plot_corners(img, kps, c=(255, 0, 0), text=False):
    cv2.circle(img, (int(kps[0]), int(kps[1])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[2]), int(kps[3])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[4]), int(kps[5])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[6]), int(kps[7])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[8]), int(kps[9])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[10]), int(kps[11])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[12]), int(kps[13])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[14]), int(kps[15])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[16]), int(kps[17])), 2, color=c, thickness=2)
    if text:
        cv2.putText(img, str(0), (int(kps[0]), int(kps[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(1), (int(kps[2]), int(kps[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(2), (int(kps[4]), int(kps[5])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(3), (int(kps[6]), int(kps[7])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(4), (int(kps[8]), int(kps[9])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(5), (int(kps[10]), int(kps[11])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(6), (int(kps[12]), int(kps[13])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(7), (int(kps[14]), int(kps[15])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(8), (int(kps[16]), int(kps[17])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
    return img


def plot_lines(img, kps, c=(255, 0, 0)):
    pts = np.reshape(kps, (9, 2))
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [3, 7]]
    for i, j in lines:
        img = cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])), (int(pts[j, 0]), int(pts[j, 1])), c, 1)
    return img


def main():
    args = parse_args()
    print(args)
    os.makedirs(args.save_results, exist_ok=True)

    """
    dataset = PASCAL3DPMulti(
        img_path=args.img_path,
        anno_path=args.anno_path,
        list_file=args.list_file,
        ext='.jpg',
        image_h=320, image_w=448
    )
    dataset.file_list = dataset.file_list[:1000]

    azim, elev, theta = [], [], []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        for j in range(len(sample['principal'])):
            azim.append(sample['azimuth'][j])
            elev.append(sample['elevation'][j])
            theta.append(sample['theta'][j])
    
    x = np.linspace(-np.pi, np.pi, num=73, endpoint=True)
    azim = [x-2*np.pi if x > np.pi else x for x in np.array(azim) % (2*np.pi)]
    azim_count = [np.sum((azim >= x[i]) & (azim < x[i+1])) for i in range(len(x)-1)]
    elev = [x-2*np.pi if x > np.pi else x for x in np.array(elev) % (2*np.pi)]
    elev_count = [np.sum((elev >= x[i]) & (elev < x[i+1])) for i in range(len(x)-1)]
    theta = [x-2*np.pi if x > np.pi else x for x in np.array(theta) % (2*np.pi)]
    theta_count = [np.sum((theta >= x[i]) & (theta < x[i+1])) for i in range(len(x)-1)]
    plt.plot(x[1:], azim_count, label='azimuth')
    plt.plot(x[1:], elev_count, label='elevation')
    plt.plot(x[1:], theta_count, label='theta')
    plt.xlabel('radian threshold')
    plt.ylabel('num of instances')

    plt.legend()
    plt.title(f'PASCAL3D+ (1000 images, {len(azim)} objects)')
    plt.tight_layout()
    plt.savefig('p3dp_pascal_poses.png')
    plt.clf()
    exit()
    """

    """
    dataset = SuperCLEVRTest(
        dataset_path='/home/wufei/Documents/pretrain_6d_pose/data/SuperCLEVR_20220506',
        prefix=f'superCLEVR_trainLower',
        category=args.category,
        transform=None
    )
    dataset.file_list = dataset.file_list[:1000]

    a, b = 900, 1000
    dataset.file_list = dataset.file_list[a:b]
    
    azim, elev, theta = [], [], []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        gt_poses = [notation_blender_to_pyt3d(os.path.join(args.mesh_path, '01.off'), s, sample, args.image_h, args.image_w)[0] for s in sample['objects'] if s['category'] == args.category]
        for g in gt_poses:
            azim.append(g['azimuth'])
            elev.append(g['elevation'])
            theta.append(g['theta'])
    
    d = {'azim': np.array(azim), 'elev': np.array(elev), 'theta': np.array(theta)}
    np.save(f'{a}_{b}.npy', d)
    exit()
    """

    azim, elev, theta = [], [], []
    for i in range(10):
        d = np.load(f'{i*100}_{i*100+100}.npy', allow_pickle=True)[()]
        azim += list(d['azim'])
        elev += list(d['elev'])
        theta += list(d['theta'])
    
    x = np.linspace(-np.pi, np.pi, num=73, endpoint=True)
    azim = [x-2*np.pi if x > np.pi else x for x in np.array(azim) % (2*np.pi)]
    azim_count = [np.sum((azim >= x[i]) & (azim < x[i+1])) for i in range(len(x)-1)]
    elev = [x-2*np.pi if x > np.pi else x for x in np.array(elev) % (2*np.pi)]
    elev_count = [np.sum((elev >= x[i]) & (elev < x[i+1])) for i in range(len(x)-1)]
    theta = [x-2*np.pi if x > np.pi else x for x in np.array(theta) % (2*np.pi)]
    theta_count = [np.sum((theta >= x[i]) & (theta < x[i+1])) for i in range(len(x)-1)]
    plt.plot(x[1:], azim_count, label='azimuth')
    plt.plot(x[1:], elev_count, label='elevation')
    plt.plot(x[1:], theta_count, label='theta')
    plt.xlabel('radian threshold')
    plt.ylabel('num of instances')

    plt.legend()
    plt.title(f'SuperCLEVR (1000 images, {len(azim)} objects)')
    plt.tight_layout()
    plt.savefig('superclevr_poses.png')
    plt.clf()


if __name__ == '__main__':
    main()
