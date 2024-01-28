import _init_paths

import argparse
import math
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras, \
    camera_position_from_spherical_angles
from scipy.optimize import least_squares
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm

from src.datasets import PASCAL3DPMulti, PASCAL3DPTest, SuperCLEVRTest,PASCAL3DPTrain
from src.models import NetE2E, NearestMemoryManager, mask_remove_near, MeshInterpolateModule
from src.optim import pre_compute_kp_coords
from src.utils import str2bool, MESH_FACE_BREAKS_1000, load_off, normalize_features, center_crop_fun, plot_score_map, \
    keypoint_score, plot_mesh, plot_loss_landscape, create_cuboid_pts, pose_error, plot_multi_mesh, \
    notation_blender_to_pyt3d, \
    add_3d, flow_warp
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Inference 6D pose estimation on SuperCLEVR')

    parser.add_argument('--category', type=str, default='car')
    parser.add_argument('--ckpt', type=str,
                        default='/home/jiahao/pretrain_6d_pose/experiments/Superclevr1014_car_100/ckpts/saved_model_2000.pth')
    parser.add_argument('--mesh_path', type=str, default='/home/jiahao/pretrain_6d_pose/data/CAD_cate_pascal')

    parser.add_argument('--img_path', type=str, default='../PASCAL3D+_release1.1/Images')
    parser.add_argument('--anno_path', type=str, default='../PASCAL3D+_release1.1/Annotations')
    parser.add_argument('--list_file', type=str, default='../PASCAL3D+_release1.1/Image_sets')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--crop_object', type=bool, default=False)
    parser.add_argument('--distance', type=float, default=8)

    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--test_results', type=str, default='lower_s1.npy')
    parser.add_argument('--save_results', type=str, default='lower_s2.npy')
    parser.add_argument('--metrics', type=str, nargs='+', default=['pose_error'])
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

    parser.add_argument('--syn_path', type=str, default='/home/jiahao/pretrain_6d_pose/data/SuperCLEVR_20220908')
    parser.add_argument('--syn_anno_path', type=str,
                        default='/home/jiahao/pretrain_6d_pose/data/superclevr0908/train/annotations')
    parser.add_argument('--num_cad', type=int, default=6)

    args = parser.parse_args()

    args.mesh_path = os.path.join(args.mesh_path)

    args.img_path = os.path.join(args.img_path, f'{args.category}_imagenet')
    args.anno_path = os.path.join(args.anno_path, f'{args.category}_imagenet')
    args.list_file = os.path.join(args.list_file, f'{args.category}_imagenet_{args.split}.txt')

    return args


def notation_blender_to_pyt3d(sample):
    min_x = [0, ] * 6
    min_x[0] = np.pi / 2 - sample['objects'][0]['theta'] / 180 * np.pi
    min_x[2] = sample['rotate']
    min_x[3] = sample['distance']
    min_x[1] = np.arcsin((sample['cam_loc'][2] - sample['objects'][0]['location'][2]) / min_x[3])
    min_x[4] = sample['objects'][0]['pixel_coords'][0][0] / 520
    min_x[5] = sample['objects'][0]['pixel_coords'][0][1] / 352

    return min_x


def main():
    args = parse_args()

    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_p3d = PASCAL3DPTrain(
        img_path=args.img_path,
        anno_path=args.anno_path,
        list_file=args.list_file,
        category=args.category,
        crop_object=False,
        mesh_path=args.mesh_path,
    )

    for i in tqdm(range(len(dataset_p3d))):
        sample = dataset_p3d[i]
        if sample['distance_orig'] == 0:
            print(sample['img_name'])


if __name__ == '__main__':
    main()
