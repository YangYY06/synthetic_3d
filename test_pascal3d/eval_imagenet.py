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

from src.datasets import PASCAL3DPMulti, PASCAL3DPTest
from src.models import NetE2E, NearestMemoryManager, mask_remove_near, MeshInterpolateModule
from src.optim import pre_compute_kp_coords
from src.utils import str2bool, MESH_FACE_BREAKS_1000, load_off, normalize_features, center_crop_fun, plot_score_map, \
    keypoint_score, plot_mesh, plot_loss_landscape, create_cuboid_pts, pose_error, plot_multi_mesh, \
    notation_blender_to_pyt3d, \
    add_3d, flow_warp

from solve_pose_multi import solve_pose_multi_obj, save_segmentation_maps


def parse_args():
    parser = argparse.ArgumentParser(description='Inference 3D pose estimation on PASCAL3D+')

    parser.add_argument('--category', type=str, default='car')
    parser.add_argument('--ckpt', type=str,
                        default='../experiments/may07_superclevr_car_lower_s1/ckpts/saved_model_25000.pth')
    parser.add_argument('--mesh_path', type=str, default='../data/CAD_cate_pascal')

    parser.add_argument('--img_path', type=str, default='../PASCAL3D+_release1.1/Images')
    parser.add_argument('--anno_path', type=str, default='../PASCAL3D+_release1.1/Annotations')
    parser.add_argument('--list_file', type=str, default='../PASCAL3D+_release1.1/Image_sets')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--crop_object', type=bool, default=False)
    parser.add_argument('--distance', type=float, default=6)

    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--test_results', type=str, default='lower_s1.npy')
    parser.add_argument('--save_results', type=str, default='lower_s2.npy')
    parser.add_argument('--metrics', type=str, nargs='+', default=['pose_error'])
    parser.add_argument('--thr', type=float, default=30.0)
    parser.add_argument('--pre_filter', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')

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

    args.img_path = os.path.join(args.img_path, f'{args.category}_imagenet')
    args.anno_path = os.path.join(args.anno_path, f'{args.category}_imagenet')
    args.list_file = os.path.join(args.list_file, f'{args.category}_imagenet_{args.split}.txt')

    return args


def main():
    args = parse_args()
    print(args)

    test_npy_fname = args.test_results
    npy = np.load(test_npy_fname, allow_pickle=True)[()]
    # npy = np.load(os.path.join('test_results', test_npy_fname), allow_pickle=True)[()]

    dataset = PASCAL3DPTest(
        img_path=args.img_path,
        anno_path=args.anno_path,
        list_file=args.list_file,
        category=args.category,
        crop_object=args.crop_object,
        mesh_path=args.mesh_path,
        dist=args.distance
    )

    # np.random.seed(11)
    # np.random.shuffle(dataset.file_list)
    print(f'len(dataset) = {len(dataset)}')
    print('first 5 samples:')
    for i in range(5):
        print(dataset.file_list[i])

    if args.num_samples > 0 and args.num_samples < len(dataset.file_list):
        dataset.file_list = dataset.file_list[:args.num_samples]

    all_errors = []
    score = []

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        all_pred = npy['pred'][sample['img_name']]

        pred = all_pred


        err = pose_error(pred, sample)
        all_errors.append(err)

        score.append(pred['score'])

    score = np.array(score)
    all_errors = np.array(all_errors)
    high_score = score.argsort()[-100:][::-1]

    print('pseudo_label')
    print('mean:', np.mean(all_errors[high_score] * 180 / np.pi))
    print('median:', np.median(all_errors[high_score] * 180 / np.pi))
    print('max:', np.max(all_errors[high_score] * 180 / np.pi))

    a = []
    for i in high_score:
        a.append(dataset[i]['img_name'])
    np.savetxt(f'{args.category}_100.txt', a, delimiter='\n',
               fmt='%s')


    print('pi/6', np.mean(all_errors <= np.pi / 6))
    print('pi/18', np.mean(all_errors <= np.pi / 18))
    print('median', np.median(all_errors) * 180 / np.pi)



if __name__ == '__main__':
    main()
