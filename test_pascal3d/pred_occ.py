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

from src.datasets import PASCAL3DPMulti, PASCAL3DPTest, PASCAL3DPOcc
from src.models import NetE2E, NearestMemoryManager, mask_remove_near, MeshInterpolateModule
from src.optim import pre_compute_kp_coords
from src.utils import str2bool, MESH_FACE_BREAKS_1000, load_off, normalize_features, center_crop_fun, plot_score_map, \
    keypoint_score, plot_mesh, plot_loss_landscape, create_cuboid_pts, pose_error, plot_multi_mesh, \
    notation_blender_to_pyt3d, \
    add_3d, flow_warp, hmap

from solve_pose_multi import solve_pose_multi_obj, save_segmentation_maps
from solve_pose import solve_pose_with_kp_features


def parse_args():
    parser = argparse.ArgumentParser(description='Inference 6D pose estimation on SuperCLEVR')

    parser.add_argument('--category', type=str, default='car')
    parser.add_argument('--ckpt', type=str,
                        default='/home/jiahao/pretrain_6d_pose/experiments/may07_superclevr_car_lower_s1/ckpts/saved_model_25000.pth')
    parser.add_argument('--mesh_path', type=str, default='/home/jiahao/pretrain_6d_pose/data/CAD_cate_pascal')
    parser.add_argument('--bg_path', type=str, default='/home/jiahao/pretrain_6d_pose/data/bg')
    parser.add_argument('--img_path', type=str,
                        default='/home/jiahao/pretrain_6d_pose/data/ccvl21_data/ccvl21_data/OccludedPASCAL3D')
    parser.add_argument('--anno_path', type=str,
                        default='/home/jiahao/pretrain_6d_pose/PASCAL3D+_release1.1/Annotations')
    parser.add_argument('--distance', type=float, default=6)
    parser.add_argument('--level', type=int, default=1)

    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--save_results', type=str, default='lower_s1.npy')
    parser.add_argument('--metrics', type=str, nargs='+', default=['pose_error'])
    parser.add_argument('--thr', type=float, default=30.0)
    parser.add_argument('--pre_filter', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')

    # Data args
    parser.add_argument('--image_h', type=int, default=352)
    parser.add_argument('--image_w', type=int, default=520)
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
    parser.add_argument('--focal_length', type=int, default=3000)
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
    parser.add_argument('--evaluate_PCK', type=str2bool, default=False)

    args = parser.parse_args()

    args.mesh_path = os.path.join(args.mesh_path, args.category)
    args.anno_path = os.path.join(args.anno_path, f'{args.category}_imagenet')

    return args


def keypoint_score2(feature_map, memory, nocs, clutter_score=None, device='cuda:0'):
    if not torch.is_tensor(feature_map):
        feature_map = torch.tensor(feature_map, device=device).unsqueeze(0)  # (1, C, H, W)
    if not torch.is_tensor(memory):
        memory = torch.tensor(memory, device=device)  # (nkpt, C)

    nkpt, c = memory.size()
    feature_map = feature_map.expand(nkpt, -1, -1, -1)
    memory = memory.view(nkpt, c, 1, 1)

    kpt_map = torch.sum(feature_map * memory, dim=1)  # (nkpt, H, W)
    kpt_map, idx = torch.max(kpt_map, dim=0)

    nocs_map = nocs[idx, :].view(kpt_map.shape[0], kpt_map.shape[1], 3).to(device)

    nocs_map = nocs_map * kpt_map.unsqueeze(2)

    if clutter_score is not None:
        nocs_map[kpt_map < clutter_score] = 0.0

    return nocs_map.detach().cpu().numpy()


def get_nocs_features(xvert):
    xvert = xvert.clone()
    xvert[:, 0] -= (torch.min(xvert[:, 0]) + torch.max(xvert[:, 0])) / 2.0
    xvert[:, 1] -= (torch.min(xvert[:, 1]) + torch.max(xvert[:, 1])) / 2.0
    xvert[:, 2] -= (torch.min(xvert[:, 2]) + torch.max(xvert[:, 2])) / 2.0
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

    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = PASCAL3DPOcc(
        img_path=os.path.join(args.img_path, 'images', f'{args.category}FGL{args.level}_BGL{args.level}'),
        anno_path=args.anno_path,
        dist=args.distance,
        bg_path=args.bg_path
    )
    # dataset.file_list = ['2008_000488', '2008_001142', '2008_000884', '2008_001208']
    print(f'found {len(dataset)} samples')

    # np.random.seed(11)
    # np.random.shuffle(dataset.file_list)
    print(f'len(dataset) = {len(dataset)}')
    print('first 5 samples:')
    for i in range(5):
        print(dataset.file_list[i])

    # print('check_annotation')
    # for i in range(len(dataset)):
    #     sample=dataset[i]
    #     if sample['distance_orig']==0:
    #         print(sample['img_name'])
    # exit()

    net = NetE2E(net_type=args.backbone, local_size=[args.local_size, args.local_size], output_dimension=args.d_feature,
                 reduce_function=None, n_noise_points=args.num_noise, pretrain=True, noise_on_mask=True)
    print(f'num params {sum(p.numel() for p in net.net.parameters())}')
    net = nn.DataParallel(net).cuda().eval()
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    net.load_state_dict(checkpoint['state'])

    if isinstance(checkpoint['memory'], torch.Tensor):
        checkpoint['memory'] = [checkpoint['memory']]

    xvert, xface = load_off(os.path.join(args.mesh_path, '01.off'), to_torch=True)
    nocs_features = get_nocs_features(xvert)
    n = int(xvert.shape[0])
    memory_bank = NearestMemoryManager(inputSize=args.d_feature, outputSize=n + args.num_noise * args.max_group, K=1,
                                       num_noise=args.num_noise,
                                       num_pos=n, momentum=args.adj_momentum)
    memory_bank = memory_bank.cuda()
    with torch.no_grad():
        memory_bank.memory.copy_(checkpoint['memory'][0][0:memory_bank.memory.shape[0]])

    memory = checkpoint['memory'][0][0:memory_bank.memory.shape[0]].detach().cpu().numpy()
    clutter = checkpoint['memory'][0][memory_bank.memory.shape[0]::].detach().cpu().numpy()  # (2560, 128)
    feature_bank = torch.from_numpy(memory)
    clutter_bank = torch.from_numpy(clutter)
    clutter_bank = clutter_bank.cuda()
    clutter_bank = normalize_features(torch.mean(clutter_bank, dim=0)).unsqueeze(0)  # (1, 128)
    kp_features = checkpoint['memory'][0][0:memory_bank.memory.shape[0]].to(args.device)
    clutter_bank = [clutter_bank]

    render_image_size = max(args.image_h, args.image_w) // args.down_sample_rate
    map_shape = (args.image_h // args.down_sample_rate, args.image_w // args.down_sample_rate)
    cameras = PerspectiveCameras(focal_length=args.focal_length // args.down_sample_rate,
                                 image_size=((render_image_size, render_image_size),),
                                 principal_point=((render_image_size // 2, render_image_size // 2),), in_ndc=False,
                                 device=args.device)
    raster_settings = RasterizationSettings(
        image_size=(render_image_size, render_image_size),
        blur_radius=args.blur_radius,
        faces_per_pixel=args.num_faces,
        bin_size=0
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    inter_module = MeshInterpolateModule(xvert, xface, feature_bank, rasterizer,
                                         post_process=center_crop_fun(map_shape, (render_image_size,) * 2))
    inter_module = inter_module.cuda()

    poses, kp_coords, kp_vis = pre_compute_kp_coords(os.path.join(args.mesh_path, '01.off'),
                                                     mesh_face_breaks=MESH_FACE_BREAKS_1000[args.category],
                                                     azimuth_samples=np.linspace(0, np.pi * 2, args.azimuth_sample,
                                                                                 endpoint=False),
                                                     elevation_samples=np.linspace(-np.pi / 2, np.pi / 2,
                                                                                   args.elevation_sample),
                                                     theta_samples=np.linspace(-np.pi / 3, np.pi / 3,
                                                                               args.theta_sample),
                                                     distance_samples=np.linspace(4, 8, args.distance_sample,
                                                                                  endpoint=True))
    # poses = poses.reshape(args.azimuth_sample, args.elevation_sample, args.theta_sample, args.distance_sample, 4)

    all_pred = {}
    # anchor_imgs = np.loadtxt('file_list_car.txt')
    # anchor_features = []
    # dis_mat = np.zeros((args.num_samples, len(anchor_imgs)))

    # for i in anchor_imgs:
    #     i = int(i)
    #     sample = dataset[i]
    #     img_tensor = norm(to_tensor(sample['img'])).unsqueeze(0)
    #     with torch.no_grad():
    #         img_tensor = img_tensor.to(args.device)
    #         feature_map = net.module.forward_test(img_tensor)
    #     anchor_features.append(feature_map)
    error = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        gt_poses = [sample]
        for g in gt_poses:
            g['used'] = False

        img_tensor = norm(to_tensor(sample['img'])).unsqueeze(0)
        # img_tensor = norm(to_tensor(transforms.RandomHorizontalFlip(p=1)(Image.fromarray(sample['img'])))).unsqueeze(0)
        with torch.no_grad():
            img_tensor = img_tensor.to(args.device)
            feature_map = net.module.forward_test(img_tensor)

        # for j, anchor_feature in enumerate(anchor_features):
        #     dis_mat[i, j] = torch.sqrt(torch.sum((feature_map - anchor_feature) ** 2)).item()

        if args.pre_filter:
            with torch.no_grad():
                nkpt, c = kp_features.size()
                b, c, hm_h, hm_w = feature_map.size()
                kpt_score_map = torch.sum(feature_map.expand(nkpt, -1, -1, -1) * kp_features.view(nkpt, c, 1, 1), dim=1)
                kpt_score_map, _ = torch.max(kpt_score_map, dim=0, keepdim=True)  # (1, h, w)
                kpt_score_map = F.unfold(kpt_score_map.unsqueeze(0), kernel_size=(3, 3), padding=1).squeeze().sum(
                    dim=0)  # (h*w, )

                px_samples = torch.from_numpy(np.linspace(0, args.image_w, args.px_sample, endpoint=True)).to(
                    args.device)
                py_samples = torch.from_numpy(np.linspace(0, args.image_h, args.py_sample, endpoint=True)).to(
                    args.device)
                xv, yv = torch.meshgrid(px_samples, py_samples)
                principal_samples = torch.stack([xv, yv], dim=2).reshape(-1, 2)
                principal_samples = torch.round(principal_samples / args.down_sample_rate)
                principal_samples[:, 0] = torch.clamp(principal_samples[:, 0], min=0, max=hm_w - 1)
                principal_samples[:, 1] = torch.clamp(principal_samples[:, 1], min=0, max=hm_h - 1)
                ind = principal_samples[:, 1] * hm_w + principal_samples[:, 0]

                corr = torch.take_along_dim(kpt_score_map, ind.long())

                ind = corr.argsort(descending=True)[:100]
                xv = xv.reshape(-1)[ind].detach().cpu().numpy()
                yv = yv.reshape(-1)[ind].detach().cpu().numpy()
            pred = solve_pose_multi_obj(
                feature_map, inter_module, kp_features, clutter_bank, poses, kp_coords, kp_vis,
                epochs=args.epochs,
                lr=args.lr,
                adam_beta_0=args.adam_beta_0,
                adam_beta_1=args.adam_beta_1,
                mode=args.mode,
                loss_type=args.loss_type,
                device=args.device,
                px_samples=px_samples,
                py_samples=py_samples,
                clutter_img_path=None,
                object_img_path=None,
                blur_radius=args.blur_radius,
                verbose=True,
                down_sample_rate=args.down_sample_rate,
                hierarchical=0,
                xv=xv,
                yv=yv,
                disable_p=True
            )
        else:
            # px_samples = np.linspace(0, args.image_w, args.px_sample, endpoint=True)
            # py_samples = np.linspace(0, args.image_h, args.py_sample, endpoint=True)
            px_samples = np.array([args.image_w / 2])
            py_samples = np.array([args.image_h / 2])

            pred = solve_pose_with_kp_features(
                feature_map, inter_module, kp_features, clutter_bank, poses, kp_coords, kp_vis,
                epochs=args.epochs,
                lr=args.lr,
                adam_beta_0=args.adam_beta_0,
                adam_beta_1=args.adam_beta_1,
                mode=args.mode,
                loss_type=args.loss_type,
                device=args.device,
                px_samples=px_samples,
                py_samples=py_samples,
                clutter_img_path=None,
                object_img_path=None,
                blur_radius=args.blur_radius,
                verbose=True,
                down_sample_rate=args.down_sample_rate,
                hierarchical=0,
                disable_p=True,
                mesh_path=args.mesh_path,
            )
        # for p in pred['final']:
        #     p['azimuth'] = p['azimuth'] - np.pi
        pred['azimuth'] = pred['azimuth'] - np.pi
        error.append(pose_error(sample, pred))

        all_pred[sample['img_name']] = pred

    # error_mat = np.zeros((len(anchor_imgs), 10))
    # img_clutter = {}
    # for i in range(len(anchor_imgs)):
    #     anchor = dataset[int(anchor_imgs[i])]
    #     idx = np.argsort(dis_mat[:, i])[0:10]
    #     imgs = []
    #     for j in range(10):
    #         sample = dataset[idx[j]]
    #         imgs.append(sample['img_name'])
    #         error = pose_error(sample, anchor) * 180 / np.pi
    #         error_mat[i, j] = error
    #     img_clutter[anchor['img_name']] = imgs
    # np.savetxt('error_mat.txt', error_mat)
    # np.save('img_clutter.npy', img_clutter)

    results = {}
    results['pred'] = all_pred
    results['args'] = vars(args)
    file_path = 'exp0811'
    os.makedirs(file_path, exist_ok=True)
    np.save(os.path.join(file_path, args.save_results), results)
    print(f'results saved to {args.save_results}')

    error = np.array(error)
    print('pi/6:', np.mean(error < np.pi / 6))
    print('pi/18:', np.mean(error < np.pi / 18))
    print('median_error:', np.median(error))


if __name__ == '__main__':
    main()
