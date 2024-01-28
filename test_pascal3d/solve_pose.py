import time

import numpy as np
from pytorch3d.renderer import OpenGLPerspectiveCameras, RasterizationSettings, MeshRasterizer, \
    camera_position_from_spherical_angles
import seaborn as sns
import torch

from src.utils import camera_position_to_spherical_angle
from src.optim import loss_func_type_a, loss_func_type_b, loss_func_type_c, loss_func_type_d
from src.utils import flow_warp, plot_mesh
import cv2
from PIL import Image
import os


def eval_loss(pred, feature_map, inter_module, clutter_bank, use_z=False, down_sample_rate=8,
              loss_type='with_clutter', mode='bilinear', blur_radius=0.0, device='cuda:0'):
    if loss_type == 'without_clutter':
        loss_func = loss_func_type_a
    elif loss_type == 'with_clutter':
        loss_func = loss_func_type_b
    elif loss_type == 'z_map':
        loss_func = loss_func_type_c
    elif loss_type == 'softmax':
        loss_func = loss_func_type_d
    else:
        raise ValueError('Unknown loss function type')
    use_z = (loss_type == 'z_map')

    b, c, hm_h, hm_w = feature_map.size()

    clutter_score = None
    for cb in clutter_bank:
        cs = torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
        if clutter_score is None:
            clutter_score = cs
        else:
            clutter_score = torch.max(clutter_score, cs)

    C = camera_position_from_spherical_angles(pred['distance'], pred['elevation'], pred['azimuth'], degrees=False,
                                              device=device)
    C = torch.nn.Parameter(C, requires_grad=True)
    theta = torch.tensor(pred['theta'], dtype=torch.float32).to(device)
    theta = torch.nn.Parameter(theta, requires_grad=True)
    max_principal = pred['principal']
    flow = torch.tensor([-(max_principal[0] - hm_w * down_sample_rate / 2) / down_sample_rate / 10.0,
                         -(max_principal[1] - hm_h * down_sample_rate / 2) / down_sample_rate / 10.0],
                        dtype=torch.float32).to(device)
    flow = torch.nn.Parameter(flow, requires_grad=True)

    if use_z:
        z = torch.nn.Parameter(
            0.5 * torch.ones((predicted_map.size(0), 1, predicted_map.size(1), predicted_map.size(2)),
                             dtype=torch.float32, device=predicted_map.device), requires_grad=True)

    projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
    flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
    projected_map = flow_warp(projected_map.unsqueeze(0), flow_map * 10.0)[0]
    object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)

    if use_z:
        loss = loss_func(object_score, clutter_score, z, device=device)
    else:
        loss = loss_func(object_score, clutter_score, device=device)
    return loss.item()


def solve_pose_with_kp_features(feature_map, inter_module, kp_features, clutter_bank, poses,
                                kp_coords, kp_vis, epochs=300, lr=5e-2, adam_beta_0=0.4,
                                adam_beta_1=0.6, mode='bilinear', loss_type='with_clutter',
                                px_samples=None, py_samples=None, disable_p=False, device='cuda:0',
                                clutter_img_path=None, object_img_path=None, blur_radius=0.0,
                                verbose=False, down_sample_rate=8, hierarchical=0, mesh_path=None):
    """ Solve object pose with keypoint-based feature pre-rendering.

    Arguments:
    feature_map -- feature map of size (1, C=128, H/8, W/8)
    kp_features -- learned keypoint features of size (K=1024, C=128)
    kp_coords -- keypoint coordinates of size (P=432, K=1024, 2)
    kp_vis -- keypoint visibility of size (P=432, K=1024)
    """

    clutter_img_path = 'clutter.png' if clutter_img_path is None else clutter_img_path
    object_img_path = 'object.png' if object_img_path is None else object_img_path

    time1 = time.time()

    if loss_type == 'without_clutter':
        loss_func = loss_func_type_a
    elif loss_type == 'with_clutter':
        loss_func = loss_func_type_b
    elif loss_type == 'z_map':
        loss_func = loss_func_type_c
    elif loss_type == 'softmax':
        loss_func = loss_func_type_d
    else:
        raise ValueError('Unknown loss function type')
    use_z = (loss_type == 'z_map')

    # Clutter score I: activation score with the center of the clutter features
    # clutter_score = torch.nn.functional.conv2d(feature_map, clutter_bank.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
    # Clutter score II: activate score with the center of two clutter features
    clutter_score = None
    for cb in clutter_bank:
        cs = torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
        if clutter_score is None:
            clutter_score = cs
        else:
            clutter_score = torch.max(clutter_score, cs)
        # print(torch.min(cs), torch.min(clutter_score))

    nkpt, c = kp_features.size()
    # feature_map = feature_map.expand(nkpt, -1, -1, -1)
    memory = kp_features.view(nkpt, c, 1, 1)
    b, c, hm_h, hm_w = feature_map.size()

    if hm_h >= 80 or hm_w >= 56:
        kpt_score_map = np.zeros((nkpt, hm_h, hm_w), dtype=np.float32)
        for i in range(nkpt):
            kpt_score_map[i] = torch.sum(feature_map[0] * memory[i], dim=0).detach().cpu().numpy()
    else:
        kpt_score_map = torch.sum(feature_map.expand(nkpt, -1, -1, -1) * memory, dim=1)  # (nkpt, H, W)
        kpt_score_map = kpt_score_map.detach().cpu().numpy()
    kpt_score_map = kpt_score_map.reshape(nkpt, -1)  # (nkpt, H x W)

    # clutter_score_np = clutter_score.detach().cpu().numpy()
    # kpt_score_map = np.maximum(kpt_score_map, clutter_score_np.reshape(-1)) - clutter_score_np.reshape(-1)

    time2 = time.time()

    # Instead of pre-rendering feature maps, we use sparse keypoint features for coarse detection
    if disable_p:
        # px_samples = np.array([hm_w*down_sample_rate/2])
        # py_samples = np.array([hm_h*down_sample_rate/2])
        px_samples = np.array([260])
        py_samples = np.array([176])

    else:
        px_samples = np.linspace(0, hm_w * down_sample_rate, 7, endpoint=True) if px_samples is None else px_samples
        py_samples = np.linspace(0, hm_h * down_sample_rate, 7, endpoint=True) if py_samples is None else py_samples
    # max_corr = -1e8
    # max_idx = -1
    # max_principal = None

    if hierarchical:
        input_kp_coords = kp_coords.copy()
        input_kp_vis = kp_vis.copy()

    if False:
        xv, yv = np.meshgrid(px_samples, py_samples, indexing='ij')
        principal_samples = np.stack([xv, yv], axis=2).reshape(-1, 2)
        principal_samples = np.repeat(np.expand_dims(principal_samples, axis=1), kp_coords.shape[1], axis=1)
        kp_coords = np.repeat(np.expand_dims(kp_coords, axis=1), len(principal_samples),
                              axis=1)  # (num_poses, num_px x num_py, 1024, 2)
        kp_coords += principal_samples
        kp_coords = kp_coords.reshape(-1, kp_coords.shape[2], 2)  # (num_samples, 1024, 2)
        kp_coords[:, :, 0] = np.rint(kp_coords[:, :, 0] / down_sample_rate)
        kp_coords[:, :, 1] = np.rint(kp_coords[:, :, 1] / down_sample_rate)

        kp_vis = np.repeat(np.expand_dims(kp_vis, axis=1), len(principal_samples),
                           axis=1)  # (num_poses, num_px x num_py, 1024)
        kp_vis = kp_vis.reshape(-1, kp_vis.shape[2])
        kp_vis[kp_coords[:, :, 0] < 0] = 0
        kp_vis[kp_coords[:, :, 0] >= hm_w - 1] = 0
        kp_vis[kp_coords[:, :, 1] < 0] = 0
        kp_vis[kp_coords[:, :, 1] >= hm_h - 1] = 0

        kp_coords[:, :, 0] = np.clip(kp_coords[:, :, 0], 0, hm_w - 1)
        kp_coords[:, :, 1] = np.clip(kp_coords[:, :, 1], 0, hm_h - 1)
        kp_coords = (kp_coords[:, :, 1:2] * hm_w + kp_coords[:, :, 0:1]).astype(np.int32)  # (num_samples, 1024, 1)

        corr = np.take_along_axis(np.expand_dims(kpt_score_map, axis=0), kp_coords, axis=2)[:, :, 0]
        corr = np.sum(corr * kp_vis, axis=1)
        # corr = np.mean(corr * kp_vis, axis=1)
    else:
        px_s, py_s = torch.from_numpy(px_samples).to(device), torch.from_numpy(py_samples).to(device)
        kpc = torch.from_numpy(kp_coords).to(device)
        kpv = torch.from_numpy(kp_vis).to(device)
        kps = torch.from_numpy(kpt_score_map).to(device)

        xv, yv = torch.meshgrid(px_s, py_s)
        principal_samples = torch.stack([xv, yv], dim=2).reshape(-1, 1, 2).repeat(1, kpc.shape[1], 1)

        kpc = kpc.unsqueeze(1).repeat(1, principal_samples.shape[0], 1, 1)
        kpc += principal_samples
        kpc = kpc.reshape(-1, kpc.shape[2], 2)
        kpc = torch.round(kpc / down_sample_rate)

        kpv = kpv.unsqueeze(1).repeat(1, principal_samples.shape[0], 1)
        kpv = kpv.reshape(-1, kpv.shape[2])
        kpv[kpc[:, :, 0] < 0] = 0
        kpv[kpc[:, :, 0] >= hm_w - 1] = 0
        kpv[kpc[:, :, 1] < 0] = 0
        kpv[kpc[:, :, 1] >= hm_h - 1] = 0

        kpc[:, :, 0] = torch.clamp(kpc[:, :, 0], min=0, max=hm_w - 1)
        kpc[:, :, 1] = torch.clamp(kpc[:, :, 1], min=0, max=hm_h - 1)
        kpc = (kpc[:, :, 1:2] * hm_w + kpc[:, :, 0:1]).long()

        corr = torch.take_along_dim(kps.unsqueeze(0), kpc, dim=2)[:, :, 0]
        corr = torch.sum(corr * kpv, dim=1).detach().cpu().numpy()

    a = np.argmax(corr)
    if disable_p:
        px_idx, py_idx = 0, 0
    else:
        p_idx = a % (len(px_samples) * len(py_samples))
        px_idx, py_idx = p_idx // len(py_samples), p_idx % len(py_samples)
        px, py = px_samples[px_idx], py_samples[py_idx]

    if not disable_p and hierarchical > 0:
        dx, dy = px_samples[1] - px_samples[0], py_samples[1] - py_samples[0]
        h_px_samples = np.linspace(px - dx * hierarchical, px + dx * hierarchical, (2 * hierarchical + 1),
                                   endpoint=True)
        h_py_samples = np.linspace(py - dy * hierarchical, py + dy * hierarchical, (2 * hierarchical + 1),
                                   endpoint=True)
        print(len(h_px_samples), len(h_py_samples))

        xv, yv = np.meshgrid(h_px_samples, h_py_samples, indexing='ij')
        principal_samples = np.stack([xv, yv], axis=2).reshape(-1, 2)
        principal_samples = np.repeat(np.expand_dims(principal_samples, axis=1), input_kp_coords.shape[1], axis=1)
        kp_coords = np.repeat(np.expand_dims(input_kp_coords, axis=1), len(principal_samples),
                              axis=1)  # (num_poses, num_px x num_py, 1024, 2)
        kp_coords += principal_samples
        kp_coords = kp_coords.reshape(-1, kp_coords.shape[2], 2)  # (num_samples, 1024, 2)
        kp_coords[:, :, 0] = np.rint(kp_coords[:, :, 0] / down_sample_rate)
        kp_coords[:, :, 1] = np.rint(kp_coords[:, :, 1] / down_sample_rate)

        kp_vis = np.repeat(np.expand_dims(input_kp_vis, axis=1), len(principal_samples),
                           axis=1)  # (num_poses, num_px x num_py, 1024)
        kp_vis = kp_vis.reshape(-1, kp_vis.shape[2])
        kp_vis[kp_coords[:, :, 0] < 0] = 0
        kp_vis[kp_coords[:, :, 0] >= hm_w - 1] = 0
        kp_vis[kp_coords[:, :, 1] < 0] = 0
        kp_vis[kp_coords[:, :, 1] >= hm_h - 1] = 0

        kp_coords[:, :, 0] = np.clip(kp_coords[:, :, 0], 0, hm_w - 1)
        kp_coords[:, :, 1] = np.clip(kp_coords[:, :, 1], 0, hm_h - 1)
        kp_coords = (kp_coords[:, :, 1:2] * hm_w + kp_coords[:, :, 0:1]).astype(np.int32)  # (num_samples, 1024, 1)

        corr = np.take_along_axis(np.expand_dims(kpt_score_map, axis=0), kp_coords, axis=2)[:, :, 0]
        corr = np.sum(corr * kp_vis, axis=1)
        # corr = np.mean(corr * kp_vis, axis=1)
        a = np.argmax(corr)
        if disable_p:
            px_idx, py_idx = 0, 0
        else:
            p_idx = a % (len(h_px_samples) * len(h_py_samples))
            px_idx, py_idx = p_idx // len(h_py_samples), p_idx % len(h_py_samples)
        px, py = h_px_samples[px_idx], h_py_samples[py_idx]

    time3 = time.time()

    if disable_p:
        pred = {
            'azimuth_pre': poses[a, 0],
            'elevation_pre': poses[a, 1],
            'theta_pre': poses[a, 2],
            'distance_pre': poses[a, 3],
            # 'principal_pre': [hm_w * down_sample_rate / 2, hm_h * down_sample_rate / 2]
            'principal_pre': [260, 176]
        }
    else:
        l = len(px_samples) * len(py_samples)
        pred = {
            'azimuth_pre': poses[a // l, 0],
            'elevation_pre': poses[a // l, 1],
            'theta_pre': poses[a // l, 2],
            'distance_pre': poses[a // l, 3],
            'principal_pre': [px, py]
        }

    """
    # Clutter score I: activation score with the center of the clutter features
    # clutter_score = torch.nn.functional.conv2d(feature_map, clutter_bank.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
    # Clutter score II: activate score with the center of two clutter features
    clutter_score = None
    for cb in clutter_bank:
        cs = torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
        if clutter_score is None:
            clutter_score = cs
        else:
            clutter_score = torch.max(clutter_score, cs)
        # print(torch.min(cs), torch.min(clutter_score))
    """

    C = camera_position_from_spherical_angles(pred['distance_pre'], pred['elevation_pre'], pred['azimuth_pre'],
                                              degrees=False, device=device)
    C = torch.nn.Parameter(C, requires_grad=True)
    theta = torch.tensor(pred['theta_pre'], dtype=torch.float32).to(device)
    theta = torch.nn.Parameter(theta, requires_grad=True)
    max_principal = pred['principal_pre']
    flow = torch.tensor([-(max_principal[0] - hm_w * down_sample_rate / 2) / down_sample_rate / 10.0,
                         -(max_principal[1] - hm_h * down_sample_rate / 2) / down_sample_rate / 10.0],
                        dtype=torch.float32).to(device)
    flow = torch.nn.Parameter(flow, requires_grad=True)

    if use_z:
        z = torch.nn.Parameter(
            0.5 * torch.ones((predicted_map.size(0), 1, predicted_map.size(1), predicted_map.size(2)),
                             dtype=torch.float32, device=predicted_map.device), requires_grad=True)

    param_list = [C, theta]
    if not disable_p:
        param_list.append(flow)
    if use_z:
        param_list.append(z)
    optim = torch.optim.Adam(params=param_list, lr=lr, betas=(adam_beta_0, adam_beta_1))
    # optim = torch.optim.Adagrad(params=param_list, lr=lr)
    # optim = torch.optim.AdamW(params=param_list, lr=lr, betas=(adam_beta_0, adam_beta_1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.2)

    for epoch in range(epochs):
        projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
        flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
        projected_map = flow_warp(projected_map.unsqueeze(0), flow_map * 10.0)[0]
        object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)

        if use_z:
            loss = loss_func(object_score, clutter_score, z, device=device)
        else:
            loss = loss_func(object_score, clutter_score, device=device)

        loss.backward()
        optim.step()
        optim.zero_grad()

        if (epoch + 1) % 100 == 0:
            scheduler.step(None)

        if verbose and ((epoch + 1) % 25 == 0):
            distance_pred, elevation_pred, azimuth_pred = camera_position_to_spherical_angle(C)
            # print(f'{epoch + 1:5d} theta={theta.item():.3f} '
            #       f'elev={elevation_pred.item():.3f} azim={azimuth_pred.item():.3f} '
            #       f'dist={distance_pred.item():.3f} px={flow[0].item():.3f} '
            #       f'py={flow[1].item():.3f} '
            #       f'loss={loss.item():.3f}')

        """
        if verbose and (epoch+1 == epochs):
            print('saving clutter.png and object.png')
            ax = sns.heatmap(clutter_score.squeeze().detach().cpu().numpy(), square=True, xticklabels=False, yticklabels=False, vmin=0.0, vmax=1.0, cbar=False)
            ax.figure.tight_layout()
            ax.figure.savefig(clutter_img_path, bbox_inches='tight', pad_inches=0.01)
            ax = sns.heatmap(object_score.squeeze().detach().cpu().numpy(), square=True, xticklabels=False, yticklabels=False, vmin=0.0, vmax=1.0, cbar=False)
            ax.figure.tight_layout()
            ax.figure.savefig(object_img_path, bbox_inches='tight', pad_inches=0.01)
            if use_z:
                ax = sns.heatmap(z[0, 0].detach().cpu().numpy(), square=True, xticklabels=False, yticklabels=False, vmin=0.0, vmax=1.0, cbar=False)
                ax.figure.tight_layout()
                ax.figure.savefig('z.png', bbox_inches='tight', pad_inches=0.01)
        """

        if epoch == 500:
            for g in optim.param_groups:
                g['lr'] = lr / 5.0

    distance_pred, elevation_pred, azimuth_pred = camera_position_to_spherical_angle(C)
    theta_pred, distance_pred, elevation_pred, azimuth_pred = theta.item(), distance_pred.item(), elevation_pred.item(), azimuth_pred.item()
    px_pred, py_pred = -flow[0].item() * 10.0, -flow[1].item() * 10.0

    time4 = time.time()

    """
    if verbose:
        print('prepare render time:', time2-time1)
        print('pre-render time:', time3-time2)
        print('post opt time:', time4-time3)
    """
    # mask = plot_mesh(np.zeros([352,520,3]),
    #                  os.path.join(mesh_path, '01.off'), size=(1, 1, 1),
    #                  focal_length=3000,
    #                  azimuth=np.float32((azimuth_pred - np.pi) % (2 * np.pi)), elevation=np.float32(elevation_pred),
    #                  theta=np.float32(theta_pred),
    #                  distance=np.float32(distance_pred),
    #                  principal=np.array([px_pred * down_sample_rate + hm_w * down_sample_rate / 2,
    #                                      py_pred * down_sample_rate + hm_h * down_sample_rate / 2]).astype(
    #                      np.float32), down_sample_rate=8)
    # obj_mask = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
    # obj_mask = np.where(obj_mask == 0, 0, 1)
    #
    # obj_mask = cv2.resize(obj_mask.astype('float32'), dsize=(obj_mask.shape[1] // 8, obj_mask.shape[0] // 8))
    #
    #
    # obj_mask = torch.tensor(obj_mask).cuda()

    # seg_map = ((object_score > clutter_score) * (object_score > 0.0))
    seg_map = object_score > 0.0
    score_map = object_score.squeeze()
    score = torch.sum(seg_map * score_map) / torch.sum(seg_map)

    # score_ = torch.sum(obj_mask * score_map) / torch.sum(obj_mask)

    score_90 = torch.sum(seg_map * score_map > 0.9) / torch.sum(seg_map)
    score_80 = torch.sum(seg_map * score_map > 0.8) / torch.sum(seg_map)

    pred.update({
        'theta': theta_pred,
        'distance': distance_pred,
        'azimuth': azimuth_pred,
        'elevation': elevation_pred,
        'px': px_pred,
        'py': py_pred,
        'loss_pred': loss.item(),
        'prepare_render_time': time2 - time1,
        'pre_render_time': time3 - time2,
        'post_opt_time': time4 - time3,
        'score': score.item(),
        # 'score_': score_.item(),
        'score_90': score_90.item(),
        'score_80': score_80.item()
    })

    pred['principal'] = [pred['px'] * down_sample_rate + hm_w * down_sample_rate / 2,
                         pred['py'] * down_sample_rate + hm_h * down_sample_rate / 2]

    return pred
