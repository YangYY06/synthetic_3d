from .blender import notation_blender_to_pyt3d
from .flow_warp import flow_warp
from .mesh import create_cuboid_pts, load_off, save_off, camera_position_to_spherical_angle, angel_gradient_modifier, \
    decompose_pose, normalize, standard_loss_func_with_clutter, verts_proj, verts_proj_matrix, rotation_theta, \
    rasterize, translation_matrix, campos_to_R_T_det, campos_to_R_T, forward_interpolate, set_bary_coords_to_nearest, \
    vertex_memory_to_face_memory, center_crop_fun, pre_process_mesh_pascal, create_keypoints_pts, meshelize
from .metrics import pose_error, pose_err, calculate_ap, add_3d, calculate_ap_2d
from .plot import (
    plot_score_map,
    plot_mesh,
    plot_multi_mesh,
    fuse,
    alpha_merge_imgs,
    plot_loss_landscape
)
from .pose import (
    get_transformation_matrix,
    rotation_theta,
    cal_rotation_matrix
)
from .transforms import Transform6DPose
from .utils import str2bool, EasyDict, load_off, save_off, normalize_features, keypoint_score, hmap

MESH_FACE_BREAKS_1000 = {
    'car': [242, 484, 583, 682, 880, 1078],
    'motorbike': [189, 378, 486, 594, 846, 1098],
    'bus': [216, 432, 512, 592, 862, 1132],
    'bicycle': [176, 352, 456, 560, 846, 1132],
    'aeroplane': [306, 612, 738, 864, 983, 1102],
    'chair':[144, 288, 492, 696, 900, 1104],
    'tvmonitor':[187, 374, 578, 782, 914, 1046],
    'sofa':[198, 396, 594, 792, 913, 1034],
    'bottle':[90, 180, 410, 640, 847, 1054],
    'boat':[133, 266, 378, 490, 794, 1098],
    'diningtable':[234, 468, 648, 828, 958, 1088],
    'train':[204, 408, 456, 504, 776, 1048]
}

subcate_to_cate = {
    'jet': 'aeroplane',         'fighter': 'aeroplane',     'airliner': 'aeroplane',    'biplane': 'aeroplane',
    'minivan': 'car',           'suv': 'car',               'wagon': 'car',             'truck': 'car',
    'sedan': 'car',             'cruiser': 'motorbike',     'dirtbike': 'motorbike',    'chopper': 'motorbike',
    'scooter': 'motorbike',     'road': 'bicycle',          'tandem': 'bicycle',        'mountain': 'bicycle',
    'utility': 'bicycle',       'double': 'bus',            'regular': 'bus',           'school': 'bus',
    'articulated': 'bus'
}
