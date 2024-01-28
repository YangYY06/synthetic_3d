import _init_paths

import argparse
import json
import os
import re

import numpy as np
import pymeshlab


def save_off(off_file_name, vertices, faces):
    out_string = 'OFF\n'
    out_string += '%d %d 0\n' % (vertices.shape[0], faces.shape[0])
    for v in vertices:
        out_string += '%.16f %.16f %.16f\n' % (v[0], v[1], v[2])
    for f in faces:
        out_string += '3 %d %d %d\n' % (f[0], f[1], f[2])
    with open(off_file_name, 'w') as fl:
        fl.write(out_string)
    return


def load_off(off_file_name, to_torch=False):
    file_handle = open(off_file_name)
    # n_points = int(file_handle.readlines(6)[1].split(' ')[0])
    # all_strings = ''.join(list(islice(file_handle, n_points)))

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(' ')[0])
    all_strings = ''.join(file_list[2:2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep='\n')

    all_strings = ''.join(file_list[2 + n_points:])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep='\n')

    array_ = array_.reshape((-1, 3))

    if not to_torch:
        return array_, array_int.reshape((-1, 7))[:, 1::]
    else:
        return torch.from_numpy(array_), torch.from_numpy(array_int.reshape((-1, 4))[:, 1::])


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .obj mesh to .off mesh')

    parser.add_argument('--mesh_path', type=str, default='/home/jiahao/pretrain_6d_pose/data/CGPart/models/car')

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    subcate_list = os.listdir(args.mesh_path)
    print(subcate_list)
    for subcate in subcate_list:
        if subcate != '.DS_Store':
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(os.path.join(args.mesh_path, subcate, 'models/model_normalized_centered.obj'))
            print(subcate, ms.current_mesh().vertex_number())
            ms.save_current_mesh(os.path.join(args.mesh_path, subcate, 'models/mesh.off'))


def rotate():
    args = parse_args()
    subcate_list = os.listdir(args.mesh_path)
    for subcate in subcate_list:
        if subcate != '.DS_Store':
            xvert, xface = load_off(os.path.join(args.mesh_path, subcate, 'models/mesh.off'))
            xvert = xvert[:, [0, 2, 1]]
            xvert[:, 1] = -xvert[:, 1]
            save_off(os.path.join(args.mesh_path, subcate, 'models/mesh_consistent.off'), xvert, xface)

if __name__ == '__main__':
    # ms = pymeshlab.MeshSet()
    # ms.load_new_mesh('car_remesh_0.04.obj')
    # print(ms.current_mesh().vertex_number())

    rotate()
