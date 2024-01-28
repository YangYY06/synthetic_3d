import argparse
import os
import sys

import bpy
import bmesh
import numpy as np

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from src.utils import save_off

from material_cycles_converter import AutoNode
from preprocess_cgpart import load_properties_json


def parse_args():
    parser = argparse.ArgumentParser(description='Create cuboid mesh model for each category')
    parser.add_argument('--out_dir', type=str, default='../data/CAD_cate2')
    parser.add_argument('--number_vertices', type=int, default=1000)
    parser.add_argument('--cgpart_dir', type=str, default='../data/CGPart')
    parser.add_argument('--linear_coverage', type=float, default=0.99)
    parser.add_argument('--properties_json', type=str, default='data/properties_cgpart.json')
    parser.add_argument('--label_dir', type=str, default='../data/CGPart/labels')
    if '--' not in sys.argv:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    return args


def main():
    args = parse_args()

    color_name_to_rgba, size_mapping, obj_info = load_properties_json(args.properties_json, args.label_dir)

    objs = bpy.data.objects
    objs.remove(objs["Cube"], do_unlink=True)
    objs.remove(objs["Camera"], do_unlink=True)
    objs.remove(objs["Lamp"], do_unlink=True)

    coords_dict = {}
    for si, obj_name in enumerate(obj_info['info_pth']):
        model_id = obj_info['info_pth'][obj_name]
        cat_name = obj_info['info_pth'][obj_name].split('/')[0]
        print(obj_name, cat_name)

        existings = list(bpy.data.objects)
        bpy.ops.import_scene.obj(filepath=os.path.join(args.cgpart_dir, 'models', model_id, 'models', 'model_normalized.obj'), use_split_groups=False, use_split_objects=False)
        added_name = list(set(bpy.data.objects) - set(existings))[0].name

        obj_car = bpy.data.objects[added_name]
        obj_car.name = obj_name
        bpy.context.scene.objects.active = obj_car
        AutoNode()

        if cat_name not in coords_dict:
            coords_dict[cat_name] = {}
        coords_dict[cat_name][obj_name] = np.array([v.co for v in obj_car.data.vertices])

        objs = bpy.data.objects
        objs.remove(objs[obj_name], do_unlink=True)
    
    for cate in coords_dict:
        print(cate, coords_dict[cate].keys())
        os.makedirs(os.path.join(args.out_dir, cate), exist_ok=True)

        vertices = np.concatenate([v for v in coords_dict[cate].values()], axis=0)[:, [0, 2, 1]]
        selected_shape = int(vertices.shape[0] * args.linear_coverage)
        out_pos = []

        for i in range(vertices.shape[1]):
            v_sorted = np.sort(vertices[:, i])
            v_group = v_sorted[selected_shape::] - v_sorted[0:-selected_shape]
            min_idx = np.argmin(v_group)
            out_pos.append((v_sorted[min_idx], v_sorted[min_idx + selected_shape]))

        xvert, xface = meshelize(*out_pos, number_vertices=args.number_vertices)
        save_off(os.path.join(args.out_dir, cate, '01.off'), xvert, xface)


def meshelize(x_range, y_range, z_range, number_vertices):
    w, h, d = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
    total_area = (w * h + h * d + w * d) * 2

    # On average, every vertice attarch 6 edges. Each triangle has 3 edges
    mesh_size = total_area / (number_vertices * 2)

    edge_length = (mesh_size * 2) ** .5

    x_samples = x_range[0] + np.linspace(0, w, int(w / edge_length + 1))
    y_samples = y_range[0] + np.linspace(0, h, int(h / edge_length + 1))
    z_samples = z_range[0] + np.linspace(0, d, int(d / edge_length + 1))

    xn = x_samples.size
    yn = y_samples.size
    zn = z_samples.size

    out_vertices = []
    out_faces = []
    base_idx = 0

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[0]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += yn * xn
    print(base_idx, end=', ')

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[-1]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += yn * xn
    print(base_idx, end=', ')

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[0], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += zn * xn
    print(base_idx, end=', ')

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[-1], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += zn * xn
    print(base_idx, end=', ')

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[0], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
            out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
    base_idx += zn * yn
    print(base_idx, end=', ')

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[-1], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
            out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
    base_idx += zn * yn
    print(base_idx)

    return np.array(out_vertices), np.array(out_faces)


if __name__ == '__main__':
    main()
