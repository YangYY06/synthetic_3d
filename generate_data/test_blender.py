import argparse
import argparse
from collections import Counter
from datetime import datetime
import json
import math
import pdb
import os
import random
import subprocess
import sys
import tempfile
import bpy
import bpy_extras
from mathutils import Vector
import numpy as np
import scipy.io as sio

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Render images with Blender")

    parser.add_argument('--base_scene_blendfile', type=str, default='data/base_scene.blend')
    parser.add_argument('--properties_json', type=str, default='data/properties_cgpart.json')
    parser.add_argument('--shape_dir', type=str, default='../data/CGPart')
    parser.add_argument('--model_dir', type=str, default='data/pascal3dp_smartuv')
    parser.add_argument('--material_dir', type=str, default='data/materials')

    parser.add_argument('--camera_x', type=float, default=None)
    parser.add_argument('--camera_y', type=float, default=None)
    parser.add_argument('--camera_z', type=float, default=None)

    parser.add_argument('--save_part_mask', type=bool, default=True)

    # DTD args
    parser.add_argument('--enable_dtd', type=bool, default=False)
    parser.add_argument('--dtd_path', type=str, default='../data/dtd')
    parser.add_argument('--dtd_mat_file', type=str, default='../data/dtd/imdb/imdb.mat')
    parser.add_argument('--dtd_split', type=str, default='train', choices=['train', 'val'])

    # Custom args
    parser.add_argument('--categories', type=str, nargs='+', default=None)
    parser.add_argument('--colors', type=str, nargs='+', default=None)
    parser.add_argument('--sizes', type=str, nargs='+', default=None)
    parser.add_argument('--materials', type=str, nargs='+', default=None)
    parser.add_argument('--textures', type=str, nargs='+', default=None)

    # Object args
    parser.add_argument('--min_objects', type=int, default=3)
    parser.add_argument('--max_objects', type=int, default=10)
    parser.add_argument('--min_dist', type=float, default=0.25)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--min_pixels_per_object', type=int, default=200)
    parser.add_argument('--min_pixels_per_part', type=int, default=20)
    parser.add_argument('--max_retries', type=int, default=150)

    # Output args
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=5)
    parser.add_argument('--filename_prefix', type=str, default='superCLEVR')
    parser.add_argument('--split', type=str, default='new')
    parser.add_argument('--output_dir', type=str, default='../data/SuperCLEVR')
    parser.add_argument('--save_blendfiles', type=bool, default=False)
    parser.add_argument('--version', type=str, default='1.0')
    parser.add_argument('--license', type=str, default='Creative Commons Attribution (CC-BY 4.0)')
    parser.add_argument('--date', type=str, default=datetime.today().strftime("%m/%d/%Y"))

    # Rendering args
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--key_light_jitter', type=float, default=1.0)
    parser.add_argument('--fill_light_jitter', type=float, default=1.0)
    parser.add_argument('--back_light_jitter', type=float, default=1.0)
    parser.add_argument('--xy_jitter', type=float, default=0.5)
    parser.add_argument('--z_jitter', type=float, default=0.5)
    parser.add_argument('--render_num_samples', type=int, default=512)
    parser.add_argument('--render_min_bounces', type=int, default=8)
    parser.add_argument('--render_max_bounces', type=int, default=8)
    parser.add_argument('--render_tile_size', type=int, default=256)
    parser.add_argument('--clevr_scene_path', type=str, default=None)

    if '--' not in sys.argv:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

    args.output_image_dir = os.path.join(args.output_dir, 'images')
    args.output_mask_dir = os.path.join(args.output_dir, 'masks')
    args.output_scene_dir = os.path.join(args.output_dir, 'scenes')
    args.output_scene_file = os.path.join(args.output_dir, 'CLEVR_scenes.json')
    args.output_blend_dir = os.path.join(args.output_dir, 'blendfiles')

    return args


def main():
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    objs = bpy.data.objects
    #objs.remove(objs["Ground"], do_unlink=True)
    utils.set_layer(bpy.data.objects['Ground'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)

    render_args = bpy.context.scene.render
    render_args.use_antialiasing = False
    render_args.engine = "CYCLES"  # BLENDER_RENDER, CYCLES
    render_args.filepath = os.path.join(args.output_dir,'img.png')
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    # utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    # utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    # utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    # utils.set_layer(bpy.data.objects['Ground'], 2)
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'

    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    if args.camera_x is not None:
        bpy.data.objects['Camera'].location[0] = args.camera_x
    if args.camera_y is not None:
        bpy.data.objects['Camera'].location[1] = args.camera_y
    if args.camera_z is not None:
        bpy.data.objects['Camera'].location[2] = args.camera_z
    if args.xy_jitter > 0:
        for i in range(2):
            bpy.data.objects['Camera'].location[i] += rand(args.xy_jitter)
    if args.z_jitter > 0:
        bpy.data.objects['Camera'].location[2] += rand(args.z_jitter)


    camera = bpy.data.objects['Camera']
    print(dir(camera.data))
    print(camera.data.lens)
    camera.data.lens = 200
    utils.delete_object(plane)


    object_dir=args.model_dir
    name = '01'
    obj_pth = 'car/473dd606c5ef340638805e546aa28d99'
    scale = 1
    loc = (0, 0, 0.2)
    theta = 0
    """
       Load an object from a file. We assume that in the directory object_dir, there
       is a file named "$name.blend" which contains a single object named "$name"
       that has unit size and is centered at the origin.
       - scale: scalar giving the size that the object should be in the scene
       - loc: tuple (x, y) giving the coordinates on the ground plane where the
           object should be placed.
       """
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    # Add the obj, and tet the name of the added object
    existings = list(bpy.data.objects)
    # filename = os.path.join(object_dir, obj_pth, 'models/model_normalized.obj')
    # filename = '/home/zhuowan/zhuowan/SuperClevr/render-3d-segmentation/CGPart/models/car/d4251f9cf7f1e0a7cac1226cb3e860ca/models/model_normalized.obj'
    # bpy.ops.import_scene.obj(filepath=filename,use_split_groups=False,use_split_objects=False)
    filepath = os.path.join(object_dir, obj_pth.split('/')[0] + '_' + name + '.blend')
    inner_path = 'Object'
    bpy.ops.wm.append(
        filepath=os.path.join(filepath, inner_path, name),
        directory=os.path.join(filepath, inner_path),
        filename=name
    )
    added_name = list(set(bpy.data.objects) - set(existings))[0].name

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[added_name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    bpy.context.scene.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[2] = theta / 180. * math.pi
    bpy.ops.transform.resize(value=(scale, scale, scale))

    ## Get the min z, and move the obj to the ground
    # # find the min z of the obj
    # zverts = []
    current_obj = bpy.context.scene.objects.active
    # # get all z coordinates of the vertices
    # for face in current_obj.data.polygons:
    #     verts_in_face = face.vertices[:]
    #     for vert in verts_in_face:
    #         local_point = current_obj.data.vertices[vert].co
    #         world_point = current_obj.matrix_world * local_point
    #         zverts.append(world_point[2])
    # # move the obj
    # x, y = loc
    # bpy.ops.transform.translate(value=(x, y, -min(zverts)))

    # Move the obj to loc
    current_obj.location += Vector(loc)
    objs = bpy.data.objects
    print(objs.keys())
    bpy.ops.render.render(write_still=True)

if __name__ == '__main__':
    main()
